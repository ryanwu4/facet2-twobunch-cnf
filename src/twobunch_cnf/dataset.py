"""Dataset + DataModule for the two-bunch conditional-flow surrogate.

Loads processed/twobunch_flow.h5 into memory and serves per-sample dicts:
  knobs           (8,)      knobs normalized to [0,1] (from sweep bounds)
  drive_parts     (P,6)     drive cloud, per-bunch StandardScaler-standardized
  witness_parts   (P,6)     witness cloud, standardized (zeros where not density-trainable)
  drive_present / witness_viable / drive_density / witness_density   bool masks
  drive_frac / witness_frac   surviving fractions (feasibility-head targets)

Normalization mirrors the photoinjector flow: knobs min-max to [0,1] (the RL box),
particles per-dim per-bunch StandardScaler (raw scales span ~x:1e-4 m to pz:1e10 eV/c).
"""
from __future__ import annotations

import json

import h5py
import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split


def group_split(groups, val_fraction: float, seed: int):
    """(train_idx, val_idx) with whole groups on one side; val gets ~val_fraction of the rows."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    sizes = {g: int((groups == g).sum()) for g in uniq}
    target, val_g, n = val_fraction * len(groups), set(), 0
    for i in order:
        if n >= target:
            break
        val_g.add(uniq[i]); n += sizes[uniq[i]]
    is_val = np.isin(groups, list(val_g))
    return np.flatnonzero(~is_val).tolist(), np.flatnonzero(is_val).tolist()


def _load(h, k):
    return np.asarray(h[k][...])


SCALER_KEYS = ("drive_mean", "drive_std", "witness_mean", "witness_std")


class TwoBunchFlowDataset(Dataset):
    """`scaler`: per-bunch particle scalers {drive_mean, drive_std, witness_mean, witness_std} that
    REPLACE the processed norm's for standardization (and in `self.norm`). Used when warm-starting
    from a checkpoint: the model decodes with its own scalers, so the training targets must be
    standardized with exactly those, or the flow learns in one frame and decodes in another."""

    def __init__(self, processed_h5: str, norm_json: str | None = None, scaler: dict | None = None,
                 truth_h5: str | None = None, resample_p: int | None = None):
        with h5py.File(processed_h5, "r") as h:
            knobs = _load(h, "knobs").astype(np.float32)              # (N,8)
            drive = _load(h, "drive_parts").astype(np.float32)        # (N,P,6)
            witness = _load(h, "witness_parts").astype(np.float32)    # (N,P,6)
            masks = {k: _load(h, k).astype(bool) for k in
                     ("drive_present", "witness_viable", "drive_density", "witness_density")}
            dfrac = _load(h, "drive_frac").astype(np.float32)
            wfrac = _load(h, "witness_frac").astype(np.float32)

        norm_json = norm_json or processed_h5.replace(".h5", "_norm.json")
        with open(norm_json) as f:
            norm = json.load(f)
        if scaler is not None:
            norm = {**norm, **{k: [float(v) for v in scaler[k]] for k in SCALER_KEYS},
                    "scaler_source": "checkpoint"}
        self.norm = norm
        groups, holdout = None, None
        with h5py.File(processed_h5, "r") as h:
            names = str(h.attrs.get("campaigns", "")).split(",")
            idx = np.asarray(h["idx"][...]) if "idx" in h else np.arange(len(knobs))
            self.row_campaign = np.asarray(h["campaign"][...]) if "campaign" in h else np.zeros(len(knobs), int)
            self.row_keys = [f"{names[c] if c < len(names) else ''}/{int(i)}"
                             for c, i in zip(self.row_campaign, idx)]
            if "group" in h:
                groups = np.asarray(h["group"][...])
            if "holdout" in h:
                holdout = np.asarray(h["holdout"][...]).astype(bool)
        # pre-registered TEST rows: never in train or val (evaluate them explicitly, once)
        self.holdout = holdout if holdout is not None else np.zeros(len(knobs), bool)
        # split groups: rows that must stay on the same side of train/val (preprocess `group`);
        # default = every row its own group (plain random split, as before)
        self.has_groups = groups is not None
        self.groups = groups if groups is not None else np.arange(len(knobs))

        lo = np.array(norm["knob_low"], np.float32)
        hi = np.array(norm["knob_high"], np.float32)
        dm = np.array(norm["drive_mean"], np.float32)
        ds = np.array(norm["drive_std"], np.float32)
        wm = np.array(norm["witness_mean"], np.float32)
        ws = np.array(norm["witness_std"], np.float32)

        self.knobs = torch.tensor((knobs - lo) / np.maximum(hi - lo, 1e-12))
        self.drive = torch.tensor((drive - dm) / ds)
        self.witness = torch.tensor((witness - wm) / ws)
        self.drive_present = torch.tensor(masks["drive_present"])
        self.witness_viable = torch.tensor(masks["witness_viable"])
        self.drive_density = torch.tensor(masks["drive_density"])
        self.witness_density = torch.tensor(masks["witness_density"])
        self.drive_frac = torch.tensor(dfrac)
        self.witness_frac = torch.tensor(wfrac)
        self.P = int(self.drive.shape[1])
        self.cond_dim = int(self.knobs.shape[1])
        # resample_p (2026-10-21): each visit returns a fresh random resample_p-particle subset of the stored P (most
        # of a big table's information at a small table's cost per step)
        self.resample_p = int(resample_p) if resample_p and int(resample_p) < self.P else None
        # truth_h5 (2026-10-21): full-particle per-row truths (scripts/build_truth_table_2026-10-21.py) -- centroids in
        # the standardized frame, spacing and drive sigma_z in physical units; NaN where unknown
        self.has_truth = truth_h5 is not None
        if truth_h5 is not None:
            with h5py.File(truth_h5, "r") as h:
                dc, wc = h["drive_centroid"][...], h["witness_centroid"][...]
                self.truth_spacing = torch.tensor(h["bunch_spacing"][...])
                self.truth_drive_sigma_z = torch.tensor(h["drive_sigma_z"][...]).float()
            if len(dc) != len(knobs):
                raise ValueError(f"{truth_h5}: {len(dc)} rows, table has {len(knobs)}")
            self.truth_drive_c = torch.tensor((dc - dm) / ds).float()
            self.truth_witness_c = torch.tensor((wc - wm) / ws).float()

    def __len__(self):
        return self.knobs.shape[0]

    def __getitem__(self, i):
        d, w = self.drive[i], self.witness[i]
        if self.resample_p is not None:
            d = d[torch.randperm(self.P)[:self.resample_p]]
            w = w[torch.randperm(self.P)[:self.resample_p]]
        extra = {}
        if self.has_truth:
            extra = {"drive_ctrue": self.truth_drive_c[i], "witness_ctrue": self.truth_witness_c[i],
                     "drive_sztrue": self.truth_drive_sigma_z[i]}
        return {
            **extra,
            "knobs": self.knobs[i],
            "drive": d, "witness": w,
            "drive_present": self.drive_present[i], "witness_viable": self.witness_viable[i],
            "drive_density": self.drive_density[i], "witness_density": self.witness_density[i],
            "drive_frac": self.drive_frac[i], "witness_frac": self.witness_frac[i],
        }


class TwoBunchFlowDataModule(L.LightningDataModule):
    def __init__(self, processed_h5="processed/twobunch_flow.h5", norm_json=None,
                 batch_size=64, val_fraction=0.1, num_workers=0, split_seed=42, scaler=None,
                 split="random", orig_split=None, truth_h5=None, resample_p=None):
        super().__init__()
        self.save_hyperparameters()
        self.full = None
        self.train = None
        self.val = None

    def setup(self, stage=None):
        if self.full is None:
            self.full = TwoBunchFlowDataset(self.hparams.processed_h5, self.hparams.norm_json,
                                            scaler=self.hparams.scaler, truth_h5=self.hparams.truth_h5,
                                            resample_p=self.hparams.resample_p)
        if self.full.holdout.any() and self.hparams.split != "group":
            raise ValueError("table has pre-registered holdout rows: use split='group' (excludes them)")
        if self.hparams.split == "random":
            n_val = int(round(len(self.full) * self.hparams.val_fraction))
            n_train = len(self.full) - n_val
            self.train, self.val = random_split(
                self.full, [n_train, n_val],
                generator=torch.Generator().manual_seed(self.hparams.split_seed))
        elif self.hparams.split == "group":
            if not self.full.has_groups:
                raise ValueError(f"{self.hparams.processed_h5} has no 'group' column (re-run preprocess)")
            # rows an earlier (warm-start) flow trained / validated on keep that role (no leak into
            # val); the rest: group split stratified by campaign; holdout rows in neither
            orig = self.hparams.orig_split or {}
            role = np.array([orig.get(k, "") for k in self.full.row_keys], dtype=object)
            # rows without a role inherit their split group's (a top-up row joins the group of the row it
            # pairs with); a group must never span train and val
            grp = self.full.groups
            for gid in np.unique(grp[role != ""]):
                members = np.flatnonzero(grp == gid)
                roles = {r for r in role[members] if r}
                if len(roles) > 1:
                    raise ValueError(f"split group {gid} spans roles {roles} (rows {members[:6].tolist()} ...)")
                role[members[role[members] == ""]] = roles.pop()
            role = role.astype(str)
            tr_idx = list(np.flatnonzero((role == "train") & ~self.full.holdout))
            va_idx = list(np.flatnonzero((role == "val") & ~self.full.holdout))
            free = (role == "") & ~self.full.holdout
            for c in np.unique(self.full.row_campaign[free]):
                rows = np.flatnonzero(free & (self.full.row_campaign == c))
                tr, va = group_split(self.full.groups[rows], self.hparams.val_fraction,
                                     self.hparams.split_seed + int(c))
                tr_idx += rows[tr].tolist(); va_idx += rows[va].tolist()
            self.train, self.val = Subset(self.full, sorted(tr_idx)), Subset(self.full, sorted(va_idx))
            self.test = Subset(self.full, np.flatnonzero(self.full.holdout).tolist())
        else:
            raise ValueError(f"split must be 'random' or 'group', got {self.hparams.split!r}")

    def _dl(self, ds, shuffle):
        return DataLoader(ds, batch_size=self.hparams.batch_size, shuffle=shuffle,
                          num_workers=self.hparams.num_workers,
                          persistent_workers=self.hparams.num_workers > 0)

    def train_dataloader(self):
        return self._dl(self.train, True)

    def val_dataloader(self):
        return self._dl(self.val, False)

    @property
    def norm(self):
        assert self.full is not None
        return self.full.norm
