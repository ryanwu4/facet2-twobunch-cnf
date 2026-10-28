"""Validation metrics that see what the injector fine-tune is for (pooled NLL does not):

  val_paired_delta_rms_um   RMS over validation pairs of (predicted - true) injector effect on the
                            bunch spacing, effect = row(injector point) - row(injector = golden) at
                            identical linac knobs (Bmad pairs share the radiation seed). Predictions
                            use common random numbers, so flow sampling noise largely cancels.
  val_old_spacing_mae_um    spacing MAE on validation rows of campaigns WITHOUT pairs (the old,
                            golden-beam data: the no-regression check).

Truth comes from the table's own (standardized) particle subsamples, de-standardized with the
model's scalers (the dataset is standardized with the same ones).
"""
from __future__ import annotations

import lightning as L
import numpy as np
import torch

from .properties import inter_bunch


class PairedDeltaMetrics(L.Callback):
    def __init__(self, datamodule, n: int = 2048, seed: int = 0, max_old: int = 512):
        self.dm, self.n, self.seed, self.max_old = datamodule, n, seed, max_old

    def _setup(self):
        ds = self.dm.full
        val = np.asarray(self.dm.val.indices)
        with __import__("h5py").File(self.dm.hparams.processed_h5, "r") as h:
            key = np.asarray(h["pair_key"][...]) if "pair_key" in h else np.full(len(ds), -1)
            ref = np.asarray(h["pair_ref"][...]) if "pair_ref" in h else np.zeros(len(ds), bool)
        camp = ds.row_campaign
        ok = (ds.drive_density.numpy() & ds.witness_density.numpy())
        pairs, by = [], {}
        for i in val:
            if key[i] >= 0 and ok[i]:
                by.setdefault(int(key[i]), {})["ref" if ref[i] else "pt"] = i     # keys are globally unique
        for d in by.values():
            if len(d) == 2:
                pairs.append((d["pt"], d["ref"]))
        paired_camps = {camp[i] for i in np.flatnonzero(key >= 0)}
        old = [i for i in val if camp[i] not in paired_camps and ok[i]][: self.max_old]
        self.pairs, self.old = pairs, old

    @torch.no_grad()
    def _true_spacing(self, model, rows):
        ds = self.dm.full
        if getattr(ds, "has_truth", False):                 # full-particle truth (2026-10-21), table subsample as fallback
            t = ds.truth_spacing[rows].double().numpy()
            if np.isfinite(t).all():
                return t
        dm_, dsd = model.drive_mean.cpu(), model.drive_std.cpu()
        wm_, wsd = model.witness_mean.cpu(), model.witness_std.cpu()
        d = ds.drive[rows] * dsd + dm_
        w = ds.witness[rows] * wsd + wm_
        return inter_bunch(d.double(), w.double())["bunch_spacing"].numpy()

    @torch.no_grad()
    def _pred_spacing(self, model, rows):
        g = torch.Generator().manual_seed(self.seed)
        zd = torch.randn(self.n, model.latent_dim, generator=g).to(model.device)
        zw = torch.randn(self.n, model.latent_dim, generator=g).to(model.device)
        kn = self.dm.full.knobs[rows].to(model.device)
        out = []
        for b in range(0, len(rows), 64):
            o = model.observables(kn[b:b + 64], n=self.n, z_drive=zd, z_witness=zw)
            out.append(o["bunch_spacing"].double().cpu())
        return torch.cat(out).numpy()

    def on_validation_epoch_end(self, trainer, model):
        if trainer.sanity_checking:
            return
        if not hasattr(self, "pairs"):
            self._setup()
        was = model.training
        model.eval()
        if self.pairs:
            pt, rf = [p[0] for p in self.pairs], [p[1] for p in self.pairs]
            dt = self._true_spacing(model, pt) - self._true_spacing(model, rf)
            dp = self._pred_spacing(model, pt) - self._pred_spacing(model, rf)
            model.log("val_paired_delta_rms_um", float(1e6 * np.sqrt(np.mean((dp - dt) ** 2))))
            model.log("val_paired_delta_signal_rms_um", float(1e6 * np.sqrt(np.mean(dt ** 2))))
        if self.old:
            e = self._pred_spacing(model, self.old) - self._true_spacing(model, self.old)
            model.log("val_old_spacing_mae_um", float(1e6 * np.mean(np.abs(e))))
        model.train(was)
