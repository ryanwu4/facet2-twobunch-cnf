"""Convert a twobunch-s2e-rl TwoBunchFlow checkpoint to this package's (pruned) model, once.

The pruned model no longer accepts the options that were removed from main (they live on
archive/unpruned-7ac3617 and experiment branches off it); TwoBunchFlow.on_load_checkpoint refuses checkpoints
that carry them (Lightning would otherwise drop unknown hparams silently). `migrate`:

  1. requires the checkpoint to record its frame (`knob_keys`, `treaty_point`);
  2. checks every removed option is OFF, i.e. equals the value the pruned model implies. In a twobunch-s2e-rl
     checkpoint an option absent from it takes the default it had before the prune (it predates the option);
     a checkpoint written by this package (`twobunch_cnf_version` / `twobunch_cnf_migration` marker, or none of
     the removed hparams at all) simply has no such options;
  3. refuses any other hparam the pruned model does not know (e.g. from an experiment branch);
  4. drops the removed keys, loads the weights STRICTLY into the pruned model, and writes a new file
     (never in place) with a `twobunch_cnf_migration` record.

  python -m twobunch_cnf.migrate SRC.ckpt --out DST.ckpt
"""

from __future__ import annotations

import argparse
import inspect
import time
from pathlib import Path

import torch

from . import __version__
from .model import TwoBunchFlow

# Options removed from TwoBunchFlow on main: name -> (value the pruned model implies, default on
# archive/unpruned-7ac3617). Generated from inspect.signature(TwoBunchFlow.__init__) there; the implied
# value is the old default (option off) except for coupling / t_two_part, whose kept path was opt-in.
REMOVED = {
    "coupling": ("rqs", "affine"),
    "t_two_part": (True, False),
    "centroid_head": (False, False),
    "w_centroid": (0.0, 0.0),
    "centroid_buffer": (512, 512),
    "centroid_head_lr": (None, None),
    "centroid_dims": (None, None),
    "centroid_param": ("per_bunch", "per_bunch"),
    "centroid_act": ("relu", "relu"),
    "curve_head": (False, False),
    "curve_dims": ((2, 0), (2, 0)),
    "curve_order": (8, 8),
    "curve_scale": (3.0, 3.0),
    "curve_head_lr": (None, None),
    "w_curve": (0.0, 0.0),
    "backbone": (False, False),
    "backbone_bins": (32, 32),
    "backbone_pca": (16, 16),
    "backbone_dims": ((2, 0), (2, 0)),
    "w_backbone_prior": (1.0, 1.0),
    "backbone_lr": (None, None),
    "backbone_shear": ("pca", "pca"),
    "bunches": ((0, 1), (0, 1)),
}
REQUIRED = ("knob_keys", "treaty_point")


def _norm(v):
    return list(v) if isinstance(v, (list, tuple)) else v


def migrate_checkpoint(ckpt: dict, source: str = "") -> dict:
    """The converted checkpoint dict (input untouched). Raises ValueError naming what blocks it."""
    hp = dict(ckpt["hyper_parameters"])
    missing = [k for k in REQUIRED if not hp.get(k)]
    if missing:
        raise ValueError(
            f"{source}: checkpoint records no {missing}; only checkpoints that record their "
            "knob frame and treaty point are supported"
        )
    native = "twobunch_cnf_version" in ckpt or "twobunch_cnf_migration" in ckpt or not set(hp) & set(REMOVED)
    used = {
        k: hp.get(k, old)
        for k, (want, old) in REMOVED.items()
        if (k in hp or not native) and _norm(hp.get(k, old)) != _norm(want)
    }
    if used:
        raise ValueError(
            f"{source}: uses option(s) removed from main {used}; load it from "
            "archive/unpruned-7ac3617 or an experiment branch off it"
        )
    known = set(inspect.signature(TwoBunchFlow.__init__).parameters) - {"self"}
    unknown = sorted(set(hp) - known - set(REMOVED))
    if unknown:
        raise ValueError(f"{source}: unknown hparam(s) {unknown} (from an experiment branch?); not converted")
    new_hp = {k: v for k, v in hp.items() if k in known}
    with torch.random.fork_rng(devices=[]):  # the strict check's random init must not move the caller's RNG
        TwoBunchFlow(**new_hp).load_state_dict(ckpt["state_dict"], strict=True)
    out = dict(ckpt)
    out["hyper_parameters"] = new_hp
    out["twobunch_cnf_migration"] = {
        "source": str(source),
        "removed": sorted(k for k in hp if k in REMOVED),
        "package_version": __version__,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    return out


def migrate_file(src, dst, overwrite: bool = False) -> Path:
    src, dst = Path(src), Path(dst)
    if dst.resolve() == src.resolve():
        raise ValueError("refusing to migrate in place; pass a different --out")
    if dst.exists() and not overwrite:
        raise FileExistsError(f"{dst} exists (pass --overwrite to replace)")
    ckpt = torch.load(src, map_location="cpu", weights_only=True)
    out = migrate_checkpoint(ckpt, source=str(src))
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="twobunch-s2e-rl TwoBunchFlow checkpoint (.ckpt)")
    ap.add_argument("--out", required=True, help="converted checkpoint (.ckpt); never the source")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing --out")
    args = ap.parse_args()
    print(f"wrote {migrate_file(args.src, args.out, args.overwrite)}")


if __name__ == "__main__":
    main()
