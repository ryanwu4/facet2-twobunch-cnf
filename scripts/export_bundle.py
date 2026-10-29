"""Export a twobunch-s2e-rl TwoBunchFlow checkpoint as a model bundle (src/twobunch_cnf/resources/<name>/).

  PYTHONPATH=<twobunch-s2e-rl>/src pixi run -e test python scripts/export_bundle.py \
      --ckpt <twobunch>/trained/<run>/checkpoints/final_ema.ckpt \
      --segment-spec <twobunch>/configs/segment/segment_baseline.json \
      --name c_b3_ema --out src/twobunch_cnf/resources/c_b3_ema \
      [--clock-beams '<twobunch>/data/<campaign>/sample_*_XTCAVF.h5' ...]

Writes model.ckpt (migrated + slim: state_dict, hyper_parameters, Lightning version, migration record),
norm.json and run.json (copied from the training run directory) and bundle.yaml (knob table with bounds
from the checkpoint, baseline from the norm's sweep set, units / descriptions from configs/knob_meta.yaml;
handoff reference values from the segment spec; provenance). twobunch_s2e_rl must be importable for the
baseline (or pass --baseline-json).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from twobunch_cnf import ELECTRON_MC2_EV, __version__
from twobunch_cnf.bundle import sha256
from twobunch_cnf.cnf import reference_time_upstream
from twobunch_cnf.migrate import migrate_checkpoint
from twobunch_cnf.model import load_flow

REPO = Path(__file__).resolve().parents[1]
SLIM_KEYS = ("state_dict", "hyper_parameters", "pytorch-lightning_version", "twobunch_cnf_migration")


def _git(*args, cwd):
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def baseline_of(norm: dict, baseline_json: str | None) -> dict:
    if baseline_json:
        return json.loads(Path(baseline_json).read_text())
    from twobunch_s2e_rl.datagen.sweep_params import resolve_sweep_set

    return dict(resolve_sweep_set(norm["sweep_set"])[3])


def clock_offset(globs: list[str], ref_time_s: float, per_glob: int) -> dict | None:
    """Charge-weighted mean arrival minus the lattice reference time, over S2E beam files at the plane."""
    if not globs:
        return None
    from beamphysics import ParticleGroup

    files = []
    for g in globs:
        fs = sorted(glob.glob(g))
        files += fs[:: max(1, len(fs) // per_glob)][:per_glob]
    d = np.array([np.average(P.t, weights=P.weight) - ref_time_s for P in map(ParticleGroup, files)])
    p5, p95 = np.percentile(d, [5, 95])
    return {"median_s": float(np.median(d)), "p5_s": float(p5), "p95_s": float(p95), "n_files": len(files),
            "note": "S2E saved beams' charge-weighted mean arrival minus the lattice reference time at the "
                    "trained plane: a constant clock offset of the input beam (the knob-dependent part is the "
                    "p5-p95 spread). Exports use the lattice reference time (bunch mean on the reference particle)."}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run-dir", default=None, help="training run dir with norm.json + run.json (default: ckpt/../..)")
    ap.add_argument("--segment-spec", required=True, help="xtcav_segment spec json (meta: start plane s / ref time / E)")
    ap.add_argument("--knob-meta", default=str(REPO / "configs" / "knob_meta.yaml"))
    ap.add_argument("--baseline-json", default=None, help="{knob: value}; default: the norm's sweep set baseline")
    ap.add_argument("--clock-beams", nargs="*", default=[], help="globs of S2E beam files at the trained plane")
    ap.add_argument("--clock-per-glob", type=int, default=30)
    ap.add_argument("--name", required=True)
    ap.add_argument("--description", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src = Path(args.ckpt).resolve()
    run_dir = Path(args.run_dir).resolve() if args.run_dir else src.parent.parent
    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"{out} is not empty (pass --overwrite)")
    out.mkdir(parents=True, exist_ok=True)

    # checkpoint: migrate + slim. The source is recorded relative to its repo (bundles are published).
    src_repo = _git("rev-parse", "--show-toplevel", cwd=src.parent)
    src_rel = str(src.relative_to(src_repo)) if src_repo else src.name
    ck = torch.load(src, map_location="cpu", weights_only=True)
    mig = migrate_checkpoint(ck, source=src_rel)
    torch.save({k: mig[k] for k in SLIM_KEYS if k in mig}, out / "model.ckpt")
    flow = load_flow(out / "model.ckpt")
    hp = flow.hparams
    for f in ("norm.json", "run.json"):
        shutil.copyfile(run_dir / f, out / f)
    norm = json.loads((out / "norm.json").read_text())
    run = json.loads((out / "run.json").read_text())

    # knob table
    keys, lo, hi = list(hp.knob_keys), flow.knob_low.tolist(), flow.knob_high.tolist()
    if keys != list(norm["knob_keys"]) or not np.allclose(lo, norm["knob_low"], rtol=1e-6, atol=0):
        raise SystemExit("checkpoint frame differs from the run's norm.json")
    base = baseline_of(norm, args.baseline_json)
    meta = yaml.safe_load(Path(args.knob_meta).read_text())
    missing = [k for k in keys if k not in base or k not in meta]
    if missing:
        raise SystemExit(f"no baseline / metadata for {missing}")
    knobs = []
    for k, a, b in zip(keys, lo, hi):
        if not (a - 1e-9 * abs(b - a) <= base[k] <= b + 1e-9 * abs(b - a)):
            raise SystemExit(f"baseline {k}={base[k]} outside the training box [{a}, {b}]")
        knobs.append({"name": k, "low": float(a), "high": float(b), "baseline": float(base[k]), **meta[k]})

    # handoff reference values (the spec's start plane = the element EXIT, Bmad's saved-beam convention)
    spec = json.loads(Path(args.segment_spec).read_text())["meta"]
    if spec["start_name"] != hp.treaty_point:
        raise SystemExit(f"segment spec starts at {spec['start_name']}, checkpoint treaty point is {hp.treaty_point}")
    e_tot, length = float(spec["ref_energy_eV"]), float(spec["tdc_length_m"])
    s_exit, t_exit = float(spec["start_s_m"]), float(spec["start_ref_time_s"])
    handoff = {
        "element": spec["start_name"],
        "trained_plane": "exit",
        "species": "electron",
        "element_length_m": length,
        "e_tot_eV": e_tot,
        "p0c_eV": math.sqrt(e_tot**2 - ELECTRON_MC2_EV**2),
        "planes": {
            "exit": {"s_m": s_exit, "ref_time_s": t_exit},
            "entrance": {"s_m": s_exit - length, "ref_time_s": reference_time_upstream(t_exit, length, e_tot)},
        },
        "source": {"segment_spec": Path(args.segment_spec).name, "lattice": "FACET2-S2E (via PyTao)",
                   "setlattice_config": spec.get("config_file"), "optics": spec.get("optics")},
        "s2e_clock_offset": clock_offset(args.clock_beams, t_exit, args.clock_per_glob),
        "note": "Reference values are FACET2-S2E's. The facet2-lattice (virtual-accelerator) values at this "
                "element are unverified (its HEAD needs a newer Bmad than 20250811), and there XTCAVF is an "
                "lcavity, not a crab_cavity.",
    }

    bundle = {
        "name": args.name,
        "description": args.description,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "package_version": __version__,
        "treaty_point": hp.treaty_point,
        "checkpoint": {"file": "model.ckpt", "sha256": sha256(out / "model.ckpt"), "slim": True,
                       "source": src_rel, "source_sha256": sha256(src)},
        "norm": "norm.json",
        "training_run": "run.json",
        "provenance": {"source_repo": Path(src_repo).name if src_repo else None,
                       "source_repo_hash_at_export": _git("rev-parse", "HEAD", cwd=src.parent),
                       "training_git_hash": run.get("git_hash"), "training_parents": run.get("parents")},
        "knobs": knobs,
        "handoff": handoff,
    }
    (out / "bundle.yaml").write_text(yaml.safe_dump(bundle, sort_keys=False, width=110))
    print(f"wrote {out}: model.ckpt ({(out / 'model.ckpt').stat().st_size / 1e6:.2f} MB), norm.json, run.json, bundle.yaml")


if __name__ == "__main__":
    main()
