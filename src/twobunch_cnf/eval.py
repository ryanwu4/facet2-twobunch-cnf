"""Evaluate a trained two-bunch flow: per-bunch/inter-bunch parity, feasibility, overlays.

  python -m twobunch_cnf.eval --ckpt '<run_dir>/checkpoints/best-*.ckpt' --processed <table>.h5 --out <dir>

Writes <--out>/{parity.png, phase_space.png, metrics.json}. --processed and --out are required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .dataset import TwoBunchFlowDataModule
from .model import load_flow
from .properties import inter_bunch, per_bunch
from .artifacts import resolve_ckpt

PARITY_KEYS = [
    ("drive_norm_emit_x", "drive ε_n,x [m]", True),
    ("witness_norm_emit_x", "witness ε_n,x [m]", True),
    ("witness_norm_emit_y", "witness ε_n,y [m]", True),
    ("drive_sigma_z", "drive σ_z [m]", True),
    ("bunch_spacing", "bunch spacing [m]", False),
    ("energy_difference", "ΔE (D−W) [eV]", False),
    ("transverse_offset", "transverse offset [m]", True),
    ("witness_energy_spread", "witness σ_E [eV]", True),
    # matching objectives (BMAG / slice-β) -- now MBRL reward terms, so gated by the R² check
    ("witness_bmag_x", "witness BMAG_x", True),
    ("witness_bmag_y", "witness BMAG_y", True),
    ("witness_slice_bmag_max", "witness slice BMAG max", True),
    ("witness_slice_beta_y_core", "witness slice β_y core [m]", True),
    ("drive_bmag_x", "drive BMAG_x", True),
]


def _destd(parts_std, mean, std):
    return parts_std * std + mean


@torch.no_grad()
def evaluate(ckpt, processed, out_dir, n=2048, device="cuda"):
    device = device if torch.cuda.is_available() else "cpu"
    model = load_flow(ckpt, map_location=device).to(device)
    dm = TwoBunchFlowDataModule(processed_h5=processed, batch_size=256, num_workers=0)
    dm.setup()
    val = dm.val
    dm_mean = {0: (model.drive_mean, model.drive_std), 1: (model.witness_mean, model.witness_std)}

    # gather val tensors
    knobs = torch.stack([val[i]["knobs"] for i in range(len(val))]).to(device)
    flags = {k: torch.stack([val[i][k] for i in range(len(val))])
             for k in ("drive_density", "witness_density", "witness_viable", "drive_present")}
    fracs = {k: torch.stack([val[i][k] for i in range(len(val))])
             for k in ("drive_frac", "witness_frac")}
    true_drive = torch.stack([val[i]["drive"] for i in range(len(val))]).to(device)
    true_witness = torch.stack([val[i]["witness"] for i in range(len(val))]).to(device)

    # surrogate observables
    pred = model.observables(knobs, n=n)

    # true observables (destandardized real clouds)
    td = _destd(true_drive, *dm_mean[0])
    tw = _destd(true_witness, *dm_mean[1])
    true = {}
    for nm, parts in (("drive", td), ("witness", tw)):
        for kk, vv in per_bunch(parts).items():
            true[f"{nm}_{kk}"] = vv
    true.update(inter_bunch(td, tw))

    dd = flags["drive_density"].cpu().numpy().astype(bool)
    wd = flags["witness_density"].cpu().numpy().astype(bool)
    both = dd & wd

    # ---- parity ----
    metrics = {}
    ncol = 4
    nrow = int(np.ceil(len(PARITY_KEYS) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 4.5 * nrow))
    for ax, (key, label, logp) in zip(axes.ravel(), PARITY_KEYS):
        bunch = key.split("_")[0]
        mask = wd if bunch == "witness" else dd
        if any(k in key for k in ("spacing", "difference", "offset")):
            mask = both
        p = pred[key].cpu().numpy()[mask]
        t = true[key].cpu().numpy()[mask]
        if logp:
            p, t = np.abs(p) + 1e-30, np.abs(t) + 1e-30
        ax.scatter(t, p, s=4, alpha=0.3)
        lo, hi = np.percentile(np.concatenate([t, p]), [1, 99])
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        if logp:
            ax.set_xscale("log"); ax.set_yscale("log")
        # robust R^2 in (log) space
        tt = np.log10(t) if logp else t
        pp = np.log10(p) if logp else p
        ss_res = np.sum((tt - pp) ** 2); ss_tot = np.sum((tt - tt.mean()) ** 2) + 1e-30
        r2 = 1 - ss_res / ss_tot
        metrics[key] = {"r2": float(r2), "n": int(mask.sum())}
        ax.set_title(f"{label}\nR²={r2:.2f} (n={mask.sum()})", fontsize=10)
        ax.set_xlabel("true"); ax.set_ylabel("surrogate")
    for ax in axes.ravel()[len(PARITY_KEYS):]:
        ax.axis("off")
    fig.suptitle("Surrogate vs truth (val split) — per-bunch & inter-bunch observables", fontsize=13)
    fig.tight_layout()
    out_dir = Path(out_dir)
    (out_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "parity.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- feasibility ----
    wv = flags["witness_viable"].cpu().numpy().astype(int)
    p_surv = pred["p_surv"].cpu().numpy()
    metrics["witness_viability_auc"] = float(roc_auc_score(wv, p_surv)) if wv.min() != wv.max() else None
    metrics["T_drive_mae"] = float(np.abs(pred["T_drive"].cpu().numpy()[dd]
                                          - fracs["drive_frac"].numpy()[dd]).mean())
    wvb = flags["witness_viable"].cpu().numpy().astype(bool)
    metrics["T_witness_mae"] = float(np.abs(pred["T_witness"].cpu().numpy()[wvb]
                                            - fracs["witness_frac"].numpy()[wvb]).mean())

    # ---- phase-space overlay (first both-density val sample) ----
    j = int(np.argmax(both))
    s_d = model.sample_bunch(knobs[j:j + 1], 0, n)[0].cpu().numpy()
    s_w = model.sample_bunch(knobs[j:j + 1], 1, n)[0].cpu().numpy()
    rd, rw = td[j].cpu().numpy(), tw[j].cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (a, b, la, lb) in zip(axes, [(0, 2, "x [m]", "z [m]"), (0, 3, "x [m]", "px [eV/c]")]):
        ax.scatter(rd[:, a], rd[:, b], s=3, alpha=0.3, color="#1f77b4", label="drive true")
        ax.scatter(s_d[:, a], s_d[:, b], s=3, alpha=0.3, color="#0b3d66", marker="x", label="drive flow")
        ax.scatter(rw[:, a], rw[:, b], s=3, alpha=0.3, color="#d62728", label="witness true")
        ax.scatter(s_w[:, a], s_w[:, b], s=3, alpha=0.3, color="#7a1518", marker="x", label="witness flow")
        ax.set_xlabel(la); ax.set_ylabel(lb)
    axes[0].legend(fontsize=8, markerscale=2)
    fig.suptitle(f"Phase-space overlay — val sample idx {j}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "phase_space.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {out_dir}/parity.png, phase_space.png, metrics.json")
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="checkpoint path (glob ok)")
    ap.add_argument("--processed", required=True, help="processed table (.h5) the flow was trained on")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--n", type=int, default=2048)
    args = ap.parse_args()
    ckpt = resolve_ckpt(args.ckpt)
    evaluate(ckpt, args.processed, args.out, n=args.n)


if __name__ == "__main__":
    main()
