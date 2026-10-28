"""Plot training/val loss curves from a CSVLogger metrics.csv (surrogate training diagnostics).

Shows total loss + each component (NLL, cov, emit, feasibility) train-vs-val over epochs, with
the best-val epoch marked and an end-slope annotation, so it's obvious whether training has
converged or is still descending (i.e. whether more epochs would help).

Usage: python -m twobunch_cnf.plot_training <run_dir> --out <loss_curves.png>
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GREEN = "#4c72b0", "#dd8452", "#55a868"

# (val_key, train_key, title, log?)
PANELS = [
    ("val_loss", "train_loss", "total loss", True),
    ("val_nll_witness", "train_nll_witness", "NLL witness (bottleneck)", False),
    ("val_nll_drive", "train_nll_drive", "NLL drive", False),
    ("val_cov_witness", "train_cov_witness", "cov witness", True),
    ("val_emit_witness", "train_emit_witness", "emit witness", True),
    ("val_bce_surv", "train_bce_surv", "feasibility BCE", True),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="training run directory (contains csv/version_*/metrics.csv)")
    ap.add_argument("--out", required=True, help="output figure path (e.g. <dir>/loss_curves.png)")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    mc = sorted(glob.glob(str(run_dir / "csv" / "version_*" / "metrics.csv")))
    if not mc:
        raise SystemExit(f"no metrics.csv under {run_dir}/csv/version_*/")
    df = pd.read_csv(mc[-1])
    tr = df.dropna(subset=["train_loss"]).groupby("epoch").last()
    va = df.dropna(subset=["val_loss"]).groupby("epoch").last()
    best_ep = int(va["val_loss"].idxmin())
    best = float(va["val_loss"].min())

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for ax, (vk, tk, title, log) in zip(axes.ravel(), PANELS):
        if tk in tr:
            ax.plot(tr.index, tr[tk], color=BLUE, lw=1.3, alpha=0.8, label="train")
        if vk in va:
            ax.plot(va.index, va[vk], color=ORANGE, lw=1.6, label="val")
        ax.axvline(best_ep, color=GREEN, ls="--", lw=1.2, alpha=0.7)
        if log:
            # shift so log works if values dip <=0 (NLL can be negative -> keep linear via guard)
            vals = np.concatenate([va[vk].dropna().values]) if vk in va else np.array([1.0])
            if (vals > 0).all():
                ax.set_yscale("log")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8)
    # end-slope annotation on total loss
    vl = va["val_loss"].values
    slope = (vl[-20] - vl[-1]) / 19 if len(vl) >= 21 else float("nan")
    axes.ravel()[0].annotate(
        f"best val {best:.3f} @ ep{best_ep}\nΔ/epoch (last 20): {slope:+.4f}",
        xy=(0.97, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", fc="white", alpha=0.8),
    )
    fig.suptitle(
        f"Surrogate training curves — {args.run_dir}  (green = best-val epoch {best_ep}/{int(va.index.max())})",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"best val_loss {best:.4f} at epoch {best_ep} / {int(va.index.max())}")
    print(f"Δval_loss/epoch over last 20: {slope:+.5f}  (negative = still improving)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
