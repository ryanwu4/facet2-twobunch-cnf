"""Write tests/golden/<name>_probe.npz from the ORIGINAL twobunch-s2e-rl checkpoint of a shipped bundle.

  PYTHONPATH=<twobunch-s2e-rl>/src pixi run -e test python scripts/make_golden.py \
      --ckpt <twobunch>/trained/<run>/checkpoints/final_ema.ckpt --name c_b3_ema

The probe (tests/_golden_probe.py) runs on twobunch-s2e-rl's own TwoBunchFlow, so tests/test_bundle.py
checks the shipped bundle against the source model, not against itself.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))

from _golden_probe import probe  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="original twobunch-s2e-rl checkpoint")
    ap.add_argument("--name", required=True, help="bundle name (output tests/golden/<name>_probe.npz)")
    args = ap.parse_args()
    from twobunch_s2e_rl.surrogate.model import TwoBunchFlow

    flow = TwoBunchFlow.load_from_checkpoint(args.ckpt, map_location="cpu").eval()
    out = REPO / "tests" / "golden" / f"{args.name}_probe.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **probe(flow))
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
