"""eval.evaluate runs end to end on a tiny table and accepts an already-loaded model."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("matplotlib")
pytest.importorskip("sklearn")
h5py = pytest.importorskip("h5py")

from twobunch_cnf.eval import evaluate  # noqa: E402
from twobunch_cnf.model import TwoBunchFlow  # noqa: E402

KEYS, LO, HI = ["a", "b"], [0.0, -1.0], [1.0, 1.0]


def _table(tmp_path, n=30, p=16):
    rng = np.random.default_rng(0)
    h5 = tmp_path / "t.h5"
    with h5py.File(h5, "w") as h:
        h["knobs"] = rng.uniform(LO, HI, size=(n, 2)).astype(np.float32)
        h["drive_parts"] = rng.normal(size=(n, p, 6)).astype(np.float32)
        h["witness_parts"] = rng.normal(size=(n, p, 6)).astype(np.float32)
        for k in ("drive_present", "witness_viable", "drive_density", "witness_density"):
            h[k] = np.ones(n, bool)
        h["drive_frac"] = np.ones(n, np.float32)
        h["witness_frac"] = np.ones(n, np.float32)
    norm = {
        "knob_keys": KEYS,
        "knob_low": LO,
        "knob_high": HI,
        "drive_mean": [0.0] * 6,
        "drive_std": [1.0] * 6,
        "witness_mean": [0.0] * 6,
        "witness_std": [1.0] * 6,
    }
    (tmp_path / "t_norm.json").write_text(json.dumps(norm))
    return str(h5)


def test_evaluate_accepts_a_loaded_model(tmp_path):
    torch.manual_seed(0)
    model = TwoBunchFlow(
        condition_dim=2, n_layers=2, hidden_dim=16, knob_keys=KEYS, knob_low=LO, knob_high=HI, treaty_point="PENT"
    )
    out = tmp_path / "eval"
    metrics = evaluate(model, _table(tmp_path), out, n=32, device="cpu")
    assert {"bunch_spacing", "witness_viability_auc", "T_drive_mae"} <= set(metrics)
    assert (out / "parity.png").exists() and (out / "metrics.json").exists()
