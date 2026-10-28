"""KnobSpace frame maps + artifact helpers (resolve_ckpt, run.json, a checkpoint's knob frame)."""

import json

import numpy as np
import pytest

from twobunch_cnf.artifacts import read_run_json, resolve_ckpt, write_run_json
from twobunch_cnf.knobs import KnobSpace

KS = KnobSpace.from_lists(["a", "b", "c"], [0.0, -10.0, 5.0], [1.0, 10.0, 5.0], {"a": 0.5})


def test_normalize_roundtrip_numpy():
    x = np.array([[0.25, 0.0, 5.0], [1.0, -10.0, 5.0]])
    np.testing.assert_allclose(KS.denormalize(KS.normalize(x)), x)
    np.testing.assert_allclose(KS.normalize(x)[0, :2], [0.25, 0.5])


def test_normalize_roundtrip_torch_and_degenerate_knob():
    torch = pytest.importorskip("torch")
    x = torch.tensor([[0.5, 5.0, 5.0]], dtype=torch.float64)
    n = KS.normalize(x)
    assert torch.is_tensor(n) and float(n[0, 2]) == 0.0  # fixed knob -> span 1, value 0
    assert torch.allclose(KS.denormalize(n), x)


def test_vector_fill_and_missing():
    np.testing.assert_allclose(KS.vector({"a": 0.1, "b": 2.0}, fill={"c": 5.0}), [0.1, 2.0, 5.0])
    with pytest.raises(KeyError):
        KS.vector({"a": 0.1})


def test_subset_contains_dict_roundtrip():
    sub = KS.subset(["c", "a"])
    assert sub.keys == ("c", "a") and sub.baseline == {"a": 0.5}
    assert KS.contains(np.array([0.5, 0.0, 5.0])) and not KS.contains(np.array([1.5, 0.0, 5.0]))
    assert KnobSpace.from_dict(json.loads(json.dumps(KS.to_dict()))) == KS


def test_invalid_spaces_rejected():
    with pytest.raises(ValueError):
        KnobSpace.from_lists(["a", "a"], [0, 0], [1, 1])
    with pytest.raises(ValueError):
        KnobSpace.from_lists(["a"], [0, 0], [1])


def _touch(d, names):
    for n in names:
        (d / n).write_text("")


def test_resolve_ckpt_numeric_and_negative_val_loss(tmp_path):
    _touch(
        tmp_path,
        [
            "best-epoch=99-val_loss=0.9.ckpt",
            "best-epoch=493-val_loss=0.5126.ckpt",
            "best-epoch=144-val_loss=-1.3655.ckpt",
        ],
    )
    assert resolve_ckpt(str(tmp_path / "best-*.ckpt")).endswith("val_loss=-1.3655.ckpt")


def test_resolve_ckpt_reward_ranking_and_errors(tmp_path):
    _touch(tmp_path, ["Xpolicy_iter100_reward157.6.pt", "Xpolicy_iter200_reward181.5.pt"])
    assert resolve_ckpt(str(tmp_path / "X*.pt")).endswith("reward181.5.pt")
    with pytest.raises(FileNotFoundError):
        resolve_ckpt(str(tmp_path / "nothing*.ckpt"))
    _touch(tmp_path, ["a.ckpt", "b.ckpt"])
    with pytest.raises(ValueError):
        resolve_ckpt(str(tmp_path / "*.ckpt"))
    assert resolve_ckpt("plain/path.ckpt") == "plain/path.ckpt"


def test_resolve_ckpt_tie_is_an_error(tmp_path):
    _touch(tmp_path, ["e1-val_loss=0.5.ckpt", "e2-val_loss=0.5.ckpt"])
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_ckpt(str(tmp_path / "*.ckpt"))


def test_run_json_roundtrip_and_no_overwrite(tmp_path):
    p = write_run_json(tmp_path, "test", {"x": 1}, parents=["a.h5"], flow_ckpt="f.ckpt")
    rec = read_run_json(tmp_path)
    assert rec["kind"] == "test" and rec["config"] == {"x": 1} and rec["flow_ckpt"] == "f.ckpt"
    assert "git_hash" in rec and rec["parents"] == ["a.h5"] and rec["package_version"]
    with pytest.raises(FileExistsError):
        write_run_json(tmp_path, "test", {})
    write_run_json(tmp_path, "test2", {}, overwrite=True)
    assert read_run_json(p)["kind"] == "test2"


def test_flow_knob_space_needs_recorded_keys_and_treaty_point():
    pytest.importorskip("torch")
    from twobunch_cnf.artifacts import flow_knob_space, flow_treaty_point
    from twobunch_cnf.model import TwoBunchFlow

    keys, lo, hi = ["a", "b", "c"], [0.0, -1.0, 2.0], [1.0, 1.0, 3.0]
    flow = TwoBunchFlow(
        condition_dim=3, n_layers=2, hidden_dim=16, knob_low=lo, knob_high=hi, knob_keys=keys, treaty_point="XTCAVF"
    )
    ks = flow_knob_space(flow)
    assert list(ks.keys) == keys and flow_treaty_point(flow) == "XTCAVF"
    np.testing.assert_allclose(ks.low, lo, rtol=1e-6)
    bare = TwoBunchFlow(condition_dim=3, n_layers=2, hidden_dim=16, knob_low=lo, knob_high=hi)
    with pytest.raises(ValueError, match="knob_keys"):
        flow_knob_space(bare)
    with pytest.raises(ValueError, match="treaty_point"):
        flow_treaty_point(bare)
