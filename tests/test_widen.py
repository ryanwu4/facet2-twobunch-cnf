"""widen_conditioning: the widened flow reproduces the old one exactly, for any new-knob value."""
import pytest

torch = pytest.importorskip("torch")
from twobunch_cnf.knobs import KnobSpace  # noqa: E402
from twobunch_cnf.model import TwoBunchFlow  # noqa: E402
from twobunch_cnf.widen import widen_conditioning  # noqa: E402

OLD = KnobSpace.from_lists(["a", "b", "c"], [0.0, -1.0, 10.0], [1.0, 1.0, 20.0])
# 'b' widened (shared knob, bounds change), 'c' moved; two new knobs, one inserted in the middle
NEW = KnobSpace.from_lists(["a", "new1", "b", "c", "new2"],
                           [0.0, 5.0, -3.0, 10.0, 0.0], [1.0, 6.0, 2.0, 30.0, 1.0])


def _flow():
    torch.manual_seed(0)
    return TwoBunchFlow(condition_dim=3, n_layers=2, hidden_dim=16,
                        knob_low=list(OLD.low), knob_high=list(OLD.high), knob_keys=list(OLD.keys)).eval()


def _obs(flow, knobs_norm, z):
    with torch.no_grad():
        return flow.observables(knobs_norm, n=64, z_drive=z[0], z_witness=z[1])


@pytest.mark.parametrize("new_vals", [(5.0, 0.0), (6.0, 1.0), (5.3, 0.7)])
def test_widened_reproduces_old_at_any_new_knob_value(new_vals):
    old_flow = _flow()
    wide = widen_conditioning(old_flow, NEW)
    g = torch.Generator().manual_seed(1)
    phys = OLD.denormalize(torch.rand(5, 3, generator=g, dtype=torch.float64))
    # include a shared knob near its OLD edge
    phys[0, 1] = OLD.high[1] - 1e-3
    phys_new = torch.stack([phys[:, 0], torch.full((5,), new_vals[0], dtype=torch.float64), phys[:, 1],
                            phys[:, 2], torch.full((5,), new_vals[1], dtype=torch.float64)], -1)
    z = (torch.randn(64, 6, generator=g), torch.randn(64, 6, generator=g))
    a = _obs(old_flow, OLD.normalize(phys).float(), z)
    b = _obs(wide, NEW.normalize(phys_new).float(), z)
    for k in ("bunch_spacing", "drive_norm_emit_x", "witness_sigma_z", "T_drive", "p_surv"):
        torch.testing.assert_close(b[k], a[k], rtol=2e-4, atol=1e-9)


def test_widened_beam_latent_model_reproduces_old():
    """Every knob-input layer is remapped: encoder, feasibility + two-part T heads, latent prior."""
    torch.manual_seed(0)
    old_flow = TwoBunchFlow(condition_dim=3, n_layers=2, hidden_dim=16, beam_latent_dim=4,
                            knob_low=list(OLD.low), knob_high=list(OLD.high), knob_keys=list(OLD.keys)).eval()
    with torch.no_grad():                         # a non-trivial (non-zero-init) latent prior and input map
        for k in range(2):
            old_flow.latent_prior[k][-1].weight.normal_(0, 0.3)
            old_flow.latent_in[k].weight.normal_(0, 0.3)
    wide = widen_conditioning(old_flow, NEW)
    g = torch.Generator().manual_seed(2)
    phys = OLD.denormalize(torch.rand(4, 3, generator=g, dtype=torch.float64))
    phys_new = torch.stack([phys[:, 0], torch.full((4,), 5.5, dtype=torch.float64), phys[:, 1],
                            phys[:, 2], torch.full((4,), 0.3, dtype=torch.float64)], -1)
    z = (torch.randn(64, 6, generator=g), torch.randn(64, 6, generator=g))
    with torch.no_grad():
        a = old_flow.observables(OLD.normalize(phys).float(), n=64, z_drive=z[0], z_witness=z[1],
                                 generator=torch.Generator().manual_seed(9))
        b = wide.observables(NEW.normalize(phys_new).float(), n=64, z_drive=z[0], z_witness=z[1],
                             generator=torch.Generator().manual_seed(9))
    for k in ("bunch_spacing", "drive_norm_emit_x", "witness_sigma_z", "T_drive", "T_witness", "p_surv"):
        torch.testing.assert_close(b[k], a[k], rtol=2e-4, atol=1e-9)


def test_widened_hparams_and_frame():
    wide = widen_conditioning(_flow(), NEW)
    assert wide.hparams.condition_dim == 5 and list(wide.hparams.knob_keys) == list(NEW.keys)
    torch.testing.assert_close(wide.knob_low, torch.tensor(NEW.low))


def test_dropping_a_knob_is_an_error():
    with pytest.raises(ValueError, match="drops"):
        widen_conditioning(_flow(), OLD.subset(["a", "b"]))


def _processed(tmp_path, space, drive_mean, n=6, P=8):
    """Minimal processed table + norm whose particle scalers DIFFER from the checkpoint's."""
    import json
    import h5py
    import numpy as np
    h5 = tmp_path / "t.h5"
    with h5py.File(h5, "w") as h:
        h["knobs"] = np.stack([np.asarray(space.low) + 0.5 * (np.asarray(space.high) - np.asarray(space.low))] * n)
        h["drive_parts"] = np.ones((n, P, 6), np.float32)
        h["witness_parts"] = np.ones((n, P, 6), np.float32)
        for k in ("drive_present", "witness_viable", "drive_density", "witness_density"):
            h[k] = np.ones(n, bool)
        h["drive_frac"] = np.ones(n, np.float32)
        h["witness_frac"] = np.ones(n, np.float32)
        h["group"] = np.array([0, 0, 1, 1, 2, 3], np.int32)
    norm = {"knob_keys": list(space.keys), "knob_low": list(space.low), "knob_high": list(space.high),
            "drive_mean": drive_mean, "drive_std": [2.0] * 6, "witness_mean": [0.3] * 6,
            "witness_std": [3.0] * 6}
    (tmp_path / "t_norm.json").write_text(json.dumps(norm))
    return str(h5)


def test_zero_epoch_finetune_reproduces_old_flow(tmp_path):
    """train.py's warm-start path (dataset standardized with the CHECKPOINT's scalers, model built
    from that norm, weights loaded) must decode exactly like the widened old flow: one frame."""
    from twobunch_cnf.dataset import SCALER_KEYS, TwoBunchFlowDataModule
    from twobunch_cnf.train import build_model, check_knob_frame
    old_flow = _flow()
    old_flow.drive_mean.fill_(0.7); old_flow.drive_std.fill_(1.5)       # old frame != new table's
    wide = widen_conditioning(old_flow, NEW)
    state = wide.state_dict()
    h5 = _processed(tmp_path, NEW, drive_mean=[-4.0] * 6)
    scaler = {k: state[k].tolist() for k in SCALER_KEYS}
    dm = TwoBunchFlowDataModule(processed_h5=h5, scaler=scaler)
    dm.setup()
    check_knob_frame(state, dm.norm, "ckpt")
    model = build_model(dm.norm, n_layers=2, hidden_dim=16).eval()
    model.load_state_dict(state, strict=False)
    torch.testing.assert_close(model.drive_mean, wide.drive_mean)
    # the training targets are standardized in the model's own frame
    torch.testing.assert_close(dm.full.drive[0, 0], (torch.ones(6) - 0.7) / 1.5)
    g = torch.Generator().manual_seed(3)
    kn = torch.rand(4, 5, generator=g)
    z = (torch.randn(64, 6, generator=g), torch.randn(64, 6, generator=g))
    a, b = _obs(wide, kn, z), _obs(model, kn, z)
    for k in ("bunch_spacing", "drive_sigma_z", "drive_norm_emit_x", "T_witness"):
        torch.testing.assert_close(b[k], a[k], rtol=1e-6, atol=1e-12)


def test_warm_start_refuses_a_different_knob_frame(tmp_path):
    from twobunch_cnf.train import check_knob_frame
    state = _flow().state_dict()                 # 3-knob frame vs a 5-knob table
    with pytest.raises(SystemExit, match="knob frame differs"):
        check_knob_frame(state, {"knob_low": list(NEW.low), "knob_high": list(NEW.high)}, "ckpt")


def test_group_split_keeps_groups_together():
    from twobunch_cnf.dataset import group_split
    import numpy as np
    groups = np.repeat(np.arange(20), 5)
    tr, va = group_split(groups, 0.2, seed=0)
    assert set(groups[tr]).isdisjoint(set(groups[va])) and 15 <= len(va) <= 25
    assert sorted(tr + va) == list(range(100))


def test_roles_kept_and_rest_group_split(tmp_path):
    """Rows given a role (--role-map: the split an earlier flow used) keep it, holdout rows are in
    neither split; the rest is a group split per campaign."""
    import h5py
    import numpy as np
    from twobunch_cnf.dataset import TwoBunchFlowDataModule
    h5 = _processed(tmp_path, NEW, drive_mean=[0.0] * 6, n=6)
    with h5py.File(h5, "a") as h:
        h["campaign"] = np.array([0, 0, 1, 1, 1, 1], np.int16)
        h["idx"] = np.array([0, 1, 0, 1, 2, 3], np.int32)
        h["holdout"] = np.array([0, 0, 0, 0, 0, 1], bool)
        h.attrs["campaigns"] = "old,new"
        del h["group"]
        h["group"] = np.array([0, 1, 2, 2, 3, 4], np.int32)
    dm = TwoBunchFlowDataModule(processed_h5=h5, split="group", val_fraction=0.4,
                                orig_split={"old/0": "val", "old/1": "train"})
    dm.setup()
    tr, va, te = set(dm.train.indices), set(dm.val.indices), set(dm.test.indices)
    assert 0 in va and 1 in tr and te == {5} and not (tr & va) and len(tr | va | te) == 6
    assert ({2, 3} <= tr) or ({2, 3} <= va)                      # group 2 kept together


def test_paired_delta_metrics_finds_val_pairs(tmp_path):
    import h5py
    import numpy as np
    from twobunch_cnf.dataset import TwoBunchFlowDataModule
    from twobunch_cnf.selection import PairedDeltaMetrics
    h5 = _processed(tmp_path, NEW, drive_mean=[0.0] * 6, n=6)
    with h5py.File(h5, "a") as h:
        h["campaign"] = np.array([0, 0, 1, 1, 1, 1], np.int16)
        h["idx"] = np.arange(6, dtype=np.int32)
        h.attrs["campaigns"] = "old,new"
        h["pair_key"] = np.array([-1, -1, 7, 7, 8, 8])
        h["pair_ref"] = np.array([0, 0, 0, 1, 0, 1], bool)
        del h["group"]
        h["group"] = np.array([0, 1, 2, 2, 3, 3], np.int32)
    dm = TwoBunchFlowDataModule(processed_h5=h5, split="group", val_fraction=0.99)
    dm.setup()
    cb = PairedDeltaMetrics(dm, n=32)
    cb._setup()
    assert sorted(cb.pairs) == [(2, 3), (4, 5)] and set(cb.old) <= {0, 1}
    m = _flow()
    wide = widen_conditioning(m, NEW)
    assert np.isfinite(cb._pred_spacing(wide, [2, 3])).all()
