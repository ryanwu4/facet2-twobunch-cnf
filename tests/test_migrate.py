"""migrate: twobunch-s2e-rl checkpoints convert once (removed options OFF) and then load strictly."""

import pytest

torch = pytest.importorskip("torch")
import lightning  # noqa: E402

from twobunch_cnf.migrate import REMOVED, migrate_checkpoint, migrate_file  # noqa: E402
from twobunch_cnf.model import TwoBunchFlow, load_flow  # noqa: E402

KEYS = ["a", "b", "c"]


def _flow(**kw):
    torch.manual_seed(0)
    return TwoBunchFlow(
        condition_dim=3,
        n_layers=2,
        hidden_dim=16,
        beam_latent_dim=2,
        knob_keys=KEYS,
        knob_low=[0.0, -1.0, 2.0],
        knob_high=[1.0, 1.0, 3.0],
        treaty_point="XTCAVF",
        **kw,
    ).eval()


def _legacy_ckpt(flow, **hp_overrides):
    """A checkpoint as twobunch-s2e-rl wrote it: the pruned options' hparams are present (at their values)."""
    hp = {**dict(flow.hparams), **{k: want for k, (want, _old) in REMOVED.items()}, **hp_overrides}
    return {"state_dict": flow.state_dict(), "hyper_parameters": hp, "pytorch-lightning_version": lightning.__version__}


def _probe(flow):
    g = torch.Generator().manual_seed(1)
    kn = torch.rand(3, 3, generator=g)
    z = (torch.randn(32, 6, generator=g), torch.randn(32, 6, generator=g))
    with torch.no_grad():
        return flow.observables(kn, n=32, z_drive=z[0], z_witness=z[1], generator=torch.Generator().manual_seed(2))


def test_migrated_checkpoint_loads_and_reproduces(tmp_path):
    flow = _flow()
    src = tmp_path / "legacy.ckpt"
    torch.save(_legacy_ckpt(flow), src)
    dst = migrate_file(src, tmp_path / "new.ckpt")
    new = load_flow(dst)
    a, b = _probe(flow), _probe(new)
    for k in a:
        assert torch.equal(a[k], b[k]), k
    rec = torch.load(dst, map_location="cpu", weights_only=True)["twobunch_cnf_migration"]
    assert rec["source"] == str(src) and set(rec["removed"]) == set(REMOVED)


def test_unmigrated_checkpoint_points_to_migrate(tmp_path):
    src = tmp_path / "legacy.ckpt"
    torch.save(_legacy_ckpt(_flow()), src)
    with pytest.raises(ValueError, match="twobunch_cnf.migrate"):
        load_flow(src)
    with pytest.raises(ValueError, match="twobunch_cnf.migrate"):  # the class itself is strict too
        TwoBunchFlow.load_from_checkpoint(str(src), map_location="cpu")


@pytest.mark.parametrize(
    "override",
    [{"centroid_head": True}, {"curve_order": 6}, {"bunches": (1,)}, {"t_two_part": False}, {"coupling": "affine"}],
)
def test_checkpoint_using_a_removed_option_is_refused(override):
    key = next(iter(override))
    with pytest.raises(ValueError, match=key):
        migrate_checkpoint(_legacy_ckpt(_flow(), **override))


def test_absent_option_takes_its_pre_prune_default():
    """A checkpoint from before t_two_part existed has no such key: its default (False) means unsupported."""
    ck = _legacy_ckpt(_flow())
    del ck["hyper_parameters"]["t_two_part"]
    with pytest.raises(ValueError, match="t_two_part"):
        migrate_checkpoint(ck)
    ok = _legacy_ckpt(_flow())
    del ok["hyper_parameters"]["curve_order"]  # off-by-default option: absent is fine
    migrate_checkpoint(ok)


def test_unknown_hparam_and_missing_frame_are_refused():
    with pytest.raises(ValueError, match="unknown hparam"):
        migrate_checkpoint(_legacy_ckpt(_flow(), new_experimental_head=True))
    ck = _legacy_ckpt(_flow())
    ck["hyper_parameters"]["treaty_point"] = None
    with pytest.raises(ValueError, match="treaty_point"):
        migrate_checkpoint(ck)


def test_never_in_place_and_no_silent_overwrite(tmp_path):
    src = tmp_path / "legacy.ckpt"
    torch.save(_legacy_ckpt(_flow()), src)
    with pytest.raises(ValueError, match="in place"):
        migrate_file(src, src)
    dst = migrate_file(src, tmp_path / "new.ckpt")
    with pytest.raises(FileExistsError):
        migrate_file(src, dst)
    migrate_file(src, dst, overwrite=True)
