"""Data-contract helpers: two-bunch ParticleGroup split / coordinate stacking, and injected charges."""
import numpy as np
import pytest

from twobunch_cnf import COORD_KEYS
from twobunch_cnf.beam_io import split_bunches, stack_coords, subsample


def _two_bunch_group(n_d=30, n_w=10):
    pmd = pytest.importorskip("pmd_beamphysics")
    rng = np.random.default_rng(0)
    n = n_d + n_w
    data = {k: rng.normal(size=n) for k in ("x", "y", "px", "py")}
    data.update(z=np.zeros(n), t=rng.normal(scale=1e-13, size=n), pz=np.full(n, 1e10),
                status=np.ones(n, dtype=int), species="electron",
                weight=np.r_[np.full(n_d, 1e-14 * 1.001), np.full(n_w, 1e-14 * 0.999)])
    data["x"][:n_d] += 1.0                             # tag the drive by position too
    return pmd.ParticleGroup(data=data)


def test_split_bunches_drive_is_the_higher_weight():
    drive, witness = split_bunches(_two_bunch_group())
    assert len(drive.x) == 30 and len(witness.x) == 10 and drive.x.mean() > witness.x.mean() + 0.5


def test_split_single_weight_is_drive_only():
    P = _two_bunch_group()
    P.weight[:] = 1e-14
    drive, witness = split_bunches(P)
    assert witness is None and len(drive.x) == 40


def test_stack_coords_order_and_dtype():
    P = _two_bunch_group()
    c = stack_coords(P)
    assert c.shape == (40, 6) and c.dtype == np.float32
    for j, k in enumerate(COORD_KEYS):
        np.testing.assert_allclose(c[:, j], getattr(P, k).astype(np.float32))


def test_subsample_no_replacement():
    c = np.arange(30 * 6, dtype=np.float32).reshape(30, 6)
    out = subsample(c, 12, np.random.default_rng(0))
    assert out.shape == (12, 6) and len(np.unique(out[:, 0])) == 12


def test_injected_charges_per_sample_and_constant():
    torch = pytest.importorskip("torch")
    from twobunch_cnf.charges import injected_charges_nC
    keys = ["L2PhaseSet", "drive_fraction", "distgen:total_charge:value"]
    lo, hi = [-40.0, 0.6, 1500.0], [-34.0, 0.8, 1700.0]
    q = injected_charges_nC({"transmission": "per_sample"}, keys, lo, hi)
    kn = torch.tensor([[0.5, 0.5, 0.5], [0.0, 1.0, 0.0]])
    qd, qw = q(kn)
    torch.testing.assert_close(qd, torch.tensor([1.6 * 0.7 * 1.001, 1.5 * 0.8 * 1.001]))
    torch.testing.assert_close(qw, torch.tensor([1.6 * 0.3 * 0.999, 1.5 * 0.2 * 0.999]))
    const = injected_charges_nC({"drive_full_charge_nC": 1.2, "witness_full_charge_nC": 0.4}, keys[:1], lo[:1], hi[:1])
    qd, qw = const(torch.rand(3, 1))
    assert torch.allclose(qd, torch.full((3,), 1.2)) and torch.allclose(qw, torch.full((3,), 0.4))
