"""TwoBunchCNF.draw / to_particle_group on a small random flow: S2E count + weight convention, fixed-z
frame, exact drift between planes, input handling."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pmd = pytest.importorskip("pmd_beamphysics")

from twobunch_cnf.beam_io import split_bunches  # noqa: E402
from twobunch_cnf.cnf import C_LIGHT, TwoBunchCNF, reference_time_upstream  # noqa: E402
from twobunch_cnf.model import TwoBunchFlow  # noqa: E402

KEYS = ["L2PhaseSet", "drive_fraction", "distgen:total_charge:value"]
LO, HI = [-40.0, 0.66, 1585.6], [-34.0, 0.80, 1614.4]
BASE = {"L2PhaseSet": -36.0, "drive_fraction": 0.75, "distgen:total_charge:value": 1600.0}
E_TOT, L_TDC, S_EXIT, T_EXIT = 9.999e9, 1.0334498, 987.131388761613, 3.29273295892451e-06
HANDOFF = {
    "element": "XTCAVF",
    "trained_plane": "exit",
    "species": "electron",
    "planes": {
        "exit": {"s_m": S_EXIT, "ref_time_s": T_EXIT},
        "entrance": {"s_m": S_EXIT - L_TDC, "ref_time_s": reference_time_upstream(T_EXIT, L_TDC, E_TOT)},
    },
}
NORM = {"transmission": "per_sample", "drive_full_charge_nC": 1.2, "witness_full_charge_nC": 0.4}


def _cnf(p_surv_bias=None):
    torch.manual_seed(0)
    flow = TwoBunchFlow(
        condition_dim=3,
        n_layers=2,
        hidden_dim=16,
        beam_latent_dim=2,
        knob_keys=KEYS,
        knob_low=LO,
        knob_high=HI,
        treaty_point="XTCAVF",
        drive_mean=[1e-4, 2e-4, 3e-5, 1e5, 2e5, 9.88e9],
        drive_std=[1e-3, 3e-4, 6e-5, 5e5, 1e6, 7e7],
        witness_mean=[-2e-3, 5e-4, -1e-4, -6e5, 2e6, 9.76e9],
        witness_std=[8e-4, 2e-4, 5e-5, 3e5, 8e5, 3.5e7],
    )
    if p_surv_bias is not None:
        with torch.no_grad():
            flow.feas_head[-1].bias[0] = p_surv_bias
    return TwoBunchCNF(flow, NORM, HANDOFF, BASE, name="test")


def test_draw_is_reproducible_by_seed():
    m = _cnf()
    a, b, c = m.draw(seed=3, n_particles=500), m.draw(seed=3, n_particles=500), m.draw(seed=4, n_particles=500)
    assert torch.equal(a.drive, b.drive) and torch.equal(a.witness, b.witness)
    assert not torch.equal(a.drive, c.drive)


def test_counts_and_weights_follow_the_s2e_convention():
    m = _cnf()
    n = 10000
    dr = m.draw(seed=0, n_particles=n)
    f, q = BASE["drive_fraction"], BASE["distgen:total_charge:value"] * 1e-3  # nC
    n_inj_d = round(n * f)
    assert int(dr.n_drive[0]) == round(n_inj_d * float(dr.T_drive[0]))
    assert int(dr.n_witness[0]) == round((n - n_inj_d) * float(dr.T_witness[0]))
    w0 = q * 1e-9 / n  # C per injected particle
    assert float(dr.weight_drive[0]) == pytest.approx(w0 * 1.001, rel=1e-12)
    assert float(dr.weight_witness[0]) == pytest.approx(w0 * 0.999, rel=1e-12)
    P = m.to_particle_group(dr, plane="exit")
    assert len(P) == int(dr.n_drive[0] + dr.n_witness[0])
    assert P.charge == pytest.approx(float(dr.charge_drive_nC[0] + dr.charge_witness_nC[0]) * 1e-9, rel=1e-12)
    w = np.unique(P.weight)
    assert len(w) == 2 and w[-1] / w[0] == pytest.approx(1.001 / 0.999, rel=1e-12)
    drive, witness = split_bunches(P)
    assert len(drive) == int(dr.n_drive[0]) and len(witness) == int(dr.n_witness[0])


def test_lost_witness_leaves_a_single_weight():
    m = _cnf(p_surv_bias=-20.0)  # p_surv ~ 0
    dr = m.draw(seed=0, n_particles=2000)
    assert not bool(dr.witness_present[0]) and int(dr.n_witness[0]) == 0 and float(dr.T_witness[0]) == 0.0
    drive, witness = split_bunches(m.to_particle_group(dr))
    assert witness is None and len(drive) == int(dr.n_drive[0])
    assert int(m.draw(seed=0, n_particles=2000, witness="always").n_witness[0]) > 0


def test_exit_plane_is_fixed_z_with_t_from_the_flow_z():
    m = _cnf()
    dr = m.draw(seed=1, n_particles=3000)
    P = m.to_particle_group(dr, plane="exit")
    assert P.in_z_coordinates and np.all(P.z == S_EXIT) and np.all(P.status == 1) and P.species == "electron"
    d, w = dr.clouds(0)
    c = torch.cat([d, w]).double().numpy()
    np.testing.assert_allclose(-C_LIGHT * (P.t - T_EXIT), c[:, 2], rtol=0, atol=1e-12)  # z_flow recovered
    for j, k in enumerate(("x", "y")):
        np.testing.assert_array_equal(getattr(P, k), c[:, j])
    np.testing.assert_array_equal(P.pz, c[:, 5])


def test_entrance_is_an_exact_back_drift_of_the_exit_plane():
    m = _cnf()
    dr = m.draw(seed=2, n_particles=3000)
    ex, en = m.to_particle_group(dr, plane="exit"), m.to_particle_group(dr, plane="entrance")
    assert np.all(en.z == S_EXIT - L_TDC)
    e = np.sqrt(en.px**2 + en.py**2 + en.pz**2 + 0.51099895e6**2)
    np.testing.assert_allclose(en.x + L_TDC * en.px / en.pz, ex.x, rtol=0, atol=1e-15)
    np.testing.assert_allclose(en.y + L_TDC * en.py / en.pz, ex.y, rtol=0, atol=1e-15)
    np.testing.assert_allclose(en.t + L_TDC * e / (C_LIGHT * en.pz), ex.t, rtol=0, atol=1e-21)
    # pmd's own drift agrees: drift the entrance beam to the exit plane
    fwd = en.copy()
    fwd.drift_to_z(S_EXIT)
    np.testing.assert_allclose(fwd.x, ex.x, rtol=0, atol=1e-12)
    np.testing.assert_allclose(fwd.t, ex.t, rtol=0, atol=1e-18)
    # the bunch mean sits on the entrance reference time (to the energy-spread level)
    assert abs(np.average(en.t, weights=en.weight) - HANDOFF["planes"]["entrance"]["ref_time_s"]) < 1e-12


def test_lume_bmad_style_drift_to_z_is_a_no_op():
    m = _cnf()
    P = m.particle_group(seed=0, n_particles=1000)
    t0 = P.t.copy()
    P.drift_to_z()
    np.testing.assert_array_equal(P.t, t0)


def test_knob_inputs_and_validation():
    m = _cnf()
    assert torch.allclose(m.knobs_norm(), m.knobs_norm(BASE))
    one = m.knobs_norm({"L2PhaseSet": -35.0})
    assert one.shape == (1, 3) and float(one[0, 1]) == pytest.approx((0.75 - 0.66) / 0.14, rel=1e-6)
    batch = m.knobs_norm({"L2PhaseSet": [-39.0, -35.0, -34.5]})
    assert batch.shape == (3, 3)
    arr = m.knobs_norm(np.array([[-36.0, 0.75, 1600.0]]))
    assert torch.allclose(arr, m.knobs_norm())
    with pytest.raises(KeyError, match="unknown"):
        m.knobs_norm({"L9PhaseSet": 1.0})
    oob = {"L2PhaseSet": -45.0}
    m.knobs_norm(oob)  # 'none': silent
    with pytest.warns(UserWarning, match="outside the training box"):
        m.knobs_norm(oob, validation="warn")
    with pytest.raises(ValueError, match="L2PhaseSet"):
        m.knobs_norm(oob, validation="error")


def test_opt_in_modes():
    m = _cnf()
    mix = m.draw(seed=0, n_particles=1000, transmission="mixture")
    with torch.no_grad():
        tail = torch.sigmoid(m.flow.feas_head(mix.knobs_norm))[:, 1]
    assert float(mix.T_drive[0]) in (1.0, pytest.approx(float(tail[0])))
    s = m.draw(seed=0, n_particles=1000, witness="sample")
    assert s.witness_present.dtype == torch.bool
    a, b = m.draw(seed=5, n_particles=500, latent="mean"), m.draw(seed=5, n_particles=500, latent="mean")
    assert torch.equal(a.drive, b.drive)
    with pytest.raises(ValueError):
        m.draw(witness="maybe")
    with pytest.raises(ValueError, match="one knob setting"):
        m.particle_group({"L2PhaseSet": [-36.0, -35.0]})
