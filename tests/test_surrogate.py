"""Surrogate unit tests: shapes, flow invertibility, NLL, and the MBRL gradient contract.

Synthetic (no checkpoint/data needed); fast on CPU. Skips if torch is absent.
"""
import pytest

torch = pytest.importorskip("torch")
from twobunch_cnf.model import TwoBunchFlow  # noqa: E402


def _model(n_layers=4, hidden=32):
    torch.manual_seed(0)
    return TwoBunchFlow(condition_dim=8, n_layers=n_layers, hidden_dim=hidden,
                        n_aux_particles=128).eval()


def test_coupling_invertible():
    m = _model()
    w = torch.randn(20, 6)
    h = torch.randn(20, m.hparams.hidden_dim)
    z, ld_inv = m._inverse_enc(w, h)
    w2, ld_fwd = m._forward_enc(z, h)
    assert torch.allclose(w, w2, atol=1e-4), "flow not invertible"
    assert torch.allclose(ld_inv + ld_fwd, torch.zeros_like(ld_inv), atol=1e-4)


def test_sample_shapes_and_observables():
    m = _model()
    knobs = torch.rand(5, 8)
    for k in (0, 1):
        assert m.sample_bunch(knobs, k, 64).shape == (5, 64, 6)
    obs = m.observables(knobs, n=128)
    for key in ("drive_norm_emit_x", "witness_norm_emit_x", "bunch_spacing",
                "energy_difference", "transverse_offset", "angular_misalignment",
                "p_surv", "T_drive", "T_witness"):
        assert obs[key].shape == (5,), key
        assert torch.isfinite(obs[key]).all(), key


def test_bunch_nll_finite_and_differentiable():
    m = _model()
    parts = torch.randn(6, 256, 6)
    knobs = torch.rand(6, 8)
    nll = m.bunch_nll(parts, knobs, 1)
    assert torch.isfinite(nll)
    nll.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in m.parameters() if p.requires_grad)


def test_gradient_flows_to_knobs():
    """MBRL contract: observables must be differentiable w.r.t. the knobs."""
    m = _model()
    knobs = torch.rand(4, 8, requires_grad=True)
    obs = m.observables(knobs, n=256)
    loss = (obs["witness_norm_emit_x"].sum() + obs["transverse_offset"].sum()
            + obs["p_surv"].sum())
    loss.backward()
    assert knobs.grad is not None
    assert torch.isfinite(knobs.grad).all()
    assert knobs.grad.abs().sum() > 0, "no gradient reached the knobs"


def test_whitening_lower_triangular_positive_diag():
    m = _model()
    h = torch.randn(7, m.hparams.hidden_dim)
    mu, Lm = m._whiten(h, 0)
    assert mu.shape == (7, 6) and Lm.shape == (7, 6, 6)
    diag = torch.diagonal(Lm, dim1=-2, dim2=-1)
    assert (diag > 0).all(), "Cholesky diagonal must be positive"
    assert torch.allclose(torch.triu(Lm, diagonal=1), torch.zeros_like(Lm), atol=1e-6)


def test_bunch_nll_decreases_on_overfit():
    """The flow+whitening must actually learn (finite NLL alone isn't enough)."""
    torch.manual_seed(0)
    m = _model(n_layers=6, hidden=64)
    knobs = torch.rand(1, 8)
    A = torch.randn(6, 6) * 0.3
    parts = torch.randn(1, 512, 6) @ A.T + torch.tensor([0.5, -0.3, 0.2, 0.0, 0.1, -0.2])
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    init = m.bunch_nll(parts, knobs, 0).item()
    for _ in range(200):
        opt.zero_grad()
        loss = m.bunch_nll(parts, knobs, 0)
        loss.backward()
        opt.step()
    final = m.bunch_nll(parts, knobs, 0).item()
    import math
    assert math.isfinite(final) and final < init - 0.5, (init, final)


def test_feasibility_learns_point_mass():
    """Witness-viability classifier must separate the ~31% destroyed mode."""
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score
    torch.manual_seed(0)
    m = _model()
    knobs = torch.rand(400, 8)
    viable = (knobs[:, 1] + 0.3 * knobs[:, 4] > 0.7).float()
    opt = torch.optim.Adam(m.feas_head.parameters(), lr=5e-3)
    for _ in range(300):
        opt.zero_grad()
        p_surv, _, _ = m.feasibility(knobs)
        F.binary_cross_entropy(p_surv.clamp(1e-6, 1 - 1e-6), viable).backward()
        opt.step()
    p_surv, _, _ = m.feasibility(knobs)
    assert roc_auc_score(viable.numpy(), p_surv.detach().numpy()) > 0.9


def _batch(B=4, P=128, drive_density=True, witness_density=True):
    f = lambda v: torch.full((B,), v, dtype=torch.bool)
    return {
        "knobs": torch.rand(B, 8),
        "drive": torch.randn(B, P, 6), "witness": torch.randn(B, P, 6),
        "drive_present": f(drive_density), "witness_viable": f(witness_density),
        "drive_density": f(drive_density), "witness_density": f(witness_density),
        "drive_frac": torch.rand(B), "witness_frac": torch.rand(B) * witness_density,
    }


def test_step_handles_empty_witness_and_empty_batch():
    m = _model()
    m.log_dict = lambda *a, **k: None  # no Trainer attached in unit test
    loss_w = m._step(_batch(witness_density=False), "train")
    assert torch.isfinite(loss_w) and loss_w.requires_grad
    # fully empty density (only the always-on BCE term contributes)
    loss_e = m._step(_batch(drive_density=False, witness_density=False), "train")
    assert torch.isfinite(loss_e)


def test_state_dict_roundtrip_buffers_and_sampling():
    """Buffers (de-norm scalers, bounds, tril idx) must travel in the state_dict, and
    sampling must actually use them (deterministic + identical after reload)."""
    a = TwoBunchFlow(condition_dim=8, n_layers=4, hidden_dim=32,
                     drive_mean=[1, 2, 3, 4, 5, 6], drive_std=[1, 1, 1, 1, 1, 1],
                     witness_mean=[0.1] * 6, witness_std=[2.0] * 6,
                     knob_low=[0.0] * 8, knob_high=[1.0] * 8).eval()
    b = TwoBunchFlow(condition_dim=8, n_layers=4, hidden_dim=32).eval()
    b.load_state_dict(a.state_dict())
    for name in ("drive_mean", "witness_std", "knob_high", "_tril_r"):
        assert torch.equal(getattr(a, name), getattr(b, name)), name
    knobs = torch.rand(3, 8)
    torch.manual_seed(7); s1 = a.sample_bunch(knobs, 1, 16)
    torch.manual_seed(7); s2 = b.sample_bunch(knobs, 1, 16)
    assert torch.allclose(s1, s2, atol=1e-6)


def test_slopes_match_pxpz():
    from twobunch_cnf.properties import _slopes
    parts = torch.randn(3, 100, 6)
    parts[..., 5] = 1e10  # pz >> clamp floor
    xp, yp = _slopes(parts)
    assert torch.allclose(xp, (parts[..., 3] / parts[..., 5]).mean(1), atol=1e-9)


def test_bunches_modeled_distinctly():
    """Guards the drive/witness mix-up class: k=0 and k=1 must route to their own scaler +
    whitening head (a bug that modeled both as the drive would pass every other test)."""
    m = TwoBunchFlow(condition_dim=8, n_layers=4, hidden_dim=32,
                     drive_mean=[10., 0, 0, 0, 0, 0], drive_std=[1.] * 6,
                     witness_mean=[-10., 0, 0, 0, 0, 0], witness_std=[1.] * 6).eval()
    knobs = torch.rand(4, 8)
    with torch.no_grad():
        d = m.sample_bunch(knobs, 0, 512).mean(dim=1)[:, 0]
        w = m.sample_bunch(knobs, 1, 512).mean(dim=1)[:, 0]
    assert d.mean() > 3 and w.mean() < -3, "bunches not routed to distinct per-bunch scalers"
    h = torch.randn(3, m.hparams.hidden_dim)
    assert not torch.allclose(m._whiten(h, 0)[0], m._whiten(h, 1)[0]), "whitening heads not distinct"


def test_witness_nll_decreases_on_overfit():
    """The witness density path (k=1) must actually learn, not just be finite."""
    import math
    torch.manual_seed(0)
    m = _model(n_layers=6, hidden=64)
    knobs = torch.rand(1, 8)
    parts = torch.randn(1, 512, 6) @ (torch.randn(6, 6) * 0.3).T + torch.tensor([0.4, -0.2, 0.3, 0.1, 0.0, -0.1])
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    init = m.bunch_nll(parts, knobs, 1).item()
    for _ in range(200):
        opt.zero_grad(); m.bunch_nll(parts, knobs, 1).backward(); opt.step()
    final = m.bunch_nll(parts, knobs, 1).item()
    assert math.isfinite(final) and final < init - 0.5, (init, final)


def test_inter_bunch_spacing_and_energy_signs():
    """Pin the inter-bunch convention: bunch_spacing = drive_z - witness_z; drive is hotter."""
    from twobunch_cnf.properties import inter_bunch
    drive = torch.randn(1, 400, 6) * 0.01
    witness = torch.randn(1, 400, 6) * 0.01
    drive[..., 2] += 5.0; witness[..., 2] += -3.0       # z centroids
    drive[..., 5] += 1e10; witness[..., 5] += 0.9e10    # pz: drive higher energy
    ib = inter_bunch(drive, witness)
    assert torch.isclose(ib["bunch_spacing"][0], torch.tensor(8.0), atol=0.1)
    assert ib["energy_difference"][0] > 0


def test_subsample_no_replacement_at_boundary():
    import numpy as np
    from twobunch_cnf.beam_io import subsample
    coords = np.arange(20 * 6, dtype=np.float32).reshape(20, 6)
    out = subsample(coords, 20, np.random.default_rng(0))
    assert out.shape == (20, 6)
    assert len(np.unique(out[:, 0])) == 20  # a permutation, no repeats


# ---- RQS spline correctness (the hand-rolled kernel) ------------------------
def _rqs_params(n, K, seed=0):
    torch.manual_seed(seed)
    return torch.randn(n, K), torch.randn(n, K), torch.randn(n, K - 1)


def test_rqs_invertible_and_logdet_cancels():
    from twobunch_cnf.model import _rqs_1d
    B, K, n = 5.0, 10, 400
    x = torch.rand(n) * 2 * B - B           # inside [-B, B)
    uw, uh, ud = _rqs_params(n, K, 1)
    y, ldf = _rqs_1d(x, uw, uh, ud, False, B)
    xr, ldi = _rqs_1d(y, uw, uh, ud, True, B)
    # atol 1e-3 (not 1e-4): the discriminant gradient-safety floor (clamp_min 1e-10 in the
    # inverse) perturbs the ~0.15% of points landing on a knot by up to ~5e-4. Median ~1e-7.
    assert torch.allclose(x, xr, atol=1e-3), (x - xr).abs().max()
    assert torch.allclose(ldf + ldi, torch.zeros_like(ldf), atol=1e-3)


def test_rqs_logdet_matches_autograd():
    from twobunch_cnf.model import _rqs_1d
    B, K, n = 5.0, 8, 300
    x = (torch.rand(n) * 2 * B - B).clamp(-B + 0.02, B - 0.02).requires_grad_(True)
    uw, uh, ud = _rqs_params(n, K, 2)
    y, ld = _rqs_1d(x, uw, uh, ud, False, B)
    (g,) = torch.autograd.grad(y.sum(), x)      # elementwise -> do_i/dx_i
    assert torch.allclose(torch.log(g.abs()), ld, atol=1e-4), (torch.log(g.abs()) - ld).abs().max()


def test_rqs_tails_identity():
    from twobunch_cnf.model import _rqs_1d
    B, K = 5.0, 8
    x = torch.cat([torch.full((40,), -B - 1.0), torch.full((40,), B + 3.0)])
    uw, uh, ud = _rqs_params(80, K, 3)
    y, ld = _rqs_1d(x, uw, uh, ud, False, B)
    assert torch.allclose(y, x) and torch.allclose(ld, torch.zeros_like(ld))


def test_rqs_monotonic_and_continuous_at_boundary():
    from twobunch_cnf.model import _rqs_1d
    B, K, n = 5.0, 8, 600
    x = torch.linspace(-B + 1e-3, B - 1e-3, n)
    uw, uh, ud = (t.expand(n, -1).contiguous() for t in _rqs_params(1, K, 4))
    y, _ = _rqs_1d(x, uw, uh, ud, False, B)
    assert (y[1:] - y[:-1] > 0).all()                       # strictly increasing
    # C0 continuity with the identity tail: spline endpoints map to +/-B
    edge = torch.tensor([-B + 1e-6, B - 1e-6])
    uw2, uh2, ud2 = (t[:1].expand(2, -1).contiguous() for t in _rqs_params(1, K, 4))
    ye, _ = _rqs_1d(edge, uw2, uh2, ud2, False, B)
    assert torch.allclose(ye, torch.tensor([-B, B]), atol=1e-3)


def test_rqs_model_nll_grad_and_sampling():
    torch.manual_seed(0)
    m = TwoBunchFlow(condition_dim=8, n_layers=4, hidden_dim=32,
                     n_bins=8, n_aux_particles=128).eval()
    # invertibility of the stacked flow (base<->w)
    w = torch.randn(16, 6)
    h = torch.randn(16, 32)
    z, _ = m._inverse_enc(w, h)
    w2, _ = m._forward_enc(z, h)
    assert torch.allclose(w, w2, atol=1e-3)
    # NLL finite + differentiable; gradient reaches the knobs
    knobs = torch.rand(4, 8, requires_grad=True)
    nll = m.bunch_nll(torch.randn(4, 256, 6), knobs, 1)
    assert torch.isfinite(nll)
    obs = m.observables(knobs, n=128)
    (obs["witness_norm_emit_x"].sum() + obs["transverse_offset"].sum()).backward()
    assert knobs.grad is not None and torch.isfinite(knobs.grad).all() and knobs.grad.abs().sum() > 0


def test_step_cov_term_and_rqs():
    m = TwoBunchFlow(condition_dim=8, n_layers=4, hidden_dim=32,
                     n_bins=8, w_cov=0.5, n_aux_particles=128)
    m.log_dict = lambda *a, **k: None
    loss = m._step(_batch(), "train")
    assert torch.isfinite(loss) and loss.requires_grad


def test_step_centroid_and_spacing_terms():
    """w_mean / w_spacing: logged, finite, gradients reach the whitening (centroid) head; off by default."""
    logs = {}
    m = TwoBunchFlow(condition_dim=8, n_layers=4, hidden_dim=32, n_aux_particles=128,
                     drive_std=[1.0, 1.0, 50e-6, 1.0, 1.0, 1.0], witness_std=[1.0, 1.0, 50e-6, 1.0, 1.0, 1.0],
                     w_mean=1.0, w_spacing=1.0)
    m.log_dict = lambda d, **k: logs.update(d)
    b = _batch()
    b["witness"][..., 2] += 3.0                       # true witness centroid offset -> a spacing to match
    loss = m._step(b, "train")
    assert torch.isfinite(loss) and loss.requires_grad
    assert {"train_mean_drive", "train_mean_witness", "train_spacing", "train_spacing_mae_um"} <= set(logs)
    m.zero_grad()
    loss.backward()
    g = sum(p.grad.abs().sum() for p in m.whiten_heads.parameters() if p.grad is not None)
    assert torch.isfinite(g) and g > 0
    # default model: no centroid terms, and the loss is the old one (same seeds)
    logs.clear()
    torch.manual_seed(1)
    d0 = TwoBunchFlow(condition_dim=8, n_layers=4, hidden_dim=32, n_aux_particles=128)
    d0.log_dict = lambda d, **k: logs.update(d)
    torch.manual_seed(2)
    l0 = d0._step(b, "train")
    assert not any(k.startswith("train_mean_") or k.startswith("train_spacing") for k in logs)
    torch.manual_seed(1)
    d1 = TwoBunchFlow(condition_dim=8, n_layers=4, hidden_dim=32, n_aux_particles=128, w_mean=0.0, w_spacing=0.0)
    d1.log_dict = lambda d, **k: None
    torch.manual_seed(2)
    assert torch.equal(l0, d1._step(b, "train"))


def test_pair_loss_and_two_part_T_head():
    logs = {}
    m = TwoBunchFlow(condition_dim=8, n_layers=4, hidden_dim=32, n_aux_particles=128,
                     drive_std=[1.0, 1.0, 50e-6, 1.0, 1.0, 1.0], witness_std=[1.0, 1.0, 50e-6, 1.0, 1.0, 1.0],
                     w_pair=1.0, pair_batch=8, pair_particles=64, t_head_lr=1e-3)
    m.log_dict = lambda d, **k: logs.update(d)
    calls = []
    def src(b, device):
        calls.append(b)
        return torch.rand(b, 8), torch.rand(b, 8), torch.full((b,), 20e-6)
    m.pair_source = src
    b = _batch()
    b["drive_frac"] = torch.tensor([1.0, 1.0, 0.9, 1.0]); b["witness_frac"] = torch.tensor([1.0, 0.8, 1.0, 1.0])
    loss = m._step(b, "train")
    assert calls == [8] and "train_pair" in logs and torch.isfinite(loss)
    assert {"train_bce_Td_full", "train_bce_Tw_full", "train_mse_Tw_tail", "train_mae_Tw"} <= set(logs)
    m.zero_grad(); loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.t_full_head.parameters())
    _, td, tw = m.feasibility(torch.rand(5, 8))
    assert ((td >= 0) & (td <= 1) & (tw >= 0) & (tw <= 1)).all()
    logs.clear(); m._step(b, "val")
    assert "val_pair" not in logs                                   # pair loss is train-only
    opt = m.configure_optimizers()["optimizer"] if isinstance(m.configure_optimizers(), dict) else m.configure_optimizers()
    assert len(opt.param_groups) == 2 and opt.param_groups[1]["lr"] == 1e-3


# ---------------------------------------------------------------------------
# CRN sampling (Stage-C, plan C0): generator/z kwargs on sample_bunch/observables
# ---------------------------------------------------------------------------
def _tiny_model():
    import torch
    from twobunch_cnf.model import TwoBunchFlow

    torch.manual_seed(0)
    return TwoBunchFlow(condition_dim=5, n_layers=2, hidden_dim=16,
                        drive_mean=[0.0] * 6, drive_std=[1.0] * 6,
                        witness_mean=[0.0] * 6, witness_std=[1.0] * 6,
                        knob_low=[0.0] * 5, knob_high=[1.0] * 5)


def test_crn_generator_determinism():
    import torch

    m = _tiny_model().eval()
    knobs = torch.rand(3, 5)
    with torch.no_grad():
        a = m.sample_bunch(knobs, 0, 64, generator=torch.Generator().manual_seed(7))
        b = m.sample_bunch(knobs, 0, 64, generator=torch.Generator().manual_seed(7))
        c = m.sample_bunch(knobs, 0, 64, generator=torch.Generator().manual_seed(8))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_crn_shared_z_across_batch():
    """A frozen (n, latent_dim) buffer must broadcast over ANY batch size (the
    batched 7-step x 2-sign fit case), and per-condition clouds must differ only
    through the conditioning, not the noise."""
    import torch

    m = _tiny_model().eval()
    z = torch.randn(32, m.latent_dim)
    with torch.no_grad():
        out3 = m.sample_bunch(torch.full((3, 5), 0.5), 0, 32, z=z)
        out1 = m.sample_bunch(torch.full((1, 5), 0.5), 0, 32, z=z)
    assert out3.shape == (3, 32, 6)
    # identical conditioning + shared z => identical clouds across the batch (up to float32 kernels that
    # differ with the batch size: ~3e-6 on O(1) values for the RQS flow)
    assert torch.allclose(out3[0], out3[1]) and torch.allclose(out3[0], out1[0], atol=1e-5)
    # repeated call with the same z is deterministic
    with torch.no_grad():
        again = m.sample_bunch(torch.full((3, 5), 0.5), 0, 32, z=z)
    assert torch.equal(out3, again)
    # wrong z shape is rejected loudly
    import pytest as _pytest
    with _pytest.raises(ValueError):
        m.sample_bunch(torch.full((2, 5), 0.5), 0, 16, z=torch.randn(8, m.latent_dim))


def test_crn_default_still_stochastic():
    import torch

    m = _tiny_model().eval()
    knobs = torch.rand(2, 5)
    with torch.no_grad():
        a = m.sample_bunch(knobs, 0, 64)
        b = m.sample_bunch(knobs, 0, 64)
    assert not torch.equal(a, b)


def test_crn_observables_two_frozen_buffers():
    import torch

    m = _tiny_model().eval()
    knobs = torch.rand(2, 5)
    zd = torch.randn(48, m.latent_dim)
    zw = torch.randn(48, m.latent_dim)
    with torch.no_grad():
        o1 = m.observables(knobs, n=48, z_drive=zd, z_witness=zw)
        o2 = m.observables(knobs, n=48, z_drive=zd, z_witness=zw)
    for k in ("drive_sigma_z", "witness_sigma_z", "bunch_spacing"):
        assert torch.equal(o1[k], o2[k]), k
