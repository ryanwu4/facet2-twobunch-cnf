"""Deterministic centroid head: the sampled cloud is centred on c_k, the NLL does not train c_k, the pair loss is
deterministic, and models without the head are unchanged."""
import pytest

torch = pytest.importorskip("torch")
from twobunch_cnf.model import TwoBunchFlow  # noqa: E402


def _flow(**kw):
    torch.manual_seed(0)
    return TwoBunchFlow(condition_dim=5, n_layers=2, hidden_dim=16, n_aux_particles=64, centroid_buffer=64, **kw)


def test_cloud_mean_is_the_centroid_head_on_the_frozen_buffer():
    f = _flow(centroid_head=True).eval()
    knobs = torch.rand(3, 5)
    for k in (0, 1):
        x = f.sample_bunch(knobs, k, 64, physical=True, z=f._z_center)
        torch.testing.assert_close(x.mean(1), f.centroid_phys(knobs, k), rtol=1e-4, atol=1e-5)
        xr = f.sample_bunch(knobs, k, 37, physical=False)            # fresh draws, any n: exact too
        torch.testing.assert_close(xr.mean(1), f.centroid_heads[k](knobs), rtol=1e-4, atol=1e-5)


def test_nll_does_not_train_the_centroid_head_and_is_finite():
    f = _flow(centroid_head=True)
    parts, knobs = torch.randn(4, 32, 6), torch.rand(4, 5)
    nll = f.bunch_nll(parts, knobs, 0)
    assert torch.isfinite(nll)
    nll.backward()
    assert all(p.grad is None or p.grad.abs().max() == 0 for p in f.centroid_heads.parameters())
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in f.flows.parameters())


def test_pair_loss_is_deterministic_with_the_head():
    f = _flow(centroid_head=True, w_pair=1.0)
    kp, kr, d = torch.rand(8, 5), torch.rand(8, 5), torch.randn(8) * 1e-5
    f.pair_source = lambda n, dev: (kp, kr, d)
    torch.testing.assert_close(f._pair_loss(), f._pair_loss())


def test_no_head_means_no_new_parameters():
    f = _flow()
    assert not hasattr(f, "centroid_heads") and "_z_center" not in dict(f.named_buffers())


def test_partial_head_owns_only_z():
    f0 = _flow().eval()
    f = _flow(centroid_head=True, centroid_dims=[2]).eval()
    f.load_state_dict({**f0.state_dict(), **{k: v for k, v in f.state_dict().items() if k.startswith("centroid_heads") or k == "_z_center"}})
    knobs = torch.rand(3, 5)
    z = torch.randn(64, 6)
    a, b = f0.sample_bunch(knobs, 0, 64, physical=False, z=z), f.sample_bunch(knobs, 0, 64, physical=False, z=z)
    torch.testing.assert_close(b.mean(1)[:, 2], f.centroid_heads[0](knobs)[:, 2], rtol=1e-4, atol=1e-5)   # z mean = head
    torch.testing.assert_close(b[..., [0, 1, 3, 4, 5]], a[..., [0, 1, 3, 4, 5]])                         # rest untouched
    torch.testing.assert_close(b[..., 2] - b[..., 2].mean(1, keepdim=True), a[..., 2] - a[..., 2].mean(1, keepdim=True))
    parts = torch.randn(4, 32, 6)
    torch.testing.assert_close(f.bunch_nll(parts, knobs[:1].expand(4, 5), 0), f0.bunch_nll(parts, knobs[:1].expand(4, 5), 0))


def test_spacing_parametrization():
    f = _flow(centroid_head=True, centroid_dims=[2], centroid_param="spacing", centroid_act="silu").eval()
    knobs = torch.rand(3, 5)
    sp = f.centroid_phys(knobs, 0)[:, 2] - f.centroid_phys(knobs, 1)[:, 2]
    torch.testing.assert_close(sp, f.centroid_heads[1](knobs)[:, 2] * 1e-5, rtol=0, atol=1e-7)   # float32: ~10 nm
    d, w = f.sample_bunch(knobs, 0, 64), f.sample_bunch(knobs, 1, 64)
    torch.testing.assert_close(d[..., 2].mean(1) - w[..., 2].mean(1), sp, rtol=0, atol=1e-7)
