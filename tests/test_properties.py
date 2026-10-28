"""Unit tests for the new differentiable matching observables (BMAG, slice-beta).

Synthetic Gaussian clouds with a prescribed trace-space beta let us check beta/BMAG against
the closed form, slice-vs-projected consistency, and gradient flow (for MBRL). No Tao needed.
"""
import math

import torch

from twobunch_cnf.properties import (
    twiss_bmag, slice_twiss_bmag, per_bunch, _bmag,
)

P0 = 1.0e10        # eV/c reference momentum (pz)
EPS = 1.0e-9       # geometric emittance scale (m-rad)


def _cloud(beta_x, beta_y, n=20000, seed=0):
    """Uncorrelated (alpha=0) Gaussian cloud with prescribed trace-space beta_x, beta_y."""
    g = torch.Generator().manual_seed(seed)
    sx, sxp = math.sqrt(EPS * beta_x), math.sqrt(EPS / beta_x)
    sy, syp = math.sqrt(EPS * beta_y), math.sqrt(EPS / beta_y)
    x = torch.randn(n, generator=g) * sx
    xp = torch.randn(n, generator=g) * sxp
    y = torch.randn(n, generator=g) * sy
    yp = torch.randn(n, generator=g) * syp
    z = torch.randn(n, generator=g) * 1e-4
    pz = torch.full((n,), P0)
    return torch.stack([x, y, z, xp * pz, yp * pz, pz], dim=-1)


def test_bmag_closed_form():
    assert abs(_bmag(torch.tensor(0.5), torch.tensor(0.0), 0.5, 0.0) - 1.0) < 1e-6
    assert abs(_bmag(torch.tensor(2.0), torch.tensor(0.0), 0.5, 0.0) - 2.125) < 1e-6


def test_projected_twiss_and_bmag():
    parts = torch.stack([_cloud(0.5, 0.5, seed=1), _cloud(2.0, 0.5, seed=2)], dim=0)  # (2,N,6)
    tb = twiss_bmag(parts, beta0=0.5, alpha0=0.0)
    # matched cloud -> beta=0.5, BMAG=1; mismatched -> beta=2, BMAG=2.125
    assert abs(tb["beta_x"][0].item() - 0.5) < 0.02
    assert abs(tb["bmag_x"][0].item() - 1.0) < 0.03
    assert abs(tb["beta_x"][1].item() - 2.0) < 0.1
    assert abs(tb["bmag_x"][1].item() - 2.125) < 0.1
    assert abs(tb["alpha_x"][0].item()) < 0.05      # uncorrelated -> alpha ~ 0


def test_slice_matches_projected_when_z_independent():
    # transverse is z-independent here, so every slice ~ the projected value
    parts = _cloud(0.5, 1.0, n=20000, seed=3).unsqueeze(0)
    st = slice_twiss_bmag(parts, n_slices=5, beta0=0.5, alpha0=0.0)
    assert abs(st["slice_beta_x_core"][0].item() - 0.5) < 0.05
    assert abs(st["slice_beta_y_core"][0].item() - 1.0) < 0.1
    assert st["slice_bmag_max"][0].item() >= 1.0


def test_per_bunch_exposes_new_keys():
    out = per_bunch(_cloud(0.5, 0.5, n=4096, seed=4).unsqueeze(0))
    for k in ("bmag_x", "bmag_y", "beta_x", "slice_beta_x_core",
              "slice_bmag_x_core", "slice_bmag_max"):
        assert k in out and out[k].shape == (1,)


def test_differentiable():
    parts = _cloud(1.5, 0.5, n=8000, seed=5).unsqueeze(0).clone().requires_grad_(True)
    loss = twiss_bmag(parts)["bmag_x"].sum() + slice_twiss_bmag(parts)["slice_bmag_max"].sum()
    loss.backward()
    assert parts.grad is not None and torch.isfinite(parts.grad).all()
