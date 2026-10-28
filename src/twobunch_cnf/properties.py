"""Differentiable beam observables on (B, N, 6) clouds in (x,y,z,px,py,pz).

Positions [m], momenta [eV/c]. All functions return (B,) and are differentiable, so
the surrogate's sampled clouds give gradients w.r.t. the knobs for MBRL. The emittance
kernel is reused verbatim from the photoinjector flow (slac-photoinjector-rl) and matches
ParticleGroup.norm_emit_x/4d to ~1e-9.

Two conventions to note vs the campaign's getBeamSpecs quantities:
- Centroids/offsets here use the MEAN (differentiable); getBeamSpecs uses the MEDIAN.
  On real data the transverse offset / angular misalignment differ by ~8-12% (median) and
  up to ~55% (tail, where halo drags the mean). Means are used deliberately for MBRL
  gradients; treat campaign-median parity as approximate.
- The PENT openPMD frame stores z = -(t - <t>)*c, so larger z = earlier arrival. bunch
  spacing below is signed to match the campaign (drive - witness in z == witness - drive
  in arrival time), i.e. positive when the witness trails the drive.
"""
from __future__ import annotations

import torch

from . import ELECTRON_MC2_EV

MC2 = ELECTRON_MC2_EV


def compute_emittance_torch(particles: torch.Tensor) -> dict[str, torch.Tensor]:
    """Geometric 2D/4D/6D emittances (sqrt det Sigma) for (B,N,6) in (x,y,z,px,py,pz)."""
    means = particles.mean(dim=1, keepdim=True)
    centered = particles - means
    cov = torch.bmm(centered.transpose(1, 2), centered) / (particles.shape[1] - 1)
    out: dict[str, torch.Tensor] = {}
    cov_x = cov[:, [0, 3]][:, :, [0, 3]]
    out["x_xp"] = torch.sqrt(torch.abs(torch.linalg.det(cov_x)) + 1e-16)
    cov_y = cov[:, [1, 4]][:, :, [1, 4]]
    out["y_yp"] = torch.sqrt(torch.abs(torch.linalg.det(cov_y)) + 1e-16)
    cov_z = cov[:, [2, 5]][:, :, [2, 5]]
    out["z_delta"] = torch.sqrt(torch.abs(torch.linalg.det(cov_z)) + 1e-16)
    cov4 = cov[:, [0, 3, 1, 4]][:, :, [0, 3, 1, 4]]
    out["fourd"] = torch.sqrt(torch.abs(torch.linalg.det(cov4)) + 1e-16)
    cov6 = cov[:, [0, 3, 1, 4, 2, 5]][:, :, [0, 3, 1, 4, 2, 5]]
    out["sixd"] = torch.sqrt(torch.abs(torch.linalg.det(cov6)) + 1e-16)
    return out


def norm_emits(parts: torch.Tensor) -> dict[str, torch.Tensor]:
    """Normalized emittances (m-rad): geo_2d/mc2, geo_4d/mc2^2 (= ParticleGroup.norm_emit_*)."""
    g = compute_emittance_torch(parts)
    return {
        "norm_emit_x": g["x_xp"] / MC2,
        "norm_emit_y": g["y_yp"] / MC2,
        "norm_emit_4d": g["fourd"] / (MC2 ** 2),
    }


def _energy(parts: torch.Tensor) -> torch.Tensor:  # (B,N) eV
    px, py, pz = parts[..., 3], parts[..., 4], parts[..., 5]
    return torch.sqrt(px * px + py * py + pz * pz + MC2 * MC2)


def _trace_twiss(u: torch.Tensor, up: torch.Tensor):
    """Geometric Twiss (beta, alpha, emit) in trace space (u, u'), reducing over the LAST axis.
    Works for projected (B,N)->(B,) and sliced (B,S,n)->(B,S)."""
    u = u - u.mean(-1, keepdim=True)
    up = up - up.mean(-1, keepdim=True)
    suu = (u * u).mean(-1)
    suup = (u * up).mean(-1)
    spp = (up * up).mean(-1)
    emit = torch.sqrt(torch.clamp(suu * spp - suup * suup, min=1e-30))
    return suu / emit, -suup / emit, emit


def _bmag(beta, alpha, beta0, alpha0):
    """Mismatch parameter vs (beta0, alpha0): 0.5*(gamma0 beta - 2 alpha0 alpha + beta0 gamma);
    == 1 when matched, > 1 otherwise."""
    gamma0 = (1.0 + alpha0 * alpha0) / beta0
    gamma = (1.0 + alpha * alpha) / beta
    return 0.5 * (gamma0 * beta - 2.0 * alpha0 * alpha + beta0 * gamma)


def twiss_bmag(parts: torch.Tensor, beta0: float = 0.5, alpha0: float = 0.0):
    """Projected trace-space Twiss (beta,alpha) and BMAG at PENT, vs golden (beta0, alpha0)."""
    pz = parts[..., 5].clamp_min(1.0)
    bx, ax, _ = _trace_twiss(parts[..., 0], parts[..., 3] / pz)
    by, ay, _ = _trace_twiss(parts[..., 1], parts[..., 4] / pz)
    return {"beta_x": bx, "alpha_x": ax, "beta_y": by, "alpha_y": ay,
            "bmag_x": _bmag(bx, ax, beta0, alpha0), "bmag_y": _bmag(by, ay, beta0, alpha0)}


def slice_twiss_bmag(parts: torch.Tensor, n_slices: int = 5,
                     beta0: float = 0.5, alpha0: float = 0.0):
    """Per-slice (z-binned, equal-count) Twiss/BMAG -> core-slice beta/BMAG + worst-slice BMAG.
    Projected BMAG over the chirped two-bunch beam is inflated, so the per-slice (esp. core,
    highest-current) value is the operational matching metric (Yiheng: witness slice-beta at
    PENT). Differentiable w.r.t. the cloud (the z-sort only produces detached gather indices)."""
    B, N, _ = parts.shape
    s = N // n_slices
    order = torch.argsort(parts[..., 2], dim=1)[:, : s * n_slices]
    g = torch.gather(parts, 1, order.unsqueeze(-1).expand(-1, -1, 6)).reshape(B, n_slices, s, 6)
    pz = g[..., 5].clamp_min(1.0)
    bx, ax, _ = _trace_twiss(g[..., 0], g[..., 3] / pz)   # (B, n_slices)
    by, ay, _ = _trace_twiss(g[..., 1], g[..., 4] / pz)
    bmx, bmy = _bmag(bx, ax, beta0, alpha0), _bmag(by, ay, beta0, alpha0)
    core = n_slices // 2
    return {"slice_beta_x_core": bx[:, core], "slice_beta_y_core": by[:, core],
            "slice_bmag_x_core": bmx[:, core], "slice_bmag_y_core": bmy[:, core],
            "slice_bmag_max": torch.maximum(bmx, bmy).amax(dim=1)}


def per_bunch(parts: torch.Tensor, beta0: float = 0.5, alpha0: float = 0.0,
              n_slices: int = 5) -> dict[str, torch.Tensor]:
    """Per-bunch scalar observables (all (B,)). Includes projected + sliced Twiss/BMAG for the
    PENT matching objective (vs golden beta0=0.5 m, alpha0=0); raw slice-beta is exposed so the
    reward can target an arbitrary witness slice-beta directly."""
    ne = norm_emits(parts)
    e = _energy(parts)
    out = {
        **ne,
        "sigma_x": parts[..., 0].std(dim=1, unbiased=False),
        "sigma_y": parts[..., 1].std(dim=1, unbiased=False),
        "sigma_z": parts[..., 2].std(dim=1, unbiased=False),
        "centroid_x": parts[..., 0].mean(dim=1),
        "centroid_y": parts[..., 1].mean(dim=1),
        "centroid_z": parts[..., 2].mean(dim=1),
        "mean_energy": e.mean(dim=1),
        "energy_spread": e.std(dim=1, unbiased=False),
    }
    out.update(twiss_bmag(parts, beta0, alpha0))
    out.update(slice_twiss_bmag(parts, n_slices, beta0, alpha0))
    return out


def _slopes(parts: torch.Tensor):  # xp, yp = px/pz, py/pz centroids (B,)
    pz = parts[..., 5].clamp_min(1.0)
    return (parts[..., 3] / pz).mean(dim=1), (parts[..., 4] / pz).mean(dim=1)


def inter_bunch(drive: torch.Tensor, witness: torch.Tensor) -> dict[str, torch.Tensor]:
    """Drive<->witness relative observables (the campaign 'quality' quantities)."""
    cz_d, cz_w = drive[..., 2].mean(1), witness[..., 2].mean(1)
    cx_d, cx_w = drive[..., 0].mean(1), witness[..., 0].mean(1)
    cy_d, cy_w = drive[..., 1].mean(1), witness[..., 1].mean(1)
    e_d, e_w = _energy(drive).mean(1), _energy(witness).mean(1)
    xpd, ypd = _slopes(drive)
    xpw, ypw = _slopes(witness)
    return {
        # stored z = -(t-<t>)*c, so (drive_z - witness_z) matches the campaign's signed
        # bunchSpacing (witness_zCentroid - drive_zCentroid in arrival-time*c): + = witness trails.
        "bunch_spacing": cz_d - cz_w,
        "energy_difference": e_d - e_w,
        "transverse_offset": torch.sqrt((cx_d - cx_w) ** 2 + (cy_d - cy_w) ** 2 + 1e-30),
        "angular_misalignment": torch.sqrt((xpd - xpw) ** 2 + (ypd - ypw) ** 2 + 1e-30),
    }
