"""TwoBunchFlow: the two-bunch conditional normalizing flow, as a Lightning module.

One rational-quadratic-spline coupling flow models both bunches, told apart by a bunch index k (0 = drive,
1 = witness). A whitening network maps (knobs, k) to each bunch's mean mu_k and covariance factor L_k, so
the flow only models the standardized residual w = L_k^-1 (x_std - mu_k), and x_std = mu_k + L_k w.
Separate heads on the knobs predict p_surv, the probability that the witness survives, and each bunch's
transmitted fraction T, in two parts: the probability that T = 1, and T when there is loss. An optional
beam latent, drawn once per beam, carries the variation between simulated beams that the knobs don't
explain.

Sampling is reparameterized, so observables of the samples are differentiable with respect to the knobs.
The coupling layers come from slac-photoinjector-rl's single-bunch flow.

Only the features of the shipped model (training run C_B3) are kept. The experimental options removed from
it (affine coupling, a single-sigmoid T head, centroid / chirp-curve / backbone heads, single-bunch
training) are on the archive/unpruned-7ac3617 branch. `twobunch_cnf.migrate` converts checkpoints that
record those options, as long as they are off.
"""

from __future__ import annotations

import inspect
import math

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn

from . import LATENT_DIM, __version__
from .properties import compute_emittance_torch, inter_bunch, per_bunch

_LOG_2PI = math.log(2.0 * math.pi)

# ---- rational-quadratic spline (Durkan et al. 2019; linear tails) -----------
_MIN_W = _MIN_H = _MIN_D = 1e-3


def _rqs_1d(x, uw, uh, ud, inverse, B):
    """Monotonic RQ spline on [-B,B] with linear identity tails. Elementwise over any
    leading shape: x:(...,), uw/uh:(...,K), ud:(...,K-1). Returns (outputs, logabsdet).

    Computed densely (no boolean-mask indexing) and the tail is selected with torch.where,
    which is GPU-friendly (no host syncs / dynamic shapes). The in-spline arithmetic stays
    finite for tail inputs because bin_idx and theta are clamped, so the masked-out branch
    never produces NaN that torch.where could leak into the gradient."""
    K = uw.shape[-1]
    inside = (x >= -B) & (x <= B)

    widths = torch.softmax(uw, dim=-1)
    widths = _MIN_W + (1 - _MIN_W * K) * widths
    cw = torch.nn.functional.pad(torch.cumsum(widths, -1), (1, 0))
    cw = 2 * B * cw - B
    cw[..., 0], cw[..., -1] = -B, B
    widths = cw[..., 1:] - cw[..., :-1]

    heights = torch.softmax(uh, dim=-1)
    heights = _MIN_H + (1 - _MIN_H * K) * heights
    ch = torch.nn.functional.pad(torch.cumsum(heights, -1), (1, 0))
    ch = 2 * B * ch - B
    ch[..., 0], ch[..., -1] = -B, B
    heights = ch[..., 1:] - ch[..., :-1]

    deriv = _MIN_D + torch.nn.functional.softplus(ud)
    deriv = torch.nn.functional.pad(deriv, (1, 1), value=1.0)  # linear tails: boundary slope 1

    edges = ch if inverse else cw
    bin_idx = (torch.sum(x[..., None] >= edges, dim=-1) - 1).clamp(0, K - 1)
    g = lambda t: t.gather(-1, bin_idx[..., None]).squeeze(-1)
    cwk, wk = g(cw), g(widths)
    chk, hk = g(ch), g(heights)
    dk = g(deriv)
    dk1 = deriv.gather(-1, (bin_idx + 1)[..., None]).squeeze(-1)
    s = hk / wk

    if inverse:
        dy = x - chk
        a = dy * (dk1 + dk - 2 * s) + hk * (s - dk)
        b = hk * dk - dy * (dk1 + dk - 2 * s)
        c = -s * dy
        # clamp_min to a small positive (not 0): sqrt'(disc) = 1/(2*sqrt(disc)) -> inf as disc->0
        # (a particle landing on a knot), a finite-forward / NaN-backward trap. A tiny floor
        # keeps the gradient finite while barely perturbing the inverse value.
        disc = (b**2 - 4 * a * c).clamp_min(1e-10)
        theta = (2 * c / (-b - torch.sqrt(disc))).clamp(0.0, 1.0)
        o = theta * wk + cwk
        tt = theta * (1 - theta)
        denom = s + (dk1 + dk - 2 * s) * tt
        dnum = s**2 * (dk1 * theta**2 + 2 * s * tt + dk * (1 - theta) ** 2)
        ld = -(torch.log(dnum) - 2 * torch.log(denom))
    else:
        theta = ((x - cwk) / wk).clamp(0.0, 1.0)
        tt = theta * (1 - theta)
        denom = s + (dk1 + dk - 2 * s) * tt
        o = chk + hk * (s * theta**2 + dk * tt) / denom
        dnum = s**2 * (dk1 * theta**2 + 2 * s * tt + dk * (1 - theta) ** 2)
        ld = torch.log(dnum) - 2 * torch.log(denom)

    return torch.where(inside, o, x), torch.where(inside, ld, torch.zeros_like(ld))


class RQSCouplingLayer(nn.Module):
    """Conditional RQ-spline coupling layer; same interface as the affine layer."""

    def __init__(self, dim, cond_dim, mlp_dim=64, n_bins=12, tail_bound=5.0, reverse_mask=False):
        super().__init__()
        self.dim, self.d = dim, dim // 2
        self.n_t = dim - self.d
        self.K, self.B, self.reverse_mask = n_bins, tail_bound, reverse_mask
        self.net = nn.Sequential(
            nn.Linear(self.d + cond_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, self.n_t * (3 * n_bins - 1)),
        )

    def _split(self, z):
        return (z[:, self.d :], z[:, : self.d]) if self.reverse_mask else (z[:, : self.d], z[:, self.d :])

    def _join(self, a, b):
        return torch.cat([a, b], dim=1) if self.reverse_mask else torch.cat([b, a], dim=1)

    def _params(self, z1, cond):
        p = self.net(torch.cat([z1, cond], dim=1)).view(-1, self.n_t, 3 * self.K - 1)
        return p[..., : self.K], p[..., self.K : 2 * self.K], p[..., 2 * self.K :]

    def _apply_spline(self, z1, z2, cond, inverse):
        # uw,uh: (M, n_t, K); ud: (M, n_t, K-1). Transform all n_t dims in one vectorized
        # _rqs_1d call (no Python loop over dims) -> (M, n_t).
        uw, uh, ud = self._params(z1, cond)
        out, ld = _rqs_1d(z2, uw, uh, ud, inverse, self.B)
        return out, ld.sum(dim=-1)

    def forward(self, z, cond):
        z1, z2 = self._split(z)
        out, lad = self._apply_spline(z1, z2, cond, inverse=False)
        return self._join(out, z1), lad

    def inverse(self, z, cond):
        z1, z2 = self._split(z)
        out, lad = self._apply_spline(z1, z2, cond, inverse=True)
        return self._join(out, z1), lad


class TwoBunchFlow(L.LightningModule):
    def __init__(
        self,
        condition_dim=8,
        latent_dim=LATENT_DIM,
        hidden_dim=128,
        n_layers=16,
        n_bins=16,
        tail_bound=8.0,
        lr=1e-4,
        weight_decay=1e-5,
        lr_schedule="plateau",
        t_max=300,
        w_cls=1.0,
        w_tr=1.0,
        w_emit=0.25,
        w_emit_z=0.0,
        w_emit_4d=0.0,
        w_emit_6d=0.0,
        w_cov=0.5,
        nll_dim_norm=6.0,
        # centroid matching (0 = off): w_mean = per-bunch sample-mean error of all 6 coordinates in the
        # standardized frame (the moment losses above are centred, and the NLL barely feels a centroid error
        # of ~0.1 sigma); w_spacing = Huber on the physical bunch spacing (difference of the z centroids) in
        # units of spacing_scale_m
        w_mean=0.0,
        w_spacing=0.0,
        spacing_scale_m=1e-5,
        # paired injector-effect loss (0 = off): each training step draws `pair_batch` (point, reference)
        # row pairs from self.pair_source (set by train.py; same linac knobs, injector point vs golden) and
        # Huber-penalizes the error of the predicted spacing DIFFERENCE (common random numbers) in units of
        # spacing_scale_m; pair_particles per bunch
        w_pair=0.0,
        pair_batch=32,
        pair_particles=512,
        # two-part transmission head: a separate head predicts P(T = 1) (BCE), the T logits are fitted on
        # lossy rows only, and the output is T = T_tail + (1 - T_tail) * sigmoid(t_sharp * full_logit)
        # (median-like for t_sharp > 1); t_head_lr = its own lr (a warm-started fine-tune's lr is tiny)
        t_sharp=4.0,
        t_head_lr=None,
        # beam-level latent (0 = off): each beam (row, bunch) carries a latent s of this dimension that the
        # flow is conditioned on (h' = h(knobs) + W s, W zero-initialized). Training: an amortized set
        # encoder over the row's particles gives q(s | cloud, knobs); loss = NLL + beta * KL(q || p(s | knobs)) /
        # (P * nll_dim_norm) (an ELBO per particle). Sampling: s ~ p(s | knobs), one draw per beam, so each
        # sampled beam can be sharp while the backbone the knobs cannot pin down varies between draws (the
        # knobs do not determine the beam's backbone, its curve in (z, pz), as sharply as the beam is thin).
        beam_latent_dim=0,
        beam_latent_beta=1.0,
        beam_latent_free_bits=0.0,
        beam_latent_lr=None,
        set_hidden=128,
        # w_sw = sliced-Wasserstein-2 between the sampled and the data cloud in (z, pz) + 1-D W2 on the z and
        # pz marginals, each row in its own data-standardized frame (NLL hedges an uncertain backbone by
        # blurring; W2 is minimized by a sharp barycenter); w_logsz = |log sigma_z,pred / sigma_z,true|
        # (full-particle truth if the batch carries it), x (1 + logsz_compressed_boost) for rows with
        # sigma_z < 30 um
        w_sw=0.0,
        sw_proj=32,
        w_logsz=0.0,
        logsz_compressed_boost=2.0,
        n_aux_particles=512,
        # per-bunch (de)standardization + knob bounds (from preprocess _norm.json)
        drive_mean=None,
        drive_std=None,
        witness_mean=None,
        witness_std=None,
        knob_low=None,
        knob_high=None,
        # knob names + treaty point of the training frame
        knob_keys=None,
        treaty_point=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.latent_dim = latent_dim

        # shared encoder takes (knobs, one-hot bunch index)
        self.encoder = nn.Sequential(
            nn.Linear(condition_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # whitening: per-bunch MLP h -> mu (6) + Cholesky params (6 diag + 15 off-diag).
        # Separate head per bunch (drive-round vs witness-thin no longer share one map) + an
        # MLP (not a single Linear) so it can track per-knob anisotropy.
        self._n_offdiag = latent_dim * (latent_dim - 1) // 2
        n_white = 2 * latent_dim + self._n_offdiag
        self.whiten_heads = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_white))
                for _ in range(2)
            ]
        )
        self.flows = nn.ModuleList(
            [
                RQSCouplingLayer(
                    latent_dim, hidden_dim, n_bins=n_bins, tail_bound=tail_bound, reverse_mask=(i % 2 == 1)
                )
                for i in range(n_layers)
            ]
        )
        # feasibility heads (knobs only): [logit p_surv, logit T_d, logit T_w]
        self.feas_head = nn.Sequential(nn.Linear(condition_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3))
        # [logit P(T_drive = 1), logit P(T_witness = 1)]
        self.t_full_head = nn.Sequential(nn.Linear(condition_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2))
        self.pair_source = None

        if beam_latent_dim:
            dl, hs = int(beam_latent_dim), int(set_hidden)
            self.set_phi = nn.ModuleList(
                [nn.Sequential(nn.Linear(latent_dim, hs), nn.SiLU(), nn.Linear(hs, hs), nn.SiLU()) for _ in range(2)]
            )
            self.set_rho = nn.ModuleList(
                [nn.Sequential(nn.Linear(2 * hs + hidden_dim, hs), nn.SiLU(), nn.Linear(hs, 2 * dl)) for _ in range(2)]
            )
            self.latent_prior = nn.ModuleList()
            self.latent_in = nn.ModuleList()
            for _ in range(2):
                last = nn.Linear(hidden_dim, 2 * dl)
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)  # prior N(0, 1) at init
                self.latent_prior.append(nn.Sequential(nn.Linear(condition_dim, hidden_dim), nn.SiLU(), last))
                lin = nn.Linear(dl, hidden_dim)
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)  # no effect at init: identical to the plain flow
                self.latent_in.append(lin)
            self._kl_log = {}

        tril = torch.tril_indices(latent_dim, latent_dim, offset=-1)
        self.register_buffer("_tril_r", tril[0])
        self.register_buffer("_tril_c", tril[1])

        def buf(v, default):
            return torch.tensor(default if v is None else v).float()

        self.register_buffer("drive_mean", buf(drive_mean, [0.0] * latent_dim))
        self.register_buffer("drive_std", buf(drive_std, [1.0] * latent_dim))
        self.register_buffer("witness_mean", buf(witness_mean, [0.0] * latent_dim))
        self.register_buffer("witness_std", buf(witness_std, [1.0] * latent_dim))
        self.register_buffer("knob_low", buf(knob_low, [0.0] * condition_dim))
        self.register_buffer("knob_high", buf(knob_high, [1.0] * condition_dim))

    # ---- conditioning + whitening -----------------------------------------
    def _encode(self, knobs, k):
        oh = torch.zeros(knobs.shape[0], 2, device=knobs.device, dtype=knobs.dtype)
        oh[:, k] = 1.0
        return self.encoder(torch.cat([knobs, oh], dim=1))

    def _whiten(self, h, k):
        out = self.whiten_heads[k](h)
        mu = out[:, : self.latent_dim]
        diag = F.softplus(out[:, self.latent_dim : 2 * self.latent_dim]) + 1e-3
        # Bound off-diagonals so the Cholesky factor stays well-conditioned (NLL uses
        # L^{-1}; an ill-conditioned L would blow up w and the loss). |off| <= 5.
        off = 5.0 * torch.tanh(out[:, 2 * self.latent_dim :])
        Lm = torch.diag_embed(diag)
        Lm[:, self._tril_r, self._tril_c] = off
        return mu, Lm

    def latent_prior_params(self, knobs, k):
        mu, lv = self.latent_prior[k](knobs).chunk(2, -1)
        return mu, lv.clamp(-8.0, 4.0)

    def latent_posterior_params(self, parts_std, knobs, k, h):
        f = self.set_phi[k](parts_std)  # (M, P, hs)
        g = torch.cat([f.mean(1), f.amax(1), h], -1)
        mu, lv = self.set_rho[k](g).chunk(2, -1)
        return mu, lv.clamp(-10.0, 4.0)

    def _cond_h(self, h, k, s):
        return h if s is None else h + self.latent_in[k](s)

    def _scaler(self, k):
        return (self.drive_mean, self.drive_std) if k == 0 else (self.witness_mean, self.witness_std)

    @staticmethod
    def _cov(parts):  # (M,P,6) -> (M,6,6) sample covariance
        c = parts - parts.mean(dim=1, keepdim=True)
        return torch.bmm(c.transpose(1, 2), c) / (parts.shape[1] - 1)

    # ---- flow primitives ---------------------------------------------------
    def _forward_enc(self, z, cond):
        ld = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for fl in self.flows:
            z, d = fl(z, cond)
            ld = ld + d
        return z, ld

    def _inverse_enc(self, x, cond):
        ld = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for fl in reversed(self.flows):
            x, d = fl.inverse(x, cond)
            ld = ld + d
        return x, ld

    # ---- NLL for one bunch -------------------------------------------------
    def bunch_nll(self, parts_std, knobs, k):
        """parts_std: (M,P,6) standardized; knobs: (M,n_knob). Returns scalar mean NLL."""
        m, p = parts_std.shape[0], parts_std.shape[1]
        h = self._encode(knobs, k)
        kl_term = None
        if self.hparams.get("beam_latent_dim", 0):
            mq, lq = self.latent_posterior_params(parts_std, knobs, k, h)
            s_ = mq + torch.exp(0.5 * lq) * torch.randn_like(mq)
            mp, lp = self.latent_prior_params(knobs, k)
            kl = 0.5 * (lp - lq + (torch.exp(lq) + (mq - mp) ** 2) / torch.exp(lp) - 1.0)  # (M, d)
            fb = float(self.hparams.get("beam_latent_free_bits", 0.0))
            kl_row = (kl.clamp_min(fb) if fb > 0 else kl).sum(-1)
            self._kl_log[k] = kl.sum(-1).mean().detach()
            kl_term = float(self.hparams.get("beam_latent_beta", 1.0)) * kl_row.mean() / (p * self.hparams.nll_dim_norm)
            h = self._cond_h(h, k, s_)
        mu, Lm = self._whiten(h, k)
        xc = parts_std - mu.unsqueeze(1)  # (M,P,6)
        w = torch.linalg.solve_triangular(Lm, xc.transpose(1, 2), upper=False).transpose(1, 2)
        h_flat = h.unsqueeze(1).expand(m, p, -1).reshape(m * p, -1)
        z, ld_flow = self._inverse_enc(w.reshape(m * p, self.latent_dim), h_flat)
        logpz = (-0.5 * z**2 - 0.5 * _LOG_2PI).sum(dim=1)
        ld_white = -torch.log(torch.diagonal(Lm, dim1=-2, dim2=-1)).sum(dim=1)  # (M,)
        logp = logpz + ld_flow + ld_white.repeat_interleave(p)
        nll = -logp.mean() / self.hparams.nll_dim_norm
        return nll if kl_term is None else nll + kl_term

    # ---- sampling (reparameterized, differentiable wrt knobs) --------------
    def sample_bunch(self, knobs, k, n, physical=True, generator=None, z=None, s=None):
        """(B,n,6) cloud for bunch k. Differentiable w.r.t. knobs.

        knobs: (B, n_knob), each knob scaled to [0, 1] over its training range (see
           `TwoBunchCNF.knobs_norm`). k: 0 = drive, 1 = witness. physical: False returns
           the standardized coordinates instead of m and eV/c.
        generator: optional torch.Generator for reproducible draws.
        z: optional frozen sample of the flow's Gaussian base distribution (not the
           longitudinal z, nor the beam latent s), of shape (n, latent_dim), SHARED across the
           batch (common random numbers): repeated calls at different knobs give
           directly comparable clouds and a loss that is deterministic in the knobs.
           Default (None) keeps the fresh per-call draw (unbiased reparameterized
           gradients for RL).
        s: optional beam latent (B, beam_latent_dim) (beam-latent models); default: one draw per
           beam from the knob-conditioned prior p(s | knobs).
        """
        b = knobs.shape[0]
        h = self._encode(knobs, k)
        if self.hparams.get("beam_latent_dim", 0):  # one latent per beam: s ~ p(s | knobs) unless given
            if s is None:
                mp, lp = self.latent_prior_params(knobs, k)
                eps = torch.randn(mp.shape, device=knobs.device, dtype=knobs.dtype, generator=generator)
                s = mp + torch.exp(0.5 * lp) * eps
            h = self._cond_h(h, k, s)
        h_flat = h.unsqueeze(1).expand(b, n, -1).reshape(b * n, -1)
        if z is None:
            z = torch.randn(b * n, self.latent_dim, device=knobs.device, dtype=knobs.dtype, generator=generator)
        else:
            if z.shape != (n, self.latent_dim):
                raise ValueError(f"z must be (n, latent_dim)=({n},{self.latent_dim}), got {tuple(z.shape)}")
            # share the (n, ld) buffer across the batch (a naive reshape(b*n, ld)
            # would crash for any b > 1 — the batched-scan fit case)
            z = z.to(device=knobs.device, dtype=knobs.dtype)
            z = z.unsqueeze(0).expand(b, n, self.latent_dim).reshape(b * n, self.latent_dim)
        w, _ = self._forward_enc(z, h_flat)
        w = w.reshape(b, n, self.latent_dim)
        mu, Lm = self._whiten(h, k)
        x_std = mu.unsqueeze(1) + torch.einsum("bij,bnj->bni", Lm, w)
        if not physical:
            return x_std
        mean, std = self._scaler(k)
        return x_std * std + mean

    def feasibility(self, knobs):
        logits = self.feas_head(knobs)
        s = torch.sigmoid(logits)
        full = torch.sigmoid(self.hparams.get("t_sharp", 4.0) * self.t_full_head(knobs))
        t_d = s[:, 1] + (1 - s[:, 1]) * full[:, 0]
        t_w = s[:, 2] + (1 - s[:, 2]) * full[:, 1]
        return s[:, 0], t_d, t_w  # p_surv(witness), T_drive, T_witness

    # ---- observables, for optimization and model-based RL -----------------
    def observables(self, knobs_norm, n=None, generator=None, z_drive=None, z_witness=None):
        """Differentiable dict of per-bunch, drive-witness and survival observables, each (B,).
        knobs_norm: (B, n_knob), each knob scaled to [0, 1] over its training range.
        n: particles sampled per bunch (default: the n_aux_particles hparam).
        z_drive/z_witness: optional frozen (n, latent_dim) base samples, one per bunch (see
        sample_bunch: common random numbers, for deterministic fits)."""
        n = n or self.hparams.n_aux_particles
        drive = self.sample_bunch(knobs_norm, 0, n, generator=generator, z=z_drive)
        witness = self.sample_bunch(knobs_norm, 1, n, generator=generator, z=z_witness)
        p_surv, t_d, t_w = self.feasibility(knobs_norm)
        out = {}
        for name, parts in (("drive", drive), ("witness", witness)):
            for kk, vv in per_bunch(parts).items():
                out[f"{name}_{kk}"] = vv
        out.update(inter_bunch(drive, witness))
        out.update(p_surv=p_surv, T_drive=t_d, T_witness=t_w)
        return out

    # ---- training ----------------------------------------------------------
    def _step(self, batch, prefix):
        knobs = batch["knobs"]
        dd, wd = batch["drive_density"], batch["witness_density"]
        dv, wv = batch["drive_present"], batch["witness_viable"]
        hp = self.hparams
        logs, loss, n_skip = {}, knobs.new_zeros(()), 0

        def add(term, w, name):
            """Accumulate w*term, skipping (and counting) a non-finite term so a rare NaN/Inf
            (e.g. the RQS inverse at a knot for one particle) can't poison the shared weights."""
            nonlocal loss, n_skip
            if torch.isfinite(term):
                loss = loss + w * term
                logs[f"{prefix}_{name}"] = term
            else:
                n_skip += 1

        if dd.any():
            add(self.bunch_nll(batch["drive"][dd], knobs[dd], 0), 1.0, "nll_drive")
        if wd.any():
            add(self.bunch_nll(batch["witness"][wd], knobs[wd], 1), 1.0, "nll_witness")
        if hp.get("beam_latent_dim", 0):
            for kk, v in getattr(self, "_kl_log", {}).items():
                logs[f"{prefix}_kl_{'drive' if kk == 0 else 'witness'}"] = v

        # feasibility heads -- BCE-with-logits is numerically stable and never asserts on an
        # out-of-[0,1] input (a NaN p_surv from upstream divergence used to hard-crash here).
        feas_logits = self.feas_head(knobs)
        t_d, t_w = torch.sigmoid(feas_logits[:, 1]), torch.sigmoid(feas_logits[:, 2])
        add(F.binary_cross_entropy_with_logits(feas_logits[:, 0], wv.float()), hp.w_cls, "bce_surv")
        full_logits = self.t_full_head(knobs)
        for j, (mask, key, tt) in enumerate(((dv, "d", t_d), (wv, "w", t_w))):
            if not mask.any():
                continue
            frac = batch["drive_frac" if j == 0 else "witness_frac"]
            is_full = frac >= 1.0 - 1e-4
            add(
                F.binary_cross_entropy_with_logits(full_logits[mask, j], is_full[mask].float()),
                hp.w_tr,
                f"bce_T{key}_full",
            )
            tail = mask & ~is_full
            if tail.any():
                add(F.mse_loss(tt[tail], frac[tail]), hp.w_tr, f"mse_T{key}_tail")
            with torch.no_grad():
                _, td_, tw_ = self.feasibility(knobs)
                logs[f"{prefix}_mae_T{key}"] = ((td_ if j == 0 else tw_)[mask] - frac[mask]).abs().mean()

        # per-bunch moment matching on the sampled cloud (standardized frame), one shared draw:
        # 2D emittances (incl. z_delta = sqrt det cov_zpz, the LPS thinness lever), 4D/6D dets,
        # and the relative 6x6 covariance -- the NLL alone tolerates a too-round Sigma_k.
        any_emit = hp.w_emit > 0 or hp.w_emit_z > 0 or hp.w_emit_4d > 0 or hp.w_emit_6d > 0
        w_mean, w_sp = float(hp.get("w_mean", 0.0)), float(hp.get("w_spacing", 0.0))
        zc = {}  # k -> (mask, predicted z centroid, true z centroid) [m]
        w_sw, w_lsz = float(hp.get("w_sw", 0.0)), float(hp.get("w_logsz", 0.0))
        if any_emit or hp.w_cov > 0 or w_mean > 0 or w_sp > 0 or w_sw > 0 or w_lsz > 0:
            eps = 1e-9
            for k, mask, key in ((0, dd, "drive"), (1, wd, "witness")):
                if not mask.any():
                    continue
                true = batch[key][mask]
                pred = self.sample_bunch(knobs[mask], k, true.shape[1], physical=False)
                tmean = true.mean(1)
                if f"{key}_ctrue" in batch:  # full-particle centroids where known
                    ct = batch[f"{key}_ctrue"][mask]
                    tmean = torch.where(torch.isfinite(ct), ct, tmean)
                if w_mean > 0:
                    dmu = pred.mean(1) - tmean  # (M, 6) standardized
                    add(F.smooth_l1_loss(dmu, torch.zeros_like(dmu), beta=0.05), w_mean, f"mean_{key}")
                if w_sp > 0:
                    mean_k, std_k = self._scaler(k)
                    zc[k] = (mask, pred[..., 2].mean(1) * std_k[2] + mean_k[2], tmean[:, 2] * std_k[2] + mean_k[2])
                if w_sw > 0:
                    tz = true[..., [2, 5]]
                    mu, sd = tz.mean(1, keepdim=True), tz.std(1, keepdim=True).clamp_min(1e-6)
                    a_, b_ = (tz - mu) / sd, (pred[..., [2, 5]] - mu) / sd
                    th = torch.rand(int(hp.get("sw_proj", 32)), device=a_.device) * math.pi
                    dirs = torch.stack([torch.cos(th), torch.sin(th)], 0)  # (2, K)
                    pa, pb = (a_ @ dirs).sort(1).values, (b_ @ dirs).sort(1).values
                    sw = ((pa - pb) ** 2).mean()
                    marg = ((a_.sort(1).values - b_.sort(1).values) ** 2).mean()
                    add(sw + marg, w_sw, f"sw_{key}")
                if w_lsz > 0 and k == 0:
                    mean_k, std_k = self._scaler(k)
                    szp = pred[..., 2].std(1) * std_k[2]
                    szt = batch["drive_sztrue"][mask] if "drive_sztrue" in batch else true[..., 2].std(1) * std_k[2]
                    szt = torch.where(torch.isfinite(szt), szt, true[..., 2].std(1) * std_k[2])
                    wgt = 1.0 + float(hp.get("logsz_compressed_boost", 2.0)) * (szt < 30e-6).float()
                    add(
                        (wgt * torch.abs(torch.log(szp.clamp_min(1e-9) / szt.clamp_min(1e-9)))).mean(),
                        w_lsz,
                        "logsz_drive",
                    )
                if any_emit:
                    ep, et = compute_emittance_torch(pred), compute_emittance_torch(true)
                    ldiff = lambda q: torch.abs(torch.log10(ep[q] + eps) - torch.log10(et[q] + eps)).mean()
                    if hp.w_emit > 0:  # transverse 2D (x-px, y-py)
                        add((ldiff("x_xp") + ldiff("y_yp")) / 2, hp.w_emit, f"emit_{key}")
                    if hp.w_emit_z > 0:  # longitudinal/LPS 2D (z-pz), own (undiluted) weight
                        add(ldiff("z_delta"), hp.w_emit_z, f"emitz_{key}")
                    if hp.w_emit_4d > 0:
                        add(ldiff("fourd"), hp.w_emit_4d, f"emit4d_{key}")
                    if hp.w_emit_6d > 0:
                        add(ldiff("sixd"), hp.w_emit_6d, f"emit6d_{key}")
                if hp.w_cov > 0:
                    cp, ct = self._cov(pred), self._cov(true)
                    lc = (2 * torch.abs(cp - ct) / (torch.abs(cp) + torch.abs(ct) + 1e-6)).mean()
                    add(lc, hp.w_cov, f"cov_{key}")

        w_pair = float(hp.get("w_pair", 0.0))
        if w_pair > 0 and prefix == "train" and self.pair_source is not None:
            add(self._pair_loss(), w_pair, "pair")

        if w_sp > 0 and 0 in zc and 1 in zc:
            both = dd & wd
            if both.any():
                full = lambda m, v: torch.zeros(knobs.shape[0], device=v.device, dtype=v.dtype).masked_scatter(m, v)
                (md, pd_, td), (mw, pw, tw) = zc[0], zc[1]
                sp_p = (full(md, pd_) - full(mw, pw))[both]
                sp_t = (full(md, td) - full(mw, tw))[both]
                scale = float(hp.get("spacing_scale_m", 1e-5))
                add(F.smooth_l1_loss(sp_p / scale, sp_t / scale, beta=0.5), w_sp, "spacing")
                logs[f"{prefix}_spacing_mae_um"] = (sp_p - sp_t).abs().mean().detach() * 1e6

        logs[f"{prefix}_loss"] = loss
        if n_skip:
            logs[f"{prefix}_nonfinite_skipped"] = float(n_skip)
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=(prefix == "val"), batch_size=knobs.shape[0])
        return loss

    def _pair_loss(self):
        """Huber on (pred - true) of the spacing difference of a random batch of (point, ref) pairs."""
        kp, kr, d_true = self.pair_source(int(self.hparams.get("pair_batch", 32)), self.device)
        n = int(self.hparams.get("pair_particles", 512))
        zd = torch.randn(n, self.latent_dim, device=self.device)
        zw = torch.randn(n, self.latent_dim, device=self.device)
        knobs = torch.cat([kp, kr], 0)
        d = self.sample_bunch(knobs, 0, n, z=zd)
        w = self.sample_bunch(knobs, 1, n, z=zw)
        sp = d[..., 2].mean(1) - w[..., 2].mean(1)  # physical spacing [m]
        m = kp.shape[0]
        scale = float(self.hparams.get("spacing_scale_m", 1e-5))
        return F.smooth_l1_loss((sp[:m] - sp[m:]) / scale, d_true / scale, beta=0.5)

    def on_save_checkpoint(self, checkpoint):
        # marks checkpoints written by this package (twobunch_cnf.migrate treats them as native)
        checkpoint["twobunch_cnf_version"] = __version__

    def on_load_checkpoint(self, checkpoint):
        """Strict hparams: Lightning silently drops saved hparams that __init__ does not accept, so a checkpoint
        written with an option removed from main (or added on an experiment branch) would load as a different
        model. Refuse it; `twobunch_cnf.migrate` converts checkpoints whose removed options are off."""
        known = set(inspect.signature(type(self).__init__).parameters) - {"self"}
        unknown = sorted(set(checkpoint.get("hyper_parameters", {})) - known)
        if unknown:
            raise ValueError(
                f"checkpoint hparams {unknown} are not options of this TwoBunchFlow; convert the "
                "checkpoint once with `python -m twobunch_cnf.migrate <ckpt> --out <new.ckpt>`"
            )

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def configure_optimizers(self):
        t_lr = self.hparams.get("t_head_lr")
        heads = []
        if t_lr:
            # the fresh two-part T head trains at its own lr (a warm-started fine-tune's lr is tiny)
            heads.append((list(self.t_full_head.parameters()), float(t_lr)))
        bl_lr = self.hparams.get("beam_latent_lr")
        if self.hparams.get("beam_latent_dim", 0) and bl_lr:
            heads.append(
                (
                    list(self.set_phi.parameters())
                    + list(self.set_rho.parameters())
                    + list(self.latent_prior.parameters())
                    + list(self.latent_in.parameters()),
                    float(bl_lr),
                )
            )
        if heads:
            ids = {id(p) for ps, _ in heads for p in ps}
            groups = [{"params": [p for p in self.parameters() if id(p) not in ids]}] + [
                {"params": ps, "lr": lr} for ps, lr in heads
            ]
            opt = torch.optim.Adam(groups, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        else:
            opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        if self.hparams.lr_schedule == "cosine":
            # smooth anneal to ~lr/100 over t_max epochs -- for warm-started fine-tuning phases
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.hparams.t_max, eta_min=self.hparams.lr * 0.01
            )
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"}}


def load_flow(path, map_location="cpu") -> TwoBunchFlow:
    """A checkpoint converted by `twobunch_cnf.migrate` (or written by this package), in eval mode."""
    return TwoBunchFlow.load_from_checkpoint(str(path), map_location=map_location).eval()
