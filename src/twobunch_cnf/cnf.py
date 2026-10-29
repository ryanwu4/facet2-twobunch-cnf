"""TwoBunchCNF: a trained flow plus what turns knob settings into a beam at the output element.

`draw()` samples one beam ("shot") per knob setting, batched; `to_particle_group()` exports a shot as an
openPMD-beamphysics ParticleGroup, following the conventions of the beams FACET2-S2E saves:

* One s for all particles: every particle has z = the s position of the plane, and the longitudinal
  coordinate is t. The flow learned the z that FACET2-S2E's getBeamAtElement writes, z = -c (t - <t>),
  relative to the charge-weighted mean arrival of both bunches. Absolute arrival is not modeled, so the
  export uses t = t_ref - z / c, with t_ref the lattice reference time at the plane the flow was trained at.
* `plane`: the trained plane (the XTCAVF exit, with the cavity off) or another plane of the same element,
  reached by an exact field-free drift. The XTCAVF entrance is a drift back through the cavity, which is
  exact because the cavity was off.
* Both bunches in one ParticleGroup. `n_particles` is the number of macroparticles INJECTED, split between
  drive and witness by their injected charges. Lost particles are removed, so round(n_injected * T)
  survive. With Q the total injected charge and N = n_particles, each macroparticle weighs Q / N * 1.001
  (drive) or Q / N * 0.999 (witness) [C], which is how `beam_io.split_bunches` and FACET2-S2E's
  getDriverAndWitness tell the bunches apart.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import torch

from . import ELECTRON_MC2_EV
from .artifacts import flow_knob_space
from .charges import injected_charges_nC
from .knobs import KnobSpace

C_LIGHT = 299792458.0
WEIGHT_TAG = {"drive": 1.001, "witness": 0.999}  # FACET2-S2E per-particle weight tag (drive = higher)
_VALIDATION = ("none", "warn", "error")


@dataclass
class BeamDraw:
    """One shot per knob setting (batch B). `drive` and `witness` hold each shot's particles ("clouds"),
    (x, y, z, px, py, pz) in m and eV/c, with z the flow's longitudinal coordinate relative to the beam's
    mean. They are (B, max count, 6): shot i is only the first n_drive[i] / n_witness[i] particles, and
    the rest are extra samples, not zeros, so use `clouds(i)`."""

    knobs_norm: torch.Tensor  # (B, n_knob) in the model's [0, 1] frame
    drive: torch.Tensor  # (B, max n_drive, 6)
    witness: torch.Tensor  # (B, max n_witness, 6)
    n_drive: torch.Tensor  # (B,) surviving macroparticles
    n_witness: torch.Tensor  # (B,)
    weight_drive: torch.Tensor  # (B,) [C] per macroparticle
    weight_witness: torch.Tensor  # (B,) [C]
    T_drive: torch.Tensor  # (B,) transmitted fraction used for the counts
    T_witness: torch.Tensor  # (B,) 0 where the witness is lost
    p_surv: torch.Tensor  # (B,) predicted probability that the witness survives
    witness_present: torch.Tensor  # (B,) bool
    n_particles: int  # injected macroparticles per shot
    seed: int | None

    def __len__(self):
        return self.knobs_norm.shape[0]

    def clouds(self, i: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """(drive (n_drive, 6), witness (n_witness, 6)) of shot i."""
        return self.drive[i, : int(self.n_drive[i])], self.witness[i, : int(self.n_witness[i])]

    @property
    def charge_drive_nC(self) -> torch.Tensor:
        return self.n_drive.double() * self.weight_drive * 1e9

    @property
    def charge_witness_nC(self) -> torch.Tensor:
        return self.n_witness.double() * self.weight_witness * 1e9


class TwoBunchCNF:
    """A TwoBunchFlow with its knobs (names, training ranges, baselines), the injected charge of each
    bunch, and the position and reference time of its output planes. Build it with
    `twobunch_cnf.load_model(name)`.

    handoff: {"element", "trained_plane", "species", "planes": {plane: {"s_m", "ref_time_s"}}, ...}
    """

    def __init__(
        self, flow, norm: dict, handoff: dict, baseline: dict | None = None, name: str = "", meta: dict | None = None
    ):
        self.flow = flow.eval()
        self.norm = norm
        frame = flow_knob_space(flow)
        self.frame = KnobSpace.from_lists(frame.keys, frame.low, frame.high, baseline or {})
        self.handoff = handoff
        if handoff["trained_plane"] not in handoff["planes"]:
            raise ValueError(f"handoff has no entry for its trained plane {handoff['trained_plane']!r}")
        self.name = name
        self.meta = meta or {}
        self._q_inj = injected_charges_nC(norm, frame.keys, flow.knob_low.cpu(), flow.knob_high.cpu())

    # ---- frame ---------------------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return next(self.flow.parameters()).device

    @property
    def knob_names(self) -> list[str]:
        return list(self.frame.keys)

    @property
    def planes(self) -> list[str]:
        return list(self.handoff["planes"])

    def to(self, device) -> TwoBunchCNF:
        self.flow.to(device)
        return self

    def knobs_norm(self, values=None, validation: str = "none") -> torch.Tensor:
        """Knob values -> the flow's input, each knob scaled to [0, 1] over its training range: (B, n_knob).

        values: None (the baseline); {name: value or 1-D array} (knobs not given take the baseline;
        unknown names are an error); or a (n_knob,) / (B, n_knob) array of physical values in the order of
        `knob_names`. validation, for values outside the training ranges: 'none' (default; the flow
        extrapolates), 'warn', or 'error'."""
        ks = self.frame
        if validation not in _VALIDATION:
            raise ValueError(f"validation must be one of {_VALIDATION}, got {validation!r}")
        if values is None or isinstance(values, dict):
            values = values or {}
            unknown = sorted(set(values) - set(ks.keys))
            if unknown:
                raise KeyError(f"unknown knob(s) {unknown}; this model's knobs: {list(ks.keys)}")
            missing = [k for k in ks.keys if k not in values and k not in ks.baseline]
            if missing:
                raise KeyError(f"knob(s) {missing} not given and without a baseline")
            cols = [np.atleast_1d(np.asarray(values.get(k, ks.baseline.get(k)), dtype=np.float64)) for k in ks.keys]
            b = max(len(c) for c in cols)
            if any(c.ndim != 1 or len(c) not in (1, b) for c in cols):
                raise ValueError("knob values must be scalars or 1-D arrays of one common length")
            phys = np.stack([np.broadcast_to(c, (b,)) for c in cols], -1)
        else:
            phys = values.detach().cpu().numpy() if torch.is_tensor(values) else np.asarray(values)
            phys = np.atleast_2d(phys.astype(np.float64))
            if phys.ndim != 2 or phys.shape[-1] != ks.n:
                raise ValueError(f"expected (B, {ks.n}) physical knob values, got shape {phys.shape}")
        if validation != "none":
            lo, hi = np.asarray(ks.low), np.asarray(ks.high)
            tol = 1e-9 * np.where(hi > lo, hi - lo, 1.0)
            bad = (phys < lo - tol) | (phys > hi + tol)
            if bad.any():
                names = [ks.keys[j] for j in np.flatnonzero(bad.any(0))]
                msg = f"knob(s) {names} outside the training box (the flow extrapolates there)"
                if validation == "error":
                    raise ValueError(msg)
                warnings.warn(msg, stacklevel=2)
        return torch.as_tensor(ks.normalize(phys), dtype=torch.float32, device=self.device)

    # ---- shots ---------------------------------------------------------------------------------
    @torch.no_grad()
    def draw(
        self,
        values=None,
        n_particles: int = 10000,
        seed: int | None = None,
        witness: str = "threshold",
        transmission: str = "given",
        latent: str = "sample",
        validation: str = "none",
    ) -> BeamDraw:
        """One shot per knob setting.

        n_particles: macroparticles injected per shot; lost ones are removed, as in FACET2-S2E.
        seed: one torch.Generator drives everything random in the draw, so the same seed and inputs give
            the same draw; None = a fresh draw.
        witness: whether the witness is kept, given p_surv, the predicted probability that it survives:
            'threshold' (default; kept iff p_surv > 0.5), 'sample' (kept with probability p_surv) or
            'always'.
        transmission: 'given' (default; a point estimate of the transmitted fraction T, close to 1
            whenever no loss is likely) or 'mixture' (per shot, T = 1 with the predicted probability of no
            loss, else the predicted T when there is loss; use it for loss statistics).
        latent: 'sample' (default; a beam latent drawn from its knob-conditioned prior) or 'mean' (the
            prior mean: a typical beam). No effect for flows without a beam latent.
        validation: see `knobs_norm`."""
        if witness not in ("threshold", "sample", "always"):
            raise ValueError(f"witness must be 'threshold', 'sample' or 'always', got {witness!r}")
        if transmission not in ("given", "mixture"):
            raise ValueError(f"transmission must be 'given' or 'mixture', got {transmission!r}")
        if latent not in ("sample", "mean"):
            raise ValueError(f"latent must be 'sample' or 'mean', got {latent!r}")
        n_particles = int(n_particles)
        if n_particles < 1:
            raise ValueError("n_particles must be >= 1")
        kn = self.knobs_norm(values, validation)
        b, dev = kn.shape[0], kn.device
        g = torch.Generator(device=dev)
        if seed is None:
            g.seed()
        else:
            g.manual_seed(int(seed))
        flow = self.flow
        p_surv, t_d, t_w = flow.feasibility(kn)
        if transmission == "mixture":
            tail = torch.sigmoid(flow.feas_head(kn))[:, 1:]
            p_full = torch.sigmoid(flow.t_full_head(kn))
            full = torch.rand(p_full.shape, generator=g, device=dev) < p_full
            t = torch.where(full, torch.ones_like(tail), tail)
            t_d, t_w = t[:, 0], t[:, 1]
        if witness == "threshold":
            present = p_surv > 0.5
        elif witness == "sample":
            present = torch.rand(b, generator=g, device=dev) < p_surv
        else:
            present = torch.ones(b, dtype=torch.bool, device=dev)
        t_w = torch.where(present, t_w, torch.zeros_like(t_w))

        # S2E injection: q_drive = Q f 1.001, q_witness = Q (1 - f) 0.999 -> the counts carry the split
        q_d, q_w = self._q_inj(kn)
        a_d, a_w = q_d.double() / WEIGHT_TAG["drive"], q_w.double() / WEIGHT_TAG["witness"]
        n_inj_d = torch.round(n_particles * a_d / (a_d + a_w)).long()
        n_inj_w = n_particles - n_inj_d
        w0 = (a_d + a_w) * 1e-9 / n_particles  # [C] per injected macroparticle
        n_d = torch.round(n_inj_d * t_d.double()).long()
        n_w = torch.round(n_inj_w * t_w.double()).long()
        drive = self._sample(kn, 0, int(n_d.max()), g, latent)
        wit = self._sample(kn, 1, int(n_w.max()), g, latent)
        return BeamDraw(
            knobs_norm=kn,
            drive=drive,
            witness=wit,
            n_drive=n_d,
            n_witness=n_w,
            weight_drive=w0 * WEIGHT_TAG["drive"],
            weight_witness=w0 * WEIGHT_TAG["witness"],
            T_drive=t_d,
            T_witness=t_w,
            p_surv=p_surv,
            witness_present=present,
            n_particles=n_particles,
            seed=seed,
        )

    def _sample(self, kn, k, n, g, latent):
        if n == 0:
            return kn.new_zeros(kn.shape[0], 0, 6)
        s = None
        if latent == "mean" and self.flow.hparams.get("beam_latent_dim", 0):
            s = self.flow.latent_prior_params(kn, k)[0]
        return self.flow.sample_bunch(kn, k, n, generator=g, s=s)

    # ---- export --------------------------------------------------------------------------------
    def to_particle_group(self, draw: BeamDraw, index: int = 0, plane: str = "entrance"):
        """Shot `index` of `draw` as a fixed-z openPMD ParticleGroup at `plane` (see the module doc)."""
        try:
            from beamphysics import ParticleGroup
        except ImportError:  # openPMD-beamphysics from before the rename of pmd_beamphysics
            from pmd_beamphysics import ParticleGroup

        planes = self.handoff["planes"]
        if plane not in planes:
            raise ValueError(f"plane must be one of {list(planes)}, got {plane!r}")
        trained = self.handoff["trained_plane"]
        d, w = draw.clouds(index)
        n_d, n = d.shape[0], d.shape[0] + w.shape[0]
        x, y, zr, px, py, pz = torch.cat([d, w]).double().cpu().numpy().T
        t = float(planes[trained]["ref_time_s"]) - zr / C_LIGHT
        ds = float(planes[plane]["s_m"]) - float(planes[trained]["s_m"])
        if ds != 0.0:  # exact field-free drift from the trained plane (negative = upstream)
            e = np.sqrt(px * px + py * py + pz * pz + ELECTRON_MC2_EV**2)
            x, y, t = x + ds * px / pz, y + ds * py / pz, t + ds * e / (C_LIGHT * pz)
        weight = np.r_[
            np.full(n_d, float(draw.weight_drive[index])), np.full(n - n_d, float(draw.weight_witness[index]))
        ]
        data = {
            "x": x,
            "px": px,
            "y": y,
            "py": py,
            "z": np.full(n, float(planes[plane]["s_m"])),
            "pz": pz,
            "t": t,
            "status": np.ones(n, dtype=int),
            "weight": weight,
            "species": self.handoff.get("species", "electron"),
        }
        return ParticleGroup(data=data)

    def particle_group(self, values=None, plane: str = "entrance", **draw_kwargs):
        """draw() + to_particle_group() for ONE knob setting."""
        dr = self.draw(values, **draw_kwargs)
        if len(dr) != 1:
            raise ValueError(f"particle_group takes one knob setting, got {len(dr)}; use draw + to_particle_group")
        return self.to_particle_group(dr, 0, plane)


def reference_time_upstream(
    ref_time_s: float, length_m: float, e_tot_eV: float, mc2_eV: float = ELECTRON_MC2_EV
) -> float:
    """Lattice reference time `length_m` upstream of a plane with reference time `ref_time_s` (drift)."""
    beta = math.sqrt(1.0 - (mc2_eV / e_tot_eV) ** 2)
    return ref_time_s - length_m / (beta * C_LIGHT)
