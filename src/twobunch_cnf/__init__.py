"""Conditional normalizing-flow surrogate of the FACET-II two-bunch beam.

Generates the drive + witness 6D phase space at a treaty point (PENT or XTCAVF) conditioned on
the machine knobs, plus per-bunch feasibility (witness viability + per-bunch transmission). One
shared coupling flow selected by a per-bunch index k, with a whitening MLP that places each bunch
(mu_k, Sigma_k) so the flow only models the standardized residual. Sampling is reparameterized,
so observables are differentiable w.r.t. the knobs (MBRL).

Split out of twobunch-s2e-rl (src/twobunch_s2e_rl/surrogate @ 7ac3617). Design: lab-notebook
2026-06-09_nf-two-bunch-surrogate.md (v3); flow/coupling math from slac-photoinjector-rl.
Submodules are imported explicitly (`from twobunch_cnf.model import TwoBunchFlow`) so that
importing this package stays light.
"""

__version__ = "0.1.0"

# 6D phase-space column order; matches compute_emittance_torch and the ParticleGroup read.
COORD_KEYS = ("x", "y", "z", "px", "py", "pz")
LATENT_DIM = 6
# Electron rest energy [eV]: geometric 4D emittance / mc^2^2 = norm_emit_4d.
ELECTRON_MC2_EV = 0.51099895e6
# Fixed macroparticles kept per bunch (subsample down; bunches with fewer live particles
# are excluded from density training but still used for the feasibility heads).
DEFAULT_P = 1024
BUNCHES = ("drive", "witness")  # index k: 0=drive, 1=witness

__all__ = [
    "COORD_KEYS",
    "LATENT_DIM",
    "ELECTRON_MC2_EV",
    "DEFAULT_P",
    "BUNCHES",
    "__version__",
    "load_model",
    "available_models",
]


def __getattr__(name):
    # lazy: `import twobunch_cnf` must not import torch / lightning
    if name in ("load_model", "available_models"):
        from . import bundle

        return getattr(bundle, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
