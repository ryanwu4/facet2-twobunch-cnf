"""Conditional normalizing-flow surrogate of the FACET-II two-bunch beam.

Given the machine settings ("knobs"), the model samples the 6D macroparticles of the drive and witness
bunches at one lattice element (XTCAVF for the shipped model) and predicts how much of each bunch reaches
it. Sampling is reparameterized, so quantities computed from the samples are differentiable with respect
to the knobs. Start with `twobunch_cnf.load_model()`; the README describes the beam conventions.

Split out of twobunch-s2e-rl (src/twobunch_s2e_rl/surrogate at 7ac3617); the coupling layers come from
slac-photoinjector-rl's single-bunch flow. Submodules are imported explicitly (`from twobunch_cnf.model
import TwoBunchFlow`) so that importing this package stays light.
"""

__version__ = "0.1.0"

# Column order of (n, 6) particle arrays: positions [m], then momenta [eV/c].
COORD_KEYS = ("x", "y", "z", "px", "py", "pz")
LATENT_DIM = 6  # dimension of the flow's base distribution, one per phase-space coordinate
# Electron rest energy [eV], for normalized emittances of momenta in eV/c.
ELECTRON_MC2_EV = 0.51099895e6
# Particles kept per bunch in a training table. Bunches with fewer surviving particles still train the
# transmission heads, but not the particle distribution.
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
