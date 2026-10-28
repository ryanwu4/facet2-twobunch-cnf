"""openPMD ParticleGroup <-> (n, 6) cloud conventions shared by preprocessing and beam export.

Ported from twobunch-s2e-rl @ 7ac3617: `split_bunches` (sim/bmad.py), `stack_coords` / `subsample`
(surrogate/preprocess.py `_stack` / `_subsample`). Functions take ParticleGroup objects but do not
import openpmd-beamphysics themselves.

Two-bunch beams follow the FACET2-S2E convention: one ParticleGroup whose drive and witness carry
the two unique per-particle weights (drive = higher weight, a +-0.1% tag), with the charge ratio in
the macroparticle counts.
"""
from __future__ import annotations

import numpy as np

from . import COORD_KEYS


def split_bunches(P):
    """(drive, witness) ParticleGroups split by per-particle weight: drive = higher weight
    (FACET2_S2E.getDriverAndWitness convention). Witness is None if only one weight survives."""
    w = np.unique(P.weight)
    if len(w) < 2:
        return P, None
    return P[P.weight == w[-1]], P[P.weight == w[0]]


def stack_coords(pg) -> np.ndarray:
    """(n, 6) in COORD_KEYS order: (x, y, z, px, py, pz). pos [m], mom [eV/c]."""
    return np.stack([getattr(pg, k) for k in COORD_KEYS], axis=1).astype(np.float32)


def subsample(coords: np.ndarray, p: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.choice(coords.shape[0], p, replace=False)
    return coords[idx]
