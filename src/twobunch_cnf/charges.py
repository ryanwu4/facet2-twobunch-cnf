"""Per-bunch injected charges as a function of the knobs.

Ported from twobunch-s2e-rl (src/twobunch_s2e_rl/rl/plants.py @ 7ac3617). The transmission heads
predict surviving FRACTIONS T; the charge at the output element is T * the injected charge returned here.
"""

from __future__ import annotations

import torch


def injected_charges_nC(norm: dict, knob_keys, knob_low, knob_high):
    """knobs_norm (B, n) -> (q_drive, q_witness) injected charges [nC], (B,) each.

    Tables preprocessed with transmission='per_sample' label T = charge / that row's injected
    charge, which varies with the injector knobs: there q_in = Q f 1.001 (drive), Q (1 - f) 0.999
    (witness) -- the generation-time weight convention of twobunch-s2e-rl sim/injector.py, exact to
    ~1e-4 against the recorded injected charges. Flows without injector knobs (or 'max' tables): the
    constant full charges of the norm."""
    keys = list(knob_keys)
    if norm.get("transmission") == "per_sample" and {"drive_fraction", "distgen:total_charge:value"} <= set(keys):
        jf, jq = keys.index("drive_fraction"), keys.index("distgen:total_charge:value")
        lo, hi = torch.as_tensor(knob_low), torch.as_tensor(knob_high)

        def q(knobs_norm):
            phys = lambda j: lo[j].to(knobs_norm) + knobs_norm[:, j] * (hi[j] - lo[j]).to(knobs_norm)
            Q, f = phys(jq) / 1000.0, phys(jf)
            return Q * f * 1.001, Q * (1 - f) * 0.999

        return q
    qd, qw = float(norm["drive_full_charge_nC"]), float(norm["witness_full_charge_nC"])
    return lambda knobs_norm: (
        torch.full(knobs_norm.shape[:1], qd, device=knobs_norm.device),
        torch.full(knobs_norm.shape[:1], qw, device=knobs_norm.device),
    )
