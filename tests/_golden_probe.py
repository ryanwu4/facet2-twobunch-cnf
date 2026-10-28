"""Deterministic probe of a TwoBunchFlow: fixed knob points, frozen base samples, seeded latent / NLL RNG.

Shared by scripts/make_golden.py (writes tests/golden/<model>_probe.npz from the ORIGINAL twobunch-s2e-rl
checkpoint) and tests/test_bundle.py (compares the shipped bundle against it).
"""

import numpy as np
import torch


@torch.no_grad()
def probe(flow, n: int = 128) -> dict[str, np.ndarray]:
    d = int(flow.hparams.condition_dim)
    g = torch.Generator().manual_seed(0)
    knobs = torch.cat([torch.full((1, d), 0.5), torch.rand(3, d, generator=g)])
    zd, zw = torch.randn(n, 6, generator=g), torch.randn(n, 6, generator=g)
    parts = torch.randn(4, n, 6, generator=g)
    obs = flow.observables(knobs, n=n, z_drive=zd, z_witness=zw, generator=torch.Generator().manual_seed(1))
    out = {f"obs.{k}": v for k, v in obs.items()}
    for k in (0, 1):
        out[f"sample{k}"] = flow.sample_bunch(knobs, k, n, generator=torch.Generator().manual_seed(2 + k))
        torch.manual_seed(5 + k)  # the latent posterior draws from the global RNG
        out[f"nll{k}"] = flow.bunch_nll(parts, knobs, k)
    out.update({f"feas{i}": t for i, t in enumerate(flow.feasibility(knobs))})
    return {k: v.double().cpu().numpy() for k, v in out.items()}
