"""Widen a trained flow's conditioning to a bigger knob space without changing what it predicts.

The layers that see the knob vector are `encoder[0]` (input = [knobs, one-hot bunch(2)]),
`feas_head[0]` and `t_full_head[0]` (input = knobs), and, for beam-latent models, the latent prior's
`latent_prior[k][0]` (input = knobs). Knobs enter normalized in the model's frame, u = (x - lo) / span.
For a new frame (lo', span') the old input is u = o + s * u' with s = span' / span and
o = (lo' - lo) / span, so

    W u + b = (W * s) u' + (b + W o)

reproduces the old layer EXACTLY. Knobs that are new get zero columns, so the widened model ignores
them until it is fine-tuned on data that varies them. Columns are matched by NAME; a knob dropped
from the new space is an error (use the campaign knob-drop rule upstream instead).
"""

from __future__ import annotations

import copy

import torch

from .artifacts import flow_knob_space
from .knobs import KnobSpace
from .model import TwoBunchFlow


def _remap_first_layer(lin: torch.nn.Linear, old: KnobSpace, new: KnobSpace, n_extra: int = 0):
    """Return (W', b') for a Linear whose first `old.n` inputs are the old knobs (then `n_extra`
    other inputs, e.g. the bunch one-hot, kept as-is)."""
    W, b = lin.weight.detach().double(), lin.bias.detach().double()
    W_new = torch.zeros(W.shape[0], new.n + n_extra, dtype=W.dtype)
    b_new = b.clone()
    for i, key in enumerate(old.keys):
        j = new.index(key)
        span_old = old.high[i] - old.low[i] or 1.0
        span_new = new.high[j] - new.low[j] or 1.0
        s = span_new / span_old
        o = (new.low[j] - old.low[i]) / span_old
        W_new[:, j] = W[:, i] * s
        b_new += W[:, i] * o
    if n_extra:
        W_new[:, new.n :] = W[:, old.n :]
    return W_new.to(lin.weight.dtype), b_new.to(lin.bias.dtype)


def widen_conditioning(flow: TwoBunchFlow, new_space: KnobSpace) -> TwoBunchFlow:
    old = flow_knob_space(flow)
    missing = [k for k in old.keys if k not in new_space.keys]
    if missing:
        raise ValueError(f"new knob space drops {missing}; widening only adds knobs")
    hp = dict(flow.hparams)
    hp.update(
        condition_dim=new_space.n,
        knob_low=list(new_space.low),
        knob_high=list(new_space.high),
        knob_keys=list(new_space.keys),
    )
    wide = TwoBunchFlow(**hp)
    state = copy.deepcopy(flow.state_dict())
    knob_layers = [("encoder.0", 2), ("feas_head.0", 0), ("t_full_head.0", 0)]
    if flow.hparams.get("beam_latent_dim", 0):
        knob_layers += [(f"latent_prior.{k}.0", 0) for k in range(2)]
    for prefix, n_extra in knob_layers:
        lin = dict(flow.named_modules())[prefix]
        W, b = _remap_first_layer(lin, old, new_space, n_extra)
        state[f"{prefix}.weight"], state[f"{prefix}.bias"] = W, b
    state["knob_low"] = torch.tensor(new_space.low, dtype=torch.float32)
    state["knob_high"] = torch.tensor(new_space.high, dtype=torch.float32)
    wide.load_state_dict(state)
    return wide.eval()
