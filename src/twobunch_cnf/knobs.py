"""KnobSpace: the single knob-frame object (names, physical bounds, baseline) for the pipeline.

Every place that maps knobs between physical units and the surrogate's normalized [0, 1] frame goes
through `normalize` / `denormalize` here. A KnobSpace is always built from what an artifact STORED
(a checkpoint's knob_low/high buffers, a processed norm json, a manifest) — never from today's knob
tables, which may have changed since the artifact was made (L2PhaseSet was widened in place from
[-38, -34] to [-40, -34] after the 8-knob campaigns; see twobunch-s2e-rl docs/architecture_plan.md).
Named knob tables (for new campaigns) live in twobunch-s2e-rl (datagen/sweep_params.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class KnobSpace:
    keys: tuple[str, ...]
    low: tuple[float, ...]
    high: tuple[float, ...]
    baseline: dict[str, float] = field(default_factory=dict)  # optional; may cover a subset

    def __post_init__(self):
        if not (len(self.keys) == len(self.low) == len(self.high)):
            raise ValueError("keys / low / high must have equal length")
        if len(set(self.keys)) != len(self.keys):
            raise ValueError("duplicate knob keys")

    # ---- construction ------------------------------------------------------------------------
    @classmethod
    def from_lists(cls, keys, low, high, baseline=None) -> "KnobSpace":
        return cls(tuple(keys), tuple(float(v) for v in low), tuple(float(v) for v in high), dict(baseline or {}))

    @classmethod
    def from_norm_json(cls, path) -> "KnobSpace":
        """The frame a processed dataset (and every model trained on it) was normalized in."""
        d = json.loads(Path(path).read_text())
        return cls.from_lists(d["knob_keys"], d["knob_low"], d["knob_high"])

    def to_dict(self) -> dict:
        return {
            "knob_keys": list(self.keys),
            "knob_low": list(self.low),
            "knob_high": list(self.high),
            "baseline": dict(self.baseline),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KnobSpace":
        return cls.from_lists(d["knob_keys"], d["knob_low"], d["knob_high"], d.get("baseline"))

    # ---- frame maps (the only implementation) --------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.keys)

    def _span(self, like):
        lo, hi = np.asarray(self.low), np.asarray(self.high)
        span = np.where(hi > lo, hi - lo, 1.0)  # degenerate (fixed) knob -> span 1
        if _is_torch(like):
            import torch

            return (
                torch.as_tensor(lo, dtype=like.dtype, device=like.device),
                torch.as_tensor(span, dtype=like.dtype, device=like.device),
            )
        return lo, span

    def normalize(self, phys):
        """physical (..., n) -> normalized [0, 1] (..., n). Works on numpy arrays or torch tensors."""
        lo, span = self._span(phys)
        return (phys - lo) / span

    def denormalize(self, norm):
        """normalized (..., n) -> physical (..., n)."""
        lo, span = self._span(norm)
        return lo + norm * span

    def index(self, key: str) -> int:
        return self.keys.index(key)

    def width(self, key: str) -> float:
        i = self.index(key)
        return self.high[i] - self.low[i]

    # ---- composition ---------------------------------------------------------------------------
    def vector(self, values: dict, fill: dict | None = None) -> np.ndarray:
        """Physical vector in this key order from a {key: value} dict. Keys missing from `values`
        are taken from `fill` (e.g. a campaign's recorded fixed settings); missing from both ->
        KeyError (never silently defaulted)."""
        fill = fill or {}
        out = []
        for k in self.keys:
            if k in values:
                out.append(float(values[k]))
            elif k in fill:
                out.append(float(fill[k]))
            else:
                raise KeyError(f"knob {k!r} missing from values and fill")
        return np.asarray(out, dtype=np.float64)

    def subset(self, keys) -> "KnobSpace":
        idx = [self.index(k) for k in keys]
        return KnobSpace.from_lists(
            [self.keys[i] for i in idx],
            [self.low[i] for i in idx],
            [self.high[i] for i in idx],
            {k: self.baseline[k] for k in keys if k in self.baseline},
        )

    def contains(self, phys, tol: float = 1e-9) -> np.ndarray:
        """Boolean (...,) — inside the box (per row), with a relative tolerance on the span."""
        lo, span = self._span(np.asarray(phys))
        x = (np.asarray(phys) - lo) / span
        return np.all((x >= -tol) & (x <= 1 + tol), axis=-1)


def _is_torch(x) -> bool:
    return type(x).__module__.startswith("torch")
