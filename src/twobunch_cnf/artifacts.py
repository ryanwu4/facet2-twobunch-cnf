"""Artifact helpers: checkpoint resolution, run.json provenance, and a checkpoint's knob frame.

Ported from twobunch-s2e-rl (src/twobunch_s2e_rl/artifacts.py @ 7ac3617). Only checkpoints that
record their knob names (`knob_keys`) and `treaty_point` are supported here; the legacy tables that
fill those in for older 8/23/26-knob checkpoints stay in twobunch-s2e-rl.
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import time
from pathlib import Path

from . import __version__

# ---- checkpoint / policy resolution ------------------------------------------------------------

_VAL_LOSS = re.compile(r"val_loss=(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
_REWARD = re.compile(r"reward(-?\d+(?:\.\d+)?)")


def resolve_ckpt(pattern: str) -> str:
    """A path, or a glob. For a glob, pick the best match by the number in the file name:
    lowest `val_loss=` (flow checkpoints; may be negative) or highest `reward` (policy files).
    Fails loudly when no file matches, when names carry no score, or on a tie."""
    if not any(c in pattern for c in "*?["):
        return pattern
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no file matches {pattern!r}")
    if len(files) == 1:
        return files[0]
    scored = []
    for f in files:
        name = Path(f).name
        m_loss, m_rew = _VAL_LOSS.search(name), _REWARD.search(name)
        if m_loss:
            scored.append((float(m_loss.group(1)), f))
        elif m_rew:
            scored.append((-float(m_rew.group(1)), f))      # higher reward is better
        else:
            raise ValueError(f"cannot rank {name!r} (no val_loss/reward in name); pass a path")
    scored.sort()
    if scored[0][0] == scored[1][0]:
        raise ValueError(f"ambiguous best match for {pattern!r}: {scored[0][1]}, {scored[1][1]}")
    return scored[0][1]


# ---- run.json provenance -----------------------------------------------------------------------

def _git(*args, cwd=None) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_root() -> Path:
    """Checkout root of this package (.../src/twobunch_cnf/artifacts.py -> parents[2]); only a git
    checkout for a source / editable install."""
    return Path(__file__).resolve().parents[2]


def git_state(path=None) -> dict:
    path = str(path or package_root())
    # tracked changes only: untracked result folders don't change the code that ran
    return {"git_hash": _git("rev-parse", "HEAD", cwd=path),
            "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no", cwd=path))}


def facet2_s2e_hash() -> str | None:
    """HEAD of the FACET2-S2E checkout named by $FACET2_S2E_ROOT (None when unset or not a checkout)."""
    env = os.environ.get("FACET2_S2E_ROOT")
    if not env:
        return None
    root = Path(env).expanduser()
    return _git("rev-parse", "HEAD", cwd=root) if root.exists() else None


def write_run_json(out_dir, kind: str, config: dict, parents=(), overwrite: bool = False,
                   **extra) -> Path:
    """Write <out_dir>/run.json: what this artifact is, how it was made, and from what.
    Refuses to overwrite an existing run.json unless overwrite=True."""
    out = Path(out_dir) / "run.json"
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists (pass overwrite=True / --overwrite to replace)")
    out.parent.mkdir(parents=True, exist_ok=True)
    rec = {"kind": kind, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **git_state(),
           "package_version": __version__, "facet2_s2e_hash": facet2_s2e_hash(), "config": config,
           "parents": [str(p) for p in parents], **extra}
    out.write_text(json.dumps(rec, indent=2, default=str))
    return out


def read_run_json(path) -> dict:
    p = Path(path)
    return json.loads((p / "run.json" if p.is_dir() else p).read_text())


# ---- a checkpoint's knob frame -----------------------------------------------------------------

def flow_knob_space(flow):
    """KnobSpace of a loaded TwoBunchFlow: names from its hparams, bounds from its own buffers."""
    from .knobs import KnobSpace
    keys = getattr(flow.hparams, "knob_keys", None)
    if not keys:
        raise ValueError("checkpoint records no knob_keys; only checkpoints that record their knob names "
                         "are supported (older ones: twobunch-s2e-rl artifacts.LEGACY_KNOB_KEYS)")
    return KnobSpace.from_lists(keys, flow.knob_low.tolist(), flow.knob_high.tolist())


def flow_treaty_point(flow) -> str:
    tp = getattr(flow.hparams, "treaty_point", None)
    if not tp:
        raise ValueError("checkpoint records no treaty_point; only checkpoints that record it are supported")
    return tp
