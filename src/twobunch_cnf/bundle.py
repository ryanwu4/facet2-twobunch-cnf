"""Model bundles: a directory with everything needed to load a trained flow.

  twobunch_cnf.load_model("c_b3_ema")        # a bundle shipped in twobunch_cnf/resources/
  twobunch_cnf.load_model("/path/to/bundle")  # any bundle directory

A bundle holds the converted checkpoint, its norm.json (particle and knob scalings, injected charges), the
training run's run.json, and bundle.yaml, which scripts/export_bundle.py writes: the checkpoint's sha256,
the knobs in the model's order with their training range, baseline, unit and description, the position
and reference time of each output plane, and provenance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

RESOURCES = Path(__file__).resolve().parent / "resources"


def available_models() -> list[str]:
    """Names of the bundles shipped with the package."""
    if not RESOURCES.is_dir():
        return []
    return sorted(p.name for p in RESOURCES.iterdir() if (p / "bundle.yaml").is_file())


def bundle_dir(name_or_path) -> Path:
    p = Path(name_or_path)
    if (p / "bundle.yaml").is_file():
        return p
    if (RESOURCES / str(name_or_path) / "bundle.yaml").is_file():
        return RESOURCES / str(name_or_path)
    raise FileNotFoundError(f"no bundle {name_or_path!r} (shipped: {available_models()})")


def read_bundle(name_or_path) -> dict:
    d = bundle_dir(name_or_path)
    meta = yaml.safe_load((d / "bundle.yaml").read_text())
    meta["_dir"] = str(d)
    return meta


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(name: str = "c_b3_ema", device="cpu", verify: bool = True):
    """Load a bundle as a TwoBunchCNF.

    name: a shipped bundle (see `available_models()`) or a bundle directory. device: where the flow runs.
    verify: check the checkpoint against bundle.yaml's sha256, and the bundle's knob table against the knob
    names and ranges stored in the checkpoint."""
    from .cnf import TwoBunchCNF
    from .model import load_flow

    meta = read_bundle(name)
    d = Path(meta["_dir"])
    ckpt = d / meta["checkpoint"]["file"]
    if verify and sha256(ckpt) != meta["checkpoint"]["sha256"]:
        raise ValueError(f"{ckpt}: sha256 differs from bundle.yaml (corrupted or replaced checkpoint)")
    flow = load_flow(ckpt, map_location=device)
    norm = json.loads((d / meta["norm"]).read_text())
    knobs = meta["knobs"]
    if verify:
        keys = list(flow.hparams.knob_keys)
        if [k["name"] for k in knobs] != keys:
            raise ValueError(f"{d}: bundle knob order differs from the checkpoint's knob_keys")
        lo, hi = flow.knob_low.cpu().numpy(), flow.knob_high.cpu().numpy()
        if not (
            np.allclose([k["low"] for k in knobs], lo, rtol=1e-6, atol=0)
            and np.allclose([k["high"] for k in knobs], hi, rtol=1e-6, atol=0)
        ):
            raise ValueError(f"{d}: bundle knob bounds differ from the checkpoint's frame")
    baseline = {k["name"]: k["baseline"] for k in knobs if k.get("baseline") is not None}
    return TwoBunchCNF(flow, norm, meta["handoff"], baseline, name=meta["name"], meta=meta)
