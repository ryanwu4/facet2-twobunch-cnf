"""The shipped C_B3 bundle: verification on load, parity with the twobunch-s2e-rl source model, a sane shot."""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")
import twobunch_cnf  # noqa: E402
from _golden_probe import probe  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "golden" / "c_b3_ema_probe.npz"
pytestmark = pytest.mark.skipif("c_b3_ema" not in twobunch_cnf.available_models(), reason="bundle not shipped")


@pytest.fixture(scope="module")
def model():
    return twobunch_cnf.load_model("c_b3_ema")


def test_bundle_metadata(model):
    assert model.name == "c_b3_ema" and len(model.knob_names) == 37 and model.flow.hparams.treaty_point == "XTCAVF"
    assert model.planes == ["exit", "entrance"] and model.handoff["trained_plane"] == "exit"
    ent, ex = model.handoff["planes"]["entrance"], model.handoff["planes"]["exit"]
    assert ex["s_m"] - ent["s_m"] == pytest.approx(model.handoff["element_length_m"], rel=1e-12)
    for k in model.meta["knobs"]:
        assert k["unit"] and k["description"] and k["low"] <= k["baseline"] <= k["high"], k["name"]


def test_bundled_flow_matches_the_source_model(model):
    """Golden written from twobunch-s2e-rl's own TwoBunchFlow on the original checkpoint."""
    want = np.load(GOLDEN)
    got = probe(model.flow)
    assert set(got) == set(want.files)
    # rtol: float32 kernels differ between torch builds (1.4e-5 seen between conda-forge 2.14 and PyPI CPU 2.15)
    for k in want.files:
        np.testing.assert_allclose(got[k], want[k], rtol=1e-4, atol=1e-9, err_msg=k)


def test_tampered_checkpoint_is_refused(tmp_path):
    src = Path(twobunch_cnf.bundle.bundle_dir("c_b3_ema"))
    dst = tmp_path / "c_b3_ema"
    shutil.copytree(src, dst)
    meta = yaml.safe_load((dst / "bundle.yaml").read_text())
    meta["checkpoint"]["sha256"] = "0" * 64
    (dst / "bundle.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    with pytest.raises(ValueError, match="sha256"):
        twobunch_cnf.load_model(dst)


def test_baseline_shot(model):
    pytest.importorskip("beamphysics")
    dr = model.draw(n_particles=10000, seed=0)
    assert bool(dr.witness_present[0])
    n = int(dr.n_drive[0] + dr.n_witness[0])
    assert 0 < n <= 10000
    q = float(dr.charge_drive_nC[0] + dr.charge_witness_nC[0])
    assert 1.5 < q <= 1.62  # ~1.6 nC injected at the baseline
    P = model.to_particle_group(dr)  # entrance plane (default)
    assert P.in_z_coordinates and len(P) == n
    assert abs(np.average(P.t, weights=P.weight) - model.handoff["planes"]["entrance"]["ref_time_s"]) < 1e-12


def test_package_import_stays_light():
    code = "import sys, twobunch_cnf; assert 'torch' not in sys.modules and 'lightning' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
