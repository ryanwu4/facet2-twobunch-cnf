# facet2-twobunch-cnf

A fast, differentiable surrogate of the FACET-II two-bunch beam.

FACET-II, at SLAC, can run with two electron bunches: a high-charge drive bunch closely followed by a
lower-charge witness bunch, the configuration used for plasma wakefield acceleration. Simulating that beam
from the cathode onwards with [FACET2-S2E](https://github.com/slaclab/FACET2-S2E) (Impact-T for the
photoinjector, Bmad for the linac) takes minutes per shot. This package replaces the simulation with a
conditional normalizing flow, a neural network that learns the distribution of particles in each bunch as
a function of the machine settings and can sample from it. Given 37 settings, it samples the macroparticles
of both bunches at XTCAVF, the X-band transverse deflecting cavity in sector 20, and predicts how much of
each bunch survives to that point.

A 10,000-particle beam takes a fraction of a second on a CPU. Sampling through the flow is differentiable
with respect to the settings (see [Gradients](#gradients)), so the model can sit inside gradient-based
optimization or control loops. The shipped model was
trained on about 13,000 FACET2-S2E simulations with the settings varied around a two-bunch baseline.

| | |
|---|---|
| Inputs | 37 settings ("knobs"): RF phases and energy offsets of L0B, L1 and L2, and the L3 energy offset; W-chicane sextupole strengths and mover offsets; final-focus quadrupoles and correctors; and photoinjector and laser settings (gun phase and field, solenoid, L0A phase offset, laser spot size and truncation radius, pulse length, common timing and drive-pulse delay, cathode mean transverse energy, total charge, and the drive/witness charge split). |
| Output | One beam holding both bunches (6D macroparticles) at the entrance or exit of XTCAVF, as an [openPMD-beamphysics](https://github.com/ChristopherMayes/openPMD-beamphysics) `ParticleGroup` or as tensors. |
| Also predicted | The probability that the witness reaches XTCAVF at all, and the fraction of each bunch transmitted. |

## Install

```bash
pip install "facet2-twobunch-cnf[beam] @ git+https://github.com/ryanwu4/facet2-twobunch-cnf@v0.1.0"
```

The model weights ship with the package. The `beam` extra adds openPMD-beamphysics, which `ParticleGroup`
output needs; the `eval` extra adds what the evaluation and plotting commands need. pip installs the
default PyTorch build. On a machine without a GPU you can install the smaller CPU build first:
`pip install --index-url https://download.pytorch.org/whl/cpu torch`.

## Quick start

```python
import twobunch_cnf
from twobunch_cnf.beam_io import split_bunches

model = twobunch_cnf.load_model("c_b3_ema")  # the shipped model, on the CPU

# One beam. Knobs that are not given keep their baseline values.
beam = model.particle_group({"L2PhaseSet": -35.0}, n_particles=10_000, seed=0)
drive, witness = split_bunches(beam)
drive.charge, witness.charge  # 1.2e-9 and 4.0e-10 C
```

`beam` is an ordinary `ParticleGroup`, so the beamphysics tools work on it, for example
`beam.plot("delta_t", "energy")` or `beam.write("beam.h5")`.

The inputs:

```python
model.knob_names     # the 37 knob names, in the model's order
model.meta["knobs"]  # a list, one dict per knob in that order: name, training range (low, high),
                     # baseline, unit and description
```

Several settings at once, one beam each:

```python
draw = model.draw({"L2PhaseSet": [-37.0, -35.0, -33.0]}, n_particles=10_000, seed=1)
draw.p_surv                    # probability that the witness reaches XTCAVF, for each setting
draw.T_drive, draw.T_witness   # transmitted fraction of each bunch
beam = model.to_particle_group(draw, index=2)
drive, witness = draw.clouds(2)  # or the particles as (n, 6) tensors: x, y, z [m], px, py, pz [eV/c],
                                 # with z relative to the beam's mean (see Gradients)
```

## The output beam

Beams follow the conventions of the beams FACET2-S2E saves, so most code written for those files works on
them (but see the note on `t` below).

- Both bunches are in one `ParticleGroup`. Drive particles carry 0.2 % more weight than witness
  particles, which is how `split_bunches` (and FACET2-S2E's `getDriverAndWitness`) tells them apart.
  Since the weights are almost equal, the drive:witness charge ratio is their particle-count ratio.
- Lost particles are removed. `n_particles` is the number of macroparticles injected at the cathode, so a
  beam with losses has fewer. When the witness is predicted lost, the beam holds only the drive bunch,
  and `split_bunches` returns `(beam, None)`.
- The training beams were saved at the XTCAVF exit with the cavity off. The default, `plane="entrance"`,
  drifts the beam back through the 1.03 m cavity, which is exact because the cavity was off; the entrance
  is where a model of the powered cavity, or another simulation code, would take over. `plane="exit"`
  gives the beam as trained. `particle_group` and `to_particle_group` take `plane`; `draw` does not.
- Every particle has the same `z`, the s position of the plane, and the longitudinal coordinate is the
  arrival time `t`. The model only learned arrival times relative to the beam's mean. The export puts that
  mean at the plane's lattice reference time (the arrival time of the design particle), so timing jitter
  of the whole beam is not modeled. FACET2-S2E's saved beams sit a constant 13.8 ns earlier
  (`s2e_clock_offset` in bundle.yaml), so compare `t` with them relative to its mean.
- Each beam is a random draw, and `seed` makes it reproducible: the same seed, inputs and device give the
  same beam. The knobs don't pin down every detail of a beam, such as the exact curve of its longitudinal
  phase space. The model draws those details from a random "beam latent", once per beam, so that each
  sampled beam stays as sharp as a simulated one.

`draw` and `particle_group` take these options:

| Option | Default | Alternatives |
|---|---|---|
| `witness` | `"threshold"`: the witness is kept when `p_surv` > 0.5 | `"sample"`: kept with probability `p_surv`; `"always"` |
| `transmission` | `"given"`: a point estimate of each bunch's transmitted fraction, close to 1 whenever no loss is likely | `"mixture"`: the model predicts transmission in two parts, the probability of no loss and the transmission when there is loss. Each beam is then lossless or lossy at random, with the predicted probability. Use it for loss statistics. |
| `latent` | `"sample"`: a new beam latent for each beam | `"mean"`: the latent's typical value, the same for every beam |
| `validation` | `"none"`: knob values are not checked | `"warn"` or `"error"` for values outside the training ranges |

## Knobs

The knobs are FACET2-S2E simulation parameters, not EPICS PVs (the machine's control-system channels). For
example, `L2PhaseSet` is the RF phase of L2 in degrees. L0 is the injector linac and L1 to L3 are the main
linac sections; the W-chicane is the bunch compressor in sector 20. Names with colons are paths into the
injector simulation's inputs: `distgen:` keys set the laser and cathode distribution
(`distgen:total_charge:value` is the injected charge in pC), and `GUNF:` and `SOL10111:` the gun and
solenoid in Impact-T.

`model.meta["knobs"]` gives each knob's unit, training range and baseline; the same table is in
[`bundle.yaml`](src/twobunch_cnf/resources/c_b3_ema/bundle.yaml). The units of `GUNF:rf_field_scale` and
`SOL10111:solenoid_field_scale` were inferred rather than read from the simulation code, and are marked
`confidence: inferred`.

## Limitations

- The model reproduces FACET2-S2E, not the machine. How well FACET2-S2E matches FACET-II is a separate
  question.
- It is only reliable inside the training ranges. Outside them it extrapolates silently, unless you pass
  `validation="warn"` or `"error"`.
- Losses are predicted per bunch, not per particle. The model samples surviving particles from the
  distribution of survivors and doesn't say which particles were lost.
- XTCAVF was off in all the training beams, so the model cannot give a streaked beam.

## Gradients

`draw` and `particle_group` run without gradients: they round particle counts and return NumPy-backed
beams. For optimization and model-based RL, call the flow directly. Its inputs are the knobs scaled to
[0, 1] over their training ranges, and everything it returns is differentiable with respect to them:

```python
knobs = model.knobs_norm({"L2PhaseSet": -35.0}).requires_grad_()  # shape (1, 37)
obs = model.flow.observables(knobs, n=2048)  # dict of tensors, one value per setting
obs["bunch_spacing"].sum().backward()        # the gradient is now in knobs.grad
```

`observables` returns per-bunch quantities (`drive_norm_emit_x`, `witness_sigma_z`,
`witness_mean_energy`, ...), drive–witness quantities (`bunch_spacing`, `energy_difference`,
`transverse_offset`, `angular_misalignment`) and the survival predictions (`p_surv`, `T_drive`,
`T_witness`), in m, eV and eV/c. The Twiss and BMAG entries are measured against FACET2-S2E's matching
target at the plasma entrance (β = 0.5 m, α = 0), not against the optics at XTCAVF.
`model.flow.sample_bunch` returns the particles themselves. Both use the flow's own coordinates rather
than the exported beam's: `z` is the longitudinal position relative to the beam's mean, z = -c (t - ⟨t⟩),
so a larger z means an earlier arrival.

## Training and evaluation

Training needs a processed table of simulated beams: an HDF5 file of knob settings with particle
subsamples of each bunch, with its `_norm.json` of scalings beside it. These tables are made by the preprocessing in
[twobunch-s2e-rl](https://github.com/ryanwu4/twobunch-s2e-rl) from its FACET2-S2E simulation campaigns,
and are not part of this repository.

| Command | What it does |
|---|---|
| `twobunch-cnf-train --processed TABLE.h5 --out RUN_DIR [--config CONFIG.yaml]` | Trains a flow. |
| `twobunch-cnf-eval --ckpt CKPT --processed TABLE.h5 --out DIR` | Compares a flow with a table: parity plots, phase-space overlays and metrics. |
| `twobunch-cnf-plot-training RUN_DIR --out FIG.png` | Plots the loss curves of a training run. |
| `twobunch-cnf-migrate SRC.ckpt --out DST.ckpt` | Converts a twobunch-s2e-rl checkpoint (see below). |

[`configs/c_b3_ema.yaml`](configs/c_b3_ema.yaml) holds the settings the shipped model was trained with,
and its header lists the data files it used. All paths are given explicitly; none default to a location in
the repository.

## Model bundles

A bundle is a directory with everything needed to load a model:

- `model.ckpt`: the weights and hyperparameters;
- `norm.json`: the particle and knob scalings, and the injected charges;
- `run.json`: the training run's settings and provenance;
- `bundle.yaml`: the checkpoint's sha256, the knob table, the position and reference time of each output
  plane, and provenance.

`load_model` takes the name of a shipped bundle or the path of any bundle directory. One bundle ships with
the package: `c_b3_ema`, the exponential-moving-average (EMA) weights of training run C_B3.

`scripts/export_bundle.py` writes a new bundle, and `scripts/make_golden.py` the reference ("golden")
outputs that the unit tests compare the bundle against. Both need twobunch-s2e-rl on the `PYTHONPATH`.

## Checkpoints from twobunch-s2e-rl

This code was split out of twobunch-s2e-rl (`src/twobunch_s2e_rl/surrogate` at `7ac3617`), which keeps
the simulation campaigns, the preprocessing and the RL controllers. Only the features the shipped model
uses were kept: spline coupling layers, the two-part transmission head, the optional beam latent and the
losses it trains with. Checkpoints trained in twobunch-s2e-rl still record the removed options, so convert
them once with `twobunch-cnf-migrate`. It refuses a checkpoint that has a removed option turned on, or
that doesn't record its knob names and output element (`knob_keys`, and `treaty_point`, FACET2-S2E's name
for an element where beams are saved). Loading is strict: a checkpoint with settings this code doesn't
know raises an error rather than loading as a different model.

## Development

```bash
pixi install               # default environment: Python 3.12, PyTorch with CUDA, Lightning
pixi run -e test test      # unit tests
pixi run -e test lint      # ruff check; `pixi run -e test fmt` formats
pixi run -e va test-va     # virtual-accelerator's tests against this package (needs ../virtual-accelerator)
```

Without pixi: `pip install -e ".[eval,beam,test]"`, then `pytest`.

Branches:

- `main`: the released code.
- `archive/unpruned-7ac3617`: the code as imported from twobunch-s2e-rl, before the unused options were
  removed. It runs, and it keeps the experimental options: affine coupling layers, a single-sigmoid
  transmission head, centroid, chirp-curve and backbone heads, and single-bunch training.
- New experimental options go on branches off `main`, or off the archive branch to revive a removed one.
  They merge into `main` once a released model uses them.

## License

Apache-2.0; see [LICENSE](LICENSE).
