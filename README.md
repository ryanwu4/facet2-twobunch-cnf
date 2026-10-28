# facet2-twobunch-cnf

Conditional normalizing-flow (CNF) surrogate of the FACET-II two-bunch beam. Given the machine
knobs (linac phases / energies, FF magnets, sextupole offsets, injector and laser settings), it
samples the drive and witness 6D phase space at a treaty point (PENT or XTCAVF) and predicts
per-bunch feasibility (witness viability, per-bunch transmission). Sampling is reparameterized, so
beam observables are differentiable with respect to the knobs.

Split out of [twobunch-s2e-rl](https://github.com/ryanwu4/twobunch-s2e-rl)
(`src/twobunch_s2e_rl/surrogate` @ `7ac3617`), which keeps the data campaigns, preprocessing and
the RL controllers.

## Install

```bash
pixi install                  # default env (python 3.12, CUDA pytorch, lightning)
pixi run -e test test         # unit tests
pixi run -e test lint         # ruff check; `pixi run -e test fmt` = ruff format
# or plain pip: pip install -e ".[eval,beam,test]"
```

## Use

```python
import twobunch_cnf

model = twobunch_cnf.load_model("c_b3_ema")         # shipped bundle (or a bundle directory); CPU by default
model.knob_names                                     # 37 knobs; model.meta["knobs"]: bounds, baseline, unit
P = model.particle_group({"L2PhaseSet": -35.0},      # one shot as an openPMD ParticleGroup; other knobs at
                         seed=0)                     #   the baseline, XTCAVF entrance plane, 10000 injected
dr = model.draw({"L2PhaseSet": [-36.0, -35.0]}, n_particles=50000, seed=1)   # batched shots
P_exit = model.to_particle_group(dr, index=1, plane="exit")
```

A shot follows the FACET2-S2E two-bunch convention: one ParticleGroup, drive = the higher of two
per-particle weights (Q/N x 1.001 vs 0.999), the charge ratio in the macroparticle counts, lost
particles removed (`n_particles` is the injected count). The frame is fixed-z (z = s of the plane, the
longitudinal coordinate in t, t = lattice reference time - z_flow / c). `plane="entrance"` (default) is the
exact field-free back-drift of the trained plane (XTCAVF exit, TCAV off). Defaults are deterministic in
the feasibility heads (witness present iff p_surv > 0.5; T as predicted); `witness="sample"`,
`transmission="mixture"` and `latent="mean"` are opt-in. Knob values are not range-checked unless
`validation="warn"|"error"` (as the virtual-accelerator injector surrogate).

Lower level, for MBRL / training (differentiable):

```python
from twobunch_cnf.model import load_flow
flow = load_flow("<ckpt>")                           # eval mode
obs = flow.observables(knobs_norm, n=1024)           # knobs_norm: (B, n_knob) in the ckpt's [0, 1] frame
```

`twobunch_cnf.artifacts.flow_knob_space(flow)` is a checkpoint's knob frame,
`twobunch_cnf.charges.injected_charges_nC` the per-bunch injected charge, and `twobunch_cnf.beam_io`
the ParticleGroup -> (n, 6) cloud convention used by preprocessing.

## Bundles

`src/twobunch_cnf/resources/<name>/`: `model.ckpt` (converted, slim: weights + hparams), `norm.json`,
`run.json` (training provenance) and `bundle.yaml` (checkpoint sha256; knob table in frame order with
bounds, baseline, unit and description; the handoff element's reference values per plane; the measured
S2E clock offset; provenance). Shipped: `c_b3_ema` (37 knobs, XTCAVF, beam latent 8). Export a new
one with `scripts/export_bundle.py` (knob units from `configs/knob_meta.yaml`) and write its golden
with `scripts/make_golden.py`; both need twobunch-s2e-rl on `PYTHONPATH`.

## Checkpoints from twobunch-s2e-rl

`main` implements the feature set of the current best model, C_B3 (rational-quadratic-spline
coupling, two-part transmission head, optional beam latent, and the loss terms C_B3 trains with).
Checkpoints written by twobunch-s2e-rl carry the hparams of options that were removed; convert them
once:

```bash
twobunch-cnf-migrate <twobunch ckpt> --out <new.ckpt>
```

`migrate` requires the checkpoint to record its knob frame (`knob_keys`, `treaty_point`) and every
removed option to be off; anything else is refused. Loading is strict: a checkpoint with hparams
this `TwoBunchFlow` does not know raises instead of loading as a different model.

`configs/c_b3_ema.yaml` reproduces C_B3's training recipe (`twobunch-cnf-train --config
configs/c_b3_ema.yaml --processed ... --out ...`; the data paths it used are in its header).

## Branches

- `main`: the released feature set only.
- `archive/unpruned-7ac3617`: the full, unpruned code as imported from twobunch-s2e-rl (runnable),
  including the experimental options removed from `main` (affine coupling, single-sigmoid T head,
  centroid / chirp-curve / backbone heads, single-bunch training).
- Experimental options live on worktree branches off `main` (or off the archive branch to revive a
  removed one) and merge into `main` only once a released model uses them.

## Command line

| command | module |
|---|---|
| `twobunch-cnf-train --processed T.h5 --out RUN [--config C.yaml]` | `twobunch_cnf.train` |
| `twobunch-cnf-eval --ckpt CKPT --processed T.h5 --out DIR` | `twobunch_cnf.eval` |
| `twobunch-cnf-plot-training RUN --out FIG.png` | `twobunch_cnf.plot_training` |
| `twobunch-cnf-migrate SRC --out DST` | `twobunch_cnf.migrate` |

All paths are explicit; nothing defaults to a repo-relative location.

## License

To be decided before the package is depended on by virtual-accelerator.
