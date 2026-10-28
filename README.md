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
from twobunch_cnf.model import load_flow
from twobunch_cnf.artifacts import flow_knob_space

flow = load_flow("<ckpt>")                          # eval mode, CPU by default
frame = flow_knob_space(flow)                       # knob names + physical bounds stored in the ckpt
knobs_norm = frame.normalize(knobs_phys)            # (B, n_knob) physical -> the ckpt's [0, 1] frame
obs = flow.observables(knobs_norm, n=1024)          # differentiable beam observables + feasibility
drive = flow.sample_bunch(knobs_norm, 0, 4096)      # (B, 4096, 6) in (x, y, z, px, py, pz), m and eV/c
```

`twobunch_cnf.charges.injected_charges_nC` gives the per-bunch injected charge (charge at the treaty
point = T x injected); `twobunch_cnf.beam_io` holds the ParticleGroup <-> (n, 6) cloud convention
(drive = higher per-particle weight, charge ratio in the macroparticle counts).

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
