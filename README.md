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
# or plain pip: pip install -e ".[eval,beam,test]"
```

## Use

```python
from twobunch_cnf.model import TwoBunchFlow

flow = TwoBunchFlow.load_from_checkpoint("<ckpt>", map_location="cpu").eval()
obs = flow.observables(knobs_norm, n=1024)          # knobs_norm: (B, n_knob) in the ckpt's [0, 1] frame
drive = flow.sample_bunch(knobs_norm, 0, 4096)      # (B, 4096, 6) in (x, y, z, px, py, pz), m and eV/c
```

The knob frame (names, physical bounds) is stored in the checkpoint: `twobunch_cnf.artifacts.flow_knob_space(flow)`.

## Command line

| command | module |
|---|---|
| `twobunch-cnf-train --processed T.h5 --out RUN [--config C.yaml]` | `twobunch_cnf.train` |
| `twobunch-cnf-eval --ckpt CKPT --processed T.h5 --out DIR` | `twobunch_cnf.eval` |
| `twobunch-cnf-plot-training RUN --out FIG.png` | `twobunch_cnf.plot_training` |

All paths are explicit; nothing defaults to a repo-relative location.

## License

To be decided before the package is depended on by virtual-accelerator.
