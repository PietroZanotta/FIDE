# Authoritative Deep Ritz GPU acceleration checkpoint

This checkpoint follows `FAST_PRODUCTION_3PCT_EVALUATION.md`. It concerns only
the nondifferentiated fixed-design Deep Ritz rescorer. It does not change the
validated fixed-feature Galerkin objective, its eta gradient, the frozen
production artifacts, any certificate, or the production incumbent.

## Outcome

The authoritative numerical graph is GPU-compatible and fixed checkpoints
evaluate equivalently on CPU and GPU to approximately machine precision. On an
NVIDIA GeForce RTX 5090 Laptop GPU, a complete fresh selection-side solve took
`1655.515 s`, versus `3833.403 s` for the same design and configuration on CPU:
a measured `2.315x` wall-clock speedup.

That is not yet enough to use GPU solves for scientific candidate ordering. A
paired GPU rerun from the same frozen initial checkpoint reversed the earlier
CPU ordering of eta0 and the previous tiny update. None of the four optimized
solutions declared L-BFGS convergence, although all passed the unchanged hard
physical certificates. The discrepancy is therefore optimizer/basin
sensitivity, not a CPU/GPU error in evaluation of a fixed solution.

## Hardware and software observation

- device: NVIDIA GeForce RTX 5090 Laptop GPU;
- device memory reported by the driver: 24,463 MiB;
- observed during a production authoritative solve: 100% GPU utilization,
  20,869 MiB used, 75 C, and approximately 144 W;
- JAX version: `0.8.3`;
- JAX detected `CudaDevice(id=0)` outside the restricted execution sandbox;
- float64 remained enabled throughout.

## Fixed-checkpoint equivalence

Two independently trained checkpoints were each evaluated without optimization
on both platforms. The objective and hard-certificate action agree at the
reported float64 precision.

| Frozen checkpoint | quantity | CPU | GPU | absolute difference |
|---|---:|---:|---:|---:|
| CPU-trained best-new | train Ritz J | -0.13896279760011462 | -0.13896279760011537 | 7.5e-16 |
| CPU-trained best-new | audit action | 0.27807001343600674 | 0.27807001343600680 | 5.6e-17 |
| GPU-trained best-new | train Ritz J | -0.13921607788389878 | -0.13921607788389950 | 7.2e-16 |
| GPU-trained best-new | audit action | 0.28023800599650034 | 0.28023800599650040 | 5.6e-17 |

For both checkpoints, CPU and GPU also produced the same validity decision and
the same projection, forcing, and certificate diagnostics to numerical
precision. Machine-readable values are in
`outputs/fast_production_3pct/gpu_authoritative_checkpoint/fixed_eval_cpu.json`
and `fixed_eval_gpu.json`.

## Timing

| Complete fresh fixed-design rescore | platform | elapsed seconds | speedup versus CPU |
|---|---:|---:|---:|
| best new continuous candidate | CPU | 3833.403 | 1.000x |
| best new continuous candidate | GPU | 1655.515 | 2.315x |
| eta0 paired rerun | GPU | 1749.166 | — |
| previous tiny update paired rerun | GPU | 1793.907 | — |

These timings include loading the already materialized isolated artifacts,
moment reconstruction, projection and forcing, Adam, full-bank L-BFGS, and the
held-out certificate. They are end-to-end timings, not asynchronous kernel
launch timings.

## Paired ordering result

The paired rerun used the same device, frozen artifact set, solver settings,
initial checkpoint, and restart policy for both designs.

| platform/run | eta0 action | tiny-update action | tiny minus eta0 | ordering |
|---|---:|---:|---:|---|
| earlier CPU authoritative cross-check | 0.2785663836 | 0.2741148392 | -0.0044515444 | tiny better |
| paired GPU rerun | 0.2763324858 | 0.2797088593 | +0.0033763735 | eta0 better |

All four rows passed exact selection risk, projection, ESS, forcing, geometry,
weak-residual, energy-residual, gauge, and moment-rate gates. The GPU eta0 and
tiny solutions had train energy-identity relative errors `0.009627` and
`0.002874`, respectively. Their L-BFGS convergence flags were both false; the
corresponding CPU convergence flags were also false.

## Interpretation

The fixed-checkpoint experiment rules out a material platform discrepancy in
the scientific equations or hard audit. The remaining explanation is that
small platform-level reduction differences alter a nonconvex optimization
trajectory whose stopping point is not demonstrably stationary. A single fresh
warm-started solve is therefore too noisy to resolve action changes of a few
`1e-3` reliably.

GPU execution is suitable for accelerating a restart/stationarity study, but a
GPU result must not replace or extend the current authoritative selection merely
because it is faster. Before another eta refinement is promoted, the isolated
workflow needs:

1. cached per-initialization authoritative solves;
2. identical initialization families for every compared eta;
3. paired action differences and hard-certificate status for every restart;
4. an explicit consensus/indeterminacy gate;
5. more efficient full-bank Adam and objective-gradient contractions, verified
   against the current implementation before being used in that study.

## Full-bank implementation benchmark

An optional compiled `lax.scan` implementation was added behind the
`compiled_full_bank` configuration flag and checked on the real production
shape `(13, 8192, 16, 2)`. With 40 Adam and 8 L-BFGS iterations it reproduced
the reference parameter vector and objective exactly, but its steady time was
`50.734 s` versus `51.024 s`, only `1.006x`. This is not a material speedup, so
the paired scientific restart study retains the established host-loop path.

Increasing the exact objective/gradient chunk size also did not help:

| chunk size | steady seconds for reduced solve | parameter relative difference | objective absolute difference |
|---:|---:|---:|---:|
| 512 | 8.718 | 0 | 0 |
| 1024 | 8.816 | 1.71e-13 | 2.78e-17 |
| 2048 | 8.768 | 4.21e-13 | 2.78e-17 |

The existing chunk size `512` remained fastest. These measurements show that
the dominant cost is the actual reverse-mode derivative work, rather than
Python launch overhead or undersized GPU batches. The benchmark is recorded in
`outputs/fast_production_3pct/authoritative_acceleration/result.json`.
The observed 20.9 GiB device footprint also leaves too little margin to justify
testing still larger production chunks on this 24.5 GiB device.

## Isolation

All new result files are below
`experiments/skyrmions_deep_ritz_full/outputs/fast_production_3pct/`. The frozen
production experiment, `src/`, and `native/` were read only. No production
incumbent was modified, and the sealed validation result was not used to tune
this checkpoint.

**Checkpoint decision: GPU FIXED-EVALUATION EQUIVALENCE AND ACCELERATION
VALIDATED; AUTHORITATIVE OPTIMIZER ORDERING NOT YET STABLE.**
