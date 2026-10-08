# 0006. Drift injection by importance resampling of real rows

Date: 2026-10-08

## Status

Accepted

## Context

The evaluation needs streams whose drift is fully known: what changed, when,
and how strongly. The naive approach, adding noise or shifting a column
arithmetically, produces rows that never existed (age 93.7, impossible
feature combinations) and breaks the joint distribution, which makes
detection artificially easy or meaningless.

## Decision

- Covariate drift is injected by importance resampling from the held-out
  stream pool with weight exp(beta * z), where z is the target feature
  standardized by reference statistics. Every emitted row is a real row;
  only the sampling distribution changes, so the marginal shifts while the
  joint structure stays intact.
- Sampling is with replacement: the pool holds ~14.6k rows and a stream
  consumes 60 x 1000, so windows could never be disjoint anyway.
- Every (stream seed, window index, purpose) triple seeds its own generator
  via SeedSequence. Windows are order- and parallelism-independent, all
  scenarios are bit-identical to control before the drift start, and a
  concept stream shares control's feature matrix for its whole length (its
  separate "labels" substream flips subgroup labels with probability p).
- The gradual ramp reaches 1/K at the drift-start window and 1 at window
  t0+K-1, so the first drifted window already carries a measurable share.
- Seasonal drift oscillates beta sinusoidally (period from config, feature:
  age, same mechanism as sudden). A window counts as "active" in the ground
  truth while |beta| is at least half the amplitude; near zero crossings
  there is nothing to detect and misses there should not count.
- Severity levels (beta 0.25/0.5/1.0, p 0.15/0.3/0.5) were validated with
  `driftmon calibrate-severity` on calibration seeds; measured shifts and
  AUC impact are in `reports/severity.md`. The levels are kept as configured
  since the three levels separate cleanly.

## Alternatives considered

- Additive shifts or synthetic noise: unreal rows, broken joints (above).
- Sampling without replacement: impossible at this pool-to-stream ratio.
- One generator per stream consumed sequentially: parallel evaluation would
  change results depending on scheduling; per-window seeding removes that.

## Consequences

Drift strength is bounded by what reweighting the pool can express: extreme
beta concentrates on the pool's tail rows and effective sample size drops.
Within the configured range this is measured, not assumed (severity report).
