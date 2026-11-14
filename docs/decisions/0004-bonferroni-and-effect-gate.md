# 0004. Bonferroni by default, with an effect-size gate

## Status

Accepted

## Context

Each monitoring window runs several tests (KS and PSI per numeric feature,
chi-square and PSI per categorical feature, prediction drift). Uncorrected,
nine-plus tests per window at alpha=0.01 would alarm far too often: with 60
windows per stream, even a 1% per-window false-positive rate means almost half
of all clean streams raise at least one false alarm. Separately, large windows
make negligible shifts statistically significant: at n=1000, a 0.02-sigma
shift can clear any alpha while being operationally meaningless.

## Decision

Two filters, applied in order inside the window decision:

1. Bonferroni correction across the p-valued tests of the window. It controls
   the family-wise error rate, which matches the product goal: a very low
   stream-level false alarm rate. Benjamini-Hochberg is implemented alongside
   and reported in the ablation, since it trades strictness for sensitivity.
2. An effect-size gate per method: KS D, PSI value, and Cramer's V must each
   clear a minimum (calibrated on held-out seeds and frozen in
   `configs/thresholds.yaml`, ADR 0008). Significance says "this is not chance"; the gate says "and it is
   big enough to matter".

A feature counts as flagged when any of its tests passes both filters; the
window alarms when at least `k` features are flagged.

## Alternatives considered

- Benjamini-Hochberg as default: controls the false discovery rate, not the
  family-wise rate; more sensitive but noisier. Kept as the ablation arm.
- No effect gate, stricter alpha instead: alpha does not scale with n; any
  fixed alpha eventually flags trivial shifts as windows grow.

## Consequences

Detection of very small real drifts is delayed or suppressed by design. The
calibration phase must tune alpha and the gates jointly on calibration seeds
only.
