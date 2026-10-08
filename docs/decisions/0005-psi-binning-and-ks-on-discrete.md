# 0005. PSI binning scheme and KS behaviour on discrete columns

Date: 2026-10-08

## Status

Accepted

## Context

PSI needs a binning; the choice drives both sensitivity and stability. The
Adult numeric columns are awkward: `capital-gain` and `capital-loss` are
mostly zero, so equal-width bins would put nearly everything in one bin, and
naive quantile bins produce duplicate edges. The KS test has its own problem
on these columns: with heavy ties the asymptotic p-value is conservative
(true type-I error below alpha), so KS alone under-detects shifts in
zero-inflated or coarsely discrete features.

## Decision

- PSI bins come from the reference distribution: 10 quantile bins, duplicate
  edges merged (a point mass collapses several quantiles into one edge).
  Values outside the reference range fall into the first or last bin rather
  than being dropped. Frequencies are clipped at epsilon = 1e-4 before the
  log, so an empty bin cannot produce an infinite PSI.
- Categorical PSI uses one bin per category; categories unseen in the
  reference get the epsilon floor.
- The textbook 0.1 / 0.25 PSI thresholds are starting points only and are
  calibrated like every other threshold.
- The KS D statistic is computed by hand from the merged sorted ECDFs. For
  the p-value, the `kstwo` distribution (two-sided KS at the effective sample
  size) is evaluated through scipy, the same way scipy's own two-sample test
  does in asymptotic mode: the classical Kolmogorov limiting series differs
  from the finite-n distribution by several percent at window sizes used
  here, and scipy dropped it for exactly that reason. Distribution functions
  are the one scipy usage this project allows itself (same as the chi-square
  survival function).
- KS is paired with PSI on every numeric feature: with heavy ties the KS
  p-value is conservative, and PSI picks up the mass-at-a-point shifts KS
  underweights. The pairing is the mitigation, not a modified KS.

## Alternatives considered

- Equal-width bins: degenerate on zero-inflated columns.
- More bins (20+): noisier per-bin frequencies at window size 1000 for no
  detection gain.
- Exact KS p-values: cost grows with sample size and the exact distribution
  is also biased under ties; not worth it when PSI covers the weak spot.

## Consequences

PSI values depend on the reference binning, so bin edges are part of the
model artifact and are versioned with it. On discrete columns, KS alone is
known-conservative; evaluations should judge numeric features by the KS+PSI
pair, not KS in isolation.
