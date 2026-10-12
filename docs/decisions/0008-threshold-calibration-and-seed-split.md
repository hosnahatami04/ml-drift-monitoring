# 0008. Threshold calibration on held-out seeds

Date: 2026-10-11

## Status

Accepted. The Page-Hinkley consequence below is addressed by ADR 0010.

## Context

Every detector has knobs (alpha, effect gates, k, Page-Hinkley delta and
lambda), and the KPIs are only honest if those knobs were not chosen while
looking at the streams the KPIs are computed on. A second risk is a harness
that measures something other than production: a re-implemented decision
rule would make the numbers meaningless.

## Decision

- Seeds 0-19 are calibration seeds, 100-129 evaluation seeds. Calibration
  refuses any evaluation seed; a test enforces it. KPIs, ablations and the
  report use evaluation seeds only.
- Calibration is a grid search (grid in `configs/default.yaml`) over control
  plus every drift scenario at medium severity. Objective: highest mean
  detection rate across the drift scenarios with a control stream false alarm
  rate at or below 10%; ties go to lower median time to detect, then fewer
  false alarms. The winner is written to `configs/thresholds.yaml` and frozen.
- The harness runs the expensive, threshold-independent half of
  `Monitor.process_window` once per stream, then replays alerting through
  `Monitor.decide`, Page-Hinkley and the `AlertManager` in the live runner's
  order. A test checks the replay against the live path window by window.
- Input and performance alerts are scored separately and combined as a union.
  This is exact, not an approximation, because the cooldown is per alert type.
- Ablations reuse the frozen thresholds and are never re-tuned. The
  window-size arms keep the traffic volume fixed (60k rows; drift start,
  horizon, label delay and scenario timing scaled in rows).
- 95% intervals are percentile bootstraps over streams (1000 resamples, fixed
  seed). `results.json` has no timestamps or paths, and `make reproduce`
  checks that two runs produce the same bytes.

## Alternatives considered

- Tune on all seeds and report on the same streams: optimistic by
  construction.
- Calibrate per scenario: thresholds would then depend on knowing the drift
  type in advance, which production never does.
- Re-tune each ablation arm: answers a different question (best achievable
  per arm) at many times the cost. The frozen comparison shows what changing
  one component does to the deployed configuration.

## Consequences

- Twenty control streams make the false alarm constraint coarse: 10% means
  at most two alarming streams, so the evaluation rate can land on either
  side of the target.
- Thresholds are specific to the 1000-row window. The window-size ablation
  shows the PSI gate and Page-Hinkley parameters do not transfer to smaller
  windows, and any change of window size needs recalibration.
- Page-Hinkley is not reset after an alarm, so one false alarm on a clean
  stream repeats after every cooldown. This is what pushes the per-window
  false alarm rate over target (see `reports/results.md`). Resetting after an
  alarm would be a design change that needs recalibration on calibration
  seeds, and it was not changed after the evaluation numbers were seen.
- Bit-for-bit reproducibility holds on one machine. XGBoost trained on a
  different CPU architecture gave a slightly different model with the same
  library versions, so `results.json` records the model file's sha256.
