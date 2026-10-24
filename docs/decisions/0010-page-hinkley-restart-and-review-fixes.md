# 0010. Restart Page-Hinkley after an alarm

## Status

Accepted. Changes the evaluation published with v1.0.0.

## Context

The v1.0.0 evaluation missed one target: 0.28% of control windows raised an
alert, against at most 0.2%. Every one of those alerts came from Page-Hinkley.
The detector was never reset after it fired, so once its statistic crossed the
threshold it stayed there, and one false alarm repeated after every cooldown.
A code review after the release found the same mechanism and a handful of
smaller defects in the evaluation itself.

This change was made after the evaluation numbers had been seen. Thresholds are
still chosen on calibration seeds only, and the v1.0.0 results stay in the git
history under the tag, so the effect of the change can be checked.

## Decision

- Page-Hinkley restarts after every alarm. The restart lives in
  `Monitor.observe_error_rate`, so the live runner, the state rebuilt after a
  restart and the evaluation harness behave identically.
- Thresholds were recalibrated on seeds 0-19 with the unchanged grid and
  objective (ADR 0008), then evaluated once on seeds 100-129.
- Review fixes that change the report:
  - KPI targets moved from code to `configs/default.yaml`.
  - The input-silence KPI on concept streams is report-only. Concept streams
    reuse control's inputs, so it is zero by construction and could never fail.
  - Window-size arms now scale the alert cooldown with the traffic, and the
    report lists each arm's real timing in rows, including where a value had to
    be rounded.
  - The CI gate bounds the per-window false alarm rate separately. One
    tolerance for every KPI let it rise fifteenfold unnoticed.

## Alternatives considered

- Keep the detector as it was and report the miss: honest, but it leaves a
  known defect in the alerting path.
- Lower lambda or raise the cooldown instead: that trades detection speed for
  fewer repeats, and does not fix a detector that stays in alarm forever.

## Consequences

After a real change the detector has to gather fresh evidence before it fires
again, so sustained performance drift produces fewer repeat alerts. Comparing
the two published evaluations shows what the restart changed; the before and
after numbers are in the pull request that introduced this record.
