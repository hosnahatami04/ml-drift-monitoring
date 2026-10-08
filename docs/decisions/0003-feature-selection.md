# 0003. Feature selection for the income model

Date: 2026-10-08

## Status

Accepted

## Context

The UCI Adult frame carries fifteen columns. Three of them are problematic as
model inputs for different reasons: `fnlwgt` is the census sampling weight,
`education` is a string duplicate of `education-num`, and `race`, `sex` and
`native-country` would turn the model into a demographic classifier, which
this project does not want to build or monitor.

## Decision

The model uses nine features. Numeric: `age`, `education-num`,
`capital-gain`, `capital-loss`, `hours-per-week`. Categorical: `workclass`,
`marital-status`, `occupation`, `relationship`.

- `fnlwgt` dropped: it describes the sampling design, not the person; any
  signal in it is an artifact of how the census was collected.
- `education` dropped, `education-num` kept: same information, and the
  ordinal encoding is directly usable by both models.
- `race`, `sex`, `native-country` dropped from model inputs. Protected
  attributes stay out of the model; drift monitoring covers model features
  only, so these columns are also outside the monitored surface.

All fifteen columns are still kept in the processed splits: the simulator and
future analyses may need them, and dropping them is the pipeline's job, not
the data layer's.

## Alternatives considered

- Keep all columns and let regularization sort it out: leaves the protected
  attributes in the model and the monitor for no measurable gain.
- Drop the mostly-zero `capital-gain`/`capital-loss`: they are legitimate
  signal (large gains strongly predict the positive class); their point mass
  at zero is a detector calibration concern, not a modeling one.

## Consequences

Slightly lower headline AUC than published results that use all columns.
Monitoring covers exactly the nine model inputs plus the prediction
distribution and the error rate.
