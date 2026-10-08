# Severity calibration

Measured on calibration seeds 0-19 (20 streams per cell), active drifted windows only.
Reference AUC: 0.9270.

Shift is the mean absolute displacement of the affected feature's
window mean, in reference standard deviations. Flip rate is the
realized label-flip share among drifted-window rows (concept only).

| Scenario | Severity | Shift (sd) | Flip rate | AUC (drifted) | AUC delta |
| --- | --- | --- | --- | --- | --- |
| concept | low | - | 0.150 | 0.9159 | -0.0110 |
| concept | medium | - | 0.302 | 0.9053 | -0.0217 |
| concept | high | - | 0.504 | 0.8907 | -0.0362 |
| gradual_covariate | low | 0.190 | - | 0.9218 | -0.0051 |
| gradual_covariate | medium | 0.467 | - | 0.9147 | -0.0123 |
| gradual_covariate | high | 1.471 | - | 0.8868 | -0.0401 |
| seasonal | low | 0.205 | - | 0.9265 | -0.0005 |
| seasonal | medium | 0.411 | - | 0.9266 | -0.0003 |
| seasonal | high | 0.815 | - | 0.9283 | +0.0013 |
| sudden_covariate | low | 0.257 | - | 0.9195 | -0.0075 |
| sudden_covariate | medium | 0.554 | - | 0.9137 | -0.0133 |
| sudden_covariate | high | 1.202 | - | 0.9053 | -0.0217 |
