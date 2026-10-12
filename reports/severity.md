# Severity calibration

Measured on calibration seeds 0-19 (20 streams per cell), active drifted windows only.
Reference AUC: 0.9272.

Shift is the mean absolute displacement of the affected feature's
window mean, in reference standard deviations. Flip rate is the
realized label-flip share among drifted-window rows inside the concept
subgroup, where the flip probability p applies (concept only).

| Scenario | Severity | Shift (sd) | Flip rate | AUC (drifted) | AUC delta |
| --- | --- | --- | --- | --- | --- |
| concept | low | - | 0.150 | 0.9163 | -0.0109 |
| concept | medium | - | 0.302 | 0.9057 | -0.0215 |
| concept | high | - | 0.504 | 0.8912 | -0.0360 |
| gradual_covariate | low | 0.190 | - | 0.9221 | -0.0050 |
| gradual_covariate | medium | 0.467 | - | 0.9149 | -0.0122 |
| gradual_covariate | high | 1.471 | - | 0.8869 | -0.0402 |
| seasonal | low | 0.205 | - | 0.9269 | -0.0003 |
| seasonal | medium | 0.411 | - | 0.9269 | -0.0002 |
| seasonal | high | 0.815 | - | 0.9289 | +0.0017 |
| sudden_covariate | low | 0.257 | - | 0.9199 | -0.0072 |
| sudden_covariate | medium | 0.554 | - | 0.9143 | -0.0129 |
| sudden_covariate | high | 1.202 | - | 0.9063 | -0.0209 |
