# Evaluation results

Written by `driftmon evaluate` from `results.json` in the same directory; every number below comes from that file.

- Evaluation seeds 100-129 (30 streams per cell), severities: low, medium, high
- Stream: 60 windows of 1000 rows, drift from window 20, label delay 3, detection horizon 20
- Thresholds frozen by calibration on seeds 0-19: alpha 0.001 (bonferroni), k=1, KS D ≥ 0.04, PSI ≥ 0.1, Cramér's V ≥ 0.1, Page-Hinkley delta 0.005 / lambda 0.05
- Model artifact `model-e0787954c486` (model.pkl sha256 `e95949f30b43`), reference AUC 0.9272; config sha256 `3ebcd2fda0bf`
- 95% intervals: percentile bootstrap over streams, 1000 resamples, seed 2026

## Detection KPIs

| KPI | Scope | Value [95% CI] | Target | Status |
| --- | --- | --- | --- | --- |
| Detection rate, sudden covariate | medium severity, evaluation seeds | 100.0% [100.0%, 100.0%] | ≥ 95.0% | met |
| Detection rate, gradual covariate | medium severity, evaluation seeds | 100.0% [100.0%, 100.0%] | ≥ 90.0% | met |
| Detection rate, concept (performance_drift only) | medium severity, evaluation seeds | 100.0% [100.0%, 100.0%] | ≥ 80.0% | met |
| Median time to detect, sudden covariate | medium severity, evaluation seeds | 0.0 [0.0, 0.0] windows | ≤ 1 | met |
| Median time to detect, gradual covariate | medium severity, evaluation seeds | 6.0 [6.0, 7.0] windows | ≤ 6 | met |
| Median time to detect, concept | medium severity, evaluation seeds | 6.0 [5.0, 7.0] windows | ≤ 7 | met |
| Stream false alarm rate (control) | control streams, evaluation seeds | 6.7% [0.0%, 16.7%] | ≤ 10.0% | met |
| Per-window false alarm rate (control) | control streams, evaluation seeds | 0.28% [0.00%, 0.78%] | ≤ 0.2% | **missed** |
| Alert precision, drifted streams | medium severity, evaluation seeds, all drift scenarios pooled | 99.3% [98.4%, 99.8%] | ≥ 90.0% | met |
| Median lead time, harmful covariate drift | covariate scenarios, all severities, streams with degradation | 9.0 [7.0, 11.0] windows | ≥ 1 | met |
| Harmful-drift share | medium severity, evaluation seeds, all drift scenarios pooled | 17.5% [11.7%, 25.0%] | report | - |
| Input-detector alert rate on concept minus control | medium severity, evaluation seeds, paired by seed | 0.0% [0.0%, 0.0%] | ≤ 2.0% | met |
| Largest deviation from scipy (KS, chi-square) | random inputs | 0.0e+00 | ≤ 1e-06 | met |

Time to detect counts windows from the drift start to the first correct alert
(0 = caught in the first drifted window). A performance alert is raised when the
window's labels arrive, so concept detection cannot be faster than the label delay.

## By scenario and severity

| Scenario | Severity | Detection rate | Median TTD | Alert precision | Harmful share | Median lead time |
| --- | --- | --- | --- | --- | --- | --- |
| concept | low | 46.7% [30.0%, 63.3%] | 11.0 [8.0, 16.0] | 93.9% [85.5%, 100.0%] | 0.0% [0.0%, 0.0%] | no harmful streams |
| concept | medium | 100.0% [100.0%, 100.0%] | 6.0 [5.0, 7.0] | 98.3% [95.0%, 100.0%] | 66.7% [50.0%, 80.1%] | 6.5 [1.5, 11.0] |
| concept | high | 100.0% [100.0%, 100.0%] | 4.0 [4.0, 5.0] | 98.4% [95.4%, 100.0%] | 100.0% [100.0%, 100.0%] | -3.0 [-3.0, -3.0] |
| gradual_covariate | low | 100.0% [100.0%, 100.0%] | 13.5 [12.0, 14.5] | 97.3% [92.3%, 100.0%] | 0.0% [0.0%, 0.0%] | no harmful streams |
| gradual_covariate | medium | 100.0% [100.0%, 100.0%] | 6.0 [6.0, 7.0] | 99.2% [97.7%, 100.0%] | 0.0% [0.0%, 0.0%] | no harmful streams |
| gradual_covariate | high | 100.0% [100.0%, 100.0%] | 2.0 [2.0, 2.0] | 99.5% [98.4%, 100.0%] | 100.0% [100.0%, 100.0%] | 7.0 [6.0, 9.0] |
| seasonal | low | 100.0% [100.0%, 100.0%] | 2.0 [2.0, 2.0] | 98.7% [96.2%, 100.0%] | 0.0% [0.0%, 0.0%] | no harmful streams |
| seasonal | medium | 100.0% [100.0%, 100.0%] | 1.0 [1.0, 1.0] | 99.3% [98.0%, 100.0%] | 0.0% [0.0%, 0.0%] | no harmful streams |
| seasonal | high | 100.0% [100.0%, 100.0%] | 1.0 [1.0, 1.0] | 99.5% [98.4%, 100.0%] | 10.0% [0.0%, 23.3%] | 37.0 [27.0, 37.0] |
| sudden_covariate | low | 100.0% [100.0%, 100.0%] | 0.0 [0.0, 0.0] | 99.2% [97.6%, 100.0%] | 0.0% [0.0%, 0.0%] | no harmful streams |
| sudden_covariate | medium | 100.0% [100.0%, 100.0%] | 0.0 [0.0, 0.0] | 99.5% [98.6%, 100.0%] | 3.3% [0.0%, 10.0%] | 12.0 [12.0, 12.0] |
| sudden_covariate | high | 100.0% [100.0%, 100.0%] | 0.0 [0.0, 0.0] | 99.5% [98.6%, 100.0%] | 63.3% [46.7%, 80.0%] | 12.0 [11.0, 28.0] |

## Control streams

- 30 streams with no injected drift
- Stream false alarm rate: 6.7% [0.0%, 16.7%]
- Per-window false alarm rate: 0.28% [0.00%, 0.78%]
- Streams with at least one alarm, by type: feature_drift 0, performance_drift 2, prediction_drift 0
- Alerts by type: feature_drift 0, performance_drift 5, prediction_drift 0
- Streams whose rolling AUC dropped below the degradation line anyway: 0.0% [0.0%, 0.0%] (noise floor for harmful share)

## Detector correctness

Largest absolute deviation from scipy over random inputs:

- chi2_p_value: 0.00e+00
- chi2_statistic: 0.00e+00
- ks_p_value: 0.00e+00
- ks_statistic: 0.00e+00

## Ablations

All arms reuse the frozen thresholds; none is re-tuned, so an arm's false alarm
rate moves along with its sensitivity. Method, gate and correction arms change only
the input-side decision and are scored on feature_drift alerts alone. The
window-size arms keep the traffic volume fixed (60k rows per stream; drift start,
horizon, label delay and scenario timing scaled in rows) and are scored on all
alerts, so their time to detect is also given in rows.

Detection counts any relevant alert inside the horizon, so an arm with a high
control false alarm rate also "detects" by chance; read detection next to the
false alarm columns.

### method (feature_drift)

| Arm | gradual_covariate detection | gradual_covariate median TTD | seasonal detection | seasonal median TTD | sudden_covariate detection | sudden_covariate median TTD | Control stream FA | Control per-window FA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all tests (frozen) | 100.0% [100.0%, 100.0%] | 6.0 [6.0, 7.0] | 100.0% [100.0%, 100.0%] | 1.0 [1.0, 1.0] | 100.0% [100.0%, 100.0%] | 0.0 [0.0, 0.0] | 0.0% [0.0%, 0.0%] | 0.00% [0.00%, 0.00%] |
| KS only | 100.0% [100.0%, 100.0%] | 6.0 [6.0, 6.5] | 100.0% [100.0%, 100.0%] | 1.0 [1.0, 1.0] | 100.0% [100.0%, 100.0%] | 0.0 [0.0, 0.0] | 0.0% [0.0%, 0.0%] | 0.00% [0.00%, 0.00%] |
| PSI only | 100.0% [100.0%, 100.0%] | 9.0 [9.0, 10.0] | 100.0% [100.0%, 100.0%] | 2.0 [2.0, 2.0] | 100.0% [100.0%, 100.0%] | 0.0 [0.0, 0.0] | 0.0% [0.0%, 0.0%] | 0.00% [0.00%, 0.00%] |

Control streams with an alarm, by type:

- all tests (frozen): feature_drift 0, performance_drift 0, prediction_drift 0
- KS only: feature_drift 0, performance_drift 0, prediction_drift 0
- PSI only: feature_drift 0, performance_drift 0, prediction_drift 0

### effect gate (feature_drift)

| Arm | gradual_covariate detection | gradual_covariate median TTD | seasonal detection | seasonal median TTD | sudden_covariate detection | sudden_covariate median TTD | Control stream FA | Control per-window FA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| with gate (frozen) | 100.0% [100.0%, 100.0%] | 6.0 [6.0, 7.0] | 100.0% [100.0%, 100.0%] | 1.0 [1.0, 1.0] | 100.0% [100.0%, 100.0%] | 0.0 [0.0, 0.0] | 0.0% [0.0%, 0.0%] | 0.00% [0.00%, 0.00%] |
| without KS/chi2 gate | 100.0% [100.0%, 100.0%] | 6.0 [6.0, 7.0] | 100.0% [100.0%, 100.0%] | 1.0 [1.0, 1.0] | 100.0% [100.0%, 100.0%] | 0.0 [0.0, 0.0] | 6.7% [0.0%, 16.7%] | 0.11% [0.00%, 0.28%] |

Control streams with an alarm, by type:

- with gate (frozen): feature_drift 0, performance_drift 0, prediction_drift 0
- without KS/chi2 gate: feature_drift 2, performance_drift 0, prediction_drift 0

### correction (feature_drift)

| Arm | gradual_covariate detection | gradual_covariate median TTD | seasonal detection | seasonal median TTD | sudden_covariate detection | sudden_covariate median TTD | Control stream FA | Control per-window FA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bonferroni (frozen) | 100.0% [100.0%, 100.0%] | 6.0 [6.0, 7.0] | 100.0% [100.0%, 100.0%] | 1.0 [1.0, 1.0] | 100.0% [100.0%, 100.0%] | 0.0 [0.0, 0.0] | 0.0% [0.0%, 0.0%] | 0.00% [0.00%, 0.00%] |
| benjamini-hochberg | 100.0% [100.0%, 100.0%] | 6.0 [6.0, 7.0] | 100.0% [100.0%, 100.0%] | 1.0 [1.0, 1.0] | 100.0% [100.0%, 100.0%] | 0.0 [0.0, 0.0] | 0.0% [0.0%, 0.0%] | 0.00% [0.00%, 0.00%] |

Control streams with an alarm, by type:

- bonferroni (frozen): feature_drift 0, performance_drift 0, prediction_drift 0
- benjamini-hochberg: feature_drift 0, performance_drift 0, prediction_drift 0

### window size (all alerts)

| Arm | concept detection | concept median TTD (rows) | gradual_covariate detection | gradual_covariate median TTD (rows) | seasonal detection | seasonal median TTD (rows) | sudden_covariate detection | sudden_covariate median TTD (rows) | Control stream FA | Control per-window FA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| n=250 | 100.0% [100.0%, 100.0%] | 2875.0 [750.0, 3250.0] | 100.0% [100.0%, 100.0%] | 500.0 [250.0, 1375.0] | 100.0% [100.0%, 100.0%] | 625.0 [250.0, 1000.0] | 100.0% [100.0%, 100.0%] | 0.0 [0.0, 0.0] | 100.0% [100.0%, 100.0%] | 14.83% [13.78%, 15.93%] |
| n=500 | 100.0% [100.0%, 100.0%] | 4500.0 [4000.0, 5000.0] | 100.0% [100.0%, 100.0%] | 7500.0 [6500.0, 8000.0] | 100.0% [100.0%, 100.0%] | 1000.0 [1000.0, 1250.0] | 100.0% [100.0%, 100.0%] | 0.0 [0.0, 0.0] | 80.0% [66.7%, 93.3%] | 2.17% [1.39%, 3.14%] |
| n=1000 | 100.0% [100.0%, 100.0%] | 6000.0 [5000.0, 7000.0] | 100.0% [100.0%, 100.0%] | 6000.0 [6000.0, 7000.0] | 100.0% [100.0%, 100.0%] | 1000.0 [1000.0, 1000.0] | 100.0% [100.0%, 100.0%] | 0.0 [0.0, 0.0] | 6.7% [0.0%, 16.7%] | 0.28% [0.00%, 0.78%] |
| n=2000 | 90.0% [76.7%, 100.0%] | 14000.0 [12000.0, 16000.0] | 100.0% [100.0%, 100.0%] | 5000.0 [4000.0, 6000.0] | 100.0% [100.0%, 100.0%] | 2000.0 [2000.0, 2000.0] | 100.0% [100.0%, 100.0%] | 0.0 [0.0, 0.0] | 0.0% [0.0%, 0.0%] | 0.00% [0.00%, 0.00%] |

Control streams with an alarm, by type:

- n=250: feature_drift 30, performance_drift 30, prediction_drift 1
- n=500: feature_drift 1, performance_drift 24, prediction_drift 1
- n=1000: feature_drift 0, performance_drift 2, prediction_drift 0
- n=2000: feature_drift 0, performance_drift 0, prediction_drift 0

## What didn't work

- Per-window false alarm rate (control): 0.28% [0.00%, 0.78%] against a target of ≤ 0.2%.
- Lead time is measured on 53 covariate streams whose rolling AUC actually fell below the degradation line.

## Figures

![Detection rate by severity](figures/detection_vs_severity.png)
![Time to detect](figures/ttd_by_scenario.png)
![Window size ablation](figures/window_size_ablation.png)
![Lead time](figures/lead_time.png)
