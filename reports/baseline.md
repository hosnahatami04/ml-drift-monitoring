# Baseline models

Logistic regression baseline and XGBoost on the UCI Adult income task
(binary classification, target: income > 50K).

## Data

- Raw rows after deduplication and split: 48790 across committed splits
- Duplicate rows removed: 52
- Seed: 42

| Split | Rows |
| --- | --- |
| reference | 4879 |
| stream_pool | 14637 |
| train | 24395 |
| validation | 4879 |

## Metrics

| Model | Split | AUC | Accuracy | F1 | Brier |
| --- | --- | --- | --- | --- | --- |
| logistic_regression | validation | 0.9097 | 0.8555 | 0.6738 | 0.1009 |
| logistic_regression | reference | 0.9051 | 0.8543 | 0.6609 | 0.1016 |
| xgboost | validation | 0.9308 | 0.8735 | 0.7192 | 0.0874 |
| xgboost | reference | 0.9270 | 0.8739 | 0.7136 | 0.0888 |

## Environment

| Library | Version |
| --- | --- |
| pandas | 3.0.6 |
| scikit-learn | 1.9.1 |
| xgboost | 3.2.0 |
| numpy | 2.4.6 |
