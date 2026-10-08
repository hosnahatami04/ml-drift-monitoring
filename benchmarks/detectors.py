"""Time all detectors on one monitoring window.

Shapes mirror production: reference ~4900 rows, window 1000 rows, five numeric
and four categorical features, plus prediction drift and the window decision.

Usage: uv run python benchmarks/detectors.py
"""

import statistics
import time

import numpy as np

from driftmon.detectors.chi2 import chi2_test
from driftmon.detectors.ks import ks_test, prediction_drift_test
from driftmon.detectors.multiple_testing import DecisionConfig, decide_window
from driftmon.detectors.psi import psi_categorical, psi_numeric, quantile_bin_edges

REFERENCE_ROWS = 4900
WINDOW_ROWS = 1000
N_NUMERIC = 5
N_CATEGORICAL = 4
N_CATEGORIES = 10
REPEATS = 50


def run_window(
    numeric_ref: list[np.ndarray],
    numeric_edges: list[list[float]],
    categorical_freqs: list[dict[str, float]],
    numeric_cur: list[np.ndarray],
    categorical_ref: list[np.ndarray],
    categorical_cur: list[np.ndarray],
    probs_ref: np.ndarray,
    probs_cur: np.ndarray,
) -> None:
    results = []
    for i in range(N_NUMERIC):
        name = f"num_{i}"
        results.append(ks_test(numeric_ref[i], numeric_cur[i], name))
        results.append(psi_numeric(numeric_ref[i], numeric_cur[i], name, numeric_edges[i]))
    for i in range(N_CATEGORICAL):
        name = f"cat_{i}"
        results.append(chi2_test(categorical_ref[i], categorical_cur[i], name))
        results.append(psi_categorical(categorical_freqs[i], categorical_cur[i], name))
    results.append(prediction_drift_test(probs_ref, probs_cur))
    decide_window(results, DecisionConfig())


def main() -> None:
    rng = np.random.default_rng(42)
    categories = np.array([f"c{i}" for i in range(N_CATEGORIES)], dtype=object)

    numeric_ref = [rng.normal(0, 1, REFERENCE_ROWS) for _ in range(N_NUMERIC)]
    numeric_edges = [quantile_bin_edges(col) for col in numeric_ref]
    categorical_ref = [rng.choice(categories, REFERENCE_ROWS) for _ in range(N_CATEGORICAL)]
    categorical_freqs = []
    for col in categorical_ref:
        values, counts = np.unique(col, return_counts=True)
        categorical_freqs.append(
            {str(v): float(c) / col.size for v, c in zip(values, counts, strict=True)}
        )
    probs_ref = rng.uniform(0, 1, REFERENCE_ROWS)

    durations = []
    for _ in range(REPEATS):
        numeric_cur = [rng.normal(0.1, 1, WINDOW_ROWS) for _ in range(N_NUMERIC)]
        categorical_cur = [rng.choice(categories, WINDOW_ROWS) for _ in range(N_CATEGORICAL)]
        probs_cur = rng.uniform(0, 1, WINDOW_ROWS)
        start = time.perf_counter()
        run_window(
            numeric_ref,
            numeric_edges,
            categorical_freqs,
            numeric_cur,
            categorical_ref,
            categorical_cur,
            probs_ref,
            probs_cur,
        )
        durations.append((time.perf_counter() - start) * 1000)

    print(f"window of {WINDOW_ROWS} rows, {REPEATS} repeats")
    print(f"median: {statistics.median(durations):.2f} ms")
    print(f"p95:    {sorted(durations)[int(0.95 * REPEATS) - 1]:.2f} ms")
    print(f"max:    {max(durations):.2f} ms")


if __name__ == "__main__":
    main()
