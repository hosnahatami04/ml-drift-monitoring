"""Chi-square test of independence for categorical features.

The statistic is computed by hand on the reference-vs-current contingency
table; only the chi-square distribution itself comes from scipy. Matches
scipy.stats.chi2_contingency(correction=False) on the same table.
"""

from collections import Counter
from typing import Any

import numpy as np
from scipy.stats import chi2 as chi2_distribution

from driftmon.detectors.base import TestResult

OTHER_CATEGORY = "__other__"

MIN_EXPECTED = 5.0

StrArray = "np.ndarray[tuple[int], np.dtype[np.object_]]"


def pearson_chi2(table: "np.ndarray[Any, np.dtype[np.float64]]") -> tuple[float, float, int]:
    """Pearson chi-square on a k x 2 contingency table: statistic, p, dof."""
    row_totals = table.sum(axis=1, keepdims=True)
    col_totals = table.sum(axis=0, keepdims=True)
    total = table.sum()
    expected = row_totals @ col_totals / total
    statistic = float(((table - expected) ** 2 / expected).sum())
    dof = (table.shape[0] - 1) * (table.shape[1] - 1)
    p_value = float(chi2_distribution.sf(statistic, dof))
    return statistic, p_value, dof


def _merged_counts(
    reference: "np.ndarray[tuple[int], np.dtype[np.object_]]",
    current: "np.ndarray[tuple[int], np.dtype[np.object_]]",
) -> tuple[list[str], list[tuple[float, float]]]:
    ref_counts = Counter(str(v) for v in reference.tolist())
    cur_counts = Counter(str(v) for v in current.tolist())

    # Categories the reference never saw cannot have a meaningful expected
    # count of their own; they are pooled into the catch-all bucket.
    categories: dict[str, tuple[float, float]] = {}
    other = [0.0, 0.0]
    for category in sorted(ref_counts):
        categories[category] = (float(ref_counts[category]), float(cur_counts.get(category, 0)))
    for category in sorted(set(cur_counts) - set(ref_counts)):
        other[1] += cur_counts[category]

    n_ref = float(sum(ref_counts.values()))
    n_cur = float(sum(cur_counts.values()))
    total = n_ref + n_cur
    kept: dict[str, tuple[float, float]] = {}
    for category, (ref_n, cur_n) in categories.items():
        category_total = ref_n + cur_n + 0.0
        expected_ref = category_total * n_ref / total
        expected_cur = category_total * n_cur / total
        if expected_ref < MIN_EXPECTED or expected_cur < MIN_EXPECTED:
            other[0] += ref_n
            other[1] += cur_n
        else:
            kept[category] = (ref_n, cur_n)

    names = list(kept)
    rows = list(kept.values())
    if other[0] + other[1] > 0:
        names.append(OTHER_CATEGORY)
        rows.append((other[0], other[1]))
    return names, rows


def chi2_test(
    reference: "np.ndarray[tuple[int], np.dtype[np.object_]]",
    current: "np.ndarray[tuple[int], np.dtype[np.object_]]",
    feature: str,
) -> TestResult:
    if reference.size == 0 or current.size == 0:
        raise ValueError("reference and current samples must be non-empty")

    _, rows = _merged_counts(reference, current)
    table = np.asarray(rows, dtype=float)
    if table.shape[0] < 2:
        # Everything collapsed into one bucket (constant feature or tiny
        # window): no distributional evidence either way.
        return TestResult(
            feature=feature, method="chi2", statistic=0.0, p_value=1.0, effect_size=0.0
        )

    statistic, p_value, _ = pearson_chi2(table)
    # Cramer's V for a k x 2 table.
    effect = float(np.sqrt(statistic / table.sum()))
    return TestResult(
        feature=feature,
        method="chi2",
        statistic=statistic,
        p_value=p_value,
        effect_size=effect,
    )
