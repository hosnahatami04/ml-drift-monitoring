"""Shared result type and the contract every drift test implements."""

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


@dataclass(frozen=True)
class TestResult:
    """`p_value` is None for PSI, which has no sampling distribution.
    `flagged` is set by the window decision, not by the test: whether a result
    counts depends on the multiple-testing context of the whole window."""

    # Not a test case, despite the name; stops pytest from collecting it.
    __test__ = False

    feature: str
    method: str
    statistic: float
    p_value: float | None
    effect_size: float
    flagged: bool = False


class TwoSampleTest(Protocol):
    def __call__(self, reference: FloatArray, current: FloatArray, feature: str) -> TestResult: ...


def require_valid_sample(values: FloatArray, name: str) -> None:
    """Detectors are pure functions on clean arrays: empty windows or
    unhandled missing values are caller bugs and fail loudly."""
    if values.size == 0:
        raise ValueError(f"{name} sample is empty")
    if np.isnan(values).any():
        raise ValueError(f"{name} sample contains NaN; handle missing values before detection")
