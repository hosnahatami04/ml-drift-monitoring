import numpy as np
import pandas as pd
import pytest

from driftmon.data import schema


def make_adult_frame(n_rows: int, seed: int = 0, positive_rate: float = 0.25) -> pd.DataFrame:
    """Synthetic frame matching the Adult schema, with a learnable signal:
    higher education-num and hours-per-week push the label towards >50K."""
    rng = np.random.default_rng(seed)
    education_num = rng.integers(1, 17, n_rows)
    hours = rng.integers(1, 100, n_rows)
    signal = (education_num - 8) / 8 + (hours - 40) / 60 + rng.normal(0, 0.8, n_rows)
    threshold = np.quantile(signal, 1 - positive_rate)
    income = np.where(signal > threshold, ">50K", "<=50K")

    df = pd.DataFrame(
        {
            "age": rng.integers(17, 91, n_rows),
            "workclass": rng.choice(schema.CATEGORIES["workclass"], n_rows),
            "fnlwgt": rng.integers(10_000, 1_000_000, n_rows),
            "education": rng.choice(schema.CATEGORIES["education"], n_rows),
            "education-num": education_num,
            "marital-status": rng.choice(schema.CATEGORIES["marital-status"], n_rows),
            "occupation": rng.choice(schema.CATEGORIES["occupation"], n_rows),
            "relationship": rng.choice(schema.CATEGORIES["relationship"], n_rows),
            "race": rng.choice(schema.CATEGORIES["race"], n_rows),
            "sex": rng.choice(schema.CATEGORIES["sex"], n_rows),
            "capital-gain": rng.integers(0, 10_000, n_rows),
            "capital-loss": rng.integers(0, 2_000, n_rows),
            "hours-per-week": hours,
            "native-country": rng.choice(schema.CATEGORIES["native-country"], n_rows),
            "income": income,
        }
    )
    return df


@pytest.fixture
def adult_frame() -> pd.DataFrame:
    return make_adult_frame(2000)
