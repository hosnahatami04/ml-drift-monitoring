from pathlib import Path

import pandas as pd

from driftmon.config import SplitFractions
from driftmon.data import schema, split

from .conftest import make_adult_frame

FRACTIONS = SplitFractions()


def _positive_rate(df: pd.DataFrame) -> float:
    return float((df[schema.LABEL] == schema.POSITIVE_LABEL).mean())


def test_deduplicate_reports_removed_count(adult_frame: pd.DataFrame) -> None:
    duplicated = pd.concat([adult_frame, adult_frame.head(25)], ignore_index=True)

    deduped, removed = split.deduplicate(duplicated)

    assert removed == 25
    assert len(deduped) == len(adult_frame)


def test_row_ids_are_stable_and_unique(adult_frame: pd.DataFrame) -> None:
    first = split.assign_row_ids(adult_frame)
    second = split.assign_row_ids(adult_frame)

    assert first["row_id"].is_unique
    assert first["row_id"].tolist() == second["row_id"].tolist()


def test_split_proportions_within_one_percent(adult_frame: pd.DataFrame) -> None:
    prepared = split.prepare(adult_frame, FRACTIONS, seed=42)

    total = sum(len(part) for part in prepared.splits.values())
    assert total == len(adult_frame)
    expected = {
        "train": FRACTIONS.train,
        "validation": FRACTIONS.validation,
        "reference": FRACTIONS.reference,
        "stream_pool": FRACTIONS.stream_pool,
    }
    for name, fraction in expected.items():
        actual = len(prepared.splits[name]) / total
        assert abs(actual - fraction) <= 0.01, (name, actual)


def test_split_preserves_class_rate_within_one_percent(
    adult_frame: pd.DataFrame,
) -> None:
    prepared = split.prepare(adult_frame, FRACTIONS, seed=42)

    overall = _positive_rate(adult_frame)
    for name, part in prepared.splits.items():
        assert abs(_positive_rate(part) - overall) <= 0.01, name


def test_splits_share_no_row_ids(adult_frame: pd.DataFrame) -> None:
    prepared = split.prepare(adult_frame, FRACTIONS, seed=42)

    seen: set[int] = set()
    for part in prepared.splits.values():
        ids = set(part["row_id"])
        assert not seen & ids
        seen |= ids


def test_split_is_deterministic(adult_frame: pd.DataFrame) -> None:
    a = split.prepare(adult_frame, FRACTIONS, seed=42)
    b = split.prepare(adult_frame, FRACTIONS, seed=42)

    for name in split.SPLIT_NAMES:
        pd.testing.assert_frame_equal(a.splits[name], b.splits[name])


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    frame = make_adult_frame(400, seed=3)
    prepared = split.prepare(frame, FRACTIONS, seed=42)

    split.save_splits(prepared.splits, tmp_path)
    split.save_summary(prepared, tmp_path)

    loaded = split.load_split(tmp_path, "reference")
    pd.testing.assert_frame_equal(loaded, prepared.splits["reference"])
    summary = split.load_summary(tmp_path)
    assert summary["n_rows_raw"] == 400
    assert summary["split_sizes"]["train"] == len(prepared.splits["train"])
