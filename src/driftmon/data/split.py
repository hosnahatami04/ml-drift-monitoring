"""Deduplication, stable row ids and the stratified four-way split."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from driftmon.config import SplitFractions
from driftmon.data import schema

logger = logging.getLogger(__name__)

SPLIT_NAMES = ("train", "validation", "reference", "stream_pool")


@dataclass(frozen=True)
class PreparedData:
    splits: dict[str, pd.DataFrame]
    n_rows_raw: int
    n_duplicates_removed: int


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop exact duplicate rows (all columns including the label), keeping the
    first occurrence in file order so the result is deterministic."""
    before = len(df)
    deduped = df.drop_duplicates(keep="first").reset_index(drop=True)
    removed = before - len(deduped)
    logger.info("removed %d duplicate rows (%d -> %d)", removed, before, len(deduped))
    return deduped, removed


def assign_row_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Stable id per row: position in the deduplicated canonical file order.

    Stable because the source files are checksum-pinned and deduplication is
    deterministic.
    """
    out = df.copy()
    out.insert(0, "row_id", range(len(out)))
    return out


def stratified_split(
    df: pd.DataFrame, fractions: SplitFractions, seed: int
) -> dict[str, pd.DataFrame]:
    """Split into train / validation / reference / stream_pool, stratified by
    the label at every step."""
    labels = df[schema.LABEL]
    train, rest = train_test_split(
        df,
        train_size=fractions.train,
        stratify=labels,
        random_state=seed,
        shuffle=True,
    )
    rest_fraction = 1.0 - fractions.train
    validation, rest2 = train_test_split(
        rest,
        train_size=fractions.validation / rest_fraction,
        stratify=rest[schema.LABEL],
        random_state=seed,
        shuffle=True,
    )
    rest2_fraction = rest_fraction - fractions.validation
    reference, stream_pool = train_test_split(
        rest2,
        train_size=fractions.reference / rest2_fraction,
        stratify=rest2[schema.LABEL],
        random_state=seed,
        shuffle=True,
    )
    splits = {
        "train": train,
        "validation": validation,
        "reference": reference,
        "stream_pool": stream_pool,
    }
    return {name: part.reset_index(drop=True) for name, part in splits.items()}


def prepare(df: pd.DataFrame, fractions: SplitFractions, seed: int) -> PreparedData:
    deduped, removed = deduplicate(df)
    with_ids = assign_row_ids(deduped)
    splits = stratified_split(with_ids, fractions, seed)
    return PreparedData(splits=splits, n_rows_raw=len(df), n_duplicates_removed=removed)


def save_splits(splits: dict[str, pd.DataFrame], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, part in splits.items():
        path = out_dir / f"{name}.parquet"
        part.to_parquet(path, index=False)
        paths[name] = path
        logger.info("wrote %s (%d rows)", path, len(part))
    return paths


def save_summary(prepared: PreparedData, out_dir: Path) -> Path:
    payload = {
        "n_rows_raw": prepared.n_rows_raw,
        "n_duplicates_removed": prepared.n_duplicates_removed,
        "split_sizes": {name: len(part) for name, part in prepared.splits.items()},
    }
    path = out_dir / "summary.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_summary(processed_dir: Path) -> dict[str, Any]:
    path = processed_dir / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"data summary not found: {path}. Run 'driftmon download-data' first."
        )
    result: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return result


def load_split(processed_dir: Path, name: str) -> pd.DataFrame:
    path = processed_dir / f"{name}.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"processed split not found: {path}. Run 'driftmon download-data' first."
        )
    return pd.read_parquet(path)
