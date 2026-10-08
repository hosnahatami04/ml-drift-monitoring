"""Checksum-verified download of the UCI Adult dataset with an OpenML fallback."""

import hashlib
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from driftmon.config import Settings
from driftmon.data import schema

logger = logging.getLogger(__name__)

RAW_FILES = ("adult.data", "adult.test")


class DownloadError(RuntimeError):
    """Raised when a dataset file cannot be fetched or fails verification."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path: Path, expected: str) -> None:
    actual = sha256_of(path)
    if actual != expected:
        raise DownloadError(
            f"checksum mismatch for {path}: expected {expected}, got {actual}. "
            "The source file has changed or the download is corrupt; refusing to use it."
        )


def fetch(url: str, dest: Path, *, retries: int = 3, timeout: float = 60.0) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            logger.info("downloading %s (attempt %d/%d)", url, attempt, retries)
            with urllib.request.urlopen(url, timeout=timeout) as response:
                tmp.write_bytes(response.read())
            tmp.replace(dest)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            logger.warning("download failed: %s", exc)
            if attempt < retries:
                time.sleep(2.0 * attempt)
    raise DownloadError(f"failed to download {url} after {retries} attempts: {last_error}")


def ensure_file(url: str, dest: Path, expected_sha256: str, *, retries: int = 3) -> Path:
    """Return a verified local copy, downloading only when the cache is absent
    or does not match the pinned checksum."""
    if dest.is_file():
        try:
            verify_checksum(dest, expected_sha256)
            logger.info("using cached %s", dest)
            return dest
        except DownloadError:
            logger.warning("cached %s fails verification, re-downloading", dest)
    fetch(url, dest, retries=retries)
    verify_checksum(dest, expected_sha256)
    return dest


def download_uci(settings: Settings) -> tuple[Path, Path]:
    uci = settings.dataset.uci
    raw_dir = settings.raw_data_dir
    data_path = ensure_file(uci.data_url, raw_dir / "adult.data", uci.data_sha256)
    test_path = ensure_file(uci.test_url, raw_dir / "adult.test", uci.test_sha256)
    return data_path, test_path


def frame_fingerprint(df: pd.DataFrame) -> str:
    """Deterministic hash of a frame, used to pin the OpenML fallback."""
    canonical = df[list(schema.COLUMNS)].astype("string").fillna("?")
    payload = canonical.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_openml(df: pd.DataFrame) -> pd.DataFrame:
    """Map the OpenML adult frame (name=adult, version=2) onto the UCI schema."""
    renamed = df.rename(columns={"class": schema.LABEL})
    renamed.columns = [str(c).strip().replace("_", "-") for c in renamed.columns]
    for column in renamed.columns:
        if not pd.api.types.is_numeric_dtype(renamed[column]):
            renamed[column] = renamed[column].astype("string").str.strip()
            renamed[column] = renamed[column].replace("?", pd.NA)
    for column in schema.NUMERIC_RANGES:
        renamed[column] = renamed[column].astype("int64")
    for column in (*schema.CATEGORIES, schema.LABEL):
        renamed[column] = renamed[column].astype(object)
    return renamed[list(schema.COLUMNS)]


def fetch_openml_fallback(settings: Settings) -> pd.DataFrame:
    from sklearn.datasets import fetch_openml

    source = settings.dataset.openml
    logger.info("falling back to OpenML (name=%s, version=%d)", source.name, source.version)
    bundle = fetch_openml(name=source.name, version=source.version, as_frame=True, parser="auto")
    df = normalize_openml(bundle.frame)
    fingerprint = frame_fingerprint(df)
    if source.sha256 is None:
        raise DownloadError(
            "OpenML fallback fetched but no sha256 is pinned for it in the config. "
            f"Computed fingerprint: {fingerprint}. Pin it under dataset.openml.sha256 "
            "after verifying the data, then retry."
        )
    if fingerprint != source.sha256:
        raise DownloadError(
            f"OpenML fallback fingerprint mismatch: expected {source.sha256}, got {fingerprint}"
        )
    return df


def load_raw_dataset(settings: Settings) -> pd.DataFrame:
    """Download (or reuse) the raw dataset and return one validated frame."""
    try:
        data_path, test_path = download_uci(settings)
    except DownloadError as exc:
        logger.warning("UCI source unavailable: %s", exc)
        df = fetch_openml_fallback(settings)
    else:
        parts = [schema.read_raw_file(data_path), schema.read_raw_file(test_path)]
        df = pd.concat(parts, ignore_index=True)
    schema.validate(df)
    return df
