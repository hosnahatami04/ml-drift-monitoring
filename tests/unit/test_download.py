import urllib.error
from pathlib import Path

import pytest

from driftmon.config import Settings
from driftmon.data import schema
from driftmon.data.download import (
    DownloadError,
    ensure_file,
    fetch,
    frame_fingerprint,
    load_raw_dataset,
    normalize_openml,
    sha256_of,
    verify_checksum,
)

from .conftest import make_adult_frame

CONTENT = b"known content\n"
CONTENT_SHA256 = "a5e29604a88ef9dace3ea3de21aa0cfb09946846146070b9a9fe17f2f9701212"


def test_sha256_of_known_content(tmp_path: Path) -> None:
    path = tmp_path / "file.txt"
    path.write_bytes(CONTENT)

    assert sha256_of(path) == CONTENT_SHA256


def test_checksum_mismatch_raises_with_both_hashes(tmp_path: Path) -> None:
    path = tmp_path / "file.txt"
    path.write_bytes(b"tampered")
    expected = "0" * 64

    with pytest.raises(DownloadError, match="checksum mismatch") as excinfo:
        verify_checksum(path, expected)

    assert expected in str(excinfo.value)
    assert sha256_of(path) in str(excinfo.value)


def test_ensure_file_uses_valid_cache_without_network(tmp_path: Path) -> None:
    dest = tmp_path / "cached.txt"
    dest.write_bytes(CONTENT)

    result = ensure_file("http://unreachable.invalid/x", dest, CONTENT_SHA256)

    assert result == dest


def test_ensure_file_redownloads_corrupt_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = tmp_path / "cached.txt"
    dest.write_bytes(b"corrupt")

    def fake_fetch(url: str, path: Path, *, retries: int = 3) -> None:
        path.write_bytes(CONTENT)

    monkeypatch.setattr("driftmon.data.download.fetch", fake_fetch)

    ensure_file("http://example.invalid/x", dest, CONTENT_SHA256)

    assert dest.read_bytes() == CONTENT


def test_fetch_retries_then_succeeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dest = tmp_path / "out.bin"
    attempts: list[int] = []

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return CONTENT

    def fake_urlopen(url: str, timeout: float = 0) -> FakeResponse:
        attempts.append(1)
        if len(attempts) < 3:
            raise urllib.error.URLError("boom")
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda _: None)

    fetch("http://example.invalid/x", dest, retries=3)

    assert len(attempts) == 3
    assert dest.read_bytes() == CONTENT


def test_fetch_gives_up_after_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(url: str, timeout: float = 0) -> None:
        raise urllib.error.URLError("down")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda _: None)

    with pytest.raises(DownloadError, match="after 2 attempts"):
        fetch("http://example.invalid/x", tmp_path / "out.bin", retries=2)


def test_frame_fingerprint_is_deterministic_and_sensitive() -> None:
    frame = make_adult_frame(50, seed=9)

    assert frame_fingerprint(frame) == frame_fingerprint(frame.copy())

    changed = frame.copy()
    changed.loc[0, "age"] = changed.loc[0, "age"] + 1
    assert frame_fingerprint(changed) != frame_fingerprint(frame)


def test_normalize_openml_maps_onto_uci_schema() -> None:
    frame = make_adult_frame(80, seed=4)
    openml_style = frame.rename(
        columns={"income": "class", **{c: c.replace("-", "_") for c in frame.columns}}
    )
    openml_style["workclass"] = openml_style["workclass"].astype("category")
    openml_style.loc[0, "occupation"] = "?"

    normalized = normalize_openml(openml_style)

    assert list(normalized.columns) == list(schema.COLUMNS)
    assert normalized["age"].dtype == "int64"
    assert normalized["occupation"].isna().iloc[0]


def test_load_raw_dataset_requires_pinned_openml_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings()
    assert settings.dataset.openml.sha256 is None

    def no_uci(_: Settings) -> tuple[Path, Path]:
        raise DownloadError("uci unavailable")

    class FakeBundle:
        frame = make_adult_frame(60, seed=8)

    monkeypatch.setattr("driftmon.data.download.download_uci", no_uci)
    monkeypatch.setattr("sklearn.datasets.fetch_openml", lambda **kwargs: FakeBundle())

    with pytest.raises(DownloadError, match="no sha256 is pinned"):
        load_raw_dataset(settings)


def test_load_raw_dataset_uses_openml_when_hash_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = make_adult_frame(60, seed=8)
    expected = frame_fingerprint(frame)
    settings = Settings(dataset={"openml": {"sha256": expected}})

    def no_uci(_: Settings) -> tuple[Path, Path]:
        raise DownloadError("uci unavailable")

    class FakeBundle:
        pass

    FakeBundle.frame = frame  # type: ignore[attr-defined]

    monkeypatch.setattr("driftmon.data.download.download_uci", no_uci)
    monkeypatch.setattr("sklearn.datasets.fetch_openml", lambda **kwargs: FakeBundle())

    result = load_raw_dataset(settings)

    assert len(result) == 60
    assert list(result.columns) == list(schema.COLUMNS)
