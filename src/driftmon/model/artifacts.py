"""Persistence of the trained model together with its reference statistics."""

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from sklearn.pipeline import Pipeline

from driftmon.data import schema
from driftmon.model.reference import ReferenceStats

logger = logging.getLogger(__name__)

PINNED_LIBRARIES = ("pandas", "scikit-learn", "xgboost", "numpy")


@dataclass(frozen=True)
class ModelArtifact:
    model: Pipeline
    baseline_model: Pipeline
    reference: ReferenceStats
    metadata: dict[str, Any]


def library_versions() -> dict[str, str]:
    return {name: version(name) for name in PINNED_LIBRARIES}


def build_metadata(
    data_hashes: dict[str, str],
    seed: int,
    metrics: dict[str, dict[str, dict[str, float]]],
    n_duplicates_removed: int,
    split_sizes: dict[str, int],
) -> dict[str, Any]:
    return {
        "data_sha256": data_hashes,
        "seed": seed,
        "features": {
            "numeric": list(schema.NUMERIC_FEATURES),
            "categorical": list(schema.CATEGORICAL_FEATURES),
        },
        "library_versions": library_versions(),
        "metrics": metrics,
        "n_duplicates_removed": n_duplicates_removed,
        "split_sizes": split_sizes,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def fingerprint(metadata: dict[str, Any]) -> str:
    """Short id for the artifact directory.

    Derived only from inputs that determine the trained model (data, seed,
    features, library versions) so re-running training yields the same id;
    the creation timestamp is deliberately excluded.
    """
    core = {
        "data_sha256": metadata["data_sha256"],
        "seed": metadata["seed"],
        "features": metadata["features"],
        "library_versions": metadata["library_versions"],
    }
    payload = json.dumps(core, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def save_artifact(artifact: ModelArtifact, artifacts_dir: Path) -> Path:
    out_dir = artifacts_dir / f"model-{fingerprint(artifact.metadata)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "model.pkl").open("wb") as fh:
        pickle.dump(artifact.model, fh)
    with (out_dir / "baseline_model.pkl").open("wb") as fh:
        pickle.dump(artifact.baseline_model, fh)
    (out_dir / "reference.json").write_text(
        json.dumps(artifact.reference.to_dict()), encoding="utf-8"
    )
    (out_dir / "metadata.json").write_text(
        json.dumps(artifact.metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.info("saved artifact to %s", out_dir)
    return out_dir


def load_artifact(artifact_dir: Path) -> ModelArtifact:
    if not (artifact_dir / "metadata.json").is_file():
        raise FileNotFoundError(f"not a model artifact directory: {artifact_dir}")
    with (artifact_dir / "model.pkl").open("rb") as fh:
        model = pickle.load(fh)
    with (artifact_dir / "baseline_model.pkl").open("rb") as fh:
        baseline = pickle.load(fh)
    reference = ReferenceStats.from_dict(
        json.loads((artifact_dir / "reference.json").read_text(encoding="utf-8"))
    )
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    return ModelArtifact(
        model=model, baseline_model=baseline, reference=reference, metadata=metadata
    )


def latest_artifact_dir(artifacts_dir: Path) -> Path:
    candidates = sorted(
        (p for p in artifacts_dir.glob("model-*") if (p / "metadata.json").is_file()),
        key=lambda p: (p / "metadata.json").stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"no model artifact found under {artifacts_dir}. Run 'driftmon train' first."
        )
    return candidates[-1]
