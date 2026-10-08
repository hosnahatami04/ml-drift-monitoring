from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from driftmon.cli import app

from .conftest import make_adult_frame

runner = CliRunner()


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    config = {
        "data_dir": str(tmp_path / "data"),
        "artifacts_dir": str(tmp_path / "artifacts"),
        "reports_dir": str(tmp_path / "reports"),
        "scenarios_dir": str(repo_root / "configs" / "scenarios"),
        "seed": 42,
        "window_size": 150,
        "stream": {"n_windows": 8, "drift_start": 3, "detection_horizon": 4},
        "seeds": {
            "calibration_start": 0,
            "calibration_count": 2,
            "evaluation_start": 100,
            "evaluation_count": 2,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    frame = make_adult_frame(2500, seed=11)
    monkeypatch.setattr("driftmon.data.download.load_raw_dataset", lambda settings: frame)
    return config_path


def test_download_data_writes_splits_and_summary(workspace: Path) -> None:
    result = runner.invoke(app, ["download-data", "--config", str(workspace)])

    assert result.exit_code == 0, result.output
    data_dir = workspace.parent / "data" / "processed"
    for name in ("train", "validation", "reference", "stream_pool"):
        assert (data_dir / f"{name}.parquet").is_file()
    assert (data_dir / "summary.json").is_file()
    assert "duplicates removed" in result.output


def test_train_writes_artifact_and_report(workspace: Path) -> None:
    prepared = runner.invoke(app, ["download-data", "--config", str(workspace)])
    assert prepared.exit_code == 0, prepared.output

    result = runner.invoke(app, ["train", "--config", str(workspace)])

    assert result.exit_code == 0, result.output
    artifacts_dir = workspace.parent / "artifacts"
    artifact_dirs = list(artifacts_dir.glob("model-*"))
    assert len(artifact_dirs) == 1
    for name in ("model.pkl", "baseline_model.pkl", "reference.json", "metadata.json"):
        assert (artifact_dirs[0] / name).is_file()
    report = workspace.parent / "reports" / "baseline.md"
    assert report.is_file()
    text = report.read_text(encoding="utf-8")
    assert "| xgboost | validation |" in text


def test_train_without_data_gives_clear_error(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"data_dir": str(tmp_path / "data")}), encoding="utf-8")

    result = runner.invoke(app, ["train", "--config", str(config_path)])

    assert result.exit_code != 0
    assert isinstance(result.exception, FileNotFoundError)
    assert "download-data" in str(result.exception)


def test_simulate_writes_stream_and_ground_truth(workspace: Path) -> None:
    prepared = runner.invoke(app, ["download-data", "--config", str(workspace)])
    assert prepared.exit_code == 0, prepared.output

    result = runner.invoke(
        app,
        [
            "simulate",
            "--scenario",
            "sudden_covariate",
            "--severity",
            "medium",
            "--seed",
            "0",
            "--config",
            str(workspace),
        ],
    )

    assert result.exit_code == 0, result.output
    simulated = workspace.parent / "data" / "simulated"
    stream_path = simulated / "sudden_covariate-medium-seed0.parquet"
    truth_path = simulated / "sudden_covariate-medium-seed0.ground_truth.json"
    assert stream_path.is_file()
    assert truth_path.is_file()
    frame = pd.read_parquet(stream_path)
    assert frame["window_index"].nunique() == 8
    assert "drift from window 3" in result.output


def test_calibrate_severity_writes_report(workspace: Path) -> None:
    for command in (["download-data"], ["train"]):
        result = runner.invoke(app, [*command, "--config", str(workspace)])
        assert result.exit_code == 0, result.output

    result = runner.invoke(app, ["calibrate-severity", "--config", str(workspace)])

    assert result.exit_code == 0, result.output
    report = workspace.parent / "reports" / "severity.md"
    assert report.is_file()
    text = report.read_text(encoding="utf-8")
    assert "| sudden_covariate | medium |" in text
    assert "| concept | high |" in text


def test_download_data_rejects_corrupt_frame(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = make_adult_frame(500, seed=5)
    frame.loc[0, "age"] = 500

    def bad_dataset(settings: object) -> pd.DataFrame:
        from driftmon.data import schema

        schema.validate(frame)
        return frame

    monkeypatch.setattr("driftmon.data.download.load_raw_dataset", bad_dataset)

    result = runner.invoke(app, ["download-data", "--config", str(workspace)])

    assert result.exit_code != 0
