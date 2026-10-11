import json
import shutil
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from driftmon.cli import app
from driftmon.config import SeedsConfig
from driftmon.evaluation import harness, report

from .conftest import make_adult_frame

runner = CliRunner()

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("evaluation")
    thresholds = root / "thresholds.yaml"
    shutil.copy(REPO_ROOT / "configs" / "thresholds.yaml", thresholds)
    config = {
        "data_dir": str(root / "data"),
        "artifacts_dir": str(root / "artifacts"),
        "reports_dir": str(root / "reports"),
        "scenarios_dir": str(REPO_ROOT / "configs" / "scenarios"),
        "thresholds_path": str(thresholds),
        "window_size": 150,
        "label_delay": 1,
        "stream": {"n_windows": 8, "drift_start": 3, "detection_horizon": 4},
        "seeds": {
            "calibration_start": 0,
            "calibration_count": 2,
            "evaluation_start": 100,
            "evaluation_count": 2,
        },
        "evaluation": {
            "n_workers": 1,
            "bootstrap_samples": 50,
            "correctness_cases": 10,
            "ablation_window_sizes": [150, 300],
            "smoke_seed_count": 1,
            "calibration": {
                "alpha": [0.001, 0.01],
                "ks_d": [0.05],
                "psi": [0.1],
                "cramers_v": [0.1],
                "k_features": [1],
                "ph_delta": [0.005],
                "ph_lambda": [0.05],
            },
        },
    }
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    frame = make_adult_frame(2500, seed=11)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("driftmon.data.download.load_raw_dataset", lambda settings: frame)
        for command in (["download-data"], ["train"]):
            result = runner.invoke(app, [*command, "--config", str(config_path)])
            assert result.exit_code == 0, result.output
    return config_path


def test_calibrate_only_ever_simulates_calibration_seeds(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[int] = []
    real = harness.run_streams

    def spy(tasks, inputs, n_workers):  # type: ignore[no-untyped-def]
        seen.extend(t.seed for t in tasks)
        return real(tasks, inputs, n_workers)

    monkeypatch.setattr(harness, "run_streams", spy)

    result = runner.invoke(app, ["calibrate", "--config", str(workspace)])

    assert result.exit_code == 0, result.output
    assert seen and set(seen) == {0, 1}
    assert not set(seen) & set(SeedsConfig(evaluation_start=100, evaluation_count=2).evaluation)
    frozen = yaml.safe_load((workspace.parent / "thresholds.yaml").read_text(encoding="utf-8"))
    assert frozen["alpha"] in (0.001, 0.01)
    assert (workspace.parent / "artifacts" / "calibration.json").is_file()


def test_evaluate_writes_validated_reports_and_figures(workspace: Path) -> None:
    out = workspace.parent / "run1"

    result = runner.invoke(app, ["evaluate", "--out-dir", str(out), "--config", str(workspace)])

    assert result.exit_code == 0, result.output
    document = report.validate_results(
        json.loads((out / "results.json").read_text(encoding="utf-8"))
    )
    assert document.provenance["seeds"] == [100, 101]
    assert {a.arm for a in document.ablations if a.group == "window_size"} == {"n=150", "n=300"}
    assert (out / "results.md").is_file()
    assert len(list((out / "figures").glob("*.png"))) == 4


def test_evaluate_is_reproducible_byte_for_byte(workspace: Path) -> None:
    outputs = [workspace.parent / "repro-a", workspace.parent / "repro-b"]
    for out in outputs:
        result = runner.invoke(app, ["evaluate", "--out-dir", str(out), "--config", str(workspace)])
        assert result.exit_code == 0, result.output

    first, second = (o / "results.json" for o in outputs)
    assert first.read_bytes() == second.read_bytes()


def test_smoke_evaluation_skips_ablations_and_defaults_outside_reports(workspace: Path) -> None:
    result = runner.invoke(app, ["evaluate", "--smoke", "--config", str(workspace)])

    assert result.exit_code == 0, result.output
    smoke = workspace.parent / "artifacts" / "smoke" / "results.json"
    document = report.validate_results(json.loads(smoke.read_text(encoding="utf-8")))
    assert document.smoke
    assert document.ablations == []
    assert document.provenance["seeds"] == [100]
    assert not (workspace.parent / "reports" / "results.json").exists()
