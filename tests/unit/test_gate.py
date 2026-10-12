import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from driftmon.cli import app
from driftmon.evaluation import gate
from driftmon.evaluation.report import ResultsDocument, validate_results

REPO_ROOT = Path(__file__).resolve().parents[2]

runner = CliRunner()


def smoke_document(**overrides: float) -> ResultsDocument:
    """The committed baseline's own KPIs, with chosen values replaced."""
    baseline = gate.load_baseline(REPO_ROOT / "tests" / "fixtures" / "kpi_smoke.json")
    values = {**baseline["kpis"], **overrides}
    estimate = {"ci_low": None, "ci_high": None, "n": 5}
    payload: dict[str, Any] = {
        "schema_version": 1,
        "smoke": True,
        "provenance": {"seeds": baseline["seeds"]},
        "kpis": [
            {
                "key": key,
                "name": key,
                "scope": "smoke",
                "unit": "share",
                "estimate": {"value": value, **estimate},
                "target": None,
                "comparator": "report",
                "met": None,
            }
            for key, value in values.items()
        ],
        "cells": [],
        "control": {
            "n_streams": 5,
            "stream_false_alarm_rate": {"value": 0.0, **estimate},
            "per_window_false_alarm_rate": {"value": 0.0, **estimate},
            "streams_with_alarm_by_type": {},
            "alerts_by_type": {},
            "degradation_rate": {"value": 0.0, **estimate},
        },
        "detector_correctness": {},
        "ablations": [],
        "streams": [],
    }
    return validate_results(payload)


@pytest.fixture
def baseline() -> dict[str, Any]:
    return gate.load_baseline(REPO_ROOT / "tests" / "fixtures" / "kpi_smoke.json")


def test_committed_baseline_covers_the_gated_kpis(baseline: dict[str, Any]) -> None:
    assert baseline["tolerance"] == 0.05
    assert set(baseline["kpis"]) == {
        "detection_rate.sudden_covariate",
        "detection_rate.gradual_covariate",
        "detection_rate.concept",
        "false_alarm.stream",
        "false_alarm.per_window",
    }


def test_unchanged_results_pass(baseline: dict[str, Any]) -> None:
    assert gate.check(smoke_document(), baseline) == []


def test_small_moves_within_tolerance_pass(baseline: dict[str, Any]) -> None:
    document = smoke_document(
        **{"detection_rate.concept": baseline["kpis"]["detection_rate.concept"] - 0.04}
    )

    assert gate.check(document, baseline) == []


def test_detection_drop_beyond_tolerance_fails(baseline: dict[str, Any]) -> None:
    document = smoke_document(**{"detection_rate.sudden_covariate": 0.8})

    violations = gate.check(document, baseline)

    assert len(violations) == 1
    assert "detection_rate.sudden_covariate" in violations[0]


def test_false_alarm_rise_beyond_tolerance_fails(baseline: dict[str, Any]) -> None:
    document = smoke_document(
        **{"false_alarm.stream": baseline["kpis"]["false_alarm.stream"] + 0.2}
    )

    assert any("false_alarm.stream" in v for v in gate.check(document, baseline))


def test_better_results_never_fail(baseline: dict[str, Any]) -> None:
    lower_alarms = {k: 0.0 for k in baseline["kpis"] if k.startswith("false_alarm")}

    assert gate.check(smoke_document(**lower_alarms), baseline) == []


def test_different_seeds_ask_for_a_new_baseline(baseline: dict[str, Any]) -> None:
    changed = {**baseline, "seeds": [1, 2, 3]}

    violations = gate.check(smoke_document(), changed)

    assert violations and "regenerate" in violations[0]


def test_baseline_only_from_smoke_runs() -> None:
    full = smoke_document().model_copy(update={"smoke": False})

    with pytest.raises(ValueError, match="smoke"):
        gate.baseline_from(full)


def test_cli_update_then_check_round_trip(tmp_path: Path) -> None:
    results = tmp_path / "results.json"
    results.write_text(json.dumps(smoke_document().model_dump(mode="json")), encoding="utf-8")
    baseline_path = tmp_path / "baseline.json"

    written = runner.invoke(
        app, ["kpi-gate", "--results", str(results), "--baseline", str(baseline_path), "--update"]
    )
    checked = runner.invoke(
        app, ["kpi-gate", "--results", str(results), "--baseline", str(baseline_path)]
    )

    assert written.exit_code == 0, written.output
    assert checked.exit_code == 0
    assert "KPI gate passed" in checked.output


def test_cli_fails_on_regression(tmp_path: Path) -> None:
    results = tmp_path / "results.json"
    regressed = smoke_document(**{"detection_rate.gradual_covariate": 0.5})
    results.write_text(json.dumps(regressed.model_dump(mode="json")), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "kpi-gate",
            "--results",
            str(results),
            "--baseline",
            str(REPO_ROOT / "tests" / "fixtures" / "kpi_smoke.json"),
        ],
    )

    assert result.exit_code == 1


def test_cli_without_results_explains_what_to_run(tmp_path: Path) -> None:
    result = runner.invoke(app, ["kpi-gate", "--results", str(tmp_path / "missing.json")])

    assert result.exit_code != 0
    assert "evaluate-smoke" in result.output


def test_dashboard_command_launches_the_packaged_app(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr("os.execv", lambda path, args: calls.append(args))

    result = runner.invoke(app, ["dashboard", "--port", "8600", "--host", "0.0.0.0"])

    assert result.exit_code == 0, result.output
    (args,) = calls
    assert args[1:4] == ["-m", "streamlit", "run"]
    assert Path(args[4]).name == "app.py" and Path(args[4]).is_file()
    assert "--server.port=8600" in args
    assert "--server.address=0.0.0.0" in args
