import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from driftmon.config import ThresholdsConfig
from driftmon.evaluation import report
from driftmon.evaluation.harness import AblationArm, EvaluationRun
from driftmon.evaluation.kpis import AlertEvent, StreamOutcome
from driftmon.simulation.scenarios import StreamParams

PARAMS = StreamParams(n_windows=60, window_size=1000, drift_start=20, label_delay=3)


def outcome(
    scenario: str,
    severity: str | None,
    seed: int,
    alerts: list[tuple[int, str]],
    degradation: int | None = None,
) -> StreamOutcome:
    return StreamOutcome(
        scenario=scenario,
        kind=scenario,
        severity=severity,
        seed=seed,
        t0=None if severity is None else 20,
        horizon=20,
        n_windows=60,
        alerts=tuple(AlertEvent(w, t) for w, t in alerts),
        degradation_window=degradation,
    )


def provenance() -> dict[str, object]:
    return {
        "artifact": "model-test",
        "model_sha256": "0" * 64,
        "reference_sha256": "1" * 64,
        "reference_auc": 0.92,
        "config_sha256": "2" * 64,
        "thresholds": ThresholdsConfig().model_dump(by_alias=True, mode="json"),
        "seeds": [100, 101],
        "calibration_seeds": [0, 1],
        "severities": ["medium"],
        "stream": {"n_windows": 60, "window_size": 1000, "drift_start": 20, "label_delay": 3},
        "horizon": 20,
        "degradation_margin": 0.03,
        "bootstrap": {"samples": 50, "seed": 1},
        "library_versions": {},
    }


@pytest.fixture
def run() -> EvaluationRun:
    outcomes = [
        outcome("control", None, 100, []),
        outcome("control", None, 101, [(30, "performance_drift"), (34, "performance_drift")]),
        outcome("sudden_covariate", "medium", 100, [(20, "feature_drift")], degradation=25),
        outcome("sudden_covariate", "medium", 101, [(21, "feature_drift")]),
        outcome("gradual_covariate", "medium", 100, [(26, "feature_drift")]),
        outcome("gradual_covariate", "medium", 101, []),
        outcome("concept", "medium", 100, [(25, "performance_drift")], degradation=22),
        outcome("concept", "medium", 101, [(30, "performance_drift")]),
    ]
    ablation = AblationArm("window_size", "n=1000", "all alerts", 1000, outcomes)
    return EvaluationRun(
        outcomes=outcomes,
        ablations=[ablation],
        thresholds=ThresholdsConfig(),
        seeds=[100, 101],
        severities=["medium"],
        params=PARAMS,
        horizon=20,
    )


def build(run: EvaluationRun) -> report.ResultsDocument:
    return report.build_results(
        run,
        smoke=False,
        target_severity="medium",
        provenance=provenance(),
        correctness={"ks_statistic": 0.0, "chi2_p_value": 2e-7},
        samples=50,
        seed=1,
    )


def test_kpi_rows_carry_values_and_verdicts(run: EvaluationRun) -> None:
    document = build(run)
    rows = {r.key: r for r in document.kpis}

    assert rows["detection_rate.sudden_covariate"].estimate.value == 1.0
    assert rows["detection_rate.gradual_covariate"].estimate.value == 0.5
    assert rows["detection_rate.gradual_covariate"].met is False
    assert rows["median_ttd.concept"].estimate.value == 7.5
    assert rows["false_alarm.stream"].estimate.value == 0.5
    assert rows["false_alarm.per_window"].estimate.value == pytest.approx(2 / 120)
    assert rows["detector_correctness"].estimate.value == 2e-7
    assert rows["detector_correctness"].met is True
    assert rows["harmful_share"].met is None
    assert document.control.streams_with_alarm_by_type["performance_drift"] == 1


def test_lead_time_headline_uses_covariate_streams_only(run: EvaluationRun) -> None:
    rows = {r.key: r for r in build(run).kpis}

    lead = rows["lead_time.harmful_covariate"].estimate
    assert lead.n == 1
    assert lead.value == 5.0


def test_results_json_validates_and_is_byte_stable(run: EvaluationRun, tmp_path: Path) -> None:
    first, second = tmp_path / "a.json", tmp_path / "b.json"

    report.write_json(first, build(run))
    report.write_json(second, build(run))

    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(first.read_text(encoding="utf-8"))
    document = report.validate_results(payload)
    assert document.schema_version == 1
    assert len(document.streams) == len(run.outcomes)


def test_schema_rejects_unknown_and_missing_fields(run: EvaluationRun) -> None:
    payload = build(run).model_dump(mode="json")

    with pytest.raises(ValidationError):
        report.validate_results({**payload, "unexpected": 1})
    incomplete = dict(payload)
    del incomplete["kpis"]
    with pytest.raises(ValidationError):
        report.validate_results(incomplete)
    wrong_version = {**payload, "schema_version": 2}
    with pytest.raises(ValidationError):
        report.validate_results(wrong_version)


def test_markdown_lists_misses_under_what_did_not_work(run: EvaluationRun) -> None:
    text = report.render_markdown(build(run))

    section = text.split("## What didn't work", 1)[1]
    assert "Detection rate, gradual covariate" in section
    assert "Undetected gradual_covariate streams at medium severity: seeds [101]" in section
    assert "| sudden_covariate | medium |" in text
    assert "### window size (all alerts)" in text


def test_figures_are_written_from_the_document(run: EvaluationRun, tmp_path: Path) -> None:
    written = report.write_figures(tmp_path, build(run))

    assert {p.name for p in written} == {
        "detection_vs_severity.png",
        "ttd_by_scenario.png",
        "window_size_ablation.png",
        "lead_time.png",
    }
    assert all(p.stat().st_size > 0 for p in written)
