import threading
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from driftmon.cli import app
from driftmon.model.artifacts import ModelArtifact, save_artifact
from driftmon.monitoring.runner import MonitorRunner, poll
from driftmon.service.api import build_state, create_app
from driftmon.service.client import InProcessClient, replay_windows
from driftmon.simulation import sampler
from driftmon.simulation.scenarios import ScenarioConfig, StreamParams, generate_stream

from .conftest import make_adult_frame
from .test_observability import REPO_ROOT, settings_for

PARAMS = StreamParams(n_windows=6, window_size=200, drift_start=2, label_delay=1)

SUDDEN = ScenarioConfig(
    name="sudden_covariate",
    kind="sudden_covariate",
    feature="age",
    severity={"high": {"beta": 1.5}},
)


def stream() -> list:  # type: ignore[type-arg]
    pool = make_adult_frame(3000, seed=90)
    stats = sampler.reference_feature_stats(make_adult_frame(1500, seed=91), ("age",))
    return list(generate_stream(pool, stats, SUDDEN, "high", 100, PARAMS))


def test_api_without_inline_monitoring_only_writes(
    tiny_artifact: ModelArtifact, tmp_path: Path
) -> None:
    api = build_state(settings_for(tmp_path, inline=False), artifact=tiny_artifact)

    replay_windows(InProcessClient(TestClient(create_app(api))), stream())

    assert api.store.prediction_count() == PARAMS.n_windows * PARAMS.window_size
    assert api.store.processed_window_count() == 0


def test_worker_processes_what_the_api_wrote(tiny_artifact: ModelArtifact, tmp_path: Path) -> None:
    settings = settings_for(tmp_path, inline=False)
    api = build_state(settings, artifact=tiny_artifact)
    replay_windows(InProcessClient(TestClient(create_app(api))), stream())
    worker = build_state(settings, artifact=tiny_artifact, role="monitor")

    reports, _ = worker.runner.run_once()

    assert [r.window_id for r in reports] == list(range(PARAMS.n_windows))
    assert worker.store.processed_window_count() == PARAMS.n_windows
    # Labels of every window have arrived, all but the last L in replay order.
    assert worker.store.checked_error_rates()
    assert worker.runner.run_once() == ([], [])


def test_restored_worker_matches_one_that_never_stopped(
    tiny_artifact: ModelArtifact, tmp_path: Path
) -> None:
    settings = settings_for(tmp_path, inline=True)
    live = build_state(settings, artifact=tiny_artifact)
    replay_windows(InProcessClient(TestClient(create_app(live))), stream())

    restarted = build_state(settings, artifact=tiny_artifact, role="monitor")
    restarted.runner.restore_state()

    assert restarted.runner.monitor.page_hinkley.score == pytest.approx(
        live.runner.monitor.page_hinkley.score
    )
    assert restarted.runner.alert_manager._last_fired == live.runner.alert_manager._last_fired


def test_monitor_command_restores_state_serves_metrics_and_stops(
    tiny_artifact: ModelArtifact, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = settings_for(tmp_path, inline=False)
    api = build_state(settings, artifact=tiny_artifact)
    replay_windows(InProcessClient(TestClient(create_app(api))), stream())
    save_artifact(tiny_artifact, tmp_path / "artifacts")
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "artifacts_dir": str(tmp_path / "artifacts"),
                "db_path": str(settings.db_path),
                "window_size": PARAMS.window_size,
                "thresholds_path": str(REPO_ROOT / "configs" / "thresholds.yaml"),
                "service": {"inline_monitoring": False},
                "telemetry": {"trace_exporter": "none"},
            }
        ),
        encoding="utf-8",
    )
    served: list[int] = []
    monkeypatch.setattr(
        "prometheus_client.start_http_server", lambda port, registry: served.append(port)
    )
    # The real setup installs a process-wide provider that would leak into
    # every later test (FastAPI emits its own spans through it).
    monkeypatch.setattr("driftmon.observability.tracing.setup_tracing", lambda config, name: None)
    # One pass of the loop instead of polling forever.
    monkeypatch.setattr(
        "driftmon.monitoring.runner.poll", lambda runner, interval, stop: runner.run_once()
    )

    result = CliRunner().invoke(app, ["monitor", "--metrics-port", "9911", "--config", str(config)])

    assert result.exit_code == 0, result.output
    assert served == [9911]
    assert api.store.processed_window_count() == PARAMS.n_windows


def test_poll_runs_until_stopped(tiny_artifact: ModelArtifact, tmp_path: Path) -> None:
    worker = build_state(
        settings_for(tmp_path, inline=False), artifact=tiny_artifact, role="monitor"
    )
    stop = threading.Event()
    passes: list[int] = []
    real_run_once = worker.runner.run_once

    def counting_run_once():  # type: ignore[no-untyped-def]
        passes.append(1)
        if len(passes) == 3:
            stop.set()
        return real_run_once()

    worker.runner.run_once = counting_run_once  # type: ignore[method-assign]

    poll(worker.runner, 0.01, stop)

    assert len(passes) == 3


def test_a_failing_window_is_logged_and_retried_without_overtaking(
    tiny_artifact: ModelArtifact, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    settings = settings_for(tmp_path, inline=False)
    api = build_state(settings, artifact=tiny_artifact)
    replay_windows(InProcessClient(TestClient(create_app(api))), stream())
    worker = build_state(settings, artifact=tiny_artifact, role="monitor")
    real = worker.runner.monitor.process_window
    calls: list[int] = []

    def fail_first(window_id, features, probabilities):  # type: ignore[no-untyped-def]
        calls.append(window_id)
        if len(calls) == 1:
            raise RuntimeError("transient failure")
        return real(window_id, features, probabilities)

    worker.runner.monitor.process_window = fail_first  # type: ignore[method-assign]

    with caplog.at_level("ERROR"):
        first, _ = worker.runner.run_once()
    second, _ = worker.runner.run_once()

    assert first == []
    assert "processing window 0 failed" in caplog.text
    assert [r.window_id for r in second] == list(range(PARAMS.n_windows))


def test_poll_survives_a_failing_tick(tiny_artifact: ModelArtifact, tmp_path: Path) -> None:
    worker = build_state(
        settings_for(tmp_path, inline=False), artifact=tiny_artifact, role="monitor"
    )
    stop = threading.Event()
    ticks: list[int] = []

    def flaky_run_once():  # type: ignore[no-untyped-def]
        ticks.append(1)
        if len(ticks) == 1:
            raise RuntimeError("store locked")
        stop.set()
        return [], []

    worker.runner.run_once = flaky_run_once  # type: ignore[method-assign]

    poll(worker.runner, 0.01, stop)

    assert len(ticks) == 2


def test_inline_service_restores_monitor_state_on_start(
    tiny_artifact: ModelArtifact, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = settings_for(tmp_path, inline=True)
    live = build_state(settings, artifact=tiny_artifact)
    replay_windows(InProcessClient(TestClient(create_app(live))), stream())
    save_artifact(tiny_artifact, tmp_path / "artifacts")
    config = tmp_path / "serve.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "artifacts_dir": str(tmp_path / "artifacts"),
                "db_path": str(settings.db_path),
                "window_size": PARAMS.window_size,
                "thresholds_path": str(REPO_ROOT / "configs" / "thresholds.yaml"),
            }
        ),
        encoding="utf-8",
    )
    started: list[object] = []
    monkeypatch.setattr("uvicorn.run", lambda app, host, port: started.append(app))
    monkeypatch.setattr("driftmon.observability.tracing.setup_tracing", lambda config, name: None)
    restored: list[float] = []
    real_restore = MonitorRunner.restore_state

    def spy(self):  # type: ignore[no-untyped-def]
        real_restore(self)
        restored.append(self.monitor.page_hinkley.score)

    monkeypatch.setattr(MonitorRunner, "restore_state", spy)

    result = CliRunner().invoke(app, ["serve", "--config", str(config)])

    assert result.exit_code == 0, result.output
    assert started
    assert restored == [pytest.approx(live.runner.monitor.page_hinkley.score)]
