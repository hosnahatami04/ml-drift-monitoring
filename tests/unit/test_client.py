import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import uvicorn
import yaml
from typer.testing import CliRunner

from driftmon.cli import app
from driftmon.data import split
from driftmon.model.artifacts import ModelArtifact, save_artifact
from driftmon.service import client as api_client
from driftmon.service.api import build_state, create_app
from driftmon.service.client import ApiError, HttpClient, replay_windows, wait_for_monitor

from .conftest import make_adult_frame
from .test_monitor_worker import PARAMS, stream
from .test_observability import REPO_ROOT, payloads, settings_for


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def server(tiny_artifact: ModelArtifact, tmp_path: Path) -> Iterator[str]:
    state = build_state(settings_for(tmp_path, inline=True), artifact=tiny_artifact)
    port = _free_port()
    config = uvicorn.Config(create_app(state), host="127.0.0.1", port=port, log_level="warning")
    instance = uvicorn.Server(config)
    thread = threading.Thread(target=instance.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not instance.started:
        if time.monotonic() > deadline:
            raise RuntimeError("test server did not start")
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    instance.should_exit = True
    thread.join(timeout=10)


def test_http_client_round_trip(server: str) -> None:
    client = HttpClient(server)

    request_id = client.predict(payloads(1)[0])
    client.label(request_id, 1)

    assert len(request_id) == 32
    assert client.health()["predictions"] == 1
    assert client.alerts(limit=10) == []
    client.close()


def test_http_client_surfaces_api_errors(server: str) -> None:
    client = HttpClient(server)

    with pytest.raises(ApiError, match="422"):
        client.predict({"age": 500})
    with pytest.raises(ApiError, match="404"):
        client.label("no-such-id", 1)


def test_http_client_reconnects_after_a_dropped_connection(server: str) -> None:
    client = HttpClient(server)
    client.health()
    client._connection.close()

    assert client.health()["status"] == "ok"


def test_http_client_requires_an_http_url() -> None:
    with pytest.raises(ValueError, match="http://"):
        HttpClient("ftp://example.org")


class FakeClient:
    def __init__(self, health: list[dict[str, int]] | None = None) -> None:
        self.calls: list[tuple[str, Any]] = []
        self._health = health or []
        self._next = 0

    def predict(self, features: dict[str, Any]) -> str:
        self._next += 1
        self.calls.append(("predict", self._next))
        return str(self._next)

    def label(self, request_id: str, label: int) -> None:
        self.calls.append(("label", int(request_id)))

    def alerts(self, limit: int) -> list[dict[str, Any]]:
        return []

    def health(self) -> dict[str, Any]:
        return self._health.pop(0) if len(self._health) > 1 else self._health[0]


def test_replay_releases_labels_after_their_delay() -> None:
    fake = FakeClient()

    sent = replay_windows(fake, stream())

    assert sent == PARAMS.n_windows * PARAMS.window_size
    first_label = next(i for i, (kind, _) in enumerate(fake.calls) if kind == "label")
    # Window 0's labels arrive at window L=1, i.e. right after window 0's rows.
    assert first_label == PARAMS.window_size
    labelled = [rid for kind, rid in fake.calls if kind == "label"]
    assert labelled == list(range(1, len(labelled) + 1))
    assert len(labelled) == (PARAMS.n_windows - PARAMS.label_delay) * PARAMS.window_size


def health(processed: int, checked: int, labelled: int = 2) -> dict[str, int]:
    return {
        "window_size": 200,
        "predictions": 400,
        "processed_windows": processed,
        "labelled_windows": labelled,
        "performance_checked_windows": checked,
    }


def test_wait_for_monitor_waits_for_windows_and_performance_checks() -> None:
    fake = FakeClient([health(1, 0), health(2, 1), health(2, 2)])

    assert wait_for_monitor(fake, timeout=5, poll=0.01)
    assert fake._health == [health(2, 2)]


def test_wait_for_monitor_gives_up_after_timeout() -> None:
    fake = FakeClient([health(0, 0)])

    assert not wait_for_monitor(fake, timeout=0.05, poll=0.01)


def test_post_is_not_resent_after_it_went_out(server: str) -> None:
    client = HttpClient(server)
    client.predict(payloads(1)[0])
    sent: list[str] = []
    real_send = client._send

    def flaky_send(method, path, payload, headers):  # type: ignore[no-untyped-def]
        sent.append(method)
        raise ConnectionResetError("connection reset after the request was sent")

    client._send = flaky_send  # type: ignore[method-assign]

    with pytest.raises(ConnectionResetError):
        client.predict(payloads(1)[0])
    assert sent == ["POST"]
    client._send = real_send  # type: ignore[method-assign]
    assert client.health()["predictions"] == 1


def test_replay_command_against_a_running_server(
    server: str, tiny_artifact: ModelArtifact, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processed = tmp_path / "data" / "processed"
    split.save_splits(
        {
            "stream_pool": make_adult_frame(3000, seed=92),
            "reference": make_adult_frame(1500, seed=93),
        },
        processed,
    )
    save_artifact(tiny_artifact, tmp_path / "artifacts")
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "data_dir": str(tmp_path / "data"),
                "artifacts_dir": str(tmp_path / "artifacts"),
                "scenarios_dir": str(REPO_ROOT / "configs" / "scenarios"),
                "window_size": 200,
                "stream": {"n_windows": 4, "drift_start": 2, "detection_horizon": 2},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(api_client, "wait_for_monitor", lambda *a, **k: True)

    result = CliRunner().invoke(
        app,
        [
            "replay",
            "--scenario",
            "sudden_covariate",
            "--severity",
            "high",
            "--url",
            server,
            "--config",
            str(config),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "replaying 4 windows of 200 rows" in result.output
    assert HttpClient(server).health()["predictions"] == 800
