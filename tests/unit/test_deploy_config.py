"""Structural checks on the compose stack and Prometheus config. CI also
runs `docker compose config`, which these do not replace."""

from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

SERVICES = {"api", "monitor", "dashboard", "prometheus"}


@pytest.fixture(scope="module")
def compose() -> dict[str, Any]:
    document: dict[str, Any] = yaml.safe_load(
        (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    )
    return document


def test_stack_has_the_four_services_each_with_a_healthcheck(compose: dict[str, Any]) -> None:
    services = compose["services"]

    assert set(services) == SERVICES
    for name, service in services.items():
        assert "healthcheck" in service, name
        assert service["healthcheck"]["test"][0] == "CMD"


def test_monitoring_runs_in_the_worker_not_the_api(compose: dict[str, Any]) -> None:
    services = compose["services"]

    assert services["api"]["environment"]["DRIFTMON_SERVICE__INLINE_MONITORING"] == "false"
    assert "monitor" in services["monitor"]["command"]
    # The API migrates the schema, so the worker must wait for it.
    assert services["monitor"]["depends_on"]["api"]["condition"] == "service_healthy"


def test_app_services_share_one_store_volume(compose: dict[str, Any]) -> None:
    services = compose["services"]
    db_paths = {
        services[s]["environment"]["DRIFTMON_DB_PATH"] for s in ("api", "monitor", "dashboard")
    }

    assert db_paths == {"/app/state/driftmon.db"}
    for name in ("api", "monitor", "dashboard"):
        assert "state:/app/state" in services[name]["volumes"]
        assert "./artifacts:/app/artifacts:ro" in services[name]["volumes"]
    assert "state" in compose["volumes"]


def test_prometheus_image_is_pinned_and_scrapes_both_processes(compose: dict[str, Any]) -> None:
    image = compose["services"]["prometheus"]["image"]
    scrape = yaml.safe_load((REPO_ROOT / "deploy" / "prometheus.yml").read_text(encoding="utf-8"))

    assert image.split(":")[1] not in ("", "latest")
    targets = {
        job["job_name"]: job["static_configs"][0]["targets"] for job in scrape["scrape_configs"]
    }
    assert targets == {"api": ["api:8000"], "monitor": ["monitor:9100"]}
    published = {
        name: [p.split(":")[1] for p in service["ports"]]
        for name, service in compose["services"].items()
    }
    assert "8000" in published["api"]
    assert "9100" in published["monitor"]


def test_image_runs_unprivileged_with_a_writable_store() -> None:
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    runtime = dockerfile.split("AS runtime", 1)[1]

    assert "USER app" in runtime
    assert "chown app:app /app/state" in runtime
    # Without this the store would default into the root-owned /app and the
    # API would crash as soon as it starts (the CI image job runs it bare).
    assert "ENV DRIFTMON_DB_PATH=/app/state/driftmon.db" in runtime
    assert "HEALTHCHECK" in runtime
    assert "@sha256:" in dockerfile.splitlines()[2]
