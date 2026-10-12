"""Clients for driving the prediction API, used by `driftmon replay`.

The HTTP client talks to a running service (the compose stack) over one
keep-alive connection using only the standard library; the in-process client
wraps FastAPI's TestClient for local demos without a server.
"""

import http.client
import json
import logging
import time
from collections.abc import Iterable
from typing import Any, Protocol
from urllib.parse import urlencode, urlsplit

from driftmon.data import schema
from driftmon.simulation.scenarios import Window

logger = logging.getLogger(__name__)


class ApiError(RuntimeError):
    pass


class ApiClient(Protocol):
    def predict(self, features: dict[str, Any]) -> str: ...

    def label(self, request_id: str, label: int) -> None: ...

    def alerts(self, limit: int) -> list[dict[str, Any]]: ...

    def health(self) -> dict[str, Any]: ...


class HttpClient:
    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        parts = urlsplit(base_url)
        if parts.scheme != "http" or not parts.hostname:
            raise ValueError(f"expected an http:// base URL, got {base_url!r}")
        self._host = parts.hostname
        self._port = parts.port or 80
        self._timeout = timeout
        self._connection = http.client.HTTPConnection(self._host, self._port, timeout=timeout)

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        payload = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        try:
            self._connection.request(method, path, body=payload, headers=headers)
            response = self._connection.getresponse()
            data = response.read()
        except (OSError, http.client.HTTPException):
            # A dropped keep-alive connection is retried once on a fresh one.
            self._connection.close()
            self._connection = http.client.HTTPConnection(
                self._host, self._port, timeout=self._timeout
            )
            self._connection.request(method, path, body=payload, headers=headers)
            response = self._connection.getresponse()
            data = response.read()
        if response.status >= 400:
            raise ApiError(f"{method} {path} failed with {response.status}: {data[:300]!r}")
        return json.loads(data)

    def predict(self, features: dict[str, Any]) -> str:
        return str(self._request("POST", "/predict", features)["request_id"])

    def label(self, request_id: str, label: int) -> None:
        self._request("POST", "/labels", {"request_id": request_id, "label": label})

    def alerts(self, limit: int) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = self._request(
            "GET", f"/alerts?{urlencode({'limit': limit})}"
        )
        return result

    def health(self) -> dict[str, Any]:
        result: dict[str, Any] = self._request("GET", "/health")
        return result

    def close(self) -> None:
        self._connection.close()


class InProcessClient:
    def __init__(self, test_client: Any) -> None:
        self._client = test_client

    def _check(self, response: Any) -> Any:
        if response.status_code >= 400:
            raise ApiError(f"request failed with {response.status_code}: {response.text[:300]}")
        return response.json()

    def predict(self, features: dict[str, Any]) -> str:
        return str(self._check(self._client.post("/predict", json=features))["request_id"])

    def label(self, request_id: str, label: int) -> None:
        self._check(self._client.post("/labels", json={"request_id": request_id, "label": label}))

    def alerts(self, limit: int) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = self._check(
            self._client.get("/alerts", params={"limit": limit})
        )
        return result

    def health(self) -> dict[str, Any]:
        result: dict[str, Any] = self._check(self._client.get("/health"))
        return result


def window_payloads(window: Window) -> list[dict[str, Any]]:
    rows = window.X[list(schema.MODEL_FEATURES)].to_dict(orient="records")
    # Missing categories are omitted, as a real client would; NaN != NaN.
    return [{k: v for k, v in row.items() if v == v} for row in rows]


def replay_windows(client: ApiClient, windows: Iterable[Window]) -> int:
    """Post each window's predictions and release its labels once their
    delay has passed, exactly as the evaluation harness orders them."""
    pending: dict[int, list[tuple[str, int]]] = {}
    sent = 0
    for window in windows:
        for arrived in sorted(w for w in pending if w <= window.index):
            for request_id, label in pending.pop(arrived):
                client.label(request_id, label)
        ids: list[tuple[str, int]] = []
        for payload, label in zip(window_payloads(window), window.y_true.tolist(), strict=True):
            ids.append((client.predict(payload), int(label)))
            sent += 1
        pending[window.label_available_at] = ids
    return sent


def wait_for_monitor(
    client: ApiClient, window_size: int, timeout: float, poll: float = 0.5
) -> bool:
    """With a separate monitor worker, alerts land asynchronously; wait until
    every complete window has been processed."""
    deadline = time.monotonic() + timeout
    while True:
        health = client.health()
        if int(health["processed_windows"]) >= int(health["predictions"]) // window_size:
            return True
        if time.monotonic() >= deadline:
            logger.warning("monitor did not catch up within %.0f s", timeout)
            return False
        time.sleep(poll)
