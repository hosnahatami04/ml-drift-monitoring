"""p95 latency of POST /predict, measured in-process against the live app.

Usage: uv run python benchmarks/api_latency.py
"""

import statistics
import tempfile
import time
from pathlib import Path

from fastapi.testclient import TestClient

from driftmon.config import load_settings
from driftmon.data import schema, split
from driftmon.service.api import build_state, create_app

N_REQUESTS = 1000

WARMUP = 50


def main() -> None:
    settings = load_settings()
    reference = split.load_split(settings.processed_data_dir, "reference")
    rows = reference[list(schema.MODEL_FEATURES)].to_dict(orient="records")

    with tempfile.TemporaryDirectory() as tmp:
        state = build_state(settings, db_path=Path(tmp) / "latency.db")
        client = TestClient(create_app(state))

        payloads = []
        for i in range(N_REQUESTS + WARMUP):
            row = rows[i % len(rows)]
            payloads.append({k: v for k, v in row.items() if v == v})

        for payload in payloads[:WARMUP]:
            client.post("/predict", json=payload).raise_for_status()

        durations = []
        for payload in payloads[WARMUP:]:
            start = time.perf_counter()
            response = client.post("/predict", json=payload)
            durations.append((time.perf_counter() - start) * 1000)
            response.raise_for_status()
        state.store.close()

    durations.sort()
    print(f"{N_REQUESTS} single-row requests, in-process")
    print(f"p50: {statistics.median(durations):.2f} ms")
    print(f"p95: {durations[int(0.95 * N_REQUESTS) - 1]:.2f} ms")
    print(f"max: {durations[-1]:.2f} ms")


if __name__ == "__main__":
    main()
