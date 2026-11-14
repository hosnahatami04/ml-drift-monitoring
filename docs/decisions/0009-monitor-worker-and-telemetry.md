# 0009. Separate monitor worker and telemetry

## Status

Accepted. Supersedes the "monitor runs synchronously in the request" part of
ADR 0007; storage and count-based windowing stand.

## Context

ADR 0007 ran the monitor inside the request that completed a window. That
keeps a single process simple, but in the deployed stack it couples request
latency to window processing, puts drift state inside the API process, and
gives Prometheus nothing to scrape for the monitor separately. The
deployment also needs metrics and traces without a paid backend.

## Decision

- A `driftmon monitor` worker polls the shared SQLite store and runs the
  same `MonitorRunner` code. The API has `service.inline_monitoring`: on by
  default (single local process, tests, `driftmon replay` in-process), off in
  docker compose, where the API only writes predictions and labels.
- On start the worker rebuilds its state from the store: it replays the
  stored error rates through Page-Hinkley and restores alert cooldowns from
  the alerts table. A restart does not reset drift detection.
- The runner serialises window processing with a lock. Inline mode is called
  from concurrent request threads, and windows must reach the detectors in
  order and exactly once.
- Metrics use prometheus-client with one registry per process, and each
  process registers only the metrics of the roles it plays. The API exposes
  request counts and latency on `/metrics`; the worker exposes drift
  statistics, alert counts and the last processed window on port 9100.
- Traces use the OpenTelemetry SDK with manual spans (`predict`,
  `monitor.process_window`, `monitor.process_labels`). The console exporter
  is the default; setting `OTEL_EXPORTER_OTLP_ENDPOINT` switches to OTLP over
  HTTP through a batch processor, so an unreachable collector only logs. The
  API samples 1% of traces in compose. FastAPI emits its own request spans
  through the same provider, and the `predict` span nests under them.

## Alternatives considered

- Keep monitoring inline everywhere: no separate service to scrape or
  restart, and window processing stays on the request path.
- A message queue between API and worker: durable hand-off, but a broker to
  run for a workload SQLite already handles. The store is the queue.
- OpenTelemetry metrics instead of prometheus-client: one SDK for both, but
  the pull model and the client library are the established Prometheus path.

## Consequences

Alerts arrive asynchronously in compose, up to one poll interval (2 s) after
a window completes; `driftmon replay --url` waits for the worker to catch up
before listing them. Every window is claimed in the store before it is
processed, so two monitoring processes cannot alert on the same window twice,
but each keeps its own Page-Hinkley state, so one monitor per store is still
the supported setup. The worker is a single point of failure for alerting.
Its healthcheck only proves the process is up; a stalled loop shows as a
`driftmon_last_processed_window` gauge that stops moving.
