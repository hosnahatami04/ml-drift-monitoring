# 0007. SQLite in WAL mode and count-based windowing

Date: 2026-10-08

## Status

Accepted. The decision to run the monitor inside the request is superseded by
ADR 0009; storage and count-based windowing stand.

## Context

The service needs durable storage for predictions, delayed labels, window
reports and alerts, and a rule for when a monitoring window is "done". The
whole system targets a single node and roughly a thousand predictions per
window; operational simplicity beats scale here.

## Decision

- SQLite with `journal_mode=WAL` is the only store. WAL lets the API threads
  write while the monitor reads, with no server process to run or secure.
  Connections are short-lived per operation and the schema migrates through
  `PRAGMA user_version` steps.
- Windows close by count, not by time: the N-th prediction (arrival order)
  belongs to window (N-1)/window_size, and a window is processed as soon as
  its row count reaches window_size. Count-based windows keep the statistics
  comparable (every test sees the same sample size) and make evaluation
  deterministic; wall-clock windows would make window contents depend on
  traffic rate and would leave half-empty windows during quiet hours with
  weaker, size-varying tests.
- The monitor runs synchronously in the request that completes a window
  (about 6 ms per window on the benchmark machine). No background scheduler
  exists to fail silently, and ordering is trivially deterministic.
- Performance tracking consumes a window only when every one of its labels
  has arrived, in window order, so Page-Hinkley sees a clean error-rate
  stream.

## Alternatives considered

- Postgres: strictly more capable, but a second process, credentials and
  migrations for a single-node lab system.
- Time-based windows: variable sample sizes break threshold calibration and
  reproducibility (see above).
- A background worker for monitoring: adds scheduling and shutdown concerns
  for work that takes milliseconds per window.

## Consequences

Throughput is bounded by SQLite's single writer, which is far above the
workload here. If a window's labels never fully arrive, that window simply
never reaches the performance detector; the input-side detectors are
unaffected.
