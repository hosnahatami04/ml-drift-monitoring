"""Command line entry point."""

from importlib.metadata import version
from pathlib import Path
from typing import Annotated

import typer

from driftmon.config import DEFAULT_CONFIG_PATH, Settings, load_settings
from driftmon.logging import setup_logging

app = typer.Typer(
    name="driftmon",
    help="Drift detection and observability for production ML.",
    add_completion=False,
    no_args_is_help=True,
)

ConfigOption = Annotated[
    Path,
    typer.Option("--config", "-c", help="Path to the YAML configuration file."),
]


def _print_version(value: bool) -> None:
    if value:
        typer.echo(version("driftmon"))
        raise typer.Exit()


@app.callback()
def main(
    show_version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_print_version,
            is_eager=True,
            help="Show the package version and exit.",
        ),
    ] = False,
) -> None:
    """Drift detection and observability for production ML."""


def _load(config: Path) -> Settings:
    settings = load_settings(config)
    setup_logging(settings.log_level)
    return settings


@app.command("download-data")
def download_data(config: ConfigOption = DEFAULT_CONFIG_PATH) -> None:
    """Download the dataset, validate it and write the four processed splits."""
    from driftmon.data import split
    from driftmon.data.download import load_raw_dataset

    settings = _load(config)
    df = load_raw_dataset(settings)
    prepared = split.prepare(df, settings.splits, settings.seed)
    paths = split.save_splits(prepared.splits, settings.processed_data_dir)
    split.save_summary(prepared, settings.processed_data_dir)

    typer.echo(f"rows: {prepared.n_rows_raw}")
    typer.echo(f"duplicates removed: {prepared.n_duplicates_removed}")
    for name, path in paths.items():
        typer.echo(f"{name}: {len(prepared.splits[name])} rows -> {path}")


@app.command()
def train(config: ConfigOption = DEFAULT_CONFIG_PATH) -> None:
    """Train the baseline models and persist the model artifact."""
    from driftmon.data import split
    from driftmon.model import artifacts
    from driftmon.model import train as training
    from driftmon.model.reference import compute_reference_stats

    settings = _load(config)
    processed = settings.processed_data_dir
    train_df = split.load_split(processed, "train")
    validation_df = split.load_split(processed, "validation")
    reference_df = split.load_split(processed, "reference")

    result = training.train_models(train_df, validation_df, settings.seed)
    for model_name, model in (
        ("logistic_regression", result.baseline_model),
        ("xgboost", result.model),
    ):
        result.metrics[model_name]["reference"] = training.evaluate(model, reference_df).as_dict()

    reference_auc = result.metrics["xgboost"]["reference"]["auc"]
    reference_stats = compute_reference_stats(reference_df, result.model, reference_auc)

    summary = split.load_summary(processed)
    uci = settings.dataset.uci
    metadata = artifacts.build_metadata(
        data_hashes={"adult.data": uci.data_sha256, "adult.test": uci.test_sha256},
        seed=settings.seed,
        metrics=result.metrics,
        n_duplicates_removed=summary["n_duplicates_removed"],
        split_sizes=summary["split_sizes"],
    )
    artifact = artifacts.ModelArtifact(
        model=result.model,
        baseline_model=result.baseline_model,
        reference=reference_stats,
        metadata=metadata,
    )
    out_dir = artifacts.save_artifact(artifact, settings.artifacts_dir)

    report_path = settings.reports_dir / "baseline.md"
    training.write_baseline_report(report_path, result.metrics, metadata)

    typer.echo(f"artifact: {out_dir}")
    typer.echo(f"report: {report_path}")
    typer.echo(f"xgboost validation auc: {result.metrics['xgboost']['validation']['auc']:.4f}")


@app.command()
def simulate(
    scenario: Annotated[str, typer.Option(help="Scenario name, e.g. sudden_covariate.")],
    severity: Annotated[str, typer.Option(help="low, medium or high.")] = "medium",
    seed: Annotated[int, typer.Option(help="Stream seed.")] = 0,
    out: Annotated[Path | None, typer.Option(help="Output parquet path.")] = None,
    config: ConfigOption = DEFAULT_CONFIG_PATH,
) -> None:
    """Generate one simulated stream and save it for manual inspection."""
    import json

    import pandas as pd

    from driftmon.data import schema, split
    from driftmon.simulation import sampler
    from driftmon.simulation.ground_truth import build_ground_truth
    from driftmon.simulation.scenarios import StreamParams, generate_stream, load_scenario

    settings = _load(config)
    pool = split.load_split(settings.processed_data_dir, "stream_pool")
    reference = split.load_split(settings.processed_data_dir, "reference")
    scenario_config = load_scenario(settings.scenarios_dir / f"{scenario}.yaml")
    params = StreamParams.from_settings(settings)
    stats = sampler.reference_feature_stats(reference, schema.NUMERIC_FEATURES)

    level = None if scenario_config.kind == "control" else severity
    frames = []
    for window in generate_stream(pool, stats, scenario_config, level, seed, params):
        frame = window.X.copy()
        frame.insert(0, "window_index", window.index)
        frame["y_true"] = window.y_true
        frame["label_available_at"] = window.label_available_at
        frames.append(frame)
    stream_df = pd.concat(frames, ignore_index=True)

    out_path = out or (
        settings.data_dir / "simulated" / f"{scenario}-{severity}-seed{seed}.parquet"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stream_df.to_parquet(out_path, index=False)
    truth = build_ground_truth(scenario_config, level, params)
    truth_path = out_path.with_suffix(".ground_truth.json")
    truth_path.write_text(json.dumps(truth.to_dict(), indent=2), encoding="utf-8")

    typer.echo(f"stream: {out_path} ({params.n_windows} windows)")
    typer.echo(f"ground truth: {truth_path}")
    if truth.t0 is not None:
        typer.echo(f"drift from window {truth.t0}")


@app.command("calibrate-severity")
def calibrate_severity(config: ConfigOption = DEFAULT_CONFIG_PATH) -> None:
    """Measure real shift size and model AUC impact per scenario and severity,
    on calibration seeds only."""
    from driftmon.data import schema, split
    from driftmon.model import artifacts
    from driftmon.simulation import sampler
    from driftmon.simulation import severity as severity_module
    from driftmon.simulation.scenarios import StreamParams, load_scenarios

    settings = _load(config)
    pool = split.load_split(settings.processed_data_dir, "stream_pool")
    reference = split.load_split(settings.processed_data_dir, "reference")
    artifact = artifacts.load_artifact(artifacts.latest_artifact_dir(settings.artifacts_dir))
    scenarios = load_scenarios(settings.scenarios_dir)
    params = StreamParams.from_settings(settings)
    stats = sampler.reference_feature_stats(reference, schema.NUMERIC_FEATURES)
    seeds = settings.seeds.calibration

    measurements = severity_module.measure_all(
        pool, stats, scenarios, seeds, params, artifact.model, artifact.reference.auc
    )
    report_path = settings.reports_dir / "severity.md"
    severity_module.write_severity_report(report_path, measurements, artifact.reference.auc, seeds)

    typer.echo(f"report: {report_path}")
    for m in measurements:
        shift = f"shift {m.mean_abs_shift_sd:.3f} sd" if m.mean_abs_shift_sd is not None else ""
        flip = f"flip {m.realized_flip_rate:.3f}" if m.realized_flip_rate is not None else ""
        detail = shift or flip
        typer.echo(
            f"{m.scenario}/{m.severity}: {detail}, auc {m.auc_drifted:.4f} ({m.auc_delta:+.4f})"
        )


WorkersOption = Annotated[
    int | None,
    typer.Option(help="Worker processes; defaults to evaluation.n_workers from the config."),
]


@app.command()
def calibrate(workers: WorkersOption = None, config: ConfigOption = DEFAULT_CONFIG_PATH) -> None:
    """Search detection thresholds on calibration seeds and freeze them in
    the thresholds file."""
    import json

    from driftmon.config import load_thresholds
    from driftmon.evaluation import calibration, harness
    from driftmon.simulation.scenarios import StreamParams, load_scenarios

    settings = _load(config)
    n_workers = workers or settings.evaluation.n_workers
    grid = settings.evaluation.calibration
    seeds = settings.seeds.calibration
    calibration.ensure_calibration_seeds(seeds, settings.seeds)

    inputs, artifact = harness.prepare_inputs(settings)
    tasks = harness.build_tasks(
        load_scenarios(settings.scenarios_dir),
        [grid.severity],
        seeds,
        StreamParams.from_settings(settings),
        settings.stream.detection_horizon,
    )
    records = harness.run_streams(tasks, inputs, n_workers)
    result = calibration.calibrate(
        records,
        artifact.reference,
        load_thresholds(settings.thresholds_path),
        grid,
        settings.seeds,
        n_workers,
    )
    calibration.write_thresholds(settings.thresholds_path, result.thresholds, seeds)
    summary_path = settings.artifacts_dir / "calibration.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result.summary(), indent=2), encoding="utf-8")

    typer.echo(f"thresholds: {settings.thresholds_path}")
    typer.echo(f"selection summary (calibration seeds, not a KPI report): {summary_path}")
    if not result.feasible:
        typer.echo("warning: no candidate met the false alarm ceiling", err=True)


@app.command()
def evaluate(
    smoke: Annotated[
        bool, typer.Option(help="Small subset (first evaluation seeds, medium severity).")
    ] = False,
    out_dir: Annotated[
        Path | None,
        typer.Option(help="Output directory; defaults to reports/ (smoke: artifacts/smoke/)."),
    ] = None,
    workers: WorkersOption = None,
    config: ConfigOption = DEFAULT_CONFIG_PATH,
) -> None:
    """Score the frozen thresholds on evaluation seeds and write the KPI report."""
    from dataclasses import asdict

    from driftmon.config import load_thresholds
    from driftmon.evaluation import harness, report
    from driftmon.evaluation.calibration import thresholds_document
    from driftmon.model import artifacts
    from driftmon.simulation.scenarios import load_scenarios

    settings = _load(config)
    evaluation = settings.evaluation
    n_workers = workers or evaluation.n_workers
    target = out_dir or (settings.artifacts_dir / "smoke" if smoke else settings.reports_dir)

    inputs, artifact = harness.prepare_inputs(settings)
    artifact_dir = artifacts.latest_artifact_dir(settings.artifacts_dir)
    scenarios = load_scenarios(settings.scenarios_dir)
    thresholds = load_thresholds(settings.thresholds_path)

    run = harness.run_evaluation(
        settings, inputs, scenarios, thresholds, smoke=smoke, n_workers=n_workers
    )
    hashed_settings = settings.model_dump(
        mode="json", exclude={"log_level": True, "evaluation": {"n_workers"}}
    )
    provenance = {
        "artifact": artifact_dir.name,
        "model_sha256": report.file_sha256(artifact_dir / "model.pkl"),
        "reference_sha256": report.file_sha256(artifact_dir / "reference.json"),
        "reference_auc": artifact.reference.auc,
        "config_sha256": report.config_sha256(
            {
                "settings": hashed_settings,
                "thresholds": thresholds_document(thresholds),
                "scenarios": {n: s.model_dump(mode="json") for n, s in scenarios.items()},
            }
        ),
        "thresholds": thresholds_document(thresholds),
        "seeds": run.seeds,
        "calibration_seeds": settings.seeds.calibration,
        "severities": run.severities,
        "stream": asdict(run.params),
        "horizon": run.horizon,
        "degradation_margin": evaluation.degradation_margin,
        "bootstrap": {"samples": evaluation.bootstrap_samples, "seed": evaluation.bootstrap_seed},
        "library_versions": artifacts.library_versions(),
    }
    document = report.build_results(
        run,
        smoke=smoke,
        target_severity=evaluation.target_severity,
        provenance=provenance,
        correctness=harness.detector_correctness(
            evaluation.bootstrap_seed, evaluation.correctness_cases
        ),
        samples=evaluation.bootstrap_samples,
        seed=evaluation.bootstrap_seed,
    )
    report.write_json(target / "results.json", document)
    report.write_markdown(target / "results.md", document)
    if not smoke:
        report.write_figures(target / "figures", document)

    typer.echo(f"results: {target / 'results.json'}")
    for row in document.kpis:
        status = {True: "met", False: "MISSED", None: "-"}[row.met]
        typer.echo(f"  {row.key:<36} {row.estimate.value!s:<24} {status}")


@app.command()
def replay(
    scenario: Annotated[str, typer.Option(help="Scenario name, e.g. sudden_covariate.")],
    severity: Annotated[str, typer.Option(help="low, medium or high.")] = "medium",
    seed: Annotated[int, typer.Option(help="Stream seed.")] = 100,
    url: Annotated[
        str | None,
        typer.Option(help="Base URL of a running service; omit to replay in-process."),
    ] = None,
    wait: Annotated[
        float, typer.Option(help="Seconds to wait for a separate monitor to catch up.")
    ] = 60.0,
    config: ConfigOption = DEFAULT_CONFIG_PATH,
) -> None:
    """Send one simulated stream through the API and show the alerts it raised."""
    from driftmon.data import schema, split
    from driftmon.service import client as api_client
    from driftmon.simulation import sampler
    from driftmon.simulation.scenarios import StreamParams, generate_stream, load_scenario

    settings = _load(config)
    pool = split.load_split(settings.processed_data_dir, "stream_pool")
    reference = split.load_split(settings.processed_data_dir, "reference")
    scenario_config = load_scenario(settings.scenarios_dir / f"{scenario}.yaml")
    params = StreamParams.from_settings(settings)
    stats = sampler.reference_feature_stats(reference, schema.NUMERIC_FEATURES)

    client: api_client.ApiClient
    db_path: Path | None = None
    if url is not None:
        client = api_client.HttpClient(url)
    else:
        from fastapi.testclient import TestClient

        from driftmon.service.api import build_state, create_app

        db_path = settings.db_path.parent / f"replay-{scenario}-{severity}-{seed}.db"
        for stale in (db_path, db_path.with_suffix(".alerts.jsonl")):
            stale.unlink(missing_ok=True)
        for suffix in ("-wal", "-shm"):
            Path(str(db_path) + suffix).unlink(missing_ok=True)
        local = settings.model_copy(
            update={"service": settings.service.model_copy(update={"inline_monitoring": True})}
        )
        client = api_client.InProcessClient(
            TestClient(create_app(build_state(local, db_path=db_path)))
        )

    before = int(client.health()["predictions"])
    if before % params.window_size:
        typer.echo(
            f"warning: the service already holds {before} predictions, not a whole number "
            f"of {params.window_size}-row windows; this stream's windows will be shifted",
            err=True,
        )
    first_window = before // params.window_size

    level = None if scenario_config.kind == "control" else severity
    typer.echo(f"replaying {params.n_windows} windows of {params.window_size} rows")
    api_client.replay_windows(
        client, generate_stream(pool, stats, scenario_config, level, seed, params)
    )
    if url is not None:
        api_client.wait_for_monitor(client, params.window_size, timeout=wait)

    alerts = [a for a in client.alerts(limit=10_000) if a["window_id"] >= first_window]
    typer.echo(f"alerts: {len(alerts)}")
    for alert in sorted(alerts, key=lambda a: a["window_id"]):
        features = ",".join(alert["features"]) or "-"
        typer.echo(
            f"  window {alert['window_id'] - first_window:>3}  {alert['type']:<18} "
            f"{alert['severity']:<8} {features}"
        )
    if db_path is not None:
        typer.echo(f"db: {db_path}")


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port.")] = 8000,
    config: ConfigOption = DEFAULT_CONFIG_PATH,
) -> None:
    """Run the prediction service."""
    import uvicorn

    from driftmon.observability.tracing import setup_tracing
    from driftmon.service.api import build_state, create_app

    settings = _load(config)
    setup_tracing(settings.telemetry, "driftmon-api")
    uvicorn.run(create_app(build_state(settings)), host=host, port=port)


@app.command()
def monitor(
    metrics_port: Annotated[
        int | None, typer.Option(help="Port for /metrics; defaults to the config value.")
    ] = None,
    config: ConfigOption = DEFAULT_CONFIG_PATH,
) -> None:
    """Run the monitoring worker against the shared prediction store."""
    import signal
    import threading

    from prometheus_client import start_http_server

    from driftmon.monitoring.runner import poll
    from driftmon.observability.tracing import setup_tracing
    from driftmon.service.api import build_state

    settings = _load(config)
    setup_tracing(settings.telemetry, "driftmon-monitor")
    state = build_state(settings, role="monitor")
    state.runner.restore_state()
    port = metrics_port or settings.service.monitor_metrics_port
    start_http_server(port, registry=state.metrics.registry)
    typer.echo(f"monitor polling {settings.db_path}, metrics on :{port}")

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    poll(state.runner, settings.service.monitor_interval_seconds, stop)
    state.store.close()
