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
