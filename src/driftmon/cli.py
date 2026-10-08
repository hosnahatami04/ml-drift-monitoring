"""Command line entry point."""

from importlib.metadata import version
from typing import Annotated

import typer

app = typer.Typer(
    name="driftmon",
    help="Drift detection and observability for production ML.",
    add_completion=False,
    no_args_is_help=True,
)


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
