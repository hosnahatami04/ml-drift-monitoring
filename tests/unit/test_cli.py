from importlib.metadata import version

from typer.testing import CliRunner

from driftmon.cli import app

runner = CliRunner()


def test_version_flag_prints_package_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.output.strip() == version("driftmon")


def test_no_arguments_shows_help() -> None:
    result = runner.invoke(app, [])

    assert "driftmon" in result.output
    assert "--help" in result.output
