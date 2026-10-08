import re
from importlib.metadata import version

from typer.testing import CliRunner

from driftmon.cli import app

runner = CliRunner()

_ANSI_CODES = re.compile(r"\x1b\[[0-9;]*m")


def _invoke_plain(args: list[str]) -> tuple[int, str]:
    result = runner.invoke(app, args, env={"NO_COLOR": "1", "TERM": "dumb"})
    return result.exit_code, _ANSI_CODES.sub("", result.output)


def test_version_flag_prints_package_version() -> None:
    exit_code, output = _invoke_plain(["--version"])

    assert exit_code == 0
    assert output.strip() == version("driftmon")


def test_help_shows_usage_and_options() -> None:
    exit_code, output = _invoke_plain(["--help"])

    assert exit_code == 0
    assert "driftmon" in output
    assert "--version" in output


def test_no_arguments_shows_help() -> None:
    _, output = _invoke_plain([])

    assert "driftmon" in output
    assert "--version" in output
