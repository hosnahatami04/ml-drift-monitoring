import re
from importlib.metadata import version
from pathlib import Path

import pytest
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


def test_console_entry_turns_operator_errors_into_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # A process-wide tracer provider would leak into every later test.
    monkeypatch.setattr("driftmon.observability.tracing.setup_tracing", lambda config, name: None)
    config = tmp_path / "config.yaml"
    config.write_text(f"artifacts_dir: {tmp_path / 'none'}\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exit_info:
        app(["serve", "--config", str(config)], prog_name="driftmon")

    assert exit_info.value.code == 1
    err = capsys.readouterr().err
    assert err.startswith("error: no model artifact found")
    assert "Traceback" not in err
