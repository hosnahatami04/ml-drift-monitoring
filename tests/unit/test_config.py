from pathlib import Path

import pytest

from driftmon.config import ConfigError, Settings, load_settings

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_default_config_file_loads() -> None:
    settings = load_settings(REPO_ROOT / "configs" / "default.yaml")

    assert settings.seed == 42
    assert settings.window_size == 1000
    assert settings.label_delay == 3
    assert settings.log_level == "INFO"
    assert settings.data_dir == Path("data")


def test_yaml_values_override_defaults(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("seed: 7\nwindow_size: 500\n", encoding="utf-8")

    settings = load_settings(config)

    assert settings.seed == 7
    assert settings.window_size == 500
    assert settings.label_delay == Settings().label_delay


def test_environment_overrides_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("seed: 7\n", encoding="utf-8")
    monkeypatch.setenv("DRIFTMON_SEED", "99")

    settings = load_settings(config)

    assert settings.seed == 99


def test_invalid_value_raises_clear_error(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("seed: not-a-number\n", encoding="utf-8")

    with pytest.raises(ConfigError, match=str(config.name)):
        load_settings(config)


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("window_sze: 1000\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="window_sze"):
        load_settings(config)


def test_missing_file_raises_clear_error(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"

    with pytest.raises(ConfigError, match="not found"):
        load_settings(missing)


def test_non_mapping_yaml_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("- just\n- a\n- list\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="mapping"):
        load_settings(config)
