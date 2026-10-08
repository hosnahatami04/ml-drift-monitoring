"""Application settings loaded from YAML with environment overrides."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

DEFAULT_CONFIG_PATH = Path("configs") / "default.yaml"

ENV_PREFIX = "DRIFTMON_"


class ConfigError(ValueError):
    """Raised when a configuration file is missing or invalid."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix=ENV_PREFIX, extra="forbid")

    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    reports_dir: Path = Path("reports")
    seed: int = 42
    window_size: int = 1000
    label_delay: int = 3
    log_level: str = "INFO"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # YAML values arrive through init kwargs; environment variables must
        # still win over the file, so env is placed first.
        return (env_settings, init_settings)


def load_settings(config_path: Path | None = None) -> Settings:
    path = config_path if config_path is not None else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")

    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"config file is not valid YAML: {path}: {exc}") from exc

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"config file must contain a mapping at the top level: {path}")

    try:
        return Settings(**raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration in {path}: {exc}") from exc
