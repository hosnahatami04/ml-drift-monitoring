"""Application settings loaded from YAML with environment overrides."""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

DEFAULT_CONFIG_PATH = Path("configs") / "default.yaml"

DEFAULT_THRESHOLDS_PATH = Path("configs") / "thresholds.yaml"

ENV_PREFIX = "DRIFTMON_"


class ConfigError(ValueError):
    """Raised when a configuration file is missing or invalid."""


class UciSource(BaseModel):
    data_url: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    data_sha256: str = "5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d"
    test_url: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
    test_sha256: str = "a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05"


class OpenmlSource(BaseModel):
    name: str = "adult"
    version: int = 2
    sha256: str | None = None


class DatasetConfig(BaseModel):
    uci: UciSource = UciSource()
    openml: OpenmlSource = OpenmlSource()


class SplitFractions(BaseModel):
    train: float = 0.5
    validation: float = 0.1
    reference: float = 0.1
    stream_pool: float = 0.3

    @model_validator(mode="after")
    def _fractions_sum_to_one(self) -> "SplitFractions":
        total = self.train + self.validation + self.reference + self.stream_pool
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"split fractions must sum to 1, got {total}")
        return self


class EffectGates(BaseModel):
    ks_d: float = 0.05
    psi: float = 0.1
    cramers_v: float = 0.05


class PageHinkleySettings(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    delta: float = 0.005
    lambda_: float = Field(default=0.05, alias="lambda")


class ThresholdsConfig(BaseModel):
    """Detection thresholds; initial values until calibration freezes them."""

    alpha: float = 0.01
    correction: Literal["bonferroni", "benjamini-hochberg"] = "bonferroni"
    k_features: int = 1
    min_effect: EffectGates = EffectGates()
    page_hinkley: PageHinkleySettings = PageHinkleySettings()


def load_thresholds(path: Path | None = None) -> ThresholdsConfig:
    resolved = path if path is not None else DEFAULT_THRESHOLDS_PATH
    if not resolved.is_file():
        raise ConfigError(f"thresholds file not found: {resolved}")
    try:
        raw: Any = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"thresholds file is not valid YAML: {resolved}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"thresholds file must contain a mapping: {resolved}")
    try:
        return ThresholdsConfig(**raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid thresholds in {resolved}: {exc}") from exc


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX, env_nested_delimiter="__", extra="forbid"
    )

    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    reports_dir: Path = Path("reports")
    seed: int = 42
    window_size: int = 1000
    label_delay: int = 3
    log_level: str = "INFO"
    dataset: DatasetConfig = DatasetConfig()
    splits: SplitFractions = SplitFractions()

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

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
