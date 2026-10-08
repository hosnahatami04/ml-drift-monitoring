"""Column definitions, parsing and validation for the UCI Adult dataset."""

from pathlib import Path

import pandas as pd

LABEL = "income"
POSITIVE_LABEL = ">50K"
LABELS = ("<=50K", ">50K")

# (min, max) observed bounds of the published dataset; anything outside is
# treated as corrupt input rather than silently accepted.
NUMERIC_RANGES: dict[str, tuple[int, int]] = {
    "age": (17, 90),
    "fnlwgt": (10_000, 1_600_000),
    "education-num": (1, 16),
    "capital-gain": (0, 99_999),
    "capital-loss": (0, 4_500),
    "hours-per-week": (1, 99),
}

CATEGORIES: dict[str, tuple[str, ...]] = {
    "workclass": (
        "Private",
        "Self-emp-not-inc",
        "Self-emp-inc",
        "Federal-gov",
        "Local-gov",
        "State-gov",
        "Without-pay",
        "Never-worked",
    ),
    "education": (
        "Bachelors",
        "Some-college",
        "11th",
        "HS-grad",
        "Prof-school",
        "Assoc-acdm",
        "Assoc-voc",
        "9th",
        "7th-8th",
        "12th",
        "Masters",
        "1st-4th",
        "10th",
        "Doctorate",
        "5th-6th",
        "Preschool",
    ),
    "marital-status": (
        "Married-civ-spouse",
        "Divorced",
        "Never-married",
        "Separated",
        "Widowed",
        "Married-spouse-absent",
        "Married-AF-spouse",
    ),
    "occupation": (
        "Tech-support",
        "Craft-repair",
        "Other-service",
        "Sales",
        "Exec-managerial",
        "Prof-specialty",
        "Handlers-cleaners",
        "Machine-op-inspct",
        "Adm-clerical",
        "Farming-fishing",
        "Transport-moving",
        "Priv-house-serv",
        "Protective-serv",
        "Armed-Forces",
    ),
    "relationship": (
        "Wife",
        "Own-child",
        "Husband",
        "Not-in-family",
        "Other-relative",
        "Unmarried",
    ),
    "race": (
        "White",
        "Asian-Pac-Islander",
        "Amer-Indian-Eskimo",
        "Other",
        "Black",
    ),
    "sex": ("Female", "Male"),
    "native-country": (
        "United-States",
        "Cambodia",
        "England",
        "Puerto-Rico",
        "Canada",
        "Germany",
        "Outlying-US(Guam-USVI-etc)",
        "India",
        "Japan",
        "Greece",
        "South",
        "China",
        "Cuba",
        "Iran",
        "Honduras",
        "Philippines",
        "Italy",
        "Poland",
        "Jamaica",
        "Vietnam",
        "Mexico",
        "Portugal",
        "Ireland",
        "France",
        "Dominican-Republic",
        "Laos",
        "Ecuador",
        "Taiwan",
        "Haiti",
        "Columbia",
        "Hungary",
        "Guatemala",
        "Nicaragua",
        "Scotland",
        "Thailand",
        "Yugoslavia",
        "El-Salvador",
        "Trinadad&Tobago",
        "Peru",
        "Hong",
        "Holand-Netherlands",
    ),
}

COLUMNS: tuple[str, ...] = (
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    LABEL,
)

# Columns where the published files use "?" for unknown values.
MAY_BE_MISSING = ("workclass", "occupation", "native-country")

# Model features per the feature selection decision: fnlwgt is a sampling
# weight, education duplicates education-num, and race/sex/native-country are
# excluded so the model is not a demographic classifier.
NUMERIC_FEATURES = ("age", "education-num", "capital-gain", "capital-loss", "hours-per-week")
CATEGORICAL_FEATURES = ("workclass", "marital-status", "occupation", "relationship")
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


class SchemaError(ValueError):
    """Raised when raw data does not match the expected schema."""


def read_raw_file(path: Path) -> pd.DataFrame:
    """Parse one raw Adult file into a clean frame.

    Handles the known quirks of the published files: a banner line starting
    with "|" in adult.test, leading spaces after commas, "?" for missing
    values, and a trailing period on adult.test labels.
    """
    df = pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        skipinitialspace=True,
        na_values=["?"],
        comment="|",
        skip_blank_lines=True,
    )
    for column in df.columns:
        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]):
            df[column] = df[column].str.strip()
    df[LABEL] = df[LABEL].str.rstrip(".")
    return df


def validate(df: pd.DataFrame) -> None:
    """Reject frames that do not match the published schema, with a message
    naming the first offending column and values."""
    missing_cols = [c for c in COLUMNS if c not in df.columns]
    if missing_cols:
        raise SchemaError(f"missing columns: {missing_cols}")
    extra_cols = [c for c in df.columns if c not in (*COLUMNS, "row_id")]
    if extra_cols:
        raise SchemaError(f"unexpected columns: {extra_cols}")

    for column, (low, high) in NUMERIC_RANGES.items():
        series = df[column]
        if not pd.api.types.is_integer_dtype(series):
            raise SchemaError(f"column {column!r} must be integer, got dtype {series.dtype}")
        if series.isna().any():
            raise SchemaError(f"column {column!r} must not contain missing values")
        out_of_range = series[(series < low) | (series > high)]
        if not out_of_range.empty:
            raise SchemaError(
                f"column {column!r} has {len(out_of_range)} values outside "
                f"[{low}, {high}], e.g. {out_of_range.iloc[0]}"
            )

    for column, allowed in CATEGORIES.items():
        series = df[column]
        if column not in MAY_BE_MISSING and series.isna().any():
            raise SchemaError(f"column {column!r} must not contain missing values")
        unknown = series.dropna()[~series.dropna().isin(allowed)].unique()
        if len(unknown) > 0:
            raise SchemaError(f"column {column!r} has unknown categories: {sorted(unknown)[:5]}")

    labels = df[LABEL]
    if labels.isna().any():
        raise SchemaError(f"column {LABEL!r} must not contain missing values")
    bad_labels = labels[~labels.isin(LABELS)].unique()
    if len(bad_labels) > 0:
        raise SchemaError(f"column {LABEL!r} has unknown labels: {sorted(bad_labels)[:5]}")
