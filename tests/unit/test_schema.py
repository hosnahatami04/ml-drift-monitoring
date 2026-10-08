from pathlib import Path

import pandas as pd
import pytest

from driftmon.data import schema

RAW_DATA_STYLE = (
    "39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical,"
    " Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K\n"
    "50, ?, 83311, HS-grad, 9, Divorced, ?,"
    " Unmarried, Black, Female, 0, 0, 13, ?, >50K\n"
)

RAW_TEST_STYLE = (
    "|1x3 Cross validator\n"
    "25, Private, 226802, 11th, 7, Never-married, Machine-op-inspct,"
    " Own-child, Black, Male, 0, 0, 40, United-States, <=50K.\n"
    "38, Private, 89814, HS-grad, 9, Married-civ-spouse, Farming-fishing,"
    " Husband, White, Male, 0, 0, 50, United-States, >50K.\n"
)


def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "raw.csv"
    path.write_text(content, encoding="utf-8")
    return path


def test_parse_strips_leading_spaces(tmp_path: Path) -> None:
    df = schema.read_raw_file(_write(tmp_path, RAW_DATA_STYLE))

    assert df.loc[0, "workclass"] == "State-gov"
    assert df.loc[0, "native-country"] == "United-States"


def test_parse_converts_question_marks_to_missing(tmp_path: Path) -> None:
    df = schema.read_raw_file(_write(tmp_path, RAW_DATA_STYLE))

    assert pd.isna(df.loc[1, "workclass"])
    assert pd.isna(df.loc[1, "occupation"])
    assert pd.isna(df.loc[1, "native-country"])


def test_parse_strips_trailing_period_and_banner(tmp_path: Path) -> None:
    df = schema.read_raw_file(_write(tmp_path, RAW_TEST_STYLE))

    assert len(df) == 2
    assert set(df["income"]) == {"<=50K", ">50K"}


def test_validate_accepts_clean_frames(tmp_path: Path) -> None:
    for content in (RAW_DATA_STYLE, RAW_TEST_STYLE):
        schema.validate(schema.read_raw_file(_write(tmp_path, content)))


def test_validate_rejects_out_of_range_numeric(tmp_path: Path) -> None:
    df = schema.read_raw_file(_write(tmp_path, RAW_DATA_STYLE))
    df.loc[0, "age"] = 300

    with pytest.raises(schema.SchemaError, match="'age'"):
        schema.validate(df)


def test_validate_rejects_unknown_category(tmp_path: Path) -> None:
    df = schema.read_raw_file(_write(tmp_path, RAW_DATA_STYLE))
    df.loc[0, "workclass"] = "Freelancer"

    with pytest.raises(schema.SchemaError, match="Freelancer"):
        schema.validate(df)


def test_validate_rejects_unknown_label(tmp_path: Path) -> None:
    df = schema.read_raw_file(_write(tmp_path, RAW_DATA_STYLE))
    df.loc[0, "income"] = "maybe"

    with pytest.raises(schema.SchemaError, match="'income'"):
        schema.validate(df)


def test_validate_rejects_missing_in_required_column(tmp_path: Path) -> None:
    df = schema.read_raw_file(_write(tmp_path, RAW_DATA_STYLE))
    df.loc[0, "marital-status"] = None

    with pytest.raises(schema.SchemaError, match="'marital-status'"):
        schema.validate(df)


def test_validate_rejects_non_integer_dtype(tmp_path: Path) -> None:
    df = schema.read_raw_file(_write(tmp_path, RAW_DATA_STYLE))
    df["age"] = df["age"].astype(float)

    with pytest.raises(schema.SchemaError, match="integer"):
        schema.validate(df)
