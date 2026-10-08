"""Request and response schemas for the prediction service."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from driftmon.data import schema as data_schema


class PredictRequest(BaseModel):
    """One row of model features. Field aliases carry the dataset's dashed
    column names; missing workclass/occupation is allowed, as in the data."""

    model_config = ConfigDict(populate_by_name=True)

    age: int = Field(ge=17, le=90)
    education_num: int = Field(alias="education-num", ge=1, le=16)
    capital_gain: int = Field(alias="capital-gain", ge=0, le=99_999)
    capital_loss: int = Field(alias="capital-loss", ge=0, le=4_500)
    hours_per_week: int = Field(alias="hours-per-week", ge=1, le=99)
    workclass: str | None = None
    marital_status: str = Field(alias="marital-status")
    occupation: str | None = None
    relationship: str

    @field_validator("workclass", "marital_status", "occupation", "relationship")
    @classmethod
    def _known_category(cls, value: str | None, info: Any) -> str | None:
        if value is None:
            return None
        column = str(info.field_name).replace("_", "-")
        allowed = data_schema.CATEGORIES[column]
        if value not in allowed:
            raise ValueError(f"unknown {column} category {value!r}")
        return value

    def to_feature_dict(self) -> dict[str, Any]:
        return {
            "age": self.age,
            "education-num": self.education_num,
            "capital-gain": self.capital_gain,
            "capital-loss": self.capital_loss,
            "hours-per-week": self.hours_per_week,
            "workclass": self.workclass,
            "marital-status": self.marital_status,
            "occupation": self.occupation,
            "relationship": self.relationship,
        }


class PredictResponse(BaseModel):
    request_id: str
    probability: float


class LabelRequest(BaseModel):
    request_id: str
    label: int = Field(ge=0, le=1)


class LabelResponse(BaseModel):
    request_id: str
    attached: bool


class HealthResponse(BaseModel):
    status: str
    model_fingerprint: str
    predictions: int
    processed_windows: int


class AlertRecord(BaseModel):
    window_id: int
    type: str
    severity: str
    features: list[str]
    details: dict[str, Any]
    created_at: str
