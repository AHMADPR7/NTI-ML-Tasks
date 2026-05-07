from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from bank_predictor.config import CATEGORICAL_FIELDS, CATEGORICAL_OPTIONS


class NumericFieldInfo(BaseModel):
    min: int
    max: int
    default: int


class ConfusionMatrixInfo(BaseModel):
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int


class MetricsInfo(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    train_rows: int
    test_rows: int
    positive_sample_weight: float
    validation_precision: float
    validation_recall: float
    validation_f1: float
    confusion_matrix: ConfusionMatrixInfo


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    age: int = Field(ge=0, le=120)
    job: str
    marital: str
    education: str
    default: str
    balance: int
    housing: str
    loan: str
    contact: str
    day: int = Field(ge=1, le=31)
    month: str
    campaign: int = Field(ge=0)
    pdays: int = Field(ge=-1)
    previous: int = Field(ge=0)
    poutcome: str

    @field_validator(*CATEGORICAL_FIELDS)
    @classmethod
    def validate_category(cls, value: str, info: ValidationInfo) -> str:
        allowed_values = CATEGORICAL_OPTIONS[info.field_name]
        if value not in allowed_values:
            raise ValueError(f"Expected one of {list(allowed_values)}")
        return value


class PredictionResponse(BaseModel):
    label: str
    probability_yes: float
    probability_no: float
    decision_threshold: float
    normalized_input: PredictionInput


class MetadataResponse(BaseModel):
    model_name: str
    created_at: str
    positive_label: str
    negative_label: str
    decision_threshold: float
    feature_order: list[str]
    field_labels: dict[str, str]
    categorical_options: dict[str, list[str]]
    numeric_fields: dict[str, NumericFieldInfo]
    metrics: MetricsInfo
    example_input: PredictionInput


class HealthResponse(BaseModel):
    status: str
    artifacts_ready: bool
