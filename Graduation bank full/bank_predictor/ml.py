from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from bank_predictor.config import (
    ARTIFACTS_DIR,
    CATEGORICAL_FIELDS,
    CATEGORICAL_OPTIONS,
    DATA_PATH,
    FEATURE_FIELDS,
    FIELD_LABELS,
    LEAKY_COLUMNS,
    METADATA_PATH,
    MODEL_PATH,
    NEGATIVE_LABEL,
    NUMERIC_FIELDS,
    POSITIVE_LABEL,
    POSITIVE_SAMPLE_WEIGHT,
    RANDOM_STATE,
    TARGET_COLUMN,
)


def load_dataset(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")


def get_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    feature_frame = df.drop(columns=[TARGET_COLUMN, *LEAKY_COLUMNS], errors="ignore").copy()
    return feature_frame.loc[:, FEATURE_FIELDS]


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(CATEGORICAL_FIELDS)),
            ("numeric", SimpleImputer(strategy="median"), list(NUMERIC_FIELDS)),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                GradientBoostingClassifier(
                    n_estimators=250,
                    learning_rate=0.08,
                    max_depth=2,
                    subsample=1.0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for field in FEATURE_FIELDS:
        value = record[field]
        if field in NUMERIC_FIELDS:
            normalized[field] = int(value)
        else:
            normalized[field] = str(value)
    return normalized


def dataframe_from_record(record: dict[str, Any]) -> pd.DataFrame:
    normalized = normalize_record(record)
    return pd.DataFrame([normalized], columns=FEATURE_FIELDS)


def predict_positive_probability(pipeline: Pipeline, X: pd.DataFrame) -> pd.Series:
    class_index = list(pipeline.classes_).index(POSITIVE_LABEL)
    probabilities = pipeline.predict_proba(X)[:, class_index]
    return pd.Series(probabilities, index=X.index)


def predict_labels_from_probability(probability_yes: pd.Series, threshold: float) -> pd.Series:
    return probability_yes.ge(threshold).map({True: POSITIVE_LABEL, False: NEGATIVE_LABEL})


def select_best_threshold(y_true: pd.Series, probability_yes: pd.Series) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    for threshold_step in range(10, 61):
        threshold = threshold_step / 100
        y_pred = predict_labels_from_probability(probability_yes, threshold)
        metrics = {
            "precision": float(precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)),
        }

        candidate_score = (metrics["f1"], metrics["recall"], metrics["precision"])
        best_score = (best_metrics["f1"], best_metrics["recall"], best_metrics["precision"])
        if candidate_score > best_score:
            best_threshold = threshold
            best_metrics = metrics

    return best_threshold, best_metrics


def build_metadata(
    df: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    y_pred: pd.Series,
    decision_threshold: float,
    validation_metrics: dict[str, float],
) -> dict[str, Any]:
    cm = confusion_matrix(y_test, y_pred, labels=[NEGATIVE_LABEL, POSITIVE_LABEL])

    numeric_metadata: dict[str, dict[str, int]] = {}
    for field in NUMERIC_FIELDS:
        series = df[field]
        numeric_metadata[field] = {
            "min": int(series.min()),
            "max": int(series.max()),
            "default": int(series.median()),
        }

    positive_examples = df.loc[df[TARGET_COLUMN] == POSITIVE_LABEL, FEATURE_FIELDS]
    example_row = positive_examples.iloc[0].to_dict() if not positive_examples.empty else df.loc[:, FEATURE_FIELDS].iloc[0].to_dict()

    return {
        "model_name": "GradientBoostingClassifier",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "positive_label": POSITIVE_LABEL,
        "negative_label": NEGATIVE_LABEL,
        "decision_threshold": float(decision_threshold),
        "feature_order": list(FEATURE_FIELDS),
        "field_labels": FIELD_LABELS,
        "categorical_options": {field: list(CATEGORICAL_OPTIONS[field]) for field in CATEGORICAL_FIELDS},
        "numeric_fields": numeric_metadata,
        "metrics": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)),
            "train_rows": int(len(y_train)),
            "test_rows": int(len(y_test)),
            "positive_sample_weight": float(POSITIVE_SAMPLE_WEIGHT),
            "validation_precision": float(validation_metrics["precision"]),
            "validation_recall": float(validation_metrics["recall"]),
            "validation_f1": float(validation_metrics["f1"]),
            "confusion_matrix": {
                "true_negative": int(cm[0, 0]),
                "false_positive": int(cm[0, 1]),
                "false_negative": int(cm[1, 0]),
                "true_positive": int(cm[1, 1]),
            },
        },
        "example_input": normalize_record(example_row),
    }


def train_model(df: pd.DataFrame) -> tuple[Pipeline, dict[str, Any]]:
    X = get_feature_frame(df)
    y = df[TARGET_COLUMN].copy()

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )

    threshold_pipeline = build_pipeline()
    train_sample_weight = y_train.map({POSITIVE_LABEL: POSITIVE_SAMPLE_WEIGHT, NEGATIVE_LABEL: 1.0}).astype(float)
    threshold_pipeline.fit(X_train, y_train, classifier__sample_weight=train_sample_weight)

    validation_probability = predict_positive_probability(threshold_pipeline, X_val)
    decision_threshold, validation_metrics = select_best_threshold(y_val, validation_probability)

    test_probability = predict_positive_probability(threshold_pipeline, X_test)
    y_pred = predict_labels_from_probability(test_probability, decision_threshold)
    metadata = build_metadata(df, y_train, y_test, y_pred, decision_threshold, validation_metrics)

    return threshold_pipeline, metadata


def save_artifacts(pipeline: Pipeline, metadata: dict[str, Any]) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_pipeline():
    return joblib.load(MODEL_PATH)


def load_metadata() -> dict[str, Any]:
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
