from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from bank_predictor.config import FEATURE_FIELDS, METADATA_PATH, MODEL_PATH, NEGATIVE_LABEL, POSITIVE_LABEL
from bank_predictor.ml import load_metadata, load_pipeline, normalize_record
from bank_predictor.schemas import HealthResponse, MetadataResponse, PredictionInput, PredictionResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise RuntimeError("Model artifacts are missing. Run `python train.py` before starting the API.")

    app.state.pipeline = load_pipeline()
    app.state.metadata = load_metadata()
    yield


app = FastAPI(
    title="Bank Marketing Predictor API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        artifacts_ready=MODEL_PATH.exists() and METADATA_PATH.exists(),
    )


@app.get("/metadata", response_model=MetadataResponse)
def get_metadata() -> MetadataResponse:
    metadata = getattr(app.state, "metadata", None)
    if metadata is None:
        raise HTTPException(status_code=503, detail="Metadata is not loaded.")
    return MetadataResponse.model_validate(metadata)


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionInput) -> PredictionResponse:
    pipeline = getattr(app.state, "pipeline", None)
    metadata = getattr(app.state, "metadata", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    if metadata is None:
        raise HTTPException(status_code=503, detail="Metadata is not loaded.")

    normalized_input = normalize_record(payload.model_dump())
    input_frame = pd.DataFrame([normalized_input], columns=FEATURE_FIELDS)

    probabilities = pipeline.predict_proba(input_frame)[0]
    class_probabilities = {str(label): float(prob) for label, prob in zip(pipeline.classes_, probabilities)}
    probability_yes = class_probabilities.get(POSITIVE_LABEL, 0.0)
    probability_no = class_probabilities.get(NEGATIVE_LABEL, 0.0)
    decision_threshold = float(metadata.get("decision_threshold", 0.5))
    predicted_label = POSITIVE_LABEL if probability_yes >= decision_threshold else NEGATIVE_LABEL

    return PredictionResponse(
        label=predicted_label,
        probability_yes=probability_yes,
        probability_no=probability_no,
        decision_threshold=decision_threshold,
        normalized_input=PredictionInput.model_validate(normalized_input),
    )
