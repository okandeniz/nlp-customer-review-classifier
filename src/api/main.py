import json

from contextlib import asynccontextmanager
from pathlib import Path

import joblib

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "customer_review_classifier.joblib"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "model_metadata.json"
)


# --------------------------------------------------
# Model loading
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):

    # Load the trained pipeline
    model = joblib.load(MODEL_PATH)

    # Load the metadata
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    # Validate model configuration
    if 1 not in model.classes_:
        raise ValueError("The model does not contain class 1.")

    threshold = float(metadata["threshold"])

    if not 0 <= threshold <= 1:
        raise ValueError("Invalid classification threshold.")

    # Store model and metadata
    app.state.model = model
    app.state.metadata = metadata

    print("Model loaded successfully.")

    yield

    # Release references when the API stops
    app.state.model = None
    app.state.metadata = None


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Customer Review Classification API",
    description="Predict product recommendations from customer reviews.",
    version="1.0.0",
    lifespan=lifespan
)


# --------------------------------------------------
# Pydantic schemas
# --------------------------------------------------

class ReviewRequest(BaseModel):
    review_text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Customer review text"
    )


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probability_recommended: float
    probability_not_recommended: float
    threshold: float


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health", tags=["Health"])
def health_check(request: Request):

    return {
        "status": "ok",
        "service": "customer-review-classifier",
        "model_loaded": request.app.state.model is not None
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"]
)
def predict(payload: ReviewRequest, request: Request):

    # Validate blank text
    review_text = payload.review_text.strip()

    if not review_text:
        raise HTTPException(
            status_code=422,
            detail="Review text cannot be empty."
        )

    # Get loaded model and metadata
    model = request.app.state.model
    metadata = request.app.state.metadata

    threshold = float(metadata["threshold"])

    # Predict class probabilities
    probabilities = model.predict_proba(
        [review_text]
    )[0]

    # Find the probability of class 1
    class_1_idx = list(model.classes_).index(1)

    prob_recommended = float(
        probabilities[class_1_idx]
    )

    # Apply classification threshold
    prediction = int(
        prob_recommended >= threshold
    )

    # Get class label
    label = metadata["class_labels"][str(prediction)]

    # Return prediction result
    return PredictionResponse(
        prediction=prediction,
        label=label,
        probability_recommended=round(
            prob_recommended, 4
        ),
        probability_not_recommended=round(
            1 - prob_recommended, 4
        ),
        threshold=threshold
    )