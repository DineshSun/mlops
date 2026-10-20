import os
import time
import logging
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("heart_disease_api")

# ---------------------------------------------------------------------------
# Model loading via lifespan (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------
# Resolve relative to the repo root (one level up from app/)
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_APP_DIR)
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(_REPO_ROOT, "models", "best_pipeline.joblib"))
model_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_pipeline
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found at {MODEL_PATH}")
        raise RuntimeError(f"Model file not found at '{MODEL_PATH}'.")
    model_pipeline = joblib.load(MODEL_PATH)
    logger.info(f"Model pipeline loaded successfully from '{MODEL_PATH}'")
    yield
    # cleanup (if needed) goes here
    model_pipeline = None
    logger.info("Model pipeline unloaded.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description=(
        "MLOps model-serving endpoint for patient heart disease risk classification "
        "using a scikit-learn pipeline trained on the UCI Heart Disease dataset."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class PatientData(BaseModel):
    age: float = Field(..., example=57, description="Age in years")
    sex: float = Field(..., example=1, description="Sex (1 = male; 0 = female)")
    cp: float = Field(..., example=3, description="Chest pain type (1-4)")
    trestbps: float = Field(..., example=145, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., example=233, description="Serum cholesterol (mg/dl)")
    fbs: float = Field(..., example=1, description="Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)")
    restecg: float = Field(..., example=0, description="Resting ECG results (0-2)")
    thalach: float = Field(..., example=150, description="Maximum heart rate achieved")
    exang: float = Field(..., example=0, description="Exercise-induced angina (1 = yes; 0 = no)")
    oldpeak: float = Field(..., example=2.3, description="ST depression induced by exercise relative to rest")
    slope: float = Field(..., example=3, description="Slope of peak exercise ST segment (1-3)")
    ca: float = Field(..., example=0, description="Number of major vessels (0-3) coloured by fluoroscopy")
    thal: float = Field(..., example=6, description="Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 57,
                "sex": 1,
                "cp": 3,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 0,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 3,
                "ca": 0,
                "thal": 6,
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Binary class label (0 = No Risk, 1 = High Risk)")
    risk_label: str = Field(..., description="Human-readable risk label")
    confidence: float = Field(..., description="Probability for the predicted class")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities for both outcomes")


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool
    model_path: str


# ---------------------------------------------------------------------------
# Middleware – request logging & latency tracking
# ---------------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        f"{request.method} {request.url.path} → {response.status_code} "
        f"({duration_ms:.2f} ms)"
    )
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["Meta"])
def root():
    """Root health-check endpoint."""
    return {
        "status": "healthy",
        "service": "Heart Disease Risk Prediction API",
        "model_loaded": model_pipeline is not None,
    }


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health_check():
    """Liveness probe — returns 503 if model is not loaded."""
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline is not loaded.")
    return HealthResponse(
        status="UP",
        model_loaded=True,
        model_path=MODEL_PATH,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(patient: PatientData):
    """
    Accepts a JSON patient record with 13 UCI Heart Disease features and returns:
    - **prediction** – 0 (No Risk) or 1 (High Risk)
    - **risk_label** – human-readable label
    - **confidence** – probability for the predicted class
    - **probabilities** – full class probability dict
    """
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline is not loaded.")

    try:
        # Build a single-row DataFrame that matches the training schema exactly
        input_df = pd.DataFrame([patient.model_dump()])

        # Enforce column order matching training data
        feature_cols = [
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
        ]
        input_df = input_df[feature_cols]

        prediction = int(model_pipeline.predict(input_df)[0])
        proba = model_pipeline.predict_proba(input_df)[0]  # shape (2,)

        confidence = round(float(proba[prediction]), 4)
        risk_label = "High Risk" if prediction == 1 else "No Risk"

        return PredictionResponse(
            prediction=prediction,
            risk_label=risk_label,
            confidence=confidence,
            probabilities={
                "no_risk": round(float(proba[0]), 4),
                "high_risk": round(float(proba[1]), 4),
            },
        )

    except Exception as exc:
        logger.error(f"Inference error: {exc}", exc_info=True)
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}")
