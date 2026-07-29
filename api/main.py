"""
main.py — FastAPI inference service for the fraud detection model.

Takes transaction features, returns a fraud probability and classification.
Loads the trained model (fraud_model.pkl) and scaler (scaler.pkl) produced
by train_model.py.
"""
import pickle
import logging
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger("fraud_api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Credit Card Fraud Detector API",
    description="Real-time fraud probability scoring using a RandomForest "
                 "model trained on the ULB credit card fraud dataset.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path(__file__).parent.parent / "fraud_model.pkl"
TIME_SCALER_PATH = Path(__file__).parent.parent / "time_scaler.pkl"
AMOUNT_SCALER_PATH = Path(__file__).parent.parent / "amount_scaler.pkl"

_model = None
_time_scaler = None
_amount_scaler = None
_load_failed = False


def _load():
    global _model, _time_scaler, _amount_scaler, _load_failed
    if _load_failed:
        return False
    if _model is not None and _time_scaler is not None and _amount_scaler is not None:
        return True
    try:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        with open(TIME_SCALER_PATH, "rb") as f:
            _time_scaler = pickle.load(f)
        with open(AMOUNT_SCALER_PATH, "rb") as f:
            _amount_scaler = pickle.load(f)
        return True
    except (FileNotFoundError, pickle.UnpicklingError, EOFError) as e:
        logger.error(f"Failed to load model/scalers: {e}")
        _load_failed = True
        return False


class Transaction(BaseModel):
    """A single transaction's features, matching the ULB dataset schema."""
    Time: float = Field(..., description="Seconds elapsed since the first transaction in the dataset")
    Amount: float = Field(..., description="Transaction amount")
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float

    class Config:
        json_schema_extra = {
            "example": {
                "Time": 0, "Amount": 149.62,
                "V1": -1.3598, "V2": -0.0728, "V3": 2.5363, "V4": 1.3782,
                "V5": -0.3383, "V6": 0.4624, "V7": 0.2396, "V8": 0.0987,
                "V9": 0.3638, "V10": 0.0908, "V11": -0.5516, "V12": -0.6178,
                "V13": -0.9914, "V14": -0.3112, "V15": 1.4682, "V16": -0.4704,
                "V17": 0.2080, "V18": 0.0258, "V19": 0.4040, "V20": 0.2514,
                "V21": -0.0183, "V22": 0.2778, "V23": -0.1105, "V24": 0.0669,
                "V25": 0.1285, "V26": -0.1891, "V27": 0.1336, "V28": -0.0211,
            }
        }


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold_used: float
    model_version: str


@app.get("/health")
def health():
    ok = _load()
    return {
        "status": "ok" if ok else "degraded",
        "model_loaded": ok,
        "service": "credit-card-fraud-detector",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction, threshold: float = 0.5):
    if not _load():
        raise HTTPException(
            status_code=503,
            detail="Model unavailable — failed to load fraud_model.pkl, time_scaler.pkl, or amount_scaler.pkl"
        )

    data = transaction.model_dump()
    time_val = data.pop("Time")
    amount_val = data.pop("Amount")

    # Scale Time and Amount using their own correctly-fit scalers
    time_scaled = _time_scaler.transform([[time_val]])[0][0]
    amount_scaled = _amount_scaler.transform([[amount_val]])[0][0]

    features = pd.DataFrame([{
        **data,
        "Time_scaled": time_scaled,
        "Amount_scaled": amount_scaled,
    }])

    # Match the exact column order the model was trained on
    expected_cols = [c for c in _model.feature_names_in_]
    features = features[expected_cols]

    proba = float(_model.predict_proba(features)[0][1])
    is_fraud = proba >= threshold

    return PredictionResponse(
        fraud_probability=round(proba, 6),
        is_fraud=is_fraud,
        threshold_used=threshold,
        model_version="RandomForest-balanced-v1",
    )


@app.get("/")
def root():
    return {
        "service": "Credit Card Fraud Detector API",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
    }
