"""
routers/predict.py
==================
POST /predict  – Upload a CSV and return predictions using a saved model.
"""

import io
import numpy as np
import pandas as pd
import joblib
import os

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from schemas.schemas import PredictResponse
from services.ml_service import load_model_by_run_id, MODEL_DIR
from utils.preprocessing import encode_features

router = APIRouter()


@router.post("", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    run_id: str = Form(...),
):
    """
    Accept a CSV (no Churn column needed), return predictions + probabilities.
    """
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    df.columns = df.columns.str.strip()

    # Drop target if accidentally included
    df.drop(columns=["Churn", "customerID"], errors="ignore", inplace=True)

    # Encode categoricals
    df_encoded, _ = encode_features(df)

    # Load model + optional scaler
    try:
        clf, scaler = load_model_by_run_id(model_name, run_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    X = df_encoded.values
    if scaler:
        X = scaler.transform(X)

    preds = clf.predict(X).tolist()
    probs = clf.predict_proba(X)[:, 1].tolist()

    return PredictResponse(
        predictions=preds,
        probabilities=[round(p, 4) for p in probs],
        model_used=model_name,
    )
