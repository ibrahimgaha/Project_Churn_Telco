"""
routers/tune.py
===============
POST /tune  – Automatic hyperparameter tuning with GridSearchCV.
"""

from fastapi import APIRouter, HTTPException
from schemas.schemas import TuneRequest, TuneResponse
from services.tune_service import tune_model

router = APIRouter()


@router.post("", response_model=TuneResponse)
def auto_tune(request: TuneRequest):
    """
    Run exhaustive grid search for the specified model.
    Returns best hyperparameters and the resulting CV F1 score.
    """
    try:
        result = tune_model(request.model_name, request.cv_folds)
        return TuneResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tuning failed: {str(e)}")
