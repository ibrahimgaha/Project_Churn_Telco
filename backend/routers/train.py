"""
routers/train.py
================
POST /train  – Train one or multiple models and return metrics.
"""

from fastapi import APIRouter, HTTPException
from schemas.schemas import TrainRequest, TrainBatchResponse, TrainResponse
from services.ml_service import train_and_evaluate

router = APIRouter()


@router.post("", response_model=TrainBatchResponse)
def train_models(request: TrainRequest):
    """
    Train one or more models in sequence.

    Frontend sends:
      {
        "models": ["logistic_regression", "svm"],
        "features": ["tenure", "MonthlyCharges", ...],
        "test_size": 0.2,
        "hyperparameters": {
          "logistic_regression": {"C": 1.0, "max_iter": 200},
          "svm": {"kernel": "rbf", "C": 1.0}
        }
      }

    Returns metrics for each trained model.
    """
    results = []

    for model_name in request.models:
        hp = request.hyperparameters.get(model_name, {})
        try:
            result = train_and_evaluate(
                model_name=model_name,
                hyperparams=hp,
                selected_features=request.features,
                test_size=request.test_size,
                random_state=request.random_state,
            )
            results.append(TrainResponse(**result))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Training failed for {model_name}: {str(e)}")

    return TrainBatchResponse(results=results)
