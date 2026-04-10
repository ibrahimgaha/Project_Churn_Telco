"""
routers/models.py
=================
GET /models       – Return all previously trained model runs.
GET /models/best  – Return the single best run (by F1-score).
"""

from fastapi import APIRouter, HTTPException
from services.ml_service import get_all_runs

router = APIRouter()


@router.get("")
def list_models():
    """Return all MLflow-tracked runs with metrics, params, timestamps."""
    runs = get_all_runs()
    return {"runs": runs, "total": len(runs)}


@router.get("/best")
def get_best_model():
    """
    Identify the best model across all runs.

    We rank by F1-score because accuracy is misleading on imbalanced
    churn datasets (e.g. 73% non-churn → a dummy 'always No' model
    gets 73% accuracy but 0% recall).
    """
    runs = get_all_runs()
    if not runs:
        raise HTTPException(status_code=404, detail="No runs found. Train models first.")

    best = max(runs, key=lambda r: r["metrics"].get("f1_score", 0))
    return {
        "best_run": best,
        "ranking_metric": "f1_score",
        "total_runs": len(runs),
    }
