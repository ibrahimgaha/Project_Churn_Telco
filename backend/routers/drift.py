"""
routers/drift.py
================
FastAPI router for Data Drift operations.
"""

from fastapi import APIRouter, HTTPException
from simulate_drift import run_drift_analysis

router = APIRouter()

@router.post("/run")
def trigger_drift():
    """
    Simulates behavior shift on customer data, runs Evidently drift reports,
    performs numerical Kolmogorov-Smirnov tests, logs artifacts to MLflow,
    and returns dataset-level metrics. Triggers automatic model retraining
    if drift share exceeds 30%.
    """
    try:
        result = run_drift_analysis()
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Drift Analysis failed: {str(e)}")
