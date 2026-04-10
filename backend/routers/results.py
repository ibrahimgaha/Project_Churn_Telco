"""
routers/results.py
==================
GET /results  – Fetch metrics + confusion matrix + ROC data for a given run.
"""

from fastapi import APIRouter, Query, HTTPException
from services.ml_service import get_all_runs

router = APIRouter()


@router.get("")
def get_results(run_id: str = Query(None)):
    """
    If run_id is provided, return that run's data.
    Otherwise return all runs (for comparison dashboard).
    """
    runs = get_all_runs()
    if run_id:
        match = [r for r in runs if r["run_id"] == run_id]
        if not match:
            raise HTTPException(status_code=404, detail="Run not found")
        return match[0]
    return {"runs": runs}
