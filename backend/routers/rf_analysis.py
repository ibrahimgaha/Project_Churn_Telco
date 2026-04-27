"""
routers/rf_analysis.py
======================
GET /rf-analysis  – Performs Task 4 Random Forest Analysis.
"""

from fastapi import APIRouter, HTTPException
from schemas.schemas import RFAnalysisResponse
from services.rf_analysis_service import run_full_rf_analysis

router = APIRouter()

@router.get("", response_model=RFAnalysisResponse)
def get_rf_analysis():
    """
    Triggers the full RF analysis pipeline including stability, bias/variance,
    and error analysis as required by University Task 4.
    """
    try:
        results = run_full_rf_analysis()
        return results
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"RF Analysis failed: {str(e)}")
