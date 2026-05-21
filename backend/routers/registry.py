"""
routers/registry.py
===================
FastAPI router for Model Registry operations.
"""

from fastapi import APIRouter, HTTPException
from services.registry_service import register_best_model_pipeline, get_registered_models_info

router = APIRouter()

@router.post("/promote")
def promote_model():
    """
    Triggers the programmatic search of the best model, registers it,
    transitions it to Staging, checks its accuracy vs a threshold of 0.85,
    and promotes it to Production if the check passes.
    """
    try:
        result = register_best_model_pipeline()
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        return result
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model Registry pipeline failed: {str(e)}")

@router.get("/status")
def get_status():
    """
    Returns latest registered versions of 'churn_model'.
    """
    try:
        versions = get_registered_models_info()
        return {"versions": versions, "total": len(versions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch model registry status: {str(e)}")
