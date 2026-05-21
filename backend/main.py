"""
main.py
=======
MLA Churn Prediction Platform – FastAPI entry point.
"""

import os
import socket
import subprocess

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import (
    train, predict, models, results,
    tune, visualize, rf_analysis, registry, drift,
)

app = FastAPI(
    title="Churn Prediction Platform",
    description="ML Experimentation Platform for Telco Churn with MLflow",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(train.router,       prefix="/train",       tags=["Training"])
app.include_router(predict.router,     prefix="/predict",     tags=["Prediction"])
app.include_router(models.router,      prefix="/models",      tags=["Models"])
app.include_router(results.router,     prefix="/results",     tags=["Results"])
app.include_router(tune.router,        prefix="/tune",        tags=["Tuning"])
app.include_router(visualize.router,   prefix="/visualize",   tags=["Visualization"])
app.include_router(rf_analysis.router, prefix="/rf-analysis", tags=["RF Analysis"])
app.include_router(registry.router,    prefix="/registry",    tags=["Model Registry"])
app.include_router(drift.router,       prefix="/drift",       tags=["Data Drift"])


@app.get("/")
def root():
    return {"message": "Churn Prediction Platform API is running 🚀"}


# FIX: liveness / readiness probe used by pipeline.ps1's Wait-ForPort
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/mlflow/launch")
def launch_mlflow():
    """
    Spawns `mlflow ui` as a background process if port 5000 is free.
    FIX: reads MLFLOW_TRACKING_URI from env so it always points at the
         same SQLite DB used by the training / registry services.
    """
    port = 5000
    host = "127.0.0.1"

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        already_running = s.connect_ex((host, port)) == 0

    if already_running:
        return {"status": "already_running", "url": f"http://{host}:{port}"}

    # FIX: derive tracking URI from environment (set by pipeline.ps1)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not tracking_uri:
        # fall back: compute DB path relative to this file
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        db_path      = os.path.join(project_root, "mlflow.db").replace("\\", "/")
        tracking_uri = f"sqlite:///{db_path}"

    backend_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        subprocess.Popen(
            [
                "python", "-m", "mlflow", "ui",
                "--backend-store-uri", tracking_uri,
                "--host", host,
                "--port", str(port),
            ],
            cwd=backend_dir,
            creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return {"status": "starting", "url": f"http://{host}:{port}"}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}
