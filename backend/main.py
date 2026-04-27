"""
MLA Churn Prediction Platform – FastAPI Backend
================================================
Entry point. Registers all routers and configures CORS.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import train, predict, models, results, tune, visualize, rf_analysis
import subprocess
import socket
import os

app = FastAPI(
    title="Churn Prediction Platform",
    description="ML Experimentation Platform for Telco Churn classification with MLflow",
    version="2.0.0",
)

# Allow React dev-server to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(train.router,      prefix="/train",      tags=["Training"])
app.include_router(predict.router,    prefix="/predict",    tags=["Prediction"])
app.include_router(models.router,     prefix="/models",     tags=["Models"])
app.include_router(results.router,    prefix="/results",    tags=["Results"])
app.include_router(tune.router,       prefix="/tune",       tags=["Tuning"])
app.include_router(visualize.router,  prefix="/visualize",  tags=["Visualization"])
app.include_router(rf_analysis.router, prefix="/rf-analysis", tags=["RF Analysis Task 4"])


@app.get("/")
def root():
    return {"message": "Churn Prediction Platform API is running 🚀"}


@app.post("/mlflow/launch")
def launch_mlflow():
    """
    Spawns mlflow ui as a background process if not already running.
    Checks port 5000 availability.
    """
    port = 5000
    host = "127.0.0.1"

    # Check if port 5000 is already in use
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        is_running = s.connect_ex((host, port)) == 0

    if not is_running:
        # Construct path to mlflow.exe in venv (portable across Windows)
        venv_path = os.path.join("..", "venv", "Scripts", "mlflow.exe")
        backend_dir = os.path.dirname(os.path.abspath(__file__))

        try:
            # Start MLflow UI as a detached process
            subprocess.Popen(
                [venv_path, "ui", "--port", str(port), "--host", host],
                cwd=backend_dir,
                creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return {"status": "starting", "url": f"http://{host}:{port}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    return {"status": "already_running", "url": f"http://{host}:{port}"}
