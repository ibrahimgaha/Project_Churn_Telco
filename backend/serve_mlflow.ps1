# PowerShell script to serve the registered Production model via MLflow models serve.
# Run this from the backend/ directory.

# Configure tracking URI to the SQLite database
$env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "🚀 Launching MLflow Model Server" -ForegroundColor Green
Write-Host "Target Model : churn_model/Production" -ForegroundColor Cyan
Write-Host "Serving Port : http://127.0.0.1:5001" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Green

# Launch the server with --no-conda since we use our current active environment
..\venv\Scripts\mlflow.exe models serve -m "models:/churn_model/Production" --port 5001 --no-conda --host 127.0.0.1
