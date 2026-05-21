# serve_mlflow.ps1
# ================
# Serves the registered Production churn model via MLflow models serve.
# Run from the PROJECT ROOT (one level above backend/).
#
# FIX NOTES:
#   1. DB path built from $PSScriptRoot so it works regardless of where
#      PowerShell's CWD is when the script is called.
#   2. --no-conda replaced by --env-manager=local (MLflow >= 2.3).
#   3. MLFLOW_TRACKING_URI exported before the mlflow subprocess so it
#      can find the registered model.

$projectRoot = $PSScriptRoot
$dbAbsPath   = (Join-Path $projectRoot "mlflow.db").Replace("\", "/")
$env:MLFLOW_TRACKING_URI = "sqlite:///$dbAbsPath"

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "  Launching MLflow Model Server" -ForegroundColor Green
Write-Host "  Model  : models:/churn_model/Production" -ForegroundColor Cyan
Write-Host "  Port   : http://127.0.0.1:5001" -ForegroundColor Cyan
Write-Host "  DB     : $env:MLFLOW_TRACKING_URI" -ForegroundColor DarkGray
Write-Host "==========================================================" -ForegroundColor Green

# FIX: use --env-manager=local instead of deprecated --no-conda
& python -m mlflow models serve `
    -m "models:/churn_model/Production" `
    --port 5001 `
    --host 127.0.0.1 `
    --env-manager local