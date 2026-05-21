# ===========================================================================
# MLOps Pipeline - FIXED STABLE VERSION (Windows PowerShell)
# Project: Churn Telco
# ===========================================================================
# Uses splatting (@{}) instead of backtick line-continuation to avoid
# the "missing closing brace" parser error on Windows PowerShell 5.x.
# ===========================================================================

$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
$projectRoot = $PSScriptRoot
$backendDir  = Join-Path $projectRoot "backend"
$pythonExe   = "python"

# Build SQLite URI with forward slashes (required by SQLite on Windows)
$dbAbsPath = (Join-Path $projectRoot "mlflow.db").Replace("\", "/")
$mlflowDb  = "sqlite:///$dbAbsPath"

# Export so ALL child processes (train, registry, serve, ui) share the same DB
$env:MLFLOW_TRACKING_URI = $mlflowDb
Write-Host "[config] MLFLOW_TRACKING_URI = $mlflowDb" -ForegroundColor DarkGray

# ---------------------------------------------------------------------------
# Helper: poll a TCP port until it accepts connections
# ---------------------------------------------------------------------------
function Wait-ForPort {
    param(
        [int]$Port,
        [int]$TimeoutSec = 90,
        [string]$Label = "service"
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    Write-Host "  Waiting for $Label on port $Port ..." -NoNewline
    while ((Get-Date) -lt $deadline) {
        try {
            $tcp = New-Object System.Net.Sockets.TcpClient
            $tcp.Connect("127.0.0.1", $Port)
            $tcp.Close()
            Write-Host " ready OK" -ForegroundColor Green
            return $true
        } catch { }
        Write-Host "." -NoNewline
        Start-Sleep -Seconds 2
    }
    Write-Host " TIMEOUT" -ForegroundColor Red
    return $false
}

# ===========================================================================
# SETUP
# ===========================================================================
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  SETUP" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

& $pythonExe -m pip install -r "$backendDir\requirements.txt" --quiet

$frontendDir = Join-Path $projectRoot "frontend"
if (Test-Path $frontendDir) {
    Push-Location $frontendDir
    npm install --silent
    Pop-Location
}

# ===========================================================================
# TRAINING
# ===========================================================================
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  TRAINING MODELS" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$trainDt = Join-Path $backendDir "_train_dt.py"
$trainRf = Join-Path $backendDir "_train_rf.py"

Set-Content $trainDt @"
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from services import ml_service as m
result = m.train_and_evaluate('decision_tree', {'max_depth': 8}, None, 0.2, 42)
print('[train] decision_tree  run_id =', result['run_id'], '| accuracy =', result['metrics'].get('accuracy'))
"@

Set-Content $trainRf @"
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from services import ml_service as m
result = m.train_and_evaluate('random_forest', {'n_estimators': 100, 'max_depth': 10}, None, 0.2, 42)
print('[train] random_forest  run_id =', result['run_id'], '| accuracy =', result['metrics'].get('accuracy'))
"@

Push-Location $backendDir
    & $pythonExe $trainDt
    if ($LASTEXITCODE -ne 0) { throw "Decision Tree training failed (exit $LASTEXITCODE)" }
    & $pythonExe $trainRf
    if ($LASTEXITCODE -ne 0) { throw "Random Forest training failed (exit $LASTEXITCODE)" }
Pop-Location

Remove-Item $trainDt -ErrorAction SilentlyContinue
Remove-Item $trainRf -ErrorAction SilentlyContinue

# ===========================================================================
# MODEL REGISTRY  (synchronous - must finish before serving starts)
# ===========================================================================
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  MODEL REGISTRY" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$registryScript = Join-Path $backendDir "_register.py"
Set-Content $registryScript @"
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from services import registry_service as r
import json, sys
result = r.register_best_model_pipeline()
print(json.dumps(result, indent=2))
if result['status'] != 'success':
    sys.exit(1)
"@

Push-Location $backendDir
    & $pythonExe $registryScript
    if ($LASTEXITCODE -ne 0) { throw "Model registration failed (exit $LASTEXITCODE)" }
Pop-Location

Remove-Item $registryScript -ErrorAction SilentlyContinue

# ===========================================================================
# MLFLOW SERVING  (port 5001)
# Uses splatting to avoid backtick parser errors
# ===========================================================================
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  MLFLOW MODEL SERVING  ->  http://127.0.0.1:5001" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$serveArgs = @(
    "-m", "mlflow",
    "models", "serve",
    "-m", "models:/churn_model/Production",
    "--port", "5001",
    "--host", "127.0.0.1",
    "--env-manager", "local"
)

$serveParams = @{
    FilePath         = $pythonExe
    ArgumentList     = $serveArgs
    WorkingDirectory = $backendDir
    PassThru         = $true
    NoNewWindow      = $true
}
$serveProc = Start-Process @serveParams

if (-not (Wait-ForPort -Port 5001 -TimeoutSec 120 -Label "MLflow serve")) {
    throw "MLflow serve did not start within 120 s. Check the console output above."
}

# ===========================================================================
# FASTAPI  (port 8000)
# ===========================================================================
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  FASTAPI  ->  http://127.0.0.1:8000" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$fastapiArgs = @("-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000", "--reload")

$fastapiParams = @{
    FilePath         = $pythonExe
    ArgumentList     = $fastapiArgs
    WorkingDirectory = $backendDir
    NoNewWindow      = $true
}
Start-Process @fastapiParams

if (-not (Wait-ForPort -Port 8000 -TimeoutSec 60 -Label "FastAPI")) {
    Write-Warning "FastAPI did not respond within 60 s - continuing anyway."
}

# ===========================================================================
# DRIFT ANALYSIS
# ===========================================================================
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  DRIFT ANALYSIS" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

Push-Location $backendDir
    & $pythonExe simulate_drift.py
    if ($LASTEXITCODE -ne 0) { Write-Warning "Drift analysis exited with code $LASTEXITCODE." }
Pop-Location

# ===========================================================================
# TEST API  (serve is already confirmed ready above - no extra sleep needed)
# ===========================================================================
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  TEST API" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

Push-Location $backendDir
    & $pythonExe test_api.py
Pop-Location

# ===========================================================================
# MLFLOW UI  (port 5000)
# ===========================================================================
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  MLFLOW UI  ->  http://127.0.0.1:5000" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$uiArgs = @(
    "-m", "mlflow", "ui",
    "--backend-store-uri", $mlflowDb,
    "--host", "127.0.0.1",
    "--port", "5000"
)

$uiParams = @{
    FilePath         = $pythonExe
    ArgumentList     = $uiArgs
    WorkingDirectory = $projectRoot
    NoNewWindow      = $true
}
Start-Process @uiParams

# ===========================================================================
# DONE
# ===========================================================================
Write-Host ""
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "  PIPELINE COMPLETED SUCCESSFULLY" -ForegroundColor Green
Write-Host "  FastAPI  : http://127.0.0.1:8000/docs" -ForegroundColor Green
Write-Host "  MLflow   : http://127.0.0.1:5000" -ForegroundColor Green
Write-Host "  Serve    : http://127.0.0.1:5001/invocations" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green