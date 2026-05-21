# Makefile for Project Churn Telco MLOps Pipeline

# Executive Paths
PYTHON = .\venv\Scripts\python.exe
PIP = .\venv\Scripts\pip.exe
UVICORN = .\venv\Scripts\uvicorn.exe

.PHONY: setup train register serve drift test pipeline

setup:
	@echo "=========================================================="
	@echo "🛠️ MLOps Setup: Installing Python and Node dependencies"
	@echo "=========================================================="
	$(PIP) install -r backend/requirements.txt
	cd frontend && npm install

train:
	@echo "=========================================================="
	@echo "🏋️ MLOps Training: Training Random Forest and Decision Tree"
	@echo "=========================================================="
	$(PYTHON) -c "import services.ml_service; services.ml_service.train_and_evaluate('random_forest', {'n_estimators': 150, 'max_depth': 12}, None, 0.2, 42)"
	$(PYTHON) -c "import services.ml_service; services.ml_service.train_and_evaluate('decision_tree', {'max_depth': 8}, None, 0.2, 42)"

register:
	@echo "=========================================================="
	@echo "📁 MLOps Registry: Registering and Promoting Best Model"
	@echo "=========================================================="
	$(PYTHON) -c "import services.registry_service; print(services.registry_service.register_best_model_pipeline())"

serve:
	@echo "=========================================================="
	@echo "🚀 MLOps Serve: Launching FastAPI Application Server"
	@echo "=========================================================="
	cd backend && $(UVICORN) main:app --reload

drift:
	@echo "=========================================================="
	@echo "📊 MLOps Drift: Running Data Drift & KS-test Simulation"
	@echo "=========================================================="
	$(PYTHON) backend/simulate_drift.py

test:
	@echo "=========================================================="
	@echo "🧪 MLOps Test: Validating MLflow Serving API Invocations"
	@echo "=========================================================="
	$(PYTHON) backend/test_api.py

pipeline:
	@echo "=========================================================="
	@echo "🏁 MLOps Pipeline: Running End-to-End lifecycle on Windows"
	@echo "=========================================================="
	powershell -ExecutionPolicy Bypass -File pipeline.ps1
