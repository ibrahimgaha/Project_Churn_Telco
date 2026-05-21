"""
utils/check_accuracy.py
=======================
Invoked by the Git pre-commit hook to verify that the latest trained model
meets the minimum accuracy threshold (0.80) before allowing commits.
"""

import os
import sys
import mlflow
from mlflow.tracking import MlflowClient

def verify_latest_model_accuracy():
    # Resolve the absolute path to the local SQLite database
    DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "mlflow.db"))
    mlflow.set_tracking_uri(f"sqlite:///{DB_PATH.replace(os.sep, '/')}")

    client = MlflowClient()
    
    # 1. Fetch experiment
    experiment = client.get_experiment_by_name("churn_prediction")
    if not experiment:
        print("❌ Git Pre-Commit Hook: No experiment 'churn_prediction' found. Train a model first!")
        sys.exit(1)

    # 2. Get latest run
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
    )
    if not runs:
        print("❌ Git Pre-Commit Hook: No runs found in MLflow database. Train a model first!")
        sys.exit(1)

    latest_run = runs[0]
    accuracy = latest_run.data.metrics.get("accuracy", 0.0)
    model_name = latest_run.data.params.get("model", "unknown")
    run_id = latest_run.info.run_id

    # 3. Check performance threshold
    threshold = 0.80
    print(f"🔍 Git Hook Check: Validating latest run {run_id} ({model_name})")
    print(f"   Accuracy: {accuracy:.4f} | Minimum required: {threshold}")

    if accuracy >= threshold:
        print("✅ Git Pre-Commit Hook: Model accuracy meets requirements. Commit allowed! 🎉")
        sys.exit(0)
    else:
        print(f"❌ Git Pre-Commit Hook: Accuracy ({accuracy:.4f}) is below threshold ({threshold})!")
        print("   Commit blocked! Retrain your model with better hyperparameters to commit your code.")
        sys.exit(1)

if __name__ == "__main__":
    verify_latest_model_accuracy()
