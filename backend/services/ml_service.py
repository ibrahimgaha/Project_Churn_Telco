"""
services/ml_service.py
======================
Core ML logic:
  - Model factory (build sklearn estimators from request params)
  - Train + evaluate a single model
  - Save / load models with joblib
  - MLflow integration for experiment tracking

Gradient Descent note:
  Logistic Regression minimises the log-loss (cross-entropy) cost function.
  sklearn's 'saga' solver implements Stochastic Average Gradient Descent —
  a variance-reduced gradient descent algorithm.
  With solver='lbfgs', it uses L-BFGS (Limited-memory BFGS), a quasi-Newton
  method that approximates the Hessian for faster convergence.
  Either way: gradient descent is the OPTIMIZATION ALGORITHM, not the model.
"""

import os
import uuid
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from utils.preprocessing import load_dataset, prepare_data
from utils.evaluation import compute_metrics

# ─── Paths ───────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_models")
MLFLOW_URI = os.path.join(os.path.dirname(__file__), "..", "mlruns_store")
MLRUNS_DIR = os.path.join(os.getcwd(), "mlruns")

# ✅ Use file:// scheme (Windows-safe)
mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR.replace(os.sep, '/')}")

# ✅ Disable model registry (student-level MLOps)
mlflow.set_registry_uri("")

mlflow.set_experiment("churn_prediction")


# ─── Model factory ───────────────────────────────────────────────────────────

def build_model(model_name: str, hyperparams: Dict[str, Any]):
    """
    Factory function: returns an sklearn estimator configured with the
    provided hyperparameters.

    All three models expose predict_proba so ROC-AUC can be computed.
    SVC requires probability=True for predict_proba.
    """
    if model_name == "logistic_regression":
        return LogisticRegression(
            C=hyperparams.get("C", 1.0),
            max_iter=hyperparams.get("max_iter", 200),
            solver=hyperparams.get("solver", "saga"),   # saga = SGD-style GD
            penalty=hyperparams.get("penalty", "l2"),
            random_state=42,
            n_jobs=-1,
        )
    elif model_name == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=hyperparams.get("max_depth", None),
            min_samples_split=hyperparams.get("min_samples_split", 2),
            min_samples_leaf=hyperparams.get("min_samples_leaf", 1),
            criterion=hyperparams.get("criterion", "gini"),
            random_state=42,
        )
    elif model_name == "svm":
        return SVC(
            C=hyperparams.get("C", 1.0),
            kernel=hyperparams.get("kernel", "rbf"),
            gamma=hyperparams.get("gamma", "scale"),
            max_iter=hyperparams.get("max_iter", 1000),
            probability=True,   # enables predict_proba
            random_state=42,
        )
    elif model_name == "knn":
        # KNN: classifies by majority vote among k nearest neighbours.
        # Distance-based → requires scaled features.
        return KNeighborsClassifier(
            n_neighbors=hyperparams.get("n_neighbors", 5),
            weights=hyperparams.get("weights", "uniform"),
            metric=hyperparams.get("metric", "euclidean"),
            n_jobs=-1,
        )
    elif model_name == "random_forest":
        # Random Forest: ensemble of decision trees with bagging.
        # Reduces overfitting vs single tree. Scale-invariant.
        return RandomForestClassifier(
            n_estimators=hyperparams.get("n_estimators", 100),
            max_depth=hyperparams.get("max_depth", None),
            min_samples_split=hyperparams.get("min_samples_split", 2),
            criterion=hyperparams.get("criterion", "gini"),
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ─── Train & evaluate ────────────────────────────────────────────────────────

def train_and_evaluate(
    model_name: str,
    hyperparams: Dict[str, Any],
    selected_features: Optional[list],
    test_size: float,
    random_state: int,
) -> dict:
    """
    Full pipeline:
      load → preprocess → train → evaluate → persist → log to MLflow
    """
    # 1. Data
    df = load_dataset()
    data = prepare_data(
        df,
        selected_features=selected_features,
        test_size=test_size,
        random_state=random_state,
        model_type=model_name,
    )

    # 2. Build estimator
    clf = build_model(model_name, hyperparams)

    # 3. MLflow run
    with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%H%M%S')}") as run:
        run_id = run.info.run_id

        # Log hyperparameters
        mlflow.log_params({**hyperparams, "model": model_name, "test_size": test_size})

        # 4. Train
        clf.fit(data["X_train"], data["y_train"])

        # 5. Evaluate
        y_pred = clf.predict(data["X_test"])
        y_prob = clf.predict_proba(data["X_test"])[:, 1]
        metrics = compute_metrics(data["y_test"], y_pred, y_prob)

        # Log scalar metrics to MLflow
        mlflow.log_metrics({
            k: v for k, v in metrics.items()
            if isinstance(v, float)
        })

        # Log model artifact
        mlflow.sklearn.log_model(clf, artifact_path="model")

    # 6. Persist locally
    model_path = os.path.join(MODEL_DIR, f"{model_name}_{run_id}.joblib")
    scaler_path = os.path.join(MODEL_DIR, f"{model_name}_{run_id}_scaler.joblib")
    joblib.dump(clf, model_path)
    if data["scaler"]:
        joblib.dump(data["scaler"], scaler_path)

    # 7. Feature importances (available for Decision Tree)
    feature_importances = None
    if hasattr(clf, "feature_importances_"):
        feature_importances = dict(
            zip(data["feature_names"], clf.feature_importances_.tolist())
        )
    elif hasattr(clf, "coef_"):
        # Logistic Regression: use absolute coefficient magnitude
        feature_importances = dict(
            zip(data["feature_names"], np.abs(clf.coef_[0]).tolist())
        )

    return {
        "run_id": run_id,
        "model_name": model_name,
        "metrics": metrics,
        "feature_importances": feature_importances,
    }


# ─── Load model for prediction ───────────────────────────────────────────────

def load_model_by_run_id(model_name: str, run_id: str):
    """Load a previously saved model + its scaler."""
    model_path = os.path.join(MODEL_DIR, f"{model_name}_{run_id}.joblib")
    scaler_path = os.path.join(MODEL_DIR, f"{model_name}_{run_id}_scaler.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No saved model for run_id={run_id}")

    clf = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    return clf, scaler


def get_all_runs() -> list:
    """Return all MLflow runs sorted by start time desc."""
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("churn_prediction")
    if not experiment:
        return []

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
    )

    results = []
    for r in runs:
        results.append({
            "run_id":    r.info.run_id,
            "model":     r.data.params.get("model", "unknown"),
            "metrics":   r.data.metrics,
            "params":    r.data.params,
            "timestamp": datetime.fromtimestamp(r.info.start_time / 1000).isoformat(),
        })
    return results
