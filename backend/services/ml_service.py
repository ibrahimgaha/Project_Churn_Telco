"""
services/ml_service.py
======================
Core ML logic:
  - Model factory (build sklearn estimators from request params)
  - Train + evaluate a single model
  - Save / load models with joblib
  - MLflow integration for experiment tracking

"""

import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from utils.preprocessing import load_dataset, prepare_data
from utils.evaluation import compute_metrics

# ─── Paths ───────────────────────────────────────────────────────────────────
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_BACKEND_DIR, ".."))

MODEL_DIR = os.path.join(_PROJECT_ROOT, "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)  # FIX: ensure directory always exists

# FIX: build the DB path once, normalise separators for SQLite URI
DB_PATH = os.path.join(_PROJECT_ROOT, "mlflow.db")
_db_uri = "sqlite:///" + DB_PATH.replace("\\", "/")

# FIX: honour MLFLOW_TRACKING_URI env-var set by pipeline.ps1 so the
# same DB is shared across the serve subprocess and the Python process.
_env_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
if _env_uri:
    mlflow.set_tracking_uri(_env_uri)
else:
    mlflow.set_tracking_uri(_db_uri)

print(f"[ml_service] MLflow tracking URI → {mlflow.get_tracking_uri()}")

# Backward-compat alias – tune_service.py imports this name
MLFLOW_URI = mlflow.get_tracking_uri()

mlflow.set_experiment("churn_prediction")


# ─── Model factory ───────────────────────────────────────────────────────────

def build_model(model_name: str, hyperparams: Dict[str, Any]):
    if model_name == "logistic_regression":
        return LogisticRegression(
            C=hyperparams.get("C", 1.0),
            max_iter=hyperparams.get("max_iter", 200),
            solver=hyperparams.get("solver", "saga"),
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
            probability=True,
            random_state=42,
        )
    elif model_name == "knn":
        return KNeighborsClassifier(
            n_neighbors=hyperparams.get("n_neighbors", 5),
            weights=hyperparams.get("weights", "uniform"),
            metric=hyperparams.get("metric", "euclidean"),
            n_jobs=-1,
        )
    elif model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=hyperparams.get("n_estimators", 100),
            max_depth=hyperparams.get("max_depth", None),
            min_samples_split=hyperparams.get("min_samples_split", 2),
            criterion=hyperparams.get("criterion", "gini"),
            random_state=42,
            n_jobs=-1,
        )
    elif model_name == "adaboost":
        return AdaBoostClassifier(
            n_estimators=hyperparams.get("n_estimators", 50),
            learning_rate=hyperparams.get("learning_rate", 1.0),
            algorithm=hyperparams.get("algorithm", "SAMME"),
            random_state=42,
        )
    elif model_name == "xgboost":
        # FIX: removed deprecated use_label_encoder (XGBoost ≥ 2 raises TypeError)
        return XGBClassifier(
            n_estimators=hyperparams.get("n_estimators", 100),
            learning_rate=hyperparams.get("learning_rate", 0.1),
            max_depth=hyperparams.get("max_depth", 6),
            subsample=hyperparams.get("subsample", 0.8),
            colsample_bytree=hyperparams.get("colsample_bytree", 0.8),
            eval_metric="logloss",
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
    run_name = f"{model_name}_{datetime.now().strftime('%H%M%S')}"
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        mlflow.log_params({**hyperparams, "model": model_name, "test_size": test_size})

        # 4. Train
        clf.fit(data["X_train"], data["y_train"])

        # 5. Evaluate
        y_pred = clf.predict(data["X_test"])
        y_prob = clf.predict_proba(data["X_test"])[:, 1]
        metrics = compute_metrics(data["y_test"], y_pred, y_prob)

        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})

        # FIX: log model with explicit input_example so MLflow can infer
        # the signature automatically – required for `mlflow models serve`
        import pandas as pd
        input_example = pd.DataFrame(data["X_test"][:3], columns=data["feature_names"])
        mlflow.sklearn.log_model(
            clf,
            artifact_path="model",
            input_example=input_example,
        )

    # 6. Persist locally
    model_path  = os.path.join(MODEL_DIR, f"{model_name}_{run_id}.joblib")
    scaler_path = os.path.join(MODEL_DIR, f"{model_name}_{run_id}_scaler.joblib")
    joblib.dump(clf, model_path)
    if data["scaler"]:
        joblib.dump(data["scaler"], scaler_path)

    # 7. Feature importances
    feature_importances = None
    if hasattr(clf, "feature_importances_"):
        feature_importances = dict(
            zip(data["feature_names"], clf.feature_importances_.tolist())
        )
    elif hasattr(clf, "coef_"):
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
    model_path  = os.path.join(MODEL_DIR, f"{model_name}_{run_id}.joblib")
    scaler_path = os.path.join(MODEL_DIR, f"{model_name}_{run_id}_scaler.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No saved model for run_id={run_id}")

    clf    = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    return clf, scaler


def get_all_runs() -> list:
    client     = mlflow.tracking.MlflowClient()
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