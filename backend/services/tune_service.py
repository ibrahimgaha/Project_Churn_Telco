"""
services/tune_service.py
========================
Automatic hyperparameter tuning using GridSearchCV.

GridSearchCV performs exhaustive cross-validated search over a param grid.
We optimize for F1-score (better suited for imbalanced churn data than accuracy).
"""

import mlflow
import mlflow.sklearn
from datetime import datetime
from sklearn.model_selection import GridSearchCV

from utils.preprocessing import load_dataset, prepare_data
from services.ml_service import build_model, MODEL_DIR, MLFLOW_URI
import joblib, os

# Predefined search grids per model
PARAM_GRIDS = {
    "logistic_regression": {
        "C":        [0.01, 0.1, 1.0, 10.0],
        "penalty":  ["l1", "l2"],
        "solver":   ["saga"],
        "max_iter": [300],
    },
    "decision_tree": {
        "max_depth":         [3, 5, 8, None],
        "min_samples_split": [2, 5, 10],
        "criterion":         ["gini", "entropy"],
    },
    "svm": {
        "C":      [0.1, 1.0, 10.0],
        "kernel": ["linear", "rbf"],
        "gamma":  ["scale", "auto"],
    },
    "knn": {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights":     ["uniform", "distance"],
        "metric":      ["euclidean", "manhattan"],
    },
    "random_forest": {
        "n_estimators":     [50, 100, 200],
        "max_depth":        [5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
        "criterion":        ["gini", "entropy"],
    },
}


def tune_model(model_name: str, cv_folds: int = 5) -> dict:
    """
    Run GridSearchCV for the given model.
    Scoring is F1 to account for class imbalance in churn datasets.
    """
    df = load_dataset()
    data = prepare_data(df, model_type=model_name)

    base_clf = build_model(model_name, {})
    param_grid = PARAM_GRIDS.get(model_name, {})

    grid_search = GridSearchCV(
        estimator=base_clf,
        param_grid=param_grid,
        cv=cv_folds,
        scoring="f1",          # optimize recall-precision balance
        n_jobs=-1,
        verbose=0,
        refit=True,
    )

    grid_search.fit(data["X_train"], data["y_train"])

    best_params = grid_search.best_params_
    best_score  = round(grid_search.best_score_, 4)

    # Log best run to MLflow
    with mlflow.start_run(run_name=f"tuned_{model_name}_{datetime.now().strftime('%H%M%S')}") as run:
        run_id = run.info.run_id
        mlflow.log_params({**best_params, "model": model_name, "tuned": True})
        mlflow.log_metric("best_cv_f1", best_score)
        mlflow.sklearn.log_model(grid_search.best_estimator_, "model")

    # Save best estimator
    model_path = os.path.join(MODEL_DIR, f"{model_name}_{run_id}.joblib")
    joblib.dump(grid_search.best_estimator_, model_path)

    return {
        "model_name":    model_name,
        "best_params":   best_params,
        "best_cv_score": best_score,
        "run_id":        run_id,
    }
