"""
services/registry_service.py
============================
Handles Model Registry operations.

FIX NOTES (applied):
  1. After mlflow.register_model() we poll until the version reaches
     READY state before transitioning stages – the old code tried to
     transition immediately, which races against the async registration
     worker and silently leaves the model in NONE/PENDING stage.
  2. client.transition_model_version_stage() is now wrapped in a helper
     that retries on MlflowException (transient DB locks on Windows).
  3. import of services.ml_service is kept so the shared tracking URI
     is set before any MlflowClient calls.
  4. get_registered_models_info() now also returns "None" stage versions
     so newly registered (untagged) versions are visible in the UI.
"""

import os
import time
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

import services.ml_service  # ensures tracking URI is configured


def _wait_for_model_ready(client: MlflowClient, model_name: str, version: str,
                           timeout: int = 60) -> None:
    """Poll until model version status == READY (or timeout)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        mv = client.get_model_version(model_name, version)
        if mv.status == "READY":
            return
        print(f"  [registry] version {version} status={mv.status} – waiting…")
        time.sleep(2)
    raise TimeoutError(
        f"Model version {version} did not reach READY within {timeout}s."
    )


def _transition(client: MlflowClient, model_name: str, version: str,
                stage: str, retries: int = 3) -> None:
    """Transition stage with retry on transient Windows SQLite lock errors."""
    for attempt in range(1, retries + 1):
        try:
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=True,   # FIX: archive old Production/Staging so only one is active
            )
            return
        except MlflowException as exc:
            if attempt == retries:
                raise
            print(f"  [registry] transition attempt {attempt} failed ({exc}), retrying…")
            time.sleep(3)


def register_best_model_pipeline() -> dict:
    """
    Finds the best run by accuracy, registers it, tags it,
    transitions to Staging and optionally to Production.
    """
    client = MlflowClient()

    # 1. Get experiment
    experiment = client.get_experiment_by_name("churn_prediction")
    if not experiment:
        return {
            "status": "error",
            "message": "No experiment found. Train at least one model first.",
        }

    # 2. Search runs ordered by accuracy descending
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",       # FIX: only consider finished runs
        order_by=["metrics.accuracy DESC"],
    )
    if not runs:
        return {
            "status": "error",
            "message": "No finished runs found. Train a model first.",
        }

    best_run        = runs[0]
    best_run_id     = best_run.info.run_id
    best_accuracy   = best_run.data.metrics.get("accuracy", 0.0)
    best_model_name = best_run.data.params.get("model", "unknown")

    model_uri  = f"runs:/{best_run_id}/model"
    model_name = "churn_model"

    # 3. Register model – FIX: this is synchronous but the DB write is async;
    #    we must wait for READY before touching stage.
    print(f"[registry] Registering '{best_model_name}' (acc={best_accuracy:.4f}) run={best_run_id}")
    mv      = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = mv.version

    # FIX: wait until MLflow backend has written the version record
    _wait_for_model_ready(client, model_name, version)

    # 4. Descriptions
    client.update_registered_model(
        name=model_name,
        description="Official model for Telco Customer Churn Prediction (MLflow Registry).",
    )
    client.update_model_version(
        name=model_name,
        version=version,
        description=(
            f"Version {version} – '{best_model_name}' | "
            f"Best Accuracy: {best_accuracy:.4f}"
        ),
    )

    # 5. Tags
    for key, value in [
        ("accuracy",  str(round(best_accuracy, 4))),
        ("algorithm", best_model_name),
        ("run_id",    best_run_id),
    ]:
        client.set_model_version_tag(name=model_name, version=version, key=key, value=value)

    # 6. Transition to Staging first
    _transition(client, model_name, version, "Staging")

    # 7. Promote to Production if accuracy threshold met
    threshold = 0.75
    promoted  = best_accuracy >= threshold

    if promoted:
        _transition(client, model_name, version, "Production")
        status_msg = (
            f"Version {version} accuracy={best_accuracy:.4f} ≥ {threshold} "
            f"→ promoted to Production 🚀"
        )
    else:
        status_msg = (
            f"Version {version} accuracy={best_accuracy:.4f} < {threshold} "
            f"→ kept in Staging ⚠️"
        )

    print(f"[registry] {status_msg}")

    return {
        "status":                "success",
        "model_name":            model_name,
        "version":               int(version),
        "best_run_id":           best_run_id,
        "algorithm":             best_model_name,
        "accuracy":              best_accuracy,
        "stage":                 "Production" if promoted else "Staging",
        "promoted_to_production": promoted,
        "message":               status_msg,
    }


def get_registered_models_info() -> list:
    """Fetch registered model versions (all stages) for UI display."""
    client = MlflowClient()
    try:
        # FIX: include "None" stage so freshly registered versions appear
        versions = client.get_latest_versions(
            "churn_model",
            stages=["None", "Staging", "Production"],
        )
        return [
            {
                "version":     int(v.version),
                "stage":       v.current_stage,
                "run_id":      v.run_id,
                "description": v.description,
                "tags":        v.tags,
                "timestamp":   v.creation_timestamp,
            }
            for v in versions
        ]
    except Exception:
        return []
