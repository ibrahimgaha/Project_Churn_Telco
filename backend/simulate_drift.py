"""
simulate_drift.py
=================
Improved Drift Detection Pipeline

Features:
- Loads Telco dataset
- Simulates realistic drift
- Runs Evidently drift analysis
- Performs KS-tests per numerical feature
- Logs everything to MLflow
- Saves HTML + CSV artifacts
- Uses configurable thresholds:
    * Stable    : drift_share <= 15%
    * Warning   : 15% < drift_share <= 30%
    * Critical  : drift_share > 30%
- Auto retraining only when CRITICAL
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import scipy.stats
import mlflow

from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
)
from evidently.metrics import DatasetDriftMetric

# ==========================================================
# PATH FIX
# ==========================================================

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from services.ml_service import (
    load_dataset,
    train_and_evaluate,
)

# ==========================================================
# CONFIG
# ==========================================================

SCRATCH_DIR = os.path.join(THIS_DIR, "scratch")

mlflow.set_experiment("churn_prediction")

# Thresholds
WARNING_THRESHOLD = 0.15
CRITICAL_THRESHOLD = 0.30

# ==========================================================
# DRIFT ANALYSIS
# ==========================================================

def run_drift_analysis():

    print("\n==================================================")
    print(" DRIFT ANALYSIS STARTED")
    print("==================================================\n")

    # ======================================================
    # LOAD DATA
    # ======================================================

    print("[1/7] Loading dataset...")

    df = load_dataset()

    half = len(df) // 2

    reference = df.iloc[:half].copy()
    current = df.iloc[half:].copy()

    # ======================================================
    # SIMULATE DRIFT
    # ======================================================

    print("[2/7] Simulating realistic drift...")

    # Mild realistic drift
    current["tenure"] = current["tenure"] * 0.85
    current["MonthlyCharges"] = current["MonthlyCharges"] * 1.10

    if "Contract" in current.columns:
        mask = np.random.rand(len(current)) < 0.20
        current.loc[mask, "Contract"] = "Month-to-month"

    os.makedirs(SCRATCH_DIR, exist_ok=True)

    drifted_csv_path = os.path.join(
        SCRATCH_DIR,
        "drifted_test_data.csv"
    )

    current.to_csv(drifted_csv_path, index=False)

    # ======================================================
    # EVIDENTLY REPORT
    # ======================================================

    print("[3/7] Running Evidently analysis...")

    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
        DatasetDriftMetric(),
    ])

    report.run(
        reference_data=reference,
        current_data=current,
    )

    report_html_path = os.path.join(
        SCRATCH_DIR,
        "drift_report.html"
    )

    report.save_html(report_html_path)

    # ======================================================
# EXTRACT DRIFT METRICS
# ======================================================

    print("[4/7] Extracting drift metrics...")

    report_dict = report.as_dict()

    drift_share = 0.0
    drifted_columns = 0
    dataset_drifted = False

    try:

        metrics = report_dict.get("metrics", [])

        for metric in metrics:

            result = metric.get("result", {})

            if "drift_share" in result:

                drift_share = result.get(
                    "drift_share",
                    0.0
                )

                drifted_columns = result.get(
                    "number_of_drifted_columns",
                    result.get(
                        "number_of_drifted_features",
                        0
                    )
                )

                dataset_drifted = result.get(
                    "dataset_drift",
                    result.get(
                        "dataset_drifted",
                        False
                    )
                )

                break

    except Exception as e:

        print(f"[WARNING] Could not parse Evidently metrics: {e}")

    # ======================================================
    # KS TESTS
    # ======================================================

    print("[5/7] Running KS-tests...")

    ks_results = []
    ks_metrics_to_log = {}

    numerical_cols = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    for col in numerical_cols:

        if col == "Churn":
            continue

        stat, pval = scipy.stats.ks_2samp(
            reference[col],
            current[col]
        )

        drift_detected = bool(pval < 0.05)

        result = {
            "feature": col,
            "ks_statistic": round(float(stat), 4),
            "p_value": round(float(pval), 6),
            "drift_detected": drift_detected,
        }

        ks_results.append(result)

        ks_metrics_to_log[f"ks_pval_{col}"] = round(
            float(pval), 6
        )

        ks_metrics_to_log[f"ks_stat_{col}"] = round(
            float(stat), 4
        )

    ks_csv_path = os.path.join(
        SCRATCH_DIR,
        "ks_drift_results.csv"
    )

    pd.DataFrame(ks_results).to_csv(
        ks_csv_path,
        index=False
    )

    # ======================================================
    # DRIFT STATUS
    # ======================================================

    print("[6/7] Evaluating thresholds...")

    retrain_triggered = False

    if drift_share > CRITICAL_THRESHOLD:

        drift_status = "CRITICAL"

        message = (
            f"CRITICAL DRIFT DETECTED "
            f"({drift_share:.2%} > 30%)"
        )

        retrain_triggered = True

    elif drift_share > WARNING_THRESHOLD:

        drift_status = "WARNING"

        message = (
            f"WARNING DRIFT DETECTED "
            f"({drift_share:.2%} > 15%)"
        )

    else:

        drift_status = "STABLE"

        message = (
            f"DATA STABLE "
            f"({drift_share:.2%} <= 15%)"
        )

    print(f"\n[STATUS] {message}")

    # ======================================================
    # MLFLOW LOGGING
    # ======================================================

    print("\n[7/7] Logging to MLflow...")

    with mlflow.start_run(
        run_name=f"Drift_Analysis_{pd.Timestamp.now().strftime('%H%M%S')}"
    ) as run:

        run_id = run.info.run_id

        mlflow.log_metrics({
            "drift_share": drift_share,
            "drifted_columns": drifted_columns,
            "dataset_drifted": int(dataset_drifted),
        })

        mlflow.log_metrics(ks_metrics_to_log)

        mlflow.log_param(
            "warning_threshold",
            WARNING_THRESHOLD
        )

        mlflow.log_param(
            "critical_threshold",
            CRITICAL_THRESHOLD
        )

        mlflow.set_tag(
            "drift_status",
            drift_status
        )

        mlflow.log_artifact(
            report_html_path,
            artifact_path="evidently_report"
        )

        mlflow.log_artifact(
            ks_csv_path,
            artifact_path="ks_tests"
        )

    # ======================================================
    # OPTIONAL RETRAINING
    # ======================================================

    if retrain_triggered:

        print("\n[ACTION] Auto retraining triggered...\n")

        train_and_evaluate(
            model_name="random_forest",
            hyperparams={
                "n_estimators": 100,
                "max_depth": 10,
            },
            selected_features=None,
            test_size=0.2,
            random_state=42,
        )

    # ======================================================
    # FINAL RESULT
    # ======================================================

    result = {
        "status": "success",
        "drift_status": drift_status,
        "drift_share": round(drift_share, 4),
        "drifted_columns": drifted_columns,
        "dataset_drifted": dataset_drifted,
        "retrain_triggered": retrain_triggered,
        "message": message,
        "run_id": run_id,
        "evidently_report_path": os.path.abspath(
            report_html_path
        ),
        "ks_results_path": os.path.abspath(
            ks_csv_path
        ),
    }

    print("\n==================================================")
    print(" DRIFT ANALYSIS COMPLETED")
    print("==================================================\n")

    print(json.dumps(result, indent=2))

    return result


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    run_drift_analysis()