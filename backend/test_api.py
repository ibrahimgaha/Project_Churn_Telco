import sys
import os

# FIX: insert backend/ into path idempotently
_backend = os.path.join(os.path.dirname(os.path.abspath(__file__)))
if _backend not in sys.path:
    sys.path.insert(0, _backend)

import json
import time
import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SERVE_URL = "http://127.0.0.1:5001/invocations"
MAX_RETRIES = 10          # total attempts
INITIAL_WAIT = 2          # seconds before first retry
BACKOFF_FACTOR = 1.5      # multiply wait by this each retry


def load_sample_data() -> pd.DataFrame:
    """Try several data locations; fall back to mock data."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "data", "telco_churn.csv"),
        os.path.join(os.path.dirname(__file__), "..", "telco_churn_cleaned.csv"),
        os.path.join(os.path.dirname(__file__), "..", "data", "telco_churn.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            print(f"  Loading data from: {path}")
            return pd.read_csv(path)

    print("  No CSV found – using mock data.")
    return pd.DataFrame([{
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
        "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85, "TotalCharges": 29.85,
    }])


def clean_record(record: dict) -> dict:
    """Convert numpy types to native Python types for JSON serialisation."""
    cleaned = {}
    for k, v in record.items():
        if isinstance(v, (np.integer,)):
            cleaned[k] = int(v)
        elif isinstance(v, (np.floating,)):
            cleaned[k] = float(v)
        elif pd.isna(v):
            cleaned[k] = None
        else:
            cleaned[k] = v
    return cleaned


def test_prediction():
    df = load_sample_data()
    df.drop(columns=["Churn", "customerID"], errors="ignore", inplace=True)
    sample_records = [clean_record(r) for r in df.iloc[:3].to_dict(orient="records")]

    payload = {"dataframe_records": sample_records}
    headers = {"Content-Type": "application/json"}

    print("=" * 60)
    print("Testing MLflow REST serving endpoint…")
    print(f"  URL         : {SERVE_URL}")
    print(f"  Sample row  : {sample_records[0]}")
    print("=" * 60)

    # FIX: retry loop with back-off
    wait = INITIAL_WAIT
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(SERVE_URL, json=payload, headers=headers, timeout=10)
            if resp.status_code == 200:
                print(f"\n✅  SUCCESS  (attempt {attempt})")
                print(f"  Status      : {resp.status_code}")
                print(f"  Predictions : {resp.json()}")
                return
            else:
                print(f"  Attempt {attempt}: HTTP {resp.status_code} – {resp.text[:200]}")
        except requests.exceptions.ConnectionError:
            print(f"  Attempt {attempt}: connection refused – server not ready yet.")
        except requests.exceptions.Timeout:
            print(f"  Attempt {attempt}: request timed out.")

        if attempt < MAX_RETRIES:
            print(f"  Retrying in {wait:.0f}s…")
            time.sleep(wait)
            wait = min(wait * BACKOFF_FACTOR, 20)

    print("\n❌  FAILED after all retries.")
    print("  Make sure MLflow serve is running:")
    print("    python -m mlflow models serve -m models:/churn_model/Production --port 5001 --env-manager local")


if __name__ == "__main__":
    test_prediction()
