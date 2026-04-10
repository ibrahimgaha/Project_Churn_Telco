"""
utils/preprocessing.py
=======================
Handles all data preprocessing steps:
  1. Load and validate the Telco CSV
  2. Encode categorical features  
  3. Scale numeric features (required for LR and SVM, not tree-based)
  4. Train / test split

Design note:
  StandardScaler is only applied for Logistic Regression and SVM because
  those models are distance/gradient sensitive.
  Decision Trees are scale-invariant (splits are threshold-based).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional
import os

# Path to the default Telco dataset
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "telco_churn.csv")

# Models that require feature scaling
SCALE_REQUIRED = {"logistic_regression", "svm", "knn"}


def load_dataset(path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load CSV and perform minimal sanity checks."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # Standard Telco fix: TotalCharges may be whitespace strings
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Drop non-informative columns
    drop_cols = ["customerID"] if "customerID" in df.columns else []
    df.drop(columns=drop_cols, inplace=True)

    return df


def encode_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Label-encode all object columns.
    Returns transformed df and a mapping dict for interpretability.
    """
    encoders = {}
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def prepare_data(
    df: pd.DataFrame,
    target_col: str = "Churn",
    selected_features: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    model_type: str = "logistic_regression",
) -> dict:
    """
    Full preprocessing pipeline.

    Returns a dict with:
      X_train, X_test, y_train, y_test, feature_names, scaler (or None)
    """
    df_encoded, encoders = encode_features(df)

    # Feature selection
    all_features = [c for c in df_encoded.columns if c != target_col]
    features = selected_features if selected_features else all_features

    # Validate requested features exist
    features = [f for f in features if f in df_encoded.columns]

    X = df_encoded[features].values
    y = df_encoded[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale only for gradient-based / distance-based models
    scaler = None
    if model_type in SCALE_REQUIRED:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": features,
        "scaler": scaler,
        "encoders": encoders,
    }
