"""
utils/evaluation.py
===================
Computes all classification metrics needed for the platform.

Why not just accuracy?
  In churn prediction, False Negatives (predicting "No Churn" when the
  customer WILL churn) are costly. Therefore:
    - Recall    → minimizes false negatives
    - F1-Score  → balances precision and recall
    - ROC-AUC   → model's overall discrimination ability
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
from typing import Tuple


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Compute full evaluation suite.

    Parameters
    ----------
    y_true : ground truth labels
    y_pred : predicted class labels (0/1)
    y_prob : predicted probability of class 1 (for ROC)

    Returns
    -------
    dict matching ClassificationMetrics schema
    """
    cm = confusion_matrix(y_true, y_pred).tolist()

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    return {
        "accuracy":         round(accuracy_score(y_true, y_pred), 4),
        "precision":        round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":           round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score":         round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc":          round(roc_auc_score(y_true, y_prob), 4),
        "confusion_matrix": cm,
        "roc_curve": {
            "fpr":        fpr.tolist(),
            "tpr":        tpr.tolist(),
            "thresholds": thresholds.tolist(),
        },
    }
