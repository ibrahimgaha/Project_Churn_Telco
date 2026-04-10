"""
schemas/schemas.py
==================
All Pydantic models (request bodies & response shapes).
These act as the contract between frontend and backend.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


# ─── Hyperparameter schemas per algorithm ────────────────────────────────────

class LogisticRegressionParams(BaseModel):
    """
    Logistic Regression is optimized via Gradient Descent.
    solver='saga' supports SGD-like updates; 'lbfgs' uses L-BFGS (quasi-Newton).
    We expose C (inverse regularization strength) and max_iter.
    """
    C: float = Field(1.0, gt=0, description="Inverse regularization strength")
    max_iter: int = Field(200, ge=10, le=2000)
    solver: str = Field("saga", description="'saga' = SGD-style gradient descent")
    penalty: str = Field("l2")


class DecisionTreeParams(BaseModel):
    max_depth: Optional[int] = Field(None, ge=1, le=50)
    min_samples_split: int = Field(2, ge=2)
    min_samples_leaf: int = Field(1, ge=1)
    criterion: str = Field("gini", description="'gini' or 'entropy'")


class SVMParams(BaseModel):
    C: float = Field(1.0, gt=0)
    kernel: str = Field("rbf", description="'linear', 'rbf', 'poly'")
    gamma: str = Field("scale")
    max_iter: int = Field(1000)


class KNNParams(BaseModel):
    """K-Nearest Neighbors: classifies by majority vote of k closest points."""
    n_neighbors: int = Field(5, ge=1, le=50, description="Number of neighbours (k)")
    weights: str = Field("uniform", description="'uniform' or 'distance'")
    metric: str = Field("euclidean", description="'euclidean' or 'manhattan'")


class RandomForestParams(BaseModel):
    """Random Forest: ensemble of decision trees trained on bootstrap samples."""
    n_estimators: int = Field(100, ge=10, le=500, description="Number of trees")
    max_depth: Optional[int] = Field(None, ge=1, le=50)
    min_samples_split: int = Field(2, ge=2)
    criterion: str = Field("gini", description="'gini' or 'entropy'")


# ─── Training request ────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    models: List[str] = Field(
        ...,
        description="List of model keys: 'logistic_regression', 'decision_tree', 'svm', 'knn', 'random_forest'"
    )
    features: Optional[List[str]] = Field(
        None,
        description="Selected feature columns. None = use all."
    )
    test_size: float = Field(0.2, gt=0.05, lt=0.5)
    random_state: int = 42
    hyperparameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dict keyed by model name, value = hyperparameter dict"
    )


# ─── Metrics & result shapes ─────────────────────────────────────────────────

class ClassificationMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: List[List[int]]
    roc_curve: Dict[str, List[float]]   # fpr, tpr, thresholds


class TrainResponse(BaseModel):
    run_id: str
    model_name: str
    metrics: ClassificationMetrics
    feature_importances: Optional[Dict[str, float]] = None


class TrainBatchResponse(BaseModel):
    results: List[TrainResponse]


# ─── Prediction ──────────────────────────────────────────────────────────────

class PredictResponse(BaseModel):
    predictions: List[int]
    probabilities: List[float]
    model_used: str


# ─── Tune ────────────────────────────────────────────────────────────────────

class TuneRequest(BaseModel):
    model_name: str
    cv_folds: int = Field(5, ge=2, le=10)


class TuneResponse(BaseModel):
    model_name: str
    best_params: Dict[str, Any]
    best_cv_score: float
    run_id: str


# ─── Model registry ──────────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    name: str
    run_id: str
    metrics: Dict[str, float]
    timestamp: str
    is_latest: bool
