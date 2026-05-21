// ─── Model Configuration ───────────────────────────────────────────────────
// Each key matches a backend model name. Includes UI metadata + param controls.

const MODEL_INFO = {
  logistic_regression: {
    label: "Logistic Regression",
    color: "#6ee7b7",
    accent: "#059669",
    tooltip:
      "Optimized via Gradient Descent (SAGA solver). Minimizes log-loss. Requires scaling. Fast & interpretable.",
    params: [
      { key: "C", label: "C (Regularization)", type: "number", default: 1.0, min: 0.001, step: 0.1 },
      { key: "max_iter", label: "Max Iterations", type: "number", default: 200, min: 50, step: 50 },
      { key: "solver", label: "Solver", type: "select", default: "saga", options: ["saga", "lbfgs"] },
      { key: "penalty", label: "Penalty", type: "select", default: "l2", options: ["l1", "l2"] },
    ],
  },
  knn: {
    label: "K-Nearest Neighbors",
    color: "#38bdf8",
    accent: "#0284c7",
    tooltip:
      "Classifies by majority vote of k nearest data points. Distance-based → needs scaling. Simple but effective.",
    params: [
      { key: "n_neighbors", label: "K (Neighbors)", type: "number", default: 5, min: 1, step: 2 },
      { key: "weights", label: "Weights", type: "select", default: "uniform", options: ["uniform", "distance"] },
      { key: "metric", label: "Distance Metric", type: "select", default: "euclidean", options: ["euclidean", "manhattan"] },
    ],
  },
  svm: {
    label: "SVM",
    color: "#c4b5fd",
    accent: "#7c3aed",
    tooltip:
      "Finds optimal hyperplane maximizing margin. Kernel trick enables non-linear boundaries. Requires scaling.",
    params: [
      { key: "C", label: "C (Margin Slack)", type: "number", default: 1.0, min: 0.001, step: 0.1 },
      { key: "kernel", label: "Kernel", type: "select", default: "rbf", options: ["rbf", "linear", "poly"] },
      { key: "gamma", label: "Gamma", type: "select", default: "scale", options: ["scale", "auto"] },
      { key: "max_iter", label: "Max Iterations", type: "number", default: 1000, min: 100, step: 100 },
    ],
  },
  decision_tree: {
    label: "Decision Tree",
    color: "#fcd34d",
    accent: "#d97706",
    tooltip:
      "Splits data recursively using Gini/Entropy. Scale-invariant. Interpretable but prone to overfitting.",
    params: [
      { key: "max_depth", label: "Max Depth", type: "number", default: 5, min: 1, step: 1 },
      { key: "min_samples_split", label: "Min Samples Split", type: "number", default: 2, min: 2, step: 1 },
      { key: "min_samples_leaf", label: "Min Samples Leaf", type: "number", default: 1, min: 1, step: 1 },
      { key: "criterion", label: "Criterion", type: "select", default: "gini", options: ["gini", "entropy"] },
    ],
  },
  random_forest: {
    label: "Random Forest",
    color: "#f9a8d4",
    accent: "#db2777",
    tooltip:
      "Ensemble of decision trees trained on bootstrap samples. Reduces overfitting. Scale-invariant. Best all-rounder.",
    params: [
      { key: "n_estimators", label: "N Trees", type: "number", default: 100, min: 10, step: 10 },
      { key: "max_depth", label: "Max Depth", type: "number", default: 10, min: 1, step: 1 },
      { key: "min_samples_split", label: "Min Samples Split", type: "number", default: 2, min: 2, step: 1 },
      { key: "criterion", label: "Criterion", type: "select", default: "gini", options: ["gini", "entropy"] },
    ],
  },
  adaboost: {
    label: "AdaBoost",
    color: "#fb923c",
    accent: "#ea580c",
    tooltip:
      "Adaptive Boosting: combines many weak learners (decision stumps) sequentially. Each round re-weights misclassified samples. Simple but effective.",
    params: [
      { key: "n_estimators", label: "N Estimators", type: "number", default: 50, min: 10, step: 10 },
      { key: "learning_rate", label: "Learning Rate", type: "number", default: 1.0, min: 0.01, step: 0.05 },
      { key: "algorithm", label: "Algorithm", type: "select", default: "SAMME", options: ["SAMME"] },
    ],
  },
  xgboost: {
    label: "XGBoost",
    color: "#f43f5e",
    accent: "#be123c",
    tooltip:
      "Extreme Gradient Boosting with L1/L2 regularization. Uses 2nd-order gradients for faster convergence. subsample + colsample act like dropout for trees.",
    params: [
      { key: "n_estimators", label: "N Rounds", type: "number", default: 100, min: 10, step: 10 },
      { key: "learning_rate", label: "Learning Rate", type: "number", default: 0.1, min: 0.01, step: 0.01 },
      { key: "max_depth", label: "Max Depth", type: "number", default: 6, min: 1, step: 1 },
      { key: "subsample", label: "Subsample", type: "number", default: 0.8, min: 0.1, step: 0.05 },
      { key: "colsample_bytree", label: "Col Sample", type: "number", default: 0.8, min: 0.1, step: 0.05 },
    ],
  },
};

export default MODEL_INFO;
