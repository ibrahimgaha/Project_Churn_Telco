// ─── API Client ────────────────────────────────────────────────────────────
// Centralised HTTP helpers for talking to the FastAPI backend.

const BASE = "http://localhost:8000";

async function api(path, opts = {}) {
  const r = await fetch(`${BASE}${path}`, opts);
  if (!r.ok) {
    const e = await r.json().catch(() => ({}));
    throw new Error(e.detail || r.statusText);
  }
  return r.json();
}

/** Train one or more models */
export const trainModels = (body) =>
  api("/train", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

/** Run GridSearchCV auto-tuning */
export const autoTune = (model_name, cv_folds = 5) =>
  api("/tune", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_name, cv_folds }),
  });

/** Fetch all MLflow runs */
export const fetchModels = () => api("/models");

/** Get the single best model run */
export const fetchBestModel = () => api("/models/best");

/** Upload CSV for prediction */
export const predictCSV = (file, model_name, run_id) => {
  const f = new FormData();
  f.append("file", file);
  f.append("model_name", model_name);
  f.append("run_id", run_id);
  return api("/predict", { method: "POST", body: f });
};

/** Fetch PCA 2D projection */
export const fetchPCA = () => api("/visualize/pca");

/** Fetch t-SNE 2D projection */
export const fetchTSNE = (perplexity = 30) =>
  api(`/visualize/tsne?perplexity=${perplexity}`);

/** Launch the MLflow UI background process */
export const launchMLflowUI = () =>
  api("/mlflow/launch", { method: "POST" });
