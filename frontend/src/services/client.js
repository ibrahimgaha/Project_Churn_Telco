// src/api/client.js
// =================
// Centralized API communication layer.
// All fetch calls to FastAPI go through here.

const BASE_URL = "http://localhost:8000";

async function request(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, options);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "API Error");
  }
  return res.json();
}

// ─── Train ────────────────────────────────────────────────────────────────
export async function trainModels({ models, features, testSize, hyperparameters }) {
  return request("/train", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      models,
      features: features.length ? features : null,
      test_size: testSize,
      random_state: 42,
      hyperparameters,
    }),
  });
}

// ─── Auto Tune ────────────────────────────────────────────────────────────
export async function autoTune(modelName, cvFolds = 5) {
  return request("/tune", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_name: modelName, cv_folds: cvFolds }),
  });
}

// ─── Get all models / runs ────────────────────────────────────────────────
export async function fetchModels() {
  return request("/models");
}

// ─── Get results ─────────────────────────────────────────────────────────
export async function fetchResults() {
  return request("/results");
}

// ─── Predict from CSV ────────────────────────────────────────────────────
export async function predictFromCSV(file, modelName, runId) {
  const form = new FormData();
  form.append("file", file);
  form.append("model_name", modelName);
  form.append("run_id", runId);
  return request("/predict", { method: "POST", body: form });
}
