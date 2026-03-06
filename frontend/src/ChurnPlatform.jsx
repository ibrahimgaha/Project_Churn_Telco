import { useState, useCallback } from "react";

// ─── API client ─────────────────────────────────────────────────────────────
const BASE = "http://localhost:8000";
async function api(path, opts = {}) {
  const r = await fetch(`${BASE}${path}`, opts);
  if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(e.detail || r.statusText); }
  return r.json();
}
const trainModels = (body) => api("/train", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
const autoTune = (model_name, cv_folds = 5) => api("/tune", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model_name, cv_folds }) });
const fetchModels = () => api("/models");
const predictCSV = (file, model_name, run_id) => { const f = new FormData(); f.append("file", file); f.append("model_name", model_name); f.append("run_id", run_id); return api("/predict", { method: "POST", body: f }); };

// ─── Constants ───────────────────────────────────────────────────────────────
const MODEL_INFO = {
  logistic_regression: {
    label: "Logistic Regression",
    color: "#6ee7b7",
    accent: "#059669",
    tooltip: "Optimized via Gradient Descent (SAGA solver — Stochastic Average Gradient). Minimizes log-loss cost function. Requires feature scaling. Fast and interpretable.",
    params: [
      { key: "C", label: "C (Regularization)", type: "number", default: 1.0, min: 0.001, step: 0.1 },
      { key: "max_iter", label: "Max Iterations", type: "number", default: 200, min: 50, step: 50 },
      { key: "solver", label: "Solver (GD variant)", type: "select", default: "saga", options: ["saga", "lbfgs"] },
      { key: "penalty", label: "Penalty", type: "select", default: "l2", options: ["l1", "l2"] },
    ],
  },
  decision_tree: {
    label: "Decision Tree",
    color: "#fcd34d",
    accent: "#d97706",
    tooltip: "Splits data recursively using Gini impurity or Entropy. Scale-invariant. Highly interpretable. Prone to overfitting without depth constraints.",
    params: [
      { key: "max_depth", label: "Max Depth (null = unlimited)", type: "number", default: 5, min: 1, step: 1 },
      { key: "min_samples_split", label: "Min Samples Split", type: "number", default: 2, min: 2, step: 1 },
      { key: "min_samples_leaf", label: "Min Samples Leaf", type: "number", default: 1, min: 1, step: 1 },
      { key: "criterion", label: "Criterion", type: "select", default: "gini", options: ["gini", "entropy"] },
    ],
  },
  svm: {
    label: "SVM",
    color: "#c4b5fd",
    accent: "#7c3aed",
    tooltip: "Finds the optimal hyperplane maximizing margin between classes. Kernel trick enables non-linear boundaries. Requires scaling. Best for high-dimensional data.",
    params: [
      { key: "C", label: "C (Margin Slack)", type: "number", default: 1.0, min: 0.001, step: 0.1 },
      { key: "kernel", label: "Kernel", type: "select", default: "rbf", options: ["rbf", "linear", "poly"] },
      { key: "gamma", label: "Gamma", type: "select", default: "scale", options: ["scale", "auto"] },
      { key: "max_iter", label: "Max Iterations", type: "number", default: 1000, min: 100, step: 100 },
    ],
  },
};

const DEMO_DATASET = [
  { customerID: "1001", tenure: 12, MonthlyCharges: 65.4, TotalCharges: 784.8, Contract: "Month-to-month", InternetService: "Fiber optic", Churn: "Yes" },
  { customerID: "1002", tenure: 48, MonthlyCharges: 45.2, TotalCharges: 2169.6, Contract: "Two year", InternetService: "DSL", Churn: "No" },
  { customerID: "1003", tenure: 3, MonthlyCharges: 89.1, TotalCharges: 267.3, Contract: "Month-to-month", InternetService: "Fiber optic", Churn: "Yes" },
  { customerID: "1004", tenure: 72, MonthlyCharges: 20.0, TotalCharges: 1440.0, Contract: "Two year", InternetService: "No", Churn: "No" },
  { customerID: "1005", tenure: 7, MonthlyCharges: 77.5, TotalCharges: 542.5, Contract: "Month-to-month", InternetService: "Fiber optic", Churn: "Yes" },
];

// ─── Sub-components ──────────────────────────────────────────────────────────

function Tag({ children, color }) {
  return (
    <span style={{ background: color + "22", color, border: `1px solid ${color}44`, borderRadius: 4, padding: "2px 8px", fontSize: 11, fontWeight: 700, letterSpacing: 1 }}>
      {children}
    </span>
  );
}

function Tooltip({ text }) {
  const [show, setShow] = useState(false);
  return (
    <span style={{ position: "relative", display: "inline-block", marginLeft: 6 }}>
      <span
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
        style={{ cursor: "help", fontSize: 12, color: "#64748b", border: "1px solid #334155", borderRadius: "50%", width: 18, height: 18, display: "inline-flex", alignItems: "center", justifyContent: "center", fontWeight: 700 }}
      >?</span>
      {show && (
        <div style={{ position: "absolute", bottom: "130%", left: "50%", transform: "translateX(-50%)", background: "#0f172a", color: "#e2e8f0", border: "1px solid #334155", borderRadius: 8, padding: "10px 14px", width: 280, fontSize: 12, lineHeight: 1.6, zIndex: 100, boxShadow: "0 8px 32px #00000080" }}>
          {text}
        </div>
      )}
    </span>
  );
}

function MetricCard({ label, value, color }) {
  return (
    <div style={{ background: "#0f172a", border: `1px solid ${color}44`, borderRadius: 10, padding: "16px 20px", textAlign: "center", flex: 1 }}>
      <div style={{ fontSize: 11, color: "#64748b", marginBottom: 6, letterSpacing: 1, textTransform: "uppercase" }}>{label}</div>
      <div style={{ fontSize: 28, fontWeight: 800, color, fontFamily: "monospace" }}>
        {(value * 100).toFixed(1)}<span style={{ fontSize: 14 }}>%</span>
      </div>
    </div>
  );
}

function ConfusionMatrix({ cm }) {
  if (!cm) return null;
  const [[tn, fp], [fn, tp]] = cm;
  const cells = [
    { label: "TN", val: tn, color: "#6ee7b7", desc: "True Negative" },
    { label: "FP", val: fp, color: "#fca5a5", desc: "False Positive" },
    { label: "FN", val: fn, color: "#fcd34d", desc: "False Negative" },
    { label: "TP", val: tp, color: "#a5b4fc", desc: "True Positive" },
  ];
  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, maxWidth: 280 }}>
        {cells.map(c => (
          <div key={c.label} style={{ background: c.color + "18", border: `1px solid ${c.color}55`, borderRadius: 8, padding: "14px", textAlign: "center" }}>
            <div style={{ fontSize: 10, color: "#64748b", marginBottom: 4 }}>{c.desc}</div>
            <div style={{ fontSize: 32, fontWeight: 900, color: c.color, fontFamily: "monospace" }}>{c.val}</div>
            <Tag color={c.color}>{c.label}</Tag>
          </div>
        ))}
      </div>
      <div style={{ marginTop: 10, fontSize: 11, color: "#475569" }}>
        ⚠ FN (False Negatives) = predicted No Churn but will churn → highest business cost
      </div>
    </div>
  );
}

function SimpleROC({ roc_curve: roc }) {
  if (!roc) return null;
  const W = 300, H = 220, pad = 36;
  const fw = W - pad * 2, fh = H - pad * 2;
  const pts = roc.fpr.map((x, i) => [pad + x * fw, pad + fh - roc.tpr[i] * fh]);
  const d = pts.map((p, i) => `${i === 0 ? "M" : "L"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");
  return (
    <svg width={W} height={H} style={{ background: "#0f172a", borderRadius: 8, border: "1px solid #1e293b" }}>
      <line x1={pad} y1={pad} x2={pad} y2={pad + fh} stroke="#334155" strokeWidth={1} />
      <line x1={pad} y1={pad + fh} x2={pad + fw} y2={pad + fh} stroke="#334155" strokeWidth={1} />
      <line x1={pad} y1={pad + fh} x2={pad + fw} y2={pad} stroke="#1e293b" strokeWidth={1} strokeDasharray="4,4" />
      <path d={d} fill="none" stroke="#6ee7b7" strokeWidth={2.5} />
      <text x={pad + fw / 2} y={H - 6} fill="#475569" fontSize={10} textAnchor="middle">False Positive Rate</text>
      <text x={10} y={pad + fh / 2} fill="#475569" fontSize={10} textAnchor="middle" transform={`rotate(-90,10,${pad + fh / 2})`}>True Positive Rate</text>
    </svg>
  );
}

function MetricsBar({ results }) {
  if (!results.length) return null;
  const metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"];
  const colors = ["#6ee7b7", "#a5b4fc", "#fcd34d", "#f9a8d4", "#7dd3fc"];
  return (
    <div style={{ overflowX: "auto" }}>
      {metrics.map((m, mi) => (
        <div key={m} style={{ marginBottom: 14 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
            <span style={{ fontSize: 11, color: "#64748b", width: 80, textTransform: "uppercase", letterSpacing: 1 }}>{m.replace("_", " ")}</span>
            <div style={{ flex: 1, display: "flex", gap: 8 }}>
              {results.map(r => (
                <div key={r.run_id} style={{ flex: 1 }}>
                  <div style={{ height: 24, background: "#1e293b", borderRadius: 4, overflow: "hidden" }}>
                    <div style={{ height: "100%", width: `${(r.metrics[m] * 100).toFixed(1)}%`, background: colors[mi], borderRadius: 4, transition: "width 0.8s ease", display: "flex", alignItems: "center", justifyContent: "flex-end", paddingRight: 6 }}>
                      <span style={{ fontSize: 10, color: "#0f172a", fontWeight: 800 }}>{(r.metrics[m] * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                  <div style={{ fontSize: 10, color: "#475569", marginTop: 2, textAlign: "center" }}>
                    {MODEL_INFO[r.model_name]?.label || r.model_name}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

// ─── Main App ────────────────────────────────────────────────────────────────
export default function ChurnPlatform() {
  const [tab, setTab] = useState("train");
  const [selectedModels, setSelectedModels] = useState({ logistic_regression: true, decision_tree: false, svm: false });
  const [hyperparams, setHyperparams] = useState({
    logistic_regression: { C: 1.0, max_iter: 200, solver: "saga", penalty: "l2" },
    decision_tree: { max_depth: 5, min_samples_split: 2, min_samples_leaf: 1, criterion: "gini" },
    svm: { C: 1.0, kernel: "rbf", gamma: "scale", max_iter: 1000 },
  });
  const [testSize, setTestSize] = useState(0.2);
  const [loading, setLoading] = useState(false);
  const [tuning, setTuning] = useState(null);
  const [trainResults, setTrainResults] = useState([]);
  const [error, setError] = useState(null);
  const [predictFile, setPredictFile] = useState(null);
  const [predictModel, setPredictModel] = useState("logistic_regression");
  const [predictRunId, setPredictRunId] = useState("");
  const [predictions, setPredictions] = useState(null);
  const [savedRuns, setSavedRuns] = useState([]);

  const S = { // styles
    app: { minHeight: "100vh", background: "#060d1a", color: "#e2e8f0", fontFamily: "'IBM Plex Mono', 'Fira Code', monospace" },
    header: { background: "#0a1628", borderBottom: "1px solid #1e293b", padding: "16px 32px", display: "flex", alignItems: "center", gap: 16 },
    logo: { fontSize: 20, fontWeight: 900, color: "#6ee7b7", letterSpacing: -1 },
    badge: { background: "#6ee7b71a", color: "#6ee7b7", border: "1px solid #6ee7b744", borderRadius: 4, padding: "2px 8px", fontSize: 11 },
    main: { maxWidth: 1200, margin: "0 auto", padding: "28px 24px" },
    tabs: { display: "flex", gap: 4, marginBottom: 28, borderBottom: "1px solid #1e293b", paddingBottom: 0 },
    section: { background: "#0a1628", border: "1px solid #1e293b", borderRadius: 12, padding: 24, marginBottom: 20 },
    sectionTitle: { fontSize: 13, fontWeight: 700, color: "#94a3b8", marginBottom: 16, textTransform: "uppercase", letterSpacing: 2 },
    btn: { background: "#6ee7b7", color: "#0f172a", border: "none", borderRadius: 8, padding: "10px 22px", fontSize: 13, fontWeight: 800, cursor: "pointer", letterSpacing: 0.5 },
    btnOutline: { background: "transparent", color: "#6ee7b7", border: "1px solid #6ee7b755", borderRadius: 8, padding: "8px 18px", fontSize: 12, fontWeight: 700, cursor: "pointer" },
    input: { background: "#0f172a", border: "1px solid #1e293b", borderRadius: 6, color: "#e2e8f0", padding: "7px 12px", fontSize: 13, width: "100%", outline: "none", fontFamily: "inherit" },
    label: { fontSize: 11, color: "#64748b", display: "block", marginBottom: 5, letterSpacing: 1, textTransform: "uppercase" },
    grid2: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 },
    errorBox: { background: "#7f1d1d22", border: "1px solid #ef444444", borderRadius: 8, padding: "12px 16px", color: "#fca5a5", fontSize: 13, marginBottom: 16 },
  };

  const tabs = [
    { id: "train", label: "⚗ Train" },
    { id: "results", label: "📊 Results" },
    { id: "predict", label: "🔮 Predict" },
    { id: "runs", label: "📁 Run History" },
  ];

  const handleTrain = async () => {
    const models = Object.entries(selectedModels).filter(([, v]) => v).map(([k]) => k);
    if (!models.length) { setError("Select at least one model."); return; }
    setLoading(true); setError(null);
    try {
      const res = await trainModels({ models, features: [], testSize, hyperparameters: hyperparams });
      setTrainResults(res.results);
      setTab("results");
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  const handleTune = async (modelName) => {
    setTuning(modelName); setError(null);
    try {
      const res = await autoTune(modelName);
      setHyperparams(h => ({ ...h, [modelName]: res.best_params }));
      alert(`✅ Best CV F1: ${(res.best_cv_score * 100).toFixed(2)}%\nBest params applied to config.`);
    } catch (e) { setError(e.message); }
    finally { setTuning(null); }
  };

  const handleLoadRuns = async () => {
    try { const d = await fetchModels(); setSavedRuns(d.runs || []); setTab("runs"); }
    catch (e) { setError(e.message); }
  };

  const handlePredict = async () => {
    if (!predictFile || !predictRunId) { setError("Select a file and enter a Run ID."); return; }
    setLoading(true); setError(null);
    try { const res = await predictCSV(predictFile, predictModel, predictRunId); setPredictions(res); }
    catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  const exportResults = () => {
    if (!trainResults.length) return;
    const rows = trainResults.map(r => ({ model: r.model_name, ...Object.fromEntries(Object.entries(r.metrics).filter(([k]) => typeof r.metrics[k] === "number")) }));
    const csv = [Object.keys(rows[0]).join(","), ...rows.map(r => Object.values(r).join(","))].join("\n");
    const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([csv])); a.download = "churn_metrics.csv"; a.click();
  };

  return (
    <div style={S.app}>
      {/* Header */}
      <div style={S.header}>
        <div style={S.logo}>◈ CHURN.ML</div>
        <span style={S.badge}>v1.0 — University Project</span>
        <div style={{ flex: 1 }} />
        <span style={{ fontSize: 11, color: "#334155" }}>Logistic Regression · Decision Tree · SVM</span>
      </div>

      <div style={S.main}>
        {/* Tabs */}
        <div style={S.tabs}>
          {tabs.map(t => (
            <button key={t.id} onClick={() => t.id === "runs" ? handleLoadRuns() : setTab(t.id)}
              style={{ background: tab === t.id ? "#6ee7b71a" : "transparent", color: tab === t.id ? "#6ee7b7" : "#475569", border: `1px solid ${tab === t.id ? "#6ee7b744" : "transparent"}`, borderBottom: "none", borderRadius: "8px 8px 0 0", padding: "9px 20px", fontSize: 12, fontWeight: 700, cursor: "pointer", marginBottom: -1, letterSpacing: 0.5 }}>
              {t.label}
            </button>
          ))}
        </div>

        {error && <div style={S.errorBox}>⚠ {error}</div>}

        {/* ── TRAIN TAB ── */}
        {tab === "train" && (
          <>
            {/* Dataset Preview */}
            <div style={S.section}>
              <div style={S.sectionTitle}>Dataset Preview — Telco Churn</div>
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                  <thead>
                    <tr>{Object.keys(DEMO_DATASET[0]).map(k => <th key={k} style={{ padding: "6px 12px", borderBottom: "1px solid #1e293b", color: "#64748b", textAlign: "left", textTransform: "uppercase", letterSpacing: 1, fontSize: 10 }}>{k}</th>)}</tr>
                  </thead>
                  <tbody>
                    {DEMO_DATASET.map((row, i) => (
                      <tr key={i} style={{ borderBottom: "1px solid #0f172a" }}>
                        {Object.values(row).map((v, j) => (
                          <td key={j} style={{ padding: "7px 12px", color: j === 6 ? (v === "Yes" ? "#fca5a5" : "#6ee7b7") : "#cbd5e1", fontWeight: j === 6 ? 700 : 400 }}>{v}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Model Selection */}
            <div style={S.section}>
              <div style={S.sectionTitle}>Model Selection</div>
              <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
                {Object.entries(MODEL_INFO).map(([key, info]) => (
                  <label key={key} style={{ display: "flex", alignItems: "center", gap: 10, padding: "12px 18px", background: selectedModels[key] ? info.color + "15" : "#0f172a", border: `2px solid ${selectedModels[key] ? info.color : "#1e293b"}`, borderRadius: 10, cursor: "pointer", transition: "all 0.2s", flex: 1, minWidth: 200 }}>
                    <input type="checkbox" checked={selectedModels[key]} onChange={e => setSelectedModels(s => ({ ...s, [key]: e.target.checked }))} style={{ accentColor: info.color, width: 16, height: 16 }} />
                    <div>
                      <div style={{ fontWeight: 700, color: selectedModels[key] ? info.color : "#94a3b8", fontSize: 13 }}>{info.label}</div>
                    </div>
                    <Tooltip text={info.tooltip} />
                  </label>
                ))}
              </div>
            </div>

            {/* Hyperparameter panels */}
            {Object.entries(MODEL_INFO).map(([key, info]) => !selectedModels[key] ? null : (
              <div key={key} style={{ ...S.section, borderColor: info.color + "33" }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
                  <div style={{ ...S.sectionTitle, marginBottom: 0, color: info.color }}>{info.label} — Hyperparameters</div>
                  <button onClick={() => handleTune(key)} disabled={tuning === key} style={{ ...S.btnOutline, borderColor: info.accent + "66", color: info.color, opacity: tuning === key ? 0.6 : 1 }}>
                    {tuning === key ? "⏳ Tuning..." : "⚡ Auto Tune"}
                  </button>
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px,1fr))", gap: 14 }}>
                  {info.params.map(p => (
                    <div key={p.key}>
                      <label style={S.label}>{p.label}</label>
                      {p.type === "select" ? (
                        <select value={hyperparams[key][p.key]} onChange={e => setHyperparams(h => ({ ...h, [key]: { ...h[key], [p.key]: e.target.value } }))} style={{ ...S.input, appearance: "none" }}>
                          {p.options.map(o => <option key={o} value={o}>{o}</option>)}
                        </select>
                      ) : (
                        <input type="number" value={hyperparams[key][p.key]} min={p.min} step={p.step} onChange={e => setHyperparams(h => ({ ...h, [key]: { ...h[key], [p.key]: parseFloat(e.target.value) || p.default } }))} style={S.input} />
                      )}
                    </div>
                  ))}
                </div>
              </div>
            ))}

            {/* Train config */}
            <div style={{ ...S.section, display: "flex", alignItems: "center", gap: 24, flexWrap: "wrap" }}>
              <div>
                <label style={S.label}>Test Split</label>
                <input type="range" min="0.1" max="0.4" step="0.05" value={testSize} onChange={e => setTestSize(parseFloat(e.target.value))} style={{ accentColor: "#6ee7b7" }} />
                <span style={{ color: "#6ee7b7", fontSize: 13, marginLeft: 10, fontWeight: 700 }}>{(testSize * 100).toFixed(0)}%</span>
              </div>
              <button onClick={handleTrain} disabled={loading} style={{ ...S.btn, padding: "12px 36px", fontSize: 14, opacity: loading ? 0.7 : 1, display: "flex", alignItems: "center", gap: 10 }}>
                {loading ? <>⏳ Training...</> : <>▶ Train Models</>}
              </button>
            </div>
          </>
        )}

        {/* ── RESULTS TAB ── */}
        {tab === "results" && (
          <>
            {!trainResults.length ? (
              <div style={{ ...S.section, color: "#475569", textAlign: "center", padding: 60 }}>No results yet. Train some models first.</div>
            ) : (
              <>
                <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 16 }}>
                  <button onClick={exportResults} style={S.btnOutline}>⬇ Export CSV</button>
                </div>
                {/* Metrics comparison */}
                <div style={S.section}>
                  <div style={S.sectionTitle}>Metrics Comparison</div>
                  <MetricsBar results={trainResults} />
                </div>
                {/* Per-model detail */}
                {trainResults.map(r => (
                  <div key={r.run_id} style={{ ...S.section, borderColor: (MODEL_INFO[r.model_name]?.color || "#6ee7b7") + "44" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 20 }}>
                      <div style={{ fontSize: 16, fontWeight: 800, color: MODEL_INFO[r.model_name]?.color || "#6ee7b7" }}>
                        {MODEL_INFO[r.model_name]?.label || r.model_name}
                      </div>
                      <Tag color="#475569">{r.run_id.slice(0, 8)}</Tag>
                    </div>
                    <div style={{ display: "flex", gap: 12, marginBottom: 24, flexWrap: "wrap" }}>
                      {["accuracy", "precision", "recall", "f1_score", "roc_auc"].map(m => (
                        <MetricCard key={m} label={m.replace("_", " ")} value={r.metrics[m]} color={MODEL_INFO[r.model_name]?.color || "#6ee7b7"} />
                      ))}
                    </div>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
                      <div>
                        <div style={S.sectionTitle}>Confusion Matrix</div>
                        <ConfusionMatrix cm={r.metrics.confusion_matrix} />
                      </div>
                      <div>
                        <div style={S.sectionTitle}>ROC Curve</div>
                        <SimpleROC roc_curve={r.metrics.roc_curve} />
                        <div style={{ fontSize: 11, color: "#475569", marginTop: 8 }}>AUC = {r.metrics.roc_auc}</div>
                      </div>
                    </div>
                    {r.feature_importances && (
                      <div style={{ marginTop: 20 }}>
                        <div style={S.sectionTitle}>Feature Importances</div>
                        {Object.entries(r.feature_importances).sort(([, a], [, b]) => b - a).slice(0, 8).map(([feat, imp]) => (
                          <div key={feat} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                            <span style={{ fontSize: 11, color: "#64748b", width: 160, flexShrink: 0 }}>{feat}</span>
                            <div style={{ flex: 1, background: "#1e293b", height: 12, borderRadius: 4, overflow: "hidden" }}>
                              <div style={{ height: "100%", width: `${Math.min(imp * 500, 100)}%`, background: MODEL_INFO[r.model_name]?.color || "#6ee7b7", borderRadius: 4 }} />
                            </div>
                            <span style={{ fontSize: 11, color: "#94a3b8", fontFamily: "monospace", width: 50, textAlign: "right" }}>{imp.toFixed(3)}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </>
            )}
          </>
        )}

        {/* ── PREDICT TAB ── */}
        {tab === "predict" && (
          <div style={S.section}>
            <div style={S.sectionTitle}>Predict from CSV</div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 20 }}>
              <div>
                <label style={S.label}>Upload CSV</label>
                <input type="file" accept=".csv" onChange={e => setPredictFile(e.target.files[0])} style={{ ...S.input, padding: "6px" }} />
              </div>
              <div>
                <label style={S.label}>Model</label>
                <select value={predictModel} onChange={e => setPredictModel(e.target.value)} style={{ ...S.input, appearance: "none" }}>
                  {Object.entries(MODEL_INFO).map(([k, v]) => <option key={k} value={k}>{v.label}</option>)}
                </select>
              </div>
              <div>
                <label style={S.label}>Run ID</label>
                <input placeholder="paste run_id from results..." value={predictRunId} onChange={e => setPredictRunId(e.target.value)} style={S.input} />
              </div>
            </div>
            <button onClick={handlePredict} disabled={loading} style={S.btn}>{loading ? "⏳ Predicting..." : "🔮 Predict"}</button>
            {predictions && (
              <div style={{ marginTop: 24 }}>
                <div style={S.sectionTitle}>Results — {predictions.model_used}</div>
                <div style={{ display: "flex", gap: 12, marginBottom: 16 }}>
                  <Tag color="#fca5a5">Churn: {predictions.predictions.filter(p => p === 1).length}</Tag>
                  <Tag color="#6ee7b7">No Churn: {predictions.predictions.filter(p => p === 0).length}</Tag>
                </div>
                <div style={{ maxHeight: 300, overflowY: "auto" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                    <thead>
                      <tr>
                        <th style={{ padding: "6px 12px", color: "#64748b", borderBottom: "1px solid #1e293b", textAlign: "left" }}>#</th>
                        <th style={{ padding: "6px 12px", color: "#64748b", borderBottom: "1px solid #1e293b", textAlign: "left" }}>Prediction</th>
                        <th style={{ padding: "6px 12px", color: "#64748b", borderBottom: "1px solid #1e293b", textAlign: "left" }}>Churn Probability</th>
                      </tr>
                    </thead>
                    <tbody>
                      {predictions.predictions.slice(0, 50).map((p, i) => (
                        <tr key={i} style={{ borderBottom: "1px solid #0f172a" }}>
                          <td style={{ padding: "6px 12px", color: "#475569" }}>{i + 1}</td>
                          <td style={{ padding: "6px 12px", color: p === 1 ? "#fca5a5" : "#6ee7b7", fontWeight: 700 }}>{p === 1 ? "⚠ Churn" : "✓ Retain"}</td>
                          <td style={{ padding: "6px 12px", color: "#94a3b8", fontFamily: "monospace" }}>{(predictions.probabilities[i] * 100).toFixed(1)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── RUNS TAB ── */}
        {tab === "runs" && (
          <div style={S.section}>
            <div style={S.sectionTitle}>MLflow Run History ({savedRuns.length} runs)</div>
            {!savedRuns.length ? (
              <div style={{ color: "#475569", padding: 40, textAlign: "center" }}>No runs yet. Train some models first.</div>
            ) : (
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                  <thead>
                    <tr>{["Model", "Accuracy", "F1", "Recall", "ROC-AUC", "Run ID", "Timestamp"].map(h => (
                      <th key={h} style={{ padding: "8px 12px", borderBottom: "1px solid #1e293b", color: "#64748b", textAlign: "left", textTransform: "uppercase", letterSpacing: 1, fontSize: 10 }}>{h}</th>
                    ))}</tr>
                  </thead>
                  <tbody>
                    {savedRuns.map((run, i) => (
                      <tr key={i} style={{ borderBottom: "1px solid #0f172a" }}>
                        <td style={{ padding: "8px 12px" }}><Tag color={MODEL_INFO[run.model]?.color || "#6ee7b7"}>{run.model}</Tag></td>
                        {["accuracy", "f1_score", "recall", "roc_auc"].map(m => (
                          <td key={m} style={{ padding: "8px 12px", color: "#94a3b8", fontFamily: "monospace" }}>{run.metrics[m] ? (run.metrics[m] * 100).toFixed(1) + "%" : "—"}</td>
                        ))}
                        <td style={{ padding: "8px 12px", color: "#475569", fontFamily: "monospace", fontSize: 10 }}>{run.run_id.slice(0, 12)}...</td>
                        <td style={{ padding: "8px 12px", color: "#475569", fontSize: 10 }}>{run.timestamp?.slice(0, 16)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* Architecture note for presentation */}
        <div style={{ ...S.section, borderColor: "#1e293b", marginTop: 32 }}>
          <div style={S.sectionTitle}>System Architecture</div>
          <div style={{ display: "flex", gap: 8, alignItems: "center", justifyContent: "center", flexWrap: "wrap", padding: "8px 0" }}>
            {[
              ["React UI", "#6ee7b7", "Tailwind components, axios calls"],
              ["→"],
              ["FastAPI", "#a5b4fc", "REST endpoints, request validation"],
              ["→"],
              ["sklearn Pipeline", "#fcd34d", "LR · DT · SVM + GridSearchCV"],
              ["→"],
              ["MLflow", "#f9a8d4", "Params, metrics, model versions"],
              ["→"],
              ["joblib", "#7dd3fc", "Model persistence & reload"],
            ].map((item, i) => item.length === 1 ? (
              <span key={i} style={{ color: "#334155", fontSize: 20 }}>→</span>
            ) : (
              <div key={i} style={{ background: item[1] + "15", border: `1px solid ${item[1]}44`, borderRadius: 8, padding: "10px 16px", textAlign: "center" }}>
                <div style={{ color: item[1], fontWeight: 800, fontSize: 13 }}>{item[0]}</div>
                <div style={{ color: "#475569", fontSize: 10, marginTop: 3 }}>{item[2]}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
