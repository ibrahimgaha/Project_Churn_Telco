import React, { useState } from "react";
import MODEL_INFO from "../models";
import S from "../styles";
import { Tag } from "./UIComponents";
import { predictCSV } from "../api";

export default function PredictTab() {
  const [file, setFile] = useState(null);
  const [modelName, setModelName] = useState("logistic_regression");
  const [runId, setRunId] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [predictions, setPredictions] = useState(null);

  const handlePredict = async () => {
    if (!file || !runId) { setError("Please select a CSV file and enter a valid Run ID."); return; }
    setLoading(true); setError(null);
    try {
      const res = await predictCSV(file, modelName, runId);
      setPredictions(res);
    } catch (e) { 
      setError(e.message); 
    } finally { 
      setLoading(false); 
    }
  };

  return (
    <div className="fade-in">
      <div style={S.section}>
        <div style={S.sectionTitle}>Inference — Predict from CSV</div>
        <p style={{ fontSize: 13, color: "#64748b", marginBottom: 20 }}>
          Upload a customer dataset (CSV) to generate churn predictions using a previously trained model.
        </p>

        {error && <div style={S.errorBox}>⚠ {error}</div>}

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 20, marginBottom: 24 }}>
          <div>
            <label style={S.label}>Upload CSV File</label>
            <input 
              type="file" 
              accept=".csv" 
              onChange={e => setFile(e.target.files[0])} 
              style={{ ...S.input, padding: "8px", borderStyle: "dashed" }} 
            />
          </div>
          <div>
            <label style={S.label}>Select Model Type</label>
            <select 
              value={modelName} 
              onChange={e => setModelName(e.target.value)} 
              style={{ ...S.input, appearance: "none" }}
            >
              {Object.entries(MODEL_INFO).map(([k, v]) => <option key={k} value={k}>{v.label}</option>)}
            </select>
          </div>
          <div>
            <label style={S.label}>Run ID (from MLflow)</label>
            <input 
              placeholder="e.g. abc123def456..." 
              value={runId} 
              onChange={e => setRunId(e.target.value)} 
              style={S.input} 
            />
          </div>
        </div>

        <button 
          onClick={handlePredict} 
          disabled={loading} 
          style={{ ...S.btn, width: "100%", padding: "14px", fontSize: 14, background: (MODEL_INFO[modelName]?.color || "#6ee7b7") }}
        >
          {loading ? "⏳ Processing Predictions..." : "🔮 Generate Predictions"}
        </button>

        {predictions && (
          <div style={{ marginTop: 32 }} className="slide-up">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
              <div style={S.sectionTitle}>Prediction Results — {MODEL_INFO[predictions.model_used]?.label}</div>
              <div style={{ display: "flex", gap: 8 }}>
                <Tag color="#fca5a5">⚠ Churn: {predictions.predictions.filter(p => p === 1).length}</Tag>
                <Tag color="#6ee7b7">✓ Retain: {predictions.predictions.filter(p => p === 0).length}</Tag>
              </div>
            </div>

            <div style={{ maxHeight: 400, overflowY: "auto", border: "1px solid #1e293b", borderRadius: 8 }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                <thead style={{ position: "sticky", top: 0, background: "#0a1628", zIndex: 1 }}>
                  <tr>
                    <th style={{ padding: "12px", color: "#64748b", borderBottom: "1px solid #1e293b", textAlign: "left" }}>#</th>
                    <th style={{ padding: "12px", color: "#64748b", borderBottom: "1px solid #1e293b", textAlign: "left" }}>Classification</th>
                    <th style={{ padding: "12px", color: "#64748b", borderBottom: "1px solid #1e293b", textAlign: "left" }}>Churn Probability</th>
                    <th style={{ padding: "12px", borderBottom: "1px solid #1e293b", width: "40%" }}>Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {predictions.predictions.slice(0, 100).map((p, i) => (
                    <tr key={i} style={{ borderBottom: "1px solid #0f172a", background: i % 2 === 0 ? "transparent" : "#0f172a" }}>
                      <td style={{ padding: "10px 12px", color: "#475569", fontFamily: "monospace" }}>{i + 1}</td>
                      <td style={{ padding: "10px 12px", color: p === 1 ? "#fca5a5" : "#6ee7b7", fontWeight: 700 }}>
                        {p === 1 ? "⚠ CHURN" : "✓ RETAIN"}
                      </td>
                      <td style={{ padding: "10px 12px", color: "#94a3b8", fontFamily: "monospace" }}>
                        {(predictions.probabilities[i] * 100).toFixed(1)}%
                      </td>
                      <td style={{ padding: "10px 12px" }}>
                        <div style={{ width: "100%", background: "#1e293b", height: 6, borderRadius: 3, overflow: "hidden" }}>
                          <div style={{ 
                            height: "100%", 
                            width: `${Math.abs(predictions.probabilities[i] - 0.5) * 200}%`, 
                            background: p === 1 ? "#fca5a5" : "#6ee7b7", 
                            borderRadius: 3 
                          }} />
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {predictions.predictions.length > 100 && (
              <p style={{ textAlign: "center", fontSize: 12, color: "#475569", marginTop: 12 }}>Showing first 100 results...</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
