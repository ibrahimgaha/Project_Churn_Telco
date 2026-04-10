import React, { useState, useEffect } from "react";
import MODEL_INFO from "../models";
import S from "../styles";
import { Tag } from "./UIComponents";
import { fetchModels, launchMLflowUI } from "../api";

export default function RunsTab() {
  const [runs, setRuns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [launching, setLaunching] = useState(false);

  const loadRuns = async () => {
    setLoading(true); setError(null);
    try {
      const d = await fetchModels();
      setRuns(d.runs || []);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleLaunchMLflow = async () => {
    setLaunching(true); setError(null);
    try {
      const res = await launchMLflowUI();
      if (res.url) {
        window.open(res.url, "_blank");
      }
    } catch (e) {
      setError("Could not launch MLflow: " + e.message);
    } finally {
      setLaunching(false);
    }
  };

  useEffect(() => {
    loadRuns();
  }, []);

  return (
    <div className="fade-in">
      <div style={{ ...S.section, borderBottom: "none", borderBottomLeftRadius: 0, borderBottomRightRadius: 0, marginBottom: 0 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div style={S.sectionTitle}>MLflow Experiment History ({runs.length} runs)</div>
          <div style={{ display: "flex", gap: 8 }}>
            <button
              onClick={handleLaunchMLflow}
              disabled={launching}
              style={{ ...S.btn, background: "#f9a8d4", fontSize: 11, padding: "8px 16px" }}
            >
              {launching ? "🚀 Launching..." : "🚀 Launch MLflow UI"}
            </button>
            <button onClick={loadRuns} disabled={loading} style={S.btnOutline}>🔄 Refresh History</button>
          </div>
        </div>
        <p style={{ fontSize: 13, color: "#64748b", marginTop: 8 }}>
          This table displays all experiments logged to MLflow. You can compare different runs or copy Run IDs for inference.
        </p>
      </div>

      <div style={{ border: "1px solid #1e293b", borderTop: "none", borderBottomLeftRadius: 12, borderBottomRightRadius: 12, overflow: "hidden", background: "#0a1628" }}>
        {error && <div style={{ ...S.errorBox, margin: 20 }}>⚠ {error}</div>}
        {loading ? (
          <div style={{ padding: 60, textAlign: "center", color: "#64748b" }}>⏳ Loading experiments...</div>
        ) : !runs.length ? (
          <div style={{ padding: 60, textAlign: "center", color: "#64748b" }}>No runs found in MLflow. Train some models to get started.</div>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
              <thead>
                <tr style={{ background: "#0f172a" }}>
                  {["Model", "Params", "Acc", "F1", "Recall", "ROC-AUC", "Run ID", "Timestamp"].map(h => (
                    <th key={h} style={{ padding: "14px 16px", borderBottom: "1px solid #1e293b", color: "#64748b", textAlign: "left", textTransform: "uppercase", letterSpacing: 1, fontSize: 10 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {runs.map((run, i) => {
                  const info = MODEL_INFO[run.model];
                  return (
                    <tr key={run.run_id} className="slide-up" style={{ animationDelay: `${i * 0.02}s`, borderBottom: "1px solid #0f172a", transition: "background 0.2s" }} onMouseEnter={e => e.currentTarget.style.background = "#0f172a"} onMouseLeave={e => e.currentTarget.style.background = "transparent"}>
                      <td style={{ padding: "14px 16px" }}><Tag color={info?.color || "#6ee7b7"}>{info?.label || run.model}</Tag></td>
                      <td style={{ padding: "14px 16px" }}>
                        <div style={{ fontSize: 10, color: "#64748b", width: 120, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }} title={JSON.stringify(run.params)}>
                          {Object.keys(run.params).length > 0 ? JSON.stringify(run.params).slice(1, -1).replace(/"/g, "") : "—"}
                        </div>
                      </td>
                      {["accuracy", "f1_score", "recall", "roc_auc"].map(m => (
                        <td key={m} style={{ padding: "14px 16px", color: (run.metrics[m] > 0.8) ? "#6ee7b7" : (run.metrics[m] > 0.6) ? "#fcd34d" : "#94a3b8", fontFamily: "monospace", fontWeight: 700 }}>
                          {run.metrics[m] ? (run.metrics[m] * 100).toFixed(1) + "%" : "—"}
                        </td>
                      ))}
                      <td style={{ padding: "14px 16px" }}>
                        <button 
                          onClick={() => {
                            navigator.clipboard.writeText(run.run_id);
                            alert(`ID ${run.run_id.slice(0, 8)} copied!`);
                          }}
                          style={{ background: "#0f172a", border: "1px solid #1e293b", color: "#475569", padding: "4px 8px", borderRadius: 4, cursor: "pointer", fontSize: 10, fontFamily: "monospace" }}
                        >
                          {run.run_id.slice(0, 10)}... 📋
                        </button>
                      </td>
                      <td style={{ padding: "14px 16px", color: "#475569", fontSize: 11 }}>{new Date(run.timestamp).toLocaleString()}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
