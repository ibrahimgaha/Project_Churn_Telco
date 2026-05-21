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
    <div className="stagger">
      <div style={{ ...S.section, marginBottom: 0, borderBottomLeftRadius: 0, borderBottomRightRadius: 0 }} className="glass-card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div style={S.sectionTitle}>
             <span style={{ color: "#f9a8d4" }}>📁</span> Experiment Registry
          </div>
          <div style={{ display: "flex", gap: 12 }}>
            <button
              onClick={handleLaunchMLflow}
              disabled={launching}
              style={{ 
                ...S.btn, 
                background: "rgba(249, 168, 212, 0.1)", 
                color: "#f9a8d4", 
                border: "1px solid rgba(249, 168, 212, 0.3)",
                fontSize: 12,
                boxShadow: "none"
              }}
            >
              {launching ? "🚀 Initializing..." : "🚀 Launch MLflow UI"}
            </button>
            <button 
              onClick={loadRuns} 
              disabled={loading} 
              style={{...S.btnOutline, background: "rgba(255,255,255,0.02)"}}
            >
              🔄 Sync Repository
            </button>
          </div>
        </div>
        <p style={{ fontSize: 13, color: "#64748b", marginTop: 12, maxWidth: 700 }}>
          The Experiment Registry tracks every cryptographic hash, hyperparameter configuration, and performance metric across all training iterations.
        </p>
      </div>

      <div style={{ 
        border: "1px solid rgba(255, 255, 255, 0.05)", 
        borderTop: "none", 
        borderBottomLeftRadius: 24, 
        borderBottomRightRadius: 24, 
        overflow: "hidden", 
        background: "rgba(15, 23, 42, 0.3)",
        backdropFilter: "blur(10px)" 
      }}>
        {error && <div style={{ ...S.errorBox, margin: 24 }}><span>⚠️</span> {error}</div>}
        
        {loading ? (
          <div style={{ padding: 100, textAlign: "center", color: "#475569" }}>
             <div style={{ fontSize: 32, marginBottom: 16 }}>📡</div>
             <div style={{ fontSize: 13, fontWeight: 600 }}>Syncing with central MLflow server...</div>
          </div>
        ) : !runs.length ? (
          <div style={{ padding: 100, textAlign: "center", color: "#475569" }}>
             <div style={{ fontSize: 32, marginBottom: 16 }}>📭</div>
             <div style={{ fontSize: 13, fontWeight: 600 }}>No telemetry detected. Run an experiment to populate the registry.</div>
          </div>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
              <thead>
                <tr style={{ background: "rgba(15, 23, 42, 0.6)", textAlign: "left" }}>
                  {["Model", "Hyperparameters", "ACC", "F1", "REC", "AUC", "Artifact ID", "Synchronized"].map(h => (
                    <th key={h} style={{ padding: "18px 20px", color: "#64748b", textTransform: "uppercase", letterSpacing: 1.5, fontSize: 10, fontWeight: 800 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {runs.map((run, i) => {
                  const info = MODEL_INFO[run.model];
                  return (
                    <tr 
                      key={run.run_id} 
                      className="slide-up" 
                      style={{ 
                        animationDelay: `${i * 0.02}s`, 
                        borderBottom: "1px solid rgba(255, 255, 255, 0.03)", 
                        transition: "background 0.2s" 
                      }} 
                      onMouseEnter={e => e.currentTarget.style.background = "rgba(255,255,255,0.02)"} 
                      onMouseLeave={e => e.currentTarget.style.background = "transparent"}
                    >
                      <td style={{ padding: "16px 20px" }}><Tag color={info?.color || "#6366f1"}>{info?.label || run.model}</Tag></td>
                      <td style={{ padding: "16px 20px" }}>
                        <div style={{ fontSize: 11, color: "#64748b", width: 140, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", fontFamily: "'JetBrains Mono', monospace" }} title={JSON.stringify(run.params)}>
                          {Object.keys(run.params).length > 0 ? JSON.stringify(run.params).slice(1, -1).replace(/"/g, "") : "—"}
                        </div>
                      </td>
                      {["accuracy", "f1_score", "recall", "roc_auc"].map(m => (
                        <td key={m} style={{ padding: "16px 20px", color: (run.metrics[m] > 0.8) ? "#10b981" : (run.metrics[m] > 0.6) ? "#fbbf24" : "#94a3b8", fontFamily: "'JetBrains Mono', monospace", fontWeight: 700 }}>
                          {run.metrics[m] ? (run.metrics[m] * 100).toFixed(1) + "%" : "—"}
                        </td>
                      ))}
                      <td style={{ padding: "16px 20px" }}>
                        <button 
                          onClick={() => {
                            navigator.clipboard.writeText(run.run_id);
                            alert(`Artifact ID ${run.run_id.slice(0, 8)} copied!`);
                          }}
                          style={{ 
                            background: "rgba(255,255,255,0.03)", 
                            border: "1px solid rgba(255,255,255,0.08)", 
                            color: "#94a3b8", 
                            padding: "6px 12px", 
                            borderRadius: 8, 
                            cursor: "pointer", 
                            fontSize: 10, 
                            fontFamily: "'JetBrains Mono', monospace",
                            transition: "all 0.2s"
                          }}
                          onMouseEnter={(e)=>e.currentTarget.style.borderColor="rgba(255,255,255,0.2)"}
                          onMouseLeave={(e)=>e.currentTarget.style.borderColor="rgba(255,255,255,0.08)"}
                        >
                          {run.run_id.slice(0, 12)}... 📋
                        </button>
                      </td>
                      <td style={{ padding: "16px 20px", color: "#475569", fontSize: 11, fontWeight: 500 }}>
                         {new Date(run.timestamp).toLocaleDateString()} <span style={{opacity: 0.5}}>{new Date(run.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span>
                      </td>
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
