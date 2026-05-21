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
    if (!file || !runId) { setError("Required: Dataset (.csv) + valid Experiment Artifact ID."); return; }
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
    <div className="stagger">
      <div style={S.section} className="glass-card">
        <div style={S.sectionTitle}>
           <span style={{ color: "#6366f1" }}>🔮</span> Batch Inference Engine
        </div>
        
        {error && <div style={S.errorBox}><span>⚠️</span> {error}</div>}

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 32, marginBottom: 40 }}>
          <div style={{ position: "relative" }}>
            <label style={S.label}>Dataset Source (.csv)</label>
            <div style={{ 
              border: "1px dashed rgba(99, 102, 241, 0.3)", 
              borderRadius: 14, 
              padding: "20px", 
              textAlign: "center",
              background: "rgba(99, 102, 241, 0.03)",
              cursor: "pointer",
              transition: "all 0.2s"
            }}>
               <input 
                 type="file" 
                 accept=".csv" 
                 onChange={e => setFile(e.target.files[0])} 
                 style={{ opacity: 0, position: "absolute", inset: 0, cursor: "pointer" }} 
               />
               <div style={{ fontSize: 24, marginBottom: 8 }}>{file ? "📄" : "📤"}</div>
               <div style={{ fontSize: 13, fontWeight: 700, color: file ? "#818cf8" : "#94a3b8" }}>
                  {file ? file.name : "Click to select data file"}
               </div>
            </div>
          </div>
          <div>
            <label style={S.label}>Model Framework</label>
            <div style={{ position: "relative" }}>
              <select 
                value={modelName} 
                onChange={e => setModelName(e.target.value)} 
                style={{ ...S.input, appearance: "none" }}
              >
                {Object.entries(MODEL_INFO).map(([k, v]) => <option key={k} value={k}>{v.label}</option>)}
              </select>
              <div style={{ position: "absolute", right: 12, top: "50%", transform: "translateY(-50%)", pointerEvents: "none", fontSize: 10, color: "#475569" }}>▼</div>
            </div>
          </div>
          <div>
            <label style={S.label}>MLflow Artifact ID (Run ID)</label>
            <input 
              placeholder="Enter cryptographic run identifier..." 
              value={runId} 
              onChange={e => setRunId(e.target.value)} 
              style={S.input} 
            />
          </div>
        </div>

        <button 
          onClick={handlePredict} 
          disabled={loading} 
          style={{ 
            ...S.btn, 
            width: "100%", 
            padding: "18px", 
            fontSize: 15, 
            background: MODEL_INFO[modelName]?.color || "#6366f1",
            boxShadow: `0 10px 30px ${MODEL_INFO[modelName]?.color}44`
          }}
        >
          {loading ? "⚙️ Deploying Model & Running Inference..." : "🔮 Execute Predictive Analysis"}
        </button>

        {predictions && (
          <div style={{ marginTop: 48 }} className="stagger">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 24, padding: "0 8px" }}>
              <div style={{ fontSize: 14, fontWeight: 800, color: "#fff", letterSpacing: -0.5 }}>
                 Batch Results Overview <span style={{ color: "#475569", fontWeight: 400, marginLeft: 8 }}>— {MODEL_INFO[predictions.model_used]?.label}</span>
              </div>
              <div style={{ display: "flex", gap: 12 }}>
                <Tag color="#fca5a5">Risk Detected: {predictions.predictions.filter(p => p === 1).length}</Tag>
                <Tag color="#10b981">Retention Likely: {predictions.predictions.filter(p => p === 0).length}</Tag>
              </div>
            </div>

            <div style={{ 
              background: "rgba(15, 23, 42, 0.4)", 
              borderRadius: 24, 
              border: "1px solid rgba(255, 255, 255, 0.05)",
              overflow: "hidden" 
            }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                <thead>
                  <tr style={{ background: "rgba(15, 23, 42, 0.6)", textAlign: "left" }}>
                    <th style={{ padding: "18px 24px", color: "#64748b", fontWeight: 700, textTransform: "uppercase", fontSize: 10 }}>Sequence</th>
                    <th style={{ padding: "18px 24px", color: "#64748b", fontWeight: 700, textTransform: "uppercase", fontSize: 10 }}>Status Output</th>
                    <th style={{ padding: "18px 24px", color: "#64748b", fontWeight: 700, textTransform: "uppercase", fontSize: 10 }}>Churn Prob.</th>
                    <th style={{ padding: "18px 24px", color: "#64748b", fontWeight: 700, textTransform: "uppercase", fontSize: 10 }}>Inference Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {predictions.predictions.slice(0, 100).map((p, i) => (
                    <tr key={i} style={{ borderBottom: "1px solid rgba(255, 255, 255, 0.03)", transition: "background 0.2s" }} onMouseEnter={(e)=>e.currentTarget.style.background="rgba(255,255,255,0.02)"} onMouseLeave={(e)=>e.currentTarget.style.background="transparent"}>
                      <td style={{ padding: "14px 24px", color: "#475569", fontFamily: "monospace", fontSize: 12 }}>{String(i + 1).padStart(4, '0')}</td>
                      <td style={{ padding: "14px 24px" }}>
                        <span style={{ 
                          color: p === 1 ? "#fca5a5" : "#10b981", 
                          background: p === 1 ? "#7f1d1d22" : "#064e3b22",
                          padding: "4px 10px",
                          borderRadius: 8,
                          fontSize: 11,
                          fontWeight: 800,
                          border: `1px solid ${p === 1 ? "#ef444422" : "#10b98122"}`
                        }}>
                          {p === 1 ? "HIGH RISK" : "STABLE"}
                        </span>
                      </td>
                      <td style={{ padding: "14px 24px", color: "#f1f5f9", fontWeight: 700, fontFamily: "monospace" }}>
                        {(predictions.probabilities[i] * 100).toFixed(1)}%
                      </td>
                      <td style={{ padding: "14px 24px" }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                           <div style={{ flex: 1, background: "rgba(255,255,255,0.05)", height: 6, borderRadius: 3, overflow: "hidden" }}>
                             <div style={{ 
                               height: "100%", 
                               width: `${Math.abs(predictions.probabilities[i] - 0.5) * 200}%`, 
                               background: p === 1 ? "#f43f5e" : "#10b981", 
                               borderRadius: 3 
                             }} />
                           </div>
                           <span style={{ fontSize: 10, color: "#64748b", fontWeight: 700 }}>{(Math.abs(predictions.probabilities[i] - 0.5) * 200).toFixed(0)}%</span>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {predictions.predictions.length > 100 && (
                <div style={{ padding: 20, textAlign: "center", fontSize: 11, color: "#475569", borderTop: "1px solid rgba(255,255,255,0.03)" }}>
                   Batch overflow: Optimized view for first 100 records.
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
