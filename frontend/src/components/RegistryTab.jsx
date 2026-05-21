import React, { useState, useEffect } from "react";
import S from "../styles";
import { Tag } from "./UIComponents";
import { promoteModel, fetchRegistryStatus } from "../api";

export default function RegistryTab() {
  const [versions, setVersions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [promoting, setPromoting] = useState(false);
  const [promotionResult, setPromotionResult] = useState(null);
  const [error, setError] = useState("");

  const loadStatus = async () => {
    setLoading(true);
    try {
      const res = await fetchRegistryStatus();
      setVersions(res.versions || []);
      setError("");
    } catch (e) {
      console.error(e);
      // Fail silently if DB is not populated yet
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadStatus();
  }, []);

  const handlePromote = async () => {
    setPromoting(true);
    setPromotionResult(null);
    setError("");
    try {
      const res = await promoteModel();
      setPromotionResult(res);
      await loadStatus();
    } catch (e) {
      setError(e.message || "Model promotion failed. Please train some models first.");
    } finally {
      setPromoting(false);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 32 }}>
      {/* Promotion Action Card */}
      <div className="glass-card" style={S.section}>
        <div style={S.sectionTitle}>
          <span style={{ fontSize: 18 }}>🏆</span> Model Promotion & Registry Control
        </div>
        <p style={{ color: "#64748b", fontSize: 13, lineHeight: 1.5, margin: "10px 0 24px 0" }}>
          MLflow Model Registry operates as a centralized store. Click below to programmatically find the 
          best run across all logged experiments based on Accuracy. The system will register it, transition 
          it to <strong>Staging</strong>, verify the performance threshold (<strong>>= 0.85</strong>), and 
          promote it to <strong>Production</strong> if eligible.
        </p>

        <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
          <button
            onClick={handlePromote}
            disabled={promoting}
            style={{
              background: promoting ? "rgba(99, 102, 241, 0.4)" : "linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)",
              color: "#fff",
              border: "none",
              borderRadius: 14,
              padding: "14px 28px",
              fontSize: 14,
              fontWeight: 750,
              cursor: promoting ? "not-allowed" : "pointer",
              boxShadow: "0 4px 20px rgba(99, 102, 241, 0.3)",
              transition: "all 0.3s ease",
            }}
          >
            {promoting ? "Promoting Best Model..." : "🔍 Find & Promote Best Model"}
          </button>

          <button
            onClick={loadStatus}
            disabled={loading}
            style={{
              background: "rgba(255,255,255,0.03)",
              color: "#fff",
              border: "1px solid rgba(255,255,255,0.08)",
              borderRadius: 14,
              padding: "14px 24px",
              fontSize: 14,
              fontWeight: 600,
              cursor: "pointer",
              transition: "all 0.3s ease",
            }}
          >
            🔄 Sync Registry
          </button>
        </div>

        {error && (
          <div style={{ 
            marginTop: 20, 
            background: "rgba(239, 68, 68, 0.1)", 
            color: "#f87171", 
            padding: 16, 
            borderRadius: 12, 
            border: "1px solid rgba(239, 68, 68, 0.2)",
            fontSize: 13,
            fontWeight: 500
          }}>
            {error}
          </div>
        )}

        {promotionResult && (
          <div style={{ 
            marginTop: 24, 
            background: "rgba(99, 102, 241, 0.05)", 
            padding: 24, 
            borderRadius: 20, 
            border: "1px solid rgba(99, 102, 241, 0.2)"
          }}>
            <h4 style={{ margin: "0 0 12px 0", color: "#818cf8", fontSize: 15, fontWeight: 800 }}>
              🎉 Model Promotion Pipeline Complete
            </h4>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16, marginBottom: 16 }}>
              <div>
                <div style={{ fontSize: 11, color: "#64748b" }}>Registered Name</div>
                <div style={{ fontSize: 14, fontWeight: 700, marginTop: 4 }}>{promotionResult.model_name}</div>
              </div>
              <div>
                <div style={{ fontSize: 11, color: "#64748b" }}>Version Created</div>
                <div style={{ fontSize: 14, fontWeight: 700, marginTop: 4 }}>v{promotionResult.version}</div>
              </div>
              <div>
                <div style={{ fontSize: 11, color: "#64748b" }}>Assigned Stage</div>
                <div style={{ marginTop: 4 }}>
                  <Tag 
                    text={promotionResult.stage} 
                    color={promotionResult.stage === "Production" ? "#10b981" : "#f59e0b"} 
                  />
                </div>
              </div>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 2fr", gap: 16, marginBottom: 16 }}>
              <div>
                <div style={{ fontSize: 11, color: "#64748b" }}>Model Run Accuracy</div>
                <div style={{ fontSize: 14, fontWeight: 800, color: "#10b981", marginTop: 4 }}>
                  {(promotionResult.accuracy * 100).toFixed(2)}%
                </div>
              </div>
              <div>
                <div style={{ fontSize: 11, color: "#64748b" }}>MLflow Run ID</div>
                <div style={{ fontSize: 11, fontFamily: "monospace", marginTop: 6, color: "#94a3b8" }}>
                  {promotionResult.best_run_id}
                </div>
              </div>
            </div>
            <div style={{ 
              fontSize: 13, 
              color: "#f1f5f9", 
              background: "rgba(15, 23, 42, 0.4)", 
              padding: 12, 
              borderRadius: 10,
              border: "1px solid rgba(255,255,255,0.03)",
              fontWeight: 500
            }}>
              {promotionResult.message}
            </div>
          </div>
        )}
      </div>

      {/* Registry Status Table */}
      <div className="glass-card" style={S.section}>
        <div style={S.sectionTitle}>
          <span style={{ fontSize: 18 }}>📁</span> Registry Version Ledger
        </div>
        
        {loading && versions.length === 0 ? (
          <div style={{ textAlign: "center", padding: "40px 0", color: "#64748b" }}>Loading model versions...</div>
        ) : versions.length === 0 ? (
          <div style={{ 
            textAlign: "center", 
            padding: "40px 0", 
            color: "#64748b",
            fontSize: 13,
            border: "1px dashed rgba(255,255,255,0.05)",
            borderRadius: 16
          }}>
            No models have been registered in "churn_model" yet. Train a model and click "Find & Promote" above.
          </div>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13, textAlign: "left" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
                  <th style={{ padding: "12px 16px", color: "#475569", fontWeight: 700 }}>VERSION</th>
                  <th style={{ padding: "12px 16px", color: "#475569", fontWeight: 700 }}>CURRENT STAGE</th>
                  <th style={{ padding: "12px 16px", color: "#475569", fontWeight: 700 }}>ALGORITHM</th>
                  <th style={{ padding: "12px 16px", color: "#475569", fontWeight: 700 }}>ACCURACY</th>
                  <th style={{ padding: "12px 16px", color: "#475569", fontWeight: 700 }}>DESCRIPTION</th>
                </tr>
              </thead>
              <tbody>
                {versions.map((v, i) => (
                  <tr 
                    key={i} 
                    style={{ 
                      borderBottom: "1px solid rgba(255,255,255,0.02)",
                      background: i % 2 === 0 ? "transparent" : "rgba(255,255,255,0.01)",
                    }}
                  >
                    <td style={{ padding: "16px", fontWeight: 800 }}>v{v.version}</td>
                    <td style={{ padding: "16px" }}>
                      <span style={{
                        background: v.stage === "Production" ? "rgba(16, 185, 129, 0.1)" : v.stage === "Staging" ? "rgba(245, 158, 11, 0.1)" : "rgba(255,255,255,0.02)",
                        color: v.stage === "Production" ? "#34d399" : v.stage === "Staging" ? "#fbbf24" : "#94a3b8",
                        padding: "4px 10px",
                        borderRadius: 8,
                        fontSize: 11,
                        fontWeight: 800,
                        border: `1px solid ${v.stage === "Production" ? "rgba(16, 185, 129, 0.2)" : v.stage === "Staging" ? "rgba(245, 158, 11, 0.2)" : "rgba(255,255,255,0.05)"}`
                      }}>
                        {v.stage.toUpperCase()}
                      </span>
                    </td>
                    <td style={{ padding: "16px", fontWeight: 600, color: "#c7d2fe" }}>
                      {v.tags.algorithm ? v.tags.algorithm.toUpperCase().replace("_", " ") : "UNKNOWN"}
                    </td>
                    <td style={{ padding: "16px", fontWeight: 800, color: "#34d399" }}>
                      {v.tags.accuracy ? `${(parseFloat(v.tags.accuracy) * 100).toFixed(2)}%` : "N/A"}
                    </td>
                    <td style={{ padding: "16px", color: "#64748b", maxWidth: 300, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {v.description || "No description provided."}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
