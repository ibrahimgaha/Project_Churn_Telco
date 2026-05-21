import React, { useState } from "react";
import S from "../styles";
import { runDriftAnalysis } from "../api";

export default function DriftTab() {
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const handleRunAnalysis = async () => {
    setAnalyzing(true);
    setResult(null);
    setError("");
    try {
      const res = await runDriftAnalysis();
      setResult(res);
    } catch (e) {
      setError(e.message || "Drift analysis failed. Make sure dependencies are fully installed.");
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 32 }}>
      {/* Drift Control Panel */}
      <div className="glass-card" style={S.section}>
        <div style={S.sectionTitle}>
          <span style={{ fontSize: 18 }}>📊</span> Evidently Data Drift Control Panel
        </div>
        <p style={{ color: "#64748b", fontSize: 13, lineHeight: 1.5, margin: "10px 0 24px 0" }}>
          In production systems, incoming customer behavior shifts over time (Data Drift), causing models 
          to degrade. Clicking below will load the baseline data, simulate a realistic pricing/tenure shift 
          on incoming customer records, compute an <strong>Evidently Data Drift & Quality Report</strong>, and 
          run <strong>Kolmogorov-Smirnov (KS) tests</strong> for every numerical feature.
        </p>

        <button
          onClick={handleRunAnalysis}
          disabled={analyzing}
          style={{
            background: analyzing ? "rgba(244, 63, 94, 0.4)" : "linear-gradient(135deg, #f43f5e 0%, #be123c 100%)",
            color: "#fff",
            border: "none",
            borderRadius: 14,
            padding: "14px 28px",
            fontSize: 14,
            fontWeight: 750,
            cursor: analyzing ? "not-allowed" : "pointer",
            boxShadow: "0 4px 20px rgba(244, 63, 94, 0.3)",
            transition: "all 0.3s ease",
          }}
        >
          {analyzing ? "🔍 Running Evidently & KS-Tests..." : "⚡ Simulate & Detect Data Drift"}
        </button>

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
      </div>

      {result && (
        <>
          {/* Main Indicators Grid */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 20 }}>
            {/* Status Indicator */}
            <div className="glass-card" style={{ ...S.section, margin: 0, padding: 24, textAlign: "center" }}>
              <div style={{ fontSize: 12, color: "#64748b", fontWeight: 700 }}>DATASET DRIFT STATUS</div>
              <div style={{ 
                fontSize: 24, 
                fontWeight: 900, 
                color: result.dataset_drifted ? "#f43f5e" : "#10b981", 
                marginTop: 12,
                textShadow: result.dataset_drifted ? "0 0 20px rgba(244, 63, 94, 0.4)" : "0 0 20px rgba(16, 185, 129, 0.4)"
              }}>
                {result.dataset_drifted ? "🚨 DRIFTED" : "✅ STABLE"}
              </div>
              <div style={{ fontSize: 11, color: "#475569", marginTop: 8 }}>
                Evidently DatasetDriftMetric
              </div>
            </div>

            {/* Drift Share Indicator */}
            <div className="glass-card" style={{ ...S.section, margin: 0, padding: 24 }}>
              <div style={{ fontSize: 12, color: "#64748b", fontWeight: 700, textAlign: "center" }}>DRIFT SHARE</div>
              <div style={{ 
                fontSize: 24, 
                fontWeight: 900, 
                color: "#c7d2fe", 
                marginTop: 12,
                textAlign: "center"
              }}>
                {(result.drift_share * 100).toFixed(0)}%
              </div>
              
              {/* Progress Bar */}
              <div style={{ width: "100%", height: 6, background: "rgba(255,255,255,0.05)", borderRadius: 3, marginTop: 12, overflow: "hidden" }}>
                <div style={{ 
                  width: `${result.drift_share * 100}%`, 
                  height: "100%", 
                  background: result.drift_share > 0.3 ? "#f43f5e" : "#818cf8",
                  boxShadow: `0 0 10px ${result.drift_share > 0.3 ? "#f43f5e" : "#818cf8"}`
                }} />
              </div>
            </div>

            {/* Retraining Indicator */}
            <div className="glass-card" style={{ ...S.section, margin: 0, padding: 24, textAlign: "center" }}>
              <div style={{ fontSize: 12, color: "#64748b", fontWeight: 700 }}>AUTO-RETRAINING</div>
              <div style={{ 
                fontSize: 16, 
                fontWeight: 800, 
                color: result.retrain_triggered ? "#34d399" : "#64748b", 
                marginTop: 18 
              }}>
                {result.retrain_triggered ? "🔄 TRIGGERED (Active)" : "😴 IDLE (Skipped)"}
              </div>
              <div style={{ fontSize: 11, color: "#475569", marginTop: 10 }}>
                Threshold: Retrain &gt; 30% · Warn &gt; 15%
              </div>
            </div>
          </div>

          {/* Retraining Message Card */}
          <div className="glass-card" style={{ 
            ...S.section, 
            background: result.retrain_triggered ? "rgba(16, 185, 129, 0.05)" : "rgba(245, 158, 11, 0.05)",
            border: `1px solid ${result.retrain_triggered ? "rgba(16, 185, 129, 0.2)" : "rgba(245, 158, 11, 0.2)"}`
          }}>
            <h4 style={{ margin: "0 0 8px 0", color: result.retrain_triggered ? "#34d399" : "#fbbf24", fontSize: 14, fontWeight: 800 }}>
              📢 Drift Orchestrator Log
            </h4>
            <div style={{ fontSize: 13, color: "#f1f5f9", fontWeight: 500 }}>
              {result.message}
            </div>
            <div style={{ display: "flex", gap: 16, marginTop: 16, fontSize: 11, color: "#64748b", fontFamily: "monospace" }}>
              <div>Run ID: {result.run_id}</div>
              <div>Report: drift_report.html (Saved & Logged)</div>
            </div>
          </div>

          {/* Kolmogorov-Smirnov Feature Breakdown */}
          <div className="glass-card" style={S.section}>
            <div style={S.sectionTitle}>
              <span style={{ fontSize: 18 }}>🌲</span> Kolmogorov-Smirnov (KS-test) Feature Analysis
            </div>
            <p style={{ color: "#64748b", fontSize: 12, margin: "8px 0 20px 0" }}>
              Below is the feature-level statistical breakdown. A features is flagged as <strong>Drifted</strong> if 
              the KS test p-value is <strong>&lt; 0.05</strong>, meaning the current data distribution deviates 
              significantly from the training reference data.
            </p>

            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13, textAlign: "left" }}>
                <thead>
                  <tr style={{ borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
                    <th style={{ padding: "12px 16px", color: "#475569", fontWeight: 700 }}>FEATURE NAME</th>
                    <th style={{ padding: "12px 16px", color: "#475569", fontWeight: 700 }}>KS STATISTIC</th>
                    <th style={{ padding: "12px 16px", color: "#475569", fontWeight: 700 }}>P-VALUE</th>
                    <th style={{ padding: "12px 16px", color: "#475569", fontWeight: 700 }}>DRIFT STATUS</th>
                  </tr>
                </thead>
                <tbody>
                  {result.ks_results.map((ks, i) => (
                    <tr 
                      key={i} 
                      style={{ 
                        borderBottom: "1px solid rgba(255,255,255,0.02)",
                        background: i % 2 === 0 ? "transparent" : "rgba(255,255,255,0.01)",
                      }}
                    >
                      <td style={{ padding: "16px", fontWeight: 700, color: "#f1f5f9" }}>{ks.feature}</td>
                      <td style={{ padding: "16px", fontFamily: "monospace" }}>{ks.ks_statistic.toFixed(4)}</td>
                      <td style={{ padding: "16px", fontFamily: "monospace", color: ks.p_value < 0.05 ? "#fca5a5" : "#a7f3d0" }}>
                        {ks.p_value.toFixed(6)}
                      </td>
                      <td style={{ padding: "16px" }}>
                        <span style={{
                          background: ks.drift_detected ? "rgba(244, 63, 94, 0.1)" : "rgba(16, 185, 129, 0.1)",
                          color: ks.drift_detected ? "#f43f5e" : "#34d399",
                          padding: "4px 10px",
                          borderRadius: 8,
                          fontSize: 10,
                          fontWeight: 800,
                          border: `1px solid ${ks.drift_detected ? "rgba(244, 63, 94, 0.2)" : "rgba(16, 185, 129, 0.2)"}`
                        }}>
                          {ks.drift_detected ? "🚨 DRIFTED" : "✅ STABLE"}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
