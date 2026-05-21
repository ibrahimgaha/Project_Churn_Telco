import React from "react";
import MODEL_INFO from "../models";
import S from "../styles";
import { MetricCard, ConfusionMatrix, SimpleROC, MetricsBar, BestModelBanner, Tag } from "./UIComponents";

export default function ResultsTab({ results }) {
  if (!results.length) {
    return (
      <div style={{ ...S.section, color: "#94a3b8", textAlign: "center", padding: "120px 20px" }} className="glass-card">
        <div style={{ fontSize: 48, marginBottom: 20 }}>📊</div>
        <p style={{ fontSize: 20, fontWeight: 700, color: "#fff", marginBottom: 8 }}>Experimental Sandbox is Empty</p>
        <p style={{ fontSize: 14 }}>Initialize your models in the <b>Training</b> tab to start gathering performance intelligence.</p>
      </div>
    );
  }

  const exportResults = () => {
    const rows = results.map(r => ({
      model: r.model_name,
      ...Object.fromEntries(Object.entries(r.metrics).filter(([k]) => typeof r.metrics[k] === "number"))
    }));
    const csv = [Object.keys(rows[0]).join(","), ...rows.map(r => Object.values(r).join(","))].join("\n");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    a.download = "churn_experiment_results.csv";
    a.click();
  };

  return (
    <div className="stagger">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 32 }}>
        <div style={S.sectionTitle}>
           <span style={{ color: "#6366f1" }}>⚙️</span> Analytics Overview
        </div>
        <button onClick={exportResults} style={S.btnOutline}>
           <span style={{ marginRight: 8 }}>⬇</span> Download CSV Report
        </button>
      </div>

      <BestModelBanner results={results} />

      <div style={S.section} className="glass-card">
        <div style={S.sectionTitle}>Global Model Benchmarking</div>
        <div style={{ marginTop: 20 }}>
          <MetricsBar results={results} />
        </div>
      </div>

      {results.map((r, idx) => {
        const color = MODEL_INFO[r.model_name]?.color || "#6366f1";
        return (
          <div 
            key={r.run_id} 
            style={{ 
              ...S.section, 
              borderTop: `4px solid ${color}`,
              background: `linear-gradient(180deg, ${color}05 0%, rgba(15, 23, 42, 0.4) 100%)`
            }}
            className="glass-card"
          >
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 32 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
                <div style={{ width: 40, height: 40, borderRadius: 12, background: `${color}15`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 20 }}>
                   {MODEL_INFO[r.model_name]?.label[0]}
                </div>
                <div>
                  <div style={{ fontSize: 20, fontWeight: 900, color: "#fff", letterSpacing: -0.5 }}>
                    {MODEL_INFO[r.model_name]?.label || r.model_name}
                  </div>
                  <div style={{ fontSize: 11, color: "#64748b", fontWeight: 700, textTransform: "uppercase", marginTop: 2 }}>
                     Instance Artifact: {r.run_id.slice(0, 12)}
                  </div>
                </div>
              </div>
              <Tag color={color}>ACTIVE RUN</Tag>
            </div>

            <div style={{ display: "flex", gap: 16, marginBottom: 40, flexWrap: "wrap" }}>
              {["accuracy", "precision", "recall", "f1_score", "roc_auc"].map(m => (
                <MetricCard 
                  key={m} 
                  label={m.replace("_", " ")} 
                  value={r.metrics[m]} 
                  color={color} 
                />
              ))}
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(360px, 1fr))", gap: 40 }}>
              <div>
                <div style={S.sectionTitle}>Performance Confusion Distribution</div>
                <div style={{ marginTop: 20 }}>
                   <ConfusionMatrix cm={r.metrics.confusion_matrix} />
                </div>
              </div>
              <div>
                <div style={S.sectionTitle}>Receiver Operating Characteristic</div>
                <div style={{ marginTop: 20, display: "flex", flexDirection: "column", alignItems: "center" }}>
                   <SimpleROC roc_curve={r.metrics.roc_curve} color={color} />
                   <div style={{ marginTop: 16, background: `${color}10`, padding: "8px 20px", borderRadius: 10, border: `1px solid ${color}22` }}>
                      <span style={{ fontSize: 12, color: "#94a3b8", fontWeight: 700 }}>AUC METRIC:</span>
                      <span style={{ fontSize: 16, fontWeight: 900, color: "#fff", marginLeft: 8 }}>{r.metrics.roc_auc.toFixed(4)}</span>
                   </div>
                </div>
              </div>
            </div>

            {r.feature_importances && (
              <div style={{ marginTop: 48, borderTop: "1px solid rgba(255,255,255,0.05)", paddingTop: 32 }}>
                <div style={S.sectionTitle}>Global Feature Contributions (Top 10)</div>
                <div style={{ display: "grid", gap: 16, marginTop: 24 }}>
                  {Object.entries(r.feature_importances)
                    .sort(([, a], [, b]) => b - a)
                    .slice(0, 10)
                    .map(([feat, imp]) => (
                      <div key={feat} style={{ display: "flex", alignItems: "center", gap: 20 }}>
                        <span style={{ fontSize: 12, color: "#cbd5e1", width: 180, fontWeight: 600, flexShrink: 0 }}>{feat}</span>
                        <div style={{ flex: 1, background: "rgba(255,255,255,0.05)", height: 8, borderRadius: 4, overflow: "hidden" }}>
                          <div style={{ 
                            height: "100%", 
                            width: `${Math.min(imp * 100, 100)}%`, 
                            background: `linear-gradient(90deg, ${color} 0%, ${color}aa 100%)`, 
                            borderRadius: 4, 
                            transition: "width 1s ease-out",
                            boxShadow: `0 0 10px ${color}33`
                          }} />
                        </div>
                        <span style={{ fontSize: 13, color: color, fontWeight: 800, width: 60, textAlign: "right" }}>{imp.toFixed(3)}</span>
                      </div>
                    ))}
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
