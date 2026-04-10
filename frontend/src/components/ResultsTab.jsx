import React from "react";
import MODEL_INFO from "../models";
import S from "../styles";
import { MetricCard, ConfusionMatrix, SimpleROC, MetricsBar, BestModelBanner, Tag } from "./UIComponents";

export default function ResultsTab({ results }) {
  if (!results.length) {
    return (
      <div style={{ ...S.section, color: "#475569", textAlign: "center", padding: "80px 20px" }}>
        <p style={{ fontSize: 18, marginBottom: 12 }}>No results yet.</p>
        <p style={{ fontSize: 14 }}>Go to the <b>Train</b> tab and start an experiment to see performance metrics.</p>
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
    <div className="fade-in">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
        <div style={S.sectionTitle}>Experiment Results</div>
        <button onClick={exportResults} style={S.btnOutline}>⬇ Export to CSV</button>
      </div>

      <BestModelBanner results={results} />

      <div style={S.section}>
        <div style={S.sectionTitle}>Quick Metrics Comparison</div>
        <MetricsBar results={results} />
      </div>

      {results.map((r, idx) => (
        <div 
          key={r.run_id} 
          style={{ 
            ...S.section, 
            borderColor: (MODEL_INFO[r.model_name]?.color || "#6ee7b7") + "44",
            borderLeft: `4px solid ${MODEL_INFO[r.model_name]?.color || "#6ee7b7"}`
          }}
          className="slide-up"
          style={{ animationDelay: `${idx * 0.1}s` }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 20 }}>
            <div style={{ fontSize: 18, fontWeight: 900, color: MODEL_INFO[r.model_name]?.color || "#6ee7b7" }}>
              {MODEL_INFO[r.model_name]?.label || r.model_name}
            </div>
            <Tag color="#475569">{r.run_id.slice(0, 8)}</Tag>
          </div>

          <div style={{ display: "flex", gap: 12, marginBottom: 24, flexWrap: "wrap" }}>
            {["accuracy", "precision", "recall", "f1_score", "roc_auc"].map(m => (
              <MetricCard 
                key={m} 
                label={m.replace("_", " ")} 
                value={r.metrics[m]} 
                color={MODEL_INFO[r.model_name]?.color || "#6ee7b7"} 
              />
            ))}
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 32 }}>
            <div>
              <div style={S.sectionTitle}>Confusion Matrix</div>
              <ConfusionMatrix cm={r.metrics.confusion_matrix} />
            </div>
            <div>
              <div style={S.sectionTitle}>ROC Curve</div>
              <SimpleROC roc_curve={r.metrics.roc_curve} color={MODEL_INFO[r.model_name]?.color} />
              <div style={{ fontSize: 11, color: "#475569", marginTop: 8, fontFamily: "monospace" }}>AUC = {r.metrics.roc_auc.toFixed(4)}</div>
            </div>
          </div>

          {r.feature_importances && (
            <div style={{ marginTop: 32 }}>
              <div style={S.sectionTitle}>Feature Importances</div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 8 }}>
                {Object.entries(r.feature_importances)
                  .sort(([, a], [, b]) => b - a)
                  .slice(0, 10)
                  .map(([feat, imp]) => (
                    <div key={feat} style={{ display: "flex", alignItems: "center", gap: 12 }}>
                      <span style={{ fontSize: 11, color: "#94a3b8", width: 140, flexShrink: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{feat}</span>
                      <div style={{ flex: 1, background: "#1e293b", height: 8, borderRadius: 4, overflow: "hidden" }}>
                        <div style={{ height: "100%", width: `${Math.min(imp * 100, 100)}%`, background: MODEL_INFO[r.model_name]?.color || "#6ee7b7", borderRadius: 4, transition: "width 0.6s ease-out" }} />
                      </div>
                      <span style={{ fontSize: 11, color: "#475569", fontFamily: "monospace", width: 50, textAlign: "right" }}>{imp.toFixed(3)}</span>
                    </div>
                  ))}
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
