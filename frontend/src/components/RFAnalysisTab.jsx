import React, { useState, useEffect } from "react";
import S from "../styles";
import { Tag } from "./UIComponents";
import { fetchRFAnalysis } from "../api";
import MODEL_INFO from "../models";

// ─── 🚀 Complexity Curve ────────────────

function ComplexitySweepChart({ study }) {
  const W = 600, H = 240, pad = 40;
  const sweepData = study.filter(i => i.n_estimators === 100).sort((a,b)=>a.max_depth - b.max_depth);
  if (!sweepData.length) return null;

  const minV = 0.6, maxV = 1.0;
  const minD = 2, maxD = 20;

  const getX = (d) => pad + ((d - minD) / (maxD - minD)) * (W - pad * 2);
  const getY = (a) => H - pad - ((a - minV) / (maxV - minV)) * (H - pad * 2);

  const trainPts = sweepData.map(p => `${getX(p.max_depth)},${getY(p.train_accuracy)}`).join(" ");
  const testPts = sweepData.map(p => `${getX(p.max_depth)},${getY(p.test_accuracy)}`).join(" ");

  return (
    <div style={{ background: "#0a1628", borderRadius: 20, padding: 24, border: "1px solid #1e293b" }}>
      <div style={{ display: "flex", gap: 20, marginBottom: 15 }}>
         <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div style={{ width: 10, height: 10, borderRadius: "50%", background: "#fcd34d" }} />
            <span style={{ fontSize: 10, color: "#64748b" }}>Train</span>
         </div>
         <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div style={{ width: 10, height: 10, borderRadius: "50%", background: "#6ee7b7" }} />
            <span style={{ fontSize: 10, color: "#64748b" }}>Test</span>
         </div>
      </div>
      <svg width="100%" height={H} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet">
        {[0, 0.5, 1].map(p => {
          const y = getY(minV + p * (maxV - minV));
          return <line key={p} x1={pad} y1={y} x2={W - pad} y2={y} stroke="#1e293b" strokeDasharray="4,4" />;
        })}
        <polyline points={trainPts} fill="none" stroke="#fcd34d" strokeWidth={2} />
        <polyline points={testPts} fill="none" stroke="#6ee7b7" strokeWidth={2} />
        {sweepData.map(p => (
          <React.Fragment key={p.max_depth}>
            <circle cx={getX(p.max_depth)} cy={getY(p.train_accuracy)} r={3} fill="#fcd34d" />
            <circle cx={getX(p.max_depth)} cy={getY(p.test_accuracy)} r={3} fill="#6ee7b7" />
          </React.Fragment>
        ))}
      </svg>
    </div>
  );
}

// ─── ⚔️ Model Battle Panel ────────────────

function ModelComparisonPanel({ dt, rf, xgb }) {
  const metrics = [
    { key: "accuracy", label: "Accuracy" },
    { key: "f1_score", label: "F1 Score" }
  ];
  
  const models = [
    { key: "dt", label: "Decision Tree", color: MODEL_INFO.decision_tree.color },
    { key: "rf", label: "Random Forest", color: MODEL_INFO.random_forest.color },
    { key: "xgb", label: "XGBoost", color: MODEL_INFO.xgboost.color }
  ];

  const data = { dt, rf, xgb };

  return (
    <div style={{ display: "grid", gap: 20 }}>
      {metrics.map(m => (
        <div key={m.key}>
          <div style={{ fontSize: 10, color: "#64748b", fontWeight: 800, textTransform: "uppercase", marginBottom: 10 }}>{m.label}</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {models.map(mod => (
              <div key={mod.key} style={{ display: "flex", alignItems: "center", gap: 12 }}>
                <div style={{ width: 90, fontSize: 10, color: "#94a3b8" }}>{mod.label}</div>
                <div style={{ flex: 1, background: "#1e293b", height: 8, borderRadius: 4, overflow: "hidden" }}>
                  <div style={{ width: `${data[mod.key][m.key]*100}%`, background: mod.color, height: "100%" }} />
                </div>
                <div style={{ width: 40, fontSize: 11, color: mod.color, fontWeight: 900 }}>{(data[mod.key][m.key]*100).toFixed(0)}%</div>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

export default function RFAnalysisTab() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchRFAnalysis().then(setData).finally(() => setLoading(false));
  }, []);

  if (loading) return <div style={{ textAlign: "center", padding: 100, color: "#64748b" }}>Loading analysis...</div>;
  if (!data) return null;

  return (
    <div className="stagger" style={{ display: "flex", flexDirection: "column", gap: 24, paddingBottom: 60 }}>
      
      {/* Metrics Row */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16 }}>
        <div style={{ background: "linear-gradient(135deg, #0a1628, #020617)", padding: 24, borderRadius: 20, border: "1px solid #1e293b" }}>
           <div style={{ fontSize: 10, color: "#64748b", textTransform: "uppercase", marginBottom: 8 }}>Top Performer</div>
           <div style={{ fontSize: 32, fontWeight: 900, color: MODEL_INFO.xgboost.color }}>XGBoost</div>
        </div>
        <div style={{ background: "linear-gradient(135deg, #0a1628, #020617)", padding: 24, borderRadius: 20, border: "1px solid #1e293b" }}>
           <div style={{ fontSize: 10, color: "#64748b", textTransform: "uppercase", marginBottom: 8 }}>Max F1-Score</div>
           <div style={{ fontSize: 32, fontWeight: 900, color: "#fff" }}>{(data.dt_vs_rf.xgb.f1_score * 100).toFixed(1)}%</div>
        </div>
        <div style={{ background: "linear-gradient(135deg, #0a1628, #020617)", padding: 24, borderRadius: 20, border: "1px solid #1e293b" }}>
           <div style={{ fontSize: 10, color: "#64748b", textTransform: "uppercase", marginBottom: 8 }}>Stability</div>
           <div style={{ fontSize: 32, fontWeight: 900, color: "#6ee7b7" }}>High</div>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1.2fr 0.8fr", gap: 24 }}>
        {/* Model Battle */}
        <div style={{ background: "#0a1628", padding: 24, borderRadius: 24, border: "1px solid #1e293b" }}>
           <div style={{ ...S.sectionTitle, marginBottom: 20 }}>Model Battle: Tree vs Ensembles</div>
           <ModelComparisonPanel dt={data.dt_vs_rf.dt} rf={data.dt_vs_rf.rf} xgb={data.dt_vs_rf.xgb} />
        </div>

        {/* Feature Importance */}
        <div style={{ background: "#0a1628", padding: 24, borderRadius: 24, border: "1px solid #1e293b" }}>
           <div style={{ ...S.sectionTitle, marginBottom: 20 }}>Key Predictors</div>
           {Object.entries(data.feature_importance).sort(([,a],[,b])=>b-a).slice(0,5).map(([f, v]) => (
              <div key={f} style={{ marginBottom: 15 }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, marginBottom: 5 }}>
                  <span style={{ color: "#94a3b8" }}>{f}</span>
                  <span style={{ color: "#6ee7b7", fontWeight: 800 }}>{(v*100).toFixed(1)}%</span>
                </div>
                <div style={{ background: "#1e293b", height: 4, borderRadius: 2 }}>
                  <div style={{ height: "100%", width: `${v*100}%`, background: "#6ee7b7", borderRadius: 2 }} />
                </div>
              </div>
           ))}
        </div>
      </div>

      {/* Complexity Sweep */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
         <div>
            <div style={{ ...S.sectionTitle, marginBottom: 12 }}>Complexity Sweep (Random Forest)</div>
            <ComplexitySweepChart study={data.bias_variance_study} />
         </div>
         <div style={{ background: "#0a1628", padding: 24, borderRadius: 24, border: "1px solid #1e293b" }}>
            <div style={S.sectionTitle}>Bias-Variance Matrix</div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10, marginTop: 20 }}>
               {data.bias_variance_study.filter((_, idx)=> idx % 4 === 0).slice(0,9).map((item, i) => (
                 <div key={i} style={{ 
                   background: "#0f172a", 
                   border: `1px solid ${item.variance > 0.05 ? "#ef444444" : "#10b98144"}`,
                   borderRadius: 12, padding: 12, textAlign: "center"
                 }}>
                    <div style={{ fontSize: 16, fontWeight: 900, color: item.variance > 0.05 ? "#fca5a5" : "#6ee7b7" }}>{(item.test_accuracy*100).toFixed(0)}%</div>
                    <div style={{ fontSize: 8, color: "#64748b", marginTop: 4 }}>Depth: {item.max_depth}</div>
                 </div>
               ))}
            </div>
         </div>
      </div>

      {/* Error Analysis */}
      <div style={{ background: "#0a1628", padding: 24, borderRadius: 24, border: "1px solid #1e293b" }}>
         <div style={S.sectionTitle}>Error Profile</div>
         <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16, marginTop: 15 }}>
            {data.error_analysis.map((err, i) => (
               <div key={i} style={{ background: "#0f172a", padding: 16, borderRadius: 16, border: "1px solid #1e293b" }}>
                  <div style={{ fontSize: 10, color: "#64748b", marginBottom: 8 }}>MISCLASSIFIED #{err.index}</div>
                  <div style={{ fontSize: 11, color: "#cbd5e1" }}>Actual: <b>{err.actual === 1 ? "Churn" : "Stay"}</b></div>
                  <div style={{ fontSize: 11, color: "#ef4444" }}>Pred: <b>{err.predicted === 1 ? "Churn" : "Stay"}</b></div>
               </div>
            ))}
         </div>
      </div>

    </div>
  );
}
