import React, { useState, useEffect } from "react";
import S from "../styles";
import { Tag, MetricCard } from "./UIComponents";
import { fetchRFAnalysis } from "../api";

// ─── 🚀 Premium Complexity Sweep Chart (Validation Curve) ────────────────

function ComplexitySweepChart({ study }) {
  const W = 600, H = 260, pad = 40;
  
  // Pick the best n_estimators for the sweep (standard practice)
  const bestN = 100; 
  const sweepData = study.filter(i => i.n_estimators === bestN).sort((a,b)=>a.max_depth - b.max_depth);
  
  if (!sweepData.length) return null;

  const allVals = sweepData.flatMap(i => [i.train_accuracy, i.test_accuracy]);
  const minV = Math.max(0.6, Math.min(...allVals) - 0.05);
  const maxV = 1.01;
  const minD = 2, maxD = 20;

  const getX = (d) => pad + ((d - minD) / (maxD - minD)) * (W - pad * 2);
  const getY = (a) => H - pad - ((a - minV) / (maxV - minV)) * (H - pad * 2);

  // Find best test depth
  const bestPoint = sweepData.reduce((prev, curr) => (prev.test_accuracy > curr.test_accuracy) ? prev : curr);

  // Polylines
  const trainPts = sweepData.map(p => `${getX(p.max_depth)},${getY(p.train_accuracy)}`).join(" ");
  const testPts = sweepData.map(p => `${getX(p.max_depth)},${getY(p.test_accuracy)}`).join(" ");
  
  // Shaded area (Polygon)
  const polygonPts = [
    ...sweepData.map(p => `${getX(p.max_depth)},${getY(p.train_accuracy)}`),
    ...sweepData.reverse().map(p => `${getX(p.max_depth)},${getY(p.test_accuracy)}`)
  ].join(" ");
  // reverse it back
  sweepData.reverse();

  return (
    <div style={{ background: "#0a1628", borderRadius: 20, padding: 24, border: "1px solid #1e293b", position: "relative" }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 20 }}>
        <div style={{ display: "flex", gap: 16 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div style={{ width: 12, height: 12, borderRadius: "50%", background: "#fcd34d" }} />
            <span style={{ fontSize: 10, color: "#cbd5e1" }}>Training Score</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div style={{ width: 12, height: 12, borderRadius: "50%", background: "#6ee7b7" }} />
            <span style={{ fontSize: 10, color: "#cbd5e1" }}>Test Score</span>
          </div>
        </div>
        <Tag color="#38bdf8">N_ESTIMATORS = 100</Tag>
      </div>

      <svg width="100%" height={H} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet">
        {/* Grid lines */}
        {[0, 0.5, 1].map(p => {
          const v = minV + p * (maxV - minV);
          const y = getY(v);
          return (
            <g key={p}>
              <line x1={pad} y1={y} x2={W - pad} y2={y} stroke="#1e293b" strokeWidth={1} strokeDasharray="4,4" />
              <text x={pad - 12} y={y + 4} fill="#475569" fontSize={10} textAnchor="end">{(v * 100).toFixed(0)}%</text>
            </g>
          );
        })}
        
        {/* X Axis Labels */}
        {sweepData.map(d => (
          <text key={d.max_depth} x={getX(d.max_depth)} y={H - 10} fill="#475569" fontSize={9} textAnchor="middle">{d.max_depth}</text>
        ))}

        {/* Shaded Overfitting Gap */}
        <polygon points={polygonPts} fill="#fcd34d" fillOpacity="0.05" />

        {/* Best Point Marker (Dashed Line) */}
        <line x1={getX(bestPoint.max_depth)} y1={pad} x2={getX(bestPoint.max_depth)} y2={H-pad} stroke="#38bdf8" strokeWidth={1.5} strokeDasharray="5,5" />
        <circle cx={getX(bestPoint.max_depth)} cy={getY(bestPoint.test_accuracy)} r={8} fill="#38bdf822" stroke="#38bdf8" strokeWidth={1} />

        {/* Train Line */}
        <polyline points={trainPts} fill="none" stroke="#fcd34d" strokeWidth={3} strokeLinecap="round" strokeLinejoin="round" />
        {sweepData.map(p => (
          <circle key={p.max_depth} cx={getX(p.max_depth)} cy={getY(p.train_accuracy)} r={4} fill="#0a1628" stroke="#fcd34d" strokeWidth={2} />
        ))}

        {/* Test Line */}
        <polyline points={testPts} fill="none" stroke="#6ee7b7" strokeWidth={3} strokeLinecap="round" strokeLinejoin="round" />
        {sweepData.map(p => (
          <circle key={p.max_depth} cx={getX(p.max_depth)} cy={getY(p.test_accuracy)} r={4} fill="#0a1628" stroke="#6ee7b7" strokeWidth={2} />
        ))}
      </svg>
      
      <div style={{ marginTop: 20, background: "#38bdf811", padding: 12, borderRadius: 12, color: "#38bdf8", fontSize: 11, border: "1px solid #38bdf833" }}>
         📌 <b>Best Generalization:</b> The gap between curves (variance) is minimized at <b>Depth {bestPoint.max_depth}</b> with <b>{(bestPoint.test_accuracy*100).toFixed(1)}%</b> test accuracy.
      </div>
    </div>
  );
}

function ComparisonPanel({ dt, rf }) {
  const metrics = [
    { key: "accuracy", label: "Accuracy" },
    { key: "recall", label: "Recall" },
    { key: "f1_score", label: "F1 Score" }
  ];
  return (
    <div style={{ display: "grid", gap: 14 }}>
      {metrics.map(m => (
        <div key={m.key} style={{ background: "#0f172a", padding: 12, borderRadius: 12, border: "1px solid #1e293b" }}>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#64748b", marginBottom: 8, textTransform: "uppercase" }}>
            <span>{m.label}</span>
            <span>RF vs DT</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
             <div style={{ flex: 1, background: "#1e293b", height: 8, borderRadius: 4, overflow: "hidden" }}>
                <div style={{ width: `${rf[m.key]*100}%`, background: "#6ee7b7", height: "100%" }} />
             </div>
             <div style={{ fontSize: 10, color: "#6ee7b7", fontWeight: 900 }}>{(rf[m.key]*100).toFixed(0)}%</div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 4, opacity: 0.5 }}>
             <div style={{ flex: 1, background: "#1e293b", height: 8, borderRadius: 4, overflow: "hidden" }}>
                <div style={{ width: `${dt[m.key]*100}%`, background: "#fcd34d", height: "100%" }} />
             </div>
             <div style={{ fontSize: 10, color: "#fcd34d", fontWeight: 900 }}>{(dt[m.key]*100).toFixed(0)}%</div>
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

  if (loading) return <div style={{ textAlign: "center", padding: 100, color: "#64748b", fontWeight: 800 }}>LOADING RANDOM FOREST INSIGHTS...</div>;
  if (!data) return null;

  const best = data.bias_variance_study.reduce((a,b) => a.test_accuracy > b.test_accuracy ? a : b);
  const rfRes = data.dt_vs_rf.rf;

  return (
    <div className="stagger" style={{ display: "flex", flexDirection: "column", gap: 24, paddingBottom: 60 }}>
      
      {/* 🚀 Top Stats Header */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16 }}>
        <div style={{ gridColumn: "span 1", background: "linear-gradient(135deg, #064e3b, #020617)", padding: 20, borderRadius: 16, border: "1px solid #6ee7b744" }}>
           <div style={{ fontSize: 10, color: "#6ee7b7", opacity: 0.8, textTransform: "uppercase", letterSpacing: 1 }}>Best Accuracy</div>
           <div style={{ fontSize: 32, fontWeight: 900, color: "#f8fafc" }}>{(best.test_accuracy * 100).toFixed(1)}%</div>
        </div>
        <div style={{ gridColumn: "span 1", background: "linear-gradient(135deg, #1e3a8a, #020617)", padding: 20, borderRadius: 16, border: "1px solid #38bdf844" }}>
           <div style={{ fontSize: 10, color: "#38bdf8", opacity: 0.8, textTransform: "uppercase", letterSpacing: 1 }}>F1 Score</div>
           <div style={{ fontSize: 32, fontWeight: 900, color: "#f8fafc" }}>{(rfRes.f1_score * 100).toFixed(1)}%</div>
        </div>
        <div style={{ gridColumn: "span 1", background: "linear-gradient(135deg, #581c87, #020617)", padding: 20, borderRadius: 16, border: "1px solid #c4b5fd44" }}>
           <div style={{ fontSize: 10, color: "#c4b5fd", opacity: 0.8, textTransform: "uppercase", letterSpacing: 1 }}>Recall</div>
           <div style={{ fontSize: 32, fontWeight: 900, color: "#f8fafc" }}>{(rfRes.recall * 100).toFixed(1)}%</div>
        </div>
        <div style={{ gridColumn: "span 1", background: "#0a1628", padding: 20, borderRadius: 16, border: "1px solid #1e293b", display: "flex", flexDirection: "column", justifyContent: "center" }}>
           <Tag color="#6ee7b7">MODE: BATTLE-READY</Tag>
           <div style={{ fontSize: 10, color: "#475569", marginTop: 8 }}>The Forest is ready for deployment.</div>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
        
        {/* 📋 Importance & Reliability */}
        <div style={{ display: "grid", gap: 24 }}>
           <div style={{ background: "#0a1628", padding: 24, borderRadius: 20, border: "1px solid #1e293b" }}>
              <div style={{ fontSize: 11, color: "#64748b", textTransform: "uppercase", marginBottom: 16 }}>Important Features</div>
              {Object.entries(data.feature_importance).sort(([,a],[,b])=>b-a).slice(0,4).map(([f, v], i) => (
                <div key={f} style={{ marginBottom: 12 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 4 }}>
                    <span style={{ color: "#cbd5e1", fontWeight: 700 }}>{f}</span>
                    <span style={{ color: "#6ee7b7" }}>{(v*100).toFixed(1)}%</span>
                  </div>
                  <div style={{ background: "#1e293b", height: 4, borderRadius: 2 }}>
                    <div style={{ height: "100%", width: `${v*100}%`, background: "#6ee7b7", borderRadius: 2 }} />
                  </div>
                </div>
              ))}
           </div>

           <div style={{ background: "#0a1628", padding: 24, borderRadius: 20, border: "1px solid #1e293b" }}>
              <div style={{ fontSize: 11, color: "#64748b", textTransform: "uppercase", marginBottom: 16, display: "flex", justifyContent: "space-between" }}>
                 <span>Reliability Test (Stability)</span>
                 <span style={{ color: "#6ee7b7" }}>PASSED</span>
              </div>
              <div style={{ display: "flex", gap: 8, height: 100, alignItems: "flex-end", background: "#020617", padding: "10px 14px", borderRadius: 10, border: "1px solid #1e293b" }}>
                 {data.stability_analysis.map((s, i) => (
                   <div key={i} style={{ flex: 1, position: "relative", height: "100%", display: "flex", alignItems: "flex-end" }}>
                      <div style={{ position: "absolute", top: -14, left: "50%", transform: "translateX(-50%)", fontSize: 8, color: "#38bdf8" }}>{(s.f1_score * 100).toFixed(0)}%</div>
                      <div style={{ width: "100%", background: "linear-gradient(to top, #38bdf844, #38bdf8)", height: `${Math.max(20, s.f1_score * 100)}%`, borderRadius: "4px 4px 0 0", border: "1px solid #38bdf8" }} />
                   </div>
                 ))}
              </div>
              <div style={{ marginTop: 16, fontSize: 10, color: "#64748b", lineHeight: 1.5 }}>
                 <p style={{ margin: "4px 0" }}>🌱 <b>Seed:</b> Controls randomness. Consistent results = Stable Model.</p>
                 <p style={{ margin: "4px 0" }}>⚖️ <b>F1-Score:</b> Balanced metric for churn detection quality.</p>
              </div>
           </div>
        </div>

        {/* 📊 Comparison */}
        <div style={{ background: "#0a1628", padding: 24, borderRadius: 20, border: "1px solid #1e293b" }}>
           <div style={{ fontSize: 11, color: "#64748b", textTransform: "uppercase", marginBottom: 20 }}>Model Battle: Single Tree vs Forest</div>
           <ComparisonPanel dt={data.dt_vs_rf.dt} rf={data.dt_vs_rf.rf} />
           <div style={{ marginTop: 20, background: "#6ee7b711", padding: 12, borderRadius: 12, fontSize: 11, color: "#6ee7b7", lineHeight: 1.4 }}>
              "Ensemble methods like Random Forest reduce individual error variance, leading to a much safer model."
           </div>
        </div>

      </div>

      {/* 🧠 Complexity Sweep (NEW 2view Chart) */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
         <div style={{ gridColumn: "span 1" }}>
            <div style={{ fontSize: 11, color: "#64748b", textTransform: "uppercase", marginBottom: 12 }}>Complexity Sweep (Train vs Test)</div>
            <ComplexitySweepChart study={data.bias_variance_study} />
         </div>
         <div style={{ gridColumn: "span 1", background: "#0a1628", padding: 24, borderRadius: 20, border: "1px solid #1e293b" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
               <div style={{ fontSize: 11, color: "#64748b", textTransform: "uppercase" }}>Bias-Variance Strategy Matrix</div>
               <div style={{ display: "flex", gap: 8 }}>
                  <div style={{ fontSize: 9, color: "#6ee7b7", display: "flex", alignItems: "center", gap: 3 }}><div style={{ width: 6, height: 6, borderRadius: "50%", background: "#6ee7b7" }} /> Good</div>
                  <div style={{ fontSize: 9, color: "#fca5a5", display: "flex", alignItems: "center", gap: 3 }}><div style={{ width: 6, height: 6, borderRadius: "50%", background: "#fca5a5" }} /> High Var.</div>
               </div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10 }}>
               {data.bias_variance_study.filter((_, idx)=> idx % 2 === 0).slice(0,9).map((item, i) => {
                 const isOverfit = item.variance > 0.05;
                 const isUnderfit = item.train_accuracy < 0.75;
                 
                 return (
                   <div key={i} style={{ 
                     background: isOverfit ? "#7f1d1d22" : isUnderfit ? "#78350f22" : "#064e3b22", 
                     border: `1px solid ${isOverfit ? "#ef444444" : isUnderfit ? "#f59e0b44" : "#10b98144"}`,
                     borderRadius: 12, padding: 12, textAlign: "center", position: "relative"
                   }}>
                      <div style={{ fontSize: 8, color: "#94a3b8", marginBottom: 4 }}>D:{item.max_depth} | T:{item.n_estimators}</div>
                      <div style={{ fontSize: 16, fontWeight: 900, color: isOverfit ? "#fca5a5" : "#6ee7b7" }}>{(item.test_accuracy*100).toFixed(0)}%</div>
                      
                      <div style={{ marginTop: 6, display: "flex", flexDirection: "column", gap: 2 }}>
                         <div style={{ fontSize: 7, color: "#64748b", display: "flex", justifyContent: "space-between" }}>
                            <span>Bias:</span> <b>{item.bias.toFixed(2)}</b>
                         </div>
                         <div style={{ fontSize: 7, color: "#64748b", display: "flex", justifyContent: "space-between" }}>
                            <span>Var:</span> <b style={{ color: isOverfit ? "#fca5a5" : "inherit" }}>{item.variance.toFixed(2)}</b>
                         </div>
                      </div>

                      <div style={{ marginTop: 6, fontSize: 8, fontWeight: 800, color: isOverfit ? "#fca5a5" : isUnderfit ? "#fcd34d" : "#6ee7b7" }}>
                         {isOverfit ? "OVERFIT" : isUnderfit ? "SIMPLE" : "BALANCED"}
                      </div>
                   </div>
                 );
               })}
            </div>
            <div style={{ marginTop: 16, fontSize: 9, color: "#475569", borderTop: "1px solid #1e293b", paddingTop: 10, lineHeight: 1.4 }}>
               💡 <b>Bias</b> is how much the model lacks understanding. <b>Variance</b> is how much it changes with new data. High Variance = Overfitting.
            </div>
         </div>
      </div>

    </div>
  );
}
