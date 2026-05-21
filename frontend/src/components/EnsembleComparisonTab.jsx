import React, { useState, useEffect } from "react";
import S from "../styles";
import { Tag } from "./UIComponents";
import { fetchModels } from "../api";
import MODEL_INFO from "../models";

// ─── 📐 SVG Schemas ─────────────────────────────────────────────────────────

function BaggingSchema() {
  return (
    <div style={{ background: "#0a1628", padding: 20, borderRadius: 16, border: "1px solid #1e293b", textAlign: "center" }}>
      <div style={{ fontSize: 11, color: "#64748b", textTransform: "uppercase", marginBottom: 15, fontWeight: 800 }}>Bagging Architecture (Random Forest)</div>
      <svg width="100%" height="160" viewBox="0 0 400 160">
        {/* Input Data */}
        <rect x="10" y="60" width="60" height="40" rx="4" fill="#1e293b" stroke="#334155" />
        <text x="40" y="85" fill="#cbd5e1" fontSize="10" textAnchor="middle">Dataset</text>
        
        {/* Parallel Trees */}
        <path d="M 70 80 L 120 30" stroke="#475569" strokeWidth="1.5" markerEnd="url(#arrow)" />
        <path d="M 70 80 L 120 80" stroke="#475569" strokeWidth="1.5" markerEnd="url(#arrow)" />
        <path d="M 70 80 L 120 130" stroke="#475569" strokeWidth="1.5" markerEnd="url(#arrow)" />
        
        <rect x="120" y="10" width="80" height="30" rx="4" fill="#f9a8d422" stroke="#f9a8d4" />
        <text x="160" y="30" fill="#f9a8d4" fontSize="10" textAnchor="middle">Tree 1</text>
        
        <rect x="120" y="65" width="80" height="30" rx="4" fill="#f9a8d422" stroke="#f9a8d4" />
        <text x="160" y="85" fill="#f9a8d4" fontSize="10" textAnchor="middle">Tree 2</text>
        
        <rect x="120" y="120" width="80" height="30" rx="4" fill="#f9a8d422" stroke="#f9a8d4" />
        <text x="160" y="140" fill="#f9a8d4" fontSize="10" textAnchor="middle">Tree N</text>
        
        {/* Majority Vote */}
        <path d="M 200 25 L 260 70" stroke="#475569" strokeWidth="1.5" markerEnd="url(#arrow)" />
        <path d="M 200 80 L 260 80" stroke="#475569" strokeWidth="1.5" markerEnd="url(#arrow)" />
        <path d="M 200 135 L 260 90" stroke="#475569" strokeWidth="1.5" markerEnd="url(#arrow)" />
        
        <circle cx="300" cy="80" r="30" fill="#6ee7b722" stroke="#6ee7b7" strokeDasharray="4,2" />
        <text x="300" y="84" fill="#6ee7b7" fontSize="10" fontWeight="900" textAnchor="middle">VOTING</text>
        
        <path d="M 330 80 L 370 80" stroke="#6ee7b7" strokeWidth="2" markerEnd="url(#arrow)" />
        <text x="385" y="84" fill="#fff" fontSize="10" fontWeight="900">PRED</text>

        <defs>
          <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orientation="auto" markerUnits="strokeWidth">
            <path d="M0,0 L0,6 L9,3 z" fill="#475569" />
          </marker>
        </defs>
      </svg>
      <div style={{ fontSize: 9, color: "#94a3b8", marginTop: 10, fontStyle: "italic" }}>
        Trees grow <b>independently</b> in parallel. Final result is the <b>average</b> or majority vote.
      </div>
    </div>
  );
}

function BoostingSchema() {
  return (
    <div style={{ background: "#0a1628", padding: 20, borderRadius: 16, border: "1px solid #1e293b", textAlign: "center" }}>
      <div style={{ fontSize: 11, color: "#64748b", textTransform: "uppercase", marginBottom: 15, fontWeight: 800 }}>Boosting Architecture (AdaBoost / XGBoost)</div>
      <svg width="100%" height="160" viewBox="0 0 400 160">
        {/* Step 1 */}
        <rect x="10" y="60" width="50" height="30" rx="4" fill="#1e293b" stroke="#334155" />
        <text x="35" y="80" fill="#cbd5e1" fontSize="9" textAnchor="middle">Data</text>
        <path d="M 60 75 L 85 75" stroke="#475569" strokeWidth="1.5" markerEnd="url(#arrow2)" />
        
        {/* Tree 1 */}
        <rect x="85" y="55" width="60" height="40" rx="4" fill="#fb923c22" stroke="#fb923c" />
        <text x="115" y="80" fill="#fb923c" fontSize="9" textAnchor="middle">Tree 1</text>
        
        {/* Update Weights */}
        <path d="M 145 75 L 175 75" stroke="#475569" strokeWidth="1.5" markerEnd="url(#arrow2)" />
        <circle cx="195" cy="75" r="20" fill="#f43f5e22" stroke="#f43f5e" strokeDasharray="3,3" />
        <text x="195" y="78" fill="#f43f5e" fontSize="8" fontWeight="700" textAnchor="middle">RE-WEIGHT</text>
        
        {/* Tree 2 */}
        <path d="M 215 75 L 245 75" stroke="#475569" strokeWidth="1.5" markerEnd="url(#arrow2)" />
        <rect x="245" y="55" width="60" height="40" rx="4" fill="#fb923c22" stroke="#fb923c" />
        <text x="275" y="80" fill="#fb923c" fontSize="9" textAnchor="middle">Tree 2</text>
        
        {/* Final */}
        <path d="M 305 75 L 335 75" stroke="#475569" strokeWidth="1.5" markerEnd="url(#arrow2)" />
        <text x="350" y="78" fill="#cbd5e1" fontSize="14" textAnchor="middle">...</text>
        <path d="M 365 75 L 385 75" stroke="#6ee7b7" strokeWidth="2" markerEnd="url(#arrow2)" />
        
        <defs>
          <marker id="arrow2" markerWidth="10" markerHeight="10" refX="8" refY="3" orientation="auto" markerUnits="strokeWidth">
            <path d="M0,0 L0,6 L9,3 z" fill="#475569" />
          </marker>
        </defs>
      </svg>
      <div style={{ fontSize: 9, color: "#94a3b8", marginTop: 10, fontStyle: "italic" }}>
        Trees grow <b>sequentially</b>. Each tree corrects the <b>errors</b> of the previous ones.
      </div>
    </div>
  );
}

// ─── 📊 Performance Comparison Chart ───────────────────────────────────────

function PerformanceComparisonChart({ runs }) {
  const modelsToCompare = ["random_forest", "adaboost", "xgboost"];
  
  // Get latest run for each model
  const latestRuns = modelsToCompare.map(m => {
    return runs.filter(r => r.model === m).sort((a,b) => new Date(b.timestamp) - new Date(a.timestamp))[0];
  }).filter(Boolean);

  if (latestRuns.length === 0) {
    return (
      <div style={{ padding: 40, textAlign: "center", background: "#0a1628", borderRadius: 16, border: "1px dashed #1e293b", color: "#475569", fontSize: 12 }}>
        ⚠️ No data found for RF, AdaBoost, or XGBoost. <br/> Train these models to see the live comparison.
      </div>
    );
  }

  const metrics = ["f1_score", "accuracy", "recall"];
  const W = 600, H = 300, pad = 50;

  return (
    <div style={{ background: "#0a1628", padding: 24, borderRadius: 20, border: "1px solid #1e293b" }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 20 }}>
        <div style={S.sectionTitle}>Live Metric Comparison</div>
        <div style={{ display: "flex", gap: 10 }}>
          {latestRuns.map(r => (
            <div key={r.model} style={{ display: "flex", alignItems: "center", gap: 5 }}>
              <div style={{ width: 8, height: 8, borderRadius: "50%", background: MODEL_INFO[r.model]?.color }} />
              <span style={{ fontSize: 9, color: "#94a3b8" }}>{MODEL_INFO[r.model]?.label}</span>
            </div>
          ))}
        </div>
      </div>

      <div style={{ display: "grid", gap: 20 }}>
        {metrics.map(m => (
          <div key={m}>
            <div style={{ fontSize: 10, color: "#64748b", textTransform: "uppercase", marginBottom: 8, fontWeight: 700 }}>{m.replace("_", " ")}</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {latestRuns.map(r => (
                <div key={r.model} style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <div style={{ width: 80, fontSize: 9, color: "#94a3b8", fontWeight: 600 }}>{MODEL_INFO[r.model]?.label}</div>
                  <div style={{ flex: 1, background: "#1e293b", height: 12, borderRadius: 6, overflow: "hidden" }}>
                    <div 
                      style={{ 
                        width: `${r.metrics[m]*100}%`, 
                        background: MODEL_INFO[r.model]?.color, 
                        height: "100%",
                        transition: "width 0.8s ease-out"
                      }} 
                    />
                  </div>
                  <div style={{ width: 40, fontSize: 10, color: "#cbd5e1", fontWeight: 800 }}>{(r.metrics[m]*100).toFixed(1)}%</div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── 🏆 Final Comparison Tab ────────────────────────────────────────────────

export default function EnsembleComparisonTab() {
  const [runs, setRuns] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchModels().then(d => setRuns(d.runs || [])).finally(() => setLoading(false));
  }, []);

  const comparisonData = [
    {
      feature: "Base Algorithm",
      rf: "Bagging (Parallel)",
      ada: "Boosting (Sequential)",
      xgb: "Gradient Boosting (Sequential)"
    },
    {
      feature: "Main Advantage",
      rf: "Robust, hard to overfit",
      ada: "Simple, good with weak learners",
      xgb: "State-of-the-art accuracy"
    },
    {
      feature: "Weakness",
      rf: "Can be slow with many trees",
      ada: "Sensitive to noise/outliers",
      xgb: "Harder to tune (many params)"
    },
    {
      feature: "Hardware",
      rf: "Easy to parallelize",
      ada: "Mostly sequential",
      xgb: "Highly optimized / Parallel"
    }
  ];

  return (
    <div className="stagger" style={{ display: "flex", flexDirection: "column", gap: 24, paddingBottom: 60 }}>
      
      {/* Introduction */}
      <div style={{ ...S.section, background: "linear-gradient(135deg, #0f172a 0%, #020617 100%)", borderColor: "#38bdf833", textAlign: "center" }}>
        <div style={{ fontSize: 24, marginBottom: 10 }}>⚔️</div>
        <div style={{ ...S.sectionTitle, marginBottom: 5 }}>Ensemble Battleground</div>
        <div style={{ fontSize: 12, color: "#64748b" }}>Bagging vs Boosting: Comparing the 3 most powerful ensemble methods.</div>
      </div>

      {/* Schemas */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
        <BaggingSchema />
        <BoostingSchema />
      </div>

      {/* Live Data Plot */}
      {loading ? (
        <div style={{ textAlign: "center", padding: 40, color: "#475569" }}>Loading metrics...</div>
      ) : (
        <PerformanceComparisonChart runs={runs} />
      )}

      {/* Deep Comparison Table */}
      <div style={{ background: "#0a1628", borderRadius: 24, border: "1px solid #1e293b", overflow: "hidden" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
          <thead>
            <tr style={{ textAlign: "left", background: "#0f172a" }}>
              <th style={{ padding: 15, color: "#475569" }}>Feature</th>
              <th style={{ padding: 15, color: MODEL_INFO.random_forest.color }}>Random Forest</th>
              <th style={{ padding: 15, color: MODEL_INFO.adaboost.color }}>AdaBoost</th>
              <th style={{ padding: 15, color: MODEL_INFO.xgboost.color }}>XGBoost</th>
            </tr>
          </thead>
          <tbody>
            {comparisonData.map((row, i) => (
              <tr key={i} style={{ borderTop: "1px solid #1e293b" }}>
                <td style={{ padding: 15, color: "#94a3b8", fontWeight: 700 }}>{row.feature}</td>
                <td style={{ padding: 15, color: "#cbd5e1" }}>{row.rf}</td>
                <td style={{ padding: 15, color: "#cbd5e1" }}>{row.ada}</td>
                <td style={{ padding: 15, color: "#cbd5e1" }}>{row.xgb}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Conclusion / Pro-Tip */}
      <div style={{ background: "#6ee7b70a", padding: 24, borderRadius: 20, border: "1px solid #6ee7b733" }}>
        <div style={{ display: "flex", gap: 15 }}>
          <div style={{ fontSize: 20 }}>💡</div>
          <div>
            <div style={{ fontSize: 12, fontWeight: 900, color: "#6ee7b7", textTransform: "uppercase", marginBottom: 8 }}>Which one to choose?</div>
            <div style={{ fontSize: 13, color: "#cbd5e1", lineHeight: 1.6 }}>
              If you want a <b>stable, reliable</b> model with zero tuning, use <b>Random Forest</b>.<br/>
              If you want the <b>highest possible accuracy</b> and have time for tuning, use <b>XGBoost</b>.<br/>
              <b>AdaBoost</b> is often a middle ground but is very sensitive to bad data (noise).
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
