import React, { useState } from "react";
import S from "./styles";
import TrainTab from "./components/TrainTab";
import ResultsTab from "./components/ResultsTab";
import PredictTab from "./components/PredictTab";
import RunsTab from "./components/RunsTab";
import VisualizeTab from "./components/VisualizeTab";
import RFAnalysisTab from "./components/RFAnalysisTab";
import EnsembleComparisonTab from "./components/EnsembleComparisonTab";
import RegistryTab from "./components/RegistryTab";
import DriftTab from "./components/DriftTab";


// ─── Demo Data ──────────────────────────────────────────────────────────────
const DEMO_DATASET = [
  { id: "1001", tenure: 12, monthly: 65.4, total: 784.8, contract: "M2M", internet: "Fiber", churn: "Yes" },
  { id: "1002", tenure: 48, monthly: 45.2, total: 2169.6, contract: "2YR", internet: "DSL", churn: "No" },
  { id: "1003", tenure: 3, monthly: 89.1, total: 267.3, contract: "M2M", internet: "Fiber", churn: "Yes" },
  { id: "1004", tenure: 72, monthly: 20.0, total: 1440.0, contract: "2YR", internet: "No", churn: "No" },
];

export default function ChurnPlatform() {
  const [tab, setTab] = useState("train");
  const [trainResults, setTrainResults] = useState([]);

  const tabs = [
    { id: "train", label: "Training", icon: "⚗️" },
    { id: "results", label: "Insights", icon: "📊" },
    { id: "visualize", label: "Explore", icon: "🎨" },
    { id: "predict", label: "Predict", icon: "🔮" },
    { id: "rf_analysis", label: "RF Deep Dive", icon: "🌲" },
    { id: "ensemble_comp", label: "Ensemble Battle", icon: "⚔️" },
    { id: "registry", label: "Model Registry", icon: "🏆" },
    { id: "drift", label: "Data Drift", icon: "📈" },
    { id: "runs", label: "MLflow UI", icon: "📁" },
  ];

  const onTrainSuccess = (results) => {
    setTrainResults(results);
    setTab("results");
  };

  return (
    <div style={{ minHeight: "100vh", paddingBottom: 100 }}>
      {/* Navigation Header */}
      <nav style={{ 
        background: "rgba(3, 7, 18, 0.8)", 
        backdropFilter: "blur(20px)", 
        borderBottom: "1px solid rgba(255, 255, 255, 0.05)", 
        padding: "0 40px", 
        display: "flex", 
        alignItems: "center", 
        justifyContent: "space-between",
        height: 80,
        position: "sticky",
        top: 0,
        zIndex: 1000
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{ 
            width: 32, height: 32, 
            background: "linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)", 
            borderRadius: 10,
            display: "flex", alignItems: "center", justifyContent: "center",
            fontWeight: 900, color: "#fff", fontSize: 18,
            boxShadow: "0 0 20px rgba(99, 102, 241, 0.4)"
          }}>C</div>
          <div style={{ fontSize: 20, fontWeight: 800, letterSpacing: -0.5, color: "#fff" }}>
            CHURN.<span style={{ color: "#6366f1" }}>AI</span>
          </div>
          <div style={{ 
            background: "rgba(99, 102, 241, 0.1)", 
            color: "#818cf8", 
            padding: "4px 10px", 
            borderRadius: 8, 
            fontSize: 10, 
            fontWeight: 800, 
            letterSpacing: 1,
            border: "1px solid rgba(99, 102, 241, 0.2)",
            marginLeft: 8
          }}>v2.5 PRO</div>
        </div>

        <div style={{ display: "flex", gap: 8 }}>
          {tabs.map(t => (
            <button 
              key={t.id} 
              onClick={() => setTab(t.id)}
              style={{ 
                background: tab === t.id ? "rgba(99, 102, 241, 0.1)" : "transparent", 
                color: tab === t.id ? "#fff" : "#64748b", 
                border: "none",
                borderRadius: 12,
                padding: "10px 18px", 
                fontSize: 13, 
                fontWeight: tab === t.id ? 700 : 500, 
                cursor: "pointer", 
                transition: "all 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
                display: "flex",
                alignItems: "center",
                gap: 8,
                border: `1px solid ${tab === t.id ? "rgba(99, 102, 241, 0.2)" : "transparent"}`
              }}
              onMouseEnter={(e) => {
                if (tab !== t.id) e.currentTarget.style.color = "#a5b4fc";
              }}
              onMouseLeave={(e) => {
                if (tab !== t.id) e.currentTarget.style.color = "#64748b";
              }}
            >
              <span>{t.icon}</span>
              <span>{t.label}</span>
            </button>
          ))}
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
           <div style={{ width: 1, height: 24, background: "rgba(255,255,255,0.05)" }} />
           <div style={{ fontSize: 11, color: "#475569", fontWeight: 600, letterSpacing: 0.5 }}>
              EN7 · SOTA · 2026
           </div>
        </div>
      </nav>

      <main style={{ maxWidth: 1200, margin: "0 auto", padding: "40px 24px" }}>
        <div className="stagger">
          {tab === "train" && (
            <>
              {/* Dataset Quick Glance */}
              <div style={S.section} className="glass-card">
                <div style={S.sectionTitle}>
                  <span style={{ fontSize: 16 }}>📁</span> Dataset Intelligence
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 20 }}>
                  {DEMO_DATASET.map((row, i) => (
                    <div key={i} style={{ background: "rgba(15, 23, 42, 0.3)", padding: 16, borderRadius: 16, border: "1px solid rgba(255,255,255,0.03)" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                         <span style={{ fontSize: 10, color: "#475569", fontWeight: 700 }}>ID: {row.id}</span>
                         <span style={{ fontSize: 10, color: row.churn === "Yes" ? "#fca5a5" : "#10b981", fontWeight: 900 }}>{row.churn.toUpperCase()}</span>
                      </div>
                      <div style={{ fontSize: 13, fontWeight: 600 }}>{row.tenure} Months</div>
                      <div style={{ fontSize: 11, color: "#64748b" }}>${row.monthly}/mo · {row.contract}</div>
                    </div>
                  ))}
                </div>
                <div style={{ marginTop: 20, fontSize: 11, color: "#475569", textAlign: "center", fontStyle: "italic" }}>
                   Showing active memory samples. 7,043 total records indexed.
                </div>
              </div>

              <TrainTab onTrainSuccess={onTrainSuccess} />
            </>
          )}

          {tab === "results" && <ResultsTab results={trainResults} />}
          {tab === "visualize" && <VisualizeTab />}
          {tab === "predict" && <PredictTab />}
          {tab === "rf_analysis" && <RFAnalysisTab />}
          {tab === "ensemble_comp" && <EnsembleComparisonTab />}
          {tab === "registry" && <RegistryTab />}
          {tab === "drift" && <DriftTab />}
          {tab === "runs" && <RunsTab />}
        </div>
      </main>

      {/* Modern Architecture Footer */}
      <footer style={{ maxWidth: 1200, margin: "0 auto", padding: "0 24px" }}>
        <div style={{ ...S.section, textAlign: "center", padding: "40px" }} className="glass-card">
          <div style={{ ...S.sectionTitle, justifyContent: "center" }}>Platform MLOps Architecture</div>
          <div style={{ display: "flex", gap: 12, alignItems: "center", justifyContent: "center", flexWrap: "wrap", marginTop: 20 }}>
            {[
              ["FastAPI", "#a5b4fc"],
              ["MLflow", "#f9a8d4"],
              ["XGBoost", "#f43f5e"],
              ["Scikit-learn", "#fcd34d"],
              ["React 18", "#6366f1"],
              ["Vite", "#38bdf8"],
            ].map(([name, color], i) => (
              <div key={i} style={{ 
                background: `${color}10`, 
                border: `1px solid ${color}33`, 
                borderRadius: 14, 
                padding: "10px 20px",
                display: "flex", alignItems: "center", gap: 8
              }}>
                <div style={{ width: 6, height: 6, borderRadius: "50%", background: color, boxShadow: `0 0 10px ${color}` }} />
                <span style={{ fontSize: 12, fontWeight: 700, color: "#f1f5f9" }}>{name}</span>
              </div>
            ))}
          </div>
        </div>
      </footer>
    </div>
  );
}
