import React, { useState } from "react";
import S from "./styles";
import TrainTab from "./components/TrainTab";
import ResultsTab from "./components/ResultsTab";
import PredictTab from "./components/PredictTab";
import RunsTab from "./components/RunsTab";
import VisualizeTab from "./components/VisualizeTab";
import RFAnalysisTab from "./components/RFAnalysisTab";

// ─── Demo Data ──────────────────────────────────────────────────────────────
const DEMO_DATASET = [
  { customerID: "1001", tenure: 12, MonthlyCharges: 65.4, TotalCharges: 784.8, Contract: "Month-to-month", InternetService: "Fiber optic", Churn: "Yes" },
  { customerID: "1002", tenure: 48, MonthlyCharges: 45.2, TotalCharges: 2169.6, Contract: "Two year", InternetService: "DSL", Churn: "No" },
  { customerID: "1003", tenure: 3, MonthlyCharges: 89.1, TotalCharges: 267.3, Contract: "Month-to-month", InternetService: "Fiber optic", Churn: "Yes" },
  { customerID: "1004", tenure: 72, MonthlyCharges: 20.0, TotalCharges: 1440.0, Contract: "Two year", InternetService: "No", Churn: "No" },
  { customerID: "1005", tenure: 7, MonthlyCharges: 77.5, TotalCharges: 542.5, Contract: "Month-to-month", InternetService: "Fiber optic", Churn: "Yes" },
];

// ─── Main App ────────────────────────────────────────────────────────────────
export default function ChurnPlatform() {
  const [tab, setTab] = useState("train");
  const [trainResults, setTrainResults] = useState([]);

  const tabs = [
    { id: "train", label: "⚗ Training", icon: "⚗" },
    { id: "results", label: "📊 Performance", icon: "📊" },
    { id: "visualize", label: "🎨 Exploration", icon: "🎨" },
    { id: "predict", label: "🔮 Inference", icon: "🔮" },
    { id: "rf_analysis", label: "🌲 RF Analysis", icon: "🌲" },
    { id: "runs", label: "📁 MLflow History", icon: "📁" },
  ];

  const onTrainSuccess = (results) => {
    setTrainResults(results);
    setTab("results");
  };

  return (
    <div style={{ minHeight: "100vh", background: "#030712", color: "#e2e8f0", fontFamily: "'Inter', sans-serif" }}>
      {/* Premium Header */}
      <div style={{ background: "#0a1628", borderBottom: "1px solid #1e293b", padding: "16px 32px", display: "flex", alignItems: "center", gap: 16, sticky: "top", zIndex: 10 }}>
        <div style={{ fontSize: 20, fontWeight: 900, color: "#6ee7b7", letterSpacing: -1, textShadow: "0 0 10px rgba(110, 231, 183, 0.3)" }}>◈ CHURN.ML</div>
        <div style={{ background: "#6ee7b71a", color: "#6ee7b7", border: "1px solid #6ee7b744", borderRadius: 4, padding: "2px 8px", fontSize: 11, fontWeight: 700, letterSpacing: 1 }}>EXPERIMENTATION v2.0</div>
        <div style={{ flex: 1 }} />
        <div style={{ display: "none", md: "block", fontSize: 11, color: "#475569", fontWeight: 500 }}>
          LR · KNN · SVM · DT · RANDOM FOREST
        </div>
      </div>

      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "28px 24px" }}>
        {/* Navigation Bar */}
        <div style={{ display: "flex", gap: 6, marginBottom: 28, borderBottom: "1px solid #1e293b", paddingBottom: 0 }}>
          {tabs.map(t => (
            <button 
              key={t.id} 
              onClick={() => setTab(t.id)}
              style={{ 
                background: tab === t.id ? "#6ee7b710" : "transparent", 
                color: tab === t.id ? "#6ee7b7" : "#64748b", 
                border: "none",
                borderBottom: `2px solid ${tab === t.id ? "#6ee7b7" : "transparent"}`,
                padding: "12px 24px", 
                fontSize: 13, 
                fontWeight: tab === t.id ? 800 : 600, 
                cursor: "pointer", 
                transition: "all 0.2s",
                display: "flex",
                alignItems: "center",
                gap: 8
              }}
            >
              <span>{t.label}</span>
            </button>
          ))}
        </div>

        {/* Content Area */}
        <div className="stagger">
          {tab === "train" && (
            <>
              {/* Dataset Preview Card */}
              <div style={S.section} className="fade-in">
                <div style={S.sectionTitle}>Dataset Preview — Telco Churn Data</div>
                <div style={{ overflowX: "auto", borderRadius: 8, border: "1px solid #1e293b" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                    <thead style={{ background: "#0f172a" }}>
                      <tr>{Object.keys(DEMO_DATASET[0]).map(k => <th key={k} style={{ padding: "10px 14px", color: "#64748b", textAlign: "left", textTransform: "uppercase", letterSpacing: 1, fontSize: 10 }}>{k}</th>)}</tr>
                    </thead>
                    <tbody>
                      {DEMO_DATASET.map((row, i) => (
                        <tr key={i} style={{ borderBottom: "1px solid #0f172a" }}>
                          {Object.values(row).map((v, j) => (
                            <td key={j} style={{ padding: "10px 14px", color: j === 6 ? (v === "Yes" ? "#fca5a5" : "#6ee7b7") : "#cbd5e1", fontWeight: j === 6 ? 700 : 400 }}>{v}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p style={{ marginTop: 12, fontSize: 11, color: "#475569" }}>
                  Displaying small sample. Full dataset contains 7,043 customer records used for training.
                </p>
              </div>

              <TrainTab onTrainSuccess={onTrainSuccess} />
            </>
          )}

          {tab === "results" && <ResultsTab results={trainResults} />}
          {tab === "visualize" && <VisualizeTab />}
          {tab === "predict" && <PredictTab />}
          {tab === "rf_analysis" && <RFAnalysisTab />}
          {tab === "runs" && <RunsTab />}
        </div>
      </div>

      {/* Modern Footer System Architecture */}
      <div style={{ maxWidth: 1200, margin: "40px auto 100px auto", padding: "0 24px" }}>
        <div style={{ background: "#0a1628", border: "1px solid #1e293b", borderRadius: 12, padding: 24 }}>
          <div style={S.sectionTitle}>Platform Architecture (MLOps Stack)</div>
          <div style={{ display: "flex", gap: 8, alignItems: "center", justifyContent: "center", flexWrap: "wrap", padding: "16px 0" }}>
            {[
              ["FastAPI", "#a5b4fc", "REST API Backend"],
              ["→"],
              ["MLflow", "#f9a8d4", "Experiment Tracking"],
              ["→"],
              ["scikit-learn", "#fcd34d", "5 Classification Models"],
              ["→"],
              ["GridSearchCV", "#6ee7b7", "Auto Hyperparameter Tuning"],
              ["→"],
              ["React + Vite", "#38bdf8", "Premium Frontend UI"],
            ].map((item, i) => item.length === 1 ? (
              <span key={i} style={{ color: "#1e293b", fontSize: 24, fontWeight: 300 }}>→</span>
            ) : (
              <div key={i} style={{ background: item[1] + "10", border: `1px solid ${item[1]}30`, borderRadius: 10, padding: "12px 18px", textAlign: "center", minWidth: 140 }}>
                <div style={{ color: item[1], fontWeight: 800, fontSize: 12, marginBottom: 4 }}>{item[0]}</div>
                <div style={{ color: "#475569", fontSize: 10 }}>{item[2]}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
