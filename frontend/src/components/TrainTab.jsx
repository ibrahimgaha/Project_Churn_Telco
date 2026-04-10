import React, { useState } from "react";
import MODEL_INFO from "../models";
import S from "../styles";
import { Tooltip } from "./UIComponents";
import { trainModels, autoTune } from "../api";

export default function TrainTab({ onTrainSuccess }) {
  const [selectedModels, setSelectedModels] = useState({ 
    logistic_regression: true, 
    knn: false,
    svm: false,
    decision_tree: false, 
    random_forest: false 
  });
  
  const [hyperparams, setHyperparams] = useState(
    Object.keys(MODEL_INFO).reduce((acc, key) => {
      acc[key] = MODEL_INFO[key].params.reduce((pAcc, p) => {
        pAcc[p.key] = p.default;
        return pAcc;
      }, {});
      return acc;
    }, {})
  );

  const [testSize, setTestSize] = useState(0.2);
  const [loading, setLoading] = useState(false);
  const [tuning, setTuning] = useState(null);
  const [error, setError] = useState(null);

  const handleTrain = async () => {
    const models = Object.entries(selectedModels).filter(([, v]) => v).map(([k]) => k);
    if (!models.length) { setError("Select at least one model."); return; }
    setLoading(true); setError(null);
    try {
      const res = await trainModels({ 
        models, 
        features: null, 
        test_size: testSize, 
        hyperparameters: hyperparams 
      });
      onTrainSuccess(res.results);
    } catch (e) { 
      setError(e.message); 
    } finally { 
      setLoading(false); 
    }
  };

  const handleTune = async (modelName) => {
    setTuning(modelName); setError(null);
    try {
      const res = await autoTune(modelName);
      setHyperparams(h => ({ ...h, [modelName]: res.best_params }));
      alert(`✅ Best CV F1: ${(res.best_cv_score * 100).toFixed(2)}%\nBest params applied to config.`);
    } catch (e) { 
      setError(e.message); 
    } finally { 
      setTuning(null); 
    }
  };

  return (
    <div className="fade-in">
      {error && <div style={S.errorBox}>⚠ {error}</div>}

      {/* Model Selection */}
      <div style={S.section}>
        <div style={S.sectionTitle}>Model Selection</div>
        <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
          {Object.entries(MODEL_INFO).map(([key, info]) => (
            <label key={key} style={{ 
              display: "flex", 
              alignItems: "center", 
              gap: 10, 
              padding: "12px 18px", 
              background: selectedModels[key] ? info.color + "15" : "#0f172a", 
              border: `2px solid ${selectedModels[key] ? info.color : "#1e293b"}`, 
              borderRadius: 10, 
              cursor: "pointer", 
              transition: "all 0.2s", 
              flex: 1, 
              minWidth: 200 
            }}>
              <input 
                type="checkbox" 
                checked={selectedModels[key]} 
                onChange={e => setSelectedModels(s => ({ ...s, [key]: e.target.checked }))} 
                style={{ accentColor: info.color, width: 16, height: 16 }} 
              />
              <div style={{ flex: 1 }}>
                <div style={{ fontWeight: 700, color: selectedModels[key] ? info.color : "#94a3b8", fontSize: 13 }}>{info.label}</div>
              </div>
              <Tooltip text={info.tooltip} />
            </label>
          ))}
        </div>
      </div>

      {/* Hyperparameter panels */}
      {Object.entries(MODEL_INFO).map(([key, info]) => !selectedModels[key] ? null : (
        <div key={key} style={{ ...S.section, borderColor: info.color + "33" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
            <div style={{ ...S.sectionTitle, marginBottom: 0, color: info.color }}>{info.label} — Hyperparameters</div>
            <button onClick={() => handleTune(key)} disabled={tuning === key} style={{ ...S.btnOutline, borderColor: info.accent + "66", color: info.color, opacity: tuning === key ? 0.6 : 1 }}>
              {tuning === key ? "⏳ Tuning..." : "⚡ Auto Tune"}
            </button>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px,1fr))", gap: 14 }}>
            {info.params.map(p => (
              <div key={p.key}>
                <label style={S.label}>{p.label}</label>
                {p.type === "select" ? (
                  <select 
                    value={hyperparams[key][p.key]} 
                    onChange={e => setHyperparams(h => ({ ...h, [key]: { ...h[key], [p.key]: e.target.value } }))} 
                    style={{ ...S.input, appearance: "none" }}
                  >
                    {p.options.map(o => <option key={o} value={o}>{o}</option>)}
                  </select>
                ) : (
                  <input 
                    type="number" 
                    value={hyperparams[key][p.key]} 
                    min={p.min} 
                    step={p.step} 
                    onChange={e => setHyperparams(h => ({ ...h, [key]: { ...h[key], [p.key]: parseFloat(e.target.value) || p.default } }))} 
                    style={S.input} 
                  />
                )}
              </div>
            ))}
          </div>
        </div>
      ))}

      {/* Train config */}
      <div style={{ ...S.section, display: "flex", alignItems: "center", gap: 24, flexWrap: "wrap", justifyContent: "center" }}>
        <div>
          <label style={S.label}>Test Split</label>
          <input type="range" min="0.1" max="0.4" step="0.05" value={testSize} onChange={e => setTestSize(parseFloat(e.target.value))} style={{ accentColor: "#6ee7b7", width: 200 }} />
          <span style={{ color: "#6ee7b7", fontSize: 13, marginLeft: 10, fontWeight: 700, fontFamily: "monospace" }}>{(testSize * 100).toFixed(0)}%</span>
        </div>
        <button onClick={handleTrain} disabled={loading} style={{ ...S.btn, padding: "14px 48px", fontSize: 14, opacity: loading ? 0.7 : 1, display: "flex", alignItems: "center", gap: 10, boxShadow: "0 0 20px rgba(110, 231, 183, 0.2)" }}>
          {loading ? "⏳ Training..." : "▶ Train Selected Models"}
        </button>
      </div>
    </div>
  );
}
