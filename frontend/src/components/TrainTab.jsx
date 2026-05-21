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
    random_forest: false,
    adaboost: false,
    xgboost: false,
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
    } catch (e) { 
      setError(e.message); 
    } finally { 
      setTuning(null); 
    }
  };

  return (
    <div className="stagger">
      {error && <div style={S.errorBox}><span>⚠️</span> {error}</div>}

      {/* Model Selection Grid */}
      <div style={S.section} className="glass-card">
        <div style={S.sectionTitle}>
          <span style={{ color: "#6366f1" }}>⚛</span> Model Configuration
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))", gap: 16 }}>
          {Object.entries(MODEL_INFO).map(([key, info]) => (
            <label key={key} style={{ 
              display: "flex", 
              alignItems: "center", 
              gap: 12, 
              padding: "16px 20px", 
              background: selectedModels[key] ? `${info.color}10` : "rgba(15, 23, 42, 0.4)", 
              border: `1px solid ${selectedModels[key] ? `${info.color}44` : "rgba(255,255,255,0.05)"}`, 
              borderRadius: 16, 
              cursor: "pointer", 
              transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
              position: "relative",
              overflow: "hidden"
            }}>
              {selectedModels[key] && (
                <div style={{ position: "absolute", top: 0, left: 0, width: 4, height: "100%", background: info.color }} />
              )}
              <input 
                type="checkbox" 
                checked={selectedModels[key]} 
                onChange={e => setSelectedModels(s => ({ ...s, [key]: e.target.checked }))} 
                style={{ accentColor: info.color, width: 18, height: 18 }} 
              />
              <div style={{ flex: 1 }}>
                <div style={{ fontWeight: 800, color: selectedModels[key] ? "#fff" : "#64748b", fontSize: 13, letterSpacing: -0.2 }}>{info.label}</div>
                <div style={{ fontSize: 10, color: "#475569", marginTop: 2 }}>{selectedModels[key] ? "Config Enabled" : "Standby"}</div>
              </div>
              <Tooltip text={info.tooltip} />
            </label>
          ))}
        </div>
      </div>

      {/* Parameter Panels */}
      <div style={{ display: "grid", gap: 20 }}>
        {Object.entries(MODEL_INFO).map(([key, info]) => !selectedModels[key] ? null : (
          <div key={key} style={{ 
            ...S.section, 
            background: "rgba(15, 23, 42, 0.3)",
            borderLeft: `4px solid ${info.color}` 
          }} className="glass-card">
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 24 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                 <div style={{ width: 10, height: 10, borderRadius: "50%", background: info.color, boxShadow: `0 0 10px ${info.color}` }} />
                 <div style={{ fontSize: 14, fontWeight: 800, color: "#fff", letterSpacing: -0.5 }}>{info.label} Tuning</div>
              </div>
              <button 
                onClick={() => handleTune(key)} 
                disabled={tuning === key} 
                style={{ 
                  ...S.btnOutline, 
                  borderColor: `${info.color}44`, 
                  color: info.color, 
                  opacity: tuning === key ? 0.6 : 1,
                  background: `${info.color}08`
                }}
              >
                {tuning === key ? "⚡ Tuning..." : "⚡ Auto Optimize"}
              </button>
            </div>
            
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 24 }}>
              {info.params.map(p => (
                <div key={p.key}>
                  <label style={S.label}>{p.label}</label>
                  {p.type === "select" ? (
                    <div style={{ position: "relative" }}>
                      <select 
                        value={hyperparams[key][p.key]} 
                        onChange={e => setHyperparams(h => ({ ...h, [key]: { ...h[key], [p.key]: e.target.value } }))} 
                        style={{ ...S.input, appearance: "none" }}
                      >
                        {p.options.map(o => <option key={o} value={o}>{o}</option>)}
                      </select>
                      <div style={{ position: "absolute", right: 12, top: "50%", transform: "translateY(-50%)", pointerEvents: "none", fontSize: 10, color: "#475569" }}>▼</div>
                    </div>
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
      </div>

      {/* Execution Controller */}
      <div style={{ 
        ...S.section, 
        display: "flex", 
        alignItems: "center", 
        gap: 40, 
        flexWrap: "wrap", 
        justifyContent: "center",
        background: "linear-gradient(135deg, rgba(99, 102, 241, 0.05) 0%, rgba(15, 23, 42, 0.4) 100%)",
        marginTop: 20
      }} className="glass-card">
        <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
          <div>
            <label style={S.label}>Validation Split Ratio</label>
            <input 
              type="range" min="0.1" max="0.4" step="0.05" 
              value={testSize} 
              onChange={e => setTestSize(parseFloat(e.target.value))} 
              style={{ accentColor: "#6366f1", width: 220, cursor: "pointer" }} 
            />
          </div>
          <div style={{ 
            background: "rgba(99, 102, 241, 0.1)", 
            padding: "8px 16px", 
            borderRadius: 12, 
            color: "#818cf8", 
            fontSize: 16, 
            fontWeight: 900,
            border: "1px solid rgba(99, 102, 241, 0.2)"
          }}>
            {(testSize * 100).toFixed(0)}%
          </div>
        </div>
        
        <button 
          onClick={handleTrain} 
          disabled={loading} 
          style={{ 
            ...S.btn, 
            padding: "16px 50px", 
            fontSize: 15, 
            opacity: loading ? 0.7 : 1,
          }}
        >
          {loading ? "⚙️ Processing..." : "🚀 Launch Experiment"}
        </button>
      </div>
    </div>
  );
}
