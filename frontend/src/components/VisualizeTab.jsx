import { useState } from "react";
import MODEL_INFO from "../models";
import S from "../styles";
import { fetchPCA, fetchTSNE } from "../api";

// ─── Scatter plot (SVG) ────────────────────────────────────────────────────
function Scatter({ points, title }) {
  if (!points || !points.length) return null;
  const W = 520, H = 400, pad = 48;
  const xs = points.map(p => p.x), ys = points.map(p => p.y);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const xR = xMax - xMin || 1, yR = yMax - yMin || 1;
  const fw = W - pad * 2, fh = H - pad * 2;

  // Sample points for performance
  const sampled = points.length > 1000
    ? points.filter((_, i) => i % Math.ceil(points.length / 1000) === 0)
    : points;

  return (
    <div className="stagger">
      <div style={{ fontSize: 11, fontWeight: 800, color: "#6366f1", marginBottom: 16, textTransform: "uppercase", letterSpacing: 1.5, display: "flex", alignItems: "center", gap: 8 }}>
         <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#6366f1" }} />
         {title}
      </div>
      <div style={{ 
        background: "rgba(15, 23, 42, 0.4)", 
        borderRadius: 24, 
        border: "1px solid rgba(255, 255, 255, 0.05)",
        padding: 24,
        boxShadow: "inset 0 0 20px rgba(0,0,0,0.2)"
      }}>
        <svg width="100%" height={H} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet">
          <defs>
            <radialGradient id="churnGrad" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="#f43f5e" stopOpacity="0.8" />
              <stop offset="100%" stopColor="#f43f5e" stopOpacity="0.1" />
            </radialGradient>
            <radialGradient id="retainGrad" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="#10b981" stopOpacity="0.8" />
              <stop offset="100%" stopColor="#10b981" stopOpacity="0.1" />
            </radialGradient>
          </defs>
          
          {/* Grid Lines */}
          {[0, 0.25, 0.5, 0.75, 1].map(p => (
            <React.Fragment key={p}>
              <line x1={pad} y1={pad + p * fh} x2={W - pad} y2={pad + p * fh} stroke="rgba(255,255,255,0.03)" />
              <line x1={pad + p * fw} y1={pad} x2={pad + p * fw} y2={H - pad} stroke="rgba(255,255,255,0.03)" />
            </React.Fragment>
          ))}

          {/* Points */}
          {sampled.map((p, i) => (
            <circle
              key={i}
              cx={pad + ((p.x - xMin) / xR) * fw}
              cy={pad + fh - ((p.y - yMin) / yR) * fh}
              r={3}
              fill={p.label === 1 ? "#f43f5e" : "#10b981"}
              style={{ mixBlendMode: "screen" }}
              opacity={0.7}
            />
          ))}

          {/* Legend Overlay */}
          <g transform={`translate(${W - 120}, 20)`}>
             <rect width={100} height={50} rx={12} fill="rgba(3,7,18,0.8)" />
             <circle cx={20} cy={16} r={4} fill="#f43f5e" />
             <text x={35} y={20} fill="#94a3b8" fontSize={10} fontWeight={700}>CHURN</text>
             <circle cx={20} cy={34} r={4} fill="#10b981" />
             <text x={35} y={38} fill="#94a3b8" fontSize={10} fontWeight={700}>RETAIN</text>
          </g>
        </svg>
      </div>
    </div>
  );
}

// ─── Visualize Tab ─────────────────────────────────────────────────────────
import React from "react";
export default function VisualizeTab() {
  const [pcaData, setPcaData] = useState(null);
  const [tsneData, setTsneData] = useState(null);
  const [loading, setLoading] = useState(null);
  const [error, setError] = useState(null);

  const handlePCA = async () => {
    setLoading("pca"); setError(null);
    try { const d = await fetchPCA(); setPcaData(d.points); }
    catch (e) { setError(e.message); }
    finally { setLoading(null); }
  };

  const handleTSNE = async () => {
    setLoading("tsne"); setError(null);
    try { const d = await fetchTSNE(); setTsneData(d.points); }
    catch (e) { setError(e.message); }
    finally { setLoading(null); }
  };

  return (
    <div className="stagger">
      <div style={S.section} className="glass-card">
        <div style={S.sectionTitle}>
           <span style={{ color: "#6366f1" }}>🎨</span> Feature Topology
        </div>
        <p style={{ fontSize: 13, color: "#94a3b8", marginBottom: 32, lineHeight: 1.6, maxWidth: 800 }}>
          Deconstruct high-dimensional customer vectors into 2D manifolds to inspect feature separation.
          <br/><span style={{ color: "#64748b" }}>Linear (PCA) and Non-linear (t-SNE) projections provide diverse structural insights.</span>
        </p>

        {error && <div style={S.errorBox}><span>⚠️</span> {error}</div>}

        <div style={{ display: "flex", gap: 16, marginBottom: 40 }}>
          <button 
            onClick={handlePCA} 
            disabled={loading === "pca"} 
            style={{ 
              ...S.btn, 
              background: "rgba(56, 189, 248, 0.1)", 
              color: "#38bdf8", 
              border: "1px solid rgba(56, 189, 248, 0.3)",
              boxShadow: "none"
            }}
          >
            {loading === "pca" ? "📐 Computing PCA..." : "📐 Launch PCA Analysis"}
          </button>
          <button 
            onClick={handleTSNE} 
            disabled={loading === "tsne"} 
            style={{ 
              ...S.btn, 
              background: "rgba(167, 139, 250, 0.1)", 
              color: "#a78bfa", 
              border: "1px solid rgba(167, 139, 250, 0.3)",
              boxShadow: "none"
            }}
          >
            {loading === "tsne" ? "🔬 Computing t-SNE..." : "🔬 Launch t-SNE Analysis"}
          </button>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: pcaData && tsneData ? "1fr 1fr" : "1fr", gap: 40 }}>
          {pcaData && <Scatter points={pcaData} title="Principle Component Analysis (PCA)" />}
          {tsneData && <Scatter points={tsneData} title="t-Distributed Stochastic Neighbor Embedding (t-SNE)" />}
        </div>

        {!pcaData && !tsneData && (
          <div style={{ 
            color: "#475569", 
            textAlign: "center", 
            padding: "100px 40px", 
            border: "1px dashed rgba(255,255,255,0.05)",
            borderRadius: 24,
            background: "rgba(255,255,255,0.01)"
          }}>
            <div style={{ fontSize: 32, marginBottom: 12 }}>📡</div>
            <div style={{ fontSize: 13, fontWeight: 600 }}>Select a projection algorithm to map the customer manifold.</div>
          </div>
        )}
      </div>
    </div>
  );
}
