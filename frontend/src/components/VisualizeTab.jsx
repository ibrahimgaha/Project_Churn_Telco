import { useState } from "react";
import MODEL_INFO from "../models";
import S from "../styles";
import { fetchPCA, fetchTSNE } from "../api";

// ─── Scatter plot (SVG) ────────────────────────────────────────────────────
function Scatter({ points, title }) {
  if (!points || !points.length) return null;
  const W = 520, H = 380, pad = 40;
  const xs = points.map(p => p.x), ys = points.map(p => p.y);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const xR = xMax - xMin || 1, yR = yMax - yMin || 1;
  const fw = W - pad * 2, fh = H - pad * 2;

  // Sample up to 800 points for performance
  const sampled = points.length > 800
    ? points.filter((_, i) => i % Math.ceil(points.length / 800) === 0)
    : points;

  return (
    <div>
      <div style={{ fontSize: 13, fontWeight: 700, color: "#94a3b8", marginBottom: 8, textTransform: "uppercase", letterSpacing: 2 }}>{title}</div>
      <svg width={W} height={H} style={{ background: "#0f172a", borderRadius: 8, border: "1px solid #1e293b" }}>
        {/* Axes */}
        <line x1={pad} y1={pad} x2={pad} y2={pad + fh} stroke="#334155" />
        <line x1={pad} y1={pad + fh} x2={pad + fw} y2={pad + fh} stroke="#334155" />
        <text x={pad + fw / 2} y={H - 8} fill="#475569" fontSize={10} textAnchor="middle">Component 1</text>
        <text x={12} y={pad + fh / 2} fill="#475569" fontSize={10} textAnchor="middle" transform={`rotate(-90,12,${pad + fh / 2})`}>Component 2</text>
        {/* Points */}
        {sampled.map((p, i) => (
          <circle
            key={i}
            cx={pad + ((p.x - xMin) / xR) * fw}
            cy={pad + fh - ((p.y - yMin) / yR) * fh}
            r={2.5}
            fill={p.label === 1 ? "#fca5a5" : "#6ee7b7"}
            opacity={0.6}
          />
        ))}
        {/* Legend */}
        <circle cx={W - 100} cy={16} r={5} fill="#fca5a5" />
        <text x={W - 90} y={20} fill="#94a3b8" fontSize={10}>Churn</text>
        <circle cx={W - 45} cy={16} r={5} fill="#6ee7b7" />
        <text x={W - 35} y={20} fill="#94a3b8" fontSize={10}>Retain</text>
      </svg>
    </div>
  );
}

// ─── Visualize Tab ─────────────────────────────────────────────────────────
export default function VisualizeTab() {
  const [pcaData, setPcaData] = useState(null);
  const [tsneData, setTsneData] = useState(null);
  const [loading, setLoading] = useState(null); // "pca" | "tsne" | null
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
    <div style={S.section}>
      <div style={S.sectionTitle}>Dimensionality Reduction — Data Exploration</div>
      <p style={{ fontSize: 12, color: "#64748b", marginBottom: 16, lineHeight: 1.7 }}>
        Reduce high-dimensional features to 2D to visualise how separable the Churn vs. Retain groups are.
        <strong style={{ color: "#94a3b8" }}> PCA</strong> is fast & linear.
        <strong style={{ color: "#94a3b8" }}> t-SNE</strong> is slower but captures non-linear structure.
      </p>

      {error && <div style={S.errorBox}>⚠ {error}</div>}

      <div style={{ display: "flex", gap: 12, marginBottom: 24 }}>
        <button onClick={handlePCA} disabled={loading === "pca"} style={{ ...S.btn, background: "#38bdf8" }}>
          {loading === "pca" ? "⏳ Computing PCA..." : "📐 Run PCA"}
        </button>
        <button onClick={handleTSNE} disabled={loading === "tsne"} style={{ ...S.btn, background: "#c4b5fd", color: "#0f172a" }}>
          {loading === "tsne" ? "⏳ Computing t-SNE..." : "🔬 Run t-SNE"}
        </button>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: pcaData && tsneData ? "1fr 1fr" : "1fr", gap: 24 }}>
        {pcaData && <Scatter points={pcaData} title="PCA Projection" />}
        {tsneData && <Scatter points={tsneData} title="t-SNE Projection" />}
      </div>

      {!pcaData && !tsneData && (
        <div style={{ color: "#475569", textAlign: "center", padding: 40 }}>
          Click a button above to generate a visualization.
        </div>
      )}
    </div>
  );
}
