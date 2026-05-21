import { useState } from "react";
import MODEL_INFO from "../models";

// ─── Tag badge ──────────────────────────────────────────────────────────────
export function Tag({ children, color }) {
  return (
    <span style={{ 
      background: `${color}15`, 
      color, 
      border: `1px solid ${color}33`, 
      borderRadius: 10, 
      padding: "4px 12px", 
      fontSize: 10, 
      fontWeight: 800, 
      letterSpacing: 0.5,
      textTransform: "uppercase"
    }}>
      {children}
    </span>
  );
}

// ─── Info Tooltip ───────────────────────────────────────────────────────────
export function Tooltip({ text }) {
  const [show, setShow] = useState(false);
  return (
    <span style={{ position: "relative", display: "inline-block", marginLeft: 8 }}>
      <span 
        onMouseEnter={() => setShow(true)} 
        onMouseLeave={() => setShow(false)}
        style={{ 
          cursor: "help", 
          fontSize: 11, 
          color: "#94a3b8", 
          border: "1px solid rgba(255, 255, 255, 0.1)", 
          borderRadius: "50%", 
          width: 20, 
          height: 20, 
          display: "inline-flex", 
          alignItems: "center", 
          justifyContent: "center", 
          fontWeight: 700,
          background: "rgba(255, 255, 255, 0.03)"
        }}>?</span>
      {show && (
        <div style={{ 
          position: "absolute", 
          bottom: "140%", 
          left: "50%", 
          transform: "translateX(-50%)", 
          background: "#0f172a", 
          color: "#f1f5f9", 
          border: "1px solid rgba(99, 102, 241, 0.3)", 
          borderRadius: 14, 
          padding: "12px 18px", 
          width: 300, 
          fontSize: 12, 
          lineHeight: 1.5, 
          zIndex: 100, 
          boxShadow: "0 10px 40px rgba(0, 0, 0, 0.5)",
          backdropFilter: "blur(10px)"
        }}>
          {text}
        </div>
      )}
    </span>
  );
}

// ─── Metric card ────────────────────────────────────────────────────────────
export function MetricCard({ label, value, color }) {
  return (
    <div style={{ 
      background: "rgba(15, 23, 42, 0.4)", 
      border: `1px solid ${color}22`, 
      borderRadius: 20, 
      padding: "24px", 
      textAlign: "center", 
      flex: 1, 
      minWidth: 140,
      backdropFilter: "blur(8px)",
      transition: "transform 0.2s"
    }} className="glass-card">
      <div style={{ fontSize: 10, color: "#64748b", marginBottom: 10, letterSpacing: 1.5, textTransform: "uppercase", fontWeight: 700 }}>{label}</div>
      <div style={{ fontSize: 32, fontWeight: 900, color, letterSpacing: -1 }}>
        {(value * 100).toFixed(1)}<span style={{ fontSize: 16, opacity: 0.6 }}>%</span>
      </div>
    </div>
  );
}

// ─── Confusion Matrix ───────────────────────────────────────────────────────
export function ConfusionMatrix({ cm }) {
  if (!cm) return null;
  const [[tn, fp], [fn, tp]] = cm;
  const cells = [
    { label: "TN", val: tn, color: "#6ee7b7", desc: "True Negative" },
    { label: "FP", val: fp, color: "#fca5a5", desc: "False Positive" },
    { label: "FN", val: fn, color: "#fcd34d", desc: "False Negative" },
    { label: "TP", val: tp, color: "#818cf8", desc: "True Positive" },
  ];
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, maxWidth: 360 }}>
      {cells.map(c => (
        <div key={c.label} style={{ 
          background: `${c.color}08`, 
          border: `1px solid ${c.color}22`, 
          borderRadius: 20, 
          padding: 20, 
          textAlign: "center",
          backdropFilter: "blur(4px)"
        }}>
          <div style={{ fontSize: 10, color: "#64748b", marginBottom: 6, textTransform: "uppercase", fontWeight: 800 }}>{c.desc}</div>
          <div style={{ fontSize: 40, fontWeight: 900, color: c.color, marginBottom: 8, letterSpacing: -1 }}>{c.val}</div>
          <Tag color={c.color}>{c.label}</Tag>
        </div>
      ))}
    </div>
  );
}

// ─── ROC Curve (SVG) ────────────────────────────────────────────────────────
export function SimpleROC({ roc_curve: roc, color = "#6366f1" }) {
  if (!roc) return null;
  const W = 360, H = 280, pad = 40;
  const fw = W - pad * 2, fh = H - pad * 2;
  const pts = roc.fpr.map((x, i) => [pad + x * fw, pad + fh - roc.tpr[i] * fh]);
  const d = pts.map((p, i) => `${i === 0 ? "M" : "L"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");
  return (
    <div style={{ position: "relative" }}>
      <svg width={W} height={H} style={{ background: "rgba(15, 23, 42, 0.4)", borderRadius: 24, border: "1px solid rgba(255, 255, 255, 0.05)" }}>
        <defs>
          <linearGradient id="rocGradient" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.8" />
            <stop offset="100%" stopColor={color} stopOpacity="0.2" />
          </linearGradient>
        </defs>
        <line x1={pad} y1={pad} x2={pad} y2={pad + fh} stroke="rgba(255,255,255,0.1)" strokeWidth={1} />
        <line x1={pad} y1={pad + fh} x2={pad + fw} y2={pad + fh} stroke="rgba(255,255,255,0.1)" strokeWidth={1} />
        <line x1={pad} y1={pad + fh} x2={pad + fw} y2={pad} stroke="rgba(255,255,255,0.05)" strokeWidth={1} strokeDasharray="4,4" />
        <path d={d} fill="none" stroke={color} strokeWidth={4} strokeLinecap="round" strokeLinejoin="round" />
        <text x={pad + fw / 2} y={H - 12} fill="#64748b" fontSize={9} fontWeight={700} textAnchor="middle" style={{ textTransform: "uppercase" }}>False Positive Rate</text>
        <text x={12} y={pad + fh / 2} fill="#64748b" fontSize={9} fontWeight={700} textAnchor="middle" transform={`rotate(-90,12,${pad + fh / 2})`} style={{ textTransform: "uppercase" }}>True Positive Rate</text>
      </svg>
    </div>
  );
}

// ─── Metrics comparison bars ────────────────────────────────────────────────
export function MetricsBar({ results }) {
  if (!results.length) return null;
  const metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"];
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      {metrics.map((m) => (
        <div key={m}>
          <div style={{ fontSize: 10, color: "#94a3b8", marginBottom: 12, textTransform: "uppercase", letterSpacing: 1.5, fontWeight: 800 }}>{m.replace("_", " ")}</div>
          <div style={{ display: "flex", gap: 12 }}>
            {results.map(r => (
              <div key={r.run_id} style={{ flex: 1 }}>
                <div style={{ height: 12, background: "rgba(255,255,255,0.05)", borderRadius: 6, overflow: "hidden", marginBottom: 8 }}>
                  <div style={{ 
                    height: "100%", 
                    width: `${(r.metrics[m] * 100).toFixed(1)}%`, 
                    background: MODEL_INFO[r.model_name]?.color || "#6366f1", 
                    borderRadius: 6, 
                    transition: "width 1s cubic-bezier(0.4, 0, 0.2, 1)",
                    boxShadow: `0 0 10px ${MODEL_INFO[r.model_name]?.color}44`
                  }} />
                </div>
                <div style={{ fontSize: 10, color: "#64748b", fontWeight: 700, textAlign: "center" }}>
                  {MODEL_INFO[r.model_name]?.label || r.model_name}
                </div>
                <div style={{ fontSize: 14, color: "#f1f5f9", fontWeight: 900, textAlign: "center", marginTop: 2 }}>
                  {(r.metrics[m] * 100).toFixed(1)}%
                </div>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

// ─── Best Model Banner ──────────────────────────────────────────────────────
export function BestModelBanner({ results }) {
  if (!results || results.length < 2) return null;
  const best = results.reduce((a, b) => (a.metrics.f1_score > b.metrics.f1_score ? a : b));
  const info = MODEL_INFO[best.model_name];
  return (
    <div style={{ 
      background: `linear-gradient(135deg, ${info?.color || "#6366f1"}22 0%, rgba(15, 23, 42, 0.8) 100%)`, 
      border: `1px solid ${info?.color || "#6366f1"}44`, 
      borderRadius: 24, 
      padding: "24px 32px", 
      marginBottom: 32, 
      display: "flex", 
      alignItems: "center", 
      gap: 24, 
      boxShadow: `0 10px 40px ${info?.color || "#6366f1"}15`
    }}>
      <div style={{ fontSize: 40, filter: "drop-shadow(0 0 10px rgba(255,255,255,0.3))" }}>🏆</div>
      <div>
        <div style={{ fontSize: 10, color: "#94a3b8", textTransform: "uppercase", letterSpacing: 2, fontWeight: 800, marginBottom: 4 }}>Optimized SOTA Model Detected</div>
        <div style={{ fontSize: 24, fontWeight: 900, color: info?.color || "#6366f1", letterSpacing: -0.5 }}>
          {info?.label || best.model_name} <span style={{ color: "#fff", opacity: 0.5 }}>—</span> {(best.metrics.f1_score * 100).toFixed(1)}% F1
        </div>
      </div>
      <div style={{ marginLeft: "auto" }}>
        <Tag color={info?.color || "#6366f1"}>ID: {best.run_id.slice(0, 8)}</Tag>
      </div>
    </div>
  );
}
