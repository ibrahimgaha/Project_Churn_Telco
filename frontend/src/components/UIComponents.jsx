import { useState } from "react";
import MODEL_INFO from "../models";

// ─── Tag badge ──────────────────────────────────────────────────────────────
export function Tag({ children, color }) {
  return (
    <span style={{ background: color + "22", color, border: `1px solid ${color}44`, borderRadius: 4, padding: "2px 8px", fontSize: 11, fontWeight: 700, letterSpacing: 1 }}>
      {children}
    </span>
  );
}

// ─── Info Tooltip ───────────────────────────────────────────────────────────
export function Tooltip({ text }) {
  const [show, setShow] = useState(false);
  return (
    <span style={{ position: "relative", display: "inline-block", marginLeft: 6 }}>
      <span onMouseEnter={() => setShow(true)} onMouseLeave={() => setShow(false)}
        style={{ cursor: "help", fontSize: 12, color: "#64748b", border: "1px solid #334155", borderRadius: "50%", width: 18, height: 18, display: "inline-flex", alignItems: "center", justifyContent: "center", fontWeight: 700 }}>?</span>
      {show && (
        <div style={{ position: "absolute", bottom: "130%", left: "50%", transform: "translateX(-50%)", background: "#0f172a", color: "#e2e8f0", border: "1px solid #334155", borderRadius: 8, padding: "10px 14px", width: 280, fontSize: 12, lineHeight: 1.6, zIndex: 100, boxShadow: "0 8px 32px #00000080" }}>
          {text}
        </div>
      )}
    </span>
  );
}

// ─── Metric card ────────────────────────────────────────────────────────────
export function MetricCard({ label, value, color }) {
  return (
    <div style={{ background: "#0f172a", border: `1px solid ${color}44`, borderRadius: 10, padding: "16px 20px", textAlign: "center", flex: 1, minWidth: 100 }}>
      <div style={{ fontSize: 11, color: "#64748b", marginBottom: 6, letterSpacing: 1, textTransform: "uppercase" }}>{label}</div>
      <div style={{ fontSize: 28, fontWeight: 800, color, fontFamily: "'IBM Plex Mono', monospace" }}>
        {(value * 100).toFixed(1)}<span style={{ fontSize: 14 }}>%</span>
      </div>
    </div>
  );
}

// ─── Confusion Matrix 2×2 ───────────────────────────────────────────────────
export function ConfusionMatrix({ cm }) {
  if (!cm) return null;
  const [[tn, fp], [fn, tp]] = cm;
  const cells = [
    { label: "TN", val: tn, color: "#6ee7b7", desc: "True Negative" },
    { label: "FP", val: fp, color: "#fca5a5", desc: "False Positive" },
    { label: "FN", val: fn, color: "#fcd34d", desc: "False Negative" },
    { label: "TP", val: tp, color: "#a5b4fc", desc: "True Positive" },
  ];
  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, maxWidth: 280 }}>
        {cells.map(c => (
          <div key={c.label} style={{ background: c.color + "18", border: `1px solid ${c.color}55`, borderRadius: 8, padding: 14, textAlign: "center" }}>
            <div style={{ fontSize: 10, color: "#64748b", marginBottom: 4 }}>{c.desc}</div>
            <div style={{ fontSize: 32, fontWeight: 900, color: c.color, fontFamily: "'IBM Plex Mono', monospace" }}>{c.val}</div>
            <Tag color={c.color}>{c.label}</Tag>
          </div>
        ))}
      </div>
      <div style={{ marginTop: 10, fontSize: 11, color: "#475569" }}>
        ⚠ FN = predicted No Churn but will churn → highest business cost
      </div>
    </div>
  );
}

// ─── ROC Curve (SVG) ────────────────────────────────────────────────────────
export function SimpleROC({ roc_curve: roc, color = "#6ee7b7" }) {
  if (!roc) return null;
  const W = 300, H = 220, pad = 36;
  const fw = W - pad * 2, fh = H - pad * 2;
  const pts = roc.fpr.map((x, i) => [pad + x * fw, pad + fh - roc.tpr[i] * fh]);
  const d = pts.map((p, i) => `${i === 0 ? "M" : "L"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");
  return (
    <svg width={W} height={H} style={{ background: "#0f172a", borderRadius: 8, border: "1px solid #1e293b" }}>
      <line x1={pad} y1={pad} x2={pad} y2={pad + fh} stroke="#334155" strokeWidth={1} />
      <line x1={pad} y1={pad + fh} x2={pad + fw} y2={pad + fh} stroke="#334155" strokeWidth={1} />
      <line x1={pad} y1={pad + fh} x2={pad + fw} y2={pad} stroke="#1e293b" strokeWidth={1} strokeDasharray="4,4" />
      <path d={d} fill="none" stroke={color} strokeWidth={2.5} />
      <text x={pad + fw / 2} y={H - 6} fill="#475569" fontSize={10} textAnchor="middle">False Positive Rate</text>
      <text x={10} y={pad + fh / 2} fill="#475569" fontSize={10} textAnchor="middle" transform={`rotate(-90,10,${pad + fh / 2})`}>True Positive Rate</text>
    </svg>
  );
}

// ─── Metrics comparison bars ────────────────────────────────────────────────
export function MetricsBar({ results }) {
  if (!results.length) return null;
  const metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"];
  const colors = ["#6ee7b7", "#a5b4fc", "#fcd34d", "#f9a8d4", "#7dd3fc"];
  return (
    <div style={{ overflowX: "auto" }}>
      {metrics.map((m, mi) => (
        <div key={m} style={{ marginBottom: 14 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
            <span style={{ fontSize: 11, color: "#64748b", width: 80, textTransform: "uppercase", letterSpacing: 1 }}>{m.replace("_", " ")}</span>
            <div style={{ flex: 1, display: "flex", gap: 8 }}>
              {results.map(r => (
                <div key={r.run_id} style={{ flex: 1 }}>
                  <div style={{ height: 24, background: "#1e293b", borderRadius: 4, overflow: "hidden" }}>
                    <div style={{ height: "100%", width: `${(r.metrics[m] * 100).toFixed(1)}%`, background: colors[mi], borderRadius: 4, transition: "width 0.8s ease", display: "flex", alignItems: "center", justifyContent: "flex-end", paddingRight: 6 }}>
                      <span style={{ fontSize: 10, color: "#0f172a", fontWeight: 800 }}>{(r.metrics[m] * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                  <div style={{ fontSize: 10, color: "#475569", marginTop: 2, textAlign: "center" }}>
                    {MODEL_INFO[r.model_name]?.label || r.model_name}
                  </div>
                </div>
              ))}
            </div>
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
    <div style={{ background: `linear-gradient(135deg, ${info?.color || "#6ee7b7"}15, #0a162800)`, border: `2px solid ${info?.color || "#6ee7b7"}55`, borderRadius: 12, padding: "16px 24px", marginBottom: 20, display: "flex", alignItems: "center", gap: 16, animation: "fadeIn 0.5s ease-out" }}>
      <span style={{ fontSize: 32 }}>🏆</span>
      <div>
        <div style={{ fontSize: 11, color: "#64748b", textTransform: "uppercase", letterSpacing: 2 }}>Best Model (by F1-Score)</div>
        <div style={{ fontSize: 18, fontWeight: 800, color: info?.color || "#6ee7b7" }}>
          {info?.label || best.model_name} — F1: {(best.metrics.f1_score * 100).toFixed(1)}%
        </div>
      </div>
      <div style={{ marginLeft: "auto", textAlign: "right" }}>
        <Tag color={info?.color || "#6ee7b7"}>{best.run_id.slice(0, 8)}</Tag>
      </div>
    </div>
  );
}
