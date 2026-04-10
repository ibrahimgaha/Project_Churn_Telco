// ─── Shared Styles ─────────────────────────────────────────────────────────
// Centralised style objects used across all tabs.

const S = {
  section: {
    background: "#0a1628",
    border: "1px solid #1e293b",
    borderRadius: 12,
    padding: 24,
    marginBottom: 20,
    animation: "fadeIn 0.4s ease-out both",
  },
  sectionTitle: {
    fontSize: 13,
    fontWeight: 700,
    color: "#94a3b8",
    marginBottom: 16,
    textTransform: "uppercase",
    letterSpacing: 2,
  },
  btn: {
    background: "#6ee7b7",
    color: "#0f172a",
    border: "none",
    borderRadius: 8,
    padding: "10px 22px",
    fontSize: 13,
    fontWeight: 800,
    cursor: "pointer",
    letterSpacing: 0.5,
    transition: "all 0.2s",
  },
  btnOutline: {
    background: "transparent",
    color: "#6ee7b7",
    border: "1px solid #6ee7b755",
    borderRadius: 8,
    padding: "8px 18px",
    fontSize: 12,
    fontWeight: 700,
    cursor: "pointer",
    transition: "all 0.2s",
  },
  input: {
    background: "#0f172a",
    border: "1px solid #1e293b",
    borderRadius: 6,
    color: "#e2e8f0",
    padding: "7px 12px",
    fontSize: 13,
    width: "100%",
    outline: "none",
    fontFamily: "inherit",
  },
  label: {
    fontSize: 11,
    color: "#64748b",
    display: "block",
    marginBottom: 5,
    letterSpacing: 1,
    textTransform: "uppercase",
  },
  errorBox: {
    background: "#7f1d1d22",
    border: "1px solid #ef444444",
    borderRadius: 8,
    padding: "12px 16px",
    color: "#fca5a5",
    fontSize: 13,
    marginBottom: 16,
  },
};

export default S;
