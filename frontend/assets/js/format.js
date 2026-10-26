// Formatting and small DOM helpers. All user-visible text from the API goes
// through escapeHtml before it is put into markup.

const usd = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 });
const day = new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
const shortDay = new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", timeZone: "UTC" });
const stamp = new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });

export function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

/** Only http(s) links from the data are rendered as links. */
export function safeUrl(value) {
  try {
    const url = new URL(String(value));
    return url.protocol === "http:" || url.protocol === "https:" ? url.href : null;
  } catch {
    return null;
  }
}

export const money = (v) => (v == null || Number.isNaN(v) ? "–" : usd.format(v));

/** Fraction -> percent text: 0.1234 -> "12.3%". */
export const pct = (v, digits = 1) => (v == null ? "–" : `${(v * 100).toFixed(digits)}%`);

/** Signed number with a real minus sign: 1.5 -> "+1.50", -2 -> "−2.00". */
export function signed(v, digits = 2) {
  if (v == null) return "–";
  const text = Math.abs(v).toFixed(digits);
  return v > 0 ? `+${text}` : v < 0 ? `−${text}` : text;
}

export const num = (v, digits = 2) => (v == null ? "–" : Number(v).toFixed(digits));

/** "2026-10-23" -> Date at UTC midnight (API dates are calendar days). */
export const parseDay = (iso) => new Date(`${String(iso).slice(0, 10)}T00:00:00Z`);
export const fmtDay = (iso) => day.format(parseDay(iso));
export const fmtShortDay = (iso) => shortDay.format(parseDay(iso));
export const fmtStamp = (iso) => stamp.format(new Date(iso));

/** Create an element from an HTML string (single root). */
export function html(markup) {
  const template = document.createElement("template");
  template.innerHTML = markup.trim();
  return template.content.firstElementChild;
}

/** Debounce: run fn after `ms` without further calls. */
export function debounce(fn, ms) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}

export const ARROW_UP = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 3l5 6H3z" fill="currentColor"/></svg>';
export const ARROW_DOWN = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 13L3 7h10z" fill="currentColor"/></svg>';
