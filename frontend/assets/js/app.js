// FinSight portal: routing, views and system status.
//
// Routes (hash-based, so nginx serves one static page):
//   #/              overview: stock list, optimised portfolio, live sentiment demo
//   #/stock/AAPL    one stock: overview, price history, forecast, news sentiment
//   #/pipeline      Pipeline Console: DAG, run history, charts, run/retrain (Day 7)
//
// Every card loads on its own and shows its own loading, empty and error
// state, so one failing dependency (say, the forecasting model) never blanks
// the whole page.

import { BACKEND_URL, MLFLOW_URL, endpoints } from "./api.js";
import { allocationPie, countsChart, predictionChart, priceChart, sentimentChart, toSlices } from "./charts.js";
import {
  ARROW_DOWN,
  ARROW_UP,
  escapeHtml,
  fmtDay,
  fmtStamp,
  html,
  money,
  num,
  pct,
  safeUrl,
  signed,
} from "./format.js";
import { renderPipeline } from "./pipeline.js";
import { initSearch } from "./search.js";

const PERIODS = ["1W", "1M", "3M", "6M", "1Y"];
const SENTIMENT_WEEKS = [4, 12, 26];
const MAX_TEXT_CHARS = 2000; // same limit as the backend
const MAX_TEXTS = 32;
const STRATEGY_NAMES = {
  max_sharpe: "Maximum Sharpe ratio",
  min_volatility: "Minimum volatility",
  equal_weight: "Equal weight",
};

const view = document.getElementById("view");
let controller = new AbortController(); // aborted when the user navigates away
let modelInfo = null; // cached /api/model-info answer

// --------------------------------------------------------------- utilities
function skeleton(kind = "block") {
  return kind === "lines"
    ? '<div class="skeleton line"></div><div class="skeleton line"></div><div class="skeleton line short"></div>'
    : '<div class="skeleton block" aria-hidden="true"></div><p class="visually-hidden">Loading…</p>';
}

/** Human message for an API error; `context` names what failed to load. */
function errorMessage(error, context) {
  if (error.unreachable) {
    return `Cannot reach the FinSight API at <code>${escapeHtml(BACKEND_URL)}</code>. The backend may still be
      starting or may be stopped. Check it with <code>docker compose ps</code>.`;
  }
  const known = {
    model_unavailable: "The model server is not responding right now, and there is no earlier result to show.",
    database_unavailable: "The database is temporarily unavailable.",
    insufficient_history: "There is not enough price history for this stock yet.",
  };
  const text = known[error.code] || escapeHtml(error.message);
  const id = error.requestId ? ` <span class="muted">(request ${escapeHtml(error.requestId)})</span>` : "";
  return `Could not load ${escapeHtml(context)}. ${text}${id}`;
}

function showError(container, error, context, retry) {
  if (error.name === "AbortError") return;
  container.innerHTML = `<div class="notice error" role="alert">${errorMessage(error, context)}
    ${retry ? '<div><button class="btn" type="button" data-retry>Try again</button></div>' : ""}</div>`;
  container.querySelector("[data-retry]")?.addEventListener("click", retry);
}

function showEmpty(container, message) {
  container.innerHTML = `<div class="notice empty">${message}</div>`;
}

const isAbort = (error) => error?.name === "AbortError" || controller.signal.aborted;

function dataTable(caption, headers, rows) {
  return `<details class="data-table"><summary>${escapeHtml(caption)}</summary>
    <div class="table-wrap"><table>
      <thead><tr>${headers.map((h) => `<th scope="col">${escapeHtml(h)}</th>`).join("")}</tr></thead>
      <tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody>
    </table></div></details>`;
}

async function getModelInfo() {
  if (modelInfo) return modelInfo;
  try {
    const data = await endpoints.modelInfo();
    modelInfo = Object.fromEntries(data.models.map((m) => [m.name, m]));
  } catch {
    modelInfo = {};
  }
  return modelInfo;
}

function setActiveNav(name) {
  document.querySelectorAll("[data-nav]").forEach((a) => {
    if (a.dataset.nav === name) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
}

// ------------------------------------------------------------- home view
function renderHome() {
  document.title = "FinSight · Overview";
  setActiveNav("home");
  view.innerHTML = `
    <h1>Market overview</h1>
    <p class="page-intro">Pick a stock to see its price history, a 14-day price forecast and the
      sentiment of recent news. Below is today's optimised portfolio across all ten stocks.</p>
    <div class="grid">
      <section class="card span-12" aria-labelledby="stocks-h">
        <div class="card-head"><h2 id="stocks-h">Stocks</h2>
          <span class="sub">Ten large-cap US stocks · press <kbd>/</kbd> to search</span></div>
        <div id="stocks-body">${skeleton("lines")}</div>
      </section>
      <section class="card span-7" aria-labelledby="pf-h">
        <div class="card-head"><h2 id="pf-h">Optimised portfolio</h2><span class="sub" id="pf-sub"></span></div>
        <div id="pf-body">${skeleton()}</div>
      </section>
      <section class="card span-5 analyze" aria-labelledby="an-h">
        <div class="card-head"><h2 id="an-h">Try the sentiment model</h2></div>
        <p class="caption">Paste one or more headlines, one per line. The production model scores each one
          from −1 (negative) to +1 (positive).</p>
        <label class="visually-hidden" for="an-text">Headlines to analyse</label>
        <textarea id="an-text" maxlength="${MAX_TEXT_CHARS * 4}"
          placeholder="Apple shares rose after strong iPhone sales&#10;Tesla recalls 50,000 vehicles over a braking fault"></textarea>
        <div class="row">
          <button class="btn primary" id="an-go" type="button" disabled>Analyse</button>
          <span class="muted" id="an-hint">Up to ${MAX_TEXTS} lines, ${MAX_TEXT_CHARS} characters each.</span>
        </div>
        <div id="an-out" aria-live="polite"></div>
      </section>
    </div>`;
  loadStocks();
  loadPortfolio();
  initAnalyze();
}

async function loadStocks() {
  const body = document.getElementById("stocks-body");
  try {
    const { stocks } = await endpoints.stocks({ signal: controller.signal });
    body.innerHTML = `<div class="stock-tiles">${stocks
      .map(
        (s) => `<a class="stock-tile" href="#/stock/${encodeURIComponent(s.symbol)}">
          <span class="sym">${escapeHtml(s.symbol)}</span>
          <span class="name">${escapeHtml(s.name)}</span>
          <span class="sector">${escapeHtml(s.sector)}</span></a>`,
      )
      .join("")}</div>`;
  } catch (error) {
    if (!isAbort(error)) showError(body, error, "the stock list", loadStocks);
  }
}

async function loadPortfolio() {
  const body = document.getElementById("pf-body");
  const sub = document.getElementById("pf-sub");
  body.innerHTML = skeleton();
  try {
    const p = await endpoints.portfolio({ signal: controller.signal });
    sub.textContent = `As of ${fmtDay(p.as_of)} · last ${p.window_days} trading days`;
    const a = p.allocation;
    const slices = toSlices(a.weights);
    const oos = p.out_of_sample || {};
    const oosRows = Object.entries(STRATEGY_NAMES)
      .filter(([key]) => oos[key])
      .map(([key, name]) => [escapeHtml(name), num(oos[key].sharpe), pct(oos[key].annual_return), pct(oos[key].annual_volatility)]);
    body.innerHTML = `
      <dl class="stats">
        <div class="stat"><dt>Sharpe ratio</dt><dd class="tabular">${num(a.sharpe)}</dd></div>
        <div class="stat"><dt>Expected return (yearly)</dt><dd class="tabular">${pct(a.expected_return)}</dd></div>
        <div class="stat"><dt>Volatility (yearly)</dt><dd class="tabular">${pct(a.volatility)}</dd></div>
      </dl>
      <div class="pie-row">
        <div class="chart-box pie"><canvas id="pf-pie" role="img"
          aria-label="Portfolio allocation: ${escapeHtml(slices.map((s) => `${s.label} ${pct(s.weight, 0)}`).join(", "))}"></canvas></div>
        <ul class="weights" id="pf-weights"></ul>
      </div>
      <p class="caption">Method: <strong>${escapeHtml(STRATEGY_NAMES[a.method] || a.method)}</strong>, long-only,
        searched over ${p.grid_size.toLocaleString("en-US")} weight combinations in 10% steps, risk-free rate
        ${pct(p.risk_free_rate)}. Window ${fmtDay(p.window_start)} to ${fmtDay(p.window_end)}.
        ${p.fallback_used ? "<br><strong>Note:</strong> every portfolio lost money in this window, so the safest (minimum-volatility) mix is shown instead." : ""}</p>
      <div class="verdict"><strong>Reality check.</strong> Re-running this optimisation every week over the past
        year, ${
          p.beats_equal_weight_out_of_sample
            ? "it <strong>did beat</strong> simply splitting money equally across all ten stocks."
            : "it <strong>did not beat</strong> simply splitting money equally across all ten stocks. Optimising on 30 days of prices tends to overfit."
        }
        ${oosRows.length ? dataTable("Past-year walk-forward results", ["Strategy", "Sharpe", "Return", "Volatility"], oosRows) : ""}
      </div>`;
    const colours = allocationPie(document.getElementById("pf-pie"), slices);
    document.getElementById("pf-weights").innerHTML = slices
      .map(
        (s, i) => `<li><span class="swatch" style="background:${colours[i]}"></span>
          ${s.other ? escapeHtml(s.label) : `<a href="#/stock/${encodeURIComponent(s.label)}">${escapeHtml(s.label)}</a>`}
          <span class="w">${pct(s.weight, 0)}</span></li>`,
      )
      .join("");
  } catch (error) {
    if (isAbort(error)) return;
    sub.textContent = "";
    if (error.code === "no_portfolio") {
      showEmpty(body, "No portfolio has been optimised yet. It appears after the daily pipeline runs (<code>dvc repro optimize_portfolio</code>).");
    } else {
      showError(body, error, "the portfolio", loadPortfolio);
    }
  }
}

function initAnalyze() {
  const text = document.getElementById("an-text");
  const go = document.getElementById("an-go");
  const hint = document.getElementById("an-hint");
  const out = document.getElementById("an-out");
  const lines = () => text.value.split("\n").map((l) => l.trim()).filter(Boolean);

  const validate = () => {
    const items = lines();
    const tooLong = items.findIndex((l) => l.length > MAX_TEXT_CHARS);
    let problem = "";
    if (items.length > MAX_TEXTS) problem = `Too many lines: ${items.length} (at most ${MAX_TEXTS}).`;
    else if (tooLong >= 0) problem = `Line ${tooLong + 1} is longer than ${MAX_TEXT_CHARS} characters.`;
    hint.textContent = problem || `Up to ${MAX_TEXTS} lines, ${MAX_TEXT_CHARS} characters each.`;
    hint.classList.toggle("muted", !problem);
    go.disabled = !items.length || Boolean(problem);
  };

  const run = async () => {
    const items = lines();
    go.disabled = true;
    go.textContent = "Analysing…";
    out.innerHTML = skeleton("lines");
    try {
      const data = await endpoints.analyze(items, { signal: controller.signal });
      out.innerHTML = `<ul class="result-list">${data.results
        .map(
          (r) => `<li><span class="chip ${r.label}">${escapeHtml(r.label)}</span>
            <span class="tabular">score ${signed(r.score)} · confidence ${pct(r.confidence, 0)}</span>
            <div>${escapeHtml(r.text)}</div></li>`,
        )
        .join("")}</ul>
        <p class="caption">Model ${escapeHtml(data.model.name)} version ${escapeHtml(data.model.version ?? "unknown")}.</p>`;
    } catch (error) {
      if (!isAbort(error)) showError(out, error, "the sentiment analysis", run);
    } finally {
      go.textContent = "Analyse";
      validate();
    }
  };

  text.addEventListener("input", validate);
  text.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && !go.disabled) run();
  });
  go.addEventListener("click", run);
}

// ------------------------------------------------------------ stock view
function renderStock(symbol) {
  setActiveNav("stock");
  document.title = `FinSight · ${symbol}`;
  view.innerHTML = `
    <div class="grid">
      <section class="card span-12" id="ov" aria-label="Overview">${skeleton("lines")}</section>
      <section class="card span-12" aria-labelledby="ph-h">
        <div class="card-head"><h2 id="ph-h">Price history</h2><span class="sub">Daily closing price, USD</span>
          <div class="tools"><div class="segmented" role="radiogroup" aria-label="Time period" id="periods">
            ${PERIODS.map((p) => `<button type="button" role="radio" aria-checked="${p === "1M"}" data-period="${p}">${p}</button>`).join("")}
          </div></div></div>
        <div id="ph-body">${skeleton()}</div>
      </section>
      <section class="card span-6" aria-labelledby="fc-h">
        <div class="card-head"><h2 id="fc-h">14-day forecast</h2><span class="sub" id="fc-sub"></span></div>
        <div id="fc-body">${skeleton()}</div>
      </section>
      <section class="card span-6" aria-labelledby="se-h">
        <div class="card-head"><h2 id="se-h">News sentiment</h2><span class="sub" id="se-sub"></span>
          <div class="tools"><div class="segmented" role="radiogroup" aria-label="Number of weeks" id="weeks">
            ${SENTIMENT_WEEKS.map((w) => `<button type="button" role="radio" aria-checked="${w === 12}" data-weeks="${w}">${w}w</button>`).join("")}
          </div></div></div>
        <div id="se-body">${skeleton()}</div>
      </section>
    </div>`;

  wireSegmented("periods", "period", (p) => loadPrices(symbol, p));
  wireSegmented("weeks", "weeks", (w) => loadSentiment(symbol, Number(w)));
  loadPrices(symbol, "1M", true);
  loadForecast(symbol);
  loadSentiment(symbol, 12);
}

/** Radio-group behaviour for a segmented control (click + arrow keys). */
function wireSegmented(id, attr, onChange) {
  const group = document.getElementById(id);
  const buttons = [...group.querySelectorAll("button")];
  const select = (button) => {
    if (button.getAttribute("aria-checked") === "true" || button.disabled) return;
    buttons.forEach((b) => {
      b.setAttribute("aria-checked", String(b === button));
      b.tabIndex = b === button ? 0 : -1;
    });
    button.focus();
    onChange(button.dataset[attr]);
  };
  buttons.forEach((b) => {
    b.tabIndex = b.getAttribute("aria-checked") === "true" ? 0 : -1;
    b.addEventListener("click", () => select(b));
    b.addEventListener("keydown", (e) => {
      const i = buttons.indexOf(b);
      if (e.key === "ArrowRight" || e.key === "ArrowDown") {
        e.preventDefault();
        select(buttons[(i + 1) % buttons.length]);
      } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
        e.preventDefault();
        select(buttons[(i - 1 + buttons.length) % buttons.length]);
      }
    });
  });
}

function setBusy(groupId, busy) {
  document.querySelectorAll(`#${groupId} button`).forEach((b) => {
    b.disabled = busy;
  });
}

let priceSeq = 0;
async function loadPrices(symbol, period, withOverview = false) {
  const body = document.getElementById("ph-body");
  const mine = ++priceSeq;
  setBusy("periods", true);
  body.innerHTML = skeleton();
  try {
    const data = await endpoints.stock(symbol, period, { signal: controller.signal });
    if (mine !== priceSeq) return;
    if (withOverview) renderOverview(data);
    if (!data.history.length) {
      showEmpty(body, `No prices in this period. Try a longer one.`);
      return;
    }
    const first = data.history[0].close;
    const last = data.history[data.history.length - 1].close;
    const change = (last - first) / first;
    body.innerHTML = `<div class="chart-box"><canvas id="ph-chart" role="img"
        aria-label="${escapeHtml(symbol)} closing price over ${period}: from ${money(first)} to ${money(last)}"></canvas></div>
      <p class="caption">Over this period: <span class="change ${change > 0 ? "up" : change < 0 ? "down" : "flat"}">${signed(change * 100, 1)}%</span>
        (${data.history.length} trading days, ${fmtDay(data.history[0].date)} to ${fmtDay(data.history[data.history.length - 1].date)}).</p>
      ${dataTable(
        "Show the prices as a table",
        ["Date", "Open", "High", "Low", "Close", "Volume"],
        data.history
          .slice()
          .reverse()
          .map((p) => [fmtDay(p.date), money(p.open), money(p.high), money(p.low), money(p.close), Math.round(p.volume).toLocaleString("en-US")]),
      )}`;
    priceChart(document.getElementById("ph-chart"), data.history);
  } catch (error) {
    if (isAbort(error) || mine !== priceSeq) return;
    if (error.code === "unknown_ticker") {
      renderNotFound(symbol);
    } else if (error.code === "no_data") {
      document.getElementById("ov").innerHTML = `<h1>${escapeHtml(symbol)}</h1>`;
      showEmpty(body, `No prices are stored for ${escapeHtml(symbol)} yet. Run the ingestion job first.`);
    } else {
      if (withOverview) showError(document.getElementById("ov"), error, `${symbol}`, () => loadPrices(symbol, period, true));
      showError(body, error, "the price history", () => loadPrices(symbol, period, withOverview));
    }
  } finally {
    if (mine === priceSeq) setBusy("periods", false);
  }
}

function renderOverview(data) {
  const o = data.overview;
  const icon = o.direction === "up" ? ARROW_UP : o.direction === "down" ? ARROW_DOWN : "";
  const words = o.direction === "up" ? "up" : o.direction === "down" ? "down" : "unchanged";
  document.title = `FinSight · ${data.symbol} ${money(o.close)}`;
  document.getElementById("ov").innerHTML = `
    <div class="overview">
      <div class="title">
        <h1>${escapeHtml(data.name)} <span class="sym-badge">${escapeHtml(data.symbol)}</span></h1>
        <div class="meta">${escapeHtml(data.sector)}</div>
      </div>
      <div>
        <div class="price tabular">${money(o.close)}</div>
        <div class="change ${o.direction}" aria-label="${words} ${money(Math.abs(o.change ?? 0))} on the day">
          ${icon}${o.change == null ? "–" : `${signed(o.change)} (${signed(o.change_pct)}%)`}
        </div>
      </div>
      <div class="meta">Close of ${fmtDay(o.date)}${o.previous_close != null ? ` · previous close ${money(o.previous_close)}` : ""}
        <br>End-of-day data, not live prices · source: ${escapeHtml(data.source)}</div>
    </div>`;
}

async function loadForecast(symbol) {
  const body = document.getElementById("fc-body");
  const sub = document.getElementById("fc-sub");
  body.innerHTML = skeleton();
  try {
    const [p, models] = await Promise.all([endpoints.prediction(symbol, { signal: controller.signal }), getModelInfo()]);
    const tags = models["finsight-lstm"]?.tags || {};
    const last = p.history[p.history.length - 1]?.price;
    const end = p.forecast[p.forecast.length - 1]?.price;
    const move = last && end ? (end - last) / last : null;
    sub.textContent = `From the close of ${fmtDay(p.as_of)}`;
    const evidence =
      tags.test_mape && tags.naive_mape
        ? `In back-tests its average error was ${num(Number(tags.test_mape))}%, against ${num(Number(tags.naive_mape))}% for simply assuming
           the price stays the same, so treat the line as a rough guide, not a prediction to trade on.`
        : "Forecasts of daily stock prices are highly uncertain; treat the line as a rough guide.";
    body.innerHTML = `
      ${p.stale ? `<div class="notice warn" role="status">The forecasting model is not responding, so this is the most recent saved forecast (made ${fmtStamp(p.generated_at)}).</div>` : ""}
      <ul class="legend" aria-hidden="true">
        <li><span class="key" style="border-color:var(--series-1)"></span>Closing price</li>
        <li><span class="key dashed" style="border-color:var(--series-2)"></span>Forecast</li>
      </ul>
      <div class="chart-box"><canvas id="fc-chart" role="img"
        aria-label="${escapeHtml(symbol)} forecast: ${money(last)} now, ${money(end)} in ${p.horizon} trading days"></canvas></div>
      <p class="caption">In ${p.horizon} trading days the model expects <strong>${money(end)}</strong>
        ${move != null ? `(${signed(move * 100, 1)}% from ${money(last)})` : ""}. ${evidence}</p>
      <p class="caption muted">Model ${escapeHtml(p.model.name)} version ${escapeHtml(p.model.version ?? "unknown")}
        · generated ${fmtStamp(p.generated_at)}${p.stale ? ' · <span class="chip stale">saved copy</span>' : ""}</p>
      ${dataTable("Show the forecast as a table", ["Date", "Forecast price"], p.forecast.map((f) => [fmtDay(f.date), money(f.price)]))}`;
    predictionChart(document.getElementById("fc-chart"), p.history, p.forecast);
  } catch (error) {
    if (isAbort(error)) return;
    sub.textContent = "";
    showError(body, error, "the forecast", () => loadForecast(symbol));
  }
}

let sentimentSeq = 0;
async function loadSentiment(symbol, weeks) {
  const body = document.getElementById("se-body");
  const sub = document.getElementById("se-sub");
  const mine = ++sentimentSeq;
  setBusy("weeks", true);
  body.innerHTML = skeleton();
  try {
    const s = await endpoints.sentiment(symbol, weeks, { signal: controller.signal });
    if (mine !== sentimentSeq) return;
    if (!s.weeks.length) {
      sub.textContent = "";
      showEmpty(body, `No scored news for ${escapeHtml(symbol)} in the last ${weeks} weeks.`);
      return;
    }
    sub.textContent = `${s.total_articles.toLocaleString("en-US")} articles`;
    const mood = s.overall_mean_score > 0.05 ? "leaning positive" : s.overall_mean_score < -0.05 ? "leaning negative" : "roughly neutral";
    body.innerHTML = `
      <div class="chart-box"><canvas id="se-chart" role="img"
        aria-label="Weekly news sentiment for ${escapeHtml(symbol)}, average ${signed(s.overall_mean_score)}"></canvas></div>
      <p class="caption"><strong>Articles per week</strong></p>
      <div class="chart-box small"><canvas id="se-counts" role="img" aria-label="Number of articles per week"></canvas></div>
      <p class="caption">Average score <strong>${signed(s.overall_mean_score)}</strong>, ${mood}. Scores run from −1 (negative)
        to +1 (positive); most news is neutral. Model version ${escapeHtml(s.model_version ?? "unknown")}.</p>
      <div class="headlines" id="se-headlines"></div>
      ${dataTable(
        "Show the weeks as a table",
        ["Week of", "Mean score", "Articles", "Positive", "Negative", "Neutral"],
        s.weeks.map((w) => [fmtDay(w.week_start), signed(w.mean_score), w.article_count, w.positive, w.negative, w.neutral]),
      )}`;
    const showWeek = (i) => renderHeadlines(s.weeks[i]);
    sentimentChart(document.getElementById("se-chart"), s.weeks, showWeek);
    countsChart(document.getElementById("se-counts"), s.weeks);
    showWeek(s.weeks.length - 1);
  } catch (error) {
    if (isAbort(error) || mine !== sentimentSeq) return;
    sub.textContent = "";
    showError(body, error, "news sentiment", () => loadSentiment(symbol, weeks));
  } finally {
    if (mine === sentimentSeq) setBusy("weeks", false);
  }
}

function renderHeadlines(week) {
  const box = document.getElementById("se-headlines");
  if (!box) return;
  const item = (h) => {
    const url = safeUrl(h.url);
    const text = escapeHtml(h.headline);
    const cls = h.score >= 0 ? "pos" : "neg";
    return `<li><span class="score ${cls}">${signed(h.score)}</span> ${url ? `<a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${text}</a>` : text}
      ${h.source ? `<span class="muted">· ${escapeHtml(h.source)}</span>` : ""}</li>`;
  };
  const list = (title, items) => (items.length ? `<h3>${title}</h3><ul>${items.map(item).join("")}</ul>` : "");
  const content = list("Most positive", week.top_positive) + list("Most negative", week.top_negative);
  box.innerHTML = `<h3>Why this week scored ${signed(week.mean_score)}: week of ${fmtDay(week.week_start)}</h3>
    ${content || '<p class="muted">No clearly positive or negative headlines this week.</p>'}
    <p class="caption muted">Click another bar to see its headlines.</p>`;
}

function renderNotFound(symbol) {
  document.title = "FinSight · Not found";
  view.innerHTML = `<section class="card"><h1>“${escapeHtml(symbol)}” is not covered</h1>
    <p>FinSight analyses a fixed set of ten large-cap US stocks. Use the search box, or go back to the
    <a href="#/">overview</a> to pick one.</p></section>`;
}

// --------------------------------------------------------------- status
const pill = document.getElementById("status-pill");
const panel = document.getElementById("status-panel");
let lastHealth = null;

async function refreshHealth() {
  try {
    lastHealth = await endpoints.health();
  } catch (error) {
    lastHealth = { status: "down", error };
  }
  const state = lastHealth.status;
  pill.dataset.state = state;
  const failing = state === "degraded" ? Object.entries(lastHealth.dependencies).filter(([, d]) => !d.ok).map(([n]) => n) : [];
  pill.querySelector(".label").textContent =
    state === "ok" ? "All systems normal" : state === "degraded" ? `Degraded: ${failing.join(", ")}` : "API unreachable";
  if (!panel.hidden) renderStatusPanel();
}

async function renderStatusPanel() {
  if (!lastHealth || lastHealth.status === "down") {
    panel.innerHTML = `<h2>System status</h2><p>${errorMessage(lastHealth?.error || { unreachable: true }, "")}</p>`;
    return;
  }
  const models = await getModelInfo();
  const deps = Object.entries(lastHealth.dependencies)
    .map(([name, d]) => `<li><span>${escapeHtml(name)}</span><span class="${d.ok ? "ok" : "bad"}">${d.ok ? "up" : "down"}</span></li>`)
    .join("");
  const versions = Object.values(models)
    .map((m) => `<li><span>${escapeHtml(m.name)}</span><span>version ${escapeHtml(m.version ?? "?")}</span></li>`)
    .join("");
  panel.innerHTML = `<h2>System status</h2><ul>${deps}</ul>
    ${versions ? `<h2>Production models</h2><ul>${versions}</ul>` : ""}
    <p class="muted">API ${escapeHtml(lastHealth.version)} at ${escapeHtml(BACKEND_URL)}. When a part is down, the rest of the
    portal keeps working; forecasts fall back to the last saved copy.</p>`;
}

function togglePanel(show) {
  panel.hidden = !show;
  pill.setAttribute("aria-expanded", String(show));
  if (show) renderStatusPanel();
}

pill.addEventListener("click", () => togglePanel(panel.hidden));
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !panel.hidden) {
    togglePanel(false);
    pill.focus();
  }
});
document.addEventListener("click", (e) => {
  if (!panel.hidden && !panel.contains(e.target) && !pill.contains(e.target)) togglePanel(false);
});

// --------------------------------------------------------------- router
// Helpers the Pipeline Console shares with the other pages.
const pipelineUi = { skeleton, showError, showEmpty, dataTable, isAbort, setActiveNav, wireSegmented };

function route() {
  controller.abort();
  controller = new AbortController();
  const match = location.hash.match(/^#\/stock\/([A-Za-z.\-]{1,10})$/);
  if (match) renderStock(decodeURIComponent(match[1]).toUpperCase());
  else if (location.hash === "#/pipeline") renderPipeline(view, controller.signal, pipelineUi);
  else renderHome();
  window.scrollTo(0, 0);
}

function rerenderCharts() {
  // Colours come from CSS tokens, so a light/dark switch needs a re-render.
  route();
}

document.getElementById("mlflow-link").href = MLFLOW_URL;
document.getElementById("api-docs-link").href = `${BACKEND_URL}/docs`;
initSearch({
  input: document.getElementById("search-input"),
  list: document.getElementById("search-results"),
  onPick: (symbol) => {
    location.hash = `#/stock/${encodeURIComponent(symbol)}`;
  },
});
window.addEventListener("hashchange", route);
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", rerenderCharts);
route();
refreshHealth();
setInterval(refreshHealth, 30000);
