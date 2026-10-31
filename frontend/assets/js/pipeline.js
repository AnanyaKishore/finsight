// Pipeline Console (#/pipeline): the DVC DAG coloured by status, run history
// with error drill-down, duration and throughput charts, and the "Run now" /
// "Retrain now" buttons.
//
// Everything is read from the backend's /api/pipeline/* endpoints, which read
// MongoDB, so the page keeps working while the scheduler container is down;
// only the buttons need the scheduler. While a run is in progress the page
// polls every 5 s, otherwise every 30 s.

import { ALERTMANAGER_URL, GRAFANA_URL, MLFLOW_URL, PROMETHEUS_URL, endpoints } from "./api.js";
import { durationChart, throughputChart } from "./charts.js";
import { escapeHtml, fmtDuration, fmtStamp } from "./format.js";

const FAST_POLL_MS = 5000;
const SLOW_POLL_MS = 30000;
const STATUS_FILTERS = [
  ["", "All"],
  ["failed", "Failed"],
  ["partial", "Partial"],
  ["success", "Succeeded"],
];
const KIND_NAMES = { daily: "Daily run", retrain: "Retrain" };
const TRIGGER_NAMES = {
  schedule: "on schedule",
  "catch-up": "catch-up after the laptop woke",
  console: "from the console",
  webhook: "by an alert",
  api: "through the API",
};
const STATUS_WORDS = {
  success: "Succeeded",
  partial: "Partly succeeded",
  failed: "Failed",
  running: "Running",
  interrupted: "Interrupted",
  pending: "Waiting",
  skipped: "Skipped",
  never: "Not run yet",
};
const METRIC_NAMES = {
  prices_added: "prices added",
  articles_added: "articles added",
  ingest_rows_per_s: "ingest rows/s",
  etl_rows: "ETL rows",
  etl_rows_per_s: "ETL rows/s",
  articles_scored: "articles scored",
  scoring_articles_per_s: "scoring articles/s",
  lstm_promoted: "LSTM promoted",
  stages_run: "DVC stages run",
};

/**
 * Render the console into `view`. `ui` carries helpers shared with the other
 * pages (skeleton, showError, dataTable, isAbort, setActiveNav, wireSegmented)
 * and `signal` is aborted when the user navigates away.
 */
export function renderPipeline(view, signal, ui) {
  document.title = "FinSight · Pipeline";
  ui.setActiveNav("pipeline");
  const state = { filter: "", durationKind: "daily", selected: null, busy: false, timer: null, runs: [] };

  view.innerHTML = `
    <h1>Pipeline console</h1>
    <p class="page-intro">Every weekday morning the scheduler fetches new prices and news, runs the DVC pipeline
      (tests, validation, Spark ETL, sentiment scoring, portfolio) and stores fresh forecasts. Retraining runs only when
      you ask for it or when a monitoring alert calls for it.</p>
    <div class="grid">
      <section class="card span-12" aria-labelledby="ps-h">
        <div class="card-head"><h2 id="ps-h">Scheduler</h2><span class="sub" id="ps-sub"></span>
          <div class="tools">
            <button class="btn primary" type="button" data-trigger="daily" disabled>Run now</button>
            <button class="btn" type="button" data-trigger="retrain" disabled>Retrain now</button>
          </div></div>
        <div id="ps-confirm"></div>
        <div id="ps-body">${ui.skeleton("lines")}</div>
        <p class="caption ops-links">Tools:
          <a href="${escapeHtml(MLFLOW_URL)}" target="_blank" rel="noopener">MLflow experiments and registry</a> ·
          <a href="${escapeHtml(PROMETHEUS_URL)}" target="_blank" rel="noopener">Prometheus</a> ·
          <a href="${escapeHtml(GRAFANA_URL)}" target="_blank" rel="noopener">Grafana dashboards</a> ·
          <a href="${escapeHtml(ALERTMANAGER_URL)}" target="_blank" rel="noopener">Alertmanager</a></p>
      </section>
      <section class="card span-12" aria-labelledby="mo-h">
        <div class="card-head"><h2 id="mo-h">Monitoring</h2><span class="sub" id="mo-sub"></span></div>
        <div id="mo-body">${ui.skeleton("lines")}</div>
      </section>
      <section class="card span-12" aria-labelledby="st-h">
        <div class="card-head"><h2 id="st-h">Last 30 days</h2></div>
        <div id="st-body">${ui.skeleton("lines")}</div>
      </section>
      <section class="card span-12" aria-labelledby="dag-h">
        <div class="card-head"><h2 id="dag-h">Pipeline stages (DVC DAG)</h2><span class="sub" id="dag-sub"></span></div>
        <ul class="legend dag-legend" aria-hidden="true">
          <li><span class="swatch st-success"></span>Last run succeeded</li>
          <li><span class="swatch st-failed"></span>Last run failed</li>
          <li><span class="swatch st-running"></span>Running</li>
          <li><span class="swatch st-never"></span>Not run yet</li>
          <li><span class="stale-dot"></span>Inputs changed since it last ran</li>
        </ul>
        <div id="dag-body">${ui.skeleton()}</div>
        <div id="dag-detail" aria-live="polite"></div>
      </section>
      <section class="card span-7" aria-labelledby="rh-h">
        <div class="card-head"><h2 id="rh-h">Run history</h2>
          <div class="tools"><div class="segmented" role="radiogroup" aria-label="Show runs" id="run-filter">
            ${STATUS_FILTERS.map(([v, l]) => `<button type="button" role="radio" aria-checked="${v === ""}" data-status="${v}">${l}</button>`).join("")}
          </div></div></div>
        <div id="rh-body">${ui.skeleton("lines")}</div>
      </section>
      <section class="card span-5" aria-labelledby="rd-h" id="rd-card">
        <div class="card-head"><h2 id="rd-h">Run details</h2><span class="sub" id="rd-sub"></span></div>
        <div id="rd-body"><div class="notice empty">Pick a run in the history to see its steps, errors and log output.</div></div>
      </section>
      <section class="card span-6" aria-labelledby="du-h">
        <div class="card-head"><h2 id="du-h">Run duration</h2><span class="sub">minutes per run</span>
          <div class="tools"><div class="segmented" role="radiogroup" aria-label="Run kind" id="du-kind">
            <button type="button" role="radio" aria-checked="true" data-kind="daily">Daily</button>
            <button type="button" role="radio" aria-checked="false" data-kind="retrain">Retrain</button>
          </div></div></div>
        <div id="du-body">${ui.skeleton()}</div>
      </section>
      <section class="card span-6" aria-labelledby="tp-h">
        <div class="card-head"><h2 id="tp-h">Throughput</h2><span class="sub">per run, higher is faster</span></div>
        <div id="tp-body">${ui.skeleton()}</div>
      </section>
    </div>`;

  // ------------------------------------------------------------ scheduler
  const buttons = [...view.querySelectorAll("[data-trigger]")];
  const confirmBox = document.getElementById("ps-confirm");

  async function loadStatus() {
    const body = document.getElementById("ps-body");
    const sub = document.getElementById("ps-sub");
    try {
      const st = await endpoints.pipelineStatus({ signal });
      const wasBusy = state.busy;
      state.busy = st.busy;
      const sched = st.schedule || {};
      sub.textContent = sched.time
        ? `Daily at ${sched.time} (${sched.timezone}), ${(sched.days || []).map(cap).join(", ")}`
        : "";
      const lines = [];
      if (!st.reachable) {
        lines.push(`<div class="notice warn" role="status">The scheduler is not reachable, so no runs can be started
          from here. History below is still up to date. Start it with <code>docker compose up -d scheduler</code>.</div>`);
      } else if (st.busy) {
        lines.push(`<p class="run-state"><span class="chip st-running">Running</span>
          Run <button class="linklike" type="button" data-open-run="${escapeHtml(st.running_run_id)}">${escapeHtml(st.running_run_id)}</button>
          is on step <strong>${escapeHtml(st.current_step || "…")}</strong>.</p>`);
      } else {
        lines.push(`<p class="run-state"><span class="chip st-success">Idle</span> Waiting for the next run.</p>`);
      }
      if (st.next_daily_run) {
        lines.push(`<p class="caption">Next scheduled run: <strong>${fmtInZone(st.next_daily_run, sched.timezone)}</strong>.</p>`);
      }
      if (st.reachable && !st.triggers_enabled) {
        lines.push(`<p class="caption">The buttons are off because <code>SCHEDULER_TOKEN</code> is not set in
          <code>.env</code>. Scheduled runs still happen.</p>`);
      }
      body.innerHTML = lines.join("");
      body.querySelector("[data-open-run]")?.addEventListener("click", (e) => openRun(e.target.dataset.openRun));
      const canStart = st.reachable && st.triggers_enabled && !st.busy;
      buttons.forEach((b) => {
        b.disabled = !canStart;
        b.title = canStart ? "" : st.busy ? "A run is in progress" : "The scheduler cannot start runs right now";
      });
      if (wasBusy && !st.busy) refreshAll(); // a run just finished
      return st;
    } catch (error) {
      if (ui.isAbort(error)) return null;
      buttons.forEach((b) => (b.disabled = true));
      ui.showError(body, error, "the scheduler status", loadStatus);
      return null;
    }
  }

  function askConfirm(kind) {
    const words =
      kind === "retrain"
        ? "Retrain the LSTM and re-run the promotion gate? This runs the full DVC pipeline and takes about 20 to 60 minutes."
        : "Start the daily run now? It fetches new data, runs the pipeline and refreshes forecasts (about 5 to 25 minutes).";
    confirmBox.innerHTML = `<div class="notice confirm" role="alertdialog" aria-label="Confirm">
      <p>${words}</p>
      <div class="row"><button class="btn primary" type="button" data-go>Start</button>
      <button class="btn" type="button" data-cancel>Cancel</button></div></div>`;
    const go = confirmBox.querySelector("[data-go]");
    go.focus();
    go.addEventListener("click", () => startRun(kind));
    confirmBox.querySelector("[data-cancel]").addEventListener("click", () => (confirmBox.innerHTML = ""));
  }

  async function startRun(kind) {
    confirmBox.innerHTML = `<p class="caption">Starting…</p>`;
    buttons.forEach((b) => (b.disabled = true));
    try {
      const res = await endpoints.pipelineTrigger(kind, `${KIND_NAMES[kind]} started from the console`, { signal });
      confirmBox.innerHTML = `<div class="notice empty" role="status">${escapeHtml(KIND_NAMES[kind])} started
        (run ${escapeHtml(res.run_id)}). Progress shows below; this page refreshes every few seconds.</div>`;
      state.selected = res.run_id;
      await tick(true);
    } catch (error) {
      if (ui.isAbort(error)) return;
      const known = {
        pipeline_busy: "Another run is already in progress. Wait for it to finish.",
        scheduler_unavailable: `The scheduler did not accept the request: ${escapeHtml(error.message)}`,
      };
      confirmBox.innerHTML = `<div class="notice error" role="alert">${known[error.code] || escapeHtml(error.message)}</div>`;
      loadStatus();
    }
  }

  buttons.forEach((b) => b.addEventListener("click", () => askConfirm(b.dataset.trigger)));

  // ---------------------------------------------------------------- stats
  async function loadStats() {
    const body = document.getElementById("st-body");
    try {
      const s = await endpoints.pipelineStats(30, { signal });
      if (!s.runs) {
        ui.showEmpty(body, "No finished runs yet. The first scheduled run fills this in, or press <strong>Run now</strong>.");
        return;
      }
      const t = s.latest_throughput || {};
      const tile = (label, value) => `<div class="stat"><dt>${label}</dt><dd class="tabular">${value}</dd></div>`;
      body.innerHTML = `<dl class="stats six">
        ${tile("Runs", s.runs)}
        ${tile("Success rate", s.success_rate == null ? "–" : `${Math.round(s.success_rate * 100)}%`)}
        ${tile("Typical daily run", fmtDuration(s.median_duration_s.daily))}
        ${tile("Typical retrain", fmtDuration(s.median_duration_s.retrain))}
        ${tile("ETL rows/s", fmtRate(t.etl_rows_per_s))}
        ${tile("Articles scored/s", fmtRate(t.scoring_articles_per_s))}
      </dl>
      <p class="caption">${Object.entries(s.by_status)
        .map(([k, v]) => `${v} ${escapeHtml((STATUS_WORDS[k] || k).toLowerCase())}`)
        .join(" · ")}. Success rate counts partial runs as successful; typical = median.</p>`;
    } catch (error) {
      if (!ui.isAbort(error)) ui.showError(body, error, "the run statistics", loadStats);
    }
  }

  // ------------------------------------------------------------------ DAG
  async function loadDag() {
    const body = document.getElementById("dag-body");
    const sub = document.getElementById("dag-sub");
    try {
      const dag = await endpoints.pipelineDag({ signal });
      sub.textContent = `${dag.nodes.length} stages · structure read ${fmtStamp(dag.generated_at)}`;
      body.innerHTML = dagSvg(dag);
      wireDag(body, dag);
    } catch (error) {
      if (ui.isAbort(error)) return;
      if (error.code === "no_dag") {
        ui.showEmpty(body, "The scheduler records the pipeline structure when it starts. Start it with <code>docker compose up -d scheduler</code>.");
      } else {
        ui.showError(body, error, "the pipeline stages", loadDag);
      }
    }
  }

  function wireDag(body, dag) {
    const byName = Object.fromEntries(dag.nodes.map((n) => [n.name, n]));
    const show = (name) => {
      body.querySelectorAll(".dag-node").forEach((g) => g.classList.toggle("selected", g.dataset.stage === name));
      document.getElementById("dag-detail").innerHTML = stageDetail(byName[name], dag.edges);
    };
    body.querySelectorAll(".dag-node").forEach((g) => {
      g.addEventListener("click", () => show(g.dataset.stage));
      g.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          show(g.dataset.stage);
        }
      });
    });
  }

  // -------------------------------------------------------------- history
  async function loadRuns() {
    const body = document.getElementById("rh-body");
    try {
      const [{ runs: filtered }, { runs: all }] = await Promise.all([
        endpoints.pipelineRuns({ status: state.filter, limit: 50 }, { signal }),
        state.filter ? endpoints.pipelineRuns({ limit: 60 }, { signal }) : Promise.resolve({ runs: null }),
      ]);
      state.runs = all || filtered;
      renderCharts();
      if (!filtered.length) {
        ui.showEmpty(body, state.filter ? "No runs with this status." : "No runs yet. The first one starts at the next scheduled time, or press <strong>Run now</strong>.");
        return;
      }
      body.innerHTML = `<div class="table-wrap runs"><table class="run-table">
        <thead><tr><th scope="col">Started</th><th scope="col">Kind</th><th scope="col">Status</th>
          <th scope="col">Duration</th><th scope="col">Problem</th></tr></thead>
        <tbody>${filtered
          .map(
            (r) => `<tr data-run="${escapeHtml(r.run_id)}" tabindex="0" aria-selected="${r.run_id === state.selected}">
              <td>${fmtStamp(r.started_at)}</td>
              <td>${escapeHtml(KIND_NAMES[r.kind] || r.kind)}<div class="muted small">${escapeHtml(TRIGGER_NAMES[r.trigger] || r.trigger)}</div></td>
              <td>${chip(r.status)}</td>
              <td>${r.status === "running" ? "…" : fmtDuration(r.duration_s)}</td>
              <td class="problem">${r.failed_steps.length ? escapeHtml(r.failed_steps.join(", ")) : r.status === "running" ? escapeHtml(r.current_step || "") : ""}</td>
            </tr>`,
          )
          .join("")}</tbody></table></div>`;
      body.querySelectorAll("tr[data-run]").forEach((tr) => {
        tr.addEventListener("click", () => openRun(tr.dataset.run));
        tr.addEventListener("keydown", (e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            openRun(tr.dataset.run);
          }
        });
      });
      if (!state.selected) openRun(filtered[0].run_id, false);
    } catch (error) {
      if (!ui.isAbort(error)) ui.showError(body, error, "the run history", loadRuns);
    }
  }

  async function openRun(runId, scroll = true) {
    state.selected = runId;
    document.querySelectorAll("tr[data-run]").forEach((tr) => tr.setAttribute("aria-selected", String(tr.dataset.run === runId)));
    const body = document.getElementById("rd-body");
    const sub = document.getElementById("rd-sub");
    try {
      const r = await endpoints.pipelineRun(runId, { signal });
      if (state.selected !== runId) return;
      sub.textContent = `run ${r.run_id}`;
      body.innerHTML = runDetail(r);
      if (scroll && window.matchMedia("(max-width: 900px)").matches) document.getElementById("rd-card").scrollIntoView({ block: "start" });
    } catch (error) {
      if (!ui.isAbort(error)) ui.showError(body, error, `run ${runId}`, () => openRun(runId));
    }
  }

  ui.wireSegmented("run-filter", "status", (value) => {
    state.filter = value;
    loadRuns();
  });

  // --------------------------------------------------------------- charts
  ui.wireSegmented("du-kind", "kind", (kind) => {
    state.durationKind = kind;
    renderCharts();
  });

  function renderCharts() {
    const finished = state.runs.filter((r) => r.status !== "running" && r.duration_s != null).slice().reverse();
    const du = document.getElementById("du-body");
    const runs = finished.filter((r) => r.kind === state.durationKind);
    if (!runs.length) {
      ui.showEmpty(du, `No finished ${state.durationKind === "daily" ? "daily runs" : "retrains"} yet.`);
    } else {
      du.innerHTML = `<ul class="legend" aria-hidden="true">
          <li><span class="swatch st-success"></span>Succeeded</li><li><span class="swatch st-partial"></span>Partly succeeded</li>
          <li><span class="swatch st-failed"></span>Failed or interrupted</li></ul>
        <div class="chart-box"><canvas id="du-chart" role="img" aria-label="Duration of the last ${runs.length} runs"></canvas></div>
        ${ui.dataTable("Show the durations as a table", ["Started", "Status", "Duration"], runs.slice().reverse().map((r) => [fmtStamp(r.started_at), escapeHtml(STATUS_WORDS[r.status] || r.status), fmtDuration(r.duration_s)]))}`;
      durationChart(document.getElementById("du-chart"), runs, STATUS_WORDS);
    }
    const tp = document.getElementById("tp-body");
    const withEtl = finished.filter((r) => r.summary?.etl_rows_per_s != null);
    const withScore = finished.filter((r) => r.summary?.scoring_articles_per_s != null);
    if (!withEtl.length && !withScore.length) {
      ui.showEmpty(tp, "Throughput appears after the first run that processes data.");
      return;
    }
    // Two small charts, one unit each (rows/s vs articles/s never share an axis).
    tp.innerHTML = `
      <p class="caption"><strong>Spark ETL</strong>, rows per second</p>
      <div class="chart-box small"><canvas id="tp-etl" role="img" aria-label="Spark ETL rows per second per run"></canvas></div>
      <p class="caption"><strong>Sentiment scoring</strong>, new articles per second</p>
      <div class="chart-box small"><canvas id="tp-score" role="img" aria-label="Articles scored per second per run"></canvas></div>
      ${ui.dataTable(
        "Show throughput as a table",
        ["Started", "ETL rows/s", "Articles scored", "Articles/s"],
        finished
          .slice()
          .reverse()
          .map((r) => [fmtStamp(r.started_at), fmtRate(r.summary?.etl_rows_per_s), r.summary?.articles_scored ?? "–", fmtRate(r.summary?.scoring_articles_per_s)]),
      )}`;
    throughputChart(document.getElementById("tp-etl"), withEtl, "etl_rows_per_s", "rows/s", 0);
    throughputChart(document.getElementById("tp-score"), withScore, "scoring_articles_per_s", "articles/s", 1);
  }

  // -------------------------------------------------------------- polling
  // ----------------------------------------------------------- monitoring
  async function loadMonitoring() {
    const body = document.getElementById("mo-body");
    const sub = document.getElementById("mo-sub");
    try {
      const m = await endpoints.monitoring({ signal });
      const r = m.report;
      sub.textContent = r ? `drift and live error, computed ${fmtStamp(r.generated_at)}` : "";
      body.innerHTML = monitoringHtml(m);
    } catch (error) {
      if (ui.isAbort(error)) return;
      if (error.code === "no_monitoring") {
        ui.showEmpty(body, "No monitoring report yet. The scheduler computes one within two minutes of starting.");
      } else {
        ui.showError(body, error, "the monitoring report", loadMonitoring);
      }
    }
  }

  function refreshAll() {
    loadMonitoring();
    loadStats();
    loadDag();
    loadRuns();
  }

  async function tick(force = false) {
    clearTimeout(state.timer);
    if (signal.aborted) return;
    const st = await loadStatus();
    loadMonitoring();
    if (st?.busy || force) {
      loadRuns();
      if (state.selected) openRun(state.selected, false);
      loadDag();
    }
    if (!signal.aborted) state.timer = setTimeout(tick, st?.busy ? FAST_POLL_MS : SLOW_POLL_MS);
  }

  signal.addEventListener("abort", () => clearTimeout(state.timer), { once: true });
  loadStats();
  loadDag();
  loadRuns();
  tick();
}

// ------------------------------------------------------------ rendering
const SEVERITY_CHIP = { critical: "st-failed", warning: "st-partial", info: "st-running" };

function monitoringHtml(m) {
  const r = m.report || {};
  const parts = [];
  if (r.drill) {
    parts.push(`<div class="notice warn" role="status"><strong>Drift drill running</strong> until ${fmtStamp(r.drill.until)}:
      recent returns are scaled ${escapeHtml(r.drill.scale)}x inside the monitor only. No stored data was changed.</div>`);
  }
  // Alerts
  if (!m.alertmanager_reachable) {
    parts.push(`<p class="caption">Alertmanager is not reachable, so active alerts cannot be shown. ${escapeHtml(m.detail)}</p>`);
  } else if (!m.alerts.length) {
    parts.push(`<p class="run-state"><span class="chip st-success">No alerts</span> Nothing is firing.</p>`);
  } else {
    parts.push(`<ul class="alert-list">${m.alerts
      .map((a) => `<li><span class="chip ${SEVERITY_CHIP[a.severity] || "st-never"}">${escapeHtml(a.severity)}</span>
        <strong>${escapeHtml(a.name)}</strong>${a.action === "retrain" ? ' <span class="muted small">(starts a retrain)</span>' : ""}
        <div class="small">${escapeHtml(a.summary)}</div></li>`)
      .join("")}</ul>`);
  }
  // Headline numbers
  const ret = r.returns || {};
  const sen = r.sentiment || {};
  const fc = r.forecast || {};
  const tile = (label, value, note = "") =>
    `<div class="stat"><dt>${label}</dt><dd class="tabular">${value}</dd>${note ? `<div class="small muted">${note}</div>` : ""}</div>`;
  const drifted = ret.per_ticker ? Object.values(ret.per_ticker).filter((v) => v.drifted).length : null;
  const scored = ret.per_ticker ? Object.values(ret.per_ticker).filter((v) => v.psi != null).length : null;
  parts.push(`<dl class="stats six">
    ${tile("Returns drift", ret.detected ? "Yes" : ret.per_ticker ? "No" : "–", scored != null ? `${drifted} of ${scored} stocks` : "")}
    ${tile("Largest PSI", ret.max_psi ?? "–", `alert above ${ret.threshold ?? 0.25}`)}
    ${tile("Sentiment PSI", sen.psi ?? "–", sen.psi != null ? `mean ${sen.mean_reference} → ${sen.mean_current}` : "not enough news yet")}
    ${tile("LSTM live MAPE", fc.mape != null ? `${fc.mape}%` : "–", fc.mape != null ? `${fc.points} realised points` : `${fc.points ?? 0} of 20 points needed`)}
    ${tile("Naive live MAPE", fc.naive_mape != null ? `${fc.naive_mape}%` : "–", "price stays the same")}
    ${tile("Prices behind", r.freshness?.max_lag ?? "–", "trading days")}
  </dl>`);
  // Per-ticker drift table
  if (ret.per_ticker) {
    const rows = Object.entries(ret.per_ticker)
      .sort((a, b) => (b[1].psi ?? -1) - (a[1].psi ?? -1))
      .map(([t, v]) => [
        escapeHtml(t),
        v.psi ?? "–",
        v.ks ?? "–",
        v.current_volatility != null ? `${(v.current_volatility * 100).toFixed(2)}%` : "–",
        v.reference_volatility != null ? `${(v.reference_volatility * 100).toFixed(2)}%` : "–",
        v.psi == null ? "not enough data" : v.drifted ? '<span class="chip st-failed">Drifted</span>' : '<span class="chip st-success">Stable</span>',
      ]);
    const head = ["Stock", "PSI", "KS", "Daily volatility, last 60 days", "Daily volatility, year before", "Status"];
    parts.push(`<div class="table-wrap"><table class="drift-table"><caption class="visually-hidden">Returns drift per stock</caption>
      <thead><tr>${head.map((h) => `<th scope="col">${h}</th>`).join("")}</tr></thead>
      <tbody>${rows.map((r) => `<tr>${r.map((c, i) => (i === 0 ? `<th scope="row">${c}</th>` : `<td class="tabular">${c}</td>`)).join("")}</tr>`).join("")}</tbody>
    </table></div>`);
  }
  parts.push(`<p class="caption">PSI compares each stock's last 60 daily returns with the year before (above 0.25 is a significant
    shift). When 30% of stocks drift, or the live forecast error is 1.25 times the naive forecast's, an alert asks the scheduler to
    retrain; the promotion gate then decides whether the new model replaces the current one. Charts over time are in
    <a href="${escapeHtml(GRAFANA_URL)}/d/finsight-models" target="_blank" rel="noopener">Grafana</a>.</p>`);
  return parts.join("");
}

const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);

/** A timestamp in the scheduler's time zone (the run time is defined there, not in the browser's). */
function fmtInZone(iso, timeZone) {
  try {
    const text = new Intl.DateTimeFormat("en-GB", {
      weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit", timeZone,
    }).format(new Date(iso));
    return `${text} (${timeZone})`;
  } catch {
    return fmtStamp(iso); // unknown zone name: fall back to the browser's zone
  }
}

function fmtRate(v) {
  if (v == null) return "–";
  return v >= 100 ? Math.round(v).toLocaleString("en-US") : Number(v).toFixed(1);
}

function chip(status) {
  return `<span class="chip st-${escapeHtml(status)}">${escapeHtml(STATUS_WORDS[status] || status)}</span>`;
}

function runDetail(r) {
  const summary = Object.entries(r.summary || {})
    .map(([k, v]) => `<li><span>${escapeHtml(METRIC_NAMES[k] || k)}</span><span class="tabular">${escapeHtml(typeof v === "number" && !Number.isInteger(v) ? fmtRate(v) : v)}</span></li>`)
    .join("");
  const steps = r.steps
    .map((st) => {
      const log = st.log_tail?.length
        ? `<details class="log"${st.status === "failed" ? " open" : ""}><summary>Last ${st.log_tail.length} log lines</summary>
            <pre>${escapeHtml(st.log_tail.join("\n"))}</pre></details>`
        : "";
      return `<li class="step st-${escapeHtml(st.status)}">
        <div class="step-head"><span class="step-dot" aria-hidden="true"></span>
          <strong>${escapeHtml(st.label)}</strong>${st.critical ? ' <span class="muted small">(stops the run if it fails)</span>' : ""}
          <span class="step-meta">${chip(st.status)} ${st.duration_s != null && st.status !== "skipped" ? fmtDuration(st.duration_s) : ""}</span></div>
        ${st.detail ? `<div class="small">${escapeHtml(st.detail)}</div>` : ""}
        ${st.error ? `<div class="step-error">${escapeHtml(st.error)}</div>` : ""}
        ${log}</li>`;
    })
    .join("");
  const stages = r.stages.length
    ? `<details class="data-table"><summary>DVC stages this run executed (${r.stages.length})</summary>
        <div class="table-wrap"><table><thead><tr><th scope="col">Stage</th><th scope="col">Status</th><th scope="col">Duration</th></tr></thead>
        <tbody>${r.stages.map((s) => `<tr><td>${escapeHtml(s.stage)}</td><td>${chip(s.status)}</td><td>${fmtDuration(s.duration_s)}</td></tr>`).join("")}</tbody></table></div></details>`
    : `<p class="caption muted">No DVC stage ran${r.status === "running" ? " yet" : ": the inputs had not changed, so DVC reused the cached outputs"}.</p>`;
  return `
    <p class="run-head">${chip(r.status)} <strong>${escapeHtml(KIND_NAMES[r.kind] || r.kind)}</strong>
      started ${escapeHtml(TRIGGER_NAMES[r.trigger] || r.trigger)}, ${fmtStamp(r.started_at)}${r.duration_s != null ? `, took ${fmtDuration(r.duration_s)}` : ""}.</p>
    ${r.reason ? `<p class="caption">Reason: ${escapeHtml(r.reason)}</p>` : ""}
    ${r.error ? `<div class="notice error">${escapeHtml(r.error)}</div>` : ""}
    <ol class="steps">${steps}</ol>
    ${summary ? `<h3 class="h3">Results</h3><ul class="kv">${summary}</ul>` : ""}
    ${stages}
    <p class="caption muted">Code version ${escapeHtml((r.git_commit || "unknown").slice(0, 10))} · full log: <code>logs/scheduler/${escapeHtml(r.run_id)}.log</code></p>`;
}

function stageDetail(node, edges) {
  if (!node) return "";
  const ups = edges.filter(([, b]) => b === node.name).map(([a]) => a);
  const downs = edges.filter(([a]) => a === node.name).map(([, b]) => b);
  const metrics = Object.entries(node.metrics || {})
    .filter(([, v]) => typeof v === "number" || typeof v === "string")
    .slice(0, 8)
    .map(([k, v]) => `<li><span>${escapeHtml(k)}</span><span class="tabular">${escapeHtml(typeof v === "number" && !Number.isInteger(v) ? Number(v).toFixed(3) : v)}</span></li>`)
    .join("");
  return `<div class="stage-detail">
    <h3 class="h3">${escapeHtml(node.name)} ${chip(node.status)} ${node.stale ? '<span class="chip stale">inputs changed</span>' : ""}</h3>
    <p class="caption">${node.last_run_at ? `Last ran ${fmtStamp(node.last_run_at)}${node.duration_s != null ? ` for ${fmtDuration(node.duration_s)}` : ""}.` : "No recorded run yet."}
      ${ups.length ? `Needs: ${ups.map(escapeHtml).join(", ")}.` : "First stage."} ${downs.length ? `Feeds: ${downs.map(escapeHtml).join(", ")}.` : ""}</p>
    ${node.error ? `<div class="step-error">${escapeHtml(typeof node.error === "string" ? node.error : JSON.stringify(node.error))}</div>` : ""}
    ${metrics ? `<ul class="kv">${metrics}</ul>` : ""}
  </div>`;
}

// --------------------------------------------------------------- DAG SVG
const NODE_W = 280;
const NODE_H = 38;
const COL_GAP = 22;
const ROW_GAP = 46;

/** Layered top-down layout: a node's row is its longest path from a source. */
export function layoutDag(names, edges) {
  const parents = Object.fromEntries(names.map((n) => [n, []]));
  edges.forEach(([a, b]) => parents[b]?.push(a));
  const depth = {};
  const visit = (n, seen = new Set()) => {
    if (depth[n] != null) return depth[n];
    if (seen.has(n)) return 0; // defensive: DVC DAGs have no cycles
    seen.add(n);
    depth[n] = parents[n].length ? 1 + Math.max(...parents[n].filter((p) => p in parents).map((p) => visit(p, seen))) : 0;
    return depth[n];
  };
  names.forEach((n) => visit(n));
  const rows = [];
  names.forEach((n) => (rows[depth[n]] ||= []).push(n));
  const x = {};
  rows.forEach((row, r) => {
    // Order each row by the mean position of its parents to reduce crossings.
    const key = (n) => {
      const xs = parents[n].map((p) => x[p]).filter((v) => v != null);
      return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
    };
    if (r > 0) row.sort((a, b) => key(a) - key(b) || a.localeCompare(b));
    else row.sort();
    row.forEach((n, i) => (x[n] = i - (row.length - 1) / 2));
  });
  const widest = Math.max(...rows.map((r) => r.length));
  const width = widest * NODE_W + (widest - 1) * COL_GAP + 16;
  const pos = {};
  names.forEach((n) => {
    pos[n] = { x: width / 2 + x[n] * (NODE_W + COL_GAP) - NODE_W / 2, y: 8 + depth[n] * (NODE_H + ROW_GAP) };
  });
  return { pos, width, height: rows.length * (NODE_H + ROW_GAP) - ROW_GAP + 16 };
}

function dagSvg(dag) {
  const names = dag.nodes.map((n) => n.name);
  const { pos, width, height } = layoutDag(names, dag.edges);
  const edges = dag.edges
    .filter(([a, b]) => pos[a] && pos[b])
    .map(([a, b]) => {
      const x1 = pos[a].x + NODE_W / 2;
      const y1 = pos[a].y + NODE_H;
      const x2 = pos[b].x + NODE_W / 2;
      const y2 = pos[b].y;
      const mid = (y1 + y2) / 2;
      return `<path class="dag-edge" d="M${x1},${y1} C${x1},${mid} ${x2},${mid} ${x2},${y2 - 6}" marker-end="url(#arrow)"/>`;
    })
    .join("");
  const nodes = dag.nodes
    .map((n) => {
      const p = pos[n.name];
      const label = n.name.length > 34 ? `${n.name.slice(0, 33)}…` : n.name;
      const words = `${n.name}: ${STATUS_WORDS[n.status] || n.status}${n.stale ? ", inputs changed" : ""}`;
      return `<g class="dag-node st-${escapeHtml(n.status)}" data-stage="${escapeHtml(n.name)}" tabindex="0" role="button"
          aria-label="${escapeHtml(words)}" transform="translate(${p.x},${p.y})">
        <title>${escapeHtml(words)}${n.last_run_at ? ` · last ran ${fmtStamp(n.last_run_at)}` : ""}</title>
        <rect width="${NODE_W}" height="${NODE_H}" rx="8"/>
        <rect class="bar" width="6" height="${NODE_H}" rx="3"/>
        <text x="16" y="${NODE_H / 2 + 4}">${escapeHtml(label)}</text>
        ${n.stale ? `<circle class="stale" cx="${NODE_W - 12}" cy="12" r="5"/>` : ""}
      </g>`;
    })
    .join("");
  return `<div class="dag-wrap"><svg class="dag" viewBox="0 0 ${width} ${height}" width="${width}" role="group"
      aria-label="DVC pipeline with ${dag.nodes.length} stages; select a stage for details">
    <defs><marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" class="dag-arrow"/></marker></defs>
    ${edges}${nodes}</svg></div>
    <p class="caption">Select a stage to see when it last ran and its metrics. Arrows point from a stage to the stages that use its output.</p>`;
}
