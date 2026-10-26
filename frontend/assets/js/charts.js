// Chart builders (Chart.js 4, vendored in assets/vendor, loaded as window.Chart).
//
// Conventions: one y-axis per chart (no dual axes), thin 2px lines, hairline
// grids, a tooltip on hover for every chart, and colours read from the same
// CSS tokens as the page so light and dark mode always match.

import { fmtDay, fmtShortDay, money, pct, signed } from "./format.js";

const Chart = window.Chart;

function token(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function theme() {
  return {
    ink: token("--ink"),
    ink2: token("--ink-2"),
    muted: token("--muted"),
    grid: token("--grid"),
    axis: token("--axis"),
    surface: token("--surface"),
    series: [1, 2, 3, 4, 5, 6, 7].map((i) => token(`--series-${i}`)),
    other: token("--series-other"),
    upFill: token("--up-fill"),
    downFill: token("--down-fill"),
  };
}

const prefersReducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function baseOptions(t) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: prefersReducedMotion() ? false : { duration: 300 },
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: { display: false }, // legends are HTML, next to the chart
      tooltip: {
        backgroundColor: t.surface,
        titleColor: t.ink,
        bodyColor: t.ink2,
        borderColor: t.axis,
        borderWidth: 1,
        padding: 10,
        boxPadding: 4,
        usePointStyle: true,
      },
    },
    scales: {
      x: {
        grid: { display: false },
        border: { color: t.axis },
        ticks: { color: t.muted, maxRotation: 0, autoSkip: true, autoSkipPadding: 16 },
      },
      y: {
        grid: { color: t.grid },
        border: { display: false },
        ticks: { color: t.muted },
      },
    },
  };
}

function mount(canvas, config) {
  Chart.getChart(canvas)?.destroy();
  Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
  return new Chart(canvas, config);
}

/** Closing-price history for the selected period. */
export function priceChart(canvas, history) {
  const t = theme();
  const options = baseOptions(t);
  options.scales.x.ticks.callback = function (value) {
    return fmtShortDay(this.getLabelForValue(value));
  };
  options.scales.y.ticks.callback = (v) => money(v);
  options.plugins.tooltip.callbacks = {
    title: (items) => fmtDay(items[0].label),
    label: (item) => ` Close ${money(item.parsed.y)}`,
  };
  return mount(canvas, {
    type: "line",
    data: {
      labels: history.map((p) => p.date),
      datasets: [
        {
          label: "Close",
          data: history.map((p) => p.close),
          borderColor: t.series[0],
          backgroundColor: t.series[0],
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 4,
          tension: 0,
        },
      ],
    },
    options,
  });
}

/** Recent history (solid) joined to the 14-day forecast (dashed). */
export function predictionChart(canvas, history, forecast) {
  const t = theme();
  const labels = [...history.map((p) => p.date), ...forecast.map((p) => p.date)];
  const hist = [...history.map((p) => p.price), ...forecast.map(() => null)];
  // The forecast line starts at the last observed price so the two lines join.
  const fc = [
    ...history.map((_, i) => (i === history.length - 1 ? history[i].price : null)),
    ...forecast.map((p) => p.price),
  ];
  const options = baseOptions(t);
  options.scales.x.ticks.callback = function (value) {
    return fmtShortDay(this.getLabelForValue(value));
  };
  options.scales.y.ticks.callback = (v) => money(v);
  options.plugins.tooltip.filter = (item) => item.raw != null && !(item.datasetIndex === 1 && item.dataIndex === history.length - 1);
  options.plugins.tooltip.callbacks = {
    title: (items) => fmtDay(items[0].label),
    label: (item) => ` ${item.dataset.label} ${money(item.parsed.y)}`,
  };
  return mount(canvas, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Closing price",
          data: hist,
          borderColor: t.series[0],
          backgroundColor: t.series[0],
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 4,
          spanGaps: false,
        },
        {
          label: "Forecast",
          data: fc,
          borderColor: t.series[1],
          backgroundColor: t.series[1],
          borderWidth: 2,
          borderDash: [6, 4],
          pointRadius: 0,
          pointHoverRadius: 5,
          spanGaps: false,
        },
      ],
    },
    options,
  });
}

/** Weekly mean sentiment, -1..+1, bars coloured by sign. */
export function sentimentChart(canvas, weeks, onSelect) {
  const t = theme();
  const options = baseOptions(t);
  options.interaction = { mode: "nearest", axis: "x", intersect: false };
  options.scales.x.ticks.callback = function (value) {
    return fmtShortDay(this.getLabelForValue(value));
  };
  options.scales.y.min = -1;
  options.scales.y.max = 1;
  options.scales.y.ticks.stepSize = 0.5;
  options.scales.y.ticks.callback = (v) => signed(v, 1);
  options.scales.y.grid.color = (ctx) => (ctx.tick.value === 0 ? t.axis : t.grid);
  options.scales.y.grid.lineWidth = (ctx) => (ctx.tick.value === 0 ? 1.5 : 1);
  options.plugins.tooltip.callbacks = {
    title: (items) => `Week of ${fmtDay(items[0].label)}`,
    label: (item) => {
      const w = weeks[item.dataIndex];
      return [
        ` Mean score ${signed(w.mean_score, 2)}`,
        ` ${w.article_count} articles: ${w.positive} positive, ${w.negative} negative, ${w.neutral} neutral`,
      ];
    },
    footer: () => "Click the bar for its headlines",
  };
  options.onClick = (_event, elements) => {
    if (elements.length) onSelect(elements[0].index);
  };
  options.onHover = (event, elements) => {
    event.native.target.style.cursor = elements.length ? "pointer" : "default";
  };
  return mount(canvas, {
    type: "bar",
    data: {
      labels: weeks.map((w) => w.week_start),
      datasets: [
        {
          label: "Mean sentiment",
          data: weeks.map((w) => w.mean_score),
          backgroundColor: weeks.map((w) => (w.mean_score >= 0 ? t.upFill : t.downFill)),
          borderRadius: 4,
          borderSkipped: false,
          maxBarThickness: 36,
        },
      ],
    },
    options,
  });
}

/** Articles per week (its own small chart: counts never share an axis with scores). */
export function countsChart(canvas, weeks) {
  const t = theme();
  const options = baseOptions(t);
  options.interaction = { mode: "nearest", axis: "x", intersect: false };
  options.scales.x.ticks.callback = function (value) {
    return fmtShortDay(this.getLabelForValue(value));
  };
  options.scales.y.beginAtZero = true;
  options.scales.y.ticks.precision = 0;
  options.scales.y.ticks.maxTicksLimit = 4;
  options.plugins.tooltip.callbacks = {
    title: (items) => `Week of ${fmtDay(items[0].label)}`,
    label: (item) => ` ${item.parsed.y} articles`,
  };
  return mount(canvas, {
    type: "bar",
    data: {
      labels: weeks.map((w) => w.week_start),
      datasets: [
        {
          label: "Articles",
          data: weeks.map((w) => w.article_count),
          backgroundColor: t.series[0],
          borderRadius: { topLeft: 4, topRight: 4 },
          borderSkipped: "bottom",
          maxBarThickness: 36,
        },
      ],
    },
    options,
  });
}

/**
 * Portfolio weights as a pie (doughnut). Slices come pre-sorted with the
 * largest first; at most seven named slices, the rest folded into "Other".
 * Returns the colour of each slice so the HTML legend can match it.
 */
export function allocationPie(canvas, slices) {
  const t = theme();
  const colours = slices.map((s, i) => (s.other ? t.other : t.series[i]));
  const options = baseOptions(t);
  delete options.scales;
  options.interaction = { mode: "nearest", intersect: true };
  options.cutout = "58%";
  options.plugins.tooltip.callbacks = {
    label: (item) => ` ${slices[item.dataIndex].label}: ${pct(item.parsed, 0)}`,
  };
  mount(canvas, {
    type: "doughnut",
    data: {
      labels: slices.map((s) => s.label),
      datasets: [
        {
          data: slices.map((s) => s.weight),
          backgroundColor: colours,
          borderColor: t.surface, // 2px surface gap between slices
          borderWidth: 2,
          hoverOffset: 6,
        },
      ],
    },
    options,
  });
  return colours;
}

/** Fold weights into at most `max` slices (largest first, zero weights dropped). */
export function toSlices(weights, max = 7) {
  const sorted = Object.entries(weights)
    .filter(([, w]) => w > 1e-9)
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  if (sorted.length <= max) return sorted.map(([label, weight]) => ({ label, weight }));
  const head = sorted.slice(0, max - 1).map(([label, weight]) => ({ label, weight }));
  const rest = sorted.slice(max - 1);
  head.push({ label: `Other (${rest.map(([s]) => s).join(", ")})`, weight: rest.reduce((a, [, w]) => a + w, 0), other: true });
  return head;
}
