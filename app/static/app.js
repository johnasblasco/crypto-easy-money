/* Dashboard: fetches /api/dashboard and renders charts, prediction and metrics. */
(function () {
  "use strict";

  const REFRESH_MS = 60_000;
  const MAX_MARKERS = 40;
  const $ = (id) => document.getElementById(id);

  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const fmtPct = (v, digits = 1) => (v == null || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(digits)}%`);
  const fmtSignedPct = (v) => (v == null ? "—" : `${v >= 0 ? "+" : ""}${(v * 100).toFixed(2)}%`);
  const priceDigits = (p) => (p >= 1000 ? 2 : p >= 1 ? 4 : 6);
  const fmtPrice = (p) =>
    p == null ? "—" : p.toLocaleString(undefined, { minimumFractionDigits: priceDigits(p), maximumFractionDigits: priceDigits(p) });
  const fmtNum = (v) => (v == null ? "—" : Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(2));
  const fmtTime = (iso) => new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });

  let charts = null;
  let lastData = null;

  function chartOptions(height, attribution) {
    return {
      height,
      layout: {
        background: { type: "solid", color: css("--surface") },
        textColor: css("--text-muted"),
        fontFamily: 'system-ui, -apple-system, "Segoe UI", sans-serif',
        fontSize: 11,
        attributionLogo: attribution,
      },
      grid: { vertLines: { visible: false }, horzLines: { color: css("--grid") } },
      // Same scale width on every pane so the time axes line up.
      rightPriceScale: { borderColor: css("--axis"), minimumWidth: 90 },
      timeScale: { borderColor: css("--axis"), timeVisible: true, secondsVisible: false },
      crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
      handleScale: { axisPressedMouseMove: { time: true, price: false } },
    };
  }

  function createCharts() {
    if (charts) Object.values(charts.instances).forEach((c) => c.remove());
    const priceEl = $("price-chart");
    const volEl = $("volume-chart");
    const rsiEl = $("rsi-chart");

    const price = LightweightCharts.createChart(priceEl, {
      ...chartOptions(priceEl.clientHeight, true),
      localization: { priceFormatter: fmtPrice },
    });
    const volume = LightweightCharts.createChart(volEl, chartOptions(volEl.clientHeight, false));
    const rsi = LightweightCharts.createChart(rsiEl, chartOptions(rsiEl.clientHeight, false));

    const up = css("--up");
    const down = css("--down");
    const candles = price.addCandlestickSeries({
      upColor: up, downColor: down, borderUpColor: up, borderDownColor: down, wickUpColor: up, wickDownColor: down,
    });
    const lineOpts = { lineWidth: 2, priceLineVisible: false, lastValueVisible: false, crosshairMarkerRadius: 4 };
    const ema12 = price.addLineSeries({ ...lineOpts, color: css("--series-1") });
    const ema26 = price.addLineSeries({ ...lineOpts, color: css("--series-2") });

    const vol = volume.addHistogramSeries({ priceFormat: { type: "volume" }, priceLineVisible: false });

    const rsiLine = rsi.addLineSeries({ ...lineOpts, color: css("--series-1"), lastValueVisible: true });
    rsi.priceScale("right").applyOptions({ scaleMargins: { top: 0.05, bottom: 0.05 } });
    rsiLine.applyOptions({ autoscaleInfoProvider: () => ({ priceRange: { minValue: 0, maxValue: 100 } }) });
    [70, 30].forEach((level) =>
      rsiLine.createPriceLine({ price: level, color: css("--axis"), lineWidth: 1, lineStyle: 2, axisLabelVisible: true })
    );

    // Keep the three panes scrolled and zoomed together.
    const all = [price, volume, rsi];
    let syncing = false;
    all.forEach((src) =>
      src.timeScale().subscribeVisibleLogicalRangeChange((range) => {
        if (syncing || !range) return;
        syncing = true;
        all.filter((c) => c !== src).forEach((c) => c.timeScale().setVisibleLogicalRange(range));
        syncing = false;
      })
    );

    price.subscribeCrosshairMove((param) => {
      if (!lastData) return;
      const bar = param.time ? lastData.candles.find((c) => c.time === param.time) : null;
      renderLegend(bar || lastData.candles[lastData.candles.length - 1]);
    });

    const ro = new ResizeObserver(() => {
      price.applyOptions({ width: priceEl.clientWidth, height: priceEl.clientHeight });
      volume.applyOptions({ width: volEl.clientWidth });
      rsi.applyOptions({ width: rsiEl.clientWidth });
    });
    ro.observe(priceEl);

    charts = { instances: { price, volume, rsi }, candles, ema12, ema26, vol, rsiLine, ro };
  }

  function renderLegend(c) {
    if (!c) return;
    const t = new Date(c.time * 1000).toISOString().replace("T", " ").slice(0, 16);
    const chg = c.close / c.open - 1;
    $("legend").innerHTML =
      `<span>${t} UTC</span>` +
      `<span>O <b>${fmtPrice(c.open)}</b></span><span>H <b>${fmtPrice(c.high)}</b></span>` +
      `<span>L <b>${fmtPrice(c.low)}</b></span><span>C <b>${fmtPrice(c.close)}</b></span>` +
      `<span class="${chg >= 0 ? "up" : "down"}">${chg >= 0 ? "▲" : "▼"} ${fmtSignedPct(chg)}</span>` +
      `<span>EMA12 <b>${fmtPrice(c.ema12)}</b></span><span>EMA26 <b>${fmtPrice(c.ema26)}</b></span>` +
      `<span>RSI <b>${fmtNum(c.rsi)}</b></span>`;
  }

  function buildMarkers(data) {
    const times = new Set(data.candles.map((c) => c.time));
    const up = css("--up");
    const down = css("--down");
    const markers = data.history
      .filter((h) => times.has(h.time))
      .slice(-MAX_MARKERS)
      .map((h) => ({
        time: h.time,
        position: h.prediction ? "belowBar" : "aboveBar",
        shape: h.prediction ? "arrowUp" : "arrowDown",
        color: h.prediction ? up : down,
        text: h.actual == null ? "" : h.actual === h.prediction ? "✓" : "✗",
      }));

    // The live call for the candle that hasn't happened yet.
    const last = data.candles[data.candles.length - 1];
    const p = data.prediction;
    const live = {
      time: last.time,
      position: p.direction === "UP" ? "belowBar" : "aboveBar",
      shape: p.direction === "UP" ? "arrowUp" : "arrowDown",
      color: p.direction === "UP" ? up : down,
      text: `next ${p.direction} ${fmtPct(p.confidence, 0)}`,
      size: 2,
    };
    return markers.filter((m) => m.time !== last.time).concat(live);
  }

  function renderCharts(data, keepRange) {
    const range = keepRange ? charts.instances.price.timeScale().getVisibleLogicalRange() : null;
    const up = css("--up");
    const down = css("--down");

    charts.candles.setData(data.candles.map(({ time, open, high, low, close }) => ({ time, open, high, low, close })));
    charts.ema12.setData(data.candles.filter((c) => c.ema12 != null).map((c) => ({ time: c.time, value: c.ema12 })));
    charts.ema26.setData(data.candles.filter((c) => c.ema26 != null).map((c) => ({ time: c.time, value: c.ema26 })));
    charts.vol.setData(
      data.candles.map((c) => ({ time: c.time, value: c.volume, color: (c.close >= c.open ? up : down) + "99" }))
    );
    charts.rsiLine.setData(data.candles.filter((c) => c.rsi != null).map((c) => ({ time: c.time, value: c.rsi })));
    charts.candles.setMarkers($("show-markers").checked ? buildMarkers(data) : []);

    if (range) {
      charts.instances.price.timeScale().setVisibleLogicalRange(range);
    } else {
      const n = data.candles.length;
      charts.instances.price.timeScale().setVisibleLogicalRange({ from: Math.max(0, n - 120), to: n + 3 });
    }
    renderLegend(data.candles[data.candles.length - 1]);
  }

  function signalGlyph(signal) {
    const bull = ["Bullish", "Increasing", "Above", "Upper half", "Above upper band", "Oversold"];
    const bear = ["Bearish", "Decreasing", "Below", "Lower half", "Below lower band", "Overbought"];
    if (bull.includes(signal)) return '<span class="glyph up">▲</span>';
    if (bear.includes(signal)) return '<span class="glyph down">▼</span>';
    return "";
  }

  function renderPrediction(data) {
    const p = data.prediction;
    const isUp = p.direction === "UP";
    $("pair").textContent = p.symbol;
    $("interval").textContent = p.interval;
    $("source").textContent = data.source;
    $("horizon").textContent = p.horizon;
    $("price").textContent = fmtPrice(p.price);
    $("change").textContent = `${p.change >= 0 ? "▲" : "▼"} ${fmtSignedPct(p.change)}`;
    $("change").className = `delta ${p.change >= 0 ? "up" : "down"}`;

    $("call").textContent = `${isUp ? "▲" : "▼"} ${p.direction}`;
    $("call").className = `call ${isUp ? "up" : "down"}`;
    $("confidence").textContent = fmtPct(p.confidence);
    const fill = $("meter-fill");
    fill.style.width = `${(p.prob_up * 100).toFixed(1)}%`;
    fill.style.background = isUp ? css("--up") : css("--down");
    $("prob-up").textContent = fmtPct(p.prob_up);
    $("model").textContent = p.model;
    $("as-of").textContent = fmtTime(p.as_of);
    $("hit-rate").textContent =
      data.history_count > 0 ? `${fmtPct(data.history_accuracy)} of ${data.history_count}` : "—";
    document.title = `${isUp ? "▲" : "▼"} ${p.symbol} ${fmtPrice(p.price)} · Crypto Direction Predictor`;

    $("indicators").innerHTML = p.indicators
      .map((ind) => {
        const value = ind.format === "percent" ? fmtSignedPct(ind.value) : fmtNum(ind.value);
        return `<tr><td>${ind.name}</td><td>${value}</td><td class="signal">${signalGlyph(ind.signal)}${ind.signal}</td></tr>`;
      })
      .join("");
  }

  function renderMetrics(m) {
    if (!m) return;
    $("test-period").textContent =
      `· held-out test ${m.test_period[0].slice(0, 10)} → ${m.test_period[1].slice(0, 10)}, ${m.rows.test} candles`;
    $("models").innerHTML = m.results
      .map((r) => {
        const cls = r.baseline ? "baseline" : r.model === m.best_model ? "best" : "";
        const name = r.model === m.best_model ? `${r.model} ★` : r.model;
        return (
          `<tr class="${cls}"><td>${name}</td><td>${fmtPct(r.cv_accuracy)}</td><td>${fmtPct(r.accuracy)}</td>` +
          `<td>${fmtPct(r.precision)}</td><td>${fmtPct(r.recall)}</td><td>${fmtPct(r.f1)}</td>` +
          `<td>${r.roc_auc == null ? "—" : r.roc_auc.toFixed(3)}</td>` +
          `<td>${fmtSignedPct(r.strategy_return)}</td><td>${fmtSignedPct(r.buy_and_hold_return)}</td></tr>`
        );
      })
      .join("");
    $("models-note").textContent =
      `★ selected by walk-forward cross-validation accuracy (not by test score). ` +
      `Test set is ${fmtPct(m.test_up_share)} UP candles, so that's what "always UP" would score. ` +
      `Strategy = long-only, hold one candle when the model says UP, ${(m.fee * 100).toFixed(2)}% fee per trade.`;
  }

  async function load(keepRange) {
    try {
      const res = await fetch("/api/dashboard");
      const body = await res.json();
      if (!res.ok) throw new Error(body.detail || res.statusText);
      lastData = body;
      $("error").hidden = true;
      renderPrediction(body);
      renderCharts(body, keepRange);
      renderMetrics(body.metrics);
    } catch (err) {
      $("error").textContent = `Could not load dashboard: ${err.message}`;
      $("error").hidden = false;
    }
  }

  $("show-markers").addEventListener("change", () => lastData && renderCharts(lastData, true));
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    createCharts();
    if (lastData) renderCharts(lastData, false);
  });

  createCharts();
  load(false);
  setInterval(() => load(true), REFRESH_MS);
})();
