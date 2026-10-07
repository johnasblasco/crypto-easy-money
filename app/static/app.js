/* Dashboard: fetches /api/dashboard and renders charts, prediction and metrics. */
(function () {
  "use strict";

  const REFRESH_MS = 60_000;
  const MAX_MARKERS = 40;
  const $ = (id) => document.getElementById(id);

  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const fmtPct = (v, digits = 1) => (v == null || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(digits)}%`);
  const fmtSignedPct = (v) => (v == null ? "—" : `${v >= 0 ? "+" : ""}${(v * 100).toFixed(2)}%`);
  // About 5 significant digits, so sub-cent meme coins (PEPE ~ $0.000004) stay readable.
  const priceDigits = (p) => {
    if (!Number.isFinite(p) || p <= 0) return 2;
    if (p >= 1000) return 2;
    if (p >= 1) return 4;
    return Math.min(12, Math.max(4, Math.ceil(-Math.log10(p)) + 4));
  };
  const fmtPrice = (p) =>
    p == null ? "—" : p.toLocaleString(undefined, { minimumFractionDigits: priceDigits(p), maximumFractionDigits: priceDigits(p) });
  const fmtNum = (v) => (v == null ? "—" : Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(2));
  const fmtTime = (iso) => new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });

  let charts = null;
  let lastData = null;
  // Which model the chart shows: null = main model, else an experiments config like "ETHUSDT_4h_h1".
  let currentConfig = decodeURIComponent(location.hash.slice(1)) || null;
  let lastScan = null;

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

  // A guess only counts as a call when the model is at least as confident as the threshold it chose on
  // training data. Weaker guesses (e.g. 51% UP) are shown as gray dots: they are coin flips, not calls.
  const isCall = (h, threshold) => Math.max(h.prob_up, 1 - h.prob_up) >= threshold;

  function buildMarkers(data) {
    const times = new Set(data.candles.map((c) => c.time));
    const up = css("--up");
    const down = css("--down");
    const gray = css("--text-muted");
    const p = data.prediction;
    const threshold = p.threshold ?? 0.5;
    const markers = data.history
      .filter((h) => times.has(h.time))
      .slice(-MAX_MARKERS)
      .map((h) =>
        isCall(h, threshold)
          ? {
              time: h.time,
              position: h.prediction ? "belowBar" : "aboveBar",
              shape: h.prediction ? "arrowUp" : "arrowDown",
              color: h.prediction ? up : down,
              text: h.actual == null ? "" : h.actual === h.prediction ? "✓" : "✗",
            }
          : { time: h.time, position: h.prediction ? "belowBar" : "aboveBar", shape: "circle", color: gray, size: 0.5 }
      );

    // The candle that hasn't happened yet: an arrow only for a validated, confident call.
    const last = data.candles[data.candles.length - 1];
    const call = p.edge_verdict === "POSSIBLE EDGE" && p.actionable;
    const live = call
      ? {
          time: last.time,
          position: p.direction === "UP" ? "belowBar" : "aboveBar",
          shape: p.direction === "UP" ? "arrowUp" : "arrowDown",
          color: p.direction === "UP" ? up : down,
          text: `next ${p.direction} ${fmtPct(p.confidence, 0)}`,
          size: 2,
        }
      : {
          time: last.time,
          position: p.direction === "UP" ? "belowBar" : "aboveBar",
          shape: "circle",
          color: gray,
          text: "no call",
          size: 1,
        };
    return markers.filter((m) => m.time !== last.time).concat(live);
  }

  function markerScore(data) {
    const threshold = data.prediction.threshold ?? 0.5;
    const scored = data.history.filter((h) => h.actual != null);
    if (!scored.length) return "";
    const calls = scored.filter((h) => isCall(h, threshold));
    const right = calls.filter((h) => h.prediction === h.actual).length;
    // Compare with the better constant guess: if 58% of candles closed down, "always DOWN" is right 58% of the time.
    const upShare = scored.filter((h) => h.actual === 1).length / scored.length;
    const constant = upShare >= 0.5 ? `"always UP" ${fmtPct(upShare, 0)}` : `"always DOWN" ${fmtPct(1 - upShare, 0)}`;
    const callsText = calls.length
      ? `${right}/${calls.length} confident calls right (${fmtPct(right / calls.length, 0)})`
      : "no confident calls";
    return `Unseen candles: ${callsText} · coin flip 50% · ${constant}. A model is only useful if it beats the constant guess.`;
  }

  function renderCharts(data, keepRange) {
    const range = keepRange ? charts.instances.price.timeScale().getVisibleLogicalRange() : null;
    const up = css("--up");
    const down = css("--down");

    // The chart's default price step is 0.01, which flattens sub-cent coins; match the coin's magnitude.
    const last = data.candles.length ? data.candles[data.candles.length - 1].close : 1;
    const digits = priceDigits(last);
    const priceFormat = { type: "price", precision: digits, minMove: Math.pow(10, -digits) };
    [charts.candles, charts.ema12, charts.ema26].forEach((s) => s.applyOptions({ priceFormat }));
    charts.candles.setData(data.candles.map(({ time, open, high, low, close }) => ({ time, open, high, low, close })));
    charts.ema12.setData(data.candles.filter((c) => c.ema12 != null).map((c) => ({ time: c.time, value: c.ema12 })));
    charts.ema26.setData(data.candles.filter((c) => c.ema26 != null).map((c) => ({ time: c.time, value: c.ema26 })));
    charts.vol.setData(
      data.candles.map((c) => ({ time: c.time, value: c.volume, color: (c.close >= c.open ? up : down) + "99" }))
    );
    charts.rsiLine.setData(data.candles.filter((c) => c.rsi != null).map((c) => ({ time: c.time, value: c.rsi })));
    charts.candles.setMarkers($("show-markers").checked ? buildMarkers(data) : []);
    $("marker-score").textContent = markerScore(data);

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
    const ts = $("trade-status");
    const validated = p.edge_verdict === "POSSIBLE EDGE";
    if (!validated) {
      ts.innerHTML = `This model failed the edge check (<b>${p.edge_verdict || "unknown"}</b>), so this call is <b>not a trading signal</b>.`;
    } else if (p.actionable) {
      ts.innerHTML = `Clears the <b>${fmtPct(p.threshold, 0)}</b> confidence threshold — a validated signal.`;
    } else {
      ts.innerHTML = `Below the <b>${fmtPct(p.threshold, 0)}</b> confidence threshold — <b>no trade</b>.`;
    }
    $("prob-up").textContent = fmtPct(p.prob_up);
    $("model").textContent = p.model;
    $("as-of").textContent = fmtTime(p.as_of);
    const scored = data.history.filter((h) => h.actual != null);
    const upShare = scored.length ? scored.filter((h) => h.actual === 1).length / scored.length : null;
    $("hit-rate").textContent =
      data.history_count > 0
        ? `${fmtPct(data.history_accuracy)} of ${data.history_count} (constant guess ${fmtPct(Math.max(upShare, 1 - upShare))})`
        : "—";
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
          `<td>${r.p_vs_baseline == null ? "—" : r.p_vs_baseline.toFixed(3)}</td>` +
          `<td>${fmtPct(r.precision)}</td><td>${fmtPct(r.recall)}</td><td>${fmtPct(r.f1)}</td>` +
          `<td>${r.roc_auc == null ? "—" : r.roc_auc.toFixed(3)}</td>` +
          `<td>${fmtSignedPct(r.strategy_return)}</td><td>${fmtSignedPct(r.buy_and_hold_return)}</td></tr>`
        );
      })
      .join("");
    $("models-note").textContent =
      `★ selected by walk-forward cross-validation accuracy (not by test score). ` +
      `Test set is ${fmtPct(m.test_up_share)} UP candles, so that's what "always UP" would score. ` +
      `p vs baseline below 0.05 means the model beat the best baseline or constant guess (${fmtPct(m.best_baseline_accuracy)}) ` +
      `by more than luck would explain. ` +
      `Strategy = long-only, hold ${m.horizon} candle(s) when the model says UP, ` +
      `${((m.cost_per_trade ?? m.fee) * 100).toFixed(2)}% cost per trade` +
      (m.spread ? ` (${(m.fee * 100).toFixed(2)}% fee + ${(m.spread * 100).toFixed(3)}% half-tick spread).` : ".");
    renderEdge(m.edge, m);
  }

  function renderEdge(edge, m) {
    if (!edge) return;
    const tone = { "POSSIBLE EDGE": "good", "NO EDGE": "bad" }[edge.verdict] || "warn";
    const icon = { good: "✓", bad: "⚠", warn: "!" }[tone];
    $("edge").className = `card wide edge ${tone}`;
    $("edge").hidden = false;
    $("edge-icon").textContent = icon;
    $("edge-verdict").textContent = edge.verdict;
    $("edge-summary").textContent =
      `${edge.summary} On unseen data at the ${fmtPct(edge.threshold, 0)} threshold: strategy ` +
      `${fmtSignedPct(edge.strategy_return)} vs buy & hold ${fmtSignedPct(edge.buy_and_hold_return)}.`;

    $("threshold-model").textContent = `· ${m.best_model}, held-out test period`;
    $("thresholds").innerHTML = edge.test_thresholds
      .map((r) => {
        const chosen = r.threshold === edge.threshold;
        return (
          `<tr class="${chosen ? "best" : ""}"><td>${fmtPct(r.threshold, 0)}${chosen ? " ★" : ""}</td>` +
          `<td>${r.calls}</td><td>${fmtPct(r.coverage)}</td><td>${fmtPct(r.accuracy)}</td>` +
          `<td>${fmtSignedPct(r.strategy_return)}</td><td>${fmtSignedPct(r.buy_and_hold_return)}</td></tr>`
        );
      })
      .join("");
    $("thresholds-note").textContent =
      `The model only trades when it is at least this confident. ★ was chosen on the training period, ` +
      `before the test data was seen; picking the best row after the fact would be cheating.`;
  }

  async function load(keepRange) {
    try {
      const url = currentConfig ? `/api/dashboard?config=${encodeURIComponent(currentConfig)}` : "/api/dashboard";
      const res = await fetch(url);
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

  const STATUS = {
    BUY: { cls: "buy", label: "▲ BUY" },
    AVOID: { cls: "avoid", label: "▼ AVOID" },
    WAIT: { cls: "wait", label: "• WAIT" },
    "NOT VALIDATED": { cls: "none", label: "not validated" },
    ERROR: { cls: "none", label: "error" },
  };

  function renderScanner(data) {
    const rows = data.rows || [];
    $("scan-time").textContent = rows.length ? `· updated ${fmtTime(data.scanned_at)}` : "";
    if (!rows.length) {
      $("scanner").innerHTML =
        '<tr><td colspan="8">No token models yet. Run <code>python -m cryptopredict.experiments</code> to train and validate them.</td></tr>';
      $("scanner-note").textContent = "";
      return;
    }
    $("scanner").innerHTML = rows
      .map((r) => {
        const st = STATUS[r.status] || STATUS.ERROR;
        const cls = [r.status === "NOT VALIDATED" ? "unvalidated" : "", r.config === currentConfig ? "active" : ""].join(" ");
        if (r.status === "ERROR") {
          return `<tr class="${cls}"><td><span class="pill none">error</span></td><td>${r.symbol}</td><td>${r.interval}</td><td colspan="5">${r.error}</td></tr>`;
        }
        const call = r.direction === "UP" ? '<span class="up">▲</span> UP' : '<span class="down">▼</span> DOWN';
        return (
          `<tr class="${cls}" data-config="${r.config}" tabindex="0">` +
          `<td><span class="pill ${st.cls}">${st.label}</span></td><td>${r.symbol}</td><td>${r.interval}</td>` +
          `<td>${r.hold}</td><td>${call}</td><td>${fmtPct(r.confidence)}</td><td>${fmtPrice(r.price)}</td>` +
          `<td>${r.verdict}</td></tr>`
        );
      })
      .join("");
    const validated = rows.filter((r) => r.verdict === "POSSIBLE EDGE").length;
    const buys = rows.filter((r) => r.status === "BUY").length;
    $("scanner-note").textContent =
      validated === 0
        ? "No token/timeframe passed the edge check on unseen data, so nothing here is a signal to act on. " +
          "Predictions are shown for information only. Click a row to see its chart."
        : `${validated} validated model(s), ${buys} BUY signal(s) right now. Only validated models can signal; ` +
          "hold for the time shown, then exit. Click a row to see its chart. Not financial advice.";
  }

  function notifyNewBuys(data) {
    if (!("Notification" in window) || Notification.permission !== "granted") return;
    let seen = [];
    try { seen = JSON.parse(localStorage.getItem("seenSignals") || "[]"); } catch (e) { seen = []; }
    const fresh = (data.rows || []).filter((r) => r.status === "BUY" && !seen.includes(`${r.config}|${r.signal_time}`));
    fresh.forEach((r) => {
      new Notification(`BUY ${r.symbol} — hold ${r.hold}`, {
        body: `${fmtPct(r.confidence)} confidence on ${r.interval} candles @ ${fmtPrice(r.price)}`,
        tag: r.config,
      });
      seen.push(`${r.config}|${r.signal_time}`);
    });
    try { localStorage.setItem("seenSignals", JSON.stringify(seen.slice(-500))); } catch (e) { /* storage unavailable */ }
  }

  async function loadScanner() {
    try {
      const res = await fetch("/api/scanner");
      if (!res.ok) throw new Error(res.statusText);
      lastScan = await res.json();
      renderScanner(lastScan);
      notifyNewBuys(lastScan);
    } catch (err) {
      $("scanner-note").textContent = `Could not run the scanner: ${err.message}`;
    }
  }

  function selectConfig(config) {
    currentConfig = config;
    history.replaceState(null, "", config ? `#${config}` : location.pathname);
    if (lastScan) renderScanner(lastScan);
    load(false);
    window.scrollTo({ top: $("price-chart").getBoundingClientRect().top + window.scrollY - 140, behavior: "smooth" });
  }

  $("scanner").addEventListener("click", (e) => {
    const row = e.target.closest("tr[data-config]");
    if (row) selectConfig(row.dataset.config);
  });
  $("scanner").addEventListener("keydown", (e) => {
    const row = e.target.closest("tr[data-config]");
    if (row && (e.key === "Enter" || e.key === " ")) {
      e.preventDefault();
      selectConfig(row.dataset.config);
    }
  });

  function updateAlertsButton() {
    const btn = $("alerts-btn");
    if (!("Notification" in window)) {
      btn.textContent = "Alerts not supported";
      btn.disabled = true;
    } else if (Notification.permission === "granted") {
      btn.textContent = "Alerts on";
      btn.disabled = true;
    } else if (Notification.permission === "denied") {
      btn.textContent = "Alerts blocked";
      btn.disabled = true;
    }
  }
  $("alerts-btn").addEventListener("click", async () => {
    await Notification.requestPermission();
    updateAlertsButton();
    if (lastScan) notifyNewBuys(lastScan);
  });
  updateAlertsButton();

  $("show-markers").addEventListener("change", () => lastData && renderCharts(lastData, true));
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    createCharts();
    if (lastData) renderCharts(lastData, false);
  });

  // ------------------------------------------------------------ signal engine
  const ACTION = {
    LONG: { cls: "buy", label: "▲ LONG" },
    SHORT: { cls: "avoid", label: "▼ SHORT" },
    FLAT: { cls: "flat", label: "— FLAT" },
    "NO TRADE": { cls: "notrade", label: "NO TRADE" },
  };
  const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  function renderEngine(data) {
    $("engine-time").textContent = `· ${fmtTime(data.generated_at)}`;
    $("specialists").innerHTML = data.specialists
      .map((sp) => {
        const st = sp.evidence.status;
        const cls = st === "VALIDATED" ? "validated" : st === "RISK_OVERLAY" ? "overlay" : "not";
        const label = { VALIDATED: "✓ validated", RISK_OVERLAY: "! risk overlay (not alpha)", NOT_VALIDATED: "✗ not validated" }[st] || st;
        const health = sp.health.state === "ACTIVE" ? "health: active" : `health: ${sp.health.state} — ${sp.health.reason}`;
        return `<div class="spec"><h3>${esc(sp.name)} <span class="muted">· ${esc(sp.horizon)}</span></h3>` +
          `<div class="status ${cls}">${label}</div><p>${esc(sp.evidence.summary)}</p><p class="muted">${esc(health)}</p></div>`;
      })
      .join("");
    const specs = data.specialists.map((s) => s.name);
    $("engine-head").innerHTML = `<tr><th>Token</th><th>Verdict</th><th>Data</th><th>Regime</th>${specs.map((n) => `<th>${esc(n)}</th>`).join("")}</tr>`;
    const verdicts = {};
    (data.consensus || []).forEach((c) => { verdicts[c.symbol] = c; });
    const bySym = {};
    data.signals.forEach((s) => { (bySym[s.symbol] = bySym[s.symbol] || {})[s.specialist] = s; });
    $("engine-body").innerHTML = Object.entries(bySym)
      .map(([sym, sigs]) => {
        const any = Object.values(sigs)[0];
        const dh = any.data_health || {};
        const data = dh.ok ? `ok · ${Math.round(dh.staleness_s || 0)}s` : `<span class="down">${esc((dh.problems || ["?"])[0])}</span>`;
        const rg = any.regime || {};
        const regime = rg.vol_regime ? `${rg.vol_regime} vol · 7d ${rg.trend_7d}` : "—";
        const cells = specs.map((n) => {
          const s = sigs[n];
          if (!s) return "<td class='cell'>—</td>";
          let a = ACTION[s.action] || ACTION["NO TRADE"];
          const overlay = (s.evidence || {}).status === "RISK_OVERLAY";
          // An overlay's LONG is position-size guidance, not a trade signal.
          if (overlay && s.action === "LONG") a = { cls: "flat", label: "SIZE" };
          let detail = "";
          if (s.exposure != null) detail = `exposure ${(s.exposure * 100).toFixed(0)}%`;
          if (s.expected_edge_bps != null) {
            detail = `edge ${s.expected_edge_bps.toFixed(1)}bp vs cost ${s.cost_bps.toFixed(1)}bp`;
            if (s.confidence != null && s.view) detail += ` · P(${s.view === "UP" ? "up" : "down"}) ${(s.confidence * 100).toFixed(1)}%`;
            if (s.confidence == null) detail = `holdout avg ${s.expected_edge_bps.toFixed(1)}bp/event vs cost ${s.cost_bps.toFixed(1)}bp`;
          }
          const why = s.veto || (s.reasons || []).slice(-1)[0] || "";
          return `<td class="cell"><span class="pill ${a.cls}">${a.label}</span> <span class="muted">${esc(detail)}</span>` +
            `<span class="why" title="${esc((s.reasons || []).join(" · "))}">${esc(why)}</span></td>`;
        });
        const c = verdicts[sym];
        let verdict = "<td class='cell'>—</td>";
        if (c) {
          const va = ACTION[c.verdict] || ACTION["NO TRADE"];
          const agree = c.agreement === "conflict" ? "<span class='down'>views conflict</span>"
            : c.agreement === "agree" ? "views agree" : esc(c.agreement);
          verdict = `<td class="cell"><span class="pill ${va.cls}">${va.label}</span> <span class="muted">${agree}</span>` +
            `<span class="why" title="${esc(c.why)}">${esc(c.why)}</span></td>`;
        }
        return `<tr><td>${esc(sym)}</td>${verdict}<td>${data}</td><td>${esc(regime)}</td>${cells.join("")}</tr>`;
      })
      .join("");
    const actionable = (data.consensus || []).filter((c) => c.verdict === "LONG" || c.verdict === "SHORT").length;
    $("engine-note").textContent =
      `${actionable} actionable verdict(s). A verdict needs a validated, healthy specialist whose edge clears costs; ` +
      "otherwise it is NO TRADE. Hover a reason for the full explanation. " +
      "Exposure = suggested fraction of capital for that coin from the daily trend overlay. Not financial advice.";
  }

  async function loadEngine() {
    try {
      const res = await fetch("/api/engine");
      const body = await res.json();
      if (!res.ok) throw new Error(body.detail || res.statusText);
      renderEngine(body);
    } catch (err) {
      $("engine-note").textContent = `Signal engine unavailable: ${err.message}`;
    }
  }

  createCharts();
  load(false);
  loadEngine();
  setInterval(loadEngine, 5 * REFRESH_MS);
  loadScanner();
  setInterval(() => load(true), REFRESH_MS);
  setInterval(loadScanner, 2 * REFRESH_MS);
})();
