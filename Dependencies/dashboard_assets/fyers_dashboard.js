"use strict";

(() => {
  const byId = (id) => document.getElementById(id);
  const view = {
    mode: byId("mode"),
    service: byId("service-status"),
    clock: byId("clock"),
    sessionDate: byId("session-date"),
    totalPnl: byId("total-pnl"),
    realizedPnl: byId("realized-pnl"),
    openPnl: byId("open-pnl"),
    tradeCount: byId("trade-count"),
    positionState: byId("position-state"),
    lossLimit: byId("loss-limit"),
    riskFill: byId("risk-fill"),
    spot: byId("spot-price"),
    barTime: byId("bar-time"),
    feedState: byId("feed-state"),
    chart: byId("price-chart"),
    chartEmpty: byId("chart-empty"),
    positionBadge: byId("position-badge"),
    positionEmpty: byId("position-empty"),
    positionDetail: byId("position-detail"),
    positionDirection: byId("position-direction"),
    positionQty: byId("position-qty"),
    positionSymbol: byId("position-symbol"),
    positionPnl: byId("position-pnl"),
    positionEntry: byId("position-entry"),
    positionMark: byId("position-mark"),
    positionSpot: byId("position-spot"),
    positionStop: byId("position-stop"),
    positionOpened: byId("position-opened"),
    signalStatus: byId("signal-status"),
    signalAction: byId("signal-action"),
    emaFast: byId("ema-fast"),
    emaMid: byId("ema-mid"),
    emaSlow: byId("ema-slow"),
    adx: byId("adx"),
    atr: byId("atr"),
    dataAge: byId("data-age"),
    closedCount: byId("closed-count"),
    tradeRows: byId("trade-rows"),
    tradesEmpty: byId("trades-empty"),
    updatedAt: byId("updated-at"),
    eventList: byId("event-list"),
    eventsEmpty: byId("events-empty"),
  };

  const money = (value) => {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return "--";
    return `${Number(value) > 0 ? "+" : Number(value) < 0 ? "−" : ""}₹${Math.abs(Number(value)).toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  };
  const price = (value) => value === null || value === undefined ? "--" : Number(value).toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const time = (value) => {
    if (!value) return "--";
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? String(value) : parsed.toLocaleTimeString("en-IN", { timeZone: "Asia/Kolkata", hour12: false });
  };
  const dateTime = (value) => {
    if (!value) return "--";
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? String(value) : parsed.toLocaleString("en-IN", { timeZone: "Asia/Kolkata", hour12: false });
  };
  const setPnl = (node, value) => {
    node.textContent = money(value);
    node.classList.toggle("pos", Number(value) > 0);
    node.classList.toggle("neg", Number(value) < 0);
  };
  const cell = (row, value, className = "") => {
    const node = document.createElement("td");
    node.textContent = value === null || value === undefined ? "--" : String(value);
    if (className) node.className = className;
    row.appendChild(node);
    return node;
  };

  let chart = null;
  let candleSeries = null;
  let fastSeries = null;
  let midSeries = null;
  let slowSeries = null;
  let hasFitted = false;

  function setupChart() {
    if (!window.LightweightCharts || !view.chart) return;
    chart = LightweightCharts.createChart(view.chart, {
      autoSize: true,
      layout: { background: { type: "solid", color: "#0c131c" }, textColor: "#8294a3", fontFamily: "IBM Plex Mono, Consolas, monospace" },
      grid: { vertLines: { color: "rgba(138,177,202,0.055)" }, horzLines: { color: "rgba(138,177,202,0.07)" } },
      rightPriceScale: { borderColor: "rgba(138,177,202,0.16)" },
      timeScale: { borderColor: "rgba(138,177,202,0.16)", timeVisible: true, secondsVisible: false },
      crosshair: { vertLine: { color: "rgba(112,220,255,0.3)" }, horzLine: { color: "rgba(112,220,255,0.3)" } },
      localization: { locale: "en-IN" },
    });
    candleSeries = chart.addSeries(LightweightCharts.CandlestickSeries, {
      upColor: "#51d4a0", downColor: "#ff667d", borderVisible: false,
      wickUpColor: "#51d4a0", wickDownColor: "#ff667d",
    });
    fastSeries = chart.addSeries(LightweightCharts.LineSeries, { color: "#70dcff", lineWidth: 1, priceLineVisible: false, lastValueVisible: false });
    midSeries = chart.addSeries(LightweightCharts.LineSeries, { color: "#c9a965", lineWidth: 1, priceLineVisible: false, lastValueVisible: false });
    slowSeries = chart.addSeries(LightweightCharts.LineSeries, { color: "#93a5b5", lineWidth: 1, priceLineVisible: false, lastValueVisible: false });
  }

  function renderChart(market) {
    const candles = Array.isArray(market.candles) ? market.candles : [];
    view.chartEmpty.hidden = candles.length > 0;
    if (!chart || candles.length === 0) return;
    candleSeries.setData(candles.map((bar) => ({ time: bar.time, open: bar.open, high: bar.high, low: bar.low, close: bar.close })));
    fastSeries.setData(candles.filter((bar) => bar.ema_fast !== null).map((bar) => ({ time: bar.time, value: bar.ema_fast })));
    midSeries.setData(candles.filter((bar) => bar.ema_mid !== null).map((bar) => ({ time: bar.time, value: bar.ema_mid })));
    slowSeries.setData(candles.filter((bar) => bar.ema_slow !== null).map((bar) => ({ time: bar.time, value: bar.ema_slow })));
    if (!hasFitted) {
      chart.timeScale().fitContent();
      hasFitted = true;
    }
  }

  function renderPosition(position) {
    const open = Boolean(position);
    view.positionEmpty.hidden = open;
    view.positionDetail.hidden = !open;
    view.positionBadge.textContent = open ? "OPEN" : "FLAT";
    view.positionBadge.classList.toggle("position-open", open);
    view.positionState.textContent = open ? `${position.direction} · ${position.quantity} QTY` : "NO OPEN POSITION";
    if (!open) {
      setPnl(view.openPnl, null);
      return;
    }
    view.positionDirection.textContent = position.direction || "--";
    view.positionQty.textContent = `${position.quantity || 0} QTY`;
    view.positionSymbol.textContent = position.display_symbol || position.symbol || "--";
    setPnl(view.openPnl, position.open_pnl);
    setPnl(view.positionPnl, position.open_pnl);
    view.positionEntry.textContent = `₹${price(position.entry_price)}`;
    view.positionMark.textContent = `₹${price(position.mark_price)}`;
    view.positionSpot.textContent = price(position.entry_spot);
    view.positionStop.textContent = price(position.stop_spot);
    view.positionOpened.textContent = time(position.opened_at);
  }

  function renderTrades(trades) {
    const rows = Array.isArray(trades) ? trades.slice().reverse() : [];
    view.closedCount.textContent = `${rows.length} FILLS`;
    view.tradesEmpty.hidden = rows.length > 0;
    view.tradeRows.replaceChildren();
    for (const trade of rows) {
      const row = document.createElement("tr");
      cell(row, time(trade.exit_time || trade.time));
      cell(row, trade.direction);
      cell(row, trade.display_symbol || trade.symbol);
      cell(row, trade.quantity, "num");
      cell(row, `₹${price(trade.entry_price)}`, "num");
      cell(row, `₹${price(trade.exit_price)}`, "num");
      const pnl = cell(row, money(trade.pnl), `num ${Number(trade.pnl) >= 0 ? "pnl-pos" : "pnl-neg"}`);
      if (trade.pnl === null || trade.pnl === undefined) pnl.classList.remove("pnl-pos", "pnl-neg");
      cell(row, trade.reason);
      view.tradeRows.appendChild(row);
    }
  }

  function renderEvents(events) {
    const rows = Array.isArray(events) ? events.slice().reverse().slice(0, 8) : [];
    view.eventsEmpty.hidden = rows.length > 0;
    view.eventList.replaceChildren();
    for (const event of rows) {
      const item = document.createElement("li");
      const stamp = document.createElement("span");
      stamp.className = "event-time";
      stamp.textContent = time(event.at);
      const level = document.createElement("span");
      level.className = `event-level ${(event.level || "INFO").toLowerCase()}`;
      level.textContent = event.level || "INFO";
      const message = document.createElement("span");
      message.className = "event-message";
      message.textContent = event.message || "";
      item.append(stamp, level, message);
      view.eventList.appendChild(item);
    }
  }

  function render(snapshot) {
    const service = snapshot.service || {};
    const market = snapshot.market || {};
    const account = snapshot.account || {};
    const position = snapshot.position || null;
    const now = new Date();
    view.clock.textContent = `${now.toLocaleTimeString("en-IN", { timeZone: "Asia/Kolkata", hour12: false })} IST`;
    view.mode.textContent = service.mode || "PAPER";
    view.mode.classList.toggle("live", service.mode === "LIVE");
    view.sessionDate.textContent = service.trading_day || "SESSION / --";
    view.service.className = "service-state";
    const serviceDot = document.createElement("i");
    view.service.replaceChildren(serviceDot, document.createTextNode(` ${service.status || "RUNNING"}`));
    const lastUpdate = service.updated_at ? new Date(service.updated_at) : null;
    if (lastUpdate && now - lastUpdate > 90000) view.service.classList.add("stale");
    if (service.status === "ERROR" || market.status === "ERROR") view.service.classList.add("dead");

    setPnl(view.totalPnl, account.total_pnl);
    setPnl(view.realizedPnl, account.realized_pnl);
    view.tradeCount.textContent = `${account.trade_count || 0} trades today`;
    view.lossLimit.textContent = account.daily_loss_limit ? `₹${price(account.daily_loss_limit)}` : "--";
    const loss = Math.max(0, -(Number(account.total_pnl) || 0));
    const limit = Number(account.daily_loss_limit) || 0;
    view.riskFill.style.width = limit > 0 ? `${Math.min(100, (loss / limit) * 100)}%` : "0%";
    view.riskFill.classList.toggle("danger", limit > 0 && loss / limit >= 0.75);

    view.spot.textContent = market.spot === null || market.spot === undefined ? "--" : price(market.spot);
    view.barTime.textContent = market.bar_at ? `BAR ${time(market.bar_at)}` : "WAITING FOR BARS";
    view.feedState.textContent = market.status || "IDLE";
    view.feedState.className = "feed-state";
    if (market.status === "RETRYING" || market.status === "STALE") view.feedState.classList.add("stale");
    if (market.status === "ERROR") view.feedState.classList.add("error");
    view.emaFast.textContent = price(market.ema_fast);
    view.emaMid.textContent = price(market.ema_mid);
    view.emaSlow.textContent = price(market.ema_slow);
    view.adx.textContent = price(market.adx);
    view.atr.textContent = price(market.atr);
    if (market.last_success_at) {
      const age = Math.max(0, Math.floor((now - new Date(market.last_success_at)) / 1000));
      view.dataAge.textContent = age < 60 ? `${age}s` : `${Math.floor(age / 60)}m`;
    } else {
      view.dataAge.textContent = "--";
    }
    view.signalAction.textContent = (snapshot.signal || {}).action || "SCANNING";
    view.signalStatus.textContent = (snapshot.signal || {}).status || "SCANNING";
    view.updatedAt.textContent = `SYNC ${time(service.updated_at)}`;

    renderPosition(position);
    renderTrades(snapshot.trades);
    renderEvents(snapshot.events);
    renderChart(market);
  }

  async function poll() {
    try {
      const response = await fetch("/api/dashboard", { cache: "no-store", credentials: "same-origin" });
      if (!response.ok) throw new Error(`Dashboard API returned ${response.status}`);
      render(await response.json());
    } catch (_error) {
      view.service.className = "service-state dead";
      const serviceDot = document.createElement("i");
      view.service.replaceChildren(serviceDot, document.createTextNode(" DISCONNECTED"));
    } finally {
      window.setTimeout(poll, 2000);
    }
  }

  setupChart();
  poll();
})();
