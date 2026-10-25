"use strict";

(() => {
  const byId = (id) => document.getElementById(id);
  const form = byId("backtest-form");
  const status = byId("status");
  const runButton = byId("run-button");
  const results = byId("results");
  const instrument = byId("instrument");
  const summary = byId("summary");
  const tradeRows = byId("trade-rows");
  const tradesEmpty = byId("trades-empty");
  let activeInstrument = "EMA9";

  const fmtMoney = (value) => value === null || value === undefined
    ? "--"
    : `${Number(value) > 0 ? "+" : Number(value) < 0 ? "−" : ""}₹${Math.abs(Number(value)).toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  const fmtTime = (value) => {
    if (!value) return "--";
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime())
      ? String(value)
      : parsed.toLocaleString("en-IN", { timeZone: "Asia/Kolkata", hour12: false });
  };
  const setStatus = (message, kind = "") => {
    status.textContent = message;
    status.className = `status ${kind}`;
  };

  async function loadActiveInstrument() {
    try {
      const response = await fetch("/api/dashboard", { cache: "no-store" });
      if (!response.ok) throw new Error(`Dashboard returned ${response.status}`);
      const snapshot = await response.json();
      if (snapshot.service?.strategy !== "EMA9") {
        runButton.disabled = true;
        setStatus("The active dashboard service is not an EMA9 forward runner.", "error");
        return;
      }
      activeInstrument = snapshot.service?.instrument || "EMA9";
      instrument.textContent = `${activeInstrument} · EMA9 FORWARD TEST`;
      byId("instrument-heading").textContent = activeInstrument;
      runButton.disabled = false;
    } catch (error) {
      runButton.disabled = true;
      setStatus(`Could not identify the active runner: ${error.message}`, "error");
    }
  }

  function renderSummary(data) {
    const fields = [
      ["DATE RANGE", `${data.start_date} → ${data.end_date}`],
      ["INDEX SIGNALS", data.summary.index_signals],
      ["OPTION TRADES", data.summary.option_trades],
      ["WINS / LOSSES", `${data.summary.wins} / ${data.summary.losses}`],
      ["WIN RATE", `${Number(data.summary.win_rate_pct).toFixed(2)}%`],
      ["GROSS OPTION P&L", fmtMoney(data.summary.gross_pnl)],
      ["PROFIT FACTOR", data.summary.profit_factor === null
        ? (Number(data.summary.option_trades) ? "∞ / no losses" : "N/A")
        : Number(data.summary.profit_factor).toFixed(2)],
      ["MAX DRAWDOWN", fmtMoney(data.summary.max_drawdown)],
      ["DAILY CAP HALTS", data.summary.daily_loss_halts],
    ];
    summary.replaceChildren();
    for (const [label, value] of fields) {
      const card = document.createElement("article");
      card.className = "metric";
      const caption = document.createElement("span");
      caption.textContent = label;
      const reading = document.createElement("strong");
      reading.textContent = String(value);
      if (label.includes("P&L") || label.includes("DRAWDOWN")) {
        reading.classList.toggle("pos", Number(label === "MAX DRAWDOWN" ? data.summary.max_drawdown : data.summary.gross_pnl) > 0);
        reading.classList.toggle("neg", Number(label === "MAX DRAWDOWN" ? data.summary.max_drawdown : data.summary.gross_pnl) < 0);
      }
      card.append(caption, reading);
      summary.appendChild(card);
    }
    tradeRows.replaceChildren();
    const rows = Array.isArray(data.trades) ? data.trades : [];
    tradesEmpty.hidden = rows.length > 0;
    byId("trade-count").textContent = `${rows.length} TRADES`;
    for (const trade of rows) {
      const row = document.createElement("tr");
      const values = [
        fmtTime(trade.entry_time), fmtTime(trade.exit_time), trade.signal,
        trade.option_symbol, trade.quantity, fmtMoney(trade.entry_option),
        fmtMoney(trade.exit_option), fmtMoney(trade.gross_pnl), trade.exit_reason,
      ];
      values.forEach((value, index) => {
        const cell = document.createElement("td");
        cell.textContent = value === null || value === undefined ? "--" : String(value);
        if (index >= 4 && index <= 7) cell.className = "num";
        if (index === 7) cell.classList.add(Number(trade.gross_pnl) >= 0 ? "pos" : "neg");
        row.appendChild(cell);
      });
      tradeRows.appendChild(row);
    }
    results.hidden = false;
  }

  async function watchJob(jobId) {
    while (true) {
      await new Promise((resolve) => window.setTimeout(resolve, 2000));
      const response = await fetch(`/api/backtest/${encodeURIComponent(jobId)}`, {
        cache: "no-store",
      });
      if (!response.ok) throw new Error(await response.text());
      const job = await response.json();
      if (job.status === "RUNNING") {
        setStatus("Backtest is running. Historical option data may take several minutes to download.");
        continue;
      }
      runButton.disabled = false;
      if (job.status !== "COMPLETED") throw new Error(job.error || "Backtest failed.");
      renderSummary(job.result);
      setStatus(`${activeInstrument} backtest completed.`, "success");
      return;
    }
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    runButton.disabled = true;
    results.hidden = true;
    setStatus("Starting backtest...");
    try {
      const response = await fetch("/api/backtest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          start_date: byId("start-date").value,
          end_date: byId("end-date").value,
        }),
      });
      const responseBody = await response.text();
      let job;
      try {
        job = JSON.parse(responseBody);
      } catch (_error) {
        if (!response.ok) throw new Error(responseBody || `Request failed (${response.status}).`);
        throw new Error("The dashboard returned an invalid backtest response.");
      }
      if (response.status === 409) throw new Error("Another backtest is already running. Wait for it to finish.");
      if (!response.ok) throw new Error(job.error || `Request failed (${response.status}).`);
      await watchJob(job.id);
    } catch (error) {
      runButton.disabled = false;
      setStatus(error.message, "error");
    }
  });

  loadActiveInstrument();
})();
