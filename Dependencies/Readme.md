# Dependencies

## Active EMA runner
`python algo.py run` starts the EMA strategy in [nifty_multi_strategy_master.py](../nifty_multi_strategy_master.py).
It gets NIFTY one-minute candles and option contracts from Fyers, resamples completed
five-minute bars, and applies the shared EMA signal logic. Bullish signals buy one
ATM CE and bearish signals buy one ATM PE for the first listed expiry in the next
ISO week. Contract symbols and lot sizes come from Fyers' NSE derivatives master.

The runner is paper-only by default. To enable real orders, configure the Fyers
credentials below and explicitly set `LIVE_TRADING_ENABLED=true` in the ignored
`Dependencies/.env` file. Live startup refuses to proceed if the Fyers account has
an open order, position, or indeterminate account response. A partial or ambiguous
order result stops the runner for manual reconciliation.

Set up the token with `python algo.py setup-token`. Historical Dhan extractors and
broker diagnostics remain separate utilities; they are not used by this EMA runner.

## Fyers EMA replay
`python algo.py fyers-backtest --start-date YYYY-MM-DD --end-date YYYY-MM-DD`
replays the live EMA process without using the legacy backtest scripts. It uses
the shared EMA signal logic on complete NIFTY five-minute candles, then uses
historical candles for the selected Fyers option contract to estimate the
paper-run quote without looking beyond the simulated action time. Entry/EMA9
exit actions use the quote at the simulated decision poll, and daily loss is
checked on each poll. The `EMA_MAX_LOSS` threshold (default ₹5,500) is per
trading day and includes realized P&L plus the open position's estimated P&L;
once reached, the open trade exits and further entries are blocked for that day.
Since checks use the simulated poll schedule and completed option candles, a
loss can cross the threshold between checks. The default poll interval and lots
come from `EMA_POLL_SECONDS` and `EMA_LOTS`.
Option candle resolution defaults to `auto` (`5S` for recent history, otherwise
`1` minute); quote fills are therefore approximations, not guaranteed live fills.
Historical candles are cached persistently under `Backtest Outputs/.fyers_cache/`
by default (`EMA_REPLAY_USE_CACHE=true`). Use `--useCache=true` to enable cache,
`--useCache=false` (or `--no-cache`) to fetch candles directly without reading or
writing cache, or `--refresh-cache` to fetch and replace cached candles. The
command writes an auditable trade CSV and summary under `Backtest Outputs/`.

To test the EMA signals against the underlying index only, use
`python algo.py fyers-backtest --instrument index --start-date YYYY-MM-DD --end-date YYYY-MM-DD`.
It defaults to NIFTY; pass `--index sensex` to use Fyers' `BSE:SENSEX-INDEX`
history instead. This mode uses the same completed five-minute signal rules and reports
directional index points per trade; it does not fetch option contracts or
premiums. Entry and exit prices use the index close available at the simulated
decision. Results are not rupee P&L, and no lot multiplier, fees, slippage, or
rupee-denominated daily max-loss rule is applied. Index-only replay defaults to
a separate 170-point daily loss cap (`--max-daily-loss-points`); it counts
realized plus close-based open directional points, exits at the completed
five-minute candle close when the cap is breached, and blocks entries for the
rest of that day. Because exits are evaluated only at candle close, a gap or
fast move can make the loss exceed the cap. Index history is
requested one calendar day at a time to avoid Fyers range-request failures.
SENSEX index testing remains the source of all entries, exits, and cap decisions;
option prices are available as an optional reporting overlay. The separate
options-driven replay remains NIFTY-only.
An index-only trailing stop can be selected with `--trailing-stop`:
`swing_2bar`, `atr_2x`, `breakeven_1r_ema9`, `breakeven_1r_ema11`, or
`combined`. The default is
`breakeven_1r_ema9`; pass `--trailing-stop none` to disable it. This default
had the strongest net points among the compared modes in the January–October
22, 2026 NIFTY replay; this is an in-sample historical result, not a guarantee.
The strategy exit defaults to EMA9 and the 170-point daily cap remains active
in all modes. For the original/no-extra-trail variant with EMA11 exits, pass
`--trailing-stop none --exit-ema 11`; entries are unchanged.
The `breakeven_1r_ema11` mode uses EMA11 for both the initial 1R distance and
the post-1R trailing level; it does not change the strategy's EMA9 exit.
EMA exits require a completed candle close below the selected EMA for a long
or above it for a short; wick touches alone do not trigger an exit. Trailing
stops and the daily loss cap are also triggered and filled at the completed
candle close, not at an intrabar price. Trailing levels tighten after a
completed bar and apply to subsequent bars.

## SENSEX EMA9-touch paper runner
This separate runner reads live SENSEX index candles and option quotes from
Fyers, but is paper-only: it does not import or call the order-execution client.
It requires `LIVE_TRADING_ENABLED=false` explicitly and refuses startup if the
flag is missing or true. It starts the existing Telegram Fyers OAuth service:
if the saved token is missing or expired, it sends an authorization link to
`FYERS_AUTH_TELEGRAM_CHAT_ID` and waits for approval; the token is then saved to
`FYERS_TOKEN_FILE`. The same service monitors the token during runtime and
requests new Telegram authorization when Fyers rejects it. Set
`FYERS_APP_ID`, `FYERS_SECRET_KEY`, `FYERS_REDIRECT_URI` (the server's public
HTTPS callback URL registered in Fyers), and `FYERS_AUTH_TELEGRAM_CHAT_ID`;
expose the platform's `PORT` to the callback server. The notification chat is
configured separately using `TELEGRAM_CHAT_ID`.

Configure the server service's start command as
`python Dependencies/fyers_sensex_paper_runner.py` (optionally pass
`--option-daily-loss-cap 3500` and `--poll-seconds 30`). Do not start it on a
local workstation when the intended deployment is the production server.

Signals use completed five-minute candles. LONG buys the nearest-expiry ATM CE;
SHORT buys the nearest-expiry ATM PE. The selected contract's official lot size
is used for one lot, and option entries/exits are paper-marked at live LTP.
EMA9-touch exits follow the historical replay rule: check each completed
one-minute SENSEX candle against the prior completed five-minute EMA9 adjusted
by the configured buffer; with the default 3-point buffer, LONG stops are 3
points below EMA9 and SHORT stops are 3 points above it. If a candle gaps
through the buffered level, use its open, otherwise use that level. The
five-minute trailing stop starts at the similarly buffered entry-bar EMA9 and
ratchets in the trade's favorable direction. The buffer defaults to 3 points
and can be changed with `SENSEX_PAPER_EMA9_BUFFER_POINTS` or `--ema9-buffer-points`.
The daily loss cap is ₹3,500 of aggregate option P&L: realized option P&L plus
the open position's current option-LTP mark-to-market. It is evaluated each
time the runner polls quotes (30 seconds by default, configurable with
`--poll-seconds` or `SENSEX_PAPER_POLL_SECONDS`). When reached, the runner
closes the paper position at the current option LTP and halts entries for the
rest of the day. Rapid moves can exceed the cap between polls. EMA9 exits and
the trailing stop remain based on completed SENSEX candles. Option P&L remains
an estimate from LTP snapshots and excludes fees, spread, and slippage.
Configure the cap with `--option-daily-loss-cap` or
`SENSEX_PAPER_OPTION_DAILY_LOSS_CAP`; the default is 3500 rupees. Old `--cap`
and `SENSEX_PAPER_DAILY_LOSS_CAP` index-point settings are rejected to avoid
silently misinterpreting their units. Telegram alerts use the existing
`TELEGRAM_ENABLED`, `TELEGRAM_BOT_TOKEN`, and `TELEGRAM_CHAT_ID` configuration.

To test stop spacing in the historical index replay, run
`python "Dependencies/fyers_sensex_touch_replay.py" --caps 300 --ema9-buffer-points 3`.
The buffer is applied below EMA9 for LONG stops and above EMA9 for SHORT stops,
including the initial stop, each five-minute trailing update, and one-minute
touch exits. The replay defaults to 3 points to match the paper runner; pass
`--ema9-buffer-points 0` to reproduce the original unbuffered replay.
The replay checks the daily cap at each completed one-minute close, before
checking that minute's EMA9-touch condition; on a same-minute collision, the
cap close is used.

To add historical nearest-expiry ATM option candle marks to each SENSEX replay
trade, pass `--option-prices`. Option marks always use Fyers Prime 5-second
candles, including for expired contracts, with no fallback to one-minute
history. For example:
`python "Dependencies/fyers_sensex_touch_replay.py" --start-date 2026-10-01 --end-date 2026-10-23 --caps 300 --option-prices`.

The resulting trades CSV includes the selected contract, expiry, strike,
resolution, entry/exit option candle close and completion timestamps, and
estimated one-lot gross premium P&L. The quote is the latest fully completed
five-second option candle available at or before the index action timestamp,
so no future option candle is used. Missing candles are labeled in
`option_price_status` and left blank rather than replaced with lower-resolution
prices. With `--option-prices` alone, this overlay does not change any
index-based signal, fill, cap, or exit. Premium P&L is indicative only, excludes
fees/spread/slippage, and uses the current derivatives-master lot size as a
proxy for archived contracts. Option history requests are cached with the rest
of the Fyers history.

To use a rupee-denominated option premium cap in the historical replay, pass
`--option-daily-loss-cap 3500`. The replay uses 5-second history, aggregates
realized P&L and the open position's premium mark-to-market, and checks the
total at every completed 5-second candle. At or below -₹3,500 it exits at that
candle close and blocks new entries for the rest of the day.
`--option-daily-loss-cap` is mutually exclusive with `--caps`; the existing
index-based replay behavior remains the default when no option cap is specified.

Option-cap trades use the latest completed one-minute SENSEX close available at
the option-cap timestamp to estimate index points for reporting; this is an
index mark, not a synchronized 5-second SENSEX fill. The option-cap decision
and estimated premium P&L use only completed 5-second option candles. This
historical option P&L remains gross and uses the current derivatives-master lot
size as a proxy for archived lot sizes.

To compare contract selection while holding the index signals and exit rules
constant, run:
`python "Dependencies/fyers_sensex_touch_replay.py" --start-date 2025-01-01 --end-date 2026-10-23 --option-daily-loss-cap 3500 --option-strike-steps 2 1 0 -1 -2`.
Zero selects nearest ATM; positive values select that many listed strike steps
ITM and negative values that many steps OTM, with CE and PE directions handled
automatically. Each selection gets an independent 5-second option-P&L cap
replay. The command writes individual trades/summary files and a combined
strike-comparison CSV with gross P&L, trade-equity drawdown, daily P&L, win
rate, profit factor, cap days, and price-mark coverage. As with other replay
results, these are estimates before fees, spread, slippage, and market impact.
Because Fyers' expired-symbol endpoint can return a sparse strike list, expired
ATM and offset strikes are reconstructed from the expiry symbol prefix using
the minimum current SENSEX master strike spacing; the replay summary records
that spacing. Historical changes to the strike interval would affect this
reconstruction.

For a server deployment, set `LIVE_TRADING_ENABLED=false`,
`TELEGRAM_ENABLED=true`, and persist the token, ledger, and state on a durable
server volume. The token file is written by Telegram authentication after you
approve the morning Fyers login; a preloaded `FYERS_ACCESS_TOKEN` is optional.
For example, with a volume mounted at `/data`, set
`FYERS_TOKEN_FILE=/data/fyers_access_token`,
`SENSEX_PAPER_TRADES_FILE=/data/fyers_sensex_paper_trades.csv`, and
`SENSEX_PAPER_STATE_FILE=/data/fyers_sensex_paper_state.json`. The default cap
and polling interval are configurable through
`SENSEX_PAPER_OPTION_DAILY_LOSS_CAP` (3500) and
`SENSEX_PAPER_POLL_SECONDS` (30), or CLI flags. Keep the same cap when resuming
a saved state. Changing from the old index-point cap requires archiving the
incompatible saved state and ledger after confirming no paper position is open.
The runner rejects incompatible state versions and ledger schemas rather than
silently resetting them. Use `--help` to see options.

The remaining sections describe retained shared utilities and legacy broker
integrations. Their `LIVE_BROKER` guidance does not apply to the active EMA runner.

Shared configuration, the safety/data primitives the front-test master imports on
every run (paper or live), and the live-execution layer. The broker subfolders only
come into play when live trading is switched on.

## Contents
```
Dependencies/
├── env.example              # template; copy to .env (git-ignored) and fill in
├── dhan_token_setup.py      # one-time DhanHQ OAuth login; writes DHAN_ACCESS_TOKEN into .env
├── check_env_config.py      # `python algo.py check-env` — read-only .env vs template vs code audit
│
│   # Shared runtime primitives (imported by the master on every run)
├── broker_contract.py       # OrderResult/OrderStatus types + the contract every broker adapter implements
├── market_data_health.py    # OHLC validation + live feed-freshness gates (entry freeze / liquidation)
├── tick_bar_builder.py      # pure websocket tick -> 1-min bar + official-candle true-up helpers
├── execution_ledger.py      # per-leg live order ledger; exposure is never inferred from a boolean
├── startup_exposure.py      # startup broker-book audit; live workers blocked until proven flat
├── trading_lifecycle.py     # safe live-session stop/square-off state machine
├── next_open_entry.py       # "enter at next candle open" pending-entry primitive (Goldmine/Money Machine)
├── risk_sizing.py           # shared risk-budget position sizing (whole lots, skip-over-budget)
├── secret_redaction.py      # strips credentials/tokens from anything headed to logs
├── diagnostic_preflight.py  # safety checks before a diagnostic test order's typed-YES prompt
├── test_*.py                # focused pytest suites for the modules above + repository policy
│
│   # Broker live-execution adapters (one folder each, same surface)
├── Kotak API/           # Kotak Neo live execution
│   ├── kotak_execution.py        # KotakExecutionClient (shared singleton)
│   └── diagnose_kotak_symbol.py  # read-only symbol check + optional test order
├── Shoonya API/         # Shoonya (Finvasia) live execution
│   ├── NorenApi.py               # vendored Finvasia NorenApi client
│   ├── shoonya_execution.py      # ShoonyaExecutionClient (shared singleton)
│   └── diagnose_shoonya_symbol.py# read-only symbol check + optional test order
├── Flattrade API/       # Flattrade Pi v2 REST live execution
│   ├── flattrade_execution.py      # FlattradeExecutionClient (shared singleton)
│   ├── diagnose_flattrade_symbol.py# read-only symbol check + optional test order
│   └── test_flattrade_execution.py # offline unit tests for the adapter
└── Dhan API/            # Dhan live execution (same SDK as market data, separate session)
    ├── dhan_execution.py           # DhanBrokerClient execution counterpart (shared singleton)
    ├── diagnose_dhan_symbol.py     # read-only symbol check + optional test order
    └── test_dhan_execution.py      # offline unit tests for the adapter
```

`.env` itself is git-ignored (it holds secrets); `env.example` is the committed template.
All execution modules load `.env` from this `Dependencies/` folder. A few files appear
here only at runtime and stay untracked: `.env`, `log_files/` (the master's append-mode
log), the nightly-refreshed `all_instrument <date>.csv` Dhan instrument master (used
for option-contract resolution by the runner and the Dhan adapter), and the Google
Sheets OAuth token cache when the EOD sheet writer is enabled.

(Tests for everything in this folder live at the mirrored path under `Tests/Dependencies/`,
alongside `Tests/test_market_data_health.py` and the master's own suite. See
`docs/adr/0010-tests-in-a-mirrored-tests-tree.md` for why.)

## Keeping `.env` honest
Configuration drifts in three directions at once: your `.env`, the committed
`env.example`, and the in-code default behind every `_env_*` call. A key that is in
the code and the template but missing from your `.env` is **not** an error — the
runner silently uses the in-code default — which is exactly what makes it easy to
miss. Audit all three at once with:
```
python algo.py check-env
```
It reports settings missing from your `.env`, mistyped or stale keys (a typo means
the setting you intended is not being applied at all), and knobs missing from the
template. It is read-only, exits non-zero on findings so it can gate a pre-flight
script, and prints key NAMES only — never a value out of your `.env` — so its output
is safe to paste when asking for help. `test_repository_policy.py` reuses the same
helpers, so CI fails when a new `_env_*` key lands without an `env.example` entry.

## How live execution works
The master file guarded-loads all four broker clients and then picks one with
`LIVE_BROKER` (`KOTAK`, `SHOONYA`, `FLATTRADE`, or `DHAN`). The rest of the runner
only touches a single generic `execution_client`, so the brokers are interchangeable.
All clients expose the **same surface**:
`ensure_logged_in()`, `preload_scrip_master()`, `resolve_option_symbol()`,
`place_market_order()` (with fill confirmation), `get_order_status()`,
`cancel_order()`, `list_open_orders()`, `list_open_positions()`,
`recover_after_reconciliation()`, `extract_order_id()`, `logout()`, and an
`is_logged_in` flag.

Safety gates (all in `.env`):
- `LIVE_TRADING_ENABLED` — global kill-switch (default `false` = everything paper).
- `LIVE_BROKER` — `KOTAK`, `SHOONYA`, `FLATTRADE`, or `DHAN`. An unrecognised value
  **fails closed** (live disabled, paper only) rather than guessing a broker.
- `<PREFIX>_LIVE_TRADING` — per strategy. A strategy goes live only when the global
  switch **and** its own flag are both true.
- A live entry falls back to paper only after an explicit typed `REJECTED` result
  with zero fill. `PARTIAL` or `UNKNOWN` means broker exposure may exist: entries
  freeze, exits remain available, and reconciliation continues. A rejected or
  ambiguous exit keeps the leg tracked and open.
- Startup queries open orders and relevant option positions before any live
  worker starts. Exposure or an indeterminate broker reply blocks all live
  workers; it is never auto-adopted or auto-flattened.

Product/exchange per broker is derived from `LIVE_BROKER`: Kotak uses `nse_fo` +
`KOTAK_PRODUCT_TYPE`; Shoonya uses `NFO` + `SHOONYA_PRODUCT_TYPE`; Flattrade uses
`NFO` + `FLATTRADE_PRODUCT_TYPE` (INTRADAY → MIS/I, NORMAL → NRML/M); Dhan uses
`NSE_FNO` + `DHAN_PRODUCT_TYPE`.

## Credentials (in `.env`)
- **Kotak Neo:** `CONSUMER_KEY`, `MOBILE`, `MPIN`, `UCC`. TOTP is typed at startup
  (not stored).
- **Shoonya:** `SHOONYA_USERID`, `SHOONYA_PASSWORD`, `SHOONYA_VENDOR_CODE`,
  `SHOONYA_API_SECRET`, `SHOONYA_IMEI`, `SHOONYA_TOTP_SECRET` (base32 seed; the TOTP
  is auto-generated via `pyotp`, or you're prompted if blank). The client is vendored
  here.
- **Flattrade:** `FLATTRADE_CLIENT_ID`, `FLATTRADE_API_KEY`, and
  `FLATTRADE_API_SECRET`. `FLATTRADE_ACCESS_TOKEN` is optional: when blank, startup
  opens Flattrade authorization and asks for the returned `request_code`. Tokens
  remain in memory unless you manually paste one into `.env`. Uses core `requests`
  and `pandas`; no extra SDK is required. Token exchange must originate from the
  static IP registered against your Flattrade API key.
- **Dhan:** reuses the market-data credentials (`DHAN_CLIENT_CODE` +
  `DHAN_ACCESS_TOKEN` from `dhan_token_setup.py`) — no extra keys. Execution runs
  on its own SDK session, separate from the market-data session, and resolves
  option contracts from the local `all_instrument <date>.csv` instrument master.

The exact upstream broker compatibility environment is recorded in:
```
pip install -r requirements-brokers.txt
```
Run that command only in a clean broker-validation environment: Kotak v2.0.1
declares older exact pandas/requests constraints that conflict with the app's
audited core runtime. In the live app environment install Shoonya with
`pip install pyotp==2.9.0 websocket-client==1.8.0`, and install Kotak with the
official tagged command plus `--no-deps` shown in the root README. Flattrade
uses the pinned core `requests`/`pandas` set.

> Note: Finvasia is decommissioning the legacy Shoonya QuickAuth endpoint (it returns
> HTTP 502), so Shoonya live login needs the new OAuth API before it works again.

## Diagnostics (read-only, with an optional REAL test order)
Each broker has a diagnostic that logs in and shows whether a contract resolves to a
valid trading symbol. Passing `--place-order` and an explicit `--qty` additionally
places a **real**, confirmation-gated BUY. It attempts the matching SELL only after
the BUY is confirmed fully filled; an ambiguous result requires broker reconciliation.
The diagnostic asks you to type `YES` before submitting anything:
```
python "Dependencies/Kotak API/diagnose_kotak_symbol.py" CE 23950 --place-order --qty <current-lot-size>
python "Dependencies/Shoonya API/diagnose_shoonya_symbol.py" CE 23950 DDMMMYY --place-order --qty <current-lot-size>
python "Dependencies/Flattrade API/diagnose_flattrade_symbol.py" CE 24150 DDMMMYY --place-order --qty <current-lot-size>
python "Dependencies/Dhan API/diagnose_dhan_symbol.py" CE 24150 DDMMMYY --place-order --qty <current-lot-size>
```
