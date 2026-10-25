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

### Separate NIFTY/SENSEX EMA9 forward test

The dedicated forward-test runner is separate from the EMA11 runner above and
does not change its behavior. Start one instrument at a time with:

```sh
python -m Dependencies.fyers_ema9_forward_launcher
```

Set `EMA9_INSTRUMENT=NIFTY` or `EMA9_INSTRUMENT=SENSEX`; the launcher refuses to
start if it is unset or invalid. `EMA9_EXECUTION_MODE` defaults to `PAPER`. The
runner uses the original completed-5-minute EMA4/11/18 entry signals, a 3-point
EMA9 touch/trailing exit checked from completed 1-minute candles, one option lot,
2 listed strikes ITM, the nearest listed expiry that has not passed, and a ₹3,500
daily realized-plus-open option loss cap. NIFTY and SENSEX use the same upcoming
expiry rule as their EMA9 replays. The two instruments have separate state and CSV
ledger files under `Forward Test Outputs/` by default.

Telegram alerts are enabled for this runner: it sends startup, entry, exit,
hourly health during the weekday market-data session, once-per-day end-of-day
option-P&L, and error notifications. The hourly health alert reports the runner
mode, market-data status and last success, open position, realized P&L, and daily
loss-cap status. It uses
`TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`, falling back to
`FYERS_AUTH_TELEGRAM_CHAT_ID` when no separate trade-alert destination is set.
Error details are redacted and repeated errors are rate-limited.

To enable live orders, set `EMA9_EXECUTION_MODE=LIVE`; leaving it unset or set to
`PAPER` keeps this runner paper-only. This is the sole execution-mode switch for
the EMA9 forward test and does not change the legacy EMA11 runner or other
strategies. Live startup requires a determinate, flat account unless the saved
runner position exactly matches the one broker position. Review paper behavior
and order sizing before setting LIVE.
An open position saved from a prior trading session blocks live restart for manual
reconciliation; an unpriceable/stale option quote while live also triggers a
market exit attempt. If Fyers confirms a fill but omits its average price, the
runner logs and records the pre-order quote as an estimate.

Forward execution follows the backtest's signal, stop, expiry, strike, and cap
rules, but cannot reproduce historical fills exactly: paper entries/exits use
fresh Fyers LTP quotes polled at `EMA9_POLL_SECONDS` (default 5 seconds), while
the replay marks options at completed 5-second candle closes. Live orders use
confirmed broker fills; slippage, polling delay, and option liquidity can change
results. Missing or stale quotes stop a paper action rather than fabricating a
price. In live mode a stale mark while holding a position triggers a market-exit
attempt instead of leaving the position unattended.

Set up the token with `python algo.py setup-token`. Historical Dhan extractors and
broker diagnostics remain separate utilities; they are not used by this EMA runner.
While the service is running, its private Telegram auth chat receives a fresh Fyers
authorization link at 9:00 AM IST Monday through Friday. Send `/auth` to request an
extra link. Before sending any authorization link, the service removes the saved
token file, any `FYERS_ACCESS_TOKEN` entry from `.env`, the process-environment copy,
and the active in-memory Fyers token. Fyers API calls will not work again until the
callback completes successfully, so finish the approval before the 9:25 AM entry
window. After the 3:15 PM square-off, the process remains alive but stops market-data
polling and waits with low activity until the next weekday's 9:25 AM session start.
The Telegram poller and `/callback` server remain available overnight. The 9:00 AM
prompt is only sent while the service is running.

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
