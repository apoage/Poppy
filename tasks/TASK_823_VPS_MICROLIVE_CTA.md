# TASK_823 — VPS micro-live CTA: the linear trend book on real Kraken capital

**Status**: REJECTED/PARKED 2026-07-06 by operator — "not putting anything that stupid on the VPS,
waste of space and resources; not even a genetic algo there." A fixed 30d-sign rule is too trivial
to deploy and at €150 the P&L is ceremony. VPS stays a services host. A trader only earns VPS space
once TASK_825/826 yield a NON-trivial edge worth deploying. Spec retained below for reference only.

---

Original goal: take the ONE shippable edge — TASK_820's linear diversified
trend CTA (`trader/` package) — from paper (gate 3) to **micro-live** (gate 4) on the VPS with
**€100–150 real Kraken capital**. NOT the MuZero planner (closed, TASK_821). This is the honest
next rung: prove the *plumbing, key handling, reconciliation, and operator discipline* with real
money at a size where a mistake costs a coffee, not a paycheck.

**Read first:** `runs/821_planner_daily/EVAL_PASS1.md` is why this is the linear book, not the
planner. `tasks/TASK_820_SHIP_TREND_CTA.md` for the strategy + gate ladder. Memory
[[project_deployment_gate_ladder]], [[project_diversified_trend_shipcandidate]].

## Honest framing (read before touching money)

At €150 the edge is **not the point** — the point is the pipeline. The CTA's realized edge is
Sharpe ~0.6–0.7 point-in-time with a CI touching zero and −57% historical drawdown; at €150
notional the fixed costs (Kraken minimums, taker fee, spread) DOMINATE any signal. **Expected P&L
at this size ≈ slightly negative after costs.** Success = "the machine placed the right orders,
reconciled to the penny, respected its risk caps, and I slept fine," NOT "it made money." Do not
scale capital until the paper+micro-live pipeline has run clean for weeks (operator decision).

## Feasibility (answers the operator's question: can it live on VPS without heavy machinery?)

**Yes, trivially.** Verified 2026-07-06:
- `trader/` imports numpy + stdlib only — no eager JAX/torch (jax is a lazy opt-in in
  `config.get_backend()`, never triggered on the numpy default path). Runtime deps: `numpy`,
  `requests` (Kraken REST), `sqlite3` (stdlib). **Do NOT install jax on the VPS.**
[redacted: VPS ops detail]
  once-daily, seconds-of-CPU, <256MB job. Bound it (`systemd MemoryMax=512M`) so it can NEVER
  threaten the production services on that box.
- **Do NOT ship the 246M-row 1-min `trading_data.db`.** The live book needs only ~90 days of
  daily warmup + a daily fetch. Ship a tiny daily-only sqlite (the `ohlcv_daily` table, ~MBs) OR
  fetch 720 daily bars per asset via Kraken public REST on first run (`fetch_daily_kraken.py`
  already does this). Storage on VPS stays trivial.

## Phase A — paper on the VPS (prove it lives; zero risk)

1. Deploy `trader/` + `tools/fetch_daily_kraken.py` to VPS (rsync or a thin git checkout of just
   those paths; NOT the whole repo). Python venv with `numpy requests` only.
2. Wire the daily loop as a systemd user timer (mirror `tools/systemd/trader-paper.*`): fetch
   fresh daily bars → `python -m trader paper-step --source db` → append report. State/ledger in
   `~/trader-live/paper/`. `MemoryMax=512M`, runs ~00:20 UTC after candle close.
3. Run ≥1 week paper on the VPS. Exit criterion: daily ledger accumulates, no crashes, numbers
   match a workstation paper run on the same days (cross-check).

## Phase B — live execution adapter (DRY_RUN; no orders sent)

Build `trader/execution/kraken_live.py` — a SMALL standalone Kraken REST client (requests + hmac
+ base64, ~80–120 lines). **Do NOT import `mcts_trader.data.kraken_client`** (async + heavy deps);
port only the auth-signing + the 4 endpoints needed: `Balance`, `AssetPairs` (for `ordermin` +
lot decimals), `AddOrder`, `OpenOrders`. Then:
1. `trader live-step --dry-run`: fetch real balances → compute target weights (frozen strategy +
   risk overlay ON) → diff target vs current holdings → generate the order list (pair, side,
   volume) → **PRINT, do not send.** Round to Kraken lot decimals; drop any order below `ordermin`.
2. Validate every generated order against live `AssetPairs` metadata. Log rejected-too-small.
3. Reconciliation harness: fetch balances, reconcile against the internal ledger, assert drift <
   tolerance, alert on mismatch. This runs every cycle in live mode.

## Phase C — micro-live (real orders, tiny)

**THE BINDING CONSTRAINT — Kraken minimums vs €150/39 assets:** €150 across the 39-asset broad
book = €3.8/position, BELOW most Kraken `ordermin` and swamped by fees. The book AS-BACKTESTED is
not executable at this size. Resolution (pick + document):
- **Shrink the universe** to the most liquid, lowest-`ordermin` majors (BTC, ETH, SOL, XRP, and
  ~3–5 more) so each position clears `ordermin` with margin. Re-run the backtest on THIS reduced
  universe first (`trader backtest --source db` with a restricted asset list) and record its own
  gate numbers — a 6–8 asset book has less breadth, so expect a weaker (honest) Sharpe. Do not
  live-trade a universe you haven't re-backtested.
- Rebalance band widened (skip sub-min deltas), and consider H=3d rebalance (819 robustness showed
  net@8 CI stronger at 3-day) to cut turnover/cost drag at micro-size.

Risk caps for micro-live (hard, in `RiskConfig`): per-asset ≤ 25% (few-asset book), gross ≤ 1.0
(no leverage at gate 4), drawdown kill-switch at −15% halts new opens + alerts, daily order-count
cap, and a **global kill file** (`~/trader-live/HALT` present → flat + no trades).

## Security (non-negotiable)

- Kraken API key: **Query Funds + Create/Modify Orders ONLY. NO Withdrawal. NO margin.** Whitelist
  the VPS IP in Kraken key settings.
- Store secret in a `chmod 600` env file or systemd credential on the VPS, **never in git**,
  never echoed to logs. `.gitignore` the live state dir.
[redacted: VPS ops detail]
  + `CPUQuota` on the systemd unit.

## Gates / exit criteria
- Phase A: 1 week clean VPS paper.
- Phase B: dry-run orders reconcile byte-sane vs manual calc; all sub-min orders correctly dropped.
- Phase C: 2+ weeks micro-live, ledger reconciles to Kraken balances every cycle, risk caps
  provably fire in a drill, zero unresolved drift. THEN operator decides on any capital increase.

## Run location
VPS (`lukas@<redacted-domain>` / WG <redacted-ip>). Deploy from workstation. Sub-agents: build + report,
never touch live keys or send real orders without operator in the loop.
