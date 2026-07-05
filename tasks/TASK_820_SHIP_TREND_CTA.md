# TASK_820 — Ship the trend CTA: production `trader/` package + gate ladder

**Status**: ACTIVE (started 2026-07-05). Takes fork branch **(a) ACCEPT** from TASK_819 FINAL:
deploy the real-but-marginal diversified trend edge as a proper CTA with risk management, through
the 4-gate ladder [[project_deployment_gate_ladder]] (all-regime historical → unseen historical →
paper → live). Operator directive 2026-07-05: "make things working — trader on OHLC data available
in database"; heavy compute on gpu-server (workstation OOMs fast — dev/orchestration only).

## Frozen strategy (pre-registered, NO sweep — from 819 Phase-3 powered run)

- **Signal**: `s_i(t) = sign(logC_i(t) − logC_i(t−30d))`, daily close, leak-free (close ≤ t only).
- **Weights**: EQUAL across present assets, `pos_i = s_i / n_present(t)`, gross ≤ 1 ($1 long/short book).
- **Rebalance**: H = 1 day, at daily close (819-frozen execution assumption: signal at close(t)
  fills at close(t); acceptable for slow daily TSMOM, flagged).
- **Universe**: POINT-IN-TIME — all USD-quoted, non-stablecoin, non-fiat crypto in `trading_data.db`;
  each asset enters from its OWN listing (first real bar), dead coins included, present-mask
  (age ≥ max(lookback, vol_window) AND trailing vol > floor) exactly as 819 `portfolio.py`.
- **Costs**: net(c) = gross − turnover·c, c bp/side ONE-SIDED on Σ|Δpos| (unit flip = 2 units).
  Report at c ∈ {0, 3, 8, 21}. 21 = Kraken spot taker16+slip5; 8 = perps/maker band.
- **Explicitly rejected** (Phase-2 lever attribution): inverse-vol sizing (−0.06), lookback
  ensemble (−0.16). Do NOT re-add as signal shaping.

## Risk overlay (the "with risk management" of the fork — execution-level, not signal shaping)

1. Per-asset position cap: |pos_i| ≤ 5% of book (caps single-coin blowups; PIT universe is wide).
2. Drawdown throttle: at −20% from equity high-water-mark → halve gross; at −35% → flat + halt
   flag (Phase-3 maxDD was −57% uncapped; live book must not ride that).
3. Rebalance band: skip trades with |Δpos_i| < 0.2%/book (turnover ≈56×/yr is the main cost drag).
4. Staleness guard: if newest daily bar for the book is > 2 days old → no new positions, log alert.
5. All overlay params in one config dataclass; backtest MUST be runnable with overlay OFF to
   preserve parity with 819 (overlay OFF = bit-comparable to research numbers).

## Deliverable: top-level `trader/` package (numpy core, swappable array backend)

**Backend contract (operator override 2026-07-05):** math kernels written against a swappable
array-module handle (`xp = numpy` default; `xp = jax.numpy` selectable via `--backend jax` /
config) so heavy batch work (full PIT backtest, bootstrap batteries, walk-forward sweeps) can run
GPU-accelerated. Parity/gate runs stay on numpy float64 (determinism vs research numbers). GPU
launches follow the correct-allocation contract: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.7`
(PREALLOCATE default), cgroup RAM cap, one GPU job at a time — allowed on the workstation.

```
trader/
  __init__.py
  config.py        # frozen StrategyConfig + RiskConfig + paths (db, state dir)
  data.py          # daily panel: aggregate 1-min ohlcv → UTC daily closes; PIT universe
                   #   (listing = first real bar; liquidity = zero-return frac, PIT-recomputed);
                   #   reads/writes `ohlcv_daily` table (see TASK_820 fetch contract below)
  strategy.py      # trend signal + equal weights + present mask (port of audited 819 fns)
  risk.py          # overlay 1–4 above
  backtest.py      # accounting: net-of-cost, per-decision H-strided Sharpe, block-bootstrap CI,
                   #   equity/maxDD (port audited fns from runs/808/backtest.py + runs/819)
  paper.py         # daily paper loop: update bars → targets → simulated fills @ close w/ cost
                   #   model → append-only ledger (runs/820_paper/ledger.csv + state.json)
  cli.py           # python -m trader {selftest|backtest|parity|update-data|paper-step|report}
  selftest.py      # port ST1–ST6 pattern from 819 portfolio.py (annualization identity,
                   #   planted trend, diversification, cost-monotone, leak-free)
```

## Daily-bar store + fetch contract

- New table **`ohlcv_daily`**(asset_id, timestamp, open, high, low, close, volume, trade_count,
  source TEXT) in `trading_data.db`, PK (asset_id, timestamp). timestamp = UTC midnight epoch.
  source = 'agg1m' (aggregated from 1-min history) or 'kraken_rest' (interval=1440 fetch).
- Standalone fetcher `tools/fetch_daily_kraken.py`: incremental, resumable, rate-limited ~1 req/s,
  uses `assets.kraken_symbol`/`wsname`; REST interval=1440 gives 720 days/call (covers the
  2025-12-12 → now gap in one call per asset). Delisted pairs → mark and skip, they are dead
  coins (PIT universe handles them). Cross-check on overlap: REST daily close vs agg1m daily
  close must match within tolerance on sampled assets; log mismatches.

## Gate ladder

- **Gate 1a (parity vs Phase-2, workstation-runnable)**: engine with overlay OFF on the cached
  `runs/819_single_asset/_close_broad_daily.npz` universe must reproduce the Phase-2 best cell
  (broad/equal/single30 H=1d): FULL net@8 Sharpe +1.09 CI[+0.21,+1.95], net@21 +0.97, ×21.69 @8bp,
  maxDD −37.8%, turnover 0.155/day — Sharpe within ±0.02, turnover within ±0.005.
- **Gate 1b (parity vs Phase-3 powered)**: PIT 276-asset 2018–2025 run must land near FULL net@8
  +0.67 CI[−0.01,+1.34], net@21 +0.56, equity ×30.8, maxDD −57%, both bears positive
  (BEAR18 +0.53, BEAR22 +1.16). Powered script/artifacts live on gpu-server (`runs/819…/POWERED_RESULT.md`
  not yet pulled) — reconcile exactly once server is reachable; until then match the task-file numbers
  within Sharpe ±0.05 (PIT reconstruction differences allowed but must be explained).
- **Gate 2 (unseen OOS)**: frozen strategy on 2025-12-13 → present (data the strategy has NEVER
  seen; research froze 2026-06-28 with data through 2025-12-12). Pre-register: report FULL-window
  net@8/net@21 Sharpe + equity + maxDD, no tuning, whatever it says. Run with overlay OFF and ON.
- **Gate 3 (paper)**: daily `paper-step` in tmux on gpu-server (fallback: workstation under
  systemd-run MemoryMax cap, CPU-only — allowed, it is I/O + numpy-trivial).
- **Gate 4 (live)**: NOT in scope. Requires operator sign-off after ≥ 4 weeks of paper.

## Constraints (non-negotiable)

- 806→819 discipline: leak-free, cost-honest, per-decision H-strided Sharpe, BEAR as judge,
  net-beat without gross-beat = artifact.
- Workstation: no unsandboxed heavy jobs. Full-DB 1-min→daily aggregation (246M rows) runs
  EITHER on gpu-server OR workstation under `systemd-run --user --scope -p MemoryMax=8G` with
  streaming SQL (GROUP BY over cursor, no full-table load into RAM).
- Sub-agents: commit + report; NEVER call deploy.sh (main agent owns server-side writes).
- Working tree carries ~750 lines of prior uncommitted work + trainer.py.bak — do NOT stage
  blindly; `git add` only files this task creates/touches.

## Status log
- 2026-07-05: spec written; build fanned out to sub-agents (engine builder + daily fetcher).
  gpu-server unreachable from workstation (TCP timeout on <redacted-ip>:22) — server-side
  steps deferred; operator pinged to re-auth/verify tunnel.
- 2026-07-05 (later): engine committed `d110118` + `f9abd5c`. **Gate 1a PASS bit-exact** (all 6
  targets). Selftests 7/7 (ST7 = shadow-throttle V-recovery). Data layer complete: ohlcv_daily
  continuous 2013-10→2026-07-04 (agg1m 398 assets + kraken_rest 376, integrity exact).
  **Gate 1b (probe): net@8 Sharpe +0.72 CI[+0.04,+1.35] vs Phase-3 +0.67 CI[−0.01,+1.34] — within
  ±0.05 tolerance PASS**; net@21 +0.61 vs +0.56 ✓; maxDD −53.9% vs −57% ✓. CAVEAT: db-source
  universe = 363 alive-on-Kraken assets (35 agg1m-only dead coins skipped) → survivorship-tinted;
  dead-coin-inclusive flag requested; exact Phase-3 reconciliation still needs server artifacts.
  **Gate 2 (OFFICIAL, pre-registered, unseen 2025-12-13→2026-07-04, 203 decisions):** the OOS
  window is a hard BEAR (B&H Sharpe −2.61, ×0.47). Frozen strategy overlay OFF: **net@8 −1.31
  CI[−4.45,+1.29], gross −1.18, ×0.75, maxDD −37.8%** — beats B&H by +1.3 Sharpe/+28pp equity but
  ABSOLUTE NEGATIVE, diverging from historical bears (BEAR18/22 were absolute-positive). CI very
  wide (thin window). Overlay ON: maxDD −32.2%, turnover halved. DECISION: proceed to gate-3
  paper with overlay ON (costless observation), flag the bear-behavior divergence as the #1 thing
  paper must watch; re-score gate 2 monthly as OOS accumulates.
