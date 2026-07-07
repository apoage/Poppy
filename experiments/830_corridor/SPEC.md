# E0 — Instrument corridor (TASK_830, pathway rung R0)

**Hypothesis**: none. This is a **calibration/instrument** experiment. It produces the two
numbers every later rung is scored against and, in doing so, validates the just-shipped
`step_vector` (TASK_828 Stage 1) by pushing a known book through it.

## Metrics (both produced INSIDE / against the vector env accounting where stated)

1. **Corridor floor** — the TASK_819 linear trend book (single-30d-sign, equal-weight,
   gross ≤ 1) replayed as a sequence of per-asset `action_vec ∈ {0..4}` through
   `step_vector` (`EnvConfig.vector_action=True`), walked over the daily panel. Metric =
   net@21 (fee16+slip5) H-strided **per-decision Sharpe** (`mean/std · √(525600/1440)`,
   the 819/backtest annualization), cost-honest, leak-free (signal at row *t* uses close
   ≤ *t*; the forward label is the only future-touching quantity, applied by the env one
   step later).

2. **Corridor ceiling** — per-asset max-extractable under the SAME cost model. Reuses
   `mcts_trader/jax_agent/path_solver.py` (greedy hindsight solver) per asset, summed into
   an equal-weight, perfect-direction book, scored with the identical `trader/` accounting.
   The oracle is a **measurement instrument / upper bound ONLY — never a teacher** (H5).

Every later experiment is reported as a position inside **[floor, ceiling]**.

## Reproduction target + pre-registered tolerance

The floor must **reproduce the `trader/` engine's net@21 Sharpe** on the same panel
(the pathway bar: **+0.97** on the real 2021–2025 24GB universe; the reference is
recomputed on whatever panel is loaded so the check is self-consistent).

- **Pre-registered PASS**: `|floor_inenv.net21 − reference.net21| ≤ 0.10` (Sharpe units),
  AND the same on cost=0 gross.
- **Justification of the tolerance.** Sharpe is invariant to constant leverage, and the 819
  book is equal-weight × sign, whose *relative* weighting `step_vector` reproduces exactly
  (fixed per-asset notional, sign direction). So gross should match to sampling noise, and
  net should differ only by the **cost-base discrepancy** (the env charges fee+slip on traded
  *notional* per fill; the book charges `turnover · c/1e4` on the $1-book weight change) —
  a few bps of turnover, i.e. < 0.10 Sharpe. Anything larger is a **structural env-semantics
  divergence, not slippage**, and is reported as the E0 finding (filed against 828 Stage 1),
  **not tuned away**.
- **Known divergence sources** (asserted, documented, not absorbed): (a) `step_vector` sizes
  every asset against ONE pre-rebalance cash snapshot — combined demand at gross → 1
  over-commits cash and triggers margin-call liquidation; the book is cash-neutral (Σ|w| ≤ 1).
  (b) directional flips (long↔short) take TWO decisions in the env (close, then open) vs an
  instantaneous weight flip in the book. (c) the env holds uninvested cash, so as the present
  count drifts the *effective* leverage drifts → time-varying vs the book's fixed unit gross.

## Replay sizing (documented modelling choice, NOT env tuning)

`position_size_pct = 0.5 / n_assets` (default). Rationale: caps env gross at ≤ 0.5 even when
all assets are present, staying clear of the simultaneous-snapshot over-commitment cliff
(empirically `pct ≥ 1/n_assets` bankrupts the book — itself a finding). Sharpe is
leverage-invariant across `pct ∈ [0.1/A, 0.5/A]` (verified in the smoke), so this choice does
not move the reported floor; it only avoids the cash-overflow artifact. It is a replay
parameter, not a change to `step_vector`.

## Windows / cost tiers / seeds / compute

- **Windows**: 2021-01-01 → 2025-09-01 FULL, plus per-regime incl. **BEAR22** (2021-11-10 →
  2022-12-31) — regimes from `trader.config.REGIMES`.
- **Cost tiers**: cost=0 (gross) then fee16/slip5 (net@21).
- **Seeds**: n/a — the walk is deterministic (no episode-start sampling; fixed start = warm).
- **Compute budget**: CPU only (`JAX_PLATFORMS=cpu`). Fixture smoke ≈ seconds–minutes; the
  real-panel run is a **scheduled CPU one-shot** over the read-only 24GB `trading_data.db`
  (NOT a factory-loop job — invariant #7).
- **Kill date**: 2026-07-22. If the runner cannot produce a finite corridor by then, E0 is
  re-scoped.

## Deliverables

- `corridor.py` — the runner (this dir). Imports the 819 book (`trader.strategy`), the
  accounting (`trader.backtest`), the panel loader (`trader.data`), the env (`step_vector`),
  and the oracle (`path_solver`). No edits to any monster file.
- Fixture smoke: `python corridor.py` (defaults to `fixtures/trading_fixture.db`), writes
  `results_fixture_smoke.json`. Real numbers come from the scheduled run.
- One `experiments/LEDGER.md` line: verdict + corridor `[floor, ceiling]` per window/cost tier
  (filled from the SCHEDULED run; fixture-smoke line marked as such).

## Attribution — the fixture-smoke tolerance miss is a replay-MAPPING artifact, NOT a `step_vector` defect

A full verify panel confirmed the env `step_vector` is faithful: an independent from-scratch
cash-account simulator reproduces it to **3–4 sig figs** (finalPV to the dollar; no money
created/destroyed, no fee/PnL mis-sign). The fixture-smoke miss (Δ gross −0.20, net@21 −0.33
vs the 819 book) is therefore **NOT** an env-semantics bug filed against 828 Stage 1 — it is a
property of how this runner **maps** the 819 book into `action_vec`. Pure-numpy decomposition
(no env), on the FULL gross Sharpe:

- 819 reference book **+0.956**; env corridor floor **+0.753**.
- **Dominant negative term** = the runner's fixed `0.5/A` per-asset weight vs the 819 book's
  renormalized `1/n_present` weighting: **0.956 → 0.612**.
- Flip-lag (long↔short flip costs 2 env decisions) actually **HELPS: +0.14**.
- Fixed-weight + flip-lag combined ≈ **0.740** → the env `step_vector` adds only **+0.013**
  over the pure mapping.
- Floor Sharpe is **leverage-invariant** across `pct = 0.1/A … 0.5/A` (affordability-binding
  contributes ~nothing; only `1.0/A` bankrupts — the documented over-commit cliff).
- **BEAR22** collapse (floor +0.225 vs ref +1.218 gross) = **faithful hold-shares
  de-leveraging**: a winning short's notional `s·P` shrinks as price falls, so the corridor
  holds fixed SHARES while the 819 book rebalances to constant DOLLAR weight daily. Different
  book, correctly simulated — not a bug.

The reframed output/LEDGER carry the number unchanged and attribute the miss to the mapping.
**Which mapping defines the official corridor floor is a pending planner decision** (a
DECISIONS row is being seeded for it) — this runner does not add a new weighting mapping.

### What the SCHEDULED real-panel run must measure to close E0 on the true universe

1. **Leverage sweep** — repeat `pct ∈ {0.1/A … 0.5/A}` on the real 2021–2025 panel to confirm
   leverage-invariance holds off-fixture (and re-confirm the `1.0/A` over-commit cliff).
2. **Both books** — report a **hold-shares** floor AND a **constant-dollar-weight daily-rebalance**
   floor (or explicitly label which book the reported floor is), so BEAR22 is interpreted against
   the matching book rather than mis-read as an env divergence.
3. **Decomposition** — recompute the fixed-weight vs renormalized + flip-lag decomposition on the
   real panel, confirming `step_vector`'s residual over the pure mapping stays ≈ +0.01 (env fidelity).

## Kill / pivot (from pathway)

E0 failing means the corridor floor cannot be produced within tolerance **for a reason that
implicates `step_vector` itself** — that must be fixed before any rung above spends GPU. The
fixture-smoke tolerance miss here does NOT meet that bar (it is a replay-mapping artifact; the
env reproduces an independent cash-account sim to 3–4 sig figs). The scheduled real-panel run
confirms the attribution on the true universe and settles which mapping is the official floor.
