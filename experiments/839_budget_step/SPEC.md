# TASK_839 — Budget-constrained vector step (the env-wall fix): acceptance SPEC

**Decision**: `DECISIONS.md` D7 — **RATIFIED option (a)** (operator, 2026-07-09).
**Ledger**: `experiments/LEDGER.md` **828s3/env-wall** (2026-07-09).
**Pattern**: flag-gated, default-OFF, canon-bit-identical (the 838 discipline). This module
is the fixture-smoke acceptance harness; the real verdict is a scheduled phase-1 re-run
under the fixed step.

## Why

`step_vector` (canon) sizes EVERY simultaneous buying/shorting asset at
`min(0.1×PV, 0.99×cash)` against ONE cash snapshot, sums the deltas, and lets
`process_margin_call` clean up the resulting overdraft → long liquidation → BANKRUPTCY.
A near-uniform action vector (the first step of any untrained/exploring policy) therefore
bankrupts at step 0 — exploration is lethal from the first uniform-policy step, which is the
real root of the phase-1 HOLD collapse (the "fees punish exploration" reading of
828s3/preflight was wrong; fees only accelerate it).

## The fix (D7 opt (a), implemented in `mcts_trader/env/jax_trading_env.py`)

Flag `vector_budget_step` (default OFF = canon, parity-pinned in
`tools/tests/test_vector_step_parity.py`). ON: one pass computes
`n_trades = Σ(opening longs + opening shorts)` (CLOSING trades are proceeds → excluded),
then every OPENING market trade sizes at
`ideal_spend_value = PV × min(position_size_pct, budget_pct / max(n_trades, 1))`
(`budget_pct` default 0.99). Σ market spend = `PV × budget_pct` when the budget binds,
which is ≤ available cash exactly at episode start (PV≈cash — the env-wall's step-0 lethal
point); margin returns to being the exception path elsewhere. LIMIT orders keep the original
`ideal_trade_value` sizing (they reserve 0 cash and fill one-at-a-time — not the
simultaneous-overdraft source). All `jnp.where`/`jnp.maximum`, no dynamic shapes; the whole
ON branch is DCE'd when OFF (config is a static jit arg).

## Pre-registered reads (fixture, CPU-only, never `trading_data.db`)

**(i) 834-repro survival walk — control + fix.** Zero-logit dithered-book proposal
(`hold_bias=0.0` ⇒ near-uniform per-asset actions), legality-masked exactly as battery
self-play (`batch_compute_action_mask`, `max_positions=16`), 4 envs, `position_size_pct=0.1`
(the training / env-wall regime), on the DIVERSIFIED daily panel (279 assets — the H6
breadth regime; the 1-min `ohlcv` fixture has only 8 unique assets, so padding to 64 would
fabricate correlated exposure and is NOT used here). Under OFF the near-uniform first action
must bankrupt at step 0 (reproduce the wall); under ON the walk must survive ≥64 steps with
ZERO bankruptcies. Also reported: step-level cash OVERDRAFT events (cash<0 BEFORE the margin
call) — the fix drives that to zero.

**(ii) E1-pattern perfect-model probe — net@21, ON vs OFF, K16.** Mirrors
`experiments/831_search_probe` (dithered-book scheme, truncated true-env HOLD-rollout value)
at the net@21 tier (fee16/slip5). Pre-registered read: **ON ≥ OFF − 0.10** (budget sizing
must not destroy the NARROW-vector regime E1 lives in — a tie/small-loss is a PASS, a
collapse is a FAIL). In the E1 regime `position_size_pct = 0.5/A`, so
`budget_pct/n_trades ≥ 0.99/A > 0.5/A = pct` for any `n_trades ≤ A`: the `min()` always hits
the pct cap ⇒ the ON sizing is a provable no-op (delta ≈ 0). This is a FINDING report (838
pattern), not a ship gate.

## Results (fixture-smoke, this module; JAX 0.6.2 — `results_fixture_smoke.json`)

Panel: 279 assets, 1704 daily rows, 2021-01-01 .. 2025-09-01. Real measured numbers:

**(i) Survival walk** (4 envs × 70 steps, first-step open fraction 0.398):
| arm | bankrupt step / env | survived | step-level overdrafts |
|-----|---------------------|----------|-----------------------|
| OFF | [0, 0, 0, 0]        | 0/4      | 4 |
| ON  | [-1, -1, -1, -1]    | 4/4      | 0 |

→ OFF reproduces the env-wall (all 4 envs bankrupt at step 0). ON survives ≥64 steps with
ZERO bankruptcies AND ZERO step-level overdrafts. **(i) PASS.**

**(ii) E1 net@21 (K16, dithered-book, 30 decisions, pct=0.5/A=0.00179):**
`net21_OFF = +7.6827`, `net21_ON = +7.6827`, `delta(ON−OFF) = +0.0000` → **PASS**
(ON ≥ OFF − 0.10; the budget clamp is provably non-binding in the narrow regime, so the
book is byte-identical). No bankruptcies in either arm.

**Factory threading:** `create_jax_env_historical(..., vector_budget_step=True/False)`
propagates to `EnvConfig` (checked ON/OFF; `propagates: true`) — the flag reaches
`step_vector` by the exact path training/eval/probes use.

## Reproduce

```bash
JAX_PLATFORMS=cpu python experiments/839_budget_step/acceptance_probe.py \
    --out experiments/839_budget_step/results_fixture_smoke.json
```

## Out of scope

Weight-ladder action space (D7 opt (c) = H6), `process_margin_call` redesign, reward
shaping, phase-1 re-training (scheduled AFTER this ships). The residual mark-to-market
margin liquidations that CAN occur far from step 0 under maximal hb0.0 churn are the D7
"margin remains the exception path elsewhere" behavior — not the simultaneous-overdraft
wall this task fixes.
