# TASK_840 — Target-weight rebalance step (the H6 env): acceptance SPEC

**Decision**: `DECISIONS.md` D8 — **RATIFIED option (c)** (the H6 env; escalated 2026-07-11).
**Ledger**: `experiments/LEDGER.md` **835/S4-p1b** (budget-step policies still bankrupt).
**Pattern**: flag-gated, default-OFF, canon-bit-identical (the 838/839 discipline). This module
is the fixture-smoke acceptance harness; the real verdict is a scheduled phase-1c re-run under
the fixed step.

## Why

Even with the 839 budget step, `step_vector` sizes each opening leg at a fixed fraction of PV
and lets exposure accumulate monotonically; sparse books still size 10%×PV/trade and shorts
bleed until the margin cascade wipes PV (835/S4-p1b: budget-trained policies bankrupt at
step 132 / 181; a wide budget-bound walk dies ~step 104). The nets learn (834/E3 DEC-1 PASS,
836/E4 value>momentum) — **the env kills.** Incremental-order semantics is the demonstrated
residual killer.

## The fix (D8 opt (c), implemented in `mcts_trader/env/jax_trading_env.py`)

Flag `vector_weight_rebalance` (default OFF = canon; the OFF path is bit-identical to BOTH the
canon path AND the 839 budget path — two parity pins: `tools/tests/test_vector_step_parity.py`
`48e1f467…` and `tools/tests/test_vector_weight_parity.py` `4f8b692e…`). ON, each per-asset
action maps to a TARGET signed weight — Buy `+w_unit`, Sell `−w_unit`, Hold/LimitSell/LimitBuy
`+0` — and `raw_target = clip(w_cur + step_w, ±w_max)` is CLIPPED for EVERY asset
(`w_max = vector_weight_max·w_unit`, default 4·w_unit; `w_unit = 1/n_assets` default). EVERY
position whose `|raw_target − w_cur| > deadband` is re-trimmed (config
`vector_rebalance_deadband`, default 0.1·w_unit) — including a held leg the clip pulled back to
the cap, which is what stops a drifted short bleeding unbounded.

TWO deliberately-distinct quantities (conflating them was the root of the two BLOCKER
mechanisms the high-tier panel found):
* the **target/cap/clip ladder** uses **signed MARKET NOTIONAL** `w_cur = ±size·price/PV` — so
  a losing short whose notional grows past the cap gets a target below `w_cur` ⇒ it re-trims
  (mechanism-2 fix: `|w|` cannot run away);
* the cash **RESERVATION of a HELD leg** uses its true **CAPITAL footprint** `posval/PV`
  (`compute_position_values`: long `size·price`, short `2·size·entry − size·price`), NOT market
  notional — a winning short's capital EXCEEDS its market notional, so reserving by `|w_cur|`
  under-counts it and over-admits new opens ⇒ overdraft (mechanism-1 fix).

Re-trimmed legs are admitted only into the remaining capital budget `1 − Σ_held posval/PV`, so
`Σ|w| ≤ 1` (no leverage) AND at fee=0 `weight_cash ≥ 0` EXACTLY
(`cash = PV·(1 − Σ_held posval/PV − Σ_reb|w_final|) ≥ 0` by induction; `Σ posval ≤ PV` entering
the step). Each re-trimmed leg is CLOSED (fee-adjusted proceeds) and RE-OPENED at its target
notional of CURRENT PV ⇒ at fee=0 PV is conserved, bankruptcy is impossible, and
`process_margin_call` is genuinely a no-op ON (kept as the canon/839 safety net, never deleted).
LIMIT orders keep the canon placement but are SIZED to one `w_unit` of PV so a breadth of
simultaneous fills can't re-introduce the margin cascade. All `jnp.where`/`jnp.clip`, no dynamic
shapes; the whole ON branch is DCE'd when OFF. Independent of / composes with `vector_budget_step`.

## Pre-registered reads (fixture, CPU-only, never `trading_data.db`)

**(i) 834-repro survival walk over the FULL panel — canon / budget / weight arms.** Zero-logit
dithered-book proposal (`hold_bias=0.0` ⇒ near-uniform per-asset actions), legality-masked
exactly as battery self-play (`batch_compute_action_mask`, `max_positions=16`), on the
DIVERSIFIED daily panel (279 assets). Read: **weight-ON must survive the FULL panel (≥1600
steps) with ZERO bankruptcies AND `final_pv > 0`** (controls: canon dies at step 0; budget dies
early). **HARD INVARIANT AUDIT (fee=0):** the probe checks EVERY step of the weight arm that
`cash ≥ 0` (so `process_margin_call` never fires) AND `Σ|w| ≤ 1` (no leverage) — the defining
guarantee, gated, not merely "no bankruptcy at the end".

**(ii) Sticky-short probe.** Constant Sell on 8 assets (Hold elsewhere), 500 steps, single env.
Read: **weight-ON keeps `PV > 50%` of initial** (the short is re-trimmed to a bounded fraction
of PV each step — no unbounded bleed).

**(iii) E1-pattern perfect-model probe, net@21 (fee16/slip5), weight-ON.** Floor = the 819
linear book (`corridor.book_target_signs` + `_to_action_vec`) replayed through the SAME
weight-ON step (apples-to-apples in weight semantics). Read: **net@21(E1) ≥ floor + 0.436**.
If the search cannot express profitable books in weight semantics, that is recorded as a
**FINDING** (verdict `FINDING`), NOT a silent pass. (i)+(ii) are the HARD gates; (iii) is
recorded but does NOT gate the exit code.

## Results (fixture-smoke, this module; JAX 0.6.2 — `results_fixture_smoke.json`)

Panel: 279 assets, 1704 daily rows, 2021-01-01 .. 2025-09-01. Real measured numbers:

**(i) Survival walk** (2 envs × 1697 steps = full panel):
| arm | bankrupt step / env | survived | min final PV |
|-----|---------------------|----------|--------------|
| canon  | [0, 0]        | 0/2 | 0.0 |
| budget | [306, 205]    | 0/2 | 0.0 |
| weight | [-1, -1]      | 2/2 | 63833.8 |

→ canon reproduces the env-wall (step 0); the 839 budget arm dies early (205/306); **weight-ON
survives the full 1697-step panel with ZERO bankruptcies and `min_final_pv = 63834 > 0`.**
HARD invariant audit (fee=0): `min_step_cash = 38642 ≥ 0`, `max_step Σ|w| = 0.5293 ≤ 1` ⇒
`process_margin_call` never fired. **(i) PASS.**

**(ii) Sticky short** (constant Sell on 8 assets, 500 steps):
`weight final_pv_frac = 1.001` (no bankruptcy) → **(ii) PASS** (> 50%). NOTE: on this fixture
window the canon control did NOT bankrupt (`final_pv_frac = 1.088`) — the 8 chosen assets did
not trend adversely enough to bleed the fixed-fraction canon short here, so the control is not
discriminating in this slice; the weight-ON pass stands on its own bounded-exposure read.

**(iii) E1 net@21, weight-ON** (K16, dithered-book, 60 decisions, pct=0.5/A):
`floor(819 book, weight-ON) = +5.6034`, `net21(E1) = +5.8384`, `edge = +0.2350` →
**FINDING** (read: `≥ floor + 0.436`). The perfect-model K16 search now BEATS the 819-book floor
in weight semantics, but not by the full pre-registered 0.436 edge in this fixture smoke.
Recorded per SPEC (not a silent pass, not a hard gate — (i)/(ii) are the hard gates). Whether
the search can express the required edge is the phase-1c scheduled-run question, not decided
here.

**Factory threading:** `create_jax_env_historical(..., vector_weight_rebalance=True/False)`
propagates to `EnvConfig` (checked ON/OFF; `propagates: true`).

## Reproduce

```bash
JAX_PLATFORMS=cpu python experiments/840_weight_rebalance/acceptance_probe.py \
    --out experiments/840_weight_rebalance/results_fixture_smoke.json
```

## Out of scope

Continuous weight actions (stays the discrete 5-action set — H6 granularity widening is a LATER
measurement), reward/proposal-scheme changes, removing the 839 budget path (it stays; the flags
are independent), phase-1c re-training (scheduled AFTER this ships).
