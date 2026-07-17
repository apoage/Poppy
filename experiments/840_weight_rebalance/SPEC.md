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

## TASK_846 — weight-budget FEE headroom (D14 opt (a); appended 2026-07-16)

**Decision**: `DECISIONS.md` D14 — RATIFIED opt (a). Same discipline (flag-gated, canon +
839-budget bit-identical, weight-ON/**fee0** bit-identical — pin `6293668f…`).

**Why.** The 840/843 budget above reserves capital FOOTPRINT but not turnover FEES. The 842
re-trim charges, on the CLOSED/OPENED delta only, two explicit fee terms (both `×fee_rate`, so
0 at fee0):
* **close leg** — `close_fee = |close_proceeds| * fee_rate` at `jax_trading_env.py:1729`
  (mirrors `execute_sell` L894 / `execute_buy` L788);
* **open-short leg** — `open_short_fee = (want_short ? open_notional) * fee_rate` at
  `jax_trading_env.py:1731` (mirrors `execute_sell` L905).

Neither is reserved in `budget` (`jax_trading_env.py:1660`), so at fee>0 a near-fully-invested
book (Σ|w|→1) overdraws by ~the unreserved turnover fee and `process_margin_call` escalates the
all-short deficit (no longs to liquidate) into a spurious FULL-BOOK wipe (D14 repro: 64 assets,
w_unit=1/64, w_max=1, all-Sell, flat prices, fee16 → PV 100000→0). Pre-existing at 842
(`project_vector_env_cash_wall`), NOT a 843 regression.

**Fix (`jax_trading_env.py:1660`, internal constant — no new flag).**
`effective_budget = max(1−held_reserve,0) · (1−1e-4) · (1 − k·fee_rate)`, `k = 2`.

**k derivation.** The worst case is a FULL flip re-trim (close the long book + open the short
book) on ≈ the whole admitted budget, which pays BOTH 842 legs. Writing `weight_cash(fee)` for
that step (snap_cash≈0, close_proceeds≈PV, open_cost≈eff_budget·PV, plus the two fee legs) and
substituting `eff_budget ≈ 1 − k·fee_rate` gives
`weight_cash(fee)/PV ≈ fee_rate·(−2 + k + k·fee_rate)`, so `weight_cash ≥ 0 ⇔ k ≥ 2/(1+fee_rate)`.
The all-short-from-flat D14 repro (open leg only) needs only `k ≥ 1/(1+fee_rate)`. `k = 2` is the
tight safe integer covering **both** legs (`2/(1+16bp) = 1.9968`). Every added term is `×fee_rate`,
so at fee0 `(1 − k·fee_rate) = 1.0` exactly ⇒ the whole weight-ON/fee0 path is **bit-identical**
(pin `6293668f…` MATCH). Composes with (does not remove) the 1e-4 haircut and the 843 `foot_mult`
footprint reservation.

**Acceptance (this fix, fixture; regression tests in `tools/tests/test_weight_rebalance.py`).**
(i) D14 all-Sell/w_max=1/flat repro at fee8/fee16/fee32 ends `cash≥0`, PV>0, full 64-short book
intact, `process_margin_call` a genuine no-op (was PV→0 at k=0/pre-fix — the tests FAIL pre-fix,
PASS post-fix). Measured post-fix (single step, na=64, slip0):
`fee0 PV=100000.00 cash=10.00` · `fee8 PV=99760.50 cash=90.08` · `fee16 PV=99522.08 cash=170.50`
· `fee32 PV=99048.25 cash=332.01` (all Σ|w|≤1.003, full book, margin no-op).
(ii) fee0 weight path **bit-identical** (pin `6293668f…`); survival walk (fee0) and 819-floor
(replay) numbers identical before/after (see Results).
(iii) 843 acceptance walks re-run green at fee16.

**(iv) FINDING — E1 teacher-search net@21 exceeds the 0.02 side-effect budget.** Same-tree
before/after (k=0 vs k=2), fixture E1 probe (21bp = fee16/slip5, K16, 60 decisions,
deterministic — reproduced exactly across runs):
* `net21_floor_819book` (the 819-book **replay**): **+5.575750 → +5.575750 (Δ 0.000000)** — the
  headroom never bites the modest-breadth 819 book (`reb_abs < budget`), so the teacher REPLAY is
  unchanged;
* `net21_e1` (the E1 perfect-model **search**): **+7.742424 → +7.792593 (Δ +0.050169)** — the
  0.32%-tighter admitted budget shifts realized PVs → MCTS value backups → a few of the 60 action
  selections, nudging the net UP (edge over floor 2.167 → 2.217; verdict PASS both, well over the
  0.436 required edge; e1_final_pv 112097.5 → 111972.2, −125.4).

The move is +0.0502 > the task's 0.02 budget on `net@21`. Per TASK_846 §2(iv) this is recorded
as a STOP-and-report FINDING for the decision-owner (fallbacks (b) margin-call buyback / (c)
short-cap are decision-owned, out of scope for this task). It is NOT a hard-gate failure: the
819-book replay is untouched, the E1 verdict stays PASS by a wide margin, and the move is a
benign IMPROVEMENT of the search under a correctly-tighter budget, not a replay distortion — but
it does exceed the pre-registered absolute tolerance, so it is surfaced rather than silently
passed. `k` cannot be lowered without breaking the flip-case `k ≥ 2/(1+fee_rate)` guarantee.

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

**(i) Survival walk** (2 envs × 1697 steps = full panel; refreshed on the 846 tree — the prior
table predated the 843 fee0 rebase, hence the different weight-arm PV):
| arm | bankrupt step / env | survived | min final PV |
|-----|---------------------|----------|--------------|
| canon  | [0, 0]        | 0/2 | 0.0 |
| budget | [306, 205]    | 0/2 | 0.0 |
| weight | [-1, -1]      | 2/2 | 43230.8 |

→ canon reproduces the env-wall (step 0); the 839 budget arm dies early (205/306); **weight-ON
survives the full 1697-step panel with ZERO bankruptcies and `min_final_pv = 43231 > 0`.**
HARD invariant audit (fee=0): `min_step_cash = 17680 ≥ 0`, `max_step Σ|w| = 0.7351 ≤ 1` ⇒
`process_margin_call` never fired. **(i) PASS.** (Identical with the 846 headroom OFF vs ON —
the walk is fee0, so the `×fee_rate` headroom is exactly 0.)

**(ii) Sticky short** (constant Sell on 8 assets, 500 steps):
`weight final_pv_frac = 1.001` (no bankruptcy) → **(ii) PASS** (> 50%). NOTE: on this fixture
window the canon control did NOT bankrupt (`final_pv_frac = 1.088`) — the 8 chosen assets did
not trend adversely enough to bleed the fixed-fraction canon short here, so the control is not
discriminating in this slice; the weight-ON pass stands on its own bounded-exposure read.

**(iii) E1 net@21, weight-ON** (K16, dithered-book, 60 decisions, pct=1/A; refreshed on the 846
tree, 21bp = fee16/slip5):
`floor(819 book, weight-ON) = +5.5758`, `net21(E1) = +7.7926`, `edge = +2.2168` → **PASS**
(read: `≥ floor + 0.436`). The perfect-model K16 search now beats the 819-book floor by well over
the pre-registered 0.436 edge in this fixture smoke. NOTE (846 side-effect, see the TASK_846
FINDING above): with the fee headroom OFF this reads `net21(E1) = +7.7424` / `edge = +2.1667`; the
headroom nudges `net21(E1)` by +0.0502 (the 819 floor is unchanged). (i)/(ii) remain the hard
gates; (iii) is recorded, not gating.

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
