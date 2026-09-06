> **STATUS (2026-09-06): PHASE A SHIPPED as `task919a-execution-turnover-grid`.** Band question
> **REFUSED FOR WANT OF POWER** (MDE 5.8–9.1× the ±0.20 band; 0/48 clear at α/48 on three windows)
> → **D85**. Hold-shares advantage **UNRESOLVED** (construction-dependent; neither construction
> clears zero). Surviving lead: **θ hysteresis = 50% of the candidate's uplift**, surviving
> de-levering. Four record corrections (incl. this file's own BEAR22 anchor, quoted backwards).
> **REMAINDERS:** phase B untouched (double-gated on D81 + D80(iii)); hour-of-day arm is
> mechanism-only, real leg SCHEDULED (planner); four follow-ups rowed in `RESULTS.md`.
> Full record: `experiments/919_execution_turnover/RESULTS.md` · 3 cycles + a disclosed D76 closing pass.

# TASK_919 — execution/turnover module (the bankable +0.10–0.15)

**Status**: PROPOSED 2026-09-05 (planner; = tactics-panel row 2 upgraded with the dead-panel
execution details in `lore/harvest_6b32_appendix/{tactics_gaps,loops_gaps}.md`).
**Size S. Risk: LOW (new module + read-only panels). CPU only.**

## Odds, stated honestly up front

0–2% of clearing the goalposts ALONE (cost reduction cannot manufacture sign); **60–70% of
adding +0.10–0.15 net Sharpe to any book that survives 915/918**. It is the only item in the
whole harvest with majority odds of a positive, additive effect.

## Phase A — the rebalance grid (no dependencies beyond the panel)

On the PIT 273-asset true-daily panel, frozen 819 trend book (and, when 918 ships, the best
library arm): **H ∈ {1,2,3,5} d × hysteresis θ ∈ {0, 0.25, 0.5, 1}·σ̂ × accounting
{target-weight, drift-inclusive, hold-shares} × fee tiers from 916 (incl. borrow)**, scored
net at 8/12/21 bp + borrow, Bonferroni over cells. Print the flip/renormalisation/drift
turnover decomposition (analytic zero-crossing rate for verification: 0.165/0.117/0.0957/0.0746
flips/day at H = 1/2/3/5; θ = 0.5σ̂ kills 40–50% of flips; per-bp slope 0.0085–0.0093).
Known anchors to re-verify, not re-trust: H=3d survivor +1.35 vs +0.97 (n=1, ~0.07 of it
cost), hold-shares BEAR22 gross +1.218 vs +0.225 (`830_corridor/SPEC.md:95-98`).
**Extra arm, zero signal cost: decision boundary moved to 13–17 UTC** (hour-of-day axis never
measured). Composed ceiling on record: H=3 + 12 bp maker + θ → PIT ≈ +0.70 (pre-restatement;
re-derive under 916).

**Pre-registered band: max uplift over the frozen rule ≤ +0.20 net@21 ⇒ the family is capped
as cost-only and FOLDS INTO any surviving book — no further standalone rows.**

## Phase B — maker execution spec (GATED twice)

Post-at-touch with 24 h deadline, cross the remainder; claim maker fees only where 907-B2's
shipped estimand supports them. **Gate 1: TASK_907 B2 shipped (needs median AS(1h) ≤ 2 bp on
the majors, else withdraw the maker claim — 907's own decision gate). Gate 2: D80 rules on the
rails conflict** — mandatory market-cross and 15–60-min rest windows are FORBIDDEN in current
code (`experiments/907_live_fill_probe/phase2_rails.py:170-178`, per D65/D78); a maker spec
that must cross after a timeout needs an explicit operator ruling before any live-adjacent
test. Facts to carry: 4 majors ≈ 1.5% of turnover; a "limit then market after 60 s" baseline
already exists (`real_trade_config.py:20,23`); under a martingale mid the paired saving of
post-at-touch over taker is p·(spread + Δfee) — a lookup, not a learner (loops panel).

## Out of scope

Learned execution (refuted 2/2 — lookup suffices); anything touching the live capture or
gpu-server processes (standing rule); the env's `borrow_rate_annual` wiring (916-adjacent, only
if a book survives).
