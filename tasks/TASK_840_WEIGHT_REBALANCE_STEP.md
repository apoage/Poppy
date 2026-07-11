> **STATUS (2026-07-11): SHIPPED as `task840-weight-rebalance-step`.** Flag-gated
> `--vector-weight-rebalance` default-OFF (canon + 839-budget both bit-identical, pins
> `48e1f467…` / `4f8b692e…`). Full high-tier panel: 1 BLOCKER (winning-short budget
> under-reservation, crossval-CONFIRMED) + 1 flow-verifier-CONFIRMED held-leg drift bleed —
> BOTH FIXED (whole-book posval/PV reservation + cap-clip re-trims every drifted leg → cash≥0,
> Σ|w|≤1, margin_call a no-op ON). Acceptance (i) survival PASS (1697 steps, invariant gate
> cash≥0/Σ|w|≤1/margin-noop), (ii) sticky-short PASS; **(iii) E1 edge +0.235 (flipped from −0.23
> but below the +0.436 bar) = FINDING, non-gating** per item 6(iii). Note: the "acceptance
> (i)–(iii) green" line below is superseded — (iii) is a recorded FINDING, not a hard gate.
> Residuals → **DECISIONS D9** (weight-path fee undercharge, gates fee16 phase-2) + **D10**
> (weight-semantics expressibility). Retrain = scheduled phase-1c.

# TASK_840 — Target-weight rebalance step (D8 escalation, opt (c) — the H6 env)

**Status**: READY 2026-07-11 — the pre-registered D8 escalation FIRED (LEDGER 835/S4-p1b).
**Size: L. Risk: high** (new step semantics in `mcts_trader/env/jax_trading_env.py`, flag-gated;
canon + 839 paths both stay bit-identical — 838/839-pattern parity discipline).
**Pathway**: R2/R3 unblock. **This is the original 828 thesis** (action = per-asset weight
vector; EV = Σ64 edges) — Stage 1 shipped incremental-order semantics instead; 839 fixed the
step-0 wall; this fixes the remaining kill mechanism.

## Why (evidence chain, all 2026-07-11)
- **835/S4-p1b**: budget-step-trained policies STILL bankrupt (s1 step 132 with 10–11
  concurrent unhedged shorts; s2 step 181). The budget only binds on wide vectors; sparse books
  still size 10%×PV/trade, accumulate monotonically, and shorts bleed until the margin cascade
  wipes PV. Even wide budget-bound walks die by step ~104 (836 walk, PV→0).
- **834/E3-p1b: DEC-1 PASS ×2** (R_model 1.30/1.12 vs bars 0.33/0.24) and **836/E4-p1b: value
  beats momentum ×2** (ρ +0.21/+0.30 vs −0.03) — **the nets learn; the env kills.** Incremental-
  order semantics is the demonstrated residual killer, pre-registered as the opt (c) trigger.

## Scope
1. Flag-gated `vector_weight_rebalance` (config + `--vector-weight-rebalance`, default OFF;
   OFF leaves BOTH the canon path and the 839 budget path bit-identical — two pins).
2. ON: the per-asset action {0=Sell,1=Hold,2=Buy,3=LimitSell,4=LimitBuy} maps to a TARGET
   weight per asset — pre-registered ladder: Sell → −w_unit, Hold → current weight (drift),
   Buy → +w_unit, LimitSell/LimitBuy → −w_unit/+w_unit at limit price (order placed, target
   applies on fill), with `w_unit = 1/n_assets` default (config `vector_weight_unit`).
   Repeated Buy on an already-long asset steps the target up one unit (cap
   `vector_weight_max`, default 4×w_unit); Sell on a long steps down/through zero to short
   (symmetric). **Each step the env re-trims every position toward its target weight of
   CURRENT PV** — the book is a bounded allocation by construction (Σ|w| ≤ 1 enforced by
   proportional scale-down when exceeded; no leverage), so with fee=0 PV can never reach 0:
   bankruptcy becomes impossible rather than merely less likely. Shorts get re-trimmed every
   step — bounded per-step loss, no unbounded bleed.
3. Turnover discipline: re-trim only when |current − target| > deadband (config
   `vector_rebalance_deadband`, default 0.1×w_unit) — otherwise fee tiers make the weight env
   a turnover furnace.
4. All JAX-pure (`jnp.where`, no dynamic shapes, no cond-in-vmap); margin-call path untouched
   (should become unreachable ON — assert-guard it, don't delete it).
5. Thread the flag through training config/cli/env_factory/`train_jax_fast.py` + 835 + 834/836
   walks (operator symmetry — the 838/839 precedent).
6. **Acceptance (fixture panel, pre-registered)**: (i) the 834-repro zero-logit walk survives
   the FULL panel (1600+ steps) with ZERO bankruptcies AND final_pv > 0 under ON (control: it
   dies ~step 104 under 839-ON, step 0 under OFF); (ii) a sticky-short probe (constant Sell on
   8 assets, 500 steps) keeps PV > 50% initial (control: current step bankrupts it); (iii)
   E1-pattern perfect-model probe ON: net@21 ≥ floor +0.436 (the search must still express
   profitable books in weight semantics — if it can't, that is a FINDING to record, not a
   silent fail).

## Gates
Both OFF-parity pins MATCH; new tests alongside existing (do not modify); acceptance (i)–(iii)
green; selftest 7/7; full high-tier panel (canon-adjacent, new semantics).

## Out of scope
Continuous weight actions (stays the discrete 5-action set — H6 granularity widening is a
LATER measurement), reward changes, proposal-scheme changes, retraining (phase-1c = scheduled
run after ship), removing the 839 budget path (it stays; the flags are independent).
