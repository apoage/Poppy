> **STATUS (2026-07-17): SHIPPED as `task846-weight-fee-headroom`.** D14 opt (a): the weight
> budget now reserves turnover-fee headroom (`budget × (1 − 2·fee_rate)`, internal `fee_headroom_k=2`,
> no new flag) — closes the all-short fee16 full-wipe. k=2 covers the full-flip worst case; every
> term ×fee_rate ⇒ **fee0 BIT-IDENTICAL** (all 3 pins MATCH incl. `6293668f…`). Acceptance (i)–(iii)
> PASS (D14 repro cash≥0/PV>0 at fee8/16/32, 843 walks green); **selftest 7/7; 36 passed**; 5 new D14
> regression tests (FAIL pre-fix / PASS post-fix). **Acceptance (iv) → DECISIONS D16 (RULED: ship
> as-is).** The k=2 shrink materially shifts the 832/845 teacher-gate on the **8-asset fixture**
> (`teacher_weight_net21` −4.687→−3.111, Δ1.576) — a **benign 8-asset binding-regime chaos artifact**
> (budget-binding scaling through deadband/snap nonlinearity; no bankruptcy; moves net@21 UP), NOT a
> real-surface distortion: the binding **64-asset real-panel** read is **Δ=0.0008** (25× under the 0.02
> bar; `runs/832_e5/gate_read_post846.json`, gate PASS rel 0.042, zero bankruptcies, host-determinism
> control clean). Refinement (a′ conditional headroom) REJECTED as unsound (band hole). Un-gates the
> E5 ExIt continuation + all phase-2 fee16. (Orchestrator note: a first transient 5-test failure was
> GPU contention from racing the still-running implementer on the single 3090 — cleared serially.)

# TASK_846 — Weight-budget fee headroom (D14 opt (a); gates ExIt + all phase-2 fee16)

**Status**: READY (re-seeded) 2026-07-17 — **D16 RULED: SHIP the existing implementation AS-IS.**
Run-11 held the ship on acceptance (iv) (fixture teacher-gate Δ1.576) → D16. The binding
64-asset real-panel distortion read (`runs/832_e5/gate_read_post846.json`, patch from the
working tree) measured **Δ=0.0008 < 0.02** with the expressibility gate still passing and a
clean host-determinism control ⇒ acceptance (iv) is SATISFIED on the binding surface; the
fixture Δ is a KNOWN-BENIGN 8-asset binding-regime chaos artifact (tiny budget shrink flips
discrete deadband/snap decisions on a hard-binding walk — record, don't gate). Wrap = commit
the gate-green implementation + 5 D14 regression tests already in the working tree, update
this banner + CHANGELOG. **No code changes requested.** (a′ "only-when-binding" was REJECTED
as unsound — see D16: it reopens a narrow D14 wipe band.)
**Size: S. Risk: medium** (TASK_840/843 weight branch of `step_vector` only; canon `48e1f467…`
+ 839-budget `4f8b692e…` pins unchanged; weight-ON/fee0 pin `6293668f…` must ALSO stay
unchanged — the headroom term must be exactly 0 at fee=0, the 842 ×fee_rate discipline).

## Why
D14 (LEDGER-adjacent, TASK_843 fail-closed panel, stash-confirmed PRE-EXISTING at 842): the
weight budget reserves capital FOOTPRINT but not FEES, so at fee>0 a near-fully-invested
all-short book (Σ|w|→1) overdraws by ~the total open-turnover fee and `process_margin_call`
escalates the all-short deficit (no longs to liquidate) into a spurious FULL-BOOK wipe.
Repro: 64 assets, w_unit=1/64, w_max=1, all-Sell, flat prices, fee16 → PV 100000→0.
Phase-1c (cost0) never triggered it; the ExIt continuation and ALL phase-2 runs are fee16
(DEC-2) — a trained policy that ever forms a broad short book self-destructs there.

## Scope
1. Shrink the admitted per-step weight budget by the expected turnover-fee fraction:
   effective_budget = budget × (1 − k·fee_rate) with k covering open+close legs of the step's
   traded notional (derive k from the 842 fee placement — cite lines in the SPEC; every new
   term ×fee_rate so fee0 is bit-identical BY CONSTRUCTION).
2. **Acceptance (fixture)**: (i) the D14 repro (all-Sell, w_max=1, flat prices, fee16) ends
   with cash ≥ 0, zero margin-call escalations, PV > 0; same at fee8 and fee32 (scaling
   check); (ii) fee0 weight path BIT-IDENTICAL (pin `6293668f…` MATCH); (iii) the 843
   acceptance walks re-run green at fee16 (no invariant violations); (iv) expressibility
   side-effect check: the teacher-gate net@21 at fee16 moves by < 0.02 (headroom must not
   materially distort the teacher replay — if it does, STOP and report, fallbacks (b)/(c)
   are decision-owned).
3. Thread nothing new: internal to the ON branch (no new flags beyond an internal constant;
   if a knob is unavoidable, `vector_fee_headroom_k` default = derived k).

## Gates
Pins: canon + budget + weight-ON/fee0 all MATCH; acceptance (i)–(iv); selftest 7/7; medium
panel (fail-closed lens mandatory — this exists because a fail-closed judge found D14).

## Out of scope
Margin-call redesign (fallback (b) is decision-owned), short-caps (fallback (c)), retuning
w_unit/w_max (that is D15's lane), deploying to gpu-server while any battery runs.
