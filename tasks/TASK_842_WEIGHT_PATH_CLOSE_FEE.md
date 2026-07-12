> **STATUS (2026-07-12): SHIPPED as `task842-weight-close-fee`.** Canon-equivalent explicit
> close/open-short fee added to the weight re-trim (`close_fee=|close_proceeds|·fee_rate` = canon
> L788/L894; `open_short_fee=open_cost·fee_rate` = L905; no open-long fee = L813). **Three pins
> MATCH** (canon `48e1f467…`, 839-budget `4f8b692e…`, new weight-ON-fee0 `1c50d58c…`) — every term
> ×fee_rate ⇒ **fee0 bit-identical, phase-1c cost0 untouched**. The ~16bps UNDERcharge (D9) is
> FIXED. Flow-verifier CONFIRMED: weight ≥ canon per re-trim (never under). NOTE: the weight env
> full-closes+reopens each re-trim with a slippage-driven residual micro-short → it OVER-taxes a
> short round-trip vs canon (pre-existing 840 geometry, fee-independent, fee0-locked) → **DECISIONS
> D12** (planner; incremental-netting cleanup is a post-phase-1c task). NIT: the "within 1e-6"
> acceptance is float32-summation fragile (float64-exact; rel ~3e-7). Medium panel: correctness+
> regression session-limit-truncated (trending clean, orchestrator-substituted on objective gates);
> flow-verifier re-run COMPLETED clean. selftest 7/7, 44 tests pass.

# TASK_842 — Weight-path canon-equivalent close fee (D9 remainder; gates phase-2 fee16)

**Status**: READY 2026-07-11 (D9 ratified: phase-1c runs cost0 as-is; THIS ships before any
fee16 run touches the weight path).
**Size: S. Risk: medium** (touches the TASK_840 weight branch of `step_vector` only; canon +
839-budget + 840-weight-at-fee0 all stay bit-identical — three cheap pins, since fee=0 makes
the new term exactly 0).

## Why
Panel-quantified (TASK_840 run-7, → DECISIONS D9): the weight-path re-trim charges only the
exec-price spread (buy at ·(1+c), sell at /(1+c)) and NEVER the canon path's separate explicit
close-fee — an equivalent canon round-trip charges ~one fee tier (~16bps) MORE per closed
notional. At fee=0 the discrepancy is exactly 0 (phase-1c unaffected); at fee16 the weight env
would systematically under-tax turnover vs the corridor's cost model → phase-2 numbers would be
cost-dishonest and NOT comparable to the 830 corridor / engine ref.

## Scope
1. On the weight-rebalance branch, charge the canon-equivalent explicit close-fee on every
   re-trim leg that CLOSES notional (full or partial): fee = closed_notional × fee_bps/1e4,
   deducted from cash at the same point in the step the canon path deducts it (mirror
   `execute_sell`/`execute_buy` close-fee placement — cite lines in the SPEC).
2. Parity: fee=0 weight path BIT-IDENTICAL to pre-edit (the term is ×0 — pin it); canon and
   839-budget paths untouched (pins).
3. **Acceptance (fixture)**: a single open→close round-trip at fee16 on the weight path now
   charges within 1e-6 of the equivalent canon scalar round-trip's total fees (the panel's own
   equivalence construction); the 840 acceptance walks re-run green at fee0 AND fee16 (fee16
   walk: no invariant violations, fees strictly reduce PV vs fee0 — sign check).
4. Thread nothing new: the flag surface is unchanged (this is a correctness fix inside the ON
   branch, not a new mode).

## Gates
Three parity pins MATCH; acceptance equivalence + sign checks green; selftest 7/7; medium
panel (single-branch arithmetic change, but cost-honesty is load-bearing for every future
fee16 number).

## Out of scope
Maker/taker fee split for the limit legs (record as a NOTE if encountered), deadband retuning,
any phase-1c interference (phase-1c trains at fee0 — this change is a provable no-op there;
do NOT redeploy over a running battery, the planner controls deploy timing).
