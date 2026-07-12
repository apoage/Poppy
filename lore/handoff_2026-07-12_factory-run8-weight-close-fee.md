# Handoff 2026-07-12 (#10) — Factory run 8: TASK_842 (weight-path canon-equivalent close fee)

Previous: `lore/handoff_2026-07-11_factory-run7-weight-env-and-watchdog.md` (#9)
Branch: `master` · Working dir: `/home/lukas/Trader` · Role: ORCHESTRATOR
Nothing pushed.

## What shipped

| Tag | Commit | Task | Panel |
|-----|--------|------|-------|
| `task842-weight-close-fee` | `30013a4` | **TASK_842** weight-path canon-equivalent close fee (D9 remainder) — fee0-locked | MEDIUM. correctness+regression session-limit-truncated (trending clean, orchestrator-substituted on objective gates); flow-verifier re-run COMPLETED + CONFIRMED. 0 BLOCKER/MAJOR vs 842 scope; cost-model question → D12 |

## The change
The TASK_840 weight re-trim charged only the exec-price spread and undercharged ~16bps/close vs canon (panel finding → D9). Added the canon-equivalent explicit fee: `close_fee = |close_proceeds|·fee_rate` (canon `execute_buy` L788 / `execute_sell` L894), `open_short_fee = open_cost·fee_rate` (L905), no open-long fee (L813). Folded into `reb_cash_delta`. Every term ×fee_rate ⇒ **exactly 0 at fee0**.

**Fee0-locked = phase-1c-safe.** THIRD parity pin captured by the orchestrator on the clean pre-edit tree (ba53294, JAX 0.6.2): weight-ON/fee0 `1c50d58c570d10cfe237e4b7c630cd01058a7b5a68766eb0001a14e301474c20` (`tools/tests/test_vector_weight_fee0_parity.py`). MATCHES after ⇒ the RUNNING phase-1c cost0 battery is provably untouched. Canon `48e1f467…` + 839-budget `4f8b692e…` pins also MATCH.

## The session-limit incident + how it was handled
The first medium panel (3 agents) all died mid-run on the session token limit (reset 2am Prague). Partials: correctness "all tests pass", regression "fee0 block unchanged, only the fee16 e1 block moved — expected" (both trending clean), and the **flow-verifier flagged a SHORT round-trip fee gap + residual micro-short** before dying. I reproduced it myself (flat-price short round-trip: weight 185 vs canon 105, residual short size 0.76) and diagnosed it; then, after the limit reset, a FRESH flow-verifier COMPLETED and **CONFIRMED** all three: (a) NO undercharge — weight ≥ canon per re-trim (short RT 260.7 vs canon 237.0, an OVER-charge); (b) fee arithmetic mirrors canon on every leg (float64-exact); (c) the gap is pre-existing TASK_840 close+reopen + market-notional-`w_cur` residual-micro-short geometry, **fee-independent** (present at fee0/slip5 where the 842 term is ×0), NOT a 842 defect. Green rests on the objective gates + my line-by-line read + the completed flow-verifier; the two truncated members were trending clean and died on rate-limit, not findings.

## Findings → planner
- **D12** (seeded, OPEN) — the weight env full-closes+reopens each re-trim (not canon-style incremental) and `w_cur` uses market notional, so slippage leaves a residual micro-short and the env OVER-taxes short turnover vs canon. This is an over-charge (D9's undercharge is fixed), fee0-locked (can't change while phase-1c runs). Rec: accept for phase-2, defer incremental-netting + snap-to-flat cleanup to a post-phase-1c task.
- NIT: the spec's "within 1e-6" round-trip equivalence is float32-summation-order fragile (`snap_cash+jnp.sum(vec)` vs canon scalar add → ~3e-7 relative at larger notional; float64-exact). Read as "canon-faithful to float32 precision".
- The TASK_840 acceptance (iii) E1 fee16 edge honestly moved +0.235 → −0.134 (the +0.235 was flattered by exactly the undercharge this task fixed); still a non-gating FINDING (D10 already ratified: proceed to phase-1c).

## Git state
```
30013a4 TASK_842: weight-path canon-equivalent close fee (D9 remainder), fee0-locked   [tag: task842-weight-close-fee]
ba53294 D9/D10/D11 ratified; TASK_842 seeded (gates fee16); phase-1c pre-registered (SPEC amendment 4) + manifest   [planner]
99838ae Handoff 2026-07-11 (#9): factory run 7 …
31a545f TASK_840 …  ·  52b155f TASK_841 …
```
Working tree: clean after the QUEUE + this handoff commit. Factory: DISARMED. Nothing pushed.

## Next action
Factory idle — Ready EMPTY. Planner-owned: phase-1c is in flight (weight env, cost0, `--stall-timeout 2700` armed per D11); phase-2 fee16 is now UN-GATED (842 shipped) and re-targets phase-1c ckpts. Watch: D12 (weight cost-model for phase-2), the 841 wedge root-cause follow-up (apply a flag-gated trainer save fix once `--debug-stall-dump` captures a real wedge stack). Resume the factory with "armed". Note for next session: the subagent session limit resets 2am Europe/Prague — a large panel late at night can be truncated; the orchestrator's independent gate re-runs are the backstop.
