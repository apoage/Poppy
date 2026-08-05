# TASK_878 — No-print bars are FAILED FILLS, not illegal actions (operator ruling, 2026-08-05)

**Status**: READY 2026-08-05 (operator ruling; resolves both QUEUE open questions at once).
**Size: S/M. Risk: medium** (changes what a no-print bar MEANS in the env — every minute-panel
numeric may move, by design and disclosed). Surface: the frozen open-gate in
`experiments/835_stage4_eval/eval_vector.py` (868 a2) + the env's `price_valid` mechanic
(`mcts_trader/env/jax_trading_env.py`) + BC target construction where it consumes the mask.

## The ruling (operator, verbatim in substance)

> "in reality if you trade on that timestamp where there is no price — you would place an order
> and the order would fail to fulfil, because the given price is reasonable for the given time."

So a bar with no print is **not** an illegal decision and **not** a fabricated trade. It is an
**attempted order that does not fill**. Both QUEUE open questions collapse into this single rule,
because they are the same problem — "define what is real, and the consequences":

**UNIFIED SEMANTICS (the target state):**
1. **Valuation** uses the last valid price (the D30 forward-fill). A stale price is a *reasonable
   quote* for a period in which nothing traded — it is REAL for marking a position. Unchanged;
   this is architecture (the panel must be dense for the JAX kernels).
2. **Execution requires a real print inside the bar.** No print ⇒ the order is placed and
   **fails to fill** ⇒ position state is UNCHANGED, cash is UNCHANGED, no fee is charged.
3. The **policy's intent is preserved**. This is the part today's implementation gets wrong: 868
   (a2) makes the action *illegal* (the legal mask is restricted to HOLD), so search never
   proposes it and — critically — the **BC target teaches "HOLD was intended"** when the truth is
   "BUY was intended and could not fill". Under this ruling the decision is recorded as the
   attempt, and only the OUTCOME is null.
4. The same rule answers the stride question: a sampled minute with no print is not a data defect,
   it is a bar in which no trade was possible. It marks fine and it does not fill.

## Scope

1. Replace the open-gate's "restrict the legal mask to HOLD" with a **fill gate at execution**:
   the action is legal and recorded; the env applies no position/cash/fee change for a slot with
   no print in the bar. Force-HOLD stays only where it is genuinely correct (a slot with NO data
   at all, i.e. inactive/zero-coverage — nothing can be quoted, so nothing can be attempted).
2. **BC targets / teacher collection must record the ATTEMPT**, not the masked substitute. Audit
   every consumer of the mask (imprint target stream, chosen-vector collection, sanity replay,
   agreement metrics) and state for each whether it now sees intent or outcome. Where a metric
   must count outcomes (net@21, PV) it uses the fill result; where it measures policy behaviour
   (agreement, action histogram) it uses the attempt.
3. **Disclose fills-that-failed** as a first-class artifact field: count, per-slot breakdown, and
   the share of decisions affected — the same disclosure discipline the frozen spans got.
4. **PV conservation must still hold exactly** (the D30 first-class smoke leg): a failed fill
   changes nothing, so flat prices at fee 0 must still end at exactly 100,000.00 both sides.
5. Re-derive every moved numeric with per-field attribution; the 864 promote gate is the write
   path for tracked baselines.

## Acceptance (gates)

- (i) A no-print bar: order attempted, **no** position/cash/fee change, decision recorded as the
  attempted action (not HOLD). Demonstrated on the fixture with a hand-checked cell.
- (ii) BC targets from a collection over no-print cells carry the ATTEMPT; a regression test
  pins that they are no longer silently rewritten to HOLD (this is the 78.8%-class hazard —
  targets that differ from trajectory must be caught, not blessed).
- (iii) PV conservation exact both books, mask/fill path ON.
- (iv) Zero-coverage slots still force-HOLD (nothing quotable ⇒ nothing attemptable) and stay
  disclosed as inactive.
- (v) Failed-fill disclosure present and asserted; all suites green; every moved numeric
  attributed.

## Out of scope

The forward-fill itself (architecture, ruled); TASK_877's alias fold (do that first if scheduled —
it reduces how many cells this rule touches); slippage/partial-fill modelling of any kind (a
failed fill here is all-or-nothing: no print, no fill).
