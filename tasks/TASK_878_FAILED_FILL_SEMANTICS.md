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

**AMENDMENT (operator, same conversation — supersedes a deterministic no-fill):** an order at a
no-print bar is **not dead, it is UNFILLED** — it can still execute on a LATER step. The faithful
model is a pending order that stays live for ~5 steps and fills at the first bar that prints
("heavy solution"); for simulation the operator's choice is a **coinflip on execution**. Operator's
open caveat, recorded as a hypothesis and NOT assumed either way: *"not sure if that helps or hurts
the model, since there are market patterns that are flat before a big move .. but not super flat"*
— i.e. no-print/flat cells may carry real signal (pre-breakout compression), so suppressing them
could destroy information rather than remove noise. **That is measurable, so it gets measured
(scope 6) instead of argued.**

**UNIFIED SEMANTICS (the target state):**
1. **Valuation** uses the last valid price (the D30 forward-fill). A stale price is a *reasonable
   quote* for a period in which nothing traded — it is REAL for marking a position. Unchanged;
   this is architecture (the panel must be dense for the JAX kernels).
2. **Execution is PROBABILISTIC on a no-print bar**: the order is placed and fills with
   probability `p` (coinflip); on a miss, position/cash/fee are UNCHANGED and the attempt is
   recorded. A per-step coinflip is the geometric special case of the pending-order queue, so the
   heavy variant remains a strict upgrade path, not a redesign. Where the bar DOES print,
   execution is unchanged and deterministic.
3. The **policy's intent is preserved**. This is the part today's implementation gets wrong: 868
   (a2) makes the action *illegal* (the legal mask is restricted to HOLD), so search never
   proposes it and — critically — the **BC target teaches "HOLD was intended"** when the truth is
   "BUY was intended and could not fill". Under this ruling the decision is recorded as the
   attempt, and only the OUTCOME is null.
4. The same rule answers the stride question: a sampled minute with no print is not a data defect,
   it is a bar in which no trade was possible. It marks fine and it does not fill.

## Scope

1. Replace the open-gate's "restrict the legal mask to HOLD" with a **probabilistic fill gate at
   execution**: the action is legal and recorded; on a no-print bar the fill succeeds with
   probability `p` and otherwise leaves position/cash/fee untouched. Force-HOLD stays only where
   it is genuinely correct (a slot with NO data at all, i.e. inactive/zero-coverage — nothing can
   be quoted, so nothing can be attempted).
1b. **`p` must come from the data, not from a magic number.** Default: the slot's trailing print
   density (fraction of recent bars that printed), computed CAUSALLY from data up to the decision
   bar — a liquid-but-momentarily-quiet slot then fills easily and a near-dead slot rarely does,
   which is the behaviour the coinflip is meant to approximate. A constant `--fill-prob` stays
   available for ablations. Whatever is used is RECORDED in the artifact.
1c. **Determinism is not optional — the whole measurement stack depends on it.** The coinflip must
   be a seeded, path-independent hash of `(eval_seed, decision_index, slot)`, never a sequential
   RNG draw: with a sequential stream, one changed decision reshuffles every later flip and the
   repo's parity/bit-identity gates (861 e2e, 863/865 suites, the 864 promote baselines) all die.
   Same seed + same panel ⇒ byte-identical artifacts, demonstrated.
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
6. **Answer the operator's help-or-hurt question by measurement — a 3-arm ablation, same
   checkpoint, same windows, one artifact each:** (A) **never fill** on a no-print bar
   (`--fill-prob 0`, the deterministic reading this task originally had), (B) **coinflip** at the
   data-derived `p` (the ruled default), (C) **always fill** (`--fill-prob 1` — the pre-868
   behaviour, i.e. trade freely at the stale quote). Report net@21, fill-attempt counts, and the
   share of decisions touched, per arm, ≥3 seeds where the arm is stochastic. This is what decides
   whether flat/no-print cells carry signal (arm C ≥ arm A) or noise (arm A ≥ arm C) — the
   "flat before a big move" hypothesis is then a measured effect rather than a belief. It also
   bounds how much any of this matters: today only ~0.94% of cells are affected, less after
   TASK_877, so a large swing between arms would itself be a finding worth stopping on.

## Acceptance (gates)

- (i) A no-print bar: order attempted; on a coinflip MISS **no** position/cash/fee change, on a
  HIT a normal fill at that bar's marked price; the decision is recorded as the attempted action
  (never rewritten to HOLD). Demonstrated on the fixture with a hand-checked cell of each kind.
- (ia) **Reproducibility**: same `eval_seed` + same panel ⇒ byte-identical artifacts across runs;
  and changing ONE earlier decision must not alter any later cell's flip (proves the hash is
  path-independent, not a sequential stream).
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
it reduces how many cells this rule touches); **the pending-order QUEUE** (the operator's "heavy
solution": an order staying live ~5 steps and filling at the first printing bar) — the per-step
coinflip is its geometric approximation and a strict special case, so the queue is a future upgrade
with its own row, and this task must not half-build it; slippage and PARTIAL fills of any kind (a
fill here is all-or-nothing).
