# TASK_878 — No-print bars are FAILED FILLS, not illegal actions (operator ruling, 2026-08-05)

> **STATUS (2026-08-17): SHIPPED as `task878-failed-fill-semantics` (factory run 35, 3 cycles).**
> A no-print bar is now an ATTEMPTED order that fills with probability `p`, not an illegal action.
> `--fill-prob {auto|<float>}` on BOTH `imprint.py` and `eval_vector.py`, **`default=None` = OFF**,
> and OFF reproduces 868 exactly.
>
> **OFF-parity at this repo's actual bar — ZERO non-volatile differing leaves.** eval_vector
> policy-argmax 248/248 · eval_vector with the D17 OOS flags (`--teacher-floor
> --vector-weight-rebalance`) 266/266 · imprint 953/953 — all **0 added, 0 removed**, differing
> leaves only `{checkpoint, checkpoint_load.resolved_path, db, ledger_row, out_pointer,
> roundtrip.ckpt_dir}`. **The disclosure keys are STRUCTURALLY ABSENT when the flag is off, not
> present-and-null** — the null-valued first version produced 4 differing leaves and turned three
> 861 legs red; promotion was REFUSED and the keys were made absent instead, so no oracle was
> re-based. Full 861 recipe **12/12, 986.2 s, tracked pins IDENTICAL**; 863 and 865 all legs;
> selftest 12/12; new `experiments/878_failed_fill/smoke_failed_fill.py` ALL LEGS PASS.
>
> **The determinism gate is the one that mattered.** The coinflip is a seeded PATH-INDEPENDENT
> `fold_in(fold_in(PRNGKey(eval_seed), decision), slot)` — never a sequential stream. Proven
> against a correct negative control (a naive conditional-draw stream that PASSES a same-seed rerun
> and FAILS path-independence), and independently by the orchestrator: recomputing with 90
> decisions leaves the first 50 rows bit-identical, which no sequential stream can do. `p` is the
> causal trailing print density — corrupting panel rows ≥45 first perturbs decision 16, and
> decision 15 IS row 45 with its window `[25,45)` correctly excluding its own row.
>
> **Gate (ii), the 78.8%-class hazard, is pinned by a REAL mutation** — the test calls the actual
> pre-878 `frame_hold_mask(..., include_frozen=True)` at its true drop site and asserts the two
> disagree. Confirmed economically on the real fixture: `teacher.final_pv` is byte-identical
> between OFF and `--fill-prob 0` (92754.42 / 87335.85 both ways) while `teacher.action_hist`
> differs sharply (122/3518/128 vs 1349/2277/142) — the trajectory unchanged, the RECORDED INTENT
> different. That is the whole ruling in one measurement.
>
> **Three cycles, eight findings, every one from a panel and not from the author.** The cycle-1
> BLOCKER: `--fill-prob` made zero-coverage slots tradeable and blew PV to 2.5e16, because one
> conditional covered two distinct populations. Cycle 2 fixed it and introduced two more (a widened
> guard whose BODY was not updated; `walk_pct_hold_agg` split from the payload it describes, making
> `imprint --teacher-vectors` emit a FALSE tampering diagnosis). Cycle 3 closed those and a stale
> `UNCONDITIONAL` claim falsified by its own gating fix. **A 26-hit sweep for the CLAIM (not the
> line numbers) then found 4 false and 2 misleading assertions across three files** — fixing them
> site-by-site is what had generated the previous three.
>
> **`step_vector` deliberately got NO guard** — not provably inert across self-play, training
> rollouts, MCTS leaf, `trader/` and both experiment engines, so proving it exceeds this row while
> risking canon bit-identity. It got a CALLER CONTRACT docstring naming the measured corruption
> plus an executable characterization leg so the contract cannot rot. `jax_trading_env.py` is
> **purely additive, 205 insertions / 0 deletions.**
>
> **`--collect-chosen-vectors` + `--fill-prob` is REFUSED at argparse, and the reason is worth
> keeping:** it is not a disclosure problem but an unsatisfiable CONTRACT. A collection promises
> that replaying its stream reproduces `provenance.walk_net21`, and `_tv_replay_check` enforces
> that; under `--fill-prob` that score is coinflip-dependent while the payload is the deterministic
> ATTEMPT stream, so no consumer can reproduce it BY CONSTRUCTION. Weakening the repo's best
> integrity gate to accommodate a combination nothing asks for yet was rejected.
>
> **REMAINDER → planner.** (1) **Scope 6's 3-arm ablation (never-fill / data-derived / always-fill)
> is a SCHEDULED real-data run and did NOT happen here** — the arms are built and demonstrated on
> the fixture only; it is the read that decides whether flat/no-print cells carry signal.
> (2) Follow-up rows: the one-line severance radius (bind fill counts to realized state, not to the
> step's input array); the arm-matched teacher floor (threading the gate through changes what
> `corridor_position` measures); (A)'s firing disclosed on stdout only, so gate (iv) rests on a
> print line — fix is a count inside the already-ON-only `fill_prob` block; and a pinned-frame
> smoke leg, since no 878 leg exercises one. (3) Other lanes: `scorecard.py` has no `frozen_fill`
> passthrough; `imprint.py --help` crashes on a literal `%` in the `--iters` help string.

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
