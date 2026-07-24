# TASK_857 — Counterfactual value imprint → v̂-leaf search (D23 step 2; the deployable-search path)

> **STATUS (2026-07-24): SHIPPED as `task857-vhat-leaf-distill`.** Deployable `--leaf-eval vhat-distilled`
> operator (= the canon model leaf on a distilled ckpt; `sampled_search.py` untouched, `net_dependent:true`)
> + `experiments/857_vhat_distill/distill.py` counterfactual value-head imprint (dynamics+value only,
> policy/reward frozen). Canon + `model`/`true-env` leaf modes BIT-IDENTICAL; selftest 12/12. Medium panel
> across THREE fix rounds (regression clean throughout): fixed pooled-vs-849-instrument calibration,
> v̂-alone-vs-deployed-Q measurement, two hollow gates, and a seed-fragile floor → now a fail-closed-safe
> deployed-Q floor (> −0.25; `--shuffle-targets` FAILS + writes nothing, normal PASSES). Fixture acceptance:
> (i) 849 instrument v̂-alone +0.2331→+0.3416; (ii) deployed-Q floor +0.1153 > −0.25.
> **LOAD-BEARING FINDING → DECISIONS D24**: v̂ improves on its own axis but the DEPLOYED-Q ranking
> (`r_model+γ·v̂`) does NOT lift at fixture scale (frozen 832 reward head caps it) — clearing the v̂-alone
> ≥+0.5 bar does not alone prove the deployable operator ranks like the true env.
> **Remainder → planner**: the real-data corridor read of vhat-distilled (warm-start `ckpt_imprint_full`)
> vs floor +0.424 / +8.02 ceiling — the first DEPLOYABLE-operator corridor read; D24 = the read-then-decide
> fork on co-distilling the reward head. Carried MINORs: (i)-before-(ii) assert order (fail-closed-safe;
> (ii) independently bites); the floor can't distinguish good-vs-noop at fixture scale (frozen-reward-head limit).

**Status**: READY 2026-07-23 (D23 ruled — operator; D19 opt (b/c) with exact targets).
**Size: M. Risk: medium** (new offline training recipe for the VALUE head + a v̂-leaf eval mode;
the policy/canon training paths untouched; flag-gated everywhere).

## Why (LEDGER 849/leaf → 832/E5-inversion3 → 832/E5-exit2)
The proven-good search (+8.02 net@21 on the frozen imprint) has zero learned parts — its leaf
is an R-step TRUE-ENV rollout = lookahead, deployable never. Every net-dependent path so far
dies at one measured number: **v̂ ranks candidates at +0.03 correlation with the true R-step
return** (849's calibration column) because it was never trained on counterfactuals. Fix the
number directly: train the value head to rank candidates the way the true env does, then run
the SAME search with `v̂` as the leaf — deployable at decision time, graded against the +8.02
teacher ceiling.

## Scope
1. **Counterfactual dataset generator** (offline, reuses the 851 leaf machinery): walk
   trajectories (teacher walk + fixed-search walk, both cost tiers) and at every decision state
   emit (state, candidate action-vector, true R=3-step discounted return) for the K candidates
   the search actually generates (dithered-book hb3.0 — the deployment distribution) plus the
   policy-argmax candidate. Store to RAM/npz per the efficiency rules (this is a bounded
   instrument dataset, ~67k tuples/trajectory-pass, NOT training-loop disk I/O).
2. **Value-head imprint** (BC-style, mirrors 832's recipe): regress `v̂(g(s,a))` on the true
   returns over the counterfactual set (dynamics+value gradients only; policy/reward heads
   frozen), from the `ckpt_imprint_full` warm start; iters/schedule per the 832 imprint scale.
3. **`--leaf-eval vhat-distilled`** on the 835 search operator: identical search, leaf = the
   distilled v̂ (no lookahead). `net_dependent: true` per TASK_856's grading convention.
4. Fixture smoke: end-to-end tiny run (generator → imprint → search-eval), all finite, schema.

## Acceptance (gates)
- (i) **Calibration (the 849 instrument, pre-registered bar)**: post-imprint v̂-vs-true-return
  correlation across candidates ≥ **+0.5** (from +0.03) on held-out states (fixture leg;
  real leg = planner run).
- (ii) Ranking transfer: the v̂-leaf search on the fixture ckpt reproduces the true-env-leaf
  search's candidate CHOICES on ≥70% of dumped states (849-style trace comparison).
- (iii) Canon + all existing operators bit-identical (flags OFF); selftest green; medium panel
  (correctness + regression + fail-closed lenses).
- The REAL graded read (planner, post-ship): v̂-leaf search on the real panel vs the corridor —
  floor bar +0.424 first, the +8.02 true-leaf ceiling as the honest reference; verdict →
  LEDGER. That number is the program's first DEPLOYABLE-OPERATOR corridor read.

## Out of scope
Policy-head training (856's lane decides whether the distilled policy is rescuable); live/paper
anything; OOS (D22's lane); changing the search's proposal/selection settings (frozen at the
sweep winner).
