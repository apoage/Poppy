# TASK_857 — counterfactual value imprint → v̂-leaf search (D23 step 2)

The deployable-search path. The proven-good search (+8.02 net@21, LEDGER 832/E5-inversion3)
has ZERO learned parts — its leaf is an R-step TRUE-ENV rollout = lookahead, deployable never.
Every net-dependent path dies at one number: v̂ ranks candidates at ~+0.03 correlation with the
true R-step return (849's calibration column). This task trains the value head on
counterfactuals so v̂ ranks candidates the way the true env does, then runs the SAME search with
v̂ as the leaf (`eval_vector.py --leaf-eval vhat-distilled`) — deployable at decision time.

## Pieces (pre-registered)

1. **Counterfactual dataset generator** (`build_counterfactual_dataset` / `collect_pass`):
   walks teacher + true-env-leaf-greedy trajectories, BOTH cost tiers; at every decision emits
   `(state, candidate action-vec, true R-step discounted return G=r0+γ·V)` for the K dithered-book
   candidates (hb=3.0 — the deployment distribution) + the policy-argmax candidate. The true return
   is computed by the SHIPPED 851 leaf machinery `SampledMuZeroVectorSearch._true_env_leaf_values`
   (tail OFF ⇒ pure truncated true-env rollout) — reused, not reinvented. RAM only; no disk I/O in
   the train loop (efficiency invariant VRAM>RAM>disk).

2. **Value-head imprint** (`distill_value_head`): BC value regression v̂(g(s,a)) → G, warm-started
   from `runs/832_e5/ckpt_imprint_full` (real) / the 832 `ckpt_smoke` (fixture). **Dynamics + value
   gradients ONLY** (optax `multi_transform`); encoder, TRM reasoning refiners, policy head, and the
   reward head are FROZEN. The reward-projection columns of `dynamics_dense2` receive exactly zero
   gradient (the loss touches only next_state→value), so the reward head is frozen in effect even
   though the Dense is in the trainable partition — verified per-run (`freeze_invariants`). PopArt
   normalizes the true-return targets across the 401-bin support (shipped value-loss two-hot form).

3. **`eval_vector.py --leaf-eval vhat-distilled`** (TASK_857 scope 3): the DEPLOYABLE read.
   Mechanically the canon MODEL leaf (value head + dynamics, no lookahead) — `build_search` maps it
   to `leaf_eval='model'`, so the search is **bit-identical** to a `--leaf-eval model` run on the
   same checkpoint. Provenance-only distinction + `net_dependent=true` even at prior_weight=0 (the v̂
   leaf IS the learned component, wired into `operator_net_dependent` per TASK_856). `sampled_search.py`
   is UNTOUCHED — the new mode lives entirely in the operator harness.

## Acceptance

- **(i) Calibration** (the 849 instrument): the headline/gated statistic is the **mean PER-STATE
  Pearson** (corr across candidates *within* a state) on **HELD-OUT** states — NOT the inflated
  pooled Pearson. Reported for TWO predictors: **`vhat_alone`** (the spec's literal (i)) and
  **`deployed_q = r_model + γ·v̂`** (what the model leaf actually ranks by — the honest deployment
  predictor that tracks ranking-transfer and gates the corridor read). Real-leg bar **≥+0.5**
  (from ~+0.03), graded on the real panel by the planner. Fixture: finite + IMPROVES over
  pre-imprint on the held-out per-state v̂-alone number.
- **(ii) Deployable ranking (fail-closed-safe floor, NOT a lift claim)**: the fixture gate is the
  **deployed-Q** (`r_model + γ·v̂`, what the model leaf actually ranks by) held-out mean per-state
  Pearson being **finite AND > −0.25** (`DEPLOYED_Q_DEGENERACY_FLOOR`). This is a FAIL-CLOSED-SAFE
  non-degeneracy floor: it reliably rejects STRONG anti-correlation / `--shuffle-targets` sabotage
  (deployed-Q ~**−0.49**, consistent across seeds) but is **NOT** a positive validation of
  deployed-ranking lift. Seed-robustness (round-3 panel): the LEGITIMATE distill's absolute
  deployed-Q is **within-noise and seed-dependent** at fixture scale — measured
  **{+0.115, −0.014, −0.119, +0.491} across seeds 0–3** (the frozen 832 reward head gives no
  reliable lift; the distill often nudges it *down* vs the warm start). An absolute `> 0` floor
  would false-fail ~half of legitimate re-runs, so the floor sits **below** the legit noise band
  (min ~−0.12) and **well above** sabotage (~−0.49). The deployed-ranking VERDICT is the planner's
  real-data corridor read, NOT this gate. The literal spec (ii) — argmax-CHOICE agreement % of the
  v̂-leaf vs true-env-leaf search — is REPORTED as a diagnostic only (**≥70%** real-leg bar is the
  planner's leg); it is NOT discriminative at fixture scale (shared rng draws + the frozen reward
  head dominate selection, so a good and a degenerate leaf both score ~58%).
- **(iii)** Canon + all existing operators (`model`, `true-env`) BIT-IDENTICAL (flags OFF); `trader
  selftest` green (ST12).

### Deployment-predictor gap (load-bearing finding for the planner)

The DEPLOYED leaf ranks by `Q = r_model + γ·v̂`, not v̂ alone. On the fixture the FROZEN 832 reward
head is a poor r0 estimator, so `deployed_q` calibration sits BELOW `vhat_alone` — clearing a
v̂-alone bar does NOT establish the deployable operator ranks like the true env. The runner prints
a LOUD note + records `deployment_predictor_gap` in the JSON. The reward head may be the next
locus; TASK_857 does NOT touch it (planner decision).

## Fixture smoke (reproduction)

```bash
source trader_env/bin/activate
python experiments/857_vhat_distill/distill.py     # -> results_fixture_smoke.json + ckpt_smoke/
```

Defaults: n_assets 8, bar_minutes 240, window 2025-09-14..2025-12-12, K=8, sims=8, R=3, 80 epochs,
warm-start = the 832 `ckpt_smoke`. Fixture result (seed 0): **849 instrument** (mean per-state
Pearson, HELD-OUT) — v̂-alone **+0.2331 → +0.3416** (the (i) improvement gate), deployed-Q
**+0.1250 → +0.1153** (the (ii) fail-closed-safe floor: **+0.1153 > −0.25**); value CE 5.36→2.21;
freeze invariants all hold; argmax-choice diagnostic 58.3% of 12 held-out states. The fixture is a
~120-decision losing slice — the held-out numbers are coarse plumbing proofs, NOT the verdict.
`--shuffle-targets` is a falsification hook (writes NO deliverable): it degrades the distill and
MUST fail the smoke gates — v̂-alone (i) drops +0.2331 → +0.0917, deployed-Q (ii) drops +0.1250 →
**−0.4940** (< −0.25) — proving the gates are non-hollow. Note the (ii) floor is fail-closed-safe
(rejects sabotage), not a lift claim: the legitimate absolute deployed-Q is within-noise/seed-
dependent ({+0.115,−0.014,−0.119,+0.491} across seeds 0–3), so the committed smoke is seed-0.
Deliverables (`ckpt_smoke`, results JSON) are written ONLY after all smoke asserts pass, so a
failing/falsification run never overwrites them.

## Deployable corridor read (SCHEDULED — planner-owned, NOT the factory)

```bash
python experiments/835_stage4_eval/eval_vector.py --operator search \
  --leaf-eval vhat-distilled --checkpoint <distilled_ckpt> ...
```

The program's first DEPLOYABLE-OPERATOR corridor read: v̂-leaf search on the real panel vs the
corridor — floor bar +0.424 first, the +8.02 true-leaf ceiling as the honest reference. Verdict →
LEDGER.

## Out of scope

Policy-head training (856's lane); live/paper; OOS (D22); changing the search proposal/selection
(frozen at the sweep winner).
