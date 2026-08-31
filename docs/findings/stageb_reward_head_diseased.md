# stageb reward head diseased

> 2026-07-05 audit: Stage-B (train_phase2) reward head CONFIRMED diseased — pure CE on an action-invariant oracle mark-to-market target, trained only on the oracle argmax action → cannot rank actions by construction. Eval-MCTS scorecard: ABSOLUTE levels compromised (joint-decode 1/64 trade cap dominates); PAIRED deltas survive but compressed/biased-toward-null. 738–784 nulls stand, weaker than assumed.

*Migrated from local agent memory 2026-08-31 (`project_stageb_reward_head_diseased.md`) so subagents and the server agent
can read it — see `lore/adr/0008`. Content verbatim; only frontmatter and wiki-links changed.*
2026-07-05 read-only audit (follow-on to [verdict_contamination_audit](verdict_contamination_audit.md)): Stage-B has its
own instance of the TASK_818 reward-head disease, via a structurally WORSE route.

**Confirmed mechanics** (`train_phase2.py:374-377,1635-1636,727,731-742,2310-2311`;
`phase2_config.py:28-30`):
- Pure `softmax_cross_entropy` on hard one-hot target; no MSE anchor, no PopArt; weight 1.0.
- Target = oracle-basket mark-to-market per step (~0.001 scale, ×100 into 401 bins) —
  **action-invariant** (a state property).
- Head trained ONLY on the oracle argmax action → 319/320 actions are untrained extrapolation.
- Smoking gun from ~90 runs' CSVs: reward_loss plateau is a pure function of `reward_scale`
  (rs025→1.45, rs050→2.02, rs100→2.58, rs200→3.15), invariant across every mechanism variant —
  the head fits state (vol clustering), is blind to action.

**Important CORRECTION to earlier framing:** the eval-MCTS trade-rate collapse (argmax ~74% →
~1-3%) is DOMINANTLY the canonical operator's `decode='joint'` returning ONE action per step for
64 assets → structural cap 1/64 ≈ 1.56% (`benchmark_easy.py:576-582,595`). Reward-blindness only
nudges the one filled slot toward HOLD. Also: the 818 "reward out-scales value 10×" component is
likely ABSENT at Stage-B eval (value is PopArt-denormalized to raw units; both same scale).

**Scorecard implications (keep the two claims distinct):**
- ABSOLUTE eval-MCTS levels (trade rate, alpha) = operator artifacts. Never quote them as skill.
- PAIRED Δα (variant−baseline) remains directionally valid (same lens both arms) BUT the
  instrument is compressed → biased toward NULL and under-powered; symmetry not guaranteed
  (759's "lifts baseline, not variants" is a counterexample). The 738–784 null verdicts STAND
  but are weaker evidence than assumed; small real effects could be masked.

**Fix shape if ever needed:** 818 Tier-A analog (hybrid loss + reward PopArt in train_phase2 +
consistent denorm in mctx_search `_reward_to_scalar`) un-sticks the loss but CANNOT create action
discrimination (target is action-invariant + no counterfactual coverage). Load-bearing =
action-attributable multi-step reward + counterfactual action coverage, or stop routing eval
through the reward head (per-asset decode / value+policy-only ranking).

**CONFIRMED BY DIRECT PROBE (2026-07-05, `runs/stageb_reward_probe_2026-07-05.md`):** on
743/seed_7 `phase2_stageB_best.pkl`, 96 REAL latents (bull period, real encoder path) × all 320
actions, exact eval decode: reward across-action median STD **5.34e-03** vs across-latent
**0.323** → **60.4× ratio**; and 32× smaller than the VALUE head's across-action STD (0.171)
from the SAME shared dynamics trunk — the trunk carries action information (value expresses it),
the reward head doesn't use it. Action-blindness is a property of the trained head, not the
architecture. Prediction verified exactly.

Related: [collapse_fixed_predictability_wall](collapse_fixed_predictability_wall.md), [oracle_design_intent](oracle_design_intent.md),
[single_mechanism_exhausted](single_mechanism_exhausted.md) (nulls stand, evidence weaker).
