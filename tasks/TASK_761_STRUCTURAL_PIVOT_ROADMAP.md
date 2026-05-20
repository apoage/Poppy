# TASK_761 — Structural Pivot Roadmap (post single-mechanism exhaustion)

**Status**: DECISION TREE — operator authoritative, no execution until 760 closes.
**Created**: 2026-05-20 (post-recal verdict, see `lore/notes_2026-05-20.md` §151).
**Trigger**: 760 sims=64 paired Δα result. If null (expected, ~60% probability based on recal asymmetric-lift finding), this roadmap activates.

## Context

Per `lore/notes_2026-05-20.md` §151: every single-mechanism perturbation tried in the 738–758 campaign is null-or-worse under eval-MCTS (deploy-truthful operator). Mechanism class verdict map is fully closed on encoder / aux head / distill / data aug / memory / dynamics loss / value target / search depth. **760 is the last small-perturbation knob without code change.**

This task is the decision tree for what comes next. It's not a single experiment SPEC — it's the framework for picking the next experiment SPEC.

## Branch criteria — 760 closes

Compute 760 sims=64 3-seed paired Δα against fresh a1b1 sims=32 baselines under eval-MCTS sims=16 joint.

| 760 paired mean Δα | Branch |
|---|---|
| ≥ +3pp AND no seed < −3pp | A: PROMOTE sims=64 as canonical, run 5-seed extension and sims=128 |
| in [+1, +3)pp | B: 5-seed extension, escalate to structural pivot only if 5-seed lands null |
| in (−1, +1)pp | C: STRUCTURAL PIVOT (default, ~60% expected) |
| < −1pp | D: STRUCTURAL PIVOT (deeper search regresses → mechanism class is anti-search) |

This task assumes branch **C or D**. Branches A/B don't activate this roadmap.

## Four structural directions

Each direction is multi-day work. The decision is which to run FIRST (one at a time, not in parallel — these are big enough that paired Δα precision matters more than wall-time throughput).

### Direction 1 — Reward shaping (LOWEST RISK, MOST DEFENSIBLE)

**Hypothesis**: Over-trade collapse signature (738/741/740/757/758 all show Δ-trade-rate ≫ 0 with Δα ≪ 0) says noise trades are under-penalized. Current `reward_scale=100` is baked in `phase2_config.py:28`; fee/cost penalty is implicit only.

**Code changes required**:
- Add `--reward-scale N` CLI flag (default 100 = current).
- Add `--fee-penalty B` CLI flag (default 0 = current).
- Wire both into `phase2_config.Phase2Config` and `trainer.py` reward computation.

**Variants to sweep**:
- `reward_scale ∈ {25, 50, 100, 200}` (4 cells × 3 seeds = 12 runs).
- After best `reward_scale` found, sweep `fee_penalty ∈ {0, 0.5bp, 1bp, 2bp}`.

**Wall budget**: ~2.5h per Stage-B run × 12 = 30h wall (2-wide ~15h).

**Promotion gate**: ≥ +3pp paired Δα at 3-seed under eval-MCTS sims=16 joint, no seed < −5pp. 5-seed extension if 3-seed clears.

**Risk**: Lowest. Reward scaling has known dynamics in policy gradient methods. Most likely to yield clear PASS/FAIL signal.

**Drafted as**: `TASK_762_REWARD_SHAPING.md` (sub-task, to be drafted before launch).

### Direction 2 — Policy entropy regularization (MEDIUM RISK, MECHANISM-ALIGNED)

**Hypothesis**: Classifier-Alpha anti-correlation pattern (`project_classifier_alpha_anti_correlation.md`) says sharper classifiers degrade Alpha via over-confident wrong trades. Entropy regularization preserves classifier sharpness budget for action selection precision instead of class probability sharpness.

**Code changes required**:
- Add `--entropy-coef F` CLI flag (currently hardcoded `entropy_coef=0.01` in `trainer.py:382,435`).
- Add `--min-entropy F` CLI flag (currently hardcoded `min_entropy=3.0`).
- Wire both through `trainer.py` and Stage-B loss computation.

**Variants to sweep**:
- `entropy_coef ∈ {0.005, 0.01, 0.02, 0.05, 0.1}` (5 cells × 3 seeds = 15 runs).
- Track Δ-trade-rate AND Δ-Bal.Acc as primary diagnostics; expect entropy_coef high → lower Bal.Acc, hopefully higher Alpha (the anti-correlation broken).

**Wall budget**: ~2.5h × 15 = 37h wall (2-wide ~19h).

**Promotion gate**: same +3pp gate. **Anti-correlation check is the early-exit gate**: if best cell does NOT show Δ-Bal.Acc < 0 AND Δα > 0, the mechanism didn't break the anti-correlation pattern even if Alpha is up — stop and pivot.

**Risk**: Medium. Entropy reg has known dynamics in RL/MuZero but the effective sweep range on this stack is unknown.

**Drafted as**: `TASK_763_POLICY_ENTROPY_REG.md` (sub-task, TBD).

### Direction 3 — Oracle k-reformulation (HIGHEST CEILING ON SINGLE-COMPONENT FIX)

**Hypothesis**: `project_bull_blindness.md` traces bull-regime failure to k=4/64 oracle class imbalance: oracle picks top-4 assets out of 64 universe, so "long top-4" labels are scarce in bull regimes where many assets rally. Larger k (k=8/64 or k=12/64) increases label density and reduces class imbalance.

**Code changes required**:
- `--oracle-k N` CLI flag (currently k=4 implicit in `mcts_trader/jax_agent/path_solver.py`).
- Phase 1 re-imprint REQUIRED (oracle labels change → encoder targets change). Phase 1 wall: ~12h on workstation, ~4h on server per seed.
- Phase 2 retrain on new Phase 1 ckpts.

**Variants to sweep**:
- `oracle_k ∈ {4 (canon), 8, 12}` (3 cells × 3 Phase 1 seeds × 3 Phase 2 seeds = 27 runs total worst case).
- Phased: imprint k=8 first, train Phase 2 a1b1 on it, paired Δα against current k=4 a1b1.

**Wall budget**: ~4h Phase 1 × 3 seeds + ~2.5h Phase 2 × 9 = ~35h wall total (single-GPU paths, can't easily 2-wide because Phase 1 dominates).

**Promotion gate**: +3pp paired Δα. **Sanity gate first**: k=8 imprint must lift bull-regime classifier Bal.Acc by ≥ +5pp on Phase 1 metrics, OR the hypothesis is dead and Phase 2 doesn't need to run.

**Risk**: Higher. Touches Phase 1 — known good imprint becomes the variable. Worst case: new Phase 1 imprint is WORSE on non-bull regimes and we regress globally. Mitigation: bull-only Phase 1 ablation first.

**Drafted as**: `TASK_764_ORACLE_K_REFORMULATION.md` (sub-task, TBD).

### Direction 4 — Incompatible-dataset retrain (HIGHEST CEILING + HIGHEST RISK)

**Hypothesis**: Daily OHLCV bars on the current 64-asset universe may have insufficient signal-to-noise. Iris-DEX (`~/Iris-DEX/`) produces per-block (12s) OHLCV from on-chain Reth archive node — 7200× the bar frequency. Different universe (DEX-listed tokens vs CEX-listed). Possibly different inductive bias entirely.

**Code changes required**:
- Data pipeline: ingest Iris-DEX block-level OHLCV into a parallel SQLite DB (or new format).
- `SegmentPool` (or new pool class) for block-bar data.
- Re-derive oracle labels at new bar interval (oracle "top-k over horizon H" needs new H).
- Phase 1 imprint from scratch on new data.
- Phase 2 a1b1 retrain on new Phase 1.
- New benchmark periods (current `config/benchmark_periods.json` is daily-anchored).

**Wall budget**: data prep ~1 week (per `prompt-lib/defi_data_pipeline.md`), then Phase 1 + Phase 2 ~3 days/seed. Total: ~2 weeks per seed for the first attempt.

**Promotion gate**: this isn't paired-Δα against the current campaign — it's a **standalone alpha-positive question**. Promotion = profitable on its own benchmark.

**Risk**: Highest. Breaks every assumption in the current stack. But if it works, it's a fundamentally different product (DEX trader, not CEX trader) — which may be more aligned with the long-term apothecary thesis (per `prompt-lib/bella_context.md`).

**Drafted as**: not yet. Needs operator-side discussion about whether this is a Poppy variant or a new project entirely.

## Recommended ordering

Default operator path (lowest-risk-to-highest-risk):

1. **First — Direction 1 (Reward shaping)**. ~15h wall. Falsifies the over-trade hypothesis cleanly. If positive, we have a published mechanism. If null, narrows the search.
2. **Second — Direction 2 (Entropy reg)**. ~19h wall. Mechanism-aligned with the anti-correlation finding. If positive, breaks a known pattern; if null, kills the "policy is wasting capacity" hypothesis.
3. **Third — Direction 3 (Oracle k-reformulation)**. ~35h wall. Touches Phase 1, so it's the last CLI-side experiment before structural rethink. Highest single-component ceiling but also requires re-imprint.
4. **Fourth (parallel to 3, on workstation/Iris-DEX side) — Direction 4 (Incompatible dataset)**. Begin data-prep groundwork in parallel with directions 1–3. Decision point on whether to fully pivot comes after 1–3 close.

Total CLI-side wall for directions 1–3 sequenced: ~70h = ~3-4 calendar days at 2-wide on server.

## Open decisions for operator

1. **Sequencing override**: operator stated openness to "full retrain on incompatible dataset if it would give some good results." If directions 1–3 are seen as low-EV cleanup rather than genuine hypothesis tests, direction 4 may justify earlier launch (data prep started immediately, in parallel with 760 close).
2. **Workstation vs server split**: directions 1–3 are server-side CLI work. Direction 4 data prep is workstation/Iris-DEX side. They can run truly in parallel.
3. **Stop-rule**: at what point do we declare the current stack architecturally dead? Proposed: if directions 1, 2, 3 all close null under eval-MCTS 5-seed paired, the campaign is closed and direction 4 (or a successor) becomes the only path forward.
4. **Bloom publication trigger**: §151 + the recal asymmetric-lift finding is publishable now as a methodology arc. Pre-publishing before structural pivot may be valuable. Alternative: bundle with whichever structural direction lands first.

## References

- `lore/notes_2026-05-20.md` §151 — recal verdict + asymmetric-lift mechanism.
- `runs/recommendation_2026-05-20/analysis.md` — server-resident synthesis.
- `experiments/760_train_mcts_sims/SPEC.md` — current in-flight experiment.
- `~/.claude/projects/-home-lukas-Trader/memory/project_bull_blindness.md` — Direction 3 evidence base.
- `~/.claude/projects/-home-lukas-Trader/memory/project_classifier_alpha_anti_correlation.md` — Direction 2 evidence base.
- `prompt-lib/defi_data_pipeline.md` — Direction 4 data-prep groundwork.
- `prompt-lib/bella_context.md` — Direction 4 universe/thesis fit.

## Status

- **DECISION TREE**: ready for operator review.
- **Sub-task SPECs**: `TASK_762_REWARD_SHAPING.md`, `TASK_763_POLICY_ENTROPY_REG.md`, `TASK_764_ORACLE_K_REFORMULATION.md` — to be drafted when operator picks ordering.
- **Direction 4 SPEC**: deferred pending operator decision on parallel data-prep launch.
