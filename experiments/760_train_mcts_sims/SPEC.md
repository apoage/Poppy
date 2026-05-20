# SPEC 760 — Train-time MCTS Sims Sweep with Eval-MCTS Scorecard

> **Status**: OPERATOR-APPROVED, IN FLIGHT.
> Promoted from server-drafted `runs/recommendation_2026-05-20/SPEC_DRAFT.md` 2026-05-20 (operator workstation).
> Wave 1 (sims=64, seeds 7/13) trained; seed 23 + paired a1b1 sims=32 baselines training as of approval.
> This SPEC is the **final small-perturbation data point** before the structural pivot (see `tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md`).
>
> **POST-RECAL NOTE (2026-05-20 11:14 UTC):** the recal sweep Tier A
> closed before this draft was finalized. Both 5-seed candidates (746
> oracle, 734 flow) landed null/slightly-negative under eval-MCTS @
> sims=16 joint (746 paired Delta-alpha = -1.20pp, 734 paired = -0.15pp).
> See `analysis.md` §10 for the full data. **Branch taken:** "in
> (-2, +2)pp" -> run 760B sims=64 3-seed BUT also train paired a1b1
> sims=32 baselines (DO NOT reuse 759 frozen baselines). Expected value
> is reduced vs the original framing; the **entropy-regularization
> code-change pivot** is now a competitive alternative the operator
> should consider in parallel.

---

## Goal

Test whether increasing train-time MCTS simulation count from the canonical
`mcts_sims=32` (default `phase2_config.py:33`) to `sims in {64, 128}`
yields paired mean Delta-alpha >= +3pp under the new eval-MCTS@16 joint
benchmark scorecard. Falsifiable in one direction: if a paired 3-seed at
sims=64 yields Delta-alpha < -1pp or all-negative under eval-MCTS, the
"deeper search at train time" hypothesis is rejected. If 3-seed yields
mean Delta-alpha >= +1pp with no seed < -3pp, escalate to 5-seed for
campaign promotion.

This is **the operator-named Path B** from 759 SPEC §101 ("Then Path B
(train-side sims sweep with eval-MCTS enabled) becomes the next probe").
Now executable because Path A (759) closed positive on eval-side.

## Hypothesis

PROMOTE-likely look: deeper train-time MCTS produces value targets that
are more consistent with the trained value head's PopArt-normalized
predictions. Under eval-MCTS the trained model is asked to perform a
slightly easier version of the same search; if sims=32 train -> sims=16
eval already lifted +3.66pp, sims=64 train -> sims=16 eval may lift more
(by making the policy more search-aligned during training, so eval-MCTS
finds plans the policy already prefers). Plausible per-seed paired
Delta-alpha range: +1 to +5pp.

REJECT-likely look:
- Mean paired Delta-alpha <= -1pp -> deeper train-search overfits to
  training distribution; eval-MCTS lift is purely a measurement artifact
  unrelated to train-time search depth.
- Variance > 8pp at 3-seed -> mechanism is unstable per-seed; not a
  campaign-grade signal at this stack's precision.
- Training time blows up beyond 5h per run -> sims=128 is structurally
  too expensive at this stack; sims=64 is the practical ceiling.

## Background — why this and not the alternatives

| Option | Argument | Drawback |
|--------|----------|----------|
| **A. Train-time `--mcts-sims` sweep (THIS SPEC)** | Operator-named next probe. Code wired (`train_phase2.py:2021`). No structural risk. Direct test of "deeper search compounds with eval-MCTS". | sims=128 train ~3x slower than sims=32; ~6h per seed. Train value/policy may already be saturated at sims=32. |
| B. Eval-MCTS sims=32/64 on existing ckpts | Cheap (no retrain). Refines 759 estimate. | Already paused by 759 SPEC §147 - operator's call. Does not test mechanism-side lift, only measurement-side. |
| C. Policy entropy regularization sweep | Direct attack on classifier-Alpha anti-correlation root cause. | NO CLI flag - requires `train_phase2.py` code change (`entropy_coef`, `min_entropy` flags need wiring). Operator-side work. |
| D. Reward shaping sweep | Direct attack on over-trade collapse signature seen in 738/741. | NO CLI flag - `reward_scale` baked in `phase2_config.py:28`. Code change. |
| E. 746 oracle + deeper sims stack | Combines two near-positive signals. | Premature - 746's paired-vs-eval-MCTS verdict only lands ~12:30 UTC today. Stacking before recal close is a coin flip. |
| F. Phase 1 retrain | Addresses bull-blindness root cause from `project_bull_blindness.md`. | 12+ hours on workstation, blocking. Out of server agent scope. |

**A is the lowest-risk highest-EV path that doesn't require code changes
and tests a hypothesis (deeper-train-search compounds with eval-MCTS)
that is mechanistically motivated by 759's +3.66pp eval-side lift.**

## Prerequisites (verified on server 2026-05-20)

| Flag | Wired? | Source | Notes |
|------|--------|--------|-------|
| `--mcts-sims N` | YES | `train_phase2.py:2021` | int, defaults to None which uses `phase2_config.mcts_sims=32` |
| `--mcts-discount` | YES | `train_phase2.py:2023` | default 0.95 (per `phase2_config.py:37`) - leave at default |
| `--eval-mcts-sims N` | YES | `tools/benchmark_easy.py:1158` | default 0 (argmax); use 16 to match 759 |
| `--eval-mcts-decode joint|per_asset` | YES | `tools/benchmark_easy.py:1160` | default 'joint' - matches `real_trade_jax.py:131` |
| a1b1 baseline ckpts | YES | `runs/741_mint/baseline_seed_{7,13,23}/phase2_stageB_best.pkl` + `runs/752_value_oracle_baseline_pair/baseline_seed_{31,42}/phase2_stageB_best.pkl` | 5 seeds total |
| `runs/759_mcts_eval/seed_{7,13,23,31,42}/sims16_joint/benchmark.json` | YES for 7/13/23/31/42 | Produced by 759 + recal Wave A1 | Reuse, do not re-eval |
| Phase 1 imprint ckpt | YES | `jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl` | unchanged from 734-759 |

No new code required. No new precompute required. No new dependencies.

## Resources

Per-run wall time estimate (seed 7, sims=32 baseline ~2.5h on this stack):
- sims=64: ~3.5-4h per seed (search is the dominant cost in Stage B)
- sims=128: ~6h per seed

3-seed wall at 2-wide (TASK_742 rule):
- sims=64: 2 * 4h = ~8h wall (seeds {7,13,23} in two waves)
- sims=128: 2 * 6h = ~12h wall

Eval (one per ckpt, sims=16): ~25 min per ckpt (from 759 timing). Run
3 evals in parallel on 3 GPUs after the train wave closes.

Total Wave 1 (sims=64 only, 3 seeds): ~9h wall.
Total Wave 1+2 (sims=64 + sims=128, 3 seeds each): ~21h wall.

GPU budget: 2 GPUs concurrent for train (TASK_742), 3 GPUs concurrent
for eval (3-wide eval is operator-authorized per recal precedent).
VRAM: sims=64 ~16 GB, sims=128 ~22 GB (each near 24GB cap; monitor).

## Plan — variants table

Single mechanism axis: `--mcts-sims`. Three values, 5 seeds each
target (3-seed screening + 2-seed extension if positive).

| Cell | mcts-sims | Seed set | Baseline pair                          | Eval                       | Status |
|------|----------:|----------|----------------------------------------|----------------------------|--------|
| 760A-32-ref | 32       | {7,13,23}| `runs/759_mcts_eval/seed_{7,13,23}/sims16_joint/` | reuse (no rerun)           | reuse |
| 760A-32-ref31 | 32     | {31,42}  | `runs/759_mcts_eval/seed_{31,42}/sims16_joint/` (recal Wave A1)   | reuse (no rerun)           | reuse |
| 760B-64 | 64        | {7,13,23}| 760A-32-ref                              | sims=16 joint, eval-MCTS   | RUN  |
| 760C-128 | 128      | {7,13,23}| 760A-32-ref                              | sims=16 joint, eval-MCTS   | RUN if 760B promotes |
| 760D-64-ext | 64    | {31,42}  | 760A-32-ref31                            | sims=16 joint, eval-MCTS   | RUN if 760B 3-seed paired Delta-alpha >= +1pp |
| 760E-128-ext | 128 | {31,42}   | 760A-32-ref31                            | sims=16 joint, eval-MCTS   | RUN if 760C 3-seed paired Delta-alpha >= +1pp AND > 760B |

Paired Delta-alpha = (variant alpha under eval-MCTS sims=16 joint) -
(baseline alpha under eval-MCTS sims=16 joint), per-seed, then averaged.

## Wave plan

### Wave 1 (Monday morning): 760B sims=64 3-seed paired

```
T=0:00   GPU 1: train sims=64 seed 7
         GPU 2: train sims=64 seed 13
         GPU 3: idle (avoid GPU 0 per `feedback_gpu0_avoid.md`)
T~4:00   Wave 1A flags drop.
T=4:00   GPU 1: train sims=64 seed 23
         GPU 2: eval sims=64 seed 7 (eval-MCTS sims=16 joint)
         GPU 3: eval sims=64 seed 13
T~8:00   GPU 2: eval sims=64 seed 23
T~8:30   Wave 1 closed. Compute paired Delta-alpha.
```

### Wave 2 (conditional): 760D 5-seed extension OR 760C 128-sims

Branch on Wave 1 paired 3-seed result:

- **IF Wave 1 paired mean Delta-alpha >= +1pp AND no seed < -3pp**:
  Launch 760D (sims=64, seeds {31,42}) in parallel with 760C (sims=128,
  seeds {7,13,23}). ~12h wall. Resolves both 5-seed extension and the
  question of whether even-deeper search compounds.
- **IF Wave 1 paired mean Delta-alpha in (-1, +1)pp**: train-time
  sims=64 is null. STOP, do not launch 760C / 760D. Pivot to next
  candidate in the IF/THEN branches below.
- **IF Wave 1 paired mean Delta-alpha <= -1pp**: train-time deeper
  sims regresses. STOP and pivot to entropy-regularization sweep (code
  change SPEC, operator side).

### Wave 3 (conditional): 760C+E 5-seed promotion test

If Wave 1 + Wave 2 = sims=64 5-seed paired Delta-alpha >= +3pp AND no
seed < -5pp: PROMOTE. Bloom-trigger. Stop sweeping; lock as new
canonical Stage-B.

## Commands

```bash
# COMMON for sims=64 variant (replicated for sims=128 by changing one flag)
COMMON_BASE="--stage B --epochs 200 \
  --periods-file config/foxglove_periods.json \
  --n-slots 17 --max-runs-per-group 10 \
  --factored --factored-temperature 0.1 \
  --no-reset-value-head \
  --value-target-mode nstep --td-discount 0.95 --value-weight 0.5 \
  --bc-weight-min 0.05 --bc-max-trade-weight 4.0 \
  --direction-weight 0.5 \
  --encoder-lr 3e-5 --encoder-version 3 --embed-dim 64 --encoder-heads 8 \
  --checkpoint jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl \
  --rank-loss mse --rank-weight 1.0 \
  --aux-price-weight 0.1"

SIMS=64
SEED=7
GPU=1
mkdir -p runs/760_train_mcts_sims/sims${SIMS}_seed_${SEED}
screen -d -m -S 760_s${SEED}_sims${SIMS} bash -c "\
  source trader_env/bin/activate && \
  CUDA_VISIBLE_DEVICES=${GPU} python -u train_phase2.py ${COMMON_BASE} \
    --mcts-sims ${SIMS} \
    --seed ${SEED} \
    --output-dir runs/760_train_mcts_sims/sims${SIMS}_seed_${SEED} \
    --name 760_s${SEED}_sims${SIMS} 2>&1 | \
    tee runs/760_train_mcts_sims/sims${SIMS}_seed_${SEED}/train.log; \
  echo 760_S${SEED}_SIMS${SIMS}_DONE > /tmp/760_s${SEED}_sims${SIMS}_done.flag"

# Eval (after training closes):
mkdir -p runs/760_train_mcts_sims/eval_sims${SIMS}_seed_${SEED}
screen -d -m -S eval_760_s${SEED}_sims${SIMS} bash -c "\
  source trader_env/bin/activate && \
  CUDA_VISIBLE_DEVICES=${GPU} python tools/benchmark_easy.py \
    --checkpoint runs/760_train_mcts_sims/sims${SIMS}_seed_${SEED}/phase2_stageB_best.pkl \
    --db trading_data.db \
    --json runs/760_train_mcts_sims/eval_sims${SIMS}_seed_${SEED}/benchmark.json \
    --eval-mcts-sims 16 --eval-mcts-decode joint 2>&1 | \
    tee runs/760_train_mcts_sims/eval_sims${SIMS}_seed_${SEED}/benchmark.log; \
  echo EVAL_760_S${SEED}_SIMS${SIMS}_DONE > /tmp/eval_760_s${SEED}_sims${SIMS}_done.flag"
```

COMMON_BASE matches the a1b1 canonical config used in 741_mint, 746,
734, 757, 758. Only `--mcts-sims` changes.

## Success criteria — explicit numerical gates with eval-MCTS as the benchmark

| Gate | Threshold | Action |
|------|-----------|--------|
| Wave 1 (sims=64, 3-seed paired Delta-alpha vs eval-MCTS baseline) | >= +1pp AND no seed < -3pp | Launch Wave 2 (5-seed extension + sims=128) |
| Wave 1 (sims=64, 3-seed paired Delta-alpha) | in (-1, +1)pp | STOP. Train-time sims is null. Pivot. |
| Wave 1 (sims=64, 3-seed paired Delta-alpha) | <= -1pp | STOP. Pivot. |
| Wave 2 (sims=64, 5-seed paired Delta-alpha) | >= +3pp AND no seed < -5pp | **PROMOTE.** Bloom-trigger. |
| Wave 2 sims=128 (3-seed paired Delta-alpha) | > sims=64 3-seed by >= +1pp | Run sims=128 5-seed extension |
| Wave 2 sims=128 (3-seed paired Delta-alpha) | <= sims=64 | sims=64 is local-best; stop deeper sweep |

All paired comparisons use eval-MCTS sims=16 joint as the scorecard.
NOT argmax. The 759 result.md established that argmax under-reports by
~+3.66pp on the a1b1 baseline.

## IF/THEN branches conditional on recal sweep outcome

The recal sweep (`/tmp/recal_orch.sh`) lands ~12:30 UTC 2026-05-20 with
eval-MCTS@16 alphas for: 746 oracle 5-seed, 734 flow 5-seed, 740/757/758
3-seed, 749 paired b128 3-seed, 743 4-seed.

### Branch IF recal: 746 oracle 5-seed paired Delta-alpha (eval-MCTS) >= +2pp AND no seed < -3pp

746 promotes under the new benchmark methodology. The new canonical
Stage-B becomes "a1b1 + oracle V*". Drop 760 as proposed and replace
with:

**760-prime** = same SPEC structure as 760, but variants stack on top of
746 oracle:
- Variant flags: COMMON_BASE + `--value-target-mode oracle` + `--mcts-sims N`
- Baselines: existing 746 oracle 5-seed under eval-MCTS (from recal)
- Goal: test whether sims=64+oracle stacks additively or substitutes

### Branch IF recal: 746 oracle 5-seed paired Delta-alpha in (-2, +2)pp **<- ACTUAL (-1.20pp)**

746 is null at all measurement methodologies. The campaign's strongest
mechanism candidate is genuinely null. Proceed with 760 as drafted
(train-time sims sweep on a1b1 baseline) but lower expectations: this
is now the last test of the "search depth lifts a null configuration"
hypothesis. If 760 is also null, the search-side mechanism class is
closed.

**Additional caveat post-recal:** the asymmetric-lift finding (eval-MCTS
boosts the baseline by ~+3.66pp, but boosts variants by less) means
training paired a1b1 sims=32 baselines is **mandatory** — do NOT reuse
the 759 baselines as the pair. The original SPEC §plan reused them; that
assumption is now invalid. Add 3 new paired a1b1 sims=32 runs (s7, s13,
s23) so each sims=64 variant has a same-train-config paired baseline.

### Branch IF recal: 746 oracle 5-seed paired Delta-alpha < -2pp

746 is dead under eval-MCTS too. The eval-MCTS lift is robust to the
baseline but DOES NOT amplify mechanism-side signals. Strategic
implication: the campaign needs a structural rethink. **Do not launch
760 as proposed.** Instead, write a code-change SPEC for one of:

a. Policy entropy regularization sweep (entropy_coef, min_entropy CLI flags).
b. Reward shaping (reward_scale, fee_penalty CLI flags).
c. Oracle k=8/64 reformulation (addresses bull blindness root cause).

These each require operator-side code work. The server agent should
not start them.

### Branch IF recal: 734 flow_matching 5-seed paired Delta-alpha (eval-MCTS) >= +2pp

Independent signal that didn't promote at argmax now does at eval-MCTS.
Run the same 760-prime variant but stacking on 734 flow_matching
(`--flow-matching`) instead of 746 oracle. Lowest-EV branch because
734's per-seed variance was 6.78pp (very noisy).

## Output expectations

```
runs/760_train_mcts_sims/
├── sims64_seed_{7,13,23}/      train.log, phase2_stageB_best.pkl, stage_b_metrics.csv
├── sims64_seed_{31,42}/        (conditional)
├── sims128_seed_{7,13,23}/     (conditional)
├── sims128_seed_{31,42}/       (conditional)
├── eval_sims64_seed_*/         benchmark.{json,log} (eval-MCTS sims=16 joint)
├── eval_sims128_seed_*/        benchmark.{json,log} (eval-MCTS sims=16 joint)
└── result.md
```

`result.md` must include:
- Paired Delta-alpha table per seed, scored under eval-MCTS sims=16 joint.
- Per-period Delta-alpha breakdown across 3 seeds (averaged).
- Comparison to 759 + the 746 5-seed + 734 5-seed paired (post-recal).
- Training loss totals at best epoch (per-seed sanity).
- Trade-rate signature on bull period (to detect over-trade / under-trade
  collapse - same diagnostic table used in 738/741/740 results).

## Notes (anti-correlation hygiene + caveats)

1. **Anti-correlation alarm.** If Wave 1 shows Delta-Bal.Acc > +3pp AND
   Delta-alpha negative, that's the classifier-Alpha anti-correlation signature.
   STOP - the mechanism is in the wrong class. Train-time sims should
   not produce this; if it does, something is structurally wrong.
2. **Single-mechanism change only.** Do NOT combine 760 with 746 oracle
   in Wave 1. The whole point is to isolate the sims-effect on a1b1
   first; stacking before isolating is what burned 734 + 748.
3. **Eval-MCTS is the scorecard.** Do NOT report argmax alpha as the
   headline. Per 759 result, argmax under-reports by ~+3.66pp on this
   stack. All paired comparisons in result.md should use eval-MCTS
   sims=16 joint.
4. **Per-seed variance > 5pp expected.** Do not interpret single-seed
   results. 3-seed is the minimum decision unit.
5. **Wall time budget.** If sims=128 per-seed wall exceeds 8h on server,
   abort the sims=128 cell and document the wall limit. sims=128 is a
   stretch experiment - it should not block the sims=64 promotion path.
6. **Eval-MCTS lift is asymmetric.** Per Wave A1 of the recal (s31 +6.20
   -> +5.87, s42 +6.92 -> +6.69), eval-MCTS may not lift high-alpha
   ckpts. If sims=64 trains a high-alpha ckpt, the eval-MCTS lift may
   be smaller than +3.66pp. Calibrate the +3pp promotion gate against
   per-seed baseline Alpha rather than the mean.
7. **Live trader uses SIMULATIONS=24** (`config/real_trade_config.py:48`).
   Eval-MCTS at sims=16 under-shoots live by 8 sims. Operator-side
   decision: bump real_trade_config.SIMULATIONS to 16 to align with the
   new benchmark methodology, OR keep eval at sims=24 and re-eval all
   ckpts (Tier A re-run). For 760 as drafted, sims=16 is the scorecard
   for consistency with 759.

## References

- `runs/759_mcts_eval/result.md` (canonical eval-MCTS evidence).
- `runs/746_value_oracle/result.md` (canonical paired-vs-variant-only
  methodology + the closest-to-promotion mechanism so far).
- `runs/recommendation_2026-05-20/analysis.md` (campaign synthesis,
  same dir as this draft).
- `experiments/759_mcts_eval/SPEC.md` (operator's Path A SPEC, names
  Path B = this draft).
- `experiments/QUEUE.md` (operator's mechanism ordering through 2026-05-19).
- `train_phase2.py:2021` (`--mcts-sims` CLI wiring).
- `tools/benchmark_easy.py:1158` (`--eval-mcts-sims` CLI wiring).
- `phase2_config.py:33` (`mcts_sims=32` default).
- `mcts_trader/jax_agent/mctx_search.py:19` (`num_simulations` parameter
  passed to mctx.gumbel_muzero_policy).
