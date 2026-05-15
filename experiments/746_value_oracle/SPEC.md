# 746_value_oracle — Oracle Value Targets (paired 3-seed)

## Goal

Re-test **`--value-target-mode oracle`** (TASK_719a) under the current paired-Δα protocol. Oracle uses Foxglove's path-solver V* directly as the value target (no TD bootstrap, no n-step rewards). Pure value-side change — does NOT touch encoder, classifier, dynamics, or policy.

Falls into the **planning-side mechanism class** that 734_flow_matching is the existence-proof for: improvements that target *what the model believes is reachable*, not *how well it predicts the next bar*.

## Background — historical caveat

TASK_719a was implemented + tested in April 2026 and called "DONE (suboptimal, plateaued at value_loss=4)" — n-step (TASK_719b) was selected as canonical Stage-B on **value_loss comparison alone**. See `lore/notes_2026-04-08.md`.

**Why re-test:**
1. TASK_741 (2026-05-14) demonstrated that training-side metrics (loss, Bal.Acc) are **anti-correlated with Alpha** on this stack. The April rejection of oracle on value_loss is no longer authoritative.
2. We now use paired Δα over 3 seeds against a fixed baseline — much higher S/N than April single-runs.
3. Stage-B stack has changed substantially since April (v3 encoder, factored MCTS, TASK_732 cross-sectional channels, TASK_736 BC-weight tuning, TASK_743 baseline established).
4. Oracle is the **least biased value signal** possible — if a planning-side fix helps Alpha, this should be the cleanest one.

## Hypothesis

Oracle value targets reduce policy aggressiveness in trending regimes (where n-step bootstrap over-estimates V from inflated future rewards), specifically attacking the bull/ftx_crash over-trading-collapse failure mode that all classifier-side mechanisms hit.

**Falsification**: if oracle gives mean Δα < −2pp paired against n-step baseline, the value-side roadmap is closed and we focus elsewhere (Bella distillation, MCTS hyperparameters, policy entropy).

## Prerequisites

- [x] Phase 1 ckpt: `jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl`
- [x] `--value-target-mode oracle` wired in `train_phase2.py:955-961` (TASK_719a code, unchanged)
- [x] `oracle_values` field populated in SegmentPool batches
- [ ] a1b1 baseline runs at seeds {7,13,23} from `runs/741_mint/baseline_seed_{7,13,23}/`

## Resources

- 1× RTX 3090 per training, ~21 GB, ~2.5h
- TASK_742 rule: ≤2 concurrent train_phase2 per host
- GPUs **1+2** (NOT 0 — VirtualBox pinned; per `feedback_gpu0_avoid.md`)
- Total: 3 variant runs × 2.5h with 2-wide = **~3.75h wall** (baseline reused)

## Plan

| Variant | value target | seeds | source |
|---|---|---|---|
| **a1b1 baseline** | nstep, td-discount 0.95 | 7, 13, 23 | reuse `runs/741_mint/baseline_seed_*/` |
| **oracle** | oracle (V* from path-solver) | 7, 13, 23 | new runs in this SPEC |

## Wave plan

```
Wave 1 (T=0:00, ~2.5h):
  GPU 1: oracle_seed_7
  GPU 2: oracle_seed_13

Wave 2 (T=2:30, ~1.25h):
  GPU 1: oracle_seed_23
  GPU 2: idle (or pick up 747_value_td0)

T=3:45 — launch 3 evals in parallel on GPUs 1+2
```

## Commands

```bash
cd ~/Trader && source trader_env/bin/activate

# COMMON is identical to 734/741 baseline EXCEPT --value-target-mode flipped:
COMMON="--stage B --epochs 200 \
  --periods-file config/foxglove_periods.json \
  --n-slots 17 --max-runs-per-group 10 \
  --factored --factored-temperature 0.1 \
  --no-reset-value-head --value-target-mode oracle \
  --value-weight 0.5 \
  --bc-weight-min 0.05 --bc-max-trade-weight 4.0 \
  --direction-weight 0.5 --encoder-lr 3e-5 \
  --encoder-version 3 --embed-dim 64 --encoder-heads 8 \
  --checkpoint jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl \
  --rank-loss mse --rank-weight 1.0 \
  --aux-price-weight 0.1"
# NOTE: --td-discount removed (oracle mode ignores it)

# For SEED in {7, 13, 23}, GPU in {1, 2} under TASK_742 rule:
mkdir -p runs/746_value_oracle/oracle_seed_${SEED}
screen -d -m -S vo_s${SEED} bash -c "source trader_env/bin/activate && \
  CUDA_VISIBLE_DEVICES=<GPU> python -u train_phase2.py $COMMON \
  --seed ${SEED} \
  --output-dir runs/746_value_oracle/oracle_seed_${SEED} \
  --name 746_vo_s${SEED} 2>&1 \
  | tee runs/746_value_oracle/oracle_seed_${SEED}/train.log; \
  echo VO_S${SEED}_DONE > /tmp/vo_s${SEED}_done.flag"
```

Eval pattern matches 734_flow_matching's: `tools/benchmark_easy.py --checkpoint <oracle_seed_N>/phase2_stageB_best.pkl --json <eval_dir>/benchmark.json` against the same 6-period benchmark set.

## Success criteria

- [ ] All 3 training runs complete 200 epochs cleanly
- [ ] No NaN losses, no rank-head deadlocks
- [ ] **value_loss trajectory**: track if value_loss plateaus higher than n-step (April finding: ~4 vs ~2 for n-step). If it does, that's expected — what matters now is Δα, not value_loss.
- [ ] **Paired Δα** vs 741_mint baseline (per seed):
      - mean(Δα) > +3pp AND all 3 positive → **CLEAR WIN**; promote oracle as canonical value-target
      - mean(Δα) ∈ (−2, +3)pp → **inconclusive**; add seeds 31, 42 for 5-seed
      - mean(Δα) < −2pp → **regression**; closes value-side roadmap, move to MCTS/policy
- [ ] Trade-rate check: per-period trade counts in benchmark should NOT explode like 738/741 (over-trading-collapse signature)

## Output expectations

```
runs/746_value_oracle/
├── oracle_seed_{7,13,23}/   train.log, phase2_stageB_best.pkl, stage_b_metrics.csv
├── eval_oracle_seed_{7,13,23}/ benchmark.json, benchmark.log
└── result.md                paired Δα table + value_loss comparison + trade-rate per period
```

## Notes

- This SPEC is independent of 734_flow_matching. If flow_matching wins its 5-seed extension and gets promoted to canonical, re-baseline this experiment against flow+a1b1.
- Multi-mechanism stack: `--value-target-mode oracle --flow-matching` is the natural Tier-3 follow-up if BOTH win independently.
- Path-solver V* is bounded in [−10, +10] post-clip (line 958) — same range as the n-step targets, so PopArt scaling is comparable.
