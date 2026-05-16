# 752_value_oracle_baseline_pair — a1b1 baselines at s31, s42 for 746 5-seed

## Goal

Run a1b1 baseline at seeds 31 and 42 to complete a properly paired 5-seed Δα for 746_value_oracle. Currently 746-x5 (queued separately) runs only the variant; this SPEC produces the matching baseline runs so the 5-seed Δα is paired-clean instead of variant-only (the lesson from 734-x5).

## Background

734's 5-seed extension was run variant-only. Its mean Δα +0.20pp used the assumption "baseline at s31/s42 ≈ 3-seed baseline mean." The variant-only assumption ate up to ~5pp of uncertainty per seed — for a single-outlier seed (s42 −9.91% absolute), the verdict ("not promote") was robust to that uncertainty, but for a closer signal it might not be.

746's 5-seed extension needs to be cleaner. Mean Δα at 3 seeds was +2.54pp (just below the +3pp threshold). If the 5-seed extension lands close to threshold, variant-only ambiguity could be decision-changing.

## Prerequisites

- [x] Phase 1 ckpt: `jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl`
- [x] 746_value_oracle/eval_oracle_seed_7/13/23 complete (the 3-seed paired against existing a1b1 baselines)
- [ ] 746-x5 extension (variant-only runs at s31, s42) — running or queued separately

## Resources

- 1× RTX 3090, GPU 2 server (sequential after 751 finishes, OR GPU 3 if 751 still running)
- Wall: 2 runs × 3h = **~6h sequential** (one GPU)
- TASK_742 rule: each is 1 of the 2-concurrent budget

## Plan

Two baseline a1b1 runs at seeds 31, 42. NO Titans, NO oracle, NO flow-matching — pure a1b1 reference matching the existing baseline_seed_7/13/23 in `runs/741_mint/`.

| Variant | seeds | titans | oracle |
|---|---|---|---|
| a1b1 baseline | 31, 42 | OFF | nstep (default) |

## Commands

```bash
cd ~/Trader && source trader_env/bin/activate

COMMON="--stage B --epochs 200 \
  --periods-file config/foxglove_periods.json \
  --n-slots 17 --max-runs-per-group 10 \
  --factored --factored-temperature 0.1 \
  --no-reset-value-head --value-target-mode nstep --td-discount 0.95 \
  --value-weight 0.5 \
  --bc-weight-min 0.05 --bc-max-trade-weight 4.0 \
  --direction-weight 0.5 --encoder-lr 3e-5 \
  --encoder-version 3 --embed-dim 64 --encoder-heads 8 \
  --checkpoint jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl \
  --rank-loss mse --rank-weight 1.0 --aux-price-weight 0.1"

# For SEED in {31, 42}, run sequentially on GPU 2 or GPU 3:
for SEED in 31 42; do
  mkdir -p runs/752_value_oracle_baseline_pair/baseline_seed_${SEED}
  screen -d -m -S vobl_s${SEED} bash -c "source trader_env/bin/activate && \
    CUDA_VISIBLE_DEVICES=<GPU> python -u train_phase2.py $COMMON \
    --seed ${SEED} \
    --output-dir runs/752_value_oracle_baseline_pair/baseline_seed_${SEED} \
    --name 752_vobl_s${SEED} 2>&1 \
    | tee runs/752_value_oracle_baseline_pair/baseline_seed_${SEED}/train.log; \
    echo VOBL_S${SEED}_DONE > /tmp/vobl_s${SEED}_done.flag"
done
```

Eval each ckpt against the standard 6-period benchmark:

```bash
for SEED in 31 42; do
  mkdir -p runs/752_value_oracle_baseline_pair/eval_baseline_seed_${SEED}
  CUDA_VISIBLE_DEVICES=<GPU> python tools/benchmark_easy.py \
    --checkpoint runs/752_value_oracle_baseline_pair/baseline_seed_${SEED}/phase2_stageB_best.pkl \
    --db trading_data.db \
    --json runs/752_value_oracle_baseline_pair/eval_baseline_seed_${SEED}/benchmark.json
done
```

## Success criteria

- [ ] Both runs complete 200 epochs cleanly
- [ ] Eval JSONs produce `summary.avg_alpha` finite per seed
- [ ] Output usable as paired baseline for 746-x5: workstation post-hoc computes properly paired 5-seed Δα for 746 using:
      - `runs/741_mint/baseline_seed_{7,13,23}/eval_baseline_seed_*` (3 existing)
      - `runs/752_value_oracle_baseline_pair/eval_baseline_seed_{31,42}/` (2 new)
      - paired against `runs/746_value_oracle/eval_oracle_seed_{7,13,23}/` + the 746-x5 results at s31, s42

## Output expectations

```
runs/752_value_oracle_baseline_pair/
├── baseline_seed_31/  train.log, phase2_stageB_best.pkl, stage_b_metrics.csv
├── baseline_seed_42/
├── eval_baseline_seed_31/ benchmark.json
├── eval_baseline_seed_42/ benchmark.json
```

No result.md needed — these are baseline runs, not a comparative experiment. The properly paired 5-seed 746 verdict is computed post-hoc by the workstation operator (or appended to `runs/746_value_oracle/result.md`).

## Notes

- **Why run separately from 746-x5**: 746-x5 was already speced as variant-only. We could refactor 746-x5 to include the baselines but that duplicates work in flight. Cleaner to leave 746-x5 as-is and add this SPEC as the missing pair.
- **Baseline reuse**: once `baseline_seed_31` and `baseline_seed_42` exist, they're reusable for ANY future 5-seed experiment that needs paired baselines at those seeds (not just 746). Investment compounds.
- **Why not seeds 7/13/23 too**: those already exist at `runs/741_mint/baseline_seed_*`. Same code, same hyperparameters, valid reference.
