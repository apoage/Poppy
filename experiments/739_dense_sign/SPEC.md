# 739_dense_sign — TASK_739 Dense Sign-of-Forward-Return Oracle

## Goal

Test whether replacing the sparse top-k HOLD oracle (93.75% HOLD by
construction) with a dense per-asset sign label breaks the 0.574 dir_acc
ceiling. Targets the per-asset directional supervision gap identified in
TASK_733.

## Source task

`tasks/TASK_739_DENSE_SIGN_ORACLE.md` (status IMPLEMENTED, ready to run).

## Prerequisites

- [x] Phase 1 ckpt: `jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl`
- [x] `--oracle-mode dense_sign --sign-epsilon 7.5e-3` flags wired in
      `train_phase2.py:1780-1785`
- [x] Sign epsilon calibrated 2026-04-24 (~13% trade density across 3
      Foxglove periods, vs the unstable 68% from ε=1e-4)

## Resources

- 1× RTX 3090 (~21 GB VRAM, ~2h20m total).
- TASK_742 rule: count toward ≤2 concurrent train_phase2 budget.

## Plan

Two variants, both stack dense_sign onto the head configs:

| Variant | rank | aux | oracle | tag |
|---|---|---|---|---|
| `dense_sign_a1b1` | ON | ON | dense_sign | matches `tools/run_739_dense_sign.sh` exactly |
| `dense_sign_no_aux` | ON | OFF | dense_sign | dense_sign on top of today's new best |

The second variant is the more informative one given TASK_737's result —
it stacks the new oracle onto the new best head config.

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
  --rank-loss mse --rank-weight 1.0 \
  --oracle-mode dense_sign --sign-epsilon 7.5e-3"

# dense_sign + a1b1 (matches the original run_739 script)
mkdir -p runs/739_dense_sign/a1b1
screen -d -m -S ds_a1b1 bash -c "cd ~/Trader && source trader_env/bin/activate && \
  CUDA_VISIBLE_DEVICES=<GPU_A> python train_phase2.py $COMMON --aux-price-weight 0.1 \
  --output-dir runs/739_dense_sign/a1b1 \
  2>&1 | tee runs/739_dense_sign/a1b1/train.log; \
  echo DS_A1B1_DONE > /tmp/ds_a1b1_done.flag"

# dense_sign + no_aux (recommended: stacks on today's new best)
mkdir -p runs/739_dense_sign/no_aux
screen -d -m -S ds_no_aux bash -c "cd ~/Trader && source trader_env/bin/activate && \
  CUDA_VISIBLE_DEVICES=<GPU_B> python train_phase2.py $COMMON --no-aux-price-head \
  --output-dir runs/739_dense_sign/no_aux \
  2>&1 | tee runs/739_dense_sign/no_aux/train.log; \
  echo DS_NO_AUX_DONE > /tmp/ds_no_aux_done.flag"
```

Pick `<GPU_A>` and `<GPU_B>` based on queue scheduling (see
`experiments/QUEUE.md` for batching).

After training:

```bash
for variant in a1b1 no_aux; do
  mkdir -p runs/739_dense_sign/eval_${variant}
  CUDA_VISIBLE_DEVICES=0 python tools/benchmark_easy.py \
    --checkpoint runs/739_dense_sign/${variant}/phase2_stageB_best.pkl \
    --db trading_data.db \
    --json runs/739_dense_sign/eval_${variant}/benchmark.json \
    2>&1 | tee runs/739_dense_sign/eval_${variant}/benchmark.log
done
```

## Success criteria

- [ ] Both variants complete 200 epochs cleanly
- [ ] `dir_acc` at E199 > 0.60 for at least one variant (the 0.574 ceiling break is the goal)
- [ ] Benchmark eval produces 6 periods × 2 variants
- [ ] result.md compares both variants vs:
      - TASK_736 reference (rank+aux, sparse oracle: 0.265 / +3.87%)
      - no_aux_solo baseline (rank only, sparse oracle: 0.266 / +6.21%)

## Output expectations

```
runs/739_dense_sign/
├── a1b1/             train.log, best.pkl, metrics.csv
├── no_aux/           train.log, best.pkl, metrics.csv
├── eval_a1b1/        benchmark.json, benchmark.log
├── eval_no_aux/      benchmark.json, benchmark.log
└── result.md         dir_acc comparison + benchmark metrics
```

## Notes

- dense_sign reframes the oracle: every asset, every step, gets a
  directional label derived from `sign(future_log_return)` with ε
  dead-zone. HOLD becomes a learned class (was a marginal artifact).
- Effective label count: ~4 per step (top-k) → 64 per step (per-asset).
- `tools/run_739_dense_sign.sh` is the reference single-variant
  workstation runner — uses a1b1 baseline. This SPEC adds the no_aux
  companion variant.
- ε=7.5e-3 was calibrated (`tools/calibrate_739_epsilon.py`). Don't
  change without re-calibrating.
