# 743_no_aux_multiseed — Multi-seed validation of `no_aux_solo`

## Goal

Replicate today's `no_aux_solo` finding (AVG Alpha +6.21%, beats-random 6/6)
across multiple seeds. Decide whether rank-only is the new Stage-B
baseline or a single-seed outlier.

## Source task

`tasks/TASK_743_NO_AUX_MULTISEED.md`. Direct follow-up to TASK_737
ablation completion (`runs/737_eval/result.md`).

## Prerequisites

- [x] `train_phase2.py` has `--seed` CLI flag (added 2026-05-13)
- [x] `jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl` on server
- [x] `trading_data.db` on server (24 GB, Jan-20 snapshot)
- [x] `config/foxglove_periods.json` on server

## Resources

- 4× RTX 3090. Each run holds ~21 GB VRAM.
- **TASK_742 operational rule: ≤2 concurrent `train_phase2.py` per host.**
  Do NOT launch all 4 seeds in parallel — that's what triggered the
  original deadlock.
- Per-run wallclock on server: ~2h20m (pool load + 200 epochs).
- Total budget: 4 seeds × ~2.5h ÷ 2-parallel = ~5h.

## Plan

Two waves, 2 concurrent each:

```
Wave 1 (start T=0):     seed 7  on GPU 0  +  seed 13 on GPU 1
Wave 2 (start T=2.5h):  seed 23 on GPU 0  +  seed 31 on GPU 1
```

Each wave waits for the previous to complete (flag drop). GPU 2 and GPU 3
stay idle for `train_phase2` purposes — could be used for benchmark eval
of completed checkpoints in parallel with training (eval is GPU-light and
won't trigger contention).

## Commands

```bash
# Server-side (run via screen / tmux / nohup):

cd ~/Trader && source trader_env/bin/activate

SEEDS=(7 13 23 31)
GPUS=(0 1 0 1)         # wave 1: 0+1, wave 2: 0+1

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
  --rank-loss mse --rank-weight 1.0 --no-aux-price-head"

# Wave 1
mkdir -p runs/743_no_aux_multiseed/seed_7 runs/743_no_aux_multiseed/seed_13
screen -d -m -S no_aux_s7  bash -c "cd ~/Trader && source trader_env/bin/activate && \
  CUDA_VISIBLE_DEVICES=0 python train_phase2.py $COMMON --seed 7 \
  --output-dir runs/743_no_aux_multiseed/seed_7 \
  2>&1 | tee runs/743_no_aux_multiseed/seed_7/train.log; \
  echo S7_DONE > /tmp/no_aux_s7_done.flag"

screen -d -m -S no_aux_s13 bash -c "cd ~/Trader && source trader_env/bin/activate && \
  CUDA_VISIBLE_DEVICES=1 python train_phase2.py $COMMON --seed 13 \
  --output-dir runs/743_no_aux_multiseed/seed_13 \
  2>&1 | tee runs/743_no_aux_multiseed/seed_13/train.log; \
  echo S13_DONE > /tmp/no_aux_s13_done.flag"

# Wait for wave 1 to complete:
until [ -f /tmp/no_aux_s7_done.flag ] && [ -f /tmp/no_aux_s13_done.flag ]; do sleep 60; done

# Wave 2
mkdir -p runs/743_no_aux_multiseed/seed_23 runs/743_no_aux_multiseed/seed_31
screen -d -m -S no_aux_s23 bash -c "cd ~/Trader && source trader_env/bin/activate && \
  CUDA_VISIBLE_DEVICES=0 python train_phase2.py $COMMON --seed 23 \
  --output-dir runs/743_no_aux_multiseed/seed_23 \
  2>&1 | tee runs/743_no_aux_multiseed/seed_23/train.log; \
  echo S23_DONE > /tmp/no_aux_s23_done.flag"

screen -d -m -S no_aux_s31 bash -c "cd ~/Trader && source trader_env/bin/activate && \
  CUDA_VISIBLE_DEVICES=1 python train_phase2.py $COMMON --seed 31 \
  --output-dir runs/743_no_aux_multiseed/seed_31 \
  2>&1 | tee runs/743_no_aux_multiseed/seed_31/train.log; \
  echo S31_DONE > /tmp/no_aux_s31_done.flag"

until [ -f /tmp/no_aux_s23_done.flag ] && [ -f /tmp/no_aux_s31_done.flag ]; do sleep 60; done

# Eval — all 4 in parallel (eval is light, no contention risk)
for S in 7 13 23 31; do
  mkdir -p runs/743_no_aux_multiseed/eval_seed_${S}
  GPU=$(( (S/7) % 4 ))   # rough load-spread
  screen -d -m -S eval_s${S} bash -c "cd ~/Trader && source trader_env/bin/activate && \
    CUDA_VISIBLE_DEVICES=${GPU} python tools/benchmark_easy.py \
    --checkpoint runs/743_no_aux_multiseed/seed_${S}/phase2_stageB_best.pkl \
    --db trading_data.db \
    --json runs/743_no_aux_multiseed/eval_seed_${S}/benchmark.json \
    2>&1 | tee runs/743_no_aux_multiseed/eval_seed_${S}/benchmark.log; \
    echo S${S}_EVAL_DONE > /tmp/no_aux_s${S}_eval_done.flag"
done

until [ -f /tmp/no_aux_s7_eval_done.flag ] && \
      [ -f /tmp/no_aux_s13_eval_done.flag ] && \
      [ -f /tmp/no_aux_s23_eval_done.flag ] && \
      [ -f /tmp/no_aux_s31_eval_done.flag ]; do sleep 60; done

echo ALL_DONE > /tmp/no_aux_multiseed_all_done.flag
```

## Success criteria

- [ ] All 4 new seed runs complete 200 epochs (no NaN, no deadlock)
- [ ] All 4 benchmark JSONs produced
- [ ] No TASK_742 recurrence (≤2 concurrent rule held)
- [ ] result.md written with AVG / MIN / MAX / STD across 5 seeds (the 4 new + seed=42 baseline from `runs/737_ablations/no_aux_solo/`)

## Output expectations

```
runs/743_no_aux_multiseed/
├── seed_7/        train.log, phase2_stageB_best.pkl, metrics.csv
├── seed_13/       train.log, phase2_stageB_best.pkl, metrics.csv
├── seed_23/       train.log, phase2_stageB_best.pkl, metrics.csv
├── seed_31/       train.log, phase2_stageB_best.pkl, metrics.csv
├── eval_seed_7/   benchmark.json, benchmark.log
├── eval_seed_13/  benchmark.json, benchmark.log
├── eval_seed_23/  benchmark.json, benchmark.log
├── eval_seed_31/  benchmark.json, benchmark.log
└── result.md      AVG/MIN/MAX/STD over the 5 seeds (incl. seed=42 from 737_ablations)
```

## Notes

- Seed=42 result lives at `runs/737_ablations/no_aux_solo/` and
  `runs/737_eval/no_aux_solo/`; not re-run.
- TASK_742 rule strictly enforced via wave structure. Do NOT launch 4
  parallel training processes.
- If any individual seed run fails (NaN, crash, deadlock), record the
  failure mode in result.md and run a replacement seed (e.g., 37).
