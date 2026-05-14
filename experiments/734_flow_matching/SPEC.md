# 734_flow_matching — TASK_734 Flow-Matching Dynamics Loss

## Goal

Test whether velocity-matching (`pred ≈ latent_{t+1} - latent_t`) gives
the dynamics MLP a stronger gradient than the current state-matching MSE.
Known MSE failure mode: identity collapse (latent_{t+1} ≈ latent_t →
loss ≈ 0 → model learns "no change"). Flow matching can't collapse this
way (identity ⇒ v=0, wrong whenever the market moves).

## Source task

`tasks/TASK_734_PREDICTION_HEAD_ALPHA.md` (status ACTIVE, α-v1 wired).

## Prerequisites

- [x] Phase 1 ckpt: `jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl`
- [x] `--flow-matching` flag wired in `train_phase2.py`
- [x] `flow_matching_velocity_weight` default 0.8 (from
      `train_phase2.py:865`)

## Resources

- 1× RTX 3090 (~21 GB, ~2h20m)
- TASK_742 rule: count toward ≤2 concurrent train_phase2 budget

## Plan

Two variants — flow_matching on top of the two head configs:

| Variant | rank | aux | flow | tag |
|---|---|---|---|---|
| `flow_a1b1` | ON | ON | ON | matches the original task's design |
| `flow_no_aux` | ON | OFF | ON | flow on top of new best |

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
  --flow-matching"

# Variant 1
mkdir -p runs/734_flow_matching/a1b1
screen -d -m -S fm_a1b1 bash -c "cd ~/Trader && source trader_env/bin/activate && \
  CUDA_VISIBLE_DEVICES=<GPU_A> python train_phase2.py $COMMON --aux-price-weight 0.1 \
  --output-dir runs/734_flow_matching/a1b1 \
  2>&1 | tee runs/734_flow_matching/a1b1/train.log; \
  echo FM_A1B1_DONE > /tmp/fm_a1b1_done.flag"

# Variant 2 (recommended)
mkdir -p runs/734_flow_matching/no_aux
screen -d -m -S fm_no_aux bash -c "cd ~/Trader && source trader_env/bin/activate && \
  CUDA_VISIBLE_DEVICES=<GPU_B> python train_phase2.py $COMMON --no-aux-price-head \
  --output-dir runs/734_flow_matching/no_aux \
  2>&1 | tee runs/734_flow_matching/no_aux/train.log; \
  echo FM_NO_AUX_DONE > /tmp/fm_no_aux_done.flag"
```

Eval pattern same as other SPECs.

## Success criteria

- [ ] Both variants complete 200 epochs cleanly
- [ ] In train.log E-lines: `flow=v<…>/s<…>` field populated
- [ ] `consistency_loss` higher than MSE-only (expected: velocity loss is
      naturally larger; total loss should still decrease)
- [ ] `dir_acc` at E199 > 0.574 for at least one variant
- [ ] Benchmark eval AVG Alpha higher than no_aux_solo baseline for the
      flow_no_aux variant

## Output expectations

```
runs/734_flow_matching/
├── a1b1/                 train.log, best.pkl, metrics.csv
├── no_aux/               train.log, best.pkl, metrics.csv
├── eval_a1b1/            benchmark.json, benchmark.log
├── eval_no_aux/          benchmark.json, benchmark.log
└── result.md             dynamics health metrics + benchmark comparison
```

## Notes

- α-v1 is a minimum-viable test. If it moves `dir_acc` above 0.60,
  escalate to α-v2 (enable `use_prediction_head=True` + wire
  NeuralMarketODE on log-price trajectory). α-v2 is a separate task.
- Flow velocity weight 0.8 means total dynamics loss is
  `0.8 * velocity_loss + 0.2 * state_loss` (both MSEs in latent space).
  Tunable via `--flow-matching-velocity-weight` if needed.
