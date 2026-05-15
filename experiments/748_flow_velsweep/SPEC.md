# 748_flow_velsweep — Flow-Matching Velocity-Weight Sweep (2 single-seed probes)

## Goal

Bracket the `--flow-matching-velocity-weight` default (0.8) used in 734 to check whether the choice was lucky or robust. Probes two endpoints:
- `0.5` — equal balance between velocity and state-matching loss
- `1.0` — pure velocity (state-matching dropped entirely)

If either probe is materially different (|Δα| > 3pp) from 734's seed-matched baseline, follow up with a paired 3-seed at that weight. Otherwise the default 0.8 stands.

## Background

734_flow_matching's paired 3-seed (2026-05-15) gave mean Δα +3.38pp with default weight 0.8. We don't know if a different weight gives a larger or smaller effect. Cheap to find out.

Single-seed is justified here because:
1. We're not making a promotion decision — only sanity-checking the weight choice for the 5-seed extension currently running.
2. The 734 main result already validated the mechanism class at 3 seeds; only the hyperparameter is in question.
3. If the sweep shows nothing surprising, default 0.8 stands; if it does show something, follow up with 3-seed.

## Hypothesis

**Most likely**: Both 0.5 and 1.0 give Δα within ±2pp of 0.8's result (i.e. weight is in a wide-flat-optimum region). Promote 734 with default 0.8.

**Alternative**: One endpoint is materially better (e.g. 1.0 = pure velocity makes the dynamics MLP train more cleanly, identity collapse fully impossible). Worth a 3-seed confirmation.

## Prerequisites

- [x] Phase 1 ckpt: `jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl`
- [x] `--flow-matching-velocity-weight` wired in `train_phase2.py`
- [x] 734_flow_matching seed 7 baseline (`runs/734_flow_matching/flow_seed_7/`) — pairs against this for matched-seed delta

## Resources

- 2 runs × 2.5h = 5h sequential or **~2.5h wall with 2-wide**
- GPUs **1+2** (NOT 0)
- No 740_bella interference (different GPU; 740 won't be running yet)

## Plan

| Variant | velocity weight | seed | pairs against |
|---|---|---|---|
| flow_v05 | 0.5 | 7 | `runs/734_flow_matching/flow_seed_7/` (weight 0.8) |
| flow_v10 | 1.0 | 7 | `runs/734_flow_matching/flow_seed_7/` (weight 0.8) |

Single seed (7) is the only fair pair against 734's existing data — same data ordering, same init randomness modulo the loss weight.

## Wave plan

Best run AFTER 746/747 finish their main waves (GPUs 1+2 free again), so it doesn't interleave with the value-target experiments.

```
Wave 1 (T=0:00, ~2.5h):
  GPU 1: flow_v05_seed_7  (--flow-matching --flow-matching-velocity-weight 0.5)
  GPU 2: flow_v10_seed_7  (--flow-matching --flow-matching-velocity-weight 1.0)

T=2:30 — launch 2 evals
```

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
  --aux-price-weight 0.1 \
  --flow-matching"

# Variant 0.5 on GPU 1:
mkdir -p runs/748_flow_velsweep/flow_v05_seed_7
screen -d -m -S fv05_s7 bash -c "source trader_env/bin/activate && \
  CUDA_VISIBLE_DEVICES=1 python -u train_phase2.py $COMMON \
  --flow-matching-velocity-weight 0.5 --seed 7 \
  --output-dir runs/748_flow_velsweep/flow_v05_seed_7 \
  --name 748_fv05_s7 2>&1 \
  | tee runs/748_flow_velsweep/flow_v05_seed_7/train.log; \
  echo FV05_S7_DONE > /tmp/fv05_s7_done.flag"

# Variant 1.0 on GPU 2:
mkdir -p runs/748_flow_velsweep/flow_v10_seed_7
screen -d -m -S fv10_s7 bash -c "source trader_env/bin/activate && \
  CUDA_VISIBLE_DEVICES=2 python -u train_phase2.py $COMMON \
  --flow-matching-velocity-weight 1.0 --seed 7 \
  --output-dir runs/748_flow_velsweep/flow_v10_seed_7 \
  --name 748_fv10_s7 2>&1 \
  | tee runs/748_flow_velsweep/flow_v10_seed_7/train.log; \
  echo FV10_S7_DONE > /tmp/fv10_s7_done.flag"
```

Eval each new checkpoint against the same 6-period benchmark.

## Success criteria

- [ ] Both runs complete 200 epochs cleanly
- [ ] `flow=v<…>/s<…>` field shows velocity component dominating in v=1.0 run (state loss should be flat)
- [ ] **Δα per probe** vs `runs/734_flow_matching/eval_flow_seed_7/`:
      - |Δα| < 3pp → weight choice is robust; default 0.8 stands
      - |Δα| ≥ 3pp on either side → follow up with paired 3-seed at the winning weight
- [ ] No NaN, no identity collapse in either configuration

## Output expectations

```
runs/748_flow_velsweep/
├── flow_v05_seed_7/    train.log, phase2_stageB_best.pkl, stage_b_metrics.csv
├── flow_v10_seed_7/
├── eval_flow_v05_seed_7/ benchmark.json
├── eval_flow_v10_seed_7/ benchmark.json
└── result.md           Δα at each weight + identity-collapse check
```

## Notes

- This is a CHEAP probe — single-seed comparisons are noise-limited at ~5pp. We're looking for large effects, not subtle ones.
- If 734's 5-seed extension (in flight) lands in the "promote" bucket, this sweep is what decides whether to retrain the canonical at a different weight before promotion.
- If 734's 5-seed lands in "inconclusive" bucket and this sweep finds a much better weight, it could reframe the whole TASK_734 question.
