# 734_flow_matching — TASK_734 Flow-Matching Dynamics Loss (paired 3-seed)

## Goal

Test whether velocity-matching (`pred ≈ latent_{t+1} - latent_t`) gives
the dynamics MLP a stronger gradient than the current state-matching MSE.
Known MSE failure mode: identity collapse (latent_{t+1} ≈ latent_t →
loss ≈ 0 → model learns "no change"). Flow matching can't collapse this
way (identity ⇒ v=0, wrong whenever the market moves).

## Source task

`tasks/TASK_734_PREDICTION_HEAD_ALPHA.md` (status ACTIVE, α-v1 wired).

## Background — multi-seed protocol

Post-TASK_743 (2026-05-14) we know single-seed Alpha gaps ≤ ~5pp are
within noise. Canonical baseline is **a1b1** (rank+aux). Every mechanism
claim needs ≥3-seed paired validation.

The a1b1 baseline runs at seeds 7/13/23 are produced by **741_mint**
Wave 1-3 (running ~08:00→15:30 today). This SPEC adds 3 variant runs and
**reuses the 741_mint baseline** for the paired comparison.

## Prerequisites

- [x] Phase 1 ckpt: `jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl`
- [x] `--flow-matching` flag wired in `train_phase2.py`
- [x] `flow_matching_velocity_weight` default 0.8
- [ ] a1b1 baseline runs at seeds {7,13,23} — produced by
      `runs/741_mint/baseline_seed_{7,13,23}/`. Verify these exist
      before launching.

## Resources

- 1× RTX 3090 per training process, ~21 GB, ~2h20m
- TASK_742 rule: ≤2 concurrent train_phase2 per host
- Total: 3 variant runs × 2.5h with 2-wide batching = **~3.75h wall**
  (baseline reused)

## Plan

| Variant | rank | aux | flow | seeds | source |
|---|---|---|---|---|---|
| **a1b1 baseline** | ON | ON | OFF | 7, 13, 23 | reuse from 741_mint |
| **flow_a1b1** | ON | ON | ON (α-v1) | 7, 13, 23 | new runs in this SPEC |

## Wave plan

```
Wave 1 (T=0:00, ~2.5h):
  GPU A:  flow_a1b1 seed 7
  GPU B:  flow_a1b1 seed 13

Wave 2 (T=2:30, ~1.25h):
  GPU A:  flow_a1b1 seed 23

T=3:45 — launch 3 evals in parallel
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

# For SEED in {7, 13, 23}, launch on free GPU under TASK_742 rule:
mkdir -p runs/734_flow_matching/flow_seed_${SEED}
screen -d -m -S fm_s${SEED} bash -c "cd ~/Trader && source trader_env/bin/activate && \
  CUDA_VISIBLE_DEVICES=<GPU> python train_phase2.py $COMMON \
  --seed ${SEED} \
  --output-dir runs/734_flow_matching/flow_seed_${SEED} \
  --name 734_fm_s${SEED} 2>&1 \
  | tee runs/734_flow_matching/flow_seed_${SEED}/train.log; \
  echo FM_S${SEED}_DONE > /tmp/fm_s${SEED}_done.flag"
```

Eval pattern same as 741_mint final block — point at new checkpoints +
shared baselines from `runs/741_mint/baseline_seed_*/`.

## Success criteria

- [ ] All 3 training runs complete 200 epochs cleanly
- [ ] `flow=v<…>/s<…>` field populated in E-line metrics
- [ ] `consistency_loss` higher than MSE-only baseline (expected:
      velocity loss is naturally larger; total loss should still decrease)
- [ ] **Paired Alpha delta** vs baseline (per seed): mean(Δα) over 3 seeds:
      - mean(Δα) > +3pp AND all 3 positive → **CLEAR WIN**; promote
        flow-matching as default dynamics loss
      - mean(Δα) ∈ (−2, +3)pp → **inconclusive**; add seeds 31, 42
      - mean(Δα) < −2pp → **regression**; check if velocity weight 0.8
        is over-weighted vs state-matching
- [ ] Dynamics-MLP identity-collapse symptom (state_loss → ~0 by E50)
      DOES NOT appear in flow runs

## Output expectations

```
runs/734_flow_matching/
├── flow_seed_7/         train.log, best.pkl, metrics.csv
├── flow_seed_13/
├── flow_seed_23/
├── eval_flow_seed_7/    benchmark.json, benchmark.log
├── eval_flow_seed_13/
├── eval_flow_seed_23/
└── result.md            paired delta table + dynamics-health metrics
```

## Notes

- α-v1 is the minimum-viable test. If 2/3 seeds move `dir_acc` above
  0.60, escalate to α-v2 (enable `use_prediction_head=True` + wire
  NeuralMarketODE on log-price trajectory). α-v2 is a separate task.
- Flow velocity weight 0.8 means total dynamics loss is
  `0.8 * velocity_loss + 0.2 * state_loss`. Tunable via
  `--flow-matching-velocity-weight` if needed.
- If baseline reuse not viable, prepend 3 baseline runs without
  `--flow-matching` — adds 3.75h.
