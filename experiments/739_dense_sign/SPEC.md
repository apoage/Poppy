# 739_dense_sign — TASK_739 Dense Sign-of-Forward-Return Oracle (paired 3-seed)

## Goal

Test whether replacing the sparse top-k HOLD oracle (93.75% HOLD by
construction) with a dense per-asset sign label breaks the 0.574 dir_acc
ceiling. Targets the per-asset directional supervision gap identified in
TASK_733.

## Source task

`tasks/TASK_739_DENSE_SIGN_ORACLE.md` (status IMPLEMENTED, ready to run).

## Background — multi-seed protocol

Post-TASK_743 (2026-05-14) we know single-seed Alpha gaps ≤ ~5pp are
within noise on this setup. Canonical baseline is **a1b1** (rank+aux).
Every mechanism claim needs ≥3-seed paired validation.

The a1b1 baseline runs at seeds 7/13/23 are produced by **741_mint**
Wave 1-3 (running ~08:00→15:30 today). This SPEC adds 3 variant runs and
**reuses the 741_mint baseline** for the paired comparison.

## Prerequisites

- [x] Phase 1 ckpt: `jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl`
- [x] `--oracle-mode dense_sign --sign-epsilon 7.5e-3` flags wired
- [x] Sign epsilon calibrated 2026-04-24 (~13% trade density across 3
      Foxglove periods, vs the unstable 68% from ε=1e-4)
- [ ] a1b1 baseline runs at seeds {7,13,23} available — produced by
      `runs/741_mint/baseline_seed_{7,13,23}/`. **Verify these exist
      before launching** (or re-create with the launch block below).

## Resources

- 1× RTX 3090 per training process, ~21 GB, ~2h20m
- TASK_742 rule: ≤2 concurrent train_phase2.py per host
- Total: 3 variant runs × 2.5h with 2-wide batching = **~3.75h wall**
  (if baseline reused from 741_mint), or **~7.5h** if baseline reruns

## Plan

| Variant | rank | aux | oracle | seeds | source |
|---|---|---|---|---|---|
| **a1b1 baseline** | ON | ON | sparse top-k | 7, 13, 23 | reuse from 741_mint |
| **dense_sign_a1b1** | ON | ON | dense_sign (ε=7.5e-3) | 7, 13, 23 | new runs in this SPEC |

Single variant only — dense_sign on top of a1b1, multi-seeded. Paired
comparison against shared baseline.

## Wave plan

Run all 3 variant seeds with **2-wide** batching, fitting under the
TASK_742 rule. The 741_mint training must be finished (or have free
slots) before launching.

```
Wave 1 (T=0:00, ~2.5h):
  GPU A:  dense_sign_a1b1 seed 7
  GPU B:  dense_sign_a1b1 seed 13

Wave 2 (T=2:30, ~1.25h with 1 train_phase2):
  GPU A:  dense_sign_a1b1 seed 23

T=3:45 — launch 3 evals in parallel (GPU-light)
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
  --oracle-mode dense_sign --sign-epsilon 7.5e-3"

# For SEED in {7, 13, 23}, launch on free GPU under TASK_742 rule:
mkdir -p runs/739_dense_sign/dense_sign_seed_${SEED}
screen -d -m -S ds_s${SEED} bash -c "cd ~/Trader && source trader_env/bin/activate && \
  CUDA_VISIBLE_DEVICES=<GPU> python train_phase2.py $COMMON \
  --seed ${SEED} \
  --output-dir runs/739_dense_sign/dense_sign_seed_${SEED} \
  --name 739_ds_s${SEED} 2>&1 \
  | tee runs/739_dense_sign/dense_sign_seed_${SEED}/train.log; \
  echo DS_S${SEED}_DONE > /tmp/ds_s${SEED}_done.flag"
```

Eval pattern same as `741_mint/SPEC.md` final block — point at the new
checkpoints + the shared baselines from `runs/741_mint/baseline_seed_*/`.

## Success criteria

- [ ] All 3 variant training runs complete 200 epochs cleanly
- [ ] `dir_acc` at E199 > 0.60 for ≥2/3 seeds (break the 0.574 ceiling
      reliably, not just by single-seed luck)
- [ ] **Paired Alpha delta** vs baseline (per seed): mean(Δα) over 3
      seeds:
      - mean(Δα) > +3pp AND all 3 deltas positive → **CLEAR WIN**;
        promote dense_sign to default oracle
      - mean(Δα) ∈ (−2, +3)pp → **inconclusive**; add seeds 31, 42
      - mean(Δα) < −2pp → **clear regression**; abandon and investigate
        epsilon calibration drift
- [ ] **Paired dir_acc delta** vs baseline ≥ +5pp on 2/3 seeds (the
      hypothesis is about dir_acc primarily, not Alpha)

## Output expectations

```
runs/739_dense_sign/
├── dense_sign_seed_7/        train.log, best.pkl, metrics.csv
├── dense_sign_seed_13/
├── dense_sign_seed_23/
├── eval_dense_sign_seed_7/   benchmark.json, benchmark.log
├── eval_dense_sign_seed_13/
├── eval_dense_sign_seed_23/
└── result.md                 paired delta table (dir_acc + Alpha) vs
                              runs/741_mint/baseline_seed_*/
```

## Notes

- dense_sign reframes the oracle: every asset, every step, gets a
  directional label derived from `sign(future_log_return)` with ε
  dead-zone. HOLD becomes a learned class.
- Effective label count: ~4 per step (top-k) → 64 per step (per-asset).
- ε=7.5e-3 was calibrated (`tools/calibrate_739_epsilon.py`). Don't
  change without re-calibrating.
- If 741_mint baseline seeds aren't available, prepend 3 baseline runs
  (omit `--oracle-mode` and `--sign-epsilon` flags) — adds 3.75h wall.
