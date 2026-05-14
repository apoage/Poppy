# 740_bella_distill — TASK_740 Bella Distillation Head

## Goal

Inject Bella's forward-rollout log-return predictions as a distillation
target to give the agent *new information* beyond the 60-step OHLCV
input. Targets the information-ceiling hypothesis behind the 0.574
dir_acc cap.

## Source task

`tasks/TASK_740_BELLA_DISTILL_HEAD.md` (status IMPLEMENTED 2026-04-23).

## Prerequisites

- [x] Phase 1 ckpt: `jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl`
- [x] Bella generator ckpt: `checkpoints/diff_teacher/generator_best/generator`
      (or equivalent — original task uses `generator_best/`; server has
      `final/generator/`. Verify path before launch.)
- [x] `--bella-distill-head --bella-distill-weight 0.1` and
      `--bella-preds-manifest` flags wired
- [ ] **`config/bella_preds_manifest.json` and Bella prediction shards
      must be PRECOMPUTED before training**. See "Step 1" below.

## Resources

- Precompute: 1× RTX 3090, ~30-60 min wallclock (depends on K samples and
  total windows across all training periods)
- Training: 1× RTX 3090, ~21 GB VRAM, ~2h20m total
- TASK_742 rule: count toward ≤2 concurrent train_phase2 budget

## Plan

**Step 1 — Bella precompute (gates training).**

```bash
cd ~/Trader && source trader_env/bin/activate

# Smoke-test one window:
python tools/precompute_bella_predictions.py --dry-run

# Full precompute:
python tools/precompute_bella_predictions.py \
  --bella-checkpoint checkpoints/diff_teacher/generator_best/generator \
  --periods-file config/foxglove_periods.json \
  --output-manifest config/bella_preds_manifest.json \
  2>&1 | tee logs/bella_precompute.log
```

If the dry-run reveals issues (path, generator config), STOP and report —
don't proceed to training.

**Step 2 — Training, two variants.**

| Variant | rank | aux | bella_distill | tag |
|---|---|---|---|---|
| `bella_a1b1` | ON | ON | ON (w=0.1) | matches `tools/run_740_bella_distill.sh` |
| `bella_no_aux` | ON | OFF | ON (w=0.1) | bella_distill on top of new best |

```bash
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
  --bella-distill-head --bella-distill-weight 0.1 \
  --bella-preds-manifest config/bella_preds_manifest.json"

# Variant 1: bella + a1b1
mkdir -p runs/740_bella_distill/a1b1
screen -d -m -S bd_a1b1 bash -c "cd ~/Trader && source trader_env/bin/activate && \
  CUDA_VISIBLE_DEVICES=<GPU_A> python train_phase2.py $COMMON --aux-price-weight 0.1 \
  --output-dir runs/740_bella_distill/a1b1 \
  2>&1 | tee runs/740_bella_distill/a1b1/train.log; \
  echo BD_A1B1_DONE > /tmp/bd_a1b1_done.flag"

# Variant 2: bella + no_aux (recommended)
mkdir -p runs/740_bella_distill/no_aux
screen -d -m -S bd_no_aux bash -c "cd ~/Trader && source trader_env/bin/activate && \
  CUDA_VISIBLE_DEVICES=<GPU_B> python train_phase2.py $COMMON --no-aux-price-head \
  --output-dir runs/740_bella_distill/no_aux \
  2>&1 | tee runs/740_bella_distill/no_aux/train.log; \
  echo BD_NO_AUX_DONE > /tmp/bd_no_aux_done.flag"
```

**Step 3 — Benchmark eval.**

Same pattern as `739_dense_sign/SPEC.md` final block — point at the new
checkpoints.

## Success criteria

- [ ] Precompute completes, manifest valid, all shards on disk
- [ ] Both training variants complete 200 epochs cleanly
- [ ] `bella_distill_loss` in metrics (visible in train.log E-lines as
      `bella=…(vf=…)`)
- [ ] `bella_valid_frac` (vf=) > 0.5 most of training — confirms manifest
      coverage is adequate
- [ ] Benchmark eval AVG Alpha higher than no_aux_solo baseline (+6.21%)
      for at least one variant → bella_distill broke the 60-step ceiling
- [ ] result.md compares both variants vs TASK_736 ref + no_aux_solo

## Output expectations

```
runs/740_bella_distill/
├── a1b1/                   train.log, best.pkl, metrics.csv
├── no_aux/                 train.log, best.pkl, metrics.csv
├── eval_a1b1/              benchmark.json, benchmark.log
├── eval_no_aux/            benchmark.json, benchmark.log
└── result.md               comparison vs baselines

config/bella_preds_manifest.json   ← shared across future Bella runs

logs/bella_precompute.log
```

## Notes

- The Bella checkpoint path on workstation may differ from server (see
  `tools/run_740_bella_distill.sh:25` for the canonical path
  `checkpoints/diff_teacher/generator_best/generator`). Server has
  `final/generator/` (orbax format). Verify before launch.
- Bella has no native input-conditioning. The precompute uses regime
  estimation from window stats to pick an unconditional Bella regime,
  then averages K rollouts. See `tools/precompute_bella_predictions.py`
  header for the method.
- `--bella-distill-weight 0.1` is the spec default. If bella_distill_loss
  is much smaller than other losses, may need increase to e.g. 0.3.
