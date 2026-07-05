# 737_ablations — TASK_737 ablation suite (3 of 4 variants)

## Goal

Run the TASK_737 ablation matrix to isolate which of TASK_730 (rank head),
TASK_735 (aux price head), and "value-weight-fix alone" contributes to the
TASK_736 baseline (Alpha +3.87% AVG, Bal.Acc 26.5%, beats-random 6/6). 4 variants
are defined in the task; 3 are runnable now (the `legacy_27ch` 4th variant
needs a 27-ch Phase 1 checkpoint that does not exist on the workstation —
deferred).

## Source task

`tasks/TASK_737_ABLATION_SUITE.md` (workstation-side, not deployed).
Reference shell runner: `tools/run_737_ablations.sh`.

## Prerequisites (operator-verified before deploy)

- [x] `checkpoints/diff_teacher/final/` present (Bella inference verified in 000_smoke)
- [ ] `jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl` (98 MB) rsync'd
      to server. **This path is gitignored / excluded from deploy** — operator
      must rsync separately:
      ```
      rsync -av --partial --progress \
        ~/Trader/jax_ckpt_easy_phase1_xs32/easy_medium/ \
        gpu-server:Trader/jax_ckpt_easy_phase1_xs32/easy_medium/
      ```
- [x] `trading_data.db` present on server (24 GB, restored from Jan-20 snapshot)
- [x] `train_phase2.py` is on training branch (verified via deploy)

## Resources

- 3 GPUs (CUDA_VISIBLE_DEVICES=0, 1, 2). GPU 3 stays free for monitoring or
  ad-hoc work.
- ≤ 20 GB VRAM per variant (TASK_736 reference fit comfortably).
- Wallclock: ~several hours per variant, parallelized → real time bounded by
  the slowest variant.

## Important deviations from `tools/run_737_ablations.sh`

The workstation runner is sequential AND has workstation-only pre-flight:

[redacted: host service detail]
  **Do NOT run `bash tools/run_737_ablations.sh` directly.** Extract per-variant
  commands and invoke each via the multi-GPU launcher.
- It runs variants sequentially. We parallelize via
  `tools/deploy/launch-experiment.sh`, one variant per GPU.
- It writes output to `jax_ckpt_easy_phase2_737_<tag>/` at repo root. Per the
  server contract (SERVER.md), training output belongs in `runs/<exp_id>/`.
  Override `--output-dir` to land in `runs/737_ablations/<tag>/`.

## Commands

Shared args, from `tools/run_737_ablations.sh`:

```
COMMON="--stage B --epochs 200 \
  --periods-file config/foxglove_periods.json \
  --n-slots 17 --max-runs-per-group 10 \
  --factored --factored-temperature 0.1 \
  --no-reset-value-head --value-target-mode nstep --td-discount 0.95 \
  --value-weight 0.5 \
  --bc-weight-min 0.05 --bc-max-trade-weight 4.0 \
  --direction-weight 0.5 --encoder-lr 3e-5 \
  --encoder-version 3 --embed-dim 64 --encoder-heads 8"

CKPT="jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl"
```

Launch the 3 variants in parallel:

```bash
# no_aux — GPU 0
tools/deploy/launch-experiment.sh 737_ablations 0 -- python train_phase2.py \
  $COMMON --checkpoint $CKPT --rank-loss mse --rank-weight 1.0 \
  --no-aux-price-head --output-dir runs/737_ablations/no_aux

# no_rank — GPU 1
tools/deploy/launch-experiment.sh 737_ablations 1 -- python train_phase2.py \
  $COMMON --checkpoint $CKPT --no-rank-head --aux-price-weight 0.1 \
  --output-dir runs/737_ablations/no_rank

# minimal — GPU 2
tools/deploy/launch-experiment.sh 737_ablations 2 -- python train_phase2.py \
  $COMMON --checkpoint $CKPT --no-rank-head --no-aux-price-head \
  --output-dir runs/737_ablations/minimal
```

Caveat: `launch-experiment.sh` enforces unique tmux session names
(`exp_<exp_id>`). Launching three variants under the same exp_id collides —
either (a) extend the launcher with a per-variant suffix, or (b) for this
suite, launch each in its own tmux session manually:

```bash
# Option (b) — direct tmux per variant:
mkdir -p runs/737_ablations/no_aux runs/737_ablations/no_rank runs/737_ablations/minimal

for variant_args in \
  "no_aux 0 --rank-loss mse --rank-weight 1.0 --no-aux-price-head" \
  "no_rank 1 --no-rank-head --aux-price-weight 0.1" \
  "minimal 2 --no-rank-head --no-aux-price-head"; do
  set -- $variant_args
  tag=$1; gpu=$2; shift 2
  tmux new-session -d -s "exp_737_${tag}" \
    "cd ~/Trader && source trader_env/bin/activate && \
     CUDA_VISIBLE_DEVICES=${gpu} python train_phase2.py \
       $COMMON --checkpoint $CKPT $* \
       --output-dir runs/737_ablations/${tag} \
       2>&1 | tee runs/737_ablations/${tag}/train.log"
done
```

Pick whichever fits — the agent decides based on what's already in tmux.

## `train_phase2.py` not yet smoke-tested

This SPEC is the first server-side run of `train_phase2.py`. Launch all 3
variants directly; if any fails at startup (checkpoint format mismatch, flag
drift, missing modules), STOP and report — that variant's startup IS the smoke
for that path. The other variants share startup logic so they will fail the
same way; killing them is cheap. No need for a separate 10-min holding run
before parallelizing.

## Success criteria

For each of `no_aux`, `no_rank`, `minimal`:

- [ ] Run completes 200 epochs without crash
- [ ] `runs/737_ablations/<tag>/` contains: `train.log`, final checkpoint,
      metrics CSV / JSON if `train_phase2.py` writes one
- [ ] No NaN loss, no CUDA OOM
- [ ] Benchmark/eval metrics extractable (look for `Bal.Acc`, `Alpha`,
      `beats-random` in the log tail or eval output)

## Output expectations

```
runs/737_ablations/
  no_aux/    train.log, phase2_stageB_best.pkl, metrics.{csv,json}, …
  no_rank/   train.log, phase2_stageB_best.pkl, metrics.{csv,json}, …
  minimal/   train.log, phase2_stageB_best.pkl, metrics.{csv,json}, …
  result.md  ← write at the end: per-variant summary table (Bal.Acc, Alpha,
              beats-random, dir_acc final, policy loss final) for operator
              to copy into the workstation's ablation_737_summary.md.
```

The summary table format expected by the operator (one row per variant):

| Tag | Bal.Acc AVG | Alpha AVG | ModelPnL AVG | Beats random | Dir.Acc (final) | Policy loss (final) |

Reference row (for context, do not recompute — from TASK_736):

| Reference (TASK_736) | 26.5% | +3.87% | … | 6/6 | … | … |

## Notes

- `legacy_27ch` variant is deferred: the 27-ch Phase 1 checkpoint
  (`jax_ckpt_easy_phase1_v3/easy_medium/phase1_best.pkl`) does not exist on
  workstation. Operator will either locate or regenerate it and add as a
  follow-up SPEC.
- The TASK_737 task description names value-weight=0.5 as the silent blocker
  resolved in TASK_736. Verify that flag is present in your launched commands
  (it's in `$COMMON`).
- Aggregating results into `result.md` is your job. Pull pertinent metric lines
  from each `train.log`. If `train_phase2.py` emits a structured eval JSON, use
  that — fall back to log grep otherwise.
