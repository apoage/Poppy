# 737_eval — benchmark eval of TASK_737 ablation checkpoints

## Goal

Fill the `Bal.Acc / Alpha / ModelPnL / beats-random` columns in the TASK_737
ablation comparison table. Training produced two completed Stage-B
checkpoints (`no_rank`, `minimal`); a third (`no_aux`) is blocked on
TASK_742. Run `tools/benchmark_easy.py` on each completed checkpoint to get
per-period and aggregate metrics on the 6 real-market benchmark periods
(`bull, bear, sideways, ftx_crash, etf_approval, boring_week`).

## Source task

Continuation of `experiments/737_ablations/SPEC.md`. Reference row from
`tasks/TASK_736_*.md` (workstation): Bal.Acc 26.5% AVG, Alpha +3.87% AVG,
beats-random 6/6.

## Prerequisites (verified on server)

- [x] `runs/737_ablations/no_rank/phase2_stageB_best.pkl` (32.9 MB, E196)
- [x] `runs/737_ablations/minimal/phase2_stageB_best.pkl` (32.6 MB, E196)
- [x] `trading_data.db` (24 GB, Jan-20 snapshot)
- [x] `config/benchmark_periods.json` (deployed via training branch)
- [x] `tools/benchmark_easy.py` (deployed)
- [ ] Workstation has not smoke-tested benchmark_easy.py on a server-trained
      checkpoint. If startup fails (config-detection, encoder mismatch),
      STOP and report — fix is workstation-side.

## Resources

- 1 GPU per checkpoint. Inference-only, ~5-10 min per checkpoint (depends on
  period lengths; ~60k total steps across 6 periods).
- Can run sequentially on a single GPU, or in parallel on 2 GPUs.
- ≤ 5 GB VRAM expected per process.

## Commands

```bash
mkdir -p runs/737_eval/no_rank runs/737_eval/minimal

# Sequential on GPU 0, OR parallel on GPU 0+1.
# Sequential is simpler — no tmux gymnastics needed and total time is
# bounded by ~20 min.

# no_rank
CUDA_VISIBLE_DEVICES=0 python tools/benchmark_easy.py \
  --checkpoint runs/737_ablations/no_rank/phase2_stageB_best.pkl \
  --db trading_data.db \
  --json runs/737_eval/no_rank/benchmark.json \
  2>&1 | tee runs/737_eval/no_rank/benchmark.log

# minimal
CUDA_VISIBLE_DEVICES=0 python tools/benchmark_easy.py \
  --checkpoint runs/737_ablations/minimal/phase2_stageB_best.pkl \
  --db trading_data.db \
  --json runs/737_eval/minimal/benchmark.json \
  2>&1 | tee runs/737_eval/minimal/benchmark.log
```

## Success criteria

For each variant:

- [ ] `benchmark.json` exists and contains 6 period blocks
- [ ] Each period has fields: `balanced_accuracy`, `model_alpha`,
      `model_pnl`, `random_pnl`, `agree_trade`, `agree_hold`
- [ ] No NaN / Inf in metrics
- [ ] Exit code 0

## Output expectations

```
runs/737_eval/
├── no_rank/   benchmark.json, benchmark.log
├── minimal/   benchmark.json, benchmark.log
└── result.md  ← write at the end:
                comparison table with AVG across 6 periods,
                beats-random = count of periods where model_pnl > random_pnl
```

Format of the table (one row per variant):

| Tag | Bal.Acc AVG | Alpha AVG | ModelPnL AVG | Beats random | log |
|---|---|---|---|---|---|
| no_rank | … | … | … | …/6 | `no_rank/benchmark.log` |
| minimal | … | … | … | …/6 | `minimal/benchmark.log` |
| Reference (TASK_736) | 26.5% | +3.87% | — | 6/6 | — |

Per-period rows are also useful — include a second per-period table if you
have the budget for it (one row per (variant, period) tuple).

## Notes

- `tools/benchmark_easy.py` loads config from checkpoint metadata
  (`metadata['config']`) — Stage B saves it there, so no separate
  `config.json` is needed in the run dir.
- DB path must be `trading_data.db` (server's restored Jan-20 snapshot).
  Train cutoff was 2024-12-31; benchmark periods are within that range, so
  the Jan-20 cutoff doesn't affect this eval.
- `no_aux` is blocked on TASK_742 (rank-head deadlock at E18). Once fixed
  and rerun, append a third row to this eval.
- `legacy_27ch` remains deferred (Phase 1 27-ch checkpoint unavailable).
