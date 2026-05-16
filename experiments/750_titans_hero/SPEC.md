# 750_titans_hero — Titans-MAG 1000-epoch single-seed hero

## Goal

Calibration run: does Titans-MAG have actual learnable headroom past the 200-epoch protocol, or does the mechanism stagnate? Single seed at 5× the standard Stage-B epoch budget. Tests whether 749's 3-seed paired comparison is fair (200 epochs may not give Titans enough wall to converge).

This is **not** a promotion experiment. It's a "how high does Titans get if we give it room" calibration.

## Background

Titans smoke (2026-05-16) showed:
- Gate moved +3.4e-3 in 10 epochs × 5 steps (50 steps total)
- Mem_norm grew 0.05 → 0.92 over the same window
- Both still strictly engaging at smoke end

Linear extrapolation suggests gate movement to ~0.05-0.2 by E200 (the standard run), and ~0.25-1.0 by E1000. We don't know if the mechanism plateaus or keeps engaging past E200.

## Prerequisites

- [x] Phase 1 ckpt: `jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl`
- [x] Titans-MAG implementation complete (commits ending `04b8632`)
- [x] `--titans-lr` flag wired (commit `04b8632`)
- [x] Baseline pair: `runs/741_mint/baseline_seed_7/` (200 epochs a1b1)

Note: the baseline only ran 200 epochs. For the Alpha comparison at E200, E500, E1000 of the hero, we use the existing baseline as the "200-epoch reference." If the hero's Alpha at E200 is similar to its 749 paired-3-seed counterpart at E200, that validates the comparison. If it diverges, the Titans hero learning curve is interesting in itself.

## Resources

- 1× RTX 3090, **GPU 1** (server stock 3090, ~25% slower than workstation)
- VRAM: ~22 GB at batch=256 (smoke verified at batch=64; server batch=256 may be tighter — fall back to batch=128 if OOM)
- Wall: ~3h per 200 epochs → **~15h for 1000 epochs**
- TASK_742 rule: counts as 1 of the 2-concurrent budget on the server
- Checkpoint every 100 epochs (default) → intermediate snapshots at E100, E200, …, E900, final at E1000

## Plan

Single run. No paired baseline pair at E1000 (would be another 15h; defer unless hero shows promise).

| Variant | seeds | epochs | source |
|---|---|---|---|
| **titans_mag_hero** | 7 | 1000 | this SPEC |
| **a1b1 baseline** | 7 | 200 | reuse `runs/741_mint/baseline_seed_7/` |

## Command

```bash
cd ~/Trader && source trader_env/bin/activate

mkdir -p runs/750_titans_hero/titans_seed_7

screen -d -m -S th_s7 bash -c "source trader_env/bin/activate && \
  CUDA_VISIBLE_DEVICES=1 python -u train_phase2.py \
  --stage B --epochs 1000 \
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
  --titans-memory --titans-variant mag \
  --titans-memory-depth 2 --titans-memory-hidden 32 \
  --titans-persistent-tokens 4 \
  --titans-gate-init -3 --titans-lr 3e-3 \
  --reset-titans-memory \
  --seed 7 \
  --output-dir runs/750_titans_hero/titans_seed_7 \
  --name 750_th_s7 2>&1 \
  | tee runs/750_titans_hero/titans_seed_7/train.log; \
  echo TH_S7_DONE > /tmp/th_s7_done.flag"
```

After completion, eval at multiple checkpoints:

```bash
for E in 200 500 1000; do
  CKPT=runs/750_titans_hero/titans_seed_7/phase2_stageB_e${E}.pkl
  [ -f "$CKPT" ] || CKPT=runs/750_titans_hero/titans_seed_7/phase2_stageB_final.pkl
  mkdir -p runs/750_titans_hero/eval_titans_seed_7_e${E}
  CUDA_VISIBLE_DEVICES=1 python tools/benchmark_easy.py \
    --checkpoint $CKPT \
    --db trading_data.db \
    --json runs/750_titans_hero/eval_titans_seed_7_e${E}/benchmark.json \
    2>&1 | tee runs/750_titans_hero/eval_titans_seed_7_e${E}/benchmark.log
done
```

## Success criteria

This is calibration, not promotion. Outputs to learn from:

- [ ] Training completes 1000 epochs (no NaN, no OOM)
- [ ] `titans_gate_mean` trajectory over epochs: plot/log E0, E100, E200, …, E1000. Look for:
      - **Monotonic engagement** (gate steadily climbing) → Titans needs longer runs, future SPECs should use ≥500 epochs
      - **Plateau by E200-E300** → 200-epoch protocol is fine; Titans simply doesn't engage further
      - **Engagement-then-collapse** → mechanism is destructive at scale, abandon
- [ ] `titans_mem_norm` trajectory: should stay bounded; watch for saturation or unbounded growth
- [ ] Alpha at E200, E500, E1000 (paired vs `runs/741_mint/baseline_seed_7/` Alpha at E200):
      - If Alpha at E1000 > Alpha at E200 by ≥ 3pp → Titans benefits from longer training
      - If flat or decreasing → mechanism doesn't unlock with more compute
- [ ] Compare to `runs/749_titans_mag/titans_seed_7/` (when that 3-seed run completes at E200) — should be similar at E200 within ±2pp

## Output expectations

```
runs/750_titans_hero/
├── titans_seed_7/
│   ├── phase2_stageB_e100.pkl, e200.pkl, e300.pkl, ..., e1000.pkl
│   ├── stage_b_metrics.csv  (1000-row trajectory)
│   ├── stage_b_training.log
│   └── train.log
├── eval_titans_seed_7_e200/   benchmark.json, benchmark.log
├── eval_titans_seed_7_e500/   ...
├── eval_titans_seed_7_e1000/  ...
└── result.md  paired Δα at each checkpoint + gate trajectory + mem_norm trajectory
```

## Notes

- **Why a single seed**: this is calibration. We're asking "does Titans have headroom" not "does Titans promote." Single seed at 1000 epochs answers the headroom question for far less compute than 3 seeds at 1000 epochs.
- **Why seed 7**: deterministically matches baseline_seed_7 (existing reference). If seed 13 or 23 were chosen, we'd also need to think about whether the baseline_seed_X reference is appropriate — seed 7 is the cleanest.
- **If hero shows headroom + 749 3-seed lands inconclusive**: hero result becomes the basis for a "1000-epoch Stage-B" follow-up at 3 seeds, replacing the 200-epoch protocol. This is a protocol-revision conditional, not implied by this SPEC.
- **VRAM caveat**: at batch=256, hero may push VRAM tighter than 749's 3-seed (which gets ~22GB per smoke math). If OOM at batch=256 on server, drop to batch=128 and re-pair against batch=128 baseline (not currently existing — would need parallel baseline_s7 at batch=128 if we want clean Δα).
