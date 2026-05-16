# 751_titans_lr_sweep — Titans LR bracket probes (seed 7)

## Goal

Confirm `--titans-lr 3e-3` (the smoke-tested default) is near a local optimum, not just the first working value. Two single-seed probes at `1e-3` (below) and `1e-2` (above). Cheap robustness check on the new mechanism's most important hyperparameter.

## Background

The default `--titans-lr 3e-3` (100× the encoder LR 3e-5) was chosen because it was the first value that successfully engaged the gate (smoke 2026-05-16, gate Δ +3.4e-3 in 10 epochs). We don't know if smaller LR (1e-3) would still engage adequately or if larger LR (1e-2) would either engage faster (better) or destabilize (worse).

Same logic as 748_flow_velsweep was to 734.

## Prerequisites

- [x] Phase 1 ckpt: `jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl`
- [x] Titans-MAG + `--titans-lr` flag wired (commit `04b8632`)
- [x] 749_titans_mag/titans_seed_7 (running or done) at `--titans-lr 3e-3` — reference for Δα

## Resources

- 1× RTX 3090, GPU 2 (server)
- Wall: 2 runs × 3h sequential = **~6h** (single GPU, sequential — same LR sweep pattern as 748)
- TASK_742 rule: count as 1 of 2-concurrent budget

## Plan

Sequential single-seed (s7) probes at two LRs vs the existing 749 3e-3 reference at the same seed.

| Variant | titans_lr | seed | pairs against |
|---|---|---|---|
| `lr_1e-3` | 1e-3 | 7 | `runs/749_titans_mag/titans_seed_7/` (3e-3 reference) |
| `lr_1e-2` | 1e-2 | 7 | `runs/749_titans_mag/titans_seed_7/` |

Run sequentially on the same GPU. No need for 2-wide.

## Commands

```bash
cd ~/Trader && source trader_env/bin/activate

COMMON_TITANS="--titans-memory --titans-variant mag \
  --titans-memory-depth 2 --titans-memory-hidden 32 \
  --titans-persistent-tokens 4 \
  --titans-gate-init -3 \
  --reset-titans-memory"

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
  --rank-loss mse --rank-weight 1.0 --aux-price-weight 0.1 \
  $COMMON_TITANS \
  --seed 7"

# Probe 1: lr=1e-3
mkdir -p runs/751_titans_lr_sweep/lr_1e-3_seed_7
screen -d -m -S tl_lo bash -c "source trader_env/bin/activate && \
  CUDA_VISIBLE_DEVICES=2 python -u train_phase2.py $COMMON \
  --titans-lr 1e-3 \
  --output-dir runs/751_titans_lr_sweep/lr_1e-3_seed_7 \
  --name 751_tl_lo 2>&1 \
  | tee runs/751_titans_lr_sweep/lr_1e-3_seed_7/train.log; \
  echo TL_LO_DONE > /tmp/tl_lo_done.flag"

# After /tmp/tl_lo_done.flag drops, launch probe 2:
mkdir -p runs/751_titans_lr_sweep/lr_1e-2_seed_7
screen -d -m -S tl_hi bash -c "source trader_env/bin/activate && \
  CUDA_VISIBLE_DEVICES=2 python -u train_phase2.py $COMMON \
  --titans-lr 1e-2 \
  --output-dir runs/751_titans_lr_sweep/lr_1e-2_seed_7 \
  --name 751_tl_hi 2>&1 \
  | tee runs/751_titans_lr_sweep/lr_1e-2_seed_7/train.log; \
  echo TL_HI_DONE > /tmp/tl_hi_done.flag"
```

Eval each with `tools/benchmark_easy.py` against the same 6-period benchmark.

## Success criteria

- [ ] Both probes complete 200 epochs cleanly
- [ ] `titans_gate_mean` trajectory captured for both
- [ ] **Δα vs 749/titans_seed_7 (the 3e-3 reference at same seed):**
      - |Δα| < 3pp at both endpoints → 3e-3 is in a wide flat optimum; default stands
      - One endpoint > 3pp better → follow up with 3-seed at the winning LR
      - Both endpoints worse → confirms 3e-3 is a local optimum
- [ ] **Gate engagement speed comparison**: gate_mean at E10 / E50 / E100 / E200 for {1e-3, 3e-3 ref, 1e-2}
      - At 1e-2 we may see faster engagement → could replace 3e-3 as default
      - At 1e-3 we may see no engagement (back to the original problem) → confirms LR matters
      - At 3e-3 (ref) we expect the smoke-validated ~0.05 engagement by E200

## Output expectations

```
runs/751_titans_lr_sweep/
├── lr_1e-3_seed_7/    train.log, best.pkl, metrics.csv
├── lr_1e-2_seed_7/
├── eval_lr_1e-3_seed_7/ benchmark.json
├── eval_lr_1e-2_seed_7/ benchmark.json
└── result.md           Δα at each LR + gate engagement curve comparison
```

## Notes

- This is single-seed; comparisons noise-limited at ~5pp. Looking for large effects.
- If both probes are clearly bad, no further LR sweep needed — 3e-3 stands.
- If 1e-2 is clearly best (large positive Δα), follow up with 3-seed at 1e-2 in a Tier-D experiment.
- Same caveat as 748: single-seed signal that "feels" significant should be validated at 3+ seeds before promoting.
