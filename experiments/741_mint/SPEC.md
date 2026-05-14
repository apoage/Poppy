# 741_mint — Mint per-asset post-pool injection (paired 3-seed)

## Goal

Test the TASK_741 hypothesis: per-asset scalar features routed past the
cross-asset temporal pool (via FiLM injection on the post-pool latents)
will improve Bal.Acc and Alpha by giving each per-asset head a clean
asset-specific signal that's not smeared by the pool.

Targets the 87% Val Agree ceiling identified by TASK_733.

## Source task

`tasks/TASK_741_PER_ASSET_INJECTION_FIX.md` — design spec.
Implementation done 2026-05-13 by sub-agent (10 files), committed in
this PR. Default OFF, zero-init byte-identical at start (γ=0, β=0).

## Background

After TASK_743 multi-seed validation (2026-05-14) we know:
- Single-seed Stage-B Alpha comparisons ≤ ~5pp are within noise.
- Canonical baseline is **a1b1** (rank+aux, TASK_736 ref).
- Any improvement claim needs ≥3-seed paired validation.

So this SPEC pairs every variant with its same-seed baseline.

## Prerequisites

- [x] TASK_741 Mint code implementation (working tree + commit landing
      with this SPEC)
- [x] Smoke test passed on workstation (Gate 1: `train_easy.py
      --cheat-channel --per-asset-injection cheat`, 5 epochs)
- [x] FiLM zero-init test passes:
      `python tools/tests/test_trm_encoder_v3.py` → `FiLM zero-init max
      diff vs baseline: 0.00e+00`
- [x] Phase 1 ckpt: `jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl`

## Resources

- 1× RTX 3090 per training process, ~21 GB, ~2h20m
- 2-wide batching per TASK_742 rule (≤2 concurrent train_phase2/host)
- Total wall: 6 runs × 2.5h with 2-wide = **~7.5h**
- Storage: ~200 MB per run (checkpoints + logs) × 6 = ~1.2 GB

## Plan

Six runs total: paired 3-seed comparison.

| Variant | rank | aux | mint | seeds |
|---|---|---|---|---|
| `a1b1` (baseline) | ON | ON | — (off) | 7, 13, 23 |
| `a1b1_mint_xs` | ON | ON | xs (`--per-asset-injection xs`) | 7, 13, 23 |

Mint XS injects the 5 per-asset scalar features from input channels 27-31.
This is the production-safe path (no cheat-channel dependency).

## Wave plan (2-concurrent per TASK_742)

Operator (or server agent) launches one wave at a time. Each wave is
2 parallel `train_phase2.py` processes; eval runs after.

```
Wave 1 (T=0:00, ~2.5h):
  GPU 0:  a1b1            seed 7
  GPU 1:  a1b1_mint_xs    seed 7

Wave 2 (T=2:30, ~2.5h):
  GPU 0:  a1b1            seed 13
  GPU 1:  a1b1_mint_xs    seed 13

Wave 3 (T=5:00, ~2.5h):
  GPU 0:  a1b1            seed 23
  GPU 1:  a1b1_mint_xs    seed 23

T=7:30 — All training done. Launch 6 evals in parallel (GPU-light, ~10
min). Aggregate result.md.
```

## Commands

```bash
cd ~/Trader && source trader_env/bin/activate

# Shared training args
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
  --aux-price-weight 0.1"

# --- Wave N pattern (for SEED in {7, 13, 23}) ---
# Variant A: a1b1 baseline (no mint)
mkdir -p runs/741_mint/baseline_seed_${SEED}
screen -d -m -S mint_base_s${SEED} bash -c "cd ~/Trader && \
  source trader_env/bin/activate && \
  CUDA_VISIBLE_DEVICES=0 python train_phase2.py $COMMON \
  --seed ${SEED} \
  --output-dir runs/741_mint/baseline_seed_${SEED} \
  --name 741_baseline_s${SEED} 2>&1 \
  | tee runs/741_mint/baseline_seed_${SEED}/train.log; \
  echo MINT_BASE_S${SEED}_DONE > /tmp/mint_base_s${SEED}_done.flag"

# Variant B: a1b1 + Mint xs
mkdir -p runs/741_mint/mint_xs_seed_${SEED}
screen -d -m -S mint_xs_s${SEED} bash -c "cd ~/Trader && \
  source trader_env/bin/activate && \
  CUDA_VISIBLE_DEVICES=1 python train_phase2.py $COMMON \
  --per-asset-injection xs \
  --seed ${SEED} \
  --output-dir runs/741_mint/mint_xs_seed_${SEED} \
  --name 741_mint_xs_s${SEED} 2>&1 \
  | tee runs/741_mint/mint_xs_seed_${SEED}/train.log; \
  echo MINT_XS_S${SEED}_DONE > /tmp/mint_xs_s${SEED}_done.flag"

# Wait for both flags
until [ -f /tmp/mint_base_s${SEED}_done.flag ] && [ -f /tmp/mint_xs_s${SEED}_done.flag ]; do
  sleep 30
done
echo "Wave for seed ${SEED} complete at $(date -u +%H:%M:%S)"
```

After all 3 waves:

```bash
# Eval all 6 in parallel (GPU-light)
for VARIANT in baseline mint_xs; do
  for SEED in 7 13 23; do
    DIR="runs/741_mint/${VARIANT}_seed_${SEED}"
    EVAL_DIR="runs/741_mint/eval_${VARIANT}_seed_${SEED}"
    mkdir -p "$EVAL_DIR"
    screen -d -m -S eval_${VARIANT}_s${SEED} bash -c "cd ~/Trader && \
      source trader_env/bin/activate && \
      python tools/benchmark_easy.py \
      --checkpoint ${DIR}/phase2_stageB_best.pkl \
      --output-dir ${EVAL_DIR} 2>&1 \
      | tee ${EVAL_DIR}/benchmark.log; \
      echo EVAL_${VARIANT}_S${SEED}_DONE > /tmp/eval_${VARIANT}_s${SEED}_done.flag"
  done
done

# Wait for all 6
until [ -f /tmp/eval_baseline_s7_done.flag ] && \
      [ -f /tmp/eval_baseline_s13_done.flag ] && \
      [ -f /tmp/eval_baseline_s23_done.flag ] && \
      [ -f /tmp/eval_mint_xs_s7_done.flag ] && \
      [ -f /tmp/eval_mint_xs_s13_done.flag ] && \
      [ -f /tmp/eval_mint_xs_s23_done.flag ]; do sleep 20; done
echo "All evals done at $(date -u +%H:%M:%S)"
```

## Success criteria

- [ ] All 6 training runs complete 200 epochs cleanly (no deadlock,
      no NaN, FiLM `gamma`/`beta` params learnable in metrics)
- [ ] Eval `Beats random` = 6/6 in all 6 runs (sanity)
- [ ] **Paired Alpha delta** (per seed):
      `mint_xs_seed_S - baseline_seed_S` averaged across 3 seeds.
      - If mean(Δα) > +3pp AND all 3 deltas positive → **CLEAR WIN**,
        promote Mint to canonical Stage-B.
      - If mean(Δα) ∈ (−2, +3)pp → inconclusive; add 3 more seeds.
      - If mean(Δα) < −2pp → **clear regression**, abandon Mint xs;
        rerun with `--per-asset-injection both` (xs + cheat) as
        fallback diagnostic.
- [ ] Val Agree (Stage A→B): variant ≥ baseline (sanity that the
      ceiling-break hypothesis isn't worse than baseline)

## Output expectations

```
runs/741_mint/
├── baseline_seed_7/         train.log, best.pkl, metrics.csv
├── baseline_seed_13/
├── baseline_seed_23/
├── mint_xs_seed_7/          train.log, best.pkl, metrics.csv
├── mint_xs_seed_13/
├── mint_xs_seed_23/
├── eval_baseline_seed_7/    benchmark.json, benchmark.log
├── eval_baseline_seed_13/
├── eval_baseline_seed_23/
├── eval_mint_xs_seed_7/
├── eval_mint_xs_seed_13/
├── eval_mint_xs_seed_23/
└── result.md                paired delta table + verdict
```

## Risks / open questions

- Per-asset scalars are sliced from `obs[:, 27:32, 0, :]` (input
  channels 27-31). Requires that the current Phase 1 ckpt
  (`phase1_xs32`) has channels 27-31 active. If not, behavior is
  undefined. **Operator should verify** with one short run before
  full Wave 1.
- FiLM is zero-init; first ~10 epochs Mint should be a no-op (γ≈0,
  β≈0). Check that variant matches baseline in E1-E5 metrics to
  confirm.
- TASK_742 multi-process rule strictly applies: ≤2 concurrent
  train_phase2 per host. Wave plan respects this.
- Single-seed flukes still possible at N=3. If results are
  inconclusive (mean Δα in the inconclusive band), do not promote;
  add seeds 31 and 42 for a 5-seed run.
