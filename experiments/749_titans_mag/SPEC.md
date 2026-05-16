# 749_titans_mag — TASK_749 Titans-MAG Memory in V3 Encoder (paired 3-seed)

## Goal

Test whether a Titans-style neural long-term memory module (MAG variant)
plugged into the V3 encoder's temporal compression stage gives Alpha
headroom over the current minGRU-only path. Titans memory is a tiny MLP
whose own weights update at *inference time* via surprise-momentum
gradient descent with weight decay (Behrouz et al. 2025, eq. 13).
MAG combines it with the existing minGRU branch via a learnable
sigmoid gate.

This is a **planning-side mechanism**, in the sense that it changes
how temporal context is compressed rather than making the encoder
"sharper" in a classifier sense. See `project_classifier_alpha_anti_correlation.md`
for why this distinction matters on this stack.

## Source task

`tasks/TASK_749_TITANS_PIVOT.md` — status DRAFTED (full implementation
spec, file:line citations, JAX scan/remat pattern, memory budget).

## Background — multi-seed protocol

Post-TASK_743 (2026-05-14) we know single-seed Alpha gaps ≤ ~5pp are
within noise. Canonical baseline is **a1b1** (rank+aux). Every
mechanism claim needs ≥3-seed paired validation.

The a1b1 baseline runs at seeds 7/13/23 already exist from
**741_mint** Wave 1-3 (2026-05-14). This SPEC adds 3 variant runs and
**reuses the 741_mint baseline** for the paired comparison — exactly
the pattern used by 734 and 739.

**Conditional baseline**: if 734-x5 (currently running, seeds 31+42)
promotes `--flow-matching` to canonical Stage-B, switch the baseline
to `runs/734_flow_matching/flow_seed_{7,13,23}/` and re-pair. The
promotion gate (mean Δα ≥ +3pp AND all 3 positive) still applies
against the new baseline.

## Prerequisites

- [x] Phase 1 ckpt: `jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl`
- [x] `--titans-memory` flag wired (commit `a55b913`)
- [x] `--titans-variant mag` implementation in `mcts_trader/jax_agent/titans_memory.py` (commit `de468a6` + 5 fix commits ending `04b8632`)
- [x] V3 encoder gate-combined Titans + minGRU (commit `8e6f581`)
- [x] **Smoke PASSED 2026-05-16** at `--titans-lr 3e-3 --titans-gate-init -3`:
  - grad_norm 3.6–69 (finite, decreasing) ✓
  - gate_mean Δ(E0→E9) = +3.4e-3 (3.4× threshold) ✓
  - mem_norm 0.05 → 0.92 (organic engagement) ✓
  - total_loss 37.4 → 17.7 monotonic ✓
  - no NaN ✓
- [x] a1b1 baseline runs at seeds {7, 13, 23} — `runs/741_mint/baseline_seed_{7,13,23}/`. Verify before launching.

## Resources

- 1× RTX 3090 per training process, ~20-22 GB
- Wall per run: ~3 h (Titans adds ~10-20% over baseline 2.5 h)
- TASK_742 rule: ≤2 concurrent train_phase2 per host
- Total: 3 variant runs × 3 h with 2-wide batching = **~4.5 h wall**
  (baseline reused)

## Plan

| Variant | encoder | titans | seeds | source |
|---|---|---|---|---|
| **a1b1 baseline** | V3 minGRU only | OFF | 7, 13, 23 | reuse from 741_mint |
| **titans_mag** | V3 minGRU + Titans-MAG gate | ON (depth=2, N_p=4) | 7, 13, 23 | new runs in this SPEC |

## Wave plan

```
Wave 1 (T=0:00, ~3.0h):
  GPU A:  titans_mag seed 7
  GPU B:  titans_mag seed 13

Wave 2 (T=3:00, ~1.5h):
  GPU A:  titans_mag seed 23

T=4:30 — launch 3 evals in parallel
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
  --titans-memory --titans-variant mag \
  --titans-memory-depth 2 --titans-memory-hidden 32 \
  --titans-persistent-tokens 4 \
  --titans-gate-init -3 --titans-lr 3e-3 \
  --reset-titans-memory"

# For SEED in {7, 13, 23}, launch on free GPU under TASK_742 rule:
mkdir -p runs/749_titans_mag/titans_seed_${SEED}
screen -d -m -S tt_s${SEED} bash -c "cd ~/Trader && source trader_env/bin/activate && \
  CUDA_VISIBLE_DEVICES=<GPU> python train_phase2.py $COMMON \
  --seed ${SEED} \
  --output-dir runs/749_titans_mag/titans_seed_${SEED} \
  --name 749_tt_s${SEED} 2>&1 \
  | tee runs/749_titans_mag/titans_seed_${SEED}/train.log; \
  echo TT_S${SEED}_DONE > /tmp/tt_s${SEED}_done.flag"
```

Eval pattern same as 741_mint final block — point at new checkpoints +
shared baselines from `runs/741_mint/baseline_seed_*/`.

## Success criteria

- [ ] All 3 training runs complete 200 epochs cleanly (no NaN, no OOM)
- [ ] `titans_mem_norm` field populated in `metrics.csv`, stays bounded in [0.01, 100.0]
- [ ] `titans_gate_mean` moves non-trivially over training (smoke showed ~3.4e-3 in 10 epochs at lr=3e-3; expect ~0.05-0.2 movement over 200 epochs)
- [ ] `titans_surprise_mean` is positive (memory updates running)
- [ ] Wall ≤ 3.5 h per run
- [ ] **Paired Δα** vs 741_mint baseline (per seed), mean over 3 seeds (updated promotion protocol per 734-x5 reversal lesson — `lore/notes_2026-05-16.md` §115):
      - mean(Δα) ≥ +3pp AND all 3 positive → **screening PASS**; extend to 5-seed (s31, s42) before promotion declaration
      - mean(Δα) ∈ (−2, +3)pp → **inconclusive**; 5-seed optional
      - mean(Δα) < −2pp → **regression**; diagnostic: check `titans_gate_mean` trajectory — flat ≈ mechanism never engaged (LR/init wrong), drifting + bad Alpha = mechanism engaged but harmful

## Output expectations

```
runs/749_titans_mag/
├── titans_seed_7/         train.log, best.pkl, metrics.csv
├── titans_seed_13/
├── titans_seed_23/
├── eval_titans_seed_7/    benchmark.json, benchmark.log
├── eval_titans_seed_13/
├── eval_titans_seed_23/
└── result.md              paired delta table + memory-health metrics
                           (mem_norm, surprise_mean, gate_mean per seed)
```

## Notes

- **Baseline status (2026-05-16)**: 734-x5 did NOT promote (mean Δα +0.20pp at 5-seed after s42 −12.71pp outlier). Canonical Stage-B baseline stays a1b1. Pair this SPEC against `runs/741_mint/baseline_seed_*/`.
- **Stacked variant (Tier-D)**: moot until at least one mechanism promotes 5-seed.
- **Critical flag for engagement**: `--titans-lr 3e-3` is REQUIRED. At default encoder_lr=3e-5 the gate moves ~1e-6/10 epochs (mechanism never engages — verified across 4 smoke iterations). Dedicated LR via `optax.multi_transform` was added in fix 5 (commit `04b8632`).
- **VRAM escape hatch**: if peak hits 24 GB at batch=256, the fallback
  is `--batch-size 128` (documented in TASK_749 §5). The 128 batch
  pairing must be re-validated against a 128-batch a1b1 reference if
  invoked — the existing 741_mint baselines are 256-batch.
- **What "Titans-style" means here**: we implement only the *memory
  module* (paper §2.1, Eqs. 2-3 with data-dependent η/θ/α and a depth-2
  MLP $\mathcal{M}$) and the MAG-style sigmoid gate (paper §3.2). The
  paper's SWA branch is replaced with the existing minGRU. We do NOT
  claim this is a paper-faithful Titans implementation. See TASK_749
  §3 + §8 R1 for the gap.
- **JAX implementation**: `lax.scan` along T=64 with `jax.grad`
  inside the scan body for the inner-loop surprise computation, wrapped
  in `jax.checkpoint` to fit the carry state in 24 GB. See
  TASK_749 §5 for the full pattern.
- **Init bias on the gate** is large-negative (~-10) so MAG output is
  byte-identical to minGRU at step 0. This is checkpoint-safe and
  guarantees the baseline reproduces if Titans never engages.
