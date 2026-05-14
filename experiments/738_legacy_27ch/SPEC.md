# 738_legacy_27ch — TASK_732 Cross-Sectional Channels Ablation (paired 3-seed)

## Goal

Test whether the cross-sectional channels added in TASK_732 (channels
27-31: ranks, beta, idio-vol, xs momentum) actually contribute to Stage-B
performance with working ExIt, or whether the pre-TASK_732 27-channel
encoder is just as good. Closes the last gap in the TASK_737 input/head
ablation matrix.

## Source task

`tasks/TASK_738_LEGACY_27CH_ABLATION.md` (status ACTIVE, originally
attached to the TASK_737 suite as Run 4).

## Background — multi-seed protocol

Post-TASK_743 (2026-05-14) we know single-seed Alpha gaps ≤ ~5pp are
within noise. Canonical baseline is **a1b1** (rank+aux on 32-ch
encoder). Every mechanism claim — including ablations — needs ≥3-seed
paired validation.

The a1b1 baseline runs at seeds 7/13/23 are produced by **741_mint**
Wave 1-3 (running ~08:00→15:30 today). This SPEC adds 3 variant runs and
**reuses the 741_mint baseline** as the 32-ch counterpart.

## Prerequisites

- [x] `--legacy-27ch` flag already wired in `train_phase2.py:1793,1958`
      and `mcts_trader/env/jax_encoder.py:336,529`
- [ ] **27-ch Phase 1 checkpoint** at
      `jax_ckpt_easy_phase1_v3/easy_medium/phase1_best.pkl`. Original
      Apr-22 ckpt (97 MB) was deleted from both workstation and server;
      user is recovering from `/mnt/storage/` evening of
      2026-05-14. Once recovered, **rsync to server** at the same path:
      ```bash
      rsync -av jax_ckpt_easy_phase1_v3/ gpu-server:Trader/jax_ckpt_easy_phase1_v3/
      ```
- [ ] a1b1 baseline runs at seeds {7,13,23} — produced by
      `runs/741_mint/baseline_seed_{7,13,23}/` (32-ch encoder side).

## Resources

- 1× RTX 3090 per training process, ~21 GB, ~2h20m
- TASK_742 rule: ≤2 concurrent train_phase2 per host
- Total: 3 variant runs × 2.5h with 2-wide batching = **~3.75h wall**

## Plan

| Variant | encoder | rank | aux | seeds | source |
|---|---|---|---|---|---|
| **a1b1_32ch** (baseline) | 32-ch (default) | ON | ON | 7, 13, 23 | reuse from 741_mint |
| **a1b1_legacy_27ch** | 27-ch (no cross-sec) | ON | ON | 7, 13, 23 | new runs in this SPEC |

Both variants warmstart from their respective Phase 1 checkpoints:
- 32-ch baseline: `jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl`
  (current default)
- 27-ch variant: `jax_ckpt_easy_phase1_v3/easy_medium/phase1_best.pkl`
  (the recovered Apr-22 ckpt)

This isolates the encoder-input dimension: rank+aux heads stay ON in
both arms, so we're testing **only** whether cross-sectional channels
contribute.

## Wave plan

Cannot start until 27-ch ckpt is rsynced to server. Then:

```
Wave 1 (T=0:00, ~2.5h):
  GPU A:  legacy_27ch seed 7
  GPU B:  legacy_27ch seed 13

Wave 2 (T=2:30, ~1.25h):
  GPU A:  legacy_27ch seed 23

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
  --checkpoint jax_ckpt_easy_phase1_v3/easy_medium/phase1_best.pkl \
  --rank-loss mse --rank-weight 1.0 \
  --aux-price-weight 0.1 \
  --legacy-27ch"

# For SEED in {7, 13, 23}, launch on free GPU under TASK_742 rule:
mkdir -p runs/738_legacy_27ch/legacy27_seed_${SEED}
screen -d -m -S l27_s${SEED} bash -c "cd ~/Trader && source trader_env/bin/activate && \
  CUDA_VISIBLE_DEVICES=<GPU> python train_phase2.py $COMMON \
  --seed ${SEED} \
  --output-dir runs/738_legacy_27ch/legacy27_seed_${SEED} \
  --name 738_l27_s${SEED} 2>&1 \
  | tee runs/738_legacy_27ch/legacy27_seed_${SEED}/train.log; \
  echo L27_S${SEED}_DONE > /tmp/l27_s${SEED}_done.flag"
```

Eval pattern same as 741_mint final block — point at new checkpoints +
shared baselines from `runs/741_mint/baseline_seed_*/`.

## Success criteria

- [ ] All 3 training runs complete 200 epochs cleanly
- [ ] **Paired Alpha delta** vs 32-ch baseline (per seed):
      - mean(Δα) ∈ (−2, +2)pp → **TASK_732 cross-sectional channels
        don't help**; keep 32-ch as default (free richness) but
        deprioritize further cross-sectional work
      - mean(Δα) < −2pp → **TASK_732 genuinely contributes**; keep
        32-ch confirmed; tier-D2 external features become attractive
      - mean(Δα) > +2pp → **surprising**: cross-sectional channels
        adding noise post-MCTS; audit which channel groups conflict
- [ ] Bal.Acc delta vs baseline also tracked (same thresholds)

## Output expectations

```
runs/738_legacy_27ch/
├── legacy27_seed_7/        train.log, best.pkl, metrics.csv
├── legacy27_seed_13/
├── legacy27_seed_23/
├── eval_legacy27_seed_7/   benchmark.json, benchmark.log
├── eval_legacy27_seed_13/
├── eval_legacy27_seed_23/
└── result.md               paired delta table + verdict on TASK_732
```

## Notes

- **Risk — P1 ckpt drift**: the 27-ch V3 P1 ckpt is from Apr 22; the
  32-ch xs32 P1 is newer (Apr 23). Subtle config/data drift between the
  two training runs is possible but should be minimal — both used the
  same Foxglove periods file. Document if loss curves diverge visibly
  at E0-E5.
- **Pool rebuild cost**: 27-ch mode recomputes encoder caches from
  scratch on first run (~5 min extra) — negligible.
- The original TASK_738 design was single-seed (one variant in the
  TASK_737 suite). Upgraded here to paired 3-seed per the post-TASK_743
  protocol. If results are inconclusive (mean Δα in the noise band),
  add seeds 31, 42 for 5-seed confirmation.
- Result of this ablation also informs **whether to invest in the
  TASK_741 Mint xs injection path** — if cross-sectional channels turn
  out to not matter, the xs scalars Mint injects from channels 27-31 are
  fighting for relevance.
