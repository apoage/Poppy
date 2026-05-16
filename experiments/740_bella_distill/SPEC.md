# 740_bella_distill — TASK_740 Bella Distillation Head (paired 3-seed)

## Goal

Use Bella's forward-rollout log-return predictions as a per-asset distillation target, giving Poppy regime-shape information beyond the 60-step OHLCV input. **Planning-side mechanism** (auxiliary loss target, not encoder change), same class as 734_flow_matching.

## Source task

`tasks/TASK_740_BELLA_DISTILL_HEAD.md` (status IMPLEMENTED 2026-04-23).

## Background — multi-seed protocol (UPDATED 2026-05-16)

Post-734 5-seed reversal (`lore/notes_2026-05-16.md` §115), the noise floor on this stack is wider than the prior 3-seed Δα protocol assumed. Updated rule:

- 3-seed paired Δα is a **screening** test, not a promotion decision
- Promotion requires **5-seed** with mean(Δα) ≥ +3pp AND no seed < −5pp
- If 3-seed shows mean(Δα) ≥ +3pp with all 3 positive, extend to 5-seed (seeds 31, 42) before promotion

This SPEC runs paired 3-seed first. Operator decides on 5-seed extension based on 3-seed signal.

Baseline reuse: `runs/741_mint/baseline_seed_{7,13,23}/` (a1b1 canonical) per the standard pattern.

## Prerequisites

- [x] Phase 1 ckpt: `jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl`
- [x] Bella precompute COMPLETE — 270/270 shards at `~/Trader/data/bella_preds/`, manifest at `~/Trader/config/bella_preds_manifest.json` (server, written 2026-05-16 05:21 UTC). Uses TASK_745 Option A per-rotation column remap with K=4 stride=64.
- [x] `--bella-distill-head --bella-distill-weight 0.1 --bella-preds-manifest` flags wired in `train_phase2.py` (lines 1865, 1867, 1869)
- [x] a1b1 baseline runs at seeds {7, 13, 23} — `runs/741_mint/baseline_seed_{7,13,23}/`

## Resources

- 1× RTX 3090 per training, ~21 GB VRAM, ~3 h per run (server stock 3090; ~25% slower than workstation EVGA 420W)
- TASK_742 rule: ≤2 concurrent train_phase2 per host
- GPUs **1+2** (NOT 0 — VirtualBox pinned, per `memory/feedback_gpu0_avoid.md`)
- Total: 3 variant runs × 3 h with 2-wide batching = **~4.5 h wall** (baseline reused)

## Plan

| Variant | rank | aux | bella_distill | seeds | source |
|---|---|---|---|---|---|
| **a1b1 baseline** | ON | ON | OFF | 7, 13, 23 | reuse `runs/741_mint/baseline_seed_*/` |
| **bella_distill** | ON | ON | ON (w=0.1) | 7, 13, 23 | new runs |

## Wave plan

```
Wave 1 (T=0:00, ~3h):
  GPU 1: bella_distill seed 7
  GPU 2: bella_distill seed 13

Wave 2 (T=3:00, ~1.5h):
  GPU 1: bella_distill seed 23

T=4:30 — 3 evals in parallel on GPUs 1+2
```

## Commands (server)

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
  --bella-distill-head --bella-distill-weight 0.1 \
  --bella-preds-manifest config/bella_preds_manifest.json"

# For SEED in {7, 13, 23}, GPU in {1, 2}:
mkdir -p runs/740_bella_distill/bella_seed_${SEED}
screen -d -m -S bd_s${SEED} bash -c "source trader_env/bin/activate && \
  CUDA_VISIBLE_DEVICES=<GPU> python -u train_phase2.py $COMMON \
  --seed ${SEED} \
  --output-dir runs/740_bella_distill/bella_seed_${SEED} \
  --name 740_bd_s${SEED} 2>&1 \
  | tee runs/740_bella_distill/bella_seed_${SEED}/train.log; \
  echo BD_S${SEED}_DONE > /tmp/bd_s${SEED}_done.flag"
```

Eval each new ckpt against the same 6-period benchmark used by Tier-2 evals:

```bash
CUDA_VISIBLE_DEVICES=<GPU> python tools/benchmark_easy.py \
  --checkpoint runs/740_bella_distill/bella_seed_${SEED}/phase2_stageB_best.pkl \
  --db trading_data.db \
  --json runs/740_bella_distill/eval_bella_seed_${SEED}/benchmark.json
```

## Success criteria

- [ ] All 3 training runs complete 200 epochs cleanly (no NaN, no OOM)
- [ ] `bella=…(vf=…)` field in E-line logs — `bella_distill_loss` populated, `bella_valid_frac` (vf) > 0.5 most of training
- [ ] **Paired Δα** vs 741_mint baseline (per seed), mean over 3 seeds:
      - mean(Δα) ≥ +3pp AND all 3 positive → **screening PASS**; recommend 5-seed extension before promotion
      - mean(Δα) ∈ (−2, +3)pp → **inconclusive**; 5-seed extension optional
      - mean(Δα) < −2pp → **regression**; check whether `bella_valid_frac` is high (mechanism actually engaged) vs near-zero (no signal getting through)
- [ ] Trade-rate per period: confirm no over-trading-collapse signature (cf. 738/741)

## Output expectations

```
runs/740_bella_distill/
├── bella_seed_{7,13,23}/      train.log, phase2_stageB_best.pkl, stage_b_metrics.csv
├── eval_bella_seed_{7,13,23}/ benchmark.json, benchmark.log
└── result.md                  paired Δα table + bella_valid_frac trajectory + per-period
```

After server agent finishes training + evals, also write `result.md` with paired Δα table. Server agent (or workstation post-hoc) is responsible for this.

## Notes

- **bella_distill weight 0.1 is the conservative default.** If `bella_distill_loss` is consistently much smaller than other losses (< 0.01), a Tier-D follow-up could raise weight to 0.3 — but only after 3-seed verdict on the default weight lands.
- **TASK_745 Option A semantics** (load-bearing for understanding the precompute): each shard is keyed `<start>__<end>__<rotation_hash[:8]>`, columns are remapped to the rotation's trainer asset order. Out-of-universe columns are NaN-masked. The trainer-side consumer (`train_easy.py:_load_bella_shard`) handles the NaN masking via `bella_valid_frac`.
- **Baseline status**: 734 flow_matching did NOT promote at 5-seed (mean Δα +0.20pp). Canonical Stage-B baseline stays `a1b1`. If a future mechanism promotes, re-pair this SPEC against the new baseline.
- **No multi-mechanism stacks queued.** Stacking is moot until at least one individual mechanism promotes 5-seed.
- **Workstation side**: the precompute shards and manifest are server-side only. The workstation manifest path doesn't exist; SPEC commands above run on server. Do NOT rsync `data/bella_preds/` to workstation (60+ MB, unnecessary).
