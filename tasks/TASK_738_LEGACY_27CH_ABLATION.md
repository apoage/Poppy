# TASK_738: Legacy 27-ch Encoder Ablation (TASK_732 Test)

**Status**: ACTIVE — attached to TASK_737 suite as Run 4 (2026-04-23).
**Priority**: HIGH — closes the last gap in the 737 ablation matrix (tests whether TASK_732's cross-sectional channels contribute once ExIt is working).
**Depends on**: `--legacy-27ch` flag (already wired in `train_phase2.py:1674,1808` and `env/jax_encoder.py:515-521`), 27-ch V3 Phase 1 checkpoint `jax_ckpt_easy_phase1_v3/easy_medium/phase1_best.pkl` (Apr 22, 97 MB).

---

## Motivation

TASK_737 isolates the contribution of TASK_735 (aux price head) and TASK_730 (rank head) but explicitly defers the TASK_732 test. The pre-work claim was "~30 min dev required"; audit of the codebase shows the flag is already fully wired (the feature snuck into the TASK_735 / TASK_736 commit window). Dev work is zero; only a training run is needed.

Adding this variant closes the ablation matrix: we now know the contribution of every head-side (730, 735) AND input-side (732) pivot individually, against the TASK_736 baseline.

---

## The Change

One variant, identical to TASK_736 config in every dimension except:
- Warmstart checkpoint: 27-ch V3 (`jax_ckpt_easy_phase1_v3/easy_medium/phase1_best.pkl`) instead of 32-ch xs32.
- Add `--legacy-27ch` to the launch command.

Rank head and aux price head stay ON so this variant isolates the encoder-input dimension alone (rank and aux continue to train; if they were the *only* load-bearing fix, 27-ch should still work).

---

## Files Modified

None. The flag and encoder gating existed already.

---

## Launch

Runs automatically as Run 4 of `tools/run_737_ablations.sh`. Manual standalone launch for reruns:

```bash
[redacted: host service detail]
mkdir -p jax_ckpt_easy_phase2_737_legacy_27ch
XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 nohup python train_phase2.py \
  --checkpoint jax_ckpt_easy_phase1_v3/easy_medium/phase1_best.pkl \
  --stage B --epochs 200 \
  --periods-file config/foxglove_periods.json --n-slots 17 --max-runs-per-group 10 \
  --factored --factored-temperature 0.1 \
  --no-reset-value-head --value-target-mode nstep --td-discount 0.95 \
  --value-weight 0.5 \
  --bc-weight-min 0.05 --bc-max-trade-weight 4.0 \
  --direction-weight 0.5 --encoder-lr 3e-5 \
  --encoder-version 3 --embed-dim 64 --encoder-heads 8 \
  --legacy-27ch \
  --rank-loss mse --rank-weight 1.0 \
  --aux-price-weight 0.1 \
  --output-dir jax_ckpt_easy_phase2_737_legacy_27ch \
  > jax_ckpt_easy_phase2_737_legacy_27ch/stage_b_nohup.log 2>&1 &
```

---

## Success Criteria

Compare Bal.Acc / Alpha / ModelPnL / beats-random against TASK_736 reference (Bal.Acc 26.5%, Alpha +3.87%, 6/6).

| Outcome | Interpretation | Next |
|---|---|---|
| legacy_27ch ≈ reference | TASK_732 cross-sectional channels don't help with working ExIt. They're valid inputs but not ceiling-breakers. | Keep 32-ch as default (free richness) but deprioritize further cross-sectional work. |
| legacy_27ch significantly worse (−2pp Alpha or −1 beats-random) | TASK_732 genuinely contributes. | Keep 32-ch. Consider Tier D2 (external features) as the next input-expansion bet. |
| legacy_27ch significantly better | Surprising — cross-sectional channels are adding noise post-MCTS. Investigate. | Audit which channel groups fight each other. Possible ordering-effect bug in encoder. |

"Significantly" = ≥ 2 pp Alpha or ≥ 1 beats-random or ≥ 3 pp Bal.Acc.

---

## Risks

- **Checkpoint mismatch**: the 27-ch V3 P1 checkpoint was trained at a different date (Apr 22) than the 32-ch xs32 (Apr 23). If there are subtle training-data or config drifts, this is not a pure encoder-input ablation. Mitigation: both P1 runs used the same Foxglove periods file; drift should be minimal but document if visible in loss curves.
- **VRAM fragmentation across sequential runs**: low; each process exit releases VRAM fully.
- **Pool rebuild cost**: 27-ch mode recomputes encoder caches from scratch; add ~5 min vs 32-ch runs. Total suite timing updates (8.5h → ~8.5h with rounded margin).

---

## Why Attached To 737 (Not a Standalone Task)

The ablation is a one-line-flag difference and the infra already supports it. Promoting it to a standalone task was unnecessary; Run 4 of the suite is the right home. This file exists to formalize the result slot so benchmark comparisons and post-mortem analysis have a canonical write-up target.
