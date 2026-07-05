# TASK_736: Turn On Value Training in Stage B

**Status**: PROPOSED (2026-04-23). Single-flag structural fix.
**Priority**: HIGHEST — supersedes all §46 / §47 / §48 options if the diagnosis in `lore/notes_2026-04-22.md §49` is correct.
**Depends on**: 32-ch Phase 1 checkpoint `jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl`.

---

## One-sentence summary

`--value-weight 0.0` has been set on every Stage B run since 729c. It silently disables MCTS by keeping the value head frozen-and-drifting, which makes MCTS visit distributions uniform, which makes `policy_loss = log(5) = 1.609` every single epoch across all 6 pivots. Flip the flag to `0.5` and the ExIt loop should actually close.

---

## Evidence (from §49)

`policy_loss` stays at 1.6118 ± 0.0002 from E0 to the last epoch of all 6 Stage B runs (728c, 729c, 730, 732, 734, 735). `log(5) = 1.6094`. The policy is producing uniform 5-way per-asset distributions the whole time because MCTS is producing uniform 5-way visit targets the whole time because the value head is producing garbage Q-values the whole time because it isn't being trained.

Turn on value training → value head tracks drifting latent → MCTS Q-values become meaningful → MCTS visit distribution concentrates → policy CE target isn't uniform → policy learns.

---

## The Change

ONE CLI flag flip. No code changes.

**Before (732 baseline):** `--value-weight 0.0`
**After (TASK_736):** `--value-weight 0.5`

Why 0.5 not 1.0: value_loss is currently ~50 (huge). At weight 1.0 it would dominate total_loss by 5-10× over policy/bc/direction/rank/aux_price combined. At 0.5 it's still significant but leaves room for the other signals to matter. Can tune up to 1.0 if it helps.

---

## Files Modified

None. Pure config change via existing CLI.

---

## Launch

[redacted: host service detail]

```bash
mkdir -p jax_ckpt_easy_phase2_736 && \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 nohup python train_phase2.py \
  --checkpoint jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl \
  --stage B --epochs 200 \
  --periods-file config/foxglove_periods.json --n-slots 17 --max-runs-per-group 10 \
  --factored --factored-temperature 0.1 \
  --no-reset-value-head --value-target-mode nstep --td-discount 0.95 \
  --value-weight 0.5 \
  --bc-weight-min 0.05 --bc-max-trade-weight 4.0 \
  --direction-weight 0.5 --encoder-lr 3e-5 \
  --encoder-version 3 --embed-dim 64 --encoder-heads 8 \
  --rank-loss mse --rank-weight 1.0 \
  --aux-price-weight 0.1 \
  --output-dir jax_ckpt_easy_phase2_736 \
  > jax_ckpt_easy_phase2_736/stage_b_nohup.log 2>&1 &
```

The *only* diff from the TASK_735 launch command is `--value-weight 0.0` → `--value-weight 0.5`.

---

## Gates

### E5 — early signal
- `policy_loss < 1.600` → policy is moving off the uniform attractor. Strongest possible early positive.
- `value_loss` dropping from its starting ~50 → value head is actively training. Prerequisite for everything else.
- If both true: you'll see these in the first 5 epochs. If neither moves in 20 epochs, the diagnosis was wrong.

### E20 — budget gate
- `dir_acc > 0.60` → ceiling broken. Declare victory early, continue to E50 for confirmation.
- `dir_acc 0.58–0.60` + policy_loss trending down → on track, run to E100.
- `dir_acc < 0.58` and policy_loss still at 1.61 → value training wasn't the blocker. Kill and diagnose what's downstream.

### E50 — decision gate
- `dir_acc > 0.60` → full run, benchmark.
- Ceiling holds at 0.57 → value_weight was a real issue but not the only one. Stack: also raise `--factored-temperature` 0.1 → 0.5 (allows MCTS to explore more given working Q-values).

### Benchmark (when E50 passes)
- `tools/benchmark_easy.py --checkpoint jax_ckpt_easy_phase2_736/phase2_stageB_best.pkl`
- Bal.Acc > 45% → ship-candidate. First time in project history.
- Bal.Acc 38–45% → big win relative to 728c's 34%; stack further fixes.
- Bal.Acc < 38% → latent ceiling is real, not a Stage B artifact.

---

## Risks

- **Value explosion in first few epochs.** value_loss is ~50 at start. With weight=0.5, total_loss jumps by ~25 immediately. If gradients blow up (grad_norm > 100), kill and retry with `--value-weight 0.25` or reset the value head via `--reset-value-head` (drop `--no-reset-value-head`).
- **PopArt stats are stale.** Phase 1 value head was trained with different PopArt running stats. Stage B will re-normalize. Transient instability expected; give 5-10 epochs before judging.
- **value_loss never drops.** If value_loss stays flat at 50 despite weight 0.5, the value head can't learn from this latent. That would be a separate problem.

---

## Why This Over Everything Else (§46 / §47 / §48)

Every prior option assumed *something is wrong with the training signal*: oracle is too sparse (§46 A1), encoder can't see cross-sectional structure (732), prediction pathway missing (α-v1/v3). All valid hypotheses. None of them addressed the hidden fact that **MCTS itself has been producing a uniform target across all 6 runs** — which means none of those training signals ever reached the policy head through MCTS.

Fixing value training restores ExIt. With ExIt working again, the older hypotheses (oracle density, encoder expressiveness, etc.) become *testable*. Right now they've all been run on a crippled MCTS and the results don't tell us anything about the underlying hypothesis.

This is the cheapest test in the project's history (one flag, no code) with the highest probability of actually moving the ceiling. Run it before anything else.
