# 747_value_td0 — TD(0) Value Targets (paired 3-seed)

## Goal

Re-test **`--value-target-mode td0`** under the current paired-Δα protocol. TD(0) uses 1-step bootstrap: `target = r_t + γ * V(s_{t+1})`. Pure value-side change — does NOT touch encoder, classifier, dynamics, or policy.

This is the **other endpoint** of the bias-variance tradeoff vs 746_value_oracle:
- Oracle = lowest bias, lowest variance (uses ground truth)
- n-step (current) = medium bias from bootstrap, lower variance from N=20 averaging
- TD(0) = highest bias from single bootstrap, highest variance

Runs together with 746 as a 3-way ablation of the value-target choice.

## Background — historical caveat

TASK_719's own docstring calls TD(0) "too weak a signal" — that was the pre-718 state that motivated the whole roadmap. See `tasks/TASK_719_VALUE_TARGET_ROADMAP.md`:

> Root cause: TD(0) with consecutive pairs is too weak a signal.
> - Per-step reward ~0.1, γ=0.95 → td_target ≈ 0.1 + 0.95 * V(s_next)
> - Value targets depend on V(s_next) which is itself being learned → moving target
> - With only 1-step lookahead, reward information propagates extremely slowly

**Why re-test anyway:**
1. Same logic as 746 — we now have paired-Δα protocol, not value_loss.
2. Confirms (or refutes) the April finding under the new metric. Even a negative result is useful as a control + completes the value-target ablation matrix.
3. The April characterization was about *learning speed* not *Alpha*. The current setup has 200 epochs (plenty of time for slow signals to propagate) and longer training than the April work.
4. Low cost: 3 runs × 2.5h = 3.75h with 2-wide. Worth the budget.

## Hypothesis

**Null hypothesis (most likely)**: TD(0) gives mean Δα < −2pp or stays in inconclusive band — confirming the April finding that 1-step bootstrap is too weak.

**Alternative**: TD(0) wins on Δα because its high variance acts as a regularizer that prevents the over-confidence pattern that 738/741 demonstrated. (Low-prior but worth checking.)

## Prerequisites

- [x] Phase 1 ckpt: `jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl`
- [x] `--value-target-mode td0` wired in `train_phase2.py:976+` (TASK_719b code, else branch)
- [ ] a1b1 baseline runs at seeds {7,13,23} from `runs/741_mint/baseline_seed_{7,13,23}/`

## Resources

- 1× RTX 3090 per training, ~21 GB, ~2.5h
- TASK_742 rule: ≤2 concurrent train_phase2 per host
- GPUs **1+2** (NOT 0 — VirtualBox pinned)
- Total: 3 variant runs × 2.5h with 2-wide = **~3.75h wall** (baseline reused)

## Plan

| Variant | value target | seeds | source |
|---|---|---|---|
| **a1b1 baseline** | nstep, td-discount 0.95 | 7, 13, 23 | reuse `runs/741_mint/baseline_seed_*/` |
| **td0** | td0, td-discount 0.95 | 7, 13, 23 | new runs in this SPEC |

## Wave plan

Most efficient run order: chain after 746_value_oracle completes (same GPUs, same flag-drop pattern).

```
[After /tmp/vo_s23_done.flag drops + eval flags drop]
Wave 1 (T=0:00, ~2.5h):
  GPU 1: td0_seed_7
  GPU 2: td0_seed_13

Wave 2 (T=2:30, ~1.25h):
  GPU 1: td0_seed_23

T=3:45 — launch 3 evals in parallel
```

## Commands

```bash
cd ~/Trader && source trader_env/bin/activate

COMMON="--stage B --epochs 200 \
  --periods-file config/foxglove_periods.json \
  --n-slots 17 --max-runs-per-group 10 \
  --factored --factored-temperature 0.1 \
  --no-reset-value-head --value-target-mode td0 --td-discount 0.95 \
  --value-weight 0.5 \
  --bc-weight-min 0.05 --bc-max-trade-weight 4.0 \
  --direction-weight 0.5 --encoder-lr 3e-5 \
  --encoder-version 3 --embed-dim 64 --encoder-heads 8 \
  --checkpoint jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl \
  --rank-loss mse --rank-weight 1.0 \
  --aux-price-weight 0.1"

# For SEED in {7, 13, 23}, GPU in {1, 2}:
mkdir -p runs/747_value_td0/td0_seed_${SEED}
screen -d -m -S td0_s${SEED} bash -c "source trader_env/bin/activate && \
  CUDA_VISIBLE_DEVICES=<GPU> python -u train_phase2.py $COMMON \
  --seed ${SEED} \
  --output-dir runs/747_value_td0/td0_seed_${SEED} \
  --name 747_td0_s${SEED} 2>&1 \
  | tee runs/747_value_td0/td0_seed_${SEED}/train.log; \
  echo TD0_S${SEED}_DONE > /tmp/td0_s${SEED}_done.flag"
```

## Success criteria

- [ ] All 3 training runs complete 200 epochs (won't NaN — td0 is the simplest mode)
- [ ] **Paired Δα** vs 741_mint baseline (per seed): same promotion thresholds as 746
- [ ] **value_loss comparison**: expected higher than nstep (~April finding); but if Δα contradicts, that's the actual result
- [ ] Trade-rate per period: again, if td0 causes over-trading-collapse, document it

## Output expectations

```
runs/747_value_td0/
├── td0_seed_{7,13,23}/      train.log, phase2_stageB_best.pkl, stage_b_metrics.csv
├── eval_td0_seed_{7,13,23}/ benchmark.json, benchmark.log
└── result.md                paired Δα table + value_loss + trade-rate
```

## Notes

- If both 746 (oracle) and 747 (td0) lose, the value-side roadmap is closed for this stack.
- If 746 wins and 747 loses → n-step bootstrap was over-biased; oracle V* was the right anchor.
- If 746 loses and 747 wins → variance regularization beats bias reduction (unlikely but informative).
- If both win → maybe the trick is "anything but n-step", and n-step is the specific failure mode.
