# TASK_734: Prediction-Head Pathway — Path α

**Status**: ACTIVE (as of 2026-04-23 ~08:30).
**Priority**: HIGHEST. First step in the prediction-head pathway (course of action in `lore/notes_2026-04-22.md §47`).
**Depends on**: 32-ch Phase 1 checkpoint at `jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl`.

---

## Scope

Minimum-viable test of the prediction-head pathway: wire the existing but unused **flow-matching loss** (`mcts_trader/jax_agent/flow_matching.py::apply_flow_matching_loss`) into Stage B training alongside the current MSE consistency loss. Tests whether velocity-matching (predict `latent_{t+1} - latent_t` as the learning target) gives the dynamics MLP a better gradient than state-matching MSE.

Known failure mode for MSE: identity collapse (`latent_{t+1} ≈ latent_t` → loss ≈ 0 on calm steps → model learns "no change"). Flow matching was designed precisely to avoid this (identity predicts v=0 which is wrong whenever the market actually moves).

This is α-v1. If it moves dir_acc above 0.60 on the existing 32-ch Phase 1 warmstart, escalate to α-v2 (enable `use_prediction_head=True`, wire NeuralMarketODE + flow-matching on log-price trajectory).

---

## The Change

### α-v1 — flow-matching loss on the dynamics MLP output (this task)

One new CLI flag (`--flow-matching`) and ~20 lines of wiring in `train_phase2.py::make_stage_b_train_step`:

```python
# Currently:
consistency_loss = jnp.mean((dyn_out['latent_state'] - latent_t1) ** 2)

# With --flow-matching:
from mcts_trader.jax_agent.flow_matching import apply_flow_matching_loss

fm_loss, fm_metrics = apply_flow_matching_loss(
    pred_states   = dyn_out['latent_state'][:, None, :],   # (B, 1, state_dim)
    actual_states = latent_t1[:, None, :],                 # (B, 1, state_dim)
    prev_states   = latent_t_sg[:, None, :],               # (B, 1, state_dim)
    use_flow_matching=True,
    velocity_weight=0.8,
)
consistency_loss = fm_loss
# + emit fm_metrics['flow_velocity_loss'], fm_metrics['flow_state_loss']
#   into the metrics dict for log/CSV.
```

Unified helper — returns a weighted velocity + state loss. State-matching component (weight 0.2) stays as regularization so we don't lose the MSE signal entirely.

### α-v2 — enable NeuralMarketODE prediction head (future follow-up, not this task)

Only if v1 moves dir_acc. Would require: flip `use_prediction_head=True`, extract future log-prices from batch windows (already present in `_prepare_numpy`'s `windows` array at indices `[ws : ws+nstep]`), call `model.predict_multi_step(...)`, apply flow-matching on predicted-vs-actual log-prices. Deferred until v1 gate.

---

## Files Modified (α-v1 scope)

| File | Change |
|---|---|
| `phase2_config.py` | Add `use_flow_matching: bool = False`, `flow_matching_velocity_weight: float = 0.8`, `flow_matching_weight: float = 1.0` (multiplier on the total flow loss, keeps consistency_w separate). |
| `train_phase2.py` | Add CLI flags `--flow-matching`, `--flow-matching-velocity-weight`. Wire into config override block. In `make_stage_b_train_step`: import `apply_flow_matching_loss`; if enabled, replace the existing `consistency_loss = jnp.mean(...)` with the flow-matching version; emit `flow_velocity_loss` + `flow_state_loss` into metrics; add to CSV schema + log line. |

No changes to `muzero_model.py`. No encoder changes. No new modules.

---

## Smoke Test

```bash
# Forward-pass sanity — same as test-init for TASK_730/732
source trader_env/bin/activate && \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 python train_phase2.py \
  --checkpoint jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl \
  --stage B --encoder-version 3 --embed-dim 64 --encoder-heads 8 \
  --flow-matching --test-init
```

Must pass the same assertions the 730/732 `--test-init` ran (dynamics init OK, recurrent_inference shapes correct). `--flow-matching` flag shouldn't affect init — just confirms the CLI routing works.

Additionally: a one-liner Python test that calls `apply_flow_matching_loss` with dummy `(B, 1, state_dim)` tensors and verifies the return tuple structure.

---

## Launch

```bash
mkdir -p jax_ckpt_easy_phase2_734 && \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 nohup python train_phase2.py \
  --checkpoint jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl \
  --stage B --epochs 200 \
  --periods-file config/foxglove_periods.json --n-slots 17 --max-runs-per-group 10 \
  --factored --factored-temperature 0.1 \
  --no-reset-value-head --value-target-mode nstep --td-discount 0.95 \
  --value-weight 0.0 --bc-weight-min 0.05 --bc-max-trade-weight 4.0 \
  --direction-weight 0.5 --encoder-lr 3e-5 \
  --encoder-version 3 --embed-dim 64 --encoder-heads 8 \
  --rank-loss mse --rank-weight 1.0 \
  --flow-matching --flow-matching-velocity-weight 0.8 \
  --output-dir jax_ckpt_easy_phase2_734 \
  > jax_ckpt_easy_phase2_734/stage_b_nohup.log 2>&1 &
```

[redacted: host service detail]

Tail (dir_acc ≥ 0.60 or velocity-loss anomaly or error):
```bash
tail -F jax_ckpt_easy_phase2_734/stage_b_nohup.log | \
  grep --line-buffered -E "acc=0\.6[0-9]|acc=0\.[789]|velocity_loss|Traceback|ERROR|NaN|Killed|OOM|aborted"
```

---

## Gates

### Diagnostic — first 20 epochs
- `velocity_loss` should drop during training (dynamics learns velocity structure). If it stays at initial value → flow-matching loss isn't flowing gradients, wiring bug.
- `state_loss` (regularizer) should also drop but more slowly than velocity_loss.
- If both drop and converge → flow matching is training the dynamics head.

### E50 gate (main question)
- `dir_acc > 0.60` → flow-matching on dynamics adds a real signal to the latent. α is a win; escalate to α-v2 (NeuralMarketODE prediction head).
- `dir_acc 0.58–0.60` → marginal; run the full 200 epochs to see if it converges higher, otherwise call it a negative.
- `dir_acc < 0.58` → flow-matching on dynamics didn't help. Prediction-head pathway may not be the answer at all. Roll back, try a different Tier (A1 sign-oracle, B1 no-BC, or C1 cheat ceiling per §46).

### Benchmark
- If E50 passes, run `tools/benchmark_easy.py` on `jax_ckpt_easy_phase2_734/phase2_stageB_best.pkl` at E50 (don't wait for E200 — budget).
- Ship-gate: Bal.Acc > 45% → ship candidate. 38-45% → stack α-v2 (Bella or NeuralMarketODE). < 38% → roll back pathway.

---

## Known Risks

- **Velocity scale mismatch**: `latent_{t+1} - latent_t` in a state_dim=64 latent may have very small magnitude. Velocity-loss could be near-zero without being meaningful. Watch `velocity_pred_norm` / `velocity_actual_norm` metrics — they should be O(1) or tunable with `velocity_weight`.
- **Double-counting consistency**: `consistency_w * consistency_loss` is the existing weighted term. If we swap `consistency_loss` for `fm_loss`, the `consistency_w=0.5` scalar still applies. Verify we're not stacking weights.
- **Stage A compatibility**: this change is Stage B only. Stage A (dynamics warmup) is untouched; if someone runs `--stage A --flow-matching`, the flag silently does nothing. Acceptable for now.
- **Rolling back is trivial**: flag-gated, default off. Ship with `--flow-matching` absent and behavior is identical to TASK_732.

---

## Why This First (Among α/β/γ)

- ~20 lines of code. Smallest possible test of the prediction-head hypothesis.
- Uses only code that already exists in the repo (flow_matching.py since January, never called).
- If this doesn't move, no Bella-integration or NeuralMarketODE-activation will help either (they use the same flow-matching math in a more complex wrapper). Saves weeks of work on β/γ if the core loss change is a no-op.
- Failure is cheap: one evening of training, then fallback to §46 Tier A (sign-oracle) or C1 (ceiling diagnostic).
