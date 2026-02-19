# TASK_677: Turtle Breaker — Oracle Temperature Sharpening

## Problem

With multipass oracle (10 passes), soft policy targets are produced by averaging 10 one-hot policies. This averaging blurs the signal:

- Each pass: `optimal_path_to_policy` produces peaked logits (`5.0/temp` for correct action, `-10.0` for wrong). After softmax within each pass, targets are near-one-hot.
- After averaging 10 passes: if 7/10 say BUY and 3/10 say HOLD, target is `[0, 0.3, 0.7, 0, 0]`.
- This 70/30 split makes the "correct" action look ambiguous to the model. The TRM doesn't get a clear signal about *which structure* caused the trade.

The temperature exists per-pass (`optimal_path_to_policy` line 654) but doesn't help — each pass is already peaked. The blur comes from *disagreement between passes*, which is the intended diversity signal.

## Solution: Post-Averaging Temperature Sharpening

Apply temperature sharpening **after** the multipass averaging, not during:

```python
# In multipass_oracle_policy, after line 322:
soft_policy = jnp.mean(all_policies, axis=0)  # (T, 320) — current

# NEW: Sharpen the averaged soft targets
# Reshape to (T, 64, 5) for per-asset sharpening
soft_reshaped = soft_policy.reshape(T, n_assets, n_actions_per_asset)
# Log + temperature rescale + softmax (per-asset)
sharpened = jax.nn.softmax(jnp.log(jnp.maximum(soft_reshaped, 1e-8)) / sharpen_temperature, axis=-1)
soft_policy = sharpened.reshape(T, n_assets * n_actions_per_asset)
```

### Effect

| Target | Before (temp=1.0) | After (temp=0.1) |
|--------|-------------------|-------------------|
| 7/10 BUY, 3/10 HOLD | `[0, 0.3, 0.7, 0, 0]` | `[0, 0.0001, 0.9999, 0, 0]` |
| 5/10 BUY, 5/10 HOLD | `[0, 0.5, 0.5, 0, 0]` | `[0, 0.5, 0.5, 0, 0]` (symmetric, unchanged) |
| 10/10 HOLD | `[0, 1.0, 0, 0, 0]` | `[0, 1.0, 0, 0, 0]` (already peaked) |

Key property: sharpening amplifies majorities. A 70/30 split becomes 99.99/0.01. A 50/50 split stays 50/50 (maximum uncertainty preserved). Already-peaked targets are unchanged.

This gives the model "commands" (sharp BUY/SELL) instead of "suggestions" (soft distributions), while still preserving genuine ambiguity when passes truly disagree.

## Implementation

### Files to Change

1. **`easy_config.py`**: Add `oracle_sharpen_temperature: float = 0.1`
2. **`mcts_trader/jax_agent/path_solver.py`**:
   - `multipass_oracle_policy`: Add sharpening step after averaging (line 322)
   - `batched_multipass_oracle_policy`: Pass through `sharpen_temperature`
   - Add `sharpen_temperature` to `static_argnames` in JIT decorators

### Config

```python
# easy_config.py
oracle_sharpen_temperature: float = 0.1  # Post-averaging sharpening (0.1 = sharp commands, 1.0 = no sharpening)
```

### Gradient Impact

- Sharp targets concentrate cross-entropy gradient on the correct action
- Instead of "learn a little about BUY and a little about HOLD," the gradient says "learn BUY, ignore HOLD"
- Combined with TASK_677 warmup=0, trade signals get both equal gradient *mass* (adaptive weight) and equal gradient *clarity* (temperature)

## Status

- [x] Warmup disabled (`weight_warmup_epochs: 50 → 0`)
- [x] Continuous trade weighting (replaces binary `is_trade` mask)
- [x] Oracle temperature sharpening (`oracle_sharpen_temperature=0.1` in config, sharpening in `path_solver.py` after multipass averaging)

## Dependencies

- TASK_673: Multipass oracle (provides the soft targets to sharpen)
- TASK_675: `lax.map` for multipass (VRAM safety, no change needed)
