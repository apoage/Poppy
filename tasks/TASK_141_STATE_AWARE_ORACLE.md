# TASK_141: State-Aware Oracle with Profitability Threshold

## Status: IMPLEMENTED
Created: 2026-02-08
Implemented: 2026-02-08

## Problem Summary

Training plateaued at ~21% agreement after 103 merges. Root cause analysis identified fundamental issues with the oracle/imprinting target in `path_solver.py`:

### Bug 1: Hyperactive Agent Target
- `greedy_lookahead_path` ALWAYS picks exactly k=4 assets every timestep
- Oracle never teaches "hold nothing" or "only trade 2 assets today"
- Agent learns it MUST trade something every minute = noisy/volatile

### Bug 2: Stateless Oracle (Critical)
- Oracle knows what SHOULD be held, not what IS currently held
- When asset should exit: oracle says HOLD (does nothing!)
- Correct behavior: SELL to close long, BUY to close short
- Current mapping in `optimal_path_to_policy()`:
  ```python
  # BUG: selected + positive return → BUY (even if already long!)
  # BUG: not selected → HOLD (doesn't close existing position!)
  ```

### Bug 3: No Profitability Threshold
- Trade logit = 5.0, Hold logit = 0.0 (sharp target)
- Doesn't consider transaction costs (0.1%)
- Teaches agent to trade for 0.01% gains (unprofitable after fees)

## Solution Options

### Option 1: Pragmatic Oracle (Minimum Alpha Threshold)

Modify `greedy_lookahead_path()` to only trade if expected gain exceeds transaction cost.

```python
def greedy_lookahead_path(
    returns: jnp.ndarray,
    k: int = 4,
    lookahead: int = 5,
    transaction_cost: float = 0.001,
    min_alpha_threshold: float = 0.002,  # NEW: 2x transaction cost
    ...
) -> OptimalPath:
    """
    Thresholded selection: picks 0 to k assets.
    If nothing exceeds threshold, stays in cash (all HOLD).
    """
    # In step():
    cumulative_abs_returns = jnp.sum(jnp.abs(future_returns), axis=0)

    # NEW: Only select if return exceeds threshold
    above_threshold = cumulative_abs_returns > min_alpha_threshold * lookahead

    # Mask returns below threshold
    masked_returns = jnp.where(above_threshold, cumulative_abs_returns, -jnp.inf)

    # Top-k from remaining (may be fewer than k)
    top_k_indices = jax.lax.top_k(masked_returns, k)[1]

    # Only set mask where actually above threshold
    new_mask = jnp.zeros(n_assets)
    for i in range(k):
        idx = top_k_indices[i]
        new_mask = jnp.where(above_threshold[idx], new_mask.at[idx].set(1.0), new_mask)
```

**Result**: Oracle picks 0-k assets. If nothing looks good → stays flat. Fixes "Hyperactive Agent."

---

### Option 2: State-Aware Teacher (Closure Logic)

Update `optimal_path_to_policy()` to accept `prev_selections` (agent's current holdings).

```python
@partial(jax.jit, static_argnames=['n_actions_per_asset', 'temperature'])
def optimal_path_to_policy(
    selections: jnp.ndarray,      # (T, n_assets) optimal holdings for next step
    returns: jnp.ndarray,         # (T, n_assets) actual returns
    prev_selections: jnp.ndarray, # (T, n_assets) current holdings BEFORE this step
    n_actions_per_asset: int = 5,
    temperature: float = 1.0,
) -> jnp.ndarray:
    """
    State-aware policy targets with explicit closure logic.

    Transition logic:
    - prev=0, new=1, ret>0 → BUY (open long)
    - prev=0, new=1, ret<0 → SELL (open short)
    - prev=1, new=1        → HOLD (maintain position)
    - prev=1, new=0        → SELL (close long)
    - prev=-1, new=0       → BUY (close short)
    - prev=0, new=0        → HOLD (stay flat)
    """
    # Opening positions
    open_long = (prev_selections == 0) & (selections == 1) & (returns > 0)
    open_short = (prev_selections == 0) & (selections == 1) & (returns < 0)

    # Closing positions
    close_long = (prev_selections == 1) & (selections == 0)
    close_short = (prev_selections == -1) & (selections == 0)

    # Maintaining positions
    maintain = (prev_selections != 0) & (selections == prev_selections)

    # Staying flat
    stay_flat = (prev_selections == 0) & (selections == 0)
```

**Result**: "Not selected" means "GET OUT" not "do nothing." Fixes "Stateless Oracle."

---

### Option 3: Soft Advantage Target (Dynamic Logits)

Instead of fixed 5.0 logit gap, use actual expected alpha from path solver.

```python
def optimal_path_to_policy(
    selections: jnp.ndarray,
    returns: jnp.ndarray,
    expected_alphas: jnp.ndarray,  # (T, n_assets) expected gains per asset
    temperature: float = 0.1,
    ...
) -> jnp.ndarray:
    """
    Dynamic logits based on expected profitability.

    logits = expected_alpha / temperature

    Big expected move → confident target (logit=3.0)
    Small expected move → weak target (logit=0.5)
    No expected move → HOLD preferred
    """
    # Normalize alphas to reasonable logit range [0, 5]
    max_alpha = jnp.maximum(jnp.max(jnp.abs(expected_alphas)), 1e-6)
    normalized = expected_alphas / max_alpha * 5.0

    # Apply to action logits
    trade_logits = jnp.abs(normalized) / temperature
```

**Result**: Agent learns confidence calibration. Big moves = confident, small = cautious. Fixes "Fixed Logit Gap."

## Implementation Order

**Recommended: Option 1 + Option 2 together (complementary fixes)**

Option 1 reduces noise (fewer trades), Option 2 fixes the closing bug.
Option 3 is a nice-to-have refinement after 1+2 are validated.

## Files to Modify

1. **`mcts_trader/jax_agent/path_solver.py`**
   - `greedy_lookahead_path()`: Add min_alpha_threshold (Option 1)
   - `optimal_path_to_policy()`: Add prev_selections param (Option 2)
   - `batched_optimal_path_to_policy()`: Update signature

2. **`train_easy.py`**
   - Update oracle callback to:
     - Pass prev_selections (agent's current holdings)
     - Track position state across steps

3. **`mcts_trader/jax_agent/trainer.py`**
   - Update any trainer code using path_solver

## Validation

### Test 1: L0 Sine Wave Mastery
Before full fix, test on L0 only:
```bash
python train_easy.py --synthetic-level 0 --epochs 500
```
Target: 90%+ agreement on pure sine waves. If this fails, there's a more fundamental issue.

### Test 2: State-Aware Oracle
After implementing, verify:
- Close actions issued when positions need to exit
- No redundant BUY when already long
- HOLD count increases (agent allowed to be flat)

### Test 3: Profitability Check
Verify trades only recommended when return > threshold:
- Count trades per episode before/after
- Should see ~50% fewer trades with threshold

## Expected Outcome

| Metric | Before | After |
|--------|--------|-------|
| Agreement ceiling | 21% | 50%+ |
| Trades/episode | ~400 (every step) | ~100-200 |
| HOLD actions | ~0% | ~40-60% |
| Training stability | Unstable (Fisher crashes) | Stable |

## Priority

**HIGH** - This is the root cause of the 21% ceiling. Without fixing this, continued training is wasted compute.

## Notes

- Consider moving to 64x5 policy head later (per-asset policy)
- Fisher merging instability may also be related to noisy targets
- FedAvg was more stable because it averages out the noise

## Implementation Notes (2026-02-08)

### Changes Made

1. **path_solver.py**
   - `greedy_lookahead_path()`: Added `min_alpha_threshold` parameter
     - Masks returns below threshold with -inf before top-k selection
     - Allows 0 to k assets selected (not forced k)
   - `optimal_path_to_policy()`: Added `prev_selections` parameter
     - Computes transition logic: open, close, maintain, stay_flat
     - Proper SELL to close long, BUY to close short

2. **easy_config.py**
   - Added `min_alpha_threshold: float = 0.002` (2x transaction cost)
   - Added `state_aware_oracle: bool = True`

3. **train_easy.py**
   - Pass `min_alpha_threshold` to path solver
   - Compute `prev_selections` from oracle's previous state
   - Pass to `optimal_path_to_policy`

### Validation Results

| Config | Hold Ratio | Notes |
|--------|------------|-------|
| Old (threshold=0) | 0% | Bug confirmed - never HOLD |
| New (threshold=0.002) | ~20% | Working - HOLD when flat |

### Ready for Production

The changes are backward compatible:
- `min_alpha_threshold=0.0` gives old behavior
- `prev_selections=None` gives stateless mode

To test with e-dojo:
```bash
python tools/orchestration/e-dojo_v1.py --run --preset medium --epochs 500
```

## References

- Previous analysis in session notes
- `mcts_trader/jax_agent/path_solver.py:469-532` (current implementation)
- TASK_119 (original optimal path solver)
