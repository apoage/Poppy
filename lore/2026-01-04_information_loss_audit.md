# 2026-01-04: Information Loss Audit

## Critical Discovery: Three Information Bottlenecks

Deep inspection of the training data flow revealed three points where information was being destroyed:

---

## 1. Value Support vs PopArt Scale Mismatch (CRITICAL)

### The Bug
- Value support: [-750, +750] with 751 bins (designed for raw MC returns)
- PopArt normalized targets: ~[-3, +3] (mean=0, std=1)
- Result: Only bins 373-377 used (5 out of 751)
- **Information utilization: 0.5%**

### Why MSE ~0.8 Was Misleading
Network learned to predict bin 375 (value=0) for everything:
- Random init → predictions ~[-50, +50], targets ~[-3, +3] → MSE ~390
- Converged → predictions ~0, targets ~0 (normalized mean) → MSE ~1.0
- This was a **local minimum**, not actual learning!

### The Fix
Changed `mcts_trader/jax_agent/__init__.py`:
```python
# Before
VALUE_SUPPORT_MIN = -750
VALUE_SUPPORT_MAX = 750
VALUE_SUPPORT_SIZE = 751

# After
VALUE_SUPPORT_MIN = -10
VALUE_SUPPORT_MAX = 10
VALUE_SUPPORT_SIZE = 201
```

**Note:** 201 bins < 320 actions. Consider increasing to 401 bins for better resolution.
More bins than actions = better value differentiation between action choices.

---

## 2. Confidence Threshold Policy Disconnect

### The Bug
When MCTS forces HOLD due to low confidence:
1. Action changed to HOLD
2. Policy stored = original uncertain distribution (flat across actions)
3. Model trained: "In this state, take HOLD, but the 'correct' policy is uncertain"

### Impact
Model never learns to be confident about HOLD. Permanently relies on heuristic override.

### The Fix
Create synthetic confident policy when forcing HOLD:
```python
synthetic_policy = np.ones_like(probs) * (0.1 / (len(probs) - 1))
synthetic_policy[hold_action] = 0.9
policy_jax = jnp.array([synthetic_policy])
```

Fixed in all three self_play functions.

---

## 3. Reward Categorical Binning Collapse

### The Bug
- Per-step symlog rewards: ~0.0005 to 0.05
- Support: [-10, +10] with 201 bins
- All rewards mapped to bins 99-100 (only 2 bins!)

### Impact
Network can't distinguish reward magnitudes. Dynamics model reward prediction useless.

### The Fix
Scale rewards by 100× before categorical mapping:
```python
# trainer.py
REWARD_SCALE = 100.0
scaled_rewards = rewards[:, t - 1] * REWARD_SCALE

# mctx_search.py (descale for MCTS)
reward_scalar = self._categorical_to_scalar(output['reward']) / 100.0
```

Result: 2 bins → 8 unique bins for different reward magnitudes.

---

## Verification Results

### Value Binning (After Fix)
```
PopArt targets [-3, +3] → bins 70-130 (60 bins used)
Information utilization: 31.6% (was 0.5%)
```

### Reward Binning (After Fix)
```
0.05% return → bin 100
0.5% return  → bin 104  (+4 bins)
1% return    → bin 109  (+9 bins)
5% return    → bin 148  (+48 bins)
```

---

## Key Insight: Bins vs Actions

**Rule:** More bins than actions for proper value differentiation.

Updated to 401 bins > 320 actions:
```python
VALUE_SUPPORT_MIN = -10
VALUE_SUPPORT_MAX = 10
VALUE_SUPPORT_SIZE = 401  # 0.05 resolution, ~1.25 bins per action
```

This ensures the value head can distinguish between all action choices.

---

## Breaking Changes

**EXISTING CHECKPOINTS INCOMPATIBLE!**
- Value head changed from 751 → 201 bins
- Must delete `jax_ckpt_fast/*` and retrain from scratch

---

## Files Modified

| File | Change |
|------|--------|
| `mcts_trader/jax_agent/__init__.py` | Value support [-10,+10] 201 bins |
| `mcts_trader/jax_agent/trainer.py` | REWARD_SCALE = 100 |
| `mcts_trader/jax_agent/mctx_search.py` | Reward descaling |
| `mcts_trader/training/self_play.py` | Synthetic HOLD policy |
| `mcts_trader/jax_agent/trm_flax.py` | Comments |
| `mcts_trader/jax_agent/muzero_model.py` | Comments |

---

## Expected Outcomes After Retraining

1. Value MSE should **decrease during training** (not plateau at ~0.8)
2. MCTS root values should **correlate with MC returns**
3. Model learns **confident HOLD** in uncertain situations
4. Reward predictions become **meaningful** for tree search
5. Overall: Network actually learns instead of "blind steering"
