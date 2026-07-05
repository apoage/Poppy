# Bug: Entropy Bonus Logic is Backwards

## Priority: CRITICAL
**Status:** FALSE POSITIVE - VERIFIED CORRECT

## Analysis (2026-01-01)

After mathematical analysis, the entropy logic is CORRECT:

```
loss = ... - entropy_coef * H + adaptive_bonus
     = ... - entropy_coef * H + 5*entropy_coef*(min_H - H)  [when H < min_H]
     = ... - 6*entropy_coef * H + constant

d(loss)/dH = -6*entropy_coef < 0
```

Since the derivative is negative, gradient descent minimizes loss by INCREASING entropy.
The implementation correctly encourages exploration.
**Component:** `mcts_trader/jax_agent/trainer.py`

## Problem Description

The entropy bonus logic is **inverted**. It applies a penalty to low-entropy (deterministic) policies instead of a bonus to high-entropy (exploratory) policies. This means it **suppresses exploration** when entropy is low instead of encouraging it.

### Location

```python
# trainer.py:532-544
if self.entropy_bonus > 0:
    # This is BACKWARDS - applying penalty to low entropy
    # instead of bonus to high entropy
    entropy_penalty = self.entropy_bonus * jnp.maximum(
        0, self.target_entropy - entropy  # When entropy < target, this is positive
    )
    total_loss = total_loss + entropy_penalty  # Adding penalty INCREASES loss
```

### What It Does (Wrong)

- When `entropy < target_entropy`: penalty is positive, loss increases
- This **punishes** low entropy states
- But increasing loss for the current policy doesn't encourage exploration
- It just makes training unstable

### What It Should Do

- When `entropy < target_entropy`: **reduce policy confidence** (flatten logits)
- OR: add **negative term** to loss to encourage entropy maximization

### Impact

- Exploration bonuses don't work as intended
- Agents may collapse to deterministic policies
- Training instability when entropy drops

## Solution

### Option A: Flip the sign (quick fix)

```python
# SUBTRACT penalty to encourage higher entropy
total_loss = total_loss - entropy_penalty
```

### Option B: Use standard entropy regularization (recommended)

```python
# Add entropy as negative term (maximize entropy = minimize negative entropy)
entropy_bonus = -self.entropy_coef * entropy.mean()
total_loss = total_loss + entropy_bonus
```

### Option C: Soft constraint approach

```python
# Only apply when entropy too low
entropy_shortfall = jnp.maximum(0, self.target_entropy - entropy)
entropy_loss = self.entropy_bonus * jnp.mean(entropy_shortfall ** 2)  # Squared penalty
total_loss = total_loss + entropy_loss
```

## Files to Modify

| File | Change |
|------|--------|
| `trainer.py:532-544` | Fix entropy bonus sign/logic |

## Testing

1. Run training with logging entropy values
2. When entropy drops below target, verify it recovers
3. Compare policy entropy curves with/without fix

## Related Issues

- BUG_001: Entropy coefficient not applied (JIT capture issue)
