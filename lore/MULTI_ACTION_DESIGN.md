# Multi-Action Per Step Design

**Status**: Future consideration (breaking change)
**Complexity**: Very High
**Prerequisite**: Stable single-action training

---

## Problem Statement

**Current**: 1 action per step (select 1 of 192/385 discrete actions)

**Limitation**: Real trading often requires simultaneous actions:
- Rebalance: Sell BTC + Buy ETH in same step
- Hedge: Long BTC + Short correlated asset
- Portfolio rotation: Exit 3 positions, enter 2 new ones

**Current workaround**: Execute one trade per step (slow, misses simultaneous opportunities)

---

## Why It's Hard

### 1. Combinatorial Explosion

```
Single action: 192 choices
Multi-action (binary per asset): 2^64 × 3^64 = astronomically large
```

### 2. MCTS Compatibility

MCTS expects discrete action → state transition. Multi-action breaks this:
- How to enumerate possible multi-actions?
- How to compute policy over exponential space?

### 3. Credit Assignment

Which action in the set caused the reward? Hard to attribute.

### 4. Order of Execution

If Buy A and Sell B, which executes first? Price impact?

---

## Possible Approaches

### Approach A: Autoregressive Actions

Select actions one-by-one until "DONE" token:

```
Step 1: Select action_1 (e.g., Buy BTC)
Step 2: Select action_2 (e.g., Sell ETH)
Step 3: Select DONE → execute all, advance time
```

**Pros**:
- Discrete actions preserved
- MCTS compatible (each sub-step is a node)

**Cons**:
- Variable-length action sequences
- Slower (multiple forward passes)
- Order matters (first action affects state for second)

### Approach B: Factorized Policy

Independent policy per asset:

```python
def multi_action_policy(latent):
    # Output: (batch, 64, 3) - probability for each asset × action
    per_asset_logits = Dense(64 * 3)(latent).reshape(-1, 64, 3)

    # Sample independently for each asset
    actions = []
    for asset in range(64):
        action = sample(softmax(per_asset_logits[:, asset, :]))
        actions.append(action)

    return actions  # List of 64 actions
```

**Pros**:
- Single forward pass
- All assets decided simultaneously

**Cons**:
- Ignores cross-asset dependencies (BTC decision affects ETH decision)
- Not compatible with standard MCTS
- May need policy gradient instead of MCTS

### Approach C: Action Set Prediction

Predict which assets to act on, then action type:

```
Stage 1: Binary mask - which assets to trade? (64 sigmoids)
Stage 2: For selected assets, what action? (selected × 3)
```

**Pros**:
- Reduces combinatorial explosion
- Can limit max trades per step

**Cons**:
- Two-stage prediction
- Still exponential in worst case

### Approach D: Portfolio Target (Continuous)

Instead of discrete actions, predict target portfolio weights:

```python
def portfolio_policy(latent):
    # Output: target allocation (batch, 64)
    target_weights = softmax(Dense(64)(latent))

    # Environment computes trades needed to reach target
    trades = compute_rebalancing_trades(current_weights, target_weights)
    return trades
```

**Pros**:
- Natural multi-action (rebalancing is multi-trade)
- Continuous optimization possible

**Cons**:
- Major architecture change
- Loses discrete MCTS benefits
- Harder to learn precise allocations

---

## Recommended Path

### Phase 1: Keep Single Action (Current)
- Focus on making single-action trading profitable
- Use fast step times (1 min) to approximate multi-action

### Phase 2: Autoregressive (If Needed)
- Add "DONE" action to signal end of multi-action
- Limit max actions per step (e.g., 5)
- MCTS treats each sub-action as separate node

### Phase 3: Factorized (Research)
- Experiment with independent per-asset policies
- May need to abandon MCTS for policy gradient

---

## Training Considerations

### Curriculum for Multi-Action

1. **Level N**: Single action only (current)
2. **Level N+1**: Allow 2 actions per step
3. **Level N+2**: Allow up to 5 actions per step

### Reward Shaping

```python
# Encourage efficient multi-action
if num_actions > 1:
    # Bonus for correlated actions (hedge/rebalance)
    correlation_bonus = compute_action_synergy(actions)
    reward += correlation_bonus

    # Penalty for random multi-action (noise)
    if not actions_are_coherent(actions):
        reward -= noise_penalty
```

---

## Dependencies

Before implementing multi-action:
- [ ] Single-action training stable (value correlation >0.9)
- [ ] Portfolio curriculum working (Levels 4-6)
- [ ] Advanced orders implemented (limit/stop)
- [ ] Cross-asset correlation understood

---

## References

- AlphaStar: Autoregressive action heads for complex action spaces
- OpenAI Five: Factorized action space for MOBA
- Portfolio optimization literature: Continuous weight targets
