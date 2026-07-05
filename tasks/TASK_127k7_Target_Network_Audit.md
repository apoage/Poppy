# TASK_127k7: Target Network Synchronization Audit

## Executive Summary

**CRITICAL FINDING:** The target network is defined, initialized, and updated periodically, but it is **NEVER ACTUALLY USED** in any computation. The `target_params` field exists in `TrainState` and is copied from `params` every 100 steps, but no code references `target_params` for value estimation, TD targets, or MCTS.

**Risk Level: HIGH** - Target network infrastructure exists but serves no purpose. Either remove it (wasted memory/computation) or use it properly (for Double Q-learning).

---

## Target Network Implementation

### ✅ Target Network Exists

**Location:** `mcts_trader/jax_agent/trainer.py:249-252`

```python
class TrainState(train_state.TrainState):
    """Extended train state with additional metrics and target network"""
    target_params: flax.core.FrozenDict = None  # RESTORED: Target network for Double Q-learning
    step_count: int = 0
```

### ✅ Target Network Initialized

**Location:** `trainer.py:588-594`

```python
state = TrainState.create(
    apply_fn=self.model.apply,
    params=params,
    target_params=params,  # RESTORED: Initialize target network with same params
    tx=self.optimizer,
    step_count=0
)
```

### ✅ Target Network Updated

**Location:** `trainer.py:1508-1519`

```python
# RESTORED: Update target network periodically (Double Q-learning stability)
should_update_target = (new_step_count % self.target_update_interval) == 0
new_target_params = jax.lax.cond(
    should_update_target,
    lambda: new_state.params,      # True: copy current params to target (HARD UPDATE)
    lambda: state.target_params    # False: keep old target params
)

new_state = new_state.replace(
    target_params=new_target_params,
    step_count=new_step_count
)
```

**Configuration:**
```python
# trainer.py:384
target_update_interval: int = 100,  # Every 100 steps
```

---

## 🔴 CRITICAL: Target Network Never Used

### Where Target Network SHOULD Be Used

| Use Case | Expected Usage | Actual Usage |
|----------|----------------|--------------|
| **Value bootstrapping** | `target_params` for TD targets | `params` used ❌ |
| **MCTS value estimation** | `target_params` for leaf values | `params` used ❌ |
| **Q-value targets** | `target_params` for stable targets | `params` used ❌ |

### Where Target Network IS Used

**NOWHERE.**

Full grep results for `target_params`:
```
mcts_trader/jax_agent/trainer.py:251     # Definition
mcts_trader/jax_agent/trainer.py:591    # Initialization  
mcts_trader/jax_agent/trainer.py:1510   # Update (read)
mcts_trader/jax_agent/trainer.py:1513   # Update (read)
mcts_trader/jax_agent/trainer.py:1517   # Update (write)
```

**Zero references for actual computation.**

---

## How Target Network Should Work

### Standard Double Q-Learning Pattern

```python
# 1. Current implementation (NO target network)
def _loss_fn(params, batch):
    # Online network computes both current and next values
    current_values = model.apply(params, batch['obs'])
    next_values = model.apply(params, batch['next_obs'])  # Same params!
    td_target = batch['reward'] + gamma * next_values
    loss = (current_values - td_target) ** 2

# 2. Proper Double Q-Learning (WITH target network)
def _loss_fn(params, target_params, batch):
    # Online network for current values
    current_values = model.apply(params, batch['obs'])
    # Target network for stable next values
    next_values = model.apply(target_params, batch['next_obs'])  # Different params!
    td_target = batch['reward'] + gamma * next_values
    loss = (current_values - td_target) ** 2
```

### Benefits of Target Network

1. **Stability**: Prevents moving target problem
2. **Reduced variance**: TD targets don't change every step
3. **Better convergence**: Decouples action selection from evaluation (Double DQN)

---

## Where Target Network Should Be Integrated

### 1. Value Bootstrapping in Returns

**Current:** `mcts_trader/jax_agent/returns.py:66`

```python
# Uses same 'values' for bootstrapping (online network)
next_value = values_with_bootstrap[t + 1] * not_done
delta = rewards[t] + discount * next_value - values[t]
```

**Should be:**

```python
# Use target network for bootstrap values
target_values = compute_values(target_params, observations)
next_value = target_values[t + 1] * not_done
delta = rewards[t] + discount * next_value - values[t]
```

### 2. MCTS Search

**Current:** `mcts_trader/jax_agent/mctx_search.py:74-78`

```python
# Root uses online params
output = self.model_apply_fn(
    {'params': params},  # Online network
    observation,
    mode='initial_inference'
)
```

**Could optionally use:**

```python
# Root could use target params for more stable MCTS values
output = self.model_apply_fn(
    {'params': target_params},  # Target network
    observation,
    mode='initial_inference'
)
```

### 3. TD Target Computation

**Current:** Uses MCTS root values from self-play (which use online params)

**Should be:** Compute TD targets using target network during training

---

## Current Value Target Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CURRENT VALUE TARGET FLOW (No Target Net)                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Self-Play                          Training                                 │
│  ─────────                          ───────                                  │
│  1. Run MCTS with online params ──▶ Store trajectory                         │
│  2. MCTS root values = V_online  ─▶ As 'values'                              │
│  3. Store (obs, V_online, ...)    ─▶ In buffer                               │
│                                                                              │
│  Training Loss:                                                              │
│  ──────────────                                                              │
│  V_pred = model(params, obs)        # Online network                         │
│  V_target = batch['values']         # Also from online network!              │
│  loss = (V_pred - V_target)²                                                 │
│                                                                              │
│  Problem: V_target changes every step as params updates                      │
│  → Moving target problem                                                     │
│  → Training instability                                                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Recommended Fix: Two Options

### Option 1: Remove Target Network (Simple)

If target network isn't needed, remove the overhead:

```python
class TrainState(train_state.TrainState):
    """Train state without unused target network"""
    step_count: int = 0
    # REMOVED: target_params

# Remove update logic
# Remove from checkpoint save/load
```

**Pros:**
- Simpler code
- Less memory (no duplicate params)
- Faster checkpoint save/load

**Cons:**
- Loses potential stability benefits
- May need to re-add later

---

### Option 2: Properly Implement Target Network (Recommended)

Use target network for TD target computation:

```python
def compute_td_targets(self, batch, target_params):
    """Compute TD targets using target network for stability."""
    # K-step unroll with target network
    targets = []
    for t in range(unroll_steps):
        # Use target params for value estimation
        output = self.model.apply(
            {'params': target_params},
            batch['observations'][:, t],
            mode='initial_inference'
        )
        value = self._categorical_to_scalar(output['value'])
        targets.append(value)
    return jnp.stack(targets, axis=1)

def _loss_fn(self, params, batch, rng, target_params):
    """Loss with target network for TD targets."""
    # Current values from online network
    current_values = self.model.apply({'params': params}, ...)
    
    # TD targets from target network
    td_targets = self.compute_td_targets(batch, target_params)
    
    # Value loss with stable targets
    value_loss = (current_values - td_targets) ** 2
    ...
```

**Also consider soft (Polyak) updates:**

```python
# Current: Hard update (copy every N steps)
new_target_params = jax.lax.cond(
    should_update_target,
    lambda: new_state.params,      # Full copy
    lambda: state.target_params
)

# Better: Soft update (Polyak averaging)
tau = 0.005  # Soft update coefficient
new_target_params = jax.tree_map(
    lambda target, online: (1 - tau) * target + tau * online,
    state.target_params,
    new_state.params
)
```

**Pros:**
- More stable training
- Reduced variance in value estimates
- Standard best practice

**Cons:**
- More memory (duplicate params)
- More computation

---

## Hard vs Soft Updates

### Current: Hard Updates

```python
# Every 100 steps: copy online → target
if step % 100 == 0:
    target_params = params  # Full copy
```

- ✅ Simple
- ⚠️ Sudden target changes can destabilize

### Recommended: Soft (Polyak) Updates

```python
# Every step: slowly blend online → target
tau = 0.005  # Or 0.01 for faster tracking
target_params = (1 - tau) * target_params + tau * params
```

- ✅ Smoother transitions
- ✅ Better stability
- ⚠️ Requires update every step (can be optimized)

---

## Memory Impact

Target network doubles parameter memory:

```python
# Example: ~10M parameter model
params_size = 10M * 4 bytes = 40 MB
target_params_size = 40 MB  # Duplicate!

# With optimizer state (Adam: 2x for m, v)
optimizer_state = 80 MB
total = params + target + optimizer = 160 MB

# Without target network:
total = params + optimizer = 120 MB  # 25% less
```

For large models, this matters. For this trading model (~few million params), it's minor.

---

## Recommendations

### Immediate Action Required

1. **Decide**: Either USE the target network or REMOVE it
2. **Don't leave dead code** - It's confusing and wastes resources

### If Using Target Network:

```python
# 1. Add target_params to value computation
target_values = self.model.apply(
    {'params': state.target_params},
    batch['observations'],
    mode='initial_inference'
)

# 2. Use for TD targets
td_targets = rewards + gamma * target_values

# 3. Consider soft updates
tau = 0.005
new_target_params = jax.tree_map(
    lambda t, p: (1 - tau) * t + tau * p,
    state.target_params,
    new_state.params
)
```

### If Removing Target Network:

```python
# 1. Remove from TrainState
class TrainState(train_state.TrainState):
    step_count: int = 0
    # Remove: target_params

# 2. Remove update logic
# 3. Update checkpoint handling
# 4. Update config
```

---

## MuZero-Specific Considerations

MuZero is different from standard Q-learning:

| Aspect | Standard DQN | MuZero |
|--------|--------------|--------|
| Value source | Q-network | MCTS search |
| Bootstrap | next Q-values | MCTS values |
| Target network benefit | High | Unclear |

**In MuZero:**
- MCTS already provides "improved" value estimates
- Values come from tree search, not direct network
- Target network benefit may be reduced

**However:**
- TD(λ) still bootstraps from value estimates
- Target network could stabilize TD learning
- Worth experimenting

---

## Acceptance Criteria

- [ ] Target network is either:
  - [ ] Used for TD target computation, OR
  - [ ] Removed from codebase
- [ ] Decision documented with rationale
- [ ] If using target network:
  - [ ] Soft updates (Polyak) considered
  - [ ] Update frequency tuned
  - [ ] Memory impact assessed

---

## References

- `mcts_trader/jax_agent/trainer.py:251` - Target params definition
- `mcts_trader/jax_agent/trainer.py:591` - Target params initialization
- `mcts_trader/jax_agent/trainer.py:1508-1519` - Target params update
- `mcts_trader/jax_agent/returns.py` - Value bootstrapping (should use target)
- `mcts_trader/jax_agent/mctx_search.py` - MCTS (could use target for values)
