# TASK_041: Flow Matching for Prediction Head

## Status: IMPLEMENTED (2026-01-06)

**Implementation:**
- `mcts_trader/jax_agent/flow_matching.py` - Core module with `apply_flow_matching_loss()`
- `train_jax_fast.py` - Added `--flow-matching`, `--flow-velocity-weight` flags
- Note: Full trainer integration pending (module ready for use)
## Priority: HIGH (Addresses prediction collapse root cause)

## Problem Statement

Current prediction head uses state-matching loss:
```python
loss = MSE(predicted_state_{t+k}, actual_state_{t+k})
```

This has a degenerate solution: **identity mapping**.
- If model predicts `state_{t+1} ≈ state_t`, loss is small when market is stable
- During HOLD-dominated training, market dynamics look like "no change"
- Prediction head learns: "just output the input" → trivial solution

**Evidence**: After warmup, prediction loss drops to ~0.01 (near-identity).

## Research Basis

From Gemini Deep Research:
> "Replace prediction state loss with a 'flow-matching' objective: train to predict
> velocity `v = (s_{t+1} - s_t) / dt` rather than next state. Identity function
> predicts v=0, which is WRONG when market actually moves. This forces the model
> to learn dynamics."

From Flow Matching literature:
> "Flow matching trains a neural network to predict the velocity field that
> transforms samples from one distribution to another."

## Solution: Velocity (Flow) Matching

Instead of predicting final state, predict the **velocity** (change):

### Current (State Matching)
```python
# Predict s_{t+1}
predicted_state = dynamics_model(state_t, action)
loss = MSE(predicted_state, actual_state_{t+1})
# Identity solution: predicted = state_t, loss ≈ 0 for stable markets
```

### Proposed (Flow Matching)
```python
# Predict velocity v = ds/dt
predicted_velocity = dynamics_model(state_t, action)
actual_velocity = actual_state_{t+1} - state_t
loss = MSE(predicted_velocity, actual_velocity)
# Identity solution: v=0, but actual_velocity ≠ 0 when market moves → HIGH loss
```

**Key insight**: Identity solution now produces v=0, which is wrong unless market is perfectly flat.

## Implementation

### Phase 1: Modify Dynamics Output

**File: `mcts_trader/jax_agent/muzero_model.py`**

```python
class DynamicsNetwork(nn.Module):
    """Dynamics model predicting velocity (flow) instead of next state."""
    state_dim: int
    action_dim: int
    hidden_dim: int
    flow_matching: bool = False  # NEW: Enable flow matching

    @nn.compact
    def __call__(self, state, action):
        # Embed action
        action_embed = nn.Dense(self.hidden_dim)(action_one_hot)

        # Combine state and action
        combined = jnp.concatenate([state, action_embed], axis=-1)

        # Process through hidden layers
        x = nn.Dense(self.hidden_dim)(combined)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)

        if self.flow_matching:
            # Output: velocity (change in state)
            velocity = nn.Dense(self.state_dim)(x)
            # Next state = current state + velocity
            next_state = state + velocity
            return next_state, velocity  # Return both for loss computation
        else:
            # Output: next state directly
            next_state = nn.Dense(self.state_dim)(x)
            return next_state, None
```

### Phase 2: Flow Matching Loss

**File: `mcts_trader/jax_agent/trainer.py`**

```python
def compute_prediction_loss(
    predicted_states: jnp.ndarray,
    predicted_velocities: jnp.ndarray,  # NEW
    actual_states: jnp.ndarray,
    prev_states: jnp.ndarray,           # NEW: for velocity computation
    flow_matching: bool = False,
) -> jnp.ndarray:
    """Compute prediction loss with optional flow matching."""

    if flow_matching and predicted_velocities is not None:
        # Flow matching: predict velocity
        actual_velocities = actual_states - prev_states
        velocity_loss = jnp.mean((predicted_velocities - actual_velocities) ** 2)

        # Optional: also include state loss for regularization
        state_loss = jnp.mean((predicted_states - actual_states) ** 2)

        # Combined loss (velocity dominates)
        return 0.8 * velocity_loss + 0.2 * state_loss
    else:
        # Standard state matching
        return jnp.mean((predicted_states - actual_states) ** 2)
```

### Phase 3: Multi-Step Flow Matching

For k-step predictions, chain velocities:

```python
def compute_multistep_flow_loss(
    model,
    params,
    initial_state: jnp.ndarray,
    actions: jnp.ndarray,        # (batch, k, action_dim)
    actual_states: jnp.ndarray,  # (batch, k+1, state_dim) - includes initial
    num_steps: int = 5,
) -> jnp.ndarray:
    """Multi-step flow matching loss."""

    total_loss = 0.0
    current_state = initial_state

    for step in range(num_steps):
        action = actions[:, step]
        target_state = actual_states[:, step + 1]
        prev_state = actual_states[:, step]

        # Predict velocity
        next_state, velocity = model.apply(
            params, current_state, action, method=model.dynamics
        )

        # Actual velocity
        actual_velocity = target_state - prev_state

        # Flow matching loss
        step_loss = jnp.mean((velocity - actual_velocity) ** 2)
        total_loss += step_loss

        # Use predicted state for next step (teacher forcing off)
        current_state = next_state

    return total_loss / num_steps
```

### Phase 4: CLI Integration

**File: `train_jax_fast.py`**

```python
parser.add_argument('--flow-matching', action='store_true', default=False,
                   help='Use flow (velocity) matching for prediction loss')
parser.add_argument('--flow-state-weight', type=float, default=0.2,
                   help='Weight for state loss in flow matching (default: 0.2)')
```

## Visualization

### State Matching (Current)
```
s_t ─────> Model ─────> s_{t+1}_predicted
                              │
                              v
            MSE(s_{t+1}_predicted, s_{t+1}_actual)

Identity solution: output ≈ input → loss ≈ 0 for stable data
```

### Flow Matching (Proposed)
```
s_t ─────> Model ─────> v_predicted (velocity)
                              │
s_t + v_predicted ────────> s_{t+1}_predicted
                              │
                              v
            MSE(v_predicted, v_actual = s_{t+1} - s_t)

Identity (v=0) solution: loss = ||s_{t+1} - s_t||² → HIGH for any change
```

## Expected Behavior

### Before (State Matching)
```
Step 1-50 (warmup): Loss ~0.5 (learning dynamics)
Step 51-100: Loss drops to ~0.01 (identity solution found)
Step 100+: Loss stays ~0.01 (trivial prediction)
```

### After (Flow Matching)
```
Step 1-50 (warmup): Loss ~0.5 (learning velocities)
Step 51-100: Loss ~0.3 (still learning, no trivial solution)
Step 100+: Loss ~0.2 (converges to actual dynamics)
```

## Configuration Options

| Flag | Default | Description |
|------|---------|-------------|
| `--flow-matching` | False | Enable velocity prediction |
| `--flow-state-weight` | 0.2 | State loss regularization weight |
| `--velocity-scale` | 1.0 | Scale factor for velocity targets |

## Interaction with Other Components

### Prediction Warmup
- Flow matching may need different warmup strategy
- Velocity prediction starts immediately (no identity to unlearn)
- Consider shorter warmup or none

### Physics Prediction (OGCM)
- Physics already predicts changes (∂p/∂t)
- Flow matching aligns neural corrector with physics formulation
- Natural integration

### Value Function
- Better predictions → better rollout values
- More informative MCTS planning

## Alternatives Considered

### Option A: Contrastive Loss
```python
# Distinguish real next states from random states
loss = contrastive(predicted, positive=actual, negatives=random_states)
```
Pro: No trivial solution. Con: Requires negative sampling, more complex.

### Option B: Reconstruction + Consistency
```python
# Predict forward and backward, enforce consistency
forward = model(s_t, a)
backward = inverse_model(forward, a)
loss = MSE(backward, s_t) + MSE(forward, actual)
```
Pro: Bidirectional constraint. Con: Requires inverse model.

### Option C: Flow Matching (Chosen)
Pro: Simple, directly prevents identity solution, aligns with physics intuition.
Con: Velocity can be noisy if states are noisy.

## Testing Plan

1. **Unit test**: Verify velocity = predicted_next - current (not state)
2. **Identity test**: Feed constant state, verify loss > 0 (would be 0 with state matching)
3. **Integration**: Run 100 epochs, verify prediction loss doesn't collapse to 0.01

## Success Criteria

1. **No collapse**: Prediction loss stays > 0.1 for 300+ epochs
2. **Meaningful gradients**: ∂loss/∂params non-trivial throughout training
3. **Downstream benefit**: Better MCTS rollout quality (measured by action diversity)

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Noisy velocities | Use EMA smoothing on velocity targets |
| Scale mismatch | Normalize velocities to unit variance |
| Harder optimization | Lower learning rate for prediction head |

## Files to Modify

1. **`mcts_trader/jax_agent/muzero_model.py`** - DynamicsNetwork output velocity
2. **`mcts_trader/jax_agent/trainer.py`** - Flow matching loss computation
3. **`train_jax_fast.py`** - CLI flags and integration
4. **`mcts_trader/jax_agent/prediction/`** - Align with physics formulation

## Dependencies

- None (pure architectural change)

## Follow-up Tasks

- Integrate with physics prediction (TASK_008-012)
- Test with different velocity normalization schemes
- Consider probabilistic flow matching (diffusion-style)

## References

- Flow Matching for Generative Modeling (Lipman et al., 2022)
- Gemini Deep Research: "Train to predict velocity rather than next state"
- Neural ODE literature: Velocity field prediction
