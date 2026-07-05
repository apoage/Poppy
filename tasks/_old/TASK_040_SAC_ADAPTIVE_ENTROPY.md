# TASK_040: SAC-style Adaptive Entropy Constraint

## Status: IMPLEMENTED (2026-01-06)

**Implementation:**
- `mcts_trader/jax_agent/sac_entropy.py` - Core module with `AdaptiveEntropyManager`
- `train_jax_fast.py` - Added `--sac-entropy`, `--sac-target-entropy-ratio`, `--sac-alpha-lr`, `--sac-init-alpha` flags
- Note: Full training loop integration pending (module ready for use)
## Priority: HIGH (Addresses root cause of entropy collapse)

## Problem Statement

Current adaptive entropy implementation is **reactive**:
```python
if hold_pct > 0.85:
    entropy_coef = boost_high  # React after collapse
else:
    entropy_coef = boost_low   # Too late to prevent
```

This threshold-based approach:
1. Only triggers **after** entropy has already collapsed
2. Creates discontinuous training dynamics (sudden coefficient jumps)
3. Doesn't maintain a consistent target entropy level
4. Can overshoot when boosting (too much entropy → random policy)

## Research Basis

From Gemini Deep Research:
> "Treat entropy as a constraint rather than just a bonus: use a learnable Lagrangian
> multiplier (SAC-style). This dynamically adjusts the entropy coefficient to keep
> entropy near a target (e.g., 50% of log(320))."

From Soft Actor-Critic paper:
> "Instead of manually tuning α, we can automate this process by formulating a
> dual problem and optimizing α alongside the policy."

## Solution: Learnable Entropy Coefficient

Transform from bonus to constraint using Lagrangian optimization:

### Current (Manual)
```python
loss = policy_loss - α * entropy  # α fixed or step-function
```

### Proposed (SAC-style)
```python
# Learnable temperature
log_α = Parameter(init=log(0.01))
α = exp(log_α)

# Entropy constraint loss
target_entropy = 0.5 * log(320)  # ~2.9 nats
entropy_loss = α * (entropy - target_entropy)

# Combined (α automatically adjusts)
total_loss = policy_loss + value_loss + entropy_loss
```

**Key insight**: When entropy < target, α increases → more entropy regularization.
When entropy > target, α decreases → less regularization.

## Implementation

### Phase 1: Learnable Temperature

**File: `mcts_trader/jax_agent/trainer.py`**

```python
class TrainerState(NamedTuple):
    """Training state with learnable entropy temperature."""
    params: Any
    opt_state: Any
    step: int
    # NEW: Entropy temperature
    log_alpha: jnp.ndarray  # Learnable log(α)
    alpha_opt_state: Any    # Separate optimizer for α


def init_trainer_state(
    model,
    learning_rate: float,
    alpha_lr: float = 3e-4,
    init_alpha: float = 0.01,
    target_entropy: float = None,
    num_actions: int = 320,
) -> TrainerState:
    """Initialize trainer state with learnable alpha."""
    params = model.init(...)

    # Default target: 50% of maximum entropy
    if target_entropy is None:
        target_entropy = 0.5 * jnp.log(num_actions)

    # Initialize log_alpha (more stable than alpha directly)
    log_alpha = jnp.array(jnp.log(init_alpha))

    # Separate optimizer for alpha (can use different LR)
    alpha_optimizer = optax.adam(alpha_lr)
    alpha_opt_state = alpha_optimizer.init(log_alpha)

    return TrainerState(
        params=params,
        opt_state=opt_state,
        step=0,
        log_alpha=log_alpha,
        alpha_opt_state=alpha_opt_state,
    )
```

### Phase 2: Alpha Update Step

```python
def update_alpha(
    log_alpha: jnp.ndarray,
    alpha_opt_state: Any,
    policy_entropy: jnp.ndarray,
    target_entropy: float,
    alpha_optimizer: optax.GradientTransformation,
) -> tuple[jnp.ndarray, Any, jnp.ndarray]:
    """Update entropy coefficient toward target."""

    def alpha_loss_fn(log_alpha):
        # α * (H(π) - H_target)
        # If H < target: positive loss → increase α
        # If H > target: negative loss → decrease α
        alpha = jnp.exp(log_alpha)
        return -alpha * (policy_entropy - target_entropy).mean()

    alpha_loss, alpha_grad = jax.value_and_grad(alpha_loss_fn)(log_alpha)

    updates, new_alpha_opt_state = alpha_optimizer.update(
        alpha_grad, alpha_opt_state, log_alpha
    )
    new_log_alpha = optax.apply_updates(log_alpha, updates)

    # Clamp alpha to reasonable range
    new_log_alpha = jnp.clip(new_log_alpha, jnp.log(1e-4), jnp.log(1.0))

    return new_log_alpha, new_alpha_opt_state, alpha_loss
```

### Phase 3: Integration with Training Loop

**File: `train_jax_fast.py`**

```python
# CLI arguments
parser.add_argument('--sac-entropy', action='store_true', default=False,
                   help='Use SAC-style learnable entropy coefficient')
parser.add_argument('--target-entropy-ratio', type=float, default=0.5,
                   help='Target entropy as ratio of max (default: 0.5 = 50%%)')
parser.add_argument('--alpha-lr', type=float, default=3e-4,
                   help='Learning rate for entropy coefficient')
parser.add_argument('--init-alpha', type=float, default=0.01,
                   help='Initial entropy coefficient')

# In training loop
if args.sac_entropy:
    target_entropy = args.target_entropy_ratio * np.log(320)

    # After computing policy loss and entropy
    new_log_alpha, new_alpha_opt_state, alpha_loss = update_alpha(
        trainer_state.log_alpha,
        trainer_state.alpha_opt_state,
        policy_entropy,
        target_entropy,
        alpha_optimizer,
    )
    alpha = jnp.exp(new_log_alpha)

    # Log current alpha
    print(f"  α (entropy coef): {alpha:.4f}")
    print(f"  Target entropy: {target_entropy:.3f}, Actual: {policy_entropy.mean():.3f}")
```

### Phase 4: Entropy Term in Policy Loss

```python
def compute_policy_loss(
    policy_logits: jnp.ndarray,
    target_policy: jnp.ndarray,
    log_alpha: jnp.ndarray,
    target_entropy: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Policy loss with entropy constraint."""

    # Standard cross-entropy with MCTS target
    ce_loss = -jnp.sum(target_policy * jax.nn.log_softmax(policy_logits), axis=-1)

    # Policy entropy
    policy_probs = jax.nn.softmax(policy_logits)
    entropy = -jnp.sum(policy_probs * jax.nn.log_softmax(policy_logits), axis=-1)

    # Entropy regularization (subtract because we minimize loss)
    alpha = jnp.exp(log_alpha)
    total_loss = ce_loss - alpha * entropy

    return total_loss.mean(), entropy.mean()
```

## Configuration Options

| Flag | Default | Description |
|------|---------|-------------|
| `--sac-entropy` | False | Enable SAC-style adaptive α |
| `--target-entropy-ratio` | 0.5 | Target as ratio of max entropy |
| `--alpha-lr` | 3e-4 | Learning rate for α |
| `--init-alpha` | 0.01 | Initial α value |
| `--alpha-clip-min` | 1e-4 | Minimum α |
| `--alpha-clip-max` | 1.0 | Maximum α |

### Target Entropy Recommendations

| Ratio | Target (320 actions) | Behavior |
|-------|---------------------|----------|
| 0.3 | ~1.73 nats | Conservative, allows some HOLD bias |
| 0.5 | ~2.88 nats | Balanced (recommended) |
| 0.7 | ~4.04 nats | Aggressive exploration |
| 1.0 | ~5.77 nats | Near-uniform policy |

## Expected Behavior

### Before (Threshold-based)
```
Epoch 50: entropy=3.8, hold=65%, α=0.01 (no boost)
Epoch 100: entropy=2.1, hold=78%, α=0.01 (still no boost)
Epoch 150: entropy=0.8, hold=91%, α=0.30 (BOOST! Too late)
```

### After (SAC-style)
```
Epoch 50: entropy=3.8, target=2.9, α=0.008 (reduce, above target)
Epoch 100: entropy=2.9, target=2.9, α=0.010 (stable at target)
Epoch 150: entropy=2.8, target=2.9, α=0.011 (slight increase)
```

**Key**: Entropy stays near target throughout training, no collapse.

## Interaction with Existing Adaptive Entropy

Two options for integration:

### Option A: Replace Current System
```python
if args.sac_entropy:
    # Use SAC-style (ignore threshold-based)
    alpha = exp(log_alpha)
else:
    # Use current threshold-based
    alpha = entropy_boost_high if hold_pct > threshold else entropy_boost_low
```

### Option B: Hybrid (Recommended)
```python
# SAC controls baseline α
alpha_sac = exp(log_alpha)

# Emergency boost still available
if hold_pct > 0.90:  # Higher threshold for emergency
    alpha = max(alpha_sac * 2, 0.3)  # Double SAC α or minimum 0.3
else:
    alpha = alpha_sac
```

## Testing Plan

1. **Unit test**: Verify α increases when entropy < target, decreases when > target
2. **Integration test**: Run 100 epochs, check entropy stays within ±0.5 of target
3. **Comparison**: Run with vs without SAC entropy, compare stability

## Success Criteria

1. **Entropy stability**: Policy entropy stays within [2.5, 3.5] for 300+ epochs
2. **No emergency boosts**: Threshold-based emergency never triggers
3. **α convergence**: α stabilizes to consistent value (not oscillating wildly)
4. **Performance**: Final reward ≥ threshold-based approach

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| α grows unbounded | Clip to [1e-4, 1.0] |
| α oscillates | Lower α learning rate (1e-4) |
| Conflicts with ε-greedy | ε-greedy is orthogonal (targets vs entropy) |
| Target too high → random | Start with ratio=0.5, adjust based on results |

## Files to Modify

1. **`mcts_trader/jax_agent/trainer.py`** - Add log_alpha to TrainerState, alpha update
2. **`train_jax_fast.py`** - Add CLI arguments, integrate with training loop
3. **Checkpoint handling** - Save/restore log_alpha and alpha_opt_state

## Dependencies

- None (uses existing optax infrastructure)

## Follow-up Tasks

- TASK_037 (ε-greedy): Orthogonal approaches, can combine
- TASK_039 (Dirichlet): Also orthogonal, all three can work together
- Consider target entropy scheduling (lower early, higher late)

## References

- Soft Actor-Critic paper (Haarnoja et al., 2018): Automatic temperature adjustment
- SAC v2 paper: Target entropy = -dim(A) heuristic
- Gemini Deep Research: "Learnable Lagrangian multiplier"
