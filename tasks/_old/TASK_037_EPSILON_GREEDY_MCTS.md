# TASK_037: ε-Greedy MCTS Training Target

## Status: IMPLEMENTED (2026-01-06)

**Implementation:**
- `mcts_trader/jax_agent/exploration.py` - `apply_epsilon_greedy()` function
- `mcts_trader/jax_agent/mctx_search.py` - Integrated into search output
- `train_jax_fast.py` - Added `--epsilon-greedy` CLI flag

## Problem Statement

The coupled policy-prediction collapse is driven by MCTS producing increasingly sharp (low-entropy) policy targets. When the policy network is trained via cross-entropy loss to match these sharp targets, it inherits the collapse. The feedback loop:

```
MCTS produces sharp π_mcts (HOLD-heavy)
    ↓
Policy trained: min CE(π_mcts, π_network)
    ↓
π_network becomes HOLD-heavy
    ↓
Data distribution shifts to HOLD-only
    ↓
Prediction head learns trivial "no change"
    ↓
Value estimates become uninformative
    ↓
MCTS produces even sharper HOLD targets
```

## Research Consensus

From `/home/lukas/Trader/docs/AI_bleed research/`:

| Source | Recommendation |
|--------|----------------|
| **Kimi** | "ε-greedy in MCTS: `π_train = 0.85 * π_mcts + 0.15 * uniform`" |
| **Claude CLI** | "Best intervention: ε-greedy MCTS" - breaks the feedback loop |
| **Compass** | "Target entropy ~2.9 nats (0.5 × log(320))" |
| **DS3** | "Entropy regularization alone is insufficient" |

## Solution: ε-Greedy Training Target

Instead of training the policy to match π_mcts directly, mix in uniform exploration:

```python
π_train = (1 - ε) * π_mcts + ε * π_uniform
```

This ensures:
1. Policy always has minimum entropy = ε × log(num_actions)
2. All actions have non-zero probability in training target
3. Data distribution remains diverse
4. Prediction head sees varied action effects

## Implementation Plan

### Phase 1: Core Implementation

**File: `mcts_trader/jax_agent/mctx_search.py`**

```python
class MuZeroMCTXSearch:
    def __init__(
        self,
        # ... existing params ...
        epsilon_greedy: float = 0.0,  # NEW: exploration mixing
    ):
        self.epsilon_greedy = epsilon_greedy

    def search_fn(self, params, rng, observation, action_mask=None):
        # ... existing MCTS search code ...

        # After getting policy_output from mctx
        actions = policy_output.action
        policy_probs = policy_output.action_weights  # (batch, num_actions)

        # NEW: Mix in uniform exploration for TRAINING targets
        if self.epsilon_greedy > 0:
            policy_probs = self._apply_epsilon_greedy(
                policy_probs, action_mask, rng
            )

        return actions, policy_probs, search_stats

    def _apply_epsilon_greedy(
        self,
        mcts_policy: jnp.ndarray,  # (batch, num_actions)
        action_mask: jnp.ndarray,  # (batch, num_actions) or None
        rng: chex.PRNGKey,
    ) -> jnp.ndarray:
        """Mix MCTS policy with uniform distribution for exploration."""
        batch_size, num_actions = mcts_policy.shape
        epsilon = self.epsilon_greedy

        # Create uniform distribution over valid actions
        if action_mask is not None:
            # Uniform only over valid actions
            valid_count = jnp.sum(action_mask, axis=-1, keepdims=True)
            uniform = action_mask / jnp.maximum(valid_count, 1.0)
        else:
            uniform = jnp.ones_like(mcts_policy) / num_actions

        # Mix: (1-ε) * π_mcts + ε * uniform
        mixed_policy = (1 - epsilon) * mcts_policy + epsilon * uniform

        # Renormalize (should already sum to 1, but ensure numerical stability)
        mixed_policy = mixed_policy / jnp.sum(mixed_policy, axis=-1, keepdims=True)

        return mixed_policy
```

### Phase 2: CLI Integration

**File: `train_jax_fast.py`**

```python
# Add argument
parser.add_argument('--epsilon-greedy', type=float, default=0.0,
                    help='Exploration mixing: (1-ε)*π_mcts + ε*uniform (default: 0.0)')

# In search initialization
search = MuZeroMCTXSearch(
    model_apply_fn=model.apply,
    num_simulations=args.sims,
    # ... other params ...
    epsilon_greedy=args.epsilon_greedy,  # NEW
)
```

### Phase 3: Epsilon Scheduling (Optional Enhancement)

```python
class EpsilonSchedule:
    """Decay epsilon over training."""

    def __init__(
        self,
        initial: float = 0.3,
        final: float = 0.05,
        decay_epochs: int = 300,
    ):
        self.initial = initial
        self.final = final
        self.decay_epochs = decay_epochs

    def get_epsilon(self, epoch: int) -> float:
        if epoch >= self.decay_epochs:
            return self.final
        progress = epoch / self.decay_epochs
        return self.initial + progress * (self.final - self.initial)
```

Usage:
```python
# In training loop
epsilon_schedule = EpsilonSchedule(initial=0.3, final=0.05, decay_epochs=300)

for epoch in range(num_epochs):
    current_epsilon = epsilon_schedule.get_epsilon(epoch)
    search.epsilon_greedy = current_epsilon
    # ... run epoch ...
```

### Phase 4: Action Selection vs Training Target

**Important distinction:**
- **Action selection**: Use pure MCTS policy (for environment interaction)
- **Training target**: Use ε-greedy mixed policy (for CE loss)

```python
# In self-play
actions, policy_probs_mcts, stats = search(params, rng, obs, action_mask)

# Use MCTS action for environment (no mixing)
next_obs, reward, done = env.step(actions)

# Store mixed policy for training
policy_probs_train = search._apply_epsilon_greedy(policy_probs_mcts, action_mask, rng)
trajectory['policy_probs'].append(policy_probs_train)  # For CE loss target
```

This way:
- Agent acts according to best MCTS estimate
- But policy network learns to maintain exploration capability

## Alternative Implementations

### Option A: Dirichlet Noise at Root (Already in MCTS)
```python
# mctx already supports this via dirichlet_alpha and dirichlet_fraction
# But this only affects search, not the output policy target
```

### Option B: Temperature on Training Target
```python
# Soften MCTS policy with temperature
def softmax_with_temp(logits, temperature=1.0):
    return jax.nn.softmax(logits / temperature)

# Apply to visit counts before normalizing
policy_probs = softmax_with_temp(jnp.log(visit_counts + 1e-8), temperature=1.5)
```

### Option C: KL-Constrained Target
```python
# Limit how far π_network can move from uniform per update
# SAC-style: max H(π) subject to E[Q] constraint
```

## Recommended Configuration

Based on research:

```bash
# Conservative start
python train_jax_fast.py \
    --synthetic-level 0 \
    --epsilon-greedy 0.2 \
    --entropy-coef 0.05 \
    --sims-warmup 200 \
    --sims-min 8 \
    --sims 32

# Aggressive exploration (if conservative fails)
python train_jax_fast.py \
    --synthetic-level 0 \
    --epsilon-greedy 0.3 \
    --entropy-coef 0.1 \
    --sims-warmup 300 \
    --sims-min 4 \
    --sims 24
```

## Success Criteria

1. **Entropy maintenance**: Policy entropy stays > 3.5 for 100+ epochs
2. **HOLD prevention**: HOLD% stays < 70% for 100+ epochs
3. **Prediction health**: Step1 MSE stays > 0.05 (non-trivial)
4. **Performance**: Reward improves over baseline (not just stable)

## Metrics to Log

```python
# Add to epoch logging
print(f"  ε-greedy: {search.epsilon_greedy:.2f}")
print(f"  Policy entropy: {policy_entropy:.3f}")
print(f"  Training target entropy: {-np.sum(policy_probs_train * np.log(policy_probs_train + 1e-10)):.3f}")
```

## Files to Modify

1. **`mcts_trader/jax_agent/mctx_search.py`**
   - Add `epsilon_greedy` parameter to `__init__`
   - Add `_apply_epsilon_greedy()` method
   - Apply mixing in `search_fn()` return

2. **`train_jax_fast.py`**
   - Add `--epsilon-greedy` CLI argument
   - Pass to search initialization
   - (Optional) Add scheduling logic

3. **`mcts_trader/jax_agent/self_play.py`** (if separate)
   - Ensure mixed policy stored in trajectory

## Testing Plan

1. **Unit test**: Verify ε-greedy mixing produces correct distribution
2. **Smoke test**: Run 10 epochs, check entropy doesn't collapse
3. **Integration**: Run with sweep script (TASK_036)

## Dependencies

- None (standalone feature)

## Follow-up Tasks

- TASK_036 (Warmup Sweep) will test various epsilon values
- Consider SAC-style automatic epsilon tuning based on target entropy

## References

- Kimi research: "ε-greedy MCTS training target"
- Claude CLI analysis: "MCTS is the collapse catalyst"
- SAC paper: Automatic entropy tuning with target entropy
- AlphaZero: Temperature scheduling in MCTS
