# TASK 023: Prioritized Experience Replay

**Severity**: FEATURE
**Priority**: MEDIUM (upgraded 2026-01-23)
**Location**: `mcts_trader/jax_agent/prioritized_buffer.py` (NEW)
**Status**: OPEN - HIGH ARCHITECTURE IMPACT
**Effort**: High (4-6 hours)
**Dependencies**: None

---

## Architecture Notes (2026-01-23)

**Potential Benefits**:
- Could help balance model weights by focusing on informative transitions
- May prevent "overheating" (overfitting to recent easy patterns)
- Better imprint of rare but important market events (crashes, breakouts)

**Concerns**:
- High impact on current JAXReplayBuffer architecture
- Need to maintain TD errors alongside transitions
- Importance sampling weights add complexity to training loop
- May interact with PopArt normalization (need careful integration)

**Recommended Approach**:
- Start with simple proportional prioritization (α=0.6)
- Use sum-tree data structure for O(log n) sampling
- Anneal β from 0.4 → 1.0 over training
- Consider integrating with ReDo (dormant neurons may correlate with low-priority regions)

---

## Problem

Current uniform sampling from replay buffer treats all transitions equally:
- High-value learning opportunities sampled at same rate as trivial ones
- TD errors vary widely - some transitions teach more than others
- Rare events (big wins/losses) may not be sampled enough

**Impact**: Slower learning, especially for rare but important transitions.

---

## Solution: Prioritized Experience Replay (PER)

Sample transitions proportional to their TD error (learning potential):
```
P(i) ∝ |δ_i|^α + ε

where:
- δ_i = TD error for transition i
- α = prioritization exponent (0=uniform, 1=full prioritization)
- ε = small constant to ensure non-zero probability
```

Use importance sampling weights to correct for bias:
```
w_i = (N * P(i))^{-β}
```

---

## Research (2025)

### Recent Advances
- [ReaPER (2025)](https://arxiv.org/abs/2506.18482): Reliability-adjusted PER, considers TD error reliability
- [Experience Replay with Random Reshuffling (2025)](https://arxiv.org/html/2503.02269v1): Extends RR to prioritized settings
- [Q-value-based Experience Replay (2025)](https://www.sciencedirect.com/science/article/pii/S0950705125000000): Uses Q-values instead of TD errors

### Original PER
- [Schaul et al. 2015](https://arxiv.org/abs/1511.05952): Original prioritized experience replay paper
- [Distributed PER](https://arxiv.org/abs/1803.00933): Scalable implementation for distributed RL

### Key Findings
- PER provides ~2x sample efficiency improvement
- α=0.6, β annealing 0.4→1.0 are common hyperparameters
- Requires careful implementation to avoid O(n) updates

---

## Implementation

### Data Structure: Sum Tree

Efficient O(log n) sampling and updates using a segment tree:

```python
# mcts_trader/jax_agent/prioritized_buffer.py

import numpy as np
from typing import Tuple

class SumTree:
    """Binary sum tree for O(log n) prioritized sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Internal nodes + leaves
        self.data_pointer = 0

    def _propagate(self, idx: int, delta: float):
        """Update parent nodes after priority change."""
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def update(self, idx: int, priority: float):
        """Update priority of leaf node."""
        tree_idx = idx + self.capacity - 1
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, delta)

    def get(self, s: float) -> Tuple[int, float]:
        """Sample leaf node given uniform random s in [0, total]."""
        idx = 0
        while idx < self.capacity - 1:  # While not a leaf
            left = 2 * idx + 1
            right = left + 1
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = idx - (self.capacity - 1)
        return data_idx, self.tree[idx]

    @property
    def total(self) -> float:
        return self.tree[0]

    @property
    def max_priority(self) -> float:
        return np.max(self.tree[self.capacity - 1:])


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer with sum tree."""

    def __init__(
        self,
        capacity: int,
        observation_shape: Tuple[int, ...],
        action_size: int,
        alpha: float = 0.6,      # Prioritization exponent
        beta_start: float = 0.4, # Initial importance sampling
        beta_end: float = 1.0,   # Final importance sampling
        beta_anneal_steps: int = 100000,
        epsilon: float = 1e-6,   # Small constant for priorities
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_anneal_steps = beta_anneal_steps
        self.epsilon = epsilon
        self.step_count = 0

        # Data storage (same as regular buffer)
        self.observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        # Priority tree
        self.tree = SumTree(capacity)
        self.position = 0
        self.size = 0

        # Max priority for new samples (optimistic initialization)
        self.max_priority = 1.0

    @property
    def beta(self) -> float:
        """Current importance sampling exponent."""
        progress = min(1.0, self.step_count / self.beta_anneal_steps)
        return self.beta_start + progress * (self.beta_end - self.beta_start)

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        value: float,
        done: float,
        td_error: float = None
    ):
        """Add transition with optional TD error for priority."""
        # Store data
        self.observations[self.position] = observation
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.values[self.position] = value
        self.dones[self.position] = done

        # Set priority (max for new samples, or based on TD error)
        if td_error is not None:
            priority = (np.abs(td_error) + self.epsilon) ** self.alpha
        else:
            priority = self.max_priority ** self.alpha

        self.tree.update(self.position, priority)

        # Update position
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[dict, np.ndarray, np.ndarray]:
        """Sample batch with priorities.

        Returns:
            batch: Dict of arrays
            indices: Sampled indices (for priority updates)
            weights: Importance sampling weights
        """
        indices = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size, dtype=np.float32)

        # Stratified sampling: divide [0, total] into segments
        segment = self.tree.total / batch_size

        for i in range(batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1))
            idx, priority = self.tree.get(s)
            indices[i] = idx
            priorities[i] = priority

        # Compute importance sampling weights
        probs = priorities / self.tree.total
        weights = (self.size * probs) ** (-self.beta)
        weights /= weights.max()  # Normalize

        # Get batch data
        batch = {
            'observations': self.observations[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'values': self.values[indices],
            'dones': self.dones[indices],
        }

        self.step_count += 1
        return batch, indices, weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on new TD errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = (np.abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
```

### Integration in Trainer

```python
# trainer.py

def train_step_prioritized(self, state, batch, indices, weights, rng):
    """Training step with importance sampling weights."""

    def loss_fn(params):
        # ... compute per-sample TD errors ...
        value_errors = predicted_values - target_values

        # Weight losses by importance sampling
        weighted_value_loss = jnp.mean(weights * value_errors ** 2)

        return total_loss, (value_errors, metrics)

    (loss, (td_errors, metrics)), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(state.params)

    # Return TD errors for priority update
    return new_state, metrics, td_errors
```

---

## CLI Arguments

```python
parser.add_argument('--prioritized-replay', action='store_true', default=False,
    help='Use prioritized experience replay')
parser.add_argument('--per-alpha', type=float, default=0.6,
    help='PER prioritization exponent (default: 0.6)')
parser.add_argument('--per-beta-start', type=float, default=0.4,
    help='PER initial importance sampling (default: 0.4)')
```

---

## Complexity Note

This is marked as **High Effort** because:
1. Requires refactoring replay buffer architecture
2. TD errors must flow back from trainer to buffer
3. Must handle trajectory storage (not just transitions)
4. JAX integration is tricky (sum tree updates are sequential)

**Consider**: Starting with simpler approaches (Huber loss, GAE) before implementing PER.

---

## Files to Create/Modify

| File | Changes |
|------|---------|
| `mcts_trader/jax_agent/prioritized_buffer.py` | NEW: PER implementation |
| `mcts_trader/jax_agent/trainer.py` | Return TD errors, use weights |
| `train_jax_fast.py` | Add CLI args, integrate buffer |

---

## Success Criteria

1. Sample efficiency improvement (~1.5-2x)
2. Faster learning on rare but important events
3. Stable training (no priority explosion)
4. Reasonable overhead (< 20% slower sampling)
