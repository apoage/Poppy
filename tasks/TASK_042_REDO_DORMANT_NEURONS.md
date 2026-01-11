# TASK_042: ReDo (Recycling Dormant Neurons)

## Status: PARTIALLY IMPLEMENTED (2026-01-11)

Note: Core framework implemented. Full activation capture requires model changes.
See `mcts_trader/jax_agent/redo.py` for implementation.
## Priority: MEDIUM (Addresses network capacity loss)

## Problem Statement

During policy collapse, large portions of the network become "dormant":
- Neurons that always output 0 (dead ReLUs)
- Neurons with tiny activation variance (near-constant)
- Weights that gradient descent has "given up on"

This happens because:
1. HOLD-dominated data provides weak gradients
2. ReLU units can "die" if they go negative and never recover
3. Collapsed policy means many action-specific pathways go unused

**Result**: Effective model capacity shrinks during collapse, making recovery harder.

## Research Basis

From Gemini Deep Research:
> "Consider 'ReDo' (Recycling Dormant Neurons): periodically check network activation
> patterns, identify 'dormant' units (low activation variance), and re-initialize
> their incoming weights to re-inject plasticity."

From ReDo paper (Sokar et al., 2023):
> "Dormant neurons, which become inactive during training, can comprise a significant
> fraction of a network's parameters. ReDo periodically recycles these neurons,
> improving sample efficiency and final performance."

## Solution: Periodic Neuron Recycling

### Algorithm Overview

```python
every N epochs:
    1. Compute activation statistics for each neuron
    2. Identify dormant neurons (low activation variance)
    3. Re-initialize incoming weights for dormant neurons
    4. Optionally reset associated optimizer state
```

### Dormancy Detection

A neuron is dormant if its activation variance is below threshold:

```python
def compute_dormancy_score(activations, axis=0):
    """Compute normalized activation variance."""
    mean = jnp.mean(activations, axis=axis)
    var = jnp.mean((activations - mean) ** 2, axis=axis)
    # Normalize by mean magnitude to get coefficient of variation
    score = var / (jnp.abs(mean) + 1e-8)
    return score

def is_dormant(score, threshold=0.01):
    """Neuron is dormant if score < threshold."""
    return score < threshold
```

## Implementation

### Phase 1: Activation Monitoring

**File: `mcts_trader/jax_agent/redo.py`** (new file)

```python
from typing import Dict, List, Tuple
import jax
import jax.numpy as jnp
from flax import linen as nn


class DormancyMonitor:
    """Track neuron activation statistics for ReDo."""

    def __init__(
        self,
        dormancy_threshold: float = 0.01,
        check_interval: int = 10,  # epochs
        min_samples: int = 1000,
    ):
        self.dormancy_threshold = dormancy_threshold
        self.check_interval = check_interval
        self.min_samples = min_samples

        # Running statistics
        self.activation_sum = {}     # {layer_name: sum}
        self.activation_sq_sum = {}  # {layer_name: sum of squares}
        self.sample_count = 0

    def update(self, activations: Dict[str, jnp.ndarray]):
        """Update running statistics with batch of activations."""
        batch_size = next(iter(activations.values())).shape[0]

        for name, act in activations.items():
            # Flatten to (batch, neurons)
            act_flat = act.reshape(batch_size, -1)

            if name not in self.activation_sum:
                self.activation_sum[name] = jnp.zeros(act_flat.shape[1])
                self.activation_sq_sum[name] = jnp.zeros(act_flat.shape[1])

            # Update sums
            self.activation_sum[name] += jnp.sum(act_flat, axis=0)
            self.activation_sq_sum[name] += jnp.sum(act_flat ** 2, axis=0)

        self.sample_count += batch_size

    def compute_dormancy_scores(self) -> Dict[str, jnp.ndarray]:
        """Compute dormancy score for each neuron."""
        scores = {}

        for name in self.activation_sum:
            mean = self.activation_sum[name] / self.sample_count
            var = (self.activation_sq_sum[name] / self.sample_count) - mean ** 2
            var = jnp.maximum(var, 0)  # Numerical stability

            # Coefficient of variation (normalized variance)
            scores[name] = jnp.sqrt(var) / (jnp.abs(mean) + 1e-8)

        return scores

    def get_dormant_mask(self) -> Dict[str, jnp.ndarray]:
        """Get boolean mask of dormant neurons per layer."""
        scores = self.compute_dormancy_scores()
        masks = {}
        for name, score in scores.items():
            masks[name] = score < self.dormancy_threshold
        return masks

    def reset(self):
        """Reset statistics after recycling."""
        self.activation_sum = {}
        self.activation_sq_sum = {}
        self.sample_count = 0
```

### Phase 2: Weight Recycling

```python
def recycle_dormant_neurons(
    params: Dict,
    dormant_masks: Dict[str, jnp.ndarray],
    rng: jax.random.PRNGKey,
    init_scale: float = 0.01,
) -> Dict:
    """Re-initialize incoming weights for dormant neurons."""

    def recycle_layer(layer_params, mask, rng):
        """Recycle single layer's dormant neurons."""
        kernel = layer_params['kernel']  # (in_features, out_features)
        bias = layer_params.get('bias', None)

        # Generate new weights for dormant neurons
        new_kernel = jax.random.normal(rng, kernel.shape) * init_scale

        # Replace only dormant columns
        # mask shape: (out_features,) -> expand to (1, out_features)
        mask_expanded = mask[None, :]
        recycled_kernel = jnp.where(mask_expanded, new_kernel, kernel)

        result = {'kernel': recycled_kernel}

        if bias is not None:
            # Reset bias for dormant neurons to 0
            recycled_bias = jnp.where(mask, 0.0, bias)
            result['bias'] = recycled_bias

        return result

    # Map layer names in dormant_masks to param paths
    recycled_params = params.copy()
    rngs = jax.random.split(rng, len(dormant_masks))

    for (layer_name, mask), layer_rng in zip(dormant_masks.items(), rngs):
        # Navigate to correct param in tree
        # This depends on your model structure
        if layer_name in recycled_params:
            recycled_params[layer_name] = recycle_layer(
                recycled_params[layer_name], mask, layer_rng
            )

    return recycled_params
```

### Phase 3: Optimizer State Reset

```python
def reset_optimizer_for_recycled(
    opt_state: optax.OptState,
    dormant_masks: Dict[str, jnp.ndarray],
) -> optax.OptState:
    """Reset optimizer momentum/variance for recycled neurons."""

    def reset_slot(slot, mask):
        """Reset a single optimizer slot (e.g., momentum)."""
        if slot is None:
            return None

        # Expand mask to match slot shape
        if slot.ndim == 2:  # kernel
            mask_expanded = mask[None, :]
        else:  # bias
            mask_expanded = mask

        return jnp.where(mask_expanded, 0.0, slot)

    # For Adam: reset both momentum (mu) and variance (nu)
    # Structure depends on optax optimizer used
    # This is a simplified example

    new_state = opt_state
    # Navigate and reset relevant slots
    # Implementation depends on optax internals

    return new_state
```

### Phase 4: Integration with Training

**File: `train_jax_fast.py`**

```python
# CLI arguments
parser.add_argument('--redo', action='store_true', default=False,
                   help='Enable ReDo (Recycling Dormant Neurons)')
parser.add_argument('--redo-interval', type=int, default=10,
                   help='Epochs between dormancy checks')
parser.add_argument('--redo-threshold', type=float, default=0.01,
                   help='Dormancy threshold (CV below this is dormant)')

# In training loop
if args.redo:
    dormancy_monitor = DormancyMonitor(
        dormancy_threshold=args.redo_threshold,
        check_interval=args.redo_interval,
    )

# During forward pass (need to capture activations)
# This requires model modification to return intermediate activations

# Every redo_interval epochs:
if args.redo and epoch % args.redo_interval == 0:
    if dormancy_monitor.sample_count >= dormancy_monitor.min_samples:
        # Compute dormancy
        dormant_masks = dormancy_monitor.get_dormant_mask()
        total_dormant = sum(jnp.sum(m) for m in dormant_masks.values())
        total_neurons = sum(m.size for m in dormant_masks.values())

        print(f"  Dormant neurons: {total_dormant}/{total_neurons} "
              f"({100*total_dormant/total_neurons:.1f}%)")

        if total_dormant > 0:
            # Recycle
            rng, recycle_rng = jax.random.split(rng)
            params = recycle_dormant_neurons(params, dormant_masks, recycle_rng)
            opt_state = reset_optimizer_for_recycled(opt_state, dormant_masks)
            print(f"  Recycled {total_dormant} dormant neurons")

        # Reset monitoring
        dormancy_monitor.reset()
```

### Phase 5: Activation Capture Hook

To monitor activations, we need to modify the model:

```python
class TRMWithActivations(nn.Module):
    """TRM model that captures intermediate activations."""
    base_model: nn.Module
    capture_layers: List[str]

    @nn.compact
    def __call__(self, x, capture=False):
        activations = {}

        def capture_hook(name):
            def hook(module, args, output):
                if capture:
                    activations[name] = output
                return output
            return hook

        # Run model with hooks
        # (Implementation depends on Flax patterns)

        output = self.base_model(x)
        return output, activations if capture else output
```

## Configuration Options

| Flag | Default | Description |
|------|---------|-------------|
| `--redo` | False | Enable ReDo |
| `--redo-interval` | 10 | Epochs between checks |
| `--redo-threshold` | 0.01 | Dormancy threshold (CV) |
| `--redo-init-scale` | 0.01 | Init scale for recycled weights |
| `--redo-reset-optimizer` | True | Reset optimizer for recycled |

## Expected Behavior

### Without ReDo
```
Epoch 50: 5% dormant neurons
Epoch 100: 15% dormant (collapse starting)
Epoch 150: 35% dormant (capacity shrinking)
Epoch 200: 50% dormant (half the network dead)
```

### With ReDo
```
Epoch 50: 5% dormant neurons
Epoch 60: (ReDo) Recycled 500 neurons, 0% dormant
Epoch 100: 3% dormant (healthy)
Epoch 110: (ReDo) Recycled 300 neurons, 0% dormant
Epoch 200: 2% dormant (network stays healthy)
```

## Interaction with Other Components

### Entropy Collapse
- ReDo maintains network plasticity
- More active neurons = more expressive policy
- Complements entropy regularization

### Value Function
- Dormant neurons in value head → poor predictions
- ReDo keeps value network healthy

### Prediction Head
- Most relevant for prediction collapse
- Dormant prediction neurons can't capture market dynamics
- ReDo forces network to use full capacity

## Testing Plan

1. **Unit test**: Verify dormancy detection (inject known dormant neurons)
2. **Recycling test**: Verify weight re-initialization
3. **Integration**: Run with/without ReDo, compare dormancy over time

## Success Criteria

1. **Dormancy control**: Dormant neurons stay < 10% throughout training
2. **No performance loss**: Recycling doesn't harm final performance
3. **Recovery ability**: Network can recover from partial collapse via ReDo

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Recycling hurts learned features | Only recycle truly dormant (threshold=0.01) |
| Too frequent recycling | Use longer interval (20+ epochs) |
| Optimizer state mismatch | Reset optimizer for recycled neurons |
| Doesn't address root cause | Combine with other interventions |

## Files to Modify

1. **`mcts_trader/jax_agent/redo.py`** (new) - DormancyMonitor, recycling logic
2. **`mcts_trader/jax_agent/trm_flax.py`** - Add activation capture
3. **`train_jax_fast.py`** - CLI flags, integration
4. **Checkpoint handling** - Consider saving dormancy stats

## Dependencies

- Requires activation capture hooks (model modification)
- May need careful handling with JIT compilation

## Follow-up Tasks

- Analyze which layers have most dormancy
- Per-layer threshold tuning
- Combine with learning rate reset for recycled params

## References

- ReDo paper (Sokar et al., 2023): "Recycling Dormant Neurons"
- "Dying ReLU problem" literature
- Gemini Deep Research: "Re-initialize incoming weights to re-inject plasticity"
