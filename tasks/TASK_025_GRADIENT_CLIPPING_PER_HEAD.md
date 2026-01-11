# TASK 025: Per-Head Gradient Clipping

**Severity**: FEATURE
**Priority**: MEDIUM
**Location**: `mcts_trader/jax_agent/trainer.py`
**Status**: IMPLEMENTED (2026-01-11)
**Effort**: Easy (1-2 hours)
**Dependencies**: None

---

## Problem

Current gradient clipping applies globally:
```python
optax.clip_by_global_norm(max_norm=1.0)
```

**Issues**:
1. One head with large gradients can starve others
2. Value head often has larger gradients than policy head
3. Dynamics network may have different scale than prediction heads
4. Multi-task optimization conflict between heads

---

## Solution: Per-Head Gradient Clipping

Clip gradients separately for each network head:
- Value head: `max_norm_value`
- Policy head: `max_norm_policy`
- Dynamics network: `max_norm_dynamics`
- (Optional) Prediction head: `max_norm_prediction`

---

## Research (2025)

### Gradient Surgery
- [PCGrad](https://arxiv.org/abs/2001.06782): Projects conflicting gradients to avoid interference
- Used in multi-task RL: SAC on Meta-World MT10/MT50 benchmarks

### Multi-Task Gradient Balance
- [MTSGD](https://www.mdpi.com/2079-9292/11/8/1217): Decomposes gradients into task-specific sub-gradients
- Uses gradient balance and clipping for each task

### Dynamic Clipping (2025)
- [CE-GPPO](https://arxiv.org/html/2509.20712v3): Adaptive clipping based on entropy
- Threshold computed from running statistics of gradient norms

---

## Implementation

### Option 1: Simple Per-Head Clipping

```python
# trainer.py

def _clip_gradients_per_head(
    self,
    grads: dict,
    max_norm_value: float = 1.0,
    max_norm_policy: float = 1.0,
    max_norm_dynamics: float = 1.0,
    max_norm_encoder: float = 1.0,
) -> dict:
    """Clip gradients separately for each network head."""

    def clip_subtree(grads_subtree, max_norm):
        """Clip gradients within a parameter subtree."""
        # Compute norm of this subtree
        leaves = jax.tree_util.tree_leaves(grads_subtree)
        total_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in leaves))
        scale = jnp.minimum(1.0, max_norm / (total_norm + 1e-8))
        return jax.tree_map(lambda g: g * scale, grads_subtree)

    clipped = {}
    for key, value in grads.items():
        if 'value_head' in key:
            clipped[key] = clip_subtree(value, max_norm_value)
        elif 'TRMReasoning' in key or 'output_refiner' in key:
            clipped[key] = clip_subtree(value, max_norm_policy)
        elif 'dynamics' in key or 'action_embedding' in key:
            clipped[key] = clip_subtree(value, max_norm_dynamics)
        elif 'encoder' in key or 'cnn' in key:
            clipped[key] = clip_subtree(value, max_norm_encoder)
        else:
            clipped[key] = value  # No clipping for unrecognized

    return clipped
```

### Option 2: Using Optax Multi-Transform

```python
import optax

def create_per_head_optimizer(
    learning_rate: float,
    max_norm_value: float = 1.0,
    max_norm_policy: float = 1.0,
    max_norm_dynamics: float = 1.0,
) -> optax.GradientTransformation:
    """Create optimizer with per-head gradient clipping."""

    def label_fn(params):
        """Label parameters by head."""
        labels = {}
        for key in params.keys():
            if 'value_head' in key:
                labels[key] = 'value'
            elif 'TRMReasoning' in key or 'output_refiner' in key:
                labels[key] = 'policy'
            elif 'dynamics' in key:
                labels[key] = 'dynamics'
            else:
                labels[key] = 'other'
        return labels

    return optax.chain(
        optax.multi_transform(
            transforms={
                'value': optax.clip_by_global_norm(max_norm_value),
                'policy': optax.clip_by_global_norm(max_norm_policy),
                'dynamics': optax.clip_by_global_norm(max_norm_dynamics),
                'other': optax.clip_by_global_norm(1.0),
            },
            param_labels=label_fn,
        ),
        optax.adam(learning_rate),
    )
```

### Option 3: Gradient Norm Logging + Adaptive Clipping

```python
def _adaptive_per_head_clipping(
    self,
    grads: dict,
    head_stats: dict,  # Running stats of gradient norms
    clip_percentile: float = 0.95,  # Clip at 95th percentile
) -> dict:
    """Adaptively clip based on running gradient norm statistics."""

    clipped = {}
    for key, value in grads.items():
        head_type = self._get_head_type(key)
        if head_type in head_stats:
            # Use percentile-based threshold
            max_norm = head_stats[head_type]['p95']
            clipped[key] = clip_subtree(value, max_norm)
        else:
            clipped[key] = value

    return clipped
```

---

## CLI Arguments

```python
parser.add_argument('--clip-grad-value', type=float, default=1.0,
    help='Gradient clip norm for value head')
parser.add_argument('--clip-grad-policy', type=float, default=1.0,
    help='Gradient clip norm for policy head')
parser.add_argument('--clip-grad-dynamics', type=float, default=1.0,
    help='Gradient clip norm for dynamics network')
parser.add_argument('--per-head-clipping', action='store_true', default=False,
    help='Enable per-head gradient clipping')
```

---

## Integration with DS24 Gradient Monitoring

This pairs well with existing `--ds24-grad-norms` flag:

```python
# Log gradient norms BEFORE clipping
if self.track_grad_norms:
    metrics['grad_norm_value_pre_clip'] = compute_norm(grads, 'value')
    metrics['grad_norm_policy_pre_clip'] = compute_norm(grads, 'policy')

# Clip
if self.per_head_clipping:
    grads = self._clip_gradients_per_head(grads, ...)

# Log gradient norms AFTER clipping
if self.track_grad_norms:
    metrics['grad_norm_value_post_clip'] = compute_norm(grads, 'value')
    metrics['grad_norm_policy_post_clip'] = compute_norm(grads, 'policy')
```

---

## Testing

```bash
# Test per-head clipping with aggressive value clipping
python train_jax_fast.py \
    --synthetic-level 0 \
    --epochs 200 \
    --per-head-clipping \
    --clip-grad-value 0.5 \
    --clip-grad-policy 1.0 \
    --ds24-grad-norms \
    --name test_perhead_clip
```

**Monitor**: Check DS24 dashboard for gradient norm balance between heads.

---

## Files to Modify

| File | Changes |
|------|---------|
| `mcts_trader/jax_agent/trainer.py` | Add `_clip_gradients_per_head()` |
| `train_jax_fast.py` | Add CLI args, pass to trainer |

---

## Success Criteria

1. Gradient norms balanced across heads (visible in DS24)
2. No single head dominating training
3. Stable value predictions during policy updates
4. No performance regression vs global clipping
