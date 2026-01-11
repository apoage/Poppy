# TASK_043: Augmented Temporal Contrast (ATC)

## Status: PARTIALLY IMPLEMENTED (2026-01-11)

Note: Core loss function implemented. Full integration requires model changes
to expose intermediate representations. See `mcts_trader/jax_agent/atc.py`.
## Priority: MEDIUM (Addresses representation collapse)

## Problem Statement

When policy collapses to HOLD, the representation network suffers:
1. **All states look similar**: HOLD-dominated trajectories have low diversity
2. **Representation collapses**: Encoder outputs similar embeddings for different market conditions
3. **Value function can't differentiate**: Same embedding → same value → no gradient signal

**Symptom**: Representation network converges to near-constant output regardless of input.

## Research Basis

From Gemini Deep Research:
> "Use representation learning that doesn't depend on reward (e.g., ATC - Augmented
> Temporal Contrast). This decouples the encoder from value and policy signals,
> forcing it to capture market structure via self-supervision."

From MuZero/EfficientZero papers:
> "Auxiliary prediction tasks can improve representation learning by providing
> additional training signal independent of sparse rewards."

From Contrastive Learning literature:
> "Contrastive losses force representations to distinguish positive pairs (related
> samples) from negative pairs (unrelated samples), learning meaningful structure."

## Solution: Temporal Contrastive Learning

Add an auxiliary loss that forces the encoder to produce **temporally consistent but distinct** representations:

### Core Idea
```
- Positive pairs: (s_t, s_{t+k}) from same trajectory (nearby in time)
- Negative pairs: (s_t, s_random) from different trajectories/times
- Loss: Pull positives together, push negatives apart
```

This forces the encoder to:
1. Learn features that are stable over short time horizons (market regime)
2. Distinguish different market conditions (not all same embedding)
3. NOT rely on reward signal (self-supervised)

## Implementation

### Phase 1: Contrastive Loss

**File: `mcts_trader/jax_agent/atc.py`** (new file)

```python
import jax
import jax.numpy as jnp
from typing import Tuple


def info_nce_loss(
    anchor: jnp.ndarray,      # (batch, dim) - s_t embeddings
    positive: jnp.ndarray,    # (batch, dim) - s_{t+k} embeddings
    negatives: jnp.ndarray,   # (batch, num_neg, dim) or (num_neg, dim)
    temperature: float = 0.1,
) -> jnp.ndarray:
    """Compute InfoNCE contrastive loss.

    Encourages: sim(anchor, positive) > sim(anchor, negative)

    Args:
        anchor: Embeddings at time t
        positive: Embeddings at time t+k (same trajectory)
        negatives: Embeddings from other times/trajectories
        temperature: Softmax temperature (lower = harder)

    Returns:
        Scalar loss value
    """
    batch_size, dim = anchor.shape

    # Normalize embeddings
    anchor = anchor / (jnp.linalg.norm(anchor, axis=-1, keepdims=True) + 1e-8)
    positive = positive / (jnp.linalg.norm(positive, axis=-1, keepdims=True) + 1e-8)

    # Handle negatives shape
    if negatives.ndim == 2:
        # Shared negatives: (num_neg, dim)
        negatives = negatives / (jnp.linalg.norm(negatives, axis=-1, keepdims=True) + 1e-8)
        # Similarity with negatives: (batch, num_neg)
        neg_sim = jnp.dot(anchor, negatives.T) / temperature
    else:
        # Per-sample negatives: (batch, num_neg, dim)
        negatives = negatives / (jnp.linalg.norm(negatives, axis=-1, keepdims=True) + 1e-8)
        # (batch, dim) @ (batch, dim, num_neg) -> (batch, num_neg)
        neg_sim = jnp.einsum('bd,bnd->bn', anchor, negatives.transpose(0, 2, 1)) / temperature

    # Positive similarity: (batch,)
    pos_sim = jnp.sum(anchor * positive, axis=-1) / temperature

    # InfoNCE: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
    # = -pos + log(exp(pos) + sum(exp(neg)))
    # = -pos + logsumexp([pos, neg_1, neg_2, ...])

    all_logits = jnp.concatenate([pos_sim[:, None], neg_sim], axis=-1)  # (batch, 1+num_neg)
    loss = -pos_sim + jax.nn.logsumexp(all_logits, axis=-1)

    return loss.mean()


def augmented_temporal_contrast_loss(
    encoder_fn,
    params,
    observations: jnp.ndarray,  # (batch, time, *obs_shape)
    k_step: int = 5,            # Temporal distance for positives
    num_negatives: int = 32,    # Number of negative samples
    temperature: float = 0.1,
    rng: jax.random.PRNGKey = None,
) -> jnp.ndarray:
    """Compute ATC loss over a batch of trajectories.

    Args:
        encoder_fn: Function (params, obs) -> embedding
        params: Encoder parameters
        observations: Batch of trajectories
        k_step: How far in future to sample positives
        num_negatives: Number of negative samples per anchor
        temperature: Contrastive loss temperature
        rng: Random key for negative sampling

    Returns:
        ATC loss value
    """
    batch_size, time_len = observations.shape[:2]

    # Sample anchor times (not too close to end)
    max_t = time_len - k_step - 1
    if max_t <= 0:
        return 0.0

    if rng is not None:
        rng, sample_rng = jax.random.split(rng)
        anchor_times = jax.random.randint(sample_rng, (batch_size,), 0, max_t)
    else:
        anchor_times = jnp.zeros(batch_size, dtype=jnp.int32)

    # Get anchor and positive observations
    anchor_obs = observations[jnp.arange(batch_size), anchor_times]
    positive_obs = observations[jnp.arange(batch_size), anchor_times + k_step]

    # Encode anchor and positive
    anchor_embed = encoder_fn(params, anchor_obs)
    positive_embed = encoder_fn(params, positive_obs)

    # Sample negatives: random observations from the batch
    if rng is not None:
        rng, neg_rng = jax.random.split(rng)
        neg_batch_idx = jax.random.randint(neg_rng, (num_negatives,), 0, batch_size)
        neg_time_idx = jax.random.randint(neg_rng, (num_negatives,), 0, time_len)
    else:
        neg_batch_idx = jnp.arange(num_negatives) % batch_size
        neg_time_idx = jnp.arange(num_negatives) % time_len

    negative_obs = observations[neg_batch_idx, neg_time_idx]
    negative_embed = encoder_fn(params, negative_obs)  # (num_neg, dim)

    # Compute InfoNCE loss
    loss = info_nce_loss(anchor_embed, positive_embed, negative_embed, temperature)

    return loss
```

### Phase 2: Projection Head (Optional but Recommended)

```python
class ATCProjector(nn.Module):
    """Small projection head for contrastive learning.

    Common practice: project to lower dim for contrastive loss,
    but keep original embedding for downstream tasks.
    """
    hidden_dim: int = 128
    output_dim: int = 64

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)
        return x
```

### Phase 3: Data Augmentation

To make contrastive learning more effective, apply augmentations:

```python
def augment_observation(obs, rng, noise_scale=0.01):
    """Apply random augmentation to observation.

    For trading: small noise injection (market microstructure noise).
    """
    noise = jax.random.normal(rng, obs.shape) * noise_scale
    return obs + noise


def augmented_positive(obs_t, obs_t_k, rng, noise_scale=0.01):
    """Create augmented positive pair."""
    rng1, rng2 = jax.random.split(rng)
    aug_t = augment_observation(obs_t, rng1, noise_scale)
    aug_t_k = augment_observation(obs_t_k, rng2, noise_scale)
    return aug_t, aug_t_k
```

### Phase 4: Integration with Training

**File: `train_jax_fast.py`**

```python
# CLI arguments
parser.add_argument('--atc-loss', action='store_true', default=False,
                   help='Enable Augmented Temporal Contrast loss')
parser.add_argument('--atc-weight', type=float, default=0.1,
                   help='Weight for ATC loss (default: 0.1)')
parser.add_argument('--atc-k-step', type=int, default=5,
                   help='Temporal distance for positive pairs')
parser.add_argument('--atc-temperature', type=float, default=0.1,
                   help='Contrastive loss temperature')
parser.add_argument('--atc-num-negatives', type=int, default=32,
                   help='Number of negative samples')

# In training loop
if args.atc_loss:
    # Need full trajectory observations
    atc_loss = augmented_temporal_contrast_loss(
        encoder_fn=lambda p, o: model.apply(p, o, method=model.encode),
        params=params,
        observations=trajectory_obs,  # (batch, time, *obs_shape)
        k_step=args.atc_k_step,
        num_negatives=args.atc_num_negatives,
        temperature=args.atc_temperature,
        rng=atc_rng,
    )
    total_loss = total_loss + args.atc_weight * atc_loss
    print(f"  ATC loss: {atc_loss:.4f}")
```

## Configuration Options

| Flag | Default | Description |
|------|---------|-------------|
| `--atc-loss` | False | Enable ATC |
| `--atc-weight` | 0.1 | Loss weight |
| `--atc-k-step` | 5 | Positive pair distance |
| `--atc-temperature` | 0.1 | Contrastive temperature |
| `--atc-num-negatives` | 32 | Negative samples |
| `--atc-aug-noise` | 0.01 | Augmentation noise scale |

### Temperature Effects

| Temperature | Behavior |
|-------------|----------|
| 0.05 | Very hard negatives, may be unstable |
| 0.1 | Standard, balanced (recommended) |
| 0.5 | Softer, more gradient flow |
| 1.0 | Very soft, weak contrastive signal |

### k-step Effects

| k-step | Behavior |
|--------|----------|
| 1 | Very similar positives, easy task |
| 5 | Moderate temporal gap (recommended) |
| 20 | Large gap, harder task, may be too hard |

## Expected Behavior

### Without ATC
```
Epoch 50: Encoder output variance = 0.5 (healthy)
Epoch 100: Encoder output variance = 0.2 (collapsing)
Epoch 150: Encoder output variance = 0.05 (collapsed)
→ All embeddings nearly identical
```

### With ATC
```
Epoch 50: Encoder variance = 0.5, ATC loss = 2.1
Epoch 100: Encoder variance = 0.4, ATC loss = 1.5
Epoch 150: Encoder variance = 0.35, ATC loss = 1.2
→ Embeddings remain distinct, capture market structure
```

## Interaction with Other Components

### Policy Collapse
- ATC maintains representation quality even with HOLD-dominated data
- Encoder forced to differentiate market states
- Better embeddings → better policy gradients

### Value Function
- Rich representations → value function can differentiate states
- Breaks "all states same value" collapse mode

### Prediction Head
- ATC complements flow matching (TASK_041)
- Encoder captures what changes, dynamics predicts how

### MCTS
- Better state representations → more informative rollouts
- Q-values differentiate between states

## Alternatives Considered

### Option A: Next-Frame Prediction
```python
loss = MSE(predicted_obs_{t+1}, actual_obs_{t+1})
```
Con: Identity solution problem (same as prediction head).

### Option B: Variational Autoencoder
```python
loss = reconstruction + KL_divergence
```
Con: May collapse to trivial reconstruction.

### Option C: BYOL-style (No Negatives)
```python
# Predictor network + stop-gradient target
loss = MSE(predictor(encoder(aug1(x))), stop_grad(encoder(aug2(x))))
```
Pro: No negative sampling. Con: More complex, requires EMA target.

### Option D: Contrastive (Chosen)
Pro: Simple, proven effective, no extra networks.
Con: Requires negative sampling.

## Testing Plan

1. **Unit test**: Verify InfoNCE loss computation
2. **Collapse test**: Train without ATC, measure embedding variance → should decrease
3. **Prevention test**: Train with ATC, verify embedding variance stays high
4. **Integration**: Full training run, compare representation quality

## Success Criteria

1. **Embedding diversity**: Encoder output variance stays > 0.3 for 300+ epochs
2. **Contrastive quality**: ATC loss decreases (model learns to distinguish)
3. **Downstream benefit**: Better policy learning (measured by entropy stability)

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| ATC dominates other losses | Use low weight (0.1) |
| Hard negatives too hard | Higher temperature, more negatives |
| Slow training | Use projection head (smaller dim) |
| Negatives not negative enough | Sample from different trajectories |

## Files to Modify

1. **`mcts_trader/jax_agent/atc.py`** (new) - ATC loss implementation
2. **`mcts_trader/jax_agent/trainer.py`** - Integrate ATC loss
3. **`train_jax_fast.py`** - CLI flags, trajectory handling
4. **Model** - Add projection head (optional)

## Dependencies

- Requires full trajectory observations (not just single-step)
- May need replay buffer modification for trajectory storage

## Follow-up Tasks

- Combine with flow matching (TASK_041) for full representation + dynamics
- Test BYOL-style alternative if negatives are problematic
- Per-asset contrastive learning (if 64 assets are distinct enough)

## References

- SimCLR (Chen et al., 2020): Contrastive learning framework
- MoCo (He et al., 2020): Momentum contrastive learning
- CURL (Laskin et al., 2020): Contrastive RL
- Gemini Deep Research: "ATC - Augmented Temporal Contrast"
