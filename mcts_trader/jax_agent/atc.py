"""Augmented Temporal Contrast (ATC) for representation learning.

TASK_043: Self-supervised contrastive learning to prevent representation collapse.

Based on:
- SimCLR (Chen et al., 2020): Contrastive learning framework
- CURL (Laskin et al., 2020): Contrastive RL
- Gemini Deep Research: "ATC decouples encoder from value/policy signals"

The core idea:
- Positive pairs: (s_t, s_{t+k}) from same trajectory (nearby in time)
- Negative pairs: (s_t, s_random) from different trajectories/times
- Loss: Pull positives together, push negatives apart

This forces the encoder to learn meaningful representations that capture
market structure, even when policy collapses to HOLD.
"""

import jax
import jax.numpy as jnp
from typing import Callable, Dict, Optional, Tuple
import flax.linen as nn


def info_nce_loss(
    anchor: jnp.ndarray,
    positive: jnp.ndarray,
    negatives: jnp.ndarray,
    temperature: float = 0.1,
) -> jnp.ndarray:
    """Compute InfoNCE contrastive loss.

    Encourages: sim(anchor, positive) > sim(anchor, negative)

    Args:
        anchor: (batch, dim) embeddings at time t
        positive: (batch, dim) embeddings at time t+k (same trajectory)
        negatives: (num_neg, dim) or (batch, num_neg, dim) negative embeddings
        temperature: Softmax temperature (lower = harder negatives)

    Returns:
        Scalar loss value
    """
    batch_size = anchor.shape[0]

    # L2 normalize embeddings
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
        neg_sim = jnp.einsum('bd,bnd->bn', anchor, jnp.transpose(negatives, (0, 2, 1))) / temperature

    # Positive similarity: (batch,)
    pos_sim = jnp.sum(anchor * positive, axis=-1) / temperature

    # InfoNCE: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
    # = -pos + logsumexp([pos, neg_1, neg_2, ...])
    all_logits = jnp.concatenate([pos_sim[:, None], neg_sim], axis=-1)  # (batch, 1+num_neg)
    loss = -pos_sim + jax.nn.logsumexp(all_logits, axis=-1)

    return jnp.mean(loss)


def compute_atc_loss(
    encoder_fn: Callable,
    params: Dict,
    observations: jnp.ndarray,  # (batch, time, *obs_shape) or (batch, *obs_shape)
    k_step: int = 5,
    num_negatives: int = 32,
    temperature: float = 0.1,
    rng: Optional[jax.random.PRNGKey] = None,
) -> jnp.ndarray:
    """Compute ATC loss over a batch of observations.

    For single-step observations (not trajectories), uses other batch elements
    as negatives.

    Args:
        encoder_fn: Function (params, obs) -> embedding
        params: Encoder parameters
        observations: Batch of observations (with or without time dimension)
        k_step: Temporal distance for positives (if time dim exists)
        num_negatives: Number of negative samples per anchor
        temperature: Contrastive loss temperature
        rng: Random key for sampling

    Returns:
        ATC loss value
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    # Check if observations have time dimension
    if observations.ndim == 5:  # (batch, time, C, H, W)
        return _atc_loss_temporal(
            encoder_fn, params, observations,
            k_step, num_negatives, temperature, rng
        )
    else:  # (batch, C, H, W) - single observations
        return _atc_loss_batch(
            encoder_fn, params, observations,
            num_negatives, temperature, rng
        )


def _atc_loss_temporal(
    encoder_fn: Callable,
    params: Dict,
    observations: jnp.ndarray,  # (batch, time, *obs_shape)
    k_step: int,
    num_negatives: int,
    temperature: float,
    rng: jax.random.PRNGKey,
) -> jnp.ndarray:
    """ATC loss for temporal sequences."""
    batch_size, time_len = observations.shape[:2]

    # Sample anchor times (not too close to end)
    max_t = time_len - k_step - 1
    if max_t <= 0:
        return jnp.array(0.0)

    rng, sample_rng = jax.random.split(rng)
    anchor_times = jax.random.randint(sample_rng, (batch_size,), 0, max_t)

    # Get anchor and positive observations
    anchor_obs = observations[jnp.arange(batch_size), anchor_times]
    positive_obs = observations[jnp.arange(batch_size), anchor_times + k_step]

    # Encode anchor and positive
    anchor_embed = encoder_fn(params, anchor_obs)
    positive_embed = encoder_fn(params, positive_obs)

    # Sample negatives: random observations from the batch
    rng, neg_rng = jax.random.split(rng)
    neg_batch_idx = jax.random.randint(neg_rng, (num_negatives,), 0, batch_size)
    rng, neg_time_rng = jax.random.split(rng)
    neg_time_idx = jax.random.randint(neg_time_rng, (num_negatives,), 0, time_len)

    negative_obs = observations[neg_batch_idx, neg_time_idx]
    negative_embed = encoder_fn(params, negative_obs)  # (num_neg, dim)

    # Compute InfoNCE loss
    return info_nce_loss(anchor_embed, positive_embed, negative_embed, temperature)


def _atc_loss_batch(
    encoder_fn: Callable,
    params: Dict,
    observations: jnp.ndarray,  # (batch, *obs_shape)
    num_negatives: int,
    temperature: float,
    rng: jax.random.PRNGKey,
) -> jnp.ndarray:
    """ATC loss for single observations (no time dimension).

    Uses data augmentation to create positive pairs.
    """
    batch_size = observations.shape[0]

    if batch_size < 2:
        return jnp.array(0.0)

    # Create positive pairs via augmentation (small noise)
    rng, noise_rng = jax.random.split(rng)
    noise = jax.random.normal(noise_rng, observations.shape) * 0.01
    augmented_obs = observations + noise

    # Encode both views
    anchor_embed = encoder_fn(params, observations)
    positive_embed = encoder_fn(params, augmented_obs)

    # Sample negatives from other batch elements
    rng, neg_rng = jax.random.split(rng)
    neg_idx = jax.random.randint(neg_rng, (num_negatives,), 0, batch_size)
    negative_obs = observations[neg_idx]
    negative_embed = encoder_fn(params, negative_obs)

    # Compute InfoNCE loss
    return info_nce_loss(anchor_embed, positive_embed, negative_embed, temperature)


class ATCProjector(nn.Module):
    """Small projection head for contrastive learning.

    Common practice: project to lower dim for contrastive loss,
    but keep original embedding for downstream tasks.
    """
    hidden_dim: int = 128
    output_dim: int = 64

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)
        return x


def create_atc_encoder_fn(model_apply_fn, projector_params=None):
    """Create encoder function for ATC loss.

    This wraps the model's representation function to get embeddings.

    Args:
        model_apply_fn: Model apply function
        projector_params: Optional projection head params

    Returns:
        Function (params, obs) -> embedding
    """
    def encoder_fn(params, obs):
        # Get representation from model
        # This assumes model has a 'representation' mode or we can extract latent
        output = model_apply_fn({'params': params}, obs, mode='initial_inference')

        # Get the latent state (after encoder, before dynamics)
        if 'hidden_state' in output:
            embed = output['hidden_state']
        else:
            # Fallback: use value logits as proxy for representation
            embed = output.get('value', jnp.zeros((obs.shape[0], 64)))

        # Flatten if needed
        if embed.ndim > 2:
            embed = embed.reshape(embed.shape[0], -1)

        # Apply projection head if available
        if projector_params is not None:
            projector = ATCProjector()
            embed = projector.apply({'params': projector_params}, embed)

        return embed

    return encoder_fn
