"""Cycle Balance: Prevent H0 dominance in TRM recursive reasoning.

TASK_110: Two mechanisms to balance TRM cycle utilization:

1. Cycle Balance Loss: Penalizes imbalanced delta norms across H-cycles
   - Measures coefficient of variation (CV) of delta norms
   - Lower CV = more balanced utilization

2. Cycle ATC: Contrastive loss between early (H0) and late (H3) cycles
   - Forces later cycles to learn complementary features
   - Prevents representation collapse where H1-H3 just copy H0

Usage:
    # In trainer, after forward pass with return_intermediates=True:
    policy, value, latent, intermediates = model(obs, return_intermediates=True)

    # Add balance loss
    balance_loss = compute_cycle_balance_loss(intermediates, h_cycles=4, l_cycles=6)
    total_loss = main_loss + balance_weight * balance_loss

    # Add cycle ATC
    cycle_atc_loss = compute_cycle_atc_loss(intermediates, temperature=0.1)
    total_loss = total_loss + cycle_atc_weight * cycle_atc_loss
"""

import jax
import jax.numpy as jnp
from typing import Dict, Optional


def compute_cycle_balance_loss(
    intermediates: Dict,
    h_cycles: int = 4,
    l_cycles: int = 6,
    target_cv: float = 0.3,
) -> jnp.ndarray:
    """
    Compute loss that penalizes imbalanced cycle utilization.

    Measures coefficient of variation (CV = std/mean) of delta norms across H-cycles.
    Penalizes when CV exceeds target_cv (default 0.3 = 30% variation allowed).

    Args:
        intermediates: Dict with 'latent_deltas' from TRM forward pass
        h_cycles: Number of H-cycles
        l_cycles: Number of L-cycles per H-cycle
        target_cv: Target coefficient of variation (0.3 = allow 30% imbalance)

    Returns:
        Scalar loss (0 if balanced, positive if imbalanced)
    """
    latent_deltas = intermediates.get('latent_deltas', {})
    if not latent_deltas:
        return jnp.array(0.0)

    # Compute total delta norm per H-cycle
    h_norms = []
    for h in range(h_cycles):
        h_total = 0.0
        for l in range(l_cycles):
            key = f'h{h}_l{l}'
            if key in latent_deltas:
                delta = latent_deltas[key]
                # Mean norm across batch
                norm = jnp.mean(jnp.linalg.norm(delta, axis=-1))
                h_total = h_total + norm
        h_norms.append(h_total)

    h_norms = jnp.stack(h_norms)

    # Coefficient of variation
    mean_norm = jnp.mean(h_norms)
    std_norm = jnp.std(h_norms)
    cv = std_norm / (mean_norm + 1e-8)

    # Penalize CV above target (hinge loss style)
    # If CV <= target_cv: loss = 0
    # If CV > target_cv: loss increases linearly
    loss = jnp.maximum(0.0, cv - target_cv)

    return loss


def compute_cycle_balance_loss_soft(
    intermediates: Dict,
    h_cycles: int = 4,
    l_cycles: int = 6,
) -> jnp.ndarray:
    """
    Softer version: directly minimize variance of H-cycle norms.

    This encourages all H-cycles to contribute equally.
    """
    latent_deltas = intermediates.get('latent_deltas', {})
    if not latent_deltas:
        return jnp.array(0.0)

    h_norms = []
    for h in range(h_cycles):
        h_total = 0.0
        for l in range(l_cycles):
            key = f'h{h}_l{l}'
            if key in latent_deltas:
                delta = latent_deltas[key]
                norm = jnp.mean(jnp.linalg.norm(delta, axis=-1))
                h_total = h_total + norm
        h_norms.append(h_total)

    h_norms = jnp.stack(h_norms)

    # Normalize to make loss scale-invariant
    mean_norm = jnp.mean(h_norms)
    normalized = h_norms / (mean_norm + 1e-8)

    # Variance of normalized norms (target: all equal to 1.0)
    variance = jnp.var(normalized)

    return variance


def compute_cycle_atc_loss(
    intermediates: Dict,
    temperature: float = 0.1,
    early_cycle: str = 'h0',
    late_cycle: str = 'h3',
) -> jnp.ndarray:
    """
    Cycle-specific Augmented Temporal Contrast.

    Forces late cycles (H3) to learn different representations from early cycles (H0).
    Uses InfoNCE-style contrastive loss where:
    - Anchor: H3 latent
    - Negative: H0 latent (from SAME sample - we want them different!)

    This is OPPOSITE of standard contrastive learning:
    - Standard: pull same-class together, push different apart
    - Cycle ATC: push early/late cycles apart (force complementary features)

    Args:
        intermediates: Dict with 'latent_per_h' from TRM forward pass
        temperature: Contrastive temperature (lower = stronger push)
        early_cycle: Key for early cycle ('h0')
        late_cycle: Key for late cycle ('h3')

    Returns:
        Scalar loss (higher = H0 and H3 are too similar)
    """
    latent_per_h = intermediates.get('latent_per_h', {})
    if early_cycle not in latent_per_h or late_cycle not in latent_per_h:
        return jnp.array(0.0)

    h0_latent = latent_per_h[early_cycle]  # (batch, dim)
    h3_latent = latent_per_h[late_cycle]   # (batch, dim)

    # L2 normalize
    h0_norm = h0_latent / (jnp.linalg.norm(h0_latent, axis=-1, keepdims=True) + 1e-8)
    h3_norm = h3_latent / (jnp.linalg.norm(h3_latent, axis=-1, keepdims=True) + 1e-8)

    # Cosine similarity between corresponding H0 and H3 (same sample)
    # We want this to be LOW (different representations)
    similarity = jnp.sum(h0_norm * h3_norm, axis=-1)  # (batch,)

    # Loss: penalize high similarity
    # Using softplus for smooth gradient: log(1 + exp(sim/temp))
    # When sim is high -> loss is high
    # When sim is low/negative -> loss is low
    loss = jnp.mean(jax.nn.softplus(similarity / temperature))

    return loss


def compute_cycle_diversity_loss(
    intermediates: Dict,
    h_cycles: int = 4,
    temperature: float = 0.1,
) -> jnp.ndarray:
    """
    Extended version: encourage ALL adjacent H-cycles to be different.

    Computes pairwise similarity between consecutive H-cycles and penalizes
    high similarity.

    Loss = sum over h: softplus(cos_sim(h, h+1) / temperature)
    """
    latent_per_h = intermediates.get('latent_per_h', {})
    if len(latent_per_h) < 2:
        return jnp.array(0.0)

    total_loss = 0.0
    count = 0

    for h in range(h_cycles - 1):
        key_curr = f'h{h}'
        key_next = f'h{h+1}'

        if key_curr not in latent_per_h or key_next not in latent_per_h:
            continue

        curr_latent = latent_per_h[key_curr]
        next_latent = latent_per_h[key_next]

        # L2 normalize
        curr_norm = curr_latent / (jnp.linalg.norm(curr_latent, axis=-1, keepdims=True) + 1e-8)
        next_norm = next_latent / (jnp.linalg.norm(next_latent, axis=-1, keepdims=True) + 1e-8)

        # Cosine similarity
        similarity = jnp.sum(curr_norm * next_norm, axis=-1)

        # Penalize high similarity
        pair_loss = jnp.mean(jax.nn.softplus(similarity / temperature))
        total_loss = total_loss + pair_loss
        count += 1

    if count == 0:
        return jnp.array(0.0)

    return total_loss / count


def get_cycle_utilization_stats(
    intermediates: Dict,
    h_cycles: int = 4,
    l_cycles: int = 6,
) -> Dict[str, float]:
    """
    Compute utilization statistics for logging/monitoring.

    Returns dict with:
    - h{i}_norm: Total delta norm for each H-cycle
    - cv: Coefficient of variation
    - h0_h3_similarity: Cosine similarity between H0 and H3 latents
    """
    stats = {}

    # Delta norms per H-cycle
    latent_deltas = intermediates.get('latent_deltas', {})
    h_norms = []
    for h in range(h_cycles):
        h_total = 0.0
        for l in range(l_cycles):
            key = f'h{h}_l{l}'
            if key in latent_deltas:
                delta = latent_deltas[key]
                norm = float(jnp.mean(jnp.linalg.norm(delta, axis=-1)))
                h_total += norm
        stats[f'h{h}_norm'] = h_total
        h_norms.append(h_total)

    # CV
    if h_norms:
        h_norms = jnp.array(h_norms)
        mean_norm = jnp.mean(h_norms)
        std_norm = jnp.std(h_norms)
        stats['cv'] = float(std_norm / (mean_norm + 1e-8))

    # H0-H3 similarity
    latent_per_h = intermediates.get('latent_per_h', {})
    if 'h0' in latent_per_h and 'h3' in latent_per_h:
        h0 = latent_per_h['h0']
        h3 = latent_per_h['h3']
        h0_norm = h0 / (jnp.linalg.norm(h0, axis=-1, keepdims=True) + 1e-8)
        h3_norm = h3 / (jnp.linalg.norm(h3, axis=-1, keepdims=True) + 1e-8)
        similarity = jnp.mean(jnp.sum(h0_norm * h3_norm, axis=-1))
        stats['h0_h3_similarity'] = float(similarity)

    return stats
