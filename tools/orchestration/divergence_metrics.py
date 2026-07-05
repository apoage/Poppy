"""
Divergence Metrics for Branch Monitoring.

TASK_667: Measure compatibility between training branches.

Primary metric: Loss barrier at midpoint (most predictive of merge success)
"""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from typing import Any, Dict, Callable, Tuple
import numpy as np


# =============================================================================
# Weight Space Metrics
# =============================================================================

def flatten_params(params: Any) -> np.ndarray:
    """Flatten a PyTree of parameters into a single 1D numpy array."""
    leaves = jtu.tree_leaves(params)
    return np.concatenate([np.asarray(leaf).ravel() for leaf in leaves])


def cosine_similarity(params_a: Any, params_b: Any) -> float:
    """
    Compute cosine similarity between two parameter sets.

    cos(A, B) = (A · B) / (||A|| * ||B||)

    Thresholds (from research):
        > 0.99: Very aligned, safe to merge
        > 0.95: Normal, proceed with merge
        < 0.95: Warning, use Fisher merge
        < 0.90: Danger, branches have diverged significantly

    Uses numpy to avoid GPU allocation (orchestrator runs alongside training).

    Args:
        params_a, params_b: Parameter PyTrees (JAX arrays or numpy)

    Returns:
        Cosine similarity in [-1, 1]
    """
    flat_a = flatten_params(params_a)
    flat_b = flatten_params(params_b)

    dot_product = np.dot(flat_a, flat_b)
    norm_a = np.linalg.norm(flat_a)
    norm_b = np.linalg.norm(flat_b)

    return float(dot_product / (norm_a * norm_b + 1e-8))


def l2_distance(params_a: Any, params_b: Any) -> float:
    """Compute L2 (Euclidean) distance between parameter sets."""
    flat_a = flatten_params(params_a)
    flat_b = flatten_params(params_b)
    return float(np.linalg.norm(flat_a - flat_b))


def per_layer_cosine(params_a: Any, params_b: Any) -> Dict[str, float]:
    """
    Compute cosine similarity per layer.

    Useful for identifying which layers have diverged most.
    Early layers typically stay aligned; later layers diverge faster.

    Returns:
        Dict mapping layer paths to cosine similarities
    """
    results = {}

    flat_a = jtu.tree_flatten_with_path(params_a)[0]
    flat_b = jtu.tree_flatten_with_path(params_b)[0]

    for (path_a, leaf_a), (path_b, leaf_b) in zip(flat_a, flat_b):
        path_str = "/".join(str(p.key) if hasattr(p, 'key') else str(p) for p in path_a)

        a_flat = np.asarray(leaf_a).ravel()
        b_flat = np.asarray(leaf_b).ravel()

        dot = np.dot(a_flat, b_flat)
        norm_a = np.linalg.norm(a_flat)
        norm_b = np.linalg.norm(b_flat)

        cos_sim = float(dot / (norm_a * norm_b + 1e-8))
        results[path_str] = cos_sim

    return results


# =============================================================================
# Loss Landscape Metrics (Most Important)
# =============================================================================

def compute_loss_barrier(
    params_a: Any,
    params_b: Any,
    loss_fn: Callable[[Any], float],
    alphas: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
) -> Dict[str, float]:
    """
    Compute loss barrier along linear interpolation path.

    This is the GOLD STANDARD metric for merge safety.

    Barrier = max(loss along path) - average(endpoint losses)

    Thresholds:
        < 1%: Very safe, branches in same basin
        < 2%: Safe, proceed with normal merge
        > 2%: Warning, use Fisher merge + distillation
        > 5%: Danger, branches in different basins

    Args:
        params_a, params_b: Parameter PyTrees
        loss_fn: Function that takes params and returns scalar loss
        alphas: Interpolation points to evaluate

    Returns:
        Dict with 'barrier', 'midpoint_loss', 'endpoint_avg', and per-alpha losses
    """
    losses = {}

    for alpha in alphas:
        # Linear interpolation: alpha * A + (1 - alpha) * B
        interpolated = jtu.tree_map(
            lambda a, b: alpha * a + (1.0 - alpha) * b,
            params_a, params_b
        )
        loss = float(loss_fn(interpolated))
        losses[f"alpha_{alpha}"] = loss

    # Compute barrier
    endpoint_avg = 0.5 * (losses["alpha_0.0"] + losses["alpha_1.0"])
    max_loss = max(losses.values())
    midpoint_loss = losses.get("alpha_0.5", max_loss)

    barrier = max_loss - endpoint_avg
    barrier_relative = barrier / (endpoint_avg + 1e-8)

    return {
        "barrier": barrier,
        "barrier_relative": barrier_relative,
        "midpoint_loss": midpoint_loss,
        "endpoint_avg": endpoint_avg,
        **losses
    }


def quick_barrier_check(
    params_a: Any,
    params_b: Any,
    loss_fn: Callable[[Any], float]
) -> Tuple[float, float]:
    """
    Quick barrier check using only midpoint.

    Sufficient for screening - barrier is generally unimodal.

    Returns:
        (midpoint_loss, relative_barrier)
    """
    # Midpoint
    midpoint = jtu.tree_map(lambda a, b: 0.5 * a + 0.5 * b, params_a, params_b)
    midpoint_loss = float(loss_fn(midpoint))

    # Endpoints
    loss_a = float(loss_fn(params_a))
    loss_b = float(loss_fn(params_b))
    endpoint_avg = 0.5 * (loss_a + loss_b)

    barrier = midpoint_loss - endpoint_avg
    barrier_relative = barrier / (endpoint_avg + 1e-8)

    return midpoint_loss, barrier_relative


# =============================================================================
# Behavioral Metrics
# =============================================================================

def behavioral_agreement(
    policy_logits_a: jnp.ndarray,
    policy_logits_b: jnp.ndarray
) -> float:
    """
    Compute behavioral agreement: fraction of inputs where both models
    predict the same action.

    Pr[argmax(π_A) == argmax(π_B)]

    Thresholds:
        > 80%: Safe to merge
        60-80%: Cautious, layer-wise merge
        < 60%: Do not merge

    Args:
        policy_logits_a, policy_logits_b: (batch, n_actions) logits

    Returns:
        Agreement rate in [0, 1]
    """
    actions_a = jnp.argmax(policy_logits_a, axis=-1)
    actions_b = jnp.argmax(policy_logits_b, axis=-1)

    agreement = jnp.mean(actions_a == actions_b)
    return float(agreement)


def policy_kl_divergence(
    policy_logits_a: jnp.ndarray,
    policy_logits_b: jnp.ndarray
) -> float:
    """
    Compute KL divergence between policy distributions.

    KL(A || B) = sum(P_A * log(P_A / P_B))

    Lower is better (more similar policies).
    """
    probs_a = jax.nn.softmax(policy_logits_a, axis=-1)
    probs_b = jax.nn.softmax(policy_logits_b, axis=-1)

    # Add small epsilon for numerical stability
    eps = 1e-8
    kl = jnp.sum(probs_a * (jnp.log(probs_a + eps) - jnp.log(probs_b + eps)), axis=-1)

    return float(jnp.mean(kl))


def jensen_shannon_divergence(
    policy_logits_a: jnp.ndarray,
    policy_logits_b: jnp.ndarray
) -> float:
    """
    Compute Jensen-Shannon divergence (symmetric KL).

    JS(A, B) = 0.5 * KL(A || M) + 0.5 * KL(B || M), where M = 0.5*(A+B)

    Bounded in [0, 1] (using log base 2).
    """
    probs_a = jax.nn.softmax(policy_logits_a, axis=-1)
    probs_b = jax.nn.softmax(policy_logits_b, axis=-1)
    probs_m = 0.5 * (probs_a + probs_b)

    eps = 1e-8

    def kl(p, q):
        return jnp.sum(p * (jnp.log(p + eps) - jnp.log(q + eps)), axis=-1)

    js = 0.5 * kl(probs_a, probs_m) + 0.5 * kl(probs_b, probs_m)
    return float(jnp.mean(js))


# =============================================================================
# Representation Metrics
# =============================================================================

def linear_cka(
    activations_a: jnp.ndarray,
    activations_b: jnp.ndarray
) -> float:
    """
    Compute Linear CKA (Centered Kernel Alignment).

    Measures representation similarity, robust to rotation and scale.

    CKA > 0.8: Compatible representations
    CKA < 0.5: Different world models

    Args:
        activations_a, activations_b: (batch, features) hidden activations

    Returns:
        CKA score in [0, 1]
    """
    # Center columns
    a = activations_a - jnp.mean(activations_a, axis=0)
    b = activations_b - jnp.mean(activations_b, axis=0)

    # HSIC-based CKA
    numerator = jnp.linalg.norm(b.T @ a, ord='fro') ** 2
    denom_a = jnp.linalg.norm(a.T @ a, ord='fro')
    denom_b = jnp.linalg.norm(b.T @ b, ord='fro')

    cka = numerator / (denom_a * denom_b + 1e-8)
    return float(cka)


# =============================================================================
# Gradient Alignment
# =============================================================================

def gradient_cosine(
    grads_a: Any,
    grads_b: Any
) -> float:
    """
    Compute cosine similarity of gradients.

    Positive alignment (> 0.1): Cooperative, beneficial merge
    Near zero: Orthogonal tasks
    Negative: Conflicting gradients, may need gradient surgery

    Args:
        grads_a, grads_b: Gradient PyTrees

    Returns:
        Gradient cosine similarity
    """
    return cosine_similarity(grads_a, grads_b)


# =============================================================================
# Comprehensive Divergence Report
# =============================================================================

def compute_divergence_report(
    params_a: Any,
    params_b: Any,
    loss_fn: Callable[[Any], float] = None,
    policy_logits_a: jnp.ndarray = None,
    policy_logits_b: jnp.ndarray = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive divergence report.

    Args:
        params_a, params_b: Parameter PyTrees
        loss_fn: Optional loss function for barrier computation
        policy_logits_a, policy_logits_b: Optional policy outputs for behavioral metrics

    Returns:
        Dict with all computed metrics
    """
    report = {
        "weight_cosine": cosine_similarity(params_a, params_b),
        "l2_distance": l2_distance(params_a, params_b),
    }

    # Loss barrier (if loss_fn provided)
    if loss_fn is not None:
        midpoint_loss, barrier_rel = quick_barrier_check(params_a, params_b, loss_fn)
        report["midpoint_loss"] = midpoint_loss
        report["barrier_relative"] = barrier_rel

    # Behavioral metrics (if policy outputs provided)
    if policy_logits_a is not None and policy_logits_b is not None:
        report["behavioral_agreement"] = behavioral_agreement(policy_logits_a, policy_logits_b)
        report["js_divergence"] = jensen_shannon_divergence(policy_logits_a, policy_logits_b)

    # Determine merge recommendation
    cosine = report["weight_cosine"]
    barrier = report.get("barrier_relative", 0)

    if cosine > 0.99 and barrier < 0.01:
        report["recommendation"] = "SAFE - increase merge interval"
    elif cosine > 0.95 and barrier < 0.02:
        report["recommendation"] = "NORMAL - proceed with merge"
    elif cosine > 0.90 or barrier < 0.05:
        report["recommendation"] = "WARNING - use Fisher merge"
    else:
        report["recommendation"] = "DANGER - branches diverged significantly"

    return report


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser(description="Compute divergence metrics")
    parser.add_argument("ckpt_a", type=str, help="Path to checkpoint A")
    parser.add_argument("ckpt_b", type=str, help="Path to checkpoint B")

    args = parser.parse_args()

    print(f"Loading checkpoints...")
    with open(args.ckpt_a, 'rb') as f:
        ckpt_a = pickle.load(f)
    with open(args.ckpt_b, 'rb') as f:
        ckpt_b = pickle.load(f)

    params_a = ckpt_a['params']
    params_b = ckpt_b['params']

    print(f"\nComputing divergence metrics...")
    report = compute_divergence_report(params_a, params_b)

    print(f"\n{'='*50}")
    print(f"DIVERGENCE REPORT")
    print(f"{'='*50}")
    for key, value in report.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print(f"{'='*50}")
