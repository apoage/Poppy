"""
Merge Utilities for Federated Easy Mode Training.

TASK_667: Research-validated merging strategies.

Key insight: Adam's v accumulator ≈ diagonal Fisher (Squisher, Li et al. 2025)
"""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pickle
from pathlib import Path


# =============================================================================
# Core Merge Functions
# =============================================================================

def fed_avg(params_a: Any, params_b: Any, alpha: float = 0.5) -> Any:
    """
    Simple Federated Averaging.

    Args:
        params_a: PyTree of parameters from Branch A
        params_b: PyTree of parameters from Branch B
        alpha: Weight for Branch A (default 0.5 = equal)

    Returns:
        Merged PyTree: alpha * A + (1 - alpha) * B
    """
    return jtu.tree_map(
        lambda a, b: alpha * a + (1.0 - alpha) * b,
        params_a, params_b
    )


def normalize_fisher(fisher: Any) -> Any:
    """
    Normalize Fisher PyTree so each branch has equal total importance.

    Without normalization, a branch with higher variance tasks (e.g., Bella)
    will dominate over a branch with lower variance (e.g., simple L0 sine).

    Args:
        fisher: Fisher information PyTree

    Returns:
        Normalized Fisher where sum(abs) = 1.0 per leaf
    """
    def normalize_leaf(f):
        total = jnp.sum(jnp.abs(f)) + 1e-10
        return f / total

    return jtu.tree_map(normalize_leaf, fisher)


def fisher_merge(
    params_a: Any,
    params_b: Any,
    fisher_a: Any,
    fisher_b: Any,
    epsilon: float = 1e-8,
    normalize: bool = True
) -> Any:
    """
    Fisher-weighted merging using importance scores.

    Uses diagonal Fisher approximation (from Adam's v accumulator).

    Formula: W_i = (F_a * W_a + F_b * W_b) / (F_a + F_b)

    Args:
        params_a, params_b: Parameter PyTrees
        fisher_a, fisher_b: Fisher information PyTrees (≈ Adam's v)
        epsilon: Small value to prevent division by zero
        normalize: Normalize Fisher before merging (prevents one branch dominating)

    Returns:
        Fisher-weighted merged parameters
    """
    if normalize:
        fisher_a = normalize_fisher(fisher_a)
        fisher_b = normalize_fisher(fisher_b)

    return jtu.tree_map(
        lambda p_a, p_b, f_a, f_b: (f_a * p_a + f_b * p_b) / (f_a + f_b + epsilon),
        params_a, params_b, fisher_a, fisher_b
    )


def performance_weighted_merge(
    params_a: Any,
    params_b: Any,
    score_a: float,
    score_b: float,
    temperature: float = 1.0
) -> Any:
    """
    Merge weighted by validation performance (agreement scores).

    Args:
        params_a, params_b: Parameter PyTrees
        score_a, score_b: Validation scores (e.g., oracle agreement)
        temperature: Softmax temperature (higher = more uniform)

    Returns:
        Performance-weighted merged parameters
    """
    scores = jnp.array([score_a, score_b])
    weights = jax.nn.softmax(scores / temperature)
    alpha = float(weights[0])

    return fed_avg(params_a, params_b, alpha)


def layer_wise_merge(
    params_a: Any,
    params_b: Any,
    layer_alphas: Dict[str, float],
    default_alpha: float = 0.5
) -> Any:
    """
    Layer-wise merging with different ratios per component.

    Args:
        params_a, params_b: Parameter PyTrees
        layer_alphas: Dict mapping layer name patterns to alpha values
            e.g., {"Encoder": 0.5, "policy": 0.7, "value": 0.6}
        default_alpha: Default alpha for unmatched layers

    Returns:
        Layer-wise merged parameters
    """
    def merge_leaf(path, a, b):
        path_str = "/".join(str(p.key) if hasattr(p, 'key') else str(p) for p in path)

        # Find matching alpha
        alpha = default_alpha
        for pattern, pattern_alpha in layer_alphas.items():
            if pattern.lower() in path_str.lower():
                alpha = pattern_alpha
                break

        return alpha * a + (1.0 - alpha) * b

    return jtu.tree_map_with_path(merge_leaf, params_a, params_b)


def ties_merge(
    params_a: Any,
    params_b: Any,
    params_init: Any,
    trim_fraction: float = 0.2
) -> Any:
    """
    TIES-Merging: Trim, Elect Sign, Disjoint Merge.

    Handles sign conflicts between branches (Yadav et al., NeurIPS 2023).

    Args:
        params_a, params_b: Fine-tuned parameter PyTrees
        params_init: Initial/base parameters (before training)
        trim_fraction: Fraction of smallest-magnitude updates to trim

    Returns:
        TIES-merged parameters
    """
    # Compute task vectors (deltas from init)
    delta_a = jtu.tree_map(lambda a, i: a - i, params_a, params_init)
    delta_b = jtu.tree_map(lambda b, i: b - i, params_b, params_init)

    def process_layer(da, db):
        # TRIM: Zero out bottom k% by magnitude
        def get_trim_mask(d):
            # BUG-4 fix: trim_fraction=0.2 means trim bottom 20%, keep top 80%
            threshold = jnp.percentile(jnp.abs(d), trim_fraction * 100)
            return jnp.abs(d) >= threshold

        mask_a = get_trim_mask(da)
        mask_b = get_trim_mask(db)

        da_trim = da * mask_a
        db_trim = db * mask_b

        # ELECT SIGN: Majority vote on direction
        total_delta = da_trim + db_trim
        elected_sign = jnp.sign(total_delta)

        # DISJOINT MERGE: Only keep updates matching elected sign
        valid_a = (jnp.sign(da_trim) == elected_sign) & mask_a
        valid_b = (jnp.sign(db_trim) == elected_sign) & mask_b

        count = valid_a.astype(jnp.float32) + valid_b.astype(jnp.float32)
        sum_updates = (da_trim * valid_a) + (db_trim * valid_b)

        final_delta = jnp.where(count > 0, sum_updates / jnp.maximum(count, 1), 0.0)
        return final_delta

    merged_deltas = jtu.tree_map(process_layer, delta_a, delta_b)

    # Add merged deltas back to init
    return jtu.tree_map(lambda i, d: i + d, params_init, merged_deltas)


# =============================================================================
# Optimizer State Handling
# =============================================================================

def merge_optimizer_state(
    opt_state_a: Any,
    opt_state_b: Any,
    reset_momentum: bool = True
) -> Tuple[Any, Any]:
    """
    Merge optimizer states correctly.

    Research-validated approach:
    - Average second moment (v) - serves as merged Fisher estimate
    - Reset first moment (m) - momentum vectors are branch-specific

    Args:
        opt_state_a, opt_state_b: Optax optimizer states
        reset_momentum: Whether to reset first moment (recommended: True)

    Returns:
        (merged_v, merged_m) tuple for reconstructing optimizer state
    """
    # Optax AdamW state structure: (ScaleByAdamState(count, mu, nu), ScaleByScheduleState, ...)
    # mu = first moment (momentum), nu = second moment (≈ Fisher)

    # Extract mu and nu from both states
    # Structure depends on optimizer chain - try common patterns
    def get_adam_state(opt_state):
        """Find ScaleByAdamState in optimizer state tree."""
        # Pattern 1: opt_state[0] is Adam (simple chain)
        if hasattr(opt_state[0], 'mu') and hasattr(opt_state[0], 'nu'):
            return opt_state[0]
        # Pattern 2: opt_state[1][0] is Adam (multi_transform wrapper)
        if len(opt_state) > 1 and isinstance(opt_state[1], tuple):
            if hasattr(opt_state[1][0], 'mu') and hasattr(opt_state[1][0], 'nu'):
                return opt_state[1][0]
        # Pattern 3: opt_state[0][0] is Adam (nested wrapper)
        if isinstance(opt_state[0], tuple) and hasattr(opt_state[0][0], 'mu'):
            return opt_state[0][0]
        return None

    try:
        adam_a = get_adam_state(opt_state_a)
        adam_b = get_adam_state(opt_state_b)
        if adam_a is None or adam_b is None:
            raise AttributeError("Could not find Adam state")
        mu_a, nu_a = adam_a.mu, adam_a.nu
        mu_b, nu_b = adam_b.mu, adam_b.nu
    except (IndexError, AttributeError) as e:
        print(f"Warning: Could not extract optimizer moments ({e}), using default merge")
        return opt_state_a, opt_state_a

    # Average second moment (Fisher approximation)
    merged_nu = jtu.tree_map(lambda a, b: 0.5 * a + 0.5 * b, nu_a, nu_b)

    # Reset or average first moment
    if reset_momentum:
        merged_mu = jtu.tree_map(lambda m: jnp.zeros_like(m), mu_a)
    else:
        merged_mu = jtu.tree_map(lambda a, b: 0.5 * a + 0.5 * b, mu_a, mu_b)

    return merged_nu, merged_mu


def extract_fisher_from_optimizer(opt_state: Any) -> Any:
    """
    Extract Fisher approximation (Adam's v/nu) from optimizer state.

    The Squisher shortcut: Adam's second moment ≈ diagonal Fisher.

    Args:
        opt_state: Optax optimizer state

    Returns:
        Fisher approximation PyTree
    """
    # Pattern 1: opt_state[0] is Adam (simple chain)
    if hasattr(opt_state[0], 'nu'):
        return opt_state[0].nu
    # Pattern 2: opt_state[1][0] is Adam (multi_transform wrapper)
    if len(opt_state) > 1 and isinstance(opt_state[1], tuple):
        if hasattr(opt_state[1][0], 'nu'):
            return opt_state[1][0].nu
    # Pattern 3: opt_state[0][0] is Adam (nested wrapper)
    if isinstance(opt_state[0], tuple) and hasattr(opt_state[0][0], 'nu'):
        return opt_state[0][0].nu
    raise ValueError("Could not extract nu (second moment) from optimizer state")


# =============================================================================
# Checkpoint I/O
# =============================================================================

def load_checkpoint(path: str, cpu: bool = False) -> Dict[str, Any]:
    """Load checkpoint from pickle file.

    Args:
        path: Path to checkpoint pickle file.
        cpu: If True, move params/opt_state to numpy arrays (avoids GPU allocation).
             Use this when loading for merge comparison in the orchestrator.
    """
    with open(path, 'rb') as f:
        ckpt = pickle.load(f)
    if cpu:
        def _to_numpy(x):
            if hasattr(x, 'shape'):
                return np.asarray(x)
            return x
        for key in ('params', 'opt_state'):
            if key in ckpt:
                ckpt[key] = jax.tree.map(_to_numpy, ckpt[key])
    return ckpt


def save_checkpoint(checkpoint: Dict[str, Any], path: str):
    """Save checkpoint to pickle file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"Saved checkpoint to {path}")


def merge_checkpoints(
    ckpt_a_path: str,
    ckpt_b_path: str,
    output_path: str,
    method: str = "fisher",
    alpha: float = 0.5,
    init_ckpt_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Merge two checkpoints using specified method.

    Args:
        ckpt_a_path, ckpt_b_path: Paths to checkpoints
        output_path: Path for merged checkpoint
        method: "fedavg", "fisher", "performance", "ties"
        alpha: Merge ratio for FedAvg (ignored for other methods)
        init_ckpt_path: Initial checkpoint for TIES (required for TIES)

    Returns:
        Merged checkpoint dict
    """
    print(f"Loading checkpoints...")
    ckpt_a = load_checkpoint(ckpt_a_path)
    ckpt_b = load_checkpoint(ckpt_b_path)

    params_a = ckpt_a['params']
    params_b = ckpt_b['params']

    # Get scores for performance-weighted
    score_a = ckpt_a.get('metrics', {}).get('oracle_agreement', 0.5)
    score_b = ckpt_b.get('metrics', {}).get('oracle_agreement', 0.5)

    print(f"Merging with method={method}...")
    print(f"  Branch A agreement: {score_a:.1%}")
    print(f"  Branch B agreement: {score_b:.1%}")

    if method == "fedavg":
        merged_params = fed_avg(params_a, params_b, alpha)

    elif method == "fisher":
        # Extract Fisher from optimizer states
        if 'opt_state' in ckpt_a and 'opt_state' in ckpt_b:
            fisher_a = extract_fisher_from_optimizer(ckpt_a['opt_state'])
            fisher_b = extract_fisher_from_optimizer(ckpt_b['opt_state'])
            merged_params = fisher_merge(params_a, params_b, fisher_a, fisher_b)
        else:
            print("  Warning: No optimizer state found, falling back to FedAvg")
            merged_params = fed_avg(params_a, params_b, alpha)

    elif method == "performance":
        merged_params = performance_weighted_merge(params_a, params_b, score_a, score_b)

    elif method == "ties":
        if init_ckpt_path is None:
            raise ValueError("TIES merge requires init_ckpt_path")
        ckpt_init = load_checkpoint(init_ckpt_path)
        merged_params = ties_merge(params_a, params_b, ckpt_init['params'])

    else:
        raise ValueError(f"Unknown merge method: {method}")

    # Handle optimizer state
    merged_opt_state = None
    if 'opt_state' in ckpt_a and 'opt_state' in ckpt_b:
        merged_nu, merged_mu = merge_optimizer_state(
            ckpt_a['opt_state'],
            ckpt_b['opt_state'],
            reset_momentum=True
        )
        # Note: Full optimizer state reconstruction requires the optimizer object
        # For now, we store the components
        merged_opt_state = {
            'merged_nu': merged_nu,
            'merged_mu': merged_mu,
            'note': 'Reconstruct with trainer.rebuild_optimizer_state()'
        }

    # Create merged checkpoint
    merged_ckpt = {
        'params': merged_params,
        'epoch': max(ckpt_a.get('epoch', 0), ckpt_b.get('epoch', 0)),
        'metrics': {
            'source': 'merged',
            'method': method,
            'a_agreement': score_a,
            'b_agreement': score_b,
        },
        'config': ckpt_a.get('config', {}),
    }

    if merged_opt_state:
        merged_ckpt['merged_opt_components'] = merged_opt_state

    save_checkpoint(merged_ckpt, output_path)
    print(f"Done! Merged checkpoint saved to {output_path}")

    return merged_ckpt


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge Easy Mode checkpoints")
    parser.add_argument("ckpt_a", type=str, help="Path to checkpoint A")
    parser.add_argument("ckpt_b", type=str, help="Path to checkpoint B")
    parser.add_argument("-o", "--output", type=str, default="merged.pkl", help="Output path")
    parser.add_argument("-m", "--method", type=str, default="fisher",
                        choices=["fedavg", "fisher", "performance", "ties"],
                        help="Merge method")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha for FedAvg")
    parser.add_argument("--init", type=str, default=None, help="Init checkpoint for TIES")

    args = parser.parse_args()

    merge_checkpoints(
        args.ckpt_a,
        args.ckpt_b,
        args.output,
        method=args.method,
        alpha=args.alpha,
        init_ckpt_path=args.init
    )
