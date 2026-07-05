# Stability modules (TASK_037-043)
from .exploration import (
    apply_epsilon_greedy,
    compute_policy_entropy,
    EpsilonSchedule,
    AdaptiveEpsilon,
    get_recommended_dirichlet_params,
    SACEntropyState,
)
from .flow_matching import (
    compute_flow_matching_loss,
    apply_flow_matching_loss,
    FlowMatchingDynamics,
)
# Deprecated: moved to .exploration
# from .sac_entropy import (...)
from .differential_reward import (
    compute_differential_reward,
    compute_differential_reward_from_values,
    DifferentialRewardWrapper,
    integrate_differential_reward_jax,
)

# TASK_119: Optimal hindsight path solvers
from .path_solver import (
    OptimalPath,
    greedy_lookahead_path,
    soft_topk_path,
    sinkhorn_topk_path,
    compute_optimal_path,
    compute_regret,
    dense_sign_oracle,
    dense_sign_labels_to_policy,
)

# Value head configuration
# Support range for categorical value representation
#
# CRITICAL: Must match the scale of training targets
#
# With PopArt normalization (default):
# - Targets normalized to mean=0, std=1
# - Typical range: [-3, +3] (99.7% of values within 3 sigma)
# - Extreme range: [-10, +10] (covers all practical cases)
# - Using 401 bins → 0.05 resolution (more bins than 320 actions)
#
# Without PopArt (raw MC returns):
# - Range would be [-750, +750] for symlog rewards
# - But PopArt is now the default, so we use normalized range
#
# IMPORTANT: If you disable PopArt, you must increase this range!
# NOTE: 401 bins > 320 actions ensures value head can distinguish all action values
VALUE_SUPPORT_MIN = -10
VALUE_SUPPORT_MAX = 10
VALUE_SUPPORT_SIZE = 401  # 0.05 resolution, more bins than actions (320)

def get_value_support():
    """Get the value support range as JAX linspace array"""
    import jax.numpy as jnp
    return jnp.linspace(VALUE_SUPPORT_MIN, VALUE_SUPPORT_MAX, VALUE_SUPPORT_SIZE)

def get_value_support_np():
    """Get the value support range as numpy array (for non-JAX code)"""
    import numpy as np
    return np.linspace(VALUE_SUPPORT_MIN, VALUE_SUPPORT_MAX, VALUE_SUPPORT_SIZE)

def get_support_range():
    """Get min, max, size tuple for value support"""
    return VALUE_SUPPORT_MIN, VALUE_SUPPORT_MAX, VALUE_SUPPORT_SIZE

def get_bin_width():
    """Get the width of each bin in the value support"""
    return (VALUE_SUPPORT_MAX - VALUE_SUPPORT_MIN) / (VALUE_SUPPORT_SIZE - 1)
