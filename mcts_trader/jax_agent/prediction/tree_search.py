"""
Physics-Aware Predictive Tree Search (TASK_012)

Monte Carlo exploration with:
- Noise scaled by learned volatility (sigma)
- Scenarios defined by physics parameters (alpha, beta)
- Full trajectory storage
- Risk-penalty MCTS integration

Based on NeuralOGCM + MCTS concepts.

All operations in LOG-SPACE for numerical stability.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Tuple, Callable, NamedTuple, Optional
from functools import partial

from .market_ode import NeuralMarketODE


class TreeSearchConfig(NamedTuple):
    """Configuration for predictive tree search."""
    depth: int = 5              # Number of prediction steps
    branches: int = 3           # Branches per node (total leaves = branches^depth)
    use_physics_scaling: bool = True   # Scale noise by learned sigma
    store_trajectory: bool = True      # Store full path, not just leaves


class TreeSearchResult(NamedTuple):
    """Result of predictive tree search."""
    leaves: jnp.ndarray           # (n_leaves, n_assets) - Leaf states
    trajectory: Optional[jnp.ndarray]  # (depth, n_nodes_at_depth, n_assets) - Full paths
    mean: jnp.ndarray             # (n_assets,) - Mean prediction
    std: jnp.ndarray              # (n_assets,) - Standard deviation
    var_5: jnp.ndarray            # (n_assets,) - 5th percentile (VaR 95%)
    var_95: jnp.ndarray           # (n_assets,) - 95th percentile
    expected_return: float        # Scalar - Mean return across scenarios
    uncertainty: float            # Scalar - Overall uncertainty (for risk penalty)


# ============================================================
# PHYSICS-AWARE TREE SEARCH
# ============================================================

def create_physics_aware_tree_search_fn(
    model: NeuralMarketODE,
    params: Dict,
    config: TreeSearchConfig = TreeSearchConfig()
) -> Callable:
    """
    Create JIT-compiled tree search with physics-aware noise scaling.

    Noise is scaled by the learned volatility (sigma) from the physics core.
    This ensures exploration adapts to market regime.

    Args:
        model: NeuralMarketODE instance
        params: Model parameters (includes physics_core.sigma)
        config: Tree search configuration

    Returns:
        tree_search_fn: Function that performs physics-aware tree search
    """
    depth = config.depth
    n_branches = config.branches
    store_trajectory = config.store_trajectory
    n_assets = model.n_assets

    # Extract learned volatility from physics core
    # sigma shape: (n_assets,)
    physics_params = params.get('params', params)
    if 'physics_core' in physics_params:
        sigma_raw = physics_params['physics_core']['sigma']
    else:
        # Fallback: try to find sigma at top level
        sigma_raw = physics_params.get('sigma', jnp.zeros(n_assets))

    sigma = nn.softplus(sigma_raw)  # Ensure positive

    def expand_level_physics_aware(
        states: jnp.ndarray,
        returns: jnp.ndarray,
        moving_avg: jnp.ndarray,
        rng: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Expand all states with physics-scaled noise.

        Args:
            states: (n_nodes, n_assets) - Current log prices
            returns: (n_nodes, n_assets) - Current log returns
            moving_avg: (n_nodes, n_assets) - Current MA
            rng: Random key

        Returns:
            next_states: (n_nodes * n_branches, n_assets)
            next_returns: (n_nodes * n_branches, n_assets)
            next_ma: (n_nodes * n_branches, n_assets)
        """
        n_nodes, n_a = states.shape

        # Tile states for each branch
        states_expanded = jnp.tile(states[:, None, :], (1, n_branches, 1))
        returns_expanded = jnp.tile(returns[:, None, :], (1, n_branches, 1))
        ma_expanded = jnp.tile(moving_avg[:, None, :], (1, n_branches, 1))

        # PHYSICS-AWARE NOISE: scale by learned volatility
        # This ensures high-volatility assets explore wider, low-volatility narrower
        base_noise = jax.random.normal(rng, (n_nodes, n_branches, n_a))
        scaled_noise = base_noise * sigma[None, None, :]  # Broadcast sigma

        # Flatten for batch prediction
        states_flat = states_expanded.reshape(n_nodes * n_branches, n_a)
        returns_flat = returns_expanded.reshape(n_nodes * n_branches, n_a)
        ma_flat = ma_expanded.reshape(n_nodes * n_branches, n_a)
        noise_flat = scaled_noise.reshape(n_nodes * n_branches, n_a)

        # Predict all branches in parallel using the model
        next_states, _ = model.apply(
            params,
            states_flat,
            returns_flat,
            ma_flat,
            noise_flat,
            features=None,  # No external features for tree search
            action=None,    # No action conditioning during search
        )

        # Update returns and moving average
        next_returns = next_states - states_flat  # Log returns
        ema_alpha = 0.1
        next_ma = ema_alpha * next_states + (1 - ema_alpha) * ma_flat

        return next_states, next_returns, next_ma

    # Pre-compile expand functions for each level size
    # This allows JIT compilation with known shapes
    expected_sizes = [n_branches ** i for i in range(depth)]

    @partial(jax.jit, static_argnums=(0,))
    def expand_at_level(n_nodes: int, states, returns, ma, rng):
        """JIT-compiled expansion for fixed size."""
        return expand_level_physics_aware(states, returns, ma, rng)

    def tree_search(
        init_prices: jnp.ndarray,
        rng: jnp.ndarray
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Run physics-aware tree search.

        Args:
            init_prices: (n_assets,) - Initial log prices
            rng: Random key

        Returns:
            leaves: (n_leaves, n_assets) - Final states
            trajectory: List of arrays if store_trajectory, else None
        """
        # Initialize
        states = init_prices[None, :]  # (1, n_assets)
        returns = jnp.zeros_like(states)
        ma = states * 0.99

        # Store trajectory if requested
        trajectory_list = [] if store_trajectory else None

        # Generate keys for each depth level
        keys = jax.random.split(rng, depth)

        # Expand tree level by level
        for d in range(depth):
            n_nodes = expected_sizes[d]
            states, returns, ma = expand_at_level(n_nodes, states, returns, ma, keys[d])
            if store_trajectory:
                trajectory_list.append(states)

        return states, trajectory_list

    return tree_search


# ============================================================
# STRUCTURAL SCENARIO SEARCH
# ============================================================

class StructuralScenarioConfig(NamedTuple):
    """
    Configuration for structural scenarios.

    Scenarios are defined by amplifying physics parameters,
    not by fixed bias values.
    """
    momentum_boost: float = 1.5     # Amplify alpha for "trend continues"
    reversion_boost: float = 1.5    # Amplify beta for "mean reversion"
    volatility_boost: float = 2.0   # Amplify sigma for "volatility spike"


def create_structural_scenario_search_fn(
    model: NeuralMarketODE,
    params: Dict,
    depth: int = 5,
    scenario_config: StructuralScenarioConfig = StructuralScenarioConfig()
) -> Callable:
    """
    Create tree search with structural scenarios.

    Instead of fixed biases, scenarios are defined by amplifying physics terms:
    - Scenario 1: Momentum dominates (alpha * 1.5)
    - Scenario 2: Reversion dominates (beta * 1.5)
    - Scenario 3: Volatility spike (sigma * 2.0)

    This explores structural uncertainty in market dynamics.

    Args:
        model: NeuralMarketODE instance
        params: Model parameters
        depth: Number of prediction steps
        scenario_config: Amplification factors

    Returns:
        scenario_tree_search_fn
    """
    n_scenarios = 3
    n_assets = model.n_assets

    # Extract physics parameters
    physics_params = params.get('params', params)
    if 'physics_core' in physics_params:
        pp = physics_params['physics_core']
    else:
        pp = physics_params

    # Get raw parameters and apply softplus
    alpha_raw = pp.get('alpha', jnp.zeros(1))
    beta_raw = pp.get('beta', jnp.zeros(1))
    sigma_raw = pp.get('sigma', jnp.zeros(n_assets))

    # Ensure alpha/beta are scalars
    alpha = float(nn.softplus(alpha_raw).flatten()[0])
    beta = float(nn.softplus(beta_raw).flatten()[0])
    sigma = nn.softplus(sigma_raw)

    # Create scenario-specific parameters (scalars for alpha/beta)
    # Scenario 0: Momentum boost
    alpha_scenarios = jnp.array([
        alpha * scenario_config.momentum_boost,  # Strong momentum
        alpha,                                    # Normal momentum
        alpha                                     # Normal momentum
    ])  # (3,)

    beta_scenarios = jnp.array([
        beta,                                     # Normal reversion
        beta * scenario_config.reversion_boost,   # Strong reversion
        beta                                      # Normal reversion
    ])  # (3,)

    sigma_scenarios = jnp.stack([
        sigma,                                    # Normal volatility
        sigma,                                    # Normal volatility
        sigma * scenario_config.volatility_boost  # High volatility
    ])  # (3, n_assets)

    def expand_structural_scenarios(
        states: jnp.ndarray,
        returns: jnp.ndarray,
        moving_avg: jnp.ndarray,
        rng: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Expand with scenario-specific physics parameters.

        Each scenario amplifies different physics components.
        """
        n_nodes, n_a = states.shape

        # Tile for scenarios
        states_expanded = jnp.tile(states[:, None, :], (1, n_scenarios, 1))
        returns_expanded = jnp.tile(returns[:, None, :], (1, n_scenarios, 1))
        ma_expanded = jnp.tile(moving_avg[:, None, :], (1, n_scenarios, 1))

        # Generate base noise
        base_noise = jax.random.normal(rng, (n_nodes, n_scenarios, n_a))

        # Scale noise by scenario-specific volatility
        # sigma_scenarios: (3, n_assets) -> broadcast to (n_nodes, 3, n_assets)
        scaled_noise = base_noise * sigma_scenarios[None, :, :]

        # Compute scenario-specific physics tendency
        # G_phys = alpha * returns - beta * (prices - ma) + sigma * noise
        # alpha_scenarios: (3,), beta_scenarios: (3,) - need to broadcast to (n_nodes, n_scenarios, n_assets)
        G_momentum = alpha_scenarios[None, :, None] * returns_expanded  # (n_nodes, 3, n_assets)
        G_reversion = -beta_scenarios[None, :, None] * (states_expanded - ma_expanded)  # (n_nodes, 3, n_assets)
        G_volatility = scaled_noise  # (n_nodes, 3, n_assets)

        # Simplified forward step (physics only, no neural corrector for scenarios)
        dt = 1.0
        G_total = G_momentum + G_reversion + G_volatility
        next_states = states_expanded + dt * G_total

        # Flatten
        next_states_flat = next_states.reshape(n_nodes * n_scenarios, n_a)
        states_flat = states_expanded.reshape(n_nodes * n_scenarios, n_a)
        ma_flat = ma_expanded.reshape(n_nodes * n_scenarios, n_a)

        next_returns = next_states_flat - states_flat  # Log returns
        next_ma = 0.1 * next_states_flat + 0.9 * ma_flat

        return next_states_flat, next_returns, next_ma

    # Pre-compile expand functions for each level size
    expected_sizes = [n_scenarios ** i for i in range(depth)]

    @partial(jax.jit, static_argnums=(0,))
    def expand_at_level(n_nodes: int, states, returns, ma, rng):
        """JIT-compiled expansion for fixed size."""
        return expand_structural_scenarios(states, returns, ma, rng)

    def structural_scenario_search(
        init_prices: jnp.ndarray,
        rng: jnp.ndarray
    ) -> Tuple[jnp.ndarray, Optional[list]]:
        """
        Run structural scenario tree search.

        Returns leaves and full trajectory.
        """
        # Initialize
        states = init_prices[None, :]
        returns = jnp.zeros_like(states)
        ma = states * 0.99

        trajectory_list = []

        keys = jax.random.split(rng, depth)

        # Expand tree level by level
        for d in range(depth):
            n_nodes = expected_sizes[d]
            states, returns, ma = expand_at_level(n_nodes, states, returns, ma, keys[d])
            trajectory_list.append(states)

        return states, trajectory_list

    return structural_scenario_search


# ============================================================
# ANALYSIS AND METRICS
# ============================================================

def analyze_tree_results(
    init_prices: jnp.ndarray,
    leaves: jnp.ndarray,
    trajectory: Optional[jnp.ndarray] = None
) -> TreeSearchResult:
    """
    Analyze tree search results to compute statistics.

    Args:
        init_prices: (n_assets,) - Initial log prices
        leaves: (n_leaves, n_assets) - All leaf states
        trajectory: Optional trajectory data

    Returns:
        TreeSearchResult with comprehensive statistics
    """
    # Basic statistics
    mean = leaves.mean(axis=0)
    std = leaves.std(axis=0)

    # Value at Risk (percentiles)
    var_5 = jnp.percentile(leaves, 5, axis=0)
    var_95 = jnp.percentile(leaves, 95, axis=0)

    # Expected return (in log space, returns are additive)
    returns = leaves - init_prices  # Log returns
    expected_return = float(returns.mean())

    # Overall uncertainty (scalar for risk penalty)
    # Use coefficient of variation averaged across assets
    cv = std / (jnp.abs(mean) + 1e-8)
    uncertainty = float(cv.mean())

    return TreeSearchResult(
        leaves=leaves,
        trajectory=trajectory,
        mean=mean,
        std=std,
        var_5=var_5,
        var_95=var_95,
        expected_return=expected_return,
        uncertainty=uncertainty
    )


# ============================================================
# MCTS INTEGRATION: RISK-ADJUSTED REWARD
# ============================================================

def create_risk_penalty_fn(
    tree_search_fn: Callable,
    risk_lambda: float = 0.1
) -> Callable:
    """
    Create function that computes risk penalty for MCTS reward adjustment.

    Instead of blending prediction with value (which is a hack), we compute
    an uncertainty penalty that adjusts the reward in MCTS dynamics.

    Args:
        tree_search_fn: Tree search function
        risk_lambda: Weight for uncertainty penalty

    Returns:
        compute_risk_penalty_fn: (prices, rng) -> risk_penalty
    """

    @jax.jit
    def compute_risk_penalty(
        prices: jnp.ndarray,
        rng: jnp.ndarray
    ) -> float:
        """
        Compute risk penalty based on future uncertainty.

        Args:
            prices: (n_assets,) - Current log prices
            rng: Random key

        Returns:
            risk_penalty: Scalar penalty (higher = more uncertain)
        """
        # Run tree search
        leaves, _ = tree_search_fn(prices, rng)

        # Improved uncertainty calculation
        returns = leaves - prices  # (n_leaves, n_assets)

        # Total variance across all scenarios and assets
        total_variance = returns.var()

        # Average of per-asset variances
        per_asset_variance = returns.var(axis=0)
        avg_asset_variance = per_asset_variance.mean()

        # VaR-style spread
        portfolio_returns = returns.mean(axis=1)
        var_5 = jnp.percentile(portfolio_returns, 5)
        var_95 = jnp.percentile(portfolio_returns, 95)
        var_spread = var_95 - var_5

        # Combined uncertainty
        uncertainty = jnp.sqrt(total_variance) + jnp.sqrt(avg_asset_variance) + 0.5 * var_spread

        # Risk penalty
        risk_penalty = risk_lambda * uncertainty

        return risk_penalty

    return compute_risk_penalty


def create_mcts_reward_adjuster(
    tree_search_fn: Callable,
    risk_lambda: float = 0.1
) -> Callable:
    """
    Create reward adjuster for MCTS dynamics.

    Adjusts reward by subtracting uncertainty penalty:
        adjusted_reward = raw_reward - lambda * uncertainty

    This encourages the agent to avoid high-uncertainty states.

    Args:
        tree_search_fn: Tree search function
        risk_lambda: Weight for uncertainty penalty

    Returns:
        adjust_reward_fn: (prices, raw_reward, rng) -> adjusted_reward
    """
    risk_penalty_fn = create_risk_penalty_fn(tree_search_fn, risk_lambda)

    @jax.jit
    def adjust_reward(
        prices: jnp.ndarray,
        raw_reward: float,
        rng: jnp.ndarray
    ) -> float:
        """
        Adjust reward by subtracting uncertainty penalty.

        Args:
            prices: (n_assets,) - Current state log prices
            raw_reward: Original reward (e.g., PnL)
            rng: Random key

        Returns:
            adjusted_reward: Risk-penalized reward
        """
        penalty = risk_penalty_fn(prices, rng)
        adjusted_reward = raw_reward - penalty
        return adjusted_reward

    return adjust_reward


# ============================================================
# BATCHED OPERATIONS
# ============================================================

def create_batched_tree_search_fn(
    model: NeuralMarketODE,
    params: Dict,
    config: TreeSearchConfig = TreeSearchConfig()
) -> Callable:
    """
    Create tree search for batch of starting states.

    Useful for evaluating multiple MCTS nodes simultaneously.

    Args:
        model: NeuralMarketODE instance
        params: Model parameters
        config: Tree search configuration

    Returns:
        batched_tree_search_fn: (batch_prices, rng) -> (batch, n_leaves, n_assets)
    """
    single_tree_search = create_physics_aware_tree_search_fn(model, params, config)

    @jax.jit
    def batched_tree_search(
        batch_prices: jnp.ndarray,
        rng: jnp.ndarray
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Run tree search for batch of states.

        Args:
            batch_prices: (batch, n_assets)
            rng: Random key

        Returns:
            results: (batch, n_leaves, n_assets)
            trajectories: Optional batch trajectories
        """
        batch_size = batch_prices.shape[0]

        # Generate keys for each batch item
        keys = jax.random.split(rng, batch_size)

        # vmap over batch
        vmapped_search = jax.vmap(single_tree_search)
        all_leaves, all_trajectories = vmapped_search(batch_prices, keys)

        return all_leaves, all_trajectories

    return batched_tree_search


# ============================================================
# LEGACY COMPATIBILITY (Simple Tree Search)
# ============================================================

def create_tree_search_fn(
    model: NeuralMarketODE,
    params: Dict,
    config: TreeSearchConfig = TreeSearchConfig()
) -> Callable:
    """
    Alias for create_physics_aware_tree_search_fn.

    Provided for backwards compatibility.
    """
    return create_physics_aware_tree_search_fn(model, params, config)


# ============================================================
# DYNAMIC PARAMS TREE SEARCH (for trained models)
# ============================================================

def create_dynamic_tree_search_fn(
    model: NeuralMarketODE,
    config: TreeSearchConfig = TreeSearchConfig()
) -> Callable:
    """
    Create tree search that accepts params at RUNTIME (not closure).

    This is critical for using trained prediction models - the params
    are updated during training and we need to use the latest values.

    Args:
        model: NeuralMarketODE instance (structure only, not params)
        config: Tree search configuration

    Returns:
        tree_search_fn: (params, prices, rng) -> (leaves, trajectory)
                        Note: takes params as first argument!
    """
    depth = config.depth
    n_branches = config.branches
    store_trajectory = config.store_trajectory
    n_assets = model.n_assets

    expected_sizes = [n_branches ** i for i in range(depth)]

    def tree_search_with_params(
        params: Dict,
        init_prices: jnp.ndarray,
        rng: jnp.ndarray
    ) -> Tuple[jnp.ndarray, Optional[list]]:
        """
        Run physics-aware tree search with provided params.

        Args:
            params: Model parameters (will use latest trained values)
            init_prices: (n_assets,) - Initial log prices
            rng: Random key

        Returns:
            leaves: (n_leaves, n_assets) - All leaf states
            trajectory: List of states at each depth (if store_trajectory)
        """
        # Extract sigma from current params
        physics_params = params.get('params', params)
        if 'physics_core' in physics_params:
            sigma_raw = physics_params['physics_core']['sigma']
        else:
            sigma_raw = physics_params.get('sigma', jnp.zeros(n_assets))
        sigma = nn.softplus(sigma_raw)

        # Initialize
        states = init_prices[None, :]
        returns = jnp.zeros_like(states)
        ma = states * 0.99

        trajectory_list = []
        keys = jax.random.split(rng, depth)

        for d in range(depth):
            n_nodes = expected_sizes[d]

            # Expand with physics-aware noise
            states_expanded = jnp.tile(states[:, None, :], (1, n_branches, 1))
            returns_expanded = jnp.tile(returns[:, None, :], (1, n_branches, 1))
            ma_expanded = jnp.tile(ma[:, None, :], (1, n_branches, 1))

            # Physics-aware noise scaling
            base_noise = jax.random.normal(keys[d], (n_nodes, n_branches, n_assets))
            scaled_noise = base_noise * sigma[None, None, :]

            # Flatten
            states_flat = states_expanded.reshape(n_nodes * n_branches, n_assets)
            returns_flat = returns_expanded.reshape(n_nodes * n_branches, n_assets)
            ma_flat = ma_expanded.reshape(n_nodes * n_branches, n_assets)
            noise_flat = scaled_noise.reshape(n_nodes * n_branches, n_assets)

            # Forward through model with CURRENT params
            next_states, _ = model.apply(
                params,
                states_flat,
                returns_flat,
                ma_flat,
                noise_flat,
                features=None,
                action=None,
            )

            states = next_states
            returns = states - states_flat
            ma = 0.1 * states + 0.9 * ma_flat

            if store_trajectory:
                trajectory_list.append(states)

        trajectory = trajectory_list if store_trajectory else None
        return states, trajectory

    return jax.jit(tree_search_with_params)


def create_dynamic_reward_adjuster(
    model: NeuralMarketODE,
    config: TreeSearchConfig = TreeSearchConfig(),
    risk_lambda: float = 0.1
) -> Callable:
    """
    Create reward adjuster that uses RUNTIME params (not baked-in).

    This is the version to use when you want tree search to use
    trained prediction model params that update during training.

    Args:
        model: NeuralMarketODE instance
        config: Tree search configuration
        risk_lambda: Uncertainty penalty weight

    Returns:
        adjust_reward_fn: (params, prices, raw_reward, rng) -> adjusted_reward
                          Note: takes params as first argument!
    """
    tree_search_fn = create_dynamic_tree_search_fn(model, config)
    n_assets = model.n_assets

    @jax.jit
    def adjust_reward_with_params(
        params: Dict,
        prices: jnp.ndarray,
        raw_reward: float,
        rng: jnp.ndarray
    ) -> float:
        """
        Adjust reward using current model params.

        Args:
            params: Current prediction model parameters
            prices: (n_assets,) - Current state log prices
            raw_reward: Original reward
            rng: Random key

        Returns:
            adjusted_reward: Risk-penalized reward
        """
        # Run tree search with current params
        leaves, _ = tree_search_fn(params, prices, rng)

        # Compute uncertainty - improved calculation
        # leaves: (n_leaves, n_assets), prices: (n_assets,)
        returns = leaves - prices  # (n_leaves, n_assets)

        # Method 1: Total variance across all scenarios and assets
        # This captures both scenario diversity AND asset-level uncertainty
        total_variance = returns.var()  # Scalar - total variance

        # Method 2: Average of per-asset variances (captures asset-level uncertainty)
        per_asset_variance = returns.var(axis=0)  # (n_assets,)
        avg_asset_variance = per_asset_variance.mean()

        # Method 3: VaR-style spread (5th to 95th percentile)
        portfolio_returns = returns.mean(axis=1)  # (n_leaves,) - average across assets
        var_5 = jnp.percentile(portfolio_returns, 5)
        var_95 = jnp.percentile(portfolio_returns, 95)
        var_spread = var_95 - var_5  # Range of likely outcomes

        # Combined uncertainty metric:
        # - sqrt(total_variance) gives std-like scale
        # - avg_asset_variance adds asset-level risk
        # - var_spread adds tail risk
        uncertainty = jnp.sqrt(total_variance) + jnp.sqrt(avg_asset_variance) + 0.5 * var_spread

        # Penalize
        adjusted = raw_reward - risk_lambda * uncertainty
        return adjusted

    return adjust_reward_with_params
