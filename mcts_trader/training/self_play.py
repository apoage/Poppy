"""
Self-play episode generation for MuZero training.

This module contains three variants of self-play:
- run_self_play: Single-environment episode generation
- run_self_play_batched: Batched Python (VectorBT) environments
- run_self_play_jax: Fully GPU-accelerated JAX environments

TASK_121: Fixed observation window to use LOOKBACK (not forward).
"""
print("[TASK_121] self_play.py loaded with LOOKBACK window fix")
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any

from mcts_trader.jax_agent import VALUE_SUPPORT_SIZE, get_value_support, get_value_support_np
from mcts_trader.jax_agent.popart import PopArtState
from mcts_trader.jax_agent.returns import compute_returns
from mcts_trader.jax_agent.path_solver import compute_optimal_path


def denormalize_value(value: float, popart_state: Optional[PopArtState]) -> float:
    """Denormalize a value prediction using PopArt statistics.

    Args:
        value: Normalized value prediction
        popart_state: PopArt state with mean/var statistics (or None to skip)

    Returns:
        Denormalized value in original scale
    """
    if popart_state is None:
        return value
    std = float(jnp.sqrt(popart_state.var + 1e-8))
    mean = float(popart_state.mean)
    return value * std + mean

# Action space: 5 actions per asset [Sell, Hold, Buy, Limit Sell, Limit Buy]
N_ACTIONS_PER_ASSET = 5


def run_self_play(
    env,
    search_fn: Callable,
    model_apply_fn: Callable,
    params: Dict,
    rng_key: jnp.ndarray,
    steps: int = 200,
    confidence_threshold: float = 0.0,
    support_size: int = VALUE_SUPPORT_SIZE,
    collect_prices: bool = False,
    risk_adjust_fn: Optional[Callable] = None,
    pred_params: Optional[Dict] = None,
    discount: float = 0.997,
    popart_state: Optional[PopArtState] = None,
    use_gae: bool = False,
    gae_lambda: float = 0.95,
    on_episode_complete: Optional[Callable[[Dict, float], None]] = None,  # TASK_125
) -> Tuple[Dict, float]:
    """Self-play with timing. Optionally collects price data for prediction loss and risk-adjusted rewards.

    Args:
        env: Trading environment instance
        search_fn: MCTS search function
        model_apply_fn: Model apply function
        params: Model parameters
        rng_key: JAX RNG key
        steps: Number of steps per episode
        confidence_threshold: Min policy probability to execute action (else HOLD)
        support_size: Value support size for categorical distribution
        collect_prices: Whether to collect price data for prediction loss
        risk_adjust_fn: Optional risk adjustment function for tree search
        pred_params: Prediction model params for dynamic tree search. If provided, risk_adjust_fn
                     will be called as risk_adjust_fn(pred_params, prices, reward, rng) instead of
                     risk_adjust_fn(prices, reward, rng).
        discount: Discount factor for returns
        popart_state: PopArt state for denormalizing MCTS root values (or None to skip)
        on_episode_complete: TASK_125 callback for Causal Expert Imprinting.
                             Called with (trajectory_dict, episode_score) after episode ends.
                             Used to add trajectory to expert buffer for hindsight analysis.

    Returns:
        Tuple of (trajectory_dict, episode_score)
    """
    obs_dict, info = env.reset()

    observations, actions, rewards, values, policy_probs, dones = [], [], [], [], [], []
    # MCTS root values for DS24 planning metrics
    mcts_root_values = []
    # Price data for prediction loss (Phase 2)
    prices_list, next_prices_list = [], []
    # Uncertainty tracking for tree search (Phase 3)
    uncertainty_samples = []
    score = 0.0
    rng = rng_key

    for step in range(steps):
        rng, step_rng = jax.random.split(rng)

        # Prepare observation
        obs_array = obs_dict['observation']
        if hasattr(obs_array, 'cpu'):
            obs_array = obs_array.cpu().numpy()
        obs_array = np.expand_dims(obs_array, axis=0)
        obs_jax = jnp.array(obs_array, dtype=jnp.float32)

        # Prepare Mask
        mask_array = obs_dict['action_mask']
        mask_array = np.expand_dims(mask_array, axis=0)
        mask_jax = jnp.array(mask_array, dtype=jnp.int8)

        # MCTS
        action_jax, policy_jax, stats = search_fn(params, step_rng, obs_jax, mask_jax)

        # Value (MCTS root value for DS24 logging)
        # TASK_031: Denormalize to original scale for meaningful diagnostics
        if 'root_value' in stats:
            value_scalar = float(stats['root_value'][0])
        else:
            output = model_apply_fn({'params': params}, obs_jax, mode='initial_inference')
            support = get_value_support()
            value_probs = jax.nn.softmax(output['value'][0])
            value_scalar = float(jnp.sum(value_probs * support))
        # Denormalize if PopArt is active
        value_denorm = denormalize_value(value_scalar, popart_state)
        mcts_root_values.append(value_denorm)

        # MCTS Tree Analytics (every 10% of episode or min 100 steps)
        log_interval = max(100, steps // 10)
        if step % log_interval == 0 and step > 0:
            # Aggregate action types from MCTS policy
            policy_np = np.array(policy_jax[0])

            # Sum probabilities by action type across all assets
            # 5-action: [0] Sell, [1] Hold, [2] Buy, [3] Limit Sell, [4] Limit Buy
            sell_prob = policy_np[0::5].sum()
            hold_prob = policy_np[1::5].sum()
            buy_prob = policy_np[2::5].sum()
            limit_sell_prob = policy_np[3::5].sum()
            limit_buy_prob = policy_np[4::5].sum()
            action_summary = f"S={sell_prob:.3f} H={hold_prob:.3f} B={buy_prob:.3f} LS={limit_sell_prob:.3f} LB={limit_buy_prob:.3f}"

            # Policy entropy (measure of exploration)
            policy_entropy = -np.sum(policy_np * np.log(policy_np + 1e-10))

            # Value head distribution analysis
            output = model_apply_fn({'params': params}, obs_jax, mode='initial_inference')
            value_probs = jax.nn.softmax(output['value'][0])
            value_probs_np = np.array(value_probs)

            # Find peak bin and concentration
            peak_bin = np.argmax(value_probs_np)
            peak_prob = value_probs_np[peak_bin]
            support_np = get_value_support_np()
            peak_value = support_np[peak_bin]

            # Entropy of value distribution
            value_entropy = -np.sum(value_probs_np * np.log(value_probs_np + 1e-10))

            print(f"  [MCTS Step {step}] Policy: {action_summary} | Entropy={policy_entropy:.2f} | Value={value_scalar:.3f}")
            print(f"    Value Head: Peak bin={peak_bin}/{len(value_probs_np)} ({peak_value:.2f}) | Peak prob={peak_prob:.3f} | Entropy={value_entropy:.2f}")

        # Execute
        action_int = int(action_jax[0])

        # Confidence Thresholding (Gate-keeping)
        # TASK_034 FIX: Also update policy when forcing HOLD to avoid action-policy mismatch
        if confidence_threshold > 0:
            probs = np.array(policy_jax[0])
            max_prob = np.max(probs)
            if max_prob < confidence_threshold:
                # Force HOLD: (asset_idx * N_ACTIONS_PER_ASSET) + 1
                asset_idx = action_int // N_ACTIONS_PER_ASSET
                hold_action = (asset_idx * N_ACTIONS_PER_ASSET) + 1
                action_int = hold_action

                # Create synthetic confident HOLD policy to match forced action
                # 90% on HOLD, 10% spread across others (with smoothing for stability)
                synthetic_policy = np.ones_like(probs) * (0.1 / (len(probs) - 1))
                synthetic_policy[hold_action] = 0.9
                policy_jax = jnp.array([synthetic_policy])  # Update policy to match action

        next_obs_dict, reward, terminated, truncated, info = env.step(action_int)
        done = terminated or truncated

        # Risk-adjusted reward using tree search (Phase 3)
        # Sample uncertainty every 50 steps to avoid overhead
        if risk_adjust_fn is not None and step % 50 == 0 and hasattr(env, 'prices') and env.prices is not None:
            rng, tree_rng = jax.random.split(rng)
            try:
                # Use correct attribute names from VectorBTTradingEnv
                current_idx = env.episode_start_idx + env.current_step
                if current_idx < len(env.prices):
                    # Convert prices to log-space for tree search
                    log_prices = jnp.log(jnp.array(env.prices[current_idx]) + 1e-8)

                    # Call reward adjuster - different signature for dynamic vs static params
                    if pred_params is not None:
                        # Dynamic: uses trained prediction model params
                        adjusted_reward = risk_adjust_fn(pred_params, log_prices, float(reward), tree_rng)
                    else:
                        # Static: uses baked-in params
                        adjusted_reward = risk_adjust_fn(log_prices, float(reward), tree_rng)

                    uncertainty = float(reward) - float(adjusted_reward)  # Penalty amount
                    uncertainty_samples.append(uncertainty)
            except Exception as e:
                # Log first error for debugging
                if len(uncertainty_samples) == 0:
                    print(f"Warning: Tree search error: {e}")
                pass  # Don't crash training on tree search errors

        # Store
        observations.append(obs_array[0])
        actions.append(action_int)
        rewards.append(reward)
        values.append(float(value_scalar))
        # BUG-005 FIX: Ensure policy targets are normalized probability distributions
        # mctx action_weights should already be normalized, but add defensive normalization
        policy_array = np.array(policy_jax[0])
        policy_sum = np.sum(policy_array)
        if policy_sum > 0:
            policy_array = policy_array / policy_sum
        policy_probs.append(policy_array)
        dones.append(done)

        # Collect price data for prediction loss (Phase 2)
        if collect_prices and hasattr(env, 'prices') and env.prices is not None:
            # Use correct attribute names from VectorBTTradingEnv
            current_idx = env.episode_start_idx + env.current_step
            if current_idx < len(env.prices) - 1:
                prices_list.append(env.prices[current_idx].copy())
                next_prices_list.append(env.prices[current_idx + 1].copy())

        score += reward
        obs_dict = next_obs_dict
        last_info = info  # Keep last info for market baseline metrics
        # BUG_003 FIX: Track truncation status for bootstrap
        was_truncated = truncated

        if done:
            break

    # Extract market baseline metrics from last step
    market_return = last_info.get('market_return', 0.0)
    agent_return = last_info.get('agent_return', 0.0)
    alpha = last_info.get('alpha', 0.0)

    # CRITICAL FIX: Compute returns (MC or GAE) for training
    rewards_arr = np.array(rewards)
    values_arr = np.array(values)  # Original value predictions
    dones_arr = np.array(dones)
    n_steps = len(rewards_arr)

    # BUG_003 FIX: Bootstrap from V(s_T) when truncated (time limit reached)
    # If terminated (true end), V(s_T) = 0. If truncated, estimate V(s_T).
    if was_truncated and n_steps > 0:
        # Get value estimate for final state (obs_dict is next_obs_dict after break)
        final_obs = obs_dict['observation']
        if hasattr(final_obs, 'cpu'):
            final_obs = final_obs.cpu().numpy()
        final_obs = np.expand_dims(final_obs, axis=0)
        final_obs_jax = jnp.array(final_obs, dtype=jnp.float32)

        # Run inference to get V(s_T)
        output = model_apply_fn({'params': params}, final_obs_jax, mode='initial_inference')
        support = get_value_support()
        value_probs = jax.nn.softmax(output['value'][0])
        bootstrap_value = float(jnp.sum(value_probs * support))
    else:
        bootstrap_value = 0.0

    # TASK_022: Use unified return computation (MC or GAE)
    returns = compute_returns(
        rewards=rewards_arr,
        values=values_arr,
        dones=dones_arr,
        bootstrap_value=bootstrap_value,
        discount=discount,
        use_gae=use_gae,
        gae_lambda=gae_lambda,
    )

    # Compute horizon-specific returns for DS24 tracking (value prediction error analysis)
    horizon_returns = {}
    for h in [1, 5, 10]:
        h_returns = np.zeros(n_steps)
        for t in range(n_steps):
            # k-step discounted return from t
            end_idx = min(t + h, n_steps)
            running = 0.0
            for i in range(end_idx - 1, t - 1, -1):
                running = rewards_arr[i] + discount * running
            h_returns[t] = running
        horizon_returns[h] = h_returns

    result = {
        'observations': np.array(observations),
        'actions': np.array(actions),
        'rewards': rewards_arr,
        'values': returns,  # TASK_022: MC or GAE returns
        'policy_probs': np.array(policy_probs),
        'dones': dones_arr,
        'mcts_root_values': np.array(mcts_root_values),  # For DS24 planning metrics
        'horizon_returns': horizon_returns,  # For DS24 horizon value error analysis
        # Market baseline metrics
        'market_return': market_return,
        'agent_return': agent_return,
        'alpha': alpha,
    }

    # Include price data if collected (Phase 2)
    if collect_prices and len(prices_list) > 0:
        result['prices'] = np.array(prices_list)
        result['next_prices'] = np.array(next_prices_list)

    # Include uncertainty samples if collected (Phase 3)
    if len(uncertainty_samples) > 0:
        result['uncertainty'] = np.array(uncertainty_samples)

    # TASK_125: Episode completion callback for Causal Expert Imprinting
    # Called AFTER episode ends to add trajectory to expert buffer
    # The hindsight-optimal path is computed in the callback (causal validity)
    if on_episode_complete is not None:
        on_episode_complete(result, score)

    return result, score


def run_self_play_batched(
    envs: List,
    search_fn: Callable,
    model_apply_fn: Callable,
    params: Dict,
    rng_key: jnp.ndarray,
    steps: int = 200,
    confidence_threshold: float = 0.0,
    support_size: int = VALUE_SUPPORT_SIZE,
    collect_prices: bool = False,
    discount: float = 0.997,
    popart_state: Optional[PopArtState] = None,
    use_gae: bool = False,
    gae_lambda: float = 0.95,
    start_offsets: Optional[List[int]] = None,
    on_episode_complete: Optional[Callable[[Dict, float], None]] = None,  # TASK_125
) -> Tuple[List[Dict], List[float]]:
    """Batched self-play across N parallel environments.

    Achieves near-linear speedup by batching MCTS neural network calls across
    all environments. Each env is independent, preserving causality.

    Args:
        envs: List of N environment instances
        search_fn: MCTS search function
        model_apply_fn: Model apply function
        params: Model parameters
        rng_key: JAX RNG key
        steps: Number of steps per episode
        confidence_threshold: Min policy probability to execute action (else HOLD)
        support_size: Value support size
        collect_prices: Whether to collect price data
        discount: Discount factor for returns
        start_offsets: Optional list of start indices for each env (TASK_064).
                       If provided, enables smart allocation with distributed tiling.
        on_episode_complete: TASK_125 callback for Causal Expert Imprinting.
                             Called per-env with (trajectory_dict, episode_score).

    Returns:
        Tuple of (list of trajectory dicts, list of episode scores)
    """
    n_envs = len(envs)

    # Reset all envs with optional smart allocation offsets (TASK_064)
    obs_dicts = []
    for i, env in enumerate(envs):
        offset = start_offsets[i] if start_offsets else None
        obs_dict, _ = env.reset(start_offset=offset)
        obs_dicts.append(obs_dict)

    # Per-env trajectory storage
    all_observations = [[] for _ in range(n_envs)]
    all_actions = [[] for _ in range(n_envs)]
    all_rewards = [[] for _ in range(n_envs)]
    all_values = [[] for _ in range(n_envs)]
    all_policy_probs = [[] for _ in range(n_envs)]
    all_dones = [[] for _ in range(n_envs)]
    all_mcts_root_values = [[] for _ in range(n_envs)]  # For DS24 planning metrics
    all_prices = [[] for _ in range(n_envs)]
    all_next_prices = [[] for _ in range(n_envs)]
    all_last_info = [{} for _ in range(n_envs)]  # Track last info for market baseline

    scores = [0.0] * n_envs
    active = [True] * n_envs  # Track which envs are still running
    rng = rng_key

    for step in range(steps):
        # Count active envs
        n_active = sum(active)
        if n_active == 0:
            break

        rng, step_rng = jax.random.split(rng)

        # BUG_004 FIX: Always use fixed batch size to avoid recompilation
        # Stack ALL env observations (active and inactive), use last obs for inactive
        active_indices = [i for i in range(n_envs) if active[i]]
        obs_batch = []
        mask_batch = []

        for i in range(n_envs):
            if active[i]:
                obs_array = obs_dicts[i]['observation']
                if hasattr(obs_array, 'cpu'):
                    obs_array = obs_array.cpu().numpy()
                obs_batch.append(obs_array)
                mask_batch.append(obs_dicts[i]['action_mask'])
            else:
                # Use dummy observation for inactive env (reuse last)
                # These results will be ignored anyway
                obs_batch.append(obs_batch[-1] if obs_batch else np.zeros_like(obs_dicts[0]['observation']))
                mask_batch.append(mask_batch[-1] if mask_batch else np.zeros_like(obs_dicts[0]['action_mask']))

        obs_batch = jnp.array(np.stack(obs_batch), dtype=jnp.float32)
        mask_batch = jnp.array(np.stack(mask_batch), dtype=jnp.int8)

        # Batched MCTS - single GPU call for all envs! (fixed batch size)
        actions_jax, policies_jax, stats = search_fn(params, step_rng, obs_batch, mask_batch)

        # Get values
        if 'root_value' in stats:
            values_batch = stats['root_value']
        else:
            output = model_apply_fn({'params': params}, obs_batch, mode='initial_inference')
            support = get_value_support()
            value_probs = jax.nn.softmax(output['value'], axis=-1)
            values_batch = jnp.sum(value_probs * support, axis=-1)

        # G1 FIX (TASK_107): Batch convert JAX arrays to numpy ONCE to avoid 60,000+ CPU-GPU sync events
        # This reduces sync events from (n_envs * steps * 3) to just 4 per step
        # Gemini analysis: GPU utilization 35% -> 80% expected improvement
        actions_np = np.array(actions_jax)
        policies_np = np.array(policies_jax)
        values_np = np.array(values_batch)
        obs_np = np.array(obs_batch)

        # Step each active env (this is still serial but env.step is fast ~2ms)
        # BUG_004 FIX: Now batch uses env_idx directly (not batch_idx) since all envs in batch
        for env_idx in active_indices:
            action_int = int(actions_np[env_idx])

            # Confidence thresholding
            # TASK_034 FIX: Also update policy when forcing HOLD to avoid action-policy mismatch
            policy_to_store = policies_np[env_idx].copy()
            if confidence_threshold > 0:
                probs = policy_to_store
                max_prob = np.max(probs)
                if max_prob < confidence_threshold:
                    asset_idx = action_int // N_ACTIONS_PER_ASSET
                    hold_action = (asset_idx * N_ACTIONS_PER_ASSET) + 1
                    action_int = hold_action

                    # Create synthetic confident HOLD policy to match forced action
                    policy_to_store = np.ones_like(probs) * (0.1 / (len(probs) - 1))
                    policy_to_store[hold_action] = 0.9

            next_obs_dict, reward, terminated, truncated, info = envs[env_idx].step(action_int)
            done = terminated or truncated

            # Store trajectory data (using pre-converted numpy arrays - G1 FIX)
            all_observations[env_idx].append(obs_np[env_idx])
            all_actions[env_idx].append(action_int)
            all_rewards[env_idx].append(reward)
            all_values[env_idx].append(float(values_np[env_idx]))
            all_policy_probs[env_idx].append(policy_to_store)
            all_dones[env_idx].append(done)
            # TASK_031: Denormalize MCTS root values for meaningful diagnostics
            raw_value = float(values_np[env_idx])
            denorm_value = denormalize_value(raw_value, popart_state)
            all_mcts_root_values[env_idx].append(denorm_value)

            # Collect price data
            if collect_prices and hasattr(envs[env_idx], 'prices') and envs[env_idx].prices is not None:
                current_idx = envs[env_idx].episode_start_idx + envs[env_idx].current_step
                if current_idx < len(envs[env_idx].prices) - 1:
                    all_prices[env_idx].append(envs[env_idx].prices[current_idx].copy())
                    all_next_prices[env_idx].append(envs[env_idx].prices[current_idx + 1].copy())

            scores[env_idx] += reward
            obs_dicts[env_idx] = next_obs_dict
            all_last_info[env_idx] = info  # Track last info for market baseline

            if done:
                active[env_idx] = False

    # Compute returns and package results for each env
    results = []

    for i in range(n_envs):
        rewards_arr = np.array(all_rewards[i])
        values_arr = np.array(all_values[i])  # Original value predictions
        dones_arr = np.array(all_dones[i])
        n_steps = len(rewards_arr)

        # TASK_022: Use unified return computation (MC or GAE)
        returns = compute_returns(
            rewards=rewards_arr,
            values=values_arr,
            dones=dones_arr,
            bootstrap_value=0.0,  # Non-JAX env doesn't have bootstrap handling yet
            discount=discount,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
        )

        # Compute horizon-specific returns for DS24 tracking
        horizon_returns = {}
        for h in [1, 5, 10]:
            h_returns = np.zeros(n_steps)
            for t in range(n_steps):
                end_idx = min(t + h, n_steps)
                running = 0.0
                for j in range(end_idx - 1, t - 1, -1):
                    running = rewards_arr[j] + discount * running
                h_returns[t] = running
            horizon_returns[h] = h_returns

        # Extract market baseline metrics from last info
        last_info = all_last_info[i]
        market_return = last_info.get('market_return', 0.0)
        agent_return = last_info.get('agent_return', 0.0)
        alpha = last_info.get('alpha', 0.0)

        result = {
            'observations': np.array(all_observations[i]),
            'actions': np.array(all_actions[i]),
            'rewards': rewards_arr,
            'values': returns,  # TASK_022: MC or GAE returns
            'policy_probs': np.array(all_policy_probs[i]),
            'dones': dones_arr,
            'mcts_root_values': np.array(all_mcts_root_values[i]),  # For DS24 planning metrics
            'horizon_returns': horizon_returns,  # For DS24 horizon value error analysis
            # Market baseline metrics
            'market_return': market_return,
            'agent_return': agent_return,
            'alpha': alpha,
        }

        if collect_prices and len(all_prices[i]) > 0:
            result['prices'] = np.array(all_prices[i])
            result['next_prices'] = np.array(all_next_prices[i])

        # TASK_125: Episode completion callback for Causal Expert Imprinting
        if on_episode_complete is not None:
            on_episode_complete(result, scores[i])

        results.append(result)

    return results, scores


def run_self_play_jax(
    jax_env_states,  # TradingState from JAX env (batched)
    prices: jnp.ndarray,  # Price array (n_steps, n_assets)
    jax_env_config,  # EnvConfig
    search_fn: Callable,
    model_apply_fn: Callable,
    params: Dict,
    rng_key: jnp.ndarray,
    steps: int = 200,
    confidence_threshold: float = 0.0,
    support_size: int = VALUE_SUPPORT_SIZE,
    collect_prices: bool = False,
    discount: float = 0.997,
    popart_state: Optional[PopArtState] = None,
    use_gae: bool = False,
    gae_lambda: float = 0.95,
    max_positions: int = 16,
    min_cash_threshold: float = 100.0,
    on_episode_complete: Optional[Callable[[Dict, float], None]] = None,  # TASK_125
    vector_action: bool = False,  # TASK_828 Stage 3 (captured STATICALLY at trace)
) -> Tuple[List[Dict], List[float]]:
    """Fully GPU-accelerated self-play using lax.scan to eliminate Python loop overhead.

    This fuses the entire episode (200-400 steps) into a single GPU kernel launch,
    maximizing GPU utilization on high-end GPUs like RTX 3090.

    Args:
        jax_env_states: Batched TradingState from jit_batch_reset
        prices: Price array (n_steps, n_assets) as JAX array
        jax_env_config: JAX EnvConfig
        search_fn: MCTS search function
        model_apply_fn: Model apply function
        params: Model parameters
        rng_key: JAX RNG key
        steps: Number of steps per episode
        confidence_threshold: MCTS confidence threshold for forcing HOLD
        support_size: Value support size
        discount: Discount factor for returns
        popart_state: PopArt state for denormalizing MCTS root values (or None to skip)
        on_episode_complete: TASK_125 callback for Causal Expert Imprinting.
                             Called per-env with (trajectory_dict, episode_score).

    Returns:
        Tuple of (list of trajectory dicts, list of episode scores)
    """
    from mcts_trader.env.jax_trading_env import (
        jit_batch_step, jit_batch_step_unique,
        jit_batch_step_vector, jit_batch_step_vector_unique,  # TASK_828 Stage 3
        compute_observation_27ch,
        compute_portfolio_value, get_current_prices,
        jit_batch_compute_action_mask,
    )
    from mcts_trader.env.jax_encoder import (
        EncoderConfig as FullEncoderConfig,
        jit_batch_encode_simple,
        batch_encode_simple,
    )

    n_envs = jax_env_states.cash.shape[0]
    n_assets = jax_env_config.n_assets
    n_actions = n_assets * N_ACTIONS_PER_ASSET

    # Detect unique prices mode (3D = per-env, 2D = shared)
    unique_prices_mode = prices.ndim == 3
    env_indices = jnp.arange(n_envs) if unique_prices_mode else None

    # Create encoder config
    encoder_config = FullEncoderConfig(
        n_assets=n_assets,
        window_size=jax_env_config.window_size,
        n_channels=27,
    )

    # Define the scan-compatible step function
    # This runs entirely on GPU without Python loop overhead
    def episode_step(carry, _):
        """Single episode step for lax.scan - runs on GPU."""
        state, key = carry

        # A. Observation & Mask Encoding (GPU)
        # TASK_121: Fix to use LOOKBACK window (not forward window)
        # TASK_124: Fix same-step observation leak - window ends at t-1, not t
        # price_idx: index of current trading price (we trade at this price)
        # window_starts: start of lookback window (we observe up to price BEFORE trading)
        price_idx = state.episode_start_idx + state.current_step
        # Window should be [price_idx - window_size, price_idx - 1] so we DON'T see execution price
        window_starts = price_idx - encoder_config.window_size  # TASK_124: Removed + 1 to fix same-step leak

        # Handle unique prices mode (3D) vs shared prices (2D)
        if unique_prices_mode:
            # Unique prices: prices[env_idx, step_idx, asset_idx]
            # For encoder, extract per-env price windows
            obs = jit_batch_encode_simple(prices, window_starts, encoder_config, per_env=True)
            current_prices = prices[env_indices, price_idx]
        else:
            # Shared prices: prices[step_idx, asset_idx]
            obs = jit_batch_encode_simple(prices, window_starts, encoder_config)
            current_prices = prices[price_idx]

        mask = jit_batch_compute_action_mask(
            state, current_prices, jax_env_config,
            max_positions=max_positions, min_cash_threshold=min_cash_threshold
        )

        # B. Neural MCTS Search (GPU)
        key, step_rng = jax.random.split(key)
        if vector_action:
            # TASK_828 Stage 3: Sampled-MuZero vector search. Derive the v1 book
            # (HOLD=1 everywhere per DEC-3 HOLD-maintains-position); pass the same
            # asset-major legal mask; deterministic=False (exploration in self-play).
            book = jnp.ones((n_envs, n_assets), dtype=jnp.int32)
            action_vec, sampled_vecs, visit_probs, stats = search_fn(
                params, step_rng, obs, mask, book)
            # C. Value Extraction (root_value is always present in stats)
            values = stats['root_value']
        else:
            actions, policies, stats = search_fn(params, step_rng, obs, mask)

            # C. Value Extraction (root_value is always present in stats)
            values = stats['root_value']

            # D. Confidence Gating (GPU-native via jnp.where)
            # Note: confidence_threshold is captured as static value at trace time.
            # Scalar HOLD-force (actions//5) has no vector analogue → guarded off.
            if confidence_threshold > 0:
                max_probs = jnp.max(policies, axis=-1)
                low_conf = max_probs < confidence_threshold
                hold_actions = (actions // N_ACTIONS_PER_ASSET) * N_ACTIONS_PER_ASSET + 1
                actions = jnp.where(low_conf, hold_actions, actions)

                # Update policy to match forced HOLD
                base_p = 0.1 / (n_actions - 1)
                synth_p = jnp.ones_like(policies) * base_p
                synth_p = synth_p.at[jnp.arange(n_envs), hold_actions].set(0.9)
                policies = jnp.where(low_conf[:, None], synth_p, policies)

        # E. Environment Step (GPU)
        if vector_action:
            if unique_prices_mode:
                new_state, _, rewards, dones = jit_batch_step_vector_unique(state, action_vec, prices, jax_env_config)
            else:
                new_state, _, rewards, dones = jit_batch_step_vector(state, action_vec, prices, jax_env_config)
        else:
            if unique_prices_mode:
                new_state, _, rewards, dones = jit_batch_step_unique(state, actions, prices, jax_env_config)
            else:
                new_state, _, rewards, dones = jit_batch_step(state, actions, prices, jax_env_config)

        # Pack step data for accumulation
        # NOTE: Don't store observations (4.4GB!) - store window_starts instead
        # and reconstruct observations on CPU
        if vector_action:
            # Factored policy target = per-asset visit MARGINALS (D-1): marginalize
            # HERE (do NOT emit sampled_vecs/visit_probs to the buffer).
            # policy_marginals[b,i,a] = Σ_k visit_probs[b,k]·1{sampled_vecs[b,k,i]==a}
            # → (B, n_assets, A) asset-major; each asset-row sums to 1.
            onehot = jax.nn.one_hot(sampled_vecs, N_ACTIONS_PER_ASSET, dtype=jnp.float32)  # (B,K,n_assets,A)
            policy_marginals = jnp.einsum('bk,bkna->bna', visit_probs, onehot)             # (B,n_assets,A)
            step_data = {
                'window_starts': window_starts,       # (n_envs,) int32 - tiny!
                'action_vecs': action_vec,            # (n_envs, n_assets)
                'rewards': rewards,
                'values': values,
                'policy_marginals': policy_marginals.astype(jnp.float16),
                'dones': dones,
            }
        else:
            step_data = {
                'window_starts': window_starts,  # (n_envs,) int32 - tiny!
                'actions': actions,
                'rewards': rewards,
                'values': values,
                'policy_probs': policies.astype(jnp.float16),
                'dones': dones,
            }

        return (new_state, key), step_data

    # Execute fused episode loop (ONE kernel launch for entire episode!)
    # This eliminates ~200 Python iterations and PCIe roundtrips
    (final_states, _), trajectory_batch = jax.lax.scan(
        episode_step, (jax_env_states, rng_key), None, length=steps
    )

    # TASK_128 FIX: Ensure GPU kernel completes before starting CPU work
    # This prevents potential deadlock when starting new JIT compilation
    # while previous GPU operations are still queued
    jax.block_until_ready(trajectory_batch['rewards'])

    # Single device-to-host transfer (massive PCIe efficiency gain)
    # Shape: (steps, n_envs, ...) -> (n_envs, steps, ...)
    def fetch(x):
        return np.array(jnp.swapaxes(x, 0, 1))

    # Fetch small arrays from GPU
    cpu_window_starts = fetch(trajectory_batch['window_starts'])  # (n_envs, steps) - tiny
    cpu_rewards = fetch(trajectory_batch['rewards'])
    cpu_values = fetch(trajectory_batch['values'])
    cpu_dones = fetch(trajectory_batch['dones'])
    if vector_action:
        # TASK_828 Stage 3: per-asset action VECTORS + factored policy marginals.
        cpu_action_vecs = fetch(trajectory_batch['action_vecs'])          # (n_envs, steps, n_assets)
        cpu_policy_marginals = fetch(trajectory_batch['policy_marginals'])  # (n_envs, steps, n_assets, A)
    else:
        cpu_actions = fetch(trajectory_batch['actions'])
        cpu_policies = fetch(trajectory_batch['policy_probs'])
    cpu_prices = np.array(prices)  # Fetch once

    # Observations will be reconstructed lazily per-env by the caller
    # This prevents allocating ~10GB of RAM for 15 envs * 1440 steps * (27*64*64) floats

    # Compute scores from accumulated rewards (TASK_126: mask post-termination)
    # BUG FIX: Previously summed ALL steps including post-done, causing -30.99 spikes
    scores = []
    for i in range(n_envs):
        dones_i = cpu_dones[i]
        if np.any(dones_i):
            n_valid = int(np.argmax(dones_i) + 1)  # Include the terminal step
        else:
            n_valid = steps
        scores.append(np.sum(cpu_rewards[i, :n_valid]))
    scores = np.array(scores)

    # CPU packaging & return computation
    results = []

    for i in range(n_envs):
        # Determine actual sequence length (first done or full episode)
        dones_i = cpu_dones[i]
        if np.any(dones_i):
            n_steps_i = int(np.argmax(dones_i) + 1)
        else:
            n_steps_i = steps

        if n_steps_i == 0:
            continue

        # Slice to actual trajectory length
        rewards_arr = cpu_rewards[i, :n_steps_i]
        values_arr = cpu_values[i, :n_steps_i]
        dones_arr = cpu_dones[i, :n_steps_i]
        
        # Store compressed observation data (lazy reconstruction)
        # Caller must reconstruct using reconstruct_observations()
        env_window_starts = cpu_window_starts[i, :n_steps_i]
        
        # BUG_003 NOTE: For JAX env, we approximate truncation as hitting step limit
        was_truncated = (n_steps_i == steps) and not dones_arr[-1]

        if was_truncated:
            bootstrap_value = float(values_arr[-1])
        else:
            bootstrap_value = 0.0

        # TASK_022: Use unified return computation (MC or GAE)
        returns = compute_returns(
            rewards=rewards_arr,
            values=values_arr,
            dones=dones_arr,
            bootstrap_value=bootstrap_value,
            discount=discount,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
        )

        # Horizon-specific returns for DS24 (vectorized)
        horizon_returns = {}
        for h in [1, 5, 10]:
            h_returns = np.zeros(n_steps_i, dtype=np.float32)
            disc_powers = discount ** np.arange(h)
            for t in range(n_steps_i):
                end_idx = min(t + h, n_steps_i)
                window = rewards_arr[t:end_idx]
                h_returns[t] = np.dot(window, disc_powers[:len(window)])
            horizon_returns[h] = h_returns

        # Get final portfolio metrics
        final_return = float(final_states.cumulative_return[i]) if n_steps_i > 0 else 0.0

        # Compute market return (buy & hold baseline)
        start_idx = int(jax_env_states.episode_start_idx[i])
        end_idx = start_idx + n_steps_i
        if end_idx < len(cpu_prices) and n_steps_i > 0:
            start_prices = cpu_prices[start_idx]
            end_prices = cpu_prices[end_idx - 1]
            asset_returns = (end_prices - start_prices) / (start_prices + 1e-8)
            market_return = float(np.mean(asset_returns)) * 100

            # TASK_126: Compute episode optimal_alpha on SAME data as market_return
            # This fixes the data source mismatch where trainer computed on batch data
            episode_prices = cpu_prices[start_idx:end_idx]
            if len(episode_prices) > 1:
                # Compute log returns for path solver (consistent with trainer)
                log_prices = np.log(episode_prices + 1e-8)
                log_returns = log_prices[1:] - log_prices[:-1]  # (T-1, n_assets)
                # Compute optimal path with no transaction cost (theoretical max)
                try:
                    optimal_path = compute_optimal_path(
                        jnp.array(log_returns), k=4, method='greedy', transaction_cost=0.0
                    )
                    episode_optimal_alpha = float(optimal_path.total_alpha) * 100.0  # Scale to %
                except Exception:
                    episode_optimal_alpha = 0.0
            else:
                episode_optimal_alpha = 0.0
        else:
            market_return = 0.0
            episode_optimal_alpha = 0.0

        # Denormalize MCTS root values for DS24
        mcts_root_values = values_arr.copy()
        if popart_state is not None:
            std = float(jnp.sqrt(popart_state.var + 1e-8))
            mean = float(popart_state.mean)
            mcts_root_values = mcts_root_values * std + mean

        # Package result (fp16 -> fp32 for trainer compatibility)
        result = {
            # 'observations': REMOVED (Lazy reconstruction)
            'window_starts': env_window_starts, # Store for reconstruction
            'rewards': rewards_arr,
            'values': returns,  # TASK_022: MC or GAE returns
            'dones': dones_arr,
            'mcts_root_values': mcts_root_values,
            'horizon_returns': horizon_returns,
            'market_return': market_return,
            'agent_return': final_return,
            'alpha': final_return - market_return,
            'episode_optimal_alpha': episode_optimal_alpha,  # TASK_126: Oracle on same data
        }

        # TASK_828 Stage 3: emit vector keys (action_vecs / policy_marginals) in
        # vector mode; the scalar path keeps actions / policy_probs (byte-identical).
        if vector_action:
            result['action_vecs'] = cpu_action_vecs[i, :n_steps_i]
            result['policy_marginals'] = cpu_policy_marginals[i, :n_steps_i].astype(np.float32)
        else:
            result['actions'] = cpu_actions[i, :n_steps_i]
            result['policy_probs'] = cpu_policies[i, :n_steps_i].astype(np.float32)

        # Add prices for prediction loss and DS24 physics logging
        if collect_prices and n_steps_i > 0 and end_idx <= len(cpu_prices):
            result['prices'] = cpu_prices[start_idx:end_idx]
            next_end = min(end_idx + 1, len(cpu_prices))
            result['next_prices'] = cpu_prices[start_idx + 1:next_end]
            if len(result['next_prices']) < n_steps_i:
                result['next_prices'] = np.concatenate([
                    result['next_prices'],
                    cpu_prices[next_end - 1:next_end]
                ])
        
        # Always pass full prices reference for observation reconstruction
        # window_starts are absolute indices into the FULL price array,
        # so reconstruction always needs the full array (not the sliced 'prices')
        result['_full_prices_ref'] = cpu_prices

        # TASK_125: Episode completion callback for Causal Expert Imprinting
        if on_episode_complete is not None:
            on_episode_complete(result, scores[i])

        results.append(result)

    return results, list(scores)


def reconstruct_observations(
    trajectory: Dict,
    n_assets: int = 64,
    window_size: int = 60,
    n_channels: int = 27,
) -> np.ndarray:
    """
    Lazy reconstruction of observations from compressed trajectory data.
    
    Used to avoid massive RAM spikes during self-play data collection.
    
    Args:
        trajectory: Trajectory dictionary with 'window_starts' and prices data
        n_assets: Number of assets
        window_size: Lookback window size
        n_channels: Number of encoder channels
        
    Returns:
        observations: (T, C, N, N) numpy array
    """
    from mcts_trader.env.jax_encoder import (
        EncoderConfig, batch_encode_simple
    )
    
    # Get prices source
    if '_full_prices_ref' in trajectory:
        prices = trajectory['_full_prices_ref']
        window_starts = trajectory['window_starts']
    elif 'prices' in trajectory:
        # 'prices' is a slice, but window_starts are absolute indices into the FULL array.
        # This shouldn't happen as run_self_play_jax now always adds _full_prices_ref.
        raise ValueError("Cannot reconstruct: missing full price reference (ensure _full_prices_ref is present)")
    else:
        raise ValueError("Cannot reconstruct: no price data found")
        
    
    # Create config
    config = EncoderConfig(
        n_assets=n_assets,
        window_size=window_size,
        n_channels=n_channels,
    )
    
    # Reconstruct on GPU -> CPU
    # This copies only ONE trajectory's observations (600MB vs 10GB)
    obs = np.array(batch_encode_simple(
        jnp.array(prices), jnp.array(window_starts), config
    ))
    
    return obs.astype(np.float32)
