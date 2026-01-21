"""
Self-play episode generation for MuZero training.

This module contains three variants of self-play:
- run_self_play: Single-environment episode generation
- run_self_play_batched: Batched Python (VectorBT) environments
- run_self_play_jax: Fully GPU-accelerated JAX environments
"""
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any

from mcts_trader.jax_agent import VALUE_SUPPORT_SIZE, get_value_support, get_value_support_np
from mcts_trader.jax_agent.popart import PopArtState
from mcts_trader.jax_agent.returns import compute_returns


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
        policy_probs.append(np.array(policy_jax[0]))
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

        # Step each active env (this is still serial but env.step is fast ~2ms)
        # BUG_004 FIX: Now batch uses env_idx directly (not batch_idx) since all envs in batch
        for env_idx in active_indices:
            action_int = int(actions_jax[env_idx])

            # Confidence thresholding
            # TASK_034 FIX: Also update policy when forcing HOLD to avoid action-policy mismatch
            policy_to_store = np.array(policies_jax[env_idx])
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

            # Store trajectory data
            all_observations[env_idx].append(obs_batch[env_idx])
            all_actions[env_idx].append(action_int)
            all_rewards[env_idx].append(reward)
            all_values[env_idx].append(float(values_batch[env_idx]))
            all_policy_probs[env_idx].append(policy_to_store)
            all_dones[env_idx].append(done)
            # TASK_031: Denormalize MCTS root values for meaningful diagnostics
            raw_value = float(values_batch[env_idx])
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
    discount: float = 0.997,
    popart_state: Optional[PopArtState] = None,
    use_gae: bool = False,
    gae_lambda: float = 0.95,
) -> Tuple[List[Dict], List[float]]:
    """Fully GPU-accelerated self-play using JAX environment.

    This replaces run_self_play_batched with a JAX-native implementation
    that keeps all computation on GPU.

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

    Returns:
        Tuple of (list of trajectory dicts, list of episode scores)
    """
    from mcts_trader.env.jax_trading_env import (
        jit_batch_step, compute_observation_27ch,
        compute_portfolio_value, get_current_prices,
        jit_batch_compute_action_mask,
    )
    from mcts_trader.env.jax_encoder import (
        EncoderConfig as FullEncoderConfig,
        jit_batch_encode_simple,
    )

    n_envs = jax_env_states.cash.shape[0]
    n_assets = jax_env_config.n_assets
    n_actions = n_assets * N_ACTIONS_PER_ASSET

    # Create encoder config
    encoder_config = FullEncoderConfig(
        n_assets=n_assets,
        window_size=jax_env_config.window_size,
        n_channels=27,
    )

    # Storage for trajectories - PRE-ALLOCATED arrays (TASK_028 Phase 1)
    # This eliminates 20,000+ list.append() calls and their memory fragmentation
    # Using fp16 for large arrays (obs, policies) to halve bandwidth
    all_observations = np.zeros((n_envs, steps, 27, n_assets, n_assets), dtype=np.float16)
    all_actions = np.zeros((n_envs, steps), dtype=np.int32)
    all_rewards = np.zeros((n_envs, steps), dtype=np.float32)
    all_values = np.zeros((n_envs, steps), dtype=np.float32)
    all_policy_probs = np.zeros((n_envs, steps, n_actions), dtype=np.float16)
    all_dones = np.zeros((n_envs, steps), dtype=np.bool_)
    all_mcts_root_values = np.zeros((n_envs, steps), dtype=np.float32)
    step_counts = np.zeros(n_envs, dtype=np.int32)  # Track actual steps per env

    scores = np.zeros(n_envs, dtype=np.float32)
    active = np.ones(n_envs, dtype=bool)  # Track active envs
    states = jax_env_states
    rng = rng_key

    # TASK_028 Phase 2: Batched GPU transfers
    # Accumulate JAX arrays on GPU, transfer in batches every BATCH_STEPS
    BATCH_STEPS = 50  # Transfer every 50 steps (4 transfers instead of 200)
    gpu_obs_buffer = []
    gpu_actions_buffer = []
    gpu_rewards_buffer = []
    gpu_values_buffer = []
    gpu_policies_buffer = []
    gpu_dones_buffer = []
    batch_start_step = 0

    def flush_gpu_buffers(batch_start, gpu_obs, gpu_actions, gpu_rewards,
                          gpu_values, gpu_policies, gpu_dones, active_states):
        """Transfer accumulated GPU buffers to CPU storage.

        WARNING: This function captures mutable closures (all_observations, etc).
        Do NOT JIT this function - it relies on external array mutation.
        """
        nonlocal all_observations, all_actions, all_rewards, all_values
        nonlocal all_policy_probs, all_dones, all_mcts_root_values

        if not gpu_obs:
            return

        # Stack on GPU (fast) then single transfer
        obs_stacked = jnp.stack(gpu_obs, axis=1)  # (n_envs, batch_steps, 27, n_assets, n_assets)
        actions_stacked = jnp.stack(gpu_actions, axis=1)  # (n_envs, batch_steps)
        rewards_stacked = jnp.stack(gpu_rewards, axis=1)
        values_stacked = jnp.stack(gpu_values, axis=1)
        policies_stacked = jnp.stack(gpu_policies, axis=1)
        dones_stacked = jnp.stack(gpu_dones, axis=1)

        # Single batch transfer (much fewer PCIe roundtrips)
        obs_np = np.array(obs_stacked, dtype=np.float16)
        actions_np = np.array(actions_stacked, dtype=np.int32)
        rewards_np = np.array(rewards_stacked, dtype=np.float32)
        values_np = np.array(values_stacked, dtype=np.float32)
        policies_np = np.array(policies_stacked, dtype=np.float16)
        dones_np = np.array(dones_stacked, dtype=np.bool_)

        batch_len = obs_np.shape[1]
        end_step = batch_start + batch_len

        # TASK_031: Denormalize MCTS root values for meaningful diagnostics
        if popart_state is not None:
            std = float(jnp.sqrt(popart_state.var + 1e-8))
            mean = float(popart_state.mean)
            values_denorm = values_np * std + mean
        else:
            values_denorm = values_np

        # Copy to pre-allocated arrays
        all_observations[:, batch_start:end_step] = obs_np
        all_actions[:, batch_start:end_step] = actions_np
        all_rewards[:, batch_start:end_step] = rewards_np
        all_values[:, batch_start:end_step] = values_np
        all_policy_probs[:, batch_start:end_step] = policies_np
        all_dones[:, batch_start:end_step] = dones_np
        all_mcts_root_values[:, batch_start:end_step] = values_denorm  # Use denormalized values

    for step in range(steps):
        n_active = int(np.sum(active))
        if n_active == 0:
            break

        rng, step_rng = jax.random.split(rng)

        # Get window starts for observation encoding
        window_starts = states.episode_start_idx + states.current_step

        # Encode observations using 27-channel encoder (batched)
        obs_batch = jit_batch_encode_simple(
            prices, window_starts, encoder_config
        )  # (n_envs, 27, n_assets, n_assets)

        # Get current prices for each env (for action masking)
        current_step_indices = states.episode_start_idx + states.current_step
        current_prices_batch = prices[current_step_indices]  # (n_envs, n_assets)

        # Compute proper action mask based on portfolio state (matches VectorBT behavior)
        mask_batch = jit_batch_compute_action_mask(
            states, current_prices_batch, jax_env_config,
            max_positions=16, min_cash_threshold=100.0
        )

        # Batched MCTS - single GPU call
        actions_jax, policies_jax, stats = search_fn(params, step_rng, obs_batch, mask_batch)

        # Get values
        if 'root_value' in stats:
            values_batch = stats['root_value']
        else:
            output = model_apply_fn({'params': params}, obs_batch, mode='initial_inference')
            support = get_value_support()
            value_probs = jax.nn.softmax(output['value'], axis=-1)
            values_batch = jnp.sum(value_probs * support, axis=-1)

        # Confidence thresholding
        # TASK_034 FIX: Also update policy when forcing HOLD to avoid action-policy mismatch
        if confidence_threshold > 0:
            max_probs = jnp.max(policies_jax, axis=-1)
            low_confidence = max_probs < confidence_threshold
            asset_indices = actions_jax // N_ACTIONS_PER_ASSET
            hold_actions = asset_indices * N_ACTIONS_PER_ASSET + 1
            actions_jax = jnp.where(low_confidence, hold_actions, actions_jax)

            # Create synthetic confident HOLD policy for low-confidence cases
            # 90% on HOLD action, 10% spread across others
            n_actions = policies_jax.shape[-1]
            base_prob = 0.1 / (n_actions - 1)
            synthetic_policies = jnp.ones_like(policies_jax) * base_prob
            # Set 90% probability on the HOLD action for each env
            # Use advanced indexing to set hold_actions probabilities
            env_indices = jnp.arange(policies_jax.shape[0])
            synthetic_policies = synthetic_policies.at[env_indices, hold_actions].set(0.9)
            # Replace policies for low-confidence envs
            policies_jax = jnp.where(low_confidence[:, None], synthetic_policies, policies_jax)

        # Batched JAX step - GPU accelerated!
        new_states, obs_new, rewards, dones = jit_batch_step(
            states, actions_jax, prices, jax_env_config
        )

        # TASK_028 Phase 2: Accumulate on GPU instead of per-step transfer
        gpu_obs_buffer.append(obs_batch)
        gpu_actions_buffer.append(actions_jax)
        gpu_rewards_buffer.append(rewards)
        gpu_values_buffer.append(values_batch)
        gpu_policies_buffer.append(policies_jax)
        gpu_dones_buffer.append(dones)

        # Flush buffer every BATCH_STEPS (4 transfers instead of 200)
        if len(gpu_obs_buffer) >= BATCH_STEPS or step == steps - 1:
            flush_gpu_buffers(batch_start_step, gpu_obs_buffer, gpu_actions_buffer,
                              gpu_rewards_buffer, gpu_values_buffer, gpu_policies_buffer,
                              gpu_dones_buffer, active)
            batch_start_step = step + 1
            gpu_obs_buffer = []
            gpu_actions_buffer = []
            gpu_rewards_buffer = []
            gpu_values_buffer = []
            gpu_policies_buffer = []
            gpu_dones_buffer = []

        # Update step counts and scores (need per-step for done detection)
        # Small transfer for dones check only
        dones_np = np.array(dones, dtype=np.bool_)
        rewards_np = np.array(rewards, dtype=np.float32)
        step_counts[active] += 1
        scores[active] += rewards_np[active]

        # Mark done envs as inactive
        active[dones_np] = False

        states = new_states

    # Compute MC returns and package results
    results = []

    # Convert prices once (not per-env!) - fixes performance bug
    prices_np = np.array(prices)  # (n_steps, n_assets)

    for i in range(n_envs):
        # Use step_counts to slice pre-allocated arrays to actual trajectory length
        n_steps_i = int(step_counts[i])
        if n_steps_i == 0:
            # Empty trajectory, skip
            continue

        # Slice to actual trajectory length (TASK_028: no extra copies)
        rewards_arr = all_rewards[i, :n_steps_i]
        values_arr = all_values[i, :n_steps_i]  # Original value predictions
        dones_arr = all_dones[i, :n_steps_i]

        # BUG_003 NOTE: For JAX env, we approximate truncation as hitting step limit
        # If env completed before max_steps, it was terminated. Otherwise, bootstrap.
        # This is approximate because the env could have terminated exactly at max_steps.
        was_truncated = (n_steps_i >= steps) and dones_arr[-1] if n_steps_i > 0 else False

        if was_truncated:
            # Bootstrap from final value estimate
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

        # Horizon-specific returns for DS24
        horizon_returns = {}
        for h in [1, 5, 10]:
            h_returns = np.zeros(n_steps_i, dtype=np.float32)
            for t in range(n_steps_i):
                end_idx = min(t + h, n_steps_i)
                running = 0.0
                for j in range(end_idx - 1, t - 1, -1):
                    running = rewards_arr[j] + discount * running
                h_returns[t] = running
            horizon_returns[h] = h_returns

        # Get final portfolio metrics
        final_return = float(states.cumulative_return[i]) if n_steps_i > 0 else 0.0

        # Compute market return (buy & hold baseline)
        start_idx = int(jax_env_states.episode_start_idx[i])
        end_idx = start_idx + n_steps_i
        if end_idx < len(prices_np) and n_steps_i > 0:
            start_prices = prices_np[start_idx]
            end_prices = prices_np[end_idx - 1]
            # Average return across all assets (equal weight buy & hold)
            asset_returns = (end_prices - start_prices) / (start_prices + 1e-8)
            market_return = float(np.mean(asset_returns)) * 100  # As percentage
        else:
            market_return = 0.0

        # Slice pre-allocated arrays to actual trajectory length
        # fp16 arrays converted back to fp32 for trainer compatibility
        result = {
            'observations': all_observations[i, :n_steps_i].astype(np.float32),
            'actions': all_actions[i, :n_steps_i],
            'rewards': rewards_arr,
            'values': returns,  # TASK_022: MC or GAE returns
            'policy_probs': all_policy_probs[i, :n_steps_i].astype(np.float32),
            'dones': dones_arr,
            'mcts_root_values': all_mcts_root_values[i, :n_steps_i],
            'horizon_returns': horizon_returns,
            'market_return': market_return,
            'agent_return': final_return,
            'alpha': final_return - market_return,  # True alpha vs market
        }

        # Add prices for prediction loss and DS24 physics logging
        # Use trajectory's actual time indices from the price array
        if n_steps_i > 0 and end_idx <= len(prices_np):
            result['prices'] = prices_np[start_idx:end_idx]
            # next_prices: shifted by 1, with last step using same price
            next_end = min(end_idx + 1, len(prices_np))
            result['next_prices'] = prices_np[start_idx + 1:next_end]
            # Pad if needed to match length
            if len(result['next_prices']) < n_steps_i:
                result['next_prices'] = np.concatenate([
                    result['next_prices'],
                    prices_np[next_end - 1:next_end]  # Repeat last price
                ])

        results.append(result)

    return results, list(scores)
