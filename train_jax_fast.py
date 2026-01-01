import os
import warnings

# =============================================================================
# GPU OPTIMIZATION: Must be set BEFORE importing JAX
# =============================================================================
# XLA Autotuning Level 4: Most thorough optimization (slower first compile, fastest runtime)
# Cache directories persist tuning results across runs
_trader_dir = os.path.dirname(os.path.abspath(__file__))
_xla_cache = os.path.join(_trader_dir, '.xla_cache')
_jax_cache = os.path.join(_trader_dir, '.jax_cache')
os.makedirs(_xla_cache, exist_ok=True)
os.makedirs(_jax_cache, exist_ok=True)

# XLA GPU Autotuning: Level 4 + persistent per-fusion cache
# --xla_gpu_per_fusion_autotune_cache_dir: caches autotune results per fusion, grows across runs
# Ref: https://openxla.org/xla/persisted_autotuning
os.environ['XLA_FLAGS'] = (
    f'--xla_gpu_autotune_level=4 '
    f'--xla_gpu_per_fusion_autotune_cache_dir={_xla_cache}'
)

# CUDA async allocator: reduces memory fragmentation
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Suppress TF/XLA verbose logging (dot_search_space, slow_operation_alarm warnings)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warn, 3=error

# JAX compilation cache (for HLO -> compiled code)
os.environ['JAX_COMPILATION_CACHE_DIR'] = _jax_cache

# Suppress FutureWarning from JAX scatter dtype promotion (bf16)
warnings.filterwarnings('ignore', category=FutureWarning, module='jax')

# Now import JAX (picks up env vars)
import jax

# Enable JAX persistent compilation cache with XLA autotuning integration
jax.config.update("jax_compilation_cache_dir", _jax_cache)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)  # Cache everything
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)  # Cache all compiles
import jax.numpy as jnp
import numpy as np
import time
import logging
import argparse

# Import JAX components
from mcts_trader.jax_agent.muzero_model import MuZeroNetworks
from mcts_trader.jax_agent.trainer import MuZeroTrainer, PredictionLossConfig
from mcts_trader.jax_agent.replay_buffer import JAXReplayBuffer
from mcts_trader.jax_agent import VALUE_SUPPORT_SIZE, get_value_support, get_value_support_np
# NOTE: Not using MCTSWorkerPool - it runs MCTS on CPU (slow!)
# Direct GPU MCTS is ~10× faster

# Import environment - VectorBT-based (clean accounting, no FX bugs)
from mcts_trader.env.vectorbt_env import VectorBTTradingEnv

# Tree Search (Phase 3) - imported conditionally when --use-tree-search is enabled
# from mcts_trader.jax_agent.prediction import (
#     NeuralMarketODE, create_physics_aware_tree_search_fn, TreeSearchConfig
# )

# DS24 Enhanced Metrics Logging - imported conditionally when --ds24-logging is enabled
# from tools.DS24.metrics_logger import DS24MetricsLogger

# Action space: 5 actions per asset [Sell, Hold, Buy, Limit Sell, Limit Buy]
N_ACTIONS_PER_ASSET = 5


def run_self_play(env, search_fn, model_apply_fn, params, rng_key, steps=200, confidence_threshold=0.0, support_size=VALUE_SUPPORT_SIZE, collect_prices=False, risk_adjust_fn=None, pred_params=None, discount=0.997):
    """Self-play with timing. Optionally collects price data for prediction loss and risk-adjusted rewards.

    Args:
        pred_params: Prediction model params for dynamic tree search. If provided, risk_adjust_fn
                     will be called as risk_adjust_fn(pred_params, prices, reward, rng) instead of
                     risk_adjust_fn(prices, reward, rng).
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
        if 'root_value' in stats:
            value_scalar = stats['root_value'][0]
            mcts_root_values.append(float(value_scalar))
        else:
            output = model_apply_fn({'params': params}, obs_jax, mode='initial_inference')
            support = get_value_support()
            value_probs = jax.nn.softmax(output['value'][0])
            value_scalar = jnp.sum(value_probs * support)
            mcts_root_values.append(float(value_scalar))

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
        if confidence_threshold > 0:
            probs = np.array(policy_jax[0])
            max_prob = np.max(probs)
            if max_prob < confidence_threshold:
                # Force HOLD: (asset_idx * N_ACTIONS_PER_ASSET) + 1
                asset_idx = action_int // N_ACTIONS_PER_ASSET
                action_int = (asset_idx * N_ACTIONS_PER_ASSET) + 1

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
                    print(f"⚠️ Tree search error: {e}")
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

        if done:
            break

    # CRITICAL FIX: Compute Monte Carlo returns (sum of future discounted rewards)
    # instead of using model's value predictions (which creates circular dependency!)
    rewards_arr = np.array(rewards)
    n_steps = len(rewards_arr)
    mc_returns = np.zeros(n_steps)
    # discount is passed as parameter now

    # Compute returns from end to start
    running_return = 0.0
    for t in range(n_steps - 1, -1, -1):
        running_return = rewards_arr[t] + discount * running_return
        mc_returns[t] = running_return

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
        'rewards': np.array(rewards),
        'values': mc_returns,  # Full MC returns (not clipped)
        'policy_probs': np.array(policy_probs),
        'dones': np.array(dones),
        'mcts_root_values': np.array(mcts_root_values),  # For DS24 planning metrics
        'horizon_returns': horizon_returns  # For DS24 horizon value error analysis
    }

    # Include price data if collected (Phase 2)
    if collect_prices and len(prices_list) > 0:
        result['prices'] = np.array(prices_list)
        result['next_prices'] = np.array(next_prices_list)

    # Include uncertainty samples if collected (Phase 3)
    if len(uncertainty_samples) > 0:
        result['uncertainty'] = np.array(uncertainty_samples)

    return result, score


def run_self_play_batched(envs, search_fn, model_apply_fn, params, rng_key, steps=200,
                          confidence_threshold=0.0, support_size=VALUE_SUPPORT_SIZE,
                          collect_prices=False, discount=0.997):
    """Batched self-play across N parallel environments.

    Achieves near-linear speedup by batching MCTS neural network calls across
    all environments. Each env is independent, preserving causality.

    Args:
        envs: List of N environment instances
        Other args same as run_self_play

    Returns:
        results: List of N trajectory dicts
        scores: List of N episode scores
    """
    n_envs = len(envs)

    # Reset all envs
    obs_dicts = []
    for env in envs:
        obs_dict, _ = env.reset()
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

    scores = [0.0] * n_envs
    active = [True] * n_envs  # Track which envs are still running
    rng = rng_key

    for step in range(steps):
        # Count active envs
        n_active = sum(active)
        if n_active == 0:
            break

        rng, step_rng = jax.random.split(rng)

        # Stack observations from active envs only
        active_indices = [i for i in range(n_envs) if active[i]]
        obs_batch = []
        mask_batch = []

        for i in active_indices:
            obs_array = obs_dicts[i]['observation']
            if hasattr(obs_array, 'cpu'):
                obs_array = obs_array.cpu().numpy()
            obs_batch.append(obs_array)
            mask_batch.append(obs_dicts[i]['action_mask'])

        obs_batch = jnp.array(np.stack(obs_batch), dtype=jnp.float32)
        mask_batch = jnp.array(np.stack(mask_batch), dtype=jnp.int8)

        # Batched MCTS - single GPU call for all envs!
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
        for batch_idx, env_idx in enumerate(active_indices):
            action_int = int(actions_jax[batch_idx])

            # Confidence thresholding
            if confidence_threshold > 0:
                probs = np.array(policies_jax[batch_idx])
                max_prob = np.max(probs)
                if max_prob < confidence_threshold:
                    asset_idx = action_int // N_ACTIONS_PER_ASSET
                    action_int = (asset_idx * N_ACTIONS_PER_ASSET) + 1  # Force HOLD

            next_obs_dict, reward, terminated, truncated, info = envs[env_idx].step(action_int)
            done = terminated or truncated

            # Store trajectory data
            all_observations[env_idx].append(obs_batch[batch_idx])
            all_actions[env_idx].append(action_int)
            all_rewards[env_idx].append(reward)
            all_values[env_idx].append(float(values_batch[batch_idx]))
            all_policy_probs[env_idx].append(np.array(policies_jax[batch_idx]))
            all_dones[env_idx].append(done)
            all_mcts_root_values[env_idx].append(float(values_batch[batch_idx]))

            # Collect price data
            if collect_prices and hasattr(envs[env_idx], 'prices') and envs[env_idx].prices is not None:
                current_idx = envs[env_idx].episode_start_idx + envs[env_idx].current_step
                if current_idx < len(envs[env_idx].prices) - 1:
                    all_prices[env_idx].append(envs[env_idx].prices[current_idx].copy())
                    all_next_prices[env_idx].append(envs[env_idx].prices[current_idx + 1].copy())

            scores[env_idx] += reward
            obs_dicts[env_idx] = next_obs_dict

            if done:
                active[env_idx] = False

    # Compute MC returns and package results for each env
    results = []
    # discount is passed as parameter now

    for i in range(n_envs):
        rewards_arr = np.array(all_rewards[i])
        n_steps = len(rewards_arr)
        mc_returns = np.zeros(n_steps)

        running_return = 0.0
        for t in range(n_steps - 1, -1, -1):
            running_return = rewards_arr[t] + discount * running_return
            mc_returns[t] = running_return

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

        result = {
            'observations': np.array(all_observations[i]),
            'actions': np.array(all_actions[i]),
            'rewards': rewards_arr,
            'values': mc_returns,
            'policy_probs': np.array(all_policy_probs[i]),
            'dones': np.array(all_dones[i]),
            'mcts_root_values': np.array(all_mcts_root_values[i]),  # For DS24 planning metrics
            'horizon_returns': horizon_returns  # For DS24 horizon value error analysis
        }

        if collect_prices and len(all_prices[i]) > 0:
            result['prices'] = np.array(all_prices[i])
            result['next_prices'] = np.array(all_next_prices[i])

        results.append(result)

    return results, scores


def main():
    parser = argparse.ArgumentParser(description='JAX MuZero Training')
    parser.add_argument('--name', type=str, default='default', help='Experiment name')
    parser.add_argument('--mode', type=str, default='MARGIN', choices=['SPOT', 'MARGIN'], help='Trading mode')
    parser.add_argument('--fee', type=int, default=0, help='Trading fee in bps (0 for synthetic testing)')
    parser.add_argument('--slippage', type=int, default=0, help='Slippage in bps (0 for synthetic testing)')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--steps', type=int, default=400, help='Steps per episode (400=6h, 200=3h)')
    parser.add_argument('--config', type=str, help='Training config JSON (dynamic assets)')
    parser.add_argument('--start-date', type=str, help='Start date filter (YYYY-MM-DD or unix ts)')
    parser.add_argument('--end-date', type=str, help='End date filter (YYYY-MM-DD or unix ts)')
    parser.add_argument('--load-checkpoint', type=str, help='Path to checkpoint to resume from (absolute or relative)')
    
    # MCTS Hyperparameters (Maniac Mode)
    parser.add_argument('--sims', type=int, default=32, help='MCTS simulations (32=better planning, 16=faster)')
    parser.add_argument('--noise', type=float, default=0.3, help='Dirichlet noise alpha (Standard MCTS only)')
    parser.add_argument('--noise-frac', type=float, default=0.25, help='Dirichlet noise fraction (Standard MCTS only)')
    parser.add_argument('--gumbel-scale', type=float, default=3.0, help='Gumbel noise scale (Gumbel MCTS) - higher=more exploration')
    parser.add_argument('--value-scale', type=float, default=1.0, help='Value calibration scale (>1.0 expands prediction range)')
    parser.add_argument('--value-shift', type=float, default=0.0, help='Value calibration shift (adjust prediction center)')
    parser.add_argument('--adaptive-calibration', action='store_true', help='Enable adaptive value calibration (auto-tune scale/shift)')

    # Advanced Training Parameters
    parser.add_argument('--batch-size', type=int, default=256, help='Training batch size')
    parser.add_argument('--parallel-envs', type=int, default=1,
                       help='Number of parallel environments for batched self-play (1=serial, 4=4x throughput)')
    parser.add_argument('--model-size', type=str, default='small', choices=['small', 'medium', 'large'],
                       help='Model size: small (~7.3M), medium (~15M), large (~36M)')
    parser.add_argument('--state-dim', type=int, default=None, help='Model state dimension (overrides --model-size)')
    parser.add_argument('--synthetic-level', type=int, default=None,
                       help='Use synthetic data (0=sine, 1=lag, 2=dip, 3=trends, None=real)')
    parser.add_argument('--synthetic-steps', type=int, default=10000,
                       help='Steps per synthetic episode')
    parser.add_argument('--schedule-path', type=str, default=None,
                       help='Load from training_schedule.json (random period/run selection)')
    parser.add_argument('--db-path', type=str, default=None,
                       help='Load data from SQLite database')
    parser.add_argument('--h-cycles', type=int, default=4, help='TRM Reasoning outer cycles (H)')
    parser.add_argument('--l-cycles', type=int, default=6, help='TRM Reasoning inner cycles (L)')
    parser.add_argument('--confidence-threshold', type=float, default=0.0, help='Minimum policy confidence to execute trade (force Hold otherwise)')
    parser.add_argument('--entropy-coef', type=float, default=0.01, help='Entropy bonus coefficient (higher=more exploration, try 1.0-10.0 for aggressive)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
    parser.add_argument('--discount', type=float, default=0.997, help='Discount factor gamma (default: 0.997)')

    # Adaptive entropy - boost exploration when policy collapses to HOLD
    parser.add_argument('--adaptive-entropy', action='store_true', default=False,
                       help='Enable adaptive entropy: boost when HOLD%% exceeds threshold')
    parser.add_argument('--entropy-hold-threshold', type=float, default=0.70,
                       help='HOLD%% threshold to trigger entropy boost (default: 0.70)')
    parser.add_argument('--entropy-boost-low', type=float, default=0.15,
                       help='Moderate entropy boost when HOLD%% > threshold (default: 0.15)')
    parser.add_argument('--entropy-boost-high', type=float, default=0.30,
                       help='Aggressive entropy boost when HOLD%% > threshold+10%% (default: 0.30)')

    # Multi-Action + Limit Orders
    parser.add_argument('--action-points', type=int, default=1,
                       help='Actions per timestep: 1=single action, 4=multi-action')
    parser.add_argument('--tp-sl-enabled', action='store_true', default=False,
                       help='Enable TP/SL guardrails (5%% SL, 10%% TP)')

    # Curriculum transfer flags
    parser.add_argument('--clear-buffer', action='store_true', default=False,
                       help='Clear replay buffer when loading checkpoint (use for new curriculum levels)')
    parser.add_argument('--reset-optimizer', action='store_true', default=False,
                       help='Reset optimizer state when loading checkpoint (fresh momentum for new regime)')

    # =========================================================================
    # Prediction Loss (Phase 2) - Auxiliary price prediction
    # =========================================================================
    parser.add_argument('--use-prediction-loss', action='store_true', default=False,
                       help='Enable auxiliary prediction loss (requires prediction head)')
    parser.add_argument('--prediction-loss-weight', type=float, default=0.2,
                       help='Final weight for auxiliary prediction loss (default: 0.2)')
    parser.add_argument('--prediction-loss-initial', type=float, default=0.01,
                       help='Initial prediction loss weight (annealed up, default: 0.01)')
    parser.add_argument('--prediction-anneal-steps', type=int, default=10000,
                       help='Steps to anneal prediction loss weight (default: 10000)')
    parser.add_argument('--multi-step-k', type=int, default=5,
                       help='Number of future steps for prediction loss (default: 5)')
    parser.add_argument('--scheduled-sampling-steps', type=int, default=20000,
                       help='Steps to anneal scheduled sampling (default: 20000)')
    parser.add_argument('--physics-reg-weight', type=float, default=0.01,
                       help='Physics regularization weight (default: 0.01)')

    # =========================================================================
    # Tree Search (Phase 3) - Predictive tree search for risk-adjusted rewards
    # =========================================================================
    parser.add_argument('--use-tree-search', action='store_true', default=False,
                       help='Enable predictive tree search for MCTS reward adjustment')
    parser.add_argument('--tree-depth', type=int, default=5,
                       help='Prediction tree depth (default: 5, gives 3^5=243 leaves)')
    parser.add_argument('--tree-branches', type=int, default=3,
                       help='Branches per node (default: 3)')
    parser.add_argument('--risk-lambda', type=float, default=0.1,
                       help='Uncertainty penalty weight for risk-adjusted rewards (default: 0.1)')

    # =========================================================================
    # Mixed Precision (Performance Optimization)
    # =========================================================================
    parser.add_argument('--bf16', action='store_true', default=False,
                       help='Use bfloat16 mixed precision (2x speedup on RTX 30/40 series)')

    # =========================================================================
    # DS24 Enhanced Metrics Logging
    # =========================================================================
    # Level-based logging system:
    #   0 = OFF (no DS24 logging)
    #   1 = CSV only (basic training metrics)
    #   2 = CSV + TRM intermediates + planning metrics (default when --ds24-logging)
    #   3 = Level 2 + physics forward pass (G_momentum, G_reversion, etc.)
    #   4 = Level 3 + latent space heatmaps + weight monitoring
    #   5 = Level 4 + 3D visualization (future, not implemented)
    #   6 = ShroomEngine - Full model introspection (future)
    parser.add_argument('--ds24-level', type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6],
                       help='DS24 logging level: 0=off, 1=csv, 2=+TRM/planning, 3=+physics, 4=+heatmaps, 5=+3D, 6=ShroomEngine')
    parser.add_argument('--ds24-logging', action='store_true', default=False,
                       help='Enable DS24 logging (shortcut for --ds24-level 2)')
    parser.add_argument('--ds24-intermediates-every', type=int, default=10,
                       help='Log TRM intermediate deltas every N epochs (default: 10)')
    parser.add_argument('--ds24-save-tensors', action='store_true', default=False,
                       help='Save periodic tensor snapshots for analysis (every 100 epochs)')
    parser.add_argument('--ds24-grad-norms', action='store_true', default=False,
                       help='Track per-head gradient norms (slower, ~20%% overhead)')

    args = parser.parse_args()

    # Resolve DS24 logging level (--ds24-logging is shortcut for --ds24-level 2)
    if args.ds24_logging and args.ds24_level == 0:
        args.ds24_level = 2
    # Create convenient level checks
    ds24_level = args.ds24_level

    # Map model size to state_dim if not explicitly provided
    if args.state_dim is None:
        if args.model_size == 'small':
            args.state_dim = 64
        elif args.model_size == 'medium':
            args.state_dim = 96
        elif args.model_size == 'large':
            args.state_dim = 160
    
    # Auto-adjust batch size for model size to prevent OOM on 24GB VRAM
    if args.model_size == 'large' and args.batch_size > 512:
        logging.warning(f"Large model + batch {args.batch_size} might OOM. Reducing to 512.")
        args.batch_size = 512
    elif args.model_size == 'medium' and args.batch_size > 384:
        logging.warning(f"Medium model + batch {args.batch_size} might OOM. Reducing to 384.")
        args.batch_size = 384

    # Dynamic Asset Configuration
    n_assets = 64
    asset_symbols = None
    min_coverage = 0.1  # Default 10%
    start_date = args.start_date
    end_date = args.end_date
    
    if args.config:
        import json
        with open(args.config) as f:
            config = json.load(f)
            if 'n_assets' in config:
                n_assets = config['n_assets']
            if 'asset_symbols' in config:
                asset_symbols = config['asset_symbols']
            if 'experiment_name' in config:
                args.name = config['experiment_name']
            if 'min_coverage' in config:
                min_coverage = config.get('min_coverage', 0.1)
            # Config overrides CLI args if present
            if 'start_date' in config:
                start_date = config['start_date']
            if 'end_date' in config:
                end_date = config['end_date']
                
            print(f"🔧 Loaded config: {args.config} (n_assets={n_assets}, coverage={min_coverage:.2f})")

    # Calculate min_required_assets
    min_required_assets = max(1, int(n_assets * min_coverage))
    print(f"DEBUG: Min required assets set to {min_required_assets} ({min_coverage*100:.1f}% coverage)")

    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join('logs', f"training_fast_{args.name}.log")), # Log per experiment
            logging.StreamHandler()
        ]
    )

    print("="*70)
    print(f"JAX TRADING AGENT | {args.mode} | Fee: {args.fee}bps | Slip: {args.slippage}bps | {n_assets} Assets")
    print("="*70)
    print(f"GPU Optimization: XLA autotune=4 | cache={_xla_cache}")
    print()

    print("[1/4] Initializing Environment...")

    # VectorBT-based environment (clean accounting, no FX bugs)
    # Data source priority: synthetic > schedule > database > error
    env_kwargs = dict(
        n_assets=n_assets,
        initial_cash=10000.0,
        trading_fee_bps=args.fee,
        slippage_bps=args.slippage,
        max_steps=args.steps,
        # Multi-Action + Limit Orders
        action_points=args.action_points,  # 1=single, 4=multi-action
        tp_sl_enabled=args.tp_sl_enabled,  # TP/SL guardrails
    )

    if args.synthetic_level is not None:
        env_kwargs['synthetic_level'] = args.synthetic_level
        env_kwargs['synthetic_n_steps'] = args.synthetic_steps
        data_source = f"synthetic level {args.synthetic_level}"
    elif hasattr(args, 'schedule_path') and args.schedule_path:
        env_kwargs['schedule_path'] = args.schedule_path
        if hasattr(args, 'db_path') and args.db_path:
            env_kwargs['db_path'] = args.db_path
        data_source = "training schedule"
    elif args.db_path:
        env_kwargs['db_path'] = args.db_path
        if args.start_date:
            env_kwargs['start_date'] = args.start_date
        if args.end_date:
            env_kwargs['end_date'] = args.end_date
        data_source = f"database {args.db_path}"
    else:
        # Default to synthetic level 0 if no data source specified
        env_kwargs['synthetic_level'] = 0
        env_kwargs['synthetic_n_steps'] = args.synthetic_steps
        data_source = "synthetic level 0 (default)"

    env = VectorBTTradingEnv(**env_kwargs)

    # Create additional parallel environments if requested
    parallel_envs = []
    if args.parallel_envs > 1:
        parallel_envs = [env]  # First env already created
        for i in range(args.parallel_envs - 1):
            # Create additional envs with different random seeds via reset
            parallel_envs.append(VectorBTTradingEnv(**env_kwargs))
        print(f"   Created {args.parallel_envs} parallel environments for batched self-play")

    # Update n_assets to match what the environment actually loaded
    n_assets = env.n_assets
    action_space_size = n_assets * N_ACTIONS_PER_ASSET  # [Sell, Hold, Buy, Limit Sell, Limit Buy]

    multi_action_mode = f"{args.action_points} action points" if args.action_points > 1 else "single action"
    parallel_mode = f", {args.parallel_envs}x parallel" if args.parallel_envs > 1 else ""
    print(f"✅ Environment ready (VectorBT, {args.mode} mode, {n_assets} assets, {args.steps} steps/episode{parallel_mode})")
    print(f"   5-action space (limit orders), {multi_action_mode}, TP/SL={'ON' if args.tp_sl_enabled else 'OFF'}")
    print(f"DEBUG: n_assets={n_assets}, total_actions={action_space_size}")

    print("[2/4] Initializing JAX model...")
    rng = jax.random.PRNGKey(42)

    # Mixed precision setup
    compute_dtype = jnp.bfloat16 if args.bf16 else jnp.float32
    dtype_str = "bfloat16" if args.bf16 else "float32"

    state_dim = args.state_dim
    model = MuZeroNetworks(
        state_dim=state_dim,
        action_space_size=action_space_size,  # n_assets * N_ACTIONS_PER_ASSET
        support_size=VALUE_SUPPORT_SIZE,  # Central config (751 bins over [-750, 750])
        H_cycles=args.h_cycles,
        L_cycles=args.l_cycles,
        # Prediction head (Phase 2)
        use_prediction_head=args.use_prediction_loss,
        n_assets=n_assets,
        n_features=10,  # External features dimension
        # Mixed precision
        dtype=compute_dtype,
        param_dtype=jnp.float32,  # Always keep params in float32 for stability
    )
    print(f"DEBUG: Model definition: state_dim={model.state_dim}, size={args.model_size}, actions={model.action_space_size}")
    print(f"DEBUG: Compute dtype: {dtype_str} ({'TensorCore enabled' if args.bf16 else 'standard precision'})")
    print(f"DEBUG: Entropy coefficient: {args.entropy_coef} (higher=more exploration)")
    if args.use_prediction_loss:
        print(f"DEBUG: Prediction head ENABLED (λ_pred: {args.prediction_loss_initial} -> {args.prediction_loss_weight})")

    # Create prediction loss config if enabled
    prediction_config = None
    if args.use_prediction_loss:
        prediction_config = PredictionLossConfig(
            weight_initial=args.prediction_loss_initial,
            weight_final=args.prediction_loss_weight,
            anneal_steps=args.prediction_anneal_steps,
            multi_step_k=args.multi_step_k,
            scheduled_sampling_steps=args.scheduled_sampling_steps,
            physics_reg_weight=args.physics_reg_weight,
        )

    trainer = MuZeroTrainer(
        model=model,
        learning_rate=args.lr,
        num_simulations=args.sims,  # Dynamic sims
        batch_size=args.batch_size,      # Dynamic batch size
        unroll_steps=5,
        td_steps=10,
        discount=args.discount,  # Discount factor for TD targets
        value_loss_weight=2.0,  # Increased but not excessive (was 1.0, tried 5.0)
        policy_loss_weight=1.0,
        reward_loss_weight=1.0,
        entropy_coef=args.entropy_coef,  # Entropy bonus to prevent policy collapse
        # Prediction loss (Phase 2)
        use_prediction_loss=args.use_prediction_loss,
        prediction_config=prediction_config,
        # DS24 gradient monitoring
        track_grad_norms=args.ds24_grad_norms,
    )

    rng, init_rng = jax.random.split(rng)
    dummy_obs = jnp.ones((1, 27, n_assets, n_assets))
    state = trainer.init_train_state(init_rng, dummy_obs)

    # Resume from checkpoint if provided
    epoch_offset = 0  # Track total epochs for CSV continuity
    if args.load_checkpoint:
        # Resolve to absolute path for Orbax
        abs_ckpt_path = os.path.abspath(args.load_checkpoint)
        print(f"📥 Loading checkpoint from: {args.load_checkpoint}")
        print(f"   (Resolved absolute path: {abs_ckpt_path})")

        try:
            # MuZeroTrainer.restore_checkpoint returns the restored state
            restored_state = trainer.restore_checkpoint(abs_ckpt_path)
            if restored_state:
                trainer.state = restored_state
                state = restored_state # Update local reference
                # Calculate epoch offset from step count
                # Assuming ~50 steps per epoch (200 env steps / 4)
                epoch_offset = state.step_count // 50
                print(f"✅ Checkpoint loaded successfully! (Step: {state.step_count}, Epoch offset: {epoch_offset})")

                # Reset optimizer state if requested (for curriculum transfer)
                if args.reset_optimizer:
                    print("🔄 Resetting optimizer state for new regime...")
                    trainer.state = trainer.reset_optimizer_state(trainer.state)
                    state = trainer.state
                    print("   (Momentum and variance reset, model weights preserved)")
            else:
                print(f"⚠️ Failed to load checkpoint (trainer returned None)")
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            # Do not crash, continue from scratch if load fails? Or crash?
            # Crashing is safer to avoid accidental scratch training
            raise e

    # Create search with Maniac params
    # PARALLEL MCTS INTEGRATION
    
    # Report model size
    if state_dim == 64:
        param_estimate = "~7.3M"
    elif state_dim == 96:
        param_estimate = "~15M"
    elif state_dim == 160:
        param_estimate = "~36M"
    else:
        param_estimate = "custom"
    print(f"Model size: state_dim={state_dim} ({param_estimate} params)")
    
    search_kwargs = {
        'num_simulations': args.sims,
        'use_gumbel': True,
        'gumbel_scale': args.gumbel_scale,
        'dirichlet_alpha': args.noise,
        'exploration_fraction': args.noise_frac,
        'value_scale': args.value_scale,
        'value_shift': args.value_shift
    }
    print(f"MCTS Search: {args.sims} sims | Gumbel scale: {args.gumbel_scale} | Use Gumbel: True")
    if args.value_scale != 1.0 or args.value_shift != 0.0:
        print(f"Value Calibration: scale={args.value_scale}, shift={args.value_shift}")
    
    print(f"🚀 Initializing GPU-native MCTS...")
    # Use direct GPU MCTS instead of CPU worker pool
    # Worker pool runs MCTS on CPU (slow!), direct call runs on GPU (fast!)
    from mcts_trader.jax_agent.mctx_search import create_mctx_search
    search = create_mctx_search(model, **search_kwargs)
    search = jax.jit(search)  # JIT compile for GPU

    print(f"   Maniac Mode: Sims={args.sims}, Noise={args.noise}, Frac={args.noise_frac}")

    # =========================================================================
    # Tree Search (Phase 3) - Predictive tree search for uncertainty metrics
    # =========================================================================
    tree_search_fn = None
    risk_adjust_fn = None
    tree_search_uses_muzero_prediction = False  # Track if we're using trained model

    if args.use_tree_search:
        print(f"🌳 Initializing Predictive Tree Search...")
        from mcts_trader.jax_agent.prediction import (
            NeuralMarketODE,
            create_physics_aware_tree_search_fn,
            create_mcts_reward_adjuster,
            create_dynamic_reward_adjuster,
            TreeSearchConfig,
        )

        tree_config = TreeSearchConfig(
            depth=args.tree_depth,
            branches=args.tree_branches,
            store_trajectory=False,
        )
        n_leaves = args.tree_branches ** args.tree_depth

        # Check if we can use MuZero's TRAINED prediction model
        # Use the config flag instead of hasattr (Flax modules don't expose submodules as attrs)
        if args.use_prediction_loss and model.use_prediction_head:
            # Use MuZero's prediction model (will be trained during training)
            print(f"   Using MuZero's trained prediction model (dynamic params)")
            tree_search_uses_muzero_prediction = True

            # Create a standalone model with same config for tree search
            # (We'll pass MuZero's trained params at runtime)
            tree_pred_model = NeuralMarketODE(
                n_assets=n_assets,
                n_features=10,
                use_transformer=True,
                use_action_conditioning=True,  # Match MuZero's config
            )

            # Create dynamic reward adjuster that takes params at runtime
            risk_adjust_fn = create_dynamic_reward_adjuster(
                tree_pred_model,
                tree_config,
                risk_lambda=args.risk_lambda
            )
            print(f"   Depth={args.tree_depth}, Branches={args.tree_branches} → {n_leaves} leaves")
            print(f"   Risk λ={args.risk_lambda} (uncertainty penalty weight)")
            print(f"   ⚡ Params will update with training!")
        else:
            # Standalone (untrained) prediction model - less effective but still works
            if args.use_prediction_loss:
                print(f"   ⚠️ use_prediction_head not enabled, using standalone model")
            else:
                print(f"   ⚠️ Using untrained prediction model (add --use-prediction-loss for better results)")

            pred_model = NeuralMarketODE(
                n_assets=n_assets,
                n_features=10,
                use_transformer=True,
                use_action_conditioning=False,
            )

            rng, pred_init_rng = jax.random.split(rng)
            dummy_prices = jnp.zeros((1, n_assets))
            pred_params = pred_model.init(
                pred_init_rng,
                dummy_prices,
                dummy_prices,
                dummy_prices,
                dummy_prices,
            )

            tree_search_fn = create_physics_aware_tree_search_fn(pred_model, pred_params, tree_config)
            risk_adjust_fn = create_mcts_reward_adjuster(tree_search_fn, risk_lambda=args.risk_lambda)

            print(f"   Depth={args.tree_depth}, Branches={args.tree_branches} → {n_leaves} leaves")
            print(f"   Risk λ={args.risk_lambda} (uncertainty penalty weight)")
    print()

    print("[3/4] Initializing replay buffer...")
    # Reduce buffer size for multi-instance training (5K per instance = 10K total with 2 instances)
    buffer_capacity = 5000
    replay_buffer = JAXReplayBuffer(
        capacity=buffer_capacity,
        observation_shape=(27, n_assets, n_assets),
        action_size=action_space_size,  # n_assets * N_ACTIONS_PER_ASSET
        batch_size=args.batch_size,
        sequence_length=5,
        # Auxiliary data for prediction loss (Phase 2)
        n_assets=n_assets,
        n_features=10,
    )
    print(f"✅ Buffer ready ({buffer_capacity} capacity, float32)")
    if args.use_prediction_loss:
        print(f"   Prediction loss auxiliary storage enabled (n_assets={n_assets}, n_features=10)")

    # Clear buffer if requested (for curriculum transfer to new level)
    if args.clear_buffer and args.load_checkpoint:
        print("🧹 Clearing replay buffer for new regime...")
        replay_buffer.clear()
        print(f"   Buffer cleared: {len(replay_buffer)} trajectories")
    print()

    print("[4/4] Warmup (JIT compilation)...")
    warmup_start = time.time()
    rng, warmup_rng = jax.random.split(rng)
    _, _ = run_self_play(env, search, model.apply, trainer.state.params, warmup_rng, steps=5, support_size=model.support_size, discount=args.discount)
    print(f"✅ Warmup done ({time.time() - warmup_start:.1f}s)")
    print()

    print("="*70)
    print("TRAINING LOOP")
    print("="*70)
    print()

    checkpoint_dir = os.path.abspath(f"jax_ckpt_fast/{args.name}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Track metrics for scheduler and mastery check
    recent_rewards = []
    all_rewards = []  # Full history for mastery check

    # CSV logging for dashboard
    import csv
    csv_log_path = f'logs/training_{args.name}.csv'

    # DS24 Enhanced Metrics Logger (optional, level-based)
    ds24_hooks = None
    ds24_columns = []
    if ds24_level >= 1:
        from tools.DS24.training_hooks import create_ds24_hooks
        ds24_hooks = create_ds24_hooks(args, model=model)
        if ds24_hooks:
            ds24_columns = ds24_hooks.get_ds24_columns()
            print(f"📈 DS24 {ds24_hooks.get_level_description()}")
        if args.ds24_save_tensors:
            print(f"   Tensor snapshots → tensor_snapshots/{args.name}/")
        if args.ds24_grad_norms:
            print(f"   ⚠️ Gradient norm tracking enabled (~20% overhead)")

    # Base CSV columns
    base_columns = [
        # Core metrics
        'epoch', 'reward', 'reward_ma20', 'fee_bps', 'steps',
        # Action counts (5-action space: S/H/B/LS/LB)
        'n_buy', 'n_hold', 'n_sell', 'n_limit_sell', 'n_limit_buy',
        # Action percentages
        'pct_buy', 'pct_hold', 'pct_sell', 'pct_limit_sell', 'pct_limit_buy',
        # Timing
        'self_play_time', 'train_time',
        # Loss components
        'total_loss', 'value_loss', 'value_mse', 'value_cat', 'policy_loss', 'reward_loss',
        # Prediction loss (Phase 2)
        'pred_loss', 'lambda_pred', 'physics_reg',
        # Policy/Value analytics
        'policy_entropy', 'avg_value', 'value_std',
        # Value calibration
        'value_scale', 'value_shift',
        # Tree search (Phase 3)
        'uncertainty',
        # Config reference
        'synthetic_level', 'model_size', 'run_id'
    ]

    # Combine base + DS24 columns
    all_csv_columns = base_columns + ds24_columns

    # Append mode if resuming from checkpoint, otherwise create new
    csv_exists = os.path.exists(csv_log_path)
    if args.load_checkpoint and csv_exists:
        csv_file = open(csv_log_path, 'a', newline='')
        csv_writer = csv.writer(csv_file)
        print(f"📊 Appending training metrics to: {csv_log_path} (resumed at epoch {epoch_offset})")
    else:
        csv_file = open(csv_log_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(all_csv_columns)
        csv_file.flush()
        print(f"📊 Logging training metrics to: {csv_log_path}")

    # Adaptive value calibration tracking
    calibration_values = []  # Predicted values
    calibration_rewards = []  # Actual rewards
    adaptive_calibration = args.adaptive_calibration
    calibration_interval = 25  # Recompute scale every N epochs
    calibration_start_epoch = 50  # Start after this many epochs
    if adaptive_calibration:
        print(f"📐 Adaptive calibration enabled (starts epoch {calibration_start_epoch}, interval {calibration_interval})")

    try:
        for epoch in range(args.epochs):
            # Self-play
            start = time.time()
            rng, search_rng = jax.random.split(rng)

            # Extract prediction model params for dynamic tree search
            pred_params_for_tree = None
            if tree_search_uses_muzero_prediction and risk_adjust_fn is not None:
                # Get prediction model params from MuZero's params
                # Path: trainer.state.params['price_predictor'] (no extra 'params' level)
                try:
                    pred_params_for_tree = {'params': trainer.state.params['price_predictor']}
                except (KeyError, TypeError) as e:
                    # Log the path error once
                    if epoch == 0:
                        print(f"⚠️ Could not extract price_predictor params: {e}")
                        print(f"   Available keys: {list(trainer.state.params.keys())[:10]}...")
                    # Create dummy params for the dynamic adjuster (won't be optimal but won't crash)
                    pred_params_for_tree = {'params': {}}

            # Use batched self-play if parallel_envs > 1
            if args.parallel_envs > 1:
                trajectories, scores = run_self_play_batched(
                    parallel_envs,
                    search,
                    model.apply,
                    trainer.state.params,
                    search_rng,
                    steps=args.steps,
                    confidence_threshold=args.confidence_threshold,
                    support_size=model.support_size,
                    collect_prices=args.use_prediction_loss,
                    discount=args.discount,
                )
                # Use average score for display, add all trajectories
                score = np.mean(scores)
                trajectory = trajectories[0]  # Use first for action stats below
            else:
                trajectory, score = run_self_play(
                    env,
                    search,
                    model.apply,
                    trainer.state.params,
                    search_rng,
                    steps=args.steps,
                    confidence_threshold=args.confidence_threshold,
                    support_size=model.support_size,
                    collect_prices=args.use_prediction_loss,  # Phase 2
                    risk_adjust_fn=risk_adjust_fn,  # Phase 3
                    pred_params=pred_params_for_tree,  # Dynamic tree search params
                    discount=args.discount,
                )
                trajectories = [trajectory]  # Wrap in list for uniform handling
                scores = [score]

            self_play_time = time.time() - start

            # Track reward (use average for batched)
            all_rewards.append(score)  # All rewards for metrics
            recent_rewards.append(score)  # Recent for display
            if len(recent_rewards) > 20:
                recent_rewards.pop(0)

            # Add all trajectories to buffer
            for traj_idx, traj in enumerate(trajectories):
                # Sanity check for NaNs in trajectory (only for numpy arrays)
                for key, arr in traj.items():
                    if isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.floating):
                        if np.isnan(arr).any():
                            print(f"!!! CRITICAL WARNING: NaNs detected in {key} from self-play (env {traj_idx})!")

                replay_buffer.add_trajectory(
                    jnp.array(traj['observations']),
                    jnp.array(traj['actions']),
                    jnp.array(traj['rewards']),
                    jnp.array(traj['values']),
                    jnp.array(traj['policy_probs']),
                    jnp.array(traj['dones'])
                )

                # Add auxiliary price data for prediction loss (Phase 2)
                if args.use_prediction_loss and 'prices' in traj:
                    replay_buffer.add_auxiliary(
                        prices=traj['prices'],
                        next_prices=traj['next_prices'],
                    )

            # Training
            if epoch >= 5:
                train_start = time.time()
                metrics = trainer.train(replay_buffer, num_iterations=50, rng=rng)
                train_time = time.time() - train_start
                loss = metrics['total_loss'][-1]
                val_loss = metrics['value_loss'][-1]
                val_mse = metrics['value_mse'][-1]
                val_cat = metrics['value_cat'][-1]
                pol_loss = metrics['policy_loss'][-1]
                rew_loss = metrics['reward_loss'][-1]

                # Prediction loss metrics (Phase 2 - defaults to 0 if not enabled)
                pred_loss = metrics.get('loss_prediction', 0.0)
                if isinstance(pred_loss, list):
                    pred_loss = pred_loss[-1] if pred_loss else 0.0
                lambda_pred = metrics.get('lambda_pred', 0.0)
                if isinstance(lambda_pred, list):
                    lambda_pred = lambda_pred[-1] if lambda_pred else 0.0
                physics_reg = metrics.get('loss_physics_reg', 0.0)
                if isinstance(physics_reg, list):
                    physics_reg = physics_reg[-1] if physics_reg else 0.0

                # Action Stats
                actions_arr = np.array(trajectory['actions'])
                total_actions = len(actions_arr)
                n_sell = np.sum(actions_arr % N_ACTIONS_PER_ASSET == 0)
                n_hold = np.sum(actions_arr % N_ACTIONS_PER_ASSET == 1)
                n_buy = np.sum(actions_arr % N_ACTIONS_PER_ASSET == 2)
                n_limit_sell = np.sum(actions_arr % N_ACTIONS_PER_ASSET == 3)
                n_limit_buy = np.sum(actions_arr % N_ACTIONS_PER_ASSET == 4)
                pct_sell = 100.0 * n_sell / total_actions
                pct_hold = 100.0 * n_hold / total_actions
                pct_buy = 100.0 * n_buy / total_actions
                pct_limit_sell = 100.0 * n_limit_sell / total_actions
                pct_limit_buy = 100.0 * n_limit_buy / total_actions

                # Adaptive entropy: boost when policy collapses to HOLD
                if args.adaptive_entropy:
                    hold_ratio = n_hold / total_actions
                    if hold_ratio > args.entropy_hold_threshold + 0.10:
                        # Very high HOLD% -> aggressive boost
                        new_entropy_coef = args.entropy_boost_high
                    elif hold_ratio > args.entropy_hold_threshold:
                        # Moderate HOLD% -> moderate boost
                        new_entropy_coef = args.entropy_boost_low
                    else:
                        # Normal -> base entropy
                        new_entropy_coef = args.entropy_coef

                    if trainer.entropy_coef != new_entropy_coef:
                        print(f"  [ADAPTIVE ENTROPY] HOLD={hold_ratio*100:.1f}% -> entropy_coef: {trainer.entropy_coef:.3f} -> {new_entropy_coef:.3f}")
                        trainer.entropy_coef = new_entropy_coef

                # Policy Analytics
                policy_probs_arr = np.array(trajectory['policy_probs'])
                policy_entropy = -np.mean(np.sum(policy_probs_arr * np.log(policy_probs_arr + 1e-10), axis=1))
                avg_value = np.mean(trajectory['values'])
                value_std = np.std(trajectory['values'])

                # Reward MA20
                reward_ma20 = np.mean(all_rewards[-20:]) if len(all_rewards) >= 20 else np.mean(all_rewards)

                # DS24 Enhanced Logging (via hooks module)
                if ds24_hooks is not None:
                    # Extract last training metrics for gradient norms (Level 4+)
                    train_metrics_last = None
                    if args.ds24_grad_norms and metrics:
                        train_metrics_last = {
                            k: v[-1] if isinstance(v, list) and v else v
                            for k, v in metrics.items()
                        }
                    ds24_hooks.on_epoch_end(
                        epoch=epoch,
                        trajectory=trajectory,
                        trainer_state=trainer.state,
                        args=args,
                        train_metrics=train_metrics_last
                    )

                calib_str = f" | Calib: {search.value_scale:.2f}/{search.value_shift:.2f}" if (search.value_scale != 1.0 or search.value_shift != 0.0) else ""

                action_str = f"S/H/B/LS/LB: {n_sell}/{n_hold}/{n_buy}/{n_limit_sell}/{n_limit_buy}"

                # Tree search uncertainty (Phase 3)
                avg_uncertainty = 0.0
                uncertainty_str = ""
                if 'uncertainty' in trajectory and len(trajectory['uncertainty']) > 0:
                    avg_uncertainty = np.mean(trajectory['uncertainty'])
                    uncertainty_str = f" | Unc: {avg_uncertainty:.4f}"

                # Prediction loss string (Phase 2)
                pred_str = ""
                if pred_loss > 0:
                    pred_str = f" | Pred: {pred_loss:.4f} (λ={lambda_pred:.3f})"

                print(f"Epoch {epoch+1:4d} | Reward: {score:+7.2f} | Fee: {env.trading_fee_bps:2d} | Steps: {len(trajectory['actions']):3d} | "
                      f"{action_str} | "
                      f"Self-play: {self_play_time:5.1f}s | Train: {train_time:4.1f}s | "
                      f"Loss: {loss:.4f} | Val: {val_loss:.4f} | Pol: {pol_loss:.4f} | Rew: {rew_loss:.4f}{pred_str}{calib_str}{uncertainty_str}")

                # Log to CSV (use global epoch count)
                global_epoch = epoch + 1 + epoch_offset
                base_row = [
                    global_epoch, score, reward_ma20, env.trading_fee_bps, len(trajectory['actions']),
                    n_buy, n_hold, n_sell, n_limit_sell, n_limit_buy,
                    pct_buy, pct_hold, pct_sell, pct_limit_sell, pct_limit_buy,
                    self_play_time, train_time,
                    loss, val_loss, val_mse, val_cat, pol_loss, rew_loss,
                    pred_loss, lambda_pred, physics_reg,
                    policy_entropy, avg_value, value_std,
                    search.value_scale, search.value_shift,
                    avg_uncertainty,
                    args.synthetic_level, args.model_size, args.name
                ]

                # Append DS24 values if enabled
                if ds24_hooks is not None:
                    ds24_values = ds24_hooks.get_current_values()
                    ds24_row = [ds24_values.get(col, '') for col in ds24_columns]
                    csv_writer.writerow(base_row + ds24_row)
                else:
                    csv_writer.writerow(base_row)

                csv_file.flush()
                os.fsync(csv_file.fileno())  # Force OS to write to disk

                if (epoch + 1) % 10 == 0:
                    trainer.save_checkpoint(checkpoint_dir)
                    print(f"  ✓ Checkpoint saved to {checkpoint_dir}")

                # Adaptive value calibration
                if adaptive_calibration:
                    calibration_values.append(avg_value)
                    calibration_rewards.append(score)

                    # Keep only recent data (last 100 epochs)
                    if len(calibration_values) > 100:
                        calibration_values.pop(0)
                        calibration_rewards.pop(0)

                    # Recompute scale every interval after warmup
                    if epoch >= calibration_start_epoch and (epoch + 1) % calibration_interval == 0:
                        if len(calibration_values) >= 25:
                            vals = np.array(calibration_values)
                            rews = np.array(calibration_rewards)

                            # Compute optimal scale: match std of predictions to std of rewards
                            val_std = np.std(vals)
                            rew_std = np.std(rews)

                            if val_std > 0.01:  # Avoid division by zero
                                optimal_scale = rew_std / val_std
                                # Smooth update (don't jump too fast)
                                new_scale = 0.7 * search.value_scale + 0.3 * optimal_scale
                                # Clamp to reasonable range
                                new_scale = np.clip(new_scale, 0.5, 4.0)
                                search.value_scale = float(new_scale)

                                # Also compute optimal shift (center alignment)
                                val_mean = np.mean(vals) * new_scale
                                rew_mean = np.mean(rews)
                                optimal_shift = rew_mean - val_mean
                                search.value_shift = float(0.7 * search.value_shift + 0.3 * optimal_shift)

                                print(f"  📐 Adaptive calibration: scale={search.value_scale:.3f}, shift={search.value_shift:.3f}")
            else:
                # Action Stats for warmup
                actions_arr = np.array(trajectory['actions'])
                total_actions = len(actions_arr)
                n_sell = np.sum(actions_arr % N_ACTIONS_PER_ASSET == 0)
                n_hold = np.sum(actions_arr % N_ACTIONS_PER_ASSET == 1)
                n_buy = np.sum(actions_arr % N_ACTIONS_PER_ASSET == 2)
                n_limit_sell = np.sum(actions_arr % N_ACTIONS_PER_ASSET == 3)
                n_limit_buy = np.sum(actions_arr % N_ACTIONS_PER_ASSET == 4)
                pct_sell = 100.0 * n_sell / total_actions
                pct_hold = 100.0 * n_hold / total_actions
                pct_buy = 100.0 * n_buy / total_actions
                pct_limit_sell = 100.0 * n_limit_sell / total_actions
                pct_limit_buy = 100.0 * n_limit_buy / total_actions

                # Policy Analytics for warmup
                policy_probs_arr = np.array(trajectory['policy_probs'])
                policy_entropy = -np.mean(np.sum(policy_probs_arr * np.log(policy_probs_arr + 1e-10), axis=1))
                avg_value = np.mean(trajectory['values'])
                value_std = np.std(trajectory['values'])

                # Reward MA20
                reward_ma20 = np.mean(all_rewards[-20:]) if len(all_rewards) >= 20 else np.mean(all_rewards)

                action_str = f"S/H/B/LS/LB: {n_sell}/{n_hold}/{n_buy}/{n_limit_sell}/{n_limit_buy}"

                print(f"Epoch {epoch+1:4d} | Reward: {score:+7.2f} | Fee: {env.trading_fee_bps:2d} | Steps: {len(trajectory['actions']):3d} | "
                      f"{action_str} | "
                      f"Self-play: {self_play_time:5.1f}s | (warmup)")

                # Tree search uncertainty for warmup (Phase 3)
                avg_uncertainty = 0.0
                if 'uncertainty' in trajectory and len(trajectory['uncertainty']) > 0:
                    avg_uncertainty = np.mean(trajectory['uncertainty'])

                # Log warmup to CSV (no train metrics, use global epoch count)
                global_epoch = epoch + 1 + epoch_offset
                base_row = [
                    global_epoch, score, reward_ma20, env.trading_fee_bps, len(trajectory['actions']),
                    n_buy, n_hold, n_sell, n_limit_sell, n_limit_buy,
                    pct_buy, pct_hold, pct_sell, pct_limit_sell, pct_limit_buy,
                    self_play_time, 0,  # train_time=0
                    0, 0, 0, 0, 0, 0,   # loss components all 0
                    0, 0, 0,            # pred_loss, lambda_pred, physics_reg all 0
                    policy_entropy, avg_value, value_std,
                    search.value_scale, search.value_shift,
                    avg_uncertainty,
                    args.synthetic_level, args.model_size, args.name
                ]

                # Append empty DS24 values if enabled (warmup has no TRM data)
                if ds24_hooks is not None:
                    ds24_row = ['' for _ in ds24_columns]
                    csv_writer.writerow(base_row + ds24_row)
                else:
                    csv_writer.writerow(base_row)

                csv_file.flush()
                      
    finally:
        # Close CSV log
        if 'csv_file' in locals():
            csv_file.close()
            print(f"📊 Training metrics saved to: {csv_log_path}")

        # No cleanup needed for GPU-native MCTS
        print("✅ MCTS cleanup complete.")

    print()
    print("="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    
    # Save metrics for scheduler and mastery check
    avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0
    recent_50_avg = np.mean(all_rewards[-50:]) if len(all_rewards) >= 50 else np.mean(all_rewards) if all_rewards else 0.0

    metrics_data = {
        'avg_reward': float(avg_reward),
        'final_loss': float(loss) if 'loss' in locals() else 0.0,
        'episode_rewards': [float(r) for r in all_rewards],  # Full history for mastery check
        'recent_avg_reward': float(recent_50_avg),  # Last 50 epochs average
        'total_epochs': args.epochs,
        'experiment_name': args.name
    }
    import json
    with open('last_run_metrics.json', 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"📊 Saved metrics: avg_reward={avg_reward:.2f}, recent_50_avg={recent_50_avg:.2f}")

    print()
    print("="*70)
    print("TRAINING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    # Needed for torch import in encoder
    import torch
    main()