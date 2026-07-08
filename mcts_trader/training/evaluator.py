"""
Generalization evaluation for MuZero training.

This module provides evaluation on out-of-distribution synthetic levels
to track how well the model generalizes during curriculum training.
"""
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Callable

from mcts_trader.jax_agent import VALUE_SUPPORT_SIZE
from mcts_trader.env.vectorbt_env import VectorBTTradingEnv

# Action space: 5 actions per asset [Sell, Hold, Buy, Limit Sell, Limit Buy]
N_ACTIONS_PER_ASSET = 5

# TASK_828 Stage 3: emit the vector-action eval-skip warning only ONCE per process.
_VECTOR_EVAL_WARNED = False


def run_generalization_eval(
    eval_level: int,
    search_fn: Callable,
    model_apply_fn: Callable,
    params: Dict,
    rng_key: jnp.ndarray,
    steps: int = 200,
    support_size: int = VALUE_SUPPORT_SIZE,
    n_assets: int = 64,
    fee: int = 0,
    slippage: int = 0,
    vector_action: bool = False,  # TASK_828 Stage 3 (default OFF = canon scalar eval)
) -> Dict[str, float]:
    """Run evaluation episode on a different synthetic level to test generalization.

    Used during curriculum training to track how well the model handles
    harder patterns (e.g., evaluate on L4 rotation while training on L0 sine).

    Args:
        eval_level: Synthetic level to evaluate on (0-4)
        search_fn: MCTS search function
        model_apply_fn: Model apply function
        params: Model parameters
        rng_key: JAX RNG key
        steps: Episode length
        support_size: Value support size
        n_assets: Number of assets
        fee: Trading fee in bps
        slippage: Slippage in bps

    Returns:
        Dict with evaluation metrics:
            - reward: Episode reward (%)
            - alpha: Agent return - market return (%)
            - market_return: Buy-and-hold return (%)
            - agent_return: Agent final return (%)
            - pct_hold: Hold action percentage
            - pct_buy: Buy action percentage
            - pct_sell: Sell action percentage
            - eval_level: The level evaluated on
            - steps: Number of steps completed
    """
    # TASK_828 Stage 3: the in-loop generalization eval runs on the LEGACY
    # VectorBTTradingEnv, which steps a scalar action index and caps a multi-action
    # list to ``action_points`` (<< 64) in ascending asset-index order — for a
    # per-asset action VECTOR that silently drops most assets and is index-biased.
    # Rather than write a systematically-biased approximation into the same
    # eval_reward/eval_alpha fields as a faithful scalar eval, SKIP it (fail-closed)
    # and emit a one-time warning. The faithful per-asset eval is the scheduled
    # JAX-env run (TASK_828 Stage 4). Returns no reward/alpha → the caller leaves
    # those CSV columns ABSENT for skipped rows (no fabricated number).
    if vector_action:
        global _VECTOR_EVAL_WARNED
        if not _VECTOR_EVAL_WARNED:
            print("[EVAL] vector-action mode: in-loop generalization eval SKIPPED — "
                  "the legacy VectorBTTradingEnv is action_points-capped/index-biased "
                  "for vector actions; the faithful per-asset eval is the scheduled "
                  "JAX-env run (TASK_828 Stage 4).")
            _VECTOR_EVAL_WARNED = True
        return {'skipped': True, 'eval_level': eval_level, 'steps': 0}

    # Create evaluation environment
    eval_env = VectorBTTradingEnv(
        n_assets=n_assets,
        trading_fee_bps=fee,
        slippage_bps=slippage,
        synthetic_level=eval_level,
        synthetic_n_steps=steps + 100,  # Extra buffer
        max_steps=steps,
        allow_short=True,  # MARGIN mode
    )

    # Run single evaluation episode
    obs_dict, _ = eval_env.reset()
    score = 0.0
    actions_list = []
    rng = rng_key

    for step in range(steps):
        rng, step_rng = jax.random.split(rng)

        # Prepare observation
        obs_array = obs_dict['observation']
        if hasattr(obs_array, 'cpu'):
            obs_array = obs_array.cpu().numpy()
        obs_array = np.expand_dims(obs_array, axis=0)
        obs_jax = jnp.array(obs_array, dtype=jnp.float32)

        # Action mask
        action_mask = obs_dict['action_mask']
        if hasattr(action_mask, 'cpu'):
            action_mask = action_mask.cpu().numpy()
        mask_jax = jnp.array(np.expand_dims(action_mask, 0), dtype=jnp.int8)

        # MCTS search (no exploration noise in eval)
        action_jax, _, _ = search_fn(params, step_rng, obs_jax, mask_jax)
        action = int(action_jax[0])
        actions_list.append(action)

        # Step environment
        obs_dict, reward, terminated, truncated, info = eval_env.step(action)
        score += reward

        if terminated or truncated:
            break

    # Compute action stats
    actions_arr = np.array(actions_list)
    total_actions = len(actions_arr)
    n_sell = np.sum(actions_arr % N_ACTIONS_PER_ASSET == 0)
    n_hold = np.sum(actions_arr % N_ACTIONS_PER_ASSET == 1)
    n_buy = np.sum(actions_arr % N_ACTIONS_PER_ASSET == 2)

    # Extract market baseline from final info
    market_return = info.get('market_return', 0.0)
    agent_return = info.get('agent_return', 0.0)
    alpha = info.get('alpha', agent_return - market_return)

    eval_env.close()

    return {
        'reward': score,
        'alpha': alpha,
        'market_return': market_return,
        'agent_return': agent_return,
        'pct_hold': 100.0 * n_hold / total_actions if total_actions > 0 else 0,
        'pct_buy': 100.0 * n_buy / total_actions if total_actions > 0 else 0,
        'pct_sell': 100.0 * n_sell / total_actions if total_actions > 0 else 0,
        'eval_level': eval_level,
        'steps': len(actions_list),
    }


def run_multi_level_eval(
    levels: list,
    search_fn: Callable,
    model_apply_fn: Callable,
    params: Dict,
    rng_key: jnp.ndarray,
    steps: int = 200,
    n_assets: int = 64,
    fee: int = 0,
    slippage: int = 0,
) -> Dict[int, Dict[str, float]]:
    """Evaluate agent on multiple synthetic levels.

    Convenience function to run generalization evaluation across
    multiple curriculum levels in a single call.

    Args:
        levels: List of synthetic levels to evaluate (e.g., [0, 2, 4])
        Other args same as run_generalization_eval

    Returns:
        Dict mapping level -> evaluation metrics dict
    """
    results = {}
    rng = rng_key

    for level in levels:
        rng, eval_rng = jax.random.split(rng)
        results[level] = run_generalization_eval(
            eval_level=level,
            search_fn=search_fn,
            model_apply_fn=model_apply_fn,
            params=params,
            rng_key=eval_rng,
            steps=steps,
            n_assets=n_assets,
            fee=fee,
            slippage=slippage,
        )

    return results
