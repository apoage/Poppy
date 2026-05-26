"""
TASK_780: Oracle-as-MCTS Distillation helpers.

This module bridges the existing multipass oracle infrastructure
(`path_solver.py:multipass_oracle_policy`) to MCTS-shaped training targets:
visit distributions, root values, per-step policies, per-action Q-values, and
dynamics-trajectory targets.

The canonical Foxglove Phase-1 path (`easy_trainer.py`) collapses the
multi-pass output to a single soft policy via `jnp.mean(all_policies, axis=0)`
at `path_solver.py:324` — discarding the per-pass detail that TASK_780 needs.
Rather than mutate that function's return tuple (which is touched by ~11 call
sites in the canon), we re-implement the per-pass loop here as a parallel
helper that exposes the un-collapsed `(n_passes, T, n_actions)` tensor.

References
----------
- `tasks/TASK_780_ORACLE_AS_MCTS_DISTILLATION.md`
- `tasks/TASK_697_ORACLE_AS_MCTS_DISTILLATION.md` §A.1 mapping table.
- `runs/780_oracle_as_mcts/AUTHORIZATION_2026-05-26.md` §"Framing".
- `mcts_trader/jax_agent/path_solver.py:222-339` (single-episode multipass).
- `mcts_trader/jax_agent/path_solver.py:342-392` (batched multipass).

Action-space convention
-----------------------
Per `mcts_trader/CLAUDE.md` and `path_solver.py:406, 473`, actions are
encoded as `asset * 5 + action_type` with:
  0 = Sell, 1 = Hold, 2 = Buy, 3 = LimitSell, 4 = LimitBuy
The oracle never emits LimitSell/LimitBuy, so visit-counts on slots 3 and 4
are identically zero by construction. TASK_780 retains the 320-slot policy
shape (5 actions × 64 assets) for byte-identical compatibility with
TASK_767's deployed action mask at Phase-2.
"""

from __future__ import annotations

from functools import partial
from typing import Dict, Tuple

import jax
import jax.numpy as jnp

from mcts_trader.jax_agent.path_solver import (
    DEFAULT_MAX_PATH_LEN,
    greedy_lookahead_path,
    optimal_path_to_policy,
)


# =============================================================================
# Multipass oracle with per-pass detail exposed
# =============================================================================

@partial(
    jax.jit,
    static_argnames=[
        'k', 'lookahead', 'max_len', 'n_passes',
        'n_actions_per_asset', 'temperature',
    ],
)
def multipass_oracle_with_passes(
    returns: jnp.ndarray,
    key: jnp.ndarray,
    k: int = 1,
    lookahead: int = 20,
    transaction_cost: float = 0.001,
    min_alpha_threshold: float = 0.001,
    max_len: int = DEFAULT_MAX_PATH_LEN,
    n_passes: int = 16,
    noise_scale: float = 0.1,
    n_actions_per_asset: int = 5,
    temperature: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Multipass oracle that keeps the un-collapsed per-pass tensor.

    Mirrors `path_solver.multipass_oracle_policy` (`:222-339`) but returns the
    full `(n_passes, T, n_assets * n_actions_per_asset)` tensor instead of
    collapsing to mean at `:324`. This is the load-bearing input for
    `oracle_to_mcts_targets()`.

    Returns
    -------
    all_policies : jnp.ndarray, shape (n_passes, T, n_assets * 5)
        Per-pass soft policies (one-hot-ish: each pass is a single deterministic
        path through noisy returns, mapped to a per-asset 5-class policy via
        `optimal_path_to_policy`). The canon `multipass_oracle_policy` takes
        the mean over the leading axis at line 324; we expose it directly.
    clean_cumulative_pnl : jnp.ndarray, shape (T,)
        Cumulative PnL of the *clean* oracle path (no noise). Used as the
        value-target trajectory per `path_solver.py:270-279`.
    clean_total_alpha : float
        Final alpha of the clean path.
    clean_selections : jnp.ndarray, shape (T, n_assets)
        TASK_689 clean oracle selections (binary mask of optimal holdings),
        reusable as the per-step dynamics-action trajectory target.
    """
    T, n_assets = returns.shape
    return_std = jnp.maximum(jnp.std(returns), 1e-6)

    # 0. Clean oracle path for value targets (matches multipass_oracle_policy:270-279)
    clean_path = greedy_lookahead_path(
        returns,
        k=k,
        lookahead=lookahead,
        transaction_cost=transaction_cost,
        min_alpha_threshold=min_alpha_threshold,
        max_len=max_len,
    )

    def single_pass(subkey):
        # 1. Perturb returns with Gaussian noise
        noise = jax.random.normal(subkey, returns.shape) * noise_scale * return_std
        noisy_returns = returns + noise

        # 2. Solve binary path on noisy returns
        path = greedy_lookahead_path(
            noisy_returns,
            k=k,
            lookahead=lookahead,
            transaction_cost=transaction_cost,
            min_alpha_threshold=min_alpha_threshold,
            max_len=max_len,
        )

        # 3. Compute prev_selections for state-aware oracle (TASK_141)
        prev_sels = jnp.zeros((T, n_assets))
        prev_sels = prev_sels.at[1:].set(
            jnp.where(
                path.selections[:-1] > 0.5,
                jnp.where(
                    returns[:-1] > 0,
                    1.0,
                    jnp.where(returns[:-1] < 0, -1.0, 0.0),
                ),
                0.0,
            )
        )

        # 4. Convert binary path → one-hot policy (state-aware, correct per-pass)
        policy = optimal_path_to_policy(
            path.selections,
            returns,
            prev_selections=prev_sels,
            n_actions_per_asset=n_actions_per_asset,
            temperature=temperature,
        )
        return policy

    # 5. Split key, lax.map over all passes (TASK_675: caps VRAM at 1-pass peak)
    keys = jax.random.split(key, n_passes)
    all_policies = jax.lax.map(single_pass, keys)  # (n_passes, T, n_assets * 5)

    return (
        all_policies,
        clean_path.cumulative_pnl,
        clean_path.total_alpha,
        clean_path.selections,
    )


# =============================================================================
# Oracle → MCTS target mapping (TASK_697 §A.1)
# =============================================================================

@partial(jax.jit, static_argnames=['n_assets', 'n_actions_per_asset'])
def oracle_to_mcts_targets(
    all_policies: jnp.ndarray,
    clean_pnl: jnp.ndarray,
    clean_selections: jnp.ndarray,
    n_assets: int = 64,
    n_actions_per_asset: int = 5,
) -> Dict[str, jnp.ndarray]:
    """Convert multipass-oracle output → MCTS-shaped training targets.

    Implements the TASK_697 §A.1 mapping in a JIT-compatible JAX form:

    | Oracle concept                        | MCTS equivalent                |
    |---------------------------------------|--------------------------------|
    | N oracle passes                       | N MCTS simulations             |
    | Per-pass asset selections             | Visited actions per simulation |
    | Aggregated selection frequency        | Visit-count distribution       |
    | Per-pass returns                      | Backed-up leaf values          |
    | Mean return across passes             | Root value estimate            |
    | Per-step oracle decisions             | Dynamics transition targets    |

    Note on Q-values (Phase-1 caveat). In the multipass oracle the *clean*
    path provides a single PnL trajectory shared across all passes; passes
    differ only in *which assets they select* under noise (not in their
    realized PnL). The Q-values returned here therefore reduce to
    `clean_pnl[-1] * visit_indicator` rather than a per-pass conditional
    mean. The Q-value loss is gated OFF in the default Phase-1 config
    (`oracle_distill_q_value_weight = 0.0`); Phase-2 may upgrade to
    per-pass actual PnL if the signal is needed. See
    `tasks/TASK_780_IMPLEMENTATION_PLAN.md` §"Surprises" for details.

    Parameters
    ----------
    all_policies : jnp.ndarray, shape (n_passes, T, n_assets * n_actions_per_asset)
        Per-pass soft policies, output of `multipass_oracle_with_passes`.
        Each pass is essentially a one-hot policy (5-class softmax peaked
        on the action selected by that pass at each step).
    clean_pnl : jnp.ndarray, shape (T,)
        Cumulative PnL of the clean oracle path.
    clean_selections : jnp.ndarray, shape (T, n_assets)
        Per-step clean oracle selections (binary mask).
    n_assets : int
        Number of assets (default 64).
    n_actions_per_asset : int
        Actions per asset (default 5).

    Returns
    -------
    Dict with keys:
      - policy_target    : (n_assets * n_actions_per_asset,)  root visit-distribution.
      - value_target     : scalar                              mean total PnL = clean episode PnL.
      - q_values         : (n_assets * n_actions_per_asset,)  per-action backed-up value (root).
      - step_policies    : (T, n_assets * n_actions_per_asset) per-step visit-distributions.
      - step_values      : (T,)                                per-step value targets
                                                              (clean remaining PnL).
      - step_actions     : (T, n_assets) int                   per-step argmax action per asset
                                                              (dynamics-trajectory target).
      - visit_counts     : (T, n_assets * n_actions_per_asset) per-step integer visit counts in float.
    """
    n_passes, T, n_actions = all_policies.shape
    assert n_actions == n_assets * n_actions_per_asset, (
        f"all_policies last dim {n_actions} != n_assets*n_actions_per_asset "
        f"({n_assets * n_actions_per_asset})"
    )

    # 1. Per-pass per-asset action ids (argmax over the 5-class softmax)
    per_pass_reshaped = all_policies.reshape(
        n_passes, T, n_assets, n_actions_per_asset
    )
    per_pass_actions = jnp.argmax(per_pass_reshaped, axis=-1)  # (n_passes, T, n_assets)

    # 2. Visit one-hot tensor: which (asset, action) slot did each pass visit?
    visit_one_hot = jax.nn.one_hot(
        per_pass_actions, n_actions_per_asset, dtype=jnp.float32
    )  # (n_passes, T, n_assets, n_actions_per_asset)

    # 3. Per-step visit counts (sum over passes)
    step_visit_counts = jnp.sum(visit_one_hot, axis=0)  # (T, n_assets, n_actions_per_asset)
    step_visit_counts_flat = step_visit_counts.reshape(T, n_actions)

    # 4. Per-step visit distributions (normalized to sum to 1.0 per asset)
    #    Each asset's 5 action slots form a valid probability distribution.
    step_visit_per_asset_sum = jnp.maximum(
        jnp.sum(step_visit_counts, axis=-1, keepdims=True), 1e-8
    )  # (T, n_assets, 1) — equals n_passes if no degeneracy
    step_policies_per_asset = step_visit_counts / step_visit_per_asset_sum
    step_policies = step_policies_per_asset.reshape(T, n_actions)

    # 5. Root visit-count distribution: mean of per-step distributions over T.
    #    This is the per-asset visit-count summed across the trajectory and
    #    normalized — what the policy head learns to mimic.
    policy_target = jnp.mean(step_policies, axis=0)  # (n_actions,)

    # 6. Root value target — final cumulative PnL of the clean oracle path.
    value_target = clean_pnl[-1]  # scalar

    # 7. Per-step value targets — clean remaining PnL.
    #    remaining_pnl[t] = final_pnl - cumulative_pnl[t]  (TASK_670 fix in train_easy.py:757)
    step_values = clean_pnl[-1] - clean_pnl  # (T,)

    # 8. Per-step trajectory action targets — argmax of per-step visit distribution.
    #    Per-asset (T, n_assets) — this is the action id (0..4) the dynamics
    #    block should be unrolled against at each step.
    step_actions = jnp.argmax(step_policies_per_asset, axis=-1).astype(jnp.int32)

    # 9. Per-action Q-values.
    #    Phase-1 simplification: per-pass realized PnL is identical across passes
    #    (only the clean path provides PnL trajectories), so Q at a visited slot
    #    reduces to clean_pnl[-1] * (slot was visited at any step).
    #    Aggregated: Q[a] = clean_pnl[-1] * (1 if any pass at any step visited a, else 0).
    any_visit = jnp.max(step_visit_counts, axis=0) > 0  # (n_assets, n_actions_per_asset)
    q_values = (
        any_visit.astype(jnp.float32).reshape(n_actions) * value_target
    )  # (n_actions,)

    # Note: clean_selections is currently unused by the math above but is
    # preserved as a return value for downstream consumers (TASK_689 state
    # channel computation, AR rollout dispatching) and to keep this helper's
    # signature aligned with the canonical multipass return tuple. Phase 2
    # may use it as a fallback dynamics-action target if the per-step argmax
    # of step_policies turns out to be too noisy.
    _ = clean_selections  # explicit acknowledgement; not silently dropped.

    return {
        'policy_target': policy_target,
        'value_target': value_target,
        'q_values': q_values,
        'step_policies': step_policies,
        'step_values': step_values,
        'step_actions': step_actions,
        'visit_counts': step_visit_counts_flat,
    }


# =============================================================================
# Batched helper: oracle_to_mcts_targets across N episodes
# =============================================================================

@partial(jax.jit, static_argnames=['n_assets', 'n_actions_per_asset'])
def batched_oracle_to_mcts_targets(
    all_policies_batch: jnp.ndarray,
    clean_pnl_batch: jnp.ndarray,
    clean_selections_batch: jnp.ndarray,
    n_assets: int = 64,
    n_actions_per_asset: int = 5,
) -> Dict[str, jnp.ndarray]:
    """Vmap of `oracle_to_mcts_targets` over N episodes.

    Parameters
    ----------
    all_policies_batch : (N, n_passes, T, n_assets * n_actions_per_asset)
    clean_pnl_batch    : (N, T)
    clean_selections_batch : (N, T, n_assets)

    Returns
    -------
    Dict with the same keys as `oracle_to_mcts_targets`, each with a leading
    batch axis of size N.
    """
    vmapped = jax.vmap(
        lambda ap, cp, cs: oracle_to_mcts_targets(
            ap, cp, cs, n_assets=n_assets, n_actions_per_asset=n_actions_per_asset,
        )
    )
    return vmapped(all_policies_batch, clean_pnl_batch, clean_selections_batch)
