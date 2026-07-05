"""
Tree Search Setup Module (H1 - Phase 3 Predictive Planning)

Handles initialization and configuration of physics-aware tree search
for MCTS reward adjustment. Extracted from train_jax_fast.py.bak.
"""

import jax
import jax.numpy as jnp
from typing import Optional, Callable, Tuple, Any

from mcts_trader.training.config import TrainingConfig


def setup_tree_search(
    config: TrainingConfig,
    model: Any,
    n_assets: int,
    rng: jnp.ndarray,
) -> Tuple[Optional[Callable], Optional[Callable], bool]:
    """
    Initialize tree search components based on config.

    Args:
        config: Training configuration
        model: MuZero model (for checking prediction head)
        n_assets: Number of assets
        rng: Random key

    Returns:
        tree_search_fn: Tree search function (or None)
        risk_adjust_fn: Risk adjustment function (or None)
        uses_muzero_prediction: Whether using trained model params
    """
    if not config.use_tree_search:
        return None, None, False

    # Import lazily to avoid circular imports
    from mcts_trader.jax_agent.prediction import (
        NeuralMarketODE,
        create_physics_aware_tree_search_fn,
        create_mcts_reward_adjuster,
        create_dynamic_reward_adjuster,
        TreeSearchConfig,
    )

    tree_mode = "PHYSICS-ONLY" if config.physics_only else "HYBRID"
    print(f"🌳 Initializing Predictive Tree Search ({tree_mode})...")

    tree_config = TreeSearchConfig(
        depth=config.tree_depth,
        branches=config.tree_branches,
        store_trajectory=False,
    )
    n_leaves = config.tree_branches ** config.tree_depth

    tree_search_fn = None
    risk_adjust_fn = None
    uses_muzero_prediction = False

    # Check if we can use MuZero's TRAINED prediction model
    if config.use_prediction_loss and getattr(model, 'use_prediction_head', False):
        # Use MuZero's prediction model (will be trained during training)
        print(f"   Using MuZero's trained prediction model (dynamic params)")
        uses_muzero_prediction = True

        # Create a standalone model with same config for tree search
        tree_pred_model = NeuralMarketODE(
            n_assets=n_assets,
            n_features=10,
            use_transformer=True,
            use_action_conditioning=True,
            physics_only=config.physics_only,
        )

        # Create dynamic reward adjuster that takes params at runtime
        risk_adjust_fn = create_dynamic_reward_adjuster(
            tree_pred_model,
            tree_config,
            risk_lambda=config.risk_lambda
        )
        print(f"   Depth={config.tree_depth}, Branches={config.tree_branches} → {n_leaves} leaves")
        print(f"   Risk λ={config.risk_lambda} (uncertainty penalty weight)")
        print(f"   ⚡ Params will update with training!")
    else:
        # Standalone (untrained) prediction model
        if config.use_prediction_loss:
            print(f"   ⚠️ use_prediction_head not enabled, using standalone model")
        else:
            print(f"   ⚠️ Using untrained prediction model (add --use-prediction-loss for better results)")

        pred_model = NeuralMarketODE(
            n_assets=n_assets,
            n_features=10,
            use_transformer=True,
            use_action_conditioning=False,
            physics_only=config.physics_only,
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
        risk_adjust_fn = create_mcts_reward_adjuster(tree_search_fn, risk_lambda=config.risk_lambda)

        print(f"   Depth={config.tree_depth}, Branches={config.tree_branches} → {n_leaves} leaves")
        print(f"   Risk λ={config.risk_lambda} (uncertainty penalty weight)")

    return tree_search_fn, risk_adjust_fn, uses_muzero_prediction
