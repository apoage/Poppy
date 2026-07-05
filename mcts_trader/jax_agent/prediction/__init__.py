"""
Prediction Model Module

Physics-based market prediction using Neural ODE approach.
Based on NeuralOGCM (arXiv:2512.11525v1) adapted for financial markets.

Components:
- physics_core: Differentiable market physics (momentum, reversion, volatility, correlation)
- neural_corrector: Neural network corrections using cross-asset transformer
- market_ode: Combined ODE model with action conditioning

All operations in LOG-SPACE for numerical stability.

Usage:
    from mcts_trader.jax_agent.prediction import (
        NeuralMarketODE,
        create_neural_market_ode,
        DifferentiableMarketPhysics,
        NeuralMarketCorrector,
    )

    # Create and initialize model
    model, params = create_neural_market_ode(
        n_assets=64,
        n_features=10,
        ode_method='rk2',
        use_transformer=True,
    )

    # Single step prediction
    next_log_prices, aux = model.apply(
        params,
        log_prices,
        log_returns,
        moving_avg,
        noise,
        features,
        action,
    )

    # Multi-step rollout
    trajectory, _ = model.apply(
        params,
        log_prices,
        log_returns,
        moving_avg,
        rng,
        features=features,
        action_sequence=actions,
        num_steps=10,
        method=model.rollout,
    )
"""

# Physics Core (TASK_008)
from .physics_core import (
    DifferentiableMarketPhysics,
    LowRankCorrelation,
    GARCHVolatility,
    PhysicsOutput,
    physics_regularization_loss,
    create_physics_core,
)

# Neural Corrector (TASK_009)
from .neural_corrector import (
    NeuralMarketCorrector,
    SimpleNeuralCorrector,
    CrossAssetAttention,
    CrossAssetTransformerBlock,
    create_neural_corrector,
)

# Market ODE (TASK_010)
from .market_ode import (
    NeuralMarketODE,
    MarketImpact,
    ODEOutput,
    create_neural_market_ode,
    create_training_loss_fn,
)

# Student-t Distribution (TASK_116)
from .student_t import (
    student_t_nll,
    student_t_loss,
    StudentTHead,
    hybrid_prediction_loss,
    maybe_student_t_loss,
)

# Tree Search (TASK_012)
from .tree_search import (
    TreeSearchConfig,
    TreeSearchResult,
    StructuralScenarioConfig,
    create_physics_aware_tree_search_fn,
    create_structural_scenario_search_fn,
    create_tree_search_fn,
    create_batched_tree_search_fn,
    create_risk_penalty_fn,
    create_mcts_reward_adjuster,
    analyze_tree_results,
    # Dynamic params versions (for trained models)
    create_dynamic_tree_search_fn,
    create_dynamic_reward_adjuster,
)

__all__ = [
    # Student-t (TASK_116)
    'student_t_nll',
    'student_t_loss',
    'StudentTHead',
    'hybrid_prediction_loss',
    'maybe_student_t_loss',
    # Physics Core
    'DifferentiableMarketPhysics',
    'LowRankCorrelation',
    'GARCHVolatility',
    'PhysicsOutput',
    'physics_regularization_loss',
    'create_physics_core',
    # Neural Corrector
    'NeuralMarketCorrector',
    'SimpleNeuralCorrector',
    'CrossAssetAttention',
    'CrossAssetTransformerBlock',
    'create_neural_corrector',
    # Market ODE
    'NeuralMarketODE',
    'MarketImpact',
    'ODEOutput',
    'create_neural_market_ode',
    'create_training_loss_fn',
    # Tree Search
    'TreeSearchConfig',
    'TreeSearchResult',
    'StructuralScenarioConfig',
    'create_physics_aware_tree_search_fn',
    'create_structural_scenario_search_fn',
    'create_tree_search_fn',
    'create_batched_tree_search_fn',
    'create_risk_penalty_fn',
    'create_mcts_reward_adjuster',
    'analyze_tree_results',
    # Dynamic params (for trained models)
    'create_dynamic_tree_search_fn',
    'create_dynamic_reward_adjuster',
]
