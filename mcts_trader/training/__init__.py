"""
Training utilities for MCTS Trader.

Modules:
    - self_play: Episode generation (single-env, batched, JAX-vectorized)
    - evaluator: Generalization evaluation
    - env_factory: Environment creation utilities
    - safe_checkpoint: Checkpoint management with atomic saves
"""

from .safe_checkpoint import SafeCheckpointManager
from .self_play import run_self_play, run_self_play_batched, run_self_play_jax
from .evaluator import run_generalization_eval
from .env_factory import create_jax_env_and_prices

__all__ = [
    'SafeCheckpointManager',
    'run_self_play',
    'run_self_play_batched',
    'run_self_play_jax',
    'run_generalization_eval',
    'create_jax_env_and_prices',
]
