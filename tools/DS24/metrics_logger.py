#!/usr/bin/env python3
"""
DS24 Metrics Logger - Enhanced Training Metrics Collection

This module provides helper functions to log the additional metrics
required by the DS24 Enhanced Dashboard for:
- TRM recursive reasoning dynamics
- Neural ODE physics components
- MCTS planning diagnostics
- Multi-component interactions

Integration:
    from tools.DS24.metrics_logger import DS24MetricsLogger

    # In your training loop:
    logger = DS24MetricsLogger(log_path='logs/training_myrun.csv')

    # After each epoch:
    logger.log_trm_dynamics(latent_deltas, output_deltas)
    logger.log_physics_components(physics_output)
    logger.log_planning_metrics(mcts_output, actual_reward)
"""

import os
import csv
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TRMDynamics:
    """TRM cycle dynamics metrics."""
    latent_delta_norms: Dict[str, float] = field(default_factory=dict)  # h0_l0, h0_l1, etc.
    output_delta_norms: Dict[str, float] = field(default_factory=dict)  # h0, h1, etc.
    latent_final_norm: float = 0.0
    trm_utilization: float = 0.0  # % of cycles with delta > threshold


@dataclass
class PhysicsMetrics:
    """Neural ODE physics component metrics."""
    G_momentum_mean: float = 0.0
    G_momentum_std: float = 0.0
    G_reversion_mean: float = 0.0
    G_reversion_std: float = 0.0
    G_volatility_mean: float = 0.0
    G_volatility_std: float = 0.0
    G_correlation_mean: float = 0.0
    G_correlation_std: float = 0.0
    physics_alpha_mean: float = 0.0
    physics_beta_mean: float = 0.0
    garch_persistence: float = 0.0


@dataclass
class PlanningMetrics:
    """MCTS planning diagnostic metrics."""
    mcts_root_value: float = 0.0
    mcts_visit_entropy: float = 0.0
    value_horizon_1_error: float = 0.0
    value_horizon_5_error: float = 0.0
    value_horizon_10_error: float = 0.0


class DS24MetricsLogger:
    """
    Enhanced metrics logger for DS24 dashboard.

    Extends standard training CSV logging with additional columns for:
    - TRM dynamics (latent/output deltas per cycle)
    - Physics components (G_momentum, G_reversion, etc.)
    - Planning metrics (horizon errors, visit entropy)

    Usage:
        logger = DS24MetricsLogger('logs/training_run.csv')

        # Standard logging (extend existing)
        logger.log_epoch(epoch_data)

        # Enhanced logging
        logger.log_trm_dynamics(trm_metrics)
        logger.log_physics(physics_metrics)
        logger.log_planning(planning_metrics)

        # Write combined row
        logger.write_row()
    """

    # Additional columns for DS24
    DS24_COLUMNS = [
        # TRM Dynamics - Full H×L matrix for heatmap (4 H-cycles × 6 L-cycles = 24)
        'latent_delta_norm_h0_l0', 'latent_delta_norm_h0_l1', 'latent_delta_norm_h0_l2',
        'latent_delta_norm_h0_l3', 'latent_delta_norm_h0_l4', 'latent_delta_norm_h0_l5',
        'latent_delta_norm_h1_l0', 'latent_delta_norm_h1_l1', 'latent_delta_norm_h1_l2',
        'latent_delta_norm_h1_l3', 'latent_delta_norm_h1_l4', 'latent_delta_norm_h1_l5',
        'latent_delta_norm_h2_l0', 'latent_delta_norm_h2_l1', 'latent_delta_norm_h2_l2',
        'latent_delta_norm_h2_l3', 'latent_delta_norm_h2_l4', 'latent_delta_norm_h2_l5',
        'latent_delta_norm_h3_l0', 'latent_delta_norm_h3_l1', 'latent_delta_norm_h3_l2',
        'latent_delta_norm_h3_l3', 'latent_delta_norm_h3_l4', 'latent_delta_norm_h3_l5',
        # TRM Dynamics - Aggregated per H-cycle (for line charts)
        'latent_delta_norm_h0', 'latent_delta_norm_h1',
        'latent_delta_norm_h2', 'latent_delta_norm_h3',
        'output_delta_norm_h0', 'output_delta_norm_h1',
        'output_delta_norm_h2', 'output_delta_norm_h3',
        'latent_final_norm', 'trm_utilization',

        # Physics Components
        'G_momentum_mean', 'G_momentum_std',
        'G_reversion_mean', 'G_reversion_std',
        'G_volatility_mean', 'G_volatility_std',
        'G_correlation_mean', 'G_correlation_std',
        'physics_alpha_mean', 'physics_beta_mean',
        'garch_persistence',

        # Planning Metrics
        'mcts_root_value', 'mcts_visit_entropy',
        'value_horizon_1_error', 'value_horizon_5_error', 'value_horizon_10_error',

        # Gradient Analysis
        'grad_norm_value', 'grad_norm_policy', 'grad_norm_dynamics',

        # TASK_022: GAE Returns Config
        'ds24_use_gae', 'ds24_gae_lambda',

        # TASK_025: Per-head Gradient Clipping
        'ds24_per_head_clipping',
        'ds24_clip_value', 'ds24_clip_policy',
        'ds24_clip_dynamics', 'ds24_clip_encoder',
        # Per-head gradient norms AFTER clipping (to verify clipping works)
        'grad_norm_value_clipped', 'grad_norm_policy_clipped',
        'grad_norm_dynamics_clipped', 'grad_norm_encoder_clipped',

        # Exploration
        'ds24_epsilon_greedy',

        # TASK_042: ReDo Dormancy Stats
        'redo_dormant_neurons', 'redo_dormant_pct', 'redo_recycled_count',

        # TASK_043: ATC Contrastive Loss
        'atc_loss', 'atc_pos_sim', 'atc_neg_sim', 'atc_temperature',

        # Note: 'alpha', 'market_return', 'agent_return' are in base columns
        # Do NOT duplicate them here - causes pandas issues with duplicate columns
    ]

    # Level descriptions for documentation
    LEVEL_DESCRIPTIONS = {
        0: "OFF - No DS24 logging",
        1: "CSV + Market baseline (alpha, market_return, agent_return)",
        2: "Level 1 + TRM intermediates + MCTS planning metrics",
        3: "Level 2 + Physics forward pass (G_momentum, G_reversion, etc.)",
        4: "Level 3 + Latent space heatmaps + weight monitoring",
        5: "Level 4 + 3D visualization",
        6: "ShroomEngine - Full model introspection with attention maps, gradient flow, decision trees",
    }

    def __init__(
        self,
        log_path: Optional[str] = None,
        tensor_dir: Optional[str] = None,
        snapshot_every: int = 100,
        level: int = 2
    ):
        """
        Initialize DS24 metrics logger.

        Args:
            log_path: Path to CSV log file (appends DS24 columns)
            tensor_dir: Directory for periodic tensor snapshots
            snapshot_every: Save tensor snapshot every N epochs
            level: Logging level (0-5):
                0 = OFF
                1 = CSV only
                2 = +TRM/planning (default)
                3 = +physics forward pass
                4 = +heatmaps
                5 = +3D visualization
        """
        self.log_path = Path(log_path) if log_path else None
        self.tensor_dir = Path(tensor_dir) if tensor_dir else None
        self.snapshot_every = snapshot_every
        self.level = level

        self.current_row: Dict[str, Any] = {}
        self.epoch = 0

        if self.tensor_dir:
            self.tensor_dir.mkdir(parents=True, exist_ok=True)

    def reset_row(self):
        """Reset current row data for new epoch."""
        self.current_row = {col: None for col in self.DS24_COLUMNS}

    def log_trm_dynamics(
        self,
        latent_deltas: Optional[Dict[str, np.ndarray]] = None,
        output_deltas: Optional[Dict[str, np.ndarray]] = None,
        latent_final: Optional[np.ndarray] = None,
        utilization_threshold: float = 0.01
    ):
        """
        Log TRM recursive reasoning dynamics.

        Args:
            latent_deltas: Dict mapping cycle names (h0_l0, etc.) to delta arrays
            output_deltas: Dict mapping H cycles (h0, etc.) to output delta arrays
            latent_final: Final latent representation
            utilization_threshold: Min delta norm to count as "utilized"
        """
        if latent_deltas:
            total_cycles = 0
            utilized_cycles = 0

            # Log individual H×L cycle norms for heatmap AND aggregate per H-cycle
            # Keys are like 'h0_l0', 'h0_l1', etc.
            h_cycle_norms = {}

            for key, delta in latent_deltas.items():
                norm = float(np.mean(np.linalg.norm(delta, axis=-1)))

                # 1. Log individual H×L value for heatmap (e.g., latent_delta_norm_h0_l0)
                col_name_full = f'latent_delta_norm_{key}'
                if col_name_full in self.current_row:
                    self.current_row[col_name_full] = norm

                # 2. Aggregate norms per H-cycle for line charts
                # Extract H-cycle index (e.g., 'h0_l2' -> 'h0')
                h_key = key.split('_')[0] if '_' in key else key
                if h_key not in h_cycle_norms:
                    h_cycle_norms[h_key] = 0.0
                h_cycle_norms[h_key] += norm

                total_cycles += 1
                if norm > utilization_threshold:
                    utilized_cycles += 1

            # Log aggregated H-cycle norms for line charts
            for h_key, agg_norm in h_cycle_norms.items():
                col_name = f'latent_delta_norm_{h_key}'
                if col_name in self.current_row:
                    self.current_row[col_name] = agg_norm

            # TRM utilization: % of cycles being used
            if total_cycles > 0:
                self.current_row['trm_utilization'] = utilized_cycles / total_cycles * 100

        if output_deltas:
            for key, delta in output_deltas.items():
                norm = float(np.mean(np.linalg.norm(delta, axis=-1)))
                col_name = f'output_delta_norm_{key}'
                if col_name in self.current_row:
                    self.current_row[col_name] = norm

        if latent_final is not None:
            self.current_row['latent_final_norm'] = float(np.mean(np.linalg.norm(latent_final, axis=-1)))

    def log_physics_components(
        self,
        G_momentum: Optional[np.ndarray] = None,
        G_reversion: Optional[np.ndarray] = None,
        G_volatility: Optional[np.ndarray] = None,
        G_correlation: Optional[np.ndarray] = None,
        alpha: Optional[np.ndarray] = None,
        beta: Optional[np.ndarray] = None,
        garch_alpha: Optional[np.ndarray] = None,
        garch_beta: Optional[np.ndarray] = None
    ):
        """
        Log Neural ODE physics component metrics.

        Args:
            G_momentum: Momentum component array
            G_reversion: Mean-reversion component array
            G_volatility: Volatility component array
            G_correlation: Correlation component array
            alpha: Momentum coefficient array
            beta: Reversion coefficient array
            garch_alpha: GARCH alpha parameter
            garch_beta: GARCH beta parameter
        """
        if G_momentum is not None:
            self.current_row['G_momentum_mean'] = float(np.mean(G_momentum))
            self.current_row['G_momentum_std'] = float(np.std(G_momentum))

        if G_reversion is not None:
            self.current_row['G_reversion_mean'] = float(np.mean(G_reversion))
            self.current_row['G_reversion_std'] = float(np.std(G_reversion))

        if G_volatility is not None:
            self.current_row['G_volatility_mean'] = float(np.mean(G_volatility))
            self.current_row['G_volatility_std'] = float(np.std(G_volatility))

        if G_correlation is not None:
            self.current_row['G_correlation_mean'] = float(np.mean(G_correlation))
            self.current_row['G_correlation_std'] = float(np.std(G_correlation))

        if alpha is not None:
            self.current_row['physics_alpha_mean'] = float(np.mean(alpha))

        if beta is not None:
            self.current_row['physics_beta_mean'] = float(np.mean(beta))

        if garch_alpha is not None and garch_beta is not None:
            self.current_row['garch_persistence'] = float(np.mean(garch_alpha) + np.mean(garch_beta))

    def log_planning_metrics(
        self,
        root_values: Optional[np.ndarray] = None,
        visit_counts: Optional[np.ndarray] = None,
        actual_rewards: Optional[np.ndarray] = None,
        horizon_predictions: Optional[Dict[int, np.ndarray]] = None,
        horizon_actuals: Optional[Dict[int, np.ndarray]] = None
    ):
        """
        Log MCTS planning diagnostic metrics.

        Args:
            root_values: MCTS root node values
            visit_counts: Child visit counts for entropy calculation
            actual_rewards: Actual rewards for calibration
            horizon_predictions: Predicted values at different horizons
            horizon_actuals: Actual values at different horizons
        """
        if root_values is not None:
            self.current_row['mcts_root_value'] = float(np.mean(root_values))

        if visit_counts is not None:
            # Calculate visit entropy (exploration indicator)
            probs = visit_counts / (visit_counts.sum(axis=-1, keepdims=True) + 1e-8)
            entropy = -np.sum(probs * np.log(probs + 1e-8), axis=-1)
            self.current_row['mcts_visit_entropy'] = float(np.mean(entropy))

        if horizon_predictions and horizon_actuals:
            for horizon in [1, 5, 10]:
                if horizon in horizon_predictions and horizon in horizon_actuals:
                    error = float(np.mean((horizon_predictions[horizon] - horizon_actuals[horizon]) ** 2))
                    self.current_row[f'value_horizon_{horizon}_error'] = error

    def log_gradient_norms(
        self,
        value_grad_norm: Optional[float] = None,
        policy_grad_norm: Optional[float] = None,
        dynamics_grad_norm: Optional[float] = None,
        # Clipped norms (TASK_025)
        value_clipped: Optional[float] = None,
        policy_clipped: Optional[float] = None,
        dynamics_clipped: Optional[float] = None,
        encoder_clipped: Optional[float] = None
    ):
        """
        Log gradient norms for multi-task analysis.

        Args:
            value_grad_norm: Gradient norm from value loss
            policy_grad_norm: Gradient norm from policy loss
            dynamics_grad_norm: Gradient norm from dynamics/prediction loss
            value_clipped: Value grad norm after per-head clipping
            policy_clipped: Policy grad norm after per-head clipping
            dynamics_clipped: Dynamics grad norm after per-head clipping
            encoder_clipped: Encoder grad norm after per-head clipping
        """
        if value_grad_norm is not None:
            self.current_row['grad_norm_value'] = value_grad_norm
        if policy_grad_norm is not None:
            self.current_row['grad_norm_policy'] = policy_grad_norm
        if dynamics_grad_norm is not None:
            self.current_row['grad_norm_dynamics'] = dynamics_grad_norm
        # Per-head clipped norms
        if value_clipped is not None:
            self.current_row['grad_norm_value_clipped'] = value_clipped
        if policy_clipped is not None:
            self.current_row['grad_norm_policy_clipped'] = policy_clipped
        if dynamics_clipped is not None:
            self.current_row['grad_norm_dynamics_clipped'] = dynamics_clipped
        if encoder_clipped is not None:
            self.current_row['grad_norm_encoder_clipped'] = encoder_clipped

    def log_training_config(
        self,
        use_gae: bool = False,
        gae_lambda: float = 0.95,
        per_head_clipping: bool = False,
        clip_value: float = 1.0,
        clip_policy: float = 1.0,
        clip_dynamics: float = 1.0,
        clip_encoder: float = 1.0,
        epsilon_greedy: float = 0.0
    ):
        """
        Log training configuration flags (TASK_022, TASK_025).

        Args:
            use_gae: Whether GAE returns are enabled
            gae_lambda: GAE lambda parameter
            per_head_clipping: Whether per-head gradient clipping is enabled
            clip_value: Clip threshold for value head
            clip_policy: Clip threshold for policy head
            clip_dynamics: Clip threshold for dynamics head
            clip_encoder: Clip threshold for encoder
            epsilon_greedy: Epsilon-greedy mixing ratio
        """
        self.current_row['ds24_use_gae'] = 1 if use_gae else 0
        self.current_row['ds24_gae_lambda'] = gae_lambda if use_gae else None
        self.current_row['ds24_per_head_clipping'] = 1 if per_head_clipping else 0
        if per_head_clipping:
            self.current_row['ds24_clip_value'] = clip_value
            self.current_row['ds24_clip_policy'] = clip_policy
            self.current_row['ds24_clip_dynamics'] = clip_dynamics
            self.current_row['ds24_clip_encoder'] = clip_encoder
        self.current_row['ds24_epsilon_greedy'] = epsilon_greedy

    def log_redo_stats(
        self,
        dormant_neurons: int = 0,
        dormant_pct: float = 0.0,
        recycled_count: int = 0
    ):
        """
        Log ReDo dormancy statistics (TASK_042).

        Args:
            dormant_neurons: Number of dormant neurons detected
            dormant_pct: Percentage of neurons that are dormant
            recycled_count: Number of neurons recycled this epoch
        """
        self.current_row['redo_dormant_neurons'] = dormant_neurons
        self.current_row['redo_dormant_pct'] = dormant_pct
        self.current_row['redo_recycled_count'] = recycled_count

    def log_atc_stats(
        self,
        loss: float = 0.0,
        pos_sim: float = 0.0,
        neg_sim: float = 0.0,
        temperature: float = 0.1
    ):
        """
        Log ATC contrastive loss statistics (TASK_043).

        Args:
            loss: Contrastive loss value
            pos_sim: Positive pair similarity
            neg_sim: Negative pair similarity
            temperature: Temperature parameter
        """
        self.current_row['atc_loss'] = loss
        self.current_row['atc_pos_sim'] = pos_sim
        self.current_row['atc_neg_sim'] = neg_sim
        self.current_row['atc_temperature'] = temperature

    def log_market_baseline(
        self,
        alpha: Optional[float] = None,
        market_return: Optional[float] = None,
        agent_return: Optional[float] = None
    ):
        """
        Log market baseline metrics for alpha analysis.

        Alpha = agent_return - market_return:
          - Positive alpha: Agent beating buy-and-hold benchmark
          - Negative alpha: Agent underperforming market
          - Zero alpha: Agent matching market (no edge)

        Args:
            alpha: Agent return minus market return (percentage)
            market_return: Buy-and-hold benchmark return (percentage)
            agent_return: Cumulative agent return from trading (percentage)
        """
        if alpha is not None:
            self.current_row['alpha'] = alpha
        if market_return is not None:
            self.current_row['market_return'] = market_return
        if agent_return is not None:
            self.current_row['agent_return'] = agent_return

    def save_tensor_snapshot(
        self,
        epoch: int,
        run_name: str,
        latent_states: Optional[np.ndarray] = None,
        correlation_matrix: Optional[np.ndarray] = None,
        attention_weights: Optional[np.ndarray] = None,
        **extra_tensors
    ):
        """
        Save periodic tensor snapshots for detailed analysis.

        Args:
            epoch: Current epoch number
            run_name: Experiment name
            latent_states: Batch of latent state vectors
            correlation_matrix: Learned correlation matrix
            attention_weights: Attention weight matrices
            **extra_tensors: Additional tensors to save
        """
        if not self.tensor_dir:
            return

        # Note: epoch filtering is done by caller (training_hooks._save_tensor_snapshot)

        run_dir = self.tensor_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        save_dict = {}

        if latent_states is not None:
            save_dict['latent_states'] = latent_states

        if correlation_matrix is not None:
            save_dict['correlation_matrix'] = correlation_matrix

        if attention_weights is not None:
            save_dict['attention_weights'] = attention_weights

        save_dict.update(extra_tensors)

        if save_dict:
            snapshot_path = run_dir / f'epoch_{epoch:06d}.npz'
            np.savez_compressed(str(snapshot_path), **save_dict)

    def get_ds24_columns(self) -> List[str]:
        """Return list of DS24-specific column names."""
        return self.DS24_COLUMNS.copy()

    def get_current_values(self) -> Dict[str, Any]:
        """Return current row values for DS24 columns."""
        return {k: v for k, v in self.current_row.items() if v is not None}


# =============================================================================
# Helper Functions for Integration
# =============================================================================

def compute_trm_deltas(
    model_apply_fn,
    params,
    state: np.ndarray,
    H_cycles: int = 4,
    L_cycles: int = 6
) -> Dict[str, np.ndarray]:
    """
    Compute TRM latent deltas by running model with intermediate outputs.

    This is a template - actual implementation depends on model architecture.

    Args:
        model_apply_fn: Model apply function
        params: Model parameters
        state: Input state tensor
        H_cycles: Number of H cycles
        L_cycles: Number of L cycles

    Returns:
        Dict with latent_deltas and output_deltas
    """
    # This would need to be implemented based on actual TRM architecture
    # to capture intermediate deltas during the recursive refinement

    # Placeholder return
    return {
        'latent_deltas': {f'h{h}': np.zeros((state.shape[0], 64)) for h in range(H_cycles)},
        'output_deltas': {f'h{h}': np.zeros((state.shape[0], 320)) for h in range(H_cycles)}
    }


def compute_physics_outputs(
    physics_model,
    params,
    log_prices: np.ndarray,
    log_returns: np.ndarray,
    moving_avg: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compute physics component outputs for logging.

    Args:
        physics_model: DifferentiableMarketPhysics model
        params: Model parameters
        log_prices: Current log prices
        log_returns: Recent log returns
        moving_avg: Moving average of log prices

    Returns:
        Dict with G_momentum, G_reversion, G_volatility, G_correlation
    """
    # Placeholder - actual implementation would call physics model
    return {
        'G_momentum': np.zeros_like(log_prices),
        'G_reversion': np.zeros_like(log_prices),
        'G_volatility': np.zeros_like(log_prices),
        'G_correlation': np.zeros_like(log_prices)
    }


# =============================================================================
# Example Integration
# =============================================================================

def example_training_loop_integration():
    """
    Example showing how to integrate DS24 logging into train_jax_fast.py.

    Add this to your training loop:
    """
    example_code = '''
    # At the top of train_jax_fast.py:
    from tools.DS24.metrics_logger import DS24MetricsLogger

    # After creating CSV log path:
    ds24_logger = DS24MetricsLogger(
        log_path=csv_log_path,
        tensor_dir='tensor_snapshots',
        snapshot_every=100
    )

    # In the training loop, after computing standard metrics:
    ds24_logger.reset_row()

    # Log TRM dynamics (if available)
    if hasattr(model, 'get_intermediate_outputs'):
        trm_outputs = model.get_intermediate_outputs(params, observations)
        ds24_logger.log_trm_dynamics(
            latent_deltas=trm_outputs.get('latent_deltas'),
            output_deltas=trm_outputs.get('output_deltas'),
            latent_final=trm_outputs.get('latent_final')
        )

    # Log physics components (if prediction model enabled)
    if prediction_model is not None:
        physics_out = prediction_model.get_physics_outputs(params, market_state)
        ds24_logger.log_physics_components(
            G_momentum=physics_out.G_momentum,
            G_reversion=physics_out.G_reversion,
            G_volatility=physics_out.G_volatility,
            G_correlation=physics_out.G_correlation
        )

    # Log planning metrics
    ds24_logger.log_planning_metrics(
        root_values=mcts_output.root_value,
        visit_counts=mcts_output.visit_counts,
        actual_rewards=episode_rewards
    )

    # Get DS24 column values to append to CSV row
    ds24_values = ds24_logger.get_current_values()

    # Extend your CSV header and row with DS24 columns
    # (First epoch only for header)
    if epoch == 0:
        existing_columns.extend(ds24_logger.get_ds24_columns())

    # Add DS24 values to row
    row_values.update(ds24_values)

    # Save tensor snapshot periodically
    ds24_logger.save_tensor_snapshot(
        epoch=epoch,
        run_name=args.name,
        latent_states=final_latent_batch,
        correlation_matrix=physics_model.get_correlation_matrix(params)
    )
    '''
    return example_code


if __name__ == '__main__':
    # Print integration example
    print("DS24 Metrics Logger - Integration Example")
    print("=" * 60)
    print(example_training_loop_integration())
