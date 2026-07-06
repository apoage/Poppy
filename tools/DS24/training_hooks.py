#!/usr/bin/env python3
"""
DS24 Training Hooks - Level-based training integration for DS24 dashboard.

This module provides clean hooks for training loops with level-based logging:
  Level 0: OFF - No logging
  Level 1: CSV + Market baseline (alpha, market_return, agent_return)
  Level 2: Level 1 + TRM intermediates + MCTS planning metrics
  Level 3: Level 2 + Physics forward pass (G_momentum, G_reversion, etc.)
  Level 4: Level 3 + Latent space heatmaps + weight monitoring
  Level 5: Level 4 + 3D visualization (future)

Usage:
    from tools.DS24.training_hooks import DS24TrainingHooks

    # Initialize once at training start
    hooks = DS24TrainingHooks(
        level=3,
        model=model,
        intermediates_every=10,
        save_tensors=True
    )

    # Call at end of each epoch
    hooks.on_epoch_end(
        epoch=epoch,
        trajectory=trajectory,
        trainer_state=trainer.state,
        args=args
    )

    # Get column values for CSV
    ds24_values = hooks.get_current_values()
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

# Lazy imports to avoid circular dependencies
jnp = None
jax = None


def _ensure_jax():
    """Lazy import JAX modules."""
    global jnp, jax
    if jnp is None:
        import jax as _jax
        import jax.numpy as _jnp
        jax = _jax
        jnp = _jnp


@dataclass
class DS24Config:
    """Configuration for DS24 hooks."""
    level: int = 2                    # Base logging level
    intermediates_every: int = 10
    save_tensors: bool = False
    tensor_dir: Optional[str] = None
    snapshot_every: int = 100
    h_cycles: int = 4
    l_cycles: int = 6
    run_name: str = "default"
    # Dynamic level boosting (reduces overhead while getting periodic deep insights)
    boost_every: int = 10             # Boost to L4 every N epochs (0=disabled)
    full_snapshot_every: int = 50     # Full L8 snapshot every M epochs (0=disabled)


class DS24TrainingHooks:
    """
    Level-based training hooks for DS24 dashboard integration.

    Provides clean separation between training loop and logging logic.
    All level-based conditional logic is encapsulated here.
    """

    LEVEL_DESCRIPTIONS = {
        0: "OFF",
        1: "CSV + market baseline (alpha)",
        2: "+TRM/planning",
        3: "+physics forward",
        4: "+heatmaps",
        5: "+3D visualization",
        6: "ShroomEngine",  # Full introspection mode
    }

    def __init__(
        self,
        level: int = 3,
        model: Any = None,
        intermediates_every: int = 10,
        save_tensors: bool = False,
        tensor_dir: Optional[str] = None,
        h_cycles: int = 4,
        l_cycles: int = 6,
        run_name: str = "default",
        boost_every: int = 10,
        full_snapshot_every: int = 50
    ):
        """
        Initialize DS24 training hooks.

        Args:
            level: Base logging level (0-6), default 3
            model: MuZero model (needed for level 2+)
            intermediates_every: Log TRM intermediates every N epochs
            save_tensors: Save tensor snapshots
            tensor_dir: Directory for tensor snapshots
            h_cycles: Number of TRM H cycles
            l_cycles: Number of TRM L cycles
            run_name: Experiment name for snapshots
            boost_every: Boost to L4 every N epochs (0=disabled)
            full_snapshot_every: Full L8 snapshot every M epochs (0=disabled)
        """
        self.base_level = level  # Store base level
        self.level = level       # Current effective level (may be boosted)
        self.model = model
        self.config = DS24Config(
            level=level,
            intermediates_every=intermediates_every,
            save_tensors=save_tensors,
            tensor_dir=tensor_dir,
            h_cycles=h_cycles,
            l_cycles=l_cycles,
            run_name=run_name,
            boost_every=boost_every,
            full_snapshot_every=full_snapshot_every
        )

        # Initialize metrics logger
        from .metrics_logger import DS24MetricsLogger
        self.logger = DS24MetricsLogger(
            tensor_dir=tensor_dir if save_tensors else None,
            snapshot_every=100,
            level=level
        )

        # Error counters (to avoid spamming logs)
        self._error_counts = {
            'trm': 0,
            'planning': 0,
            'physics': 0,
            'heatmap': 0
        }

    def get_level_description(self) -> str:
        """Get human-readable description of current level."""
        return f"Level {self.level} ({self.LEVEL_DESCRIPTIONS.get(self.level, 'unknown')})"

    def get_effective_level(self, epoch: int) -> Tuple[int, str]:
        """
        Get effective logging level for this epoch with dynamic boosting.

        Levels:
          - Base (L3): Normal training epochs
          - Boost (L4): Every boost_every epochs - more detailed heatmaps
          - Full (L6+): Every full_snapshot_every epochs - comprehensive snapshot

        Returns:
            Tuple of (level, reason_string)
        """
        # Check for full snapshot epoch first (takes priority)
        if self.config.full_snapshot_every > 0 and epoch > 0 and epoch % self.config.full_snapshot_every == 0:
            return 6, f"📸 Full snapshot (every {self.config.full_snapshot_every})"

        # Check for L4 boost epoch
        if self.config.boost_every > 0 and epoch > 0 and epoch % self.config.boost_every == 0:
            return max(4, self.base_level), f"📊 L4 boost (every {self.config.boost_every})"

        # Default: base level
        return self.base_level, ""

    def get_ds24_columns(self) -> List[str]:
        """Get list of DS24 column names for CSV header."""
        return self.logger.get_ds24_columns()

    def get_current_values(self) -> Dict[str, Any]:
        """Get current row values for CSV output."""
        return self.logger.get_current_values()

    def on_epoch_start(self, epoch: int):
        """Called at start of each training epoch."""
        self.logger.reset_row()
        self.logger.epoch = epoch

    def on_epoch_end(
        self,
        epoch: int,
        trajectory: Dict[str, Any],
        trainer_state: Any,
        args: Any = None,
        train_metrics: Optional[Dict[str, Any]] = None,
        redo_stats: Optional[Dict[str, Any]] = None,
        atc_stats: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Called at end of each training epoch.

        This is the main entry point for DS24 logging.
        All level-based logic is handled internally.
        Dynamic level boosting: L3 default, L4 every boost_every, L6 every full_snapshot_every.

        Args:
            epoch: Current epoch number
            trajectory: Self-play trajectory dict with observations, actions, etc.
            trainer_state: Trainer state with params
            args: Training args (optional, for h_cycles/l_cycles override)
            train_metrics: Training metrics dict (optional, for gradient norms)
            redo_stats: ReDo dormancy statistics (TASK_042)
            atc_stats: ATC contrastive loss statistics (TASK_043)

        Returns:
            Dict with logged metrics (for optional inspection)
        """
        self.on_epoch_start(epoch)

        # Dynamic level boosting: determine effective level for this epoch
        effective_level, boost_reason = self.get_effective_level(epoch)
        self.level = effective_level  # Update current level for this epoch

        if self.level == 0:
            return {}

        # Log boost reason if any
        if boost_reason:
            print(f"  {boost_reason}")

        logged_metrics = {}

        # Level 1+: Basic CSV metrics + Market baseline + Training config
        self._log_market_baseline(trajectory, train_metrics)
        self._log_training_config(args)

        # Log ReDo and ATC stats if provided (TASK_042, TASK_043)
        self._log_redo_stats(redo_stats)
        self._log_atc_stats(atc_stats)

        # Level 2+: TRM intermediates + planning metrics
        if self.level >= 2:
            self._log_trm_intermediates(epoch, trajectory, trainer_state, args)
            self._log_planning_metrics(epoch, trajectory)

        # Level 3+: Physics forward pass
        if self.level >= 3:
            self._log_physics_forward(epoch, trajectory, trainer_state)

        # Level 4+: Heatmaps, weight monitoring, gradient norms
        if self.level >= 4:
            self._log_heatmaps(epoch, trajectory, trainer_state)
            self._log_gradient_norms(train_metrics)

        # Level 6+: Full comprehensive snapshot (ShroomEngine territory)
        # Save tensor snapshots on: first epoch, regular interval, OR full_snapshot_every
        is_full_snapshot_epoch = (
            self.config.full_snapshot_every > 0 and
            epoch > 0 and
            epoch % self.config.full_snapshot_every == 0
        )
        should_save_tensors = (
            self.config.save_tensors and (epoch == 1 or epoch % self.config.snapshot_every == 0)
        ) or is_full_snapshot_epoch

        if should_save_tensors:
            self._save_tensor_snapshot(epoch, trajectory, trainer_state)

        return self.get_current_values()

    def _log_trm_intermediates(
        self,
        epoch: int,
        trajectory: Dict[str, Any],
        trainer_state: Any,
        args: Any
    ):
        """Level 2+: Log TRM recursive reasoning intermediates."""
        if epoch % self.config.intermediates_every != 0:
            return

        if self.model is None:
            return

        try:
            _ensure_jax()

            # Get sample observation
            observations = trajectory.get('observations')
            if observations is None or len(observations) == 0:
                return

            sample_obs = jnp.array(observations[:1])

            # Get latent state from initial inference
            encoder_output = self.model.apply(
                {'params': trainer_state.params},
                sample_obs,
                mode='initial_inference'
            )
            latent_state = encoder_output.get('latent_state')

            if latent_state is None:
                return

            # Navigate to TRM reasoning params
            trm_params = trainer_state.params.get('trm', {})
            reasoning_params = trm_params.get('reasoning', trm_params)

            # Get cycles and model dimensions from args or config
            h_cycles = getattr(args, 'h_cycles', self.config.h_cycles) if args else self.config.h_cycles
            l_cycles = getattr(args, 'l_cycles', self.config.l_cycles) if args else self.config.l_cycles

            # Instantiate TRM reasoning module directly (Flax submodules can't be accessed from outside)
            from mcts_trader.jax_agent.trm_flax import TRMReasoning
            state_dim = latent_state.shape[-1]
            n_actions = 320  # Default for 64 assets × 5 actions

            reasoning_module = TRMReasoning(
                state_dim=state_dim,
                n_actions=n_actions
            )

            # Call TRM reasoning with return_intermediates=True
            _, _, _, trm_intermediates = reasoning_module.apply(
                {'params': reasoning_params},
                latent_state,
                h_cycles,
                l_cycles,
                return_intermediates=True
            )

            if trm_intermediates:
                self.logger.log_trm_dynamics(
                    latent_deltas={k: np.array(v) for k, v in trm_intermediates.get('latent_deltas', {}).items()},
                    output_deltas={k: np.array(v) for k, v in trm_intermediates.get('output_deltas', {}).items()},
                    latent_final=np.array(trm_intermediates.get('latent_final')) if trm_intermediates.get('latent_final') is not None else None
                )
                self._error_counts['trm'] = 0

        except Exception as e:
            self._error_counts['trm'] += 1
            if self._error_counts['trm'] <= 3 or epoch % 50 == 0:
                print(f"  ⚠️ DS24 TRM intermediates error: {e}")

    def _log_planning_metrics(self, epoch: int, trajectory: Dict[str, Any]):
        """Level 2+: Log MCTS planning metrics with horizon value errors."""
        try:
            mcts_root_values = trajectory.get('mcts_root_values')
            policy_probs = trajectory.get('policy_probs')
            rewards = trajectory.get('rewards')
            horizon_returns = trajectory.get('horizon_returns', {})

            if mcts_root_values is not None and len(mcts_root_values) > 0:
                # Basic planning metrics
                self.logger.log_planning_metrics(
                    root_values=np.array(mcts_root_values),
                    visit_counts=np.array(policy_probs) if policy_probs is not None else None,
                    actual_rewards=np.array(rewards) if rewards is not None else None
                )

                # Horizon value prediction errors
                # Compare MCTS root value predictions with actual k-step returns
                if horizon_returns:
                    horizon_predictions = {}
                    horizon_actuals = {}
                    mcts_vals = np.array(mcts_root_values)

                    for h in [1, 5, 10]:
                        if h in horizon_returns:
                            h_returns = horizon_returns[h]
                            # Align lengths (mcts values may be shorter)
                            min_len = min(len(mcts_vals), len(h_returns))
                            if min_len > 0:
                                horizon_predictions[h] = mcts_vals[:min_len]
                                horizon_actuals[h] = h_returns[:min_len]

                    if horizon_predictions:
                        self.logger.log_planning_metrics(
                            horizon_predictions=horizon_predictions,
                            horizon_actuals=horizon_actuals
                        )

                self._error_counts['planning'] = 0

        except Exception as e:
            self._error_counts['planning'] += 1
            if self._error_counts['planning'] <= 3 or epoch % 50 == 0:
                print(f"  ⚠️ DS24 planning metrics error: {e}")

    def _log_physics_forward(
        self,
        epoch: int,
        trajectory: Dict[str, Any],
        trainer_state: Any
    ):
        """Level 3+: Log physics components with forward pass."""
        if self.model is None:
            return

        if not hasattr(self.model, 'use_prediction_head') or not self.model.use_prediction_head:
            return

        try:
            _ensure_jax()
            from mcts_trader.jax_agent.prediction.physics_core import DifferentiableMarketPhysics

            pred_params = trainer_state.params.get('price_predictor', {})
            if not pred_params:
                return

            # Navigate to physics_core params
            physics_dict = pred_params.get('physics_core', pred_params)

            # 1. Extract learned parameters (α, β, GARCH)
            physics_info = DifferentiableMarketPhysics.get_learned_params_from_dict({'params': physics_dict})

            # 2. Run forward pass to get G components (if price data available)
            G_momentum, G_reversion, G_volatility, G_correlation = None, None, None, None

            has_prices = 'prices' in trajectory and len(trajectory['prices']) > 0
            # Debug: log once per 50 epochs if prices missing
            if not has_prices and (epoch == 0 or epoch % 50 == 0):
                print(f"  [DS24] Physics: No prices in trajectory (keys: {list(trajectory.keys())[:5]}...)")

            if has_prices:
                # Use sample price data from trajectory (up to 32 steps for better MA)
                sample_prices = jnp.array(trajectory['prices'][:min(32, len(trajectory['prices']))])
                log_prices = jnp.log(sample_prices + 1e-8)

                # Compute returns
                log_returns = jnp.diff(log_prices, axis=0, prepend=log_prices[:1])

                # Compute real moving average (EMA with decay=0.9)
                if len(log_prices) > 5:
                    # Exponential moving average for each asset
                    def ema_scan(carry, x):
                        ema_next = 0.9 * carry + 0.1 * x
                        return ema_next, ema_next
                    _, moving_avg = jax.lax.scan(ema_scan, log_prices[0], log_prices)
                else:
                    # Not enough data - use mean as fallback
                    moving_avg = jnp.broadcast_to(
                        jnp.mean(log_prices, axis=0, keepdims=True),
                        log_prices.shape
                    )

                # Create physics_core module and run forward pass directly
                # (Flax submodules can't be accessed from outside - must instantiate fresh)
                n_assets = log_prices.shape[-1]

                # Infer n_factors from params shape (factor_loadings is [n_assets, n_factors])
                n_factors = 8  # default
                corr_params = physics_dict.get('correlation', {})
                if 'factor_loadings' in corr_params:
                    factor_loadings = corr_params['factor_loadings']
                    if hasattr(factor_loadings, 'shape'):
                        n_factors = factor_loadings.shape[-1]

                physics_core = DifferentiableMarketPhysics(n_assets=n_assets, n_factors=n_factors)

                # Use unit noise to see the MAGNITUDE of volatility (1-sigma tendency)
                # This shows how much the volatility component would contribute per σ
                noise = jnp.ones_like(log_prices)
                physics_output, _ = physics_core.apply(
                    {'params': physics_dict},
                    log_prices, log_returns, moving_avg, noise
                )

                G_momentum = np.array(physics_output.G_momentum)
                G_reversion = np.array(physics_output.G_reversion)
                G_volatility = np.array(physics_output.G_volatility)
                G_correlation = np.array(physics_output.G_correlation)

            # Log all physics metrics
            self.logger.log_physics_components(
                G_momentum=G_momentum,
                G_reversion=G_reversion,
                G_volatility=G_volatility,
                G_correlation=G_correlation,
                alpha=np.array([physics_info.get('alpha_mean', 0.0)]),
                beta=np.array([physics_info.get('beta_mean', 0.0)]),
                garch_alpha=np.array([physics_info.get('garch_persistence', 0.0) * 0.3]),
                garch_beta=np.array([physics_info.get('garch_persistence', 0.0) * 0.7])
            )
            self._error_counts['physics'] = 0

        except Exception as e:
            self._error_counts['physics'] += 1
            if self._error_counts['physics'] <= 3 or epoch % 50 == 0:
                print(f"  ⚠️ DS24 physics forward error: {e}")

    def _log_heatmaps(
        self,
        epoch: int,
        trajectory: Dict[str, Any],
        trainer_state: Any
    ):
        """Level 4+: Log latent space and weight heatmaps."""
        if self.model is None:
            return

        heatmap_data = {}

        try:
            _ensure_jax()
            params = trainer_state.params

            # 1. TRM Cycle Utilization Heatmap (H x L matrix)
            trm_cycle_matrix = self._compute_trm_cycle_matrix()
            if trm_cycle_matrix is not None:
                heatmap_data['trm_cycle_utilization'] = trm_cycle_matrix

            # 2. Extract weight matrices (sampled)
            # Navigate to TRM reasoning params: params['trm']['reasoning']
            trm_params = params.get('trm', {})
            reasoning_params = trm_params.get('reasoning', trm_params)

            # Value head weights
            if isinstance(reasoning_params, dict) and 'value_head_1' in reasoning_params:
                heatmap_data['value_head_weights'] = np.array(
                    reasoning_params['value_head_1']['kernel']
                )

            # Policy/TRM output weights
            if isinstance(reasoning_params, dict) and 'output_refiner_h0_1' in reasoning_params:
                heatmap_data['policy_refiner_weights'] = np.array(
                    reasoning_params['output_refiner_h0_1']['kernel']
                )

            # 3. Correlation matrix from physics (if available)
            if self.level >= 3 and hasattr(self.model, 'use_prediction_head') and self.model.use_prediction_head:
                pred_params = params.get('price_predictor', {})
                physics_params = pred_params.get('physics_core', {})
                corr_params = physics_params.get('correlation', {})
                if 'factor_loadings' in corr_params:
                    L = corr_params['factor_loadings']
                    if L is not None:
                        L_np = np.array(L)
                        corr_matrix = L_np @ L_np.T
                        heatmap_data['correlation_matrix'] = corr_matrix

            # 4. Latent space batch (for PCA projection and DS56 dynamics)
            observations = trajectory.get('observations', [])
            if len(observations) >= 10:
                # Use more samples for dynamics analysis (DS56 needs trajectory data)
                max_samples = min(len(observations), 200)  # Cap at 200 for memory
                sample_obs = jnp.array(observations[:max_samples])
                encoder_output = self.model.apply(
                    {'params': params},
                    sample_obs,
                    mode='initial_inference'
                )
                latent_states = encoder_output.get('latent_state')
                if latent_states is not None:
                    # Save full trajectory for DS56 dynamics analysis
                    heatmap_data['latent_trajectory'] = np.array(latent_states)
                    # Also save small batch for backwards compat
                    heatmap_data['latent_batch'] = np.array(latent_states[:10])

                    # Simple PCA projection (top 2 components)
                    latent_np = np.array(latent_states)
                    centered = latent_np - latent_np.mean(axis=0)
                    try:
                        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
                        heatmap_data['latent_pca_2d'] = centered @ Vt[:2].T
                        # Store explained variance ratio for DS56
                        total_var = np.sum(S**2)
                        heatmap_data['pca_explained_variance'] = (S[:2]**2 / total_var).tolist()
                    except np.linalg.LinAlgError:
                        pass  # SVD didn't converge, skip PCA

            # Save heatmap data via tensor snapshot
            if heatmap_data and self.config.save_tensors:
                self.logger.save_tensor_snapshot(
                    epoch=epoch,
                    run_name=self.config.run_name,
                    **heatmap_data
                )

        except Exception as e:
            self._error_counts['heatmap'] += 1
            if self._error_counts['heatmap'] <= 3:
                print(f"  ⚠️ DS24 heatmap error: {e}")

    def _compute_trm_cycle_matrix(self) -> Optional[np.ndarray]:
        """Compute H×L matrix of TRM delta norms from logged values.

        Returns an array of shape (H, L) with individual cycle norms.
        This is used for the TRM Cycle Utilization heatmap visualization.
        """
        h_cycles = self.config.h_cycles
        l_cycles = self.config.l_cycles

        # Build full H×L matrix from individual cycle values
        matrix = np.zeros((h_cycles, l_cycles))
        current = self.logger.current_row

        has_data = False
        for h in range(h_cycles):
            for l in range(l_cycles):
                key = f'latent_delta_norm_h{h}_l{l}'
                if key in current and current[key] is not None:
                    matrix[h, l] = current[key]
                    has_data = True

        return matrix if has_data else None

    def _log_gradient_norms(self, train_metrics: Optional[Dict[str, Any]]):
        """Level 4+: Log per-head gradient norms from training metrics."""
        if train_metrics is None:
            return

        self.logger.log_gradient_norms(
            value_grad_norm=train_metrics.get('grad_norm_value'),
            policy_grad_norm=train_metrics.get('grad_norm_policy'),
            dynamics_grad_norm=train_metrics.get('grad_norm_dynamics')
        )

    def _log_market_baseline(self, trajectory: Dict[str, Any], train_metrics: Optional[Dict[str, Any]] = None):
        """Level 1+: Log market baseline metrics (alpha, market/agent returns).

        Alpha = agent_return - market_return:
          - Positive: Agent beating buy-and-hold
          - Negative: Agent underperforming market
          - Zero: Agent matching market (no edge)

        TASK_119: Also logs optimal hindsight path metrics if available:
          - optimal_alpha: Maximum achievable alpha with perfect hindsight
          - optimal_trades: Trade count in optimal path
          - alpha_capture: agent_pnl / optimal_alpha (efficiency ratio)
        """
        alpha = trajectory.get('alpha')
        market_return = trajectory.get('market_return')
        agent_return = trajectory.get('agent_return')

        # TASK_119: Extract optimal path metrics from train_metrics
        optimal_alpha = None
        optimal_trades = None
        alpha_capture = None
        if train_metrics is not None:
            optimal_alpha = train_metrics.get('optimal_alpha')
            optimal_trades = train_metrics.get('optimal_trades')
            alpha_capture = train_metrics.get('alpha_capture')

        self.logger.log_market_baseline(
            alpha=alpha,
            market_return=market_return,
            agent_return=agent_return,
            optimal_alpha=optimal_alpha,
            optimal_trades=optimal_trades,
            alpha_capture=alpha_capture
        )

    def _log_training_config(self, args: Any):
        """Log training configuration flags (TASK_022, TASK_025)."""
        if args is None:
            return

        self.logger.log_training_config(
            use_gae=getattr(args, 'use_gae', False),
            gae_lambda=getattr(args, 'gae_lambda', 0.95),
            per_head_clipping=getattr(args, 'per_head_clipping', False),
            clip_value=getattr(args, 'clip_grad_value', 1.0),
            clip_policy=getattr(args, 'clip_grad_policy', 1.0),
            clip_dynamics=getattr(args, 'clip_grad_dynamics', 1.0),
            clip_encoder=getattr(args, 'clip_grad_encoder', 1.0),
            epsilon_greedy=getattr(args, 'epsilon_greedy', 0.0),
        )

    def _log_redo_stats(self, redo_stats: Optional[Dict[str, Any]] = None):
        """Log ReDo dormancy statistics (TASK_042)."""
        if redo_stats is None:
            return

        self.logger.log_redo_stats(
            dormant_neurons=redo_stats.get('dormant_neurons', 0),
            dormant_pct=redo_stats.get('dormancy_percent', 0.0),
            recycled_count=redo_stats.get('recycled_count', 0),
        )

    def _log_atc_stats(self, atc_stats: Optional[Dict[str, Any]] = None):
        """Log ATC contrastive loss statistics (TASK_043)."""
        if atc_stats is None:
            return

        self.logger.log_atc_stats(
            loss=atc_stats.get('loss', 0.0),
            pos_sim=atc_stats.get('pos_similarity', 0.0),
            neg_sim=atc_stats.get('neg_similarity', 0.0),
            temperature=atc_stats.get('temperature', 0.1),
        )

    def _save_tensor_snapshot(
        self,
        epoch: int,
        trajectory: Dict[str, Any],
        trainer_state: Any
    ):
        """Save periodic tensor snapshots for detailed analysis."""
        observations = trajectory.get('observations', [])
        if len(observations) == 0:
            return

        latent_states = None
        latent_trajectory = None

        try:
            _ensure_jax()

            # Get actual latent states by running observations through model
            if self.model is not None and trainer_state is not None:
                params = trainer_state.params
                obs_batch = jnp.array(observations)

                # Run through representation network to get latent states
                # Use initial_inference mode to get latent_state
                batch_size = min(len(observations), 100)  # Cap batch size
                sample_obs = obs_batch[:batch_size]

                output = self.model.apply(
                    {'params': params},
                    sample_obs,
                    mode='initial_inference'
                )

                if 'latent_state' in output:
                    latent_states = np.array(output['latent_state'])

                # Also get trajectory of latent states (first dim over time)
                latent_trajectory = latent_states

        except Exception as e:
            self._error_counts['heatmap'] += 1
            if self._error_counts['heatmap'] <= 3:
                print(f"  Warning: DS24 latent extraction error: {e}")
            # Fallback: save raw observations shape info
            latent_states = None

        self.logger.save_tensor_snapshot(
            epoch=epoch,
            run_name=self.config.run_name,
            latent_states=latent_states,
            latent_trajectory=latent_trajectory,
            num_observations=len(observations)
        )


def create_ds24_hooks(args, model=None) -> Optional[DS24TrainingHooks]:
    """
    Factory function to create DS24 hooks from training args.

    Args:
        args: Parsed command line arguments with ds24_level, etc.
        model: MuZero model instance

    Returns:
        DS24TrainingHooks instance or None if level is 0
    """
    level = getattr(args, 'ds24_level', 0)

    # Handle --ds24-logging shortcut
    if getattr(args, 'ds24_logging', False) and level == 0:
        level = 2

    # Default to L3 if enabled but not specified
    if level == 0 and (getattr(args, 'ds24_boost_every', 0) > 0 or
                       getattr(args, 'ds24_snapshot_every', 0) > 0):
        level = 3  # Base level for dynamic boosting

    if level == 0:
        return None

    return DS24TrainingHooks(
        level=level,
        model=model,
        intermediates_every=getattr(args, 'ds24_intermediates_every', 10),
        save_tensors=getattr(args, 'ds24_save_tensors', False),
        tensor_dir='tensor_snapshots' if getattr(args, 'ds24_save_tensors', False) else None,
        h_cycles=getattr(args, 'h_cycles', 4),
        l_cycles=getattr(args, 'l_cycles', 6),
        run_name=getattr(args, 'name', 'default'),
        boost_every=getattr(args, 'ds24_boost_every', 10),
        full_snapshot_every=getattr(args, 'ds24_snapshot_every', 50),
    )
