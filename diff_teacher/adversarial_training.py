"""
Adversarial Training for Market Generation.

TASK_110 Phase 3: Trains the flow matching generator to create market sequences
that maximize agent regret (agent struggles to profit).

Core idea (PAIRED-style):
    1. Generator creates market sequences
    2. Agent trades on sequences, achieves actual_pnl
    3. Oracle computes optimal_pnl (hindsight)
    4. Regret = (optimal_pnl - actual_pnl) / |optimal_pnl|
    5. Train generator to MAXIMIZE regret (minimize -regret)
    6. Realism constraint keeps sequences plausible

Usage:
    trainer = AdversarialTrainer(
        teacher_checkpoint="checkpoints/diff_teacher_small",
        agent_checkpoint="jax_ckpt_fast/latest",
    )
    trainer.train(epochs=50)
"""

import os
import json
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Deque
from pathlib import Path
import logging
import time

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from .teacher import DiffusionTeacher
from .config import TeacherConfig
from .agent_evaluator import AgentEvaluator, EvaluationResult
from .optimal_solver import optimal_pnl_jax
from .flow_matching import REGIME_TO_IDX

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AdversarialConfig:
    """Configuration for adversarial training."""

    # Training
    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 1e-5  # Lower LR for fine-tuning
    warmup_steps: int = 100

    # Loss weights
    lambda_realism: float = 0.1  # Weight for realism constraint
    lambda_diversity: float = 0.01  # Weight for diversity term

    # Realism constraints
    min_volatility: float = 0.001  # Min daily volatility
    max_volatility: float = 0.15   # Max daily volatility
    max_jump: float = 0.15         # Max single-step return

    # Curriculum
    regime_sample_hard: bool = True  # Bias towards hard regimes
    hard_regime_boost: float = 2.0   # Multiplier for hard regime sampling

    # Checkpointing
    checkpoint_every: int = 10
    log_every: int = 5

    # Paths
    checkpoint_dir: str = "checkpoints/diff_teacher_adversarial"


@dataclass
class AdversarialMetrics:
    """Metrics from adversarial training step."""

    total_loss: float
    adv_loss: float          # Negative mean regret
    realism_loss: float      # Realism constraint violation
    diversity_loss: float    # Encourages variety

    mean_regret: float       # Key metric: higher = agent struggles
    max_regret: float
    min_regret: float

    mean_actual_pnl: float   # Agent's PnL
    mean_optimal_pnl: float  # Oracle's PnL

    hold_rate: float         # Agent's hold rate


def compute_realism_loss(prices: jnp.ndarray, config: AdversarialConfig) -> jnp.ndarray:
    """
    Compute realism loss to keep generated sequences plausible.

    Checks:
        - Price positivity: prices > 0
        - Volatility bounds: not too extreme
        - Smoothness: no discontinuous jumps

    Args:
        prices: (batch, steps, n_assets) price sequences
        config: Adversarial config with bounds

    Returns:
        Scalar realism loss
    """
    # 1. Positivity: penalize negative prices heavily
    pos_loss = jnp.mean(jnp.maximum(-prices, 0) ** 2) * 100.0

    # 2. Volatility bounds
    # Compute per-step returns
    returns = jnp.diff(prices, axis=1) / (prices[:, :-1, :] + 1e-8)

    # Per-asset volatility
    vol = jnp.std(returns, axis=1)  # (batch, n_assets)

    # Penalize if vol < min or vol > max
    vol_low_loss = jnp.mean(jnp.maximum(config.min_volatility - vol, 0) ** 2)
    vol_high_loss = jnp.mean(jnp.maximum(vol - config.max_volatility, 0) ** 2)
    vol_loss = vol_low_loss + vol_high_loss

    # 3. Smoothness: penalize large jumps
    abs_returns = jnp.abs(returns)
    jump_loss = jnp.mean(jnp.maximum(abs_returns - config.max_jump, 0) ** 2)

    # 4. Price range: penalize extreme prices
    price_mean = jnp.mean(prices, axis=(1, 2), keepdims=True)
    price_deviation = jnp.abs(prices - price_mean) / (price_mean + 1e-8)
    range_loss = jnp.mean(jnp.maximum(price_deviation - 2.0, 0) ** 2)  # >2x from mean

    total = pos_loss + vol_loss + jump_loss + range_loss

    return total


def compute_diversity_loss(latents: jnp.ndarray) -> jnp.ndarray:
    """
    Encourage diversity in generated sequences.

    Penalizes when all latents are too similar (mode collapse).

    Args:
        latents: (batch, window_size, latent_dim)

    Returns:
        Scalar diversity loss (lower = more diverse)
    """
    # Compute pairwise distances in latent space
    # Flatten to (batch, window_size * latent_dim)
    flat = latents.reshape(latents.shape[0], -1)

    # Mean latent
    mean_latent = jnp.mean(flat, axis=0, keepdims=True)

    # Variance from mean (higher = more diverse)
    variance = jnp.mean(jnp.sum((flat - mean_latent) ** 2, axis=1))

    # We want high variance, so loss = -variance (bounded)
    # Use softplus to keep it positive
    diversity_loss = jax.nn.softplus(1.0 - variance)

    return diversity_loss


class AdversarialTrainer:
    """
    Trains generator to create adversarial market sequences.

    The generator learns to create scenarios where the agent has high regret
    (difference between optimal and actual PnL), while maintaining realistic
    market properties.
    """

    def __init__(
        self,
        teacher_checkpoint: str,
        agent_checkpoint: str,
        config: Optional[AdversarialConfig] = None,
    ):
        """
        Initialize adversarial trainer.

        Args:
            teacher_checkpoint: Path to trained DiffusionTeacher
            agent_checkpoint: Path to TRM+MuZero checkpoint
            config: Adversarial training config
        """
        self.config = config or AdversarialConfig()

        # Load teacher
        logger.info(f"Loading teacher from {teacher_checkpoint}")
        self.teacher = self._load_teacher(teacher_checkpoint)

        # Load agent evaluator
        logger.info(f"Loading agent from {agent_checkpoint}")
        self.agent = AgentEvaluator(
            checkpoint_path=agent_checkpoint,
            n_simulations=8,  # Fewer sims for speed during training
        )

        # Tracking - MED-006 FIX: Use bounded deque to prevent memory leak
        self.metrics_history: Deque[AdversarialMetrics] = deque(maxlen=1000)
        self.regime_performance: Dict[str, Deque[float]] = {
            regime: deque(maxlen=500) for regime in REGIME_TO_IDX.keys()
        }

        # RNG - HIGH-009 FIX: Use base_rng + step counter with fold_in pattern
        # This is reproducible and thread-safe unlike mutable self.rng splitting
        self.base_rng = jax.random.PRNGKey(42)
        self.rng_step = 0

        # Create checkpoint dir
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def _load_teacher(self, checkpoint_path: str) -> DiffusionTeacher:
        """Load DiffusionTeacher from checkpoint."""
        checkpoint_path = os.path.abspath(checkpoint_path)

        # Load config
        config_path = os.path.join(checkpoint_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            teacher_config = TeacherConfig(**config_dict)
        else:
            # Use default config
            logger.warning("No config.json found, using defaults")
            teacher_config = TeacherConfig()

        # Create teacher
        teacher = DiffusionTeacher(teacher_config)
        teacher.init_training()

        # Load checkpoint
        teacher.load_checkpoint(checkpoint_path)

        return teacher

    def _sample_regimes(self, batch_size: int) -> jnp.ndarray:
        """
        Sample regimes, biased towards hard ones if configured.

        Hard regimes are those where agent historically has high regret.
        """
        regimes = list(REGIME_TO_IDX.keys())

        if self.config.regime_sample_hard and self.regime_performance:
            # Compute weights based on historical performance
            weights = []
            for regime in regimes:
                if self.regime_performance[regime]:
                    # Higher regret = higher sampling weight
                    mean_regret = np.mean(self.regime_performance[regime][-50:])
                    weight = 1.0 + self.config.hard_regime_boost * max(0, mean_regret)
                else:
                    weight = 1.0
                weights.append(weight)

            weights = np.array(weights)
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(regimes)) / len(regimes)

        # Sample - HIGH-009 FIX: Use fold_in for reproducible RNG
        sample_key = jax.random.fold_in(self.base_rng, self.rng_step)
        self.rng_step += 1
        indices = jax.random.categorical(
            sample_key,
            jnp.log(jnp.array(weights)),
            shape=(batch_size,)
        )

        return indices

    def _latents_to_prices(
        self,
        latents: jnp.ndarray,
    ) -> np.ndarray:
        """
        Convert latent sequences to price arrays.

        For now, uses a simple transformation since we don't have
        a fully trained decoder. In production, this would use the
        VAE decoder.

        Args:
            latents: (batch, window_size, latent_dim)

        Returns:
            prices: (batch, window_size, n_assets) as numpy
            NOTE: Always returns 64 assets for agent compatibility
        """
        # Simple transformation: latent -> prices
        # This is a placeholder - real implementation would use VAE decoder
        batch_size, window_size, latent_dim = latents.shape

        # Always produce 64 assets for agent compatibility
        target_n_assets = 64

        # Project latent to target_n_assets
        if latent_dim >= target_n_assets:
            price_factors = latents[:, :, :target_n_assets]
        else:
            # Tile to reach target_n_assets
            repeats = (target_n_assets + latent_dim - 1) // latent_dim
            price_factors = jnp.tile(latents, (1, 1, repeats))[:, :, :target_n_assets]

        # Transform to positive prices
        # Start at 100, apply cumulative returns from latent
        returns = jnp.tanh(price_factors * 0.1) * 0.05  # Bounded returns
        cum_returns = jnp.cumsum(returns, axis=1)
        prices = 100.0 * jnp.exp(cum_returns)

        return np.array(prices)

    def _compute_batch_regret(
        self,
        prices: np.ndarray,
        regimes: List[str],
    ) -> Tuple[np.ndarray, List[EvaluationResult]]:
        """
        Compute regret for a batch of price sequences.

        Args:
            prices: (batch, window_size, n_assets) prices
            regimes: List of regime names

        Returns:
            regrets: (batch,) regret values
            results: List of EvaluationResult
        """
        batch_size = prices.shape[0]

        # Evaluate agent
        results = self.agent.evaluate_batch(prices, regimes)
        actual_pnls = np.array([r.actual_pnl for r in results])

        # Compute optimal
        optimal_pnls = np.array(optimal_pnl_jax(jnp.array(prices)))

        # Scale optimal to match agent's scale (percentage of initial cash)
        initial_cash = self.agent.initial_cash
        optimal_pnls_scaled = optimal_pnls * initial_cash

        # Compute regret
        # regret = (optimal - actual) / |optimal|
        # Higher = agent did worse relative to optimal
        regrets = (optimal_pnls_scaled - actual_pnls) / (np.abs(optimal_pnls_scaled) + 1e-6)

        return regrets, results

    def train_step(self) -> AdversarialMetrics:
        """
        Single adversarial training step.

        1. Sample regimes (biased towards hard)
        2. Generate sequences
        3. Evaluate agent
        4. Compute regret + realism loss
        5. Update generator

        Returns:
            AdversarialMetrics for this step
        """
        batch_size = self.config.batch_size

        # 1. Sample regimes
        regime_indices = self._sample_regimes(batch_size)
        regime_names = [list(REGIME_TO_IDX.keys())[int(i)] for i in regime_indices]

        # 2. Generate latents - HIGH-009 FIX: Use fold_in for reproducible RNG
        gen_key = jax.random.fold_in(self.base_rng, self.rng_step)
        self.rng_step += 1
        latents, _ = self.teacher.generate(
            batch_size=batch_size,
            num_steps=1,  # Fast single-step generation
        )

        # 3. Convert to prices
        prices = self._latents_to_prices(latents)

        # 4. Compute regret
        regrets, results = self._compute_batch_regret(prices, regime_names)

        # 5. Compute losses
        # Adversarial loss: we want HIGH regret, so minimize -regret
        adv_loss = -np.mean(regrets)

        # Realism loss
        realism_loss = float(compute_realism_loss(jnp.array(prices), self.config))

        # Diversity loss
        diversity_loss = float(compute_diversity_loss(latents))

        # Total loss
        total_loss = (
            adv_loss +
            self.config.lambda_realism * realism_loss +
            self.config.lambda_diversity * diversity_loss
        )

        # 6. Update generator (simplified - in practice use JAX gradients)
        # For now, track metrics only - full gradient update requires more setup
        # TODO: Implement proper gradient-based update

        # 7. Track regime performance
        for regime, regret in zip(regime_names, regrets):
            self.regime_performance[regime].append(float(regret))

        # 8. Collect metrics
        metrics = AdversarialMetrics(
            total_loss=float(total_loss),
            adv_loss=float(adv_loss),
            realism_loss=float(realism_loss),
            diversity_loss=float(diversity_loss),
            mean_regret=float(np.mean(regrets)),
            max_regret=float(np.max(regrets)),
            min_regret=float(np.min(regrets)),
            mean_actual_pnl=float(np.mean([r.actual_pnl for r in results])),
            mean_optimal_pnl=float(np.mean([r.actual_pnl + r.actual_pnl * regrets[i]
                                            for i, r in enumerate(results)])),
            hold_rate=float(np.mean([r.hold_rate for r in results])),
        )

        self.metrics_history.append(metrics)

        return metrics

    def train(self, epochs: Optional[int] = None) -> None:
        """
        Run adversarial training loop.

        Args:
            epochs: Number of epochs (defaults to config)
        """
        epochs = epochs or self.config.epochs

        logger.info(f"Starting adversarial training for {epochs} epochs")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  Lambda realism: {self.config.lambda_realism}")

        start_time = time.time()

        for epoch in range(epochs):
            epoch_start = time.time()

            # Training step
            metrics = self.train_step()

            epoch_time = time.time() - epoch_start

            # Logging
            if (epoch + 1) % self.config.log_every == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Regret: {metrics.mean_regret:.3f} (max: {metrics.max_regret:.3f}) | "
                    f"Loss: {metrics.total_loss:.3f} | "
                    f"Agent PnL: ${metrics.mean_actual_pnl:.2f} | "
                    f"Hold: {metrics.hold_rate:.1%} | "
                    f"Time: {epoch_time:.1f}s"
                )

            # Checkpointing
            if (epoch + 1) % self.config.checkpoint_every == 0:
                self.save_checkpoint(epoch + 1)

        total_time = time.time() - start_time
        logger.info(f"Training complete in {total_time / 60:.1f} minutes")

        # Final checkpoint
        self.save_checkpoint(epochs)

        # Summary
        self._print_summary()

    def save_checkpoint(self, epoch: int) -> None:
        """Save adversarial trainer checkpoint."""
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"checkpoint_epoch_{epoch}"
        )
        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

        # Save metrics
        metrics_path = os.path.join(checkpoint_path, "metrics.json")
        metrics_data = [
            {
                "total_loss": m.total_loss,
                "adv_loss": m.adv_loss,
                "mean_regret": m.mean_regret,
                "max_regret": m.max_regret,
                "mean_actual_pnl": m.mean_actual_pnl,
                "hold_rate": m.hold_rate,
            }
            for m in self.metrics_history
        ]
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)

        # Save regime performance
        regime_path = os.path.join(checkpoint_path, "regime_performance.json")
        with open(regime_path, 'w') as f:
            json.dump(self.regime_performance, f, indent=2)

        # Save config
        config_path = os.path.join(checkpoint_path, "config.json")
        with open(config_path, 'w') as f:
            json.dump(vars(self.config), f, indent=2)

        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def _print_summary(self) -> None:
        """Print training summary."""
        if not self.metrics_history:
            return

        # Overall stats
        final_metrics = self.metrics_history[-1]
        initial_metrics = self.metrics_history[0]

        logger.info("\n" + "=" * 60)
        logger.info("ADVERSARIAL TRAINING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Initial regret: {initial_metrics.mean_regret:.3f}")
        logger.info(f"Final regret:   {final_metrics.mean_regret:.3f}")
        logger.info(f"Regret change:  {final_metrics.mean_regret - initial_metrics.mean_regret:+.3f}")
        logger.info("")

        # Per-regime breakdown
        logger.info("Regime Performance (mean regret):")
        for regime, regrets in self.regime_performance.items():
            if regrets:
                logger.info(f"  {regime:15s}: {np.mean(regrets):.3f}")

        logger.info("=" * 60)


def train_adversarial_cli():
    """CLI entry point for adversarial training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train adversarial market generator")
    parser.add_argument("--teacher-checkpoint", required=True,
                       help="Path to DiffusionTeacher checkpoint")
    parser.add_argument("--agent-checkpoint", required=True,
                       help="Path to TRM+MuZero checkpoint")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--lambda-realism", type=float, default=0.1,
                       help="Weight for realism constraint")
    parser.add_argument("--checkpoint-dir", type=str,
                       default="checkpoints/diff_teacher_adversarial",
                       help="Directory for checkpoints")

    args = parser.parse_args()

    config = AdversarialConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lambda_realism=args.lambda_realism,
        checkpoint_dir=args.checkpoint_dir,
    )

    trainer = AdversarialTrainer(
        teacher_checkpoint=args.teacher_checkpoint,
        agent_checkpoint=args.agent_checkpoint,
        config=config,
    )

    trainer.train()


if __name__ == "__main__":
    train_adversarial_cli()
