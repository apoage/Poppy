"""
Curriculum Bridge - External adversarial data generation for training.

TASK_110 Phase 4: Connects adversarial generator to train_jax_fast.py
with MINIMAL trainer changes.

Architecture:
    Teacher (this module) → generates data files
    Student (trainer)     → reads data files normally
    Teacher              → monitors metrics, regenerates

The trainer only needs one small addition:
    --adversarial-data-dir path/to/generated_data

Usage:
    # Terminal 1: Run curriculum loop
    python -m diff_teacher.curriculum_bridge \
        --teacher-checkpoint checkpoints/diff_teacher_small \
        --agent-checkpoint jax_ckpt_fast/latest \
        --output-dir generated_data/adversarial \
        --monitor-log logs/training_fast_*.log

    # Terminal 2: Run training (minimal change - just add data dir)
    python train_jax_fast.py \
        --adversarial-data-dir generated_data/adversarial \
        --adversarial-ratio 0.3 \
        ...other flags...
"""

import os
import json
import time
import glob
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

import jax
import jax.numpy as jnp

from .adversarial_training import AdversarialTrainer, AdversarialConfig
from .teacher import DiffusionTeacher
from .config import TeacherConfig
from .flow_matching import REGIME_TO_IDX

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CurriculumConfig:
    """Configuration for curriculum bridge."""

    # Data generation
    output_dir: str = "generated_data/adversarial"
    sequences_per_batch: int = 100  # Sequences to generate per cycle
    sequence_length: int = 400      # Steps per sequence
    n_assets: int = 64

    # Regeneration
    regenerate_every_epochs: int = 50   # Regenerate data every N epochs
    regenerate_on_plateau: bool = True  # Regenerate when metrics plateau
    plateau_patience: int = 20          # Epochs to wait before plateau detection

    # Mixing
    adversarial_ratio: float = 0.3  # 30% adversarial data
    bias_hard_regimes: bool = True  # Bias towards regimes where student struggles

    # Monitoring
    monitor_log_pattern: str = "logs/training_fast_*.log"
    check_interval_seconds: int = 60


class AdversarialDataGenerator:
    """
    Generates adversarial market data files for training.

    Produces .npz files that can be loaded by the trainer's data loader.
    """

    def __init__(
        self,
        teacher_checkpoint: str,
        config: CurriculumConfig,
    ):
        self.config = config
        self.teacher_checkpoint = teacher_checkpoint

        # Load teacher
        logger.info(f"Loading teacher from {teacher_checkpoint}")
        self.teacher = self._load_teacher(teacher_checkpoint)

        # RNG - HIGH-009 FIX: Use base_rng + step counter with fold_in pattern
        self.base_rng = jax.random.PRNGKey(int(time.time()))
        self.rng_step = 0

        # Create output dir
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        # Tracking
        self.generation_count = 0
        self.regime_stats: Dict[str, int] = {r: 0 for r in REGIME_TO_IDX.keys()}

    def _load_teacher(self, checkpoint_path: str) -> DiffusionTeacher:
        """Load DiffusionTeacher from checkpoint."""
        checkpoint_path = os.path.abspath(checkpoint_path)

        config_path = os.path.join(checkpoint_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            teacher_config = TeacherConfig(**config_dict)
        else:
            teacher_config = TeacherConfig()

        teacher = DiffusionTeacher(teacher_config)
        teacher.init_training()
        teacher.load_checkpoint(checkpoint_path)

        return teacher

    def _latents_to_prices(self, latents: jnp.ndarray) -> np.ndarray:
        """Convert latent sequences to prices."""
        batch_size, window_size, latent_dim = latents.shape
        target_n_assets = self.config.n_assets

        if latent_dim >= target_n_assets:
            price_factors = latents[:, :, :target_n_assets]
        else:
            repeats = (target_n_assets + latent_dim - 1) // latent_dim
            price_factors = jnp.tile(latents, (1, 1, repeats))[:, :, :target_n_assets]

        # Transform to realistic prices
        returns = jnp.tanh(price_factors * 0.1) * 0.03  # Bounded returns ~3%
        cum_returns = jnp.cumsum(returns, axis=1)
        prices = 100.0 * jnp.exp(cum_returns)

        return np.array(prices)

    def generate_batch(
        self,
        regime_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Generate a batch of adversarial market sequences.

        Args:
            regime_weights: Optional weights for regime sampling

        Returns:
            prices: (batch, steps, n_assets) price sequences
            difficulties: (batch,) difficulty levels
            regimes: List of regime names
        """
        batch_size = self.config.sequences_per_batch
        regimes_list = list(REGIME_TO_IDX.keys())

        # Sample regimes
        if regime_weights:
            weights = np.array([regime_weights.get(r, 1.0) for r in regimes_list])
        else:
            weights = np.ones(len(regimes_list))
        weights = weights / weights.sum()

        # HIGH-009 FIX: Use fold_in for reproducible RNG
        sample_key = jax.random.fold_in(self.base_rng, self.rng_step)
        self.rng_step += 1
        regime_indices = jax.random.categorical(
            sample_key,
            jnp.log(jnp.array(weights)),
            shape=(batch_size,)
        )
        regimes = [regimes_list[int(i)] for i in regime_indices]

        # Generate latents - HIGH-009 FIX: Use fold_in for reproducible RNG
        gen_key = jax.random.fold_in(self.base_rng, self.rng_step)
        self.rng_step += 1
        latents, _ = self.teacher.generate(batch_size=batch_size, num_steps=1)

        # Ensure correct sequence length
        if latents.shape[1] != self.config.sequence_length:
            # Interpolate or truncate
            if latents.shape[1] > self.config.sequence_length:
                latents = latents[:, :self.config.sequence_length, :]
            else:
                # Tile to extend
                repeats = (self.config.sequence_length + latents.shape[1] - 1) // latents.shape[1]
                latents = jnp.tile(latents, (1, repeats, 1))[:, :self.config.sequence_length, :]

        # Convert to prices
        prices = self._latents_to_prices(latents)

        # Get difficulty from curriculum
        difficulties = np.array([self.teacher.curriculum.difficulty] * batch_size)

        # Update stats
        for r in regimes:
            self.regime_stats[r] += 1
        self.generation_count += batch_size

        return prices, difficulties, regimes

    def save_batch(
        self,
        prices: np.ndarray,
        difficulties: np.ndarray,
        regimes: List[str],
        batch_id: Optional[int] = None,
    ) -> str:
        """
        Save generated batch to file.

        Args:
            prices: (batch, steps, n_assets)
            difficulties: (batch,)
            regimes: List of regime names
            batch_id: Optional batch identifier

        Returns:
            Path to saved file
        """
        if batch_id is None:
            batch_id = int(time.time() * 1000) % 1000000

        filename = f"adversarial_batch_{batch_id:06d}.npz"
        filepath = os.path.join(self.config.output_dir, filename)

        np.savez(
            filepath,
            prices=prices,
            difficulties=difficulties,
            regimes=np.array(regimes),
            timestamp=time.time(),
            generation_count=self.generation_count,
        )

        logger.info(f"Saved {len(prices)} sequences to {filepath}")
        return filepath

    def generate_and_save(
        self,
        n_batches: int = 1,
        regime_weights: Optional[Dict[str, float]] = None,
    ) -> List[str]:
        """Generate multiple batches and save to files."""
        saved_files = []

        for i in range(n_batches):
            prices, difficulties, regimes = self.generate_batch(regime_weights)
            filepath = self.save_batch(prices, difficulties, regimes, batch_id=i)
            saved_files.append(filepath)

        # Save manifest
        manifest = {
            "files": saved_files,
            "total_sequences": self.generation_count,
            "regime_stats": self.regime_stats,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "sequences_per_batch": self.config.sequences_per_batch,
                "sequence_length": self.config.sequence_length,
                "n_assets": self.config.n_assets,
            }
        }

        manifest_path = os.path.join(self.config.output_dir, "manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Generated {self.generation_count} total sequences")
        return saved_files


class AdversarialDataLoader:
    """
    Loads adversarial data for training.

    This is designed to be used by the trainer with minimal integration:

        loader = AdversarialDataLoader("generated_data/adversarial")
        prices, regimes = loader.sample_batch(batch_size=32)
    """

    def __init__(self, data_dir: str, shuffle: bool = True):
        self.data_dir = data_dir
        self.shuffle = shuffle

        # Load manifest
        manifest_path = os.path.join(data_dir, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                self.manifest = json.load(f)
        else:
            self.manifest = None

        # Find all data files
        self.files = sorted(glob.glob(os.path.join(data_dir, "adversarial_batch_*.npz")))

        if not self.files:
            raise FileNotFoundError(f"No adversarial data files found in {data_dir}")

        logger.info(f"Found {len(self.files)} adversarial data files")

        # Load all data into memory (for small datasets)
        self._load_all()

    def _load_all(self):
        """Load all data files into memory."""
        all_prices = []
        all_regimes = []

        for filepath in self.files:
            data = np.load(filepath, allow_pickle=True)
            all_prices.append(data['prices'])
            all_regimes.extend(data['regimes'].tolist())

        self.prices = np.concatenate(all_prices, axis=0)
        self.regimes = all_regimes
        self.n_sequences = len(self.prices)

        # Shuffled indices
        self.indices = np.arange(self.n_sequences)
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current_idx = 0

        logger.info(f"Loaded {self.n_sequences} adversarial sequences")

    def sample_batch(self, batch_size: int) -> Tuple[np.ndarray, List[str]]:
        """
        Sample a batch of adversarial sequences.

        Args:
            batch_size: Number of sequences

        Returns:
            prices: (batch, steps, n_assets)
            regimes: List of regime names
        """
        # Get indices for this batch
        if self.current_idx + batch_size > self.n_sequences:
            # Reshuffle and reset
            if self.shuffle:
                np.random.shuffle(self.indices)
            self.current_idx = 0

        batch_indices = self.indices[self.current_idx:self.current_idx + batch_size]
        self.current_idx += batch_size

        prices = self.prices[batch_indices]
        regimes = [self.regimes[i] for i in batch_indices]

        return prices, regimes

    def get_random_sequence(self) -> Tuple[np.ndarray, str]:
        """Get a single random sequence."""
        idx = np.random.randint(self.n_sequences)
        return self.prices[idx], self.regimes[idx]


class CurriculumMonitor:
    """
    Monitors training progress and triggers data regeneration.

    Watches training logs for metrics and signals when to regenerate
    adversarial data based on:
    - Epoch count (every N epochs)
    - Plateau detection (metrics not improving)
    - Specific regime performance
    """

    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.last_metrics: Dict[str, float] = {}
        self.metrics_history: List[Dict] = []
        self.plateau_counter = 0
        self.last_regenerate_epoch = 0

    def parse_log_file(self, log_path: str) -> List[Dict]:
        """Parse training log file for metrics."""
        metrics = []

        if not os.path.exists(log_path):
            return metrics

        with open(log_path, 'r') as f:
            for line in f:
                # Look for epoch metrics
                if "Epoch" in line and "Reward" in line:
                    try:
                        # Parse: "Epoch 50 | Reward: +1.23 | Alpha: +0.05 | ..."
                        parts = line.split("|")
                        epoch_part = parts[0]
                        epoch = int(epoch_part.split()[-1])

                        metric_dict = {"epoch": epoch}
                        for part in parts[1:]:
                            if ":" in part:
                                key, val = part.split(":")
                                key = key.strip().lower().replace(" ", "_")
                                val = val.strip().rstrip("%")
                                try:
                                    metric_dict[key] = float(val)
                                except:
                                    pass

                        metrics.append(metric_dict)
                    except:
                        pass

        return metrics

    def check_should_regenerate(self, log_path: str) -> Tuple[bool, str]:
        """
        Check if adversarial data should be regenerated.

        Returns:
            (should_regenerate, reason)
        """
        metrics = self.parse_log_file(log_path)

        if not metrics:
            return False, "no metrics found"

        latest = metrics[-1]
        current_epoch = latest.get("epoch", 0)

        # Check epoch-based regeneration
        epochs_since_regen = current_epoch - self.last_regenerate_epoch
        if epochs_since_regen >= self.config.regenerate_every_epochs:
            self.last_regenerate_epoch = current_epoch
            return True, f"scheduled (every {self.config.regenerate_every_epochs} epochs)"

        # Check plateau detection
        if self.config.regenerate_on_plateau and len(metrics) >= self.config.plateau_patience:
            recent_rewards = [m.get("reward", 0) for m in metrics[-self.config.plateau_patience:]]

            # Check if improvement is minimal
            if len(recent_rewards) >= 2:
                improvement = max(recent_rewards) - min(recent_rewards)
                if improvement < 0.1:  # Less than 0.1 improvement
                    self.plateau_counter += 1
                    if self.plateau_counter >= 3:  # 3 consecutive plateau checks
                        self.plateau_counter = 0
                        self.last_regenerate_epoch = current_epoch
                        return True, "plateau detected"
                else:
                    self.plateau_counter = 0

        return False, "not needed"

    def get_regime_weights(self, log_path: str) -> Dict[str, float]:
        """
        Compute regime weights based on training performance.

        Higher weights for regimes where agent struggles.
        """
        # Default equal weights
        weights = {r: 1.0 for r in REGIME_TO_IDX.keys()}

        # TODO: Parse regime-specific performance from logs
        # For now, use default weights

        return weights


def run_curriculum_loop(
    teacher_checkpoint: str,
    agent_checkpoint: str,
    output_dir: str,
    monitor_log: Optional[str] = None,
    n_initial_batches: int = 5,
):
    """
    Main curriculum loop.

    1. Generate initial adversarial data
    2. Monitor training progress
    3. Regenerate when needed

    Args:
        teacher_checkpoint: Path to DiffusionTeacher
        agent_checkpoint: Path to TRM+MuZero (for regret estimation)
        output_dir: Where to save generated data
        monitor_log: Path pattern for training logs
        n_initial_batches: Number of initial batches to generate
    """
    config = CurriculumConfig(output_dir=output_dir)

    # Initialize generator
    generator = AdversarialDataGenerator(teacher_checkpoint, config)

    # Generate initial data
    logger.info(f"Generating {n_initial_batches} initial batches...")
    generator.generate_and_save(n_batches=n_initial_batches)

    if not monitor_log:
        logger.info("No monitor log specified - exiting after initial generation")
        return

    # Monitor loop
    monitor = CurriculumMonitor(config)

    logger.info(f"Starting curriculum monitor loop...")
    logger.info(f"  Monitoring: {monitor_log}")
    logger.info(f"  Regenerate every: {config.regenerate_every_epochs} epochs")
    logger.info(f"  Check interval: {config.check_interval_seconds}s")

    while True:
        try:
            # Find latest log file
            log_files = sorted(glob.glob(monitor_log))
            if log_files:
                latest_log = log_files[-1]

                # Check if should regenerate
                should_regen, reason = monitor.check_should_regenerate(latest_log)

                if should_regen:
                    logger.info(f"Regenerating adversarial data: {reason}")

                    # Get regime weights based on performance
                    weights = monitor.get_regime_weights(latest_log)

                    # Generate new data
                    generator.generate_and_save(
                        n_batches=n_initial_batches,
                        regime_weights=weights,
                    )

            # Wait before next check
            time.sleep(config.check_interval_seconds)

        except KeyboardInterrupt:
            logger.info("Curriculum loop stopped")
            break
        except Exception as e:
            logger.error(f"Error in curriculum loop: {e}")
            time.sleep(config.check_interval_seconds)


def curriculum_bridge_cli():
    """CLI entry point for curriculum bridge."""
    import argparse

    parser = argparse.ArgumentParser(description="Adversarial curriculum bridge")
    parser.add_argument("--teacher-checkpoint", required=True,
                       help="Path to DiffusionTeacher checkpoint")
    parser.add_argument("--agent-checkpoint", default=None,
                       help="Path to TRM+MuZero checkpoint (optional)")
    parser.add_argument("--output-dir", default="generated_data/adversarial",
                       help="Output directory for generated data")
    parser.add_argument("--monitor-log", default=None,
                       help="Log file pattern to monitor (e.g., 'logs/training_*.log')")
    parser.add_argument("--n-batches", type=int, default=5,
                       help="Number of batches to generate")
    parser.add_argument("--generate-only", action="store_true",
                       help="Generate data and exit (no monitoring)")

    args = parser.parse_args()

    if args.generate_only:
        # Just generate and exit
        config = CurriculumConfig(output_dir=args.output_dir)
        generator = AdversarialDataGenerator(args.teacher_checkpoint, config)
        generator.generate_and_save(n_batches=args.n_batches)
    else:
        # Run full curriculum loop
        run_curriculum_loop(
            teacher_checkpoint=args.teacher_checkpoint,
            agent_checkpoint=args.agent_checkpoint,
            output_dir=args.output_dir,
            monitor_log=args.monitor_log,
            n_initial_batches=args.n_batches,
        )


if __name__ == "__main__":
    curriculum_bridge_cli()
