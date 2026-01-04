#!/usr/bin/env python3
"""
Dojo: Curriculum Training Manager

Orchestrates modular curriculum training with:
- Synthetic market generators (progressive difficulty)
- Composable tactics (basic → morphing → pairs → market making)
- Progressive complexity (teach one concept at a time)

Philosophy: Modular architecture over monolithic integration.
"""

import argparse
import logging
import json
import os
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

# Synthetic generators (for reference, not used directly since we call via subprocess)
# from mcts_trader.data.synthetic_market import SyntheticMarket

# Tactics (for reference)
# from mcts_trader.tactics import BasicTrading, PortfolioMorphing

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [DOJO] - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'dojo_training.log')),
        logging.StreamHandler()
    ]
)


class DojoManager:
    """
    Curriculum Training Orchestrator.

    Manages:
    - Level progression (0 → 22+)
    - Tactic composition (basic → morphing → pairs → ...)
    - Checkpoint management
    - Progress tracking
    """

    def __init__(self, curriculum_config: List[Dict], checkpoint_dir: str = 'dojo_checkpoints',
                 adaptive_calibration: bool = False, value_scale: float = 1.0,
                 load_checkpoint: str = None, model_size: str = 'large',
                 base_name: str = 'pretrain'):
        """
        Args:
            curriculum_config: List of level configs
                [
                    {'level': 0, 'generator': 'sine', 'tactics': ['basic'], 'epochs': 500},
                    {'level': 1, 'generator': 'lagged', 'tactics': ['basic'], 'epochs': 1000},
                    ...
                ]
            checkpoint_dir: Directory for curriculum checkpoints
            adaptive_calibration: Enable adaptive value calibration
            value_scale: Initial value calibration scale
            load_checkpoint: Path to checkpoint to load for first level
            model_size: Model size (small/medium/large) - overrides curriculum config
            base_name: Base name for run directories
        """
        self.curriculum = curriculum_config
        self.checkpoint_dir = checkpoint_dir
        self.current_level = 0
        self.adaptive_calibration = adaptive_calibration
        self.value_scale = value_scale
        self.initial_checkpoint = load_checkpoint
        self.model_size = model_size
        self.base_name = base_name

        os.makedirs(checkpoint_dir, exist_ok=True)
        logging.info(f"Dojo initialized with {len(curriculum_config)} curriculum levels")
        logging.info(f"Model size: {model_size}")
        if adaptive_calibration:
            logging.info(f"Adaptive calibration enabled (initial scale={value_scale})")

    def run(self):
        """Run full curriculum sequence"""
        logging.info(f"Starting curriculum training ({len(self.curriculum)} levels)")

        previous_checkpoint = self.initial_checkpoint  # Track checkpoint between levels

        for i, level_config in enumerate(self.curriculum):
            logging.info(f"\n{'='*60}")
            logging.info(f"LEVEL {i}: {level_config['generator']}")
            logging.info(f"Tactics: {level_config['tactics']}")
            logging.info(f"Epochs: {level_config['epochs']}")
            logging.info(f"{'='*60}\n")

            # Pass previous level's checkpoint
            checkpoint_from_level = self._train_level(
                level_config, level_idx=i,
                previous_level_checkpoint=previous_checkpoint
            )

            # Update checkpoint for next level
            if checkpoint_from_level:
                previous_checkpoint = checkpoint_from_level

            # Save curriculum progress
            self._save_progress(i)

        logging.info("🎓 Curriculum COMPLETE!")

    def _train_level(self, config: Dict, level_idx: int,
                     previous_level_checkpoint: str = None) -> str:
        """
        Train single curriculum level with mastery check.

        Args:
            config: Level configuration
                {
                    'level': 0,
                    'generator': 'sine',
                    'tactics': ['basic'],
                    'epochs': 500,
                    'pass_threshold': 1.5,  # Required avg reward
                    'retry_epochs': 250,     # Extra epochs if failed
                    'max_retries': 2,        # Max retry attempts
                    'n_assets': 64,
                    'n_steps': 10000,
                }
            level_idx: Index in curriculum
            previous_level_checkpoint: Checkpoint from previous level (for handoff)

        Returns:
            Path to this level's checkpoint (for next level)
        """
        start_time = time.time()

        # Extract config parameters
        generator_type = config['generator']
        level = config.get('level', level_idx)
        n_assets = config.get('n_assets', 64)
        n_steps = config.get('n_steps', 10000)
        epochs = config.get('epochs', 500)
        batch_size = config.get('batch_size', 384)  # Balance: fits 24GB VRAM
        model_size = self.model_size  # Use from CLI args (overrides config)
        retry_epochs = config.get('retry_epochs', 250)
        max_retries = config.get('max_retries', 2)
        disable_fees = config.get('disable_fees', False)  # NEW: For easier early levels

        # NEW: Extended curriculum parameters
        sims = config.get('sims', 32)                    # MCTS simulations
        slippage_bps = config.get('slippage_bps', 0)     # Slippage in basis points
        steps = config.get('steps', 400)                  # Episode length
        parallel_envs = config.get('parallel_envs', 4)    # Parallel environments
        bf16 = config.get('bf16', True)                   # BFloat16 mixed precision
        adaptive_entropy = config.get('adaptive_entropy', True)  # Adaptive entropy

        # Run naming: base_name + stage
        stage = config.get('stage', f'{level_idx}')

        # Determine tactics
        tactics = config['tactics']
        enable_swaps = 'morphing' in tactics

        logging.info(f"Training Level {level} ({generator_type})")
        logging.info(f"Tactics: {tactics} (enable_swaps={enable_swaps})")
        logging.info(f"Config: {epochs} epochs, {n_assets} assets, {steps} steps/ep, {sims} sims")

        # Run naming (defined before loop so it's available for return)
        run_name = f'{self.base_name}_{stage}_{generator_type}'

        # Training loop with retries
        attempt = 0
        passed = False
        total_epochs_trained = 0

        while not passed and attempt <= max_retries:
            if attempt > 0:
                logging.info(f"🔄 RETRY {attempt}/{max_retries}: Adding {retry_epochs} more epochs")
                current_epochs = retry_epochs
            else:
                current_epochs = epochs

            # Build training command with tournament-validated configurations (2026-01-03)
            import subprocess

            # Optimal env config based on data type:
            # Synthetic: 50 envs × 400 steps = 20,000 samples (diversity)
            # Real-time: 15 envs × 1440 steps = 21,600 samples
            optimal_envs = parallel_envs if parallel_envs > 4 else 50
            optimal_steps = steps if steps > 400 else 400

            cmd = [
                'python', 'train_jax_fast.py',
                '--synthetic-level', str(level),
                '--synthetic-steps', str(n_steps),
                '--epochs', str(current_epochs),
                '--batch-size', str(batch_size),
                '--model-size', model_size,
                '--name', run_name,
                '--sims', str(sims),
                '--steps', str(optimal_steps),
                '--parallel-envs', str(optimal_envs),
                '--jax-env',  # Use JAX environment with TASK_028 optimizations
                # Tournament winner: PopArt normalization
                '--use-popart',
                '--popart-beta', '0.0001',
                # Tournament winner: Slow sims warmup
                '--sims-warmup', '100',
                '--sims-min', '8',
                # Tournament winner: Prediction loss
                '--use-prediction-loss',
                '--prediction-loss-weight', '0.2',
                '--pred-head-size', 'large',
                # DS24 logging (L3 full metrics)
                '--ds24-level', '3',
                '--ds24-intermediates-every', '5',
                '--ds24-snapshot-every', '10',
                '--ds24-save-tensors',  # Save tensor snapshots every 10 epochs
                # Data regeneration for diversity
                '--regenerate-data-every', '50',
            ]

            # BFloat16 mixed precision
            if bf16:
                cmd.append('--bf16')

            # Tournament winner: Adaptive entropy pumping with tuned thresholds
            if adaptive_entropy:
                cmd.extend([
                    '--adaptive-entropy',
                    '--entropy-hold-threshold', '0.85',
                    '--entropy-boost-low', '0.05',
                    '--entropy-boost-high', '0.3',
                ])

            # Adaptive value calibration for curriculum stability
            cmd.append('--adaptive-calibration')

            if enable_swaps:
                cmd.append('--enable-swaps')

            # Fee and slippage handling
            fee_bps = config.get('fee_bps', None)
            if fee_bps is not None:
                cmd.extend(['--fee', str(fee_bps)])
            elif disable_fees:
                cmd.extend(['--fee', '0'])

            # Slippage handling
            if slippage_bps > 0:
                cmd.extend(['--slippage', str(slippage_bps)])
            elif disable_fees or fee_bps == 0:
                cmd.extend(['--slippage', '0'])

            # Adaptive calibration parameters
            if self.adaptive_calibration:
                cmd.append('--adaptive-calibration')
            if self.value_scale != 1.0:
                cmd.extend(['--value-scale', str(self.value_scale)])

            # Safe checkpoint parameters (for advanced levels)
            # Safe checkpoints for level 4+ (protect learned knowledge)
            safety_config = config.get('safety', {})
            if safety_config.get('enabled', False) or level >= 4:
                cmd.append('--safe-checkpoints')
                # Default safety params for higher levels
                rollback_patience = safety_config.get('rollback_patience', 50)
                rollback_threshold = safety_config.get('rollback_threshold', 0.5)
                max_rollbacks = safety_config.get('max_rollbacks', 3)
                keep_best_n = safety_config.get('keep_best_n', 3)
                lr_decay = safety_config.get('lr_decay_on_rollback', 0.5)

                cmd.extend([
                    '--rollback-patience', str(rollback_patience),
                    '--rollback-threshold', str(rollback_threshold),
                    '--max-rollbacks', str(max_rollbacks),
                    '--keep-best-n', str(keep_best_n),
                    '--lr-decay-on-rollback', str(lr_decay),
                ])

                # Periodic evaluation on rotation level for higher levels
                if level >= 5:
                    cmd.extend([
                        '--eval-every', '50',
                        '--eval-level', '4',  # Evaluate on rotation (tests multi-asset)
                        '--eval-steps', '200',
                    ])
                logging.info(f"Safe checkpoints enabled for level {level}")

            # Checkpoint loading logic
            checkpoint_dir = f'jax_ckpt_fast/{run_name}'
            is_first_attempt = (attempt == 0)

            if is_first_attempt and previous_level_checkpoint and os.path.exists(previous_level_checkpoint):
                # First attempt of NEW level - load from previous level, reset optimizer
                cmd.extend(['--load-checkpoint', previous_level_checkpoint])
                cmd.append('--reset-optimizer')
                cmd.append('--clear-buffer')
                logging.info(f"📂 Starting new level from previous checkpoint: {previous_level_checkpoint}")
                logging.info(f"   🔄 Resetting optimizer for new level")
            elif not is_first_attempt and os.path.exists(checkpoint_dir):
                # Retry - resume from same level's checkpoint
                cmd.extend(['--load-checkpoint', checkpoint_dir])
                logging.info(f"📂 Resuming from checkpoint (retry {attempt})")

            logging.info(f"Running: {' '.join(cmd)}")

            result = subprocess.run(cmd, capture_output=False)

            if result.returncode != 0:
                logging.error(f"Level {level_idx} FAILED with code {result.returncode}")
                raise RuntimeError(f"Training failed at level {level_idx}")

            total_epochs_trained += current_epochs

            # Check mastery
            passed = self._check_level_mastery(level_idx, config, 'last_run_metrics.json')

            if not passed and attempt < max_retries:
                logging.warning(f"❌ Level {level_idx} did not pass mastery check")
                logging.info(f"Will retry with {retry_epochs} additional epochs")
            elif not passed:
                logging.error(f"❌ Level {level_idx} FAILED after {max_retries} retries")
                raise RuntimeError(f"Could not pass Level {level_idx} after {total_epochs_trained} total epochs")

            attempt += 1

        duration = time.time() - start_time
        logging.info(f"✅ Level {level_idx} PASSED in {duration/60:.1f} minutes ({total_epochs_trained} total epochs)")

        # Return checkpoint path for next level
        return f'jax_ckpt_fast/{run_name}'

    def _check_level_mastery(self, level_idx: int, config: Dict, metrics_file: str) -> bool:
        """
        Check if agent passed level based on reward criteria.

        Args:
            level_idx: Current level index
            config: Level configuration (contains pass_threshold)
            metrics_file: Path to last_run_metrics.json

        Returns:
            True if passed, False otherwise
        """
        # Load metrics
        try:
            with open(metrics_file) as f:
                metrics = json.load(f)
        except FileNotFoundError:
            logging.error(f"Metrics file not found: {metrics_file}")
            return False

        episode_rewards = metrics.get('episode_rewards', [])
        recent_avg = metrics.get('recent_avg_reward', 0.0)

        # Get pass threshold from config or use defaults
        default_thresholds = {
            # Phase 1: Basic Mechanics
            0: 2.0,   # Sine waves (no fees)
            1: 1.5,   # Lagged pairs
            2: 1.0,   # Dip crashes
            3: 0.5,   # Noisy trends
            4: 0.5,   # Rotation
            # Phase 2: Statistical Arbitrage
            5: 0.3,   # Pairs trading
            6: 0.2,   # Triangle arb
            7: 0.2,   # Index arb
            # Phase 3: Momentum & Reversal
            8: 0.3,   # Momentum
            9: 0.3,   # Mean reversion
            10: 0.2,  # Breakout
            # Phase 4: Market Microstructure
            11: 0.1,  # Bid-ask bounce
            12: 0.1,  # Order imbalance
            13: 0.1,  # Market making
            # Phase 5: Volatility
            14: 0.1,  # Volatility clustering
        }

        threshold = config.get('pass_threshold', default_thresholds.get(level_idx, 0.1))

        # Check if we have enough data
        if len(episode_rewards) < 50:
            logging.warning(f"Only {len(episode_rewards)} epochs - using all rewards for average")
            if episode_rewards:
                recent_avg = sum(episode_rewards) / len(episode_rewards)
            else:
                logging.error("No episode rewards found!")
                return False

        # Determine pass/fail
        passed = recent_avg >= threshold

        # Log results
        logging.info("")
        logging.info("="*60)
        logging.info(f"MASTERY CHECK - Level {level_idx}")
        logging.info("="*60)
        logging.info(f"Recent Avg Reward (last 50 epochs): {recent_avg:+.2f}%")
        logging.info(f"Pass Threshold:                      {threshold:+.2f}%")
        logging.info(f"Total Epochs Completed:              {len(episode_rewards)}")
        logging.info(f"Result: {'✅ PASS' if passed else '❌ FAIL'}")
        logging.info("="*60)
        logging.info("")

        return passed

    def _save_progress(self, level_idx: int):
        """Save curriculum progress checkpoint"""
        progress_file = os.path.join(self.checkpoint_dir, 'curriculum_progress.json')

        progress = {
            'current_level': level_idx,
            'completed_levels': level_idx + 1,
            'total_levels': len(self.curriculum),
            'timestamp': time.time()
        }

        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)

        logging.info(f"Progress saved: {level_idx + 1}/{len(self.curriculum)} levels complete")

    def load_progress(self) -> int:
        """Load curriculum progress from checkpoint"""
        progress_file = os.path.join(self.checkpoint_dir, 'curriculum_progress.json')

        if not os.path.exists(progress_file):
            return 0

        with open(progress_file) as f:
            progress = json.load(f)

        return progress.get('current_level', 0)


def load_curriculum_from_file(filepath: str) -> List[Dict]:
    """Load curriculum config from JSON file"""
    with open(filepath) as f:
        return json.load(f)


def create_default_curriculum() -> List[Dict]:
    """
    Create full curriculum (Levels 0-14) with mastery checks.

    Uses tournament-validated optimal configurations (2026-01-03):
    - 50 envs × 400 steps = 20,000 samples/epoch (synthetic diversity)
    - PopArt, sims warmup, prediction loss, adaptive entropy

    Returns:
        List of level configs
    """
    # Base config for all levels
    # Optimal config: 50 envs × 400 steps, batch 256 (fits 24GB VRAM)
    base_config = {
        'n_assets': 64,
        'n_steps': 10000,
        'parallel_envs': 50,  # 50 envs × 400 steps = 20,000 samples/epoch
        'steps': 400,
        'retry_epochs': 300,
        'max_retries': 2,
    }

    return [
        # Phase 1: Basic Mechanics (Levels 0-4)
        {
            **base_config,
            'level': 0,
            'generator': 'sine',
            'tactics': ['basic'],
            'epochs': 500,
            'pass_threshold': 2.0,
            'disable_fees': True,
            'description': 'Sine waves - trend detection (no fees)'
        },
        {
            **base_config,
            'level': 1,
            'generator': 'lagged',
            'tactics': ['basic'],
            'epochs': 750,
            'pass_threshold': 1.5,
            'description': 'Lagged pairs - correlation detection'
        },
        {
            **base_config,
            'level': 2,
            'generator': 'dip',
            'tactics': ['basic'],
            'epochs': 1000,
            'pass_threshold': 1.0,
            'description': 'Dip crashes - patience and timing'
        },
        {
            **base_config,
            'level': 3,
            'generator': 'noisy_trends',
            'tactics': ['basic', 'morphing'],
            'epochs': 1000,
            'pass_threshold': 0.5,
            'description': 'Noisy trends + swaps - fee optimization'
        },
        {
            **base_config,
            'level': 4,
            'generator': 'rotation',
            'tactics': ['basic', 'morphing'],
            'epochs': 1000,
            'pass_threshold': 0.5,
            'description': 'Sector rotation - multi-asset timing'
        },
        # Phase 2: Statistical Arbitrage (Levels 5-7)
        {
            **base_config,
            'level': 5,
            'generator': 'pairs_trading',
            'tactics': ['basic', 'morphing', 'pairs'],
            'epochs': 1500,
            'pass_threshold': 0.3,
            'description': 'Pairs trading - cointegrated pairs'
        },
        {
            **base_config,
            'level': 6,
            'generator': 'triangle_arbitrage',
            'tactics': ['basic', 'morphing', 'pairs'],
            'epochs': 1500,
            'pass_threshold': 0.2,
            'description': 'Triangle arbitrage opportunities'
        },
        {
            **base_config,
            'level': 7,
            'generator': 'index_arbitrage',
            'tactics': ['basic', 'morphing', 'pairs'],
            'epochs': 1500,
            'pass_threshold': 0.2,
            'description': 'Index vs component arbitrage'
        },
        # Phase 3: Momentum & Reversal (Levels 8-10)
        {
            **base_config,
            'level': 8,
            'generator': 'momentum',
            'tactics': ['basic', 'morphing'],
            'epochs': 1500,
            'pass_threshold': 0.3,
            'description': 'Momentum with trend persistence'
        },
        {
            **base_config,
            'level': 9,
            'generator': 'mean_reversion',
            'tactics': ['basic', 'morphing'],
            'epochs': 1500,
            'pass_threshold': 0.3,
            'description': 'Mean-reverting prices (OU process)'
        },
        {
            **base_config,
            'level': 10,
            'generator': 'breakout',
            'tactics': ['basic', 'morphing'],
            'epochs': 1500,
            'pass_threshold': 0.2,
            'description': 'Consolidation → Breakout patterns'
        },
        # Phase 4: Market Microstructure (Levels 11-13)
        {
            **base_config,
            'level': 11,
            'generator': 'bid_ask_bounce',
            'tactics': ['basic', 'morphing', 'market_making'],
            'epochs': 2000,
            'pass_threshold': 0.1,
            'description': 'Bid-ask bounce microstructure'
        },
        {
            **base_config,
            'level': 12,
            'generator': 'order_imbalance',
            'tactics': ['basic', 'morphing', 'market_making'],
            'epochs': 2000,
            'pass_threshold': 0.1,
            'description': 'Order book imbalance signals'
        },
        {
            **base_config,
            'level': 13,
            'generator': 'market_making',
            'tactics': ['basic', 'morphing', 'market_making'],
            'epochs': 2000,
            'pass_threshold': 0.1,
            'description': 'Market making with adverse selection'
        },
        # Phase 5: Volatility (Level 14)
        {
            **base_config,
            'level': 14,
            'generator': 'volatility_clustering',
            'tactics': ['basic', 'morphing'],
            'epochs': 2000,
            'pass_threshold': 0.1,
            'description': 'GARCH(1,1) volatility dynamics'
        },
    ]


def main():
    parser = argparse.ArgumentParser(description='Dojo Curriculum Training Manager')

    parser.add_argument('--curriculum', help='Curriculum config JSON file')
    parser.add_argument('--level', type=int, help='Train single level (0-3)')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    parser.add_argument('--checkpoint-dir', default='dojo_checkpoints', help='Checkpoint directory')
    parser.add_argument('--adaptive-calibration', action='store_true', help='Enable adaptive value calibration')
    parser.add_argument('--value-scale', type=float, default=1.0, help='Initial value calibration scale')
    parser.add_argument('--load-checkpoint', help='Load model from specific checkpoint path')
    parser.add_argument('--model-size', default='large', choices=['small', 'medium', 'large'],
                        help='Model size (overrides curriculum config)')
    parser.add_argument('--name', default='pretrain', help='Base name for run')

    args = parser.parse_args()

    # Load curriculum
    if args.curriculum:
        curriculum = load_curriculum_from_file(args.curriculum)
    elif args.level is not None:
        # Single level training
        default_curriculum = create_default_curriculum()
        if args.level >= len(default_curriculum):
            logging.error(f"Level {args.level} not in default curriculum (0-{len(default_curriculum)-1})")
            return
        curriculum = [default_curriculum[args.level]]
    else:
        # Full default curriculum
        curriculum = create_default_curriculum()

    # Create Dojo manager
    dojo = DojoManager(
        curriculum,
        checkpoint_dir=args.checkpoint_dir,
        adaptive_calibration=args.adaptive_calibration,
        value_scale=args.value_scale,
        load_checkpoint=args.load_checkpoint,
        model_size=args.model_size,
        base_name=args.name
    )

    # Resume from checkpoint if requested
    if args.resume:
        start_level = dojo.load_progress()
        logging.info(f"Resuming from level {start_level}")
        curriculum = curriculum[start_level:]

    # Run training
    dojo.run()


if __name__ == '__main__':
    main()
