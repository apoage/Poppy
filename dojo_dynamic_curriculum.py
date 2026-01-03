#!/usr/bin/env python3
"""
Dojo: Dynamic Curriculum Training Manager

New rules:
- Check mastery every 250 epochs
- Adjust difficulty (fees, slippage, level) based on performance
- Progressive difficulty within each level
"""

import argparse
import logging
import json
import os
import time
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [DOJO] - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'dojo_dynamic.log')),
        logging.StreamHandler()
    ]
)


class DynamicDojo:
    """
    Dynamic curriculum manager with adaptive difficulty.

    Rules:
    1. Check mastery every 250 epochs
    2. If passing → Increase difficulty (add fees/slippage)
    3. If struggling → Give more time or reduce difficulty
    4. Progressive phases within each level
    """

    def __init__(self, checkpoint_dir: str = 'dojo_checkpoints', model_size: str = 'small',
                 ds24_boost_every: int = 10, ds24_snapshot_every: int = 50):
        self.checkpoint_dir = checkpoint_dir
        self.model_size = model_size
        # DS24 dynamic level boosting
        self.ds24_boost_every = ds24_boost_every        # Boost to L4 every N epochs
        self.ds24_snapshot_every = ds24_snapshot_every  # Full L6 snapshot every N epochs
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Difficulty progression
        self.difficulty_phases = [
            {'name': 'Phase 1', 'fee': 0, 'slippage': 0, 'threshold': 2.5},    # No friction
            {'name': 'Phase 2', 'fee': 5, 'slippage': 0, 'threshold': 2.0},    # Tiny fee
            {'name': 'Phase 3', 'fee': 10, 'slippage': 2, 'threshold': 1.5},   # Light costs
            {'name': 'Phase 4', 'fee': 16, 'slippage': 5, 'threshold': 1.0},   # Realistic
            {'name': 'Phase 5', 'fee': 26, 'slippage': 5, 'threshold': 0.5},   # Full costs
        ]

    def train_level_dynamic(self, level: int, generator: str, tactics: List[str],
                            previous_level_checkpoint: Optional[str] = None):
        """
        Train level with dynamic difficulty adjustment.

        Args:
            level: Curriculum level (0-3)
            generator: Synthetic generator type
            tactics: List of tactics to enable
            previous_level_checkpoint: Path to checkpoint from previous level (for level transitions)
        """
        logging.info("="*60)
        logging.info(f"DYNAMIC TRAINING - Level {level} ({generator})")
        logging.info("="*60)

        current_phase = 0
        total_epochs = 0
        max_total_epochs = 2000  # Safety limit
        checkpoint_interval = 250

        enable_swaps = 'morphing' in tactics
        experiment_name = f'dojo_dynamic_L{level}_{generator}'

        while current_phase < len(self.difficulty_phases) and total_epochs < max_total_epochs:
            phase = self.difficulty_phases[current_phase]

            logging.info("")
            logging.info(f"🎯 {phase['name']}")
            logging.info(f"   Fee: {phase['fee']} bps | Slippage: {phase['slippage']} bps")
            logging.info(f"   Pass Threshold: +{phase['threshold']:.1f}%")
            logging.info("")

            # Train for checkpoint_interval epochs
            success = self._train_phase(
                level=level,
                generator=generator,
                enable_swaps=enable_swaps,
                fee=phase['fee'],
                slippage=phase['slippage'],
                epochs=checkpoint_interval,
                experiment_name=experiment_name,
                total_epochs_so_far=total_epochs,
                previous_level_checkpoint=previous_level_checkpoint
            )

            if not success:
                logging.error(f"Training failed in {phase['name']}")
                return False

            total_epochs += checkpoint_interval

            # Check mastery
            passed = self._check_mastery(
                threshold=phase['threshold'],
                metrics_file='last_run_metrics.json'
            )

            if passed:
                logging.info(f"✅ {phase['name']} PASSED!")
                logging.info(f"   Advancing to next difficulty phase...")
                current_phase += 1

                if current_phase >= len(self.difficulty_phases):
                    logging.info("")
                    logging.info("🎓 LEVEL COMPLETE! All phases mastered!")
                    logging.info(f"   Total epochs: {total_epochs}")
                    logging.info("")
                    return True
            else:
                logging.warning(f"❌ {phase['name']} not passed yet")
                logging.info(f"   Continuing with same difficulty for +{checkpoint_interval} epochs")

                # Safety check: don't loop forever on same phase
                if total_epochs >= max_total_epochs:
                    logging.error(f"Hit max epoch budget ({max_total_epochs})")
                    return False

        return True

    def _train_phase(self, level: int, generator: str, enable_swaps: bool,
                     fee: int, slippage: int, epochs: int,
                     experiment_name: str, total_epochs_so_far: int,
                     previous_level_checkpoint: Optional[str] = None) -> bool:
        """
        Train for specified epochs with given difficulty settings.

        Args:
            previous_level_checkpoint: Path to checkpoint from previous level (for level transitions)

        Returns:
            True if training succeeded, False otherwise
        """
        import subprocess

        # Tournament-validated optimal configurations (2026-01-03)
        # Sweet spot: ~20,000 samples/epoch
        cmd = [
            'python', 'train_jax_fast.py',
            '--synthetic-level', str(level),
            '--synthetic-steps', '10000',
            '--epochs', str(epochs),
            '--batch-size', '512',
            '--model-size', self.model_size,
            '--name', experiment_name,
            '--fee', str(fee),
            '--slippage', str(slippage),
            # Env config: 20 envs × 400 steps = 8,000 samples (avoid OOM with large pred head)
            '--parallel-envs', '20',
            '--steps', '400',
            '--jax-env',  # Use JAX environment with batched transfers
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
            # Tournament winner: Adaptive entropy pumping
            '--adaptive-entropy',
            '--entropy-hold-threshold', '0.7',
            '--entropy-boost-low', '0.05',
            '--entropy-boost-high', '0.3',
            # Adaptive value calibration for curriculum stability
            '--adaptive-calibration',
            # BFloat16 mixed precision
            '--bf16',
            # DS24 logging with dynamic level boosting
            # L3 base (full logging), L4 every boost_every
            '--ds24-level', '3',
            '--ds24-intermediates-every', '5',
            '--ds24-boost-every', str(self.ds24_boost_every),
            '--ds24-snapshot-every', '10',
            '--ds24-save-tensors',  # Save tensor snapshots every snapshot-every epochs
            # Data regeneration for diversity
            '--regenerate-data-every', '50',
        ]

        if enable_swaps:
            cmd.append('--enable-swaps')

        # Safe checkpoints for level 4+ (protect learned knowledge)
        if level >= 4:
            cmd.extend([
                '--safe-checkpoints',
                '--rollback-patience', '50',
                '--rollback-threshold', '0.5',
                '--max-rollbacks', '3',
                '--keep-best-n', '3',
                '--lr-decay-on-rollback', '0.5',
            ])
            # Periodic evaluation for levels 5+
            if level >= 5:
                cmd.extend([
                    '--eval-every', '50',
                    '--eval-level', '4',  # Evaluate on rotation
                    '--eval-steps', '200',
                ])
            logging.info(f"🛡️ Safe checkpoints enabled for level {level}")

        # Checkpoint loading logic
        checkpoint_dir = f'jax_ckpt_fast/{experiment_name}'
        is_first_phase = (total_epochs_so_far == 0)

        if is_first_phase and previous_level_checkpoint and os.path.exists(previous_level_checkpoint):
            # First phase of NEW level - load from previous level, reset optimizer
            cmd.extend(['--load-checkpoint', previous_level_checkpoint])
            cmd.append('--reset-optimizer')
            cmd.append('--clear-buffer')
            logging.info(f"📂 Starting new level from previous checkpoint: {previous_level_checkpoint}")
            logging.info(f"   🔄 Resetting optimizer for new level")
        elif not is_first_phase and os.path.exists(checkpoint_dir):
            # Continuing same level - resume without reset
            cmd.extend(['--load-checkpoint', checkpoint_dir])
            logging.info(f"📂 Resuming from checkpoint (after {total_epochs_so_far} epochs)")

        logging.info(f"🚀 Training: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=False)

        return result.returncode == 0

    def _check_mastery(self, threshold: float, metrics_file: str) -> bool:
        """
        Check if recent performance meets threshold.

        Args:
            threshold: Required average reward (%)
            metrics_file: Path to metrics JSON

        Returns:
            True if passed, False otherwise
        """
        try:
            with open(metrics_file) as f:
                metrics = json.load(f)
        except FileNotFoundError:
            logging.error(f"Metrics file not found: {metrics_file}")
            return False

        episode_rewards = metrics.get('episode_rewards', [])

        if not episode_rewards:
            logging.error("No episode rewards in metrics")
            return False

        # Use last 50 epochs or all if less than 50
        recent_rewards = episode_rewards[-50:] if len(episode_rewards) >= 50 else episode_rewards
        recent_avg = np.mean(recent_rewards)

        passed = recent_avg >= threshold

        logging.info("")
        logging.info("─" * 60)
        logging.info("MASTERY CHECK")
        logging.info("─" * 60)
        logging.info(f"Recent Avg Reward: {recent_avg:+.2f}%")
        logging.info(f"Pass Threshold:    {threshold:+.2f}%")
        logging.info(f"Epochs Evaluated:  {len(recent_rewards)}")
        logging.info(f"Result:            {'✅ PASS' if passed else '❌ FAIL'}")
        logging.info("─" * 60)
        logging.info("")

        return passed


def create_dynamic_curriculum() -> List[Dict]:
    """
    Create curriculum for dynamic training (Levels 0-14).

    Each level uses progressive difficulty (phases).
    Optimal env config: 50 envs × 400 steps = 20,000 samples/epoch
    """
    return [
        # Phase 1: Basic Mechanics (Levels 0-4)
        {
            'level': 0,
            'generator': 'sine',
            'tactics': ['basic'],
            'description': 'Sine waves - learn trend detection'
        },
        {
            'level': 1,
            'generator': 'lagged',
            'tactics': ['basic'],
            'description': 'Lagged pairs - learn correlation detection'
        },
        {
            'level': 2,
            'generator': 'dip',
            'tactics': ['basic'],
            'description': 'Dip crashes - learn patience and timing'
        },
        {
            'level': 3,
            'generator': 'noisy_trends',
            'tactics': ['basic', 'morphing'],
            'description': 'Noisy trends + swaps - learn fee optimization'
        },
        {
            'level': 4,
            'generator': 'rotation',
            'tactics': ['basic', 'morphing'],
            'description': 'Sector rotation - multi-asset timing'
        },
        # Phase 2: Statistical Arbitrage (Levels 5-7)
        {
            'level': 5,
            'generator': 'pairs_trading',
            'tactics': ['basic', 'morphing', 'pairs'],
            'description': 'Pairs trading - cointegrated pairs'
        },
        {
            'level': 6,
            'generator': 'triangle_arbitrage',
            'tactics': ['basic', 'morphing', 'pairs'],
            'description': 'Triangle arbitrage opportunities'
        },
        {
            'level': 7,
            'generator': 'index_arbitrage',
            'tactics': ['basic', 'morphing', 'pairs'],
            'description': 'Index vs component arbitrage'
        },
        # Phase 3: Momentum & Reversal (Levels 8-10)
        {
            'level': 8,
            'generator': 'momentum',
            'tactics': ['basic', 'morphing'],
            'description': 'Momentum with trend persistence'
        },
        {
            'level': 9,
            'generator': 'mean_reversion',
            'tactics': ['basic', 'morphing'],
            'description': 'Mean-reverting prices (OU process)'
        },
        {
            'level': 10,
            'generator': 'breakout',
            'tactics': ['basic', 'morphing'],
            'description': 'Consolidation → Breakout patterns'
        },
        # Phase 4: Market Microstructure (Levels 11-13)
        {
            'level': 11,
            'generator': 'bid_ask_bounce',
            'tactics': ['basic', 'morphing', 'market_making'],
            'description': 'Bid-ask bounce microstructure'
        },
        {
            'level': 12,
            'generator': 'order_imbalance',
            'tactics': ['basic', 'morphing', 'market_making'],
            'description': 'Order book imbalance signals'
        },
        {
            'level': 13,
            'generator': 'market_making',
            'tactics': ['basic', 'morphing', 'market_making'],
            'description': 'Market making with adverse selection'
        },
        # Phase 5: Volatility (Level 14)
        {
            'level': 14,
            'generator': 'volatility_clustering',
            'tactics': ['basic', 'morphing'],
            'description': 'GARCH(1,1) volatility dynamics'
        },
    ]


def main():
    parser = argparse.ArgumentParser(description='Dojo Dynamic Curriculum Manager')

    parser.add_argument('--level', type=int, default=0, help='Start level (0-14)')
    parser.add_argument('--model-size', default='small', choices=['small', 'medium', 'large'],
                        help='Model size (small=15M, medium=28M, large=50M+)')
    parser.add_argument('--checkpoint-dir', default='dojo_checkpoints', help='Checkpoint directory')
    parser.add_argument('--previous-checkpoint', type=str, default=None,
                        help='Path to checkpoint from previous level (resets optimizer on load)')
    # DS24 dynamic level boosting settings
    parser.add_argument('--ds24-boost-every', type=int, default=10,
                        help='Boost DS24 to L4 every N epochs (default: 10)')
    parser.add_argument('--ds24-snapshot-every', type=int, default=50,
                        help='Full DS24 L6 snapshot every N epochs (default: 50)')

    args = parser.parse_args()

    # Create curriculum
    curriculum = create_dynamic_curriculum()

    if args.level >= len(curriculum):
        logging.error(f"Invalid level {args.level} (max: {len(curriculum)-1})")
        return

    # Create Dojo manager
    dojo = DynamicDojo(
        checkpoint_dir=args.checkpoint_dir,
        model_size=args.model_size,
        ds24_boost_every=args.ds24_boost_every,
        ds24_snapshot_every=args.ds24_snapshot_every,
    )

    # Train selected level
    level_config = curriculum[args.level]

    logging.info(f"Starting training with {args.model_size.upper()} model")
    logging.info(f"📈 DS24: L3 base → L4 every {args.ds24_boost_every} epochs → L6 every {args.ds24_snapshot_every} epochs")
    if args.previous_checkpoint:
        logging.info(f"📂 Will load from previous level: {args.previous_checkpoint}")
        logging.info(f"   🔄 Optimizer will be reset for new level")

    success = dojo.train_level_dynamic(
        level=level_config['level'],
        generator=level_config['generator'],
        tactics=level_config['tactics'],
        previous_level_checkpoint=args.previous_checkpoint
    )

    if success:
        logging.info("🎉 Training completed successfully!")
    else:
        logging.error("❌ Training failed")


if __name__ == '__main__':
    main()
