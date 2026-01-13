#!/usr/bin/env python3
"""
Dojo v2: Full Curriculum Training Manager

TASK_045: Complete curriculum from Level 0 to 14 with:
- Three-phase training per level (Basic -> Noise -> Fees)
- Multi-period benchmark after each level (Bull/Bear/Sideways)
- Kitchen Sink v2 production configuration
- Auto-graduation with performance thresholds

Training Structure per Level:
  Phase A: 300 epochs (basic, no noise, no fees)
  Phase B: 250 epochs (+ synthetic-noise 0.3)
  Phase C: 250 epochs (+ noise + fees 16bps + slip 5bps)
  Benchmark: 21 epochs real data (7 bull + 7 bear + 7 sideways)
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [DOJO-V2] - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'dojo_v2.log')),
        logging.StreamHandler()
    ]
)

# ============================================================================
# CURRICULUM DEFINITION
# ============================================================================

CURRICULUM = [
    # Phase 1: Basic Mechanics (Levels 0-4)
    {'level': 0, 'generator': 'sine', 'description': 'Sine waves - learn trend detection'},
    {'level': 1, 'generator': 'lagged', 'description': 'Lagged pairs - correlation detection'},
    {'level': 2, 'generator': 'dip', 'description': 'Dip crashes - patience and timing'},
    {'level': 3, 'generator': 'noisy_trends', 'description': 'Noisy trends - fee optimization'},
    {'level': 4, 'generator': 'rotation', 'description': 'Sector rotation - multi-asset timing'},
    # Phase 2: Statistical Arbitrage (Levels 5-7)
    {'level': 5, 'generator': 'pairs_trading', 'description': 'Pairs trading - cointegrated pairs'},
    {'level': 6, 'generator': 'triangle_arbitrage', 'description': 'Triangle arbitrage'},
    {'level': 7, 'generator': 'index_arbitrage', 'description': 'Index vs component arbitrage'},
    # Phase 3: Momentum & Reversal (Levels 8-10)
    {'level': 8, 'generator': 'momentum', 'description': 'Momentum with trend persistence'},
    {'level': 9, 'generator': 'mean_reversion', 'description': 'Mean-reverting prices (OU process)'},
    {'level': 10, 'generator': 'breakout', 'description': 'Consolidation to breakout patterns'},
    # Phase 4: Market Microstructure (Levels 11-13)
    {'level': 11, 'generator': 'bid_ask_bounce', 'description': 'Bid-ask bounce microstructure'},
    {'level': 12, 'generator': 'order_imbalance', 'description': 'Order book imbalance signals'},
    {'level': 13, 'generator': 'market_making', 'description': 'Market making with adverse selection'},
    # Phase 5: Volatility (Level 14)
    {'level': 14, 'generator': 'volatility_clustering', 'description': 'GARCH(1,1) volatility dynamics'},
]

# ============================================================================
# PHASE CONFIGURATIONS
# ============================================================================

PHASES = [
    {
        'name': 'Phase A (Basic)',
        'epochs': 300,
        'synthetic_noise': 0.0,
        'fee': 0,
        'slippage': 0,
        'description': 'Learn basic pattern recognition without friction',
    },
    {
        'name': 'Phase B (Noise)',
        'epochs': 250,
        'synthetic_noise': 0.3,
        'fee': 0,
        'slippage': 0,
        'description': 'Build robustness to price perturbations',
    },
    {
        'name': 'Phase C (Fees)',
        'epochs': 250,
        'synthetic_noise': 0.3,
        'fee': 16,
        'slippage': 5,
        'description': 'Learn to overcome transaction costs',
    },
]

# ============================================================================
# KITCHEN SINK V2 BASE FLAGS
# ============================================================================

KITCHEN_SINK_V2_FLAGS = [
    # Core
    '--batch-size', '384',
    '--model-size', 'small',
    '--parallel-envs', '50',
    '--jax-env',
    '--bf16',
    '--synthetic-steps', '10000',

    # PopArt + GAE
    '--use-popart', '--popart-beta', '0.0001',
    '--use-gae', '--gae-lambda', '0.95',

    # MCTS
    '--sims', '24',
    '--sims-warmup', '300',
    '--sims-min', '4',

    # Prediction
    '--use-prediction-loss',
    '--prediction-loss-weight', '0.5',
    '--physics-warmup-steps', '10000',

    # Entropy control
    '--adaptive-entropy',
    '--entropy-coef', '0.2',
    '--entropy-hold-threshold', '0.55',
    '--entropy-boost-low', '0.4',
    '--entropy-boost-high', '0.6',

    # LR schedule
    '--lr-schedule', 'cosine',
    '--lr-warmup-epochs', '20',
    '--lr-min', '1e-6',

    # Gradient clipping
    '--per-head-clipping',
    '--clip-grad-value', '1.0',
    '--clip-grad-policy', '1.0',
    '--clip-grad-dynamics', '1.0',
    '--clip-grad-encoder', '1.0',

    # Exploration
    '--epsilon-greedy', '0.1',

    # Realism
    '--probabilistic-fills',
    '--borrow-rate', '0.03',
    '--episode-variance', '30',

    # Maintenance
    '--redo', '--redo-interval', '50',

    # ATC contrastive learning
    '--atc-loss',
    '--atc-weight', '0.1',

    # DS24 monitoring
    '--ds24-level', '3',
    '--ds24-intermediates-every', '10',
]

# Graduation thresholds per level
GRADUATION_THRESHOLDS = {
    0: 2.0,   # Easy: sine waves
    1: 1.5,   # Lagged pairs
    2: 1.5,   # Dip crashes
    3: 1.0,   # Noisy trends
    4: 1.0,   # Rotation
    5: 0.8,   # Pairs trading
    6: 0.8,   # Triangle arb
    7: 0.8,   # Index arb
    8: 0.5,   # Momentum
    9: 0.5,   # Mean reversion
    10: 0.5,  # Breakout
    11: 0.3,  # Bid-ask bounce
    12: 0.3,  # Order imbalance
    13: 0.3,  # Market making
    14: 0.3,  # Volatility clustering
}


class DojoV2:
    """
    Dojo v2: Full curriculum training manager.

    Trains models from Level 0 to 14 with 3-phase structure per level
    and multi-period benchmarks after each level.
    """

    def __init__(
        self,
        checkpoint_dir: str = 'jax_ckpt_fast',
        model_size: str = 'small',
        db_path: str = 'trading_data.db',
    ):
        self.checkpoint_dir = checkpoint_dir
        self.model_size = model_size
        self.db_path = db_path
        self.benchmark_config = self._load_benchmark_config()

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs('logs/benchmarks', exist_ok=True)

    def _load_benchmark_config(self) -> Dict:
        """Load benchmark periods configuration."""
        config_path = PROJECT_ROOT / 'config' / 'benchmark_periods.json'
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        else:
            logging.warning(f"Benchmark config not found at {config_path}")
            return {}

    def run_curriculum(self, start_level: int = 0, end_level: int = 14):
        """
        Run full curriculum from start_level to end_level.

        Args:
            start_level: First level to train (0-14)
            end_level: Last level to train (0-14)
        """
        logging.info("=" * 70)
        logging.info("DOJO V2 - FULL CURRICULUM TRAINING")
        logging.info("=" * 70)
        logging.info(f"Levels: {start_level} -> {end_level}")
        logging.info(f"Phases per level: {len(PHASES)}")
        logging.info(f"Model size: {self.model_size}")
        logging.info("=" * 70)

        checkpoint = None  # Start fresh

        for level_idx in range(start_level, end_level + 1):
            level_info = CURRICULUM[level_idx]
            level = level_info['level']
            generator = level_info['generator']
            description = level_info['description']

            logging.info("")
            logging.info("=" * 70)
            logging.info(f"LEVEL {level}: {generator.upper()}")
            logging.info(f"Description: {description}")
            logging.info("=" * 70)

            # Train all phases for this level
            for phase_idx, phase in enumerate(PHASES):
                phase_name = phase['name']
                phase_epochs = phase['epochs']

                logging.info("")
                logging.info(f">>> {phase_name} ({phase_epochs} epochs)")
                logging.info(f"    Noise: {phase['synthetic_noise']}, Fee: {phase['fee']}bps, Slip: {phase['slippage']}bps")

                # First phase of new level: reset optimizer
                reset_optimizer = (phase_idx == 0 and level > start_level)

                checkpoint = self._train_phase(
                    level=level,
                    phase=phase,
                    phase_idx=phase_idx,
                    checkpoint=checkpoint,
                    reset_optimizer=reset_optimizer,
                )

                if checkpoint is None:
                    logging.error(f"Training failed at Level {level} {phase_name}")
                    return False

            # Run benchmark after all phases complete
            logging.info("")
            logging.info(f">>> BENCHMARK: Level {level}")
            benchmark_results = self._run_benchmark(checkpoint, level)
            self._save_benchmark_results(level, benchmark_results)

            # Check graduation
            threshold = GRADUATION_THRESHOLDS.get(level, 0.5)
            avg_alpha = sum(r.get('alpha', 0) for r in benchmark_results.values()) / len(benchmark_results)

            if avg_alpha >= threshold:
                logging.info(f"LEVEL {level} GRADUATED! (alpha={avg_alpha:.2f}% >= {threshold}%)")
            else:
                logging.warning(f"Level {level} below threshold (alpha={avg_alpha:.2f}% < {threshold}%)")
                logging.warning("Continuing to next level anyway...")

        logging.info("")
        logging.info("=" * 70)
        logging.info("CURRICULUM COMPLETE!")
        logging.info("=" * 70)
        return True

    def _train_phase(
        self,
        level: int,
        phase: Dict,
        phase_idx: int,
        checkpoint: Optional[str],
        reset_optimizer: bool,
    ) -> Optional[str]:
        """
        Train a single phase of a level.

        Returns:
            Path to checkpoint after training, or None on failure
        """
        phase_char = chr(ord('A') + phase_idx)  # A, B, C
        experiment_name = f'dojo_v2_L{level}_phase_{phase_char}'

        # Build command
        cmd = [
            'python', 'train_jax_fast.py',
            '--synthetic-level', str(level),
            '--epochs', str(phase['epochs']),
            '--synthetic-noise', str(phase['synthetic_noise']),
            '--fee', str(phase['fee']),
            '--slippage', str(phase['slippage']),
            '--name', experiment_name,
            '--model-size', self.model_size,
        ]

        # Add kitchen sink v2 flags
        cmd.extend(KITCHEN_SINK_V2_FLAGS)

        # Handle checkpoint loading
        if checkpoint:
            cmd.extend(['--load-checkpoint', checkpoint])
            if reset_optimizer:
                cmd.append('--reset-optimizer')
                cmd.append('--clear-buffer')
                logging.info(f"    Loading from {checkpoint} (reset optimizer for new level)")
            else:
                logging.info(f"    Continuing from {checkpoint}")

        # Run training
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=False,
                timeout=3600 * 8,  # 8 hour timeout per phase
            )

            elapsed = time.time() - start_time
            logging.info(f"    Phase completed in {elapsed/60:.1f} minutes")

            if result.returncode != 0:
                logging.error(f"Training failed with return code {result.returncode}")
                return None

        except subprocess.TimeoutExpired:
            logging.error("Training timed out!")
            return None
        except Exception as e:
            logging.error(f"Training error: {e}")
            return None

        # Return checkpoint path
        new_checkpoint = os.path.join(self.checkpoint_dir, experiment_name)
        if os.path.exists(new_checkpoint):
            return new_checkpoint
        else:
            logging.warning(f"Checkpoint not found at {new_checkpoint}")
            return checkpoint  # Return previous checkpoint

    def _run_benchmark(self, checkpoint: str, level: int) -> Dict[str, Dict]:
        """
        Run benchmark on real data (bull/bear/sideways periods).

        Returns:
            Dict with results for each period
        """
        results = {}

        if not self.benchmark_config:
            logging.warning("No benchmark config, skipping benchmark")
            return results

        periods = self.benchmark_config.get('periods', [])
        bench_cfg = self.benchmark_config.get('benchmark_config', {})

        epochs_per_period = bench_cfg.get('epochs_per_period', 7)
        steps = bench_cfg.get('steps_per_epoch', 1440)
        parallel_envs = bench_cfg.get('parallel_envs', 4)
        warmup_epochs = bench_cfg.get('warmup_epochs', 7)

        for period in periods:
            period_name = period['name']
            start_date = period['start_date']
            end_date = period['end_date']

            logging.info(f"    Benchmark: {period_name} ({start_date} to {end_date})")

            # Run evaluation (no training, no checkpoint save)
            cmd = [
                'python', 'train_jax_fast.py',
                '--db-path', self.db_path,
                '--start-date', start_date,
                '--end-date', end_date,
                '--epochs', str(warmup_epochs + epochs_per_period),
                '--steps', str(steps),
                '--parallel-envs', str(parallel_envs),
                '--load-checkpoint', checkpoint,
                '--no-checkpoint',  # Don't save
                '--jax-env',
                '--bf16',
                '--model-size', self.model_size,
                '--name', f'benchmark_L{level}_{period_name}',
            ]

            try:
                result = subprocess.run(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    timeout=1800,  # 30 min timeout
                )

                # Parse results from output
                output = result.stdout + result.stderr
                metrics = self._parse_benchmark_output(output)
                results[period_name] = metrics

                logging.info(f"      {period_name}: reward={metrics.get('reward', 0):.2f}, alpha={metrics.get('alpha', 0):.2f}%")

            except Exception as e:
                logging.error(f"      Benchmark {period_name} failed: {e}")
                results[period_name] = {'reward': 0, 'alpha': 0, 'error': str(e)}

        return results

    def _parse_benchmark_output(self, output: str) -> Dict:
        """Parse training output to extract comprehensive benchmark metrics."""
        import re

        metrics = {
            # Core performance
            'reward': 0.0,
            'reward_std': 0.0,
            'alpha': 0.0,
            'market_return': 0.0,
            'agent_return': 0.0,
            # Action distribution
            'n_sell': 0,
            'n_hold': 0,
            'n_buy': 0,
            'n_limit_sell': 0,
            'n_limit_buy': 0,
            'pct_sell': 0.0,
            'pct_hold': 0.0,
            'pct_buy': 0.0,
            'pct_limit_sell': 0.0,
            'pct_limit_buy': 0.0,
            # Training metrics
            'entropy': 0.0,
            'value_loss': 0.0,
            'policy_loss': 0.0,
            'total_loss': 0.0,
            # Timing
            'self_play_time': 0.0,
            'epochs_completed': 0,
        }

        # Collect all epoch data for statistics
        rewards = []
        alphas = []

        # Pattern: "Epoch   30 | Reward:  +42.50 | α: +318.70% (Mkt: +2.6%)"
        epoch_pattern = r'Epoch\s+(\d+)\s+\|\s+Reward:\s+([\+\-]?\d+\.?\d*)\s+\|\s+α:\s+([\+\-]?\d+\.?\d*)%\s+\(Mkt:\s+([\+\-]?\d+\.?\d*)%\)'
        epoch_matches = re.findall(epoch_pattern, output)

        for match in epoch_matches:
            epoch_num = int(match[0])
            reward = float(match[1])
            alpha = float(match[2])
            mkt = float(match[3])
            rewards.append(reward)
            alphas.append(alpha)
            metrics['epochs_completed'] = epoch_num
            metrics['market_return'] = mkt  # Last market return

        if rewards:
            metrics['reward'] = sum(rewards) / len(rewards)
            metrics['reward_std'] = (sum((r - metrics['reward'])**2 for r in rewards) / len(rewards)) ** 0.5
            metrics['alpha'] = sum(alphas) / len(alphas)
            # Approximate agent return from alpha and market
            metrics['agent_return'] = metrics['market_return'] + metrics['alpha']

        # Action distribution: "S/H/B/LS/LB: 7/23/6/6/8"
        action_pattern = r'S/H/B/LS/LB:\s*(\d+)/(\d+)/(\d+)/(\d+)/(\d+)'
        action_matches = re.findall(action_pattern, output)
        if action_matches:
            # Aggregate all action counts
            totals = [0, 0, 0, 0, 0]
            for match in action_matches:
                for i, val in enumerate(match):
                    totals[i] += int(val)

            metrics['n_sell'] = totals[0]
            metrics['n_hold'] = totals[1]
            metrics['n_buy'] = totals[2]
            metrics['n_limit_sell'] = totals[3]
            metrics['n_limit_buy'] = totals[4]

            total = sum(totals)
            if total > 0:
                metrics['pct_sell'] = totals[0] / total * 100
                metrics['pct_hold'] = totals[1] / total * 100
                metrics['pct_buy'] = totals[2] / total * 100
                metrics['pct_limit_sell'] = totals[3] / total * 100
                metrics['pct_limit_buy'] = totals[4] / total * 100

        # Self-play time: "Self-play:  10.2s"
        time_pattern = r'Self-play:\s+([\d\.]+)s'
        time_matches = re.findall(time_pattern, output)
        if time_matches:
            metrics['self_play_time'] = sum(float(t) for t in time_matches)

        # Training metrics from DS24 or logs
        # Loss pattern (if available)
        loss_pattern = r'Loss:\s+([\d\.]+)'
        loss_matches = re.findall(loss_pattern, output)
        if loss_matches:
            metrics['total_loss'] = float(loss_matches[-1])

        # Entropy pattern
        entropy_pattern = r'Ent:\s+([\d\.]+)'
        entropy_matches = re.findall(entropy_pattern, output)
        if entropy_matches:
            metrics['entropy'] = float(entropy_matches[-1])

        return metrics

    def _save_benchmark_results(self, level: int, results: Dict[str, Dict]):
        """Save comprehensive benchmark results to CSV."""
        csv_path = f'logs/benchmarks/level_{level:02d}_benchmark.csv'

        # Define all columns
        columns = [
            'period', 'reward', 'reward_std', 'alpha', 'market_return', 'agent_return',
            'n_sell', 'n_hold', 'n_buy', 'n_limit_sell', 'n_limit_buy',
            'pct_sell', 'pct_hold', 'pct_buy', 'pct_limit_sell', 'pct_limit_buy',
            'entropy', 'total_loss', 'self_play_time', 'epochs_completed'
        ]

        with open(csv_path, 'w') as f:
            f.write(','.join(columns) + '\n')
            for period, metrics in results.items():
                row = [period]
                for col in columns[1:]:
                    val = metrics.get(col, 0)
                    if isinstance(val, float):
                        row.append(f'{val:.4f}')
                    else:
                        row.append(str(val))
                f.write(','.join(row) + '\n')

        logging.info(f"    Benchmark results saved to {csv_path}")

        # Also save a JSON with full details for analytics tool
        json_path = f'logs/benchmarks/level_{level:02d}_benchmark.json'
        with open(json_path, 'w') as f:
            json.dump({
                'level': level,
                'timestamp': datetime.now().isoformat(),
                'results': results,
            }, f, indent=2)

        logging.info(f"    Benchmark JSON saved to {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Dojo v2: Full curriculum training (Level 0-14)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full curriculum from scratch
  python dojo_v2.py

  # Start from level 5
  python dojo_v2.py --start-level 5

  # Only levels 0-4
  python dojo_v2.py --end-level 4

  # Large model
  python dojo_v2.py --model-size large
"""
    )

    parser.add_argument('--start-level', type=int, default=0,
                        help='Start level (0-14, default: 0)')
    parser.add_argument('--end-level', type=int, default=14,
                        help='End level (0-14, default: 14)')
    parser.add_argument('--model-size', default='small',
                        choices=['small', 'medium', 'large'],
                        help='Model size (default: small)')
    parser.add_argument('--checkpoint-dir', default='jax_ckpt_fast',
                        help='Checkpoint directory (default: jax_ckpt_fast)')
    parser.add_argument('--db-path', default='trading_data.db',
                        help='Database path for benchmarks (default: trading_data.db)')

    args = parser.parse_args()

    # Validate levels
    if args.start_level < 0 or args.start_level > 14:
        parser.error(f"start-level must be 0-14, got {args.start_level}")
    if args.end_level < 0 or args.end_level > 14:
        parser.error(f"end-level must be 0-14, got {args.end_level}")
    if args.start_level > args.end_level:
        parser.error(f"start-level ({args.start_level}) > end-level ({args.end_level})")

    # Run curriculum
    dojo = DojoV2(
        checkpoint_dir=args.checkpoint_dir,
        model_size=args.model_size,
        db_path=args.db_path,
    )

    success = dojo.run_curriculum(
        start_level=args.start_level,
        end_level=args.end_level,
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
