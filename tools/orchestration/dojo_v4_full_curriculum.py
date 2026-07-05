#!/usr/bin/env python3
"""
Dojo v4: Simplified Curriculum Training (Synthetic + Historical)

Automated curriculum that trains a fresh model through:

PHASE 1: Per-Level Training (L0-L14)
    For each level (100 epochs per level, 1500 total):
    - 50 epochs: Basic training (0% noise)
    - 50 epochs: Progressive noise (0%->100% over 50 epochs)

    Auto-enables burning_money flag if hold rate > 60%

PHASE 2: Speed Training (300 epochs)
    Single continuous run with weighted level cycling:
    - --weighted-synthetic flag for random level selection
    - Weighted distribution: L0-L4 (~12%), L5-L9 (~32%), L10-L14 (~56%)
    - Adaptive noise always on (multi-modal)
    - Adaptive burn controller (auto-escalates)
    - 50 parallel envs, single CSV, progress bar

PHASE 3: Historical Data
    Run through all historical data (96 years):
    - Uses dojo_v3_historical stratified mode
    - Cycles through periods to prevent forgetting

Configuration:
    - 50 parallel envs throughout synthetic phases
    - Fresh model (no checkpoint loading initially)
    - One-shot execution (no looping)
    - Kitchen Sink v2 flags + GPU optimizations
"""

# ============================================================================
# MODEL NAMING
# ============================================================================
MODEL_NAME = "phoenix"  # New model series for fresh training
MODEL_VERSION = "1"

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [DOJO-V4] - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'dojo_v4_full_curriculum.log')),
        logging.StreamHandler()
    ]
)

# ============================================================================
# CURRICULUM DEFINITION
# ============================================================================

CURRICULUM_LEVELS = [
    # Phase 1: Basic Mechanics (Levels 0-4)
    {'level': 0, 'generator': 'sine', 'description': 'Sine waves - trend detection'},
    {'level': 1, 'generator': 'lagged', 'description': 'Lagged pairs - correlation'},
    {'level': 2, 'generator': 'dip', 'description': 'Dip crashes - timing'},
    {'level': 3, 'generator': 'noisy_trends', 'description': 'Noisy trends - fee optimization'},
    {'level': 4, 'generator': 'rotation', 'description': 'Sector rotation - multi-asset timing'},
    # Phase 2: Statistical Arbitrage (Levels 5-7)
    {'level': 5, 'generator': 'pairs_trading', 'description': 'Pairs trading - cointegrated pairs'},
    {'level': 6, 'generator': 'triangle_arbitrage', 'description': 'Triangle arbitrage'},
    {'level': 7, 'generator': 'index_arbitrage', 'description': 'Index vs component arbitrage'},
    # Phase 3: Momentum & Reversal (Levels 8-10)
    {'level': 8, 'generator': 'momentum', 'description': 'Momentum with trend persistence'},
    {'level': 9, 'generator': 'mean_reversion', 'description': 'Mean-reverting prices (OU)'},
    {'level': 10, 'generator': 'breakout', 'description': 'Consolidation to breakout'},
    # Phase 4: Market Microstructure (Levels 11-13)
    {'level': 11, 'generator': 'bid_ask_bounce', 'description': 'Bid-ask bounce microstructure'},
    {'level': 12, 'generator': 'order_imbalance', 'description': 'Order book imbalance signals'},
    {'level': 13, 'generator': 'market_making', 'description': 'Market making with adverse selection'},
    # Phase 5: Volatility (Level 14)
    {'level': 14, 'generator': 'volatility_clustering', 'description': 'GARCH(1,1) volatility dynamics'},
]

# ============================================================================
# SYNTHETIC TRAINING SCHEDULE (Phase 1)
# ============================================================================

# Per-level: 300 epochs basic + 450 epochs with progressive noise = 750 epochs per level
# Must achieve positive MA50 reward before advancing to next level
SYNTHETIC_SCHEDULE = {
    'epochs_basic': 300,         # Epochs with 0% noise (learn clean pattern)
    'epochs_noise': 450,         # Epochs with progressive noise (0%->100%)
    'min_ma50_to_advance': 0.0,  # Must have positive MA50 reward to advance
    'max_extra_epochs': 100,     # Max extra epochs if MA50 not met
}

# Hold rate threshold to trigger burning money
HOLD_RATE_THRESHOLD = 0.60  # 60% hold actions triggers burning_money

# ============================================================================
# SPEED TRAINING SCHEDULE (Phase 2)
# ============================================================================

SPEED_TRAINING = {
    'total_epochs': 300,         # Total epochs for speed training
    'min_reward_threshold': 5.0, # If avg reward < 5, continue training
    'max_retries': 3,            # Max retry rounds if below threshold
}

# Weighted level distribution: fewer easy, more hard
# Format: {level: weight} - higher weight = more likely to be selected
LEVEL_WEIGHTS = {
    0: 0.5,   # Very easy - minimal sampling
    1: 0.5,   # Easy
    2: 1.0,   # Easy
    3: 1.0,   # Easy
    4: 1.5,   # Medium-easy
    5: 2.0,   # Medium
    6: 2.0,   # Medium
    7: 2.0,   # Medium
    8: 3.0,   # Medium-hard
    9: 3.0,   # Medium-hard
    10: 3.5,  # Hard
    11: 4.0,  # Hard (microstructure)
    12: 4.0,  # Hard (order imbalance)
    13: 4.5,  # Very hard (market making)
    14: 5.0,  # Hardest (volatility clustering)
}

# ============================================================================
# ADAPTIVE DIFFICULTY THRESHOLDS (Dojo V2 style)
# ============================================================================

# Features are ENABLED based on performance thresholds, not fixed stages
ADAPTIVE_THRESHOLDS = {
    # Noise: Enable --adaptive-noise when reward > 5 AND rising
    # AdaptiveNoiseController manages 4 noise types in order:
    #   1. Price noise (0%→100%)
    #   2. Temporal jitter (0→5 steps)
    #   3. Time dilation (0%→30%)
    #   4. Phase shift (0%→50%)
    'noise_enable_reward': 5.0,

    # Fees: Enable when reward > 5 AND rising (after adaptive noise active)
    'fees_enable_reward': 5.0,
    'fees_max_bps': 28,            # Max maker fee (bps)
    'slippage_max_bps': 2,         # Max slippage (bps)
    'fees_ramp_rate': 0.5,         # BPS increase per epoch when conditions met

    # Burning money: Enable when hold rate > 50%
    'burning_money_hold_threshold': 0.50,
}

# Phase 2 configuration
SPEED_TRAINING_CONFIG = {
    'total_epochs': 300,
    'min_reward_threshold': 5.0,   # Per-epoch reward threshold for retry
    'max_retries': 3,
    'reward_window': 10,           # Epochs to track for "rising" detection
}

# Benchmark config
BENCHMARK_MIN_REWARD = 5.0  # Minimum acceptable reward to pass benchmark

# ============================================================================
# BASE FLAGS FOR ALL PHASES
# ============================================================================

BASE_FLAGS = [
    # Core
    '--batch-size', '384',
    '--model-size', 'small',
    '--parallel-envs', '50',
    '--jax-env',
    '--bf16',
    '--synthetic-steps', '10000',
    '--steps', '400',

    # PopArt + GAE
    '--popart-beta', '0.0001',
    '--use-gae', '--gae-lambda', '0.95',

    # MCTS (extended sims for better value estimates)
    '--sims', '32',
    '--sims-warmup', '0',
    '--sims-min', '12',

    # Prediction (balanced with physics)
    '--use-prediction-loss',
    '--use-student-t',  # TASK_116: Fat-tailed output distribution
    '--prediction-loss-weight', '0.1',  # Lower to prevent domination
    '--physics-warmup-steps', '5000',

    # TASK_119: Optimal hindsight comparison (DS24 Market Baseline chart)
    '--amax', 'greedy',  # Track optimal alpha for alpha_capture metric
    '--amax-regret-weight', '0.05',  # Safe hindsight feedback
    '--amax-imprint-weight', '0.01',  # Soft policy guidance toward optimal patterns

    # TASK_125: Causal Expert Imprinting (oracle demos from successful episodes)
    '--imprint',

    # Entropy control
    '--adaptive-entropy',
    '--entropy-coef', '0.2',
    '--entropy-hold-threshold', '0.55',
    '--entropy-boost-low', '0.4',
    '--entropy-boost-high', '0.6',

    # LR schedule (higher initial LR for fresh training without data leaks)
    '--lr-schedule', 'cosine',
    '--lr-warmup-epochs', '20',
    '--lr-min', '1e-6',
    '--lr', '3e-4',

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

    # TASK_110: Cycle Balance (prevent H0 dominance)
    '--cycle-balance',
    '--cycle-balance-weight', '0.1',
    '--cycle-atc',
    '--cycle-atc-weight', '0.05',

    # DS24 monitoring (level 3 for prediction head + TRM heatmap)
    '--ds24-level', '3',
    '--ds24-intermediates-every', '10',

    # Physics targets for crypto (TASK_066)
    '--sigma-target', '0.03',
    '--alpha-target', '0.03',
    '--physics-reg-weight', '0.5',  # Was 0.01 (negligible) - boost to make physics meaningful
]

# ============================================================================
# DOJO V4 ORCHESTRATOR
# ============================================================================

class DojoV4:
    """
    Dojo v4: Full curriculum training (synthetic + historical).

    Trains from scratch through:
    1. Per-level warmup (L0-L14, 300 epochs)
    2. Speed training (140 epochs, cycling L1-L14)
    3. Historical data (96 years)
    """

    def __init__(
        self,
        checkpoint_dir: str = 'jax_ckpt_fast',
        model_size: str = 'small',
        db_path: str = 'trading_data.db',
        model_name: str = MODEL_NAME,
        model_version: str = MODEL_VERSION,
        historical_schedule: str = 'training_schedule.json',
    ):
        self.checkpoint_dir = checkpoint_dir
        self.model_size = model_size
        self.db_path = db_path
        self.model_name = model_name
        self.model_version = model_version
        self.historical_schedule = historical_schedule

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs('logs/benchmarks', exist_ok=True)

        self.total_epochs_trained = 0

    def run_full_curriculum(
        self,
        skip_phase1: bool = False,
        skip_phase2: bool = False,
        skip_phase3: bool = False,
        start_checkpoint: Optional[str] = None,
        start_level: int = 0,
    ) -> bool:
        """
        Run the complete curriculum.

        Args:
            skip_phase1: Skip per-level synthetic training (L0-L14)
            skip_phase2: Skip speed training (300 epochs)
            skip_phase3: Skip historical training
            start_checkpoint: Resume from checkpoint
            start_level: Start from this synthetic level (0-14)
        """
        epochs_per_level = SYNTHETIC_SCHEDULE['epochs_basic'] + SYNTHETIC_SCHEDULE['epochs_noise']
        total_synthetic = len(CURRICULUM_LEVELS) * epochs_per_level

        logging.info("=" * 70)
        logging.info("DOJO V4 - FULL CURRICULUM TRAINING")
        speed_epochs = SPEED_TRAINING['total_epochs']
        basic_epochs = SYNTHETIC_SCHEDULE['epochs_basic']
        noise_epochs = SYNTHETIC_SCHEDULE['epochs_noise']

        logging.info("=" * 70)
        logging.info(f"Model: {self.model_name} v{self.model_version}")
        logging.info(f"Checkpoint dir: {self.checkpoint_dir}")
        phase1_str = 'SKIP' if skip_phase1 else f'Per-level L0-L14 ({total_synthetic} epochs)'
        phase2_str = 'SKIP' if skip_phase2 else f'Speed training ({speed_epochs} epochs)'
        phase3_str = 'SKIP' if skip_phase3 else 'Historical (96 years)'
        logging.info(f"Phase 1: {phase1_str}")
        logging.info(f"Phase 2: {phase2_str}")
        logging.info(f"Phase 3: {phase3_str}")
        logging.info(f"Schedule: {basic_epochs} basic + {noise_epochs} noise per level")
        logging.info(f"Burning money: Auto-enabled if hold rate > {HOLD_RATE_THRESHOLD:.0%}")
        logging.info("=" * 70)

        checkpoint = start_checkpoint
        self.burning_money_enabled = False  # Track burning money state

        # Phase 1: Per-Level Synthetic Training (L0-L14)
        if not skip_phase1:
            logging.info("")
            logging.info("=" * 70)
            logging.info("PHASE 1: PER-LEVEL TRAINING (L0-L14)")
            logging.info("=" * 70)

            checkpoint = self._run_synthetic_training(checkpoint, start_level)
            if checkpoint is None:
                logging.error("Per-level training failed!")
                return False

            # Save phase 1 complete checkpoint
            phase1_ckpt = os.path.join(self.checkpoint_dir, f'{self.model_name}_v{self.model_version}_phase1_complete')
            self._copy_checkpoint(checkpoint, phase1_ckpt)

            # Run Phase 1 benchmark
            logging.info("")
            logging.info(">>> PHASE 1 BENCHMARK")
            benchmark_results = self._run_benchmark(checkpoint, "phase1")
            avg_reward = self._calculate_avg_benchmark_reward(benchmark_results)
            self._save_benchmark_to_csv("phase1", benchmark_results, avg_reward)
            logging.info(f"  Phase 1 avg reward: {avg_reward:.2f}")

        # Phase 2: Speed Training
        if not skip_phase2:
            logging.info("")
            logging.info("=" * 70)
            logging.info("PHASE 2: SPEED TRAINING (Weighted Synthetic + Adaptive Noise/Burn)")
            logging.info("=" * 70)

            # Use phase 1 checkpoint if no current checkpoint
            if checkpoint is None:
                checkpoint = os.path.join(
                    self.checkpoint_dir,
                    f'{self.model_name}_v{self.model_version}_phase1_complete'
                )
                if not os.path.exists(checkpoint):
                    logging.error(f"Phase 1 checkpoint not found: {checkpoint}")
                    return False

            checkpoint = self._run_speed_training(checkpoint)
            if checkpoint is None:
                logging.error("Speed training failed!")
                return False

            # Save synthetic complete checkpoint
            self._copy_checkpoint(
                checkpoint,
                os.path.join(self.checkpoint_dir, f'{self.model_name}_v{self.model_version}_synthetic_complete')
            )

        # Phase 3: Historical
        if not skip_phase3:
            logging.info("")
            logging.info("=" * 70)
            logging.info("PHASE 3: HISTORICAL TRAINING")
            logging.info("=" * 70)

            # Use synthetic complete checkpoint
            if checkpoint is None:
                checkpoint = os.path.join(
                    self.checkpoint_dir,
                    f'{self.model_name}_v{self.model_version}_synthetic_complete'
                )
                if not os.path.exists(checkpoint):
                    logging.error(f"Synthetic checkpoint not found: {checkpoint}")
                    return False

            success = self._run_historical_training(checkpoint)
            if not success:
                logging.error("Historical training failed!")
                return False

        logging.info("")
        logging.info("=" * 70)
        logging.info("FULL CURRICULUM COMPLETE!")
        logging.info(f"Total epochs trained: {self.total_epochs_trained}")
        logging.info("=" * 70)

        return True

    def _run_synthetic_training(self, start_checkpoint: Optional[str] = None, start_level: int = 0) -> Optional[str]:
        """
        Synthetic training: 100 epochs basic + 150 epochs noise = 250 epochs per level.

        For each level (L0-L14):
        - 100 epochs: Basic training (0% noise)
        - 150 epochs: Progressive noise (0%->100%)
        - Must achieve positive MA50 reward to advance to next level
        - Up to 100 extra epochs if MA50 condition not met

        Auto-enables burning_money if hold rate > 60%
        """
        checkpoint = start_checkpoint
        epochs_basic = SYNTHETIC_SCHEDULE['epochs_basic']
        epochs_noise = SYNTHETIC_SCHEDULE['epochs_noise']
        min_ma50 = SYNTHETIC_SCHEDULE.get('min_ma50_to_advance', 0.0)
        max_extra = SYNTHETIC_SCHEDULE.get('max_extra_epochs', 100)

        for level_info in CURRICULUM_LEVELS:
            level = level_info['level']
            description = level_info['description']

            # Skip levels before start_level
            if level < start_level:
                logging.info(f">>> SKIPPING LEVEL {level} (starting from L{start_level})")
                continue

            logging.info("")
            logging.info(f">>> LEVEL {level}: {description}")
            logging.info(f"    Schedule: {epochs_basic} basic + {epochs_noise} noise epochs")
            logging.info(f"    Condition: MA50 >= {min_ma50} to advance")
            if self.burning_money_enabled:
                logging.info(f"    🔥 Burning money: ENABLED")

            # Part A: Basic training (0% noise)
            logging.info(f"    Part A: {epochs_basic} epochs at 0% noise")
            checkpoint, epochs, hold_rate = self._train_synthetic(
                level=level,
                epochs=epochs_basic,
                noise=0.0,
                checkpoint=checkpoint,
                name_suffix=f'L{level:02d}_basic',
                reset_optimizer=(level > start_level and checkpoint is not None),
                burning_money=self.burning_money_enabled,
            )
            if checkpoint is None:
                return None
            self.total_epochs_trained += epochs

            # Check hold rate and enable burning money if needed
            self._check_hold_rate(hold_rate)

            # Part B: Progressive noise (0% -> 100%)
            logging.info(f"    Part B: {epochs_noise} epochs with progressive noise (0%->100%)")

            checkpoint, epochs, hold_rate = self._train_synthetic(
                level=level,
                epochs=epochs_noise,
                noise=1.0,  # Full noise (schedule handles ramping)
                checkpoint=checkpoint,
                name_suffix=f'L{level:02d}_noise',
                reset_optimizer=False,
                burning_money=self.burning_money_enabled,
                noise_schedule=True,  # Ramp from 0% to 100% over epochs
            )
            if checkpoint is None:
                return None
            self.total_epochs_trained += epochs
            self._check_hold_rate(hold_rate)

            # Check MA50 condition before advancing
            csv_path = f'logs/training_{self.model_name}_v{self.model_version}_L{level:02d}_noise.csv'
            ma50 = self._get_ma50_from_csv(csv_path)
            logging.info(f"    MA50 reward: {ma50:.2f} (threshold: {min_ma50})")

            # If MA50 not met, train extra epochs
            extra_trained = 0
            extra_round = 0
            while ma50 < min_ma50 and extra_trained < max_extra:
                extra_epochs = min(50, max_extra - extra_trained)
                extra_round += 1
                logging.warning(f"    ⚠️ MA50 {ma50:.2f} < {min_ma50} - training {extra_epochs} extra epochs (round {extra_round})")

                checkpoint, epochs, hold_rate = self._train_synthetic(
                    level=level,
                    epochs=extra_epochs,
                    noise=1.0,
                    checkpoint=checkpoint,
                    name_suffix=f'L{level:02d}_extra{extra_round}',
                    reset_optimizer=False,
                    burning_money=self.burning_money_enabled,
                    noise_schedule=False,  # Full noise for extra epochs
                )
                if checkpoint is None:
                    return None
                self.total_epochs_trained += epochs
                extra_trained += extra_epochs

                # Re-check MA50
                csv_path = f'logs/training_{self.model_name}_v{self.model_version}_L{level:02d}_extra{extra_round}.csv'
                ma50 = self._get_ma50_from_csv(csv_path)
                logging.info(f"    MA50 after extra: {ma50:.2f}")

            if ma50 >= min_ma50:
                logging.info(f"    ✓ Level {level} PASSED (MA50: {ma50:.2f})")
            else:
                logging.warning(f"    ⚠️ Level {level} did not reach MA50 threshold after {extra_trained} extra epochs")

            # Save level checkpoint
            level_checkpoint = os.path.join(
                self.checkpoint_dir,
                f'{self.model_name}_v{self.model_version}_L{level:02d}_complete'
            )
            self._copy_checkpoint(checkpoint, level_checkpoint)
            logging.info(f"    Level {level} complete! Checkpoint: {level_checkpoint}")

        return checkpoint

    def _get_ma50_from_csv(self, csv_path: str) -> float:
        """Get MA50 (moving average of last 50 rewards) from training CSV."""
        import csv
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if not rows:
                    return -999.0
                last_rows = rows[-50:] if len(rows) >= 50 else rows
                rewards = [float(r.get('reward', 0) or 0) for r in last_rows]
                return sum(rewards) / len(rewards) if rewards else -999.0
        except Exception as e:
            logging.warning(f"Could not read MA50 from {csv_path}: {e}")
            return -999.0

    def _check_hold_rate(self, hold_rate: Optional[float]):
        """Check hold rate and enable burning money if too high."""
        if hold_rate is None:
            return

        if not self.burning_money_enabled and hold_rate > HOLD_RATE_THRESHOLD:
            logging.warning(f"🔥 Hold rate {hold_rate:.1%} > {HOLD_RATE_THRESHOLD:.0%} - ENABLING BURNING MONEY")
            self.burning_money_enabled = True

    def _run_speed_training(self, checkpoint: str) -> Optional[str]:
        """
        Phase 2: Speed training - single 300-epoch run with weighted level cycling.

        Uses --weighted-synthetic flag for random level selection per epoch.
        Features:
        - Weighted level distribution (more hard, fewer easy)
        - Adaptive noise always on
        - Adaptive burn controller
        - Single CSV, progress bar
        """
        total_epochs = SPEED_TRAINING_CONFIG['total_epochs']
        min_reward = SPEED_TRAINING_CONFIG['min_reward_threshold']
        max_retries = SPEED_TRAINING_CONFIG['max_retries']

        # Build weighted level selection info for logging
        levels = list(LEVEL_WEIGHTS.keys())
        weights = list(LEVEL_WEIGHTS.values())
        total_weight = sum(weights)

        logging.info(f"Speed training: {total_epochs} epochs with weighted synthetic levels")
        logging.info(f"  Level weights: L0-L4 (easy): ~{sum(weights[:5])/total_weight:.0%}, "
                     f"L5-L9 (medium): ~{sum(weights[5:10])/total_weight:.0%}, "
                     f"L10-L14 (hard): ~{sum(weights[10:])/total_weight:.0%}")
        logging.info(f"  Adaptive noise: ON (multi-modal)")
        logging.info(f"  Adaptive burn: ON (auto-escalates when hold > 50%)")

        retry_count = 0
        best_checkpoint = checkpoint

        while retry_count <= max_retries:
            logging.info("")
            logging.info(f">>> PHASE 2 RUN {retry_count + 1}/{max_retries + 1}")

            name_suffix = f'{self.model_name}_v{self.model_version}_speed_r{retry_count}'

            # Build command - single 300-epoch run
            cmd = [
                'python', 'train_jax_fast.py',
                '--weighted-synthetic',           # Pre-generates all L0-L14, instant swap
                '--regenerate-data-every', '1',   # Swap level each epoch (instant from cache)
                '--epochs', str(total_epochs),
                '--parallel-envs', '50',
                '--steps', '400',
                '--synthetic-steps', '20000',     # 50 envs * 400 steps
                '--batch-size', '256',
                '--name', name_suffix,
                '--load-checkpoint', checkpoint,
                '--clear-buffer',

                # Kitchen sink v2
                '--jax-env',
                '--unique-data-per-env',            # Each env gets independent price data
                '--data-gen-workers', '8',          # Parallel generation
                '--popart-beta', '0.0001',
                '--use-gae', '--gae-lambda', '0.95',
                '--adaptive-entropy', '--entropy-coef', '0.2',
                '--epsilon-greedy', '0.1',
                '--sims', '32', '--sims-warmup', '300', '--sims-min', '12',
                '--use-prediction-loss', '--use-student-t', '--prediction-loss-weight', '0.1',
                '--pred-loss-type', 'huber',
                '--physics-reg-weight', '0.5',       # Boost physics regularization
                '--redo', '--redo-interval', '50',
                '--atc-loss', '--atc-weight', '0.1',
                # TASK_110: Cycle Balance
                '--cycle-balance', '--cycle-balance-weight', '0.1',
                '--cycle-atc', '--cycle-atc-weight', '0.05',

                # TASK_125: Causal Expert Imprinting
                '--imprint',

                # Adaptive noise always on (escalate aggressively)
                '--adaptive-noise',
                '--target-reward-low', '5',     # Below this: decrease difficulty
                '--target-reward-high', '12',   # Above this: increase difficulty (was 20, too lenient)
                '--noise-adjustment-rate', '0.1',
                '--max-temporal-jitter', '10',
                '--max-time-dilation', '0.5',
                '--max-phase-shift', '0.15',

                # Adaptive burn controller (needs --burning-money to activate)
                '--burning-money',
                '--adaptive-burn',
                '--burn-hold-threshold', '0.50',
                '--burn-patience', '5',
                # Random burns (wife with debit card mode)
                '--burn-random',
                '--burn-random-prob', '0.02',

                # Additional stress mechanisms
                '--probabilistic-fills',        # Limit orders may not fill
                '--borrow-rate', '0.05',        # 5% annual cost for shorts
                '--episode-variance', '50',     # ±50 steps random episode length

                # DS24 logging
                '--ds24-level', '3',
            ]

            logging.info(f"  Running: {' '.join(cmd[:10])}...")

            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=False,
            )

            if result.returncode != 0:
                logging.error(f"Training failed with return code {result.returncode}")
                return None

            # Update checkpoint path
            new_checkpoint = os.path.join(self.checkpoint_dir, name_suffix)
            if os.path.exists(new_checkpoint):
                checkpoint = new_checkpoint

            self.total_epochs_trained += total_epochs

            # Parse final metrics from training CSV
            csv_path = f'logs/training_{name_suffix}.csv'
            avg_reward = self._get_avg_reward_from_csv(csv_path, last_n=50)
            logging.info(f"  Phase 2 avg reward (last 50): {avg_reward:.2f}")

            # MANDATORY: Run benchmark after Phase 2 completion
            logging.info("")
            logging.info(">>> PHASE 2 BENCHMARK (MANDATORY)")
            benchmark_results = self._run_benchmark(checkpoint, f"phase2_r{retry_count}")
            benchmark_avg = self._calculate_avg_benchmark_reward(benchmark_results)
            self._save_benchmark_to_csv(f"phase2_r{retry_count}", benchmark_results, benchmark_avg)

            # Check if rewards meet threshold
            if avg_reward >= min_reward:
                logging.info(f"  Phase 2 PASSED! Avg reward: {avg_reward:.2f} >= {min_reward}")
                best_checkpoint = checkpoint
                break
            else:
                logging.warning(f"  Phase 2 RETRY NEEDED! Avg reward: {avg_reward:.2f} < {min_reward}")
                retry_count += 1
                if retry_count <= max_retries:
                    logging.info(f"  Retrying Phase 2 (attempt {retry_count + 1}/{max_retries + 1})...")
                else:
                    logging.warning(f"  Max retries reached. Proceeding with best checkpoint.")
                    best_checkpoint = checkpoint

        return best_checkpoint

    def _get_avg_reward_from_csv(self, csv_path: str, last_n: int = 50) -> float:
        """Get average reward from last N rows of training CSV."""
        import csv
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if not rows:
                    return 0.0
                last_rows = rows[-last_n:] if len(rows) >= last_n else rows
                rewards = [float(r.get('reward', 0) or 0) for r in last_rows]
                return sum(rewards) / len(rewards) if rewards else 0.0
        except Exception as e:
            logging.warning(f"Could not read {csv_path}: {e}")
            return 0.0

    def _train_synthetic_advanced(
        self,
        level: int,
        epochs: int,
        noise: float,
        noise_type: str,
        checkpoint: Optional[str],
        name_suffix: str,
        reset_optimizer: bool = False,
        use_fees: bool = False,
        burning_money: bool = False,
    ) -> Tuple[Optional[str], int, Optional[float]]:
        """
        Advanced synthetic training with progressive difficulty.

        noise_type: Controls difficulty stage (uses --synthetic-noise for all types)
        """
        experiment_name = f'{self.model_name}_v{self.model_version}_{name_suffix}'

        cmd = [
            'python', 'train_jax_fast.py',
            '--synthetic-level', str(level),
            '--epochs', str(epochs),
            '--name', experiment_name,
            '--model-size', self.model_size,
        ]

        # Add base flags
        cmd.extend(BASE_FLAGS)

        # Set noise level (synthetic_market.py handles all noise types internally)
        cmd.extend(['--synthetic-noise', str(noise)])

        # Fees configuration based on stage
        if use_fees:
            cmd.extend(['--fee', '5'])      # 5 bps maker fee
            cmd.extend(['--slippage', '2'])  # 2 bps slippage
        else:
            cmd.extend(['--fee', '0', '--slippage', '0'])

        # Burning money (cash opportunity cost)
        if burning_money:
            cmd.append('--burning-money')

        # Handle checkpoint
        if checkpoint:
            cmd.extend(['--load-checkpoint', checkpoint])
            if reset_optimizer:
                cmd.append('--reset-optimizer')
                cmd.append('--clear-buffer')

        # Run training with live output streaming
        start_time = time.time()
        hold_rate = None
        reward = None
        import re

        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            output_lines = []
            for line in process.stdout:
                print(line, end='', flush=True)
                output_lines.append(line)

                # Parse hold rate
                match = re.search(r'S/H/B/LS/LB:\s*(\d+)/(\d+)/(\d+)/(\d+)/(\d+)', line)
                if match:
                    sell, hold, buy, ls, lb = map(int, match.groups())
                    total = sell + hold + buy + ls + lb
                    if total > 0:
                        hold_rate = hold / total

                # Parse reward
                reward_match = re.search(r'Reward:\s*([\+\-]?\d+\.?\d*)', line)
                if reward_match:
                    reward = float(reward_match.group(1))

            process.wait(timeout=3600 * 2)
            elapsed = time.time() - start_time

            hold_str = f"{hold_rate:.1%}" if hold_rate else "N/A"
            reward_str = f"{reward:.2f}" if reward else "N/A"
            logging.info(f"      Completed in {elapsed/60:.1f} min (hold: {hold_str}, reward: {reward_str})")

            if process.returncode != 0:
                logging.error(f"Training failed with return code {process.returncode}")
                return None, 0, hold_rate

        except subprocess.TimeoutExpired:
            process.kill()
            logging.error("Training timed out!")
            return None, 0, None
        except Exception as e:
            logging.error(f"Training error: {e}")
            return None, 0, None

        # Return checkpoint path
        new_checkpoint = os.path.join(self.checkpoint_dir, experiment_name)
        if os.path.exists(new_checkpoint):
            return new_checkpoint, epochs, reward if reward else hold_rate
        else:
            logging.warning(f"Checkpoint not found at {new_checkpoint}")
            return checkpoint, epochs, reward if reward else hold_rate

    def _train_synthetic_adaptive(
        self,
        level: int,
        epochs: int,
        use_adaptive_noise: bool,
        fee_bps: int,
        slippage_bps: int,
        checkpoint: Optional[str],
        name_suffix: str,
        burning_money: bool = False,
    ) -> Tuple[Optional[str], int, Optional[float], Optional[float]]:
        """
        Adaptive synthetic training with fine-grained fee/slippage control.

        Uses --adaptive-noise for multi-modal noise progression:
        1. Price noise (0%→100%)
        2. Temporal jitter (0→5 steps)
        3. Time dilation (0%→30%)
        4. Phase shift (0%→50%)

        Returns:
            (checkpoint_path, epochs_trained, reward, hold_rate)
        """
        experiment_name = f'{self.model_name}_v{self.model_version}_{name_suffix}'

        cmd = [
            'python', 'train_jax_fast.py',
            '--synthetic-level', str(level),
            '--epochs', str(epochs),
            '--name', experiment_name,
            '--model-size', self.model_size,
        ]

        # Add base flags
        cmd.extend(BASE_FLAGS)

        # Adaptive noise mode (handles price → temporal → time dilation → phase shift)
        if use_adaptive_noise:
            cmd.extend([
                '--adaptive-noise',
                '--target-reward-low', '5',    # Below this: decrease difficulty
                '--target-reward-high', '15',  # Above this: increase difficulty
            ])
        else:
            cmd.extend(['--synthetic-noise', '0'])

        # Fees configuration (fine-grained control)
        cmd.extend(['--fee', str(fee_bps)])
        cmd.extend(['--slippage', str(slippage_bps)])

        # Burning money with ADAPTIVE ESCALATION (Phase 2 hardcore mode)
        if burning_money:
            cmd.extend([
                '--burning-money',
                '--adaptive-burn',
                '--burn-hold-threshold', '0.60',
                '--burn-patience', '5',
                # Random burns (wife with debit card mode)
                '--burn-random',
                '--burn-random-prob', '0.02',  # 2% chance per step
            ])

        # Additional stress mechanisms (Phase 2 hardcore)
        cmd.extend([
            '--probabilistic-fills',        # Limit orders may not fill
            '--borrow-rate', '0.05',        # 5% annual cost for shorts
            '--episode-variance', '50',     # ±50 steps random episode length
        ])

        # Handle checkpoint
        if checkpoint:
            cmd.extend(['--load-checkpoint', checkpoint])

        # Run training with live output streaming
        start_time = time.time()
        hold_rate = None
        reward = None
        import re

        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            output_lines = []
            for line in process.stdout:
                # Don't print every line for single-epoch training (too verbose)
                output_lines.append(line)

                # Parse hold rate
                match = re.search(r'S/H/B/LS/LB:\s*(\d+)/(\d+)/(\d+)/(\d+)/(\d+)', line)
                if match:
                    sell, hold, buy, ls, lb = map(int, match.groups())
                    total = sell + hold + buy + ls + lb
                    if total > 0:
                        hold_rate = hold / total

                # Parse reward
                reward_match = re.search(r'Reward:\s*([\+\-]?\d+\.?\d*)', line)
                if reward_match:
                    reward = float(reward_match.group(1))

            process.wait(timeout=3600)  # 1 hour timeout for single epoch
            elapsed = time.time() - start_time

            if process.returncode != 0:
                logging.error(f"Training failed with return code {process.returncode}")
                # Print last few lines for debugging
                for line in output_lines[-20:]:
                    print(line, end='')
                return None, 0, None, None

        except subprocess.TimeoutExpired:
            process.kill()
            logging.error("Training timed out!")
            return None, 0, None, None
        except Exception as e:
            logging.error(f"Training error: {e}")
            return None, 0, None, None

        # Return checkpoint path
        new_checkpoint = os.path.join(self.checkpoint_dir, experiment_name)
        if os.path.exists(new_checkpoint):
            return new_checkpoint, epochs, reward, hold_rate
        else:
            logging.warning(f"Checkpoint not found at {new_checkpoint}")
            return checkpoint, epochs, reward, hold_rate

    def _run_historical_training(self, checkpoint: str) -> bool:
        """
        Phase 3: Historical training using dojo_v3_historical.

        Runs stratified mode (cycles through periods).
        """
        if not os.path.exists(self.historical_schedule):
            logging.warning(f"Historical schedule not found: {self.historical_schedule}")
            logging.warning("Skipping Phase 3")
            return True

        logging.info(f"Starting historical training from: {checkpoint}")
        logging.info(f"Schedule: {self.historical_schedule}")

        cmd = [
            'python', 'tools/orchestration/dojo_v3_historical.py',
            '--schedule', self.historical_schedule,
            '--checkpoint', checkpoint,
            '--model-name', self.model_name,
            '--model-version', self.model_version,
            '--stratified',
            '--periods-per-batch', '4',
            '--epochs-per-period', '15',
            # One-shot mode (no --loop flag)
        ]

        try:
            # No timeout - runs until data exhausted, model collapse, or hardware failure
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=False,
            )

            if result.returncode != 0:
                logging.error(f"Historical training failed with return code {result.returncode}")
                return False

            return True

        except Exception as e:
            logging.error(f"Historical training error: {e}")
            return False

    def _train_synthetic(
        self,
        level: int,
        epochs: int,
        noise: float,
        checkpoint: Optional[str],
        name_suffix: str,
        reset_optimizer: bool = False,
        burning_money: bool = False,
        noise_schedule: bool = False,
    ) -> Tuple[Optional[str], int, Optional[float]]:
        """
        Train on synthetic data for given parameters.

        Returns:
            (checkpoint_path, epochs_trained, hold_rate)
        """
        experiment_name = f'{self.model_name}_v{self.model_version}_{name_suffix}'

        cmd = [
            'python', 'train_jax_fast.py',
            '--synthetic-level', str(level),
            '--epochs', str(epochs),
            '--synthetic-noise', str(noise),
            '--name', experiment_name,
            '--model-size', self.model_size,
            '--fee', '0',
            '--slippage', '0',
        ]

        # Add base flags
        cmd.extend(BASE_FLAGS)

        # Add noise schedule if enabled (ramps from 0% to noise over epochs)
        if noise_schedule:
            cmd.append('--noise-schedule')

        # Add burning money if enabled
        if burning_money:
            cmd.append('--burning-money')

        # Handle checkpoint
        if checkpoint:
            cmd.extend(['--load-checkpoint', checkpoint])
            if reset_optimizer:
                cmd.append('--reset-optimizer')
                cmd.append('--clear-buffer')

        # Run training with live output streaming
        start_time = time.time()
        hold_rate = None
        import re

        try:
            # Use Popen for live output streaming
            process = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
            )

            # Stream output and parse hold rate
            output_lines = []
            for line in process.stdout:
                print(line, end='', flush=True)  # Live output
                output_lines.append(line)
                # Parse hold rate from output (format: "S/H/B/LS/LB: 100/500/200/50/50")
                match = re.search(r'S/H/B/LS/LB:\s*(\d+)/(\d+)/(\d+)/(\d+)/(\d+)', line)
                if match:
                    sell, hold, buy, ls, lb = map(int, match.groups())
                    total = sell + hold + buy + ls + lb
                    if total > 0:
                        hold_rate = hold / total

            process.wait(timeout=3600 * 2)  # 2 hour timeout
            elapsed = time.time() - start_time

            hold_str = f"{hold_rate:.1%}" if hold_rate else "N/A"
            logging.info(f"      Completed in {elapsed/60:.1f} minutes (hold rate: {hold_str})")

            if process.returncode != 0:
                logging.error(f"Training failed with return code {process.returncode}")
                return None, 0, hold_rate

        except subprocess.TimeoutExpired:
            process.kill()
            logging.error("Training timed out!")
            return None, 0, None
        except Exception as e:
            logging.error(f"Training error: {e}")
            return None, 0, None

        # Return checkpoint path
        new_checkpoint = os.path.join(self.checkpoint_dir, experiment_name)
        if os.path.exists(new_checkpoint):
            return new_checkpoint, epochs, hold_rate
        else:
            logging.warning(f"Checkpoint not found at {new_checkpoint}")
            return checkpoint, epochs, hold_rate

    def _copy_checkpoint(self, src: str, dst: str):
        """Copy checkpoint directory."""
        import shutil
        if os.path.exists(dst):
            shutil.rmtree(dst)
        if src and os.path.exists(src):
            shutil.copytree(src, dst)
            logging.info(f"      Saved checkpoint: {dst}")

    def _run_benchmark(self, checkpoint: str, phase_name: str) -> Dict[str, Dict]:
        """
        Run benchmark on all benchmark periods.

        Returns dict of {period_name: {reward, alpha, market_return, ...}}
        """
        import re

        results = {}

        # Load benchmark periods
        benchmark_config_path = PROJECT_ROOT / 'config' / 'benchmark_periods.json'
        if not benchmark_config_path.exists():
            logging.warning(f"Benchmark config not found: {benchmark_config_path}")
            return results

        with open(benchmark_config_path) as f:
            benchmark_config = json.load(f)

        periods = benchmark_config.get('periods', [])
        bench_cfg = benchmark_config.get('benchmark_config', {})

        epochs_per_period = bench_cfg.get('epochs_per_period', 7)
        steps = bench_cfg.get('steps_per_epoch', 1440)
        parallel_envs = bench_cfg.get('parallel_envs', 4)
        warmup_epochs = bench_cfg.get('warmup_epochs', 7)

        logging.info(f"Running benchmark on {len(periods)} periods...")

        for period in periods:
            period_name = period['name']
            start_date = period['start_date']
            end_date = period['end_date']
            description = period.get('description', '')

            logging.info(f"  {period_name}: {start_date} → {end_date} ({description})")

            cmd = [
                'python', 'train_jax_fast.py',
                '--db-path', self.db_path,
                '--start-date', start_date,
                '--end-date', end_date,
                '--epochs', str(warmup_epochs + epochs_per_period),
                '--steps', str(steps),
                '--parallel-envs', str(parallel_envs),
                '--load-checkpoint', checkpoint,
                '--no-checkpoint',
                '--jax-env',
                '--bf16',
                '--model-size', self.model_size,
                '--name', f'{self.model_name}_benchmark_{phase_name}_{period_name}',
                # Inference flags
                '--popart-beta', '0.0001',
                '--use-gae', '--gae-lambda', '0.95',
                '--sims', '24',
                '--use-prediction-loss', '--use-student-t',
                '--prediction-loss-weight', '0.1',
                '--physics-reg-weight', '0.5',
                '--adaptive-entropy',
                '--entropy-coef', '0.2',
                '--entropy-hold-threshold', '0.55',
                '--ds24-level', '3',
            ]

            try:
                result = subprocess.run(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    timeout=1800,  # 30 min timeout per benchmark
                )

                output = result.stdout + result.stderr
                metrics = self._parse_benchmark_output(output)
                results[period_name] = metrics

                logging.info(f"    → reward={metrics.get('reward', 0):.2f}, "
                             f"α={metrics.get('alpha', 0):.1f}%, "
                             f"hold={metrics.get('hold_pct', 0):.1f}%")

            except subprocess.TimeoutExpired:
                logging.error(f"    Benchmark {period_name} timed out!")
                results[period_name] = {'reward': 0, 'alpha': 0, 'error': 'timeout'}
            except Exception as e:
                logging.error(f"    Benchmark {period_name} failed: {e}")
                results[period_name] = {'reward': 0, 'alpha': 0, 'error': str(e)}

        return results

    def _parse_benchmark_output(self, output: str) -> Dict:
        """Parse training output to extract benchmark metrics."""
        import re

        metrics = {
            'reward': 0.0,
            'alpha': 0.0,
            'market_return': 0.0,
            'hold_pct': 0.0,
            'entropy': 0.0,
        }

        rewards = []
        alphas = []

        # Pattern: "Ep   50 | Reward:  +33.61 | α: +241.02% (Mkt: +9.7%)"
        # Note: Output uses "Ep" not "Epoch"
        epoch_pattern = r'Ep\s+(\d+)\s+\|\s+Reward:\s*([\+\-]?\d+\.?\d*)\s+\|\s+α:\s*([\+\-]?\d+\.?\d*)%'
        for match in re.findall(epoch_pattern, output):
            rewards.append(float(match[1]))
            alphas.append(float(match[2]))

        if rewards:
            # Use last half of rewards (skip warmup)
            half_idx = len(rewards) // 2
            metrics['reward'] = sum(rewards[half_idx:]) / max(1, len(rewards) - half_idx)
            metrics['alpha'] = sum(alphas[half_idx:]) / max(1, len(alphas) - half_idx)

        # HOLD percentage - pattern: "HOLD: 45.2%"
        hold_pattern = r'HOLD:\s*(\d+\.?\d*)%'
        hold_matches = re.findall(hold_pattern, output)
        if hold_matches:
            metrics['hold_pct'] = float(hold_matches[-1])

        # Entropy - pattern: "Ent: 1.234"
        entropy_pattern = r'Ent:\s*(\d+\.?\d*)'
        entropy_matches = re.findall(entropy_pattern, output)
        if entropy_matches:
            metrics['entropy'] = float(entropy_matches[-1])

        return metrics

    def _calculate_avg_benchmark_reward(self, results: Dict[str, Dict]) -> float:
        """Calculate average reward across all benchmark periods."""
        if not results:
            return 0.0

        rewards = [r.get('reward', 0) for r in results.values() if 'error' not in r]
        if not rewards:
            return 0.0

        return sum(rewards) / len(rewards)

    def _save_benchmark_to_csv(self, phase_name: str, results: Dict[str, Dict], avg_reward: float):
        """
        Save benchmark results to CSV for DS24 dashboard.

        CSV format: timestamp, model, phase, period, reward, alpha, hold_pct, avg_reward
        """
        csv_path = os.path.join('logs', 'benchmarks', 'benchmark_results.csv')
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        # Check if file exists to write header
        write_header = not os.path.exists(csv_path)

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with open(csv_path, 'a', newline='') as f:
            import csv
            writer = csv.writer(f)

            if write_header:
                writer.writerow([
                    'timestamp', 'model', 'version', 'phase', 'period',
                    'reward', 'alpha', 'hold_pct', 'entropy', 'avg_reward', 'passed'
                ])

            for period_name, metrics in results.items():
                passed = 'Y' if avg_reward >= BENCHMARK_MIN_REWARD else 'N'
                writer.writerow([
                    timestamp,
                    self.model_name,
                    self.model_version,
                    phase_name,
                    period_name,
                    f"{metrics.get('reward', 0):.2f}",
                    f"{metrics.get('alpha', 0):.1f}",
                    f"{metrics.get('hold_pct', 0):.1f}",
                    f"{metrics.get('entropy', 0):.3f}",
                    f"{avg_reward:.2f}",
                    passed,
                ])

        logging.info(f"  Benchmark results saved to {csv_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Dojo v4: Full Curriculum Training (Synthetic + Historical)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full curriculum from scratch
  python dojo_v4_full_curriculum.py

  # Skip phase 1 (resume from phase 1 complete)
  python dojo_v4_full_curriculum.py --skip-phase1

  # Only synthetic (phases 1+2)
  python dojo_v4_full_curriculum.py --skip-phase3

  # Resume from checkpoint
  python dojo_v4_full_curriculum.py --checkpoint jax_ckpt_fast/phoenix_v1_L05_complete

  # Custom model name
  python dojo_v4_full_curriculum.py --model-name lotus --model-version 1
"""
    )

    parser.add_argument('--model-name', default=MODEL_NAME,
                        help=f'Model name (default: {MODEL_NAME})')
    parser.add_argument('--model-version', default=MODEL_VERSION,
                        help=f'Model version (default: {MODEL_VERSION})')
    parser.add_argument('--model-size', default='small',
                        choices=['small', 'medium', 'large'],
                        help='Model size (default: small)')
    parser.add_argument('--checkpoint-dir', default='jax_ckpt_fast',
                        help='Checkpoint directory (default: jax_ckpt_fast)')
    parser.add_argument('--checkpoint', default=None,
                        help='Resume from specific checkpoint')
    parser.add_argument('--db-path', default='trading_data.db',
                        help='Database path for historical (default: trading_data.db)')
    parser.add_argument('--historical-schedule', default='training_schedule.json',
                        help='Historical training schedule (default: training_schedule.json)')

    # Phase skipping
    parser.add_argument('--skip-phase1', action='store_true',
                        help='Skip Phase 1 (per-level training)')
    parser.add_argument('--skip-phase2', action='store_true',
                        help='Skip Phase 2 (speed training)')
    parser.add_argument('--skip-phase3', action='store_true',
                        help='Skip Phase 3 (historical)')
    parser.add_argument('--start-level', type=int, default=0,
                        help='Start from this synthetic level (0-14, default: 0)')

    args = parser.parse_args()

    dojo = DojoV4(
        checkpoint_dir=args.checkpoint_dir,
        model_size=args.model_size,
        db_path=args.db_path,
        model_name=args.model_name,
        model_version=args.model_version,
        historical_schedule=args.historical_schedule,
    )

    success = dojo.run_full_curriculum(
        skip_phase1=args.skip_phase1,
        skip_phase2=args.skip_phase2,
        skip_phase3=args.skip_phase3,
        start_checkpoint=args.checkpoint,
        start_level=args.start_level,
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
