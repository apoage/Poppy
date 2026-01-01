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

# Training imports
import jax
import jax.numpy as jnp
import numpy as np
from train_jax_fast import run_training  # Reuse existing training loop

# Synthetic generators
from mcts_trader.data.synthetic_market import SyntheticMarket

# Tactics
from mcts_trader.tactics import BasicTrading, PortfolioMorphing

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

    def __init__(self, curriculum_config: List[Dict], checkpoint_dir: str = 'dojo_checkpoints'):
        """
        Args:
            curriculum_config: List of level configs
                [
                    {'level': 0, 'generator': 'sine', 'tactics': ['basic'], 'epochs': 500},
                    {'level': 1, 'generator': 'lagged', 'tactics': ['basic'], 'epochs': 1000},
                    ...
                ]
            checkpoint_dir: Directory for curriculum checkpoints
        """
        self.curriculum = curriculum_config
        self.checkpoint_dir = checkpoint_dir
        self.current_level = 0

        os.makedirs(checkpoint_dir, exist_ok=True)
        logging.info(f"Dojo initialized with {len(curriculum_config)} curriculum levels")

    def run(self):
        """Run full curriculum sequence"""
        logging.info(f"Starting curriculum training ({len(self.curriculum)} levels)")

        for i, level_config in enumerate(self.curriculum):
            logging.info(f"\n{'='*60}")
            logging.info(f"LEVEL {i}: {level_config['generator']}")
            logging.info(f"Tactics: {level_config['tactics']}")
            logging.info(f"Epochs: {level_config['epochs']}")
            logging.info(f"{'='*60}\n")

            self._train_level(level_config, level_idx=i)

            # Save curriculum progress
            self._save_progress(i)

        logging.info("🎓 Curriculum COMPLETE!")

    def _train_level(self, config: Dict, level_idx: int):
        """
        Train single curriculum level.

        Args:
            config: Level configuration
                {
                    'level': 0,
                    'generator': 'sine',
                    'tactics': ['basic'],
                    'epochs': 500,
                    'n_assets': 64,
                    'n_steps': 10000,
                }
            level_idx: Index in curriculum
        """
        start_time = time.time()

        # 1. Create synthetic generator
        generator_type = config['generator']
        level = config.get('level', level_idx)
        n_assets = config.get('n_assets', 64)
        n_steps = config.get('n_steps', 10000)
        seed = config.get('seed', 42 + level_idx)

        logging.info(f"Creating synthetic generator: {generator_type} (Level {level})")
        # For now, use existing SyntheticMarket class
        # TODO: Refactor into separate generator classes
        generator = SyntheticMarket(
            level=level,
            n_assets=n_assets,
            n_steps=n_steps,
            seed=seed
        )

        # 2. Determine tactics
        tactics = config['tactics']
        enable_swaps = 'morphing' in tactics

        logging.info(f"Tactics enabled: {tactics} (enable_swaps={enable_swaps})")

        # 3. Build training command
        # For now, we call train_jax_fast.py with appropriate flags
        # TODO: Refactor to use modular environment composition

        epochs = config.get('epochs', 500)
        batch_size = config.get('batch_size', 512)
        model_size = config.get('model_size', 'tiny')

        training_args = {
            'synthetic_level': level,
            'synthetic_steps': n_steps,
            'enable_swaps': enable_swaps,
            'epochs': epochs,
            'batch_size': batch_size,
            'model_size': model_size,
            'name': f'dojo_level_{level_idx}_{generator_type}',
        }

        logging.info(f"Training args: {training_args}")

        # 4. Run training
        # Option A: Call train_jax_fast as subprocess (current approach)
        # Option B: Import run_training and call directly (future refactor)

        import subprocess
        cmd = [
            'python', 'train_jax_fast.py',
            '--synthetic-level', str(level),
            '--synthetic-steps', str(n_steps),
            '--epochs', str(epochs),
            '--batch-size', str(batch_size),
            '--model-size', model_size,
            '--name', training_args['name'],
        ]

        if enable_swaps:
            cmd.append('--enable-swaps')

        logging.info(f"Running command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=False)

        if result.returncode != 0:
            logging.error(f"Level {level_idx} FAILED with code {result.returncode}")
            raise RuntimeError(f"Training failed at level {level_idx}")

        duration = time.time() - start_time
        logging.info(f"Level {level_idx} COMPLETED in {duration/60:.1f} minutes")

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
    Create default 4-level curriculum (Levels 0-3).

    Returns:
        List of level configs
    """
    return [
        {
            'level': 0,
            'generator': 'sine',
            'tactics': ['basic'],
            'epochs': 500,
            'n_assets': 64,
            'n_steps': 10000,
            'description': 'Pure sine waves - learn trend detection'
        },
        {
            'level': 1,
            'generator': 'lagged',
            'tactics': ['basic'],
            'epochs': 1000,
            'n_assets': 64,
            'n_steps': 10000,
            'description': 'Lagged pairs - learn correlation detection'
        },
        {
            'level': 2,
            'generator': 'dip',
            'tactics': ['basic'],
            'epochs': 1500,
            'n_assets': 64,
            'n_steps': 10000,
            'description': 'Dip crashes - learn patience and timing'
        },
        {
            'level': 3,
            'generator': 'noisy_trends',
            'tactics': ['basic', 'morphing'],
            'epochs': 2000,
            'n_assets': 64,
            'n_steps': 10000,
            'description': 'Noisy trends + swaps - learn fee optimization'
        },
    ]


def main():
    parser = argparse.ArgumentParser(description='Dojo Curriculum Training Manager')

    parser.add_argument('--curriculum', help='Curriculum config JSON file')
    parser.add_argument('--level', type=int, help='Train single level (0-3)')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    parser.add_argument('--checkpoint-dir', default='dojo_checkpoints', help='Checkpoint directory')

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
    dojo = DojoManager(curriculum, checkpoint_dir=args.checkpoint_dir)

    # Resume from checkpoint if requested
    if args.resume:
        start_level = dojo.load_progress()
        logging.info(f"Resuming from level {start_level}")
        curriculum = curriculum[start_level:]

    # Run training
    dojo.run()


if __name__ == '__main__':
    main()
