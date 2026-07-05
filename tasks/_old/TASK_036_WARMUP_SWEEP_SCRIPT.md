# TASK_036: Warmup Configuration Sweep Script

## Status: PLANNED

## Problem Statement

The training process exhibits "dual collapse" - coupled policy entropy collapse and prediction head collapse. Research indicates this is caused by:
1. MCTS sharpening amplifying small value biases toward HOLD
2. Physics warmup timing mismatch (policy collapses before warmup ends)
3. Neural corrector learning trivial "no change" solution

We need to find stable warmup configurations that allow all network components to learn before collapse occurs.

## Research Background

From `/home/lukas/Trader/docs/AI_bleed research/`:
- **Compass**: "Breaking the loop requires simultaneous intervention at multiple points"
- **Kimi**: "More simulations → sharper policies → faster collapse"
- **Claude CLI**: "MCTS is the collapse catalyst" - sims warmup delays but doesn't prevent
- **Gemini**: "Martingale Trap" - identity function is strong local optimum

## Objective

Create a script that systematically tests warmup configurations to find:
1. Configurations that maintain entropy > 3.5 for longest duration
2. Configurations where prediction loss stays > 0.1 (non-trivial)
3. Configurations achieving highest reward plateau

## Search Space

```python
SWEEP_CONFIG = {
    # MCTS simulation warmup
    'sims_warmup_epochs': [50, 100, 200, 500],
    'sims_min': [4, 8, 16],
    'sims_max': [16, 32, 64],

    # Physics/neural warmup
    'physics_warmup_steps': [0, 2500, 5000, 10000],  # steps, not epochs

    # Entropy control
    'entropy_coef': [0.01, 0.05, 0.1, 0.2],
    'entropy_hold_threshold': [0.60, 0.70, 0.80],  # when adaptive kicks in

    # Exploration mixing (NEW - key intervention)
    'epsilon_greedy': [0.0, 0.1, 0.2, 0.3],  # mix into MCTS policy

    # Prediction loss weight
    'prediction_loss_weight': [0.1, 0.2, 0.5, 1.0],
}
```

## Implementation Plan

### Phase 1: Sweep Script Structure

```python
# warmup_sweep.py

import itertools
import subprocess
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

class WarmupSweep:
    def __init__(self, output_dir: str = "sweep_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []

    def generate_configs(self) -> list:
        """Generate all config combinations to test."""
        # Start with focused subset (not full cartesian product)
        focused_configs = [
            # Baseline (current)
            {'sims_warmup': 100, 'sims_min': 8, 'sims_max': 32,
             'entropy_coef': 0.01, 'epsilon': 0.0},

            # High epsilon exploration
            {'sims_warmup': 100, 'sims_min': 8, 'sims_max': 32,
             'entropy_coef': 0.01, 'epsilon': 0.2},
            {'sims_warmup': 100, 'sims_min': 8, 'sims_max': 32,
             'entropy_coef': 0.01, 'epsilon': 0.3},

            # Extended low-sims phase
            {'sims_warmup': 200, 'sims_min': 4, 'sims_max': 32,
             'entropy_coef': 0.05, 'epsilon': 0.15},
            {'sims_warmup': 500, 'sims_min': 4, 'sims_max': 16,
             'entropy_coef': 0.05, 'epsilon': 0.2},

            # High entropy coefficient
            {'sims_warmup': 100, 'sims_min': 8, 'sims_max': 32,
             'entropy_coef': 0.1, 'epsilon': 0.1},
            {'sims_warmup': 100, 'sims_min': 8, 'sims_max': 32,
             'entropy_coef': 0.2, 'epsilon': 0.1},

            # Extended physics warmup
            {'sims_warmup': 100, 'sims_min': 8, 'sims_max': 32,
             'entropy_coef': 0.05, 'epsilon': 0.2,
             'physics_warmup': 10000},

            # Combined interventions
            {'sims_warmup': 300, 'sims_min': 4, 'sims_max': 24,
             'entropy_coef': 0.1, 'epsilon': 0.25,
             'physics_warmup': 5000},
        ]
        return focused_configs

    def run_single_config(self, config: dict, run_id: int, epochs: int = 50):
        """Run training with specific config, collect metrics."""
        name = f"sweep_{run_id:03d}"

        cmd = [
            "python", "train_jax_fast.py",
            "--synthetic-level", "0",
            "--epochs", str(epochs),
            "--steps", "400",
            "--parallel-envs", "20",
            "--batch-size", "384",
            "--model-size", "small",
            "--sims-warmup", str(config.get('sims_warmup', 100)),
            "--sims-min", str(config.get('sims_min', 8)),
            "--sims", str(config.get('sims_max', 32)),
            "--entropy-coef", str(config.get('entropy_coef', 0.01)),
            "--adaptive-entropy",
            "--use-prediction-loss",
            "--prediction-loss-weight", str(config.get('pred_weight', 0.2)),
            "--name", name,
        ]

        # Add epsilon if supported (TASK_037)
        if config.get('epsilon', 0) > 0:
            cmd.extend(["--epsilon-greedy", str(config['epsilon'])])

        # Add physics warmup if specified
        if config.get('physics_warmup'):
            cmd.extend(["--physics-warmup-steps", str(config['physics_warmup'])])

        # Run and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)

        return self.parse_training_output(result.stdout, config, run_id)

    def parse_training_output(self, output: str, config: dict, run_id: int) -> dict:
        """Extract key metrics from training output."""
        metrics = {
            'run_id': run_id,
            'config': config,
            'epochs_until_hold_70': None,  # When HOLD% first exceeds 70%
            'epochs_until_entropy_3': None,  # When entropy drops below 3.0
            'final_reward': None,
            'max_reward': None,
            'final_entropy': None,
            'final_hold_pct': None,
            'final_pred_loss': None,
        }
        # Parse logic here...
        return metrics

    def run_sweep(self, epochs_per_run: int = 50):
        """Run full sweep."""
        configs = self.generate_configs()

        for i, config in enumerate(configs):
            print(f"\n{'='*60}")
            print(f"Running config {i+1}/{len(configs)}: {config}")
            print(f"{'='*60}\n")

            try:
                result = self.run_single_config(config, i, epochs_per_run)
                self.results.append(result)
                self.save_intermediate()
            except Exception as e:
                print(f"Config {i} failed: {e}")

        self.save_final_report()

    def save_intermediate(self):
        """Save results after each run."""
        df = pd.DataFrame(self.results)
        df.to_csv(self.output_dir / "sweep_progress.csv", index=False)

    def save_final_report(self):
        """Generate final analysis report."""
        df = pd.DataFrame(self.results)

        # Rank by stability metrics
        df['stability_score'] = (
            df['epochs_until_hold_70'].fillna(100) * 0.4 +
            df['epochs_until_entropy_3'].fillna(100) * 0.3 +
            df['max_reward'] * 0.3
        )

        df_sorted = df.sort_values('stability_score', ascending=False)
        df_sorted.to_csv(self.output_dir / "sweep_final.csv", index=False)

        # Print top 3
        print("\n" + "="*60)
        print("TOP 3 CONFIGURATIONS:")
        print("="*60)
        for i, row in df_sorted.head(3).iterrows():
            print(f"\n{row['config']}")
            print(f"  Stability Score: {row['stability_score']:.2f}")
            print(f"  Epochs until HOLD>70%: {row['epochs_until_hold_70']}")
            print(f"  Max Reward: {row['max_reward']:.2f}")
```

### Phase 2: Metrics Collection

Key metrics to track per epoch:
- `hold_pct`: Percentage of HOLD actions
- `policy_entropy`: From training metrics
- `pred_loss`: Prediction head loss
- `step1_mse`: 1-step prediction MSE
- `reward`: Episode reward
- `alpha`: Agent return - market return

### Phase 3: Analysis Output

Generate report with:
1. Stability ranking (epochs until collapse)
2. Performance ranking (max reward achieved)
3. Pareto frontier (stability vs performance)
4. Recommended configuration for TASK_037

## Success Criteria

1. Find at least one config that maintains entropy > 3.5 for 50+ epochs
2. Find config where prediction loss stays > 0.1 for 30+ epochs
3. Identify whether epsilon-greedy is key intervention

## Files to Create/Modify

- **CREATE**: `warmup_sweep.py` - Main sweep script
- **CREATE**: `sweep_results/` - Output directory
- **MODIFY**: `train_jax_fast.py` - Add `--epsilon-greedy` flag (placeholder for TASK_037)

## Dependencies

- TASK_037 (ε-greedy MCTS) - For full sweep, but can run baseline without it

## Estimated Runtime

- 10 configs × 50 epochs × ~2 min/epoch = ~17 hours overnight
- Can reduce to 30 epochs for faster feedback: ~10 hours

## Notes

- Run on synthetic level 0 (sine waves) for fast iteration
- Use small model to minimize GPU time per run
- Save checkpoints for promising configs to resume later
