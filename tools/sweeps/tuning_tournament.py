#!/usr/bin/env python3
# Add project root to path for imports
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))
"""
Tuning Tournament - Compare stability features across winning configs.

Uses the top 3 configs from hyperparameter tournament as base,
then tests combinations of:
- Prediction mode: no_pred, physics_only, hybrid
- Value stabilization: baseline, huber, popart
- MCTS warmup: none, fast, slow

Run: python tuning_tournament.py --round A  (or B, C, D, all)
"""

import argparse
import subprocess
import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import csv

# ============================================================
# BASE CONFIGURATIONS (from hyperparameter tournament winners)
# ============================================================

@dataclass
class BaseConfig:
    """Base configuration from hyperparameter tournament."""
    name: str
    sims: int
    lr: float
    h_cycles: int
    prediction_weight: float  # 0 = no prediction
    gumbel_scale: float
    score: float  # Tournament score

    def to_args(self) -> List[str]:
        """Convert to CLI arguments."""
        args = [
            f"--sims={self.sims}",
            f"--lr={self.lr}",
            f"--h-cycles={self.h_cycles}",
            f"--gumbel-scale={self.gumbel_scale}",
        ]
        if self.prediction_weight > 0:
            args.append("--use-prediction-loss")
            args.append(f"--prediction-loss-weight={self.prediction_weight}")
        return args


# Top 3 winners from hyperparameter tournament
BASE_CONFIGS = [
    BaseConfig(
        name="s16_lr10_h2_np",
        sims=16,
        lr=1e-4,
        h_cycles=2,
        prediction_weight=0.0,
        gumbel_scale=1.5,
        score=-2.35
    ),
    BaseConfig(
        name="s16_lr10_h4_p10",
        sims=16,
        lr=1e-4,
        h_cycles=4,
        prediction_weight=0.1,
        gumbel_scale=1.5,
        score=-2.73
    ),
    BaseConfig(
        name="s32_lr10_h2_p3",
        sims=32,
        lr=1e-4,
        h_cycles=2,
        prediction_weight=0.03,
        gumbel_scale=1.5,
        score=-3.37
    ),
]


# ============================================================
# TUNING VARIATIONS
# ============================================================

@dataclass
class TuningVariation:
    """A tuning variation to test."""
    name: str
    args: List[str]
    description: str


# Round A: Prediction Mode
PREDICTION_VARIATIONS = [
    TuningVariation(
        name="no_pred",
        args=["--prediction-loss-weight=0"],
        description="No prediction loss (baseline)"
    ),
    TuningVariation(
        name="physics",
        args=["--use-prediction-loss", "--physics-only", "--prediction-loss-weight=0.2"],
        description="Physics-only prediction (interpretable)"
    ),
    TuningVariation(
        name="hybrid",
        args=["--use-prediction-loss", "--prediction-loss-weight=0.2"],
        description="Hybrid physics+neural prediction"
    ),
    TuningVariation(
        name="tree",
        args=["--use-prediction-loss", "--prediction-loss-weight=0.2",
              "--use-tree-search", "--tree-depth=5", "--risk-lambda=0.1"],
        description="Hybrid + predictive tree search (risk-adjusted MCTS)"
    ),
]

# Round B: Value Stabilization
VALUE_VARIATIONS = [
    TuningVariation(
        name="baseline",
        args=[],
        description="No value stabilization"
    ),
    TuningVariation(
        name="huber",
        args=["--value-loss-type=huber", "--value-huber-delta=1.0"],
        description="Huber loss (robust to outliers)"
    ),
    TuningVariation(
        name="no_popart",
        args=["--no-popart"],
        description="Disable PopArt (now default=True) for comparison"
    ),
]

# Round C: MCTS Warmup
WARMUP_VARIATIONS = [
    TuningVariation(
        name="no_warmup",
        args=[],
        description="No sims warmup"
    ),
    TuningVariation(
        name="warmup_fast",
        args=["--sims-warmup=50", "--sims-min=8"],
        description="Fast warmup (50 epochs)"
    ),
    TuningVariation(
        name="warmup_slow",
        args=["--sims-warmup=100", "--sims-min=8"],
        description="Slow warmup (100 epochs)"
    ),
]


# ============================================================
# TOURNAMENT RUNNER
# ============================================================

@dataclass
class RunResult:
    """Result of a single run."""
    config_name: str
    variation_name: str
    round_name: str
    final_reward: float
    reward_ma50: float
    hold_pct: float
    value_loss: float
    value_std: float
    pred_loss: float  # Prediction loss (how much prediction helps)
    training_time: float
    success: bool
    error: Optional[str] = None


def run_training(
    base_config: BaseConfig,
    variation: TuningVariation,
    round_name: str,
    epochs: int = 150,
    steps: int = 200,
    synthetic_level: int = 0,
    dry_run: bool = False
) -> RunResult:
    """Run a single training configuration."""

    run_name = f"tune_{round_name}_{base_config.name}_{variation.name}"

    # Build command
    cmd = [
        "python", "-u", "train_jax_fast.py",
        f"--synthetic-level={synthetic_level}",
        f"--epochs={epochs}",
        f"--steps={steps}",
        "--model-size=small",
        "--batch-size=256",
        "--parallel-envs=4",
        "--adaptive-entropy",  # Critical for stability - prevents entropy collapse
        f"--name={run_name}",
    ]

    # Add base config args
    cmd.extend(base_config.to_args())

    # Add variation args (may override base config)
    cmd.extend(variation.args)

    print(f"\n{'='*60}")
    print(f"Running: {run_name}")
    print(f"Base: {base_config.name} (score={base_config.score})")
    print(f"Variation: {variation.name} - {variation.description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    if dry_run:
        return RunResult(
            config_name=base_config.name,
            variation_name=variation.name,
            round_name=round_name,
            final_reward=0.0,
            reward_ma50=0.0,
            hold_pct=0.0,
            value_loss=0.0,
            value_std=0.0,
            pred_loss=0.0,
            training_time=0.0,
            success=True,
            error="DRY RUN"
        )

    start_time = time.time()

    try:
        # Run training with live output streaming
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # Line buffered
        )

        # Collect output while streaming to terminal
        output_lines = []
        for line in process.stdout:
            print(line, end='', flush=True)  # Stream to terminal
            output_lines.append(line)

        process.wait(timeout=3600)  # 1 hour timeout
        returncode = process.returncode

        training_time = time.time() - start_time

        # Parse output for metrics
        output = ''.join(output_lines)

        # Extract final metrics from last epoch
        final_reward = 0.0
        reward_ma50 = 0.0
        hold_pct = 0.0
        value_loss = 0.0
        value_std = 0.0
        pred_loss = 0.0

        for line in reversed(output.split('\n')):
            if 'Epoch' in line and 'Reward:' in line:
                # Parse: Epoch  150 | Reward:  +1.23 | ... | HOLD: 75% | ... | Pred: 0.0234 |
                try:
                    parts = line.split('|')
                    for part in parts:
                        part = part.strip()
                        if part.startswith('Reward:'):
                            final_reward = float(part.split(':')[1].strip().split()[0])
                        elif 'HOLD:' in part or 'Hold:' in part:
                            hold_str = part.split(':')[1].strip().replace('%', '').split()[0]
                            hold_pct = float(hold_str)
                        elif part.startswith('Val:'):
                            value_loss = float(part.split(':')[1].strip().split()[0])
                        elif part.startswith('Pred:'):
                            # Parse prediction loss: "Pred: 0.0234 (λ=0.200)"
                            pred_str = part.split(':')[1].strip().split()[0]
                            pred_loss = float(pred_str)
                except:
                    pass
                break

        # Try to get MA50 from near-end output
        for line in reversed(output.split('\n')[-100:]):
            if 'MA50' in line or 'ma50' in line:
                try:
                    # Various formats possible
                    if 'MA50:' in line:
                        reward_ma50 = float(line.split('MA50:')[1].split()[0])
                except:
                    pass
                break

        if reward_ma50 == 0.0:
            reward_ma50 = final_reward  # Fallback

        return RunResult(
            config_name=base_config.name,
            variation_name=variation.name,
            round_name=round_name,
            final_reward=final_reward,
            reward_ma50=reward_ma50,
            hold_pct=hold_pct,
            value_loss=value_loss,
            value_std=value_std,
            pred_loss=pred_loss,
            training_time=training_time,
            success=returncode == 0,
            error=None if returncode == 0 else output[-500:]
        )

    except subprocess.TimeoutExpired:
        return RunResult(
            config_name=base_config.name,
            variation_name=variation.name,
            round_name=round_name,
            final_reward=0.0,
            reward_ma50=0.0,
            hold_pct=0.0,
            value_loss=0.0,
            value_std=0.0,
            pred_loss=0.0,
            training_time=3600.0,
            success=False,
            error="TIMEOUT"
        )
    except Exception as e:
        return RunResult(
            config_name=base_config.name,
            variation_name=variation.name,
            round_name=round_name,
            final_reward=0.0,
            reward_ma50=0.0,
            hold_pct=0.0,
            value_loss=0.0,
            value_std=0.0,
            pred_loss=0.0,
            training_time=time.time() - start_time,
            success=False,
            error=str(e)
        )


def run_round(
    round_name: str,
    variations: List[TuningVariation],
    base_configs: List[BaseConfig],
    epochs: int = 150,
    steps: int = 200,
    synthetic_level: int = 0,
    dry_run: bool = False
) -> List[RunResult]:
    """Run all combinations for a round."""

    results = []
    total_runs = len(base_configs) * len(variations)
    current_run = 0

    print(f"\n{'#'*60}")
    print(f"# ROUND {round_name}: {len(variations)} variations x {len(base_configs)} configs = {total_runs} runs")
    print(f"{'#'*60}\n")

    for base_config in base_configs:
        for variation in variations:
            current_run += 1
            print(f"\n[{current_run}/{total_runs}] ", end="")

            result = run_training(
                base_config=base_config,
                variation=variation,
                round_name=round_name,
                epochs=epochs,
                steps=steps,
                synthetic_level=synthetic_level,
                dry_run=dry_run
            )
            results.append(result)

            # Print result
            status = "✓" if result.success else "✗"
            pred_str = f", pred={result.pred_loss:.4f}" if result.pred_loss > 0 else ""
            print(f"{status} {result.config_name}+{result.variation_name}: "
                  f"reward={result.final_reward:+.2f}, hold={result.hold_pct:.0f}%{pred_str}, "
                  f"time={result.training_time:.0f}s")

    return results


def save_results(results: List[RunResult], filename: str):
    """Save results to CSV."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'round', 'base_config', 'variation', 'final_reward', 'reward_ma50',
            'hold_pct', 'value_loss', 'pred_loss', 'training_time', 'success', 'error'
        ])
        for r in results:
            writer.writerow([
                r.round_name, r.config_name, r.variation_name, r.final_reward,
                r.reward_ma50, r.hold_pct, r.value_loss, r.pred_loss, r.training_time,
                r.success, r.error or ''
            ])
    print(f"\nResults saved to: {filename}")


def print_round_summary(results: List[RunResult], round_name: str):
    """Print summary of round results."""

    print(f"\n{'='*60}")
    print(f"ROUND {round_name} SUMMARY")
    print(f"{'='*60}\n")

    # Group by variation
    by_variation: Dict[str, List[RunResult]] = {}
    for r in results:
        if r.variation_name not in by_variation:
            by_variation[r.variation_name] = []
        by_variation[r.variation_name].append(r)

    # Check if any results have pred_loss (Round A)
    has_pred = any(r.pred_loss > 0 for r in results)

    if has_pred:
        print(f"{'Variation':<15} {'Avg Reward':>12} {'Avg Hold%':>10} {'Avg Pred':>10} {'Wins':>6}")
        print("-" * 55)
    else:
        print(f"{'Variation':<15} {'Avg Reward':>12} {'Avg Hold%':>10} {'Wins':>6}")
        print("-" * 45)

    variation_scores = []
    for var_name, var_results in by_variation.items():
        successful = [r for r in var_results if r.success]
        if successful:
            avg_reward = sum(r.final_reward for r in successful) / len(successful)
            avg_hold = sum(r.hold_pct for r in successful) / len(successful)
            avg_pred = sum(r.pred_loss for r in successful) / len(successful)
            wins = len(successful)
        else:
            avg_reward = float('-inf')
            avg_hold = 0
            avg_pred = 0
            wins = 0

        variation_scores.append((var_name, avg_reward, avg_hold, avg_pred, wins))
        if has_pred:
            print(f"{var_name:<15} {avg_reward:>+12.2f} {avg_hold:>10.1f} {avg_pred:>10.4f} {wins:>6}/{len(var_results)}")
        else:
            print(f"{var_name:<15} {avg_reward:>+12.2f} {avg_hold:>10.1f} {wins:>6}/{len(var_results)}")

    # Best variation
    best = max(variation_scores, key=lambda x: x[1])
    print(f"\n🏆 Best variation: {best[0]} (avg reward: {best[1]:+.2f})")

    return best[0]


def main():
    parser = argparse.ArgumentParser(description="Tuning Tournament")
    parser.add_argument('--round', type=str, default='all',
                       choices=['A', 'B', 'C', 'D', 'all'],
                       help='Which round to run (A=prediction, B=value, C=warmup, D=final, all=all)')
    parser.add_argument('--epochs', type=int, default=150,
                       help='Epochs per run (default: 150)')
    parser.add_argument('--steps', type=int, default=200,
                       help='Steps per episode (default: 200)')
    parser.add_argument('--synthetic-level', type=int, default=0,
                       help='Synthetic level (default: 0)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print commands without running')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file (default: tuning_results_TIMESTAMP.csv)')
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output or f"tuning_results_{timestamp}.csv"

    all_results = []
    best_variations = {}

    print(f"\n{'#'*60}")
    print(f"# TUNING TOURNAMENT")
    print(f"# Base configs: {len(BASE_CONFIGS)}")
    print(f"# Epochs: {args.epochs}, Steps: {args.steps}")
    print(f"# Synthetic Level: {args.synthetic_level}")
    print(f"{'#'*60}\n")

    # Round A: Prediction Mode
    if args.round in ['A', 'all']:
        results = run_round(
            round_name='A',
            variations=PREDICTION_VARIATIONS,
            base_configs=BASE_CONFIGS,
            epochs=args.epochs,
            steps=args.steps,
            synthetic_level=args.synthetic_level,
            dry_run=args.dry_run
        )
        all_results.extend(results)
        best_variations['A'] = print_round_summary(results, 'A')

    # Round B: Value Stabilization
    if args.round in ['B', 'all']:
        results = run_round(
            round_name='B',
            variations=VALUE_VARIATIONS,
            base_configs=BASE_CONFIGS,
            epochs=args.epochs,
            steps=args.steps,
            synthetic_level=args.synthetic_level,
            dry_run=args.dry_run
        )
        all_results.extend(results)
        best_variations['B'] = print_round_summary(results, 'B')

    # Round C: MCTS Warmup
    if args.round in ['C', 'all']:
        results = run_round(
            round_name='C',
            variations=WARMUP_VARIATIONS,
            base_configs=BASE_CONFIGS,
            epochs=args.epochs,
            steps=args.steps,
            synthetic_level=args.synthetic_level,
            dry_run=args.dry_run
        )
        all_results.extend(results)
        best_variations['C'] = print_round_summary(results, 'C')

    # Round D: Final (combine best from A, B, C)
    if args.round in ['D', 'all'] and len(best_variations) >= 3:
        print(f"\n{'#'*60}")
        print(f"# ROUND D: FINAL COMBINATION")
        print(f"# Best from A: {best_variations.get('A', 'N/A')}")
        print(f"# Best from B: {best_variations.get('B', 'N/A')}")
        print(f"# Best from C: {best_variations.get('C', 'N/A')}")
        print(f"{'#'*60}\n")

        # Build combined variation from winners
        combined_args = []

        # Add prediction variation
        for v in PREDICTION_VARIATIONS:
            if v.name == best_variations.get('A'):
                combined_args.extend(v.args)
                break

        # Add value variation
        for v in VALUE_VARIATIONS:
            if v.name == best_variations.get('B'):
                combined_args.extend(v.args)
                break

        # Add warmup variation
        for v in WARMUP_VARIATIONS:
            if v.name == best_variations.get('C'):
                combined_args.extend(v.args)
                break

        combined_variation = TuningVariation(
            name="combined_best",
            args=combined_args,
            description=f"Best combo: {best_variations.get('A')}+{best_variations.get('B')}+{best_variations.get('C')}"
        )

        results = run_round(
            round_name='D',
            variations=[combined_variation],
            base_configs=BASE_CONFIGS,
            epochs=args.epochs,
            steps=args.steps,
            synthetic_level=args.synthetic_level,
            dry_run=args.dry_run
        )
        all_results.extend(results)
        print_round_summary(results, 'D')

    # Save all results
    if all_results and not args.dry_run:
        save_results(all_results, output_file)

    # Final summary
    print(f"\n{'#'*60}")
    print(f"# TOURNAMENT COMPLETE")
    print(f"# Total runs: {len(all_results)}")
    print(f"# Results: {output_file}")
    print(f"{'#'*60}\n")


if __name__ == '__main__':
    main()
