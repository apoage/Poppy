#!/usr/bin/env python3
"""
Cross-Validation Framework for Training Changes

Usage:
    python cross_validate.py --baseline run1 --experiment run2
    python cross_validate.py --compare logs/training_*.csv
    python cross_validate.py --quick-test --epochs 50
"""

import argparse
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import subprocess
import json
from datetime import datetime


def load_training_log(csv_path: str) -> pd.DataFrame:
    """Load and validate training CSV log."""
    df = pd.read_csv(csv_path)

    # Ensure required columns exist
    required = ['epoch', 'reward', 'value_loss', 'policy_entropy', 'avg_value']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Warning: Missing columns in {csv_path}: {missing}")

    return df


def compute_metrics(df: pd.DataFrame, name: str = "run") -> Dict:
    """Compute summary metrics from training log."""

    # Skip warmup epochs (first 5)
    df_train = df[df['epoch'] > 5].copy()

    if len(df_train) == 0:
        return {"name": name, "error": "No training epochs"}

    # Core metrics
    metrics = {
        "name": name,
        "total_epochs": len(df),
        "training_epochs": len(df_train),

        # Reward metrics
        "reward_mean": df_train['reward'].mean(),
        "reward_std": df_train['reward'].std(),
        "reward_final_ma20": df_train['reward'].tail(20).mean() if len(df_train) >= 20 else df_train['reward'].mean(),
        "reward_max": df_train['reward'].max(),
        "reward_min": df_train['reward'].min(),

        # Value head metrics
        "value_loss_initial": df_train['value_loss'].head(10).mean() if len(df_train) >= 10 else df_train['value_loss'].mean(),
        "value_loss_final": df_train['value_loss'].tail(10).mean() if len(df_train) >= 10 else df_train['value_loss'].mean(),

        # Policy metrics
        "entropy_initial": df_train['policy_entropy'].head(10).mean() if len(df_train) >= 10 else df_train['policy_entropy'].mean(),
        "entropy_final": df_train['policy_entropy'].tail(10).mean() if len(df_train) >= 10 else df_train['policy_entropy'].mean(),

        # Timing
        "avg_self_play_time": df_train['self_play_time'].mean() if 'self_play_time' in df_train else 0,
        "avg_train_time": df_train['train_time'].mean() if 'train_time' in df_train else 0,
    }

    # Value-Reward correlation (key metric for value head quality)
    if 'avg_value' in df_train.columns and df_train['avg_value'].std() > 0:
        metrics["value_reward_corr"] = df_train['avg_value'].corr(df_train['reward'])
    else:
        metrics["value_reward_corr"] = 0.0

    # Compute derived metrics
    metrics["value_loss_reduction"] = (
        (metrics["value_loss_initial"] - metrics["value_loss_final"]) /
        metrics["value_loss_initial"] * 100 if metrics["value_loss_initial"] > 0 else 0
    )

    metrics["entropy_change"] = (
        (metrics["entropy_final"] - metrics["entropy_initial"]) /
        metrics["entropy_initial"] * 100 if metrics["entropy_initial"] > 0 else 0
    )

    # Action distribution (if available)
    if 'pct_buy' in df_train.columns:
        metrics["avg_pct_buy"] = df_train['pct_buy'].tail(20).mean()
        metrics["avg_pct_hold"] = df_train['pct_hold'].tail(20).mean()
        metrics["avg_pct_sell"] = df_train['pct_sell'].tail(20).mean()

    # MSE tracking (if available)
    if 'value_mse' in df_train.columns:
        metrics["value_mse_initial"] = df_train['value_mse'].head(10).mean()
        metrics["value_mse_final"] = df_train['value_mse'].tail(10).mean()
        metrics["value_mse_reduction"] = (
            (metrics["value_mse_initial"] - metrics["value_mse_final"]) /
            metrics["value_mse_initial"] * 100 if metrics["value_mse_initial"] > 0 else 0
        )

    return metrics


def compare_runs(baseline: Dict, experiment: Dict) -> Dict:
    """Compare two runs and compute deltas."""

    comparison = {
        "baseline": baseline["name"],
        "experiment": experiment["name"],
        "deltas": {}
    }

    # Key metrics to compare
    compare_keys = [
        ("reward_final_ma20", "higher_better"),
        ("value_reward_corr", "higher_better"),
        ("value_loss_final", "lower_better"),
        ("entropy_final", "in_range"),  # Should be 3.0-4.5
        ("avg_self_play_time", "lower_better"),
    ]

    for key, direction in compare_keys:
        if key in baseline and key in experiment:
            base_val = baseline[key]
            exp_val = experiment[key]

            if base_val != 0:
                pct_change = (exp_val - base_val) / abs(base_val) * 100
            else:
                pct_change = 0

            # Determine if change is improvement
            if direction == "higher_better":
                is_better = exp_val > base_val
            elif direction == "lower_better":
                is_better = exp_val < base_val
            else:  # in_range
                is_better = 3.0 <= exp_val <= 4.5

            comparison["deltas"][key] = {
                "baseline": base_val,
                "experiment": exp_val,
                "pct_change": pct_change,
                "is_improvement": is_better
            }

    # Overall assessment
    improvements = sum(1 for d in comparison["deltas"].values() if d["is_improvement"])
    comparison["improvements"] = improvements
    comparison["total_metrics"] = len(comparison["deltas"])
    comparison["verdict"] = "BETTER" if improvements > len(comparison["deltas"]) / 2 else "WORSE"

    return comparison


def print_metrics(metrics: Dict):
    """Pretty print metrics."""
    print(f"\n{'='*60}")
    print(f"Run: {metrics['name']}")
    print(f"{'='*60}")

    print(f"\n📊 Reward Performance:")
    print(f"   Mean:     {metrics['reward_mean']:+.2f}%")
    print(f"   Std:      {metrics['reward_std']:.2f}%")
    print(f"   Final MA: {metrics['reward_final_ma20']:+.2f}%")
    print(f"   Range:    [{metrics['reward_min']:+.2f}, {metrics['reward_max']:+.2f}]%")

    print(f"\n🎯 Value Head:")
    print(f"   Correlation: {metrics['value_reward_corr']:.3f}", end="")
    if metrics['value_reward_corr'] > 0.8:
        print(" ✓ (excellent)")
    elif metrics['value_reward_corr'] > 0.5:
        print(" ~ (learning)")
    else:
        print(" ✗ (poor)")

    print(f"   Loss: {metrics['value_loss_initial']:.2f} → {metrics['value_loss_final']:.2f} ({metrics['value_loss_reduction']:+.1f}%)")

    if 'value_mse_final' in metrics:
        print(f"   MSE:  {metrics['value_mse_initial']:.2f} → {metrics['value_mse_final']:.2f} ({metrics['value_mse_reduction']:+.1f}%)")

    print(f"\n🎲 Policy:")
    print(f"   Entropy: {metrics['entropy_initial']:.2f} → {metrics['entropy_final']:.2f} ({metrics['entropy_change']:+.1f}%)")

    if 'avg_pct_buy' in metrics:
        print(f"   Actions: B={metrics['avg_pct_buy']:.1f}% H={metrics['avg_pct_hold']:.1f}% S={metrics['avg_pct_sell']:.1f}%")

    print(f"\n⏱️  Timing:")
    print(f"   Self-play: {metrics['avg_self_play_time']:.1f}s")
    print(f"   Training:  {metrics['avg_train_time']:.1f}s")


def print_comparison(comp: Dict):
    """Pretty print comparison."""
    print(f"\n{'='*60}")
    print(f"COMPARISON: {comp['baseline']} vs {comp['experiment']}")
    print(f"{'='*60}")

    print(f"\n{'Metric':<25} {'Baseline':>12} {'Experiment':>12} {'Change':>10} {'Better?':>8}")
    print("-" * 70)

    for key, delta in comp["deltas"].items():
        better = "✓" if delta["is_improvement"] else "✗"
        print(f"{key:<25} {delta['baseline']:>12.3f} {delta['experiment']:>12.3f} {delta['pct_change']:>+9.1f}% {better:>8}")

    print("-" * 70)
    print(f"\nVerdict: {comp['verdict']} ({comp['improvements']}/{comp['total_metrics']} metrics improved)")


def run_quick_test(epochs: int = 50, configs: List[Dict] = None):
    """Run quick A/B test with two configurations."""

    if configs is None:
        # Default: test current vs previous
        configs = [
            {"name": "baseline", "args": "--entropy-coef 0.01"},
            {"name": "experiment", "args": "--entropy-coef 0.02"},
        ]

    results = []

    for config in configs:
        name = config["name"]
        args = config["args"]

        print(f"\n🚀 Running {name}...")

        cmd = f"python train_jax_fast.py --synthetic-level 0 --model-size large --epochs {epochs} --name cv_{name} {args}"

        try:
            subprocess.run(cmd.split(), check=True)

            csv_path = f"logs/training_cv_{name}.csv"
            if os.path.exists(csv_path):
                df = load_training_log(csv_path)
                metrics = compute_metrics(df, name)
                results.append(metrics)
                print_metrics(metrics)
        except subprocess.CalledProcessError as e:
            print(f"Error running {name}: {e}")

    if len(results) >= 2:
        comp = compare_runs(results[0], results[1])
        print_comparison(comp)

        # Save comparison
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"logs/cv_comparison_{timestamp}.json", "w") as f:
            json.dump(comp, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Cross-validation framework")
    parser.add_argument("--baseline", type=str, help="Baseline run name or CSV path")
    parser.add_argument("--experiment", type=str, help="Experiment run name or CSV path")
    parser.add_argument("--compare", type=str, nargs="+", help="Compare multiple CSV files")
    parser.add_argument("--quick-test", action="store_true", help="Run quick A/B test")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs for quick test")

    args = parser.parse_args()

    if args.quick_test:
        run_quick_test(args.epochs)
        return

    if args.compare:
        # Compare multiple runs
        all_metrics = []
        for csv_path in args.compare:
            if os.path.exists(csv_path):
                df = load_training_log(csv_path)
                name = Path(csv_path).stem.replace("training_", "")
                metrics = compute_metrics(df, name)
                all_metrics.append(metrics)
                print_metrics(metrics)

        # Pairwise comparisons
        if len(all_metrics) >= 2:
            print("\n" + "="*60)
            print("PAIRWISE COMPARISONS")
            print("="*60)
            for i in range(len(all_metrics) - 1):
                comp = compare_runs(all_metrics[i], all_metrics[i+1])
                print_comparison(comp)
        return

    if args.baseline and args.experiment:
        # Compare two specific runs
        def get_csv_path(name):
            if os.path.exists(name):
                return name
            return f"logs/training_{name}.csv"

        base_path = get_csv_path(args.baseline)
        exp_path = get_csv_path(args.experiment)

        if os.path.exists(base_path) and os.path.exists(exp_path):
            base_df = load_training_log(base_path)
            exp_df = load_training_log(exp_path)

            base_metrics = compute_metrics(base_df, args.baseline)
            exp_metrics = compute_metrics(exp_df, args.experiment)

            print_metrics(base_metrics)
            print_metrics(exp_metrics)

            comp = compare_runs(base_metrics, exp_metrics)
            print_comparison(comp)
        else:
            print(f"Error: Could not find CSV files")
            if not os.path.exists(base_path):
                print(f"  Missing: {base_path}")
            if not os.path.exists(exp_path):
                print(f"  Missing: {exp_path}")
        return

    # Default: show available logs
    print("Cross-Validation Framework")
    print("-" * 40)
    print("\nAvailable training logs:")

    log_dir = Path("logs")
    if log_dir.exists():
        for csv in sorted(log_dir.glob("training_*.csv")):
            df = pd.read_csv(csv)
            print(f"  {csv.name}: {len(df)} epochs")

    print("\nUsage:")
    print("  python cross_validate.py --baseline run1 --experiment run2")
    print("  python cross_validate.py --compare logs/training_*.csv")
    print("  python cross_validate.py --quick-test --epochs 50")


if __name__ == "__main__":
    main()
