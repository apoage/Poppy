#!/usr/bin/env python3
"""
Analyze stability sweep results and rank configurations.

Usage:
    python analyze_sweep.py logs/stability_sweep_20260106_220809
    python analyze_sweep.py --latest  # Auto-find most recent sweep
"""

import argparse
import glob
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ConfigScore:
    """Scores for a single configuration."""
    name: str
    csv_path: str

    # Raw metrics
    epochs_completed: int
    max_reward: float
    final_reward: float
    mean_reward: float

    # Entropy metrics
    min_entropy: float
    mean_entropy: float
    final_entropy: float
    entropy_std: float  # Lower = more stable

    # Hold metrics
    hold_pct_at_end: float
    epoch_hold_70: Optional[int]  # First epoch where HOLD > 70%
    epoch_hold_80: Optional[int]  # First epoch where HOLD > 80%
    mean_hold_pct: float = 50.0

    # Prediction model metrics
    pred_loss_final: float = 0.0
    pred_loss_mean: float = 0.0
    pred_loss_std: float = 0.0  # Lower = more stable
    pred_loss_min: float = 0.0

    # Alpha (outperformance vs market)
    max_alpha: float = 0.0
    mean_alpha: float = 0.0
    final_alpha: float = 0.0

    # Computed scores (0-1, higher is better)
    reward_score: float = 0.0
    entropy_stability_score: float = 0.0  # High mean + low std
    pred_stability_score: float = 0.0     # Low loss + low std
    hold_delay_score: float = 0.0
    alpha_score: float = 0.0
    composite_score: float = 0.0


def find_latest_sweep_dir(base_dir: str = "logs") -> Optional[str]:
    """Find the most recent stability_sweep directory."""
    pattern = os.path.join(base_dir, "stability_sweep_*")
    dirs = sorted(glob.glob(pattern), reverse=True)
    return dirs[0] if dirs else None


def parse_csv(csv_path: str) -> Optional[pd.DataFrame]:
    """Parse a training CSV file."""
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"  Warning: Could not parse {csv_path}: {e}")
        return None


def find_first_epoch_above(series: pd.Series, threshold: float, window: int = 5) -> Optional[int]:
    """Find first epoch where rolling median exceeds threshold.

    Uses rolling median to filter out spikes - requires sustained behavior.
    """
    if len(series) < window:
        # Too short for rolling, use raw values
        above = series > threshold
        if above.any():
            return int(above.idxmax())
        return None

    # Rolling median filters spikes
    rolling_median = series.rolling(window=window, center=True, min_periods=1).median()
    above = rolling_median > threshold
    if above.any():
        return int(above.idxmax())
    return None


def analyze_config(csv_path: str, total_epochs: int = 300) -> Optional[ConfigScore]:
    """Analyze a single configuration's results."""
    df = parse_csv(csv_path)
    if df is None or len(df) == 0:
        return None

    name = Path(csv_path).stem

    # Extract metrics - handle different column naming conventions
    # Primary columns from train_jax_fast.py
    reward_col = None
    for col in ['reward', 'reward_ma20', 'mean_reward', 'avg_reward', 'episode_reward']:
        if col in df.columns:
            reward_col = col
            break

    entropy_col = None
    for col in ['policy_entropy', 'entropy', 'mean_entropy']:
        if col in df.columns:
            entropy_col = col
            break

    hold_col = None
    for col in ['pct_hold', 'hold_pct', 'hold_percent', 'hold_%', 'action_hold_pct']:
        if col in df.columns:
            hold_col = col
            break

    # Also extract alpha if available (agent outperformance vs market)
    alpha_col = 'alpha' if 'alpha' in df.columns else None

    # Prediction loss column
    pred_col = 'pred_loss' if 'pred_loss' in df.columns else None

    # If we can't find reward column, try to compute from other columns
    if reward_col is None:
        print(f"  Warning: No reward column found in {name}")
        print(f"  Available columns: {list(df.columns)}")
        return None

    rewards = df[reward_col].values
    epochs_completed = len(df)

    # Reward metrics
    max_reward = float(np.max(rewards))
    final_reward = float(rewards[-1]) if len(rewards) > 0 else 0.0
    mean_reward = float(np.mean(rewards))

    # Entropy metrics (default if not found)
    if entropy_col:
        entropies = df[entropy_col].values
        # Filter out NaN/inf
        entropies = entropies[np.isfinite(entropies)]
        if len(entropies) > 0:
            min_entropy = float(np.min(entropies))
            mean_entropy = float(np.mean(entropies))
            final_entropy = float(entropies[-1])
            entropy_std = float(np.std(entropies))
        else:
            min_entropy = mean_entropy = final_entropy = 4.0
            entropy_std = 0.0
    else:
        min_entropy = mean_entropy = final_entropy = 4.0
        entropy_std = 0.0

    # Hold percentage metrics
    if hold_col:
        hold_pcts = df[hold_col].values
        hold_pcts = hold_pcts[np.isfinite(hold_pcts)]
        if len(hold_pcts) > 0:
            hold_pct_at_end = float(hold_pcts[-1])
            mean_hold_pct = float(np.mean(hold_pcts))
            # Find when HOLD crossed thresholds
            hold_series = pd.Series(hold_pcts)
            epoch_hold_70 = find_first_epoch_above(hold_series, 70.0)
            epoch_hold_80 = find_first_epoch_above(hold_series, 80.0)
        else:
            hold_pct_at_end = mean_hold_pct = 50.0
            epoch_hold_70 = epoch_hold_80 = None
    else:
        hold_pct_at_end = mean_hold_pct = 50.0
        epoch_hold_70 = epoch_hold_80 = None

    # Prediction loss metrics (key for stability!)
    if pred_col:
        pred_losses = df[pred_col].values
        # Filter out NaN/inf and zeros (warmup period)
        pred_losses = pred_losses[np.isfinite(pred_losses) & (pred_losses > 0)]
        if len(pred_losses) > 0:
            pred_loss_final = float(pred_losses[-1])
            pred_loss_mean = float(np.mean(pred_losses))
            pred_loss_std = float(np.std(pred_losses))
            pred_loss_min = float(np.min(pred_losses))
        else:
            pred_loss_final = pred_loss_mean = pred_loss_std = pred_loss_min = 0.0
    else:
        pred_loss_final = pred_loss_mean = pred_loss_std = pred_loss_min = 0.0

    # Alpha metrics (agent vs market outperformance)
    if alpha_col:
        alphas = df[alpha_col].values
        alphas = alphas[np.isfinite(alphas)]
        if len(alphas) > 0:
            max_alpha = float(np.max(alphas))
            mean_alpha = float(np.mean(alphas))
            final_alpha = float(alphas[-1])
        else:
            max_alpha = mean_alpha = final_alpha = 0.0
    else:
        max_alpha = mean_alpha = final_alpha = 0.0

    score = ConfigScore(
        name=name,
        csv_path=csv_path,
        epochs_completed=epochs_completed,
        max_reward=max_reward,
        final_reward=final_reward,
        mean_reward=mean_reward,
        min_entropy=min_entropy,
        mean_entropy=mean_entropy,
        final_entropy=final_entropy,
        entropy_std=entropy_std,
        hold_pct_at_end=hold_pct_at_end,
        mean_hold_pct=mean_hold_pct,
        epoch_hold_70=epoch_hold_70,
        epoch_hold_80=epoch_hold_80,
        pred_loss_final=pred_loss_final,
        pred_loss_mean=pred_loss_mean,
        pred_loss_std=pred_loss_std,
        pred_loss_min=pred_loss_min,
        max_alpha=max_alpha,
        mean_alpha=mean_alpha,
        final_alpha=final_alpha,
    )

    return score


def compute_scores(configs: list[ConfigScore], total_epochs: int = 300) -> list[ConfigScore]:
    """Compute normalized scores for all configurations.

    Priority: Stability > Everything else
    - Prediction model stability (low loss, low variance)
    - Entropy stability (high mean, low variance)
    - Hold delay (later collapse = better)
    """
    if not configs:
        return []

    # Get ranges for normalization
    max_rewards = [c.max_reward for c in configs]
    max_alphas = [c.max_alpha for c in configs]
    entropy_stds = [c.entropy_std for c in configs if c.entropy_std > 0]
    pred_stds = [c.pred_loss_std for c in configs if c.pred_loss_std > 0]
    pred_means = [c.pred_loss_mean for c in configs if c.pred_loss_mean > 0]

    max_reward_range = max(max_rewards) - min(max_rewards) if len(set(max_rewards)) > 1 else 1.0
    max_alpha_range = max(max_alphas) - min(max_alphas) if len(set(max_alphas)) > 1 else 1.0

    # For std metrics, lower is better - get max for normalization
    max_entropy_std = max(entropy_stds) if entropy_stds else 1.0
    max_pred_std = max(pred_stds) if pred_stds else 1.0
    max_pred_mean = max(pred_means) if pred_means else 1.0

    for config in configs:
        # Reward score: combination of max and mean (0-1)
        if max_reward_range > 0:
            config.reward_score = (config.max_reward - min(max_rewards)) / max_reward_range
        else:
            config.reward_score = 1.0

        # Alpha score: how much better than market (normalized)
        if max_alpha_range > 0:
            config.alpha_score = (config.max_alpha - min(max_alphas)) / max_alpha_range
        else:
            config.alpha_score = 0.5

        # ENTROPY STABILITY SCORE (key metric!)
        # Components: high mean entropy + low std (stable)
        entropy_level = min(1.0, config.mean_entropy / 4.5)  # Target ~4.5
        if max_entropy_std > 0 and config.entropy_std > 0:
            entropy_variance_penalty = config.entropy_std / max_entropy_std
        else:
            entropy_variance_penalty = 0.0
        config.entropy_stability_score = 0.6 * entropy_level + 0.4 * (1.0 - entropy_variance_penalty)

        # PREDICTION MODEL STABILITY SCORE (key metric!)
        # Components: low mean loss + low std (stable)
        if config.pred_loss_mean > 0 and max_pred_mean > 0:
            pred_level = 1.0 - (config.pred_loss_mean / max_pred_mean)  # Lower loss = higher score
        else:
            pred_level = 0.5  # Neutral if no pred data
        if max_pred_std > 0 and config.pred_loss_std > 0:
            pred_variance_penalty = config.pred_loss_std / max_pred_std
        else:
            pred_variance_penalty = 0.0
        config.pred_stability_score = 0.5 * pred_level + 0.5 * (1.0 - pred_variance_penalty)

        # Hold delay score: later collapse is better
        if config.epoch_hold_70 is None:
            config.hold_delay_score = 1.0  # Never collapsed
        else:
            config.hold_delay_score = config.epoch_hold_70 / total_epochs

        # COMPOSITE SCORE - Stability focused!
        config.composite_score = (
            0.30 * config.entropy_stability_score +  # Entropy stability (HIGH priority)
            0.25 * config.pred_stability_score +     # Prediction stability (HIGH priority)
            0.20 * config.hold_delay_score +         # Hold delay (medium priority)
            0.15 * config.reward_score +             # Reward (lower priority)
            0.10 * config.alpha_score                # Alpha (lowest priority)
        )

    return configs


def print_results(configs: list[ConfigScore], total_epochs: int = 300):
    """Print ranked results."""
    if not configs:
        print("No configurations to analyze!")
        return

    # Sort by composite score
    ranked = sorted(configs, key=lambda c: c.composite_score, reverse=True)

    print("\n" + "=" * 100)
    print("STABILITY SWEEP ANALYSIS RESULTS")
    print("=" * 100)
    print(f"Analyzed {len(configs)} configurations\n")

    # Summary table - stability focused
    print("RANKINGS (by stability-focused composite score):")
    print("-" * 130)
    print(f"{'Rank':<4} {'Config':<32} {'Score':>6} {'EntStab':>7} {'PredStab':>8} "
          f"{'EntStd':>6} {'PredStd':>7} {'HOLD>70':>8} {'MaxRwd':>7}")
    print("-" * 130)

    for i, config in enumerate(ranked, 1):
        hold_70_str = str(config.epoch_hold_70) if config.epoch_hold_70 else "never"
        name_short = config.name[:32] if len(config.name) > 32 else config.name
        print(f"{i:<4} {name_short:<32} {config.composite_score:>6.3f} "
              f"{config.entropy_stability_score:>7.3f} {config.pred_stability_score:>8.3f} "
              f"{config.entropy_std:>6.2f} {config.pred_loss_std:>7.4f} "
              f"{hold_70_str:>8} {config.max_reward:>7.1f}")

    print("-" * 100)

    # Detailed breakdown of top 5
    print("\n" + "=" * 100)
    print("TOP 5 DETAILED BREAKDOWN")
    print("=" * 100)

    for i, config in enumerate(ranked[:5], 1):
        print(f"\n#{i}: {config.name}")
        print(f"    Composite Score: {config.composite_score:.4f}")
        print(f"    ├─ Entropy Stability:  {config.entropy_stability_score:.4f} (mean={config.mean_entropy:.2f}, std={config.entropy_std:.3f})")
        print(f"    ├─ Pred Stability:     {config.pred_stability_score:.4f} (mean={config.pred_loss_mean:.4f}, std={config.pred_loss_std:.4f})")
        print(f"    ├─ Hold Delay:         {config.hold_delay_score:.4f} (HOLD>70% at epoch {config.epoch_hold_70 or 'never'})")
        print(f"    ├─ Reward Score:       {config.reward_score:.4f} (max={config.max_reward:.2f})")
        print(f"    └─ Alpha Score:        {config.alpha_score:.4f} (max={config.max_alpha:.2f}%)")
        print(f"    Final state: entropy={config.final_entropy:.2f}, pred_loss={config.pred_loss_final:.4f}, HOLD%={config.hold_pct_at_end:.1f}%")

    # Winner announcement
    winner = ranked[0]
    print("\n" + "=" * 100)
    print("RECOMMENDED CONFIGURATION")
    print("=" * 100)
    print(f"\n  >>> {winner.name} <<<\n")
    print(f"  Composite Score: {winner.composite_score:.4f}")
    print(f"  Max Reward: {winner.max_reward:.2f}")
    print(f"  HOLD>70% delayed until: epoch {winner.epoch_hold_70 or 'never'}")
    print(f"  Minimum entropy: {winner.min_entropy:.2f}")

    # Category winners - stability focused
    print("\n" + "-" * 100)
    print("CATEGORY WINNERS (Stability Focus):")

    best_entropy_stab = max(configs, key=lambda c: c.entropy_stability_score)
    print(f"  Best Entropy Stability:  {best_entropy_stab.name}")
    print(f"                           (mean={best_entropy_stab.mean_entropy:.2f}, std={best_entropy_stab.entropy_std:.3f})")

    best_pred_stab = max(configs, key=lambda c: c.pred_stability_score)
    print(f"  Best Pred Stability:     {best_pred_stab.name}")
    print(f"                           (mean_loss={best_pred_stab.pred_loss_mean:.4f}, std={best_pred_stab.pred_loss_std:.4f})")

    # Lowest entropy std (most stable entropy)
    configs_with_entropy = [c for c in configs if c.entropy_std > 0]
    if configs_with_entropy:
        lowest_ent_std = min(configs_with_entropy, key=lambda c: c.entropy_std)
        print(f"  Lowest Entropy Std:      {lowest_ent_std.name} (std={lowest_ent_std.entropy_std:.3f})")

    # Lowest pred loss std (most stable prediction)
    configs_with_pred = [c for c in configs if c.pred_loss_std > 0]
    if configs_with_pred:
        lowest_pred_std = min(configs_with_pred, key=lambda c: c.pred_loss_std)
        print(f"  Lowest Pred Loss Std:    {lowest_pred_std.name} (std={lowest_pred_std.pred_loss_std:.4f})")

    latest_collapse = max(configs, key=lambda c: c.epoch_hold_70 or total_epochs)
    collapse_str = f"epoch {latest_collapse.epoch_hold_70}" if latest_collapse.epoch_hold_70 else "never"
    print(f"  Latest HOLD Collapse:    {latest_collapse.name} (HOLD>70% at {collapse_str})")

    best_reward = max(configs, key=lambda c: c.max_reward)
    print(f"  Best Max Reward:         {best_reward.name} ({best_reward.max_reward:.2f})")

    print("=" * 100)


def export_summary(configs: list[ConfigScore], output_path: str):
    """Export summary to CSV for further analysis."""
    if not configs:
        return

    rows = []
    for config in sorted(configs, key=lambda c: c.composite_score, reverse=True):
        rows.append({
            'rank': len(rows) + 1,
            'config': config.name,
            'composite_score': config.composite_score,
            # Stability scores (key metrics)
            'entropy_stability_score': config.entropy_stability_score,
            'pred_stability_score': config.pred_stability_score,
            'hold_delay_score': config.hold_delay_score,
            'reward_score': config.reward_score,
            'alpha_score': config.alpha_score,
            # Entropy metrics
            'entropy_mean': config.mean_entropy,
            'entropy_std': config.entropy_std,
            'entropy_min': config.min_entropy,
            'entropy_final': config.final_entropy,
            # Prediction metrics
            'pred_loss_mean': config.pred_loss_mean,
            'pred_loss_std': config.pred_loss_std,
            'pred_loss_min': config.pred_loss_min,
            'pred_loss_final': config.pred_loss_final,
            # Hold metrics
            'hold_pct_mean': config.mean_hold_pct,
            'hold_pct_final': config.hold_pct_at_end,
            'epoch_hold_70': config.epoch_hold_70,
            'epoch_hold_80': config.epoch_hold_80,
            # Reward metrics
            'max_reward': config.max_reward,
            'mean_reward': config.mean_reward,
            'final_reward': config.final_reward,
            # Alpha metrics
            'max_alpha': config.max_alpha,
            'mean_alpha': config.mean_alpha,
            'final_alpha': config.final_alpha,
            # Meta
            'epochs_completed': config.epochs_completed,
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nSummary exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze stability sweep results")
    parser.add_argument("sweep_dir", nargs="?", help="Path to sweep output directory")
    parser.add_argument("--latest", action="store_true", help="Use most recent sweep directory")
    parser.add_argument("--epochs", type=int, default=300, help="Total epochs per config")
    parser.add_argument("--export", type=str, help="Export summary CSV to this path")
    parser.add_argument("--watch", action="store_true", help="Watch mode: refresh every 60s")
    parser.add_argument("--min-epochs", type=int, default=10, help="Minimum epochs to include config")
    args = parser.parse_args()

    # Find sweep directory
    if args.latest:
        sweep_dir = find_latest_sweep_dir()
        if not sweep_dir:
            print("Error: No stability_sweep directories found in logs/")
            sys.exit(1)
        print(f"Using latest sweep: {sweep_dir}")
    elif args.sweep_dir:
        sweep_dir = args.sweep_dir
    else:
        # Try latest by default
        sweep_dir = find_latest_sweep_dir()
        if not sweep_dir:
            print("Usage: python analyze_sweep.py <sweep_dir> OR --latest")
            sys.exit(1)
        print(f"Using latest sweep: {sweep_dir}")

    if not os.path.isdir(sweep_dir):
        print(f"Error: Directory not found: {sweep_dir}")
        sys.exit(1)

    def run_analysis():
        """Run one analysis pass."""
        csv_files = []

        # Strategy 1: CSVs in sweep directory
        csv_pattern = os.path.join(sweep_dir, "*.csv")
        csv_files.extend(glob.glob(csv_pattern))

        # Strategy 2: Subdirectories
        csv_pattern = os.path.join(sweep_dir, "*", "*.csv")
        csv_files.extend(glob.glob(csv_pattern))

        # Strategy 3: Look in logs/ for training_*.csv matching sweep config names
        # Parse sweep_config.json to get config names
        sweep_config_path = os.path.join(sweep_dir, "sweep_config.json")
        if os.path.exists(sweep_config_path):
            import json
            with open(sweep_config_path) as f:
                sweep_config = json.load(f)
            # Get run names from config
            timestamp = os.path.basename(sweep_dir).replace("stability_sweep_", "")
            for config_name in sweep_config.get("configs", []):
                run_name = f"sweep_{timestamp}_{config_name}"
                csv_path = f"logs/training_{run_name}.csv"
                if os.path.exists(csv_path) and csv_path not in csv_files:
                    csv_files.append(csv_path)

        # Strategy 3b: Parse sweep_config.txt for level sweeps (L1, L2, etc.)
        sweep_config_txt = os.path.join(sweep_dir, "sweep_config.txt")
        if os.path.exists(sweep_config_txt):
            with open(sweep_config_txt) as f:
                content = f.read()
            # Extract config names (lines ending with ":")
            import re
            config_names = re.findall(r'^(\w+):$', content, re.MULTILINE)
            # Determine level prefix from directory name
            dir_name = os.path.basename(sweep_dir)
            if dir_name.startswith("level"):
                level_match = re.search(r'level(\d+)', dir_name)
                if level_match:
                    level_prefix = f"L{level_match.group(1)}_"
                    for config_name in config_names:
                        csv_path = f"logs/training_{level_prefix}{config_name}.csv"
                        if os.path.exists(csv_path) and csv_path not in csv_files:
                            csv_files.append(csv_path)

        # Strategy 4: Match any training_sweep_*.csv in logs/ (only if no files found yet)
        if not csv_files:
            sweep_csvs = glob.glob("logs/training_sweep_*.csv")
            for path in sweep_csvs:
                if path not in csv_files:
                    csv_files.append(path)

        # Remove duplicates
        csv_files = list(set(csv_files))

        # Filter out analysis output files (not training runs)
        csv_files = [f for f in csv_files if 'analysis_summary' not in f]

        if not csv_files:
            print(f"No CSV files found in {sweep_dir} yet...")
            return []

        print(f"Found {len(csv_files)} CSV files to analyze...")

        # Analyze each config
        configs = []
        for csv_path in sorted(csv_files):
            config = analyze_config(csv_path, args.epochs)
            if config:
                # Filter by min-epochs
                if config.epochs_completed >= args.min_epochs:
                    configs.append(config)
                    print(f"  ✓ {config.name}: {config.epochs_completed} epochs, max_reward={config.max_reward:.2f}")
                else:
                    print(f"  ⏳ {config.name}: only {config.epochs_completed} epochs (waiting for {args.min_epochs})")
            else:
                print(f"  ✗ Could not analyze: {csv_path}")

        if not configs:
            print("No configurations with sufficient epochs yet...")
            return []

        # Compute scores
        configs = compute_scores(configs, args.epochs)

        # Print results
        print_results(configs, args.epochs)

        return configs

    # Watch mode: loop with refresh
    if args.watch:
        import time
        print(f"\n👀 Watch mode enabled - refreshing every 60 seconds (Ctrl+C to stop)\n")
        try:
            while True:
                os.system('clear' if os.name == 'posix' else 'cls')
                print(f"[{time.strftime('%H:%M:%S')}] Analyzing sweep: {sweep_dir}\n")
                configs = run_analysis()
                if configs:
                    export_path = args.export or os.path.join(sweep_dir, "analysis_summary.csv")
                    export_summary(configs, export_path)
                print(f"\n⏰ Next refresh in 60 seconds... (Ctrl+C to stop)")
                time.sleep(60)
        except KeyboardInterrupt:
            print("\n\n👋 Watch mode stopped.")
    else:
        # Single run
        configs = run_analysis()
        if configs:
            export_path = args.export or os.path.join(sweep_dir, "analysis_summary.csv")
            export_summary(configs, export_path)


if __name__ == "__main__":
    main()
