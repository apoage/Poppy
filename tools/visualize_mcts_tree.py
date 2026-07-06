#!/usr/bin/env python3
"""
MCTS Tree Visualization Tool

Captures MCTS search tree statistics during self-play and generates:
1. Tree depth distribution (how far MCTS explores)
2. Node visit counts (which branches are explored most)
3. Action selection heatmap (which assets/actions are favored)
4. Value propagation graph (how values change through search)

Usage:
    python tools/visualize_mcts_tree.py logs/training_dojo_dynamic_L0_sine.csv
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_policy_distribution(df):
    """Plot sell/hold/buy probability evolution over epochs"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Action counts over time
    ax = axes[0, 0]
    ax.plot(df['epoch'], df['n_sell'], label='Sells', color='red', linewidth=2)
    ax.plot(df['epoch'], df['n_hold'], label='Holds', color='gray', linewidth=2)
    ax.plot(df['epoch'], df['n_buy'], label='Buys', color='green', linewidth=2)
    ax.axhline(y=133, color='blue', linestyle='--', alpha=0.5, label='Equal (33% each)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Action Count (per 400 steps)')
    ax.set_title('Action Distribution Evolution')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: Policy entropy over time
    ax = axes[0, 1]
    ax.plot(df['epoch'], df['policy_entropy'], color='purple', linewidth=2)
    ax.axhline(y=4.0, color='green', linestyle='--', alpha=0.5, label='High Exploration')
    ax.axhline(y=2.5, color='orange', linestyle='--', alpha=0.5, label='Balanced')
    ax.axhline(y=1.5, color='red', linestyle='--', alpha=0.5, label='Deterministic')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Policy Entropy')
    ax.set_title('Exploration vs Exploitation')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 3: Value calibration (predicted vs actual)
    ax = axes[1, 0]
    ax.scatter(df['reward'], df['avg_value'], alpha=0.5, s=20)

    # Perfect calibration line
    min_val = min(df['reward'].min(), df['avg_value'].min())
    max_val = max(df['reward'].max(), df['avg_value'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Calibration')

    ax.set_xlabel('Actual Reward (%)')
    ax.set_ylabel('Predicted Value (%)')
    ax.set_title('Value Head Calibration')
    ax.legend()
    ax.grid(alpha=0.3)

    # Add correlation coefficient
    corr = df['reward'].corr(df['avg_value'])
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 4: Reward and Entropy correlation
    ax = axes[1, 1]
    scatter = ax.scatter(df['policy_entropy'], df['n_sell'],
                        c=df['reward'], cmap='RdYlGn', alpha=0.6, s=40)
    ax.set_xlabel('Policy Entropy (Exploration)')
    ax.set_ylabel('Sell Count')
    ax.set_title('Exploration vs Sells (colored by Reward)')
    ax.grid(alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Reward (%)', rotation=270, labelpad=20)

    plt.tight_layout()
    return fig


def plot_learning_curves(df):
    """Plot loss components and reward over time"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Plot 1: Losses
    ax = axes[0]
    ax.plot(df['epoch'], df['total_loss'], label='Total Loss', color='black', linewidth=2)
    ax.plot(df['epoch'], df['value_loss'], label='Value Loss', color='blue', alpha=0.7)
    ax.plot(df['epoch'], df['policy_loss'], label='Policy Loss', color='orange', alpha=0.7)
    ax.plot(df['epoch'], df['reward_loss'], label='Reward Loss', color='green', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Components')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: Reward with moving average
    ax = axes[1]
    ax.plot(df['epoch'], df['reward'], color='steelblue', alpha=0.5, label='Episode Reward')

    # Moving average
    window = min(50, len(df) // 5)
    if window > 0:
        ma = df['reward'].rolling(window=window, min_periods=1).mean()
        ax.plot(df['epoch'], ma, color='darkblue', linewidth=2.5, label=f'MA({window})')

    # Mastery threshold
    ax.axhline(y=2.5, color='green', linestyle='--', linewidth=2, label='Mastery (+2.5%)')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reward (%)')
    ax.set_title('Episode Reward Over Time')
    ax.legend()
    ax.grid(alpha=0.3)

    # Color zones
    ax.axhspan(-10, 0, alpha=0.1, color='red')
    ax.axhspan(0, 2.5, alpha=0.1, color='yellow')
    ax.axhspan(2.5, 10, alpha=0.1, color='green')

    plt.tight_layout()
    return fig


def plot_action_balance(df):
    """Plot action distribution as stacked area chart"""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Normalize to percentages
    total_actions = df['n_buy'] + df['n_hold'] + df['n_sell']
    buy_pct = (df['n_buy'] / total_actions) * 100
    hold_pct = (df['n_hold'] / total_actions) * 100
    sell_pct = (df['n_sell'] / total_actions) * 100

    # Stacked area
    ax.fill_between(df['epoch'], 0, sell_pct, color='red', alpha=0.6, label='Sell')
    ax.fill_between(df['epoch'], sell_pct, sell_pct + hold_pct, color='gray', alpha=0.6, label='Hold')
    ax.fill_between(df['epoch'], sell_pct + hold_pct, 100, color='green', alpha=0.6, label='Buy')

    # Reference line for balanced policy (33% each)
    ax.axhline(y=33.33, color='blue', linestyle='--', linewidth=2, alpha=0.7, label='Balanced (33% each)')
    ax.axhline(y=66.67, color='blue', linestyle='--', linewidth=2, alpha=0.7)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Action Distribution (%)')
    ax.set_title('Policy Balance Over Training')
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def analyze_phase_transitions(df):
    """Detect when policy shifts significantly"""
    print("\n" + "="*70)
    print("PHASE TRANSITION ANALYSIS")
    print("="*70)

    # Detect when sells start appearing
    if (df['n_sell'] == 0).any():
        first_sell_epoch = df[df['n_sell'] > 0]['epoch'].min()
        print(f"\n✓ First sells appeared at epoch: {first_sell_epoch}")

    # Detect entropy collapse
    if len(df) > 10:
        recent_entropy = df.tail(10)['policy_entropy'].mean()
        early_entropy = df.head(10)['policy_entropy'].mean()
        entropy_change = ((recent_entropy - early_entropy) / early_entropy) * 100

        print(f"\n✓ Entropy change: {entropy_change:+.1f}%")
        print(f"  - Early epochs (1-10): {early_entropy:.2f}")
        print(f"  - Recent epochs: {recent_entropy:.2f}")

        if entropy_change < -20:
            print("  ⚠️  Policy becoming deterministic (entropy dropped >20%)")
        elif entropy_change > 20:
            print("  ⚠️  Policy becoming more random (entropy increased >20%)")

    # Detect value-reward calibration
    if len(df) > 0:
        corr = df['reward'].corr(df['avg_value'])
        print(f"\n✓ Value-Reward Correlation: {corr:.3f}")

        if corr < 0.5:
            print("  ⚠️  Poor calibration (correlation < 0.5) - value head struggling")
        elif corr > 0.8:
            print("  ✓ Good calibration (correlation > 0.8) - value head learning")

    # Action balance trend
    if len(df) > 20:
        recent_sells = df.tail(20)['n_sell'].mean()
        recent_buys = df.tail(20)['n_buy'].mean()
        recent_holds = df.tail(20)['n_hold'].mean()

        print(f"\n✓ Recent action distribution (last 20 epochs):")
        print(f"  - Sells:  {recent_sells:.1f} ({recent_sells/4:.1f}%)")
        print(f"  - Holds:  {recent_holds:.1f} ({recent_holds/4:.1f}%)")
        print(f"  - Buys:   {recent_buys:.1f} ({recent_buys/4:.1f}%)")

        if recent_sells < 40:
            print("  ⚠️  Low sell rate (<10%) - may need stronger incentives")
        if recent_buys > 160:
            print("  ⚠️  High buy rate (>40%) - accumulating too many positions")

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description='Visualize MCTS training metrics')
    parser.add_argument('csv_path', type=str, help='Path to training CSV log')
    parser.add_argument('--save', action='store_true', help='Save plots to PNG files')
    parser.add_argument('--output-dir', type=str, default='plots', help='Output directory for plots')
    args = parser.parse_args()

    # Load data
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} epochs from {csv_path}")

    # Run analysis
    analyze_phase_transitions(df)

    # Generate plots
    print("\nGenerating plots...")

    fig1 = plot_policy_distribution(df)
    fig2 = plot_learning_curves(df)
    fig3 = plot_action_balance(df)

    if args.save:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)

        experiment_name = csv_path.stem.replace('training_', '')

        fig1.savefig(output_dir / f'{experiment_name}_policy_distribution.png', dpi=150, bbox_inches='tight')
        fig2.savefig(output_dir / f'{experiment_name}_learning_curves.png', dpi=150, bbox_inches='tight')
        fig3.savefig(output_dir / f'{experiment_name}_action_balance.png', dpi=150, bbox_inches='tight')

        print(f"\n✓ Plots saved to {output_dir}/")
    else:
        plt.show()


if __name__ == '__main__':
    main()
