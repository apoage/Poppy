#!/usr/bin/env python3
"""
Paper Trading Audit using VectorBT Environment + QuantStats

Uses the SAME VectorBT-based environment as training for consistent accounting.
Generates comprehensive performance reports using QuantStats.

Usage:
    python tools/paper_trade_vectorbt.py --checkpoint jax_ckpt_fast/default/checkpoint_250 --synthetic-level 0 --steps 500
    python tools/paper_trade_vectorbt.py --checkpoint jax_ckpt_fast/default/checkpoint_250 --output-dir reports/
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import jax
import jax.numpy as jnp

# Import central config
from mcts_trader.jax_agent import VALUE_SUPPORT_SIZE

# Suppress gym deprecation warnings
import warnings
warnings.filterwarnings('ignore', message='.*Gym.*')

def run_paper_trading(args):
    """Run paper trading audit with VectorBT environment"""

    print("=" * 70)
    print("PAPER TRADING AUDIT (VectorBT)")
    print("=" * 70)
    print()

    # Import after path setup
    from mcts_trader.env.vectorbt_env import VectorBTTradingEnv
    from mcts_trader.jax_agent.muzero_model import MuZeroNetworks
    from mcts_trader.jax_agent.trainer import MuZeroTrainer
    from mcts_trader.jax_agent.mctx_search import create_mctx_search

    # Create environment
    print("[1/4] Creating VectorBT environment...")
    env = VectorBTTradingEnv(
        n_assets=args.n_assets,
        initial_cash=args.initial_cash,
        trading_fee_bps=args.fee,
        slippage_bps=args.slippage,
        max_steps=args.steps,
        synthetic_level=args.synthetic_level,
        synthetic_n_steps=args.synthetic_steps,
    )
    print(f"  Environment: {env.n_assets} assets, {args.steps} steps")

    # Load model
    print("[2/4] Loading model...")
    model = MuZeroNetworks(
        state_dim=args.state_dim,
        action_space_size=env.n_assets * 5,  # 5-action space
        support_size=VALUE_SUPPORT_SIZE,
        H_cycles=args.h_cycles,
        L_cycles=args.l_cycles,
    )

    trainer = MuZeroTrainer(
        model=model,
        learning_rate=1e-4,
        num_simulations=args.sims,
        batch_size=256,
    )

    # Initialize with dummy input
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    dummy_obs = jnp.ones((1, 27, env.n_assets, env.n_assets))
    state = trainer.init_train_state(init_rng, dummy_obs)

    # Load checkpoint
    if args.checkpoint:
        abs_path = os.path.abspath(args.checkpoint)
        print(f"  Loading checkpoint: {abs_path}")
        restored = trainer.restore_checkpoint(abs_path)
        if restored:
            state = restored
            print("  Checkpoint loaded successfully")
        else:
            print("  WARNING: Checkpoint load failed, using random weights")
    else:
        print("  No checkpoint provided, using random weights")

    # Create MCTS search
    print("[3/4] Creating MCTS search...")
    search = create_mctx_search(
        model=model,
        num_simulations=args.sims,
        use_gumbel=True,
        gumbel_scale=args.gumbel_scale,
    )

    # Run paper trading episodes
    print("[4/4] Running paper trading episodes...")
    print()

    all_episode_data = []

    for episode in range(args.episodes):
        print(f"Episode {episode + 1}/{args.episodes}")

        obs_dict, _ = env.reset()
        episode_returns = []
        episode_values = [env.initial_cash]
        actions_taken = {'buy': 0, 'hold': 0, 'sell': 0}

        for step in range(args.steps):
            # Get observation
            obs = obs_dict['observation']
            action_mask = obs_dict['action_mask']

            # Model inference with MCTS
            obs_batch = jnp.expand_dims(obs, 0)
            action_mask_batch = jnp.expand_dims(action_mask, 0)

            rng, search_rng = jax.random.split(rng)
            # search() returns (actions, policy_probs, stats)
            action_jax, policy_jax, stats = search(
                state.params,
                search_rng,
                obs_batch,
                action_mask_batch
            )

            # Get action from MCTS (already selected best action)
            action = int(action_jax[0])

            # Track action type
            action_type = action % 3
            if action_type == 0:
                actions_taken['sell'] += 1
            elif action_type == 1:
                actions_taken['hold'] += 1
            else:
                actions_taken['buy'] += 1

            # Step environment
            obs_dict, reward, terminated, truncated, info = env.step(action)
            episode_returns.append(reward / 100)  # Convert percentage to decimal
            episode_values.append(info['portfolio_value'])

            if terminated or truncated:
                break

        # Episode summary
        final_stats = env.get_portfolio_stats()
        total_return = final_stats['return_pct']
        sharpe_approx = np.mean(episode_returns) / (np.std(episode_returns) + 1e-8) * np.sqrt(252 * 24 * 60)

        print(f"  Return: {total_return:+.2f}% | Sharpe (approx): {sharpe_approx:.2f}")
        print(f"  Actions: BUY={actions_taken['buy']}, HOLD={actions_taken['hold']}, SELL={actions_taken['sell']}")
        print(f"  Final positions: {final_stats['n_positions']}")
        print()

        # Store episode data
        all_episode_data.append({
            'episode': episode + 1,
            'return_pct': total_return,
            'sharpe': sharpe_approx,
            'n_positions': final_stats['n_positions'],
            'returns': episode_returns,
            'values': episode_values,
            'actions': actions_taken.copy(),
        })

    # Generate report
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    returns_all = [ep['return_pct'] for ep in all_episode_data]
    print(f"Episodes: {len(returns_all)}")
    print(f"Avg Return: {np.mean(returns_all):+.2f}%")
    print(f"Std Return: {np.std(returns_all):.2f}%")
    print(f"Best Episode: {max(returns_all):+.2f}%")
    print(f"Worst Episode: {min(returns_all):+.2f}%")
    print(f"Win Rate: {sum(1 for r in returns_all if r > 0) / len(returns_all) * 100:.1f}%")

    # Generate QuantStats report if multiple episodes
    if args.generate_report and len(all_episode_data) > 0:
        generate_quantstats_report(all_episode_data, args)

    return all_episode_data


def generate_quantstats_report(episode_data, args):
    """Generate QuantStats HTML report"""

    try:
        import quantstats as qs
    except ImportError:
        print("\nQuantStats not installed. Run: pip install quantstats")
        return

    print()
    print("Generating QuantStats report...")

    # Combine all episode returns into a single series
    all_returns = []
    timestamps = []
    base_time = datetime.now()

    for ep in episode_data:
        for i, ret in enumerate(ep['returns']):
            timestamps.append(base_time + timedelta(minutes=i + len(all_returns)))
            all_returns.append(ret)

    returns_series = pd.Series(all_returns, index=pd.DatetimeIndex(timestamps))

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate HTML report
    report_path = os.path.join(args.output_dir, f"paper_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")

    try:
        qs.reports.html(
            returns_series,
            output=report_path,
            title=f"Paper Trading Audit (VectorBT)",
        )
        print(f"Report saved to: {report_path}")
    except Exception as e:
        print(f"Warning: Could not generate HTML report: {e}")

    # Print key metrics (handle Series or scalar return values)
    def to_scalar(val):
        if hasattr(val, 'item'):
            return val.item()
        if hasattr(val, 'iloc'):
            return val.iloc[0]
        return val

    print()
    print("Key Metrics:")
    print(f"  Total Return: {(1 + returns_series).prod() - 1:.2%}")
    print(f"  Sharpe Ratio: {to_scalar(qs.stats.sharpe(returns_series)):.2f}")
    print(f"  Sortino Ratio: {to_scalar(qs.stats.sortino(returns_series)):.2f}")
    print(f"  Max Drawdown: {to_scalar(qs.stats.max_drawdown(returns_series)):.2%}")
    print(f"  Win Rate: {to_scalar(qs.stats.win_rate(returns_series)):.2%}")
    print(f"  Volatility: {to_scalar(qs.stats.volatility(returns_series)):.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Paper Trading Audit (VectorBT)')

    # Required
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')

    # Environment
    parser.add_argument('--n-assets', type=int, default=64, help='Number of assets')
    parser.add_argument('--initial-cash', type=float, default=10000, help='Initial cash')
    parser.add_argument('--fee', type=int, default=26, help='Trading fee in bps')
    parser.add_argument('--slippage', type=int, default=5, help='Slippage in bps')
    parser.add_argument('--steps', type=int, default=500, help='Steps per episode')
    parser.add_argument('--synthetic-level', type=int, default=0, help='Synthetic data level (0-3)')
    parser.add_argument('--synthetic-steps', type=int, default=10000, help='Synthetic data length')

    # Model
    parser.add_argument('--state-dim', type=int, default=64, help='Model state dimension')
    parser.add_argument('--h-cycles', type=int, default=4, help='TRM H cycles')
    parser.add_argument('--l-cycles', type=int, default=6, help='TRM L cycles')
    parser.add_argument('--sims', type=int, default=32, help='MCTS simulations')
    parser.add_argument('--gumbel-scale', type=float, default=3.0, help='Gumbel scale')

    # Audit
    parser.add_argument('--episodes', type=int, default=3, help='Number of episodes to run')
    parser.add_argument('--generate-report', action='store_true', help='Generate QuantStats HTML report')
    parser.add_argument('--output-dir', type=str, default='reports', help='Output directory for reports')

    args = parser.parse_args()

    run_paper_trading(args)
