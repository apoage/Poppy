#!/usr/bin/env python3
"""Verify timestamps are consecutive in episodes"""

import sys
import os
from datetime import datetime

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

from mcts_trader.env.trading_env import TradingEnv

def verify_temporal_correlation(num_episodes=3, steps_per_episode=20):
    """Check if episode timestamps are consecutive"""
    print("="*70)
    print("Timestamp Correlation Verification")
    print("="*70)
    print(f"Running {num_episodes} episodes with {steps_per_episode} steps each")
    print()

    env = TradingEnv(n_assets=64, lookback=60)
    env.encoder.device = 'cpu'

    all_consecutive = True

    for ep in range(num_episodes):
        print(f"Episode {ep+1}/{num_episodes}:")

        obs, info = env.reset()

        # Access market timestamp
        start_time = env.market.current_time
        start_dt = datetime.fromtimestamp(start_time)

        print(f"  Start: {start_dt.strftime('%Y-%m-%d %H:%M:%S')} "
              f"({start_dt.strftime('%A')})")
        print(f"         Timestamp: {start_time}")

        timestamps = [start_time]

        # Run steps
        for step in range(steps_per_episode):
            action = 1  # Hold action (neutral)
            obs, reward, terminated, truncated, info = env.step(action)
            timestamps.append(env.market.current_time)

            if terminated or truncated:
                break

        # Check consecutiveness
        deltas = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]

        if all(d == 60 for d in deltas):
            print(f"  ✅ All {len(deltas)} timestamps consecutive (60s intervals)")
            consecutive = True
        else:
            print(f"  ❌ Timestamps NOT consecutive!")
            print(f"     Deltas (seconds): {deltas}")
            all_consecutive = False
            consecutive = False

        # Show time range
        end_time = timestamps[-1]
        end_dt = datetime.fromtimestamp(end_time)
        print(f"  End:   {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"         Timestamp: {end_time}")
        print(f"  Duration: {(end_time - start_time) / 60:.0f} minutes")

        # Show temporal features
        print(f"  Temporal features:")
        print(f"    Weekday: {start_dt.strftime('%A')} (index {start_dt.weekday()})")
        print(f"    Hour: {start_dt.hour}:00")
        print(f"    Minute: {start_dt.minute}")

        # Check if crosses day boundary
        if start_dt.day != end_dt.day:
            print(f"  ℹ️  Episode crosses day boundary")

        print()

    print("="*70)
    print("Summary")
    print("="*70)

    if all_consecutive:
        print("✅ Temporal correlation VERIFIED!")
        print("   All episodes have consecutive 60-second timestamps.")
        print()
        print("Benefits for training:")
        print("  - Model can learn temporal patterns (hour-of-day effects)")
        print("  - Adding time features will help capture:")
        print("    • Market open/close effects (volume spikes)")
        print("    • Weekday patterns (Monday volatility, Friday closes)")
        print("    • Intraday cycles (Asia/Europe/US sessions)")
    else:
        print("❌ Temporal correlation FAILED!")
        print("   Some episodes have non-consecutive timestamps.")
        print("   This may indicate a bug in HistoricalMarket.step()")

    print()
    print("Recommended next steps:")
    print("  1. Add time-of-day features to encoder (hour, weekday)")
    print("  2. Use cyclic encoding (sin/cos) to capture periodicity")
    print("  3. Verify model learns temporal patterns in training")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Verify timestamp consistency')
    parser.add_argument('--episodes', type=int, default=3,
                        help='Number of episodes to test')
    parser.add_argument('--steps', type=int, default=20,
                        help='Steps per episode')
    args = parser.parse_args()

    verify_temporal_correlation(args.episodes, args.steps)
