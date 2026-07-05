#!/usr/bin/env python3
"""Debug why step() returns empty data"""

from mcts_trader.data.history import HistoricalMarket

print("Testing HistoricalMarket step behavior...\n")

market = HistoricalMarket("trading_data.db", n_assets=64, min_required_assets=10)

print(f"Total valid timestamps: {len(market.valid_timestamps):,}")
print(f"Time range: {market.min_time} - {market.max_time}\n")

# Reset
print("Resetting with lookahead=300...")
data = market.reset(lookahead_steps=300)
print(f"After reset: current_time={market.current_time}, got {len(data)} assets\n")

# Find current index
current_idx = None
for i, ts in enumerate(market.valid_timestamps):
    if ts == market.current_time:
        current_idx = i
        break

if current_idx is not None:
    remaining = len(market.valid_timestamps) - current_idx
    print(f"Current index: {current_idx:,} / {len(market.valid_timestamps):,}")
    print(f"Remaining timestamps: {remaining:,}\n")

# Test stepping
print("Testing 10 steps:")
for i in range(10):
    next_data = market.step()
    print(f"  Step {i+1}: current_time={market.current_time}, got {len(next_data)} assets")
    if not next_data:
        print(f"  ❌ Empty data on step {i+1}!")
        break

print("\n✅ Debug complete")
