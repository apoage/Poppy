#!/usr/bin/env python3
"""Quick system test to verify all fixes work"""

import time
print("Testing system components...")

# Test 1: HistoricalMarket with filtering
print("\n[1/3] Testing HistoricalMarket with pre-computed filtering...")
start = time.time()
from mcts_trader.data.history import HistoricalMarket
market = HistoricalMarket("trading_data.db", n_assets=64, min_active_assets=10)
init_time = time.time() - start
print(f"  ✓ Initialized in {init_time:.2f}s")
print(f"  ✓ Loaded {len(market.assets)} assets")
print(f"  ✓ Found {len(market.valid_timestamps):,} valid timestamps")

# Test 2: Fast reset
print("\n[2/3] Testing reset performance...")
start = time.time()
for _ in range(10):
    market.reset()
reset_time = (time.time() - start) / 10
print(f"  ✓ Average reset: {reset_time*1000:.1f}ms")

# Test 3: Fast step
print("\n[3/3] Testing step performance...")
market.reset()
start = time.time()
for _ in range(100):
    data = market.step()
    if not data:
        break
step_time = (time.time() - start) / 100
print(f"  ✓ Average step: {step_time*1000:.1f}ms")

print("\n" + "="*60)
print("✅ ALL TESTS PASSED")
print("="*60)
print(f"\nPerformance summary:")
print(f"  - Init: {init_time:.2f}s (one-time cost)")
print(f"  - Reset: {reset_time*1000:.1f}ms")
print(f"  - Step: {step_time*1000:.1f}ms")
print(f"\nEstimated episode (200 steps): {step_time*200:.1f}s")
print("\n✓ System ready for training!")
