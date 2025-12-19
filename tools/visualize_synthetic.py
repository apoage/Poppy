import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mcts_trader.data.synthetic_market import SyntheticMarket

def visualize_level(level: int):
    print(f"Generating visualization for Level {level}...")
    market = SyntheticMarket(level=level, n_assets=64, n_steps=1000)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i in range(4):
        # Pick different assets to show variety
        # For Level 1 (Lagged), show a leader (0) and follower (32)
        if level == 1:
             indices = [0, 32, 1, 33]
             idx = indices[i]
        else:
             idx = i * 4 # spread them out
             
        symbol = f'SYN_{idx:02d}'
        if symbol not in market.data:
            symbol = list(market.data.keys())[i] # Fallback
            
        prices = market.data[symbol]['close']

        axes[i].plot(prices[:200], linewidth=1.5)
        axes[i].set_title(f'{symbol} (Level {level})')
        axes[i].set_xlabel('Step')
        axes[i].set_ylabel('Price')
        axes[i].grid(True, alpha=0.3)

    plt.suptitle(f'Synthetic Level {level} Preview', fontsize=16, fontweight='bold')
    plt.tight_layout()
    output_file = f'synthetic_level_{level}_preview.png'
    plt.savefig(output_file, dpi=150)
    print(f"Saved: {output_file}")

if __name__ == "__main__":
    for level in [0, 1, 2, 3]:
        visualize_level(level)
