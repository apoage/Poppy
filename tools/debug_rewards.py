import numpy as np
import jax.numpy as jnp
import time
import os
import logging
from typing import Dict
from mcts_trader.env.trading_env import TradingEnv
from mcts_trader.agent.portfolio import PortfolioManager # To inspect defaults
import jax

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def debug_rewards_analysis(
    db_path: str = "trading_data.db",
    n_assets: int = 64,
    n_actions_per_asset: int = 3,
    initial_capital: float = 10000.0,
    max_steps: int = 100
):
    """
    Runs a simple environment simulation, printing detailed portfolio and reward
    information at each step to diagnose the constant 10.0 reward issue.
    """
    print("="*60)
    print("DEBUGGING REWARD CALCULATION")
    print("="*60)

    # Initialize environment with encoder forced to CPU
    env = TradingEnv(db_path=db_path, n_assets=n_assets, n_actions_per_asset=n_actions_per_asset, initial_capital=initial_capital)
    env.encoder.device = 'cpu'
    print("Environment initialized (Encoder forced to CPU).")
    
    # Let's override default portfolio manager settings for debugging
    # Reduce max_position_pct to make rewards smaller
    env.portfolio_manager_cls = lambda **kwargs: PortfolioManager(
        max_position_pct=0.01, # 1% max per position
        max_total_exposure=0.5, # 50% total exposure
        initial_capital=initial_capital,
        n_assets=n_assets
    )

    obs_dict, info = env.reset()
    portfolio = env.portfolio # Get access to the portfolio manager

    print(f"\n--- Initial State ---")
    print(f"Time: {env.market.current_time}")
    print(f"Cash: {portfolio.state.cash:.2f}, Total Value: {portfolio.state.total_value:.2f}")
    print(f"Initial Max Position %: {portfolio.max_position_pct}")
    print(f"Initial max_total_exposure: {portfolio.max_total_exposure}")


    # Simple strategy: Alternate between a single Buy and a single Sell
    # Or just random actions
    rng = jax.random.PRNGKey(0)

    for step_idx in range(max_steps):
        rng, action_rng = jax.random.split(rng)
        
        # Pick a random asset and random action (Buy/Sell/Hold)
        asset_idx = jax.random.randint(action_rng, (), 0, n_assets).item()
        action_type = jax.random.randint(action_rng, (), 0, n_actions_per_asset).item() # 0:Sell, 1:Hold, 2:Buy

        # Override action_type to force some actions
        # For first few steps, let's try to buy.
        if step_idx == 0:
            action_type = 2 # Force Buy
        elif step_idx == 1:
            action_type = 1 # Hold
        elif step_idx == 2:
            action_type = 0 # Force Sell

        action = asset_idx * n_actions_per_asset + action_type
        
        prev_value = portfolio.state.total_value

        next_obs_dict, reward, terminated, truncated, info = env.step(action)
        
        new_value = portfolio.state.total_value

        print(f"\n--- Step {step_idx+1} ---")
        print(f"Time: {env.market.current_time}")
        current_prices_for_step = env.market.get_current_data()
        print(f"Market prices for step: {current_prices_for_step.keys()}")
        print(f"Action: Asset {asset_idx}, Type {['Sell', 'Hold', 'Buy'][action_type]}")
        print(f"Prev Value: {prev_value:.2f}, New Value: {new_value:.2f}")
        print(f"Portfolio Change: {(new_value - prev_value):.2f}")
        print(f"Calculated Reward: {reward:.2f} (Clipped: {np.clip(reward, -10, 10):.2f})")
        print(f"Cash: {portfolio.state.cash:.2f}")
        print(f"Positions: {len(portfolio.state.positions)}")
        for pos_id, pos in portfolio.state.positions.items():
            print(f"  - {pos.symbol}: {pos.side} {pos.size:.2f} @ {pos.entry_price:.2f} (Current: {pos.current_price:.2f}, PnL: {pos.pnl:.2f})")

        if terminated or truncated:
            print(f"Episode terminated/truncated at step {step_idx+1}.")
            break

    env.close()
    print("\nReward debugging complete.")

if __name__ == "__main__":
    debug_rewards_analysis()