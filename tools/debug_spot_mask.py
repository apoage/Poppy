
import sys
import os
import numpy as np
import logging

# Ensure project root is in path
sys.path.insert(0, os.getcwd())

from mcts_trader.agent.portfolio import PortfolioManager, Position

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_spot_mask():
    logging.info("--- Testing PortfolioManager Spot Mode Mask ---")
    
    # 1. Initialize in SPOT mode
    pm = PortfolioManager(
        initial_capital=1000.0,
        mode='SPOT',
        n_assets=10 # Small number for easier debugging
    )
    
    # 2. Add a dummy position (Asset 5)
    # Simulate buying Asset 5
    pm.state.positions[5] = Position(
        asset_idx=5,
        symbol="TEST/USD",
        side="long",
        size=10.0,
        entry_price=10.0,
        current_price=10.0 # Set current price
    )
    logging.info(f"Added Long Position for Asset 5. Cash: ${pm.state.cash}")
    
    # 2b. CRITICAL: Update prices so mask logic works
    prices = {5: 10.0, 6: 10.0}
    pm.update_prices(prices)
    logging.info("Updated portfolio prices.")
    
    # 3. Get Mask
    mask = pm.get_valid_actions()
    
    # 4. Verify
    # Actions: 0=Sell, 1=Hold, 2=Buy
    
    # Asset 5 (Held Long): Should allow Sell (0), Hold (1), Buy (2)
    asset_5_sell = mask[5, 0]
    asset_5_buy = mask[5, 2]
    
    logging.info(f"Asset 5 (Held): Sell Allowed? {asset_5_sell} [EXPECTED: True]")
    logging.info(f"Asset 5 (Held): Buy Allowed?  {asset_5_buy}  [EXPECTED: True]")
    
    if not asset_5_sell:
        logging.error("FAILURE: Cannot sell held asset in Spot mode!")
    else:
        logging.info("SUCCESS: Held asset can be sold.")

    # Asset 6 (Empty): Should NOT allow Sell (0), Should allow Buy (2)
    asset_6_sell = mask[6, 0]
    asset_6_buy = mask[6, 2]
    
    logging.info(f"Asset 6 (Empty): Sell Allowed? {asset_6_sell} [EXPECTED: False]")
    logging.info(f"Asset 6 (Empty): Buy Allowed?  {asset_6_buy}  [EXPECTED: True]")
    
    if asset_6_sell:
        logging.error("FAILURE: Can sell empty asset in Spot mode (Shorting enabled?)")
    else:
        logging.info("SUCCESS: Empty asset cannot be sold.")

    # 5. Test Cash Limit (optional check logic, though mask doesn't check cash amount explicitly yet unless we added it)
    # The mask checks 'drawdown' and 'max_position_pct', but not strictly 'cash > 0' for mask generation
    # (that logic is in execute_actions). 
    # But wait, did I add cash check to mask? No, I added it to execute_actions.
    
    return

if __name__ == "__main__":
    test_spot_mask()
