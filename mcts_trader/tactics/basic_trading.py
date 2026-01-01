"""
Basic Trading Tactic: Buy/Sell/Hold via Cash Intermediate

Action Space: n_assets × 3 (Sell, Hold, Buy)
"""

import numpy as np
import logging
from typing import Tuple, Dict, Any


class BasicTrading:
    """
    Basic trading through cash intermediate.
    All trades: Asset ↔ Cash
    """

    @staticmethod
    def get_action_space_size(n_assets: int) -> int:
        """
        Returns:
            Total actions for this tactic (n_assets × 3)
        """
        return n_assets * 3

    @staticmethod
    def decode_action(action: int, n_assets: int, n_actions_per_asset: int = 3) -> Tuple[int, int]:
        """
        Decode action into (asset_idx, action_type).

        Args:
            action: Discrete action [0, n_assets*3)
            n_assets: Number of tradeable assets
            n_actions_per_asset: Actions per asset (3: Sell/Hold/Buy)

        Returns:
            (asset_idx, action_type)
            - asset_idx: [0, n_assets)
            - action_type: 0=Sell, 1=Hold, 2=Buy
        """
        asset_idx = action // n_actions_per_asset
        action_type = action % n_actions_per_asset
        return asset_idx, action_type

    @staticmethod
    def execute(env, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute basic trading action.

        Args:
            env: Trading environment (FastTradingEnv)
            action: Discrete action index

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # Decode action
        asset_idx, action_type = BasicTrading.decode_action(
            action, env.n_assets, env.n_actions_per_asset
        )

        # Get current portfolio value (for reward calculation)
        prev_value = env.portfolio.state.total_value
        terminated = False

        # Get current market data
        current_data = env.market.get_current_data()

        # Build actions array (hold for all except selected asset)
        actions = np.ones(env.n_assets, dtype=np.int32)  # Default: Hold
        sizes = np.full(env.n_assets, 0.5, dtype=np.float32)  # Position size

        # Set action for selected asset
        actions[asset_idx] = action_type  # 0=Sell, 1=Hold, 2=Buy

        # Build price map (asset_idx → price)
        current_prices_by_idx = {}
        fx_rates = {}

        for symbol, data in current_data.items():
            price = data['close']

            # Map Kraken FX symbols to standard rates
            if hasattr(env, 'fx_map') and symbol in env.fx_map:
                std_pair = env.fx_map[symbol]

                if std_pair == 'EUR/USD':
                    fx_rates['EUR/USD'] = price
                elif std_pair == 'GBP/USD':
                    fx_rates['GBP/USD'] = price
                elif std_pair == 'AUD/USD':
                    fx_rates['AUD/USD'] = price
                elif std_pair == 'USD/CAD':
                    fx_rates['CAD/USD'] = 1.0 / price if price > 0 else 0
                elif std_pair == 'USD/CHF':
                    fx_rates['CHF/USD'] = 1.0 / price if price > 0 else 0
                elif std_pair == 'USD/JPY':
                    fx_rates['JPY/USD'] = 1.0 / price if price > 0 else 0

            # Store symbol prices (for FX conversions)
            if hasattr(env, '_normalize_kraken_symbol'):
                norm_sym = env._normalize_kraken_symbol(symbol)
                fx_rates[norm_sym] = price
                if norm_sym != symbol:
                    fx_rates[symbol] = price

        # Map prices to asset indices
        if hasattr(env, 'action_idx_to_symbol'):
            for idx, sym in env.action_idx_to_symbol.items():
                if sym in current_data:
                    current_prices_by_idx[idx] = current_data[sym]['close']

        # Execute via Portfolio Manager
        env.portfolio.execute_actions(
            actions=actions,
            sizes=sizes,
            prices=current_prices_by_idx,
            fx_rates=fx_rates
        )

        # Advance market
        env.market.step()

        # Update portfolio value
        new_value = env.portfolio.state.total_value

        # Calculate reward
        reward = (new_value - prev_value) / prev_value * 100
        reward = np.clip(reward, -10, 10)

        # Check episode termination
        env.steps_taken += 1
        truncated = env.steps_taken >= env.max_steps

        return env._get_observation(), reward, terminated, truncated, {}

    @staticmethod
    def get_action_mask(env) -> np.ndarray:
        """
        Get valid action mask for basic trading.

        Args:
            env: Trading environment

        Returns:
            Boolean mask (n_assets * 3,)
        """
        if env.portfolio:
            mask = env.portfolio.get_valid_actions().flatten().astype(np.int8)
        else:
            mask = np.ones(env.n_assets * 3, dtype=np.int8)

        return mask
