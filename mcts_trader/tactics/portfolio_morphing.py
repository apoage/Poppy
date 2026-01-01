"""
Portfolio Morphing Tactic: Direct Asset→Asset Swaps

Action Space: n_assets × n_assets (all possible swaps)
Benefit: 50% fee reduction (1 swap vs 2 trades through cash)
"""

import numpy as np
import logging
from typing import Tuple, Dict, Any
from mcts_trader.agent.portfolio import Position


class PortfolioMorphing:
    """
    Direct asset-to-asset swaps (portfolio morphing).

    Example: BTC → ETH direct (1 fee) vs BTC → USD → ETH (2 fees)
    """

    @staticmethod
    def get_action_space_size(n_assets: int) -> int:
        """
        Returns:
            Total swap actions (n_assets × n_assets)
        """
        return n_assets * n_assets

    @staticmethod
    def decode_swap(swap_action: int, n_assets: int, base_action_count: int) -> Tuple[int, int]:
        """
        Decode swap action into (source_asset, target_asset).

        Args:
            swap_action: Action index from combined action space
            n_assets: Number of assets
            base_action_count: Offset (basic actions count)

        Returns:
            (source_asset, target_asset) indices
        """
        swap_idx = swap_action - base_action_count
        source_asset = swap_idx // n_assets
        target_asset = swap_idx % n_assets
        return source_asset, target_asset

    @staticmethod
    def execute_swap(env, source_asset: int, target_asset: int):
        """
        Execute direct asset swap (source → target).

        Args:
            env: Trading environment
            source_asset: Index of asset to sell
            target_asset: Index of asset to buy
        """
        if not env.portfolio:
            return

        # Get symbols
        source_symbol = env.padded_symbols[source_asset]
        target_symbol = env.padded_symbols[target_asset]
        current_data = env.market.get_current_data()

        # Validate symbols exist in market
        if source_symbol not in current_data or target_symbol not in current_data:
            logging.warning(f"Swap invalid: {source_symbol} or {target_symbol} not in market data")
            return

        source_price = current_data[source_symbol]['close']
        target_price = current_data[target_symbol]['close']

        # Check if source position exists
        if source_asset not in env.portfolio.state.positions:
            logging.debug(f"Swap skipped: No position in {source_symbol}")
            return

        # FIX: Use .size instead of .shares
        source_shares = env.portfolio.state.positions[source_asset].size
        if source_shares <= 0:
            logging.debug(f"Swap skipped: Zero shares in {source_symbol}")
            return

        # Calculate swap value (single fee)
        source_value = source_shares * source_price
        fee_rate = env.trading_fee_bps / 10000.0
        net_value = source_value * (1 - fee_rate)
        target_shares = net_value / target_price

        # Update Portfolio State
        # Close source position
        del env.portfolio.state.positions[source_asset]

        # Update or create target position
        if target_asset in env.portfolio.state.positions:
            # FIX: Use .size instead of .shares
            env.portfolio.state.positions[target_asset].size += target_shares
            env.portfolio.state.positions[target_asset].current_price = target_price
            # Note: Cost basis update is complex (weighted average)
            # For simplicity, we keep existing entry_price
        else:
            # Create new position
            # FIX: Use correct Position fields (asset_idx, size, not asset_id, shares)
            env.portfolio.state.positions[target_asset] = Position(
                asset_idx=target_asset,
                symbol=target_symbol,
                side='long',
                size=target_shares,
                entry_price=target_price,
                current_price=target_price
            )

        logging.debug(
            f"SWAP: {source_symbol}({source_shares:.4f}) → {target_symbol}({target_shares:.4f}) "
            f"| Fee: {fee_rate*100:.2f}%"
        )

    @staticmethod
    def execute(env, swap_action: int, base_action_count: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute swap action.

        Args:
            env: Trading environment
            swap_action: Swap action index (from combined action space)
            base_action_count: Offset for swap actions

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # Get current portfolio value
        prev_value = env.portfolio.state.total_value
        terminated = False

        # Decode swap
        source_asset, target_asset = PortfolioMorphing.decode_swap(
            swap_action, env.n_assets, base_action_count
        )

        # Execute swap
        PortfolioMorphing.execute_swap(env, source_asset, target_asset)

        # Advance market
        env.market.step()

        # Update portfolio value
        current_data = env.market.get_current_data()
        env.portfolio.update_from_market(current_data)
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
        Get valid swap action mask.

        Returns:
            Boolean mask (n_assets × n_assets,)
            - True if swap valid (source held, target exists)
            - False otherwise
        """
        mask = np.zeros(env.n_assets * env.n_assets, dtype=np.int8)

        if not env.portfolio:
            return mask

        current_data = env.market.get_current_data()

        # Iterate held positions
        for i, pos in env.portfolio.state.positions.items():
            # FIX: Use .size instead of .shares
            if pos.size > 1e-6:
                base_idx = i * env.n_assets
                for j in range(env.n_assets):
                    if i == j:
                        continue  # Can't swap to self

                    # Check if target exists in market
                    target_symbol = env.padded_symbols[j]
                    if target_symbol in current_data:
                        mask[base_idx + j] = 1

        return mask
