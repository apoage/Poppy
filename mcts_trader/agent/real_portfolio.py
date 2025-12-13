import logging
import time
import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from mcts_trader.data.kraken_client import KrakenClient
from mcts_trader.agent.portfolio import PortfolioState, Position, extract_quote_currency

@dataclass
class PendingOrder:
    order_id: str
    asset_idx: int
    side: str  # 'buy' or 'sell'
    order_type: str # 'limit' or 'market'
    price: float
    size: float
    timestamp: float
    confidence: float

@dataclass
class StopLossOrder:
    """Tracks exchange-level stop-loss orders"""
    order_id: str
    asset_idx: int
    symbol: str
    entry_side: str  # 'buy' (long) or 'sell' (short)
    entry_price: float
    stop_price: float
    volume: float
    timestamp: float

class RealPortfolioManager:
    """
    Manages real trading operations via Kraken API.
    Acts as an adapter between the Agent and the Exchange.
    """
    def __init__(
        self,
        api_client: KrakenClient,
        symbols: Dict[int, str], # ID -> Symbol map
        mode: str = 'SPOT',      # 'SPOT' or 'MARGIN'
        leverage: int = 1,       # Leverage for margin mode (1 = none/spot-margin)
        market_threshold: float = 0.8, # Confidence threshold for market orders
        limit_timeout: int = 60, # Seconds before cancelling unfilled limit order
        max_drawdown: float = 0.0, # 0 = disabled
        stop_loss_pct: float = 5.0,  # Stop-loss percentage for longs
        stop_loss_short_pct: float = 3.0,  # Stop-loss percentage for shorts (tighter)
        cash_reserve_pct: float = 0.50,  # Percentage of equity to keep as cash reserve (anti-liquidation)
        dry_run: bool = False    # If True, log orders but don't send
    ):
        self.client = api_client
        self.symbols = symbols
        self.mode = mode.upper()
        self.leverage = str(leverage) if self.mode == 'MARGIN' else 'none'

        self.market_threshold = market_threshold
        self.limit_timeout = limit_timeout
        self.max_drawdown = max_drawdown
        self.stop_loss_pct = stop_loss_pct
        self.stop_loss_short_pct = stop_loss_short_pct
        self.cash_reserve_pct = cash_reserve_pct
        self.dry_run = dry_run

        # Inverted map for lookups
        self.symbol_to_id = {v: k for k, v in symbols.items()}
        # Handle Kraken format variations (XBT/USD vs XXBTZUSD)
        self.kraken_map = {}

        # State
        self.state = PortfolioState(cash=0.0)
        self.pending_orders: Dict[str, PendingOrder] = {} # order_id -> Info
        self.stop_loss_orders: Dict[str, StopLossOrder] = {} # order_id -> StopLossOrder
        self.position_to_sl: Dict[int, str] = {} # asset_idx -> sl_order_id mapping
        self.initial_equity = 0.0
        self.equity = 0.0
        self.margin_level = 0.0

        # Cash Reserve Management (Anti-Liquidation Protection)
        self.cash_reserve_locked = 0.0  # USD amount locked as reserve
        self.available_capital = 0.0    # USD available for trading
        
        # Mappings for asset names (e.g. ZUSD -> USD)
        self.asset_map = {
            'ZUSD': 'USD', 'ZEUR': 'EUR', 'ZGBP': 'GBP', 'ZCAD': 'CAD', 'ZJPY': 'JPY',
            'XXBT': 'BTC', 'XBT': 'BTC', 'XETH': 'ETH', 'XLTC': 'LTC', 'XXRP': 'XRP', 'XDG': 'DOGE'
        }

    async def initialize(self):
        """Initial sync to establish baseline and set up cash reserve"""
        logging.info("Initializing Real Portfolio Manager...")
        await self.sync()
        self.initial_equity = self.equity

        # Establish cash reserve (anti-liquidation protection)
        self.cash_reserve_locked = self.equity * self.cash_reserve_pct
        self.available_capital = self.equity - self.cash_reserve_locked

        logging.info(f"Initial Equity: ${self.initial_equity:.2f}")
        logging.info(f"Cash Reserve Locked: ${self.cash_reserve_locked:.2f} ({self.cash_reserve_pct*100:.0f}%)")
        logging.info(f"Available for Trading: ${self.available_capital:.2f}")

        if self.cash_reserve_pct > 0:
            logging.warning(f"⚠️  CASH RESERVE ACTIVE: {self.cash_reserve_pct*100:.0f}% of equity is protected from trading")
            logging.warning(f"   This prevents liquidation but limits position sizes")

    def _normalize_asset(self, asset: str) -> str:
        """Normalize Kraken asset name to standard (XXBT -> BTC)"""
        return self.asset_map.get(asset, asset)

    def get_valid_actions(self) -> np.ndarray:
        """
        Get mask of valid actions based on current holdings and mode.
        Returns: (n_assets, 3) boolean mask
        """
        n_assets = len(self.symbols)
        mask = np.ones((n_assets, 3), dtype=bool)
        
        if self.mode == 'SPOT':
            # In Spot mode, can only Sell if we have a position
            # Iterate through all assets managed by this manager
            for idx in self.symbols.keys():
                # Check if we have this asset in our state positions
                # Note: sync() needs to populate state.positions correctly for this to work!
                # Currently sync() only does this for MARGIN.
                # For SPOT, we need to map balances to positions.
                
                # If idx is NOT in state.positions, we assume we hold 0.
                if idx not in self.state.positions:
                     mask[idx, 0] = False # Disable Sell
        
        # In Margin mode, generally we can open Short (Sell) or Long (Buy) freely
        # unless margin limits are hit, but checking that is complex.
        # So we leave mask as True for Margin.
        
        return mask

    async def sync(self):
        """
        Synchronize local state with Kraken Exchange.
        Fetches Balances, Positions, and Open Orders.
        """
        try:
            # 1. Get Balances (Cash & Spot Assets)
            balances = await self.client.get_account_balance()
            if not balances:
                logging.warning("Sync: Failed to fetch balances")
                return

            # Update Cash (USD)
            usd_bal = float(balances.get('ZUSD', 0.0)) + float(balances.get('USD', 0.0))
            self.state.cash = usd_bal
            
            # 2. Get Trade Balance (Equity, Margin)
            # This gives us the total "Portfolio Value" accurately
            trade_bal = await self.client.get_trade_balance(asset="ZUSD")
            self.equity = float(trade_bal.get('eb', 0.0)) # Equivalent Balance (Equity)
            self.margin_level = float(trade_bal.get('ml', 0.0)) if self.mode == 'MARGIN' else 0.0
            
            self.state.total_value = self.equity
            
            self.state.positions.clear()

            # 3. Get Open Positions
            if self.mode == 'MARGIN':
                positions = await self.client.get_open_positions()
                for pid, info in positions.items():
                    # TODO: Robust mapping of 'pair' to asset_idx
                    pass
            else:
                # SPOT MODE: Map Balances to Positions
                # We need to know which asset_idx corresponds to which balance key (e.g. 'XXBT')
                # This requires a map we built in init or via symbol map.
                # For now, we iterate symbols and see if we have balance.
                for idx, symbol in self.symbols.items():
                    # Symbol is like 'XBT/USD'. We need 'XXBT' or 'XBT'.
                    # This is hard without the pairs info.
                    # Heuristic: Split by '/' and check.
                    base = symbol.split('/')[0] if '/' in symbol else symbol
                    
                    # Check obvious variants
                    variants = [base, f"X{base}", f"Z{base}"]
                    if base == 'BTC': variants.extend(['XBT', 'XXBT'])
                    
                    size = 0.0
                    for v in variants:
                        if v in balances:
                            size = float(balances[v])
                            break
                    
                    if size > 0.0001: # Threshold for dust
                        # Create a "dummy" position object for tracking
                        self.state.positions[idx] = Position(
                            asset_idx=idx,
                            symbol=symbol,
                            side='long',
                            size=size,
                            entry_price=0.0 # Unknown for spot sync
                        )

            # 4. Get Open Orders
            open_orders = await self.client.get_open_orders()
            
            # Reconcile Pending Orders
            # Remove any local pending orders that are no longer in open_orders
            # This means they were filled or cancelled
            current_order_ids = set(open_orders.keys())
            
            removed_orders = []
            for oid in list(self.pending_orders.keys()):
                if oid not in current_order_ids:
                    # Order is gone. Assume filled for now (or cancelled externally).
                    # In a real system, we'd query ClosedOrders to verify fill vs cancel.
                    removed_orders.append(oid)
                    del self.pending_orders[oid]
            
            if removed_orders:
                logging.info(f"Orders no longer open (Filled/Cancelled): {removed_orders}")

            # Check for timed-out limit orders
            now = time.time()
            for oid, order in list(self.pending_orders.items()):
                if now - order.timestamp > self.limit_timeout:
                    logging.info(f"Order {oid} timed out ({self.limit_timeout}s). Cancelling...")
                    await self.client.cancel_order(oid)
                    # We'll remove it from the list on next sync when it disappears from API

            # 5. Reconcile Stop-Loss Orders
            # Check which SL orders are still open vs triggered/filled
            removed_sl_orders = []
            for sl_oid in list(self.stop_loss_orders.keys()):
                if sl_oid not in current_order_ids:
                    # Stop-loss order is gone - either triggered or manually cancelled
                    sl_order = self.stop_loss_orders[sl_oid]
                    logging.warning(f"⚠️  STOP-LOSS TRIGGERED or CANCELLED: {sl_order.symbol} {sl_order.entry_side} @ {sl_order.stop_price:.6f}")

                    # Remove from tracking
                    removed_sl_orders.append(sl_oid)
                    del self.stop_loss_orders[sl_oid]

                    # Remove from position mapping
                    if sl_order.asset_idx in self.position_to_sl:
                        del self.position_to_sl[sl_order.asset_idx]

            if removed_sl_orders:
                logging.info(f"Stop-Loss Orders Removed (Triggered/Cancelled): {removed_sl_orders}")

            # 6. CRITICAL SAFETY CHECK: Positions without SL protection
            # If we have a position but no corresponding SL order, that's dangerous
            for idx, pos in self.state.positions.items():
                if idx not in self.position_to_sl:
                    logging.error(f"⚠️ ⚠️ ⚠️  DANGER: Position {pos.symbol} has NO STOP-LOSS protection!")
                    # Could auto-place SL here, but that requires knowing entry price
                    # For now, just alert loudly

            # 7. Update Cash Reserve and Available Capital
            # Recalculate after positions are updated
            if self.cash_reserve_pct > 0:
                self.cash_reserve_locked = self.equity * self.cash_reserve_pct
                self.available_capital = self.equity - self.cash_reserve_locked

                # Log if capital is getting low
                utilization = (self.cash_reserve_locked + self._calculate_margin_used()) / self.equity if self.equity > 0 else 0
                if utilization > 0.9:
                    logging.warning(f"⚠️  Capital Utilization: {utilization*100:.1f}% - Approaching limits!")

                logging.debug(f"Cash Reserve: ${self.cash_reserve_locked:.2f} | Available: ${self.available_capital:.2f}")

        except Exception as e:
            logging.error(f"Sync Error: {e}")

    def _calculate_margin_used(self) -> float:
        """Calculate total margin currently used by open positions"""
        # For MARGIN mode, sum up the margin used by positions
        # For SPOT mode, sum up the value of holdings
        total_margin = 0.0
        for pos in self.state.positions.values():
            # Simplified: position value = size * entry_price
            # Real margin calculation would query Kraken's margin info
            pos_value = getattr(pos, 'size', 0) * getattr(pos, 'entry_price', 0)
            total_margin += pos_value
        return total_margin

    async def execute_actions(
        self,
        actions: Dict[int, int], # idx -> action (0,1,2)
        confidences: Dict[int, float], # idx -> confidence
        prices: Dict[int, float] # idx -> current_price
    ):
        """
        Execute trading decisions.
        Handles Market vs Limit logic based on confidence.
        """
        if self.max_drawdown > 0 and self.initial_equity > 0:
            current_dd = (self.initial_equity - self.equity) / self.initial_equity
            if current_dd >= self.max_drawdown:
                logging.warning(f"Hard Stop Drawdown Triggered ({current_dd:.2%} >= {self.max_drawdown:.2%}). Trading halted.")
                return

        for idx, action in actions.items():
            if action == 1: # Hold
                continue
                
            symbol = self.symbols.get(idx)
            if not symbol:
                continue

            price = prices.get(idx)
            confidence = confidences.get(idx, 0.0)

            # Skip if no price available
            if price is None or price <= 0:
                continue

            # Determine Side
            side = 'buy' if action == 2 else 'sell'

            # Determine Size - CRITICAL: Use available_capital, NOT total equity
            # This respects the cash reserve and prevents over-leveraging
            target_pct = 0.25 # 25% per trade (of available capital, not total equity)

            # Use available capital (equity - cash reserve)
            capital_base = self.available_capital if self.cash_reserve_pct > 0 else self.equity
            usd_size = capital_base * target_pct

            # Check if we have enough available capital
            if usd_size > self.available_capital and self.cash_reserve_pct > 0:
                logging.warning(f"Position size ${usd_size:.2f} exceeds available capital ${self.available_capital:.2f}")
                logging.warning(f"Reducing to available capital (cash reserve protection)")
                usd_size = self.available_capital * 0.95  # Leave 5% buffer

            volume = usd_size / price
            
            # Determine Order Type
            if confidence >= self.market_threshold:
                order_type = 'market'
                limit_price = None
                logging.info(f"CRITICAL SIGNAL ({confidence:.2f}): Placing MARKET {side} for {symbol}")
            else:
                order_type = 'limit'
                # Chase logic: 
                # Buy: Place at current price (assuming it's close to bid/ask)
                # Sell: Place at current price
                limit_price = price 
                logging.info(f"STANDARD SIGNAL ({confidence:.2f}): Placing LIMIT {side} for {symbol} @ {limit_price}")

            if self.dry_run:
                logging.info(f"[DRY RUN] Would {side} {volume:.4f} {symbol} ({order_type})")
                continue

            # Execute Entry Order
            try:
                resp = await self.client.add_order(
                    pair=symbol,
                    side=side,
                    order_type=order_type,
                    volume=volume,
                    price=limit_price,
                    leverage=self.leverage if self.mode == 'MARGIN' else None
                )

                if resp.get('error'):
                    logging.error(f"Order Failed: {resp['error']}")
                else:
                    txid = resp['result']['txid'][0]
                    logging.info(f"Order Placed: {txid}")

                    if order_type == 'limit':
                        self.pending_orders[txid] = PendingOrder(
                            order_id=txid,
                            asset_idx=idx,
                            side=side,
                            order_type=order_type,
                            price=limit_price,
                            size=volume,
                            timestamp=time.time(),
                            confidence=confidence
                        )

                    # CRITICAL: Place Stop-Loss Order immediately after entry
                    # This protects the position even if our script crashes
                    await self._place_stop_loss_order(idx, symbol, side, price, volume)

            except Exception as e:
                logging.error(f"Execution Exception: {e}")

    async def _place_stop_loss_order(
        self,
        asset_idx: int,
        symbol: str,
        entry_side: str,  # 'buy' or 'sell'
        entry_price: float,
        volume: float
    ):
        """
        Place exchange-level stop-loss order to protect position.

        For LONG (buy): Stop-loss triggers sell when price drops 5%
        For SHORT (sell): Stop-loss triggers buy when price rises 3%
        """
        try:
            # Calculate stop price based on position type
            if entry_side == 'buy':  # Long position
                stop_price = entry_price * (1 - self.stop_loss_pct / 100)
                sl_side = 'sell'
                risk_pct = self.stop_loss_pct
            else:  # Short position
                stop_price = entry_price * (1 + self.stop_loss_short_pct / 100)
                sl_side = 'buy'
                risk_pct = self.stop_loss_short_pct

            if self.dry_run:
                logging.info(f"[DRY RUN] Would place SL order: {sl_side} {volume:.4f} {symbol} @ {stop_price:.6f} (-{risk_pct}%)")
                return

            # Place stop-loss order on exchange
            logging.info(f"Placing STOP-LOSS: {sl_side} {volume:.4f} {symbol} @ {stop_price:.6f} (-{risk_pct}%)")

            resp = await self.client.add_order(
                pair=symbol,
                side=sl_side,
                order_type='stop-loss',
                volume=volume,
                price=stop_price,
                leverage=self.leverage if self.mode == 'MARGIN' else None
            )

            if resp.get('error'):
                logging.error(f"CRITICAL: Stop-Loss Order Failed: {resp['error']}")
                logging.error(f"POSITION AT RISK: {symbol} {entry_side} {volume} without SL protection!")
            else:
                sl_txid = resp['result']['txid'][0]
                logging.info(f"✓ Stop-Loss Order Placed: {sl_txid}")

                # Track the SL order
                self.stop_loss_orders[sl_txid] = StopLossOrder(
                    order_id=sl_txid,
                    asset_idx=asset_idx,
                    symbol=symbol,
                    entry_side=entry_side,
                    entry_price=entry_price,
                    stop_price=stop_price,
                    volume=volume,
                    timestamp=time.time()
                )
                self.position_to_sl[asset_idx] = sl_txid

        except Exception as e:
            logging.error(f"CRITICAL: Exception placing stop-loss for {symbol}: {e}")
            logging.error(f"POSITION AT RISK without SL protection!")

    async def close_all_positions(self):
        """PANIC BUTTON: Cancel all orders and flatten portfolio"""
        logging.warning("!!! KILL SWITCH ACTIVATED !!!")
        
        # 1. Cancel All Open Orders
        try:
            logging.info("Cancelling all open orders...")
            await self.client.cancel_all_orders()
        except Exception as e:
            logging.error(f"Failed to cancel orders: {e}")
            
        # 2. Close Positions
        # For Spot: Sell all non-USD balances
        # For Margin: Close all open positions
        
        try:
            if self.mode == 'MARGIN':
                logging.info("Closing all margin positions...")
                # Unfortunately Kraken doesn't have "CloseAll". We must iterate.
                positions = await self.client.get_open_positions()
                for txid, info in positions.items():
                    # Close by sending opposing order
                    pair = info['pair']
                    vol = info['vol']
                    side = 'sell' if info['type'] == 'buy' else 'buy'
                    
                    logging.info(f"Closing {pair} position ({vol})...")
                    await self.client.add_order(
                        pair=pair, side=side, order_type='market', volume=vol, leverage=self.leverage
                    )
            else:
                # SPOT Mode: Sell everything to USD
                logging.info("Selling all spot assets...")
                balances = await self.client.get_account_balance()
                for asset, bal in balances.items():
                    if float(bal) > 0 and asset not in ['ZUSD', 'USD', 'KFEE']:
                        # Construct pair (e.g. XETH -> ETHUSD)
                        # This mapping is hard. Ideally we use known pairs list.
                        # Naive attempt for standard assets:
                        norm = self._normalize_asset(asset)
                        pair = f"{norm}USD"
                        
                        logging.info(f"Selling {bal} {asset} ({pair})...")
                        await self.client.add_order(
                            pair=pair, side='sell', order_type='market', volume=bal
                        )
                        
        except Exception as e:
            logging.error(f"Failed to close positions: {e}")
