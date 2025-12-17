"""
Paper trading exchange implementation.

Simulates exchange functionality for backtesting and paper trading.
Uses historical data or live data feed without real execution.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Callable
import uuid
import logging

from .base import (
    ExchangeAPI, Order, OrderType, OrderSide, OrderStatus,
    Position, Ticker, OHLCV
)


class PaperExchange(ExchangeAPI):
    """
    Paper trading exchange - simulates trading without real money.

    Can be backed by:
    - Historical database (for backtesting)
    - Live data feed from real exchange (for paper trading)
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        base_currency: str = "USD",
        fee_rate: float = 0.001,  # 0.1% per trade
        slippage: float = 0.0005  # 0.05% slippage
    ):
        super().__init__()
        self.initial_balance = initial_balance
        self.base_currency = base_currency
        self.fee_rate = fee_rate
        self.slippage = slippage

        # State
        self.balances: Dict[str, float] = {base_currency: initial_balance}
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.tickers: Dict[str, Ticker] = {}

        # Data source callbacks
        self._data_source: Optional[Callable] = None

        self._connected = False
        self._listening = False
        logging.info(f"PaperExchange initialized with ${initial_balance} {base_currency}")

    # === Connection ===

    async def connect(self) -> bool:
        """Paper exchange is always 'connected'"""
        self._connected = True
        logging.info("PaperExchange connected")
        return True

    async def disconnect(self) -> None:
        self._connected = False
        self._listening = False
        logging.info("PaperExchange disconnected")

    # === Data Source Integration ===

    def set_data_source(self, source: 'ExchangeAPI') -> None:
        """
        Set a real exchange as data source for paper trading.
        Paper exchange will mirror prices but not execute real trades.
        """
        self._data_source = source

        # Forward ticker callbacks
        source.on_ticker(self._handle_ticker_update)
        source.on_ohlc(self._handle_ohlc_update)

    async def _handle_ticker_update(self, ticker: Ticker) -> None:
        """Handle ticker from data source"""
        self.tickers[ticker.symbol] = ticker
        await self._emit_ticker(ticker)

        # Update position prices
        if ticker.symbol in self.positions:
            pos = self.positions[ticker.symbol]
            pos.current_price = ticker.last
            pos.unrealized_pnl = (ticker.last - pos.entry_price) * pos.quantity

    async def _handle_ohlc_update(self, symbol: str, ohlc: OHLCV) -> None:
        """Handle OHLC from data source"""
        # Update internal ticker
        if symbol not in self.tickers:
            self.tickers[symbol] = Ticker(
                symbol=symbol,
                bid=ohlc.close * 0.9999,
                ask=ohlc.close * 1.0001,
                last=ohlc.close,
                volume_24h=ohlc.volume
            )
        else:
            self.tickers[symbol].last = ohlc.close
            self.tickers[symbol].bid = ohlc.close * 0.9999
            self.tickers[symbol].ask = ohlc.close * 1.0001

        await self._emit_ohlc(symbol, ohlc)

    def update_price(self, symbol: str, price: float, volume: float = 0.0) -> None:
        """
        Manually update price (for backtesting from historical data).
        """
        self.tickers[symbol] = Ticker(
            symbol=symbol,
            bid=price * 0.9999,
            ask=price * 1.0001,
            last=price,
            volume_24h=volume
        )

        # Update position
        if symbol in self.positions:
            pos = self.positions[symbol]
            pos.current_price = price
            pos.unrealized_pnl = (price - pos.entry_price) * pos.quantity

    # === Market Data ===

    async def get_ticker(self, symbol: str) -> Ticker:
        if symbol in self.tickers:
            return self.tickers[symbol]

        if self._data_source:
            ticker = await self._data_source.get_ticker(symbol)
            self.tickers[symbol] = ticker
            return ticker

        raise ValueError(f"No price data for {symbol}")

    async def get_tickers(self, symbols: List[str]) -> Dict[str, Ticker]:
        result = {}
        for symbol in symbols:
            try:
                result[symbol] = await self.get_ticker(symbol)
            except ValueError:
                pass
        return result

    async def get_ohlcv(
        self,
        symbol: str,
        interval: int = 1,
        limit: int = 100,
        since: Optional[int] = None
    ) -> List[OHLCV]:
        if self._data_source:
            return await self._data_source.get_ohlcv(symbol, interval, limit, since)
        return []

    async def get_tradeable_pairs(self) -> Dict[str, Dict]:
        if self._data_source:
            return await self._data_source.get_tradeable_pairs()
        return {}

    # === Streaming ===

    async def subscribe_ticker(self, symbols: List[str]) -> None:
        if self._data_source:
            await self._data_source.subscribe_ticker(symbols)

    async def subscribe_ohlc(self, symbols: List[str], interval: int = 1) -> None:
        if self._data_source:
            await self._data_source.subscribe_ohlc(symbols, interval)

    async def unsubscribe(self, symbols: List[str]) -> None:
        if self._data_source:
            await self._data_source.unsubscribe(symbols)

    # === Trading ===

    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        **kwargs
    ) -> Order:
        """
        Place a simulated order.
        Market orders fill immediately at current price +/- slippage.
        """
        order_id = str(uuid.uuid4())[:8]

        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            status=OrderStatus.PENDING
        )

        self.orders[order_id] = order

        # Market orders fill immediately
        if order_type == OrderType.MARKET:
            await self._fill_order(order)

        logging.info(f"Order placed: {side.value} {quantity} {symbol} @ {price or 'MARKET'}")
        return order

    async def _fill_order(self, order: Order) -> None:
        """Execute order fill"""
        try:
            ticker = await self.get_ticker(order.symbol)
        except ValueError:
            order.status = OrderStatus.REJECTED
            logging.error(f"Order rejected: No price for {order.symbol}")
            return

        # Determine execution price with slippage
        if order.side == OrderSide.BUY:
            exec_price = ticker.ask * (1 + self.slippage)
        else:
            exec_price = ticker.bid * (1 - self.slippage)

        # For limit orders, check price
        if order.order_type == OrderType.LIMIT and order.price:
            if order.side == OrderSide.BUY and exec_price > order.price:
                return  # Don't fill, price too high
            if order.side == OrderSide.SELL and exec_price < order.price:
                return  # Don't fill, price too low
            exec_price = order.price

        # Calculate cost/proceeds
        total_value = exec_price * order.quantity
        fee = total_value * self.fee_rate

        # Extract base/quote currencies
        base, quote = self._parse_symbol(order.symbol)

        if order.side == OrderSide.BUY:
            # Check balance
            required = total_value + fee
            if self.balances.get(quote, 0) < required:
                order.status = OrderStatus.REJECTED
                logging.warning(f"Order rejected: Insufficient {quote} balance")
                return

            # Execute
            self.balances[quote] = self.balances.get(quote, 0) - required
            self.balances[base] = self.balances.get(base, 0) + order.quantity

            # Update position
            if order.symbol in self.positions:
                pos = self.positions[order.symbol]
                total_qty = pos.quantity + order.quantity
                pos.entry_price = (pos.entry_price * pos.quantity + exec_price * order.quantity) / total_qty
                pos.quantity = total_qty
            else:
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    entry_price=exec_price,
                    current_price=exec_price
                )

        else:  # SELL
            # Check balance
            if self.balances.get(base, 0) < order.quantity:
                order.status = OrderStatus.REJECTED
                logging.warning(f"Order rejected: Insufficient {base} balance")
                return

            # Execute
            self.balances[base] = self.balances.get(base, 0) - order.quantity
            self.balances[quote] = self.balances.get(quote, 0) + (total_value - fee)

            # Update position
            if order.symbol in self.positions:
                pos = self.positions[order.symbol]
                pos.quantity -= order.quantity
                pos.realized_pnl += (exec_price - pos.entry_price) * order.quantity

                if abs(pos.quantity) < 1e-8:
                    del self.positions[order.symbol]

        # Update order status
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = exec_price
        self.order_history.append(order)

        logging.info(f"Order filled: {order.side.value} {order.quantity} {order.symbol} @ {exec_price:.6f}")

    def _parse_symbol(self, symbol: str) -> tuple:
        """Parse symbol into base/quote currencies"""
        if '/' in symbol:
            return symbol.split('/')
        # Default assumption: last 3 chars are quote
        return symbol[:-3], symbol[-3:]

    async def cancel_order(self, order_id: str) -> bool:
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status == OrderStatus.PENDING:
                order.status = OrderStatus.CANCELLED
                return True
        return False

    async def get_order(self, order_id: str) -> Order:
        if order_id in self.orders:
            return self.orders[order_id]
        raise ValueError(f"Order not found: {order_id}")

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        orders = [o for o in self.orders.values() if o.status == OrderStatus.PENDING]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    # === Account ===

    async def get_balance(self) -> Dict[str, float]:
        return self.balances.copy()

    async def get_positions(self) -> List[Position]:
        return list(self.positions.values())

    def get_portfolio_value(self) -> float:
        """Total portfolio value in base currency"""
        total = self.balances.get(self.base_currency, 0)

        for pos in self.positions.values():
            total += pos.value

        return total

    def get_pnl(self) -> Dict[str, float]:
        """Get P&L summary"""
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        realized = sum(p.realized_pnl for p in self.positions.values())
        total_value = self.get_portfolio_value()

        return {
            'total_value': total_value,
            'unrealized_pnl': unrealized,
            'realized_pnl': realized,
            'total_pnl': total_value - self.initial_balance,
            'return_pct': (total_value - self.initial_balance) / self.initial_balance * 100
        }

    def reset(self) -> None:
        """Reset to initial state"""
        self.balances = {self.base_currency: self.initial_balance}
        self.positions.clear()
        self.orders.clear()
        self.order_history.clear()
        self.tickers.clear()
        logging.info("PaperExchange reset to initial state")
