"""
Abstract base classes for exchange API integration.

Implement these interfaces for specific exchanges (Kraken, Binance, etc.)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
import asyncio


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Unified order representation"""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None  # None for market orders
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    exchange_order_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Unified position representation"""
    symbol: str
    quantity: float  # Positive for long, negative for short
    entry_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    @property
    def side(self) -> str:
        return "long" if self.quantity > 0 else "short"

    @property
    def value(self) -> float:
        return abs(self.quantity) * self.current_price


@dataclass
class Ticker:
    """Unified ticker/price data"""
    symbol: str
    bid: float
    ask: float
    last: float
    volume_24h: float
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass
class OHLCV:
    """OHLCV candle data"""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class ExchangeAPI(ABC):
    """
    Abstract base class for exchange APIs.

    Implement this for each exchange (Kraken, Binance, etc.)
    """

    def __init__(self, api_key: str = "", api_secret: str = ""):
        self.api_key = api_key
        self.api_secret = api_secret
        self._connected = False
        self._callbacks: Dict[str, List[Callable]] = {
            'ticker': [],
            'ohlc': [],
            'order': [],
            'trade': [],
        }

    # === Connection Management ===

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to exchange"""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection"""
        pass

    @property
    def is_connected(self) -> bool:
        return self._connected

    # === Market Data ===

    @abstractmethod
    async def get_ticker(self, symbol: str) -> Ticker:
        """Get current ticker for symbol"""
        pass

    @abstractmethod
    async def get_tickers(self, symbols: List[str]) -> Dict[str, Ticker]:
        """Get tickers for multiple symbols"""
        pass

    @abstractmethod
    async def get_ohlcv(
        self,
        symbol: str,
        interval: int = 1,  # minutes
        limit: int = 100,
        since: Optional[int] = None  # Unix timestamp
    ) -> List[OHLCV]:
        """Get historical OHLCV candles"""
        pass

    @abstractmethod
    async def get_tradeable_pairs(self) -> Dict[str, Dict]:
        """Get all tradeable pairs with their info"""
        pass

    # === Streaming ===

    @abstractmethod
    async def subscribe_ticker(self, symbols: List[str]) -> None:
        """Subscribe to real-time ticker updates"""
        pass

    @abstractmethod
    async def subscribe_ohlc(self, symbols: List[str], interval: int = 1) -> None:
        """Subscribe to real-time OHLC updates"""
        pass

    @abstractmethod
    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from updates"""
        pass

    def on_ticker(self, callback: Callable[[Ticker], None]) -> None:
        """Register ticker callback"""
        self._callbacks['ticker'].append(callback)

    def on_ohlc(self, callback: Callable[[str, OHLCV], None]) -> None:
        """Register OHLC callback"""
        self._callbacks['ohlc'].append(callback)

    async def _emit_ticker(self, ticker: Ticker) -> None:
        """Emit ticker to all callbacks"""
        for callback in self._callbacks['ticker']:
            if asyncio.iscoroutinefunction(callback):
                await callback(ticker)
            else:
                callback(ticker)

    async def _emit_ohlc(self, symbol: str, ohlc: OHLCV) -> None:
        """Emit OHLC to all callbacks"""
        for callback in self._callbacks['ohlc']:
            if asyncio.iscoroutinefunction(callback):
                await callback(symbol, ohlc)
            else:
                callback(symbol, ohlc)

    # === Trading ===

    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        **kwargs
    ) -> Order:
        """Place an order"""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        pass

    @abstractmethod
    async def get_order(self, order_id: str) -> Order:
        """Get order status"""
        pass

    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders"""
        pass

    # === Account ===

    @abstractmethod
    async def get_balance(self) -> Dict[str, float]:
        """Get account balances"""
        pass

    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get open positions"""
        pass

    # === Helpers ===

    def normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol to exchange format.
        Override in subclass for exchange-specific normalization.

        Example: "BTC/USD" -> "XXBTZUSD" for Kraken
        """
        return symbol

    def denormalize_symbol(self, symbol: str) -> str:
        """
        Convert exchange symbol to standard format.
        Override in subclass for exchange-specific conversion.

        Example: "XXBTZUSD" -> "BTC/USD"
        """
        return symbol
