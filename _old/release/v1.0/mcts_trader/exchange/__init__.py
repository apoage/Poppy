"""
Exchange abstraction layer.

Provides unified API for different cryptocurrency exchanges.
"""

from .base import ExchangeAPI, OrderType, OrderSide, Order, Position, Ticker
from .paper import PaperExchange

__all__ = [
    'ExchangeAPI',
    'OrderType',
    'OrderSide',
    'Order',
    'Position',
    'Ticker',
    'PaperExchange'
]
