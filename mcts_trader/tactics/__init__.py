"""
Portfolio Tactics - Modular Trading Strategies

Each tactic extends the action space with specific trading capabilities.
Tactics are composable via Gym wrappers.
"""

from .basic_trading import BasicTrading
from .portfolio_morphing import PortfolioMorphing

__all__ = ['BasicTrading', 'PortfolioMorphing']
