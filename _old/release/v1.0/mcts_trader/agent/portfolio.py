import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

@dataclass
class Position:
    asset_idx: int
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    current_price: float = 0.0
    
    @property
    def pnl(self) -> float:
        multiplier = 1 if self.side == 'long' else -1
        return multiplier * self.size * (self.current_price - self.entry_price)
    
    @property
    def pnl_percent(self) -> float:
        if self.entry_price == 0:
            return 0.0
        multiplier = 1 if self.side == 'long' else -1
        return multiplier * (self.current_price - self.entry_price) / self.entry_price * 100


@dataclass
class PortfolioState:
    cash: float
    positions: Dict[int, Position] = field(default_factory=dict)
    total_value: float = 0.0
    max_value: float = 0.0
    
    @property
    def drawdown(self) -> float:
        if self.max_value == 0:
            return 0.0
        return (self.max_value - self.total_value) / self.max_value


class PortfolioManager:
    """
    Manages positions, risk limits, and order execution.
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        max_position_pct: float = 0.1,      # Max 10% per position
        max_total_exposure: float = 2.0,     # Max 200% total (with leverage)
        max_drawdown: float = 0.15,          # Stop trading at 15% drawdown
        n_assets: int = 64
    ):
        self.initial_capital = initial_capital
        self.max_position_pct = max_position_pct
        self.max_total_exposure = max_total_exposure
        self.max_drawdown = max_drawdown
        self.n_assets = n_assets
        
        self.state = PortfolioState(cash=initial_capital)
        self.state.total_value = initial_capital
        self.state.max_value = initial_capital
        
        # Asset info
        self.symbols: Dict[int, str] = {}
        self.prices: Dict[int, float] = {}
        
    def register_assets(self, symbols: Dict[int, str]):
        self.symbols = symbols
    
    def update_prices(self, prices: Dict[int, float]):
        """Update current prices and portfolio value"""
        self.prices = prices
        
        # Update position values
        position_value = 0.0
        for idx, pos in self.state.positions.items():
            if idx in prices:
                pos.current_price = prices[idx]
                position_value += abs(pos.size * pos.current_price)
        
        # Calculate total value
        self.state.total_value = self.state.cash + sum(
            abs(pos.size * pos.current_price) for pos in self.state.positions.values()
        )
        
        self.state.max_value = max(self.state.max_value, self.state.total_value)
    
    def get_valid_actions(self) -> np.ndarray:
        """
        Get mask of valid actions considering risk limits.
        
        Returns:
            mask: (n_assets, 3) - True if action is valid
        """
        mask = np.ones((self.n_assets, 3), dtype=bool)
        
        # Check drawdown limit
        if self.state.drawdown >= self.max_drawdown:
            # Only allow closing positions (action depends on current position)
            for idx in range(self.n_assets):
                if idx in self.state.positions:
                    pos = self.state.positions[idx]
                    if pos.side == 'long':
                        mask[idx, 2] = False  # Can't buy more
                    else:
                        mask[idx, 0] = False  # Can't sell more
                else:
                    mask[idx, 0] = False  # Can't open short
                    mask[idx, 2] = False  # Can't open long
        
        # Check per-position limits
        for idx in range(self.n_assets):
            if idx not in self.prices:
                mask[idx, :] = [False, True, False]  # Only hold if no price
                continue
            
            current_exposure = 0.0
            if idx in self.state.positions:
                pos = self.state.positions[idx]
                current_exposure = abs(pos.size * pos.current_price) / self.state.total_value
            
            if current_exposure >= self.max_position_pct:
                # Can't increase position
                if idx in self.state.positions:
                    if self.state.positions[idx].side == 'long':
                        mask[idx, 2] = False
                    else:
                        mask[idx, 0] = False
        
        return mask
    
    def execute_actions(
        self,
        actions: np.ndarray,      # (n_assets,) - 0=sell, 1=hold, 2=buy
        sizes: np.ndarray,        # (n_assets,) - position sizes (0-1)
        prices: Dict[int, float]
    ) -> List[Dict]:
        """
        Execute trading actions.
        
        Returns:
            List of executed orders
        """
        self.update_prices(prices)
        orders = []
        
        for idx in range(self.n_assets):
            action = actions[idx]
            size = sizes[idx]
            
            if action == 1:  # Hold
                continue
            
            if idx not in prices:
                continue
            
            price = prices[idx]
            
            # Calculate target position value
            max_value = self.state.total_value * self.max_position_pct * size
            target_size = max_value / price
            
            current_pos = self.state.positions.get(idx)
            
            if action == 2:  # Buy
                order = self._create_buy_order(idx, target_size, price, current_pos)
            else:  # Sell
                order = self._create_sell_order(idx, target_size, price, current_pos)
            
            if order:
                orders.append(order)
                self._apply_order(order)
        
        return orders
    
    def _create_buy_order(
        self,
        idx: int,
        target_size: float,
        price: float,
        current_pos: Optional[Position]
    ) -> Optional[Dict]:
        """Create buy order"""
        if current_pos is None:
            # Open long
            return {
                'asset_idx': idx,
                'symbol': self.symbols.get(idx, f'ASSET_{idx}'),
                'side': 'buy',
                'size': target_size,
                'price': price,
                'type': 'open_long'
            }
        elif current_pos.side == 'short':
            # Close short
            return {
                'asset_idx': idx,
                'symbol': self.symbols.get(idx, f'ASSET_{idx}'),
                'side': 'buy',
                'size': current_pos.size,
                'price': price,
                'type': 'close_short'
            }
        else:
            # Already long - could increase but respecting limits
            return None
    
    def _create_sell_order(
        self,
        idx: int,
        target_size: float,
        price: float,
        current_pos: Optional[Position]
    ) -> Optional[Dict]:
        """Create sell order"""
        if current_pos is None:
            # Open short
            return {
                'asset_idx': idx,
                'symbol': self.symbols.get(idx, f'ASSET_{idx}'),
                'side': 'sell',
                'size': target_size,
                'price': price,
                'type': 'open_short'
            }
        elif current_pos.side == 'long':
            # Close long
            return {
                'asset_idx': idx,
                'symbol': self.symbols.get(idx, f'ASSET_{idx}'),
                'side': 'sell',
                'size': current_pos.size,
                'price': price,
                'type': 'close_long'
            }
        else:
            # Already short
            return None
    
    def _apply_order(self, order: Dict):
        """
        Apply executed order to state.
        order: {'asset_idx': int, 'side': 'buy'/'sell', 'size': float, 'price': float}
        """
        idx = order['asset_idx']
        cost = order['size'] * order['price']
        
        if order['side'] == 'buy':
            # Check if we already have a position
            if idx in self.state.positions:
                pos = self.state.positions[idx]
                # Average entry price
                total_cost = pos.entry_price * pos.size + cost
                new_size = pos.size + order['size']
                pos.entry_price = total_cost / new_size
                pos.size = new_size
                pos.current_price = order['price']
            else:
                self.state.positions[idx] = Position(
                    asset_idx=idx,
                    symbol=order.get('symbol', f'ASSET_{idx}'),
                    side='long',  # Buy creates long position
                    size=order['size'],
                    entry_price=order['price'],
                    current_price=order['price']
                )
            # DEDUCT CASH (Fix)
            self.state.cash -= cost
                
        elif order['side'] == 'sell':
            if idx in self.state.positions:
                pos = self.state.positions[idx]
                if pos.size <= order['size']:
                    # Full close
                    del self.state.positions[idx]
                else:
                    # Partial close
                    pos.size -= order['size']
                # ADD CASH (Fix)
                self.state.cash += cost
            else:
                # Open short position
                self.state.positions[idx] = Position(
                    asset_idx=idx,
                    symbol=order.get('symbol', f'ASSET_{idx}'),
                    side='short',
                    size=order['size'],
                    entry_price=order['price'],
                    current_price=order['price']
                )
                # Short sale - receive cash but have liability
                self.state.cash += cost
    
    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary"""
        return {
            'total_value': self.state.total_value,
            'cash': self.state.cash,
            'n_positions': len(self.state.positions),
            'drawdown': self.state.drawdown,
            'return_pct': (self.state.total_value - self.initial_capital) / self.initial_capital * 100,
            'positions': [
                {
                    'symbol': pos.symbol,
                    'side': pos.side,
                    'size': pos.size,
                    'pnl': pos.pnl,
                    'pnl_pct': pos.pnl_percent
                }
                for pos in self.state.positions.values()
            ]
        }
