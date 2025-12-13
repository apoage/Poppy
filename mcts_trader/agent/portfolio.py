import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

# Kraken minimum order sizes (in base currency)
# Source: Kraken API get_tradeable_pairs() ordermin field
# For pairs not listed, we use a conservative USD-equivalent minimum
MIN_ORDER_SIZES = {
    # BTC pairs
    'XBT/USD': 0.0001, 'XBT/EUR': 0.0001, 'XBT/USDT': 0.0001, 'XBT/USDC': 0.0001,
    'BTC/USD': 0.0001, 'BTC/EUR': 0.0001, 'BTC/CAD': 0.0001, 'BTC/GBP': 0.0001, 'BTC/CHF': 0.0001,
    'BTCCAD': 0.0001, 'BTCCHF': 0.0001, 'BTCGBP': 0.0001,
    # ETH pairs
    'ETH/USD': 0.01, 'ETH/EUR': 0.01, 'ETH/USDT': 0.01, 'ETH/GBP': 0.01,
    'ETHBTC': 0.01, 'ETHGBP': 0.01,
    # Major alts
    'SOL/USD': 0.02, 'SOL/EUR': 0.02,
    'ADA/USD': 5.0, 'ADA/EUR': 5.0,
    'DOT/USD': 0.6, 'DOT/EUR': 0.6,
    'LINK/USD': 0.3, 'LINK/EUR': 0.3,
    'ATOM/USD': 0.5, 'ATOM/EUR': 0.5,
    'XRP/USD': 2.0, 'XRP/EUR': 2.0, 'XRPBTC': 2.0,
    'LTC/USD': 0.05, 'LTC/EUR': 0.05, 'LTCBTC': 0.05,
    'BCH/USD': 0.01, 'BCH/EUR': 0.01, 'BCHBTC': 0.01,
    'ETC/USD': 0.25, 'ETC/EUR': 0.25,
    'XLM/USD': 12.0, 'XLM/EUR': 12.0, 'XLMBTC': 12.0,
    'XMR/USD': 0.02, 'XMR/EUR': 0.02, 'XMRBTC': 0.02,
    'XTZ/USD': 3.0,
    'KSM/USD': 0.12,
    'ALGO/USD': 10.0,
    'TRX/USD': 15.0, 'TRX/EUR': 15.0,
    'EOS/USD': 1.0,
    # Doge
    'XDG/USD': 15.0, 'XDG/EUR': 15.0, 'DOGE/USD': 15.0, 'DOGE/EUR': 15.0,
    # Stablecoins / Forex
    'USDT/USD': 5.0, 'USDC/USD': 5.0, 'DAI/USD': 5.0, 'DAI/EUR': 5.0,
    'USDC/USDT': 5.0,
    'EUR/USD': 1.0, 'GBP/USD': 5.0,
    'USDCAD': 5.0, 'USDCHF': 5.0, 'USDCEUR': 5.0, 'USDCGBP': 5.0,
    'USDTCHF': 5.0, 'USDTEUR': 5.0, 'USDTGBP': 5.0,
    'EURGBP': 1.0,
    # MATIC -> POL migration
    'MATIC/USD': 5.0, 'MATIC/EUR': 5.0, 'POL/USD': 5.0, 'POL/EUR': 5.0,
}

# Default minimum in USD value if symbol not found
DEFAULT_MIN_ORDER_USD = 5.0


def extract_quote_currency(symbol: str) -> str:
    """
    Extract quote currency from trading pair symbol.
    Examples:
        ETH/USD -> USD
        ETHBTC -> BTC
        XBT/EUR -> EUR
        USDTGBP -> GBP
    """
    # Handle slash format: BASE/QUOTE
    if '/' in symbol:
        return symbol.split('/')[1].upper()

    # Handle no-slash format: try to identify quote
    symbol = symbol.upper()

    # Common quote currencies (order matters - check longer ones first)
    quotes = ['USDT', 'USDC', 'USD', 'EUR', 'GBP', 'CAD', 'CHF', 'BTC', 'XBT', 'ETH']

    for quote in quotes:
        if symbol.endswith(quote):
            # Handle XBT -> BTC normalization
            if quote == 'XBT':
                return 'BTC'
            return quote

    # Default to USD if can't determine
    return 'USD'


def get_quote_to_usd_rate(quote_currency: str, fx_rates: dict) -> float:
    """
    Get conversion rate from quote currency to USD.

    Args:
        quote_currency: The quote currency (EUR, GBP, BTC, etc.)
        fx_rates: Dict of current FX rates, keyed by symbol

    Returns:
        Multiplier to convert quote currency to USD
    """
    quote = quote_currency.upper()

    # Normalize stablecoin variants to underlying currency
    # CEUR/TEUR = Circle/Tether EUR stablecoins, etc.
    # Also handle USDT pair notations (e.g., USDTCHF = USDT/CHF pair, quote is CHF)
    stablecoin_map = {
        'USDT': 'USD', 'USDC': 'USD', 'DAI': 'USD', 'BUSD': 'USD', 'TUSD': 'USD',
        'CEUR': 'EUR', 'TEUR': 'EUR', 'EURT': 'EUR',
        'CGBP': 'GBP', 'TGBP': 'GBP',
        'CAUD': 'AUD', 'TAUD': 'AUD',
        'CCAD': 'CAD', 'TCAD': 'CAD',
        'TCHF': 'CHF',
        # USDT pairs (e.g., /USDTCHF means USDT/CHF, quote currency is CHF)
        'USDTCHF': 'CHF', 'USDTGBP': 'GBP', 'USDTCAD': 'CAD',
        'USDTAUD': 'AUD', 'USDTEUR': 'EUR', 'USDTJPY': 'JPY',
        # USDC variants
        'USDCCHF': 'CHF', 'USDCGBP': 'GBP', 'USDCCAD': 'CAD',
        'USDCAUD': 'AUD', 'USDCEUR': 'EUR',
    }
    quote = stablecoin_map.get(quote, quote)

    if quote in ('USD', 'USDT', 'USDC'):
        return 1.0

    # Try to find direct rate
    # Look for QUOTE/USD or USD/QUOTE patterns
    direct_symbols = [
        f"{quote}/USD", f"{quote}USD",
        f"X{quote}/USD" if quote == 'BTC' else None,  # XBT/USD
        f"XBT/USD" if quote == 'BTC' else None,
    ]

    for sym in direct_symbols:
        if sym and sym in fx_rates:
            return fx_rates[sym]

    # Try inverse (USD/QUOTE means we divide)
    inverse_symbols = [f"USD/{quote}", f"USD{quote}"]
    for sym in inverse_symbols:
        if sym in fx_rates and fx_rates[sym] > 0:
            return 1.0 / fx_rates[sym]

    # No rate found - for training, use fallback. For real trading, this should fail.
    # TODO: Download actual FX pair data (EUR/USD, GBP/USD) from Kraken
    # These are rough 2024 estimates - better than crashing, but not accurate historically
    fallback_rates = {
        'EUR': 1.08, 'GBP': 1.27, 'CAD': 0.74, 'CHF': 1.13,
        'JPY': 0.0067, 'AUD': 0.65, 'BTC': 100000.0, 'XBT': 100000.0, 'ETH': 4000.0,
    }

    if quote in fallback_rates:
        # Only warn once per currency to avoid log spam
        if not hasattr(get_quote_to_usd_rate, '_warned'):
            get_quote_to_usd_rate._warned = set()
        if quote not in get_quote_to_usd_rate._warned:
            logging.warning(f"Using fallback FX rate for {quote}/USD: {fallback_rates[quote]} (inaccurate for historical data)")
            get_quote_to_usd_rate._warned.add(quote)
        return fallback_rates[quote]

    # Last resort - log once per currency to avoid spam
    if not hasattr(get_quote_to_usd_rate, '_warned_unknown'):
        get_quote_to_usd_rate._warned_unknown = set()
    if quote not in get_quote_to_usd_rate._warned_unknown:
        logging.warning(f"Unknown currency {quote}, using 1.0")
        get_quote_to_usd_rate._warned_unknown.add(quote)
    return 1.0


@dataclass
class Position:
    asset_idx: int
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    current_price: float = 0.0
    stop_loss: float = 0.0      # Price to close at loss
    take_profit: float = 0.0    # Price to close at profit
    highest_price: float = 0.0  # Track highest for trailing (longs)
    lowest_price: float = 0.0   # Track lowest for trailing (shorts)
    quote_currency: str = "USD" # Quote currency for FX conversion

    def __post_init__(self):
        # Initialize tracking prices
        if self.highest_price == 0.0:
            self.highest_price = self.entry_price
        if self.lowest_price == 0.0:
            self.lowest_price = self.entry_price

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

    def should_stop_loss(self) -> bool:
        """Check if stop loss triggered"""
        if self.stop_loss <= 0:
            return False
        if self.side == 'long':
            return self.current_price <= self.stop_loss
        else:  # short
            return self.current_price >= self.stop_loss

    def should_take_profit(self) -> bool:
        """Check if take profit triggered"""
        if self.take_profit <= 0:
            return False
        if self.side == 'long':
            return self.current_price >= self.take_profit
        else:  # short
            return self.current_price <= self.take_profit

    def update_trailing(self, current_price: float):
        """Update highest/lowest price for trailing calculations"""
        self.current_price = current_price
        if self.side == 'long':
            self.highest_price = max(self.highest_price, current_price)
        else:
            self.lowest_price = min(self.lowest_price, current_price) if self.lowest_price > 0 else current_price


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
    Includes realistic simulation of fees and slippage.
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        max_position_pct: float = 0.1,      # Max 10% per position
        max_total_exposure: float = 2.0,     # Max 200% total (with leverage)
        max_drawdown: float = 0.15,          # Stop trading at 15% drawdown
        n_assets: int = 64,
        trading_fee_bps: int = 10,           # 10 bps = 0.1%
        slippage_bps: int = 5,               # 5 bps = 0.05%
        # Stop-loss / Take-profit defaults (percentage from entry)
        default_stop_loss_pct: float = 5.0,  # 5% stop loss
        default_take_profit_pct: float = 10.0,  # 10% take profit
        # Tighter limits for shorts (unlimited loss potential)
        short_stop_loss_pct: float = 3.0,    # 3% stop loss for shorts
        short_take_profit_pct: float = 5.0,   # 5% take profit for shorts
        mode: str = 'MARGIN' # 'SPOT' or 'MARGIN'
    ):
        self.initial_capital = initial_capital
        self.max_position_pct = max_position_pct
        self.max_total_exposure = max_total_exposure
        self.max_drawdown = max_drawdown
        self.n_assets = n_assets
        self.trading_fee_rate = trading_fee_bps / 10000.0
        self.slippage_rate = slippage_bps / 10000.0
        self.mode = mode.upper()

        # SL/TP settings
        self.default_stop_loss_pct = default_stop_loss_pct
        self.default_take_profit_pct = default_take_profit_pct
        self.short_stop_loss_pct = short_stop_loss_pct
        self.short_take_profit_pct = short_take_profit_pct

        self.state = PortfolioState(cash=initial_capital)
        self.state.total_value = initial_capital
        self.state.max_value = initial_capital

        # Asset info
        self.symbols: Dict[int, str] = {}
        self.prices: Dict[int, float] = {}

        # FX rates for multi-currency conversion (symbol -> USD price)
        self.fx_rates: Dict[str, float] = {}
        
    def register_assets(self, symbols: Dict[int, str]):
        self.symbols = symbols

    def get_min_order_size(self, symbol: str, price: float) -> float:
        """
        Get minimum order size for a symbol.
        Returns size in base currency units.
        """
        # Try exact match first
        if symbol in MIN_ORDER_SIZES:
            return MIN_ORDER_SIZES[symbol]

        # Try without slash
        no_slash = symbol.replace('/', '')
        if no_slash in MIN_ORDER_SIZES:
            return MIN_ORDER_SIZES[no_slash]

        # Try common substitutions
        variants = [
            symbol.replace('BTC', 'XBT'),
            symbol.replace('DOGE', 'XDG'),
        ]
        for v in variants:
            if v in MIN_ORDER_SIZES:
                return MIN_ORDER_SIZES[v]

        # Fallback: use USD equivalent minimum
        # e.g., if price is $100 and min is $5, return 0.05 units
        if price > 0:
            return DEFAULT_MIN_ORDER_USD / price
        return 1.0  # Safe fallback
    
    def update_prices(self, prices: Dict[int, float], fx_rates: Dict[str, float] = None):
        """Update current prices and portfolio value with proper FX conversion"""
        self.prices = prices

        if fx_rates:
            self.fx_rates.update(fx_rates)

        long_value_usd = 0.0
        short_liability_usd = 0.0  # Cost to close short positions

        # Update position values with FX conversion
        for idx, pos in self.state.positions.items():
            if idx in prices:
                pos.current_price = prices[idx]

                # Get FX rate for this position's quote currency
                fx_rate = get_quote_to_usd_rate(pos.quote_currency, self.fx_rates)

                if pos.side == 'long':
                    # Long: current market value in quote currency -> USD
                    value_in_quote = pos.size * pos.current_price
                    long_value_usd += value_in_quote * fx_rate
                else:
                    # Short: calculate liability (cost to buy back)
                    # Liability = size * current_price
                    liability_in_quote = pos.size * pos.current_price
                    short_liability_usd += liability_in_quote * fx_rate

        # Total = Cash (USD) + Long Market Value (USD) - Short Liability (USD)
        # Note: Cash already includes the proceeds from opening the short
        self.state.total_value = self.state.cash + long_value_usd - short_liability_usd

        self.state.max_value = max(self.state.max_value, self.state.total_value)

    def reinforce_position(self, idx: int, action: int, current_price: float) -> bool:
        """
        When model signals same direction on existing position, trail the TP.
        action: 0=sell, 2=buy
        Returns True if TP was updated.
        """
        if idx not in self.state.positions:
            return False

        pos = self.state.positions[idx]

        # Check if signal reinforces position direction
        # Long position + Buy signal = reinforce
        # Short position + Sell signal = reinforce
        reinforces = (
            (pos.side == 'long' and action == 2) or
            (pos.side == 'short' and action == 0)
        )

        if not reinforces:
            return False

        # Update trailing tracker
        pos.update_trailing(current_price)

        # Calculate new TP from current price
        old_tp = pos.take_profit

        if pos.side == 'long':
            # Trail TP upward - new TP is current price + TP%
            new_tp = current_price * (1 + self.default_take_profit_pct / 100)
            # Only move TP up, never down
            if new_tp > old_tp:
                pos.take_profit = new_tp

            # BUG FIX [2025-12-12]: True Trailing Stop for Longs
            # New SL = Current * (1 - SL%)
            new_sl = current_price * (1 - self.default_stop_loss_pct / 100)
            # For Longs, tighter SL is HIGHER.
            if new_sl > pos.stop_loss:
                 pos.stop_loss = new_sl

        else:  # short
            # Trail TP downward - new TP is current price - TP%
            new_tp = current_price * (1 - self.short_take_profit_pct / 100)
            # Only move TP down (lower = more profit for short), never up
            if new_tp < old_tp:
                pos.take_profit = new_tp

            # BUG FIX [2025-12-12]: True Trailing Stop for Shorts
            # New SL = Current * (1 + SL%)
            new_sl = current_price * (1 + self.short_stop_loss_pct / 100)
            # For Shorts, tighter SL is LOWER.
            if new_sl < pos.stop_loss:
                 pos.stop_loss = new_sl

        return pos.take_profit != old_tp

    def check_stop_loss_take_profit(
        self,
        prices: Dict[int, float],
        fx_rates: Optional[Dict[str, float]] = None
    ) -> List[Dict]:
        """
        Check all positions for SL/TP triggers and close them.
        Returns list of closed orders.
        """
        self.update_prices(prices, fx_rates)
        closed_orders = []

        positions_to_close = []
        for idx, pos in self.state.positions.items():
            trigger_reason = None
            if pos.should_stop_loss():
                trigger_reason = 'stop_loss'
            elif pos.should_take_profit():
                trigger_reason = 'take_profit'

            if trigger_reason:
                positions_to_close.append((idx, pos, trigger_reason))

        # Close triggered positions
        for idx, pos, reason in positions_to_close:
            price = prices.get(idx, pos.current_price)
            
            # Get FX rate for this position's quote currency
            fx_rate = get_quote_to_usd_rate(pos.quote_currency, self.fx_rates)

            # Apply slippage
            if pos.side == 'long':
                exec_price = price * (1 - self.slippage_rate)
                value_quote = pos.size * exec_price
                value_usd = value_quote * fx_rate
                
                fee_usd = value_usd * self.trading_fee_rate
                self.state.cash += (value_usd - fee_usd)
                order_type = f'close_long_{reason}'
                
                # PnL in USD
                pnl_usd = (exec_price - pos.entry_price) * pos.size * fx_rate
                
            else: # Short
                exec_price = price * (1 + self.slippage_rate)
                value_quote = pos.size * exec_price
                value_usd = value_quote * fx_rate
                
                fee_usd = value_usd * self.trading_fee_rate
                self.state.cash -= (value_usd + fee_usd)
                order_type = f'close_short_{reason}'
                
                # PnL in USD (Entry - Exit)
                pnl_usd = (pos.entry_price - exec_price) * pos.size * fx_rate

            order = {
                'asset_idx': idx,
                'symbol': pos.symbol,
                'side': 'sell' if pos.side == 'long' else 'buy',
                'size': pos.size,
                'price': exec_price,
                'type': order_type,
                'reason': reason,
                'pnl': pnl_usd
            }
            closed_orders.append(order)
            del self.state.positions[idx]

        # Recalculate total value after closes
        if closed_orders:
            self.update_prices(prices)

        return closed_orders
    
    def get_valid_actions(self) -> np.ndarray:
        """
        Get mask of valid actions considering risk limits.
        
        Returns:
            mask: (n_assets, 3) - True if action is valid
        """
        mask = np.ones((self.n_assets, 3), dtype=bool)

        # Count current short positions (MAX 4 SHORTS LIMIT)
        n_shorts = sum(1 for pos in self.state.positions.values() if pos.side == 'short')
        n_longs = sum(1 for pos in self.state.positions.values() if pos.side == 'long')

        # Block opening new shorts if we have 4 already
        if n_shorts >= 4:
            for idx in range(self.n_assets):
                if idx not in self.state.positions:
                    # Can't open new short
                    mask[idx, 0] = False

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

        # BUG FIX [2025-12-12]: Enforce Global Leverage Limits
        # Calculate total exposure correctly with FX
        total_exposure_usd = 0.0
        for pos in self.state.positions.values():
             fx_rate = get_quote_to_usd_rate(pos.quote_currency, self.fx_rates)
             total_exposure_usd += pos.size * pos.current_price * fx_rate
        
        current_leverage = total_exposure_usd / self.state.total_value if self.state.total_value > 0 else 999.0

        # Check Total Exposure Limit
        if current_leverage >= self.max_total_exposure:
             for idx in range(self.n_assets):
                if idx not in self.state.positions:
                     mask[idx, 0] = False # Block Open Short
                     mask[idx, 2] = False # Block Open Long
                else:
                     pos = self.state.positions[idx]
                     if pos.side == 'long':
                          mask[idx, 2] = False # Block buying more
                     else:
                          mask[idx, 0] = False # Block selling more

        # Check Cash Limit (Spot Mode)
        if self.mode == 'SPOT' and self.state.cash < 10.0:
             for idx in range(self.n_assets):
                  if idx in self.state.positions:
                       if self.state.positions[idx].side == 'long':
                            mask[idx, 2] = False # Block buying more
                  else:
                       mask[idx, 2] = False # Block Open Long
        
        # Check per-position limits
        for idx in range(self.n_assets):
            if idx not in self.prices:
                mask[idx, :] = [False, True, False]  # Only hold if no price
                continue
            
            current_exposure = 0.0
            if idx in self.state.positions:
                pos = self.state.positions[idx]
                # CRITICAL: Must use FX-adjusted value for exposure calculation
                fx_rate = get_quote_to_usd_rate(pos.quote_currency, self.fx_rates)
                value_usd = pos.size * pos.current_price * fx_rate
                current_exposure = value_usd / self.state.total_value if self.state.total_value > 0 else 0
            
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
        prices: Dict[int, float],
        fx_rates: Optional[Dict[str, float]] = None # New parameter for FX rates
    ) -> List[Dict]:
        """
        Execute trading actions.
        
        Returns:
            List of executed orders
        """
        self.update_prices(prices, fx_rates)
        orders = []
        
        for idx in range(self.n_assets):
            action = actions[idx]
            size = sizes[idx]
            
            if action == 1:  # Hold
                continue
            
            symbol = self.symbols.get(idx, f'ASSET_{idx}')

            if idx not in prices:
                continue
            
            # SPOT MODE CHECK: Block Short Selling (Selling what we don't own)
            # We unmasked this in get_valid_actions to see the signal, but we must block it here.
            if self.mode == 'SPOT' and action == 0 and idx not in self.state.positions:
                # Log happens in caller (paper_trade_jax) usually, but we can log here too for clarity
                # actually paper_trade_jax handles the logging of blocked signals if 'executed_orders' is empty for that asset
                # But here we just skip it.
                # To ensure paper_trade_jax logs it as "SIGNAL", we just don't return an order.
                continue

            # Apply Slippage to Price
            # Buy acts at Ask (Price * 1+slippage)
            # Sell acts at Bid (Price * 1-slippage)
            raw_price = prices[idx]
            if action == 2: # Buy
                execution_price = raw_price * (1 + self.slippage_rate)
            else: # Sell
                execution_price = raw_price * (1 - self.slippage_rate)

            # Get quote currency and FX rate for proper sizing
            symbol = self.symbols.get(idx, f'ASSET_{idx}')
            quote_currency = extract_quote_currency(symbol)
            fx_rate = get_quote_to_usd_rate(quote_currency, self.fx_rates)

            # Calculate target position value in USD, then convert to quote currency units
            # max_value is in USD, but we need to divide by price*fx_rate to get units
            # Example: $500 USD / (0.0358 ETHBTC * $97000 USD/BTC) = 0.144 ETH units
            
            # Base desired value from model's size input
            base_desired_value_usd = self.state.total_value * self.max_position_pct * size

            # Adjust for SPOT mode: Cannot spend more cash than available
            if self.mode == 'SPOT' and action == 2: # Only applies to BUY in SPOT
                # Ensure we have enough cash, accounting for fees
                max_spendable_cash = self.state.cash / (1 + self.trading_fee_rate)
                if base_desired_value_usd > max_spendable_cash:
                    logging.info(f"Capping buy order for {symbol} due to insufficient cash. Desired: ${base_desired_value_usd:.2f}, Available: ${self.state.cash:.2f}")
                    max_value_usd = max_spendable_cash
                else:
                    max_value_usd = base_desired_value_usd
            else:
                max_value_usd = base_desired_value_usd

            # Ensure minimum size
            if max_value_usd < DEFAULT_MIN_ORDER_USD and max_value_usd > 0:
                logging.info(f"Order for {symbol} too small (desired ${max_value_usd:.2f}). Skipping.")
                continue

            target_size = max_value_usd / (execution_price * fx_rate)
            
            current_pos = self.state.positions.get(idx)
            
            if action == 2:  # Buy
                order = self._create_buy_order(idx, target_size, execution_price, current_pos)
            else:  # Sell
                order = self._create_sell_order(idx, target_size, execution_price, current_pos)
            
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
        symbol = self.symbols.get(idx, f'ASSET_{idx}')
        min_size = self.get_min_order_size(symbol, price)

        if current_pos is None:
            # Open long - check minimum
            if target_size < min_size:
                return None  # Order too small
            return {
                'asset_idx': idx,
                'symbol': symbol,
                'side': 'buy',
                'size': target_size,
                'price': price,
                'type': 'open_long'
            }
        elif current_pos.side == 'short':
            # Close short - always allow (closing positions)
            return {
                'asset_idx': idx,
                'symbol': symbol,
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
        symbol = self.symbols.get(idx, f'ASSET_{idx}')
        min_size = self.get_min_order_size(symbol, price)

        if current_pos is None:
            # Open short - check minimum
            if target_size < min_size:
                return None  # Order too small
            return {
                'asset_idx': idx,
                'symbol': symbol,
                'side': 'sell',
                'size': target_size,
                'price': price,
                'type': 'open_short'
            }
        elif current_pos.side == 'long':
            # Close long - always allow (closing positions)
            return {
                'asset_idx': idx,
                'symbol': symbol,
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
        order: {'asset_idx': int, 'side': 'buy'/'sell', 'size': float, 'price': float, 'type': str}
        """
        idx = order['asset_idx']
        price = order['price']
        size = order['size']
        order_type = order['type']
        
        # Transaction Value
        value = size * price
        
        # Fee
        fee = value * self.trading_fee_rate
        
        # Extract quote currency from symbol for FX conversion
        symbol = order.get('symbol', f'ASSET_{idx}')
        quote_currency = extract_quote_currency(symbol)

        # Get FX rate for proper USD accounting
        fx_rate = get_quote_to_usd_rate(quote_currency, self.fx_rates)

        # Transaction value in USD
        value_usd = value * fx_rate

        # Execute logic based on type
        if order_type == 'open_long':
            # Calculate SL/TP prices for longs
            stop_loss = price * (1 - self.default_stop_loss_pct / 100)
            take_profit = price * (1 + self.default_take_profit_pct / 100)

            self.state.positions[idx] = Position(
                asset_idx=idx,
                symbol=symbol,
                side='long',
                size=size,
                entry_price=price,
                current_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                quote_currency=quote_currency
            )
            # Cash decreases by cost + fee (in USD)
            fee_usd = value_usd * self.trading_fee_rate
            self.state.cash -= (value_usd + fee_usd)

        elif order_type == 'close_long':
            if idx in self.state.positions:
                del self.state.positions[idx]
                # Cash increases by proceeds - fee (in USD)
                fee_usd = value_usd * self.trading_fee_rate
                self.state.cash += (value_usd - fee_usd)

        elif order_type == 'open_short':
            # Calculate SL/TP prices for shorts (tighter limits!)
            stop_loss = price * (1 + self.short_stop_loss_pct / 100)
            take_profit = price * (1 - self.short_take_profit_pct / 100)

            self.state.positions[idx] = Position(
                asset_idx=idx,
                symbol=symbol,
                side='short',
                size=size,
                entry_price=price,
                current_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                quote_currency=quote_currency
            )
            # Short sale: Cash increases by proceeds - fee (in USD)
            fee_usd = value_usd * self.trading_fee_rate
            self.state.cash += (value_usd - fee_usd)

        elif order_type == 'close_short':
            if idx in self.state.positions:
                del self.state.positions[idx]
                # Buy to cover: Cash decreases by cost + fee (in USD)
                fee_usd = value_usd * self.trading_fee_rate
                self.state.cash -= (value_usd + fee_usd)
    
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
