import copy
from easydict import EasyDict
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Tuple, Optional, List
import torch
import logging

from ..agent.portfolio import PortfolioManager
from ..encoder.state import MultiAssetStateEncoder
from ..data.history import HistoricalMarket
from ding.utils import ENV_REGISTRY

@ENV_REGISTRY.register('trading_env')
class TradingEnv(gym.Env):
    """
    Gym environment for Multi-Asset Trading with TRM+MCTS.
    Uses historical data from local SQLite DB.
    """
    config = dict(
        db_path="trading_data.db",
        n_assets=64,
        n_actions_per_asset=3,
        lookback=60,
        initial_capital=10000.0,
        collector_env_num=8,
        evaluator_env_num=3,
        n_evaluator_episode=3,
        stop_value=100000,
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    @classmethod
    def create_collector_env_cfg(cls: type, cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num', 1)
        cfg = copy.deepcopy(cfg)
        cfg.is_train = True
        return [cfg for _ in range(collector_env_num)]

    @classmethod
    def create_evaluator_env_cfg(cls: type, cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num', 1)
        cfg = copy.deepcopy(cfg)
        cfg.is_train = False
        return [cfg for _ in range(evaluator_env_num)]
    
    def __init__(
        self,
        db_path: str = "trading_data.db",
        n_assets: int = 64,
        n_actions_per_asset: int = 3, # Sell, Hold, Buy
        lookback: int = 60,
        initial_capital: float = 10000.0,
        device: str = 'cpu', # Default to cpu to avoid PyTorch/JAX GPU conflicts
        trading_fee_bps: int = 26, # Conservative default (0.26% Taker)
        slippage_bps: int = 5,     # Conservative default (0.05%)
        asset_symbols: list = None, # NEW: explicit list of symbols
        **kwargs
    ):
        super().__init__()
        self.n_assets = n_assets
        self.n_actions_per_asset = n_actions_per_asset
        self.initial_capital = initial_capital
        self.device = device 
        self.trading_fee_bps = trading_fee_bps
        self.slippage_bps = slippage_bps
        
        # Action space: select (asset, action) pair
        self.action_space = spaces.Discrete(n_assets * n_actions_per_asset)
        
        # Observation space: Dict for MuZero
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(27, 64, 64), # Fixed size assumption? Or should use n_assets?
                # Ideally use n_assets, but for CNN we often fix it.
                # Assuming n_assets matches 64 for now or caller handles it.
                dtype=np.float32
            ),
            'action_mask': spaces.Box(
                low=0, high=1, 
                shape=(n_assets * n_actions_per_asset,), 
                dtype=np.int8
            ),
            'to_play': spaces.Box(
                low=-1, high=1, 
                shape=(1,), 
                dtype=np.int8
            )
        })
        
        self.reward_space = spaces.Box(
            low=-float("inf"), high=float("inf"), 
            shape=(1,), dtype=np.float32
        )
        
        # Components
        # Handle duplicate assets for padding
        self.padded_symbols = asset_symbols
        unique_symbols = None
        
        if self.padded_symbols:
            # If explicit list provided, use it (and respect length including duplicates)
            self.n_assets = len(self.padded_symbols)
            # Filter unique for DB query
            unique_symbols = list(set(self.padded_symbols))
        
        # Init Market with UNIQUE symbols only (DB doesn't understand duplicates)
        self.market = HistoricalMarket(
            db_path, 
            n_assets=len(unique_symbols) if unique_symbols else n_assets, 
            specific_symbols=unique_symbols
        )
        
        # If we didn't have symbols before, get them from market
        if not self.padded_symbols:
            self.padded_symbols = list(self.market.assets.values())
            self.n_assets = len(self.padded_symbols)

        # Use specified device for encoder
        # Encoder uses the PADDED list (so it generates 64 rows if padded)
        self.encoder = MultiAssetStateEncoder(n_assets=self.n_assets, lookback=lookback, device=self.device)
        self.portfolio = None # Init in reset
        
        # Register assets with encoder
        self.encoder.register_assets(self.padded_symbols)
        self.portfolio_manager_cls = PortfolioManager
        
        # Create mapping: Action Index -> Symbol (DB ID no longer primary key for Agent)
        self.action_idx_to_symbol = {i: sym for i, sym in enumerate(self.padded_symbols)}
        
    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        # Handle seed from either arg or self._seed
        if seed is None and hasattr(self, '_seed'):
            seed = self._seed
            
        super().reset(seed=seed)
        
        # Reset Market time
        market_data = self.market.reset()
        
        # Reset components
        self.portfolio = self.portfolio_manager_cls(
            initial_capital=self.initial_capital, 
            n_assets=self.n_assets,
            trading_fee_bps=self.trading_fee_bps,
            slippage_bps=self.slippage_bps
        )
        
        # Register assets using Action Index as ID
        # Portfolio expects {id: symbol}
        self.portfolio.register_assets(self.action_idx_to_symbol)
        
        # Warm up encoder (feed 60 minutes of data)
        # In a real efficient impl, we'd load this in batch from DB
        # For now, just step 60 times
        # logging.info("Warming up encoder...")
        for _ in range(60):
            self.step_market_only()
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info

    def step_market_only(self):
        """Advance market and update encoder without executing actions"""
        data = self.market.step()
        if data:
            self.encoder.update(data)

    def step(self, action: int):
        # Decode action
        asset_idx = action // self.n_actions_per_asset
        action_type = action % self.n_actions_per_asset
        # action_type: 0=Sell, 1=Hold, 2=Buy
        
        # Get current data from market
        current_data = self.market.get_current_data()
        
        # Map Market Data (Key=Symbol) to Portfolio Input (Key=ActionIndex)
        current_prices_for_portfolio = {}
        fx_rates = {}
        
        for idx, sym in self.action_idx_to_symbol.items():
            if sym in current_data:
                current_prices_for_portfolio[idx] = current_data[sym]['close']
                
                # Capture FX rates
                if 'USD' in sym:
                    fx_rates[sym] = current_data[sym]['close']
        
        # Execute action
        actions = np.ones(self.n_assets, dtype=int) # Default Hold (1)
        sizes = np.zeros(self.n_assets, dtype=float)
        
        actions[asset_idx] = action_type
        sizes[asset_idx] = 0.5 # Fixed size for now
        
        # Capture value before execution/step
        prev_value = self.portfolio.state.total_value
        
        # Execute orders
        self.portfolio.execute_actions(actions, sizes, current_prices_for_portfolio, fx_rates=fx_rates)
        
        # Advance time
        self.step_market_only()
        
        # Update portfolio prices to new market state to calc Reward
        # We need new prices for ALL padded assets
        new_data = self.market.get_current_data() # Market already stepped
        new_prices = {}
        new_fx_rates = {}
        
        for idx, sym in self.action_idx_to_symbol.items():
            if sym in new_data:
                new_prices[idx] = new_data[sym]['close']
                if 'USD' in sym:
                    new_fx_rates[sym] = new_data[sym]['close']

        self.portfolio.update_prices(new_prices, fx_rates=new_fx_rates)
        
        # Calculate Reward
        new_value = self.portfolio.state.total_value
        reward = (new_value - prev_value) / prev_value * 100 # Percentage return
        
        # Clip huge rewards (artifacts)
        reward = np.clip(reward, -10, 10)
        
        terminated = False 
        truncated = False
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> Dict:
        state = self.encoder.encode()
        obs = state['combined']
        
        # If on GPU, move to CPU numpy for returning to LightZero Collector
        # (Collector expects numpy, but we encode on GPU for speed)
        if hasattr(obs, 'cpu'):
            obs = obs.cpu().numpy()

        # Simple action mask - all actions valid (faster than 192 lookups)
        # Market data gaps are rare in our filtered dataset
        mask = np.ones(self.action_space.n, dtype=np.int8)
        
        # MuZero expects dict with observation and action_mask
        return {
            'observation': obs,
            'action_mask': mask,
            'to_play': -1 # Scalar for 1-player/environment games
        }
    
    def _get_info(self) -> Dict:
        return self.portfolio.get_portfolio_summary()
    
    def close(self):
        self.market.close()