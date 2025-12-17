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
        **kwargs
    ):
        super().__init__()
        self.n_assets = n_assets
        self.n_actions_per_asset = n_actions_per_asset
        self.initial_capital = initial_capital
        self.device = device # Store device config
        
        # Action space: select (asset, action) pair
        self.action_space = spaces.Discrete(n_assets * n_actions_per_asset)
        
        # Observation space: Dict for MuZero
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(27, 64, 64), 
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
        self.market = HistoricalMarket(db_path, n_assets=n_assets)
        # Use specified device for encoder
        self.encoder = MultiAssetStateEncoder(n_assets=n_assets, lookback=lookback, device=self.device)
        self.portfolio = None # Init in reset
        
        self.market_asset_ids_ordered = list(self.market.assets.keys()) # Store ordered list of db_ids
        self.db_id_to_action_idx = {db_id: i for i, db_id in enumerate(self.market_asset_ids_ordered)}
        self.action_idx_to_db_id = {i: db_id for i, db_id in enumerate(self.market_asset_ids_ordered)}
        
        # Register assets with encoder
        asset_symbols = list(self.market.assets.values())
        self.encoder.register_assets(asset_symbols)
        self.portfolio_manager_cls = PortfolioManager
        
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
            n_assets=self.n_assets
        )
        self.portfolio.register_assets(self.market.assets)
        
        # Warm up encoder (feed 60 minutes of data)
        # In a real efficient impl, we'd load this in batch from DB
        # For now, just step 60 times
        # logging.info("Warming up encoder...")
        for _ in range(60):
            data = self.market.step()
            if data:
                self.encoder.update(data)
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int):
        # Decode action
        asset_idx = action // self.n_actions_per_asset
        action_type = action % self.n_actions_per_asset
        # action_type: 0=Sell, 1=Hold, 2=Buy
        
        # Get current prices from encoder buffer (last close)
        # or fetch from market (which is current step)
        # Note: market.step() gives us data for the NEW minute (future relative to decision)
        # RL Convention: Observation(t) -> Action(t) -> Env updates -> Reward(t+1), Obs(t+1)
        # So we execute action at current price, THEN step market.
        
        current_data = self.market.get_current_data()
        current_prices_by_db_id = {
            db_id: current_data[sym]['close'] 
            for db_id, sym in self.market.assets.items() 
            if sym in current_data
        }
        
        # Convert keys from db_id to action_idx for PortfolioManager
        current_prices_for_portfolio = {
            self.db_id_to_action_idx[db_id]: price
            for db_id, price in current_prices_by_db_id.items()
            if db_id in self.db_id_to_action_idx # Ensure mapping exists
        }
        
        # Execute action
        actions = np.ones(self.n_assets, dtype=int) # Default Hold (1)
        sizes = np.zeros(self.n_assets, dtype=float)
        
        actions[asset_idx] = action_type
        sizes[asset_idx] = 0.5 # Fixed size for now
        
        # Capture value before execution/step
        prev_value = self.portfolio.state.total_value
        
        # Execute orders
        self.portfolio.execute_actions(actions, sizes, current_prices_for_portfolio)
        
        # Advance time
        new_data = self.market.step()
        if new_data:
            self.encoder.update(new_data)
            
            # Update portfolio prices to new market state to calc Reward
            new_prices = {
                idx: new_data[sym]['close'] 
                for idx, sym in self.market.assets.items() 
                if sym in new_data
            }
            self.portfolio.update_prices(new_prices)
        
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