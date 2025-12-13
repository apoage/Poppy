import gymnasium as gym
import logging
from ding.utils import ENV_REGISTRY
from ding.envs import BaseEnvTimestep
from mcts_trader.env.trading_env import TradingEnv

@ENV_REGISTRY.register('wrapped_trading_env')
class DICompatibleTradingEnv(gym.Wrapper):
    """
    A wrapper class registered with DI-engine that ensures compatibility 
    between Gymnasium (used by TradingEnv) and the Gym API expected by DI-engine.
    
    It proxies configuration methods to the underlying TradingEnv class.
    """
    
    # Proxy class methods required by DI-engine configuration system
    @classmethod
    def default_config(cls):
        return TradingEnv.default_config()

    @classmethod
    def create_collector_env_cfg(cls, cfg):
        return TradingEnv.create_collector_env_cfg(cfg)

    @classmethod
    def create_evaluator_env_cfg(cls, cfg):
        return TradingEnv.create_evaluator_env_cfg(cfg)

    def __init__(self, cfg=None):
        # Instantiate the actual environment using the config
        # Note: TradingEnv.__init__ expects kwargs or specific args, cfg is usually passed as dict
        if cfg is None:
            cfg = {}
        
        # TradingEnv expects arguments, not just a cfg dict sometimes, 
        # but DI-engine passes cfg as dict.
        # Let's check TradingEnv.__init__ signature:
        # def __init__(self, db_path=..., n_assets=..., ...)
        # DI-engine instantiates as: env_cls(cfg) if cfg is dict
        
        # We unpack cfg if it's a dictionary to pass as kwargs
        # But TradingEnv handles kwargs.
        
        self.env = TradingEnv(**cfg)
        super().__init__(self.env)
        
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.metadata = self.env.metadata
        self.reward_space = self.env.reward_space

    def step(self, action):
        # Unpack the 5-tuple from Gymnasium's step
        observation, reward, terminated, truncated, info = self.env.step(action)
        # Combine terminated and truncated into the old 'done' flag
        done = terminated or truncated
        return BaseEnvTimestep(observation, reward, done, info)

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        if hasattr(self.env, 'seed'):
            self.env.seed(seed, dynamic_seed)
        else:
            self._pending_seed = seed
    
    def reset(self, seed=None, options=None):
        if seed is None and hasattr(self, '_pending_seed'):
            seed = self._pending_seed
            del self._pending_seed
        # Call reset on wrapper (which calls env.reset), getting tuple
        # Then return just obs
        obs, info = self.env.reset(seed=seed, options=options)
        return obs