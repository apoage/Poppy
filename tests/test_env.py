
import pytest
import gym
import numpy as np
from easydict import EasyDict
from mcts_trader.env.trading_env import TradingEnv
from ding.envs import BaseEnvTimestep

# Mock config
mock_cfg = EasyDict(
    env_id='MultiAssetTrading-v0',
    n_assets=64,
    n_actions_per_asset=3,
    collector_env_num=1,
    evaluator_env_num=1,
    n_evaluator_episode=1,
    stop_value=100000,
    db_path='trading_data.db' 
)

def test_env_initialization():
    env = TradingEnv(**mock_cfg)
    assert env.n_assets == 64
    assert env.action_space.n == 192
    env.close()

def test_env_reset():
    env = TradingEnv(**mock_cfg)
    obs = env.reset()
    
    # Check observation structure for MuZero
    assert isinstance(obs, dict)
    assert 'observation' in obs
    assert 'action_mask' in obs
    assert 'to_play' in obs
    
    # Check shapes
    assert obs['observation'].shape == (27, 64, 64)
    assert obs['action_mask'].shape == (192,)
    env.close()

def test_env_step():
    env = TradingEnv(**mock_cfg)
    env.reset()
    
    # Random action
    action = env.action_space.sample()
    timestep = env.step(action)
    
    assert isinstance(timestep, BaseEnvTimestep)
    assert timestep.obs['observation'].shape == (27, 64, 64)
    assert isinstance(timestep.reward, (float, np.float32, np.float64))
    assert isinstance(timestep.done, bool)
    env.close()
