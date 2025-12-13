#!/usr/bin/env python3
"""
JAX MuZero Training - ULTRA FAST VERSION

Key optimizations:
1. FastMultiAssetStateEncoder (no correlations, 10-20x faster)
2. Reduced MCTS simulations (50 -> 16)
3. Larger batch size (128 -> 256) for GPU efficiency
4. CPU encoder (avoids PyTorch/JAX GPU conflict)
5. Warmup compilation step

Expected performance:
- Self-play: ~30-60 seconds per episode (vs 9 minutes before)
- Training: ~2-3 seconds per iteration
- Total: ~1 minute per epoch
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import os
import sys
import logging

# Configure JAX
cache_dir = os.path.expanduser('~/.cache/jax_compilation_cache')
os.makedirs(cache_dir, exist_ok=True)
os.environ['JAX_COMPILATION_CACHE_DIR'] = cache_dir

# Import JAX components
from mcts_trader.jax_agent.muzero_model import MuZeroNetworks
from mcts_trader.jax_agent.trainer import MuZeroTrainer
from mcts_trader.jax_agent.replay_buffer import JAXReplayBuffer
from mcts_trader.jax_agent.mctx_search import create_mctx_search

# Import environment components
from mcts_trader.data.history import HistoricalMarket
from mcts_trader.encoder.state_fast import FastMultiAssetStateEncoder  # Fast encoder!
from mcts_trader.agent.portfolio import PortfolioManager
import gymnasium as gym
from gymnasium import spaces

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('training_fast.log'),
        logging.StreamHandler()
    ]
)


class FastTradingEnv(gym.Env):
    """Lightweight trading environment with fast encoder"""

    def __init__(
        self,
        db_path: str = "trading_data.db",
        n_assets: int = 64,
        lookback: int = 60,
        initial_capital: float = 10000.0
    ):
        super().__init__()
        self.n_assets = n_assets
        self.n_actions_per_asset = 3
        self.initial_capital = initial_capital

        # Action space
        self.action_space = spaces.Discrete(n_assets * 3)

        # Observation space
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(27, 64, 64),
                dtype=np.float32
            ),
            'action_mask': spaces.Box(
                low=0, high=1,
                shape=(n_assets * 3,),
                dtype=np.int8
            ),
            'to_play': spaces.Box(
                low=-1, high=1,
                shape=(1,),
                dtype=np.int8
            )
        })

        # Components (CPU encoder!)
        self.market = HistoricalMarket(db_path, n_assets=n_assets)
        self.encoder = FastMultiAssetStateEncoder(n_assets=n_assets, lookback=lookback, device='cpu')
        self.portfolio = None

        # Asset mapping
        self.market_asset_ids_ordered = list(self.market.assets.keys())
        self.db_id_to_action_idx = {db_id: i for i, db_id in enumerate(self.market_asset_ids_ordered)}
        self.action_idx_to_db_id = {i: db_id for i, db_id in enumerate(self.market_asset_ids_ordered)}

        # Register assets in encoder
        symbols = [self.market.id_to_symbol[db_id] for db_id in self.market_asset_ids_ordered]
        self.encoder.register_assets(symbols)

        self.steps_taken = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset market
        self.market.reset()

        # Reset portfolio
        self.portfolio = PortfolioManager(self.initial_capital)

        # Warmup encoder (60 steps minimum)
        for _ in range(60):
            data = self.market.get_current_data()
            if data:
                self.encoder.update(data)
            self.market.step()

        self.steps_taken = 0
        return self._get_observation(), {}

    def step(self, action: int):
        # Decode action
        asset_idx = action // self.n_actions_per_asset
        action_type = action % self.n_actions_per_asset

        # Get current portfolio value
        prev_value = self.portfolio.state.total_value

        # Execute action using portfolio's execute_actions API
        current_data = self.market.get_current_data()

        # Build actions and sizes arrays for ALL 64 assets (portfolio expects full arrays)
        actions = np.ones(self.n_assets, dtype=np.int32)  # Default: Hold for all
        sizes = np.full(self.n_assets, 0.5, dtype=np.float32)  # Position size

        # Set the specific action for the selected asset
        actions[asset_idx] = action_type  # 0=Sell, 1=Hold, 2=Buy

        # Convert current_data from {symbol: data} to {db_id: price}
        # Portfolio expects prices keyed by database ID (int), not symbol (str)
        current_prices_by_id = {}
        for symbol, data in current_data.items():
            # Find the db_id for this symbol
            for db_id, sym in self.market.id_to_symbol.items():
                if sym == symbol:
                    # Map asset index to price (portfolio uses 0-based indices)
                    if db_id in self.db_id_to_action_idx:
                        idx = self.db_id_to_action_idx[db_id]
                        current_prices_by_id[idx] = data['close']
                    break

        # Execute if we have price data
        if len(current_prices_by_id) > 0:
            self.portfolio.execute_actions(actions, sizes, current_prices_by_id)

        # Step market
        next_data = self.market.step()
        terminated = len(next_data) == 0

        if not terminated:
            self.encoder.update(next_data)

        # Calculate reward
        new_value = self.portfolio.state.total_value

        reward = (new_value - prev_value) / prev_value * 100
        reward = np.clip(reward, -10, 10)

        self.steps_taken += 1
        truncated = self.steps_taken >= 200

        return self._get_observation(), reward, terminated, truncated, {}

    def _get_observation(self):
        obs = self.encoder.encode()

        if hasattr(obs, 'cpu'):
            obs = obs.cpu().numpy()
        elif isinstance(obs, torch.Tensor):
            obs = obs.numpy()

        # Dynamic action mask
        current_data = self.market.get_current_data()
        available_symbols = set(current_data.keys())

        mask = np.zeros(self.action_space.n, dtype=np.int8)
        for action_idx in range(self.action_space.n):
            asset_idx = action_idx // self.n_actions_per_asset
            if hasattr(self, 'action_idx_to_db_id'):
                db_id = self.action_idx_to_db_id.get(asset_idx)
                if db_id:
                    symbol = self.market.id_to_symbol.get(db_id)
                    if symbol and symbol in available_symbols:
                        mask[action_idx] = 1
            else:
                mask[action_idx] = 1

        return {
            'observation': obs,
            'action_mask': mask,
            'to_play': -1
        }


def run_self_play(env, search_fn, model_apply_fn, params, rng_key, steps=200):
    """Self-play with timing"""
    obs_dict, info = env.reset()

    observations, actions, rewards, values, policy_probs, dones = [], [], [], [], [], []
    score = 0.0
    rng = rng_key

    for step in range(steps):
        rng, step_rng = jax.random.split(rng)

        # Prepare observation
        obs_array = obs_dict['observation']
        if hasattr(obs_array, 'cpu'):
            obs_array = obs_array.cpu().numpy()
        obs_array = np.expand_dims(obs_array, axis=0)
        obs_jax = jnp.array(obs_array, dtype=jnp.float32)

        # MCTS
        action_jax, policy_jax, stats = search_fn(params, step_rng, obs_jax)

        # Value
        if 'root_value' in stats:
            value_scalar = stats['root_value'][0]
        else:
            output = model_apply_fn({'params': params}, obs_jax, mode='initial_inference')
            support = jnp.linspace(-10, 10, 601)
            value_probs = jax.nn.softmax(output['value'][0])
            value_scalar = jnp.sum(value_probs * support)

        # Execute
        action_int = int(action_jax[0])
        next_obs_dict, reward, terminated, truncated, info = env.step(action_int)
        done = terminated or truncated

        # Store
        observations.append(obs_array[0])
        actions.append(action_int)
        rewards.append(reward)
        values.append(float(value_scalar))
        policy_probs.append(np.array(policy_jax[0]))
        dones.append(done)

        score += reward
        obs_dict = next_obs_dict

        if done:
            break

    return {
        'observations': np.array(observations),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'values': np.array(values),
        'policy_probs': np.array(policy_probs),
        'dones': np.array(dones)
    }, score


def main():
    print("="*70)
    print("JAX MuZero Training - ULTRA FAST VERSION")
    print("="*70)
    print()

    # Initialize
    print("[1/4] Initializing environment...")
    env = FastTradingEnv(n_assets=64, lookback=60)
    print(f"✅ Environment ready (CPU encoder, 27 channels)")
    print()

    print("[2/4] Initializing JAX model...")
    rng = jax.random.PRNGKey(42)

    model = MuZeroNetworks(state_dim=64, action_space_size=192, support_size=601)
    trainer = MuZeroTrainer(
        model=model,
        learning_rate=2e-5,  # Reduced from 1e-4 for stability
        num_simulations=16,  # Reduced for speed
        batch_size=256,      # Increased for GPU
        unroll_steps=5,
        td_steps=10
    )

    rng, init_rng = jax.random.split(rng)
    dummy_obs = jnp.ones((1, 27, 64, 64))
    state = trainer.init_train_state(init_rng, dummy_obs)

    search = create_mctx_search(model, num_simulations=16, use_gumbel=True)
    print(f"✅ Model ready ({sum(x.size for x in jax.tree_util.tree_leaves(state.params)):,} params)")
    print()

    print("[3/4] Initializing replay buffer...")
    replay_buffer = JAXReplayBuffer(
        capacity=10000,
        observation_shape=(27, 64, 64),
        batch_size=256,
        sequence_length=5
    )
    print("✅ Buffer ready (10K capacity, float32)")
    print()

    print("[4/4] Warmup (JIT compilation)...")
    warmup_start = time.time()
    rng, warmup_rng = jax.random.split(rng)
    _, _ = run_self_play(env, search, model.apply, trainer.state.params, warmup_rng, steps=5)
    print(f"✅ Warmup done ({time.time() - warmup_start:.1f}s)")
    print()

    print("="*70)
    print("TRAINING LOOP")
    print("="*70)
    print()

    checkpoint_dir = os.path.abspath("jax_ckpt_fast")

    for epoch in range(1000):
        # Self-play
        start = time.time()
        rng, search_rng = jax.random.split(rng)
        trajectory, score = run_self_play(env, search, model.apply, trainer.state.params, search_rng, steps=200)
        self_play_time = time.time() - start

        # Add to buffer
        # Sanity check for NaNs in trajectory
        for key, arr in trajectory.items():
            if np.isnan(arr).any():
                print(f"!!! CRITICAL WARNING: NaNs detected in {key} from self-play!")
                # Optional: Don't add to buffer if bad?
                # continue
        
        replay_buffer.add_trajectory(
            jnp.array(trajectory['observations']),
            jnp.array(trajectory['actions']),
            jnp.array(trajectory['rewards']),
            jnp.array(trajectory['values']),
            jnp.array(trajectory['policy_probs']),
            jnp.array(trajectory['dones'])
        )

        # Training
        if epoch >= 5:
            train_start = time.time()
            metrics = trainer.train(replay_buffer, num_iterations=50, rng=rng)
            train_time = time.time() - train_start
            loss = metrics['total_loss'][-1]
            val_loss = metrics['value_loss'][-1]
            pol_loss = metrics['policy_loss'][-1]
            rew_loss = metrics['reward_loss'][-1]

            print(f"Epoch {epoch+1:4d} | Reward: {score:+7.2f} | Steps: {len(trajectory['actions']):3d} | "
                  f"Self-play: {self_play_time:5.1f}s | Train: {train_time:4.1f}s | "
                  f"Loss: {loss:.4f} | Val: {val_loss:.4f} | Pol: {pol_loss:.4f} | Rew: {rew_loss:.4f}")

            if (epoch + 1) % 10 == 0:
                trainer.save_checkpoint(checkpoint_dir)
                print(f"  ✓ Checkpoint saved")
        else:
            print(f"Epoch {epoch+1:4d} | Reward: {score:+7.2f} | Steps: {len(trajectory['actions']):3d} | "
                  f"Self-play: {self_play_time:5.1f}s | (warmup)")

    print()
    print("="*70)
    print("TRAINING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    # Needed for torch import in encoder
    import torch
    main()
