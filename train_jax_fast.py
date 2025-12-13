import jax
import jax.numpy as jnp
import numpy as np
import time
import os
import sys
import logging
import argparse

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


class FastTradingEnv(gym.Env):
    """Lightweight trading environment with fast encoder"""

    def __init__(
        self,
        db_path: str = "trading_data.db",
        n_assets: int = 64,
        lookback: int = 60,
        initial_capital: float = 10000.0,
        trading_fee_bps: int = 26,
        slippage_bps: int = 5,
        mode: str = 'MARGIN',
        asset_symbols: list = None,
        max_steps: int = 200,  # Steps per episode (200=3h, 1440=24h)
        min_required_assets: int = 6,
        start_date: str = None,
        end_date: str = None
    ):
        super().__init__()
        self.n_actions_per_asset = 3
        self.initial_capital = initial_capital
        self.trading_fee_bps = trading_fee_bps
        self.max_steps = max_steps
        self.slippage_bps = slippage_bps
        self.mode = mode

        # Components (CPU encoder!)
        # Initialize market FIRST to determine actual available assets
        self.market = HistoricalMarket(
            db_path, 
            n_assets=n_assets, 
            min_required_assets=min_required_assets,
            specific_symbols=asset_symbols,
            start_date=start_date,
            end_date=end_date
        )
        # Update n_assets to match actual market assets (in case fewer were found)
        self.n_assets = self.market.n_assets
        
        self.encoder = FastMultiAssetStateEncoder(n_assets=self.n_assets, lookback=lookback, device='cpu')
        self.portfolio = None


        # Action space
        self.action_space = spaces.Discrete(self.n_assets * 3)

        # Observation space
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(27, self.n_assets, self.n_assets),
                dtype=np.float32
            ),
            'action_mask': spaces.Box(
                low=0, high=1,
                shape=(self.n_assets * 3,),
                dtype=np.int8
            ),
            'to_play': spaces.Box(
                low=-1, high=1,
                shape=(1,),
                dtype=np.int8
            )
        })

        # Asset mapping
        self.market_asset_ids_ordered = list(self.market.assets.keys())
        self.db_id_to_action_idx = {db_id: i for i, db_id in enumerate(self.market_asset_ids_ordered)}
        self.action_idx_to_db_id = {i: db_id for i, db_id in enumerate(self.market_asset_ids_ordered)}

        # Register assets in encoder
        symbols = [self.market.id_to_symbol[db_id] for db_id in self.market_asset_ids_ordered]
        self.encoder.register_assets(symbols)

        # Build symbol dict for portfolio logging (idx -> symbol)
        self.symbol_map = {i: sym for i, sym in enumerate(symbols)}

        self.steps_taken = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset market
        self.market.reset()

        # Reset portfolio with fees and slippage
        self.portfolio = PortfolioManager(
            initial_capital=self.initial_capital,
            trading_fee_bps=self.trading_fee_bps,
            slippage_bps=self.slippage_bps,
            mode=self.mode,
            n_assets=self.n_assets  # Must match actual loaded assets
        )
        # Register symbols for readable logging
        self.portfolio.register_assets(self.symbol_map)

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
        fx_rates = {}

        for symbol, data in current_data.items():
            price = data['close']

            # Extract FX rates for non-USD quote currencies
            # e.g., BTC/EUR price tells us EUR/USD rate indirectly
            # But we need direct rates like EUR/USD, GBP/USD
            if symbol.endswith('/USD') or symbol.endswith('USD'):
                # This gives us the base asset price in USD directly
                pass
            elif '/EUR' in symbol or '/GBP' in symbol or '/CAD' in symbol or '/CHF' in symbol or '/JPY' in symbol:
                # Extract quote currency for FX tracking
                parts = symbol.split('/')
                if len(parts) == 2:
                    quote = parts[1]
                    # Store the pair price (we'll derive FX from USD pairs)
                    fx_rates[symbol] = price

            # Find the db_id for this symbol
            for db_id, sym in self.market.id_to_symbol.items():
                if sym == symbol:
                    # Map asset index to price (portfolio uses 0-based indices)
                    if db_id in self.db_id_to_action_idx:
                        idx = self.db_id_to_action_idx[db_id]
                        current_prices_by_id[idx] = price
                    break

        # Build FX rates - first check for direct FX pairs
        for symbol, data in current_data.items():
            if symbol in ['EUR/USD', 'EURUSD', 'GBP/USD', 'GBPUSD', 'CHF/USD', 'CHFUSD',
                         'CAD/USD', 'CADUSD', 'JPY/USD', 'JPYUSD', 'AUD/USD', 'AUDUSD']:
                fx_rates[symbol] = data['close']
            elif symbol.startswith('USD/') or symbol.startswith('USD'):
                quote = symbol.replace('USD/', '').replace('USD', '')
                if quote:
                    fx_rates[f'{quote}/USD'] = 1.0 / data['close'] if data['close'] > 0 else 1.0

        # Derive FX rates from crypto pairs if not available
        # If we have BTC/USD and BTC/EUR, then EUR/USD = BTC/USD / BTC/EUR
        base_usd_prices = {}  # base_asset -> USD price
        base_other_prices = {}  # (base_asset, quote_currency) -> price

        for symbol, data in current_data.items():
            parts = symbol.replace('/', ' ').split()
            if len(parts) != 2:
                continue
            base, quote = parts[0], parts[1]

            if quote == 'USD':
                base_usd_prices[base] = data['close']
            elif quote in ['EUR', 'GBP', 'CAD', 'CHF', 'JPY', 'AUD']:
                base_other_prices[(base, quote)] = data['close']

        # Calculate cross rates
        for (base, quote), price in base_other_prices.items():
            if base in base_usd_prices and price > 0:
                # quote/USD = base/USD / base/quote
                derived_rate = base_usd_prices[base] / price
                rate_key = f'{quote}/USD'
                if rate_key not in fx_rates:
                    fx_rates[rate_key] = derived_rate

        # Update portfolio FX rates
        self.portfolio.fx_rates = fx_rates

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
        truncated = self.steps_taken >= self.max_steps

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

        # Use Portfolio's mask logic now (includes Spot vs Margin constraints)
        # Note: We need to flatten it because gym space is flattened
        # Portfolio returns (n_assets, 3)
        if self.portfolio:
            mask = self.portfolio.get_valid_actions().flatten().astype(np.int8)
        else:
            # Fallback if portfolio not init yet
            mask = np.ones(self.action_space.n, dtype=np.int8)

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
        
        # Prepare Mask
        mask_array = obs_dict['action_mask']
        mask_array = np.expand_dims(mask_array, axis=0)
        mask_jax = jnp.array(mask_array, dtype=jnp.int8)

        # MCTS
        action_jax, policy_jax, stats = search_fn(params, step_rng, obs_jax, mask_jax)

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
    parser = argparse.ArgumentParser(description='JAX MuZero Training')
    parser.add_argument('--name', type=str, default='default', help='Experiment name')
    parser.add_argument('--mode', type=str, default='MARGIN', choices=['SPOT', 'MARGIN'], help='Trading mode')
    parser.add_argument('--fee', type=int, default=26, help='Trading fee in bps')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--steps', type=int, default=200, help='Steps per episode (200=3h, 1440=24h)')
    parser.add_argument('--config', type=str, help='Training config JSON (dynamic assets)')
    parser.add_argument('--start-date', type=str, help='Start date filter (YYYY-MM-DD or unix ts)')
    parser.add_argument('--end-date', type=str, help='End date filter (YYYY-MM-DD or unix ts)')
    args = parser.parse_args()

    # Dynamic Asset Configuration
    n_assets = 64
    asset_symbols = None
    min_coverage = 0.1  # Default 10%
    start_date = args.start_date
    end_date = args.end_date
    
    if args.config:
        import json
        with open(args.config) as f:
            config = json.load(f)
            if 'n_assets' in config:
                n_assets = config['n_assets']
            if 'asset_symbols' in config:
                asset_symbols = config['asset_symbols']
            if 'experiment_name' in config:
                args.name = config['experiment_name']
            if 'min_coverage' in config:
                min_coverage = config.get('min_coverage', 0.1)
            # Config overrides CLI args if present
            if 'start_date' in config:
                start_date = config['start_date']
            if 'end_date' in config:
                end_date = config['end_date']
                
            print(f"🔧 Loaded config: {args.config} (n_assets={n_assets}, coverage={min_coverage:.2f})")

    # Calculate min_required_assets
    min_required_assets = max(1, int(n_assets * min_coverage))
    print(f"DEBUG: Min required assets set to {min_required_assets} ({min_coverage*100:.1f}% coverage)")

    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join('logs', f"training_fast_{args.name}.log")), # Log per experiment
            logging.StreamHandler()
        ]
    )

    print("="*70)
    print(f"JAX TRADING AGENT | {args.mode} | {args.fee}bps | {n_assets} Assets")
    print("="*70)
    print()

    print("[1/4] Initializing Environment...")
    env = FastTradingEnv(
        n_assets=n_assets,
        initial_capital=10000.0,
        trading_fee_bps=args.fee,
        slippage_bps=5,
        mode=args.mode,
        asset_symbols=asset_symbols,
        max_steps=args.steps,
        min_required_assets=min_required_assets,
        start_date=start_date,
        end_date=end_date
    )
    # Update n_assets to match what the environment actually loaded
    n_assets = env.n_assets
    print(f"✅ Environment ready ({args.mode} mode, {n_assets} assets, {args.steps} steps/episode)")
    print(f"DEBUG: n_assets={n_assets}, expected_actions={n_assets*3}")

    print("[2/4] Initializing JAX model...")
    rng = jax.random.PRNGKey(42)

    model = MuZeroNetworks(state_dim=n_assets, action_space_size=n_assets*3, support_size=601)
    # We can't access model.trm yet because setup() is called during apply/init
    print(f"DEBUG: Model definition: state_dim={model.state_dim}, actions={model.action_space_size}")
    
    trainer = MuZeroTrainer(
        model=model,
        learning_rate=2e-5,  # Reduced from 1e-4 for stability
        num_simulations=16,  # Reduced for speed
        batch_size=256,      # Increased for GPU
        unroll_steps=5,
        td_steps=10
    )

    rng, init_rng = jax.random.split(rng)
    dummy_obs = jnp.ones((1, 27, n_assets, n_assets))
    state = trainer.init_train_state(init_rng, dummy_obs)

    search = create_mctx_search(model, num_simulations=16, use_gumbel=True)
    print(f"✅ Model ready ({sum(x.size for x in jax.tree_util.tree_leaves(state.params)):,} params)")
    print()

    print("[3/4] Initializing replay buffer...")
    replay_buffer = JAXReplayBuffer(
        capacity=10000,
        observation_shape=(27, n_assets, n_assets),
        action_size=n_assets * 3,  # Must match actual loaded assets
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

    checkpoint_dir = os.path.abspath(f"jax_ckpt_fast/{args.name}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(args.epochs):
        # Self-play
        start = time.time()
        rng, search_rng = jax.random.split(rng)
        trajectory, score = run_self_play(env, search, model.apply, trainer.state.params, search_rng, steps=args.steps)
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
                print(f"  ✓ Checkpoint saved to {checkpoint_dir}")
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
