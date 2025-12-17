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
from mcts_trader.jax_agent.mcts_worker_pool import MCTSWorkerPool

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

        # Handle duplicates for padding
        self.padded_symbols = asset_symbols
        unique_symbols = None
        if self.padded_symbols:
            unique_symbols = list(set(self.padded_symbols))
            self.n_assets = len(self.padded_symbols) # Force size to match padded list
        else:
            self.n_assets = n_assets

        # FORCE LOAD FX PAIRS (Standard Format for DB)
        # We found that EUR/USD has 2M+ rows, while ZEURZUSD has 700.
        # Use standard symbols for loading, but map correctly.
        self.fx_map = {
            'EUR/USD': 'EUR/USD',
            'GBP/USD': 'GBP/USD',
            'USD/CAD': 'USD/CAD', # Check if exists
            '/USDCAD': 'USD/CAD',
            'USD/CHF': 'USD/CHF',
            '/USDCHF': 'USD/CHF',
            'USD/JPY': 'USD/JPY',
            '/USDJPY': 'USD/JPY',
            'AUD/USD': 'AUD/USD',
            'AUDUSD': 'AUD/USD',
            # Keep Kraken ones just in case, but mapped to standard
            'ZEURZUSD': 'EUR/USD',
            'ZGBPZUSD': 'GBP/USD'
        }
        
        # Add FX pairs to unique_symbols if they aren't already there
        if unique_symbols:
            for fx_sym in self.fx_map.keys():
                if fx_sym not in unique_symbols:
                    unique_symbols.append(fx_sym)

        # Components (CPU encoder!)
        # Initialize market with UNIQUE symbols (DB limitation)
        self.market = HistoricalMarket(
            db_path, 
            n_assets=len(unique_symbols) if unique_symbols else n_assets, 
            min_required_assets=min_required_assets,
            specific_symbols=unique_symbols,
            start_date=start_date,
            end_date=end_date
        )
        
        # If no explicit symbols provided, grab what market found
        if not self.padded_symbols:
            self.padded_symbols = list(self.market.assets.values())
            # Filter out the forced FX pairs from the "tradeable" set if we auto-discovered
            # (Optional, but keeps action space clean if we don't want to trade FX)
            # For now, let's just keep them if they were auto-discovered.
            self.n_assets = len(self.padded_symbols)
        
        # Encoder gets the FULL PADDED list so it generates correct shape (e.g. 64x64)
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

        # Asset mapping: Action Index -> Symbol (handles duplicates)
        self.action_idx_to_symbol = {i: sym for i, sym in enumerate(self.padded_symbols)}
        
        # We assume HistoricalMarket loads what we asked.
        # But we need to map from our Padded Index to DB ID if we needed exact IDs.
        # But step() below uses symbol string lookup, so we are good.

        # Register assets in encoder (FULL PADDED LIST)
        self.encoder.register_assets(self.padded_symbols)

        # Build symbol dict for portfolio logging (idx -> symbol)
        self.symbol_map = self.action_idx_to_symbol

        self.steps_taken = 0
        
        # Forward Fill cache for FX rates (handles weekends/gaps)
        self.last_fx_rates = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset market
        self.market.reset()
        
        # Reset FX cache on new episode (or keep? No, new episode might jump in time)
        self.last_fx_rates = {}

        # Reset portfolio with fees and slippage
        self.portfolio = PortfolioManager(
            initial_capital=self.initial_capital,
            trading_fee_bps=self.trading_fee_bps,
            slippage_bps=self.slippage_bps,
            mode=self.mode,
            n_assets=self.n_assets  # Must match padded count
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

    def _normalize_kraken_symbol(self, symbol):
        """Convert Kraken symbols (XXBTZUSD) to standard (BTC/USD)"""
        # Common mappings
        if symbol == 'XXBTZUSD': return 'BTC/USD'
        if symbol == 'XETHZUSD': return 'ETH/USD'
        if symbol == 'ZEURZUSD': return 'EUR/USD'
        if symbol == 'ZGBPZUSD': return 'GBP/USD'
        if symbol == 'ZUSDZCAD': return 'USD/CAD'
        if symbol == 'ZUSDZJPY': return 'USD/JPY'
        if symbol == 'XXRPZUSD': return 'XRP/USD'
        if symbol == 'XLTCZUSD': return 'LTC/USD'
        if symbol == 'XXMRZUSD': return 'XMR/USD'
        if symbol == 'XZECZUSD': return 'ZEC/USD'
        
        # Generic cleaning
        s = symbol
        if len(s) == 8:
            # Try removing X/Z prefixes for standard ISO/Crypto pairs
            # e.g. XXBT ZUSD -> XBT USD -> BTC USD
            # This is heuristics, safer to use explicit map above for majors
            pass
            
        return symbol

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

        # Convert current_data from {symbol: data} to {action_idx: price}
        # This handles duplicates correctly: Both 'BTC' at idx 0 and 'BTC' at idx 60 get the same price
        current_prices_by_idx = {}
        fx_rates = {}

        for symbol, data in current_data.items():
            price = data['close']
            
            # Map Kraken FX symbols to standard rates
            if symbol in self.fx_map:
                std_pair = self.fx_map[symbol]
                
                if std_pair == 'EUR/USD':
                    fx_rates['EUR/USD'] = price
                elif std_pair == 'GBP/USD':
                    fx_rates['GBP/USD'] = price
                elif std_pair == 'AUD/USD':
                    fx_rates['AUD/USD'] = price
                elif std_pair == 'USD/CAD':
                    fx_rates['CAD/USD'] = 1.0 / price if price > 0 else 0
                elif std_pair == 'USD/CHF':
                    fx_rates['CHF/USD'] = 1.0 / price if price > 0 else 0
                elif std_pair == 'USD/JPY':
                    fx_rates['JPY/USD'] = 1.0 / price if price > 0 else 0

            # Store ALL symbols as potential FX rates (e.g. BTC/USD is needed for ETH/BTC quote conversion)
            # NORMALIZE KEYS: XXBTZUSD -> BTC/USD
            norm_sym = self._normalize_kraken_symbol(symbol)
            fx_rates[norm_sym] = price
            # Also keep original just in case
            if norm_sym != symbol:
                fx_rates[symbol] = price
        
        # Map prices to ALL indices that use this symbol
        for idx, sym in self.action_idx_to_symbol.items():
            if sym in current_data:
                current_prices_by_idx[idx] = current_data[sym]['close']

        # Build FX rates - first check for direct FX pairs
        for symbol, data in current_data.items():
            if symbol in ['EUR/USD', 'EURUSD', 'GBP/USD', 'GBPUSD', 'CHF/USD', 'CHFUSD',
                         'CAD/USD', 'CADUSD', 'JPY/USD', 'JPYUSD', 'AUD/USD', 'AUDUSD']:
                fx_rates[symbol] = data['close']
            elif symbol.startswith('USD/') or symbol.startswith('USD'):
                quote = symbol.replace('USD/', '').replace('USD', '')
                if quote:
                    fx_rates[f'{quote}/USD'] = 1.0 / data['close'] if data['close'] > 0 else 1.0

        # --- KEY UPDATE: Load authoritative FX rates from DB ---
        # This replaces the ad-hoc derivation from crypto bridges
        db_fx_rates = self.market.get_current_fx_rates()
        if db_fx_rates:
            fx_rates.update(db_fx_rates)
            
        # Update last known FX rates with current ones
        self.last_fx_rates.update(fx_rates)
        
        # Use merged rates (Forward Fill)
        # This prevents 'Using fallback FX rate' warnings on weekends when EUR/USD is closed but Crypto is open
        effective_fx_rates = self.last_fx_rates.copy()
        effective_fx_rates.update(fx_rates)

        # Update portfolio FX rates
        self.portfolio.fx_rates = effective_fx_rates

        # Execute if we have price data
        if len(current_prices_by_idx) > 0:
            self.portfolio.execute_actions(actions, sizes, current_prices_by_idx)

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


def run_self_play(env, search_fn, model_apply_fn, params, rng_key, steps=200, confidence_threshold=0.0):
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

        # Confidence Thresholding (Gate-keeping)
        if confidence_threshold > 0:
            probs = np.array(policy_jax[0])
            max_prob = np.max(probs)
            if max_prob < confidence_threshold:
                # Force HOLD: (asset_idx * 3) + 1
                asset_idx = action_int // 3
                action_int = (asset_idx * 3) + 1

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
    parser.add_argument('--slippage', type=int, default=5, help='Slippage in bps')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--steps', type=int, default=200, help='Steps per episode (200=3h, 1440=24h)')
    parser.add_argument('--config', type=str, help='Training config JSON (dynamic assets)')
    parser.add_argument('--start-date', type=str, help='Start date filter (YYYY-MM-DD or unix ts)')
    parser.add_argument('--end-date', type=str, help='End date filter (YYYY-MM-DD or unix ts)')
    parser.add_argument('--load-checkpoint', type=str, help='Path to checkpoint to resume from (absolute or relative)')
    
    # MCTS Hyperparameters (Maniac Mode)
    parser.add_argument('--sims', type=int, default=16, help='MCTS simulations')
    parser.add_argument('--noise', type=float, default=0.3, help='Dirichlet noise alpha')
    parser.add_argument('--noise-frac', type=float, default=0.25, help='Dirichlet noise fraction')
    
    # Advanced Training Parameters
    parser.add_argument('--batch-size', type=int, default=256, help='Training batch size')
    parser.add_argument('--confidence-threshold', type=float, default=0.0, help='Minimum policy confidence to execute trade (force Hold otherwise)')
    
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
    print(f"JAX TRADING AGENT | {args.mode} | Fee: {args.fee}bps | Slip: {args.slippage}bps | {n_assets} Assets")
    print("="*70)
    print()

    print("[1/4] Initializing Environment...")
    env = FastTradingEnv(
        n_assets=n_assets,
        initial_capital=10000.0,
        trading_fee_bps=args.fee,
        slippage_bps=args.slippage,
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
        num_simulations=args.sims,  # Dynamic sims
        batch_size=args.batch_size,      # Dynamic batch size
        unroll_steps=5,
        td_steps=10
    )

    rng, init_rng = jax.random.split(rng)
    dummy_obs = jnp.ones((1, 27, n_assets, n_assets))
    state = trainer.init_train_state(init_rng, dummy_obs)

    # Resume from checkpoint if provided
    if args.load_checkpoint:
        # Resolve to absolute path for Orbax
        abs_ckpt_path = os.path.abspath(args.load_checkpoint)
        print(f"📥 Loading checkpoint from: {args.load_checkpoint}")
        print(f"   (Resolved absolute path: {abs_ckpt_path})")
        
        try:
            # MuZeroTrainer.restore_checkpoint returns the restored state
            restored_state = trainer.restore_checkpoint(abs_ckpt_path)
            if restored_state:
                trainer.state = restored_state
                state = restored_state # Update local reference
                print(f"✅ Checkpoint loaded successfully! (Step: {state.step_count})")
            else:
                print(f"⚠️ Failed to load checkpoint (trainer returned None)")
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            # Do not crash, continue from scratch if load fails? Or crash?
            # Crashing is safer to avoid accidental scratch training
            raise e

    # Create search with Maniac params
    # PARALLEL MCTS INTEGRATION
    model_kwargs = {
        'state_dim': n_assets,
        'action_space_size': n_assets*3,
        'support_size': 601
    }
    
    search_kwargs = {
        'num_simulations': args.sims,
        'use_gumbel': True,
        'dirichlet_alpha': args.noise,
        'exploration_fraction': args.noise_frac
    }
    
    print(f"🚀 Initializing Parallel MCTS with 4 workers...")
    search = MCTSWorkerPool(
        model_cls=MuZeroNetworks,
        model_kwargs=model_kwargs,
        search_kwargs=search_kwargs,
        n_workers=4
    )
    
    print(f"✅ Model ready ({sum(x.size for x in jax.tree_util.tree_leaves(state.params)):,} params)")
    print(f"   Maniac Mode: Sims={args.sims}, Noise={args.noise}, Frac={args.noise_frac}")
    print()

    print("[3/4] Initializing replay buffer...")
    replay_buffer = JAXReplayBuffer(
        capacity=10000,
        observation_shape=(27, n_assets, n_assets),
        action_size=n_assets * 3,  # Must match actual loaded assets
        batch_size=args.batch_size,
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

    # Track metrics for scheduler
    recent_rewards = []

    try:
        for epoch in range(args.epochs):
            # Self-play
            start = time.time()
            rng, search_rng = jax.random.split(rng)
            trajectory, score = run_self_play(
                env, 
                search, 
                model.apply, 
                trainer.state.params, 
                search_rng, 
                steps=args.steps,
                confidence_threshold=args.confidence_threshold
            )
            self_play_time = time.time() - start

            # Track reward
            recent_rewards.append(score)
            if len(recent_rewards) > 20:
                recent_rewards.pop(0)

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

                # Action Stats
                actions_arr = np.array(trajectory['actions'])
                n_sell = np.sum(actions_arr % 3 == 0)
                n_hold = np.sum(actions_arr % 3 == 1)
                n_buy = np.sum(actions_arr % 3 == 2)

                print(f"Epoch {epoch+1:4d} | Reward: {score:+7.2f} | Fee: {env.trading_fee_bps:2d} | Steps: {len(trajectory['actions']):3d} | "
                      f"B/H/S: {n_buy}/{n_hold}/{n_sell} | "
                      f"Self-play: {self_play_time:5.1f}s | Train: {train_time:4.1f}s | "
                      f"Loss: {loss:.4f} | Val: {val_loss:.4f} | Pol: {pol_loss:.4f} | Rew: {rew_loss:.4f}")

                if (epoch + 1) % 10 == 0:
                    trainer.save_checkpoint(checkpoint_dir)
                    print(f"  ✓ Checkpoint saved to {checkpoint_dir}")
            else:
                # Action Stats for warmup too
                actions_arr = np.array(trajectory['actions'])
                n_sell = np.sum(actions_arr % 3 == 0)
                n_hold = np.sum(actions_arr % 3 == 1)
                n_buy = np.sum(actions_arr % 3 == 2)
                
                print(f"Epoch {epoch+1:4d} | Reward: {score:+7.2f} | Fee: {env.trading_fee_bps:2d} | Steps: {len(trajectory['actions']):3d} | "
                      f"B/H/S: {n_buy}/{n_hold}/{n_sell} | "
                      f"Self-play: {self_play_time:5.1f}s | (warmup)")
                      
    finally:
        search.close()
        print("🛑 MCTS Worker Pool closed.")

    print()
    print("="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    
    # Save metrics for scheduler
    avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0
    metrics_data = {
        'avg_reward': float(avg_reward),
        'final_loss': float(loss) if 'loss' in locals() else 0.0
    }
    import json
    with open('last_run_metrics.json', 'w') as f:
        json.dump(metrics_data, f)
    print(f"📊 Saved metrics: {metrics_data}")

    print()
    print("="*70)
    print("TRAINING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    # Needed for torch import in encoder
    import torch
    main()