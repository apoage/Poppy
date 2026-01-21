import os
import sys
import warnings
import time
import logging
import jax
import jax.numpy as jnp
import numpy as np

# GPU Optimization settings
_trader_dir = os.path.dirname(os.path.abspath(__file__))
_xla_cache = os.path.join(_trader_dir, '.xla_cache')
_jax_cache = os.path.join(_trader_dir, '.jax_cache')
os.makedirs(_xla_cache, exist_ok=True)
os.makedirs(_jax_cache, exist_ok=True)

os.environ['XLA_FLAGS'] = f'--xla_gpu_autotune_level=4 --xla_gpu_per_fusion_autotune_cache_dir={_xla_cache}'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['JAX_COMPILATION_CACHE_DIR'] = _jax_cache

warnings.filterwarnings('ignore', category=FutureWarning, module='jax')

# JAX Configuration
jax.config.update("jax_compilation_cache_dir", _jax_cache)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

# Import refactored components
from mcts_trader.training.cli import parse_args
from mcts_trader.training.setup import setup_environment, setup_model_and_trainer, setup_jax_environment
from mcts_trader.jax_agent.mctx_search import create_mctx_search
from mcts_trader.jax_agent.replay_buffer import JAXReplayBuffer
from mcts_trader.jax_agent.smart_allocation import compute_env_start_offsets
from mcts_trader.training.self_play import run_self_play, run_self_play_batched, run_self_play_jax, N_ACTIONS_PER_ASSET

def main():
    # 1. Parse configuration
    config = parse_args()
    
    # 2. Setup logging
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join('logs', f"training_fast_{config.name}.log")),
            logging.StreamHandler()
        ]
    )
    
    print("="*70)
    print(f"JAX TRADING AGENT | {config.mode} | Refactored Core")
    print("="*70)

    # 3. Initialize Environment
    print("[1/4] Initializing Environment...")
    
    jax_env_states = None
    jax_prices = None
    jax_env_config = None
    env = None
    parallel_envs = []
    period_data = None
    n_assets = 64

    use_jax_env = getattr(config, 'use_jax_env', False)
    if use_jax_env:
        if config.synthetic_level is not None:
            print("   Initializing JAX-vectorized environment...")
            jax_env_states, jax_prices, jax_env_config = setup_jax_environment(config)
            n_assets = jax_env_config.n_assets
        else:
            logging.warning("--jax-env requires --synthetic-level. Falling back to Python env.")
            use_jax_env = False

    if not use_jax_env:
        env, parallel_envs, period_data = setup_environment(config)
        n_assets = env.n_assets
    
    action_space_size = n_assets * 5
    
    # 4. Initialize Model and Trainer
    print("[2/4] Initializing JAX model...")
    model, trainer = setup_model_and_trainer(config, n_assets)
    
    # 5. Initialize Search and Buffer
    search = create_mctx_search(model, num_simulations=config.sims, use_gumbel=True)
    search = jax.jit(search)
    
    replay_buffer = JAXReplayBuffer(
        capacity=5000,
        observation_shape=(27, n_assets, n_assets),
        action_size=action_space_size,
        batch_size=config.batch_size,
        n_assets=n_assets
    )
    
    # 6. Initialize parameters
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    dummy_obs = jnp.ones((1, 27, n_assets, n_assets))
    state = trainer.init_train_state(init_rng, dummy_obs)
    
    # 7. Warmup
    print("[4/4] Warmup (JIT compilation)...")
    if jax_env_states is not None:
        _, _ = run_self_play_jax(
            jax_env_states, jax_prices, jax_env_config,
            search, model.apply, trainer.state.params, rng,
            steps=5, collect_prices=config.use_prediction_loss
        )
    else:
        _, _ = run_self_play(env, search, model.apply, trainer.state.params, rng, steps=5)
    print("✅ Warmup done")

    # 8. Simplified Training Loop (Router logic)
    print("="*70)
    print("TRAINING LOOP")
    print("="*70)
    
    all_rewards = []
    checkpoint_dir = os.path.abspath(f"jax_ckpt_fast/{config.name}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    try:
        for epoch in range(config.epochs):
            start = time.time()
            rng, search_rng = jax.random.split(rng)
            
            # TASK_064: Smart allocation & Period Swapping
            start_offsets = None
            if config.smart_allocation and period_data:
                current_period_idx = period_data['current_idx']
                period_lengths = period_data['lengths']
                
                new_period_idx, start_offsets = compute_env_start_offsets(
                    period_lengths=period_lengths,
                    n_envs=config.parallel_envs,
                    n_steps=config.steps,
                    epoch=epoch,
                    jitter_minutes=config.jitter_minutes,
                    seed=epoch
                )
                
                # Check for period switch
                if new_period_idx != current_period_idx:
                    print(f"   [STRATIFIED] Switching to period P{new_period_idx}")
                    new_prices = period_data['prices'][new_period_idx]
                    for penv in parallel_envs:
                        penv.set_prices(new_prices)
                    period_data['current_idx'] = new_period_idx

            # Self-play
            if jax_env_states is not None:
                trajectories, scores = run_self_play_jax(
                    jax_env_states, jax_prices, jax_env_config,
                    search, model.apply, trainer.state.params, search_rng,
                    steps=config.steps,
                    collect_prices=config.use_prediction_loss
                )
                score = np.mean(scores)
            elif config.parallel_envs > 1:
                trajectories, scores = run_self_play_batched(
                    parallel_envs, search, model.apply, trainer.state.params, search_rng, steps=config.steps,
                    collect_prices=config.use_prediction_loss,
                    start_offsets=start_offsets
                )
                score = np.mean(scores)
            else:
                trajectory, score = run_self_play(
                    env, search, model.apply, trainer.state.params, search_rng, steps=config.steps,
                    collect_prices=config.use_prediction_loss
                )
                trajectories = [trajectory]
            
            self_play_time = time.time() - start
            all_rewards.append(score)
            
            # Buffer update
            for traj in trajectories:
                replay_buffer.add_trajectory(
                    jnp.array(traj['observations']), jnp.array(traj['actions']),
                    jnp.array(traj['rewards']), jnp.array(traj['values']),
                    jnp.array(traj['policy_probs']), jnp.array(traj['dones'])
                )
                if config.use_prediction_loss and 'prices' in traj:
                    replay_buffer.add_auxiliary(prices=traj['prices'], next_prices=traj['next_prices'])

            # Train
            train_start = time.time()
            metrics = trainer.train(replay_buffer, num_iterations=50, rng=rng)
            train_time = time.time() - train_start
            
            # Quick summary log
            loss = metrics['total_loss'][-1]
            print(f"Epoch {epoch+1:4d} | Reward: {score:+7.2f} | Loss: {loss:.4f} | "
                  f"Self-play: {self_play_time:5.1f}s | Train: {train_time:4.1f}s")
            
            # Checkpoint
            if (epoch + 1) % 10 == 0 and not config.no_checkpoint:
                trainer.save_checkpoint(checkpoint_dir)

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted.")
    finally:
        if not config.no_checkpoint:
            trainer.save_checkpoint(checkpoint_dir)

if __name__ == "__main__":
    import torch # Required for encoder
    main()
