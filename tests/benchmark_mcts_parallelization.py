import time
import jax
import jax.numpy as jnp
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath('.'))

from mcts_trader.jax_agent.mctx_search import create_mctx_search
from mcts_trader.jax_agent.muzero_model import MuZeroNetworks
from mcts_trader.jax_agent.mcts_worker_pool import MCTSWorkerPool

def benchmark():
    print("🚀 Starting MCTS Benchmark...")
    
    # Setup
    n_assets = 64
    action_space = n_assets * 3
    model = MuZeroNetworks(state_dim=n_assets, action_space_size=action_space, support_size=601)
    
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    
    # Init dummy params
    dummy_obs = jnp.ones((1, 27, n_assets, n_assets))
    dummy_action = jnp.zeros((1,), dtype=jnp.int32)
    
    # Use 'initialize_parameters' mode to ensure dynamics params are created
    variables = model.init(init_rng, dummy_obs, action=dummy_action, mode='initialize_parameters')
    params = variables['params']
    
    # Create dummy batch
    batch_size = 256 # Must be big enough to justify parallel overhead
    obs = jnp.ones((batch_size, 27, n_assets, n_assets))
    masks = jnp.ones((batch_size, action_space), dtype=jnp.int8)
    
    print(f"📦 Batch size: {batch_size}, Actions: {action_space}")

    # Test 1: Serial (current)
    print("\n[1/2] Running Serial MCTS...")
    base_search = create_mctx_search(
        model, 
        num_simulations=16, # Reduced sims for quicker bench
        use_gumbel=True,
        dirichlet_alpha=0.3,
        exploration_fraction=0.25
    )
    
    # JIT compile first
    print("   Compiling Serial...")
    start = time.time()
    _ = base_search(params, rng, obs[:2], masks[:2]) # Small batch for JIT
    print(f"   Compile time: {time.time() - start:.2f}s")

    print("   Running Benchmark...")
    start = time.time()
    _ = base_search(params, rng, obs, masks)
    serial_time = time.time() - start
    print(f"   ⏱️  Serial Time: {serial_time:.2f}s")

    # Test 2: Parallel (new)
    print("\n[2/2] Running Parallel MCTS (4 workers)...")
    
    model_kwargs = {
        'state_dim': n_assets,
        'action_space_size': action_space,
        'support_size': 601
    }
    
    search_kwargs = {
        'num_simulations': 16,
        'use_gumbel': True,
        'dirichlet_alpha': 0.3,
        'exploration_fraction': 0.25
    }
    
    search_parallel = MCTSWorkerPool(
        model_cls=MuZeroNetworks,
        model_kwargs=model_kwargs,
        search_kwargs=search_kwargs,
        n_workers=4
    )

    print("   Running Benchmark...")
    start = time.time()
    _ = search_parallel(params, rng, obs, masks)
    parallel_time = time.time() - start
    print(f"   ⏱️  Parallel Time: {parallel_time:.2f}s")

    print("\n" + "="*30)
    print(f"Speedup: {serial_time / parallel_time:.2f}x")
    print("="*30)

    search_parallel.close()

if __name__ == "__main__":
    benchmark()
