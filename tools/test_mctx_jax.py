import jax
import jax.numpy as jnp
from mcts_trader.jax_agent.muzero_model import MuZeroNetworks
from mcts_trader.jax_agent.mctx_search import create_mctx_search
import time

def test_mctx_search():
    print("🚀 Testing JAX/MCTX MuZero Search on GPU...")
    
    # Initialize random key
    rng = jax.random.PRNGKey(42)

    # Create model
    model = MuZeroNetworks(
        state_dim=64,
        action_space_size=192,
        support_size=601
    )

    # Initialize model parameters
    print("Initializing model...")
    rng, init_rng = jax.random.split(rng)
    print(f"DEBUG: init_rng type: {type(init_rng)}, shape: {init_rng.shape}")
    dummy_obs = jnp.ones((4, 27, 64, 64))  # batch=4
    # Use mode='initialize_parameters' to initialize all parameters including dynamics
    # Pass observation positionally to match __call__ signature
    variables = model.init({'params': init_rng}, dummy_obs, mode='initialize_parameters')
    params = variables['params']

    # Create search
    print("JIT-compiling MCTS search...")
    search = create_mctx_search(
        model,
        num_simulations=50,  # 50 sims
        use_gumbel=True
    )

    # Prepare input
    # MCTX expects observations to be ready for root function
    
    # Warmup compilation
    start_time = time.time()
    rng, search_rng = jax.random.split(rng)
    actions, probs, stats = search(params, search_rng, dummy_obs)
    jax.block_until_ready(actions)
    compile_time = time.time() - start_time
    print(f"✅ Compilation time: {compile_time:.2f}s")

    # Benchmark
    print("Running benchmark (100 steps)...")
    start_time = time.time()
    for _ in range(100):
        rng, search_rng = jax.random.split(rng)
        actions, probs, stats = search(params, search_rng, dummy_obs)
        jax.block_until_ready(actions)
    
    total_time = time.time() - start_time
    avg_time = total_time / 100 * 1000  # ms
    
    print(f"✅ Benchmark completed:")
    print(f"   Batch size: 4")
    print(f"   Simulations: 50")
    print(f"   Avg Latency: {avg_time:.2f} ms per batch")
    print(f"   Throughput: {4 * 1000 / avg_time:.1f} inferences/sec")
    
    print(f"Actions shape: {actions.shape}")
    print(f"Probs shape: {probs.shape}")
    
    assert actions.shape == (4,)
    assert probs.shape == (4, 192)
    
    print("✅ MCTX Search Test Passed!")

if __name__ == "__main__":
    test_mctx_search()
