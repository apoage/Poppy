#!/usr/bin/env python3
"""
Benchmark MCTS Parallelization Performance

Tests different batch sizes and parallelization strategies to find bottlenecks.
"""

import time
import jax
import jax.numpy as jnp
import numpy as np
from mcts_trader.jax_agent.mcts_worker_pool import MCTSWorkerPool
from mcts_trader.jax_agent.muzero_model import MuZeroNetworks
from mcts_trader.jax_agent.mctx_search import create_mctx_search

def benchmark_batch_size(batch_size: int, n_workers: int = 4, num_simulations: int = 32):
    """Benchmark MCTS with specific batch size"""

    print(f"\n{'='*60}")
    print(f"Benchmarking: batch_size={batch_size}, workers={n_workers}, sims={num_simulations}")
    print(f"{'='*60}")

    # Initialize model
    model_kwargs = {
        'observation_shape': (27, 64, 64),
        'num_actions': 192,
        'num_channels': 64,
        'num_blocks': 4,
        'support_size': 10,
    }

    search_kwargs = {
        'num_simulations': num_simulations,
        'max_depth': 10,
        'discount': 0.997,
        'pb_c_init': 1.25,
        'use_gumbel': True,
    }

    # Create worker pool
    print("Creating worker pool...")
    pool = MCTSWorkerPool(
        model_cls=MuZeroNetworks,
        model_kwargs=model_kwargs,
        search_kwargs=search_kwargs,
        n_workers=n_workers
    )

    # Initialize model for local comparison
    print("Creating local model...")
    model = MuZeroNetworks(**model_kwargs)
    local_search = create_mctx_search(model, **search_kwargs)
    local_search_jit = jax.jit(local_search)

    # Create dummy data
    print(f"Creating dummy batch (size={batch_size})...")
    rng = jax.random.PRNGKey(0)
    params_rng, obs_rng, rng = jax.random.split(rng, 3)

    params = model.init(params_rng, jnp.zeros((1, 27, 64, 64)))
    observations = jax.random.normal(obs_rng, (batch_size, 27, 64, 64))
    action_masks = jnp.ones((batch_size, 192), dtype=jnp.float32)

    # Warmup
    print("Warming up...")
    _ = pool(params, rng, observations[:4], action_masks[:4])
    _ = local_search_jit(params, rng, observations[:1], action_masks[:1])

    # Benchmark worker pool
    print(f"\nBenchmarking WORKER POOL ({n_workers} workers)...")
    times_pool = []
    for i in range(5):
        start = time.time()
        actions, policies, stats = pool(params, rng, observations, action_masks)
        jax.block_until_ready(actions)  # Ensure computation is done
        elapsed = time.time() - start
        times_pool.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.3f}s ({batch_size/elapsed:.1f} samples/s)")

    avg_pool = np.mean(times_pool)
    std_pool = np.std(times_pool)

    # Benchmark local (serial) for comparison
    print(f"\nBenchmarking LOCAL (serial, single core)...")
    times_local = []
    for i in range(5):
        start = time.time()
        actions, policies, stats = local_search_jit(params, rng, observations, action_masks)
        jax.block_until_ready(actions)
        elapsed = time.time() - start
        times_local.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.3f}s ({batch_size/elapsed:.1f} samples/s)")

    avg_local = np.mean(times_local)
    std_local = np.std(times_local)

    # Calculate speedup
    speedup = avg_local / avg_pool

    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"  Worker Pool: {avg_pool:.3f}s ± {std_pool:.3f}s ({batch_size/avg_pool:.1f} samples/s)")
    print(f"  Local:       {avg_local:.3f}s ± {std_local:.3f}s ({batch_size/avg_local:.1f} samples/s)")
    print(f"  SPEEDUP:     {speedup:.2f}×")
    print(f"{'='*60}")

    pool.close()

    return {
        'batch_size': batch_size,
        'n_workers': n_workers,
        'num_simulations': num_simulations,
        'avg_pool': avg_pool,
        'avg_local': avg_local,
        'speedup': speedup,
        'throughput_pool': batch_size / avg_pool,
        'throughput_local': batch_size / avg_local,
    }

def test_worker_count(batch_size: int = 512):
    """Test different worker counts"""

    print("\n" + "="*60)
    print("TESTING WORKER COUNT SCALING")
    print("="*60)

    results = []
    for n_workers in [1, 2, 4, 8, 12, 16]:
        print(f"\n\nTesting {n_workers} workers...")
        try:
            result = benchmark_batch_size(batch_size, n_workers=n_workers, num_simulations=32)
            results.append(result)
        except Exception as e:
            print(f"ERROR with {n_workers} workers: {e}")
            continue

    # Print summary
    print("\n" + "="*60)
    print("WORKER COUNT SUMMARY")
    print("="*60)
    print(f"{'Workers':<10} {'Time (s)':<12} {'Speedup':<10} {'Throughput':<15}")
    print("-"*60)
    for r in results:
        print(f"{r['n_workers']:<10} {r['avg_pool']:<12.3f} {r['speedup']:<10.2f}× {r['throughput_pool']:<15.1f} samples/s")

    return results

def test_batch_sizes(n_workers: int = 4):
    """Test different batch sizes"""

    print("\n" + "="*60)
    print("TESTING BATCH SIZE SCALING")
    print("="*60)

    results = []
    for batch_size in [64, 128, 256, 512, 1024, 2048]:
        print(f"\n\nTesting batch size {batch_size}...")
        try:
            result = benchmark_batch_size(batch_size, n_workers=n_workers, num_simulations=32)
            results.append(result)
        except Exception as e:
            print(f"ERROR with batch {batch_size}: {e}")
            continue

    # Print summary
    print("\n" + "="*60)
    print("BATCH SIZE SUMMARY")
    print("="*60)
    print(f"{'Batch Size':<12} {'Time (s)':<12} {'Speedup':<10} {'Throughput':<15}")
    print("-"*60)
    for r in results:
        print(f"{r['batch_size']:<12} {r['avg_pool']:<12.3f} {r['speedup']:<10.2f}× {r['throughput_pool']:<15.1f} samples/s")

    return results

def analyze_vram_usage():
    """Analyze VRAM usage for multi-epoch training"""

    print("\n" + "="*60)
    print("VRAM USAGE ANALYSIS")
    print("="*60)

    # Check GPU memory
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            used, total = map(int, result.stdout.strip().split(', '))
            print(f"\nCurrent GPU Memory:")
            print(f"  Used:  {used} MB")
            print(f"  Total: {total} MB")
            print(f"  Free:  {total - used} MB")

            # Model size estimation
            model_size_mb = 7  # 7M params * 4 bytes (float32) / 1024^2 ≈ 27MB, but with overhead ~50MB
            batch_512_mb = 200  # Estimated activation memory for batch 512

            print(f"\nEstimated VRAM Requirements:")
            print(f"  Model (7M params):        ~50 MB")
            print(f"  Activations (batch 512):  ~200 MB")
            print(f"  Replay Buffer:            ~1000 MB")
            print(f"  MCTS Trees (CPU):         0 MB (on CPU)")
            print(f"  Total per training:       ~1250 MB")

            parallel_instances = (total - used - 2000) // 1250  # Leave 2GB buffer
            print(f"\nCould run ~{parallel_instances} parallel training instances")
            print(f"(Each uses ~1.25 GB VRAM)")

    except Exception as e:
        print(f"Could not query GPU: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark MCTS parallelization")
    parser.add_argument('--test', choices=['batch', 'workers', 'single', 'vram', 'all'],
                       default='single', help='Which test to run')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size for single test')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')

    args = parser.parse_args()

    if args.test == 'single' or args.test == 'all':
        print("\n### SINGLE BENCHMARK ###")
        benchmark_batch_size(args.batch_size, n_workers=args.workers)

    if args.test == 'batch' or args.test == 'all':
        print("\n### BATCH SIZE SCALING TEST ###")
        test_batch_sizes(n_workers=args.workers)

    if args.test == 'workers' or args.test == 'all':
        print("\n### WORKER COUNT SCALING TEST ###")
        test_worker_count(batch_size=args.batch_size)

    if args.test == 'vram' or args.test == 'all':
        print("\n### VRAM ANALYSIS ###")
        analyze_vram_usage()

    print("\n\nDONE!")
