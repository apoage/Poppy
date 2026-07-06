"""Test batched self-play speedup"""
import time
import numpy as np
import jax
import jax.numpy as jnp

import sys
sys.path.insert(0, '/home/lukas/Trader')

from mcts_trader.env.vectorbt_env import VectorBTTradingEnv
from mcts_trader.jax_agent.muzero_model import MuZeroNetworks
from mcts_trader.jax_agent.mctx_search import create_mctx_search
from mcts_trader.jax_agent import VALUE_SUPPORT_SIZE

# Import the functions we want to test
from train_jax_fast import run_self_play, run_self_play_batched

print("=" * 60)
print("BATCHED SELF-PLAY SPEEDUP TEST")
print("=" * 60)

# Setup
STEPS = 100  # Short episode for testing
SIMS = 32

print("\nSetting up model (large, bfloat16)...")
model = MuZeroNetworks(
    state_dim=160,
    action_space_size=320,
    support_size=VALUE_SUPPORT_SIZE,
    H_cycles=4,
    L_cycles=6,
    dtype=jnp.bfloat16,
)

rng = jax.random.PRNGKey(42)
dummy_obs = jnp.zeros((1, 27, 64, 64), dtype=jnp.float32)
dummy_action = jnp.zeros((1,), dtype=jnp.int32)
params = model.init(rng, dummy_obs, action=dummy_action, mode='initialize_parameters')['params']

print("Setting up MCTS search...")
search_fn = create_mctx_search(model, num_simulations=SIMS)

# Warmup JIT
print("Warming up JIT compilation...")
env_warmup = VectorBTTradingEnv(synthetic_level=0, n_assets=64, max_steps=10)
obs_dict, _ = env_warmup.reset()
obs = jnp.array(np.expand_dims(obs_dict['observation'], 0), dtype=jnp.float32)
mask = jnp.array(np.expand_dims(obs_dict['action_mask'], 0), dtype=jnp.int8)
_ = search_fn(params, rng, obs, mask)

# Also warmup batched version
obs_batch = jnp.concatenate([obs, obs, obs, obs], axis=0)
mask_batch = jnp.concatenate([mask, mask, mask, mask], axis=0)
_ = search_fn(params, rng, obs_batch, mask_batch)
print("JIT warmup done.\n")

# Test configurations
configs = [
    ("Serial (1 env)", 1),
    ("Batched (2 envs)", 2),
    ("Batched (4 envs)", 4),
    ("Batched (8 envs)", 8),
]

results = []

for name, n_envs in configs:
    print(f"\nTesting {name}...")

    # Create environments
    envs = [VectorBTTradingEnv(synthetic_level=0, n_assets=64, max_steps=STEPS)
            for _ in range(n_envs)]

    rng, test_rng = jax.random.split(rng)

    start = time.perf_counter()

    if n_envs == 1:
        trajectory, score = run_self_play(
            envs[0], search_fn, model.apply, params, test_rng,
            steps=STEPS, support_size=VALUE_SUPPORT_SIZE
        )
        n_transitions = len(trajectory['observations'])
    else:
        trajectories, scores = run_self_play_batched(
            envs, search_fn, model.apply, params, test_rng,
            steps=STEPS, support_size=VALUE_SUPPORT_SIZE
        )
        n_transitions = sum(len(t['observations']) for t in trajectories)
        score = np.mean(scores)

    elapsed = time.perf_counter() - start

    transitions_per_sec = n_transitions / elapsed
    time_per_transition = elapsed / n_transitions * 1000

    results.append({
        'name': name,
        'n_envs': n_envs,
        'elapsed': elapsed,
        'transitions': n_transitions,
        'tps': transitions_per_sec,
        'ms_per_trans': time_per_transition,
    })

    print(f"  Time: {elapsed:.2f}s | Transitions: {n_transitions} | {transitions_per_sec:.1f} trans/s")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
baseline_tps = results[0]['tps']
for r in results:
    speedup = r['tps'] / baseline_tps
    print(f"{r['name']:20s}: {r['tps']:6.1f} trans/s | {r['ms_per_trans']:.2f}ms/trans | {speedup:.2f}x speedup")

print("\n" + "=" * 60)
print(f"RECOMMENDATION: Use --parallel-envs 4 for ~{results[2]['tps']/baseline_tps:.1f}x throughput boost")
print("=" * 60)
