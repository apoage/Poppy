"""Profile MCTS internals in detail"""
import time
import numpy as np
import jax
import jax.numpy as jnp
import mctx
from functools import partial

import sys
sys.path.insert(0, '/home/lukas/Trader')

from mcts_trader.env.vectorbt_env import VectorBTTradingEnv
from mcts_trader.jax_agent.muzero_model import MuZeroNetworks
from mcts_trader.jax_agent import VALUE_SUPPORT_SIZE

print("Setting up...")
model = MuZeroNetworks(
    state_dim=160,  # Large model
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

# Test individual components
print("\n" + "="*60)
print("INDIVIDUAL COMPONENT TIMING (100 iterations each)")
print("="*60)

# 1. Representation (encode observation)
@jax.jit
def rep_fn(params, obs):
    return model.apply({'params': params}, obs, mode='initial_inference')

print("\nWarming up representation...")
dummy = rep_fn(params, dummy_obs)
dummy['latent_state'].block_until_ready()

print("Timing representation (encode observation -> latent)...")
times = []
for _ in range(100):
    start = time.perf_counter()
    out = rep_fn(params, dummy_obs)
    out['latent_state'].block_until_ready()
    times.append(time.perf_counter() - start)
avg_rep = np.mean(times) * 1000
print(f"  representation: {avg_rep:.3f}ms/call")

# 2. Prediction (latent -> policy, value)
@jax.jit
def pred_fn(params, latent):
    policy, value = model.apply({'params': params}, latent, method=model.prediction)
    return policy, value

latent = dummy['latent_state']
print("\nWarming up prediction...")
p, v = pred_fn(params, latent)
p.block_until_ready()

print("Timing prediction (latent -> policy, value)...")
times = []
for _ in range(100):
    start = time.perf_counter()
    p, v = pred_fn(params, latent)
    p.block_until_ready()
    times.append(time.perf_counter() - start)
avg_pred = np.mean(times) * 1000
print(f"  prediction: {avg_pred:.3f}ms/call")

# 3. Dynamics (latent + action -> next_latent, reward)
@jax.jit
def dyn_fn(params, latent, action):
    next_lat, reward = model.apply({'params': params}, latent, action, method=model.dynamics)
    return next_lat, reward

dummy_act = jnp.array([42])
print("\nWarming up dynamics...")
nl, r = dyn_fn(params, latent, dummy_act)
nl.block_until_ready()

print("Timing dynamics (latent + action -> next_latent, reward)...")
times = []
for _ in range(100):
    start = time.perf_counter()
    nl, r = dyn_fn(params, latent, dummy_act)
    nl.block_until_ready()
    times.append(time.perf_counter() - start)
avg_dyn = np.mean(times) * 1000
print(f"  dynamics: {avg_dyn:.3f}ms/call")

# 4. Recurrent inference (dynamics + prediction combined)
@jax.jit
def rec_fn(params, latent, action):
    return model.apply({'params': params}, latent, action, mode='recurrent_inference')

print("\nWarming up recurrent_inference...")
out = rec_fn(params, latent, dummy_act)
out['latent_state'].block_until_ready()

print("Timing recurrent_inference (dynamics + prediction)...")
times = []
for _ in range(100):
    start = time.perf_counter()
    out = rec_fn(params, latent, dummy_act)
    out['latent_state'].block_until_ready()
    times.append(time.perf_counter() - start)
avg_rec = np.mean(times) * 1000
print(f"  recurrent_inference: {avg_rec:.3f}ms/call")

# 5. Full MCTS search with different simulation counts
print("\n" + "="*60)
print("MCTS SEARCH TIMING (varying simulations)")
print("="*60)

def scalar_value_fn(value_logits: jnp.ndarray) -> jnp.ndarray:
    """Convert categorical value logits to scalar values."""
    support_size = value_logits.shape[-1]
    half_support = support_size // 2
    support = jnp.arange(-half_support, half_support + 1)
    probs = jax.nn.softmax(value_logits, axis=-1)
    return jnp.sum(probs * support, axis=-1)

env = VectorBTTradingEnv(synthetic_level=0, n_assets=64, max_steps=100)
obs_dict, _ = env.reset()
obs = jnp.array(np.expand_dims(obs_dict['observation'], 0), dtype=jnp.float32)
mask = jnp.array(np.expand_dims(obs_dict['action_mask'], 0), dtype=jnp.int8)

for num_sims in [8, 16, 32, 64, 128]:
    def root_fn(params, rng, obs):
        out = model.apply({'params': params}, obs, mode='initial_inference')
        return mctx.RootFnOutput(
            prior_logits=out['policy_logits'],
            value=scalar_value_fn(out['value']),
            embedding=out['latent_state']
        )

    def recurrent_fn(params, rng, action, embedding):
        out = model.apply({'params': params}, embedding, action, mode='recurrent_inference')
        return mctx.RecurrentFnOutput(
            reward=scalar_value_fn(out['reward']),
            discount=jnp.full(action.shape, 0.997),
            prior_logits=out['policy_logits'],
            value=scalar_value_fn(out['value'])
        ), out['latent_state']

    @jax.jit
    def search_fn(params, rng, obs, mask):
        root = root_fn(params, rng, obs)
        root = root.replace(
            prior_logits=jnp.where(mask == 1, root.prior_logits, -1e9)
        )
        policy_output = mctx.gumbel_muzero_policy(
            params=params,
            rng_key=rng,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=num_sims,
            max_depth=10,
            gumbel_scale=1.0,
        )
        return policy_output

    # Warmup
    rng, key = jax.random.split(rng)
    out = search_fn(params, key, obs, mask)
    out.action.block_until_ready()

    # Time
    times = []
    for _ in range(20):
        rng, key = jax.random.split(rng)
        start = time.perf_counter()
        out = search_fn(params, key, obs, mask)
        out.action.block_until_ready()
        times.append(time.perf_counter() - start)

    avg_time = np.mean(times) * 1000
    time_per_sim = avg_time / num_sims
    print(f"  {num_sims:3d} simulations: {avg_time:7.2f}ms total | {time_per_sim:.3f}ms/sim")

# Summary
print("\n" + "="*60)
print("COST ESTIMATE PER MCTS SEARCH (32 sims)")
print("="*60)
print(f"  1× representation:       {avg_rep:.3f}ms")
print(f"  32× recurrent_inference: {32 * avg_rec:.3f}ms")
print(f"  Tree overhead:           ~{33.5 - avg_rep - 32*avg_rec:.1f}ms")
print(f"  ---")
print(f"  Total expected:          ~{avg_rep + 32*avg_rec:.1f}ms")
print("="*60)
