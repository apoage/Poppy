"""Profile self-play bottlenecks"""
import time
import numpy as np
import jax
import jax.numpy as jnp
from functools import wraps

# Timing decorator
class Timer:
    def __init__(self):
        self.times = {}
        self.counts = {}
    
    def time(self, name):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                if name not in self.times:
                    self.times[name] = 0
                    self.counts[name] = 0
                self.times[name] += elapsed
                self.counts[name] += 1
                return result
            return wrapper
        return decorator
    
    def report(self):
        total = sum(self.times.values())
        print("\n" + "="*60)
        print("PROFILING RESULTS")
        print("="*60)
        for name, t in sorted(self.times.items(), key=lambda x: -x[1]):
            count = self.counts[name]
            pct = 100 * t / total if total > 0 else 0
            avg_ms = 1000 * t / count if count > 0 else 0
            print(f"{name:30s}: {t:8.3f}s ({pct:5.1f}%) | {count:5d} calls | {avg_ms:.3f}ms/call")
        print(f"{'TOTAL':30s}: {total:8.3f}s")
        print("="*60)

timer = Timer()

# Import after timer setup
import sys
sys.path.insert(0, '/home/lukas/Trader')

from mcts_trader.env.vectorbt_env import VectorBTTradingEnv
from mcts_trader.jax_agent.muzero_model import MuZeroNetworks
from mcts_trader.jax_agent.mctx_search import create_mctx_search
from mcts_trader.jax_agent import VALUE_SUPPORT_SIZE

# Patch encoder.encode to time it
original_encode = None
def patch_encoder(env):
    global original_encode
    if hasattr(env, 'encoder') and env.encoder is not None:
        original_encode = env.encoder.encode
        def timed_encode():
            start = time.perf_counter()
            result = original_encode()
            timer.times.setdefault('encoder.encode', 0)
            timer.counts.setdefault('encoder.encode', 0)
            timer.times['encoder.encode'] += time.perf_counter() - start
            timer.counts['encoder.encode'] += 1
            return result
        env.encoder.encode = timed_encode

# Patch indicators.compute_all
def patch_indicators(env):
    if hasattr(env, 'encoder') and env.encoder is not None:
        if hasattr(env.encoder, 'indicators'):
            original_compute = env.encoder.indicators.compute_all
            def timed_compute(*args, **kwargs):
                start = time.perf_counter()
                result = original_compute(*args, **kwargs)
                timer.times.setdefault('indicators.compute_all', 0)
                timer.counts.setdefault('indicators.compute_all', 0)
                timer.times['indicators.compute_all'] += time.perf_counter() - start
                timer.counts['indicators.compute_all'] += 1
                return result
            env.encoder.indicators.compute_all = timed_compute

print("Setting up environment...")
env = VectorBTTradingEnv(
    synthetic_level=0,
    n_assets=64,
    max_steps=200,  # Short test
)
patch_encoder(env)
patch_indicators(env)

print("Setting up model...")
model = MuZeroNetworks(
    state_dim=160,  # Large model
    action_space_size=320,
    support_size=VALUE_SUPPORT_SIZE,
    H_cycles=4,
    L_cycles=6,
    dtype=jnp.bfloat16,
)

# Initialize - use 'initialize_parameters' to init ALL networks (including dynamics)
rng = jax.random.PRNGKey(42)
dummy_obs = jnp.zeros((1, 27, 64, 64), dtype=jnp.float32)
dummy_action = jnp.zeros((1,), dtype=jnp.int32)
params = model.init(rng, dummy_obs, action=dummy_action, mode='initialize_parameters')['params']

print("Setting up MCTS...")
search_fn = create_mctx_search(model, num_simulations=32)

# Warmup
print("Warming up JIT...")
obs_dict, _ = env.reset()
obs = jnp.array(np.expand_dims(obs_dict['observation'], 0), dtype=jnp.float32)
mask = jnp.array(np.expand_dims(obs_dict['action_mask'], 0), dtype=jnp.int8)
_ = search_fn(params, rng, obs, mask)
print("Warmup done.\n")

# Profile 200 steps
print("Profiling 200 steps...")
obs_dict, _ = env.reset()

for step in range(200):
    # Time observation prep
    start = time.perf_counter()
    obs_array = obs_dict['observation']
    obs_array = np.expand_dims(obs_array, axis=0)
    obs_jax = jnp.array(obs_array, dtype=jnp.float32)
    mask_array = obs_dict['action_mask']
    mask_array = np.expand_dims(mask_array, axis=0)
    mask_jax = jnp.array(mask_array, dtype=jnp.int8)
    timer.times.setdefault('obs_prep_numpy_to_jax', 0)
    timer.counts.setdefault('obs_prep_numpy_to_jax', 0)
    timer.times['obs_prep_numpy_to_jax'] += time.perf_counter() - start
    timer.counts['obs_prep_numpy_to_jax'] += 1
    
    # Time MCTS
    start = time.perf_counter()
    rng, step_rng = jax.random.split(rng)
    action_jax, policy_jax, stats = search_fn(params, step_rng, obs_jax, mask_jax)
    # Force sync to get accurate timing
    action_jax.block_until_ready()
    timer.times.setdefault('mcts_search', 0)
    timer.counts.setdefault('mcts_search', 0)
    timer.times['mcts_search'] += time.perf_counter() - start
    timer.counts['mcts_search'] += 1
    
    # Time env step
    start = time.perf_counter()
    action_int = int(action_jax[0])
    obs_dict, reward, terminated, truncated, info = env.step(action_int)
    timer.times.setdefault('env.step', 0)
    timer.counts.setdefault('env.step', 0)
    timer.times['env.step'] += time.perf_counter() - start
    timer.counts['env.step'] += 1
    
    if terminated or truncated:
        break

timer.report()
