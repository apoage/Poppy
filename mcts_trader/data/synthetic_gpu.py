"""
GPU Synthetic Price Generation — JAX replacement for SyntheticMarket.

Generates close prices directly on GPU, eliminating:
- CPU Python loops (O(n_steps * n_assets) iterations)
- CPU→GPU PCIe transfers (especially slow on Gen1 x4 cards like 90HX)
- Background thread pool overhead

All 15 curriculum levels (L0-L14) are supported.
Matches SyntheticMarket output exactly (close prices only — OHLC is wasteful
since only close is consumed downstream).

Usage:
    key = jax.random.PRNGKey(42)
    prices = generate_synthetic_on_device(key, n_envs=2, level=3,
                                          n_assets=64, n_steps=489,
                                          noise_level=0.1)
    # prices: (2, 489, 64) float32 on device
"""

import jax
import jax.numpy as jnp
from functools import partial


# =============================================================================
# Helpers
# =============================================================================

def _gbm(key, n_assets, n_steps, drift=0.0, vol=0.01):
    """Geometric Brownian Motion. Returns (n_steps, n_assets)."""
    returns = jax.random.normal(key, (n_steps, n_assets)) * vol + drift
    return 100.0 * jnp.exp(jnp.cumsum(returns, axis=0))


def _apply_noise(prices, key, noise_level):
    """Anti-memorization noise (mirrors SyntheticMarket._apply_noise).

    Only price noise is applied (time_dilation, phase_shift, temporal_jitter
    all default to zero in the training config).

    Args:
        prices: (n_steps,) single asset close prices
        key: JAX PRNG key
        noise_level: float (0.0 = none, 0.1 = default)
    Returns:
        (n_steps,) noisy prices
    """
    n = prices.shape[0]
    k1, k2, k3 = jax.random.split(key, 3)
    # Random walk (cumulative drift)
    walk = jnp.cumsum(jax.random.normal(k1, (n,)) * (0.0005 * noise_level))
    # Per-step noise
    step = jax.random.normal(k2, (n,)) * (0.005 * noise_level)
    # Slow amplitude variation
    amp_period = jnp.maximum(100.0, n / 10.0)
    amp_phase = jax.random.uniform(k3, (), minval=0.0, maxval=2.0 * jnp.pi)
    t = jnp.arange(n, dtype=jnp.float32)
    amp = 1.0 + 0.2 * noise_level * jnp.sin(2.0 * jnp.pi * t / amp_period + amp_phase)
    noisy = prices * (1.0 + walk + step) * amp
    return jnp.maximum(noisy, prices * 0.5)


def _apply_noise_all(prices, key, noise_level):
    """Apply noise to all assets. prices: (n_steps, n_assets)."""
    n_assets = prices.shape[1]
    keys = jax.random.split(key, n_assets)
    return jax.vmap(
        lambda p, k: _apply_noise(p, k, noise_level), in_axes=(1, 0), out_axes=1
    )(prices, keys)


# =============================================================================
# Level 0: Pure Sine Waves
# =============================================================================

def _level_0(key, n_assets, n_steps):
    k1, k2, k3 = jax.random.split(key, 3)
    periods = jax.random.randint(k1, (n_assets,), 20, 60).astype(jnp.float32)
    amps = jax.random.uniform(k2, (n_assets,), minval=0.08, maxval=0.15)
    g_phase = jax.random.uniform(k3, (), minval=0.0, maxval=2.0 * jnp.pi)
    base_ph = (jnp.arange(n_assets, dtype=jnp.float32) * jnp.pi / 4.0) % (2.0 * jnp.pi)
    t = jnp.arange(n_steps, dtype=jnp.float32)[:, None]
    return 100.0 * (1.0 + amps[None] * jnp.sin(2.0 * jnp.pi * t / periods[None] + (base_ph + g_phase)[None]))


# =============================================================================
# Level 1: Lagged Pairs
# =============================================================================

def _level_1(key, n_assets, n_steps):
    LAG = 10
    n_leaders = n_assets // 2
    k1, k2, k3, k4 = jax.random.split(key, 4)
    periods = jax.random.randint(k1, (n_leaders,), 15, 40).astype(jnp.float32)
    amps = jax.random.uniform(k2, (n_leaders,), minval=0.10, maxval=0.25)
    phases = jax.random.uniform(k3, (n_leaders,), minval=0.0, maxval=2.0 * jnp.pi)
    t = jnp.arange(n_steps, dtype=jnp.float32)[:, None]
    leaders = 100.0 * (1.0 + amps[None] * jnp.sin(2.0 * jnp.pi * t / periods[None] + phases[None]))
    # Followers: lagged + noise
    followers = jnp.roll(leaders, LAG, axis=0).at[:LAG].set(leaders[:1])
    followers = followers * (1.0 + jax.random.normal(k4, (n_steps, n_leaders)) * 0.01)
    return jnp.concatenate([leaders, followers[:, :n_assets - n_leaders]], axis=1)


# =============================================================================
# Level 2: Dip Crashes
# =============================================================================

def _level_2(key, n_assets, n_steps):
    k1, k2 = jax.random.split(key)
    crashes = jax.random.uniform(k1, (n_steps, n_assets)) < 0.02
    noise = jax.random.normal(k2, (n_steps, n_assets)) * 0.005
    return jnp.maximum(jnp.where(crashes, 90.0, 100.0) * (1.0 + noise), 0.01)


# =============================================================================
# Level 3: Noisy Trends (Markov trend switching + mean reversion)
# =============================================================================

def _level_3(key, n_assets, n_steps):
    SWITCH_P, DRIFT, NOISE, REVERT, BASE = 0.05, 0.003, 0.008, 0.002, 100.0
    k1, k2, k3 = jax.random.split(key, 3)
    # Pre-generate all randomness: (n_steps-1, n_assets)
    switch_rng = jax.random.uniform(k1, (n_steps - 1, n_assets))
    trend_choices = (jax.random.randint(k2, (n_steps - 1, n_assets), 0, 3) - 1).astype(jnp.float32)
    noises = jax.random.normal(k3, (n_steps - 1, n_assets)) * NOISE

    def step(carry, data):
        prices, trends = carry
        sw, tc, n = data
        trends = jnp.where(sw < SWITCH_P, tc, trends)
        mr = -REVERT * jnp.log(jnp.maximum(prices, 1e-8) / BASE)
        new_prices = jnp.clip(prices * (1.0 + trends * DRIFT + mr + n), 50.0, 200.0)
        return (new_prices, trends), new_prices

    _, all_p = jax.lax.scan(step, (jnp.full(n_assets, BASE), jnp.zeros(n_assets)),
                            (switch_rng, trend_choices, noises))
    return jnp.concatenate([jnp.full((1, n_assets), BASE), all_p], axis=0)


# =============================================================================
# Level 4: Portfolio Rotation
# =============================================================================

def _level_4(key, n_assets, n_steps):
    SECTOR_SIZE, ROTATION_PERIOD, PUMP = 8, 50, 0.002
    N_SECTORS = n_assets // SECTOR_SIZE
    prices = _gbm(key, n_assets, n_steps, drift=0.0, vol=0.01)
    # Sector pump: leading sector gets +0.2% per step
    t = jnp.arange(n_steps)[:, None]
    asset_sector = jnp.arange(n_assets)[None, :] // SECTOR_SIZE
    leading = (jnp.arange(n_steps)[:, None] // ROTATION_PERIOD) % N_SECTORS
    pump_mask = (leading == asset_sector).astype(jnp.float32)
    # Cumulative multiplicative pump
    return prices * jnp.exp(jnp.cumsum(pump_mask * PUMP, axis=0))


# =============================================================================
# Level 5: Pairs Trading (Cointegrated via OU spread)
# =============================================================================

def _level_5(key, n_assets, n_steps):
    THETA, MU, SIGMA = 0.15, 0.0, 0.05
    N_PAIRS = n_assets // 2
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    # Common factor
    common = _gbm(k1, N_PAIRS, n_steps, drift=0.0001, vol=0.01)
    # OU spread via scan
    spread_noise = jax.random.normal(k2, (n_steps - 1, N_PAIRS)) * SIGMA
    spread_init = jax.random.normal(k3, (N_PAIRS,)) * SIGMA

    def ou_step(s, n):
        return s + (-THETA * (s - MU) + n), s
    _, spreads = jax.lax.scan(ou_step, spread_init, spread_noise)
    spreads = jnp.concatenate([spread_init[None], spreads], axis=0)

    betas = 1.0 + jax.random.uniform(k4, (N_PAIRS,), minval=-0.1, maxval=0.1)
    # Idiosyncratic noise
    idio = jax.random.normal(k5, (n_steps, N_PAIRS * 2)) * 0.002
    price_a = common * (1.0 + spreads / 2.0) * (1.0 + idio[:, :N_PAIRS])
    price_b = common * (1.0 - spreads / (2.0 * betas[None])) * (1.0 + idio[:, N_PAIRS:])
    return jnp.maximum(jnp.concatenate([price_a, price_b], axis=1), 0.01)


# =============================================================================
# Level 6: Triangle Arbitrage
# =============================================================================

def _level_6(key, n_assets, n_steps):
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    # 16 base currencies
    base = _gbm(k1, 16, n_steps, drift=0.0001, vol=0.015)
    # Quote pairs
    quotes = base * (1.0 + jax.random.normal(k2, (n_steps, 16)) * 0.002)
    # Cross pairs with mispricing
    cross_keys = jax.random.split(k3, 16)
    fair_rates = []
    for i in range(16):
        j = (i + 1) % 16
        fair = base[:, i] / jnp.maximum(base[:, j], 0.01)
        ck1, ck2, ck3 = jax.random.split(cross_keys[i], 3)
        rng = jax.random.uniform(ck1, (n_steps,))
        direction = jax.random.bernoulli(ck2, 0.5, (n_steps,)) * 2.0 - 1.0
        magnitude = jax.random.uniform(ck3, (n_steps,), minval=0.01, maxval=0.03)
        misprice = jnp.where(rng < 0.05, 1.0 + direction * magnitude, 1.0)
        fair_rates.append(jnp.maximum(fair * misprice, 0.01))
    crosses = jnp.stack(fair_rates, axis=1)
    # Fill remaining with GBM
    n_remaining = max(n_assets - 48, 1)
    fill = _gbm(k4, n_remaining, n_steps, drift=0.0, vol=0.01)
    all_prices = jnp.concatenate([base, quotes, crosses, fill], axis=1)
    return all_prices[:, :n_assets]


# =============================================================================
# Level 7: Index Arbitrage
# =============================================================================

def _level_7(key, n_assets, n_steps):
    N_COMP, N_IDX, COMP_PER_IDX, LAG = 56, 8, 7, 4
    k1, k2, k3 = jax.random.split(key, 3)
    # Component prices
    components = _gbm(k1, N_COMP, n_steps, drift=0.0002, vol=0.012)
    # Index = lagged equal-weight basket + premium events
    indices = []
    idx_keys = jax.random.split(k2, N_IDX)
    for idx in range(N_IDX):
        basket = components[:, idx * COMP_PER_IDX:(idx + 1) * COMP_PER_IDX].mean(axis=1)
        # Lag
        index_p = jnp.roll(basket, LAG).at[:LAG].set(basket[:LAG])
        # Premium/discount events (3%)
        ik1, ik2 = jax.random.split(idx_keys[idx])
        event_rng = jax.random.uniform(ik1, (n_steps,))
        premium = jax.random.uniform(ik2, (n_steps,), minval=-0.02, maxval=0.02)
        index_p = index_p * jnp.where(event_rng < 0.03, 1.0 + premium, 1.0)
        indices.append(jnp.maximum(index_p, 0.01))
    idx_arr = jnp.stack(indices, axis=1)
    return jnp.concatenate([components, idx_arr], axis=1)[:, :n_assets]


# =============================================================================
# Level 8: Momentum (Trend persistence)
# =============================================================================

def _level_8(key, n_assets, n_steps):
    LOOKBACK, PERSIST, DRIFT, VOL = 20, 0.70, 0.003, 0.01
    k1, k2, k3 = jax.random.split(key, 3)
    # Pre-generate randomness
    persist_rng = jax.random.uniform(k1, (n_steps - 1, n_assets))
    new_trends = (jax.random.randint(k2, (n_steps - 1, n_assets), 0, 3) - 1).astype(jnp.float32)
    noises = jax.random.normal(k3, (n_steps - 1, n_assets)) * VOL

    def step(carry, data):
        prices, trends, counter = carry
        p_rng, nt, noise = data
        counter = counter + 1
        # Switch check at LOOKBACK intervals
        at_boundary = (counter % LOOKBACK == 0)
        should_switch = at_boundary & (p_rng > PERSIST)
        trends = jnp.where(should_switch, nt, trends)
        new_prices = jnp.maximum(prices * (1.0 + trends * DRIFT + noise), 0.01)
        return (new_prices, trends, counter), new_prices

    init_trends = (jax.random.randint(key, (n_assets,), 0, 3) - 1).astype(jnp.float32)
    _, all_p = jax.lax.scan(
        step, (jnp.full(n_assets, 100.0), init_trends, jnp.zeros(n_assets)),
        (persist_rng, new_trends, noises))
    return jnp.concatenate([jnp.full((1, n_assets), 100.0), jnp.maximum(all_p, 0.01)], axis=0)


# =============================================================================
# Level 9: Mean Reversion (OU process)
# =============================================================================

def _level_9(key, n_assets, n_steps):
    THETA, MU, SIGMA = 0.20, 100.0, 3.0
    k1, k2 = jax.random.split(key)
    init = MU + jax.random.normal(k1, (n_assets,)) * SIGMA
    noise = jax.random.normal(k2, (n_steps - 1, n_assets))

    def step(prices, n):
        new = prices + (-THETA * (prices - MU) + SIGMA * 0.1 * n)
        return jnp.maximum(new, 0.01), new

    _, all_p = jax.lax.scan(step, init, noise)
    return jnp.concatenate([init[None], jnp.maximum(all_p, 0.01)], axis=0)


# =============================================================================
# Level 10: Breakout (Consolidation → Trend)
# =============================================================================

def _level_10(key, n_assets, n_steps):
    CONS_P, BREAK_P, RANGE, BREAK_DRIFT = 30, 20, 0.02, 0.0025
    k1, k2, k3 = jax.random.split(key, 3)
    range_noise = jax.random.uniform(k1, (n_steps - 1, n_assets), minval=-RANGE, maxval=RANGE)
    break_noise = jax.random.normal(k2, (n_steps - 1, n_assets)) * 0.005
    break_dirs = jax.random.bernoulli(k3, 0.5, (n_steps - 1, n_assets)) * 2.0 - 1.0

    def step(carry, data):
        prices, base, phase, counter, bdir = carry
        rn, bn, bd = data
        counter = counter + 1
        # Consolidation: price around base
        cons_p = base * (1.0 + rn)
        # Breakout: trending
        brk_p = prices * (1.0 + bdir * BREAK_DRIFT + bn)
        new_prices = jnp.where(phase == 0, cons_p, brk_p)
        new_prices = jnp.maximum(new_prices, 0.01)
        # Transitions
        to_break = (phase == 0) & (counter >= CONS_P)
        to_cons = (phase == 1) & (counter >= BREAK_P)
        new_phase = jnp.where(to_break, 1.0, jnp.where(to_cons, 0.0, phase))
        new_counter = jnp.where(to_break | to_cons, 0.0, counter)
        new_base = jnp.where(to_cons, new_prices, base)
        new_bdir = jnp.where(to_break, bd, bdir)
        return (new_prices, new_base, new_phase, new_counter, new_bdir), new_prices

    z = jnp.zeros(n_assets)
    init = (jnp.full(n_assets, 100.0), jnp.full(n_assets, 100.0), z, z, jnp.ones(n_assets))
    _, all_p = jax.lax.scan(step, init, (range_noise, break_noise, break_dirs))
    return jnp.concatenate([jnp.full((1, n_assets), 100.0), all_p], axis=0)


# =============================================================================
# Level 11: Bid-Ask Bounce
# =============================================================================

def _level_11(key, n_assets, n_steps):
    SPREAD = 10.0 / 10000.0  # 10 bps
    k1, k2 = jax.random.split(key)
    mid_returns = jax.random.normal(k1, (n_steps, n_assets)) * 0.001
    mid = 100.0 * jnp.exp(jnp.cumsum(mid_returns, axis=0))
    order_flow = jax.random.bernoulli(k2, 0.5, (n_steps, n_assets)) * 2.0 - 1.0
    return jnp.maximum(mid * (1.0 + order_flow * SPREAD / 2.0), 0.01)


# =============================================================================
# Level 12: Order Imbalance
# =============================================================================

def _level_12(key, n_assets, n_steps):
    THRESHOLD, ACCURACY, MOVE = 0.3, 0.80, 0.005
    k1, k2, k3 = jax.random.split(key, 3)
    imbalances = jax.random.uniform(k1, (n_steps - 1, n_assets), minval=-0.5, maxval=0.5)
    accuracy_rng = jax.random.uniform(k2, (n_steps - 1, n_assets))
    noises = jax.random.normal(k3, (n_steps - 1, n_assets)) * 0.002

    def step(prices, data):
        imb, acc_rng, noise = data
        strong = jnp.abs(imb) > THRESHOLD
        direction = jnp.where(acc_rng < ACCURACY, jnp.sign(imb), -jnp.sign(imb))
        drift = jnp.where(strong, direction * MOVE, 0.0)
        new = jnp.maximum(prices * (1.0 + drift + noise), 0.01)
        return new, new

    _, all_p = jax.lax.scan(step, jnp.full(n_assets, 100.0), (imbalances, accuracy_rng, noises))
    return jnp.concatenate([jnp.full((1, n_assets), 100.0), all_p], axis=0)


# =============================================================================
# Level 13: Market Making (Informed vs Noise flow)
# =============================================================================

def _level_13(key, n_assets, n_steps):
    NOISE_SIZE, INFORMED_PROB, INFORMED_SIZE = 0.001, 0.05, 0.01
    k1, k2, k3 = jax.random.split(key, 3)
    is_informed = jax.random.uniform(k1, (n_steps - 1, n_assets)) < INFORMED_PROB
    direction = jax.random.bernoulli(k2, 0.5, (n_steps - 1, n_assets)) * 2.0 - 1.0
    noise_moves = jax.random.uniform(k3, (n_steps - 1, n_assets), minval=-NOISE_SIZE, maxval=NOISE_SIZE)

    def step(prices, data):
        informed, d, nm = data
        move = jnp.where(informed, d * INFORMED_SIZE, nm)
        new = jnp.maximum(prices * (1.0 + move), 0.01)
        return new, new

    _, all_p = jax.lax.scan(step, jnp.full(n_assets, 100.0), (is_informed, direction, noise_moves))
    return jnp.concatenate([jnp.full((1, n_assets), 100.0), all_p], axis=0)


# =============================================================================
# Level 14: GARCH(1,1) Volatility Clustering
# =============================================================================

def _level_14(key, n_assets, n_steps):
    OMEGA, ALPHA, BETA = 0.000001, 0.10, 0.85
    k1 = key
    epsilons = jax.random.normal(k1, (n_steps - 1, n_assets))

    def step(carry, eps):
        prices, variances, prev_ret = carry
        # GARCH(1,1)
        new_var = jnp.clip(OMEGA + ALPHA * prev_ret ** 2 + BETA * variances, 0.000001, 0.001)
        vol = jnp.sqrt(new_var)
        ret = vol * eps
        new_prices = jnp.maximum(prices * jnp.exp(ret), 0.01)
        return (new_prices, new_var, ret), new_prices

    init = (jnp.full(n_assets, 100.0), jnp.full(n_assets, 0.0001), jnp.zeros(n_assets))
    _, all_p = jax.lax.scan(step, init, epsilons)
    return jnp.concatenate([jnp.full((1, n_assets), 100.0), all_p], axis=0)


# =============================================================================
# Dispatcher
# =============================================================================

_LEVEL_FNS = {
    0: _level_0, 1: _level_1, 2: _level_2, 3: _level_3,
    4: _level_4, 5: _level_5, 6: _level_6, 7: _level_7,
    8: _level_8, 9: _level_9, 10: _level_10, 11: _level_11,
    12: _level_12, 13: _level_13, 14: _level_14,
}


def _generate_single(key, level, n_assets, n_steps, noise_level):
    """Generate one episode. Returns (n_steps, n_assets) close prices."""
    gen_key, noise_key = jax.random.split(key)
    prices = _LEVEL_FNS[level](gen_key, n_assets, n_steps)
    return _apply_noise_all(prices, noise_key, noise_level)


@partial(jax.jit, static_argnames=('n_envs', 'level', 'n_assets', 'n_steps'))
def generate_synthetic_on_device(key, n_envs, level, n_assets, n_steps, noise_level):
    """Generate n_envs synthetic episodes entirely on GPU.

    Args:
        key: JAX PRNG key
        n_envs: Number of parallel episodes
        level: Curriculum level (0-14)
        n_assets: Number of assets (64)
        n_steps: Episode length (typically 489 = 400+64+20+5)
        noise_level: Anti-memorization noise (0.1 default)

    Returns:
        (n_envs, n_steps, n_assets) float32 close prices on device
    """
    keys = jax.random.split(key, n_envs)
    return jax.vmap(
        lambda k: _generate_single(k, level, n_assets, n_steps, noise_level)
    )(keys)
