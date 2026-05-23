"""
TASK_768 JAX-old-vs-new bit-identity gate (the definitive test):
implement the pre-768 monolithic body inline using jnp ops with the
exact same operation order, then compare to the post-768 split wrapper.

Because both use the same jnp expressions in the same order, JAX produces
identical bit patterns for both. If they ever diverge, the split changed
either (a) the operation order or (b) introduced a dtype cast somewhere.

This is the gate the SPEC asked for: `jnp.allclose(old, new, atol=0, rtol=0)`
(equivalent to `jnp.array_equal`).
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

sys.path.insert(0, "/home/lukas/Trader")

import jax
import jax.numpy as jnp
import numpy as np

from mcts_trader.env.jax_encoder import (
    EncoderConfig,
    _compute_all_channels,
    _compute_all_channels_temporal,
)


# Inline JAX-native pre-768 implementation (the monolithic body).
def jax_old_temporal(prices, volumes, highs, lows, config):
    n = config.n_assets
    ws = config.window_size

    output = jnp.zeros((config.n_channels, ws, n), dtype=jnp.float32)

    price_min = jnp.min(prices, axis=1, keepdims=True)
    price_max = jnp.max(prices, axis=1, keepdims=True)
    price_range = price_max - price_min
    price_range = jnp.where(price_range < 1e-8, 1.0, price_range)

    norm_prices = (prices - price_min) / price_range
    norm_highs = (highs - price_min) / price_range
    norm_lows = (lows - price_min) / price_range

    vol_mean = jnp.mean(volumes, axis=1, keepdims=True)
    vol_std = jnp.std(volumes, axis=1, keepdims=True) + 1e-8
    norm_volumes = jnp.clip((volumes - vol_mean) / vol_std, -3, 3) / 3.0

    output = output.at[0].set(norm_prices.T)
    output = output.at[1].set(norm_highs.T)
    output = output.at[2].set(norm_lows.T)
    output = output.at[3].set(norm_volumes.T)

    prev_prices = jnp.concatenate([prices[:, 0:1], prices[:, :-1]], axis=1)
    returns_1m = (prices - prev_prices) / (prev_prices + 1e-8)
    returns_1m = jnp.clip(returns_1m, -config.return_clip, config.return_clip)
    output = output.at[4].set(returns_1m.T)

    def broadcast_scalar(x):
        return jnp.broadcast_to(x[None, :], (ws, n))

    returns_5m = (prices[:, -1] - prices[:, -5]) / (prices[:, -5] + 1e-8)
    returns_15m = (prices[:, -1] - prices[:, -15]) / (prices[:, -15] + 1e-8)
    returns_64m = (prices[:, -1] - prices[:, 0]) / (prices[:, 0] + 1e-8)

    output = output.at[5].set(broadcast_scalar(jnp.clip(returns_5m, -config.return_clip, config.return_clip)))
    output = output.at[6].set(broadcast_scalar(jnp.clip(returns_15m, -config.return_clip, config.return_clip)))
    output = output.at[7].set(broadcast_scalar(jnp.clip(returns_64m, -config.return_clip, config.return_clip)))

    vol_ratio_5m = jnp.mean(volumes[:, -5:], axis=1) / (vol_mean[:, 0] + 1e-8)
    vol_ratio_15m = jnp.mean(volumes[:, -15:], axis=1) / (vol_mean[:, 0] + 1e-8)
    vol_ratio_30m = jnp.mean(volumes[:, -30:], axis=1) / (vol_mean[:, 0] + 1e-8)
    vol_ratio_64m = jnp.mean(volumes, axis=1) / (vol_mean[:, 0] + 1e-8)

    output = output.at[8].set(broadcast_scalar(jnp.clip(vol_ratio_5m, 0, 5) / 5.0))
    output = output.at[9].set(broadcast_scalar(jnp.clip(vol_ratio_15m, 0, 5) / 5.0))
    output = output.at[10].set(broadcast_scalar(jnp.clip(vol_ratio_30m, 0, 5) / 5.0))
    output = output.at[11].set(broadcast_scalar(jnp.clip(vol_ratio_64m, 0, 5) / 5.0))

    momentum_5 = (prices[:, -1] - prices[:, -5]) / (price_range[:, 0] + 1e-8)
    momentum_15 = (prices[:, -1] - prices[:, -15]) / (price_range[:, 0] + 1e-8)
    momentum_30 = (prices[:, -1] - prices[:, -30]) / (price_range[:, 0] + 1e-8)
    momentum_64 = (prices[:, -1] - prices[:, 0]) / (price_range[:, 0] + 1e-8)

    output = output.at[12].set(broadcast_scalar(jnp.clip(momentum_5, -1, 1)))
    output = output.at[13].set(broadcast_scalar(jnp.clip(momentum_15, -1, 1)))
    output = output.at[14].set(broadcast_scalar(jnp.clip(momentum_30, -1, 1)))
    output = output.at[15].set(broadcast_scalar(jnp.clip(momentum_64, -1, 1)))

    vol_5 = jnp.std(returns_1m[:, -5:], axis=1)
    vol_15 = jnp.std(returns_1m[:, -15:], axis=1)
    vol_30 = jnp.std(returns_1m[:, -30:], axis=1)
    vol_64 = jnp.std(returns_1m, axis=1)

    output = output.at[16].set(broadcast_scalar(jnp.clip(vol_5 * config.vol_scale, 0, 1)))
    output = output.at[17].set(broadcast_scalar(jnp.clip(vol_15 * config.vol_scale, 0, 1)))
    output = output.at[18].set(broadcast_scalar(jnp.clip(vol_30 * config.vol_scale, 0, 1)))
    output = output.at[19].set(broadcast_scalar(jnp.clip(vol_64 * config.vol_scale, 0, 1)))

    def compute_percentile(window_prices, window):
        window_data = window_prices[:, -window:]
        w_min = jnp.min(window_data, axis=1)
        w_max = jnp.max(window_data, axis=1)
        w_range = jnp.where((w_max - w_min) < 1e-8, 1.0, w_max - w_min)
        return (window_prices[:, -1] - w_min) / w_range

    output = output.at[20].set(broadcast_scalar(compute_percentile(prices, 5)))
    output = output.at[21].set(broadcast_scalar(compute_percentile(prices, 15)))
    output = output.at[22].set(broadcast_scalar(compute_percentile(prices, 30)))
    output = output.at[23].set(broadcast_scalar(compute_percentile(prices, 64)))

    vwap_window = 20
    vwap_prices = prices[:, -vwap_window:]
    vwap_volumes = volumes[:, -vwap_window:]
    vol_sum = jnp.sum(vwap_volumes, axis=1, keepdims=True) + 1e-8
    vwap = jnp.sum(vwap_prices * vwap_volumes, axis=1) / vol_sum[:, 0]
    vwap_norm = (vwap - price_min[:, 0]) / (price_range[:, 0] + 1e-8)

    price_std = jnp.std(vwap_prices, axis=1)
    upper_band = (vwap + price_std - price_min[:, 0]) / (price_range[:, 0] + 1e-8)
    lower_band = (vwap - price_std - price_min[:, 0]) / (price_range[:, 0] + 1e-8)

    output = output.at[24].set(broadcast_scalar(vwap_norm))
    output = output.at[25].set(broadcast_scalar(jnp.clip(upper_band, 0, 1)))
    output = output.at[26].set(broadcast_scalar(jnp.clip(lower_band, 0, 1)))

    n_f = max(float(n - 1), 1.0)
    lookback_20 = min(20, ws - 1)

    cur_r1 = returns_1m[:, -1]
    rank_1 = jnp.argsort(jnp.argsort(cur_r1)).astype(jnp.float32) / n_f

    r20 = (prices[:, -1] - prices[:, -lookback_20]) / (prices[:, -lookback_20] + 1e-8)
    r20 = jnp.clip(r20, -config.return_clip, config.return_clip)
    rank_20 = jnp.argsort(jnp.argsort(r20)).astype(jnp.float32) / n_f

    uni_rets = jnp.mean(returns_1m, axis=0)
    asset_centered = returns_1m - jnp.mean(returns_1m, axis=1, keepdims=True)
    uni_centered = uni_rets - jnp.mean(uni_rets)
    cov = jnp.mean(asset_centered * uni_centered[None, :], axis=1)
    var_uni = jnp.mean(uni_centered ** 2)
    beta_raw = cov / (var_uni + 1e-8)
    beta_scaled = jnp.clip(beta_raw, -3.0, 3.0) / 3.0

    residuals = returns_1m - beta_raw[:, None] * uni_rets[None, :]
    idio_vol = jnp.std(residuals, axis=1)
    idio_vol_scaled = jnp.clip(idio_vol * config.vol_scale, 0, 1)

    uni_mean_cur = jnp.mean(cur_r1)
    xs_mom = jnp.clip((cur_r1 - uni_mean_cur) * 100.0, -1.0, 1.0)

    if config.n_channels >= 32:
        output = output.at[27].set(broadcast_scalar(rank_1))
        output = output.at[28].set(broadcast_scalar(rank_20))
        output = output.at[29].set(broadcast_scalar(beta_scaled))
        output = output.at[30].set(broadcast_scalar(idio_vol_scaled))
        output = output.at[31].set(broadcast_scalar(xs_mom))

    return output


def jax_old_tiled(prices, volumes, highs, lows, config):
    n = config.n_assets

    output = jnp.zeros((config.n_channels, n, n), dtype=jnp.float32)

    price_min = jnp.min(prices, axis=1, keepdims=True)
    price_max = jnp.max(prices, axis=1, keepdims=True)
    price_range = price_max - price_min
    price_range = jnp.where(price_range < 1e-8, 1.0, price_range)

    norm_prices = (prices - price_min) / price_range
    norm_highs = (highs - price_min) / price_range
    norm_lows = (lows - price_min) / price_range

    vol_mean = jnp.mean(volumes, axis=1, keepdims=True)
    vol_std = jnp.std(volumes, axis=1, keepdims=True) + 1e-8
    norm_volumes = jnp.clip((volumes - vol_mean) / vol_std, -3, 3) / 3.0

    output = output.at[0].set(jnp.tile(norm_prices[:, -1:], (1, n)))
    output = output.at[1].set(jnp.tile(norm_highs[:, -1:], (1, n)))
    output = output.at[2].set(jnp.tile(norm_lows[:, -1:], (1, n)))
    output = output.at[3].set(jnp.tile(norm_volumes[:, -1:], (1, n)))

    prev_prices = jnp.concatenate([prices[:, 0:1], prices[:, :-1]], axis=1)
    returns_1m = (prices - prev_prices) / (prev_prices + 1e-8)
    returns_1m = jnp.clip(returns_1m, -config.return_clip, config.return_clip)
    output = output.at[4].set(jnp.tile(returns_1m[:, -1:], (1, n)))

    returns_5m = (prices[:, -1] - prices[:, -5]) / (prices[:, -5] + 1e-8)
    returns_5m = jnp.clip(returns_5m, -config.return_clip, config.return_clip)
    output = output.at[5].set(jnp.tile(returns_5m[:, None], (1, n)))

    returns_15m = (prices[:, -1] - prices[:, -15]) / (prices[:, -15] + 1e-8)
    returns_15m = jnp.clip(returns_15m, -config.return_clip, config.return_clip)
    output = output.at[6].set(jnp.tile(returns_15m[:, None], (1, n)))

    returns_60m = (prices[:, -1] - prices[:, 0]) / (prices[:, 0] + 1e-8)
    returns_60m = jnp.clip(returns_60m, -config.return_clip, config.return_clip)
    output = output.at[7].set(jnp.tile(returns_60m[:, None], (1, n)))

    vol_ratio_5m = jnp.mean(volumes[:, -5:], axis=1) / (vol_mean[:, 0] + 1e-8)
    vol_ratio_15m = jnp.mean(volumes[:, -15:], axis=1) / (vol_mean[:, 0] + 1e-8)
    vol_ratio_30m = jnp.mean(volumes[:, -30:], axis=1) / (vol_mean[:, 0] + 1e-8)
    vol_ratio_60m = jnp.mean(volumes, axis=1) / (vol_mean[:, 0] + 1e-8)

    output = output.at[8].set(jnp.tile(jnp.clip(vol_ratio_5m[:, None], 0, 5) / 5.0, (1, n)))
    output = output.at[9].set(jnp.tile(jnp.clip(vol_ratio_15m[:, None], 0, 5) / 5.0, (1, n)))
    output = output.at[10].set(jnp.tile(jnp.clip(vol_ratio_30m[:, None], 0, 5) / 5.0, (1, n)))
    output = output.at[11].set(jnp.tile(jnp.clip(vol_ratio_60m[:, None], 0, 5) / 5.0, (1, n)))

    momentum_5 = (prices[:, -1] - prices[:, -5]) / (price_range[:, 0] + 1e-8)
    momentum_15 = (prices[:, -1] - prices[:, -15]) / (price_range[:, 0] + 1e-8)
    momentum_30 = (prices[:, -1] - prices[:, -30]) / (price_range[:, 0] + 1e-8)
    momentum_60 = (prices[:, -1] - prices[:, 0]) / (price_range[:, 0] + 1e-8)

    output = output.at[12].set(jnp.tile(jnp.clip(momentum_5[:, None], -1, 1), (1, n)))
    output = output.at[13].set(jnp.tile(jnp.clip(momentum_15[:, None], -1, 1), (1, n)))
    output = output.at[14].set(jnp.tile(jnp.clip(momentum_30[:, None], -1, 1), (1, n)))
    output = output.at[15].set(jnp.tile(jnp.clip(momentum_60[:, None], -1, 1), (1, n)))

    vol_5 = jnp.std(returns_1m[:, -5:], axis=1)
    vol_15 = jnp.std(returns_1m[:, -15:], axis=1)
    vol_30 = jnp.std(returns_1m[:, -30:], axis=1)
    vol_60 = jnp.std(returns_1m, axis=1)

    output = output.at[16].set(jnp.tile(jnp.clip(vol_5[:, None] * config.vol_scale, 0, 1), (1, n)))
    output = output.at[17].set(jnp.tile(jnp.clip(vol_15[:, None] * config.vol_scale, 0, 1), (1, n)))
    output = output.at[18].set(jnp.tile(jnp.clip(vol_30[:, None] * config.vol_scale, 0, 1), (1, n)))
    output = output.at[19].set(jnp.tile(jnp.clip(vol_60[:, None] * config.vol_scale, 0, 1), (1, n)))

    def compute_percentile(window_prices, window):
        window_data = window_prices[:, -window:]
        w_min = jnp.min(window_data, axis=1)
        w_max = jnp.max(window_data, axis=1)
        w_range = w_max - w_min
        w_range = jnp.where(w_range < 1e-8, 1.0, w_range)
        return (window_prices[:, -1] - w_min) / w_range

    pct_5 = compute_percentile(prices, 5)
    pct_15 = compute_percentile(prices, 15)
    pct_30 = compute_percentile(prices, 30)
    pct_60 = compute_percentile(prices, 60)

    output = output.at[20].set(jnp.tile(pct_5[:, None], (1, n)))
    output = output.at[21].set(jnp.tile(pct_15[:, None], (1, n)))
    output = output.at[22].set(jnp.tile(pct_30[:, None], (1, n)))
    output = output.at[23].set(jnp.tile(pct_60[:, None], (1, n)))

    vwap_window = 20
    vwap_prices = prices[:, -vwap_window:]
    vwap_volumes = volumes[:, -vwap_window:]
    vol_sum = jnp.sum(vwap_volumes, axis=1, keepdims=True) + 1e-8
    vwap = jnp.sum(vwap_prices * vwap_volumes, axis=1) / vol_sum[:, 0]
    vwap_norm = (vwap - price_min[:, 0]) / (price_range[:, 0] + 1e-8)

    price_std = jnp.std(vwap_prices, axis=1)
    upper_band = (vwap + price_std - price_min[:, 0]) / (price_range[:, 0] + 1e-8)
    lower_band = (vwap - price_std - price_min[:, 0]) / (price_range[:, 0] + 1e-8)

    output = output.at[24].set(jnp.tile(vwap_norm[:, None], (1, n)))
    output = output.at[25].set(jnp.tile(jnp.clip(upper_band[:, None], 0, 1), (1, n)))
    output = output.at[26].set(jnp.tile(jnp.clip(lower_band[:, None], 0, 1), (1, n)))

    n_f = max(float(n - 1), 1.0)
    lookback_20 = min(20, prices.shape[-1] - 1)

    cur_r1 = returns_1m[:, -1]
    rank_1 = jnp.argsort(jnp.argsort(cur_r1)).astype(jnp.float32) / n_f
    r20 = (prices[:, -1] - prices[:, -lookback_20]) / (prices[:, -lookback_20] + 1e-8)
    r20 = jnp.clip(r20, -config.return_clip, config.return_clip)
    rank_20 = jnp.argsort(jnp.argsort(r20)).astype(jnp.float32) / n_f

    uni_rets = jnp.mean(returns_1m, axis=0)
    asset_centered = returns_1m - jnp.mean(returns_1m, axis=1, keepdims=True)
    uni_centered = uni_rets - jnp.mean(uni_rets)
    cov = jnp.mean(asset_centered * uni_centered[None, :], axis=1)
    var_uni = jnp.mean(uni_centered ** 2)
    beta_raw = cov / (var_uni + 1e-8)
    beta_scaled = jnp.clip(beta_raw, -3.0, 3.0) / 3.0

    residuals = returns_1m - beta_raw[:, None] * uni_rets[None, :]
    idio_vol = jnp.std(residuals, axis=1)
    idio_vol_scaled = jnp.clip(idio_vol * config.vol_scale, 0, 1)

    uni_mean_cur = jnp.mean(cur_r1)
    xs_mom = jnp.clip((cur_r1 - uni_mean_cur) * 100.0, -1.0, 1.0)

    if config.n_channels >= 32:
        output = output.at[27].set(jnp.tile(rank_1[:, None], (1, n)))
        output = output.at[28].set(jnp.tile(rank_20[:, None], (1, n)))
        output = output.at[29].set(jnp.tile(beta_scaled[:, None], (1, n)))
        output = output.at[30].set(jnp.tile(idio_vol_scaled[:, None], (1, n)))
        output = output.at[31].set(jnp.tile(xs_mom[:, None], (1, n)))

    return output


def run():
    overall_ok = True
    for seed_label, seed_v, sigma, vol_min, vol_max in [
        ("seed=7", 7, 0.01, 0.5, 2.0),
        ("seed=42", 42, 0.02, 0.1, 5.0),
        ("seed=123", 123, 0.005, 1.0, 3.0),
    ]:
        rng = np.random.default_rng(seed_v)
        n_assets, ws = 64, 64
        base = 100.0
        log_walk = np.cumsum(rng.normal(0, sigma, size=(n_assets, ws)), axis=1)
        prices_np = (base * np.exp(log_walk)).astype(np.float32)
        volumes_np = (rng.uniform(vol_min, vol_max, size=(n_assets, ws)) * 1000.0).astype(np.float32)
        highs_np = (prices_np * (1.0 + rng.uniform(0, 0.005, size=prices_np.shape))).astype(np.float32)
        lows_np = (prices_np * (1.0 - rng.uniform(0, 0.005, size=prices_np.shape))).astype(np.float32)

        prices = jnp.asarray(prices_np)
        volumes = jnp.asarray(volumes_np)
        highs = jnp.asarray(highs_np)
        lows = jnp.asarray(lows_np)

        cfg = EncoderConfig(n_assets=n_assets, window_size=ws, n_channels=32)
        cfg27 = EncoderConfig(n_assets=n_assets, window_size=ws, n_channels=27)

        # Temporal: jax_old vs jax_new (post-768 split-wrapper)
        old_t = jax.device_get(jax_old_temporal(prices, volumes, highs, lows, cfg))
        new_t = jax.device_get(_compute_all_channels_temporal(prices, volumes, highs, lows, cfg))
        eq_t = np.array_equal(old_t, new_t)
        diff_t = float(np.max(np.abs(old_t - new_t)))

        # Tiled: jax_old vs jax_new
        old_tl = jax.device_get(jax_old_tiled(prices, volumes, highs, lows, cfg))
        new_tl = jax.device_get(_compute_all_channels(prices, volumes, highs, lows, cfg))
        eq_tl = np.array_equal(old_tl, new_tl)
        diff_tl = float(np.max(np.abs(old_tl - new_tl)))

        # 27ch legacy
        old_t27 = jax.device_get(jax_old_temporal(prices, volumes, highs, lows, cfg27))
        new_t27 = jax.device_get(_compute_all_channels_temporal(prices, volumes, highs, lows, cfg27))
        eq_t27 = np.array_equal(old_t27, new_t27)
        diff_t27 = float(np.max(np.abs(old_t27 - new_t27)))

        old_tl27 = jax.device_get(jax_old_tiled(prices, volumes, highs, lows, cfg27))
        new_tl27 = jax.device_get(_compute_all_channels(prices, volumes, highs, lows, cfg27))
        eq_tl27 = np.array_equal(old_tl27, new_tl27)
        diff_tl27 = float(np.max(np.abs(old_tl27 - new_tl27)))

        print(f"[{seed_label}] JAX-old vs JAX-new TEMPORAL 32ch: array_equal={eq_t}, max abs diff={diff_t}")
        print(f"[{seed_label}] JAX-old vs JAX-new TILED    32ch: array_equal={eq_tl}, max abs diff={diff_tl}")
        print(f"[{seed_label}] JAX-old vs JAX-new TEMPORAL 27ch: array_equal={eq_t27}, max abs diff={diff_t27}")
        print(f"[{seed_label}] JAX-old vs JAX-new TILED    27ch: array_equal={eq_tl27}, max abs diff={diff_tl27}")

        if not (eq_t and eq_tl and eq_t27 and eq_tl27):
            overall_ok = False

    print()
    print("JAX-OLD vs JAX-NEW BIT IDENTITY:", "PASS" if overall_ok else "FAIL")
    return overall_ok


if __name__ == "__main__":
    ok = run()
    raise SystemExit(0 if ok else 1)
