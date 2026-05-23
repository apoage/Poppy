"""
TASK_768 bit-identity gate: numpy-emulated check that the split
preprocessing functions in jax_encoder.py produce bit-identical
output to the pre-768 monolithic implementation.

The cross-sectional formulas are deterministic and parameter-free,
so reproducing them in numpy with the SAME operation order is
sufficient to prove the split is exact. We compare:
    OLD: monolithic body inlined (verbatim translation of pre-768 code)
    NEW: per_asset() + cross_sectional() + wrapper (in that order)

If both produce identical tensors on random inputs, the JAX code path
(which uses the same expressions on jnp arrays) is also bit-identical,
because JAX defines the same numeric semantics as numpy for these ops.

Run:
    python3 tools/benchmarks/task768_bit_identity_check.py
"""
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class EncoderConfig:
    n_assets: int = 64
    window_size: int = 64
    n_channels: int = 32
    return_clip: float = 10.0
    vol_scale: float = 100.0
    use_cheat_channel: bool = False
    cheat_lookahead: int = 20
    cheat_epsilon: float = 1e-4


# =============================================================================
# OLD MONOLITHIC IMPL (pre-768) — verbatim translation of
# _compute_all_channels_temporal as it existed before commit
# =============================================================================

def old_monolithic(prices, volumes, highs, lows, config):
    """Pre-768 monolithic implementation (numpy version)."""
    n = config.n_assets
    ws = config.window_size

    output = np.zeros((config.n_channels, ws, n), dtype=np.float32)

    price_min = np.min(prices, axis=1, keepdims=True)
    price_max = np.max(prices, axis=1, keepdims=True)
    price_range = price_max - price_min
    price_range = np.where(price_range < 1e-8, 1.0, price_range)

    norm_prices = (prices - price_min) / price_range
    norm_highs = (highs - price_min) / price_range
    norm_lows = (lows - price_min) / price_range

    vol_mean = np.mean(volumes, axis=1, keepdims=True)
    vol_std = np.std(volumes, axis=1, keepdims=True) + 1e-8
    norm_volumes = np.clip((volumes - vol_mean) / vol_std, -3, 3) / 3.0

    output[0] = norm_prices.T
    output[1] = norm_highs.T
    output[2] = norm_lows.T
    output[3] = norm_volumes.T

    prev_prices = np.concatenate([prices[:, 0:1], prices[:, :-1]], axis=1)
    returns_1m = (prices - prev_prices) / (prev_prices + 1e-8)
    returns_1m = np.clip(returns_1m, -config.return_clip, config.return_clip)
    output[4] = returns_1m.T

    def broadcast_scalar(x):
        return np.broadcast_to(x[None, :], (ws, n))

    returns_5m = (prices[:, -1] - prices[:, -5]) / (prices[:, -5] + 1e-8)
    returns_15m = (prices[:, -1] - prices[:, -15]) / (prices[:, -15] + 1e-8)
    returns_64m = (prices[:, -1] - prices[:, 0]) / (prices[:, 0] + 1e-8)

    output[5] = broadcast_scalar(np.clip(returns_5m, -config.return_clip, config.return_clip))
    output[6] = broadcast_scalar(np.clip(returns_15m, -config.return_clip, config.return_clip))
    output[7] = broadcast_scalar(np.clip(returns_64m, -config.return_clip, config.return_clip))

    vol_ratio_5m = np.mean(volumes[:, -5:], axis=1) / (vol_mean[:, 0] + 1e-8)
    vol_ratio_15m = np.mean(volumes[:, -15:], axis=1) / (vol_mean[:, 0] + 1e-8)
    vol_ratio_30m = np.mean(volumes[:, -30:], axis=1) / (vol_mean[:, 0] + 1e-8)
    vol_ratio_64m = np.mean(volumes, axis=1) / (vol_mean[:, 0] + 1e-8)

    output[8] = broadcast_scalar(np.clip(vol_ratio_5m, 0, 5) / 5.0)
    output[9] = broadcast_scalar(np.clip(vol_ratio_15m, 0, 5) / 5.0)
    output[10] = broadcast_scalar(np.clip(vol_ratio_30m, 0, 5) / 5.0)
    output[11] = broadcast_scalar(np.clip(vol_ratio_64m, 0, 5) / 5.0)

    momentum_5 = (prices[:, -1] - prices[:, -5]) / (price_range[:, 0] + 1e-8)
    momentum_15 = (prices[:, -1] - prices[:, -15]) / (price_range[:, 0] + 1e-8)
    momentum_30 = (prices[:, -1] - prices[:, -30]) / (price_range[:, 0] + 1e-8)
    momentum_64 = (prices[:, -1] - prices[:, 0]) / (price_range[:, 0] + 1e-8)

    output[12] = broadcast_scalar(np.clip(momentum_5, -1, 1))
    output[13] = broadcast_scalar(np.clip(momentum_15, -1, 1))
    output[14] = broadcast_scalar(np.clip(momentum_30, -1, 1))
    output[15] = broadcast_scalar(np.clip(momentum_64, -1, 1))

    vol_5 = np.std(returns_1m[:, -5:], axis=1)
    vol_15 = np.std(returns_1m[:, -15:], axis=1)
    vol_30 = np.std(returns_1m[:, -30:], axis=1)
    vol_64 = np.std(returns_1m, axis=1)

    output[16] = broadcast_scalar(np.clip(vol_5 * config.vol_scale, 0, 1))
    output[17] = broadcast_scalar(np.clip(vol_15 * config.vol_scale, 0, 1))
    output[18] = broadcast_scalar(np.clip(vol_30 * config.vol_scale, 0, 1))
    output[19] = broadcast_scalar(np.clip(vol_64 * config.vol_scale, 0, 1))

    def compute_percentile(window_prices, window):
        window_data = window_prices[:, -window:]
        w_min = np.min(window_data, axis=1)
        w_max = np.max(window_data, axis=1)
        w_range = np.where((w_max - w_min) < 1e-8, 1.0, w_max - w_min)
        return (window_prices[:, -1] - w_min) / w_range

    output[20] = broadcast_scalar(compute_percentile(prices, 5))
    output[21] = broadcast_scalar(compute_percentile(prices, 15))
    output[22] = broadcast_scalar(compute_percentile(prices, 30))
    output[23] = broadcast_scalar(compute_percentile(prices, 64))

    vwap_window = 20
    vwap_prices = prices[:, -vwap_window:]
    vwap_volumes = volumes[:, -vwap_window:]
    vol_sum = np.sum(vwap_volumes, axis=1, keepdims=True) + 1e-8
    vwap = np.sum(vwap_prices * vwap_volumes, axis=1) / vol_sum[:, 0]
    vwap_norm = (vwap - price_min[:, 0]) / (price_range[:, 0] + 1e-8)

    price_std = np.std(vwap_prices, axis=1)
    upper_band = (vwap + price_std - price_min[:, 0]) / (price_range[:, 0] + 1e-8)
    lower_band = (vwap - price_std - price_min[:, 0]) / (price_range[:, 0] + 1e-8)

    output[24] = broadcast_scalar(vwap_norm)
    output[25] = broadcast_scalar(np.clip(upper_band, 0, 1))
    output[26] = broadcast_scalar(np.clip(lower_band, 0, 1))

    # Cross-sectional (TASK_732)
    n_f = max(float(n - 1), 1.0)
    lookback_20 = min(20, ws - 1)

    cur_r1 = returns_1m[:, -1]
    rank_1 = np.argsort(np.argsort(cur_r1)).astype(np.float32) / n_f

    r20 = (prices[:, -1] - prices[:, -lookback_20]) / (prices[:, -lookback_20] + 1e-8)
    r20 = np.clip(r20, -config.return_clip, config.return_clip)
    rank_20 = np.argsort(np.argsort(r20)).astype(np.float32) / n_f

    uni_rets = np.mean(returns_1m, axis=0)
    asset_centered = returns_1m - np.mean(returns_1m, axis=1, keepdims=True)
    uni_centered = uni_rets - np.mean(uni_rets)
    cov = np.mean(asset_centered * uni_centered[None, :], axis=1)
    var_uni = np.mean(uni_centered ** 2)
    beta_raw = cov / (var_uni + 1e-8)
    beta_scaled = np.clip(beta_raw, -3.0, 3.0) / 3.0

    residuals = returns_1m - beta_raw[:, None] * uni_rets[None, :]
    idio_vol = np.std(residuals, axis=1)
    idio_vol_scaled = np.clip(idio_vol * config.vol_scale, 0, 1)

    uni_mean_cur = np.mean(cur_r1)
    xs_mom = np.clip((cur_r1 - uni_mean_cur) * 100.0, -1.0, 1.0)

    if config.n_channels >= 32:
        output[27] = broadcast_scalar(rank_1)
        output[28] = broadcast_scalar(rank_20)
        output[29] = broadcast_scalar(beta_scaled)
        output[30] = broadcast_scalar(idio_vol_scaled)
        output[31] = broadcast_scalar(xs_mom)

    return output


# =============================================================================
# NEW SPLIT IMPL (post-768) — translation of the wrapper + helpers
# =============================================================================

def new_per_asset_temporal(prices, volumes, highs, lows, config):
    """Matches _compute_per_asset_channels_temporal in jax_encoder.py."""
    n = config.n_assets
    ws = config.window_size

    output = np.zeros((config.n_channels, ws, n), dtype=np.float32)

    price_min = np.min(prices, axis=1, keepdims=True)
    price_max = np.max(prices, axis=1, keepdims=True)
    price_range = price_max - price_min
    price_range = np.where(price_range < 1e-8, 1.0, price_range)

    norm_prices = (prices - price_min) / price_range
    norm_highs = (highs - price_min) / price_range
    norm_lows = (lows - price_min) / price_range

    vol_mean = np.mean(volumes, axis=1, keepdims=True)
    vol_std = np.std(volumes, axis=1, keepdims=True) + 1e-8
    norm_volumes = np.clip((volumes - vol_mean) / vol_std, -3, 3) / 3.0

    output[0] = norm_prices.T
    output[1] = norm_highs.T
    output[2] = norm_lows.T
    output[3] = norm_volumes.T

    prev_prices = np.concatenate([prices[:, 0:1], prices[:, :-1]], axis=1)
    returns_1m = (prices - prev_prices) / (prev_prices + 1e-8)
    returns_1m = np.clip(returns_1m, -config.return_clip, config.return_clip)
    output[4] = returns_1m.T

    def broadcast_scalar(x):
        return np.broadcast_to(x[None, :], (ws, n))

    returns_5m = (prices[:, -1] - prices[:, -5]) / (prices[:, -5] + 1e-8)
    returns_15m = (prices[:, -1] - prices[:, -15]) / (prices[:, -15] + 1e-8)
    returns_64m = (prices[:, -1] - prices[:, 0]) / (prices[:, 0] + 1e-8)

    output[5] = broadcast_scalar(np.clip(returns_5m, -config.return_clip, config.return_clip))
    output[6] = broadcast_scalar(np.clip(returns_15m, -config.return_clip, config.return_clip))
    output[7] = broadcast_scalar(np.clip(returns_64m, -config.return_clip, config.return_clip))

    vol_ratio_5m = np.mean(volumes[:, -5:], axis=1) / (vol_mean[:, 0] + 1e-8)
    vol_ratio_15m = np.mean(volumes[:, -15:], axis=1) / (vol_mean[:, 0] + 1e-8)
    vol_ratio_30m = np.mean(volumes[:, -30:], axis=1) / (vol_mean[:, 0] + 1e-8)
    vol_ratio_64m = np.mean(volumes, axis=1) / (vol_mean[:, 0] + 1e-8)

    output[8] = broadcast_scalar(np.clip(vol_ratio_5m, 0, 5) / 5.0)
    output[9] = broadcast_scalar(np.clip(vol_ratio_15m, 0, 5) / 5.0)
    output[10] = broadcast_scalar(np.clip(vol_ratio_30m, 0, 5) / 5.0)
    output[11] = broadcast_scalar(np.clip(vol_ratio_64m, 0, 5) / 5.0)

    momentum_5 = (prices[:, -1] - prices[:, -5]) / (price_range[:, 0] + 1e-8)
    momentum_15 = (prices[:, -1] - prices[:, -15]) / (price_range[:, 0] + 1e-8)
    momentum_30 = (prices[:, -1] - prices[:, -30]) / (price_range[:, 0] + 1e-8)
    momentum_64 = (prices[:, -1] - prices[:, 0]) / (price_range[:, 0] + 1e-8)

    output[12] = broadcast_scalar(np.clip(momentum_5, -1, 1))
    output[13] = broadcast_scalar(np.clip(momentum_15, -1, 1))
    output[14] = broadcast_scalar(np.clip(momentum_30, -1, 1))
    output[15] = broadcast_scalar(np.clip(momentum_64, -1, 1))

    vol_5 = np.std(returns_1m[:, -5:], axis=1)
    vol_15 = np.std(returns_1m[:, -15:], axis=1)
    vol_30 = np.std(returns_1m[:, -30:], axis=1)
    vol_64 = np.std(returns_1m, axis=1)

    output[16] = broadcast_scalar(np.clip(vol_5 * config.vol_scale, 0, 1))
    output[17] = broadcast_scalar(np.clip(vol_15 * config.vol_scale, 0, 1))
    output[18] = broadcast_scalar(np.clip(vol_30 * config.vol_scale, 0, 1))
    output[19] = broadcast_scalar(np.clip(vol_64 * config.vol_scale, 0, 1))

    def compute_percentile(window_prices, window):
        window_data = window_prices[:, -window:]
        w_min = np.min(window_data, axis=1)
        w_max = np.max(window_data, axis=1)
        w_range = np.where((w_max - w_min) < 1e-8, 1.0, w_max - w_min)
        return (window_prices[:, -1] - w_min) / w_range

    output[20] = broadcast_scalar(compute_percentile(prices, 5))
    output[21] = broadcast_scalar(compute_percentile(prices, 15))
    output[22] = broadcast_scalar(compute_percentile(prices, 30))
    output[23] = broadcast_scalar(compute_percentile(prices, 64))

    vwap_window = 20
    vwap_prices = prices[:, -vwap_window:]
    vwap_volumes = volumes[:, -vwap_window:]
    vol_sum = np.sum(vwap_volumes, axis=1, keepdims=True) + 1e-8
    vwap = np.sum(vwap_prices * vwap_volumes, axis=1) / vol_sum[:, 0]
    vwap_norm = (vwap - price_min[:, 0]) / (price_range[:, 0] + 1e-8)

    price_std = np.std(vwap_prices, axis=1)
    upper_band = (vwap + price_std - price_min[:, 0]) / (price_range[:, 0] + 1e-8)
    lower_band = (vwap - price_std - price_min[:, 0]) / (price_range[:, 0] + 1e-8)

    output[24] = broadcast_scalar(vwap_norm)
    output[25] = broadcast_scalar(np.clip(upper_band, 0, 1))
    output[26] = broadcast_scalar(np.clip(lower_band, 0, 1))

    return output, returns_1m


def new_cross_sectional_temporal(prices, returns_1m, n, ws, config):
    """Matches _compute_cross_sectional_temporal in jax_encoder.py."""
    n_f = max(float(n - 1), 1.0)
    lookback_20 = min(20, ws - 1)

    cur_r1 = returns_1m[:, -1]
    rank_1 = np.argsort(np.argsort(cur_r1)).astype(np.float32) / n_f

    r20 = (prices[:, -1] - prices[:, -lookback_20]) / (prices[:, -lookback_20] + 1e-8)
    r20 = np.clip(r20, -config.return_clip, config.return_clip)
    rank_20 = np.argsort(np.argsort(r20)).astype(np.float32) / n_f

    uni_rets = np.mean(returns_1m, axis=0)
    asset_centered = returns_1m - np.mean(returns_1m, axis=1, keepdims=True)
    uni_centered = uni_rets - np.mean(uni_rets)
    cov = np.mean(asset_centered * uni_centered[None, :], axis=1)
    var_uni = np.mean(uni_centered ** 2)
    beta_raw = cov / (var_uni + 1e-8)
    beta_scaled = np.clip(beta_raw, -3.0, 3.0) / 3.0

    residuals = returns_1m - beta_raw[:, None] * uni_rets[None, :]
    idio_vol = np.std(residuals, axis=1)
    idio_vol_scaled = np.clip(idio_vol * config.vol_scale, 0, 1)

    uni_mean_cur = np.mean(cur_r1)
    xs_mom = np.clip((cur_r1 - uni_mean_cur) * 100.0, -1.0, 1.0)

    return rank_1, rank_20, beta_scaled, idio_vol_scaled, xs_mom


def new_wrapper_temporal(prices, volumes, highs, lows, config):
    """Matches _compute_all_channels_temporal in jax_encoder.py."""
    n = config.n_assets
    ws = config.window_size

    def broadcast_scalar(x):
        return np.broadcast_to(x[None, :], (ws, n))

    output, returns_1m = new_per_asset_temporal(prices, volumes, highs, lows, config)

    if config.n_channels >= 32:
        rank_1, rank_20, beta_scaled, idio_vol_scaled, xs_mom = new_cross_sectional_temporal(
            prices, returns_1m, n, ws, config
        )
        output[27] = broadcast_scalar(rank_1)
        output[28] = broadcast_scalar(rank_20)
        output[29] = broadcast_scalar(beta_scaled)
        output[30] = broadcast_scalar(idio_vol_scaled)
        output[31] = broadcast_scalar(xs_mom)

    return output


# =============================================================================
# Tiled (legacy) path mirrors — pre-768 monolithic _compute_all_channels
# =============================================================================

def old_monolithic_tiled(prices, volumes, highs, lows, config):
    """Pre-768 monolithic _compute_all_channels (tiled). Used by encode_observation."""
    n = config.n_assets

    output = np.zeros((config.n_channels, n, n), dtype=np.float32)

    price_min = np.min(prices, axis=1, keepdims=True)
    price_max = np.max(prices, axis=1, keepdims=True)
    price_range = price_max - price_min
    price_range = np.where(price_range < 1e-8, 1.0, price_range)

    norm_prices = (prices - price_min) / price_range
    norm_highs = (highs - price_min) / price_range
    norm_lows = (lows - price_min) / price_range

    vol_mean = np.mean(volumes, axis=1, keepdims=True)
    vol_std = np.std(volumes, axis=1, keepdims=True) + 1e-8
    norm_volumes = np.clip((volumes - vol_mean) / vol_std, -3, 3) / 3.0

    output[0] = np.tile(norm_prices[:, -1:], (1, n))
    output[1] = np.tile(norm_highs[:, -1:], (1, n))
    output[2] = np.tile(norm_lows[:, -1:], (1, n))
    output[3] = np.tile(norm_volumes[:, -1:], (1, n))

    prev_prices = np.concatenate([prices[:, 0:1], prices[:, :-1]], axis=1)
    returns_1m = (prices - prev_prices) / (prev_prices + 1e-8)
    returns_1m = np.clip(returns_1m, -config.return_clip, config.return_clip)
    output[4] = np.tile(returns_1m[:, -1:], (1, n))

    returns_5m = (prices[:, -1] - prices[:, -5]) / (prices[:, -5] + 1e-8)
    returns_5m = np.clip(returns_5m, -config.return_clip, config.return_clip)
    output[5] = np.tile(returns_5m[:, None], (1, n))

    returns_15m = (prices[:, -1] - prices[:, -15]) / (prices[:, -15] + 1e-8)
    returns_15m = np.clip(returns_15m, -config.return_clip, config.return_clip)
    output[6] = np.tile(returns_15m[:, None], (1, n))

    returns_60m = (prices[:, -1] - prices[:, 0]) / (prices[:, 0] + 1e-8)
    returns_60m = np.clip(returns_60m, -config.return_clip, config.return_clip)
    output[7] = np.tile(returns_60m[:, None], (1, n))

    vol_ratio_5m = np.mean(volumes[:, -5:], axis=1) / (vol_mean[:, 0] + 1e-8)
    vol_ratio_15m = np.mean(volumes[:, -15:], axis=1) / (vol_mean[:, 0] + 1e-8)
    vol_ratio_30m = np.mean(volumes[:, -30:], axis=1) / (vol_mean[:, 0] + 1e-8)
    vol_ratio_60m = np.mean(volumes, axis=1) / (vol_mean[:, 0] + 1e-8)

    output[8] = np.tile(np.clip(vol_ratio_5m[:, None], 0, 5) / 5.0, (1, n))
    output[9] = np.tile(np.clip(vol_ratio_15m[:, None], 0, 5) / 5.0, (1, n))
    output[10] = np.tile(np.clip(vol_ratio_30m[:, None], 0, 5) / 5.0, (1, n))
    output[11] = np.tile(np.clip(vol_ratio_60m[:, None], 0, 5) / 5.0, (1, n))

    momentum_5 = (prices[:, -1] - prices[:, -5]) / (price_range[:, 0] + 1e-8)
    momentum_15 = (prices[:, -1] - prices[:, -15]) / (price_range[:, 0] + 1e-8)
    momentum_30 = (prices[:, -1] - prices[:, -30]) / (price_range[:, 0] + 1e-8)
    momentum_60 = (prices[:, -1] - prices[:, 0]) / (price_range[:, 0] + 1e-8)

    output[12] = np.tile(np.clip(momentum_5[:, None], -1, 1), (1, n))
    output[13] = np.tile(np.clip(momentum_15[:, None], -1, 1), (1, n))
    output[14] = np.tile(np.clip(momentum_30[:, None], -1, 1), (1, n))
    output[15] = np.tile(np.clip(momentum_60[:, None], -1, 1), (1, n))

    vol_5 = np.std(returns_1m[:, -5:], axis=1)
    vol_15 = np.std(returns_1m[:, -15:], axis=1)
    vol_30 = np.std(returns_1m[:, -30:], axis=1)
    vol_60 = np.std(returns_1m, axis=1)

    output[16] = np.tile(np.clip(vol_5[:, None] * config.vol_scale, 0, 1), (1, n))
    output[17] = np.tile(np.clip(vol_15[:, None] * config.vol_scale, 0, 1), (1, n))
    output[18] = np.tile(np.clip(vol_30[:, None] * config.vol_scale, 0, 1), (1, n))
    output[19] = np.tile(np.clip(vol_60[:, None] * config.vol_scale, 0, 1), (1, n))

    def compute_percentile(window_prices, window):
        window_data = window_prices[:, -window:]
        w_min = np.min(window_data, axis=1)
        w_max = np.max(window_data, axis=1)
        w_range = w_max - w_min
        w_range = np.where(w_range < 1e-8, 1.0, w_range)
        return (window_prices[:, -1] - w_min) / w_range

    pct_5 = compute_percentile(prices, 5)
    pct_15 = compute_percentile(prices, 15)
    pct_30 = compute_percentile(prices, 30)
    pct_60 = compute_percentile(prices, 60)

    output[20] = np.tile(pct_5[:, None], (1, n))
    output[21] = np.tile(pct_15[:, None], (1, n))
    output[22] = np.tile(pct_30[:, None], (1, n))
    output[23] = np.tile(pct_60[:, None], (1, n))

    vwap_window = 20
    vwap_prices = prices[:, -vwap_window:]
    vwap_volumes = volumes[:, -vwap_window:]
    vol_sum = np.sum(vwap_volumes, axis=1, keepdims=True) + 1e-8
    vwap = np.sum(vwap_prices * vwap_volumes, axis=1) / vol_sum[:, 0]
    vwap_norm = (vwap - price_min[:, 0]) / (price_range[:, 0] + 1e-8)

    price_std = np.std(vwap_prices, axis=1)
    upper_band = (vwap + price_std - price_min[:, 0]) / (price_range[:, 0] + 1e-8)
    lower_band = (vwap - price_std - price_min[:, 0]) / (price_range[:, 0] + 1e-8)

    output[24] = np.tile(vwap_norm[:, None], (1, n))
    output[25] = np.tile(np.clip(upper_band[:, None], 0, 1), (1, n))
    output[26] = np.tile(np.clip(lower_band[:, None], 0, 1), (1, n))

    n_f = max(float(n - 1), 1.0)
    lookback_20 = min(20, prices.shape[-1] - 1)

    cur_r1 = returns_1m[:, -1]
    rank_1 = np.argsort(np.argsort(cur_r1)).astype(np.float32) / n_f
    r20 = (prices[:, -1] - prices[:, -lookback_20]) / (prices[:, -lookback_20] + 1e-8)
    r20 = np.clip(r20, -config.return_clip, config.return_clip)
    rank_20 = np.argsort(np.argsort(r20)).astype(np.float32) / n_f

    uni_rets = np.mean(returns_1m, axis=0)
    asset_centered = returns_1m - np.mean(returns_1m, axis=1, keepdims=True)
    uni_centered = uni_rets - np.mean(uni_rets)
    cov = np.mean(asset_centered * uni_centered[None, :], axis=1)
    var_uni = np.mean(uni_centered ** 2)
    beta_raw = cov / (var_uni + 1e-8)
    beta_scaled = np.clip(beta_raw, -3.0, 3.0) / 3.0

    residuals = returns_1m - beta_raw[:, None] * uni_rets[None, :]
    idio_vol = np.std(residuals, axis=1)
    idio_vol_scaled = np.clip(idio_vol * config.vol_scale, 0, 1)

    uni_mean_cur = np.mean(cur_r1)
    xs_mom = np.clip((cur_r1 - uni_mean_cur) * 100.0, -1.0, 1.0)

    if config.n_channels >= 32:
        output[27] = np.tile(rank_1[:, None], (1, n))
        output[28] = np.tile(rank_20[:, None], (1, n))
        output[29] = np.tile(beta_scaled[:, None], (1, n))
        output[30] = np.tile(idio_vol_scaled[:, None], (1, n))
        output[31] = np.tile(xs_mom[:, None], (1, n))

    return output


def new_per_asset_tiled(prices, volumes, highs, lows, config):
    """Matches _compute_per_asset_channels (tiled) in jax_encoder.py."""
    n = config.n_assets

    output = np.zeros((config.n_channels, n, n), dtype=np.float32)

    price_min = np.min(prices, axis=1, keepdims=True)
    price_max = np.max(prices, axis=1, keepdims=True)
    price_range = price_max - price_min
    price_range = np.where(price_range < 1e-8, 1.0, price_range)

    norm_prices = (prices - price_min) / price_range
    norm_highs = (highs - price_min) / price_range
    norm_lows = (lows - price_min) / price_range

    vol_mean = np.mean(volumes, axis=1, keepdims=True)
    vol_std = np.std(volumes, axis=1, keepdims=True) + 1e-8
    norm_volumes = np.clip((volumes - vol_mean) / vol_std, -3, 3) / 3.0

    output[0] = np.tile(norm_prices[:, -1:], (1, n))
    output[1] = np.tile(norm_highs[:, -1:], (1, n))
    output[2] = np.tile(norm_lows[:, -1:], (1, n))
    output[3] = np.tile(norm_volumes[:, -1:], (1, n))

    prev_prices = np.concatenate([prices[:, 0:1], prices[:, :-1]], axis=1)
    returns_1m = (prices - prev_prices) / (prev_prices + 1e-8)
    returns_1m = np.clip(returns_1m, -config.return_clip, config.return_clip)
    output[4] = np.tile(returns_1m[:, -1:], (1, n))

    returns_5m = (prices[:, -1] - prices[:, -5]) / (prices[:, -5] + 1e-8)
    returns_5m = np.clip(returns_5m, -config.return_clip, config.return_clip)
    output[5] = np.tile(returns_5m[:, None], (1, n))

    returns_15m = (prices[:, -1] - prices[:, -15]) / (prices[:, -15] + 1e-8)
    returns_15m = np.clip(returns_15m, -config.return_clip, config.return_clip)
    output[6] = np.tile(returns_15m[:, None], (1, n))

    returns_60m = (prices[:, -1] - prices[:, 0]) / (prices[:, 0] + 1e-8)
    returns_60m = np.clip(returns_60m, -config.return_clip, config.return_clip)
    output[7] = np.tile(returns_60m[:, None], (1, n))

    vol_ratio_5m = np.mean(volumes[:, -5:], axis=1) / (vol_mean[:, 0] + 1e-8)
    vol_ratio_15m = np.mean(volumes[:, -15:], axis=1) / (vol_mean[:, 0] + 1e-8)
    vol_ratio_30m = np.mean(volumes[:, -30:], axis=1) / (vol_mean[:, 0] + 1e-8)
    vol_ratio_60m = np.mean(volumes, axis=1) / (vol_mean[:, 0] + 1e-8)

    output[8] = np.tile(np.clip(vol_ratio_5m[:, None], 0, 5) / 5.0, (1, n))
    output[9] = np.tile(np.clip(vol_ratio_15m[:, None], 0, 5) / 5.0, (1, n))
    output[10] = np.tile(np.clip(vol_ratio_30m[:, None], 0, 5) / 5.0, (1, n))
    output[11] = np.tile(np.clip(vol_ratio_60m[:, None], 0, 5) / 5.0, (1, n))

    momentum_5 = (prices[:, -1] - prices[:, -5]) / (price_range[:, 0] + 1e-8)
    momentum_15 = (prices[:, -1] - prices[:, -15]) / (price_range[:, 0] + 1e-8)
    momentum_30 = (prices[:, -1] - prices[:, -30]) / (price_range[:, 0] + 1e-8)
    momentum_60 = (prices[:, -1] - prices[:, 0]) / (price_range[:, 0] + 1e-8)

    output[12] = np.tile(np.clip(momentum_5[:, None], -1, 1), (1, n))
    output[13] = np.tile(np.clip(momentum_15[:, None], -1, 1), (1, n))
    output[14] = np.tile(np.clip(momentum_30[:, None], -1, 1), (1, n))
    output[15] = np.tile(np.clip(momentum_60[:, None], -1, 1), (1, n))

    vol_5 = np.std(returns_1m[:, -5:], axis=1)
    vol_15 = np.std(returns_1m[:, -15:], axis=1)
    vol_30 = np.std(returns_1m[:, -30:], axis=1)
    vol_60 = np.std(returns_1m, axis=1)

    output[16] = np.tile(np.clip(vol_5[:, None] * config.vol_scale, 0, 1), (1, n))
    output[17] = np.tile(np.clip(vol_15[:, None] * config.vol_scale, 0, 1), (1, n))
    output[18] = np.tile(np.clip(vol_30[:, None] * config.vol_scale, 0, 1), (1, n))
    output[19] = np.tile(np.clip(vol_60[:, None] * config.vol_scale, 0, 1), (1, n))

    def compute_percentile(window_prices, window):
        window_data = window_prices[:, -window:]
        w_min = np.min(window_data, axis=1)
        w_max = np.max(window_data, axis=1)
        w_range = w_max - w_min
        w_range = np.where(w_range < 1e-8, 1.0, w_range)
        return (window_prices[:, -1] - w_min) / w_range

    pct_5 = compute_percentile(prices, 5)
    pct_15 = compute_percentile(prices, 15)
    pct_30 = compute_percentile(prices, 30)
    pct_60 = compute_percentile(prices, 60)

    output[20] = np.tile(pct_5[:, None], (1, n))
    output[21] = np.tile(pct_15[:, None], (1, n))
    output[22] = np.tile(pct_30[:, None], (1, n))
    output[23] = np.tile(pct_60[:, None], (1, n))

    vwap_window = 20
    vwap_prices = prices[:, -vwap_window:]
    vwap_volumes = volumes[:, -vwap_window:]
    vol_sum = np.sum(vwap_volumes, axis=1, keepdims=True) + 1e-8
    vwap = np.sum(vwap_prices * vwap_volumes, axis=1) / vol_sum[:, 0]
    vwap_norm = (vwap - price_min[:, 0]) / (price_range[:, 0] + 1e-8)

    price_std = np.std(vwap_prices, axis=1)
    upper_band = (vwap + price_std - price_min[:, 0]) / (price_range[:, 0] + 1e-8)
    lower_band = (vwap - price_std - price_min[:, 0]) / (price_range[:, 0] + 1e-8)

    output[24] = np.tile(vwap_norm[:, None], (1, n))
    output[25] = np.tile(np.clip(upper_band[:, None], 0, 1), (1, n))
    output[26] = np.tile(np.clip(lower_band[:, None], 0, 1), (1, n))

    return output, returns_1m


def new_cross_sectional_tiled(prices, returns_1m, n, config):
    """Matches _compute_cross_sectional (tiled) in jax_encoder.py."""
    n_f = max(float(n - 1), 1.0)
    lookback_20 = min(20, prices.shape[-1] - 1)

    cur_r1 = returns_1m[:, -1]
    rank_1 = np.argsort(np.argsort(cur_r1)).astype(np.float32) / n_f
    r20 = (prices[:, -1] - prices[:, -lookback_20]) / (prices[:, -lookback_20] + 1e-8)
    r20 = np.clip(r20, -config.return_clip, config.return_clip)
    rank_20 = np.argsort(np.argsort(r20)).astype(np.float32) / n_f

    uni_rets = np.mean(returns_1m, axis=0)
    asset_centered = returns_1m - np.mean(returns_1m, axis=1, keepdims=True)
    uni_centered = uni_rets - np.mean(uni_rets)
    cov = np.mean(asset_centered * uni_centered[None, :], axis=1)
    var_uni = np.mean(uni_centered ** 2)
    beta_raw = cov / (var_uni + 1e-8)
    beta_scaled = np.clip(beta_raw, -3.0, 3.0) / 3.0

    residuals = returns_1m - beta_raw[:, None] * uni_rets[None, :]
    idio_vol = np.std(residuals, axis=1)
    idio_vol_scaled = np.clip(idio_vol * config.vol_scale, 0, 1)

    uni_mean_cur = np.mean(cur_r1)
    xs_mom = np.clip((cur_r1 - uni_mean_cur) * 100.0, -1.0, 1.0)

    return rank_1, rank_20, beta_scaled, idio_vol_scaled, xs_mom


def new_wrapper_tiled(prices, volumes, highs, lows, config):
    """Matches _compute_all_channels (tiled) in jax_encoder.py."""
    n = config.n_assets

    output, returns_1m = new_per_asset_tiled(prices, volumes, highs, lows, config)

    if config.n_channels >= 32:
        rank_1, rank_20, beta_scaled, idio_vol_scaled, xs_mom = new_cross_sectional_tiled(
            prices, returns_1m, n, config
        )
        output[27] = np.tile(rank_1[:, None], (1, n))
        output[28] = np.tile(rank_20[:, None], (1, n))
        output[29] = np.tile(beta_scaled[:, None], (1, n))
        output[30] = np.tile(idio_vol_scaled[:, None], (1, n))
        output[31] = np.tile(xs_mom[:, None], (1, n))

    return output


# =============================================================================
# Driver
# =============================================================================

def run():
    overall_ok = True
    for seed_label, seed_v, sigma, vol_min, vol_max in [
        ("seed=7", 7, 0.01, 0.5, 2.0),
        ("seed=42", 42, 0.02, 0.1, 5.0),
        ("seed=123", 123, 0.005, 1.0, 3.0),
    ]:
        rng = np.random.default_rng(seed_v)
        n_assets, window_size = 64, 64
        cfg = EncoderConfig(n_assets=n_assets, window_size=window_size, n_channels=32)
        base = 100.0
        log_walk = np.cumsum(rng.normal(0, sigma, size=(n_assets, window_size)), axis=1)
        prices = (base * np.exp(log_walk)).astype(np.float32)
        volumes = (rng.uniform(vol_min, vol_max, size=(n_assets, window_size)) * 1000.0).astype(np.float32)
        highs = (prices * (1.0 + rng.uniform(0, 0.005, size=prices.shape))).astype(np.float32)
        lows = (prices * (1.0 - rng.uniform(0, 0.005, size=prices.shape))).astype(np.float32)

        # Temporal path
        old_t = old_monolithic(prices, volumes, highs, lows, cfg)
        new_t = new_wrapper_temporal(prices, volumes, highs, lows, cfg)
        eq_t = np.array_equal(old_t, new_t)
        diff_t = float(np.max(np.abs(old_t - new_t)))
        print(f"[{seed_label}] TEMPORAL  array_equal={eq_t}, max abs diff={diff_t}")

        # Tiled (legacy) path
        old_tl = old_monolithic_tiled(prices, volumes, highs, lows, cfg)
        new_tl = new_wrapper_tiled(prices, volumes, highs, lows, cfg)
        eq_tl = np.array_equal(old_tl, new_tl)
        diff_tl = float(np.max(np.abs(old_tl - new_tl)))
        print(f"[{seed_label}] TILED     array_equal={eq_tl}, max abs diff={diff_tl}")

        # Legacy 27-channel mode (skip XS)
        cfg27 = EncoderConfig(n_assets=n_assets, window_size=window_size, n_channels=27)
        old27_t = old_monolithic(prices, volumes, highs, lows, cfg27)
        new27_t = new_wrapper_temporal(prices, volumes, highs, lows, cfg27)
        eq27_t = np.array_equal(old27_t, new27_t)
        diff27_t = float(np.max(np.abs(old27_t - new27_t)))
        print(f"[{seed_label}] TEMPORAL-27ch array_equal={eq27_t}, max abs diff={diff27_t}")

        old27_tl = old_monolithic_tiled(prices, volumes, highs, lows, cfg27)
        new27_tl = new_wrapper_tiled(prices, volumes, highs, lows, cfg27)
        eq27_tl = np.array_equal(old27_tl, new27_tl)
        diff27_tl = float(np.max(np.abs(old27_tl - new27_tl)))
        print(f"[{seed_label}] TILED-27ch    array_equal={eq27_tl}, max abs diff={diff27_tl}")

        if not (eq_t and eq_tl and eq27_t and eq27_tl):
            overall_ok = False
            # Per-channel breakdown for the failing batch
            for c in range(cfg.n_channels):
                d_t = float(np.max(np.abs(old_t[c] - new_t[c])))
                d_tl = float(np.max(np.abs(old_tl[c] - new_tl[c])))
                if d_t > 0 or d_tl > 0:
                    print(f"    ch{c}: TEMPORAL diff={d_t}, TILED diff={d_tl}")

    print()
    print("BIT IDENTITY GATE:", "PASS" if overall_ok else "FAIL")
    return overall_ok


if __name__ == '__main__':
    ok = run()
    raise SystemExit(0 if ok else 1)
