"""TASK_801 — token-timeline window slicing with the TASK_800 serving-lag convention.

This is the ONE place the serving lag lives. The model never knows about lag; it
just consumes `token_ids` of shape (B, T, A). The lag is baked into how a
per-bar token timeline is sliced into the (window_start .. window_start+win_len)
window that aligns with the raw-OHLCV window the encoder sees.

Serving-lag convention (LOAD-BEARING, p99-firm from the TASK_800 re-cert on the
frozen 512-merge vocab cb=1376):
  * L1 inputs are DELAYED 3 bars: token_input[t] = token_id[t-3]. The re-cert on
    the deeper merge chain firms 99.3% of cuts within 2 bars (p99 lag 2), so the
    final-only horizon is bars <= t-3. The first 3 bars of the sliced window map
    to the reserved invalid id (-1). NOTE: the old t-2 came from the 256-merge
    fit (p99/max lag 1); the empirical max (6 on a single 0.1% cut) is a sample
    statistic, NOT a bound — strict max+1 is rejected, p99-firm is the convention.
    The ~0.7% residual against live prefix-causal serving is superseded at ladder
    gate 3 by the sealed-flag streaming annotator (lag-adaptive, zero skew). Within
    804 the residual is moot: train and eval both consume full-stream annotations
    (C3 proves internal skew is exactly zero at any lag).
  * L8 inputs are UNSHIFTED (LAG-0 exact; churn == 0, unchanged by the re-cert).
  * TARGETS stay UNSHIFTED everywhere — only *inputs* carry the delay. (Target
    slicing is the caller's job; this module only produces input token windows.)

Invalid bars (gaps, pre-first-token, and the lag-introduced leading bars) carry
token_id == -1. The model maps -1 -> reserved sentinel row, zero-masked
(see StreamingAssetEncoder._embed_token_contrib). Nothing here needs the vocab.
"""
from __future__ import annotations

import numpy as np

INVALID_TOKEN_ID = -1

# Per-language input serving lag (bars). Targets are NEVER shifted.
# Re-cert 2026-06-11 on frozen vocab runs/799_grammar/L1_RESULT_vocab.json (cb 1376,
# 512 merges): L1 lag_hist {0:960,1:22,2:11,3:3,4:3,6:1} → p99 lag 2 → input lag 3
# (p99-firm; max_lag=6 is a single-cut sample stat, not a bound — see module docstring).
LANGUAGE_INPUT_LAG = {
    "L1": 3,   # p99-firm: token_input[t] = token_id[t-3]
    "L8": 0,   # LAG-0 exact
}


def input_lag_for(language: str) -> int:
    """Return the input serving lag (in bars) for a language code.

    Accepts the short code ('L1', 'L8') or the audit-style long form
    ('L1_bpe', 'L8_haar') — normalized to the prefix before the first '_'.

    Unknown languages RAISE (801-audit hardening). Defaulting to lag 0 is NOT
    safe: for any lagged language a silent 0 serves provisional tokens —
    exactly the train/serve skew the TASK_800 convention exists to prevent.
    Lag is resolved once at timeline-prep time, never per-bar, so raising here
    is not a hot-path concern.
    """
    key = str(language).split("_")[0].upper()
    if key not in LANGUAGE_INPUT_LAG:
        raise ValueError(
            f"Unknown token language {language!r} (normalized {key!r}); "
            f"known: {sorted(LANGUAGE_INPUT_LAG)}. Refusing to default to "
            f"lag 0 — that silently serves provisional tokens.")
    return int(LANGUAGE_INPUT_LAG[key])


def lag_shift_per_bar(token_id_per_bar: np.ndarray, lag: int) -> np.ndarray:
    """Apply the input serving lag to a full per-bar token timeline.

    output[t] = token_id_per_bar[t - lag]; the leading `lag` bars become
    INVALID_TOKEN_ID (no final-only annotation is available yet).

    Args:
        token_id_per_bar: (T,) int — per-bar last-completed token ids
            (already -1 at gaps / pre-first-token per annotate_asset).
        lag: non-negative integer bar delay.
    Returns:
        (T,) int32 — lag-shifted token ids, leading `lag` bars = -1.
    """
    ids = np.asarray(token_id_per_bar)
    T = ids.shape[0]
    lag = int(lag)
    if lag <= 0:
        return ids.astype(np.int32, copy=True)
    out = np.empty(T, dtype=np.int32)
    out[:lag] = INVALID_TOKEN_ID
    if lag < T:
        out[lag:] = ids[: T - lag]
    return out


def slice_token_window(token_id_per_bar: np.ndarray,
                       window_start: int,
                       win_len: int,
                       language: str) -> np.ndarray:
    """Slice the serving-lag-shifted token timeline to one encoder window.

    Mirrors the raw-OHLCV slice `cache[window_start : window_start + win_len]`
    (win_len = ws + nstep = 84 in Phase 2). The lag is applied to the FULL
    timeline first, then sliced, so the leading-bar invalidation is global, not
    per-window — a window that starts inside the stream still sees the correct
    (already-firmed) ids, and only a window at the true stream start sees the
    leading -1 bars.

    Args:
        token_id_per_bar: (T,) int per-bar token ids for ONE asset.
        window_start: first bar index of the window (same index used for OHLCV).
        win_len: window length (ws + nstep).
        language: 'L1' (lag 3, p99-firm) or 'L8' (lag 0).
    Returns:
        (win_len,) int32 token ids for this asset's window, lag-applied,
        -1 at any bar that has no final-only annotation.
    """
    shifted = lag_shift_per_bar(token_id_per_bar, input_lag_for(language))
    lo = int(window_start)
    hi = lo + int(win_len)
    out = np.full(int(win_len), INVALID_TOKEN_ID, dtype=np.int32)
    # Guard window bounds; bars beyond the timeline stay -1.
    src_lo = max(lo, 0)
    src_hi = min(hi, shifted.shape[0])
    if src_hi > src_lo:
        out[src_lo - lo: src_hi - lo] = shifted[src_lo:src_hi]
    return out
