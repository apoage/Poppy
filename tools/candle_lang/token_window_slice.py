"""TASK_801 — token-timeline window slicing with the TASK_800 serving-lag convention.

This is the ONE place the serving lag lives. The model never knows about lag; it
just consumes `token_ids` of shape (B, T, A). The lag is baked into how a
per-bar token timeline is sliced into the (window_start .. window_start+win_len)
window that aligns with the raw-OHLCV window the encoder sees.

Serving-lag convention (LOAD-BEARING, from the TASK_800 verdict):
  * L1 inputs are DELAYED 2 bars: token_input[t] = token_id[t-2]. At decision
    time t the annotation of bar t-1 is still provisional (final-only horizon is
    bars <= t-2). The first 2 bars of the sliced window map to the reserved
    invalid id (-1). "Use tokens up to t-1" is the WRONG off-by-one.
  * L8 inputs are UNSHIFTED (LAG-0 exact; churn == 0 over 1000 cuts).
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
LANGUAGE_INPUT_LAG = {
    "L1": 2,   # final-only horizon: token_input[t] = token_id[t-2]
    "L8": 0,   # LAG-0 exact
}


def input_lag_for(language: str) -> int:
    """Return the input serving lag (in bars) for a language code.

    Unknown languages default to 0 (no shift) — callers should pass an explicit
    known code; the default exists only so an unconfigured language degrades to
    the safe unshifted behavior rather than raising in the hot path.
    """
    return int(LANGUAGE_INPUT_LAG.get(str(language), 0))


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
        language: 'L1' (lag 2) or 'L8' (lag 0).
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
