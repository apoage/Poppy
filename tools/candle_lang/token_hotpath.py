"""
token_hotpath.py — TASK_801b-ii (Fork A) hot-path token wiring helpers.

ONE place that turns a training/eval price source into the per-bar candle-language `token_ids`
the StreamingAssetEncoder consumes, reusing the 801b-i producers (token_timeline.annotate_* +
join_timeline_to_grid) with ZERO parallel tokenization logic. Both train_phase2.py (SegmentPool
slot build) and tools/benchmark_easy.py (eval period) call THIS module, so the same
(asset, window_start) yields a byte-identical token window on both paths (the C3 train/serve
parity skew check the whole serving-lag convention exists for).

Design contract (do not revisit — anchored on the 801b-i frozen conventions):

  * Tokenize ONCE per slot / per period into a RAM `(T_or_G, n_assets) int32` grid (annotate-once,
    cache-to-RAM — C4). The batch loop only SLICES this grid; ZERO re-annotation in any hot loop.
  * The grid is on the slot/period PRICE GRID (HistoricalMarket.valid_timestamps for a real slot,
    or the synthetic dt==60 grid). `join_timeline_to_grid` masks forward-filled rows to -1 (R2) and
    applies the serving lag as a grid-index shift (R4) — the slice is POST-lag, so callers MUST NOT
    re-apply lag.
  * Window slice mirrors the OHLCV window EXACTLY: `grid[c : c+win_len]` for cursor c, then a per-obs
    `ws`-bar gather at the matching start offset (obs_t @ +0, obs_t1 @ +1, obs_tN @ +nstep). Tokens
    align bar-for-bar with the obs window they ride with.
  * Vocab is loaded from a frozen codebook JSON (CandleTokenizer.from_path) — the real
    runs/799_grammar/L1_RESULT_vocab.json at train time (server-only), or a bounded LOCAL fit for
    CPU acceptance. The wiring is vocab-agnostic; the codebook_size derives from the loaded vocab.

CPU ONLY. Pure numpy at build time; the per-obs window gather returns numpy too (the train loop
converts to jnp alongside the prices). No JAX/GPU here.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

from . import config as C
from . import token_timeline as TT
from . import tokenizer_api as TK
from .token_window_slice import INVALID_TOKEN_ID

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# tokenizer loading (frozen codebook JSON, or a bounded local fit for tests)
# --------------------------------------------------------------------------- #
def load_tokenizer(vocab_file: str, cfg: Optional["C.CandleConfig"] = None):
    """Load a frozen L1 CandleTokenizer from a codebook JSON path.

    vocab_file : path to a frozen vocab JSON (tokenizer_api.save_vocab format), e.g. the pinned
                 runs/799_grammar/L1_RESULT_vocab.json (cb 1376, server-only at real train time).
    cfg        : optional CandleConfig; if None the cfg recorded in the frozen vocab is used (the
                 window/halflife/sigma_floor the vocab was fit under — MUST match).

    Returns the CandleTokenizer. `tok.codebook_size` is the real codebook size; the model's
    token_vocab_size is that + 1 (reserved sentinel for invalid bars), wired in create_model.
    """
    tok = TK.CandleTokenizer.from_path(vocab_file)
    return tok


def fit_local_tokenizer(cfg: "C.CandleConfig", build_assets: Sequence[int],
                        fit_mwpa: int = 4000, max_merges: int = 256, p_min: int = 50):
    """Fit a BOUNDED, audit-local L1 vocab (for CPU acceptance only). Mirrors audit_join.build_vocabs.

    Returns a CandleTokenizer. NOT the pinned 799 vocab — flagged by every acceptance check that
    uses it. The hot-path wiring is vocab-agnostic, so a local cb≈1120 vocab exercises every code
    path the real cb-1376 vocab would.
    """
    tmp = "/tmp/801bii_local_l1_vocab.json"
    vocab, _ = TK.fit_and_save(cfg, build_assets, tmp,
                               max_windows_per_asset=fit_mwpa,
                               p_min=p_min, max_merges=max_merges, verbose=False)
    return TK.CandleTokenizer.from_bpe_vocab(vocab, cfg)


# --------------------------------------------------------------------------- #
# slot / period tokenization (annotate ONCE -> RAM (T, n_assets) int32 grid)
# --------------------------------------------------------------------------- #
def _column_asset_ids(hm, n_assets: int) -> list:
    """Resolve column a -> DB asset_id for a HistoricalMarket / BellaMarket `hm`.

    `hm.asset_ids` is the column order (history.py:84 / bella_market.py:348). Cyclic-padding
    (segment_pool.py:332-335) TILES cache_data to n_assets columns WITHOUT extending asset_ids, so
    column a wraps: asset_ids[a % len(asset_ids)] — the SAME cyclic identity the prices carry.
    """
    aids = list(getattr(hm, "asset_ids", []) or [])
    if not aids:
        return [-1] * n_assets
    m = len(aids)
    return [int(aids[a % m]) for a in range(n_assets)]


def tokenize_grid(hm, source: str, tokenizer, cfg: "C.CandleConfig", db_path: str,
                  language: str = "L1_bpe") -> np.ndarray:
    """Tokenize ALL n_assets columns of one slot/period into a `(T, n_assets) int32` grid on the
    slot's price grid (hm.valid_timestamps). -1 at filled/gap/leading-lag rows (post serving-lag).

    Dispatch by `source`:
      * 'historical'        -> annotate_asset(db, asset_id, ...) per column (real DB stream).
      * 'bella' / synthetic -> annotate_ohlcv(hm.cache_data[:, a, :], ts=<grid>, ...) per column.

    Annotate ONCE per column here (C4). The returned grid is cached on the slot dict; the batch loop
    ONLY slices it.

    Returns: (T, n_assets) int32, T == len(hm.valid_timestamps) == hm.cache_data.shape[0].
    """
    cache = np.asarray(hm.cache_data)              # (T, n_assets, 5)
    T = cache.shape[0]
    n_assets = cache.shape[1]
    grid_ts = np.asarray(getattr(hm, "valid_timestamps"), dtype=np.int64)
    if grid_ts.shape[0] != T:
        # Defensive: valid_timestamps must be the per-row time axis. BellaMarket sets
        # valid_timestamps = range(T) (dt==1, NOT 60). For the synthetic path we re-key on a dt==60
        # grid below; for the historical path this must already be the real per-row ts.
        grid_ts = np.arange(T, dtype=np.int64)

    out = np.full((T, n_assets), INVALID_TOKEN_ID, dtype=np.int32)

    if source == "historical":
        col_aids = _column_asset_ids(hm, n_assets)
        # annotate-once cache keyed by asset_id so cyclic-padded duplicate columns reuse one join.
        join_cache: dict = {}
        for a in range(n_assets):
            aid = col_aids[a]
            if aid < 0:
                continue
            if aid not in join_cache:
                tl = TT.annotate_asset(db_path, aid, tokenizer, cfg)
                tok_col, _valid = TT.join_timeline_to_grid(tl, grid_ts, language)
                join_cache[aid] = tok_col
            out[:, a] = join_cache[aid]
        return out

    # synthetic / Bella: in-memory OHLCV per column. ts=None uses the FROZEN synthetic convention
    # (arange(T)*60) inside annotate_ohlcv; we join onto the SAME dt==60 grid so the join is exact.
    syn_grid_ts = np.arange(T, dtype=np.int64) * C.BAR_SECONDS
    for a in range(n_assets):
        ohlcv_a = cache[:, a, :]                    # (T, 5)
        tl = TT.annotate_ohlcv(ohlcv_a, None, tokenizer, cfg)
        tok_col, _valid = TT.join_timeline_to_grid(tl, syn_grid_ts, language)
        out[:, a] = tok_col
    return out


# --------------------------------------------------------------------------- #
# batch slice + per-obs encoder-window gather
# --------------------------------------------------------------------------- #
def slice_token_windows(token_grid: np.ndarray, cursors: np.ndarray, win_len: int) -> np.ndarray:
    """Slice per-sample (win_len, n_assets) token windows at cursors `c` (SAME c / win_len as OHLCV).

    token_grid : (T, n_assets) int32 — the slot's lag-applied token grid (post serving-lag).
    cursors    : (n,) int — per-sample window start (the SAME cursor the OHLCV window uses).
    win_len    : ws + nstep (84 in Phase 2).

    Returns: (n, win_len, n_assets) int32. Out-of-range tail bars are -1 (defensive; the cursor
    sampler already bounds c <= T - ws - nstep, so this should not trigger on the canon path).
    """
    cursors = np.asarray(cursors, dtype=np.int64)
    n = cursors.shape[0]
    T, n_assets = token_grid.shape
    offsets = np.arange(win_len, dtype=np.int64)
    idx = cursors[:, None] + offsets[None, :]                 # (n, win_len)
    valid = idx < T
    idx_clamped = np.minimum(idx, T - 1)
    windows = token_grid[idx_clamped]                          # (n, win_len, n_assets)
    windows = np.where(valid[:, :, None], windows, INVALID_TOKEN_ID)
    return windows.astype(np.int32)


def gather_obs_token_window(token_windows: np.ndarray, start_offset: int, ws: int) -> np.ndarray:
    """Gather the `ws`-bar token window at a within-block start offset, matching one obs encode call.

    The obs encoder gathers prices[base + start_offset : base + start_offset + ws] for each sample
    (jax_encoder.encode_observation_temporal dynamic_slice). The token window MUST gather the SAME
    `ws` bars at the SAME offset so it aligns bar-for-bar with that obs.

    token_windows : (B, win_len, n_assets) int32 — the per-sample win_len token slice.
    start_offset  : within-block obs start (0 for obs_t, 1 for obs_t1, nstep for obs_tN).
    ws            : window_size (64).

    Returns: (B, ws, n_assets) int32 — the encoder's token_ids kwarg shape (B, T=ws, A).
    """
    lo = int(start_offset)
    hi = lo + int(ws)
    return np.ascontiguousarray(token_windows[:, lo:hi, :]).astype(np.int32)
