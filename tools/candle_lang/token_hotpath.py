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

import hashlib
import json
import logging
import os
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
def _attach_vocab_sha(tok) -> None:
    """Stamp a stable `vocab_sha` (sha256 hex) onto a CandleTokenizer, computed ONCE at load.

    CONTENT-CANONICAL: the sha is derived from the LIVE vocab content — the ordered BPE merge table +
    the base-alphabet table (via the same _table_to_jsonable save_vocab uses) + the fit cfg + the
    codebook_size + the schema tag — serialized as canonical (sorted-key, compact) JSON. This is the
    SAME for a vocab whether it was loaded from a file (CandleTokenizer.from_path) or constructed
    in-process from the same content (CandleTokenizer.from_bpe_vocab), so the train path (in-process
    tokenizer) and the eval path (file-loaded tokenizer) share ONE cache entry per asset. Two distinct
    vocabs (different merges / table / cfg) NEVER collide; any content change flips the sha. Hashing
    the raw file BYTES would be brittle (whitespace / field-order / extra fields flip it with no token
    change) AND would split the train vs eval caches — content-canonical avoids both. Idempotent.
    """
    if getattr(tok, "vocab_sha", None):
        return
    cfg = tok.cfg
    canon = {
        "schema": TK.SCHEMA,
        "schema_version": TK.SCHEMA_VERSION,
        "base_alphabet": TK._table_to_jsonable(tok.table),
        "merges": [[int(a), int(b)] for (a, b) in tok.vocab.merges],
        "codebook_size": int(tok.codebook_size),
        "config": {
            "window": int(cfg.window),
            "ewma_halflife": int(cfg.ewma_halflife),
            "sigma_floor": float(cfg.sigma_floor),
        },
    }
    payload = json.dumps(canon, sort_keys=True, separators=(",", ":")).encode("utf-8")
    tok.vocab_sha = hashlib.sha256(payload).hexdigest()


def vocab_sha_for(tokenizer) -> str:
    """Return the stable vocab sha for a tokenizer (computing + caching it on first use)."""
    _attach_vocab_sha(tokenizer)
    return str(tokenizer.vocab_sha)


def load_tokenizer(vocab_file: str, cfg: Optional["C.CandleConfig"] = None):
    """Load a frozen L1 CandleTokenizer from a codebook JSON path.

    vocab_file : path to a frozen vocab JSON (tokenizer_api.save_vocab format), e.g. the pinned
                 runs/799_grammar/L1_RESULT_vocab.json (cb 1376, server-only at real train time).
    cfg        : optional CandleConfig; if None the cfg recorded in the frozen vocab is used (the
                 window/halflife/sigma_floor the vocab was fit under — MUST match).

    Returns the CandleTokenizer. `tok.codebook_size` is the real codebook size; the model's
    token_vocab_size is that + 1 (reserved sentinel for invalid bars), wired in create_model.
    A stable `tok.vocab_sha` (content-canonical sha256 of the frozen vocab — merges + table + cfg) is
    stamped ONCE here so the per-asset timeline cache can key on (asset_id, vocab_sha) and never
    collide across vocabs (and so the file-loaded eval tokenizer shares the cache with the in-process
    train tokenizer of the same content).
    """
    tok = TK.CandleTokenizer.from_path(vocab_file)
    _attach_vocab_sha(tok)
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
    tok = TK.CandleTokenizer.from_bpe_vocab(vocab, cfg)
    _attach_vocab_sha(tok)
    return tok


# --------------------------------------------------------------------------- #
# pool-level per-asset timeline cache  (TASK_801b-ii hot-path perf fix)
# --------------------------------------------------------------------------- #
# annotate_asset(db, asset_id, ...) scans the asset's FULL DB history (~108s/asset — intentional,
# it gives real slots a warmed causal sigma). Its output (a TimelineResult) is FULL-HISTORY,
# vocab-dependent, and SLOT-INDEPENDENT: it does NOT depend on which slot's grid we join onto. So
# across SegmentPool slots that share the asset universe (the common case — all slots see the same
# 64 assets), the per-slot join_cache in tokenize_grid re-annotates each asset ONCE PER SLOT — pure
# waste (64 assets × N slots × full-history annotate ⇒ hours on the first flag-ON pool build).
#
# Fix: a PER-PROCESS RAM cache keyed by (int asset_id, str vocab_sha) -> TimelineResult, so each
# unique asset is annotated ONCE PER POOL (per process). Per-slot work then becomes just
# join_timeline_to_grid(cached_timeline, slot_grid_ts, language) (cheap, no DB scan). vocab_sha keys
# the cache so two different vocabs (or a refit) NEVER collide on the same asset_id.
#
# Lifecycle: module-level dict, alive for the process. Pool lifecycle is per-process, so this matches
# the operator's "per-process RAM cache" spec exactly. clear_timeline_cache() drops it (tests + a
# defensive reset between independent pools that change vocab).
#
# CORRECTNESS: the cache stores ONLY annotate_asset's output. The per-slot join (lag + grid mask) is
# recomputed every slot from the cached timeline, so the served token grid is BYTE-IDENTICAL to the
# uncached per-slot path — the cache is pure perf, tokens unchanged (proven by C-cache / C2 / C3).
# The synthetic / Bella path (annotate_ohlcv on slot-unique cache_data) is NOT cached here — its
# input is slot-unique in-memory OHLCV, not asset_id-shareable.
_TIMELINE_CACHE: dict = {}
_TIMELINE_CACHE_STATS = {"hits": 0, "misses": 0, "annotate_calls": 0, "disk_loads": 0,
                         "disk_writes": 0}


def clear_timeline_cache() -> None:
    """Drop the per-process per-asset timeline cache (and reset stats). Call between independent
    pools that swap the vocab, or in tests that count annotate calls."""
    _TIMELINE_CACHE.clear()
    for k in _TIMELINE_CACHE_STATS:
        _TIMELINE_CACHE_STATS[k] = 0


def timeline_cache_stats() -> dict:
    """Return a shallow copy of the cache counters (hits/misses/annotate_calls/disk_loads/writes)
    plus the current number of cached (asset_id, vocab_sha) entries."""
    s = dict(_TIMELINE_CACHE_STATS)
    s["entries"] = len(_TIMELINE_CACHE)
    return s


def _get_cached_timeline(asset_id: int, vocab_sha: str, db_path: str, tokenizer,
                         cfg: "C.CandleConfig") -> "TT.TimelineResult":
    """Return the FULL-HISTORY TimelineResult for one asset under one vocab, annotating ONCE per
    (asset_id, vocab_sha) per process. Subsequent calls (other slots, this asset) hit RAM."""
    key = (int(asset_id), str(vocab_sha))
    tl = _TIMELINE_CACHE.get(key)
    if tl is not None:
        _TIMELINE_CACHE_STATS["hits"] += 1
        return tl
    _TIMELINE_CACHE_STATS["misses"] += 1
    _TIMELINE_CACHE_STATS["annotate_calls"] += 1
    tl = TT.annotate_asset(db_path, int(asset_id), tokenizer, cfg)
    _TIMELINE_CACHE[key] = tl
    return tl


# --------------------------------------------------------------------------- #
# optional one-time DISK ARTIFACT keyed by vocab_sha (sanctioned load-once-to-RAM; NEVER disk-in-loop)
# --------------------------------------------------------------------------- #
# At pool init, if a cache_dir is configured AND an artifact for THIS vocab_sha exists, load every
# per-asset timeline into the RAM cache ONCE and skip re-annotation across RUNS (15 arm runs × ~2h
# saved ≈ 30h). If absent, the pool annotates normally and writes the artifact after the first pass
# (warm_disk_artifact). This is the CLAUDE.md sanctioned pattern: load-once-to-RAM at startup, NEVER
# a disk read inside any per-slot / per-batch loop. Opt-in via a config field; default off.
def _artifact_path(cache_dir: str, vocab_sha: str) -> str:
    return os.path.join(str(cache_dir), f"timelines_{vocab_sha}.npz")


def load_disk_artifact(cache_dir: str, vocab_sha: str) -> int:
    """Load a per-vocab timeline artifact (if present) into the RAM cache ONCE. Returns the number
    of asset timelines loaded (0 if no artifact / cache_dir off). Pure load-to-RAM at pool init."""
    if not cache_dir:
        return 0
    path = _artifact_path(cache_dir, vocab_sha)
    if not os.path.isfile(path):
        return 0
    npz = np.load(path, allow_pickle=False)
    asset_ids = npz["asset_ids"].astype(np.int64)
    n = 0
    for i, aid in enumerate(asset_ids.tolist()):
        tl = TT.TimelineResult(
            npz[f"ts_{aid}"], npz[f"tok_{aid}"], npz[f"code_{aid}"], npz[f"val_{aid}"],
            asset_id=int(aid), language=str(npz["language"].item()),
        )
        _TIMELINE_CACHE[(int(aid), str(vocab_sha))] = tl
        n += 1
    _TIMELINE_CACHE_STATS["disk_loads"] += n
    logger.info("[TOKEN-CACHE] loaded %d per-asset timelines from %s", n, path)
    return n


def write_disk_artifact(cache_dir: str, vocab_sha: str, asset_ids: Sequence[int]) -> Optional[str]:
    """Write the cached per-asset timelines for `asset_ids` (under vocab_sha) to a single .npz keyed
    by vocab_sha. ~130 MB int32 for the 64-asset universe. Called ONCE after the first annotate pass;
    NEVER inside a loop. Returns the path written (or None if cache_dir off / nothing cached)."""
    if not cache_dir:
        return None
    aids = sorted({int(a) for a in asset_ids if int(a) >= 0})
    payload = {}
    kept = []
    lang = None
    for aid in aids:
        tl = _TIMELINE_CACHE.get((int(aid), str(vocab_sha)))
        if tl is None:
            continue
        payload[f"ts_{aid}"] = np.asarray(tl.ts, dtype=np.int64)
        payload[f"tok_{aid}"] = np.asarray(tl.token_id, dtype=np.int32)
        payload[f"code_{aid}"] = np.asarray(tl.codelength, dtype=np.float32)
        payload[f"val_{aid}"] = np.asarray(tl.valid, dtype=bool)
        kept.append(int(aid))
        lang = tl.language
    if not kept:
        return None
    os.makedirs(str(cache_dir), exist_ok=True)
    path = _artifact_path(cache_dir, vocab_sha)
    payload["asset_ids"] = np.asarray(kept, dtype=np.int64)
    payload["language"] = np.asarray(lang if lang is not None else "L1_bpe")
    np.savez(path, **payload)
    _TIMELINE_CACHE_STATS["disk_writes"] += len(kept)
    logger.info("[TOKEN-CACHE] wrote %d per-asset timelines -> %s", len(kept), path)
    return path


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
        vocab_sha = vocab_sha_for(tokenizer)
        # Per-slot join cache keyed by asset_id so cyclic-padded duplicate COLUMNS within THIS slot
        # reuse one join. The expensive full-history annotate_asset goes through the POOL-LEVEL
        # (asset_id, vocab_sha) RAM cache (_get_cached_timeline): annotate ONCE per unique asset per
        # POOL, not once per slot. The per-slot, slot-grid-dependent join is recomputed here from the
        # cached timeline, so the served grid is byte-identical to the uncached path.
        join_cache: dict = {}
        for a in range(n_assets):
            aid = col_aids[a]
            if aid < 0:
                continue
            if aid not in join_cache:
                tl = _get_cached_timeline(aid, vocab_sha, db_path, tokenizer, cfg)
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
