"""TASK_803 Phase 1 — eval-only, default-OFF token trade gate (branch C).

A host-computable, CAUSAL "informativeness" gate overlaid on EXISTING trader checkpoints with NO
training and NO model change. During quiet bars the gate forces HOLD, attacking the trade-rate
explosion failure mode directly ([[project_classifier_alpha_anti_correlation]], TASK_761 Direction 1).

Gate rule v1 — per (window index i, asset a) force HOLD if EITHER:
  * the token annotating the decision bar is a MEGA / FLAT token (a long quiet run), OR
  * the trailing-k mean codelength at the decision bar is below threshold theta.

This module does ONE thing: turn a benchmark period's (db, symbols, per-row timestamps) into a
precomputed boolean ``hold_grid`` of shape (n_windows, n_assets). ``hold_grid[i, a] == True`` means
"force actions[i, a] = HOLD (1)". The benchmark loop reads the grid; ZERO tokenization happens inside
the window loop (CLAUDE.md anti-pattern — pre-encode once to RAM).

Serving lag (LOAD-BEARING, TASK_800 verdict): the per-bar token timeline is read as a DECISION-TIME
input, so the lag MUST be applied via token_window_slice.input_lag_for(language) — the ONLY allowed
lag source. L1 => 2-bar final-only delay; L8 => unshifted. We never hand-roll a shift.

Alignment (matches run_period_eval's own indexing):
  * The benchmark holds ohlcv/prices on a (T_prices, n_assets) grid whose row r is the price at
    timestamp prices_ts[r] (HistoricalMarket.valid_timestamps).
  * obs[i] covers prices[i : i+window_size]; its action applies to returns[i+window_size-1]. The
    DECISION BAR is the last obs bar, price-row index  i + window_size - 1.
  * annotate_asset produces a per-FEATURE-BAR timeline (ts, token_id, codelength). A feature bar's
    ts equals the raw bar's timestamp it describes, so we join the timeline onto the benchmark grid
    by TIMESTAMP (robust to the benchmark's forward-fill / row layout).

PURE NUMPY / CPU ONLY. No JAX, no GPU, no model. Reuses token_timeline + token_window_slice; no
parallel implementations of either the annotation or the lag.
"""
from __future__ import annotations

import dataclasses
from collections import deque
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import token_timeline as TT
from . import token_window_slice as TWS

# Default mega/flat heuristic parameters (explicit + parameterized per the spec). A token is treated
# as a "mega / flat" quiet token when its per-bar annotation PERSISTS for >= MEGA_BARS consecutive
# bars (a long quiet run where the same token id keeps annotating the stream), OR when its codelength
# is at/below CODELEN_FLOOR bits (a near-degenerate, low-information token under the BUILD unigram).
DEFAULT_MEGA_BARS = 16
DEFAULT_CODELEN_FLOOR = 0.0   # bits; 0.0 => disabled by default (codelength floor is opt-in)


# --------------------------------------------------------------------------- #
# per-bar quietness signals (language-agnostic, derived from the timeline)
# --------------------------------------------------------------------------- #
def token_run_length_per_bar(token_id: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Length of the maximal run of the identical (valid) token id ENDING at each bar.

    A quiet market re-emits the same last-completed token across many consecutive bars, so a long
    run of one token id is exactly the "mega / flat" quiet signal — language-agnostic (works for L1
    BPE and L8 Haar). ``run_len[t]`` counts how many consecutive bars up to and including t carry the
    same valid token id as bar t. Invalid bars (id < 0) get run_len 0 and reset the run.

    Args:
        token_id: (T,) int per-bar token ids (-1 at gaps / pre-first-token).
        valid:    (T,) bool per-bar validity.
    Returns:
        (T,) int32 trailing run length (>=1 on valid bars, 0 on invalid bars).
    """
    ids = np.asarray(token_id)
    val = np.asarray(valid, dtype=bool)
    T = ids.shape[0]
    out = np.zeros(T, dtype=np.int32)
    run = 0
    prev = -1
    for t in range(T):
        if (not val[t]) or ids[t] < 0:
            run = 0
            prev = -1
            continue
        if int(ids[t]) == prev:
            run += 1
        else:
            run = 1
            prev = int(ids[t])
        out[t] = run
    return out


def trailing_mean(x: np.ndarray, valid: np.ndarray, k: int) -> np.ndarray:
    """Trailing-k mean of x over the last k BARS ending at each bar, averaging VALID samples only.

    Causal (no lookahead). Bars with no valid sample in their trailing k-bar window get NaN (the
    caller treats NaN as "no gate signal"). k<=1 reduces to the per-bar value on valid bars.

    Args:
        x:     (T,) float per-bar values (e.g. codelength bits).
        valid: (T,) bool per-bar validity (invalid bars excluded from the mean).
        k:     trailing window length in bars (>=1).
    Returns:
        (T,) float64 trailing mean, NaN where no valid sample falls in the window.
    """
    x = np.asarray(x, dtype=np.float64)
    val = np.asarray(valid, dtype=bool)
    T = x.shape[0]
    k = max(1, int(k))
    out = np.full(T, np.nan, dtype=np.float64)
    win: deque = deque()      # holds (value_or_nan) for the last k BARS
    s = 0.0
    n = 0
    for t in range(T):
        if val[t]:
            win.append(float(x[t]))
            s += float(x[t])
            n += 1
        else:
            win.append(np.nan)   # keep the window length aligned to BARS, not samples
        if len(win) > k:
            old = win.popleft()
            if not np.isnan(old):
                s -= old
                n -= 1
        if n > 0:
            out[t] = s / n
    return out


# --------------------------------------------------------------------------- #
# lag-shift helpers (codelength + validity move in lock-step with the token)
# --------------------------------------------------------------------------- #
def _shift_float(x: np.ndarray, lag: int, fill: float = 0.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    lag = int(lag)
    if lag <= 0:
        return x.astype(np.float64, copy=True)
    T = x.shape[0]
    out = np.full(T, fill, dtype=np.float64)
    if lag < T:
        out[lag:] = x[: T - lag]
    return out


def _shift_bool(x: np.ndarray, lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=bool)
    lag = int(lag)
    if lag <= 0:
        return x.copy()
    T = x.shape[0]
    out = np.zeros(T, dtype=bool)
    if lag < T:
        out[lag:] = x[: T - lag]
    return out


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class TokenGate:
    """Precomputed per-(window, asset) HOLD-forcing grid for one benchmark period.

    hold_grid[i, a] == True  =>  force actions[i, a] = HOLD (1) BEFORE position stepping.
    """
    hold_grid: np.ndarray              # (n_windows, n_assets) bool
    language: str
    theta: float
    k: int
    mega_bars: int
    codelen_floor: float
    lag: int
    # diagnostics (per asset): how many decision bars each asset gated, and how many had a timeline.
    n_gated_per_asset: np.ndarray      # (n_assets,) int
    n_covered_per_asset: np.ndarray    # (n_assets,) int  (decision bars with a valid joined token)

    def gated_fraction(self) -> float:
        g = self.hold_grid
        return float(g.mean()) if g.size else 0.0


def _asset_hold_mask(
    timeline: "TT.TimelineResult",
    lag: int,
    prices_ts: np.ndarray,
    window_size: int,
    n_windows: int,
    theta: float,
    k: int,
    mega_bars: int,
    codelen_floor: float,
):
    """Return (hold(n_windows,) bool, n_covered int) for ONE asset. All steps causal.

      1. Apply the serving lag to the per-bar token-id timeline via token_window_slice (the ONLY
         lag source); shift codelength + validity in lock-step.
      2. Derive per-bar mega/flat run-length and trailing-k mean codelength on the LAGGED timeline.
      3. Join onto the benchmark price grid by timestamp; the decision bar for window i is price-row
         (i + window_size - 1). Gate window i if (mega/flat OR trailing-mean codelength < theta) at
         that decision bar's joined token. Price rows with no joined token never gate.
    """
    ts = np.asarray(timeline.ts, dtype=np.int64)
    tok = timeline.token_id
    code = timeline.codelength
    valid = timeline.valid

    # --- 1. serving lag (LOAD-BEARING) via the canonical helper, never hand-rolled. ---
    tok_lag = TWS.lag_shift_per_bar(tok, lag)              # (T,) int32, leading `lag` bars -> -1
    code_lag = _shift_float(code, lag, fill=0.0)
    valid_lag = _shift_bool(valid, lag) & (tok_lag >= 0)

    # --- 2. per-bar quietness signals on the lagged timeline ---
    run_len = token_run_length_per_bar(tok_lag, valid_lag)
    tmean = trailing_mean(code_lag, valid_lag, k)

    mega = (run_len >= int(mega_bars)) & valid_lag
    if codelen_floor > 0.0:
        mega = mega | (valid_lag & (code_lag <= float(codelen_floor)))

    # --- 3. join timeline bars onto benchmark price rows by timestamp ---
    ts_to_bar = {int(t): j for j, t in enumerate(ts.tolist())}
    hold = np.zeros(n_windows, dtype=bool)
    n_covered = 0
    n_dec = min(n_windows, max(0, prices_ts.shape[0] - (window_size - 1)))
    for i in range(n_dec):
        r = i + (window_size - 1)
        b = ts_to_bar.get(int(prices_ts[r]), -1)
        if b < 0 or not bool(valid_lag[b]):
            continue
        n_covered += 1
        theta_hit = np.isfinite(tmean[b]) and (tmean[b] < float(theta))
        if bool(mega[b]) or theta_hit:
            hold[i] = True
    return hold, n_covered


def build_gate_from_timelines(
    timelines: Dict[int, "TT.TimelineResult"],
    symbol_to_timeline_key: Sequence[int],
    prices_ts: np.ndarray,
    n_windows: int,
    window_size: int,
    *,
    language: str,
    theta: float,
    k: int,
    mega_bars: int = DEFAULT_MEGA_BARS,
    codelen_floor: float = DEFAULT_CODELEN_FLOOR,
) -> TokenGate:
    """Assemble a TokenGate from PRE-COMPUTED per-asset timelines (no DB / tokenization here).

    This is the pure, testable core: hand it synthetic TimelineResults + a synthetic price-row
    timestamp axis and it returns the exact HOLD grid. ``symbol_to_timeline_key[a]`` gives the
    timelines-dict key for benchmark asset column a (or a negative sentinel => no gate for that asset).

    The serving lag is resolved ONCE here from `language` via token_window_slice.input_lag_for.
    """
    lag = TWS.input_lag_for(language)
    n_assets = len(symbol_to_timeline_key)
    hold_grid = np.zeros((n_windows, n_assets), dtype=bool)
    n_gated = np.zeros(n_assets, dtype=np.int64)
    n_covered = np.zeros(n_assets, dtype=np.int64)
    prices_ts = np.asarray(prices_ts, dtype=np.int64)

    for a in range(n_assets):
        key = int(symbol_to_timeline_key[a])
        tl = timelines.get(key) if key >= 0 else None
        if tl is None or len(tl) == 0:
            continue
        hold, cov = _asset_hold_mask(
            tl, lag, prices_ts, window_size, n_windows, theta, k, mega_bars, codelen_floor,
        )
        n_covered[a] = cov
        hold_grid[:, a] = hold
        n_gated[a] = int(hold.sum())

    return TokenGate(
        hold_grid=hold_grid, language=language, theta=float(theta), k=int(k),
        mega_bars=int(mega_bars), codelen_floor=float(codelen_floor), lag=int(lag),
        n_gated_per_asset=n_gated, n_covered_per_asset=n_covered,
    )


# --------------------------------------------------------------------------- #
# DB-backed builder (used at eval time; pre-encodes timelines ONCE before the loop)
# --------------------------------------------------------------------------- #
def _l8_codebook_size(tokenizer) -> int:
    """Best-effort codebook size for an L8 HaarVocab (for the codelength unigram table)."""
    for attr in ("codebook_size", "vocab_size", "n_codes", "size"):
        v = getattr(tokenizer, attr, None)
        if v:
            return int(v)
    return 0


def _tokenizer_codebook_size(tokenizer) -> int:
    K = int(getattr(tokenizer, "codebook_size", 0) or 0)
    if K:
        return K
    vocab = getattr(tokenizer, "vocab", None)
    K = int(getattr(vocab, "codebook_size", 0) or 0)
    if K:
        return K
    return _l8_codebook_size(tokenizer)


def precompute_timelines(
    db: str,
    asset_ids: Sequence[int],
    tokenizer,
    cfg=None,
    *,
    build_code_model: bool = True,
    code_alpha: float = 1.0,
) -> Dict[int, "TT.TimelineResult"]:
    """Annotate every asset ONCE into RAM (token_timeline.annotate_asset), returning a dict
    {asset_id -> TimelineResult}. A shared BUILD-unigram codelength model is fit across the period's
    assets (so `codelength` is populated; without it the timeline codelength is zeros and the theta
    arm of the gate is inert — only the mega/flat run-length arm fires).

    Zero tokenization happens after this call — the eval loop only indexes the returned arrays.
    """
    cfg = cfg if cfg is not None else getattr(tokenizer, "cfg", None)
    code_model = None
    if build_code_model:
        K = _tokenizer_codebook_size(tokenizer)
        if K > 0:
            code_model = TT._CodelengthModel(K, alpha=code_alpha)

    # First pass: annotate, fitting the BUILD unigram from the period's own ids.
    raw: Dict[int, "TT.TimelineResult"] = {}
    for aid in asset_ids:
        tl = TT.annotate_asset(db, int(aid), tokenizer, cfg)
        raw[int(aid)] = tl
        if code_model is not None:
            ids = tl.token_id[tl.valid]
            if ids.size:
                code_model.fold_ids(ids.astype(np.int64))
    if code_model is None:
        return raw
    code_model.finalize()
    # Second pass: refill codelength from the fitted model (pure lookup, no re-tokenization).
    out: Dict[int, "TT.TimelineResult"] = {}
    for aid, tl in raw.items():
        cl = code_model.bits_array(tl.token_id).astype(np.float32)
        cl[~tl.valid] = 0.0
        out[int(aid)] = TT.TimelineResult(tl.ts, tl.token_id, cl, tl.valid, tl.asset_id, tl.language)
    return out


def build_gate(
    db: str,
    symbols: List[str],
    symbol_to_id: Dict[str, int],
    prices_ts: np.ndarray,
    n_windows: int,
    window_size: int,
    tokenizer,
    *,
    language: str,
    theta: float,
    k: int,
    mega_bars: int = DEFAULT_MEGA_BARS,
    codelen_floor: float = DEFAULT_CODELEN_FLOOR,
    cfg=None,
) -> TokenGate:
    """Eval-time entry: pre-encode timelines for the period's assets, then assemble the HOLD grid.

    `symbols[a]` is the benchmark column-a symbol; `symbol_to_id` maps it to a DB asset_id (the key
    annotate_asset reads). A symbol with no resolvable asset_id is left ungated (sentinel -1).
    """
    keys: List[int] = [int(symbol_to_id.get(sym, -1)) for sym in symbols]
    unique_ids = sorted({k for k in keys if k >= 0})
    timelines = precompute_timelines(db, unique_ids, tokenizer, cfg)
    return build_gate_from_timelines(
        timelines, keys, prices_ts, n_windows, window_size,
        language=language, theta=theta, k=k, mega_bars=mega_bars, codelen_floor=codelen_floor,
    )
