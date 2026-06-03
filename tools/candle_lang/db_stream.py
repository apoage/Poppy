"""
Streaming DB -> bounded RAM ring -> VRAM. THE HARD-REQ#2 core (design 04 build plan; leak/OOM
audit 05). Reads minute OHLC directly from `ohlcv` (read-only), pages the cursor by timestamp
(NEVER fetchall a multi-million-row asset), splits each asset into maximal dt==60 contiguous runs
(NO forward-fill — gaps are missing rows, not flat bars — history.py:375-389 forward-fill is
FORBIDDEN), slices fixed-W windows WITHIN one run, and emits FIXED-SHAPE batches of exactly
cfg.batch with a `valid` mask (the final tail batch is PADDED, never emitted short, so the jit
step never retraces — audit finding 3c).

The bounded queue.Queue(maxsize=cfg.prefetch) is THE RAM cap + backpressure (producer/consumer
modeled on train_phase2.py:513-535). Worker is numpy-only (no jnp) so JAX never touches the
background thread. Worker exceptions are re-raised through the queue (train_phase2.py:877-895).

DB schema (verified): ohlcv(asset_id, timestamp, open, high, low, close, volume, trade_count),
PK(asset_id, timestamp). Minute data is GAPPY: dt!=60 between consecutive rows is normal.

Public surface (contract):
    survey_row_counts(db, limit=None) -> list[(asset_id:int, n_rows:int)] sorted desc
    iter_runs(db, asset_id, cfg)       -> yields (ts,o,h,l,c) np arrays, ONE per maximal dt==60 run
    iter_windows(db, asset_id, cfg, max_windows=None, stride=1)
                                       -> yields per-asset dict(X(n,W,6), C(n,W), end_ts(n,)) chunks
    class BatchStreamer(cfg, assets, shuffle=False, max_windows_per_asset=None)
                                       -> iterable of dict{X,C,asset_id,end_ts,valid}, B==cfg.batch
"""
import gc
import queue
import sqlite3
import threading

import numpy as np

from . import config as C
from .volunit import features_for_run

# Worker housekeeping cadence — gc.collect() every N emitted batches (bounded RAM hygiene).
_GC_EVERY = 64

# --------------------------------------------------------------------------- #
# Per-process windowed-corpus cache (load-once).
#
# WHY: each symbolic member re-streams the build corpus ~9x (alphabet fit, BPE/segmentation fit,
# A2 template, A1/A2/A8/A9 evals) and L4 re-streams it once per epoch. With max_windows set the
# corpus is tiny (~0.5 GB for the default 20k-window panel), so reading it from the 25 GB DB on
# every pass is the CLAUDE.md anti-pattern "repeated disk reads in the loop". We cache the single
# most-recent-`max_windows` chunk per (db, asset, window, max_windows, stride) so the ~9 passes
# collapse to ONE disk read + RAM replays (and L4 epochs 2+ replay from RAM).
#
# GATED ON max_windows being set: max_windows=None => the UNBOUNDED full-resolution path, NEVER
# cached (true streaming / HARD-REQ#2 preserved — caching the full 64-asset minute history would
# defeat the bounded-RAM contract). Cached arrays are READ-ONLY so an accidental in-place mutation
# of a shared window raises loudly instead of silently corrupting later passes.
_WIN_CACHE = {}
_WIN_CACHE_LOCK = threading.Lock()


def clear_window_cache():
    """Drop the per-process windowed-corpus cache (frees the ~0.5 GB resident corpus)."""
    with _WIN_CACHE_LOCK:
        _WIN_CACHE.clear()


# --------------------------------------------------------------------------- #
# read-only connect (donor data.py:43-44)
# --------------------------------------------------------------------------- #
def _connect(db: str) -> sqlite3.Connection:
    """Read-only URI connect. NEVER opens the DB for write (no accidental WAL/journal churn)."""
    return sqlite3.connect(f"file:{db}?mode=ro", uri=True)


def quick_check(db: str) -> bool:
    """One-time cheap PRAGMA quick_check. Returns True on 'ok', else False. NON-FATAL by contract
    (BatchStreamer.__init__ prints a warning and proceeds)."""
    try:
        con = _connect(db)
        try:
            row = con.execute("PRAGMA quick_check").fetchone()
        finally:
            con.close()
        return bool(row) and str(row[0]).lower() == "ok"
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# survey
# --------------------------------------------------------------------------- #
def survey_row_counts(db: str, limit=None):
    """Row count per asset, sorted desc. Single grouped aggregate (the count is index-cheap on the
    PK); does NOT pull any OHLC rows. Returns list[(asset_id:int, n_rows:int)]."""
    con = _connect(db)
    try:
        rows = con.execute(
            "SELECT asset_id, COUNT(*) AS n FROM ohlcv GROUP BY asset_id ORDER BY n DESC"
        ).fetchall()
    finally:
        con.close()
    out = [(int(a), int(n)) for (a, n) in rows]
    if limit is not None:
        out = out[: int(limit)]
    return out


def _asset_ts_span(con: sqlite3.Connection, asset_id: int):
    """(min_ts, max_ts) for one asset, or (None, None) if absent. Index-cheap (PK on asset_id,ts)."""
    row = con.execute(
        "SELECT MIN(timestamp), MAX(timestamp) FROM ohlcv WHERE asset_id=?",
        (int(asset_id),),
    ).fetchone()
    if row is None or row[0] is None:
        return None, None
    return int(row[0]), int(row[1])


def _bar_valid(o, h, l, c):
    """Bar-valid = all OHLC present, finite, and strictly positive (vectorized, one run's rows)."""
    o = np.asarray(o, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    finite = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
    positive = (o > 0) & (h > 0) & (l > 0) & (c > 0)
    return finite & positive


# --------------------------------------------------------------------------- #
# paged row reader (NEVER fetchall a whole asset)
# --------------------------------------------------------------------------- #
def _tail_start_ts(con, asset_id, cfg, max_windows, stride):
    """Tail-seek: the timestamp to BEGIN a forward scan from so it yields >= max_windows windows
    without reading the asset's whole history. Uses the PK index to jump to a row near the tail via
    ORDER BY timestamp DESC LIMIT 1 OFFSET <budget>, over-fetching generously (factor 3 + slack) to
    absorb gaps and run-boundary losses — recent minute data is dense, so this is ample. Returns
    None (=> scan from the asset start) when the asset has fewer rows than the over-fetch budget."""
    W = int(cfg.window)
    need = int(max_windows) * max(int(stride), 1) + W + 1
    budget = need * 3 + 10000
    row = con.execute(
        "SELECT timestamp FROM ohlcv WHERE asset_id=? ORDER BY timestamp DESC LIMIT 1 OFFSET ?",
        (int(asset_id), int(budget)),
    ).fetchone()
    return None if row is None else int(row[0])


def _iter_pages(con, asset_id, cfg, start_ts=None):
    """Yield (ts,o,h,l,c) numpy arrays page by page, ordered by timestamp, paging by a moving
    timestamp lower bound so we never materialize a multi-million-row asset. Each SQL fetch is
    bounded to cfg.page_rows rows (LIMIT), advancing `lo` past the last seen timestamp.

    The PK (asset_id, timestamp) makes the BETWEEN+ORDER+LIMIT seek index-only on the cursor side.
    `start_ts` (tail-seek) begins the scan near the tail instead of at the asset's first bar.
    """
    lo, hi = _asset_ts_span(con, asset_id)
    if lo is None:
        return
    if start_ts is not None:
        lo = max(lo, int(start_ts))
    page_rows = int(cfg.page_rows)
    cur_lo = lo
    while cur_lo <= hi:
        rows = con.execute(
            """
            SELECT timestamp, open, high, low, close
            FROM ohlcv
            WHERE asset_id=? AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp
            LIMIT ?
            """,
            (int(asset_id), int(cur_lo), int(hi), page_rows),
        ).fetchall()
        if not rows:
            break
        n = len(rows)
        ts = np.empty(n, dtype=np.int64)
        o = np.empty(n, dtype=np.float64)
        h = np.empty(n, dtype=np.float64)
        l = np.empty(n, dtype=np.float64)
        c = np.empty(n, dtype=np.float64)
        for i, r in enumerate(rows):
            ts[i] = r[0]
            o[i] = r[1] if r[1] is not None else np.nan
            h[i] = r[2] if r[2] is not None else np.nan
            l[i] = r[3] if r[3] is not None else np.nan
            c[i] = r[4] if r[4] is not None else np.nan
        yield ts, o, h, l, c
        last_ts = int(ts[-1])
        if n < page_rows:
            break
        # advance strictly past the last timestamp seen (PK unique on (asset,ts) -> +1 second is safe;
        # minute grid step is 60s so no real bar is skipped).
        cur_lo = last_ts + 1


# --------------------------------------------------------------------------- #
# runs (maximal dt==60 contiguous segments)
# --------------------------------------------------------------------------- #
def iter_runs(db: str, asset_id: int, cfg: "C.CandleConfig", start_ts=None):
    """Yield (ts,o,h,l,c) numpy arrays, ONE per maximal dt==60 contiguous run of bar-VALID rows.

    Pages the cursor (never fetchall). A run breaks whenever consecutive timestamps differ by != 60,
    OR a row is not bar-valid (the invalid bar terminates the current run and is itself dropped). NO
    forward-fill. Page boundaries are stitched: a carry buffer holds the open run tail across pages
    so a run that spans a page boundary is emitted whole. `start_ts` tail-seeks the underlying scan.
    """
    con = _connect(db)
    try:
        # carry = the still-open run accumulated so far (lists), flushed when a break is hit.
        c_ts, c_o, c_h, c_l, c_c = [], [], [], [], []

        def _flush():
            if len(c_ts) >= 2:
                return (
                    np.asarray(c_ts, dtype=np.int64),
                    np.asarray(c_o, dtype=np.float64),
                    np.asarray(c_h, dtype=np.float64),
                    np.asarray(c_l, dtype=np.float64),
                    np.asarray(c_c, dtype=np.float64),
                )
            return None

        for ts, o, h, l, c in _iter_pages(con, asset_id, cfg, start_ts=start_ts):
            valid = _bar_valid(o, h, l, c)
            n = len(ts)
            for i in range(n):
                if not valid[i]:
                    # invalid bar terminates the open run (and is dropped, no fill)
                    run = _flush()
                    if run is not None:
                        yield run
                    c_ts.clear(); c_o.clear(); c_h.clear(); c_l.clear(); c_c.clear()
                    continue
                if c_ts and (int(ts[i]) - c_ts[-1]) != C.BAR_SECONDS:
                    # gap (or non-monotone / duplicate) -> break the run before this bar
                    run = _flush()
                    if run is not None:
                        yield run
                    c_ts.clear(); c_o.clear(); c_h.clear(); c_l.clear(); c_c.clear()
                c_ts.append(int(ts[i]))
                c_o.append(float(o[i])); c_h.append(float(h[i]))
                c_l.append(float(l[i])); c_c.append(float(c[i]))
        run = _flush()
        if run is not None:
            yield run
    finally:
        con.close()


# --------------------------------------------------------------------------- #
# windows (sliced WITHIN one run only)
# --------------------------------------------------------------------------- #
def _windows_from_run(ts, o, h, l, c, cfg, stride):
    """All length-W windows of all-finite vol-unit features sliced WITHIN one run.

    features_for_run yields L-1 feature bars (feats[k] describes bar ts[k+1]). A window of W feature
    bars therefore needs W+1 raw bars. We require ALL W feature rows in the window to be finite
    (drop any window touching a non-finite feature bar). Returns (X(n,W,6)f32, C(n,W)f32,
    end_ts(n,)i64) or (None,None,None) if no window fits.
    """
    W = int(cfg.window)
    if len(ts) < W + 1:
        return None, None, None
    feats, close, ts_out, finite = features_for_run(ts, o, h, l, c, cfg)
    F = feats.shape[0]  # == L-1
    if F < W:
        return None, None, None
    starts = np.arange(0, F - W + 1, int(stride), dtype=np.int64)
    if starts.size == 0:
        return None, None, None
    # vectorized window index matrix (n, W); n is bounded by run length (<< whole asset).
    offs = np.arange(W, dtype=np.int64)
    idx = starts[:, None] + offs[None, :]            # (n, W)
    X = feats[idx]                                    # (n, W, 6) f32
    Cp = close[idx]                                   # (n, W)    f32 raw close path (A2 denorm)
    end_ts = ts_out[idx[:, -1]]                       # (n,)      i64 end-bar timestamp
    win_finite = finite[idx].all(axis=1)              # (n,)      window all-finite guard
    if not win_finite.all():
        X = X[win_finite]; Cp = Cp[win_finite]; end_ts = end_ts[win_finite]
    if X.shape[0] == 0:
        return None, None, None
    return (
        np.ascontiguousarray(X, dtype=np.float32),
        np.ascontiguousarray(Cp, dtype=np.float32),
        np.ascontiguousarray(end_ts, dtype=np.int64),
    )


def iter_windows(db: str, asset_id: int, cfg: "C.CandleConfig", max_windows=None, stride=1):
    """Yield per-asset dict(X(n,W,6)f32, C(n,W)f32, end_ts(n,)i64) in chunks, ONE chunk per run.

    Windows are sliced WITHIN a single dt==60 run only and are all-finite. `max_windows` caps the
    total windows yielded for the asset, keeping the MOST-RECENT ones.

    Two paths:
      * max_windows is None  -> UNBOUNDED, run-by-run TRUE streaming, NEVER cached (HARD-REQ#2:
        the full-resolution corpus must not be materialized).
      * max_windows is set   -> bounded + CACHEABLE. The single most-recent-`cap` chunk is built
        ONCE (tail-seeked so we read ~cap windows from the tail, not the whole asset) and replayed
        from a per-process RAM cache on every later pass (the ~9 re-reads each member does, and L4
        epochs 2+, become RAM replays). Cached arrays are READ-ONLY.
    """
    stride = int(stride)
    if max_windows is None:
        for ts, o, h, l, c in iter_runs(db, asset_id, cfg):
            X, Cp, end_ts = _windows_from_run(ts, o, h, l, c, cfg, stride)
            if X is not None:
                yield {"X": X, "C": Cp, "end_ts": end_ts}
        return

    cap = int(max_windows)
    if cap <= 0:
        return

    # cache lookup (load-once). A cached value of None means "built, but the asset has no windows".
    key = (str(db), int(asset_id), int(cfg.window), cap, stride)
    with _WIN_CACHE_LOCK:
        cached = key in _WIN_CACHE
        chunk = _WIN_CACHE.get(key)
    if cached:
        if chunk is not None:
            yield chunk
        return

    # MISS -> build once. Tail-seek the scan to ~cap windows from the tail (not the whole asset).
    con = _connect(db)
    try:
        start_ts = _tail_start_ts(con, asset_id, cfg, cap, stride)
    finally:
        con.close()

    ring_X, ring_C, ring_ts = [], [], []
    held = 0

    def _append(X, Cp, end_ts):
        nonlocal held
        ring_X.append(X); ring_C.append(Cp); ring_ts.append(end_ts)
        held += X.shape[0]
        # trim from the FRONT (oldest) whenever we exceed cap.
        while held - ring_X[0].shape[0] >= cap:
            held -= ring_X[0].shape[0]
            ring_X.pop(0); ring_C.pop(0); ring_ts.pop(0)

    for ts, o, h, l, c in iter_runs(db, asset_id, cfg, start_ts=start_ts):
        X, Cp, end_ts = _windows_from_run(ts, o, h, l, c, cfg, stride)
        if X is not None:
            _append(X, Cp, end_ts)

    if not ring_X:
        with _WIN_CACHE_LOCK:
            _WIN_CACHE[key] = None
        return
    X = np.concatenate(ring_X, axis=0)
    Cp = np.concatenate(ring_C, axis=0)
    end_ts = np.concatenate(ring_ts, axis=0)
    if X.shape[0] > cap:                              # keep the most-recent cap rows
        X = X[-cap:]; Cp = Cp[-cap:]; end_ts = end_ts[-cap:]
    # store READ-ONLY so a shared cached window can't be mutated in place (every member already
    # copies before its phase-shuffle control; read-only turns any future aliasing bug into a loud raise).
    X = np.ascontiguousarray(X); Cp = np.ascontiguousarray(Cp); end_ts = np.ascontiguousarray(end_ts)
    X.flags.writeable = False
    Cp.flags.writeable = False
    end_ts.flags.writeable = False
    chunk = {"X": X, "C": Cp, "end_ts": end_ts}
    with _WIN_CACHE_LOCK:
        _WIN_CACHE[key] = chunk
    yield chunk


# --------------------------------------------------------------------------- #
# batch streamer (producer/consumer, bounded queue = RAM cap)
# --------------------------------------------------------------------------- #
class BatchStreamer:
    """Producer/consumer streamer (modeled on train_phase2.py:513-535 SegmentBatchGenerator).

    A single numpy-only worker thread round-robins over `assets`, accumulates window rows, and emits
    FIXED-shape batches of EXACTLY cfg.batch into a queue.Queue(maxsize=cfg.prefetch) — that bounded
    queue IS the RAM cap + backpressure (HARD-REQ#2). The consumer iterates via __iter__/__next__.

    Batch dict (B == cfg.batch ALWAYS):
        X        : (B, W, 6) float32
        C        : (B, W)    float32   raw close path (A2 denorm)
        asset_id : (B,)      int32     bookkeeping only
        end_ts   : (B,)      int64     end-bar timestamp
        valid    : (B,)      bool       False for PAD rows in the final tail batch

    The FINAL tail batch is PADDED to cfg.batch with valid=False (NEVER a short batch -> no jit
    retrace, audit 3c). Worker exceptions re-raise through the queue (train_phase2.py:877-895).
    """

    _SENTINEL = object()  # end-of-stream marker placed on the queue by the worker

    def __init__(self, cfg: "C.CandleConfig", assets, shuffle=False, max_windows_per_asset=None):
        self.cfg = cfg
        self.assets = [int(a) for a in assets]
        self.shuffle = bool(shuffle)
        self.max_windows_per_asset = (
            None if max_windows_per_asset is None else int(max_windows_per_asset)
        )
        self.batch = int(cfg.batch)
        self.W = int(cfg.window)
        self._queue = queue.Queue(maxsize=int(cfg.prefetch))
        self._stop = threading.Event()
        self._lock = threading.RLock()  # CLAUDE.md: RLock not Lock for reentrant contexts
        self._exhausted = False

        # one-time cheap PRAGMA quick_check (non-fatal warning by contract)
        if not quick_check(cfg.db):
            print(f"[BatchStreamer] WARNING: PRAGMA quick_check did not return 'ok' for {cfg.db}")

        self._rng = np.random.default_rng(0)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    # ----- worker (numpy ONLY; no jnp) ----- #
    def _worker(self):
        cfg = self.cfg
        B = self.batch
        W = self.W
        try:
            order = list(self.assets)
            if self.shuffle:
                self._rng.shuffle(order)

            # round-robin generators, one window-stream per asset
            gens = [
                iter_windows(cfg.db, a, cfg, max_windows=self.max_windows_per_asset)
                for a in order
            ]
            alive = [True] * len(gens)

            # accumulation buffers (lists of row-blocks); flushed in EXACT-B slices
            buf_X, buf_C, buf_ts, buf_aid = [], [], [], []
            buf_rows = 0
            emitted = 0

            def _take(n):
                """Pop the first n rows out of the accumulation buffers (returns X,C,ts,aid)."""
                nonlocal buf_rows
                outX, outC, outTs, outAid = [], [], [], []
                need = n
                while need > 0 and buf_X:
                    bx = buf_X[0]; take = min(need, bx.shape[0])
                    outX.append(bx[:take]); outC.append(buf_C[0][:take])
                    outTs.append(buf_ts[0][:take]); outAid.append(buf_aid[0][:take])
                    if take == bx.shape[0]:
                        buf_X.pop(0); buf_C.pop(0); buf_ts.pop(0); buf_aid.pop(0)
                    else:
                        buf_X[0] = bx[take:]; buf_C[0] = buf_C[0][take:]
                        buf_ts[0] = buf_ts[0][take:]; buf_aid[0] = buf_aid[0][take:]
                    need -= take
                    buf_rows -= take
                return (
                    np.concatenate(outX, axis=0),
                    np.concatenate(outC, axis=0),
                    np.concatenate(outTs, axis=0),
                    np.concatenate(outAid, axis=0),
                )

            def _emit_full():
                nonlocal emitted
                X, Cp, ts, aid = _take(B)
                valid = np.ones(B, dtype=bool)
                self._put({
                    "X": X, "C": Cp, "asset_id": aid.astype(np.int32),
                    "end_ts": ts.astype(np.int64), "valid": valid,
                })
                emitted += 1
                if emitted % _GC_EVERY == 0:
                    gc.collect()

            # round-robin drain
            while any(alive) and not self._stop.is_set():
                progressed = False
                for gi, g in enumerate(gens):
                    if not alive[gi] or self._stop.is_set():
                        continue
                    try:
                        chunk = next(g)
                    except StopIteration:
                        alive[gi] = False
                        continue
                    Xc = chunk["X"]
                    if Xc.shape[0] == 0:
                        continue
                    buf_X.append(Xc)
                    buf_C.append(chunk["C"])
                    buf_ts.append(chunk["end_ts"])
                    buf_aid.append(np.full(Xc.shape[0], order[gi], dtype=np.int32))
                    buf_rows += Xc.shape[0]
                    progressed = True
                    while buf_rows >= B and not self._stop.is_set():
                        _emit_full()
                if not progressed:
                    # every generator exhausted on this pass
                    break

            # tail: PAD the final partial batch to B with valid=False (never emit short)
            if buf_rows > 0 and not self._stop.is_set():
                X, Cp, ts, aid = _take(buf_rows)
                m = X.shape[0]
                pad = B - m
                Xp = np.zeros((B, W, C.N_FEAT), dtype=np.float32)
                Cp_p = np.zeros((B, W), dtype=np.float32)
                ts_p = np.zeros(B, dtype=np.int64)
                aid_p = np.zeros(B, dtype=np.int32)
                valid = np.zeros(B, dtype=bool)
                Xp[:m] = X.astype(np.float32, copy=False)
                Cp_p[:m] = Cp.astype(np.float32, copy=False)
                ts_p[:m] = ts.astype(np.int64, copy=False)
                aid_p[:m] = aid.astype(np.int32, copy=False)
                valid[:m] = True
                self._put({
                    "X": Xp, "C": Cp_p, "asset_id": aid_p, "end_ts": ts_p, "valid": valid,
                })

            self._put(self._SENTINEL)
        except Exception as e:  # re-raise through the queue (train_phase2.py:877-895)
            try:
                self._queue.put(e)
            except Exception:
                pass

    def _put(self, item):
        """Blocking put with a stop-aware poll so .close() can unwedge a full queue."""
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=1.0)
                return
            except queue.Full:
                continue

    # ----- consumer ----- #
    def __iter__(self):
        return self

    def __next__(self):
        if self._exhausted:
            raise StopIteration
        item = self._queue.get()
        if isinstance(item, Exception):
            self._exhausted = True
            raise item
        if item is self._SENTINEL:
            self._exhausted = True
            raise StopIteration
        return item

    def close(self):
        """Stop the worker and join. Idempotent."""
        self._stop.set()
        # drain anything queued so a blocked worker .put() can return
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
