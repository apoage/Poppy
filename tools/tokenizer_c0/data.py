"""
C0 data loading for the Temporal Token Middle-Layer viability gate (TASK_789).

Reads OHLC directly from trading_data.db via sqlite3 (read-only).
The DB stores 1-minute candles in a single `ohlcv` table keyed by
(asset_id, timestamp). Minute coverage is gappy, so we aggregate to a
regular hourly grid (open=first, high=max, low=min, close=last) and carry a
validity mask. We NEVER build a feature window that straddles a gap hour --
windows containing an invalid hour are dropped. This keeps the tokenizer from
learning artifacts of forward-filled flat bars.

Feature vector per bar (scale-free, per the locked design -- NOT raw price):
    f0 = log return of close:        log(close_t / close_{t-1})
    f1 = body:    (close - open) / prev_close
    f2 = upper wick: (high - max(open,close)) / prev_close
    f3 = lower wick: (min(open,close) - low) / prev_close
    f4 = range:   (high - low) / prev_close

All five are dimensionless / scale-free. f0 captures direction+magnitude of
the close-to-close move; f1..f4 capture intrabar geometry (candlestick shape).

We also retain the raw close path so decode fidelity can be measured on
de-normalized PRICE paths, not just on returns.
"""
import sqlite3
import numpy as np

DB_PATH = "/home/lukas/Trader/trading_data.db"

# Liquid, well-spanned assets chosen from the row-count survey.
# (asset_id, human symbol). 3 crypto majors + 2 FX.
ASSETS = [
    (1091, "XBT/USD"),
    (359,  "ETH/USD"),
    (1121, "XRP/USD"),
    (384,  "EUR/USD"),
    (431,  "GBP/USD"),
]

BAR_SECONDS = 3600  # aggregate 1m -> 1h


def _connect():
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


def load_hourly_ohlc(asset_id):
    """Return (ts, o, h, l, c) numpy arrays on a regular hourly grid plus a
    boolean `valid` mask (True where the hour had real minute data)."""
    con = _connect()
    cur = con.cursor()
    cur.execute(
        """
        SELECT timestamp/? AS bucket,
               MIN(timestamp) AS first_ts,
               MAX(timestamp) AS last_ts,
               MAX(high) AS hi,
               MIN(low)  AS lo,
               COUNT(*)  AS n
        FROM ohlcv WHERE asset_id=?
        GROUP BY bucket ORDER BY bucket
        """,
        (BAR_SECONDS, asset_id),
    )
    rows = cur.fetchall()
    # We need open of first minute and close of last minute per bucket; do it
    # with correlated lookups in one pass using the recorded first/last ts.
    buckets = [r[0] for r in rows]
    first_ts = {r[0]: r[1] for r in rows}
    last_ts = {r[0]: r[2] for r in rows}
    hi = {r[0]: r[3] for r in rows}
    lo = {r[0]: r[4] for r in rows}

    # Pull open@first_ts and close@last_ts. A huge `timestamp IN (?,?,...)` blows
    # past SQLite's variable limit on multi-year assets, so fetch all
    # (ts, open, close) rows for the asset once and index them in Python.
    fts_set = set(first_ts.values())
    lts_set = set(last_ts.values())
    open_at = {}
    close_at = {}
    cur.execute(
        "SELECT timestamp, open, close FROM ohlcv WHERE asset_id=?",
        (asset_id,),
    )
    for t, o, c in cur.fetchall():
        if t in fts_set:
            open_at[t] = o
        if t in lts_set:
            close_at[t] = c
    con.close()

    if not buckets:
        raise ValueError(f"no data for asset {asset_id}")

    b0, b1 = buckets[0], buckets[-1]
    n = b1 - b0 + 1
    ts = np.arange(b0, b1 + 1) * BAR_SECONDS
    o = np.full(n, np.nan)
    h = np.full(n, np.nan)
    lw = np.full(n, np.nan)
    c = np.full(n, np.nan)
    for b in buckets:
        i = b - b0
        o[i] = open_at.get(first_ts[b], np.nan)
        c[i] = close_at.get(last_ts[b], np.nan)
        h[i] = hi[b]
        lw[i] = lo[b]
    valid = ~np.isnan(c) & ~np.isnan(o) & ~np.isnan(h) & ~np.isnan(lw)
    return ts, o, h, lw, c, valid


def make_features(o, h, l, c, valid):
    """Build the 5-channel scale-free feature array, aligned to bar t (t>=1).
    Returns features[T-1,5], close[T-1] (the close at bar t), and a
    feat_valid[T-1] mask that is True only when bars t-1 and t are both valid.
    """
    prev_c = c[:-1]
    cur_o, cur_h, cur_l, cur_c = o[1:], h[1:], l[1:], c[1:]
    denom = np.where(np.abs(prev_c) > 1e-12, prev_c, np.nan)

    f0 = np.log(np.clip(cur_c / np.where(prev_c > 0, prev_c, np.nan), 1e-12, None))
    f1 = (cur_c - cur_o) / denom
    f2 = (cur_h - np.maximum(cur_o, cur_c)) / denom
    f3 = (np.minimum(cur_o, cur_c) - cur_l) / denom
    f4 = (cur_h - cur_l) / denom
    feats = np.stack([f0, f1, f2, f3, f4], axis=1).astype(np.float32)

    feat_valid = valid[:-1] & valid[1:] & np.all(np.isfinite(feats), axis=1)
    return feats, cur_c.astype(np.float32), feat_valid


def make_windows(feats, close, feat_valid, window, horizon):
    """Slice contiguous windows of length `window` whose bars are ALL valid AND
    whose following `horizon` bars are valid (so the next-return label exists).
    Returns:
        X        [N, window, 5]   feature windows
        closes   [N, window]      close prices over the window (for denorm decode)
        fut_ret  [N]              log return from last window close to close at +horizon
        end_idx  [N]              global index of the last bar of each window (for purge split)
    """
    T = feats.shape[0]
    Xs, Cs, R, E = [], [], [], []
    need = window + horizon
    for end in range(window - 1, T - horizon):
        s = end - window + 1
        if not feat_valid[s:end + 1].all():
            continue
        if not feat_valid[end + 1:end + 1 + horizon].all():
            continue
        last_close = close[end]
        fut_close = close[end + horizon]
        if last_close <= 0 or fut_close <= 0:
            continue
        Xs.append(feats[s:end + 1])
        Cs.append(close[s:end + 1])
        R.append(np.log(fut_close / last_close))
        E.append(end)
    if not Xs:
        return (np.zeros((0, window, 5), np.float32), np.zeros((0, window), np.float32),
                np.zeros((0,), np.float32), np.zeros((0,), np.int64))
    return (np.stack(Xs).astype(np.float32), np.stack(Cs).astype(np.float32),
            np.asarray(R, np.float32), np.asarray(E, np.int64))


def phase_shuffle_returns(feats, feat_valid, rng):
    """Returns-shuffled CONTROL: destroy temporal structure of the return
    channel while preserving its marginal distribution, via phase
    randomization (surrogate data). We FFT the f0 (log-return) channel over
    each contiguous valid run, randomize phases, invert. Geometry channels
    (f1..f4) are independently permuted across time within the run. This keeps
    per-bar marginals but kills any predictable temporal token structure --
    the proper null for "do tokens carry temporal edge".
    """
    feats = feats.copy()
    T = feats.shape[0]
    # find contiguous valid runs
    i = 0
    while i < T:
        if not feat_valid[i]:
            i += 1
            continue
        j = i
        while j < T and feat_valid[j]:
            j += 1
        seg = slice(i, j)
        L = j - i
        if L >= 8:
            # phase randomize f0
            x = feats[seg, 0]
            Xf = np.fft.rfft(x)
            mag = np.abs(Xf)
            ph = np.angle(Xf)
            rand_ph = rng.uniform(-np.pi, np.pi, size=ph.shape)
            rand_ph[0] = ph[0]  # keep DC phase
            if L % 2 == 0:
                rand_ph[-1] = ph[-1]  # keep Nyquist phase real
            Xnew = mag * np.exp(1j * rand_ph)
            feats[seg, 0] = np.fft.irfft(Xnew, n=L).astype(np.float32)
            # permute geometry channels in time
            perm = rng.permutation(L)
            feats[seg, 1:] = feats[seg, 1:][perm]
        i = j
    return feats
