"""
audit_timeline.py — TASK_800 Deliverable 2: prefix-stability (causality) audit + gating smoke.

Runs ENTIRELY on CPU / pure numpy (no JAX, no GPU). Process asset-by-asset; never holds all assets'
full streams in RAM at once. Reuses token_timeline.annotate_asset + the frozen L1/L8 vocabs fit on
the config BUILD assets.

What it does:
  1. fit_vocabs       : fit a frozen L1 BPE vocab (tokenizer_api.fit_and_save in-process) + a frozen
                        L8 Haar vocab (l8_wavelet.fit_vocab_from_assets) on the BUILD assets, and fit
                        a BUILD unigram codelength model per language.
  2. smoke            : one asset, ~10k bars, both languages — assert timeline length == bar count,
                        all ids < V, valid False exactly at gaps + pre-first-token bars, runtime < 60s.
  3. churn audit      : for >=20 price-stratified assets, >=50 random cut points per asset, measure
                        churn(t,k) = [token at bar t-k from prefix <= t] != [same from full stream];
                        report the churn-lag distribution + p50/p99 lag in bars per language.
  4. A0 through path  : rescale ONE sampled asset's prices x1000 -> assert byte-identical token_id.
  5. write RESULT.md + .json via result_writer with the per-language SERVABLE verdict.

The prefix re-annotation reuses the SAME per-run encode primitives as the full-stream annotation:
for a cut point t we slice each run's feature bars to <= t and re-run the per-bar tokenizer on that
prefix, then compare token_id[t-k] against the full-stream token_id[t-k].
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time

import numpy as np

from . import config as C
from . import base_alphabet as BA
from . import db_stream as DS
from .volunit import features_for_run
from .symbolic import l1_bpe as L1
from .symbolic import l8_wavelet as L8
from . import tokenizer_api as TK
from . import token_timeline as TT
from . import result_writer as RW


# --------------------------------------------------------------------------- #
# vocab + codelength fitting (BUILD assets)
# --------------------------------------------------------------------------- #
def fit_vocabs(cfg, build_assets, max_windows_per_asset=4000, p_min=50, max_merges=256, verbose=True):
    """Fit a frozen L1 BPE vocab + L8 Haar vocab on the BUILD assets (bounded per-asset windows), and
    fit a BUILD unigram codelength model for each language by folding the per-bar token streams.
    Returns dict(l1_vocab, l8_vocab, l1_code, l8_code)."""
    t0 = time.time()
    # ---- L1 BPE vocab (in-process fit; no save/load round-trip needed for the audit) ----
    if verbose:
        print(f"[audit] fitting L1 BPE vocab on {len(build_assets)} build assets "
              f"(mwpa={max_windows_per_asset}, p_min={p_min}, max_merges={max_merges}) ...", flush=True)
    # fit_and_save writes a json; we only need the in-process vocab, so call its pieces via fit_and_save
    # to a temp path (it returns the vocab). Keeps us on the exact frozen-vocab path.
    tmp = "/tmp/800_l1_vocab.json"
    l1_vocab, _ = TK.fit_and_save(cfg, build_assets, tmp,
                                  max_windows_per_asset=max_windows_per_asset,
                                  p_min=p_min, max_merges=max_merges, verbose=verbose)

    # ---- L8 Haar vocab ----
    if verbose:
        print("[audit] fitting L8 Haar vocab on build assets ...", flush=True)
    l8_vocab = L8.fit_vocab_from_assets(
        cfg.db, build_assets, cfg,
        n_levels=L8.DEFAULT_N_LEVELS, win_len=int(cfg.window), stride=1,
        max_windows=max_windows_per_asset, detail_bands_kept=L8.DEFAULT_DETAIL_BANDS_KEPT, seed=0,
    )

    # ---- BUILD unigram codelength models (fold per-bar token streams over a few build assets) ----
    l1_code = TT._CodelengthModel(int(l1_vocab.codebook_size))
    l8_code = TT._CodelengthModel(int(l8_vocab.codebook_size))
    for a in build_assets[: min(4, len(build_assets))]:
        tl1 = TT.annotate_asset(cfg.db, a, TK.CandleTokenizer.from_bpe_vocab(l1_vocab, cfg), cfg,
                                max_feature_bars=max_windows_per_asset)
        ids1 = tl1.token_id[tl1.valid].astype(np.int64)
        if ids1.size:
            l1_code.fold_ids(ids1)
        tl8 = TT.annotate_asset(cfg.db, a, l8_vocab, cfg, max_feature_bars=max_windows_per_asset)
        ids8 = tl8.token_id[tl8.valid].astype(np.int64)
        if ids8.size:
            l8_code.fold_ids(ids8)
    l1_code.finalize()
    l8_code.finalize()
    if verbose:
        print(f"[audit] vocabs fit in {time.time()-t0:.0f}s  "
              f"L1 cb={l1_vocab.codebook_size}  L8 cb={l8_vocab.codebook_size}", flush=True)
    return dict(l1_vocab=l1_vocab, l8_vocab=l8_vocab, l1_code=l1_code, l8_code=l8_code)


# --------------------------------------------------------------------------- #
# prefix re-annotation (the causality probe)
# --------------------------------------------------------------------------- #
def _load_runs(db, asset_id, cfg, max_feature_bars=None):
    """Load ONE asset's feature runs into RAM as a list of (feats, ts_out, finite). Bounded by
    max_feature_bars (kept from the tail). This is the only place a single asset's stream is held."""
    runs = []
    total = 0
    for feats, ts_out, finite in TT._iter_feature_runs(db, asset_id, cfg):
        runs.append((feats, ts_out, finite))
        total += feats.shape[0]
    if max_feature_bars is not None and total > max_feature_bars:
        # keep most-recent runs covering ~max_feature_bars bars.
        kept = []
        acc = 0
        for r in reversed(runs):
            kept.append(r)
            acc += r[0].shape[0]
            if acc >= max_feature_bars:
                break
        runs = list(reversed(kept))
    return runs


def _l1_token_at_bars_from_prefix(feats, finite, vocab, cut_idx):
    """L1: re-annotate the prefix feats[:cut_idx+1] of ONE run, return the per-bar last-completed
    token array over [0..cut_idx]. Uses the SAME _bpe_prefix_token_per_bar primitive."""
    syms = BA.encode_symbols(feats[: cut_idx + 1].astype(np.float64), vocab.table)
    return TT._bpe_prefix_token_per_bar(syms, vocab)  # (cut_idx+1,)


def _l8_token_at_bars_from_prefix(feats, finite, vocab, cut_idx):
    """L8: the token at bar s depends ONLY on the window [s-W+1 .. s]; truncating the run at cut_idx
    cannot change any token at s <= cut_idx. We recompute over the prefix to PROVE it (the audit must
    measure, not assume)."""
    W = int(vocab.win_len)
    mv = feats[: cut_idx + 1, 0].astype(np.float64)
    F = mv.shape[0]
    out = np.full(F, -1, dtype=np.int64)
    for s in range(W - 1, F):
        lo = s - W + 1
        if not finite[lo : s + 1].all():
            continue
        out[s] = int(vocab.encode_window(mv[lo : s + 1]))
    return out


def _merged_id_threshold(vocab):
    """The merged-token id threshold: token ids >= this are BPE MERGE (multi-unit) tokens; ids below
    are leaf (single reduced-unit) tokens. Derived from the tokenizer, NOT hardcoded: l1_bpe.BPEVocab
    lays out [0, n_base) = leaf region (= N_SYMBOLS(table) * N_LEN_BUCKETS) and [n_base, codebook) =
    merge tokens (BPEVocab.children keyed at n_base + merge_index). So the threshold IS vocab.n_base.
    Returns None for non-L1 vocabs (no merge mechanism)."""
    return int(vocab.n_base) if hasattr(vocab, "n_base") else None


def churn_for_asset(db, asset_id, cfg, vocab, language, n_cuts=50, max_k=128,
                    max_feature_bars=None, seed=0, stress_merges=False, merge_thresh=None):
    """Measure churn-lag for ONE asset under ONE language. For n_cuts cut points t (within a run, with
    enough lead-in), compare the per-bar token at bars t-k (k=0..max_k) computed from the PREFIX <= t
    against the FULL-stream value. Returns a dict with the churn-lag samples (the largest k at which a
    bar t-k still disagrees, per cut) and counts.

    stress_merges (L1 only): instead of sampling cut points uniformly, OVERSAMPLE cut points whose
    full-stream last-completed token is a MERGE token (id >= merge_thresh), or that sit just before
    one. Merges span completed units, so a merge firing once its right-hand neighbour completes is the
    ONLY mechanism that can produce lag>0 for L1 — this aims the audit at exactly that mechanism rather
    than diluting it across the (overwhelmingly leaf) uniform population. n_stress_eligible reports how
    many merge-adjacent cut points existed so under-sampling is never silent."""
    runs = _load_runs(db, asset_id, cfg, max_feature_bars=max_feature_bars)
    if not runs:
        return dict(n_cuts=0, churn_lags=[], n_churned_cuts=0, n_bars_compared=0,
                    n_stress_cuts=0, n_stress_eligible=0)
    rng = np.random.default_rng(seed + int(asset_id))

    # full-stream per-run token arrays (computed once per run).
    full_tokens = []
    for feats, ts_out, finite in runs:
        if language == "L1_bpe":
            syms = BA.encode_symbols(feats.astype(np.float64), vocab.table)
            ft = TT._bpe_prefix_token_per_bar(syms, vocab)
        else:  # L8
            W = int(vocab.win_len)
            mv = feats[:, 0].astype(np.float64)
            F = mv.shape[0]
            ft = np.full(F, -1, dtype=np.int64)
            for s in range(W - 1, F):
                lo = s - W + 1
                if finite[lo : s + 1].all():
                    ft[s] = int(vocab.encode_window(mv[lo : s + 1]))
        full_tokens.append(ft)

    # pick cut points: a run index + a bar index t with at least max_k+W lead-in and not the last bar.
    eligible = []
    stress_eligible = []   # cut points adjacent to a merge token (L1 stress)
    W = int(getattr(vocab, "win_len", cfg.window))
    lead = max_k + W + 2
    do_stress = bool(stress_merges and language == "L1_bpe" and merge_thresh is not None)
    for ri, (feats, _ts, _fin) in enumerate(runs):
        F = feats.shape[0]
        if F <= lead + 1:
            continue
        ft = full_tokens[ri]
        for t in range(lead, F - 1):
            eligible.append((ri, t))
            if do_stress:
                # cut at t is "merge-adjacent" if the last-completed token at t (or within a small
                # forward window that a longer prefix could complete) is a merge token. Aim at bars
                # whose own annotation, or the next-completing annotation just after t, is a merge.
                fwd_hi = min(F, t + W + 1)
                seg = ft[t:fwd_hi]
                if seg.size and int(seg.max()) >= int(merge_thresh):
                    stress_eligible.append((ri, t))

    if not eligible:
        return dict(n_cuts=0, churn_lags=[], n_churned_cuts=0, n_bars_compared=0,
                    n_stress_cuts=0, n_stress_eligible=len(stress_eligible))

    # build the cut-point selection. When stressing, draw merge-adjacent cuts FIRST (up to n_cuts),
    # then top up uniformly so n_cuts is always met (the hard floor is enforced by the caller).
    sel = []
    n_stress_used = 0
    if do_stress and stress_eligible:
        s_take = min(int(n_cuts), len(stress_eligible))
        s_idx = rng.choice(len(stress_eligible), size=s_take, replace=False)
        sel.extend(stress_eligible[i] for i in s_idx)
        n_stress_used = s_take
    if len(sel) < int(n_cuts):
        remaining = int(n_cuts) - len(sel)
        pool = eligible if not sel else [e for e in eligible if e not in set(sel)]
        u_take = min(remaining, len(pool))
        u_idx = rng.choice(len(pool), size=u_take, replace=False)
        sel.extend(pool[i] for i in u_idx)
    take = len(sel)

    churn_lags = []
    n_churned = 0
    n_bars_cmp = 0
    for (ri, t) in sel:
        feats, _ts, finite = runs[ri]
        ft = full_tokens[ri]
        if language == "L1_bpe":
            pref = _l1_token_at_bars_from_prefix(feats, finite, vocab, t)
        else:
            pref = _l8_token_at_bars_from_prefix(feats, finite, vocab, t)
        # compare bars t-k for k=0..max_k
        max_disagree_k = -1
        for k in range(0, max_k + 1):
            b = t - k
            if b < 0:
                break
            n_bars_cmp += 1
            if int(pref[b]) != int(ft[b]):
                if k > max_disagree_k:
                    max_disagree_k = k
        if max_disagree_k >= 0:
            n_churned += 1
            churn_lags.append(int(max_disagree_k))
        else:
            churn_lags.append(0)  # no churn at this cut -> lag 0
    return dict(n_cuts=take, churn_lags=churn_lags, n_churned_cuts=n_churned,
                n_bars_compared=n_bars_cmp, n_stress_cuts=int(n_stress_used),
                n_stress_eligible=int(len(stress_eligible)))


# --------------------------------------------------------------------------- #
# A0 through the timeline path (rescale x1000 -> byte-identical token_id)
# --------------------------------------------------------------------------- #
def a0_through_timeline(db, asset_id, cfg, vocabs, lam=1000.0, max_feature_bars=20000):
    """Rescale ONE asset's prices by `lam` and assert byte-identical token_id arrays for BOTH
    languages. We build a TEMP in-memory rescaled DB for the asset (prices x lam) and annotate it the
    SAME way, then compare token_id arrays. Returns dict(l1_identical, l8_identical, ...)."""
    # pull the asset's raw rows (bounded by max_feature_bars*~3 to keep RAM small) into a temp DB.
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        # most-recent rows: tail-seek via OFFSET.
        budget = int(max_feature_bars) * 2 + 10000
        row = con.execute(
            "SELECT timestamp FROM ohlcv WHERE asset_id=? ORDER BY timestamp DESC LIMIT 1 OFFSET ?",
            (int(asset_id), budget)).fetchone()
        start_ts = int(row[0]) if row else None
        q = "SELECT timestamp,open,high,low,close FROM ohlcv WHERE asset_id=?"
        params = [int(asset_id)]
        if start_ts is not None:
            q += " AND timestamp>=?"
            params.append(start_ts)
        q += " ORDER BY timestamp"
        rows = con.execute(q, params).fetchall()
    finally:
        con.close()
    if not rows:
        return dict(l1_identical=None, l8_identical=None, note="no rows")

    # build two temp DBs: identity and x-lam.
    def _make_db(path, scale):
        if os.path.exists(path):
            os.remove(path)
        c2 = sqlite3.connect(path)
        c2.execute("CREATE TABLE ohlcv(asset_id INT, timestamp INT, open REAL, high REAL, low REAL, "
                   "close REAL, volume REAL, trade_count INT, PRIMARY KEY(asset_id,timestamp))")
        c2.executemany(
            "INSERT INTO ohlcv(asset_id,timestamp,open,high,low,close,volume,trade_count) "
            "VALUES(?,?,?,?,?,?,1.0,1)",
            [(int(asset_id), int(ts), o * scale, h * scale, l * scale, cc * scale)
             for (ts, o, h, l, cc) in rows])
        c2.commit(); c2.close()
        return path

    db1 = _make_db("/tmp/800_a0_identity.db", 1.0)
    dbL = _make_db("/tmp/800_a0_scaled.db", float(lam))
    # the window cache keys on db path so the two temp dbs don't collide; clear to be safe.
    DS.clear_window_cache()

    out = {}
    for lang, vocab_or_tok in (("L1_bpe", TK.CandleTokenizer.from_bpe_vocab(vocabs["l1_vocab"], cfg)),
                               ("L8_haar", vocabs["l8_vocab"])):
        t1 = TT.annotate_asset(db1, asset_id, vocab_or_tok, cfg, max_feature_bars=max_feature_bars)
        DS.clear_window_cache()
        tL = TT.annotate_asset(dbL, asset_id, vocab_or_tok, cfg, max_feature_bars=max_feature_bars)
        DS.clear_window_cache()
        same = (t1.token_id.shape == tL.token_id.shape) and bool(np.array_equal(t1.token_id, tL.token_id))
        out[f"{'l1' if lang=='L1_bpe' else 'l8'}_identical"] = bool(same)
        out[f"{'l1' if lang=='L1_bpe' else 'l8'}_n_bars"] = int(t1.token_id.shape[0])
    for p in (db1, dbL):
        try:
            os.remove(p)
        except OSError:
            pass
    return out


# --------------------------------------------------------------------------- #
# gating smoke
# --------------------------------------------------------------------------- #
def run_smoke(db, asset_id, cfg, vocabs, n_bars=10000):
    """One asset, ~n_bars, both languages: timeline length == bar count (== the feature-bar grid),
    all ids < V, valid False exactly at gaps + pre-first-token bars, runtime < 60s."""
    t0 = time.time()
    res = {}
    ok = True
    for lang, tok in (("L1_bpe", TK.CandleTokenizer.from_bpe_vocab(vocabs["l1_vocab"], cfg)),
                      ("L8_haar", vocabs["l8_vocab"])):
        code = vocabs["l1_code"] if lang == "L1_bpe" else vocabs["l8_code"]
        tl = TT.annotate_asset(db, asset_id, tok, cfg, code_model=code, max_feature_bars=n_bars)
        V = int(vocabs["l1_vocab"].codebook_size if lang == "L1_bpe" else vocabs["l8_vocab"].codebook_size)
        T = len(tl)
        ids_valid = tl.token_id[tl.valid]
        ids_ok = bool((ids_valid >= 0).all() and (ids_valid < V).all())
        invalid_neg1 = bool(np.array_equal(tl.token_id[~tl.valid], np.full((~tl.valid).sum(), -1, np.int32))) \
            if (~tl.valid).any() else True
        # valid must be False at gaps: recompute dt and check every dt!=60 bar is invalid.
        dt = np.empty(T, dtype=np.int64); dt[0] = 0; dt[1:] = tl.ts[1:] - tl.ts[:-1]
        gap_bars = dt != C.BAR_SECONDS
        gap_bars[0] = True
        gaps_invalid = bool((~tl.valid[gap_bars]).all())
        member_ok = ids_ok and invalid_neg1 and gaps_invalid and T > 0
        ok = ok and member_ok
        res[lang] = dict(T=int(T), V=V, ids_in_range=ids_ok, invalid_is_neg1=invalid_neg1,
                         gaps_are_invalid=gaps_invalid, n_valid=int(tl.valid.sum()),
                         n_invalid=int((~tl.valid).sum()), member_ok=bool(member_ok))
    wall = time.time() - t0
    res["wall_s"] = float(wall)
    res["runtime_ok"] = bool(wall < 60.0)
    res["smoke_pass"] = bool(ok and wall < 60.0)
    return res


# --------------------------------------------------------------------------- #
# price-stratified asset sample
# --------------------------------------------------------------------------- #
def _survey_prices(db, min_rows=50000):
    """Survey the DB once: return (assets_ascending_by_price, prices_dict) over assets with >= min_rows
    rows. Shared by both the split-file path and the DB-reconstruction path."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT asset_id, COUNT(*) n FROM ohlcv GROUP BY asset_id "
                           "ORDER BY n DESC").fetchall()
        cand = [int(a) for a, n in rows if int(n) >= int(min_rows)]
        px = {}
        for a in cand:
            r = con.execute("SELECT close FROM ohlcv WHERE asset_id=? ORDER BY timestamp DESC LIMIT 1",
                            (a,)).fetchone()
            if r and r[0] and float(r[0]) > 0:
                px[a] = float(r[0])
    finally:
        con.close()
    items = sorted(px.items(), key=lambda kv: kv[1])           # ascending price
    return [a for a, _ in items], {int(a): float(px[a]) for a in px}


def load_split_file(path):
    """Load a canonical asset_split.json with `build`/`holdout` keys (the runs/796_scaled artifact).
    Returns (build_ids:list[int], holdout_ids:list[int]) or None if the file is absent/unreadable."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            d = json.load(f)
        b = [int(x) for x in (d.get("build") or d.get("build_assets") or [])]
        h = [int(x) for x in (d.get("holdout") or d.get("holdout_assets") or [])]
        if not b and not h:
            return None
        return b, h
    except Exception:
        return None


def eligible_run_bars(db, asset_id, cfg, max_feature_bars=None):
    """Longest contiguous valid (dt==60, all-finite) feature-bar run for ONE asset, in feature bars.
    An asset is 'eligible' for the audit only if this is >= (window + audit margin) so cut points with
    enough lead-in exist. Uses the SAME _iter_feature_runs / finite mask as the timeline path."""
    best = 0
    for feats, _ts, finite in TT._iter_feature_runs(db, int(asset_id), cfg):
        # within a run all bars are dt==60; the eligible span is the longest all-finite contiguous
        # block (finite gates token validity exactly as the timeline does).
        fin = np.asarray(finite, dtype=bool)
        if fin.all():
            run_best = fin.size
        else:
            # longest True run
            run_best = 0
            cur = 0
            for v in fin:
                cur = cur + 1 if v else 0
                if cur > run_best:
                    run_best = cur
        if run_best > best:
            best = run_best
        if max_feature_bars is not None and best >= max_feature_bars:
            return int(min(best, max_feature_bars))
    return int(best)


def price_stratified_pool(db, n_total=20, min_rows=50000, split_file=None):
    """Build an ORDERED candidate pool, price-stratified across BOTH split halves, that the caller
    draws from until it has N EFFECTIVE assets (the draw-until loop replaces thin assets rather than
    silently counting a 0). Returns dict(order(list of (asset, half) in draw priority), prices(dict),
    split_source(str)) where half in {"build_like","holdout_like"}.

    Split source resolution (recorded in meta.deviations):
      * split_file present + parseable -> use its build/holdout ids (canonical 796_scaled split);
        build -> build_like half, holdout -> holdout_like half. Assets are price-sorted within half.
      * else -> reconstruct a price-stratified split from the DB survey (high-price half = build_like,
        low-price half = holdout_like), spreading across price quantiles within each half.

    The draw `order` interleaves the two halves and spreads within each half so price-stratification is
    preserved even as thin assets get skipped and deeper candidates pulled in."""
    assets_asc, px = _survey_prices(db, min_rows=min_rows)
    asc_set = set(assets_asc)

    loaded = load_split_file(split_file)
    if loaded is not None:
        b_ids, h_ids = loaded
        # keep only surveyed assets (need >= min_rows + a close price); preserve price order within half
        def _by_price(ids):
            ids = [a for a in ids if a in asc_set]
            return sorted(ids, key=lambda a: px[a])
        build_like = _by_price(b_ids)
        holdout_like = _by_price(h_ids)
        split_source = f"split_file:{split_file}"
        # carry prices for any split asset missing a close (defensive; usually all present)
        for a in (build_like + holdout_like):
            px.setdefault(int(a), float("nan"))
    else:
        n = len(assets_asc)
        half = n // 2
        holdout_like = assets_asc[:half]            # low price half
        build_like = assets_asc[half:]              # high price half
        split_source = "db_survey_reconstructed"

    def _spread_order(pool):
        """Return pool re-ordered to spread across price quantiles (quantile-stratified draw order):
        pick the endpoints and midpoints first so early draws span the price range."""
        pool = list(pool)
        if len(pool) <= 2:
            return pool
        idxs = list(range(len(pool)))
        order, taken = [], set()
        # repeatedly halve the unfilled gaps -> quantile-spread draw order
        frontier = [(0, len(pool) - 1)]
        for end in (0, len(pool) - 1):
            if end not in taken:
                order.append(end); taken.add(end)
        while len(order) < len(pool):
            new_frontier = []
            for lo, hi in frontier:
                mid = (lo + hi) // 2
                if mid not in taken and lo < mid < hi:
                    order.append(mid); taken.add(mid)
                    new_frontier.append((lo, mid)); new_frontier.append((mid, hi))
            if not new_frontier:
                for i in idxs:
                    if i not in taken:
                        order.append(i); taken.add(i)
                break
            frontier = new_frontier
        return [pool[i] for i in order]

    b_order = _spread_order(build_like)
    h_order = _spread_order(holdout_like)
    # interleave halves so the draw alternates build-like / holdout-like (keeps both halves sampled)
    order = []
    bi = hi = 0
    while bi < len(b_order) or hi < len(h_order):
        if bi < len(b_order):
            order.append((int(b_order[bi]), "build_like")); bi += 1
        if hi < len(h_order):
            order.append((int(h_order[hi]), "holdout_like")); hi += 1
    return dict(order=order, prices={int(a): float(px.get(int(a), float("nan"))) for a, _ in order},
                split_source=split_source)


# --------------------------------------------------------------------------- #
# verdict
# --------------------------------------------------------------------------- #
def _verdict_for(language, p50, p99, n_churned, n_cuts, a0_identical):
    """SERVABLE (window-rate exact, churn==0) / SERVABLE-WITH-LAG-k (bounded measured lag) /
    NOT-SERVABLE (unbounded/large churn). A0 must hold (byte-identical) or it's NOT-SERVABLE."""
    if a0_identical is False:
        return "NOT-SERVABLE", "A0 rescale broke byte-identity through the timeline path"
    if language == "L8_haar":
        if n_churned == 0 and p99 == 0:
            return "SERVABLE", "window-rate exact: churn == 0 over all sampled cut points"
        return "NOT-SERVABLE", f"window-rate member churned (p99={p99}) — a per-window leak crept in"
    # L1 / variable-rate: bounded lag is SERVABLE-WITH-LAG-k.
    if n_churned == 0 and p99 == 0:
        return "SERVABLE", "no measured churn at any cut point"
    # treat 'large' as p99 lag exceeding one window (would mean unbounded firm-up).
    if p99 <= int(C.DEFAULT_WINDOW):
        return f"SERVABLE-WITH-LAG-{int(p99)}", f"bounded firm-up lag: p50={p50} p99={p99} bars"
    return "NOT-SERVABLE", f"firm-up lag exceeds one window (p99={p99} bars) — unbounded churn risk"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="TASK_800 token-timeline causality audit (CPU/numpy).")
    ap.add_argument("--db", default=None)
    ap.add_argument("--out", default="/home/lukas/Trader/runs/800_timeline/RESULT.md")
    ap.add_argument("--n-assets", type=int, default=20,
                    help="DEPRECATED alias for --max-effective-assets (kept for back-compat).")
    ap.add_argument("--max-effective-assets", type=int, default=20,
                    help="target number of EFFECTIVE assets (assets that actually produced cuts). The "
                         "draw-until loop replaces thin assets until this many effective are reached.")
    ap.add_argument("--n-cuts", type=int, default=50,
                    help="cut points per asset per language.")
    ap.add_argument("--min-cuts", type=int, default=30,
                    help="HARD FLOOR: every effective asset must yield >= this many cuts or FAIL.")
    ap.add_argument("--max-k", type=int, default=128)
    ap.add_argument("--audit-max-bars", type=int, default=40000,
                    help="cap on feature bars per asset held in RAM during the audit (RAM safety).")
    ap.add_argument("--fit-mwpa", type=int, default=4000)
    ap.add_argument("--l1-stress-cuts", type=int, default=300,
                    help="target number of L1 merge-adjacent STRESS cut points (oversample cuts next to "
                         "merge tokens, aiming the audit at the deep-merge firm-up mechanism).")
    ap.add_argument("--split-file", default="/home/lukas/Trader/runs/796_scaled/asset_split.json",
                    help="canonical asset_split.json (build/holdout keys). Falls back to a DB-survey "
                         "reconstruction if absent; the path taken is recorded in meta.deviations[].")
    ap.add_argument("--max-draws", type=int, default=0,
                    help="cap on assets DRAWN while hunting for effective ones (0 = pool size).")
    ap.add_argument("--smoke-only", action="store_true")
    args = ap.parse_args(argv)
    # --n-assets back-compat: if the user set it but left the new flag at default, honor the old name.
    if args.n_assets != 20 and args.max_effective_assets == 20:
        args.max_effective_assets = int(args.n_assets)

    db = args.db if args.db else C.db_path()
    cfg = C.CandleConfig(db=db)
    build_assets = list(cfg.build_assets)

    print(f"[audit] db={db}")
    print(f"[audit] build_assets={build_assets}")
    vocabs = fit_vocabs(cfg, build_assets, max_windows_per_asset=int(args.fit_mwpa))

    # ---- smoke (must pass before the full audit) ----
    smoke_asset = build_assets[0]
    print(f"[audit] SMOKE on asset {smoke_asset} (~10k bars, both languages) ...", flush=True)
    smoke = run_smoke(db, smoke_asset, cfg, vocabs, n_bars=10000)
    print(f"[audit] smoke: pass={smoke['smoke_pass']} wall={smoke['wall_s']:.1f}s "
          f"L1={smoke['L1_bpe']} L8={smoke['L8_haar']}", flush=True)
    if not smoke["smoke_pass"]:
        raise SystemExit("[audit] SMOKE FAILED — aborting before the full audit.")
    if args.smoke_only:
        print("[audit] --smoke-only: stopping after smoke.")
        return smoke

    # ---- eligibility-aware, price-stratified candidate pool (draw-until N EFFECTIVE) ----
    merge_thresh = _merged_id_threshold(vocabs["l1_vocab"])
    print(f"[audit] L1 merged-id threshold (vocab.n_base) = {merge_thresh} "
          f"(ids >= {merge_thresh} are MERGE tokens; codebook={vocabs['l1_vocab'].codebook_size})")
    pool = price_stratified_pool(db, n_total=int(args.max_effective_assets),
                                 split_file=args.split_file)
    print(f"[audit] split source: {pool['split_source']}  pool size: {len(pool['order'])}")

    # eligibility margin: a cut needs max_k + window + lead-in bars; require a contiguous valid run
    # at least this long so we don't draw assets that can only ever yield 0 cuts.
    W0 = int(cfg.window)
    elig_margin = int(args.max_k) + W0 + 8
    target_eff = int(args.max_effective_assets)
    n_cuts = int(args.n_cuts)
    min_cuts = int(args.min_cuts)
    if n_cuts < min_cuts:
        raise SystemExit(f"[audit] --n-cuts ({n_cuts}) < --min-cuts ({min_cuts}); "
                         f"raise --n-cuts so the per-asset hard floor is satisfiable.")

    deviations = [
        dict(name="asset_split_source", value=pool["split_source"],
             detail=("canonical runs/796_scaled/asset_split.json used"
                     if pool["split_source"].startswith("split_file")
                     else "runs/796_scaled/asset_split.json absent on this checkout; "
                          "price-stratified split reconstructed from the DB survey (both halves, "
                          "quantile-spread). Point --split-file at the server copy to use the canonical split.")),
    ]

    import subprocess
    def _free_g():
        try:
            out = subprocess.check_output(["free", "-g"]).decode().splitlines()[1].split()
            return int(out[6])  # available
        except Exception:
            return 999
    avail = _free_g()
    print(f"[audit] free -g available: {avail} GB")
    if avail < 4 and target_eff > 8:
        target_eff = 8
        deviations.append(dict(name="low_ram_shrink", value=target_eff,
                               detail=f"free -g={avail}G < 4G -> target effective assets capped at 8 (RAM safety)."))
        print(f"[audit] LOW RAM ({avail}G) -> target effective assets -> {target_eff}")

    # accumulators per language; we draw assets one at a time, run BOTH languages, and only count an
    # asset as "effective" if it produced cuts (L1 is the gating language for effectiveness since L8
    # eligibility is a strict superset). A drawn-but-thin asset is REPORTED, never silently counted.
    langs = (("L8_haar", vocabs["l8_vocab"]), ("L1_bpe", vocabs["l1_vocab"]))
    acc = {lang: dict(all_lags=[], n_churned=0, n_cuts_tot=0, per_asset=[],
                      n_stress_cuts=0, n_stress_eligible=0) for lang, _ in langs}
    effective_assets = []     # assets that produced cuts (count toward the headline)
    thin_assets = []          # drawn but ineligible/0-cut (reported, not counted)
    used_prices = {}
    used_halves = {"build_like": 0, "holdout_like": 0}
    max_draws = int(args.max_draws) if int(args.max_draws) > 0 else len(pool["order"])
    n_drawn = 0

    for (a, half) in pool["order"]:
        if len(effective_assets) >= target_eff or n_drawn >= max_draws:
            break
        n_drawn += 1
        # pre-filter on contiguous valid run length (cheap eligibility check before the full churn run)
        elig = eligible_run_bars(db, a, cfg, max_feature_bars=int(args.audit_max_bars))
        if elig < elig_margin:
            DS.clear_window_cache()
            thin_assets.append(dict(asset=int(a), half=half, reason="ineligible",
                                    eligible_bars=int(elig), need=int(elig_margin)))
            print(f"[audit] DRAW asset {a} ({half}) INELIGIBLE: longest valid run {elig} < {elig_margin} -> replaced", flush=True)
            continue

        per_lang_cuts = {}
        per_lang_r = {}
        for lang, vocab in langs:
            stress = (lang == "L1_bpe")
            r = churn_for_asset(db, a, cfg, vocab, lang, n_cuts=n_cuts,
                                max_k=int(args.max_k), max_feature_bars=int(args.audit_max_bars),
                                seed=0, stress_merges=stress, merge_thresh=merge_thresh)
            DS.clear_window_cache()
            per_lang_cuts[lang] = int(r["n_cuts"])
            per_lang_r[lang] = r

        # effectiveness gated on the binding language (L1: the variable-rate / merge mechanism).
        l1_cuts = per_lang_cuts["L1_bpe"]
        if l1_cuts <= 0:
            thin_assets.append(dict(asset=int(a), half=half, reason="zero_cuts",
                                    eligible_bars=int(elig)))
            print(f"[audit] DRAW asset {a} ({half}) produced 0 L1 cuts despite eligibility -> replaced", flush=True)
            continue

        # HARD FLOOR: an effective asset must clear min_cuts in BOTH languages, else FAIL loudly.
        for lang in ("L8_haar", "L1_bpe"):
            if per_lang_cuts[lang] < min_cuts:
                raise SystemExit(
                    f"[audit] HARD FLOOR FAIL: effective asset {a} yielded {per_lang_cuts[lang]} "
                    f"{lang} cuts < min_cuts={min_cuts}. The harness refuses to under-sample. "
                    f"(eligible_bars={elig}; raise --audit-max-bars or lower --min-cuts deliberately.)")

        # commit this effective asset
        for lang, _v in langs:
            r = per_lang_r[lang]
            A = acc[lang]
            A["all_lags"].extend(r["churn_lags"])
            A["n_churned"] += r["n_churned_cuts"]
            A["n_cuts_tot"] += r["n_cuts"]
            A["n_stress_cuts"] += r.get("n_stress_cuts", 0)
            A["n_stress_eligible"] += r.get("n_stress_eligible", 0)
            A["per_asset"].append(dict(asset=int(a), half=half,
                                       **{k: r[k] for k in ("n_cuts", "n_churned_cuts")},
                                       n_stress_cuts=int(r.get("n_stress_cuts", 0))))
        effective_assets.append(int(a))
        used_prices[int(a)] = float(pool["prices"].get(int(a), float("nan")))
        used_halves[half] += 1
        print(f"[audit] EFFECTIVE {len(effective_assets)}/{target_eff}: asset {a} ({half}) "
              f"L1 cuts={l1_cuts} L1 stress={per_lang_r['L1_bpe'].get('n_stress_cuts',0)} "
              f"L8 cuts={per_lang_cuts['L8_haar']}", flush=True)

    n_effective = len(effective_assets)
    print(f"[audit] effective assets: {n_effective}/{target_eff} (drawn {n_drawn}, "
          f"thin/replaced {len(thin_assets)}; halves {used_halves})", flush=True)
    if n_effective < target_eff:
        # loud, machine-readable under-sampling (NOT a silent cap).
        deviations.append(dict(name="effective_assets_shortfall", value=n_effective,
                               detail=f"only {n_effective} of {target_eff} target effective assets reached "
                                      f"after drawing {n_drawn} (pool exhausted or --max-draws hit). "
                                      f"thin/replaced: {len(thin_assets)}."))
        print(f"[audit] WARNING: effective-asset shortfall {n_effective}<{target_eff} "
              f"recorded in deviations[].", flush=True)

    # roll up per-language stats from the effective accumulators.
    lang_results = {}
    for lang, _v in langs:
        A = acc[lang]
        lags = np.asarray(A["all_lags"], dtype=np.int64) if A["all_lags"] else np.zeros(0, np.int64)
        p50 = int(np.percentile(lags, 50)) if lags.size else 0
        p99 = int(np.percentile(lags, 99)) if lags.size else 0
        pmax = int(lags.max()) if lags.size else 0
        lang_results[lang] = dict(
            p50_lag=p50, p99_lag=p99, max_lag=pmax,
            n_cuts=int(A["n_cuts_tot"]), n_churned_cuts=int(A["n_churned"]),
            churn_rate=float(A["n_churned"] / max(A["n_cuts_tot"], 1)),
            n_effective_assets=int(n_effective),
            n_assets_drawn=int(n_drawn),
            n_stress_cuts=int(A["n_stress_cuts"]),
            n_stress_eligible=int(A["n_stress_eligible"]),
            lag_histogram={int(k): int(v) for k, v in zip(*np.unique(lags, return_counts=True))} if lags.size else {},
            per_asset=A["per_asset"],
        )
        print(f"[audit] {lang}: eff_assets={n_effective} cuts={A['n_cuts_tot']} "
              f"churned={A['n_churned']} stress_cuts={A['n_stress_cuts']} "
              f"p50_lag={p50} p99_lag={p99} max_lag={pmax}", flush=True)

    # synthesize a `samp`-shaped object for the result writer (effective sample only).
    half_of = {pa["asset"]: pa["half"] for pa in lang_results["L1_bpe"]["per_asset"]}
    samp = dict(
        build_like=[a for a in effective_assets if half_of.get(a) == "build_like"],
        holdout_like=[a for a in effective_assets if half_of.get(a) == "holdout_like"],
        prices={int(a): float(used_prices.get(int(a), float("nan"))) for a in effective_assets},
        thin_assets=thin_assets,
        split_source=pool["split_source"],
        n_effective=int(n_effective),
        n_drawn=int(n_drawn),
        target_effective=int(target_eff),
    )

    if not effective_assets:
        raise SystemExit("[audit] FAIL: zero effective assets — cannot produce a verdict.")

    # ---- A0 through the path (one sampled asset, x1000) ----
    a0_asset = effective_assets[0]
    print(f"[audit] A0 rescale x1000 through timeline on asset {a0_asset} ...", flush=True)
    a0 = a0_through_timeline(db, a0_asset, cfg, vocabs, lam=1000.0,
                             max_feature_bars=int(args.audit_max_bars))
    DS.clear_window_cache()
    print(f"[audit] A0: {a0}", flush=True)

    # ---- verdicts ----
    l8_verdict, l8_reason = _verdict_for("L8_haar", lang_results["L8_haar"]["p50_lag"],
                                         lang_results["L8_haar"]["p99_lag"],
                                         lang_results["L8_haar"]["n_churned_cuts"],
                                         lang_results["L8_haar"]["n_cuts"], a0.get("l8_identical"))
    l1_verdict, l1_reason = _verdict_for("L1_bpe", lang_results["L1_bpe"]["p50_lag"],
                                         lang_results["L1_bpe"]["p99_lag"],
                                         lang_results["L1_bpe"]["n_churned_cuts"],
                                         lang_results["L1_bpe"]["n_cuts"], a0.get("l1_identical"))

    _write_result(args.out, cfg, vocabs, samp, smoke, lang_results, a0,
                  dict(L8_haar=(l8_verdict, l8_reason), L1_bpe=(l1_verdict, l1_reason)),
                  deviations, merge_thresh)
    print(f"[audit] L8 verdict: {l8_verdict} ({l8_reason})")
    print(f"[audit] L1 verdict: {l1_verdict} ({l1_reason})")
    return dict(smoke=smoke, lang_results=lang_results, a0=a0,
                verdicts=dict(L8_haar=l8_verdict, L1_bpe=l1_verdict))


# --------------------------------------------------------------------------- #
# RESULT.md / .json
# --------------------------------------------------------------------------- #
_L2_L5_MEMO = """\
L2 ABBA and L5 matrix-profile are NOT built here (TASK_800 scope: build exactly L1 + L8). A ≤1-page
feasibility memo on the per-bar alignment each would require:

**L2 ABBA (adaptive Brownian-bridge aggregation).** ABBA segments a series into variable-length
linear pieces, then symbolizes each piece's (length, value-delta) by a CLUSTERING fit over the WHOLE
series. Two causality hazards stack: (i) like L1 BPE, the segment covering bar t is boundary-sensitive
— the current piece keeps GROWING until the bridge error exceeds tolerance, so the symbol for the
in-flight segment is unknown until it closes (a firm-up lag at least as large as L1's, plausibly
larger because ABBA pieces are longer than BPE tokens); (ii) the symbol ALPHABET itself comes from a
k-means over the segment (len, Δ) cloud — a GLOBAL fit. For a frozen serving vocab the centroids must
be fit ONCE on BUILD (as L1's merges and L8's MCB bins are), then segments assigned by nearest frozen
centroid. With a frozen alphabet the only per-bar hazard is segment-boundary churn, identical in
character to L1: emit the last CLOSED segment's symbol per bar, expect a bounded firm-up lag of order
the mean segment length. Build cost: a frozen-ABBA fit path (segment + frozen-centroid assign) does
not yet exist; the segmenter is causal but the centroid fit is not wired as a frozen artifact.
Expected churn behaviour: SERVABLE-WITH-LAG-k with k ≈ mean segment length (likely larger than L1).

**L5 matrix-profile.** The matrix profile annotates each subsequence with the distance to its nearest
neighbour ELSEWHERE in the series — an inherently NON-CAUSAL, all-pairs object (the nearest neighbour
of the window ending at t can lie in the FUTURE). A causal "left matrix profile" variant restricts the
neighbour search to subsequences ending at or before t, which IS prefix-causal but (a) changes the
semantics (the token is "closest past analogue"), and (b) its value at bar t can still firm up as new
future windows are excluded — actually the left-MP at t is fixed once computed (only past neighbours),
so it is window-rate-exact like L8 IF the discretization of the MP distance into a token uses a FROZEN
BUILD quantizer. The real obstacle is cost, not causality: a streaming left-matrix-profile (STAMP/STOMP
incremental) is O(n) per bar but heavy, and the token would be a quantized scalar distance — low
information vs L1/L8. Expected behaviour for a frozen left-MP: SERVABLE (window-rate exact) but
expensive to serve and likely weak as a standalone language. Recommend deferring both; L1 + L8 cover
the variable-rate and window-rate causality archetypes the trader needs.
"""


def _write_result(out, cfg, vocabs, samp, smoke, lang_results, a0, verdicts,
                  deviations=None, merge_thresh=None):
    l8v, l8r = verdicts["L8_haar"]
    l1v, l1r = verdicts["L1_bpe"]
    deviations = list(deviations or [])
    n_eff = int(samp.get("n_effective", len(samp["build_like"]) + len(samp["holdout_like"])))
    n_drawn = int(samp.get("n_drawn", n_eff))
    thin = samp.get("thin_assets", [])
    l1lr = lang_results["L1_bpe"]
    # headline uses the EFFECTIVE asset count x cuts, never the nominal drawn count.
    headline_cuts = int(l1lr["n_cuts"])
    prices = [p for p in samp["prices"].values() if p == p]  # drop NaN
    pspan = (f"{min(prices):.4g}..{max(prices):.4g}" if prices else "n/a")
    sections = [
        ("Summary — per-language SERVABLE verdict (selects the trader language)", [
            f"- **L8 Haar (window-rate)**: **{l8v}** — {l8r}",
            f"- **L1 BPE (variable-rate)**: **{l1v}** — {l1r}",
            f"- coverage: **{n_eff} EFFECTIVE assets × {headline_cuts} L1 cuts** "
            f"(drawn {n_drawn}; thin/replaced {len(thin)}). Headline = EFFECTIVE count, not nominal.",
            f"- L1 merge stress: {l1lr.get('n_stress_cuts', 0)} merge-adjacent cut points "
            f"(merge-id threshold = {merge_thresh}); these aim the audit at the deep-merge firm-up "
            f"mechanism — the ONLY path to L1 lag>0.",
            "- Convention: emit the LAST COMPLETED token at or before t; no <PENDING> vocab; "
            "pre-first-token + gap bars get token_id=-1, valid=False.",
        ]),
        ("Gating smoke (one asset, ~10k bars, both languages)", [
            f"- smoke_pass: {smoke['smoke_pass']}  (wall {smoke['wall_s']:.1f}s, <60s required)",
            f"- L1: {smoke['L1_bpe']}",
            f"- L8: {smoke['L8_haar']}",
        ]),
        ("Prefix-stability (causality) churn audit", [
            f"- sample: {n_eff} effective assets "
            f"({len(samp['build_like'])} build-like + {len(samp['holdout_like'])} holdout-like), "
            f"price-stratified across both split halves; split source: {samp.get('split_source','?')}; "
            f"price span {pspan}",
            f"- L8 Haar: cuts={lang_results['L8_haar']['n_cuts']} "
            f"churned={lang_results['L8_haar']['n_churned_cuts']} "
            f"p50_lag={lang_results['L8_haar']['p50_lag']} "
            f"**p99_lag={lang_results['L8_haar']['p99_lag']}** "
            f"max_lag={lang_results['L8_haar']['max_lag']} (churn must be 0)",
            f"- L1 BPE: cuts={l1lr['n_cuts']} "
            f"stress_cuts={l1lr.get('n_stress_cuts', 0)} "
            f"churned={l1lr['n_churned_cuts']} "
            f"p50_lag={l1lr['p50_lag']} "
            f"**p99_lag={l1lr['p99_lag']}** "
            f"max_lag={l1lr['max_lag']}  lag_hist={l1lr.get('lag_histogram', {})}",
            f"- churn(t,k) = [token at bar t-k from prefix<=t] != [same from full stream]; "
            f"lag = greatest k still disagreeing per cut.",
        ]),
        ("A0 rescale-invariance through the timeline path (x1000)", [
            f"- L1 token_id byte-identical under x1000: {a0.get('l1_identical')}",
            f"- L8 token_id byte-identical under x1000: {a0.get('l8_identical')}",
            f"- bars compared: L1={a0.get('l1_n_bars')} L8={a0.get('l8_n_bars')}",
        ]),
        ("L2 / L5 feasibility memo (NOT built — TASK_800 scope is L1 + L8)",
         _L2_L5_MEMO.strip().splitlines()),
    ]
    meta = dict(
        task="TASK_800",
        convention="last_completed_token_at_or_before_t; invalid=-1; no_PENDING_vocab",
        window=int(cfg.window),
        l1_codebook_size=int(vocabs["l1_vocab"].codebook_size),
        l1_n_merges=int(len(vocabs["l1_vocab"].merges)),
        l8_codebook_size=int(vocabs["l8_vocab"].codebook_size),
        # machine-readable per-language verdict (this SELECTS the trader language downstream).
        verdict=dict(
            L8_haar=dict(verdict=l8v, reason=l8r, **lang_results["L8_haar"]),
            L1_bpe=dict(verdict=l1v, reason=l1r, **lang_results["L1_bpe"]),
        ),
        merged_id_threshold=(int(merge_thresh) if merge_thresh is not None else None),
        a0_rescale=a0,
        smoke=smoke,
        coverage=dict(
            n_effective_assets=n_eff,
            n_assets_drawn=n_drawn,
            target_effective=int(samp.get("target_effective", n_eff)),
            n_thin_replaced=len(thin),
            l1_stress_cuts=int(l1lr.get("n_stress_cuts", 0)),
            l1_stress_eligible=int(l1lr.get("n_stress_eligible", 0)),
            l1_lag_histogram=l1lr.get("lag_histogram", {}),
            l8_lag_histogram=lang_results["L8_haar"].get("lag_histogram", {}),
        ),
        sample=dict(build_like=samp["build_like"], holdout_like=samp["holdout_like"],
                    prices=samp["prices"], split_source=samp.get("split_source")),
        thin_assets=thin,
        # machine-readable split / coverage deviations (NOT a prose footnote).
        deviations=deviations,
    )
    RW.write_result(out, "TASK_800 — Token Timeline + Serving-Path Causality Audit",
                    sections, f"L8={l8v} | L1={l1v}", meta)


if __name__ == "__main__":
    main()
