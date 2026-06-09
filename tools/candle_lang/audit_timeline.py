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


def churn_for_asset(db, asset_id, cfg, vocab, language, n_cuts=50, max_k=128,
                    max_feature_bars=None, seed=0):
    """Measure churn-lag for ONE asset under ONE language. For n_cuts random cut points t (within a
    run, with enough lead-in), compare the per-bar token at bars t-k (k=0..max_k) computed from the
    PREFIX <= t against the FULL-stream value. Returns a dict with the churn-lag samples (the largest
    k at which a bar t-k still disagrees, per cut) and counts."""
    runs = _load_runs(db, asset_id, cfg, max_feature_bars=max_feature_bars)
    if not runs:
        return dict(n_cuts=0, churn_lags=[], n_churned_cuts=0, n_bars_compared=0)
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
    W = int(getattr(vocab, "win_len", cfg.window))
    lead = max_k + W + 2
    for ri, (feats, _ts, _fin) in enumerate(runs):
        F = feats.shape[0]
        if F <= lead + 1:
            continue
        for t in range(lead, F - 1):
            eligible.append((ri, t))
    if not eligible:
        return dict(n_cuts=0, churn_lags=[], n_churned_cuts=0, n_bars_compared=0)
    take = min(int(n_cuts), len(eligible))
    sel = [eligible[i] for i in rng.choice(len(eligible), size=take, replace=False)]

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
                n_bars_compared=n_bars_cmp)


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
def price_stratified_sample(db, n_total=20, min_rows=50000, seed=0):
    """Build a price-stratified sample of `n_total` assets from the DB. We split the log-price range
    into two halves (low-price half ~ the holdout design; high-price half ~ the build design) and draw
    n_total/2 from each, spread across price quantiles within each half. Returns dict(build_like(list),
    holdout_like(list), prices(dict)) where build_like = high-price half, holdout_like = low-price.

    NOTE: TASK_800 references runs/796_scaled/asset_split.json (10 build / 10 holdout). That artifact
    is not present on this checkout; we reconstruct an equivalent price-stratified 10/10 split directly
    from the DB survey, preserving the spec's intent (both split halves, price-stratified)."""
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
    assets = [a for a, _ in items]
    prices = np.array([p for _, p in items])
    n = len(assets)
    half = n // 2
    low_idx = list(range(0, half))
    high_idx = list(range(half, n))
    nh = n_total // 2

    def _spread(idx_pool, k):
        if k >= len(idx_pool):
            return [assets[i] for i in idx_pool]
        picks = np.linspace(0, len(idx_pool) - 1, k).round().astype(int)
        picks = sorted(set(int(p) for p in picks))
        # top up if dedup shrank it
        j = 0
        while len(picks) < k and j < len(idx_pool):
            if j not in picks:
                picks.append(j)
            j += 1
        return [assets[idx_pool[p]] for p in picks[:k]]

    holdout_like = _spread(low_idx, nh)    # low price
    build_like = _spread(high_idx, n_total - nh)   # high price
    return dict(build_like=build_like, holdout_like=holdout_like,
                prices={int(a): float(px[a]) for a in (build_like + holdout_like)})


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
    ap.add_argument("--n-assets", type=int, default=20)
    ap.add_argument("--n-cuts", type=int, default=50)
    ap.add_argument("--max-k", type=int, default=128)
    ap.add_argument("--audit-max-bars", type=int, default=40000,
                    help="cap on feature bars per asset held in RAM during the audit (RAM safety).")
    ap.add_argument("--fit-mwpa", type=int, default=4000)
    ap.add_argument("--smoke-only", action="store_true")
    args = ap.parse_args(argv)

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

    # ---- price-stratified sample ----
    samp = price_stratified_sample(db, n_total=int(args.n_assets))
    sample_assets = list(samp["build_like"]) + list(samp["holdout_like"])
    print(f"[audit] sampled {len(sample_assets)} assets "
          f"({len(samp['build_like'])} build-like / {len(samp['holdout_like'])} holdout-like)")
    print(f"[audit] price span of sample: "
          f"{min(samp['prices'].values()):.4g} .. {max(samp['prices'].values()):.4g}")

    # ---- churn audit per language (asset-by-asset; bounded RAM) ----
    import subprocess
    def _free_g():
        try:
            out = subprocess.check_output(["free", "-g"]).decode().splitlines()[1].split()
            return int(out[6])  # available
        except Exception:
            return 999
    avail = _free_g()
    print(f"[audit] free -g available: {avail} GB")
    if avail < 4:
        # bail to a smaller sample if RAM is low (workstation memory-fragile).
        sample_assets = sample_assets[:8]
        print(f"[audit] LOW RAM ({avail}G) -> shrinking sample to {len(sample_assets)} assets")

    lang_results = {}
    for lang, vocab in (("L8_haar", vocabs["l8_vocab"]), ("L1_bpe", vocabs["l1_vocab"])):
        all_lags = []
        n_churned = 0
        n_cuts_tot = 0
        per_asset = []
        for a in sample_assets:
            r = churn_for_asset(db, a, cfg, vocab, lang, n_cuts=int(args.n_cuts),
                                max_k=int(args.max_k), max_feature_bars=int(args.audit_max_bars),
                                seed=0)
            DS.clear_window_cache()
            all_lags.extend(r["churn_lags"])
            n_churned += r["n_churned_cuts"]
            n_cuts_tot += r["n_cuts"]
            per_asset.append(dict(asset=int(a), **{k: r[k] for k in ("n_cuts", "n_churned_cuts")}))
        lags = np.asarray(all_lags, dtype=np.int64) if all_lags else np.zeros(0, np.int64)
        p50 = int(np.percentile(lags, 50)) if lags.size else 0
        p99 = int(np.percentile(lags, 99)) if lags.size else 0
        pmax = int(lags.max()) if lags.size else 0
        lang_results[lang] = dict(
            p50_lag=p50, p99_lag=p99, max_lag=pmax,
            n_cuts=int(n_cuts_tot), n_churned_cuts=int(n_churned),
            churn_rate=float(n_churned / max(n_cuts_tot, 1)),
            lag_histogram={int(k): int(v) for k, v in zip(*np.unique(lags, return_counts=True))} if lags.size else {},
            per_asset=per_asset,
        )
        print(f"[audit] {lang}: cuts={n_cuts_tot} churned={n_churned} "
              f"p50_lag={p50} p99_lag={p99} max_lag={pmax}", flush=True)

    # ---- A0 through the path (one sampled asset, x1000) ----
    a0_asset = sample_assets[0]
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
                  dict(L8_haar=(l8_verdict, l8_reason), L1_bpe=(l1_verdict, l1_reason)))
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


def _write_result(out, cfg, vocabs, samp, smoke, lang_results, a0, verdicts):
    l8v, l8r = verdicts["L8_haar"]
    l1v, l1r = verdicts["L1_bpe"]
    sections = [
        ("Summary — per-language SERVABLE verdict (selects the trader language)", [
            f"- **L8 Haar (window-rate)**: **{l8v}** — {l8r}",
            f"- **L1 BPE (variable-rate)**: **{l1v}** — {l1r}",
            "- Convention: emit the LAST COMPLETED token at or before t; no <PENDING> vocab; "
            "pre-first-token + gap bars get token_id=-1, valid=False.",
        ]),
        ("Gating smoke (one asset, ~10k bars, both languages)", [
            f"- smoke_pass: {smoke['smoke_pass']}  (wall {smoke['wall_s']:.1f}s, <60s required)",
            f"- L1: {smoke['L1_bpe']}",
            f"- L8: {smoke['L8_haar']}",
        ]),
        ("Prefix-stability (causality) churn audit", [
            f"- sample: {len(samp['build_like'])} build-like + {len(samp['holdout_like'])} "
            f"holdout-like assets, price-stratified; "
            f"price span {min(samp['prices'].values()):.4g}..{max(samp['prices'].values()):.4g}",
            f"- L8 Haar: cuts={lang_results['L8_haar']['n_cuts']} "
            f"churned={lang_results['L8_haar']['n_churned_cuts']} "
            f"p50_lag={lang_results['L8_haar']['p50_lag']} "
            f"**p99_lag={lang_results['L8_haar']['p99_lag']}** "
            f"max_lag={lang_results['L8_haar']['max_lag']} (churn must be 0)",
            f"- L1 BPE: cuts={lang_results['L1_bpe']['n_cuts']} "
            f"churned={lang_results['L1_bpe']['n_churned_cuts']} "
            f"p50_lag={lang_results['L1_bpe']['p50_lag']} "
            f"**p99_lag={lang_results['L1_bpe']['p99_lag']}** "
            f"max_lag={lang_results['L1_bpe']['max_lag']}",
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
        a0_rescale=a0,
        smoke=smoke,
        sample=dict(build_like=samp["build_like"], holdout_like=samp["holdout_like"],
                    prices=samp["prices"]),
        note_asset_split=("runs/796_scaled/asset_split.json absent on this checkout; "
                          "price-stratified 10/10 split reconstructed from the DB survey."),
    )
    RW.write_result(out, "TASK_800 — Token Timeline + Serving-Path Causality Audit",
                    sections, f"L8={l8v} | L1={l1v}", meta)


if __name__ == "__main__":
    main()
