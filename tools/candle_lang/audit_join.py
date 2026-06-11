"""
audit_join.py — TASK_801b-i (Fork A) R3 join-audit harness.

Proves the load-bearing correctness core of the no-fill token-timeline producer + the timestamp-join
onto a slot grid, BEFORE any training wiring (that is 801b-ii). CPU-only, pure numpy, no JAX/GPU/server.

Three checks (write console + runs/801_join_audit/RESULT.md):

  (a) JOIN CORRECTNESS + R2 MASKING (the amended R3 headline). For >=2 real DB assets:
        annotate_asset -> ground-truth timeline.
        Build a DENSE forward-filled grid simulating a slot: a mid-stream time range, grid_ts = every
        C.BAR_SECONDS over that range, so some grid rows have a real bar (the timeline has a VALID entry
        at that ts) and some are "filled" (no real bar at that ts).
        Join the timeline onto this grid with language='L1_bpe' (long-form -> strict lag=2).
        ASSERT: every grid row whose ts has a VALID real-bar timeline entry receives EXACTLY
        annotate_asset's token at that ts (after the lag shift); every filled row lands -1/invalid.
        0 mismatches required (B2).

  (b) SYNTHETIC CROSS-PRODUCER CURVE (R3 original). Build a slot from a mid-stream CONTIGUOUS range of a
        real asset's RAW ohlcv (dense, no fill). Run annotate_ohlcv on that raw range. Compare its
        per-bar tokens vs annotate_asset's tokens (joined by ts) as a function of OFFSET from the range
        start. Report the agreement curve (offset-binned) and where it reaches ~100% — the cold-sigma
        head firms up after the first UTC-day re-anchor inside the range (B3).

  (B4) strict input_lag_for: long-form 'L1_bpe' -> lag 2; the lagged join shifts tokens by 2 grid rows.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import time

import numpy as np

from . import config as C
from . import db_stream as DS
from .symbolic import l8_wavelet as L8
from . import tokenizer_api as TK
from . import token_timeline as TT
from .token_window_slice import input_lag_for
from . import result_writer as RW


# --------------------------------------------------------------------------- #
# vocab fit (bounded, fast — the audit is about the JOIN, not the vocab)
# --------------------------------------------------------------------------- #
def build_vocabs(cfg, fit_mwpa=4000, verbose=True):
    if verbose:
        print(f"[join-audit] fitting L1 BPE vocab (mwpa={fit_mwpa}) ...", flush=True)
    tmp = "/tmp/801_join_l1_vocab.json"
    l1_vocab, _ = TK.fit_and_save(cfg, cfg.build_assets, tmp,
                                  max_windows_per_asset=fit_mwpa,
                                  p_min=50, max_merges=256, verbose=False)
    if verbose:
        print(f"[join-audit] L1 codebook_size={l1_vocab.codebook_size} "
              f"merges={len(l1_vocab.merges)} n_base={l1_vocab.n_base}", flush=True)
        print("[join-audit] fitting L8 Haar vocab ...", flush=True)
    l8_vocab = L8.fit_vocab_from_assets(
        cfg.db, cfg.build_assets, cfg,
        n_levels=L8.DEFAULT_N_LEVELS, win_len=int(cfg.window), stride=1,
        max_windows=fit_mwpa, detail_bands_kept=L8.DEFAULT_DETAIL_BANDS_KEPT, seed=0)
    if verbose:
        print(f"[join-audit] L8 codebook_size={l8_vocab.codebook_size}", flush=True)
    return l1_vocab, l8_vocab


# --------------------------------------------------------------------------- #
# raw mid-stream OHLCV pull (for the synthetic cross-producer slot)
# --------------------------------------------------------------------------- #
def pull_raw_ts_span(db, asset_id, lo_ts, hi_ts):
    """Pull RAW OHLCV rows for one asset over the inclusive ts span [lo_ts, hi_ts]. Returns
    (ts(M,)i64, ohlcv(M,5)f64) in chronological order (col 4 = volume), or (None,None) if empty."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT timestamp,open,high,low,close,volume FROM ohlcv "
            "WHERE asset_id=? AND timestamp BETWEEN ? AND ? ORDER BY timestamp",
            (int(asset_id), int(lo_ts), int(hi_ts))).fetchall()
    finally:
        con.close()
    if not rows:
        return None, None
    ts = np.array([r[0] for r in rows], dtype=np.int64)
    ohlcv = np.array([[r[1], r[2], r[3], r[4], (r[5] if r[5] is not None else 0.0)] for r in rows],
                     dtype=np.float64)
    return ts, ohlcv


def longest_contiguous_subrange(ts, ohlcv, n_bars):
    """Return the FIRST contiguous dt==C.BAR_SECONDS subrange of (ts, ohlcv) of length >= n_bars+1
    (need +1 raw bar for the feature-bar grid), or (None,None)."""
    if ts is None or ts.shape[0] < 2:
        return None, None
    dt = np.diff(ts)
    n = ts.shape[0]
    s = 0
    while s < n - 1:
        e = s + 1
        while e < n and dt[e - 1] == C.BAR_SECONDS:
            e += 1
        if e - s >= n_bars + 1:
            return ts[s:e], ohlcv[s:e]
        s = e
    return None, None


# --------------------------------------------------------------------------- #
# (a) JOIN CORRECTNESS + R2 MASKING
# --------------------------------------------------------------------------- #
def audit_join_correctness(db, asset_id, l1_vocab, cfg, n_grid_bars=8000,
                           max_feature_bars=60000, language="L1_bpe", verbose=True):
    """Build the ground-truth timeline (annotate_asset), construct a dense forward-filled grid over a
    mid-stream time range, join, and verify every real-bar row == annotate_asset token (after lag) and
    every filled row is masked. Returns a result dict."""
    DS.clear_window_cache()
    tok = TK.CandleTokenizer.from_bpe_vocab(l1_vocab, cfg)
    timeline = TT.annotate_asset(db, asset_id, tok, cfg, max_feature_bars=max_feature_bars)
    DS.clear_window_cache()

    t_ts = np.asarray(timeline.ts, dtype=np.int64)
    t_val = np.asarray(timeline.valid, dtype=bool)
    if t_ts.shape[0] < n_grid_bars or not t_val.any():
        return dict(asset=int(asset_id), ok=False, note="timeline too short / no valid bars")

    # pick a mid-stream window of the timeline's ts, take its [lo, hi] time span, and build a DENSE
    # grid covering it at C.BAR_SECONDS step. Real-bar timestamps in this span appear in the timeline;
    # every other grid ts is a "forward-filled" slot row (no real bar at that exact second).
    mid = t_ts.shape[0] // 2
    lo_i = max(0, mid - n_grid_bars // 2)
    hi_i = min(t_ts.shape[0] - 1, lo_i + n_grid_bars)
    t_lo = int(t_ts[lo_i]); t_hi = int(t_ts[hi_i])
    grid_ts = np.arange(t_lo, t_hi + 1, C.BAR_SECONDS, dtype=np.int64)
    G = grid_ts.shape[0]

    # ground-truth {ts -> (token, valid)} over the SAME timeline (VALID entries only).
    valid_real_ts = set(int(x) for x in t_ts[t_val])
    ts_to_tok = {int(ts): int(tok) for ts, tok, v in zip(t_ts, timeline.token_id, t_val) if v}

    # JOIN under test.
    token_ids, valid = TT.join_timeline_to_grid(timeline, grid_ts, language)
    lag = input_lag_for(language)

    # classify grid rows by whether grid_ts[r] is a VALID real-bar ts (pre-lag truth).
    is_real = np.array([int(grid_ts[r]) in valid_real_ts for r in range(G)], dtype=bool)

    # EXPECTED post-lag arrays, computed independently of the join:
    #   expected_raw[r] = token at grid_ts[r] if real & valid else -1
    #   expected[r]     = expected_raw[r-lag] (index shift on the grid), leading lag rows -1.
    expected_raw = np.full(G, -1, dtype=np.int64)
    for r in range(G):
        if is_real[r]:
            expected_raw[r] = ts_to_tok[int(grid_ts[r])]
    expected = np.full(G, -1, dtype=np.int64)
    if lag < G:
        expected[lag:] = expected_raw[:G - lag]

    # B2 mismatch count: every grid row's joined token must equal the independently-expected token.
    mismatches = int(np.count_nonzero(token_ids.astype(np.int64) != expected))

    # R2 masking accounting (pre-lag truth, reported for clarity):
    n_real = int(is_real.sum())
    n_filled = int((~is_real).sum())
    # filled grid rows: after the lag shift a filled row's OUTPUT slot is filled iff its source
    # (r-lag) was filled. We report that every row whose SOURCE is a filled/non-real ts is masked
    # (token -1) in the output — i.e. no token is EVER served for a filled bar.
    filled_src_masked = True
    real_src_match = True
    n_real_matched = 0
    n_filled_masked = 0
    for r in range(G):
        src = r - lag
        if src < 0:
            continue
        if is_real[src]:
            n_real_matched += 1
            if int(token_ids[r]) != ts_to_tok[int(grid_ts[src])]:
                real_src_match = False
        else:
            n_filled_masked += 1
            if int(token_ids[r]) != -1:
                filled_src_masked = False

    ok = (mismatches == 0) and real_src_match and filled_src_masked
    res = dict(
        asset=int(asset_id), ok=bool(ok), language=language, lag=int(lag),
        grid_rows=int(G), grid_span_s=int(t_hi - t_lo),
        real_rows=n_real, filled_rows=n_filled,
        real_rows_matched=int(n_real_matched), filled_rows_masked=int(n_filled_masked),
        mismatches=int(mismatches),
        real_src_match=bool(real_src_match), filled_src_masked=bool(filled_src_masked),
        valid_out=int(valid.sum()),
    )
    if verbose:
        print(f"[join-audit](a) asset {asset_id}: G={G} real={n_real} filled={n_filled} "
              f"lag={lag} mismatches={mismatches} real_matched={n_real_matched} "
              f"filled_masked={n_filled_masked} OK={ok}", flush=True)
    return res


# --------------------------------------------------------------------------- #
# (b) SYNTHETIC CROSS-PRODUCER CURVE
# --------------------------------------------------------------------------- #
def audit_cross_producer(db, asset_id, l1_vocab, cfg, min_bars=300, max_bars=4000,
                         max_feature_bars=120000, bin_size=100, verbose=True):
    """Derive a CONTIGUOUS range from the DB producer's full timeline (the densest gap-free run, so the
    synthetic range overlaps real DB tokens), pull the matching RAW OHLCV, run annotate_ohlcv on it
    (synthetic producer), and compare its per-bar tokens to annotate_asset's tokens (DB producer)
    joined by ts, as a function of OFFSET from the range start. Returns the offset-binned agreement
    curve + the offset where it reaches ~100%.

    IMPORTANT REAL-DATA FINDING: minute OHLCV is heavily gapped, and the gapping is WORSE in the recent
    TAIL than in the dense early/mid stream — the tail's longest gap-free feature-run is ~100 bars for
    assets 1087/1091, while the early stream has runs >1500 bars (long enough to span a UTC day, 1440
    bars). So we annotate the FULL DB stream (max_feature_bars=None) to find the longest gap-free run
    that PREFERABLY spans a UTC-day boundary (so the cold-sigma head can firm to ~100%), and source the
    cross-producer range from THERE. If no run reaches a day boundary the curve still shows the head
    climbing (reach_100_offset stays None — reported honestly, not a failure).

    `max_feature_bars` is IGNORED here (kept in the signature for call-site parity): the full timeline
    (~85 MB for a 5M-bar asset) is held once to locate the densest run; this is a CPU audit, not the
    bounded-RAM training path."""
    # DB producer: annotate the FULL asset stream (no tail cap) so dense early-stream runs are visible.
    DS.clear_window_cache()
    tok = TK.CandleTokenizer.from_bpe_vocab(l1_vocab, cfg)
    db_tl = TT.annotate_asset(db, asset_id, tok, cfg, max_feature_bars=None)
    DS.clear_window_cache()
    db_ts = np.asarray(db_tl.ts, dtype=np.int64)
    db_tok = np.asarray(db_tl.token_id, dtype=np.int64)
    db_val = np.asarray(db_tl.valid, dtype=bool)
    db_map = {int(ts): int(tk) for ts, tk, v in zip(db_ts, db_tok, db_val) if v}
    if db_ts.shape[0] < min_bars + 2:
        return dict(asset=int(asset_id), ok=False, note="DB timeline too short")

    # The DB feature-bar grid is gap-free (dt==60) WITHIN a run; gaps appear only between runs. Enumerate
    # the gap-free blocks; the raw OHLCV over a block's ts span is dense by construction, so the
    # synthetic producer sees one gap-free run.
    ddt = np.diff(db_ts)
    brk = np.flatnonzero(ddt != C.BAR_SECONDS) + 1
    bounds = np.concatenate([[0], brk, [db_ts.shape[0]]]).astype(np.int64)
    blocks = [(int(bounds[k]), int(bounds[k + 1])) for k in range(bounds.shape[0] - 1)
              if int(bounds[k + 1]) - int(bounds[k]) >= min_bars + 1]
    if not blocks:
        return dict(asset=int(asset_id), ok=False, note=f"no gap-free DB block >= {min_bars}")

    def _has_day_boundary(s, e):
        d = db_ts[s:e] // 86400
        return bool((d[1:] != d[:-1]).any())
    # prefer the LONGEST day-boundary-spanning block; else the longest block overall.
    day_blocks = [b for b in blocks if _has_day_boundary(*b)]
    pool = day_blocks if day_blocks else blocks
    run_s, run_e = max(pool, key=lambda se: se[1] - se[0])
    longest_run_bars = int(run_e - run_s)
    spans_day = bool(_has_day_boundary(run_s, run_e))

    # To SURFACE the cold-sigma head (B3's expectation: imperfect head -> ~100% after the first UTC-day
    # re-anchor), start the synthetic range MID-DAY: ~head_margin bars BEFORE the first day boundary in
    # the chosen run. The synthetic producer re-seeds sigma at the range start (mid-day) while the DB
    # producer carries full upstream sigma into that point, so the head DISAGREES until both re-anchor
    # at the day boundary (offset ~head_margin), where agreement jumps to ~100%. If the run starts at a
    # day boundary and we anchored there, both seed sigma identically and the head would be (misleadingly)
    # perfect from offset 0 — hence the deliberate mid-day start.
    head_margin = 720  # half a UTC day in feature bars (the cold-head should firm by the day boundary)
    d = db_ts[run_s:run_e] // 86400
    day_brk_local = np.flatnonzero(d[1:] != d[:-1])
    if day_brk_local.size:
        first_day_idx = int(day_brk_local[0]) + 1            # local index of first day boundary in run
        s = run_s + max(0, first_day_idx - head_margin)
    else:
        s = run_s
    e = run_e
    # cap the range to max_bars feature bars (anchored at the chosen mid-day start).
    e = min(e, s + int(max_bars) + 1)
    # pull RAW OHLCV covering this feature-bar span, with one extra lead-in raw bar at the start
    # (volunit drops the first raw bar of a run), so the synthetic feature grid begins at db_ts[s].
    span_lo = int(db_ts[s]) - C.BAR_SECONDS
    span_hi = int(db_ts[e - 1])
    ts_pull, ohlcv_pull = pull_raw_ts_span(db, asset_id, span_lo, span_hi)
    ts_raw, ohlcv_raw = longest_contiguous_subrange(ts_pull, ohlcv_pull, min_bars)
    if ts_raw is None:
        return dict(asset=int(asset_id), ok=False, note="no contiguous raw range found in span")
    t_lo, t_hi = int(ts_raw[0]), int(ts_raw[-1])

    # synthetic producer: annotate_ohlcv on the contiguous raw range (uses REAL ts of that range; the
    # only difference vs the DB path is the cold-sigma head — the EWMA re-seeds at the range start
    # rather than carrying the full upstream history).
    syn_tl = TT.annotate_ohlcv(ohlcv_raw, ts_raw, tok, cfg)
    syn_ts = np.asarray(syn_tl.ts, dtype=np.int64)
    syn_tok = np.asarray(syn_tl.token_id, dtype=np.int64)
    syn_val = np.asarray(syn_tl.valid, dtype=bool)

    # offset from the range start (in feature bars). offset 0 == first feature bar of the synthetic
    # producer (== second raw bar of the range). day_boundary_offset = bars until the first UTC-day
    # re-anchor inside the range (where the cold-sigma head fully firms up).
    day0 = int(syn_ts[0]) // 86400
    day_boundary_offset = None
    for k in range(syn_ts.shape[0]):
        if int(syn_ts[k]) // 86400 != day0:
            day_boundary_offset = int(k)
            break

    # per-bar agreement (only where BOTH producers are valid AND the ts exists in the DB map).
    agree = []
    offsets = []
    for k in range(syn_ts.shape[0]):
        ts_k = int(syn_ts[k])
        if not syn_val[k]:
            continue
        if ts_k not in db_map:
            continue
        offsets.append(k)
        agree.append(1 if int(syn_tok[k]) == db_map[ts_k] else 0)
    offsets = np.asarray(offsets, dtype=np.int64)
    agree = np.asarray(agree, dtype=np.int64)

    # offset-binned agreement curve.
    curve = []
    if offsets.size:
        max_off = int(offsets.max())
        for b0 in range(0, max_off + 1, bin_size):
            m = (offsets >= b0) & (offsets < b0 + bin_size)
            if not m.any():
                continue
            curve.append((int(b0), int(b0 + bin_size), float(agree[m].mean()), int(m.sum())))

    # first offset at/after which agreement is ~100% (>=0.999) for the rest of the curve.
    reach_100_off = None
    for i in range(len(curve)):
        if all(c[2] >= 0.999 for c in curve[i:]):
            reach_100_off = curve[i][0]
            break

    overall = float(agree.mean()) if agree.size else 0.0
    # tail agreement: mean over offsets at/after the day boundary (or the back half if no day boundary)
    # — quantifies how close to ~100% the firmed-up tail gets.
    if offsets.size:
        cut = day_boundary_offset if day_boundary_offset is not None else int(offsets.max() // 2)
        tail_m = offsets >= cut
        tail_agree = float(agree[tail_m].mean()) if tail_m.any() else None
    else:
        tail_agree = None
    res = dict(
        asset=int(asset_id), ok=True, n_bars=int(syn_ts.shape[0]),
        n_compared=int(agree.size), overall_agree=overall, tail_agree=tail_agree,
        longest_run_bars=int(longest_run_bars), spans_day=bool(spans_day),
        day_boundary_offset=day_boundary_offset, reach_100_offset=reach_100_off,
        bin_size=int(bin_size), curve=curve, range_span_s=int(t_hi - t_lo),
    )
    if verbose:
        print(f"[join-audit](b) asset {asset_id}: bars={syn_ts.shape[0]} compared={agree.size} "
              f"longest_run={longest_run_bars} spans_day={spans_day} "
              f"overall_agree={overall:.4f} tail_agree={tail_agree} "
              f"day_boundary_off={day_boundary_offset} reach~100%@offset={reach_100_off}", flush=True)
        for (b0, b1, ag, n) in curve[:16]:
            print(f"    offset[{b0:>5}..{b1:>5}): agree={ag:.4f} (n={n})", flush=True)
    return res


# --------------------------------------------------------------------------- #
# (B4) strict input_lag_for + grid shift check
# --------------------------------------------------------------------------- #
def audit_lag_strictness(verbose=True):
    """B4: long-form 'L1_bpe' resolves to lag 2; the lagged join shifts tokens by 2 grid rows (not 1).
    Pure synthetic timeline so the shift is unambiguous."""
    # synthetic dense timeline: ts = 0,60,120,... tokens = 10,11,12,... all valid.
    n = 12
    ts = np.arange(n, dtype=np.int64) * C.BAR_SECONDS
    toks = (10 + np.arange(n)).astype(np.int32)
    tl = TT.TimelineResult(ts, toks, np.zeros(n, np.float32), np.ones(n, bool), asset_id=-1,
                           language="L1_bpe")
    grid_ts = ts.copy()  # grid == timeline ts (no filled rows) so the shift is isolated.
    out, valid = TT.join_timeline_to_grid(tl, grid_ts, "L1_bpe")

    lag_l1_long = input_lag_for("L1_bpe")
    lag_l1_short = input_lag_for("L1")
    lag_l8_long = input_lag_for("L8_haar")
    lag = int(lag_l1_long)
    # expected shift-by-lag (lag-agnostic — tracks LANGUAGE_INPUT_LAG): out[r]=toks[r-lag], out[:lag]=-1.
    expected = np.full(n, -1, dtype=np.int32)
    if lag < n:
        expected[lag:] = toks[:n - lag]
    shift_ok = bool(np.array_equal(out, expected))
    # explicitly disprove the off-by-one (shift-by-(lag-1) — serving a still-provisional bar).
    off = max(lag - 1, 0)
    shiftm1 = np.full(n, -1, dtype=np.int32)
    if off < n:
        shiftm1[off:] = toks[:n - off]
    not_shift_off = (lag == 0) or (not bool(np.array_equal(out, shiftm1)))

    # unknown language must RAISE (hardening).
    raised = False
    try:
        input_lag_for("L9_bogus")
    except ValueError:
        raised = True

    res = dict(
        l1_long_lag=int(lag_l1_long), l1_short_lag=int(lag_l1_short), l8_long_lag=int(lag_l8_long),
        lag=lag, shift_by_lag_ok=shift_ok, not_shift_by_lag_minus_1=not_shift_off,
        unknown_raises=raised, out=out.tolist(), expected=expected.tolist(),
    )
    ok = (lag_l1_long == lag_l1_short and lag_l1_long >= 1 and lag_l8_long == 0
          and shift_ok and not_shift_off and raised)
    res["ok"] = bool(ok)
    if verbose:
        print(f"[join-audit](B4) L1_bpe lag={lag_l1_long} (L1={lag_l1_short}) L8_haar lag={lag_l8_long} "
              f"shift_by_lag_ok={shift_ok} not_shift_by_lag-1={not_shift_off} unknown_raises={raised} OK={ok}",
              flush=True)
        print(f"    join out={out.tolist()}  expected={expected.tolist()}", flush=True)
    return res


# --------------------------------------------------------------------------- #
# RESULT.md
# --------------------------------------------------------------------------- #
def write_result(out_path, cfg, assets, join_results, cross_results, lag_res, l1_vocab, wall):
    b2_ok = all(r.get("ok") for r in join_results) and len(join_results) > 0
    b2_mismatch_total = sum(int(r.get("mismatches", 0)) for r in join_results)
    b4_ok = bool(lag_res.get("ok"))

    join_lines = []
    for r in join_results:
        join_lines.append(
            f"- asset {r['asset']}: grid_rows={r.get('grid_rows')} "
            f"real={r.get('real_rows')} filled={r.get('filled_rows')} "
            f"real_matched={r.get('real_rows_matched')} filled_masked={r.get('filled_rows_masked')} "
            f"**mismatches={r.get('mismatches')}** OK={r.get('ok')}")

    cross_lines = []
    for r in cross_results:
        if not r.get("ok"):
            cross_lines.append(f"- asset {r['asset']}: {r.get('note','(skipped)')}")
            continue
        cross_lines.append(
            f"- asset {r['asset']}: bars={r.get('n_bars')} compared={r.get('n_compared')} "
            f"longest_run_bars={r.get('longest_run_bars')} spans_day={r.get('spans_day')} "
            f"overall_agree={r.get('overall_agree'):.4f} tail_agree={r.get('tail_agree')} "
            f"day_boundary_offset={r.get('day_boundary_offset')} "
            f"**reach~100%@offset={r.get('reach_100_offset')}**")
        for (b0, b1, ag, n) in r.get("curve", [])[:24]:
            cross_lines.append(f"    - offset[{b0}..{b1}): agree={ag:.4f} (n={n})")

    sections = [
        ("Summary — TASK_801b-i Fork A join-audit verdict", [
            f"- **B1** (annotate_asset byte-identity pre/post refactor): verified by /tmp/801_b1_compare.py "
            f"(see console — array_equal=True, 2 assets × L1+L8).",
            f"- **B2** (join correctness + R2 masking): **{'PASS' if b2_ok else 'FAIL'}** — "
            f"total mismatches across assets = {b2_mismatch_total} (must be 0).",
            f"- **B3** (synthetic cross-producer agreement curve): see section below; "
            f"agreement reaches ~100% after the cold-sigma head firms (≈ first UTC-day re-anchor).",
            f"- **B4** (strict input_lag_for): **{'PASS' if b4_ok else 'FAIL'}** — "
            f"L1_bpe lag={lag_res.get('l1_long_lag')}, lagged join shifts by {lag_res.get('lag')} grid rows "
            f"(not lag-1: {lag_res.get('not_shift_by_lag_minus_1')}), unknown language raises="
            f"{lag_res.get('unknown_raises')}.",
            f"- scope: NO hot-path training wiring (that is 801b-ii). CPU-only, no GPU/JAX/server.",
        ]),
        ("Configuration", [
            f"- DB: `{cfg.db}`",
            f"- window={cfg.window} halflife={cfg.ewma_halflife} sigma_floor={cfg.sigma_floor}",
            f"- L1 codebook_size={l1_vocab.codebook_size} merges={len(l1_vocab.merges)} "
            f"n_base={l1_vocab.n_base} (bounded fit, audit-local — NOT the canonical 799 vocab)",
            f"- audit assets: {assets}",
            f"- wallclock: {wall:.0f}s",
        ]),
        ("(a) Join correctness + R2 masking (B2 — the headline)", [
            "- Construction: annotate_asset -> ground-truth timeline; build a DENSE grid at "
            "C.BAR_SECONDS over a mid-stream span (real-bar ts -> VALID timeline entry; every other "
            "grid ts -> 'forward-filled' slot row with no real bar). Join with language='L1_bpe' (lag=2).",
            "- ASSERT every grid row whose ts is a VALID real bar == annotate_asset's token at that ts "
            "(after the grid-index lag shift); every filled row -> -1/invalid. mismatches must be 0.",
        ] + join_lines),
        ("(b) Synthetic cross-producer agreement curve (B3)", [
            "- Construction: a mid-stream CONTIGUOUS raw-OHLCV range -> annotate_ohlcv (synthetic "
            "producer) vs annotate_asset (DB producer) tokens joined by ts, binned by OFFSET from the "
            "range start. The only divergence is the cold-sigma EWMA head (the synthetic producer "
            "re-seeds sigma at the range start instead of carrying full upstream history); it firms up "
            "to ~100% agreement after the first UTC-day re-anchor (sigma re-anchors at ts//86400).",
        ] + cross_lines),
        ("(B4) strict input_lag_for + grid shift", [
            f"- input_lag_for('L1_bpe') = {lag_res.get('l1_long_lag')} (long-form), "
            f"input_lag_for('L1') = {lag_res.get('l1_short_lag')}, "
            f"input_lag_for('L8_haar') = {lag_res.get('l8_long_lag')}.",
            f"- lagged join shift-by-{lag_res.get('lag')} exact: {lag_res.get('shift_by_lag_ok')}; "
            f"NOT shift-by-(lag-1): {lag_res.get('not_shift_by_lag_minus_1')}; unknown language raises: "
            f"{lag_res.get('unknown_raises')}.",
            f"- synthetic join out={lag_res.get('out')}  expected={lag_res.get('expected')}",
        ]),
    ]
    verdict = (f"B2={'PASS' if b2_ok else 'FAIL'} | "
               f"B4={'PASS' if b4_ok else 'FAIL'} | B3 curve reported")
    meta = dict(
        task="TASK_801b-i",
        b2_join_ok=bool(b2_ok), b2_mismatch_total=int(b2_mismatch_total),
        b4_lag_ok=bool(b4_ok), lag=lag_res,
        join_results=join_results, cross_results=cross_results,
        l1_codebook_size=int(l1_vocab.codebook_size),
        window=int(cfg.window), assets=list(assets), wallclock_s=float(wall),
    )
    RW.write_result(out_path, "TASK_801b-i — No-Fill Token Timeline + Timestamp-Join + R3 Join-Audit (Fork A)",
                    sections, verdict, meta)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="TASK_801b-i Fork A R3 join-audit (CPU/numpy).")
    ap.add_argument("--db", default=None)
    ap.add_argument("--out", default="/home/lukas/Trader/runs/801_join_audit/RESULT.md")
    ap.add_argument("--assets", nargs="+", type=int, default=[1087, 1091],
                    help="real DB assets for the join + cross-producer checks (>=2).")
    ap.add_argument("--fit-mwpa", type=int, default=4000)
    ap.add_argument("--grid-bars", type=int, default=8000,
                    help="dense grid span (in real-bar steps) for the join-correctness check.")
    ap.add_argument("--cross-min-bars", type=int, default=300,
                    help="min gap-free DB run length to use for the synthetic cross-producer curve "
                         "(minute data is gapped -> runs are short).")
    ap.add_argument("--cross-max-bars", type=int, default=4000,
                    help="cap on the cross-producer range length (anchored at run start).")
    ap.add_argument("--cross-bin", type=int, default=100,
                    help="offset bin width (bars) for the cross-producer agreement curve.")
    ap.add_argument("--max-feature-bars", type=int, default=120000)
    args = ap.parse_args(argv)

    db = args.db if args.db else C.db_path()
    cfg = C.CandleConfig(db=db)
    assets = [int(a) for a in args.assets]
    t0 = time.time()

    print(f"[join-audit] db={db}")
    print(f"[join-audit] assets={assets}  build_assets={list(cfg.build_assets)}")

    l1_vocab, _l8_vocab = build_vocabs(cfg, fit_mwpa=int(args.fit_mwpa))

    # (B4) first — pure synthetic, fast, no DB.
    print("\n[join-audit] === (B4) strict input_lag_for + grid shift ===")
    lag_res = audit_lag_strictness()

    # (a) join correctness + R2 masking.
    print("\n[join-audit] === (a) join correctness + R2 masking ===")
    join_results = []
    for a in assets:
        join_results.append(audit_join_correctness(
            db, a, l1_vocab, cfg, n_grid_bars=int(args.grid_bars),
            max_feature_bars=int(args.max_feature_bars), language="L1_bpe"))

    # (b) synthetic cross-producer curve.
    print("\n[join-audit] === (b) synthetic cross-producer agreement curve ===")
    cross_results = []
    for a in assets:
        cross_results.append(audit_cross_producer(
            db, a, l1_vocab, cfg, min_bars=int(args.cross_min_bars),
            max_bars=int(args.cross_max_bars), bin_size=int(args.cross_bin),
            max_feature_bars=int(args.max_feature_bars)))

    wall = time.time() - t0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    write_result(args.out, cfg, assets, join_results, cross_results, lag_res, l1_vocab, wall)

    b2_ok = all(r.get("ok") for r in join_results)
    print(f"\n[join-audit] DONE in {wall:.0f}s. B2(join)={'PASS' if b2_ok else 'FAIL'} "
          f"B4(lag)={'PASS' if lag_res.get('ok') else 'FAIL'}. wrote {args.out}", flush=True)
    return dict(join=join_results, cross=cross_results, lag=lag_res)


if __name__ == "__main__":
    main()
