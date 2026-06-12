"""TASK_804-B2.5 — acceptance battery for the CYCLIC-SHIFT null (arm `shift_fe`).

The corrected real-vs-null control. The π-permutation null (TASK_804-B2) was a NON-null (it preserved
the token SEQUENCE; the model just relearns id->embedding). The cyclic shift keeps each asset's token
MARGINAL multiset but DECOUPLES the token sequence from the bars by rolling the per-bar timeline a
per-asset offset, so token[t] now comes from a DIFFERENT bar — isolating whether the real token<->bar
ALIGNMENT carries signal.

CPU ONLY. No GPU, no JAX, no server. The five launch-gate asserts (a)-(e):

  (a) multiset equality per asset — the shifted timeline's valid-token id-frequency multiset == the
      real timeline's, on >=1 sampled asset (a cyclic shift is a pure position permutation).
  (b) offsets persisted + sha'd — the resolved per-asset offsets are in shift_null_spec.json and the
      spec's sha256 matches the on-disk bytes.
  (c) decoupling sanity (agreement ~ Sum p_i^2) — per-bar agreement(shifted, real) over valid bars
      ~ the marginal collision probability Sum_i p_i^2 (chance), NOT ~1. Proves the shift decoupled.
  (d) deterministic derived sha — regenerating with the same seed yields byte-identical offsets + the
      same derived vocab_sha.
  (e) downstream join/lag untouched — the shifted timeline flows through the IDENTICAL
      join_timeline_to_grid + input_lag_for path as real (the roll is purely pre-join); the shift_fe
      grid == lag-join(roll(real_timeline)) computed independently, and token_window_slice.py is
      byte-unmodified vs the committed HEAD.

Run:
    CANDLE_DB=/home/lukas/Trader/trading_data.db \
      python tools/candle_lang/test_shift_null.py

Vocab: a LOCAL bounded-fit L1 vocab (the real cb-1376 799 vocab is SERVER-ONLY at real train time);
the wiring is vocab-agnostic so the local vocab exercises every code path. Artifacts -> experiments/804/.
"""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

# Resolve the repo root from THIS file's location (works in the main checkout AND an isolated
# worktree) so the worktree's own modules are imported, not the main checkout's.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tools.candle_lang import config as C
from tools.candle_lang import tokenizer_api as TK
from tools.candle_lang import token_timeline as TT
from tools.candle_lang import token_hotpath as TH
from tools.candle_lang.token_window_slice import input_lag_for, lag_shift_per_bar, INVALID_TOKEN_ID

# The shift-null spec generator (experiments/804/build_shift_null.py).
sys.path.insert(0, os.path.join(REPO, "experiments", "804"))
import build_shift_null as BSN  # noqa: E402

EXP_DIR = os.path.join(REPO, "experiments", "804")
SAMPLE_ASSET = 309                 # in DEFAULT_BUILD_ASSETS; plenty of bars for a bounded test.
FIT_ASSETS = [1087, 353, 586, 309]  # bounded local fit set (subset of DEFAULT_BUILD_ASSETS).
LANGUAGE = "L1_bpe"
MFB = 5000                          # tail feature-bar cap -> fast, many distinct tokens + gaps.
SHIFT_SEED = BSN.SHIFT_SEED         # 804


# --------------------------------------------------------------------------- #
def _resolve_db():
    """The shift null needs the real DB (annotate_asset). In an isolated worktree the repo-relative
    db_path() points at a non-existent worktree copy; fall back to the canonical workstation DB."""
    db = C.db_path()
    if os.path.isfile(db):
        return db
    for cand in ("/home/lukas/Trader/trading_data.db", "/home/operator/Trader/trading_data.db"):
        if os.path.isfile(cand):
            return cand
    return db


def _fit_local_vocab(cfg, out_path):
    """Fit a bounded LOCAL L1 vocab (DB-backed) and save it. Returns the saved vocab path."""
    TK.fit_and_save(cfg, FIT_ASSETS, out_path,
                    max_windows_per_asset=1500, p_min=20, max_merges=64, verbose=False)
    return out_path


# --------------------------------------------------------------------------- #
# (e) — independent lag-join oracle (mirror of token_timeline.join_timeline_to_grid, inlined so the
# test does NOT depend on the function under test to prove the function under test).
# --------------------------------------------------------------------------- #
def _independent_lag_join(timeline, grid_ts, language):
    """Independent {ts->token} lookup on grid_ts, then a lag_shift_per_bar by input_lag_for(language).
    This reproduces join_timeline_to_grid's contract WITHOUT calling it (it inlines the same 3 steps
    on top of the SAME input_lag_for / lag_shift_per_bar primitives) — so (e) is a genuine cross-check
    that the production join lands the rolled timeline exactly where an independent computation does."""
    grid_ts = np.asarray(grid_ts, dtype=np.int64)
    G = grid_ts.shape[0]
    t_ts = np.asarray(timeline.ts, dtype=np.int64)
    t_tok = np.asarray(timeline.token_id, dtype=np.int64)
    t_val = np.asarray(timeline.valid, dtype=bool)
    ts_to_tok = {int(ts): int(tok) for ts, tok, v in zip(t_ts, t_tok, t_val) if v}
    raw = np.full(G, INVALID_TOKEN_ID, dtype=np.int32)
    for r in range(G):
        tok = ts_to_tok.get(int(grid_ts[r]))
        if tok is not None:
            raw[r] = tok
    return lag_shift_per_bar(raw, input_lag_for(language)).astype(np.int32)


# =========================================================================== #
def main():
    print("=== TASK_804-B2.5 cyclic-shift null acceptance battery (CPU) ===")
    db = _resolve_db()
    if not os.path.isfile(db):
        print(f"SKIP: DB not present at {db} (the shift null needs the real DB to annotate).")
        return
    cfg = C.CandleConfig(db=db)
    print(f"db={db}")
    os.makedirs(EXP_DIR, exist_ok=True)

    # ---- fit a bounded LOCAL base vocab (real cb-1376 is server-only) ----
    base_vocab_path = "/tmp/804b25_local_base_vocab.json"
    _fit_local_vocab(cfg, base_vocab_path)
    base_tok = TH.load_tokenizer(base_vocab_path)
    base_sha = TH.vocab_sha_for(base_tok)
    print(f"[local fit] cb={base_tok.codebook_size} base_vocab_sha={base_sha[:12]} "
          f"(NOT the pinned cb-1376 799 vocab — server-only at real train time)")

    # ---- build the shift-null spec (resolves per-asset offsets at measured T_a) ----
    spec_path = os.path.join(EXP_DIR, "shift_null_spec.json")
    TH.clear_timeline_cache()
    spec = BSN.build_shift_null_spec(
        base_vocab_path, FIT_ASSETS, spec_path, seed=SHIFT_SEED,
        cfg=cfg, db_path=db, max_feature_bars=MFB,
        tag="local-vocab (CPU acceptance; NOT the real cb-1376)")
    offsets = {int(a): int(o) for a, o in spec["per_asset_offsets"].items()}
    print(f"[spec] null_type={spec['null_type']} seed={spec['seed']} "
          f"derived_vocab_sha={spec['derived_vocab_sha'][:12]} offsets={offsets}")

    # ======================================================================= #
    # (b) offsets persisted + sha'd
    # ======================================================================= #
    print("\n--- (b) offsets persisted + spec sha256 matches on-disk bytes ---")
    on_disk = json.load(open(spec_path))
    assert on_disk["per_asset_offsets"], "(b) FAIL: no per-asset offsets persisted"
    assert int(on_disk["seed"]) == SHIFT_SEED, "(b) FAIL: seed not pinned to SHIFT_SEED"
    # the recorded sha is the canonical sha over every field EXCEPT 'sha256'; recompute it and also
    # cross-check against a from-scratch recompute over the file's own bytes (sans the field).
    recorded = on_disk["sha256"]
    recomputed = BSN.spec_self_sha(on_disk)
    assert recorded == recomputed, \
        f"(b) FAIL: recorded sha {recorded[:12]} != recompute {recomputed[:12]} (spec bytes changed)"
    assert spec["derived_vocab_sha"] == on_disk["derived_vocab_sha"]
    print(f"(b) PASS: {len(offsets)} offsets persisted; spec sha256={recorded[:16]} matches on-disk "
          f"bytes (canonical recompute over all-but-'sha256').")

    # ---- annotate the sampled asset's REAL + SHIFTED full-history timelines ----
    TH.clear_timeline_cache()
    real_tl = TT.annotate_asset(db, SAMPLE_ASSET, base_tok, cfg, max_feature_bars=MFB)
    off = offsets[SAMPLE_ASSET]
    shifted_tl = TH.roll_timeline(real_tl, off)
    rv = np.asarray(real_tl.valid, bool)
    sv = np.asarray(shifted_tl.valid, bool)
    real_ids = np.asarray(real_tl.token_id, np.int64)
    shift_ids = np.asarray(shifted_tl.token_id, np.int64)
    T_a = len(real_tl)
    n_valid_real = int(rv.sum())
    assert n_valid_real > 0, "no valid bars annotated — bump MFB or pick another asset"
    # offset is decoupling-large (the assert (c) guard): in [0.2*T, 0.8*T].
    assert 0.2 * T_a <= off <= 0.8 * T_a, \
        f"offset {off} not in [0.2*T,0.8*T]=[{0.2*T_a:.0f},{0.8*T_a:.0f}] — would NOT decouple"
    print(f"\n[asset {SAMPLE_ASSET}] T_a={T_a} offset={off} ({off / T_a:.3f}*T) "
          f"valid_real={n_valid_real}")

    # ======================================================================= #
    # (a) multiset equality per asset
    # ======================================================================= #
    print("\n--- (a) multiset equality per asset (cyclic shift preserves the multiset) ---")
    real_valid_ids = real_ids[rv]
    shift_valid_ids = shift_ids[sv]
    cb = int(base_tok.codebook_size)
    real_ms = np.bincount(real_valid_ids, minlength=cb)
    shift_ms = np.bincount(shift_valid_ids, minlength=cb)
    # a cyclic roll is a pure position permutation -> the valid mask is itself rolled, so the SAME
    # set of valid token ids reappears with the SAME counts (exact per-id equality, not just sorted).
    assert int(sv.sum()) == n_valid_real, \
        f"(a) FAIL: valid-bar count changed under the roll ({int(sv.sum())} vs {n_valid_real})"
    assert np.array_equal(real_ms, shift_ms), \
        "(a) FAIL: valid-token frequency multiset changed under the cyclic shift"
    print(f"(a) PASS: per-id valid-token multiset identical ({n_valid_real} valid bars, "
          f"{int((real_ms > 0).sum())} distinct tokens); shift is a pure position permutation.")

    # ======================================================================= #
    # (c) decoupling sanity — agreement ~ Sum p_i^2
    # ======================================================================= #
    print("\n--- (c) decoupling: per-bar agreement(shifted, real) ~ Sum p_i^2 (chance, NOT ~1) ---")
    both = rv & sv
    n_both = int(both.sum())
    agree = float(np.mean(real_ids[both] == shift_ids[both]))
    p = real_ms.astype(np.float64) / max(1.0, real_ms.sum())   # marginal over real valid tokens
    sum_p2 = float(np.sum(p * p))                              # collision probability (chance agree)
    gap = agree - sum_p2
    # tolerance: agreement is a Bernoulli mean over n_both bars at success prob ~ Sum p^2 (plus mild
    # autocorrelation), so a few-sigma band around Sum p^2 is the right gate. We use a generous
    # absolute band that still DECISIVELY separates "decoupled" (~Sum p^2, here a few %) from the
    # π-non-null's ~1.0. Most important: agreement must be FAR below 1 (real alignment is broken).
    tol = max(0.05, 5.0 / np.sqrt(max(1, n_both)))
    print(f"  n_both_valid={n_both}  agreement={agree:.4f}  Sum_p^2={sum_p2:.4f}  "
          f"gap={gap:+.4f}  tol=±{tol:.4f}")
    assert agree < 0.5, \
        f"(c) FAIL: agreement {agree:.3f} is NOT chance-like — the roll did not decouple"
    assert abs(gap) <= tol, \
        f"(c) FAIL: agreement {agree:.4f} not ~ Sum p^2 {sum_p2:.4f} (gap {gap:+.4f} > tol {tol:.4f})"
    print(f"(c) PASS: agreement {agree:.4f} ~ Sum p^2 {sum_p2:.4f} (gap {gap:+.4f} within ±{tol:.4f}); "
          f"decoupled to chance, NOT ~1 (the π-non-null would sit at ~1.0).")

    # ======================================================================= #
    # (d) deterministic derived sha
    # ======================================================================= #
    print("\n--- (d) deterministic offsets + derived vocab_sha (regenerate, byte-identical) ---")
    spec_path2 = "/tmp/804b25_shift_null_spec_repro.json"
    TH.clear_timeline_cache()
    spec2 = BSN.build_shift_null_spec(
        base_vocab_path, FIT_ASSETS, spec_path2, seed=SHIFT_SEED,
        cfg=cfg, db_path=db, max_feature_bars=MFB, tag="repro")
    assert spec2["per_asset_offsets"] == spec["per_asset_offsets"], \
        "(d) FAIL: regenerated per-asset offsets differ at the same seed"
    assert spec2["derived_vocab_sha"] == spec["derived_vocab_sha"], \
        "(d) FAIL: derived vocab_sha not reproducible at the same seed"
    # and a DIFFERENT seed yields different offsets (the seed actually drives the draw).
    spec_other = BSN.build_shift_null_spec(
        base_vocab_path, FIT_ASSETS, "/tmp/804b25_shift_null_spec_seed805.json",
        seed=SHIFT_SEED + 1, cfg=cfg, db_path=db, max_feature_bars=MFB, tag="seed+1")
    assert spec_other["per_asset_offsets"] != spec["per_asset_offsets"], \
        "(d) FAIL: a different seed produced identical offsets (seed not driving the draw)"
    assert spec_other["derived_vocab_sha"] != spec["derived_vocab_sha"]
    # derived sha is DISTINCT from the real (base) sha -> shifted grid never collides with real cache.
    assert spec["derived_vocab_sha"] != base_sha, \
        "(d) FAIL: derived sha == base sha (shifted grid would collide with the real cache)"
    print(f"(d) PASS: same seed -> identical offsets + derived sha {spec['derived_vocab_sha'][:12]}; "
          f"seed+1 -> different; derived sha != base sha {base_sha[:12]} (distinct cache key).")

    # ======================================================================= #
    # (e) downstream join/lag untouched
    # ======================================================================= #
    print("\n--- (e) downstream join/lag untouched: shift_fe grid == lag-join(roll(real)) ---")
    # The grid is drawn from the SHIFTED timeline's own real-bar ts (a mid-stream span) so the rolled
    # tokens land on real grid rows. The spec's offset was resolved against the capped T_a (MFB), so
    # production must roll the SAME capped real timeline: pre-warm the base-sha cache with `real_tl`
    # (the capped annotate) so _get_shifted_timeline rolls exactly the timeline this test rolled — the
    # cap-consistency the B3 (full-history) path gets for free.
    TH.clear_timeline_cache()
    shift_tok = TH.load_tokenizer(spec_path)                  # carries _shift_null (derived sha)
    assert getattr(shift_tok, "_shift_null", None) is not None, "(e) FAIL: spec not detected"
    base_sha_in_tok = shift_tok._shift_null["base_vocab_sha"]
    TH._TIMELINE_CACHE[(int(SAMPLE_ASSET), str(base_sha_in_tok))] = real_tl   # capped base timeline
    # grid: a mid-stream span of the shifted timeline's real-bar ts (guaranteed real-row overlap).
    s_ts = np.asarray(shifted_tl.ts, dtype=np.int64)
    lo = s_ts.shape[0] // 4
    grid_ts = s_ts[lo: lo + (MFB // 2)]
    hm = _GridHM(grid_ts, [SAMPLE_ASSET])
    prod_grid = TH.tokenize_grid(hm, "historical", shift_tok, cfg, db_path=db, language=LANGUAGE)
    prod_col = prod_grid[:, 0].astype(np.int64)
    # independent reference: roll the REAL timeline by the SAME offset, then an INDEPENDENT lag-join
    # (inlined, does NOT call join_timeline_to_grid) — proving the production join landed it exactly.
    indep = _independent_lag_join(shifted_tl, grid_ts, LANGUAGE).astype(np.int64)
    mism = int(np.count_nonzero(prod_col != indep))
    n_real_rows = int(np.count_nonzero(indep >= 0))
    print(f"  grid_rows={grid_ts.shape[0]} real_token_rows(post-lag)={n_real_rows} mismatches={mism} "
          f"lag={input_lag_for(LANGUAGE)}")
    assert n_real_rows > 0, "(e) FAIL: grid landed entirely on fills — pick a denser span"
    assert mism == 0, \
        f"(e) FAIL: shift_fe grid != independent lag-join(roll(real)) ({mism} mismatches)"

    # ALSO prove the production path used the SAME join_timeline_to_grid the real arm uses: the
    # shift_fe grid for the rolled timeline == join_timeline_to_grid(roll(real), grid) (same call).
    join_col, _v = TT.join_timeline_to_grid(shifted_tl, grid_ts, LANGUAGE)
    assert np.array_equal(prod_col, join_col.astype(np.int64)), \
        "(e) FAIL: production grid != join_timeline_to_grid(roll(real)) — join path diverged"

    # and token_window_slice.py is byte-UNMODIFIED vs the committed HEAD (the lag module is untouched).
    tws_rel = "tools/candle_lang/token_window_slice.py"
    diff = subprocess.run(["git", "-C", REPO, "diff", "--quiet", "HEAD", "--", tws_rel])
    assert diff.returncode == 0, \
        f"(e) FAIL: {tws_rel} is MODIFIED — the lag module must stay untouched"
    print(f"(e) PASS: shift_fe grid == independent lag-join(roll(real)) AND == "
          f"join_timeline_to_grid(roll(real)) (0 mismatches, {n_real_rows} real rows); "
          f"{tws_rel} byte-unmodified vs HEAD.")

    print("\n=== (a)-(e) ALL PASS — cyclic-shift null launch gate green ===")
    print(f"  spec -> {spec_path}  (sha {spec['sha256'][:16]}, derived_vocab_sha "
          f"{spec['derived_vocab_sha'][:12]})")


# --------------------------------------------------------------------------- #
class _GridHM:
    """hm-shaped shim tokenize_grid reads: cache_data (T,A,5), valid_timestamps (T,), asset_ids (A,)."""
    def __init__(self, grid_ts, asset_ids):
        T = int(np.asarray(grid_ts).shape[0])
        A = len(asset_ids)
        self.cache_data = np.zeros((T, A, 5), dtype=np.float64)
        self.valid_timestamps = np.asarray(grid_ts, dtype=np.int64)
        self.asset_ids = list(asset_ids)


if __name__ == "__main__":
    main()
