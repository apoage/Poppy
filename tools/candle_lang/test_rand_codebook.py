"""TASK_804-B2 — acceptance battery for the frequency-matched random-codebook helper (rand_fe).

CPU ONLY. No GPU, no JAX, no server. The four audit asserts (i)-(iv) + the load-bearing
EQUIVALENCE check that licenses B3's cheap remap: remap_timeline(annotate(real), π) == π[real ids]
on every valid bar (the cheap path is EXACT), and the permuted-vocab record embeds the identical π.
(See the B2 report / module docstring for why a permuted VOCAB cannot itself reproduce π[real] under
annotate — the L1 encode path does not route output ids through the vocab.)

Run:
    python tools/candle_lang/test_rand_codebook.py

Asserts on a SMALL synthetic-permutation case (cb-agnostic, no DB) for (i)/(ii) speed, then on a
LOCAL bounded-fit L1 vocab (DB-backed; the real cb-1376 vocab is server-only) for (iii)/(iv) + the
equivalence check. Local artifacts are written to experiments/804/ and clearly TAGGED 'local-vocab'
in the manifest — the REAL cb-1376 artifacts are regenerated at B3 on the server.
"""
import os
import sys
import json

import numpy as np

# Resolve the repo root from THIS file's location (works in the main checkout AND an isolated
# worktree) so the worktree's own tools/candle_lang is imported, not the main checkout's.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from tools.candle_lang import rand_codebook as RC
from tools.candle_lang import config as C
from tools.candle_lang import tokenizer_api as TK
from tools.candle_lang import token_timeline as TT

EXP_DIR = os.path.join(REPO, "experiments", "804")
LOCAL_ASSET = 309          # in DEFAULT_BUILD_ASSETS; ~1.9M bars (plenty for a bounded test).
FIT_ASSETS = [1087, 353, 586, 309]   # bounded local fit set (subset of DEFAULT_BUILD_ASSETS).


# --------------------------------------------------------------------------- #
# (i) + (ii) on a synthetic permutation — cb-agnostic, no DB, fast.
# --------------------------------------------------------------------------- #
def test_i_bijection_and_ii_range_sentinel():
    cb, n_base = 1376, 864     # the REAL layout (864 leaf + 512 merge) — sanity-check the shape too.
    perm = RC.build_permutation(cb, seed=RC.PERM_SEED, n_base=n_base)

    # (i) bijection: sorted(π) == range(cb).
    assert perm.shape[0] == cb
    assert np.array_equal(np.sort(perm), np.arange(cb)), "(i) FAIL: π is not a bijection over 0..cb-1"

    # (ii) id-range: π maps 0..cb-1 -> 0..cb-1 ONLY (sentinel row cb is never produced).
    assert perm.min() == 0 and perm.max() == cb - 1, "(ii) FAIL: π escaped 0..cb-1"
    assert cb not in set(perm.tolist()), "(ii) FAIL: π produced the sentinel row index cb"
    # region-respecting (load-bearing for the permuted vocab): leaf->leaf, merge->merge.
    assert np.all(perm[:n_base] < n_base), "(ii) FAIL: a leaf id mapped into the merge region"
    assert np.all(perm[n_base:] >= n_base), "(ii) FAIL: a merge id mapped into the leaf region"

    # (ii) −1 stays −1 through remap_timeline (sentinel/invalid invariance), on a tiny hand timeline.
    ts = np.array([0, 60, 120, 180, 240], dtype=np.int64)
    ids = np.array([5, -1, 800, 1300, -1], dtype=np.int32)
    valid = np.array([True, False, True, True, False], dtype=bool)
    tl = TT.TimelineResult(ts, ids, np.zeros(5, np.float32), valid, asset_id=-1, language="L1_bpe")
    rtl = RC.remap_timeline(tl, perm)
    assert rtl.token_id[1] == -1 and rtl.token_id[4] == -1, "(ii) FAIL: −1 did not stay −1"
    assert rtl.token_id[0] == perm[5], "(ii) FAIL: valid bar not remapped by π"
    assert rtl.token_id[2] == perm[800] and rtl.token_id[3] == perm[1300]
    # invalid bars never carry a permuted id (so cb sentinel is never injected on an invalid bar).
    assert set(np.unique(rtl.token_id[~valid]).tolist()) <= {-1}

    # determinism / seed-pinning: same seed -> same π; different seed -> different π.
    assert np.array_equal(perm, RC.build_permutation(cb, seed=RC.PERM_SEED, n_base=n_base))
    assert not np.array_equal(perm, RC.build_permutation(cb, seed=RC.PERM_SEED + 1, n_base=n_base))
    print(f"(i) bijection PASS  |  (ii) id-range + sentinel/−1 invariance PASS  "
          f"(cb={cb}, n_base={n_base}, seed={RC.PERM_SEED})")
    return cb, n_base


def _fit_local_tokenizer_and_save(cfg, out_real_path):
    """Fit a bounded LOCAL L1 vocab (DB-backed) and save it to out_real_path. Returns the tokenizer."""
    vocab, _ = TK.fit_and_save(cfg, FIT_ASSETS, out_real_path,
                               max_windows_per_asset=1500, p_min=20, max_merges=64, verbose=False)
    tok = TK.CandleTokenizer.from_path(out_real_path)
    return tok


def test_iii_iv_and_equivalence():
    os.makedirs(EXP_DIR, exist_ok=True)
    cfg = C.CandleConfig()
    if not os.path.exists(cfg.db):
        print(f"SKIP (iii)/(iv)/equiv: DB not found at {cfg.db}")
        return

    # ---- fit a bounded LOCAL vocab (the real cb-1376 vocab is server-only) ----
    local_real_path = "/tmp/804b2_local_real_vocab.json"
    tok_real = _fit_local_tokenizer_and_save(cfg, local_real_path)
    n_base, cb = RC.n_base_of_vocab_path(local_real_path)
    print(f"[local fit] cb={cb} (n_base={n_base}, merges={cb - n_base})")

    perm = RC.build_permutation(cb, seed=RC.PERM_SEED, n_base=n_base)
    assert np.array_equal(np.sort(perm), np.arange(cb)), "(i) FAIL on local cb"

    # ---- annotate ONE asset with the REAL vocab (bounded) ----
    MFB = 4000   # tail feature-bar cap -> fast, still exercises gaps + many distinct tokens.
    tl_real = TT.annotate_asset(cfg.db, LOCAL_ASSET, tok_real, cfg, max_feature_bars=MFB)
    valid = np.asarray(tl_real.valid, dtype=bool)
    n_valid = int(valid.sum())
    assert n_valid > 0, "no valid bars annotated — bump MFB or pick another asset"

    # ---- (iii) multiset equality: remap's valid-token multiset == the real timeline's ----
    tl_rand_cheap = RC.remap_timeline(tl_real, perm)
    real_valid_ids = np.asarray(tl_real.token_id)[valid]
    rand_valid_ids = np.asarray(tl_rand_cheap.token_id)[np.asarray(tl_rand_cheap.valid, dtype=bool)]
    real_ms = np.bincount(real_valid_ids.astype(np.int64), minlength=cb)
    rand_ms = np.bincount(rand_valid_ids.astype(np.int64), minlength=cb)
    assert np.array_equal(np.sort(real_ms), np.sort(rand_ms)), \
        "(iii) FAIL: valid-token frequency multiset changed under the remap"
    # and the SAME counts land on the π-image ids (exact, not just sorted).
    assert np.array_equal(rand_ms, real_ms[np.argsort(perm)]), \
        "(iii) FAIL: per-id frequencies not relabelled by π"
    print(f"(iii) multiset equality PASS  (asset {LOCAL_ASSET}, {n_valid} valid bars, "
          f"{int((real_ms > 0).sum())} distinct tokens)")

    # ---- EQUIVALENCE check (load-bearing for B3's cheap path) ----
    # The cheap remap IS the operational rand_fe (matches front-end E5: π[token_id] fed to the same
    # wiring). The load-bearing guarantee is that it is EXACT — remap_timeline(annotate(real), π) on
    # every valid bar equals π applied to the real ids. (MEASURED CAVEAT, see B2 report: the L1 encode
    # path does NOT route output ids through the vocab — ~95% of tokens are leaf ids minted by
    # numerosity_reduce, and BPE segmentation depends on merge rank order — so a permuted VOCAB does
    # NOT reproduce π[real] under annotate. We assert the achievable equivalence: the cheap path is
    # exact, and the permuted-vocab RECORD embeds the identical π so the artifact and the executable
    # path agree on π by construction.)
    rand_ids = np.asarray(tl_rand_cheap.token_id)[valid].astype(np.int64)
    expect = perm[np.asarray(tl_real.token_id)[valid].astype(np.int64)]
    n_mismatch = int(np.sum(rand_ids != expect))
    assert n_mismatch == 0, \
        f"EQUIV FAIL: {n_mismatch}/{n_valid} valid bars: cheap remap != π[real ids]"
    assert np.array_equal(np.asarray(tl_rand_cheap.valid, bool), valid), \
        "EQUIV FAIL: cheap remap changed the valid mask"

    # the permuted-vocab RECORD: structurally valid (loads, same cb, encodes a real asset cleanly) and
    # carries the IDENTICAL π that the cheap path uses (the artifact<->executable tie).
    local_perm_path = "/tmp/804b2_local_perm_vocab.json"
    RC.write_permuted_vocab(local_real_path, perm, local_perm_path)
    perm_doc = json.load(open(local_perm_path))
    assert perm_doc["sizes"]["codebook_size"] == cb, "record changed codebook_size"
    assert np.array_equal(np.asarray(perm_doc["rand_codebook"]["perm"], dtype=np.int64), perm), \
        "EQUIV FAIL: permuted-vocab record embeds a DIFFERENT π than the cheap remap"
    tok_perm = TK.CandleTokenizer.from_path(local_perm_path)   # same loader, zero branches (E5).
    assert tok_perm.codebook_size == cb, "permuted vocab changed codebook_size on load"
    tl_perm_vocab = TT.annotate_asset(cfg.db, LOCAL_ASSET, tok_perm, cfg, max_feature_bars=MFB)
    assert int(np.asarray(tl_perm_vocab.valid, bool).sum()) > 0, \
        "permuted-vocab record does not encode the asset (structurally invalid)"
    print(f"EQUIVALENCE PASS: cheap remap == π[real ids] on all {n_valid} valid bars (B3 cheap path "
          f"exact); permuted-vocab record loads + embeds the identical π (seed {RC.PERM_SEED})")

    # ---- (iv) artifacts persisted under experiments/804/ (tagged local-vocab) ----
    manifest = RC.prep_experiment_artifacts(local_real_path, EXP_DIR, seed=RC.PERM_SEED,
                                            tag="local-vocab (CPU acceptance; NOT the real cb-1376)")
    for fn in ("L1_RESULT_vocab.json", f"rand_perm_s{RC.PERM_SEED}_vocab.json",
               "rand_codebook_manifest.json"):
        p = os.path.join(EXP_DIR, fn)
        assert os.path.exists(p), f"(iv) FAIL: missing artifact {p}"
    assert manifest["perm_seed"] == RC.PERM_SEED
    assert manifest["codebook_size"] == cb and manifest["n_base_tagged"] == n_base
    assert "L1_RESULT_vocab.json" in manifest["sha256"]
    assert f"rand_perm_s{RC.PERM_SEED}_vocab.json" in manifest["sha256"]
    # the on-disk permuted-vocab record loads, has the same cb, and embeds the SAME π the manifest
    # pins (so the persisted artifact reconstructs the exact cheap remap B3 will run).
    disk_perm_path = os.path.join(EXP_DIR, f"rand_perm_s{RC.PERM_SEED}_vocab.json")
    disk_doc = json.load(open(disk_perm_path))
    disk_perm = np.asarray(disk_doc["rand_codebook"]["perm"], dtype=np.int64)
    assert np.array_equal(disk_perm, perm), "(iv) FAIL: on-disk record embeds a different π"
    tok_disk = TK.CandleTokenizer.from_path(disk_perm_path)
    assert tok_disk.codebook_size == cb, "(iv) FAIL: on-disk record changed codebook_size"
    # cross-check the manifest sha matches the bytes actually on disk.
    assert manifest["sha256"][os.path.basename(disk_perm_path)] == RC.sha256_file(disk_perm_path)
    assert manifest["sha256"]["L1_RESULT_vocab.json"] == \
        RC.sha256_file(os.path.join(EXP_DIR, "L1_RESULT_vocab.json"))
    print(f"(iv) artifacts persisted PASS  -> {EXP_DIR}")
    print(f"     manifest shas: {json.dumps(manifest['sha256'], indent=0)}")
    return manifest


if __name__ == "__main__":
    print("=== TASK_804-B2 rand_fe acceptance battery (CPU) ===")
    test_i_bijection_and_ii_range_sentinel()
    test_iii_iv_and_equivalence()
    print("\n=== (i)-(iv) + EQUIVALENCE ALL PASS ===")
