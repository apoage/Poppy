#!/usr/bin/env python3
"""prebuild_token_cache.py — TASK_804 standalone token-timeline cache pre-builder.

WHY THIS EXISTS
---------------
The token front-end (real_fe / shift_fe arms) annotates each asset's FULL DB history ONCE per pool
(_get_cached_timeline). For the B3 universe (64 long-history assets, cb-1376 vocab) the FIRST slot's
cold annotate takes ~47 min — long enough that the orchestrator's stale-log watchdog kills the run
before it ever trains. The fix (already wired in the hot path) is a one-time DISK ARTIFACT keyed by
vocab_sha: if `experiments/804/token_cache/timelines_<sha>.npz` exists, pool init LOADS every per-asset
timeline into RAM (load_disk_artifact) and skips all annotation. This script BUILDS that artifact
out-of-band (before launch), so the training run only ever LOADS.

This is a DRIVER ONLY. It reuses the EXISTING hot-path mechanism verbatim:
  * load the real tokenizer (token_hotpath.load_tokenizer) -> real_sha = vocab_sha_for(tok)
  * _get_cached_timeline(aid, real_sha, ...) per asset       (populates the module RAM cache)
  * write_disk_artifact(cache_dir, real_sha, ids)            -> the REAL artifact (serves real_fe)
  * for shift_fe: load_tokenizer(spec) -> shift tokenizer; _get_shifted_timeline(aid, ...) per asset
    (rolls the cached BASE timeline, caches under the DERIVED sha); write_disk_artifact(...,derived_sha)
NOTHING in token_hotpath.py / token_timeline.py / the trainer is touched.

CORRECTNESS GUARD (--verify): the artifact is a TRANSPARENT cache. After writing, clear the RAM cache,
load the artifact from disk, and assert the loaded per-asset timeline is BYTE-IDENTICAL to a FRESH cold
annotate (token_id / valid / ts via np.array_equal) for a sample of assets — the same "disk-artifact
PASS" invariant the C-cache test (tools/candle_lang/test_token_hotpath.py) proves. The shift artifact is
checked against roll_timeline(fresh_base_annotate, offset) for the spec's per-asset offset. Any mismatch
FAILS LOUDLY: load-from-disk MUST equal cold-annotate or the cache is not transparent.

ASSET UNIVERSE
--------------
Pass `--assets-file` (one DB asset id per line; on the server this is /tmp/b3_asset_ids.txt) or
`--assets` (comma-separated). Default: the loaded base vocab's CandleConfig.build_assets (small, for
local mechanism tests only — NOT the 64-asset B3 universe).

LOCAL ACCEPTANCE: `--max-feature-bars N` caps each annotate to the most-recent N bars (the tail-keep the
smoke path uses) so the mechanism runs in seconds on the workstation's multi-million-row assets. The cap
applies IDENTICALLY to the prebuild and to the verify re-annotate, so byte-identity holds. OMIT the cap
at B3 (full history — that is the artifact the real run consumes). CPU ONLY; no GPU, no model.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional, Sequence

import numpy as np

# Resolve the repo root from THIS file (works in the main checkout AND an isolated worktree) so the
# worktree's own tools/candle_lang is imported, mirroring build_shift_null.py.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tools.candle_lang import config as C
from tools.candle_lang import token_hotpath as TH
from tools.candle_lang import token_timeline as TT


# --------------------------------------------------------------------------- #
# asset universe resolution
# --------------------------------------------------------------------------- #
def resolve_assets(args, base_tok) -> List[int]:
    """Resolve the DB asset universe to pre-build, in priority order:
      1. --assets-file  : one int asset id per line (blank lines / '#' comments ignored). On the
                          server this is /tmp/b3_asset_ids.txt (the 64-asset B3 universe).
      2. --assets       : comma-separated ids (quick local override).
      3. base vocab's CandleConfig.build_assets (small; local mechanism tests only).
    De-duplicated, negatives dropped, SORTED (write_disk_artifact sorts ids anyway; sort here so the
    progress log and the artifact agree)."""
    if args.assets_file:
        ids: List[int] = []
        with open(args.assets_file) as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                ids.append(int(line))
        src = f"--assets-file {args.assets_file}"
    elif args.assets.strip():
        ids = [int(x) for x in args.assets.split(",") if x.strip()]
        src = "--assets"
    else:
        ids = [int(a) for a in base_tok.cfg.build_assets]
        src = "base vocab CandleConfig.build_assets (LOCAL DEFAULT — not the B3 universe)"
    ids = sorted({int(a) for a in ids if int(a) >= 0})
    print(f"[assets] {len(ids)} ids from {src}: {ids if len(ids) <= 16 else ids[:16] + ['...']}",
          flush=True)
    return ids


# --------------------------------------------------------------------------- #
# annotate-into-cache helpers
# --------------------------------------------------------------------------- #
def _annotate_into_cache(asset_id: int, vocab_sha: str, db: str, tok, cfg,
                         max_feature_bars: Optional[int]) -> "TT.TimelineResult":
    """Annotate ONE asset under the BASE vocab into the module RAM cache, keyed by (asset_id, sha).

    For the REAL (uncapped) run this IS _get_cached_timeline — the exact hot-path call the pool makes,
    so the served timeline is the cached timeline. For the LOCAL ACCEPTANCE path (max_feature_bars set)
    _get_cached_timeline can't take a cap (it always full-annotates), so we annotate directly with the
    cap and stash it under the SAME key the hot path / write_disk_artifact read. The cap is a test-speed
    knob only; OMITTED at B3."""
    key = (int(asset_id), str(vocab_sha))
    cached = TH._TIMELINE_CACHE.get(key)
    if cached is not None:
        return cached
    if max_feature_bars is None:
        return TH._get_cached_timeline(int(asset_id), vocab_sha, db, tok, cfg)
    tl = TT.annotate_asset(db, int(asset_id), tok, cfg, max_feature_bars=int(max_feature_bars))
    TH._TIMELINE_CACHE[key] = tl
    return tl


# --------------------------------------------------------------------------- #
# real artifact
# --------------------------------------------------------------------------- #
def build_real_artifact(base_vocab: str, ids: Sequence[int], cache_dir: str,
                        cfg=None, db=None, max_feature_bars: Optional[int] = None,
                        log_every: int = 4) -> tuple:
    """Annotate every asset under the REAL base vocab into RAM and write the real disk artifact.

    Returns (real_sha, artifact_path, n_assets). Progress is logged every `log_every` assets so a
    human / the watchdog sees the build advancing (the whole point — no silent 47-min stretch)."""
    tok = TH.load_tokenizer(base_vocab)
    cfg = cfg if cfg is not None else tok.cfg
    db = db if db is not None else cfg.db
    real_sha = TH.vocab_sha_for(tok)
    print(f"[real] base vocab {base_vocab}", flush=True)
    print(f"[real] real_sha={real_sha}  cb={tok.codebook_size}  db={db}", flush=True)
    if max_feature_bars is not None:
        print(f"[real] NOTE max_feature_bars={max_feature_bars} (LOCAL ACCEPTANCE cap — "
              f"OMIT at B3 for full history)", flush=True)

    n = len(ids)
    t0 = time.time()
    for i, aid in enumerate(ids, 1):
        ta = time.time()
        _annotate_into_cache(aid, real_sha, db, tok, cfg, max_feature_bars)
        if i % log_every == 0 or i == n:
            el = time.time() - t0
            rate = el / i
            eta = rate * (n - i)
            print(f"[real] annotated {i}/{n} (asset {aid}, {time.time()-ta:.1f}s; "
                  f"elapsed {el/60:.1f}m, eta {eta/60:.1f}m)", flush=True)

    path = TH.write_disk_artifact(cache_dir, real_sha, ids)
    if not path:
        raise RuntimeError("build_real_artifact: write_disk_artifact returned None "
                           "(nothing cached / cache_dir empty)")
    print(f"[real] WROTE {path}  ({n} assets, sha {real_sha})", flush=True)
    return real_sha, path, n


# --------------------------------------------------------------------------- #
# shift artifact
# --------------------------------------------------------------------------- #
def build_shift_artifact(shift_spec: str, ids: Sequence[int], cache_dir: str,
                         cfg=None, db=None, max_feature_bars: Optional[int] = None,
                         log_every: int = 4) -> tuple:
    """Build the SHIFT (cyclic-shift null) artifact: load the shift tokenizer (load_tokenizer detects
    null_type==cyclic_shift), then for every asset roll the cached BASE timeline and cache it under the
    DERIVED sha (_get_shifted_timeline). Write the derived-sha artifact (serves shift_fe).

    The base annotate reuses the REAL cache entry (base_sha) populated by build_real_artifact — so under
    the uncapped B3 path NO asset is annotated twice. Under the LOCAL cap we pre-warm the base cache with
    the cap (so the rolled timeline matches the capped fresh-annotate the verifier compares against).

    Returns (derived_sha, base_sha, artifact_path, n_assets)."""
    shift_tok = TH.load_tokenizer(shift_spec)
    sn = getattr(shift_tok, "_shift_null", None)
    if sn is None:
        raise ValueError(f"--shift-spec {shift_spec} is not a cyclic_shift null spec "
                         f"(load_tokenizer returned a plain tokenizer)")
    cfg = cfg if cfg is not None else shift_tok.cfg
    db = db if db is not None else cfg.db
    base_sha = sn["base_vocab_sha"]
    derived_sha = sn["derived_vocab_sha"]
    print(f"[shift] spec {shift_spec}", flush=True)
    print(f"[shift] base_sha={base_sha}  derived_sha={derived_sha}", flush=True)

    n = len(ids)
    t0 = time.time()
    for i, aid in enumerate(ids, 1):
        # Ensure the BASE timeline is in the cache under base_sha so _get_shifted_timeline reuses it.
        # Under the cap this also guarantees the base cache holds the CAPPED timeline (matches verify).
        _annotate_into_cache(aid, base_sha, db, shift_tok, cfg, max_feature_bars)
        TH._get_shifted_timeline(int(aid), shift_tok, db, cfg)
        if i % log_every == 0 or i == n:
            el = time.time() - t0
            print(f"[shift] rolled {i}/{n} (asset {aid}; elapsed {el/60:.1f}m)", flush=True)

    path = TH.write_disk_artifact(cache_dir, derived_sha, ids)
    if not path:
        raise RuntimeError("build_shift_artifact: write_disk_artifact returned None")
    print(f"[shift] WROTE {path}  ({n} assets, derived sha {derived_sha})", flush=True)
    return derived_sha, base_sha, path, n


# --------------------------------------------------------------------------- #
# verification (the hard correctness guard)
# --------------------------------------------------------------------------- #
def _tl_equal(a: "TT.TimelineResult", b: "TT.TimelineResult") -> bool:
    """Byte-identity of two timelines on the grid-relevant arrays (token_id, valid, ts)."""
    return (np.array_equal(np.asarray(a.token_id), np.asarray(b.token_id))
            and np.array_equal(np.asarray(a.valid), np.asarray(b.valid))
            and np.array_equal(np.asarray(a.ts), np.asarray(b.ts)))


def verify_real(base_vocab: str, ids: Sequence[int], cache_dir: str, sample: Sequence[int],
                cfg=None, db=None, max_feature_bars: Optional[int] = None) -> None:
    """Assert the REAL artifact is a transparent cache: clear RAM, load the artifact from disk, and for
    each sampled asset assert loaded == a FRESH cold annotate (token_id/valid/ts byte-identical).
    Mirrors the test_token_hotpath disk-artifact PASS. FAILS LOUDLY on any mismatch."""
    tok = TH.load_tokenizer(base_vocab)
    cfg = cfg if cfg is not None else tok.cfg
    db = db if db is not None else cfg.db
    real_sha = TH.vocab_sha_for(tok)

    TH.clear_timeline_cache()
    n_loaded = TH.load_disk_artifact(cache_dir, real_sha)
    print(f"[verify-real] loaded {n_loaded} timelines from "
          f"{TH._artifact_path(cache_dir, real_sha)} (expect {len(ids)})", flush=True)
    if n_loaded != len(set(int(a) for a in ids)):
        raise AssertionError(f"[verify-real] artifact holds {n_loaded} assets, expected {len(ids)}")

    for aid in sample:
        loaded = TH._TIMELINE_CACHE.get((int(aid), str(real_sha)))
        if loaded is None:
            raise AssertionError(f"[verify-real] asset {aid} missing from loaded artifact")
        # FRESH cold annotate (NOT via the cache) — the independent ground truth.
        fresh = TT.annotate_asset(db, int(aid), tok, cfg,
                                  max_feature_bars=max_feature_bars) if max_feature_bars is not None \
            else TT.annotate_asset(db, int(aid), tok, cfg)
        ok = _tl_equal(loaded, fresh)
        print(f"[verify-real] asset {aid}: byte-identical(token_id,valid,ts) = {ok} "
              f"(T={len(loaded)})", flush=True)
        if not ok:
            raise AssertionError(f"[verify-real] asset {aid} loaded artifact != fresh annotate — "
                                 f"the cache is NOT transparent")
    print(f"[verify-real] PASS: disk artifact load == cold annotate, {len(sample)} sample assets "
          f"byte-identical.", flush=True)


def verify_shift(shift_spec: str, ids: Sequence[int], cache_dir: str, sample: Sequence[int],
                 cfg=None, db=None, max_feature_bars: Optional[int] = None) -> None:
    """Assert the SHIFT artifact is transparent: clear RAM, load the derived-sha artifact from disk, and
    for each sampled asset assert loaded == roll_timeline(FRESH base annotate, spec offset). FAILS
    LOUDLY on any mismatch."""
    shift_tok = TH.load_tokenizer(shift_spec)
    sn = shift_tok._shift_null
    cfg = cfg if cfg is not None else shift_tok.cfg
    db = db if db is not None else cfg.db
    derived_sha = sn["derived_vocab_sha"]
    offsets = sn["offsets"]

    TH.clear_timeline_cache()
    n_loaded = TH.load_disk_artifact(cache_dir, derived_sha)
    print(f"[verify-shift] loaded {n_loaded} timelines from "
          f"{TH._artifact_path(cache_dir, derived_sha)} (expect {len(ids)})", flush=True)
    if n_loaded != len(set(int(a) for a in ids)):
        raise AssertionError(f"[verify-shift] artifact holds {n_loaded} assets, expected {len(ids)}")

    for aid in sample:
        loaded = TH._TIMELINE_CACHE.get((int(aid), str(derived_sha)))
        if loaded is None:
            raise AssertionError(f"[verify-shift] asset {aid} missing from loaded artifact")
        # FRESH base annotate, then roll by the spec's persisted offset — the independent ground truth.
        fresh_base = TT.annotate_asset(db, int(aid), shift_tok, cfg,
                                       max_feature_bars=max_feature_bars) \
            if max_feature_bars is not None else TT.annotate_asset(db, int(aid), shift_tok, cfg)
        off = int(offsets.get(int(aid), 0))
        expected = TH.roll_timeline(fresh_base, off)
        ok = _tl_equal(loaded, expected)
        print(f"[verify-shift] asset {aid}: byte-identical(token_id,valid,ts) = {ok} "
              f"(offset {off}, T={len(loaded)})", flush=True)
        if not ok:
            raise AssertionError(f"[verify-shift] asset {aid} loaded artifact != roll(fresh base, "
                                 f"{off}) — the shift cache is NOT transparent")
    print(f"[verify-shift] PASS: disk artifact load == roll(cold base annotate, offset), "
          f"{len(sample)} sample assets byte-identical.", flush=True)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_arg_parser():
    p = argparse.ArgumentParser(
        description="TASK_804 standalone token-timeline disk-artifact pre-builder (real + shift).")
    p.add_argument("--base-vocab", default="experiments/804/L1_RESULT_vocab.json",
                   help="real frozen L1 vocab JSON (default the campaign L1_RESULT vocab).")
    p.add_argument("--shift-spec", default="",
                   help="optional cyclic-shift null spec (experiments/804/shift_null_spec.json) — also "
                        "build the shift_fe artifact.")
    p.add_argument("--cache-dir", default="experiments/804/token_cache",
                   help="artifact dir (timelines_<sha>.npz written here). Default the campaign dir.")
    p.add_argument("--assets-file", default="",
                   help="file with one DB asset id per line (server: /tmp/b3_asset_ids.txt).")
    p.add_argument("--assets", default="",
                   help="comma-separated DB asset ids (overridden by --assets-file).")
    p.add_argument("--db", default="", help="DB path override (default config.db_path()).")
    p.add_argument("--max-feature-bars", type=int, default=None,
                   help="LOCAL ACCEPTANCE tail cap on annotate length (seconds-fast on big assets). "
                        "OMIT at B3 for full history.")
    p.add_argument("--log-every", type=int, default=4,
                   help="progress log cadence in assets (default 4).")
    p.add_argument("--verify", action="store_true",
                   help="after writing, verify the artifact load == cold annotate (byte-identical) for "
                        "--verify-sample assets. Recommended.")
    p.add_argument("--verify-sample", type=int, default=3,
                   help="how many assets to byte-compare in --verify (default 3).")
    p.add_argument("--verify-only", action="store_true",
                   help="skip the build; only verify an EXISTING artifact in --cache-dir.")
    return p


def main(argv=None) -> int:
    args = _build_arg_parser().parse_args(argv)
    base_tok = TH.load_tokenizer(args.base_vocab)
    cfg = base_tok.cfg
    db = args.db if args.db else cfg.db
    if not os.path.isfile(db):
        print(f"FATAL: DB not found at {db}", flush=True)
        return 2

    ids = resolve_assets(args, base_tok)
    if not ids:
        print("FATAL: no assets resolved.", flush=True)
        return 2

    # the verify sample: first/middle/last of the universe (deterministic, spread).
    k = max(1, min(int(args.verify_sample), len(ids)))
    if k >= len(ids):
        sample = list(ids)
    else:
        sample = sorted({ids[0], ids[len(ids) // 2], ids[-1]} | set(ids[:k]))[:k]

    if not args.verify_only:
        TH.clear_timeline_cache()
        real_sha, real_path, n_real = build_real_artifact(
            args.base_vocab, ids, args.cache_dir, cfg=cfg, db=db,
            max_feature_bars=args.max_feature_bars, log_every=args.log_every)
        if args.shift_spec:
            derived_sha, base_sha, shift_path, n_shift = build_shift_artifact(
                args.shift_spec, ids, args.cache_dir, cfg=cfg, db=db,
                max_feature_bars=args.max_feature_bars, log_every=args.log_every)

    if args.verify or args.verify_only:
        print("\n=== VERIFY: disk artifact load == cold annotate (byte-identical) ===", flush=True)
        verify_real(args.base_vocab, ids, args.cache_dir, sample, cfg=cfg, db=db,
                    max_feature_bars=args.max_feature_bars)
        if args.shift_spec:
            verify_shift(args.shift_spec, ids, args.cache_dir, sample, cfg=cfg, db=db,
                         max_feature_bars=args.max_feature_bars)

    # final summary (paths + shas + counts).
    print("\n=== ARTIFACTS ===", flush=True)
    real_sha = TH.vocab_sha_for(TH.load_tokenizer(args.base_vocab))
    print(f"REAL  : {TH._artifact_path(args.cache_dir, real_sha)}", flush=True)
    print(f"        sha={real_sha}  assets={len(ids)}", flush=True)
    if args.shift_spec:
        st = TH.load_tokenizer(args.shift_spec)
        dsha = st._shift_null["derived_vocab_sha"]
        print(f"SHIFT : {TH._artifact_path(args.cache_dir, dsha)}", flush=True)
        print(f"        derived_sha={dsha}  base_sha={st._shift_null['base_vocab_sha']}  "
              f"assets={len(ids)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
