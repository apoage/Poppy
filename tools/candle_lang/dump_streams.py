"""dump_streams.py — member-agnostic per-asset token-stream dumper (candle-language panel, branch C).

NEW EXPLORATORY RESEARCH CODE (CPU ONLY — pure numpy, no JAX/GPU). Companion to token_lm.py.

The token-LM grammar probe (token_lm.py) was hard-wired to the L1-BPE tokenizer. To run the SAME
probe over the tournament-champion members (L2 ABBA, L5 matrix-profile) we DUMP, not inline-integrate:
each member's NATIVE, FAITHFULLY-SCORED encode path produces per-asset int token-id streams which we
persist to an `.npz`; token_lm.py then just consumes the ids. Everything downstream of
"encode -> token-id stream" inside token_lm.py is member-agnostic, so this dump keeps each champion's
scored encode contract intact (no divergence-prone reimplementation in the LM file).

WHY A DUMP (not inline): L2 and L5 have DIFFERENT encode contracts than L1. Reproducing each
champion's scored encode path inline would be fragile. Instead each member's REAL encode (the same
functions its eval-battery run calls) is invoked here once, ids flattened per asset, written to disk.

FAITHFUL ENCODE PATHS (mirror exactly how each member is scored — see source):
  * L1 BPE  — tokenizer_api.fit_and_save -> CandleTokenizer + token_lm._encode_asset_flat (the SAME
              per-asset flat encode token_lm uses natively). codebook = tok.codebook_size. Lets us
              sanity-check dump-L1 ~= native-L1.
  * L2 ABBA — l2.fit_vocab(cfg, build_assets, k=DEFAULT_K, scl=DEFAULT_SCL, ...); per asset concat the
              chunks from l2.encode_asset_ids(cfg, asset, vocab, control=None, ...). The cfg carries
              the module DEFAULT_TOL (we do NOT set a custom tol). codebook = vocab.codebook_size (256).
  * L5 MP   — l5.fit_base_alphabet(db, build_assets, cfg); per asset
              l5._asset_anchor_symbols(db, asset, cfg, table, subseq_len=DEFAULT_SUBSEQ_LEN, ...),
              already a time-ordered 1-D int array of anchor-centre base symbols.
              codebook = l5.BA.N_SYMBOLS(table) (~288).

L5 CAVEAT (be honest in the writeup): L5's LM tokens are BASE-ALPHABET symbols (~288) sampled at
matrix-profile ANCHOR positions — i.e. the L5 "language" handed to the LM is base-symbols at
motif-anchor cadence, a THINNER vocabulary than L2's adaptive shape codebook. This is expected (L5
mints no new vocabulary; it validates motifs over the shared base alphabet), but it means the L5
token-LM probe measures grammar over a coarser symbol stream than L1/L2.

OUTPUT .npz (np.savez, int32 streams to bound size — token ids are small):
  * one array per asset under key `b{asset_id}` (build) / `h{asset_id}` (holdout) — empty streams
    are dropped (no key written).
  * `codebook_size` (0-d int array), `build_assets` / `holdout_assets` (int arrays), `member` (0-d
    string array).

MEMORY DISCIPLINE: stream per-asset, never materialize the whole corpus at once beyond the per-asset
array; int32 dtype. CPU only (the L1/L2/L5 symbolic encode paths never touch a device).

CLI:
    python -m tools.candle_lang.dump_streams --member L2 \
        --build-assets 1087 1091 353 --holdout-assets 365 1118 \
        --out /tmp/l2_streams.npz --max-windows-per-asset 2000
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from . import config as C
from .symbolic import l2_abba as l2
from .symbolic import l5_matrix_profile as l5


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Dump per-asset token-id streams for a candle-language member (L1/L2/L5) to an "
                    ".npz for consumption by token_lm.py. CPU only."
    )
    p.add_argument("--member", choices=("L1", "L2", "L5"), required=True,
                   help="which panel member's native, scored encode path to dump.")
    p.add_argument("--db", type=str, default=None,
                   help="SQLite DB path (default: config.db_path() / $CANDLE_DB).")
    p.add_argument("--build-assets", nargs="+", type=int, default=None,
                   help="BUILD asset ids (vocab is FIT on these). default => config.DEFAULT_BUILD_ASSETS.")
    p.add_argument("--holdout-assets", nargs="+", type=int, default=None,
                   help="HOLDOUT asset ids. default => config.DEFAULT_HOLDOUT_ASSETS.")
    p.add_argument("--out", type=str, required=True, help="output .npz path.")
    p.add_argument("--max-windows-per-asset", type=int, default=20000,
                   help="cap windows/pieces/anchors kept per asset; 0 => unbounded.")
    p.add_argument("--window", type=int, default=C.DEFAULT_WINDOW, help="vol-unit window length W.")
    p.add_argument("--seed", type=int, default=0, help="PRNG seed (vocab fit + encode determinism).")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# per-member fit + per-asset encode
# --------------------------------------------------------------------------- #
def _dump_l1(db, build_assets, holdout_assets, cfg, max_windows, out_path, seed):
    """L1 BPE: fit+save a fresh vocab on BUILD, then encode each asset with token_lm's flat encode."""
    from .tokenizer_api import fit_and_save, CandleTokenizer
    from .token_lm import _encode_asset_flat

    vocab_path = out_path[:-4] if out_path.endswith(".npz") else out_path
    vocab_path = vocab_path + "_l1_vocab.json"
    print(f"[L1] fitting fresh L1-BPE vocab on BUILD -> {vocab_path}", flush=True)
    _vocab, vocab_path = fit_and_save(
        cfg, build_assets, vocab_path,
        max_windows_per_asset=(max_windows or 20000), verbose=True,
    )
    tok = CandleTokenizer.from_path(vocab_path)
    codebook_size = int(tok.codebook_size)

    def encode(asset_id):
        return _encode_asset_flat(tok, db, int(asset_id), tok.cfg, max_windows)

    return encode, codebook_size


def _dump_l2(db, build_assets, holdout_assets, cfg, max_windows, out_path, seed):
    """L2 ABBA: fit the codebook on BUILD (module DEFAULT_TOL via cfg), then concat encode chunks."""
    print(f"[L2] fitting ABBA codebook on BUILD (k={l2.DEFAULT_K} scl={l2.DEFAULT_SCL} "
          f"tol={l2.DEFAULT_TOL}) ...", flush=True)
    vocab = l2.fit_vocab(cfg, build_assets, k=l2.DEFAULT_K, scl=l2.DEFAULT_SCL,
                         max_windows_per_asset=max_windows, seed=seed)
    codebook_size = int(vocab.codebook_size)

    def encode(asset_id):
        chunks = list(l2.encode_asset_ids(cfg, int(asset_id), vocab, control=None,
                                          max_windows_per_asset=max_windows))
        if not chunks:
            return np.zeros(0, dtype=np.int64)
        return np.concatenate(chunks).astype(np.int64)

    return encode, codebook_size


def _dump_l5(db, build_assets, holdout_assets, cfg, max_windows, out_path, seed):
    """L5 matrix-profile: fit the shared base alphabet on BUILD, then per-asset anchor base symbols.

    NOTE (thin-vocab caveat in the module docstring): the dumped ids are base-alphabet symbols
    (~288) at matrix-profile anchor cadence, a coarser language than L2's adaptive shape codebook.
    """
    print("[L5] fitting shared base alphabet on BUILD ...", flush=True)
    table = l5.fit_base_alphabet(db, build_assets, cfg, seed=seed)
    codebook_size = int(l5.BA.N_SYMBOLS(table))

    def encode(asset_id):
        ids = l5._asset_anchor_symbols(db, int(asset_id), cfg, table,
                                       subseq_len=l5.DEFAULT_SUBSEQ_LEN, seed=seed, control=None)
        return np.asarray(ids, dtype=np.int64).ravel()

    return encode, codebook_size


_DUMPERS = {"L1": _dump_l1, "L2": _dump_l2, "L5": _dump_l5}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None):
    args = parse_args(argv)
    db = args.db if args.db else C.db_path()
    build_assets = tuple(args.build_assets) if args.build_assets else C.DEFAULT_BUILD_ASSETS
    holdout_assets = tuple(args.holdout_assets) if args.holdout_assets is not None \
        else C.DEFAULT_HOLDOUT_ASSETS
    max_windows = None if int(args.max_windows_per_asset) <= 0 else int(args.max_windows_per_asset)
    cfg = C.CandleConfig(window=int(args.window), build_assets=build_assets,
                         holdout_assets=holdout_assets, db=db)

    print(f"[dump] member={args.member} db={db} W={cfg.window} "
          f"build={list(build_assets)} holdout={list(holdout_assets)} "
          f"max_windows_per_asset={max_windows}", flush=True)

    t0 = time.time()
    encode, codebook_size = _DUMPERS[args.member](
        db, build_assets, holdout_assets, cfg, max_windows, args.out, int(args.seed))
    print(f"[dump] {args.member} codebook_size={codebook_size}", flush=True)

    # ---- per-asset encode -> int32 streams; empty streams dropped (no key) ----
    arrays = {}
    n_build_written = 0
    n_hold_written = 0
    build_tok = 0
    hold_tok = 0
    for a in build_assets:
        s = encode(a)
        if s.size > 0:
            arrays[f"b{int(a)}"] = s.astype(np.int32)
            n_build_written += 1
            build_tok += int(s.size)
            print(f"[dump] build asset {a}: {s.size} tokens", flush=True)
        else:
            print(f"[dump] build asset {a}: EMPTY (dropped)", flush=True)
    for a in holdout_assets:
        s = encode(a)
        if s.size > 0:
            arrays[f"h{int(a)}"] = s.astype(np.int32)
            n_hold_written += 1
            hold_tok += int(s.size)
            print(f"[dump] holdout asset {a}: {s.size} tokens", flush=True)
        else:
            print(f"[dump] holdout asset {a}: EMPTY (dropped)", flush=True)

    arrays["codebook_size"] = np.asarray(int(codebook_size), dtype=np.int64)
    arrays["build_assets"] = np.asarray([int(a) for a in build_assets], dtype=np.int64)
    arrays["holdout_assets"] = np.asarray([int(a) for a in holdout_assets], dtype=np.int64)
    arrays["member"] = np.asarray(str(args.member))

    np.savez(args.out, **arrays)
    wall = time.time() - t0

    print(f"[dump] wrote {args.out}", flush=True)
    print(f"[dump] member={args.member}  codebook_size={codebook_size}  "
          f"build_assets_written={n_build_written}/{len(build_assets)}  "
          f"holdout_assets_written={n_hold_written}/{len(holdout_assets)}  "
          f"build_tokens={build_tok}  holdout_tokens={hold_tok}  ({wall:.1f}s)", flush=True)
    if n_build_written == 0 or n_hold_written == 0:
        print("[dump] WARNING: a split has zero non-empty streams — token_lm will reject this dump.",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
