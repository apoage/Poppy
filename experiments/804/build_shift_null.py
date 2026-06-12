#!/usr/bin/env python3
"""build_shift_null.py — TASK_804-B2.5 cyclic-shift null-spec generator for arm `shift_fe`.

WHAT shift_fe IS (the CORRECTED real-vs-null control — replaces the dead-end π-permutation)
-------------------------------------------------------------------------------------------
The π-permutation null (TASK_804-B2, rand_codebook.py) is a NON-null: relabelling ids with a fresh
embedding preserves the token SEQUENCE, so the model just relearns id->embedding ("the control that
couldn't fail"). The correct null is a CYCLIC SHIFT: keep each asset's token MARGINAL multiset but
DECOUPLE the token sequence from the bars by rolling the per-bar timeline a per-asset offset — so
token[t] now comes from a DIFFERENT bar. This isolates whether the real token<->bar ALIGNMENT carries
signal (a real "tokens help" effect must die under the shift; a capacity/embedding placebo survives).

THE SPEC THIS WRITES (shift_null_spec.json)
-------------------------------------------
A self-describing, sha256-able record the hot-path detects (token_hotpath.load_tokenizer):

    {
      "null_type":       "cyclic_shift",
      "base_vocab":      "<L1_RESULT path>",     # real vocab the timeline is annotated WITH
      "base_vocab_sha":  "<content-canonical sha>",
      "seed":            804,                     # SHIFT_SEED (pinned)
      "per_asset_offsets": {asset_id: offset, ...},   # RESOLVED absolute roll offsets (bars)
      "per_asset_lengths": {asset_id: T_a, ...},      # timeline lengths the offsets were resolved at
      "per_asset_fracs":   {asset_id: f_a, ...},       # the seed-derived fraction (offset = round(f*T))
      "derived_vocab_sha": "<hash(base_sha, 'cyclic_shift', seed, offsets)>",
      "sha256":          "<sha256 of the spec bytes (sans this field)>"
    }

PER-ASSET OFFSETS — seed-derived, LARGE, and decoupling
-------------------------------------------------------
Each asset's offset is deterministic from (seed, asset_id): a fraction f_a in [0.2, 0.8] drawn from a
seed+asset_id-keyed RNG, resolved to an ABSOLUTE bar offset round(f_a * T_a). Picking f in [0.2, 0.8]
(NOT a tiny offset) is the guard for assert (c): a small roll would leave token[t] still ~ the bar it
came from in quiet stretches, so agreement would stay near 1 and the control would NOT decouple. A
large roll lands token[t] on an unrelated bar, so per-bar agreement falls to the marginal collision
probability Sum_i p_i^2 (chance) — proving the decoupling. T_a is measured by annotating each asset
ONCE with the base vocab (the same annotate_asset the hot-path uses), so the resolved offset is exact
for the timeline the wiring will roll.

DETERMINISM (assert (d)): same seed + same base vocab + same asset set + same DB -> byte-identical
offsets (the fraction is (seed, asset_id)-keyed; T_a is a property of the data) -> the same derived
vocab_sha. Regenerating reproduces the spec exactly.

CPU ONLY. annotate_asset needs the DB; no GPU, no server, no model. At B3 on the server this runs
against the REAL pinned cb-1376 vocab (runs/799_grammar/L1_RESULT_vocab.json); locally (acceptance)
against a bounded local-fit vocab, clearly tagged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

# Resolve the repo root from THIS file's location (works in the main checkout AND an isolated
# worktree) so the worktree's own tools/candle_lang is imported, not the main checkout's.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tools.candle_lang import config as C
from tools.candle_lang import token_hotpath as TH
from tools.candle_lang import token_timeline as TT


SHIFT_SEED = 804          # PINNED campaign seed (matches the rand_fe lineage; distinct purpose).
NULL_TYPE = "cyclic_shift"

# Offset fraction band: each per-asset offset is round(f_a * T_a) with f_a in [FRAC_LO, FRAC_HI].
# LARGE by design so the roll decouples token[t] from its bar (assert (c)); a tiny offset would NOT.
FRAC_LO = 0.2
FRAC_HI = 0.8


# --------------------------------------------------------------------------- #
# seed-derived per-asset fraction  (deterministic from (seed, asset_id))
# --------------------------------------------------------------------------- #
def offset_fraction(seed: int, asset_id: int) -> float:
    """Deterministic fraction f_a in [FRAC_LO, FRAC_HI] for one asset, keyed by (seed, asset_id).

    Uses a per-asset RNG seeded by SeedSequence([seed, asset_id]) so two assets never share a draw
    and the fraction depends ONLY on (seed, asset_id) — not on the asset universe size, ordering, or
    T_a. The absolute offset is then round(f_a * T_a) (T_a folded in at resolve time), so the offset
    is large + decoupling AND reproducible from the seed alone.
    """
    ss = np.random.SeedSequence([int(seed), int(asset_id)])
    rng = np.random.default_rng(ss)
    return float(FRAC_LO + (FRAC_HI - FRAC_LO) * rng.random())


def resolve_offset(seed: int, asset_id: int, timeline_len: int) -> int:
    """Resolve the ABSOLUTE roll offset (bars) for one asset at its measured timeline length T_a.

    offset = round(f_a * T_a), clamped to [1, T_a-1] so the roll is non-trivial and stays cyclic
    (a 0 or T_a roll is the identity). Deterministic from (seed, asset_id, T_a)."""
    T = int(timeline_len)
    if T <= 1:
        return 0
    f = offset_fraction(seed, asset_id)
    off = int(round(f * T))
    # keep strictly inside (0, T): a 0 / T offset is the identity roll (no decoupling).
    off = max(1, min(off, T - 1))
    return off


# --------------------------------------------------------------------------- #
# derived vocab sha  (distinct from the real vocab, reproducible)
# --------------------------------------------------------------------------- #
def derive_vocab_sha(base_vocab_sha: str, seed: int,
                     per_asset_offsets: Dict[int, int]) -> str:
    """Deterministic derived vocab_sha for the SHIFTED timelines.

    A content-canonical sha256 of (base_vocab_sha, NULL_TYPE, seed, the resolved per-asset offsets).
    DISTINCT from the real vocab_sha (so the shifted timelines never collide with the real cache) and
    REPRODUCIBLE (same inputs -> same hash, assert (d)). Offsets are serialized sorted-by-asset-id so
    dict ordering can never flip the hash.
    """
    canon = {
        "base_vocab_sha": str(base_vocab_sha),
        "null_type": str(NULL_TYPE),
        "seed": int(seed),
        "per_asset_offsets": {str(int(a)): int(o) for a, o in sorted(per_asset_offsets.items())},
    }
    payload = json.dumps(canon, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------- #
# timeline-length measurement (annotate ONCE with the base vocab)
# --------------------------------------------------------------------------- #
def measure_timeline_len(db_path: str, asset_id: int, base_tok, cfg,
                         max_feature_bars: Optional[int] = None) -> int:
    """Annotate ONE asset with the BASE vocab (via the SAME annotate_asset the hot-path uses) and
    return its timeline length T_a, the length the wiring's roll offset is resolved against.

    At B3 (max_feature_bars=None, the real path) the annotation goes through the pool RAM cache
    (_get_cached_timeline), so the subsequent hot-path build of the same asset+base vocab reuses it
    (no double annotate) — the served timeline and the offset-resolution timeline are the SAME full
    history. For the CPU-acceptance path a tail cap is supplied (max_feature_bars): T_a MUST then be
    measured on the SAME capped timeline the test rolls, so we annotate directly with the cap (the
    pool cache is a full-history perf cache and is intentionally bypassed under a cap)."""
    if max_feature_bars is not None:
        tl = TT.annotate_asset(db_path, int(asset_id), base_tok, cfg,
                               max_feature_bars=int(max_feature_bars))
        return int(len(tl))
    vocab_sha = TH.vocab_sha_for(base_tok)
    tl = TH._get_cached_timeline(int(asset_id), vocab_sha, db_path, base_tok, cfg)
    return int(len(tl))


# --------------------------------------------------------------------------- #
# spec assembly + persistence
# --------------------------------------------------------------------------- #
def build_shift_null_spec(base_vocab_path: str, asset_ids: Sequence[int],
                          out_path: str, seed: int = SHIFT_SEED,
                          cfg: Optional["C.CandleConfig"] = None,
                          db_path: Optional[str] = None,
                          max_feature_bars: Optional[int] = None,
                          tag: str = "") -> dict:
    """Build the cyclic-shift null spec for `asset_ids` against `base_vocab_path` and write it to
    out_path. Returns the spec dict (with its own sha256 stamped).

    base_vocab_path  : the real L1 vocab JSON the timeline is annotated WITH (server-only cb-1376 at
                       B3; a bounded local-fit vocab for CPU acceptance).
    asset_ids        : DB asset ids to resolve offsets for (the slot/period asset universe).
    out_path         : where to write shift_null_spec.json.
    seed             : SHIFT_SEED (pinned 804).
    cfg / db_path    : CandleConfig + DB path (default: cfg from the loaded base tokenizer + cfg.db).
    max_feature_bars : optional tail cap (acceptance speed; None at B3 for full-history T_a).
    """
    base_tok = TH.load_tokenizer(base_vocab_path)
    cfg = cfg if cfg is not None else base_tok.cfg
    db = db_path if db_path is not None else cfg.db
    base_vocab_sha = TH.vocab_sha_for(base_tok)

    aids = sorted({int(a) for a in asset_ids if int(a) >= 0})
    per_asset_offsets: Dict[int, int] = {}
    per_asset_lengths: Dict[int, int] = {}
    per_asset_fracs: Dict[int, float] = {}
    for aid in aids:
        T_a = measure_timeline_len(db, aid, base_tok, cfg, max_feature_bars=max_feature_bars)
        off = resolve_offset(seed, aid, T_a)
        per_asset_offsets[aid] = off
        per_asset_lengths[aid] = T_a
        per_asset_fracs[aid] = offset_fraction(seed, aid)

    derived_sha = derive_vocab_sha(base_vocab_sha, seed, per_asset_offsets)

    spec = {
        "task": "TASK_804-B2.5 shift_fe (cyclic-shift null)",
        "tag": tag,
        "null_type": NULL_TYPE,
        "base_vocab": os.path.abspath(base_vocab_path),
        "base_vocab_sha": base_vocab_sha,
        "seed": int(seed),
        "frac_band": [FRAC_LO, FRAC_HI],
        "per_asset_offsets": {str(a): int(o) for a, o in per_asset_offsets.items()},
        "per_asset_lengths": {str(a): int(t) for a, t in per_asset_lengths.items()},
        "per_asset_fracs": {str(a): float(f) for a, f in per_asset_fracs.items()},
        "derived_vocab_sha": derived_sha,
        "convention": (
            "rolled_token_id = np.roll(token_id, off); rolled_valid = np.roll(valid, off); ts NOT "
            "rolled. The shifted timeline flows through the SAME join_timeline_to_grid + lag as real "
            "(the roll is purely pre-join). offset = round(f_a * T_a), f_a in [0.2,0.8] keyed by "
            "(seed, asset_id) — large so the roll DECOUPLES token from bar (agreement ~ Sum p_i^2)."
        ),
    }
    # stamp the spec's own sha256 over the canonical bytes (sans the sha field itself).
    spec["sha256"] = spec_self_sha(spec)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(spec, f, indent=2)
    return spec


def spec_self_sha(spec: dict) -> str:
    """Recompute the canonical sha256 of a spec dict (over every field EXCEPT 'sha256'), the way
    build_shift_null_spec stamps it. Used by assert (b) to prove the recorded sha matches the bytes."""
    spec_for_hash = {k: v for k, v in spec.items() if k != "sha256"}
    spec_bytes = json.dumps(spec_for_hash, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(spec_bytes).hexdigest()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_arg_parser():
    p = argparse.ArgumentParser(
        description="TASK_804-B2.5 cyclic-shift null-spec generator (shift_fe, seed 804).")
    p.add_argument("--base-vocab", required=True,
                   help="path to the real frozen L1 vocab JSON (e.g. runs/799_grammar/L1_RESULT_vocab.json).")
    p.add_argument("--assets", default="",
                   help="comma-separated DB asset ids (default: the base vocab's CandleConfig.build_assets).")
    p.add_argument("--out", default="experiments/804/shift_null_spec.json",
                   help="output spec path (default experiments/804/shift_null_spec.json).")
    p.add_argument("--seed", type=int, default=SHIFT_SEED, help="shift seed (default 804).")
    p.add_argument("--max-feature-bars", type=int, default=None,
                   help="optional tail cap on T_a (acceptance speed; omit at B3 for full history).")
    p.add_argument("--tag", default="", help="free-text tag stamped in the spec.")
    return p


def main(argv=None) -> int:
    args = _build_arg_parser().parse_args(argv)
    base_tok = TH.load_tokenizer(args.base_vocab)
    if args.assets.strip():
        assets = [int(x) for x in args.assets.split(",") if x.strip()]
    else:
        assets = list(base_tok.cfg.build_assets)
    spec = build_shift_null_spec(
        args.base_vocab, assets, args.out, seed=args.seed,
        cfg=base_tok.cfg, db_path=base_tok.cfg.db,
        max_feature_bars=args.max_feature_bars, tag=args.tag,
    )
    print(json.dumps({k: v for k, v in spec.items()
                      if k not in ("per_asset_offsets", "per_asset_lengths", "per_asset_fracs")},
                     indent=2), flush=True)
    print(f"per-asset offsets: {spec['per_asset_offsets']}", flush=True)
    print(f"spec sha256: {spec['sha256']}  ->  {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
