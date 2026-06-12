"""
rand_codebook.py — frequency-matched random-codebook helper for TASK_804 arm `rand_fe`.

ARCHIVED DEAD-END (TASK_804-B2.5): the π-permutation null is a NON-null (relabelling ids with a fresh
embedding preserves the token SEQUENCE, so the model just relearns id->embedding — "the control that
couldn't fail"). SUPERSEDED by the cyclic-shift null (experiments/804/build_shift_null.py + arm
`shift_fe`), which decouples the token sequence from the bars by rolling the timeline per asset. This
module is KEPT only as the archived dead-end record; it now drives NOTHING.

WHAT rand_fe IS (TASK_804 arm 3, the capacity/embedding placebo)
---------------------------------------------------------------
A frequency-matched random codebook keeps the EXACT token-shapes / frequencies of the real L1
vocab but assigns each shape a DIFFERENT id, via a seed-pinned bijection π over the real ids
(0..cb-1). This breaks every id<->shape alignment (so any "tokens help" signal that survives is a
capacity/embedding placebo, not real shared structure) while preserving the build frequency
MULTISET by construction — exactly the A8 cross-asset-transfer null
(`eval_battery.py::_codelength_perm_controls`, which uses `np.random.default_rng(seed).permutation(k)`).
We reuse that convention; the campaign seed is PINNED at `perm_seed=804` (distinct from A8's
seed0=12345). cb=1376 for the real vocab (864 leaf + 512 merge) but every function here is
cb-agnostic.

The OPERATIONAL rand_fe is `token_id_rand = π[token_id_real]` on the served token ids — exactly what
the front-end arm does (`test_token_frontend.py::test_e5_random_codebook_swap_no_branches`: a fixed
permutation of the real ids fed to the identical wiring, zero code branches). So the executable
object is the CHEAP REMAP below; the permuted-vocab JSON is the hash-answerable RECORD of the same π.

TWO PRODUCTS
------------
1. CHEAP CACHE REMAP (`remap_timeline`) — the OPERATIONAL path B3 uses. `token_id_rand = π[token_id]`
   on VALID bars; −1 stays −1; codelength / ts / valid carried through. ~2h saved: B3 remaps the
   cached real timeline instead of re-annotating the DB. This is exact by construction (π applied to
   the real ids), which is the load-bearing guarantee for B3.

2. PERMUTED VOCAB JSON (`write_permuted_vocab`) — the self-describing, sha256-able RECORD of π: the
   real frozen vocab (`tokenizer_api.save_vocab` schema) with every BPE-merge id reference relabelled
   under π (constituent ids AND each merge's positional result id), plus the FULL π array embedded in
   a `rand_codebook` block so the file alone reconstructs the cheap remap. It loads with the SAME
   `tokenizer_api.CandleTokenizer.from_path` loader, zero code branches (TASK_801 E5).

   IMPORTANT, MEASURED CAVEAT (do not silently assume otherwise — see the B2 report): the L1 encode
   path does NOT route the OUTPUT id of a token through the vocab. ~95% of timeline tokens are LEAF
   ids, and a leaf's output id comes straight from `l1_bpe.numerosity_reduce`
   (base_symbol * N_LEN_BUCKETS + bucket), NOT from any vocab field; and the BPE segmentation depends
   on merge RANK ORDER (list position), so relabelling merge result ids by repositioning them changes
   the segmentation. Therefore `annotate_asset(permuted_vocab)` does NOT reproduce
   `π[annotate_asset(real_vocab)]` for a non-identity π — the encode path has no π indirection. The
   permuted vocab is thus a RECORD/hash artifact, not an alternate encode driver. B3 must use the
   CHEAP REMAP (product 1), which IS exact. The acceptance battery proves this directly:
   `remap_timeline(annotate(real), π)` == `π[real ids]` on valid bars (the cheap path is exact), and
   the on-disk permuted vocab embeds the identical π (so the artifact and the executable path agree
   on π by construction).

WHY π IS REGION-RESPECTING (leaf<->leaf, merge<->merge)
-------------------------------------------------------
The flat id space is [0, n_base) leaf ids + [n_base, n_base+|merges|) merge ids (l1_bpe layout). For
the cheap remap a FREE permutation over 0..cb-1 is already a valid frequency-matched null. But for
the permuted-vocab RECORD to be STRUCTURALLY VALID (loadable), a merge's result id must stay in the
merge region (a merge result can never be a leaf) — so `write_permuted_vocab` requires a
REGION-RESPECTING π (leaf region onto itself, merge region onto itself). `build_permutation(cb, seed,
n_base=...)` builds exactly that (two independent A8-style permutations, one per region, from one
seeded RNG). Region-respecting is strictly stronger than needed for the remap and is what both
products share, so the manifest pins ONE π for both.

The MODEL-SIDE SENTINEL ROW (index cb, the reserved invalid-bar embedding row wired in
token_hotpath.create_model as token_vocab_size = cb + 1) is NEVER in 0..cb-1, so π never touches it;
−1 (the timeline's invalid marker, mapped to the sentinel at embed time) stays −1 through
remap_timeline.

PURE NUMPY / CPU ONLY. No JAX, no GPU, no DB needed for the helper itself (annotate_asset, used only
by the experiment-prep equivalence check, needs the DB).
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from typing import List, Tuple

import numpy as np

from . import tokenizer_api as TK  # noqa: F401  (re-exported convenience for callers)
from .token_timeline import TimelineResult


# --------------------------------------------------------------------------- #
# π — the seed-pinned bijection over 0..cb-1
# --------------------------------------------------------------------------- #
PERM_SEED = 804   # PINNED campaign seed (distinct from A8's seed0=12345).


def build_permutation(cb: int, seed: int = PERM_SEED, n_base: int | None = None) -> np.ndarray:
    """Seed-pinned bijection π over the real ids 0..cb-1 (int64 array of length cb; π[i] is the rand
    id assigned to real id i).

    Reuses the A8-null convention exactly: `np.random.default_rng(seed).permutation(k)`.

    cb     : codebook size (number of real ids). 1376 for the real L1 vocab; cb-agnostic here.
    seed   : permutation seed (default PERM_SEED=804, the pinned campaign seed).
    n_base : size of the leaf region [0, n_base). If given, π is REGION-RESPECTING — one A8-style
             permutation within [0, n_base) and an INDEPENDENT one within [n_base, cb), drawn from the
             SAME seeded RNG (leaf first, then merge). This is REQUIRED to relabel a vocab JSON (a
             merge result id must stay in the merge region). If None, a single free A8 permutation
             over 0..cb-1 is returned — valid as a cheap-remap null, but NOT usable for
             `write_permuted_vocab`.

    Invariants: sorted(π) == range(cb) (bijection, assert (i)); ids 0..cb-1 only (assert (ii));
    region-respecting when n_base is supplied (so `write_permuted_vocab` can place every result id).
    """
    cb = int(cb)
    if cb <= 0:
        return np.zeros(0, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    if n_base is None:
        return rng.permutation(cb).astype(np.int64)

    n_base = int(n_base)
    if not (0 <= n_base <= cb):
        raise ValueError(f"build_permutation: n_base {n_base} out of range for cb {cb}")
    perm = np.empty(cb, dtype=np.int64)
    if n_base > 0:                                   # leaf region [0, n_base) -> itself (drawn FIRST)
        perm[:n_base] = rng.permutation(n_base).astype(np.int64)
    n_merge = cb - n_base
    if n_merge > 0:                                  # merge region [n_base, cb) -> itself (SAME rng)
        perm[n_base:] = n_base + rng.permutation(n_merge).astype(np.int64)
    return perm


# --------------------------------------------------------------------------- #
# product 1 — cheap cache remap (the OPERATIONAL ~2h-saving path B3 uses)
# --------------------------------------------------------------------------- #
def remap_timeline(tl: TimelineResult, perm: np.ndarray) -> TimelineResult:
    """Return a NEW TimelineResult with token ids relabelled token_id_rand = π[token_id_real] on
    VALID bars; −1 (invalid) stays −1. codelength / ts / valid / asset_id / language carried through.

    This is the OPERATIONAL rand_fe (matches the front-end E5 swap) and the cheap path: instead of
    re-annotating the DB with a permuted vocab, B3 remaps the cached real timeline. Exact by
    construction.

    INVARIANTS:
      * −1 stays −1 (invalid bars are never permuted; they map to the model sentinel row at embed).
      * the model sentinel row (index cb) is never produced here (π maps 0..cb-1 -> 0..cb-1).
      * the valid-token id-frequency MULTISET is preserved (π is a bijection) — assert (iii).
    """
    perm = np.asarray(perm, dtype=np.int64)
    cb = perm.shape[0]
    ids = np.asarray(tl.token_id, dtype=np.int64)
    valid = np.asarray(tl.valid, dtype=bool)
    # any valid bar whose id is out of 0..cb-1 is a contract violation (the real timeline must only
    # ever carry ids 0..cb-1 or −1); surface it loudly rather than silently mis-mapping.
    bad = valid & ((ids < 0) | (ids >= cb))
    if np.any(bad):
        raise ValueError(f"remap_timeline: {int(bad.sum())} valid bars carry an id outside 0..cb-1 "
                         f"(cb={cb}); perm cb does not match the timeline's vocab.")
    out_ids = ids.copy()
    in_band = valid & (ids >= 0) & (ids < cb)
    out_ids[in_band] = perm[ids[in_band]]
    out_ids[~valid] = -1   # −1 stays −1 (and any invalid bar is forced back to −1).
    return dataclasses.replace(tl, token_id=out_ids.astype(np.int32))


# --------------------------------------------------------------------------- #
# product 2 — permuted vocab JSON (the sha256-able RECORD of π)
# --------------------------------------------------------------------------- #
def permute_vocab_doc(doc: dict, perm: np.ndarray) -> dict:
    """Relabel a frozen-vocab JSON (tokenizer_api.save_vocab schema) under the region-respecting π and
    return the new doc. The full π is embedded under `rand_codebook.perm` so the file alone
    reconstructs the cheap remap. Id references live ONLY in `merges` (the base-alphabet table carries
    no token ids); the schema's merge result id is POSITIONAL (merge i -> id n_base+i).

    Relabel rule (region-respecting, leaf<->leaf and merge<->merge):
      * constituents: (l, r) -> (π[l], π[r]).
      * result id   : real n_base+i -> π[n_base+i]; since result ids are positional, the relabelled
        merge is PLACED at output index j = π[n_base+i] - n_base (scatter by new result id).

    NOTE (see module docstring): this yields a STRUCTURALLY VALID, loadable record, but because the
    L1 encode path does not route output ids through the vocab (leaf ids come from numerosity reduce;
    segmentation depends on merge rank order), it is NOT an alternate encode driver — the executable
    rand_fe is the cheap remap. The embedded π ties the record to that remap.
    """
    perm = np.asarray(perm, dtype=np.int64)
    out = copy.deepcopy(doc)
    sizes = doc.get("sizes", {})
    n_base = int(sizes["n_base_tagged"])
    n_merges = int(sizes["n_merges"])
    cb = int(sizes["codebook_size"])
    if perm.shape[0] != cb:
        raise ValueError(f"permute_vocab_doc: perm length {perm.shape[0]} != codebook_size {cb}")
    if not (np.all(perm[:n_base] < n_base) and np.all(perm[n_base:] >= n_base)):
        raise ValueError(
            "permute_vocab_doc: π is NOT region-respecting (leaf id mapped into the merge region or "
            "vice versa). Build π with build_permutation(cb, seed, n_base=<n_base>)."
        )
    merges = doc["merges"]
    if len(merges) != n_merges:
        raise ValueError(f"permute_vocab_doc: |merges| {len(merges)} != sizes.n_merges {n_merges}")

    new_merges: List = [None] * n_merges
    for i, (l, r) in enumerate(merges):
        new_rid = int(perm[n_base + i])            # rand result id under π (in the merge region)
        j = new_rid - n_base                       # output position for the relabelled merge
        if not (0 <= j < n_merges):
            raise ValueError(f"permute_vocab_doc: result id {new_rid} out of merge region")
        if new_merges[j] is not None:
            raise ValueError(f"permute_vocab_doc: two merges collided on output slot {j} — π not a "
                             "bijection on the merge region")
        new_merges[j] = [int(perm[int(l)]), int(perm[int(r)])]
    if any(m is None for m in new_merges):
        raise ValueError("permute_vocab_doc: not every merge slot was filled — π not a bijection")
    out["merges"] = new_merges
    out["rand_codebook"] = {
        "perm_seed": int(PERM_SEED),
        "perm": [int(x) for x in perm.tolist()],   # full π: file reconstructs the cheap remap.
        "region_respecting": True,
        "note": "frequency-matched random codebook (TASK_804 rand_fe): ids relabelled by a "
                "region-respecting seed-pinned bijection π over 0..cb-1; shapes/frequencies unchanged. "
                "RECORD only — the executable rand_fe is the cheap remap π[token_id] (the L1 encode "
                "path does not route output ids through the vocab).",
    }
    return out


def write_permuted_vocab(real_vocab_path: str, perm: np.ndarray, out_path: str) -> str:
    """Load the real frozen vocab JSON, relabel every id reference under π (permute_vocab_doc),
    embed the full π, and write the permuted vocab JSON to out_path. Returns out_path.

    The output is a valid tokenizer_api frozen-vocab JSON: `CandleTokenizer.from_path` loads it with
    ZERO code branches (TASK_801 E5 — rand_fe swaps only the id mapping).
    """
    with open(real_vocab_path) as f:
        doc = json.load(f)
    new_doc = permute_vocab_doc(doc, perm)
    with open(out_path, "w") as f:
        json.dump(new_doc, f, indent=2)
    return out_path


# --------------------------------------------------------------------------- #
# sha256 + vocab-size helpers
# --------------------------------------------------------------------------- #
def sha256_file(path: str) -> str:
    """sha256 hex digest of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def n_base_of_vocab_path(real_vocab_path: str) -> Tuple[int, int]:
    """Read (n_base_tagged, codebook_size) from a frozen-vocab JSON's `sizes` block without
    reconstructing the BPEVocab — used to build the region-respecting π for that exact vocab."""
    with open(real_vocab_path) as f:
        sizes = json.load(f)["sizes"]
    return int(sizes["n_base_tagged"]), int(sizes["codebook_size"])


# --------------------------------------------------------------------------- #
# experiment-prep entry — writes the rand_fe artifacts at RUN TIME into experiments/804/
# --------------------------------------------------------------------------- #
def prep_experiment_artifacts(real_vocab_path: str, out_dir: str, seed: int = PERM_SEED,
                              tag: str = "") -> dict:
    """Generate and persist the rand_fe artifacts for ONE real vocab into out_dir.

    Writes (acceptance (iv)):
      * <out_dir>/L1_RESULT_vocab.json          — a verbatim COPY of the real vocab (inputs of record).
      * <out_dir>/rand_perm_s<seed>_vocab.json  — the PERMUTED vocab record (product 2, embeds π).
      * <out_dir>/rand_codebook_manifest.json   — perm_seed + cb + n_base + both shas + tag.

    The permutation is the region-respecting π for THIS vocab's (n_base, cb). No DB / GPU — artifact
    minting only. Returns the manifest dict.

    At B3 on the server this runs against the REAL pinned cb-1376 vocab
    (runs/799_grammar/L1_RESULT_vocab.json); locally (acceptance) it runs against a bounded fit vocab
    and the outputs are clearly tagged local-vocab.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    n_base, cb = n_base_of_vocab_path(real_vocab_path)
    perm = build_permutation(cb, seed=seed, n_base=n_base)

    real_copy = os.path.join(out_dir, "L1_RESULT_vocab.json")
    perm_path = os.path.join(out_dir, f"rand_perm_s{int(seed)}_vocab.json")
    manifest_path = os.path.join(out_dir, "rand_codebook_manifest.json")

    with open(real_vocab_path, "rb") as fsrc, open(real_copy, "wb") as fdst:
        fdst.write(fsrc.read())                    # verbatim real-vocab copy (source sha)
    write_permuted_vocab(real_vocab_path, perm, perm_path)

    manifest = {
        "task": "TASK_804-B2 rand_fe (frequency-matched random codebook)",
        "tag": tag,
        "perm_seed": int(seed),
        "codebook_size": int(cb),
        "n_base_tagged": int(n_base),
        "n_merges": int(cb - n_base),
        "region_respecting": True,
        "convention": "A8 null — np.random.default_rng(seed).permutation(k), per region (leaf, merge)",
        "operational_path": "cheap remap remap_timeline(real_timeline, π) — π[token_id] on valid bars",
        "real_vocab_source": os.path.abspath(real_vocab_path),
        "artifacts": {
            "real_vocab_copy": os.path.basename(real_copy),
            "permuted_vocab": os.path.basename(perm_path),
        },
        "sha256": {
            "L1_RESULT_vocab.json": sha256_file(real_copy),
            os.path.basename(perm_path): sha256_file(perm_path),
        },
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def _build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(
        description="TASK_804-B2 rand_fe artifact prep: permuted vocab + manifest (seed 804).")
    p.add_argument("--real-vocab", required=True,
                   help="path to the real frozen L1 vocab JSON (e.g. runs/799_grammar/L1_RESULT_vocab.json).")
    p.add_argument("--out-dir", default="experiments/804",
                   help="output dir for artifacts (default experiments/804).")
    p.add_argument("--seed", type=int, default=PERM_SEED, help="permutation seed (default 804).")
    p.add_argument("--tag", default="", help="free-text tag stamped in the manifest (e.g. 'local-vocab').")
    return p


def main(argv=None) -> int:
    args = _build_arg_parser().parse_args(argv)
    manifest = prep_experiment_artifacts(args.real_vocab, args.out_dir, seed=args.seed, tag=args.tag)
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
