"""
test_tokenizer_api.py — the HARD GATE for the frozen L1 tokenizer (TASK_797 Stage 1).

Run (CPU, synthetic, no DB needed for the gate):
    JAX_PLATFORMS=cpu python -m tools.candle_lang.test_tokenizer_api

The whole point of freezing the tokenizer is that a saved+loaded vocab encodes BYTE-IDENTICALLY to
the in-process vocab. If it does not, the freeze is wrong (an edge/merge/layout was dropped or
reordered). This test fits L1 on a synthetic in-memory unit stream, encodes a synthetic input with
the in-process vocab (ids_A), then save -> load -> encode the SAME input (ids_B), and asserts
np.array_equal(ids_A, ids_B). It also exercises the features -> token-ids path (real edges) so the
base-alphabet freeze is covered too, not just the merge table.

A second, OPTIONAL section (only if the local DB exists) does a cheap one-asset encode_asset smoke
check: that it returns a non-empty token-id stream without error. The ids are NOT validated beyond
non-empty (per the TASK_797 Stage-1 contract).
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

from . import config as C
from . import base_alphabet as BA
from .symbolic import l1_bpe as L1
from . import tokenizer_api as TA


# --------------------------------------------------------------------------- #
# synthetic fixtures (no DB)
# --------------------------------------------------------------------------- #
def _synthetic_features(n_rows=200_000, seed=0):
    """A synthetic fat-tailed pooled vol-unit corpus (N, N_FEAT), like base_alphabet's self-test."""
    rng = np.random.default_rng(seed)
    mv = rng.standard_t(df=3, size=n_rows)
    gap = 0.3 * rng.standard_t(df=3, size=n_rows)
    body = rng.standard_t(df=4, size=n_rows)
    rngc = np.abs(rng.standard_t(df=4, size=n_rows)) + 0.1
    uw = rng.random(n_rows)
    lw = rng.random(n_rows)
    return np.stack([mv, gap, body, rngc, uw, lw], axis=1).astype(np.float32)


def _motif_unit_sequences(table, n_seqs, seed):
    """Generate reduced-unit sequences with STRONG recurring structure: a small set of leaf "motifs"
    is repeated across sequences so frequency-greedy BPE has frequent adjacent pairs to merge AND the
    same motifs reappear at encode time (so the merge SERVING path actually fires — a pure-iid stream
    would merge at fit but never at serving, leaving the harder serving path untested).

    Leaf ids are drawn from the valid leaf region [0, n_base_tagged); motifs are short fixed leaf
    runs. Returns list[list[int]]. Pure units — no features needed (the merge path is unit-only)."""
    n_base = L1.n_base_tagged(table)
    rng = np.random.default_rng(seed)
    # a small bank of recurring 2-4 leaf motifs (the "words" BPE should learn)
    motifs = [list(rng.integers(0, n_base, size=int(rng.integers(2, 5)))) for _ in range(12)]
    seqs = []
    for _ in range(n_seqs):
        s = []
        for _k in range(int(rng.integers(8, 20))):
            if rng.random() < 0.7:
                s.extend(motifs[int(rng.integers(0, len(motifs)))])   # recurring motif
            else:
                s.append(int(rng.integers(0, n_base)))                # iid noise leaf
        if len(s) >= 2:
            seqs.append([int(x) for x in s])
    return seqs


def _fit_inprocess_vocab(seed=0):
    """Fit a base alphabet on a synthetic feature corpus, then fit L1 BPE on a POOLED synthetic
    reduced-unit stream WITH recurring motifs — WITHOUT a DB. l1_bpe.fit_bpe streams from a DB, so we
    replicate its core (pool reduced sequences -> frequency-greedy BPE) here. The RESULTING object is
    a genuine l1_bpe.BPEVocab. Returns (vocab, table).
    """
    cfg = C.CandleConfig()
    feats = _synthetic_features(seed=seed)
    table = BA.fit_alphabet(feats, cfg)

    seqs = _motif_unit_sequences(table, n_seqs=4000, seed=seed + 1)

    # frequency-greedy BPE over the pooled synthetic sequences (same algorithm as l1_bpe.fit_bpe,
    # inlined here only to avoid needing a DB; the RESULTING vocab is a genuine l1_bpe.BPEVocab).
    from collections import Counter
    n_base = L1.n_base_tagged(table)

    def _count_pairs(sequences):
        cnt = Counter()
        for s in sequences:
            for i in range(len(s) - 1):
                cnt[(s[i], s[i + 1])] += 1
        return cnt

    pair_counts = _count_pairs(seqs)
    merges = []
    p_min, max_merges = 10, 128
    for _m in range(max_merges):
        if not pair_counts:
            break
        best_pair, best_n = max(pair_counts.items(), key=lambda kv: kv[1])
        if best_n < p_min:
            break
        new_id = n_base + len(merges)
        merges.append(best_pair)
        L1._apply_merge(seqs, best_pair, new_id, pair_counts)

    vocab = L1.BPEVocab(merges, table)
    return vocab, table


def _synthetic_encode_input(table, fit_seed=0, n_windows=200):
    """A synthetic reduced-unit input to encode. Drawn from the SAME motif bank the vocab was fit on
    (seed = fit_seed + 1, matching _fit_inprocess_vocab) so the learned merges actually fire on it —
    this makes the round-trip gate exercise the BPE merge SERVING path, not just the leaf path.

    Also returns a synthetic feature corpus for the features->ids arm (covers the base-alphabet edge
    freeze independently of the unit path).
    """
    seqs = _motif_unit_sequences(table, n_seqs=n_windows, seed=fit_seed + 1)
    units_all = np.concatenate([np.asarray(s, dtype=np.int64) for s in seqs])
    feats = _synthetic_features(n_rows=40_000, seed=fit_seed + 99)
    return units_all.astype(np.int64), feats


# --------------------------------------------------------------------------- #
# the hard gate
# --------------------------------------------------------------------------- #
def test_roundtrip_byte_identical():
    vocab, table = _fit_inprocess_vocab(seed=0)
    units, feats = _synthetic_encode_input(table, fit_seed=0)

    # ---- ids_A: in-process encode (the eval-battery path) ----
    tok_in = TA.CandleTokenizer.from_bpe_vocab(vocab, cfg=C.CandleConfig())
    ids_A = tok_in.encode_units(units)

    # sanity: the synthetic input + vocab actually exercise the BPE merge SERVING path (else the gate
    # only proves the leaf path is frozen, not the harder merge path). Require >0 merge tokens.
    n_merge_tokens = int(np.sum(ids_A >= vocab.n_base))
    assert vocab.codebook_size > vocab.n_base, "vocab has no merges — test would be vacuous"
    assert ids_A.size > 0, "empty encode — test would be vacuous"
    assert n_merge_tokens > 0, (
        "ids_A contains NO merge tokens — the gate would only test the leaf path, not the BPE merge "
        "serving path. Fix the synthetic input so learned merges fire."
    )

    # ---- save -> load -> ids_B: frozen encode ----
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "l1_vocab.json")
        TA.save_vocab(vocab, path, cfg=C.CandleConfig())
        frozen = TA.load_vocab(path)
        tok_frozen = TA.CandleTokenizer(frozen)
        ids_B = tok_frozen.encode_units(units)

        # also exercise the FULL features->ids path through both (covers the base-alphabet edge freeze)
        ids_A_feat = tok_in.features_to_token_ids(feats[:64])
        ids_B_feat = tok_frozen.features_to_token_ids(feats[:64])

    ok_units = bool(np.array_equal(ids_A, ids_B))
    ok_feat = bool(np.array_equal(ids_A_feat, ids_B_feat))

    print("==== L1 frozen-tokenizer round-trip gate ====")
    print(f"codebook_size       : {tok_frozen.codebook_size} "
          f"(n_base={vocab.n_base}, merges={len(vocab.merges)})")
    print(f"input reduced units : {units.size}")
    print(f"ids_A (in-process)  : len {ids_A.size}, merge-tokens {n_merge_tokens}, "
          f"first8 {ids_A[:8].tolist()}")
    print(f"ids_B (frozen)      : len {ids_B.size}, first8 {ids_B[:8].tolist()}")
    print(f"np.array_equal(ids_A, ids_B)            = {ok_units}")
    print(f"np.array_equal(features path A/B)       = {ok_feat}")

    assert ok_units, (
        "FROZEN ENCODE NOT BYTE-IDENTICAL on encode_units — the freeze dropped/reordered the merge "
        "table or base-alphabet edges."
    )
    assert ok_feat, (
        "FROZEN ENCODE NOT BYTE-IDENTICAL on features->ids — the base-alphabet EDGES were not frozen "
        "faithfully."
    )
    print("ROUND-TRIP GATE: PASS (byte-identical)")
    return ok_units and ok_feat


def test_codebook_size_exposed():
    vocab, _ = _fit_inprocess_vocab(seed=1)
    tok = TA.CandleTokenizer.from_bpe_vocab(vocab)
    assert tok.codebook_size == vocab.codebook_size
    assert tok.codebook_size > 0
    print(f"codebook_size exposed: {tok.codebook_size}  (matches BPEVocab) — OK")
    return True


# --------------------------------------------------------------------------- #
# optional one-asset encode_asset smoke (only if the local DB is present)
# --------------------------------------------------------------------------- #
def smoke_encode_asset(db=None, asset_id=1087, max_windows=200):
    db = db or C.db_path()
    if not os.path.exists(db):
        print(f"[smoke] DB not found at {db} — skipping encode_asset smoke (gate already passed).")
        return None
    cfg = C.CandleConfig()
    # fit a quick frozen vocab on this one asset (small), save+load, encode the SAME asset.
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "vocab.json")
        vocab, _ = TA.fit_and_save(cfg, [asset_id], path,
                                   max_windows_per_asset=max_windows, p_min=20, max_merges=128,
                                   verbose=False)
        tok = TA.CandleTokenizer.from_path(path)
        chunks = list(tok.encode_asset(db, asset_id, cfg, max_windows_per_asset=max_windows))
    total = int(sum(c.size for c in chunks))
    print(f"[smoke] encode_asset(asset={asset_id}, max_windows={max_windows}): "
          f"{len(chunks)} chunks, {total} total token ids, codebook_size {tok.codebook_size}")
    assert len(chunks) > 0 and total > 0, "encode_asset returned an empty token stream!"
    print("[smoke] encode_asset: PASS (non-empty token-id stream)")
    return total


if __name__ == "__main__":
    test_roundtrip_byte_identical()
    test_codebook_size_exposed()
    smoke_encode_asset()
    print("\nALL TOKENIZER-API CHECKS DONE.")
