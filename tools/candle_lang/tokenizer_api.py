"""
tokenizer_api.py — the FROZEN L1 tokenizer encode API (TASK_797 Stage 1, branch C).

Until now the L1 BPE tokenizer existed ONLY as private eval-battery plumbing inside
`symbolic/l1_bpe.py`: every run RE-FITS the base alphabet + BPE merges in-process and throws
the vocab away. Downstream consumers (the TASK_797 asset-fingerprint tool, later 793/794) need a
*serializable, frozen, stateless* encode path: fit ONCE, save, then encode any asset with a fixed
vocab — no re-fit, no DB scan to rebuild the alphabet.

This module is that contract. It is a THIN serialization + serving wrapper around l1_bpe's existing
primitives; it does NOT duplicate the encode logic. The frozen encode reconstructs a genuine
`l1_bpe.BPEVocab` from the saved (base-alphabet edges + ordered merge table + config) and calls the
SAME `l1_bpe.encode_units_bpe` / `l1_bpe.numerosity_reduce`, so frozen encode is BYTE-IDENTICAL to
the in-process encode by construction (the hard gate — see test_tokenizer_api.py).

What gets frozen (everything needed to go candles -> token ids with NO re-fit):
  * base-alphabet EDGES (the quantile breakpoints from base_alphabet.fit_alphabet) + the FIXED axis
    order + per-axis cardinalities + N_SYMBOLS  -> maps vol-unit features to base symbols.
  * the ORDERED BPE merge-rank table (rank 0 first) -> the leaf/merge flat-id layout + serving
    merge selection. n_base_tagged and codebook_size derive from these.
  * the relevant config (window, ewma_halflife, sigma_floor, N_SYMBOLS, n_base_tagged, codebook_size,
    plus the numerosity len-bucket params) so a loader can rebuild an identical CandleConfig and
    cross-check the layout it deserialized.

Serialization format: a single JSON file (vocab tables are tiny — a few hundred merges + ~288*N
edges — so JSON is plenty and human-inspectable; no .npz needed). Schema version tagged.

CPU ONLY. Pure numpy. No JAX / GPU (the L1 symbolic path never touches a device).
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import config as C
from . import base_alphabet as BA
from . import db_stream as DS
from .symbolic import l1_bpe as L1


SCHEMA = "candle_lang.l1_bpe.tokenizer"
SCHEMA_VERSION = 1


# --------------------------------------------------------------------------- #
# (de)serialize the base-alphabet table (base_alphabet.fit_alphabet output)
# --------------------------------------------------------------------------- #
def _table_to_jsonable(table: dict) -> dict:
    """base_alphabet table -> plain-JSON dict (edges arrays -> lists of float)."""
    return {
        "axes": list(table["axes"]),
        "cardinalities": {ax: int(table["cardinalities"][ax]) for ax in table["axes"]},
        "edges": {ax: [float(x) for x in np.asarray(table["edges"][ax]).ravel()]
                  for ax in table["axes"]},
        "n_symbols": int(BA.N_SYMBOLS(table)),
        "n_fit": int(table.get("n_fit", 0)),
    }


def _table_from_json(d: dict) -> dict:
    """Inverse of _table_to_jsonable — rebuilds the exact table base_alphabet.encode_symbols wants.

    Edges are restored as float64 np arrays (encode_symbols re-casts to float64 anyway). Axis order
    and cardinalities are taken verbatim from the file (the FIXED mixed-radix layout)."""
    axes = tuple(d["axes"])
    cards = {ax: int(d["cardinalities"][ax]) for ax in axes}
    edges = {ax: np.asarray(d["edges"][ax], dtype=np.float64) for ax in axes}
    return {
        "axes": axes,
        "cardinalities": cards,
        "edges": edges,
        "n_symbols": int(d["n_symbols"]),
        "n_fit": int(d.get("n_fit", 0)),
    }


# --------------------------------------------------------------------------- #
# save / load
# --------------------------------------------------------------------------- #
def save_vocab(vocab: "L1.BPEVocab", path: str, cfg: "C.CandleConfig | None" = None) -> str:
    """Serialize EVERYTHING needed to encode with this frozen vocab to a single JSON file.

    vocab : an l1_bpe.BPEVocab (from l1_bpe.fit_bpe or fit_and_save below).
    path  : output .json path (parent dirs created).
    cfg   : the CandleConfig the vocab was fit under (for window / ewma_halflife / sigma_floor /
            len-bucket params). If None, the module defaults (CandleConfig()) are recorded; pass the
            real cfg whenever the fit used non-default window/halflife/sigma_floor so the frozen
            encode rebuilds the SAME substrate.

    Layout written: base-alphabet edges + cardinalities + the ordered BPE merge table + config +
    derived sizes (n_symbols, n_base_tagged, codebook_size) + the numerosity len-bucket params, under
    a schema/version tag. Returns the path written.
    """
    if cfg is None:
        cfg = C.CandleConfig()
    table = vocab.table
    doc = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        # --- base alphabet (frozen quantile edges that map vol-unit -> base symbols) ---
        "base_alphabet": _table_to_jsonable(table),
        # --- ordered BPE merge-rank table (rank 0 first): list of [left_id, right_id] ---
        "merges": [[int(a), int(b)] for (a, b) in vocab.merges],
        # --- numerosity-reduction params (the leaf id layout depends on these) ---
        "numerosity": {
            "n_len_buckets": int(L1.N_LEN_BUCKETS),
            "len_edges": [int(x) for x in L1._LEN_EDGES],
        },
        # --- config needed to rebuild the substrate + cross-check the layout ---
        "config": {
            "window": int(cfg.window),
            "ewma_halflife": int(cfg.ewma_halflife),
            "sigma_floor": float(cfg.sigma_floor),
            "bar_seconds": int(C.BAR_SECONDS),
            "n_feat": int(C.N_FEAT),
        },
        # --- derived sizes (redundant; stored for fast loading + a load-time consistency check) ---
        "sizes": {
            "n_symbols": int(BA.N_SYMBOLS(table)),
            "n_base_tagged": int(vocab.n_base),
            "n_merges": int(len(vocab.merges)),
            "codebook_size": int(vocab.codebook_size),
        },
    }
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    return path


def load_vocab(path: str) -> dict:
    """Load a frozen-vocab JSON into a dict the CandleTokenizer consumes.

    Returns a frozen vocab dict:
        {
          "table"        : base_alphabet table (axes/cardinalities/edges/n_symbols),
          "vocab"        : a reconstructed l1_bpe.BPEVocab (genuine — encode_units_bpe runs on it),
          "cfg"          : a CandleConfig rebuilt from the saved window/halflife/sigma_floor,
          "codebook_size": int,
          "doc"          : the raw deserialized document (schema/version/sizes/...),
        }

    Reconstructing a real BPEVocab from (merges, table) is what makes the frozen encode byte-identical
    to the in-process encode: BPEVocab.__init__ derives ranks/children/n_base deterministically from
    the ordered merge list, so the serving merge selection in encode_units_bpe is reproduced exactly.
    """
    with open(path) as f:
        doc = json.load(f)
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"load_vocab: schema mismatch — got {doc.get('schema')!r}, want {SCHEMA!r}")
    if int(doc.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(
            f"load_vocab: schema_version mismatch — got {doc.get('schema_version')}, "
            f"want {SCHEMA_VERSION}"
        )

    # numerosity-reduction params must match the running module (the leaf-id layout baked into the
    # merge ids depends on N_LEN_BUCKETS / _LEN_EDGES; a mismatch would silently corrupt ids).
    num = doc.get("numerosity", {})
    if (int(num.get("n_len_buckets", L1.N_LEN_BUCKETS)) != int(L1.N_LEN_BUCKETS)
            or tuple(num.get("len_edges", L1._LEN_EDGES)) != tuple(L1._LEN_EDGES)):
        raise ValueError(
            "load_vocab: numerosity-reduction params in the file "
            f"(buckets={num.get('n_len_buckets')}, edges={num.get('len_edges')}) do not match the "
            f"running l1_bpe module (buckets={L1.N_LEN_BUCKETS}, edges={L1._LEN_EDGES}). "
            "The frozen leaf-id layout would not match — refit or align the module."
        )

    table = _table_from_json(doc["base_alphabet"])
    merges = [(int(a), int(b)) for (a, b) in doc["merges"]]
    vocab = L1.BPEVocab(merges, table)

    # cross-check the derived layout against the stored sizes (catches a corrupted / hand-edited file).
    sizes = doc.get("sizes", {})
    if sizes:
        if int(sizes.get("n_base_tagged", vocab.n_base)) != int(vocab.n_base):
            raise ValueError(
                f"load_vocab: n_base_tagged mismatch — file {sizes.get('n_base_tagged')} vs "
                f"reconstructed {vocab.n_base}"
            )
        if int(sizes.get("codebook_size", vocab.codebook_size)) != int(vocab.codebook_size):
            raise ValueError(
                f"load_vocab: codebook_size mismatch — file {sizes.get('codebook_size')} vs "
                f"reconstructed {vocab.codebook_size}"
            )

    cc = doc.get("config", {})
    cfg = C.CandleConfig(
        window=int(cc.get("window", C.DEFAULT_WINDOW)),
        ewma_halflife=int(cc.get("ewma_halflife", C.DEFAULT_EWMA_HALFLIFE)),
        sigma_floor=float(cc.get("sigma_floor", C.DEFAULT_SIGMA_FLOOR)),
    )

    return {
        "table": table,
        "vocab": vocab,
        "cfg": cfg,
        "codebook_size": int(vocab.codebook_size),
        "doc": doc,
    }


# --------------------------------------------------------------------------- #
# the frozen, stateless tokenizer
# --------------------------------------------------------------------------- #
class CandleTokenizer:
    """Frozen, stateless L1 tokenizer constructed from a loaded vocab. NO re-fit anywhere.

    Construct from a loaded-vocab dict (load_vocab(...)), a raw .json path, or directly from an
    in-process l1_bpe.BPEVocab (CandleTokenizer.from_bpe_vocab). Exposes:
        .encode_units(base_symbol_units)            -> 1-D np.int64 token ids
        .encode_asset(db, asset_id, cfg, ...)       -> yields 1-D np.int64 token-id chunks (streamed)
        .symbols_to_units(base_symbols)             -> numerosity-reduced leaf units (helper)
        .features_to_token_ids(feats)               -> one window's features -> token ids (helper)
        .codebook_size                              -> int

    All encode work delegates to l1_bpe primitives so serving == the eval-battery path exactly.
    """

    def __init__(self, frozen):
        if isinstance(frozen, str):
            frozen = load_vocab(frozen)
        self._frozen = frozen
        self.table = frozen["table"]
        self.vocab = frozen["vocab"]
        self.cfg = frozen.get("cfg") or C.CandleConfig()
        self.codebook_size = int(frozen["codebook_size"])

    # ----- constructors ----- #
    @classmethod
    def from_path(cls, path: str) -> "CandleTokenizer":
        return cls(load_vocab(path))

    @classmethod
    def from_bpe_vocab(cls, vocab: "L1.BPEVocab", cfg: "C.CandleConfig | None" = None) -> "CandleTokenizer":
        """Wrap an in-process BPEVocab directly (no save/load round-trip) — used by the round-trip
        test's 'in-process' arm and by callers that fit + encode in one process."""
        frozen = {
            "table": vocab.table,
            "vocab": vocab,
            "cfg": cfg or C.CandleConfig(),
            "codebook_size": int(vocab.codebook_size),
            "doc": None,
        }
        return cls(frozen)

    # ----- encode primitives (delegate to l1_bpe) ----- #
    def symbols_to_units(self, base_symbols) -> np.ndarray:
        """Base symbols (T,) -> numerosity-reduced leaf units (M,). Drops the per-unit bar coverage."""
        units, _bars = L1.numerosity_reduce(np.asarray(base_symbols, dtype=np.int64))
        return units

    def encode_units(self, base_symbol_units) -> np.ndarray:
        """Reduced base-symbol leaf units -> 1-D np.int64 token ids via the EXACT BPE merge-rank
        serving path (l1_bpe.encode_units_bpe — NOT greedy longest-match). O(n) per the frozen
        merge table; no re-fit. This is the byte-identical core the hard gate locks down."""
        return L1.encode_units_bpe(base_symbol_units, self.vocab)

    def features_to_token_ids(self, feats) -> np.ndarray:
        """One window's vol-unit features (W,N_FEAT) -> token ids. features -> base symbols via the
        FROZEN edges -> numerosity reduce -> BPE merge. Convenience for single-window encoders."""
        feats = np.asarray(feats, dtype=np.float64)
        syms = BA.encode_symbols(feats, self.table)
        units, _bars = L1.numerosity_reduce(syms)
        return L1.encode_units_bpe(units, self.vocab)

    # ----- real-asset streaming encode ----- #
    def encode_asset(self, db, asset_id, cfg=None, max_windows_per_asset=None, stride=1):
        """Stream a real asset's candles -> token-id chunks, ONE 1-D np.int64 chunk per source window.

        candles (db_stream.iter_windows) -> vol-unit features -> base symbols via the FROZEN edges
        -> numerosity reduce -> EXACT BPE merge. Streamed O(window) on the host (never materializes
        all of the asset's windows or its full token stream); db_stream itself bounds RAM via the
        tail-seek + per-process window cache.

        db       : DB path (str). The frozen edges are reused; the DB is read ONLY to fetch candles —
                   the alphabet is NOT re-fit.
        asset_id : int.
        cfg      : CandleConfig for the stream (window/halflife/sigma_floor MUST match what the vocab
                   was fit under, or the substrate differs). Defaults to the cfg recorded in the
                   frozen vocab (self.cfg) so a caller can simply pass the db + asset id.
        max_windows_per_asset : cap windows (keeps the MOST-RECENT; None = unbounded true stream).
        stride   : window stride (default 1).

        Yields 1-D np.int64 token-id arrays (empty chunks suppressed). To get one flat array:
            ids = np.concatenate(list(tok.encode_asset(db, aid, max_windows_per_asset=200)) or
                                  [np.zeros(0, np.int64)])
        """
        cfg = cfg if cfg is not None else self.cfg
        table = self.table
        vocab = self.vocab
        for chunk in DS.iter_windows(db, int(asset_id), cfg,
                                     max_windows=max_windows_per_asset, stride=int(stride)):
            X = chunk["X"]                                  # (n, W, N_FEAT) f32
            for w in range(X.shape[0]):
                syms = BA.encode_symbols(X[w], table)       # base symbols via FROZEN edges
                units, _bars = L1.numerosity_reduce(syms)
                ids = L1.encode_units_bpe(units, vocab)
                if ids.size:
                    yield ids


# --------------------------------------------------------------------------- #
# fit-and-save convenience (reuses l1_bpe.fit_alphabet + fit_bpe; no logic duplicated)
# --------------------------------------------------------------------------- #
def fit_and_save(cfg, build_assets, out_path,
                 max_windows_per_asset=20000, p_min=50, max_merges=512,
                 pooled_cap=2_000_000, verbose=True):
    """Fit the base alphabet + L1 BPE vocab on `build_assets`, then save_vocab to out_path.

    This is exactly the fit prologue of l1_bpe.main (alphabet on a bounded pooled sample, then BPE),
    factored so the TASK_797 tooling can mint a frozen vocab without running the whole eval battery.
    Returns (vocab, out_path).

    cfg                   : CandleConfig (db, window, halflife, sigma_floor).
    build_assets          : iterable of asset ids to fit/pool the vocab over.
    out_path              : where to write the frozen-vocab JSON.
    max_windows_per_asset : per-asset window cap for BOTH the alphabet pool and the BPE corpus.
    p_min / max_merges    : BPE floor + merge cap (l1_bpe.fit_bpe).
    pooled_cap            : cap on rows pooled for the alphabet fit (bounded RAM).
    """
    build_assets = [int(a) for a in build_assets]
    mwpa = int(max_windows_per_asset)

    # ----- 1) base alphabet on a bounded pooled vol-unit sample (mirrors l1_bpe.main step 1) ----- #
    pooled = []
    pooled_rows = 0
    for a in build_assets:
        for chunk in DS.iter_windows(cfg.db, a, cfg, max_windows=mwpa):
            flat = chunk["X"].reshape(-1, C.N_FEAT)
            pooled.append(flat)
            pooled_rows += flat.shape[0]
            if pooled_rows >= int(pooled_cap):
                break
        if pooled_rows >= int(pooled_cap):
            break
    if not pooled:
        raise RuntimeError("fit_and_save: no windows pooled for alphabet fit — check db/assets/window.")
    feats_pooled = np.concatenate(pooled, axis=0)
    table = BA.fit_alphabet(feats_pooled, cfg)
    del pooled, feats_pooled
    if verbose:
        print(f"[tokenizer_api] base alphabet fitted: N_SYMBOLS={BA.N_SYMBOLS(table)} "
              f"n_base_tagged={L1.n_base_tagged(table)}", flush=True)

    # ----- 2) BPE merges pooled across build (l1_bpe.fit_bpe) ----- #
    vocab = L1.fit_bpe(cfg, table, build_assets, mwpa,
                       p_min=int(p_min), max_merges=int(max_merges), verbose=verbose)

    # ----- 3) freeze ----- #
    save_vocab(vocab, out_path, cfg=cfg)
    if verbose:
        print(f"[tokenizer_api] saved frozen vocab -> {out_path} "
              f"(codebook_size={vocab.codebook_size}, merges={len(vocab.merges)})", flush=True)
    return vocab, out_path
