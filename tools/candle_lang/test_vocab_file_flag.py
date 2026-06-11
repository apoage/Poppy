"""
test_vocab_file_flag.py — acceptance gate for the audit_timeline.py `--vocab-file` flag (TASK_800
re-cert on a frozen, pre-fit L1 BPE vocab).

Run (CPU only, no real DB / real vocab needed — builds a tiny synthetic ohlcv DB + tiny vocab):
    JAX_PLATFORMS=cpu python -m tools.candle_lang.test_vocab_file_flag

The change under test: audit_timeline can LOAD a frozen L1 vocab from disk (via
CandleTokenizer.from_path) instead of fitting one in-process, so the prefix-stability (causality)
audit re-certifies the SERVABLE-WITH-LAG bound on the canonical runs/799_grammar vocab. Only the
L1 vocab SOURCE changes; the audit harness is byte-identical.

What this asserts:
  A1 (fit branch untouched when flag absent): fit_vocabs(..., l1_vocab_loaded=None) takes the
     in-process L1 fit path; load_l1_vocab_frozen is never invoked. Proven by behaviour: with the flag
     absent the L1 vocab equals the in-process fit (NOT the loaded one) and the source string says
     "in-process fit".
  A2 (load-roundtrip fidelity): fit a TINY L1 vocab in-process, fit_and_save it to JSON,
     CandleTokenizer.from_path it back, and assert the loaded vocab is byte-identical to the fit one.
       * Structural equality ALWAYS (same merges, same unit_length over all ids, same codebook_size,
         same n_base, same base-alphabet edges) — runs with no DB.
       * Token-id byte-identity through annotate_asset (the real server check) — runs only if a DB
         (the synthetic one built here) is available; we build one so this fires locally too.
  A3 (flag plumbs through): load_l1_vocab_frozen returns the loaded BPEVocab + the substrate cfg
     (window/halflife/sigma_floor from the loaded tokenizer, db/build_assets preserved), and
     fit_vocabs(..., l1_vocab_loaded=loaded) uses it verbatim (codebook_size matches the loaded one,
     NOT a re-fit). Prints codebook_size so the operator confirms 1376 on the server.
"""
from __future__ import annotations

import dataclasses
import os
import sqlite3
import tempfile

import numpy as np

from . import config as C
from . import base_alphabet as BA
from . import token_timeline as TT
from . import tokenizer_api as TK
from . import audit_timeline as AT
from .symbolic import l1_bpe as L1


# --------------------------------------------------------------------------- #
# tiny synthetic ohlcv DB (gap-free minute bars) so the real annotate_asset /
# fit_and_save path runs end-to-end on CPU with no real DB.
# --------------------------------------------------------------------------- #
def _make_synthetic_db(path, asset_id=1, n_rows=6000, seed=0):
    """Build a tiny gap-free (dt==60) ohlcv DB for ONE asset with structured, recurring candle shapes
    so the BPE merge serving path actually fires (recurring motifs -> frequent adjacent pairs)."""
    rng = np.random.default_rng(seed)
    if os.path.exists(path):
        os.remove(path)
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE ohlcv(asset_id INT, timestamp INT, open REAL, high REAL, low REAL, "
                "close REAL, volume REAL, trade_count INT, PRIMARY KEY(asset_id,timestamp))")
    # a recurring price path: drift + sinusoid + small noise -> repeating local candle shapes.
    t = np.arange(n_rows)
    base = 100.0 + 0.01 * t + 5.0 * np.sin(t / 23.0) + 2.0 * np.sin(t / 7.0)
    noise = 0.3 * rng.standard_normal(n_rows)
    close = base + noise
    open_ = np.empty(n_rows)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    hi = np.maximum(open_, close) + np.abs(0.2 * rng.standard_normal(n_rows))
    lo = np.minimum(open_, close) - np.abs(0.2 * rng.standard_normal(n_rows))
    rows = [(int(asset_id), int(60 * (i + 1)), float(open_[i]), float(hi[i]), float(lo[i]),
             float(close[i]), 1.0, 1) for i in range(n_rows)]
    con.executemany("INSERT INTO ohlcv(asset_id,timestamp,open,high,low,close,volume,trade_count) "
                    "VALUES(?,?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()
    return path


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _vocab_structural_signature(vocab):
    """A structural signature for byte-identity of two BPEVocabs WITHOUT a DB: the ordered merge table,
    the per-id unit_length over the full codebook, n_base, codebook_size, and the base-alphabet edges."""
    cb = int(vocab.codebook_size)
    unit_lens = tuple(int(vocab.unit_length(i)) for i in range(cb))
    table = vocab.table
    edges = tuple((ax, tuple(round(float(x), 12) for x in np.asarray(table["edges"][ax]).ravel()))
                  for ax in table["axes"])
    return dict(
        merges=tuple((int(a), int(b)) for (a, b) in vocab.merges),
        unit_lens=unit_lens,
        n_base=int(vocab.n_base),
        codebook_size=cb,
        edges=edges,
        axes=tuple(table["axes"]),
        cards=tuple((ax, int(table["cardinalities"][ax])) for ax in table["axes"]),
    )


# --------------------------------------------------------------------------- #
# the acceptance checks
# --------------------------------------------------------------------------- #
def run(db_asset=1):
    results = {}
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "synth.db")
        _make_synthetic_db(db, asset_id=db_asset, n_rows=6000, seed=0)
        cfg = C.CandleConfig(db=db)
        # NON-default substrate so A3 proves the cfg comes from the LOADED tokenizer, not a default.
        fit_cfg = dataclasses.replace(cfg, ewma_halflife=int(cfg.ewma_halflife) + 5)

        # ---- fit a TINY L1 vocab in-process + save it (the same fit_and_save the audit fit path uses)
        vocab_path = os.path.join(d, "tiny_L1_vocab.json")
        fit_vocab, _ = TK.fit_and_save(fit_cfg, [db_asset], vocab_path,
                                       max_windows_per_asset=4000, p_min=20, max_merges=64,
                                       verbose=False)
        print(f"[setup] fit tiny L1 vocab: codebook_size={fit_vocab.codebook_size} "
              f"merges={len(fit_vocab.merges)} n_base={fit_vocab.n_base} "
              f"(substrate halflife={fit_cfg.ewma_halflife})")

        # ============================ A2: load-roundtrip fidelity ============================
        loaded_vocab, loaded_cfg = AT.load_l1_vocab_frozen(vocab_path, cfg, verbose=False)

        # (A2a) structural byte-identity — always runs, no DB needed.
        sig_fit = _vocab_structural_signature(fit_vocab)
        sig_load = _vocab_structural_signature(loaded_vocab)
        struct_ok = (sig_fit == sig_load)
        assert struct_ok, "A2 FAIL: loaded vocab structurally differs from the fit vocab"
        assert int(loaded_vocab.codebook_size) == int(fit_vocab.codebook_size)
        assert int(loaded_vocab.n_base) == int(fit_vocab.n_base)
        assert tuple(loaded_vocab.merges) == tuple(fit_vocab.merges)
        print(f"[A2a] STRUCTURAL byte-identity: PASS "
              f"(merges, unit_length over all {fit_vocab.codebook_size} ids, n_base, edges all equal)")

        # (A2b) token-id byte-identity THROUGH annotate_asset — the real server check. Runs here because
        # we built a synthetic DB; on the server this is the same assertion against the real DB.
        tl_fit = TT.annotate_asset(db, db_asset,
                                   TK.CandleTokenizer.from_bpe_vocab(fit_vocab, fit_cfg), fit_cfg,
                                   max_feature_bars=6000)
        tl_load = TT.annotate_asset(db, db_asset,
                                    TK.CandleTokenizer.from_path(vocab_path), loaded_cfg,
                                    max_feature_bars=6000)
        tok_id_ok = bool(tl_fit.token_id.shape == tl_load.token_id.shape
                         and np.array_equal(tl_fit.token_id, tl_load.token_id)
                         and np.array_equal(tl_fit.valid, tl_load.valid))
        assert tok_id_ok, "A2b FAIL: annotate_asset token_id differs between fit and loaded vocab"
        n_valid = int(tl_fit.valid.sum())
        print(f"[A2b] TOKEN-ID byte-identity through annotate_asset: PASS "
              f"(bars={len(tl_fit)}, valid={n_valid}, all token_id + valid equal)")
        results["A2"] = dict(struct_ok=struct_ok, token_id_ok=tok_id_ok, n_valid=n_valid)

        # ============================ A3: flag plumbs through ============================
        # substrate cfg MUST come from the loaded tokenizer (the non-default halflife), NOT a default.
        assert int(loaded_cfg.ewma_halflife) == int(fit_cfg.ewma_halflife), \
            "A3 FAIL: loaded substrate cfg did not carry the fit halflife"
        assert int(loaded_cfg.window) == int(fit_cfg.window)
        assert abs(float(loaded_cfg.sigma_floor) - float(fit_cfg.sigma_floor)) < 1e-12
        # db / build_assets preserved from the RUNTIME cfg (so DB access stays on the real DB)
        assert loaded_cfg.db == cfg.db, "A3 FAIL: loaded cfg lost the runtime db path"
        assert tuple(loaded_cfg.build_assets) == tuple(cfg.build_assets)
        print(f"[A3] substrate cfg from LOADED tokenizer: PASS "
              f"(halflife={loaded_cfg.ewma_halflife} window={loaded_cfg.window} "
              f"sigma_floor={loaded_cfg.sigma_floor}); db/build_assets preserved from runtime")

        # fit_vocabs(..., l1_vocab_loaded=loaded) must use the loaded vocab VERBATIM (no re-fit).
        # (L8 is still fit; point build_assets at the synthetic asset so the L8 fit finds windows. On
        #  the server the real build_assets exist in the real DB — here only asset db_asset does.)
        vocabs_loaded = AT.fit_vocabs(loaded_cfg, [db_asset],
                                      max_windows_per_asset=2000, verbose=False,
                                      l1_vocab_loaded=loaded_vocab)
        assert vocabs_loaded["l1_vocab"] is loaded_vocab, \
            "A3 FAIL: fit_vocabs did not use the loaded L1 vocab object"
        assert int(vocabs_loaded["l1_vocab"].codebook_size) == int(loaded_vocab.codebook_size)
        print(f"[A3] fit_vocabs(l1_vocab_loaded=...) uses loaded vocab VERBATIM: PASS "
              f"(L1 codebook_size={vocabs_loaded['l1_vocab'].codebook_size}; "
              f"operator confirms 1376 on the server)")
        results["A3"] = dict(codebook_size=int(vocabs_loaded["l1_vocab"].codebook_size),
                             halflife=int(loaded_cfg.ewma_halflife))

        # ============================ A1: fit branch untouched (flag absent) ============================
        # With l1_vocab_loaded=None, fit_vocabs MUST fit L1 in-process (NOT use the loaded vocab). We
        # prove it by behaviour: the resulting L1 vocab equals the in-process fit and is a DIFFERENT
        # object from loaded_vocab. (load_l1_vocab_frozen is only reachable via --vocab-file in main.)
        vocabs_fit = AT.fit_vocabs(fit_cfg, [db_asset],
                                   max_windows_per_asset=4000, p_min=20, max_merges=64,
                                   verbose=False, l1_vocab_loaded=None)
        assert vocabs_fit["l1_vocab"] is not loaded_vocab, \
            "A1 FAIL: default path returned the loaded vocab object"
        # byte-identical to the standalone in-process fit (same data/params -> same vocab)
        sig_default = _vocab_structural_signature(vocabs_fit["l1_vocab"])
        a1_ok = (sig_default == sig_fit)
        assert a1_ok, "A1 FAIL: default-path L1 vocab differs from the standalone in-process fit"
        print(f"[A1] default path (l1_vocab_loaded=None) is the in-process fit: PASS "
              f"(codebook_size={vocabs_fit['l1_vocab'].codebook_size}, "
              f"byte-identical to standalone fit; loaded vocab NOT used)")
        results["A1"] = dict(ok=a1_ok, codebook_size=int(vocabs_fit["l1_vocab"].codebook_size))

    print("\nALL --vocab-file ACCEPTANCE CHECKS PASS (A1, A2a, A2b, A3).")
    return results


if __name__ == "__main__":
    run()
