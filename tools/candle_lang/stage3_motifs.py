"""stage3_motifs.py — TASK_797 Stage 3: motif n-gram mining over the candle-language token substrate.
MODEL-UNBOUND, CPU-ONLY. No neural net, no GPU, no trained embedding — pure streaming token statistics
on top of the FROZEN L1 tokenizer (tokenizer_api.CandleTokenizer).

GOAL — the reusable "phrases" of the market language. Stage 1 froze the per-window token alphabet;
Stage 2 turned per-asset token fingerprints into an N×N similarity surface. Stage 3 asks a different
question: which token n-grams (candle "phrases") RECUR ACROSS MANY ASSETS *and* are OVER-REPRESENTED
vs chance? Those are the language's motifs — the shapes that aren't just frequent, but structurally
preferred over what an independent (unigram) token process would emit, and shared widely enough to be
a property of the market rather than of one ticker.

METHOD (all streaming, host memory O(codebook + tracked-ngrams), never O(all windows)):
  For each asset, encode it with the frozen tokenizer and concatenate its per-window token chunks into a
  single ordered stream (per asset; n-grams never straddle two assets). Over that stream:
    * accumulate the UNIGRAM counts (the chance model's marginals);
    * for each n in --ngram-n (default {2,3}) count token n-grams (hashed tuple -> int key);
    * record per-asset PRESENCE so document-frequency DF[ngram] = #assets whose stream contains it.
  After the fold, for each n-gram:
    * LIFT = p_observed(ngram) / p_expected_under_unigram_independence(ngram)
             = (count/total_ngrams_n) / Π_i (unigram_p(token_i))
      LIFT > 1 == the phrase occurs MORE than independent tokens would predict (a real bond between
      the component candles); LIFT < 1 == anti-association.
    * SCORE = DF * log(LIFT)  (over-represented AND widely shared; log so a huge lift on one asset can't
      dominate breadth, DF so a motif present in many assets outranks an asset-private quirk).
  Rank by SCORE (desc), report the top --top per n with example assets.

MEMORY GUARD — the count of DISTINCT n-grams is unbounded a priori, so we cap the tracked set: when a
per-n table exceeds a soft cap we PRUNE to the top-M by count (a count-min-ish floor; rare n-grams can't
be high-DF/high-lift motifs anyway, so pruning the long tail is safe for the *top* motif table). DF and
presence are tracked only for surviving n-grams. Periodic pruning keeps the table bounded regardless of
corpus size.

REUSE: tokenizer_api.CandleTokenizer / fit_and_save (frozen L1 encode + fit-if-absent), config,
asset_select (price-stratified candidate pool), eval_battery._entropy_bits_from_counts (codebook-space
entropy, never on the raw stream), the fingerprint.py oversample-to-absorb-gappy-skips + runs/797_*
output convention.

CLI:
  --db --vocab --n-assets --max-windows --min-bars --out-dir --ngram-n (default 2 3) --top
SMOKE (CPU, tiny): ~6 assets, --max-windows 300 -> proves it mines n-grams, computes DF + lift, ranks,
writes a non-empty motif table to runs/797_motifs/RESULT.md + RESULT.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

from . import config as C
from . import eval_battery as EB
from . import asset_select
from .tokenizer_api import CandleTokenizer, fit_and_save


# soft cap on DISTINCT n-grams tracked PER n. When exceeded we prune to PRUNE_KEEP by count. Rare
# n-grams (the pruned tail) cannot be high-DF/high-lift motifs, so the top table is unaffected; this
# only bounds host RAM on a large corpus. Default chosen so each per-n dict stays well under ~1e6 keys.
TRACK_CAP_DEFAULT = 400_000
PRUNE_KEEP_DEFAULT = 200_000


# --------------------------------------------------------------------------- #
# n-gram keying — pack a small token n-gram into ONE python int (exact, collision-free) so the count
# table is a plain dict[int,int] (cheap, hashable) rather than dict[tuple,int]. Key = mixed-radix of the
# component ids in base K (the codebook size). For n<=3 and K up to a few thousand this stays a native
# int; even at K=4096, n=3 -> < 7e10, comfortably an int. Decoded back to the id tuple for reporting.
# --------------------------------------------------------------------------- #
def _key_of(ids_slice: np.ndarray, K: int) -> int:
    k = 0
    for v in ids_slice:
        k = k * K + int(v)
    return k


def _decode_key(key: int, n: int, K: int) -> tuple:
    out = []
    for _ in range(n):
        out.append(int(key % K))
        key //= K
    return tuple(reversed(out))


# --------------------------------------------------------------------------- #
class _NgramTable:
    """Streaming counter for ONE n: total-count + per-key count + per-key document-frequency, with a
    bounded distinct-key set (periodic prune-to-top-M by count). Presence within the CURRENT asset is
    tracked in a per-asset set and flushed into DF at end_asset, so DF counts ASSETS, not occurrences."""

    def __init__(self, n: int, K: int, track_cap: int, prune_keep: int):
        self.n = int(n)
        self.K = int(K)
        self.track_cap = int(track_cap)
        self.prune_keep = int(prune_keep)
        self.count: dict[int, int] = {}     # key -> total occurrences across all assets
        self.df: dict[int, int] = {}        # key -> #assets containing it (document frequency)
        self.total = 0                      # total n-gram occurrences seen (for p_observed)
        self._cur: set[int] = set()         # keys seen in the current asset (presence, dedup)
        self.n_pruned = 0

    def add_stream(self, ids: np.ndarray) -> None:
        """Fold one asset's full token stream into this n's counts. O(len(stream)). Updates total +
        per-key count and records per-key presence for the current asset."""
        n = self.n
        if ids.size < n:
            return
        K = self.K
        cnt = self.count
        cur = self._cur
        # vectorized mixed-radix key for all length-n windows: key[t] = Σ ids[t+i] * K**(n-1-i)
        keys = np.zeros(ids.size - n + 1, dtype=np.int64)
        for i in range(n):
            keys = keys * K + ids[i: ids.size - n + 1 + i].astype(np.int64)
        self.total += int(keys.size)
        # count occurrences (np.unique on this asset's keys -> one dict update per distinct key)
        uk, uc = np.unique(keys, return_counts=True)
        for k, c in zip(uk.tolist(), uc.tolist()):
            cnt[k] = cnt.get(k, 0) + int(c)
            cur.add(k)
        if len(cnt) > self.track_cap:
            self._prune()

    def end_asset(self) -> None:
        """Flush the current asset's presence set into DF (each surviving key += 1), then reset."""
        df = self.df
        for k in self._cur:
            if k in self.count:           # key may have been pruned out mid-asset
                df[k] = df.get(k, 0) + 1
        self._cur = set()

    def _prune(self) -> None:
        """Prune distinct keys to the top prune_keep by count (drop the rare tail). DF entries for
        dropped keys are removed too so DF stays consistent with the surviving count table."""
        if len(self.count) <= self.prune_keep:
            return
        # partial top-K by count
        items = list(self.count.items())
        counts = np.fromiter((c for _, c in items), dtype=np.int64, count=len(items))
        keep_idx = np.argpartition(counts, len(items) - self.prune_keep)[-self.prune_keep:]
        keep = {items[i][0]: items[i][1] for i in keep_idx.tolist()}
        dropped = len(self.count) - len(keep)
        self.n_pruned += dropped
        self.count = keep
        self.df = {k: v for k, v in self.df.items() if k in keep}
        # the current asset's presence set must also drop pruned keys (else end_asset re-adds them)
        self._cur = {k for k in self._cur if k in keep}


# --------------------------------------------------------------------------- #
def mine_motifs(tok: CandleTokenizer, db: str, cfg, assets, ngram_ns, max_windows: int,
                track_cap: int, prune_keep: int, verbose: bool = True):
    """Stream every asset's frozen-tokenizer output and fold n-gram counts + DF + unigram marginals.

    Returns (tables, uni_counts, n_assets_used, kept_assets). `tables` is {n: _NgramTable}; `uni_counts`
    is the (K,) unigram accumulator (the chance model). One pass per asset; host memory O(codebook +
    Σ tracked n-grams). Assets that produce no tokens (gappy) are skipped — DF only counts assets that
    actually contributed a stream."""
    K = int(tok.codebook_size)
    tables = {int(n): _NgramTable(int(n), K, track_cap, prune_keep) for n in ngram_ns}
    uni = np.zeros(K, dtype=np.int64)
    kept = []
    t0 = time.time()
    for a in assets:
        chunks = list(tok.encode_asset(db, int(a), cfg, max_windows_per_asset=int(max_windows)))
        if not chunks:
            continue
        stream = np.concatenate([np.asarray(c, dtype=np.int64) for c in chunks])
        if stream.size == 0:
            continue
        stream = np.clip(stream, 0, K - 1)
        EB._bincount_into(uni, stream, K)
        for n in tables:
            tables[n].add_stream(stream)
            tables[n].end_asset()
        kept.append(int(a))
        if verbose and len(kept) % 10 == 0:
            print(f"[797m] {len(kept)} assets mined ({time.time()-t0:.0f}s)", flush=True)
    return tables, uni, len(kept), kept


def rank_motifs(table: _NgramTable, uni: np.ndarray, top: int, min_df: int = 2, min_count: int = 2):
    """Score + rank one n's tracked n-grams. For each surviving key compute LIFT (observed prob over
    unigram-independence expected prob) and SCORE = DF * log(LIFT) (over-represented AND widely shared).
    Returns a list of dicts sorted by score desc (top entries), and a couple of summary stats.

    A MOTIF is a phrase that recurs across MANY assets — a DF=1 n-gram is an asset-private quirk, not a
    shared motif, and with a sparse codebook its single-asset lift can be astronomically large and would
    swamp genuinely-shared phrases under DF·log(lift). So we GATE the ranked table on df >= min_df and
    count >= min_count (min_df is clamped to the asset count by the caller so a tiny smoke still yields a
    non-empty table). The full per-key stats remain in the table; only the *reported* motif list is gated.
    """
    K = table.K
    n = table.n
    total = float(table.total)
    uni_total = float(uni.sum())
    if total <= 0 or uni_total <= 0:
        return [], dict(n=n, total_ngrams=int(total), distinct_tracked=len(table.count),
                        n_pruned=table.n_pruned, min_df=int(min_df), min_count=int(min_count))
    p_uni = uni.astype(np.float64) / uni_total          # (K,) unigram marginal == the chance model

    rows = []
    for key, cnt in table.count.items():
        df = int(table.df.get(key, 0))
        if df < int(min_df) or int(cnt) < int(min_count):
            continue
        toks = _decode_key(key, n, K)
        p_obs = cnt / total
        # expected under unigram independence = Π p_uni[token_i]; guard against a zero-marginal token
        p_exp = 1.0
        ok = True
        for t in toks:
            pt = p_uni[t]
            if pt <= 0.0:
                ok = False
                break
            p_exp *= pt
        if not ok or p_exp <= 0.0:
            continue
        lift = p_obs / p_exp
        # SCORE: breadth (DF) times log-lift. log(lift)<=0 (under-represented) -> non-positive score; we
        # keep them but they sink to the bottom. A motif must be over-represented (lift>1) to score > 0.
        score = float(df) * math.log(lift) if lift > 0.0 else float("-inf")
        rows.append(dict(
            tokens=list(toks), n=n, count=int(cnt), df=df,
            lift=float(lift), log_lift=float(math.log(lift)) if lift > 0 else float("-inf"),
            p_obs=float(p_obs), p_exp=float(p_exp), score=score,
        ))
    rows.sort(key=lambda r: (r["score"], r["df"], r["count"]), reverse=True)
    stats = dict(
        n=n, total_ngrams=int(total), distinct_tracked=len(table.count),
        n_pruned=int(table.n_pruned), min_df=int(min_df), min_count=int(min_count),
        n_motifs_passing=len(rows),                                    # # over the DF/count gate
        max_df=int(max(table.df.values(), default=0)),                 # max DF over the FULL table
    )
    return rows[: int(top)], stats


# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(
        description="TASK_797 Stage 3: motif n-gram mining over candle-language tokens (CPU, model-unbound).")
    p.add_argument("--db", type=str, default=None, help="DB path (default config.db_path()).")
    p.add_argument("--vocab", type=str, default=None,
                   help="frozen L1 vocab .json (load it); if absent, fit one on DEFAULT_BUILD_ASSETS + save it.")
    p.add_argument("--n-assets", type=int, default=40, help="#assets to mine (price-stratified, oversampled).")
    p.add_argument("--max-windows", type=int, default=5000, help="windows/asset to encode (CPU bound).")
    p.add_argument("--fit-assets", type=int, nargs="+", default=None,
                   help="assets to FIT the vocab on if --vocab absent (default config.DEFAULT_BUILD_ASSETS).")
    p.add_argument("--fit-max-windows", type=int, default=10000, help="windows/asset for vocab fitting.")
    p.add_argument("--min-bars", type=int, default=20000, help="eligible-asset bar floor for the pool.")
    p.add_argument("--out-dir", type=str, default=None, help="output dir (default runs/797_motifs).")
    p.add_argument("--ngram-n", type=int, nargs="+", default=[2, 3], help="n-gram orders to mine.")
    p.add_argument("--top", type=int, default=40, help="top motifs to report per n.")
    p.add_argument("--track-cap", type=int, default=TRACK_CAP_DEFAULT,
                   help="soft cap on distinct n-grams tracked per n (prune-to-top when exceeded).")
    p.add_argument("--prune-keep", type=int, default=PRUNE_KEEP_DEFAULT,
                   help="#n-grams kept (top by count) after a prune.")
    p.add_argument("--examples", type=int, default=2, help="#example assets to list per top motif.")
    p.add_argument("--min-df", type=int, default=2,
                   help="a motif must appear in >= this many assets (gates out asset-private quirks). "
                        "Auto-clamped down so a tiny run still yields a non-empty table.")
    p.add_argument("--min-count", type=int, default=2,
                   help="a motif must occur >= this many times total (gates out one-off n-grams).")
    args = p.parse_args(argv)

    db = args.db if args.db else C.db_path()
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_dir = args.out_dir if args.out_dir else os.path.join(repo_root, "runs", "797_motifs")
    os.makedirs(out_dir, exist_ok=True)
    cfg = C.CandleConfig(db=db)
    ngram_ns = sorted(set(int(n) for n in args.ngram_n if int(n) >= 2))
    if not ngram_ns:
        raise SystemExit("[797m] --ngram-n must contain at least one n >= 2.")
    t0 = time.time()

    # --- 1) frozen shared vocab (the "language") — load if given/exists, else fit + save (fingerprint.py) ---
    if args.vocab and os.path.exists(args.vocab):
        print(f"[797m] loading frozen vocab {args.vocab}", flush=True)
        tok = CandleTokenizer.from_path(args.vocab)
        vocab_path = args.vocab
    else:
        fit_assets = tuple(args.fit_assets) if args.fit_assets else C.DEFAULT_BUILD_ASSETS
        vocab_path = args.vocab if args.vocab else os.path.join(out_dir, "l1_vocab.json")
        print(f"[797m] fitting frozen L1 vocab on {len(fit_assets)} assets -> {vocab_path}", flush=True)
        _vocab, vocab_path = fit_and_save(cfg, list(fit_assets), vocab_path,
                                          max_windows_per_asset=int(args.fit_max_windows))
        tok = CandleTokenizer.from_path(vocab_path)
    K = int(tok.codebook_size)
    print(f"[797m] codebook_size={K}; mining n-grams n={ngram_ns}", flush=True)

    # --- 2) price-stratified CANDIDATE pool, OVERSAMPLED to absorb gappy-asset skips (fingerprint.py
    # pattern: an asset can have >=min_bars total bars but no gap-free run >= window -> no tokens). We
    # mine candidates until --n-assets actually contribute a token stream. The skip-rate is HIGHER at
    # small --max-windows (we keep only the most-recent windows, and the cheap/sub-cent tail is gappy
    # there), so oversample more aggressively than the 3x fingerprint default AND keep a generous floor
    # (>= n_assets+24) so a tiny smoke run still has headroom. Capped at the eligible universe. ---
    elig = len(asset_select.asset_stats(db, int(args.min_bars)))
    n_cand = min(max(args.n_assets * 5, args.n_assets + 24), max(elig - 1, args.n_assets))
    half = n_cand // 2
    split = asset_select.select_split(db, half, n_cand - half, min_bars=int(args.min_bars))
    candidates = sorted(set(split["build"]) | set(split["holdout"]))
    print(f"[797m] candidate pool: {len(candidates)} assets (target {args.n_assets} that tokenize)", flush=True)

    # mine candidates, keeping the first --n-assets that tokenize (stream as we go; skip gappy ones).
    tables = {int(n): _NgramTable(int(n), K, int(args.track_cap), int(args.prune_keep)) for n in ngram_ns}
    uni = np.zeros(K, dtype=np.int64)
    kept, skipped = [], 0
    for a in candidates:
        if len(kept) >= args.n_assets:
            break
        try:
            chunks = list(tok.encode_asset(db, int(a), cfg, max_windows_per_asset=int(args.max_windows)))
        except Exception as e:
            print(f"[797m]   asset {a}: SKIP ({e})", flush=True)
            skipped += 1
            continue
        if not chunks:
            skipped += 1
            continue
        stream = np.concatenate([np.asarray(c, dtype=np.int64) for c in chunks])
        if stream.size == 0:
            skipped += 1
            continue
        stream = np.clip(stream, 0, K - 1)
        EB._bincount_into(uni, stream, K)
        for n in tables:
            tables[n].add_stream(stream)
            tables[n].end_asset()
        kept.append(int(a))
        if len(kept) % 10 == 0:
            print(f"[797m]   {len(kept)}/{args.n_assets} mined ({time.time()-t0:.0f}s, {skipped} skipped)", flush=True)
    n_used = len(kept)
    print(f"[797m] {n_used} assets mined, {skipped} skipped (gappy/no-tokens)", flush=True)
    if n_used < 2:
        raise SystemExit("[797m] too few assets produced tokens to mine motifs.")

    uni_entropy = EB._entropy_bits_from_counts(uni)
    uni_total = int(uni.sum())

    # --- 3) score + rank per n. Clamp --min-df DOWN to the asset count so a tiny smoke (few assets ->
    # most n-grams DF=1) still yields a non-empty motif table; a real run keeps the requested floor. ---
    eff_min_df = max(2, min(int(args.min_df), n_used)) if n_used >= 2 else 1
    if eff_min_df != int(args.min_df):
        print(f"[797m] clamping min-df {args.min_df} -> {eff_min_df} (only {n_used} assets)", flush=True)
    ranked = {}
    stats = {}
    for n in ngram_ns:
        rows, st = rank_motifs(tables[n], uni, int(args.top),
                               min_df=eff_min_df, min_count=int(args.min_count))
        ranked[n] = rows
        stats[n] = st
        print(f"[797m] n={n}: {st['distinct_tracked']} distinct tracked "
              f"(pruned {st['n_pruned']}), top score={rows[0]['score']:.3f} "
              f"lift={rows[0]['lift']:.2f} df={rows[0]['df']} tokens={rows[0]['tokens']}"
              if rows else f"[797m] n={n}: no motifs", flush=True)

    # --- 4) example assets per top motif: re-scan kept assets ONCE per n and record, for each top key,
    # the first --examples assets whose stream contains it (presence). Streaming, bounded: we only test
    # membership of the (small) top-key set per asset. ---
    examples = {n: {} for n in ngram_ns}
    if args.examples > 0:
        top_keys = {}
        for n in ngram_ns:
            tk = []
            for r in ranked[n]:
                key = _key_of(np.asarray(r["tokens"], dtype=np.int64), K)
                tk.append(key)
            top_keys[n] = set(tk)
        for a in kept:
            need = any(len(examples[n]) < len(top_keys[n]) or
                       any(len(examples[n].get(k, [])) < args.examples for k in top_keys[n])
                       for n in ngram_ns)
            if not need:
                break
            chunks = list(tok.encode_asset(db, int(a), cfg, max_windows_per_asset=int(args.max_windows)))
            if not chunks:
                continue
            stream = np.clip(np.concatenate([np.asarray(c, np.int64) for c in chunks]), 0, K - 1)
            for n in ngram_ns:
                if stream.size < n:
                    continue
                keys = np.zeros(stream.size - n + 1, dtype=np.int64)
                for i in range(n):
                    keys = keys * K + stream[i: stream.size - n + 1 + i]
                present = set(keys.tolist()) & top_keys[n]
                for key in present:
                    lst = examples[n].setdefault(key, [])
                    if len(lst) < args.examples and int(a) not in lst:
                        lst.append(int(a))

    wall = time.time() - t0
    print(f"[797m] done in {wall:.0f}s", flush=True)

    # --- 5) write JSON + RESULT.md ---
    def _rows_jsonable(n):
        out = []
        for r in ranked[n]:
            key = _key_of(np.asarray(r["tokens"], dtype=np.int64), K)
            rr = dict(r)
            rr["examples"] = examples[n].get(key, [])
            # JSON can't hold -inf; clamp the rare under-represented log_lift/score for serialization
            if not math.isfinite(rr["log_lift"]):
                rr["log_lift"] = None
            if not math.isfinite(rr["score"]):
                rr["score"] = None
            out.append(rr)
        return out

    js = dict(
        task="TASK_797_stage3_motif_ngram_mining", db=db, vocab=vocab_path,
        codebook_size=K, n_assets=n_used, assets=kept, skipped=skipped,
        max_windows=int(args.max_windows), ngram_n=ngram_ns, top=int(args.top),
        unigram_entropy_bits=float(uni_entropy), unigram_total_tokens=int(uni_total),
        track_cap=int(args.track_cap), prune_keep=int(args.prune_keep),
        stats={int(n): stats[n] for n in ngram_ns},
        motifs={int(n): _rows_jsonable(n) for n in ngram_ns},
        wall_s=wall,
    )
    with open(os.path.join(out_dir, "RESULT.json"), "w") as f:
        json.dump(js, f)

    lines = [
        "# TASK_797 Stage 3 — Motif n-gram mining (candle-language phrases)", "",
        f"_Model-unbound, CPU-only. Frozen L1 vocab (codebook {K}) over {n_used} price-stratified assets._", "",
        "## Method",
        "- Each asset is encoded with the FROZEN L1 tokenizer (`tokenizer_api.CandleTokenizer`); its",
        "  per-window token chunks are concatenated into one ordered per-asset stream (n-grams never",
        "  straddle two assets).",
        "- For each n we count token n-grams (streaming, hashed-tuple keys) + track per-asset **presence**",
        "  -> document-frequency **DF** = #assets containing the n-gram.",
        "- **LIFT** = p_observed(n-gram) / Π p_unigram(token_i)  (observed over unigram-independence chance).",
        "- **SCORE** = DF · log(LIFT) — over-represented (lift>1) **and** widely shared (high DF).",
        f"- Ranked table gated to motifs with DF >= {eff_min_df} and count >= {args.min_count} "
        "(asset-private one-offs excluded; a DF=1 phrase is not a shared motif).",
        f"- DB: `{db}` | windows/asset: {args.max_windows} | unigram entropy: {uni_entropy:.3f} bits"
        f" | total tokens: {uni_total:,} | wall: {wall:.0f}s", "",
    ]
    for n in ngram_ns:
        st = stats[n]
        rows = ranked[n]
        lines.append(f"## Top {min(args.top, len(rows))} motifs — n={n}")
        lines.append(f"_distinct {n}-grams tracked: {st['distinct_tracked']} "
                     f"(pruned {st['n_pruned']}); total occurrences: {st['total_ngrams']:,}; "
                     f"motifs passing gate: {st['n_motifs_passing']}; max DF: {st['max_df']}/{n_used}_")
        lines.append("")
        if not rows:
            lines.append("_(no motifs)_\n")
            continue
        lines.append("| rank | tokens | count | DF | lift | score | examples |")
        lines.append("|-----:|--------|------:|---:|-----:|------:|----------|")
        for i, r in enumerate(rows, 1):
            key = _key_of(np.asarray(r["tokens"], dtype=np.int64), K)
            ex = examples[n].get(key, [])
            toks = "(" + ", ".join(str(t) for t in r["tokens"]) + ")"
            sc = f"{r['score']:.3f}" if math.isfinite(r["score"]) else "—"
            lines.append(f"| {i} | `{toks}` | {r['count']} | {r['df']}/{n_used} | "
                         f"{r['lift']:.2f} | {sc} | {', '.join(map(str, ex)) if ex else '—'} |")
        lines.append("")
    lines += ["## Meta", f"- assets mined: {kept}", f"- vocab: {vocab_path}",
              f"- skipped (gappy/no-tokens): {skipped}",
              "- full ranked motif tables (with p_obs/p_exp/log_lift) in RESULT.json", ""]
    with open(os.path.join(out_dir, "RESULT.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[797m] wrote {out_dir}/RESULT.md + RESULT.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
