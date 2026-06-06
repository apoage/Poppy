"""stage3_regimes.py — TASK_797 Stage 3: cross-asset REGIME CLUSTERING on the candle-language token
substrate. MODEL-UNBOUND, CPU-ONLY. No neural net, no GPU, no trained embedding.

Stage 1 froze the L1 tokenizer; Stage 2 turned each ASSET into a token fingerprint -> N×N asset
similarity. Stage 3 zooms IN one level: it forgets the asset boundary and asks what recurring
*market states* the candle language sees across the whole universe. Each per-WINDOW token chunk from
`CandleTokenizer.encode_asset` IS that window's token bag — its little sentence of candle-shape words.
We pool a CAPPED, price-stratified sample of those window-bags across many assets, TF-IDF weight the
codebook columns (down-weight ubiquitous tokens so a regime is defined by its DISTINCTIVE shapes, not
the filler), and k-means them into K shared "market regimes".

Output (runs/797_regimes/RESULT.md + RESULT.json):
  - K regime centroids described by their TOP tokens (top token-ids by centroid weight),
  - per-regime population (window count + fraction of the pooled sample),
  - a per-asset regime MIX (each asset's distribution over the K regimes) so one can read off which
    assets live in which regimes.

STREAMING / MEMORY: window-bags are folded into O(codebook)-wide sparse rows one window at a time; we
never materialize all windows of all assets. The pooled matrix is hard-capped (--sample-cap rows,
default ~50k) via reservoir sampling, so RAM stays bounded regardless of universe size. The DB is
GAPPY (an asset can have many bars but no gap-free run >= window -> no tokens), so we OVERSAMPLE the
candidate pool and keep the first --n-assets that actually tokenize, exactly like fingerprint.py.

A frozen vocab need not pre-exist: if --vocab is absent we fit_and_save one on
config.DEFAULT_BUILD_ASSETS (mirrors fingerprint.py), so this runs standalone.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from . import config as C
from . import asset_select
from .tokenizer_api import CandleTokenizer, fit_and_save


# --------------------------------------------------------------------------- #
# per-window bag-of-tokens vectors (streamed, capped via reservoir sampling)
# --------------------------------------------------------------------------- #
def _window_bag(chunk: np.ndarray, K: int) -> np.ndarray:
    """One window's token-id chunk -> dense bag-of-token-ids count vector (K,), float64.

    `chunk` is a 1-D np.int64 array of token ids (one source window). The bag is np.bincount over the
    codebook — order-free, the window's word-frequency sentence. Ids are clipped defensively into
    [0,K) (mirrors eval_battery's clip) so a stray id never crashes a long stream."""
    ids = np.asarray(chunk, dtype=np.int64).ravel()
    if ids.size == 0:
        return np.zeros(K, dtype=np.float64)
    ids = np.clip(ids, 0, K - 1)
    return np.bincount(ids, minlength=K).astype(np.float64)


def collect_window_bags(tok: CandleTokenizer, db: str, assets, cfg, max_windows: int,
                        sample_cap: int, seed: int = 797, target_assets: int | None = None,
                        verbose: bool = True):
    """Pool a CAPPED, reservoir-sampled set of per-window bag-of-token vectors across `assets`.

    For each asset we stream `tok.encode_asset` (one chunk == one source window) and turn each window
    into its K-wide bag (`_window_bag`). The pooled matrix is reservoir-sampled to at most
    `sample_cap` rows: we keep every window until the reservoir is full, then replace a random slot
    with decaying probability — an unbiased uniform sample of the (unbounded) pooled window stream
    with O(sample_cap * K) RAM, never materializing all windows of all assets at once.

    `target_assets`: stop streaming candidates once this many have tokenized (the OVERSAMPLE-then-keep
    pattern from fingerprint.py — `assets` is over-provided to absorb gappy skips; we don't need to
    drain the whole pool). None == stream every candidate.

    Returns:
        X        : (M, K) float64 pooled window-bag matrix (M <= sample_cap),
        row_asset: (M,)   int64 asset id each kept row came from,
        kept     : list of asset ids that produced >=1 tokenized window,
        per_asset_n: {asset_id: n_windows_seen} (BEFORE the cap — the true window supply),
        skipped  : count of candidate assets that yielded no tokens (gappy).
    """
    K = int(tok.codebook_size)
    cap = int(sample_cap)
    rng = np.random.default_rng(int(seed))
    # reservoir of K-wide rows + the asset id behind each slot.
    reservoir = np.zeros((cap, K), dtype=np.float64)
    res_asset = np.zeros(cap, dtype=np.int64)
    filled = 0          # rows currently in the reservoir
    seen = 0            # total windows streamed so far (the reservoir denominator)
    kept, skipped = [], 0
    per_asset_n = {}
    tgt = int(target_assets) if target_assets else None

    for a in assets:
        if tgt is not None and len(kept) >= tgt:
            break
        n_this = 0
        try:
            for chunk in tok.encode_asset(db, int(a), cfg, max_windows_per_asset=int(max_windows)):
                bag = _window_bag(chunk, K)
                if bag.sum() <= 0:
                    continue
                n_this += 1
                if filled < cap:
                    reservoir[filled] = bag
                    res_asset[filled] = int(a)
                    filled += 1
                else:
                    # standard reservoir replacement: slot j in [0, seen] -> keep iff it lands in-range.
                    j = int(rng.integers(0, seen + 1))
                    if j < cap:
                        reservoir[j] = bag
                        res_asset[j] = int(a)
                seen += 1
        except Exception as e:  # gappy / unreadable asset -> skip (matches fingerprint.py)
            if verbose:
                print(f"[797-regimes]   asset {a}: SKIP ({e})", flush=True)
            skipped += 1
            continue
        if n_this > 0:
            kept.append(int(a))
            per_asset_n[int(a)] = n_this
        else:
            skipped += 1
        if verbose and len(kept) % 10 == 0 and len(kept) > 0:
            print(f"[797-regimes]   {len(kept)} assets, {min(seen, cap)} windows pooled "
                  f"(seen {seen}, {skipped} skipped)", flush=True)

    X = reservoir[:filled].copy()
    row_asset = res_asset[:filled].copy()
    return X, row_asset, kept, per_asset_n, skipped


# --------------------------------------------------------------------------- #
# TF-IDF weighting (down-weight ubiquitous tokens) + L2 row-normalize
# --------------------------------------------------------------------------- #
def tfidf_weight(X: np.ndarray):
    """TF-IDF weight a (M,K) window-bag count matrix; return (Xw, idf).

    TF  : per-row L1 normalize (a window's bag -> token frequencies; long and short windows comparable).
    IDF : log((M+1)/(df+1)) + 1 (smoothed, sklearn-style), df = #windows containing each token. Ubiquitous
          tokens (df ~ M) get idf ~ 1; rare/distinctive tokens get a larger weight -> regimes are defined
          by their DISTINCTIVE shapes, not by the filler tokens every window has. Columns never seen
          (df==0) get idf = log(M+1)+1 but stay all-zero so they don't perturb distances.
    Rows are then L2-normalized so k-means (Euclidean) acts on cosine geometry — the right metric for
    sparse frequency profiles (a window's REGIME is its token-mix direction, not its raw length)."""
    M, K = X.shape
    row_sum = X.sum(axis=1, keepdims=True)
    tf = X / np.maximum(row_sum, 1e-12)                      # L1 / term-frequency
    df = (X > 0).sum(axis=0).astype(np.float64)             # document frequency per token (K,)
    idf = np.log((M + 1.0) / (df + 1.0)) + 1.0              # smoothed idf
    Xw = tf * idf[None, :]
    # L2-normalize rows -> cosine geometry under Euclidean k-means.
    nrm = np.sqrt((Xw * Xw).sum(axis=1, keepdims=True))
    Xw = Xw / np.maximum(nrm, 1e-12)
    return Xw, idf


# --------------------------------------------------------------------------- #
# k-means (scipy.cluster.vq.kmeans2; vendored fallback if unavailable)
# --------------------------------------------------------------------------- #
def _kmeans_numpy(X: np.ndarray, k: int, seed: int, n_iter: int = 50):
    """Tiny Lloyd's k-means fallback (used only if scipy is unavailable). k-means++ init, fixed seed,
    empty-cluster reseed to the farthest point. Returns (centroids(k,K), labels(M,))."""
    rng = np.random.default_rng(int(seed))
    M = X.shape[0]
    k = min(int(k), M)
    # k-means++ seeding.
    centers = np.empty((k, X.shape[1]), dtype=np.float64)
    centers[0] = X[rng.integers(0, M)]
    d2 = ((X - centers[0]) ** 2).sum(axis=1)
    for c in range(1, k):
        probs = d2 / max(d2.sum(), 1e-12)
        centers[c] = X[rng.choice(M, p=probs)]
        d2 = np.minimum(d2, ((X - centers[c]) ** 2).sum(axis=1))
    labels = np.zeros(M, dtype=np.int64)
    for _ in range(int(n_iter)):
        # assign: argmin ||x - c||^2 == argmax x.c - 0.5||c||^2 (rows are L2-normed but keep it general).
        dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2) if M * k < 2_000_000 \
            else _chunked_dists(X, centers)
        new_labels = dists.argmin(axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            labels = new_labels
            break
        labels = new_labels
        for c in range(k):
            mask = labels == c
            if mask.any():
                centers[c] = X[mask].mean(axis=0)
            else:  # empty cluster -> reseed to the current farthest point.
                far = (((X - centers[labels]) ** 2).sum(axis=1)).argmax()
                centers[c] = X[far]
    return centers, labels


def _chunked_dists(X, centers, chunk=4096):
    out = np.empty((X.shape[0], centers.shape[0]), dtype=np.float64)
    for i in range(0, X.shape[0], chunk):
        xb = X[i:i + chunk]
        out[i:i + chunk] = ((xb[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return out


def cluster_regimes(Xw: np.ndarray, k: int, seed: int = 797):
    """k-means Xw into k regimes. Prefer scipy.cluster.vq.kmeans2 (fast C); fall back to a vendored
    numpy Lloyd if scipy is unavailable. Returns (centroids(k,K), labels(M,))."""
    k = int(min(k, Xw.shape[0]))
    try:
        from scipy.cluster.vq import kmeans2
        # '++' init, fixed seed via np seeding; 'warn' would raise on empty clusters, so use 'points'
        # to reseed empties to random points (keeps k non-degenerate on small smokes).
        centroids, labels = kmeans2(Xw, k, seed=int(seed), minit="++", missing="warn", iter=50)
        # kmeans2 can still return an empty cluster (label never assigned); repair by reseeding.
        present = set(int(x) for x in np.unique(labels))
        if len(present) < k:
            centroids, labels = _kmeans_numpy(Xw, k, seed)
        return np.asarray(centroids, dtype=np.float64), np.asarray(labels, dtype=np.int64)
    except Exception as e:  # scipy missing or kmeans2 hiccup -> vendored numpy k-means.
        print(f"[797-regimes] scipy kmeans2 unavailable/failed ({e}); using vendored numpy k-means",
              flush=True)
        return _kmeans_numpy(Xw, k, seed)


# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(
        description="TASK_797 Stage 3: cross-asset token regime clustering (CPU, model-unbound).")
    p.add_argument("--db", type=str, default=None, help="DB path (default config.db_path()).")
    p.add_argument("--vocab", type=str, default=None,
                   help="frozen L1 vocab .json; if absent, fit one on DEFAULT_BUILD_ASSETS and save it.")
    p.add_argument("--fit-assets", type=int, nargs="+", default=None,
                   help="assets to FIT the shared vocab on (default: config.DEFAULT_BUILD_ASSETS).")
    p.add_argument("--fit-max-windows", type=int, default=10000, help="windows/asset for vocab fitting.")
    p.add_argument("--n-assets", type=int, default=40,
                   help="size of the price-stratified panel that tokenizes (oversampled to absorb gaps).")
    p.add_argument("--max-windows", type=int, default=5000, help="windows/asset streamed into the pool.")
    p.add_argument("--sample-cap", type=int, default=50000,
                   help="hard cap on pooled window-bags (reservoir-sampled). RAM ~ cap*codebook floats.")
    p.add_argument("--k", type=int, default=8, help="number of market regimes (k-means K).")
    p.add_argument("--min-bars", type=int, default=20000, help="eligible-asset bar floor for the panel.")
    p.add_argument("--top-tokens", type=int, default=15, help="top token-ids reported per regime centroid.")
    p.add_argument("--seed", type=int, default=797, help="sampling + k-means seed (fixed; reproducible).")
    p.add_argument("--out-dir", type=str, default=None, help="output dir (default runs/797_regimes).")
    args = p.parse_args(argv)

    db = args.db if args.db else C.db_path()
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_dir = args.out_dir if args.out_dir else os.path.join(repo_root, "runs", "797_regimes")
    os.makedirs(out_dir, exist_ok=True)
    cfg = C.CandleConfig(db=db)
    t0 = time.time()

    # --- 1) frozen shared vocab (the "language") ---
    if args.vocab and os.path.exists(args.vocab):
        print(f"[797-regimes] loading frozen vocab {args.vocab}", flush=True)
        tok = CandleTokenizer.from_path(args.vocab)
        vocab_path = args.vocab
    else:
        fit_assets = tuple(args.fit_assets) if args.fit_assets else C.DEFAULT_BUILD_ASSETS
        vocab_path = args.vocab if args.vocab else os.path.join(out_dir, "l1_vocab.json")
        print(f"[797-regimes] fitting frozen L1 vocab on {len(fit_assets)} assets -> {vocab_path}",
              flush=True)
        _vocab, vocab_path = fit_and_save(cfg, list(fit_assets), vocab_path,
                                          max_windows_per_asset=int(args.fit_max_windows))
        tok = CandleTokenizer.from_path(vocab_path)
    K = int(tok.codebook_size)
    print(f"[797-regimes] codebook_size={K}", flush=True)

    # --- 2) price-stratified CANDIDATE pool, oversampled to absorb gappy-asset skips ---
    # The DB is GAPPY: a price-stratified pick lands on many sub-cent assets that have >=min_bars total
    # bars but NO gap-free run >= window -> no tokens. Like fingerprint.py we OVERSAMPLE the candidate
    # pool (here 6x the target — the low-price end skips hard) and keep streaming until --n-assets
    # actually tokenize (target_assets early-stop in collect_window_bags). The split stays price-
    # stratified so the kept panel still spans the price range.
    elig = len(asset_select.asset_stats(db, int(args.min_bars)))
    n_cand = min(max(args.n_assets * 6, args.n_assets), max(elig - 1, args.n_assets))
    half = n_cand // 2
    split = asset_select.select_split(db, half, n_cand - half, min_bars=int(args.min_bars))
    candidates = sorted(set(split["build"]) | set(split["holdout"]))
    print(f"[797-regimes] candidate pool: {len(candidates)} assets (target {args.n_assets} that tokenize)",
          flush=True)

    # --- 3) pool a capped, reservoir-sampled set of per-window bags across the panel ---
    # collect_window_bags streams candidates, skips the gappy ones, and STOPS once n_assets tokenize;
    # the reservoir bounds RAM to ~sample_cap*codebook floats regardless of universe size.
    bags, row_asset, kept, per_asset_n, skipped = collect_window_bags(
        tok, db, candidates, cfg, args.max_windows, args.sample_cap,
        seed=args.seed, target_assets=args.n_assets, verbose=True)
    M = bags.shape[0]
    print(f"[797-regimes] pooled {M} window-bags from {len(kept)} assets "
          f"({skipped} skipped gappy/no-tokens); {time.time()-t0:.0f}s", flush=True)
    if M < args.k:
        raise SystemExit(f"[797-regimes] too few pooled windows ({M}) for k={args.k}.")
    if len(kept) < 2:
        raise SystemExit(f"[797-regimes] too few assets tokenized ({len(kept)}) — lower --min-bars.")

    # --- 4) TF-IDF weight + L2 normalize, then k-means into K regimes ---
    Xw, idf = tfidf_weight(bags)
    centroids, labels = cluster_regimes(Xw, args.k, seed=args.seed)
    k_eff = centroids.shape[0]
    print(f"[797-regimes] clustered {M} windows into k={k_eff} regimes ({time.time()-t0:.0f}s)",
          flush=True)

    # --- 5) describe regimes: population + top tokens (by centroid weight) ---
    counts = np.bincount(labels, minlength=k_eff).astype(np.int64)
    order = np.argsort(counts)[::-1]                         # largest regime first for readability
    regimes = []
    top_n = int(args.top_tokens)
    for new_idx, c in enumerate(order):
        cen = centroids[c]
        top_ids = np.argsort(cen)[::-1][:top_n]
        top_ids = [int(t) for t in top_ids if cen[int(t)] > 0]
        regimes.append(dict(
            regime=new_idx,
            orig_label=int(c),
            n_windows=int(counts[c]),
            fraction=float(counts[c]) / float(max(M, 1)),
            top_tokens=top_ids,
            top_token_weights=[float(cen[int(t)]) for t in top_ids],
        ))
    relabel = {int(c): new_idx for new_idx, c in enumerate(order)}  # orig label -> readable rank

    # --- 6) per-asset regime MIX (distribution over regimes for each asset's pooled windows) ---
    asset_mix = {}
    for a in kept:
        mask = row_asset == int(a)
        n_a = int(mask.sum())
        if n_a == 0:
            continue
        labs = labels[mask]
        hist = np.bincount(labs, minlength=k_eff).astype(np.float64)
        # reorder to the readable regime ranking + normalize.
        mix = np.zeros(k_eff, dtype=np.float64)
        for orig, rank in relabel.items():
            mix[rank] = hist[orig]
        mix = mix / max(mix.sum(), 1e-12)
        asset_mix[int(a)] = dict(
            n_windows=n_a,
            mix=[float(x) for x in mix],
            dominant_regime=int(mix.argmax()),
        )

    # --- 7) cluster-quality diagnostics (non-degeneracy: not all windows in one cluster) ---
    frac = counts.astype(np.float64) / max(M, 1)
    nz = frac[frac > 0]
    pop_entropy_bits = float(-np.sum(nz * np.log2(nz))) if nz.size else 0.0
    max_pop_frac = float(frac.max()) if frac.size else 1.0
    n_nonempty = int((counts > 0).sum())
    # cosine inertia: mean (1 - cos) of each row to its centroid (lower == tighter regimes).
    cen_n = centroids / np.maximum(np.sqrt((centroids ** 2).sum(axis=1, keepdims=True)), 1e-12)
    cos_to_own = (Xw * cen_n[labels]).sum(axis=1)
    mean_within_cos = float(cos_to_own.mean())

    wall = time.time() - t0
    print(f"[797-regimes] done {wall:.0f}s | k={k_eff} nonempty={n_nonempty} "
          f"max_pop_frac={max_pop_frac:.3f} pop_entropy={pop_entropy_bits:.2f}bits "
          f"mean_within_cos={mean_within_cos:.3f}", flush=True)

    # --- write JSON + RESULT.md ---
    js = dict(
        task="TASK_797_stage3_token_regimes", db=db, codebook_size=K, vocab=vocab_path,
        k=k_eff, k_requested=int(args.k), n_assets=len(kept), assets=kept,
        n_windows_pooled=M, sample_cap=int(args.sample_cap),
        max_windows=int(args.max_windows), min_bars=int(args.min_bars),
        seed=int(args.seed), top_tokens=top_n,
        per_asset_window_supply=per_asset_n, skipped=skipped,
        regimes=regimes, asset_mix=asset_mix,
        diagnostics=dict(
            n_nonempty_regimes=n_nonempty, max_population_fraction=max_pop_frac,
            population_entropy_bits=pop_entropy_bits, mean_within_cluster_cosine=mean_within_cos,
        ),
        wall_s=wall,
    )
    with open(os.path.join(out_dir, "RESULT.json"), "w") as f:
        json.dump(js, f)

    lines = [
        "# TASK_797 Stage 3 — Cross-asset token regime clustering", "",
        f"_Model-unbound, CPU-only. Frozen L1 vocab (codebook {K}). "
        f"{M} window-bags pooled from {len(kept)} price-stratified assets -> k={k_eff} regimes._", "",
        "## Method",
        "- Each per-WINDOW token chunk from the FROZEN tokenizer (`CandleTokenizer.encode_asset`) -> a",
        f"  bag-of-token-ids vector (np.bincount over the {K}-code codebook).",
        f"- Pool a reservoir-sampled cap of {args.sample_cap} window-bags across the panel (RAM-bounded),",
        "  TF-IDF weight columns (down-weight ubiquitous tokens), L2-normalize (cosine geometry).",
        f"- k-means into k={args.k} regimes (scipy kmeans2 / vendored numpy fallback; seed {args.seed}).",
        f"- DB: `{db}` | windows/asset: {args.max_windows} | wall: {wall:.0f}s", "",
        "## Non-degeneracy (a collapsed clustering would put nearly all windows in one regime)",
        f"- non-empty regimes: **{n_nonempty}/{k_eff}**",
        f"- largest regime population fraction: **{max_pop_frac:.3f}** (1.0 == total collapse)",
        f"- population entropy: **{pop_entropy_bits:.2f} bits** (max {np.log2(max(k_eff,1)):.2f} = perfectly even)",
        f"- mean within-cluster cosine (tightness): **{mean_within_cos:.3f}**", "",
        "## Regimes (largest first): population + top tokens by centroid weight",
    ]
    for r in regimes:
        toks = ", ".join(str(t) for t in r["top_tokens"][:top_n])
        lines.append(
            f"- **regime {r['regime']}** — {r['n_windows']} windows ({100*r['fraction']:.1f}%) | "
            f"top tokens: {toks}")
    lines += ["", "## Per-asset regime mix (dominant regime + distribution over regimes)"]
    # sort assets by dominant regime then id for readability.
    for a in sorted(asset_mix, key=lambda x: (asset_mix[x]["dominant_regime"], x)):
        am = asset_mix[a]
        mixstr = " ".join(f"{i}:{v:.2f}" for i, v in enumerate(am["mix"]) if v > 0.005)
        lines.append(f"- **{a}** (n={am['n_windows']}, dom={am['dominant_regime']}): {mixstr}")
    lines += ["", "## Meta",
              f"- assets: {kept}", f"- vocab: {vocab_path}",
              "- full centroids/labels-derived mixes + diagnostics in RESULT.json"]
    with open(os.path.join(out_dir, "RESULT.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[797-regimes] wrote {out_dir}/RESULT.md + RESULT.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
