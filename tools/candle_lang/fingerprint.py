"""fingerprint.py — TASK_797 Stage 2: per-asset candle-language token fingerprints -> N×N
asset-similarity surface. MODEL-UNBOUND, CPU-ONLY. No neural net, no GPU, no trained embedding.

Each asset is encoded with the FROZEN L1 tokenizer (tokenizer_api.CandleTokenizer) and folded into its
token fingerprint = (unigram histogram, bigram transition table) via eval_battery._fold_with_bigram (the
exact, A8-validated machinery). Asset-vs-asset distance is the Jensen–Shannon divergence between
fingerprints (a proper, symmetric, bounded metric):
    D_uni[a,b] = JS(unigram_a, unigram_b)   — token-frequency fingerprint (shape marginals)
    D_bi [a,b] = JS(bigram_a,  bigram_b)     — transition fingerprint (the phase-shuffle-proof axis)
plus the validated bigram cross-entropy EXCESS as a cross-check. Output: an N×N matrix, a nearest-asset
table, and an agglomerative dendrogram (scipy linkage) -> runs/797_fingerprint/RESULT.md + JSON.

This gives the candle-language tokens their first real, model-unbound use: an asset-similarity surface
(asset selection, regime-matched backtesting, transfer screening, near-duplicate detection), and it
validates the transfer claim on REAL asset relationships, outside the synthetic A8 power harness.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from . import config as C
from . import eval_battery as EB
from . import asset_select
from .tokenizer_api import CandleTokenizer, fit_and_save, load_vocab


# --------------------------------------------------------------------------- #
def _js(p_counts: np.ndarray, q_counts: np.ndarray) -> float:
    """Jensen–Shannon divergence (bits, in [0,1]) between two count vectors. Symmetric proper metric
    (sqrt is a metric); 0 == identical distribution, 1 == disjoint support."""
    p = np.asarray(p_counts, dtype=np.float64).ravel()
    q = np.asarray(q_counts, dtype=np.float64).ravel()
    sp, sq = p.sum(), q.sum()
    if sp <= 0 or sq <= 0:
        return 1.0
    p = p / sp
    q = q / sq
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return float(0.5 * _kl(p, m) + 0.5 * _kl(q, m))


def asset_fingerprint(tok: CandleTokenizer, db: str, asset_id: int, cfg, max_windows: int,
                      shuffle_rng=None):
    """(unigram(K,), bigram(K,K), bigram_shuffled(K,K)|None) token fingerprint for one asset.

    shuffle_rng (a np.random.Generator): if given, ALSO returns a bigram from a token-order-SHUFFLED
    copy of the stream — destroys transitions, KEEPS marginals. The validation control: if the real
    bigram similarity structure is genuine transition structure, the shuffled bigram must collapse to
    the unigram (its distances should track D_unigram, losing the independent structure)."""
    K = int(tok.codebook_size)
    if shuffle_rng is None:
        uni, big = EB._fold_with_bigram(
            tok.encode_asset(db, int(asset_id), cfg, max_windows_per_asset=int(max_windows)), K)
        return uni, big, None
    chunks = [np.asarray(c, dtype=np.int64) for c in
              tok.encode_asset(db, int(asset_id), cfg, max_windows_per_asset=int(max_windows))]
    stream = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int64)
    uni, big = EB._fold_with_bigram([stream], K)
    sh = stream.copy(); shuffle_rng.shuffle(sh)
    _, big_shuf = EB._fold_with_bigram([sh], K)
    return uni, big, big_shuf


def _bigram_self_entropy(big):
    """H(asset | itself) — the conditional entropy of an asset's own transitions (codelength floor)."""
    d, _ = EB._bigram_codelength(big, big)
    return d


# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(description="TASK_797: token-fingerprint asset-similarity surface (CPU).")
    p.add_argument("--db", type=str, default=None, help="DB path (default config.db_path()).")
    p.add_argument("--vocab", type=str, default=None,
                   help="frozen L1 vocab .json (load it); if absent, fit one on --fit-assets and save it.")
    p.add_argument("--n-assets", type=int, default=40, help="size of the fingerprint panel (diverse, price-stratified).")
    p.add_argument("--fit-assets", type=int, nargs="+", default=None,
                   help="assets to FIT the shared vocab on (default: config.DEFAULT_BUILD_ASSETS).")
    p.add_argument("--max-windows", type=int, default=5000, help="windows/asset for fingerprinting (CPU bound).")
    p.add_argument("--fit-max-windows", type=int, default=10000, help="windows/asset for vocab fitting.")
    p.add_argument("--min-bars", type=int, default=20000, help="eligible-asset bar floor for the panel.")
    p.add_argument("--out-dir", type=str, default=None, help="output dir (default runs/797_fingerprint).")
    p.add_argument("--k-clusters", type=int, nargs="+", default=[5, 8], help="flat-cluster cuts to report.")
    p.add_argument("--nearest", type=int, default=3, help="nearest neighbours per asset in the table.")
    p.add_argument("--shuffle-control", action="store_true",
                   help="ALSO build a token-order-shuffled bigram per asset (destroys transitions, keeps "
                        "marginals); the real bigram structure must collapse toward the unigram under it.")
    args = p.parse_args(argv)

    db = args.db if args.db else C.db_path()
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_dir = args.out_dir if args.out_dir else os.path.join(repo_root, "runs", "797_fingerprint")
    os.makedirs(out_dir, exist_ok=True)
    cfg = C.CandleConfig(db=db)
    t0 = time.time()

    # --- 1) frozen shared vocab (the "language") ---
    if args.vocab and os.path.exists(args.vocab):
        print(f"[797] loading frozen vocab {args.vocab}", flush=True)
        tok = CandleTokenizer.from_path(args.vocab)
        vocab_path = args.vocab
    else:
        fit_assets = tuple(args.fit_assets) if args.fit_assets else C.DEFAULT_BUILD_ASSETS
        vocab_path = args.vocab if args.vocab else os.path.join(out_dir, "l1_vocab.json")
        print(f"[797] fitting frozen L1 vocab on {len(fit_assets)} assets -> {vocab_path}", flush=True)
        _vocab, vocab_path = fit_and_save(cfg, list(fit_assets), vocab_path,
                                          max_windows_per_asset=int(args.fit_max_windows))
        tok = CandleTokenizer.from_path(vocab_path)
    K = int(tok.codebook_size)
    print(f"[797] codebook_size={K}", flush=True)

    # --- 2) pick a diverse, price-stratified CANDIDATE pool, oversampled to absorb gappy-asset skips
    # (an asset can have >=min_bars total bars but no gap-free run >= window -> no tokens). We keep the
    # first --n-assets candidates that actually tokenize. ---
    elig = len(asset_select.asset_stats(db, int(args.min_bars)))
    n_cand = min(max(args.n_assets * 3, args.n_assets), max(elig - 1, args.n_assets))
    half = n_cand // 2
    split = asset_select.select_split(db, half, n_cand - half, min_bars=int(args.min_bars))
    candidates = sorted(set(split["build"]) | set(split["holdout"]))
    print(f"[797] candidate pool: {len(candidates)} assets (target {args.n_assets} that tokenize)", flush=True)

    # --- 3) fingerprint candidates until we have --n-assets that produce tokens ---
    unis, bigs, bigs_shuf, kept = {}, {}, {}, []
    sh_rng = np.random.default_rng(797) if args.shuffle_control else None
    skipped = 0
    for i, a in enumerate(candidates):
        if len(kept) >= args.n_assets:
            break
        try:
            uni, big, big_shuf = asset_fingerprint(tok, db, a, cfg, args.max_windows, shuffle_rng=sh_rng)
        except Exception as e:
            print(f"[797]   asset {a}: SKIP ({e})", flush=True)
            skipped += 1
            continue
        if uni.sum() <= 0:
            skipped += 1
            continue
        unis[a] = uni
        bigs[a] = big
        if big_shuf is not None:
            bigs_shuf[a] = big_shuf
        kept.append(a)
        if len(kept) % 10 == 0:
            print(f"[797]   {len(kept)}/{args.n_assets} fingerprinted ({time.time()-t0:.0f}s, {skipped} skipped)", flush=True)
    n = len(kept)
    print(f"[797] {n} assets fingerprinted, {skipped} skipped (gappy/no-tokens)", flush=True)
    print(f"[797] {n} assets fingerprinted", flush=True)
    if n < 3:
        raise SystemExit("[797] too few assets fingerprinted to build a similarity surface.")

    # --- 4) N×N distance matrices: JS over unigram + bigram, plus bigram cross-entropy excess ---
    D_uni = np.zeros((n, n)); D_bi = np.zeros((n, n)); D_xe = np.zeros((n, n))
    self_ent = {a: _bigram_self_entropy(bigs[a]) for a in kept}
    for i in range(n):
        ai = kept[i]
        for j in range(i + 1, n):
            aj = kept[j]
            duni = _js(unis[ai], unis[aj])
            dbi = _js(bigs[ai], bigs[aj])
            # bigram cross-entropy EXCESS (validated A8 metric), symmetrized, >=0, self=0:
            xe_ij, _ = EB._bigram_codelength(bigs[ai], bigs[aj])   # bits of j under i
            xe_ji, _ = EB._bigram_codelength(bigs[aj], bigs[ai])   # bits of i under j
            xe = 0.5 * ((xe_ij - self_ent[aj]) + (xe_ji - self_ent[ai]))
            D_uni[i, j] = D_uni[j, i] = duni
            D_bi[i, j] = D_bi[j, i] = dbi
            D_xe[i, j] = D_xe[j, i] = max(xe, 0.0)

    # --- 5) nearest-asset table + agglomerative dendrogram (scipy) on the bigram-JS distance ---
    nearest = {}
    for i in range(n):
        order = np.argsort(D_bi[i])
        nn = [(int(kept[j]), float(D_bi[i, j])) for j in order if j != i][: args.nearest]
        nearest[int(kept[i])] = nn

    flat = {}
    linkage_list = None
    try:
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        condensed = squareform(D_bi, checks=False)
        Z = linkage(condensed, method="average")
        linkage_list = Z.tolist()
        for kc in args.k_clusters:
            labels = fcluster(Z, t=int(kc), criterion="maxclust")
            groups = {}
            for a, lab in zip(kept, labels):
                groups.setdefault(int(lab), []).append(int(a))
            flat[int(kc)] = {int(g): sorted(v) for g, v in groups.items()}
    except Exception as e:
        print(f"[797] scipy linkage unavailable ({e}); skipping dendrogram/flat clusters", flush=True)

    # --- 6) validation: bigram should add structure beyond unigram; structure must be real ---
    iu = np.triu_indices(n, 1)
    corr_uni_bi = float(np.corrcoef(D_uni[iu], D_bi[iu])[0, 1]) if n > 2 else float("nan")
    # spread of the off-diagonal distances (a flat/structureless surface -> ~0 spread)
    spread_bi = float(np.std(D_bi[iu]))
    spread_uni = float(np.std(D_uni[iu]))

    # shuffle-collapse control: with token order destroyed (marginals kept), the bigram distances must
    # COLLAPSE toward the unigram -> corr(uni, bi_shuffled) jumps toward ~1.0 and the bigram's
    # INDEPENDENT structure (1 - corr) vanishes. That confirms the real bigram structure is genuine
    # transition structure, not an artifact.
    shuffle = None
    if bigs_shuf and len(bigs_shuf) == n:
        D_bi_shuf = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = _js(bigs_shuf[kept[i]], bigs_shuf[kept[j]])
                D_bi_shuf[i, j] = D_bi_shuf[j, i] = d
        corr_uni_bishuf = float(np.corrcoef(D_uni[iu], D_bi_shuf[iu])[0, 1]) if n > 2 else float("nan")
        shuffle = dict(
            corr_unigram_bigram_real=corr_uni_bi,
            corr_unigram_bigram_shuffled=corr_uni_bishuf,
            indep_structure_real=float(1.0 - corr_uni_bi),
            indep_structure_shuffled=float(1.0 - corr_uni_bishuf),
            spread_bigram_shuffled=float(np.std(D_bi_shuf[iu])),
            collapsed=bool(corr_uni_bishuf > corr_uni_bi + 0.10),
        )
        print(f"[797] shuffle-collapse: corr(uni,bi) real={corr_uni_bi:.3f} -> shuffled="
              f"{corr_uni_bishuf:.3f}  (collapsed={shuffle['collapsed']})", flush=True)

    wall = time.time() - t0
    print(f"[797] done in {wall:.0f}s; D_bi spread={spread_bi:.4f} corr(uni,bi)={corr_uni_bi:.3f}", flush=True)

    # --- write JSON + RESULT.md ---
    js = dict(
        task="TASK_797_token_asset_fingerprint", db=db, codebook_size=K, vocab=vocab_path,
        n_assets=n, assets=kept, max_windows=int(args.max_windows),
        distance="jensen_shannon_bits", primary="bigram",
        D_unigram=D_uni.tolist(), D_bigram=D_bi.tolist(), D_bigram_xe_excess=D_xe.tolist(),
        nearest=nearest, flat_clusters=flat, linkage_average_Dbi=linkage_list,
        corr_unigram_bigram=corr_uni_bi, spread_bigram=spread_bi, spread_unigram=spread_uni,
        shuffle_control=shuffle, wall_s=wall,
    )
    with open(os.path.join(out_dir, "RESULT.json"), "w") as f:
        json.dump(js, f)

    lines = [
        "# TASK_797 — Token-fingerprint asset-similarity surface", "",
        f"_Model-unbound, CPU-only. Frozen L1 vocab (codebook {K}) over {n} price-stratified assets._", "",
        "## Method",
        "- Per-asset fingerprint = (unigram histogram, bigram transition table) from the FROZEN L1 tokenizer",
        "  (`tokenizer_api.CandleTokenizer` + `eval_battery._fold_with_bigram`).",
        "- Distance = Jensen–Shannon divergence (bits) between fingerprints; primary = **bigram** (transitions).",
        f"- DB: `{db}`  | windows/asset: {args.max_windows}  | wall: {wall:.0f}s", "",
        "## Validation",
        f"- bigram-JS distance spread (std of off-diagonal): **{spread_bi:.4f}** "
        f"(unigram {spread_uni:.4f}) — a structureless surface would be ~0.",
        f"- corr(unigram-JS, bigram-JS) = **{corr_uni_bi:.3f}** — bigram adds structure beyond marginals "
        "when this is well below 1.0.", "",
    ]
    if shuffle:
        lines += [
            "### Shuffle-collapse control (destroy token order, keep marginals)",
            f"- corr(unigram, bigram): real **{shuffle['corr_unigram_bigram_real']:.3f}** -> "
            f"shuffled **{shuffle['corr_unigram_bigram_shuffled']:.3f}**",
            f"- bigram INDEPENDENT structure (1-corr): real **{shuffle['indep_structure_real']:.3f}** -> "
            f"shuffled **{shuffle['indep_structure_shuffled']:.3f}**",
            f"- **collapsed = {shuffle['collapsed']}** — if True, the real bigram structure is genuine "
            "transition structure (it vanishes when order is destroyed), not an artifact.", "",
        ]
    if flat:
        lines.append("## Flat clusters (agglomerative, average-linkage on bigram-JS)")
        for kc in sorted(flat):
            lines.append(f"### K={kc}")
            for g in sorted(flat[kc]):
                mem = flat[kc][g]
                lines.append(f"- cluster {g} ({len(mem)}): {', '.join(map(str, mem[:30]))}"
                             f"{' …' if len(mem) > 30 else ''}")
            lines.append("")
    lines.append("## Nearest-asset table (3 nearest by bigram-JS)")
    for a in kept:
        nn = nearest[int(a)]
        lines.append(f"- **{a}** -> " + ", ".join(f"{b} ({d:.3f})" for b, d in nn))
    lines += ["", "## Meta", f"- assets: {kept}", f"- vocab: {vocab_path}",
              "- full N×N matrices (unigram/bigram/xe) + linkage in RESULT.json"]
    with open(os.path.join(out_dir, "RESULT.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[797] wrote {out_dir}/RESULT.md + RESULT.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
