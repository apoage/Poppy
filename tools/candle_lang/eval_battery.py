"""
Non-predictive evaluation battery for the candle-language panel (TASK_792, branch C).

This is the SHARED scorecard every panel member (L1 BPE / L2 ABBA / L3 SAX-grammar /
L4 FSQ-VQ / L5 Matrix-Profile) is judged by. It is deliberately NON-PREDICTIVE: the old
RQ0 next-token sign-probe (tokenizer_c0/run_gate.py:215-297) is DELETED. We do not ask
"do tokens predict returns"; we ask "is this a well-formed, transferable, faithful,
non-degenerate *language* of relative candle shapes". The five surviving research
questions (design 03 §battery):

    A0  rescale-invariance      a pure price rescale c->lam*c must yield IDENTICAL tokens
    A1  MDL / description length bits-per-candle vs a per-bar null (is the code economical?)
    A2  reconstruction fidelity can we rebuild the price path from the token stream?
    A8  cross-asset transfer    does a vocab fit on BUILD assets cover HELD-OUT assets,
                                beating a structure-destroying control?
    A9  length non-degeneracy   tokens must not all collapse to length-1 (or one mega-token)

HARD RULE (blocking #3, "host memory O(codebook), NEVER O(windows)"):
ALL reductions here are ONLINE. We accumulate per-code counts with np.bincount into a
fixed-size (codebook_size,) accumulator and stream every other statistic as running
sums / means. We NEVER list-concat the full token stream, NEVER materialize all windows.
Every function that consumes a stream takes an ITERATOR and folds it; the only host
arrays that grow with the codebook (small) are bincount accumulators.

Pure numpy. No JAX, no GPU. CPU-only — safe to import and run on the workstation.
"""
from __future__ import annotations

import numpy as np

from . import config as C


# --------------------------------------------------------------------------- #
# small private helpers (never a contracted signature)
# --------------------------------------------------------------------------- #
def _as_1d_int(a) -> np.ndarray:
    """Coerce a token-id chunk to a contiguous 1-D int64 array (cheap, no copy if already so)."""
    a = np.asarray(a)
    if a.ndim != 1:
        a = a.reshape(-1)
    if a.dtype != np.int64:
        a = a.astype(np.int64)
    return a


def _bincount_into(acc: np.ndarray, ids: np.ndarray, codebook_size: int) -> None:
    """In-place online accumulate of a token-id chunk into a fixed (codebook_size,) counter.
    This is THE mechanism that keeps host memory O(codebook), not O(windows)."""
    if ids.size == 0:
        return
    # guard: ids must be in [0, codebook_size); clip defensively rather than crash a long stream.
    lo = int(ids.min())
    hi = int(ids.max())
    if lo < 0 or hi >= codebook_size:
        ids = np.clip(ids, 0, codebook_size - 1)
    acc += np.bincount(ids, minlength=codebook_size).astype(np.int64)


def _entropy_bits_from_counts(counts: np.ndarray) -> float:
    """Shannon entropy (bits) of the empirical token distribution from a counts vector.
    Operates on the O(codebook) counts accumulator — never on the raw stream."""
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    p = counts.astype(np.float64) / total
    nz = p[p > 0.0]
    return float(-np.sum(nz * np.log2(nz)))


def _ratio_logmse_pair(recon_path: np.ndarray, true_path: np.ndarray):
    """For one (recon, true) price/return pair return (n, log_ratio_array, None), where
    log_ratio_array is the PER-BAR np.log(|recon|/|true|) (0 == perfect). The caller
    (a2_reconstruction) does the within-tol count and sum-of-squares reduction online, so this
    helper stays a pure per-pair transform. Scale-free: the log-ratio is rescale-invariant
    (mirrors the A2 substrate property). The trailing None is reserved/unused."""
    recon = np.asarray(recon_path, dtype=np.float64).reshape(-1)
    true = np.asarray(true_path, dtype=np.float64).reshape(-1)
    n = min(recon.size, true.size)
    if n == 0:
        return 0, 0.0, 0.0
    recon = recon[:n]
    true = true[:n]
    # log-ratio: 0 means perfect. Guard both sides away from zero/sign flips.
    r = np.maximum(np.abs(recon), C.DIV_EPS)
    t = np.maximum(np.abs(true), C.DIV_EPS)
    log_ratio = np.log(r / t)
    return n, log_ratio, None


# --------------------------------------------------------------------------- #
# A0 — rescale invariance (THE load-bearing property)
# --------------------------------------------------------------------------- #
def a0_rescale_invariance(tokenize_window_fn, sample_ohlc_windows, lambdas=(0.01, 100.0)) -> dict:
    """A0: a pure price rescale must leave the token UNCHANGED.

    tokenize_window_fn : callable (ohlc(W,4) float) -> token_id (int). The member's full
                         tokenize path for ONE window (volunit -> ... -> id).
    sample_ohlc_windows: list/iterable of raw-price OHLC windows, each (W,4) float
                         [open, high, low, close] (a SMALL diagnostic sample — A0 is a unit
                         check, not a stream fold).
    lambdas            : multiplicative rescales to apply to ALL of o/h/l/c.

    For each window and each lambda, tokenize x and lam*x; they MUST be equal. We report the
    agreement fraction over (window x lambda) trials and a hard pass at exact agreement.
    """
    agree = 0
    total = 0
    for w in sample_ohlc_windows:
        ohlc = np.asarray(w, dtype=np.float64)
        if ohlc.ndim != 2 or ohlc.shape[1] != 4:
            raise ValueError(f"a0: each window must be (W,4) [o,h,l,c]; got {ohlc.shape}")
        base_id = int(tokenize_window_fn(ohlc))
        for lam in lambdas:
            scaled_id = int(tokenize_window_fn(ohlc * float(lam)))
            agree += int(scaled_id == base_id)
            total += 1
    frac = float(agree) / float(total) if total else 1.0
    return dict(agree=frac, n_trials=int(total), n_agree=int(agree),
                lambdas=tuple(float(l) for l in lambdas), pass_=bool(total > 0 and agree == total),
                **{"pass": bool(total > 0 and agree == total)})


# --------------------------------------------------------------------------- #
# A1 — MDL / description length (online bincount over the FULL stream)
# --------------------------------------------------------------------------- #
def a1_mdl(token_id_iter, codebook_size, null_bits_per_bar, n_boundaries, codebook_bits,
           n_bars_covered=None) -> dict:
    """A1: how economically does this language describe the corpus?

    token_id_iter   : iterator yielding token-id CHUNKS (1-D arrays) over the whole corpus.
                      We fold it with an O(codebook) bincount accumulator — never concat it.
    codebook_size   : size of the token alphabet (== prod(fsq_levels), or N_SYMBOLS, etc.).
    null_bits_per_bar: bits/bar of the per-bar NULL model (e.g. log2(base-alphabet card)) —
                      the description length WITHOUT the language, per *candle bar covered*.
    n_boundaries    : total number of token boundaries / segment starts emitted by the member
                      (= number of tokens for fixed-length members; segment count for ABBA/BPE).
                      Used to charge a boundary/length cost so trivially-short tokens don't win.
    codebook_bits   : one-time bits to ship the vocabulary itself (model cost in the MDL sense).
    n_bars_covered  : TASK_796 — the true number of raw candle BARS the token stream covers
                      (= sum of token unit-lengths in bars). The per-bar null MUST be charged
                      over covered bars, NOT over token count: a variable-length member that
                      merges k bars into one token replaces k null-bars with one token, and the
                      null must reflect those k bars or `compression_ratio` is structurally pinned
                      below 1 (the v1 accounting artifact). If None, falls back to token count
                      (correct only for fixed-length-1 members).

    Returns the empirical coding cost in bits-per-candle (covered), plus the null and the
    boundary/codebook overheads, so callers can compare languages on a like-for-like MDL.
    """
    counts = np.zeros(int(codebook_size), dtype=np.int64)
    n_tokens = 0
    for chunk in token_id_iter:
        ids = _as_1d_int(chunk)
        _bincount_into(counts, ids, int(codebook_size))
        n_tokens += int(ids.size)

    total_tokens = int(counts.sum())
    # entropy (bits/token) of the empirical token distribution, from the counts accumulator.
    h_bits_per_token = _entropy_bits_from_counts(counts)
    # total content bits to code the token stream under its own empirical distribution.
    covered_bits = float(h_bits_per_token * total_tokens)
    # boundary cost: log2(codebook) per boundary is a generous upper bound; charge the
    # explicit boundary count the member reports (segment starts for variable-length members).
    # NOTE (TASK_796): boundary_bits is spec-mandated (L(boundaries)) and is RETAINED — the
    # "drop it" proposal was refuted; the real A1 fix is the per-bar (not per-token) null below.
    boundary_bits = float(int(n_boundaries) * np.log2(max(int(codebook_size), 2)))
    # the per-bar NULL: bits to describe each COVERED CANDLE BAR without the language. Charged
    # over the true bar coverage (TASK_796), so merging bars into longer tokens is rewarded.
    bars_covered = int(n_bars_covered) if n_bars_covered is not None else total_tokens
    null_bits = float(float(null_bits_per_bar) * bars_covered)

    denom = max(bars_covered, 1)
    bits_per_candle = (covered_bits + boundary_bits + float(codebook_bits)) / float(denom)

    return dict(
        bits_per_candle=float(bits_per_candle),
        covered_bits=float(covered_bits),
        null_bits=float(null_bits),
        boundary_bits=float(boundary_bits),
        codebook_bits=float(codebook_bits),
        entropy_bits_per_token=float(h_bits_per_token),
        n_tokens=int(total_tokens),
        n_bars_covered=int(bars_covered),
        codebook_size=int(codebook_size),
        # compression vs null (>1 means the language is cheaper than the per-bar null over the
        # SAME covered bars). Now a like-for-like per-bar comparison (TASK_796).
        compression_ratio=float(null_bits / max(covered_bits + boundary_bits + float(codebook_bits), C.DIV_EPS)),
    )


# --------------------------------------------------------------------------- #
# A2 — reconstruction fidelity (streamed online means)
# --------------------------------------------------------------------------- #
def a2_reconstruction(recon_pairs_iter, ratio_tol=0.02, mode="ratio", abs_tol=0.25) -> dict:
    """A2: can the price path be rebuilt from the token stream?

    recon_pairs_iter : iterator of (recon, true) pairs, streamed one window/run at a time.
    mode             : TASK_796 — how to score each bar.
        "ratio"    (legacy): per-bar |log(|recon|/|true|)| <= ratio_tol. ONLY valid for
                   strictly-POSITIVE paths (prices). Ill-posed for SIGNED vol-unit move paths:
                   it explodes near zero-crossings (|true|->0) and is blind to sign flips, which
                   is what made L1's A2 read 0.071 (a missing-offset artifact, not a serving bug).
        "additive" (TASK_796 default for signed mv members L1/L2/L3): per-bar
                   |recon - true| <= abs_tol, computed on the SIGNED per-bar quantity directly
                   (vol-units, std~1) — well-posed at zero-crossings AND sensitive to sign.
    ratio_tol        : multiplicative log-tolerance (mode="ratio").
    abs_tol          : additive vol-unit tolerance (mode="additive").

    Online accumulators (host memory O(1)). Use ONE consistent (mode, tol) across the panel so
    member A2 numbers are comparable.
    """
    mode = str(mode)
    n_bars = 0
    n_within = 0
    sum_sq_err = 0.0
    sum_abs_err = 0.0
    n_pairs = 0
    for recon_path, true_path in recon_pairs_iter:
        if mode == "additive":
            n, err = _additive_pair(recon_path, true_path)
            if n == 0:
                continue
            within = np.abs(err) <= float(abs_tol)
        else:
            n, log_ratio, _ = _ratio_logmse_pair(recon_path, true_path)
            if n == 0:
                continue
            err = log_ratio
            within = np.abs(log_ratio) <= float(ratio_tol)
        n_pairs += 1
        n_bars += int(n)
        n_within += int(np.count_nonzero(within))
        sum_sq_err += float(np.sum(err * err))
        sum_abs_err += float(np.sum(np.abs(err)))

    frac_within = float(n_within) / float(n_bars) if n_bars else 0.0
    mean_sq_err = float(sum_sq_err / n_bars) if n_bars else 0.0
    mean_abs_err = float(sum_abs_err / n_bars) if n_bars else 0.0
    return dict(
        frac_within_tol=float(frac_within),
        mean_log_ratio_mse=float(mean_sq_err),  # name kept for back-compat; = MSE in mode units
        mean_abs_err=float(mean_abs_err),
        mode=mode,
        ratio_tol=float(ratio_tol),
        abs_tol=float(abs_tol),
        n_bars=int(n_bars),
        n_pairs=int(n_pairs),
    )


def _additive_pair(recon, true):
    """Per-bar signed additive error (recon - true), well-posed for signed vol-unit paths."""
    recon = np.asarray(recon, dtype=np.float64).reshape(-1)
    true = np.asarray(true, dtype=np.float64).reshape(-1)
    n = min(recon.size, true.size)
    if n == 0:
        return 0, np.zeros(0, dtype=np.float64)
    return n, (recon[:n] - true[:n])


# --------------------------------------------------------------------------- #
# A8 — cross-asset transfer (vocab fit on BUILD must cover HOLDOUT, beat a control)
# --------------------------------------------------------------------------- #
def _top_k_megatokens(counts: np.ndarray, k: int) -> set:
    """Indices of the k most frequent tokens (the mega-tokens to EXCLUDE from transfer
    overlap so a single dominant 'flat-bar' code can't fake coverage)."""
    if k <= 0 or counts.size == 0:
        return set()
    k = min(int(k), int(counts.size))
    top = np.argpartition(counts, -k)[-k:]
    return set(int(i) for i in top.tolist())


def a8_cross_asset_transfer(build_vocab_fn, encode_assets_fn, build_assets, holdout_assets,
                            controls=("shuffle", "random_merge"), exclude_top_k=8,
                            n_perm_seeds=99, band=(0.10, 0.90), alpha=1.0,
                            codelen_margin=0.02, control=None, per_asset_matrix=False) -> dict:
    """A8 (TASK_796 v2): does a language fit on BUILD assets transfer to UNSEEN HOLDOUT assets?

    The v1 gate `bool(cov_real > cov_ctrl)` was a measurement artifact: mass coverage saturates to
    ~1.0 for both vocab AND any control in the small-codebook/large-corpus regime, so the strict
    zero-margin `>` decayed to a float-noise coin flip defaulting to False. This v2 gate is built
    around a NON-SATURATING continuous statistic and a margin'd multi-control verdict.

    HEADLINE  : cross-asset held-out transfer CODELENGTH = bits-per-(in-band)-token of the HOLDOUT
                stream under the BUILD vocab's Laplace-smoothed unigram distribution, restricted to
                a MID-FREQUENCY band (drops the universal head AND the singleton tail). Low == the
                holdout's token mass concentrates where build put its mass == genuine shared
                structure. Continuous codelength -> cannot saturate.
    CONTROLS  (real must beat ALL):
        (1) frequency-matched RANDOM CODEBOOK — keep build's frequency multiset, randomly reassign
            which id holds which frequency (n_perm_seeds permutations). Real beats it iff its
            codelength is BELOW every permutation (one-sided rank test, M=1/alpha-1=19 => p<=0.05
            for 19 seeds). This is the cheap, in-process, marginal-preserving null with teeth.
        (2..n) each substrate control in `controls` (e.g. 'shuffle' phase-shuffle, 'random_merge'):
            re-encode build+holdout through the structure-destroying transform and recompute the
            codelength; real must be lower by `codelen_margin` bits.
    `beats_control` (kept for member back-compat) == beats_all = (1) AND all of (2..n).

    Secondary, REPORTED (not solely gating): mid-frequency coverage / Jaccard / Spearman — these
    are de-saturated by the band but kept as supporting evidence, not the headline.

    All folds are O(codebook) bincount accumulators — streams are never materialized.
    """
    # back-compat: a single `control=` kwarg overrides the controls tuple.
    if control is not None:
        controls = (control,)
    controls = tuple(c for c in controls if c)

    vocab = build_vocab_fn(build_assets) if _wants_arg(build_vocab_fn) else build_vocab_fn()
    cb = int(_vocab_codebook_size(vocab))

    # --- 1) real substrate: build + holdout counts ---
    # per_asset_matrix: encode each asset SEPARATELY into its own O(codebook) count vector so we can
    # report (a) which holdout assets transfer and (b) the A-by-B build-asset x holdout-asset matrix.
    # The pooled aggregates are recovered by summing the per-asset vectors, so the headline metric is
    # identical to the non-matrix path (the real substrate is still encoded exactly once per asset).
    # bigram transition matrices accumulate alongside the unigram counts (ONE ordered pass each).
    build_trans = np.zeros((cb, cb), dtype=np.int64) if cb <= _BIGRAM_MAX_K else None
    hold_trans = np.zeros((cb, cb), dtype=np.int64) if cb <= _BIGRAM_MAX_K else None
    build_by_asset = hold_by_asset = None
    if per_asset_matrix:
        build_by_asset = {}
        for a in build_assets:
            uni, big = _fold_with_bigram(encode_assets_fn(vocab, [a], None), cb)
            build_by_asset[int(a)] = uni
            if build_trans is not None and big is not None:
                build_trans += big
        hold_by_asset = {}
        for a in holdout_assets:
            uni, big = _fold_with_bigram(encode_assets_fn(vocab, [a], None), cb)
            hold_by_asset[int(a)] = uni
            if hold_trans is not None and big is not None:
                hold_trans += big
        build_counts = np.sum(list(build_by_asset.values()), axis=0).astype(np.int64) \
            if build_by_asset else np.zeros(cb, dtype=np.int64)
        hold_counts = np.sum(list(hold_by_asset.values()), axis=0).astype(np.int64) \
            if hold_by_asset else np.zeros(cb, dtype=np.int64)
    else:
        build_counts, build_trans = _fold_with_bigram(encode_assets_fn(vocab, build_assets, None), cb)
        hold_counts, hold_trans = _fold_with_bigram(encode_assets_fn(vocab, holdout_assets, None), cb)

    mask = _nonzero_quantile_band(build_counts, lo_q=band[0], hi_q=band[1], exclude_top_k=exclude_top_k)
    cov_real, jac_real, sp_real = _transfer_stats(build_counts, hold_counts, mask)
    d_real, n_band = _transfer_codelength(build_counts, hold_counts, mask, alpha=alpha)

    # --- 2) freq-matched random-codebook permutation null (unigram, one-sided rank test) ---
    perm = _codelength_perm_controls(build_counts, hold_counts, mask, n_seeds=n_perm_seeds, alpha=alpha)
    perm_min = float(perm.min()) if perm.size else float("inf")
    perm_mean = float(perm.mean()) if perm.size else float("inf")
    beats_random_codebook = bool(perm.size and np.isfinite(d_real) and d_real < perm_min)

    # --- 2b) BIGRAM transition codelength (TASK_796): the metric a phase-shuffle CANNOT preserve ---
    dbi_real, n_trans = _bigram_codelength(build_trans, hold_trans, alpha=alpha)
    bperm = _bigram_perm_controls(build_trans, hold_trans, n_seeds=n_perm_seeds, alpha=alpha)
    bperm_min = float(bperm.min()) if bperm.size else float("inf")
    beats_random_codebook_bigram = bool(bperm.size and np.isfinite(dbi_real) and dbi_real < bperm_min)

    # --- 3) substrate controls: re-encode through each structure-destroying transform ---
    # Each control is scored on BOTH the unigram codelength (marginals — a phase-shuffle preserves
    # these, so real rarely wins) AND the bigram codelength (transitions — a phase-shuffle DESTROYS
    # these, so real beating the control here is the meaningful transfer signal).
    per_control = {}
    beats_substrate = True
    beats_substrate_bigram = True
    for ctl in controls:
        c_counts, c_trans = _fold_with_bigram(encode_assets_fn(vocab, build_assets, ctl), cb)
        h_counts, h_trans = _fold_with_bigram(encode_assets_fn(vocab, holdout_assets, ctl), cb)
        cmask = _nonzero_quantile_band(c_counts, lo_q=band[0], hi_q=band[1], exclude_top_k=exclude_top_k)
        d_ctl, _ = _transfer_codelength(c_counts, h_counts, cmask, alpha=alpha)
        cov_c, jac_c, sp_c = _transfer_stats(c_counts, h_counts, cmask)
        beats_c = bool(np.isfinite(d_real) and np.isfinite(d_ctl) and d_real < d_ctl - float(codelen_margin))
        beats_substrate = beats_substrate and beats_c
        dbi_ctl, _ = _bigram_codelength(c_trans, h_trans, alpha=alpha)
        beats_c_bi = bool(np.isfinite(dbi_real) and np.isfinite(dbi_ctl) and dbi_real < dbi_ctl - float(codelen_margin))
        beats_substrate_bigram = beats_substrate_bigram and beats_c_bi
        per_control[str(ctl)] = dict(
            transfer_bits=float(d_ctl), coverage=float(cov_c), jaccard=float(jac_c),
            spearman=float(sp_c), beats=beats_c,
            transfer_bits_bigram=float(dbi_ctl), beats_bigram=beats_c_bi,
        )

    # unigram verdict (marginal transfer) and the stronger BIGRAM verdict (transition transfer).
    beats_all = bool(beats_random_codebook and beats_substrate)
    # headline TASK_796 verdict: real transfers if it beats the random-codebook null on marginals AND
    # beats the structure-destroying substrate controls on TRANSITIONS (the bigram axis they cannot fake).
    beats_all_bigram = bool(beats_random_codebook_bigram and beats_substrate_bigram and (build_trans is not None))

    # --- 4) per-asset breakdown + A-by-B matrix (TASK_796 scale-up) ---
    # per_holdout: each holdout asset's transfer_bits under the POOLED build vocab + whether it beats
    #   its OWN freq-matched random-codebook null (the actionable "how many of the N holdout assets
    #   show real cross-asset transfer"). matrix[a][b]: holdout b under build asset a's own profile.
    per_holdout = None
    transfer_matrix = None
    holdout_transfer_rate = None
    if per_asset_matrix and build_by_asset and hold_by_asset:
        per_holdout = {}
        n_beat = 0
        for b, hc in hold_by_asset.items():
            d_b, _ = _transfer_codelength(build_counts, hc, mask, alpha=alpha)
            perm_b = _codelength_perm_controls(build_counts, hc, mask, n_seeds=n_perm_seeds, alpha=alpha)
            beats_b = bool(perm_b.size and np.isfinite(d_b) and d_b < perm_b.min())
            n_beat += int(beats_b)
            per_holdout[int(b)] = dict(transfer_bits=float(d_b),
                                       perm_min_bits=float(perm_b.min()) if perm_b.size else float("inf"),
                                       beats_random_codebook=beats_b)
        holdout_transfer_rate = float(n_beat) / float(max(len(hold_by_asset), 1))
        # A-by-B matrix: holdout b coded under build asset a's OWN profile (band fit on a).
        transfer_matrix = {}
        for a, bc_a in build_by_asset.items():
            mask_a = _nonzero_quantile_band(bc_a, lo_q=band[0], hi_q=band[1], exclude_top_k=exclude_top_k)
            row = {}
            for b, hc_b in hold_by_asset.items():
                d_ab, _ = _transfer_codelength(bc_a, hc_b, mask_a, alpha=alpha)
                row[int(b)] = float(d_ab)
            transfer_matrix[int(a)] = row

    return dict(
        # headline (non-saturating)
        transfer_bits=float(d_real),
        n_band_tokens=int(n_band),
        # freq-matched random-codebook rank test
        perm_min_bits=perm_min,
        perm_mean_bits=perm_mean,
        n_perm_seeds=int(n_perm_seeds),
        beats_random_codebook=beats_random_codebook,
        # per substrate control
        controls=list(per_control.keys()),
        per_control=per_control,
        beats_substrate=bool(beats_substrate),
        # BIGRAM / transition transfer (TASK_796) — the axis a phase-shuffle cannot preserve.
        transfer_bits_bigram=float(dbi_real),
        n_transitions=int(n_trans),
        bigram_perm_min_bits=bperm_min,
        beats_random_codebook_bigram=beats_random_codebook_bigram,
        beats_substrate_bigram=bool(beats_substrate_bigram),
        beats_all_bigram=beats_all_bigram,
        bigram_available=bool(build_trans is not None),
        # secondary de-saturated stats (reported, not the sole gate)
        coverage=float(cov_real),
        jaccard=float(jac_real),
        spearman=float(sp_real),
        # per-asset breakdown + A-by-B matrix (None unless per_asset_matrix=True)
        per_holdout=per_holdout,
        holdout_transfer_rate=holdout_transfer_rate,
        n_holdout_assets=(len(hold_by_asset) if hold_by_asset else None),
        transfer_matrix=transfer_matrix,
        # band/config
        band=tuple(float(b) for b in band),
        exclude_top_k=int(exclude_top_k),
        codebook_size=int(cb),
        # verdict (back-compat key: beats_control == beats_all)
        beats_all=beats_all,
        beats_control=beats_all,
    )


def _nonzero_quantile_band(build_counts: np.ndarray, lo_q=0.10, hi_q=0.90, exclude_top_k=8) -> np.ndarray:
    """Mid-frequency band mask: keep tokens whose BUILD count is within [lo_q, hi_q] quantiles of
    the NONZERO build counts, minus the top-k mega-tokens. De-saturates coverage/codelength by
    dropping the universal head (every-asset tokens) AND the singleton tail (noise tokens)."""
    cb = int(build_counts.shape[0])
    mask = np.zeros(cb, dtype=bool)
    nz = np.nonzero(build_counts > 0)[0]
    if nz.size == 0:
        return mask
    vals = build_counts[nz].astype(np.float64)
    lo = float(np.quantile(vals, float(lo_q)))
    hi = float(np.quantile(vals, float(hi_q)))
    keep = nz[(vals >= lo) & (vals <= hi)]
    mask[keep] = True
    for i in _top_k_megatokens(build_counts, exclude_top_k):
        mask[i] = False
    # degenerate guard: if the band emptied (tiny alphabet), fall back to all nonzero minus head.
    if not mask.any():
        mask[nz] = True
        for i in _top_k_megatokens(build_counts, exclude_top_k):
            mask[i] = False
    return mask


def _transfer_codelength(build_counts: np.ndarray, hold_counts: np.ndarray, mask: np.ndarray,
                         alpha=1.0):
    """Bits-per-(in-band)-token of the HOLDOUT stream under the BUILD vocab's Laplace(add-alpha)
    smoothed unigram distribution, restricted to `mask`. The non-saturating A8 headline."""
    bc = build_counts[mask].astype(np.float64)
    hc = hold_counts[mask].astype(np.float64)
    k = int(bc.size)
    n = float(hc.sum())
    if k == 0 or n <= 0.0:
        return float("inf"), 0.0
    p = (bc + float(alpha)) / (bc.sum() + float(alpha) * k)
    bits = -np.log2(p)
    return float(np.sum(hc * bits) / n), n


def _codelength_perm_controls(build_counts: np.ndarray, hold_counts: np.ndarray, mask: np.ndarray,
                              n_seeds=19, alpha=1.0, seed0=12345) -> np.ndarray:
    """Frequency-matched random-codebook null: keep build's frequency MULTISET but randomly
    reassign which in-band id holds which frequency, breaking the build<->hold id alignment while
    preserving marginals. Returns n_seeds holdout codelengths. Real beating ALL of these (one-sided
    rank test, 19 seeds => p<=0.05) is genuine shared-id structure, not matched marginals."""
    bc = build_counts[mask].astype(np.float64)
    hc = hold_counts[mask].astype(np.float64)
    k = int(bc.size)
    n = float(hc.sum())
    if k == 0 or n <= 0.0:
        return np.zeros(0, dtype=np.float64)
    rng = np.random.default_rng(int(seed0))
    out = np.empty(int(n_seeds), dtype=np.float64)
    for s in range(int(n_seeds)):
        bc_perm = bc[rng.permutation(k)]
        p = (bc_perm + float(alpha)) / (bc_perm.sum() + float(alpha) * k)
        out[s] = float(np.sum(hc * (-np.log2(p))) / n)
    return out


# --------------------------------------------------------------------------- #
# BIGRAM / transition codelength (TASK_796 upgrade) — the metric that makes the
# substrate-control axis meaningful. The unigram codelength depends only on token MARGINALS, which a
# phase-shuffle PRESERVES (so real can't beat the shuffle control). A bigram model codes each token
# conditioned on its predecessor, p_A(curr|prev) — i.e. shared TRANSITIONS — which a phase-shuffle
# DESTROYS. So real beating the shuffle/random-merge controls on bigram codelength is genuine
# transferable temporal structure, not matched marginals.
# --------------------------------------------------------------------------- #
_BIGRAM_MAX_K = 2048   # dense KxK transition matrix; skip bigram above this codebook size (memory).


def _fold_with_bigram(token_id_iter, codebook_size, max_k=_BIGRAM_MAX_K):
    """ONE ordered pass over a token-id stream -> (unigram_counts(K,), bigram_counts(K,K) or None).
    Counts adjacent (prev,curr) transitions, tracking the last id ACROSS chunk boundaries within this
    iterator. ONE iterator == ONE contiguous stream: callers fold per-asset (or accept a negligible
    cross-asset leak on a pooled stream). Returns bigram=None when K>max_k (dense KxK too large)."""
    K = int(codebook_size)
    uni = np.zeros(K, dtype=np.int64)
    big_flat = np.zeros(K * K, dtype=np.int64) if K <= int(max_k) else None
    prev = -1
    for chunk in token_id_iter:
        ids = _as_1d_int(chunk)
        if ids.size == 0:
            continue
        ids = np.clip(ids, 0, K - 1)
        uni += np.bincount(ids, minlength=K).astype(np.int64)
        if big_flat is not None:
            seq = ids if prev < 0 else np.concatenate(([prev], ids))
            if seq.size >= 2:
                # O(tokens) scatter-add into the persistent KxK accumulator (no per-chunk O(K^2)
                # bincount allocation — matters for the K=1376 BPE member).
                np.add.at(big_flat, seq[:-1] * K + seq[1:], 1)
            prev = int(ids[-1])
    big = big_flat.reshape(K, K) if big_flat is not None else None
    return uni, big


def _bigram_codelength(build_trans, hold_trans, alpha=1.0):
    """bits/transition of the HOLDOUT transitions under the BUILD Laplace-smoothed bigram model
    p(curr|prev). Low == holdout's transitions follow build's transition structure == shared temporal
    structure. Non-saturating; this is what a phase-shuffle cannot preserve."""
    if build_trans is None or hold_trans is None:
        return float("inf"), 0
    bt = build_trans.astype(np.float64)
    ht = hold_trans.astype(np.float64)
    n = float(ht.sum())
    if n <= 0.0:
        return float("inf"), 0
    K = bt.shape[0]
    row = bt.sum(axis=1, keepdims=True)
    P = (bt + float(alpha)) / (row + float(alpha) * K)        # conditional p(curr|prev)
    bits = -np.log2(P)
    return float(np.sum(ht * bits) / n), int(n)


def _bigram_perm_controls(build_trans, hold_trans, n_seeds=99, alpha=1.0, seed0=4242):
    """Freq-matched random-codebook null for the BIGRAM: keep build's SET of conditional
    distributions but randomly REASSIGN which 'prev' owns which conditional (row-permute the per-prev
    codelength), breaking the prev->curr alignment while preserving the conditional multiset. Real
    beating ALL perms == the SPECIFIC transition structure transfers (not generic transition density)."""
    if build_trans is None or hold_trans is None:
        return np.zeros(0, dtype=np.float64)
    bt = build_trans.astype(np.float64)
    ht = hold_trans.astype(np.float64)
    n = float(ht.sum())
    if n <= 0.0:
        return np.zeros(0, dtype=np.float64)
    K = bt.shape[0]
    row = bt.sum(axis=1, keepdims=True)
    bits = -np.log2((bt + float(alpha)) / (row + float(alpha) * K))   # (K,K)
    rng = np.random.default_rng(int(seed0))
    out = np.empty(int(n_seeds), dtype=np.float64)
    for s in range(int(n_seeds)):
        out[s] = float(np.sum(ht * bits[rng.permutation(K)]) / n)     # code prev under a random prev's dist
    return out


def a8_bigram_power_check(K=32, n=20000, rng_seed=0) -> dict:
    """POSITIVE CONTROL for the BIGRAM upgrade (TASK_796). A phase-shuffle preserves token marginals
    but destroys transitions, so the BIGRAM codelength MUST distinguish an ordered Markov stream from
    its shuffled copy, where the UNIGRAM cannot. Asserts: real (ordered build->ordered holdout) bigram
    codelength is LOWER than the shuffled control's, AND the unigram codelength is ~unchanged by
    shuffling. bigram_power_ok must be True or the substrate-control (bigram) verdict is meaningless."""
    rng = np.random.default_rng(int(rng_seed))
    M = rng.random((K, K)) ** 5 + 1e-3
    M /= M.sum(axis=1, keepdims=True)

    def _chain(nn, seed):
        r = np.random.default_rng(seed)
        s = np.empty(int(nn), dtype=np.int64)
        s[0] = int(r.integers(K))
        for t in range(1, int(nn)):
            s[t] = int(r.choice(K, p=M[s[t - 1]]))
        return s

    def _trans(s):
        t = np.zeros((K, K), dtype=np.int64)
        np.add.at(t, (s[:-1], s[1:]), 1)
        return t

    def _uni(s):
        return np.bincount(s, minlength=K).astype(np.int64)

    b = _chain(n, 1)
    h = _chain(n, 2)                                   # build + holdout share the SAME chain M
    bs = b.copy(); rng.shuffle(bs)                     # shuffle: keep marginals, destroy transitions
    hs = h.copy(); rng.shuffle(hs)

    d_bi_real, _ = _bigram_codelength(_trans(b), _trans(h))
    d_bi_ctrl, _ = _bigram_codelength(_trans(bs), _trans(hs))
    # unigram (band-free) cross-entropy: should be ~identical ordered vs shuffled (same marginals).
    def _uni_cl(bu, hu):
        p = (bu.astype(np.float64) + 1.0) / (bu.sum() + float(K))
        return float(np.sum(hu * (-np.log2(p))) / max(hu.sum(), 1))
    d_uni_real = _uni_cl(_uni(b), _uni(h))
    d_uni_ctrl = _uni_cl(_uni(bs), _uni(hs))

    bigram_distinguishes = bool(d_bi_real < d_bi_ctrl - 0.05)
    unigram_blind = bool(abs(d_uni_real - d_uni_ctrl) < 0.05)
    return dict(
        bigram_power_ok=bool(bigram_distinguishes and unigram_blind),
        bigram_distinguishes=bigram_distinguishes, unigram_blind=unigram_blind,
        d_bigram_real=d_bi_real, d_bigram_shuffled=d_bi_ctrl,
        d_unigram_real=d_uni_real, d_unigram_shuffled=d_uni_ctrl, K=int(K), n=int(n),
    )


def a8_power_check(codebook_size=512, n_seeds=19, alpha=1.0, band=(0.10, 0.90),
                   exclude_top_k=8, rng_seed=0) -> dict:
    """POSITIVE-CONTROL POWER HARNESS (TASK_796): prove the A8 codelength metric has power BEFORE
    any real verdict is trusted. Build synthetic count-pairs and assert the metric PASSES a
    KNOWN-transferable pair (holdout ~ build distribution) and FAILS a KNOWN-non-transferable pair
    (holdout independent of build). power_ok must be True or every real A8 verdict is meaningless.
    CPU-only, no DB — runs on the workstation."""
    rng = np.random.default_rng(int(rng_seed))
    cb = int(codebook_size)
    base = rng.random(cb) ** 3 + 1e-3
    build = (base / base.sum() * 2_000_000.0).astype(np.int64) + 1
    hold_T = (base / base.sum() * 500_000.0).astype(np.int64) + 1            # shares build's shape
    other = rng.random(cb) ** 3 + 1e-3
    hold_N = (other / other.sum() * 500_000.0).astype(np.int64) + 1          # independent shape
    mask = _nonzero_quantile_band(build, lo_q=band[0], hi_q=band[1], exclude_top_k=exclude_top_k)

    def _beats(hold):
        d, _ = _transfer_codelength(build, hold, mask, alpha=alpha)
        perm = _codelength_perm_controls(build, hold, mask, n_seeds=n_seeds, alpha=alpha)
        beats = bool(perm.size and np.isfinite(d) and d < perm.min())
        return beats, float(d), (float(perm.min()) if perm.size else None)

    pass_T, dT, mT = _beats(hold_T)
    pass_N, dN, mN = _beats(hold_N)
    power_ok = bool(pass_T and not pass_N)
    return dict(
        power_ok=power_ok,
        transferable_pass=pass_T, nontransferable_pass=pass_N,
        d_transferable=dT, perm_min_transferable=mT,
        d_nontransferable=dN, perm_min_nontransferable=mN,
        codebook_size=cb, n_seeds=int(n_seeds), band=tuple(float(b) for b in band),
    )


def _wants_arg(fn) -> bool:
    """Best-effort: does build_vocab_fn accept a positional arg? Default to yes if unsure."""
    try:
        import inspect
        sig = inspect.signature(fn)
        return len(sig.parameters) >= 1
    except (ValueError, TypeError):
        return True


def _vocab_codebook_size(vocab) -> int:
    """Extract the alphabet size from a member's vocab object, tolerant of several shapes."""
    for attr in ("codebook_size", "n_symbols", "vocab_size", "size"):
        v = getattr(vocab, attr, None)
        if isinstance(v, (int, np.integer)):
            return int(v)
    if isinstance(vocab, dict):
        for key in ("codebook_size", "n_symbols", "vocab_size", "size"):
            if key in vocab and isinstance(vocab[key], (int, np.integer)):
                return int(vocab[key])
    if hasattr(vocab, "__len__"):
        return int(len(vocab))
    raise ValueError("a8: cannot determine codebook size from vocab object; "
                     "expose .codebook_size / .n_symbols / .vocab_size.")


def _fold_counts(token_id_iter, codebook_size) -> np.ndarray:
    """Fold a token-id stream into an O(codebook) counts accumulator (shared A8 reducer)."""
    counts = np.zeros(int(codebook_size), dtype=np.int64)
    for chunk in token_id_iter:
        _bincount_into(counts, _as_1d_int(chunk), int(codebook_size))
    return counts


def _transfer_stats(build_counts: np.ndarray, hold_counts: np.ndarray, mask):
    """Coverage (holdout mass explained by build vocab), Jaccard of token sets, Spearman of
    shared-token frequencies — all from the two O(codebook) counts vectors, restricted to the
    mid-frequency band `mask` (TASK_796: replaces top-k-only exclusion, which could not
    de-saturate coverage). `mask` may be a boolean array (band) or an int k (legacy top-k drop)."""
    if np.isscalar(mask):  # legacy: an int exclude_top_k
        m = np.ones(build_counts.shape[0], dtype=bool)
        for i in _top_k_megatokens(build_counts, int(mask)):
            m[i] = False
        mask = m
    mask = np.asarray(mask, dtype=bool)

    bc = build_counts * mask
    hc = hold_counts * mask

    build_set = bc > 0
    hold_set = hc > 0

    # coverage: fraction of HOLDOUT token mass that lands on tokens the BUILD vocab knows.
    hold_total = float(hc.sum())
    covered_mass = float(hc[build_set].sum())
    coverage = covered_mass / hold_total if hold_total > 0 else 0.0

    # Jaccard of token-presence sets.
    inter = int(np.count_nonzero(build_set & hold_set))
    union = int(np.count_nonzero(build_set | hold_set))
    jaccard = float(inter) / float(union) if union > 0 else 0.0

    # Spearman rank-corr of frequencies on the SHARED support.
    shared = build_set & hold_set
    if int(np.count_nonzero(shared)) >= 3:
        spearman = _spearman(bc[shared].astype(np.float64), hc[shared].astype(np.float64))
    else:
        spearman = 0.0
    return coverage, jaccard, spearman


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (no scipy dependency): Pearson on the ranks, average-ranked."""
    def _rank(a):
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty(a.shape[0], dtype=np.float64)
        ranks[order] = np.arange(a.shape[0], dtype=np.float64)
        # average ranks for ties
        a_sorted = a[order]
        i = 0
        n = a.shape[0]
        while i < n:
            j = i
            while j < n and a_sorted[j] == a_sorted[i]:
                j += 1
            if j - i > 1:
                ranks[order[i:j]] = np.mean(ranks[order[i:j]])
            i = j
        return ranks
    rx = _rank(x)
    ry = _rank(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
    if denom < C.DIV_EPS:
        return 0.0
    return float(np.sum(rx * ry) / denom)


# --------------------------------------------------------------------------- #
# A9 — length non-degeneracy (no all-length-1 collapse, no single mega-token)
# --------------------------------------------------------------------------- #
def a9_length_nondegeneracy(length_counts) -> dict:
    """A9: the language must not be degenerate.

    length_counts : EITHER a dict {token_length:int -> count:int}, OR a np.bincount-style
                    array where index == token length (in base symbols / bars) and value ==
                    how many emitted tokens have that length. (For fixed-length members this is
                    a single spike at the window length, which is itself non-degenerate.)

    Two degeneracy failure modes:
      - all_len1 : every token is length 1 (numerosity reduction did nothing -> not a language).
      - max_token_share : one token id dominates the corpus. We DON'T have ids here, only
                    lengths, so we surface the share of the single most common LENGTH bucket as a
                    proxy mega-length detector, and report the length distribution's median.
    """
    if isinstance(length_counts, dict):
        lengths = np.array(sorted(length_counts.keys()), dtype=np.int64)
        counts = np.array([length_counts[int(k)] for k in lengths], dtype=np.int64)
    else:
        arr = np.asarray(length_counts, dtype=np.int64).reshape(-1)
        lengths = np.nonzero(arr)[0].astype(np.int64)
        counts = arr[lengths]

    total = int(counts.sum())
    if total == 0:
        return dict(median_len=0.0, max_token_share=1.0, all_len1=True, pass_=False,
                    n_lengths=0, **{"pass": False})

    # weighted median length
    order = np.argsort(lengths)
    lengths_s = lengths[order]
    counts_s = counts[order]
    cum = np.cumsum(counts_s)
    half = total / 2.0
    median_idx = int(np.searchsorted(cum, half, side="left"))
    median_idx = min(median_idx, lengths_s.shape[0] - 1)
    median_len = float(lengths_s[median_idx])

    # share of the most populated single length bucket (mega-length proxy).
    max_token_share = float(counts.max()) / float(total)

    # all tokens length 1?
    all_len1 = bool(lengths_s.size == 1 and int(lengths_s[0]) == 1)

    # pass: not collapsed to all-length-1 AND no single length bucket eats the whole corpus.
    passed = bool((not all_len1) and (max_token_share < 0.999))
    return dict(
        median_len=float(median_len),
        max_token_share=float(max_token_share),
        all_len1=bool(all_len1),
        n_lengths=int(lengths_s.size),
        total_tokens=int(total),
        pass_=passed,
        **{"pass": passed},
    )


if __name__ == "__main__":
    # CPU self-test (no DB, no GPU): exercise every battery function on synthetic streams so a
    # sanity import + run is meaningful. This is a SHAPE/contract test, not a real eval.
    rng = np.random.default_rng(0)

    # --- A0: a tokenizer that is rescale-invariant by construction (uses log-ratios) ---
    def toy_tokenize(ohlc):
        o, h, l, c = ohlc[:, 0], ohlc[:, 1], ohlc[:, 2], ohlc[:, 3]
        lr = np.log(np.maximum(c[1:], C.EPS) / np.maximum(c[:-1], C.EPS))
        return int(np.clip(np.round(np.sum(np.sign(lr))) + 64, 0, 127))
    windows = []
    for _ in range(16):
        steps = rng.normal(0, 0.01, size=33).cumsum()
        close = 50.0 * np.exp(steps)
        op = close * np.exp(rng.normal(0, 0.002, size=33))
        hi = np.maximum(op, close) * (1 + np.abs(rng.normal(0, 0.003, size=33)))
        lo = np.minimum(op, close) * (1 - np.abs(rng.normal(0, 0.003, size=33)))
        windows.append(np.stack([op, hi, lo, close], axis=1))
    a0 = a0_rescale_invariance(toy_tokenize, windows, lambdas=(0.01, 100.0))
    print("A0:", a0)
    assert a0["pass"], "A0 toy tokenizer should be rescale-invariant by construction"

    # --- A1: online MDL over a chunked token stream ---
    cb = 288
    def tok_stream():
        for _ in range(20):
            yield rng.integers(0, cb, size=4096)
    a1 = a1_mdl(tok_stream(), codebook_size=cb,
                null_bits_per_bar=np.log2(cb), n_boundaries=20 * 4096, codebook_bits=cb * 4.0)
    print("A1:", {k: round(v, 4) if isinstance(v, float) else v for k, v in a1.items()})

    # --- A2: streamed reconstruction pairs (price paths) ---
    def recon_pairs():
        for _ in range(50):
            true = 100.0 * np.exp(rng.normal(0, 0.01, size=63).cumsum())
            recon = true * np.exp(rng.normal(0, 0.005, size=63))
            yield recon, true
    a2 = a2_reconstruction(recon_pairs(), ratio_tol=0.02)
    print("A2:", a2)

    # --- A8: POSITIVE-CONTROL POWER HARNESS (TASK_796) — the metric must pass this to be trusted ---
    power = a8_power_check(codebook_size=512, n_seeds=19)
    print("A8 power-check:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in power.items()})
    assert power["power_ok"], (
        "A8 metric FAILED its positive-control power check: it must PASS a known-transferable pair "
        "and FAIL a known-non-transferable pair. Do not trust any real A8 verdict until this holds."
    )
    bipow = a8_bigram_power_check(K=32, n=20000)
    print("A8 bigram power-check:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in bipow.items()})
    assert bipow["bigram_power_ok"], (
        "BIGRAM metric FAILED its positive control: it must distinguish an ordered Markov stream from "
        "its phase-shuffled copy (where the unigram is blind). The substrate-control bigram verdict is "
        "meaningless until this holds."
    )

    # --- A8: end-to-end shape test through the member API (shared-structure build/holdout) ---
    _shared_p = (rng.random(64) ** 4 + 1e-3)
    _shared_p = _shared_p / _shared_p.sum()
    class _Vocab:
        codebook_size = 64
    def build_vocab_fn(assets=None):
        return _Vocab()
    def encode_assets_fn(vocab, assets, control):
        # real: every asset draws from the SAME shared distribution (shared candle 'language');
        # control: structure destroyed -> id alignment broken via an independent distribution.
        for _ in range(len(assets)):
            if control is None:
                yield rng.choice(64, size=4096, p=_shared_p)
            else:
                q = rng.random(64) ** 4 + 1e-3
                yield rng.choice(64, size=4096, p=q / q.sum())
    a8 = a8_cross_asset_transfer(build_vocab_fn, encode_assets_fn,
                                 build_assets=(1, 2, 3), holdout_assets=(4, 5),
                                 controls=("shuffle", "random_merge"), exclude_top_k=4, n_perm_seeds=19)
    print("A8:", {k: (round(v, 4) if isinstance(v, float) else v)
                  for k, v in a8.items() if k != "per_control"})

    # --- A9: length non-degeneracy (both dict and bincount forms) ---
    a9_dict = a9_length_nondegeneracy({1: 100, 2: 400, 3: 250, 4: 80})
    print("A9 (dict):", a9_dict)
    assert a9_dict["pass"]
    a9_collapse = a9_length_nondegeneracy({1: 1000})
    print("A9 (collapse):", a9_collapse)
    assert not a9_collapse["pass"]
    a9_arr = a9_length_nondegeneracy(np.bincount([2, 2, 3, 3, 3, 5, 64]))
    print("A9 (bincount):", a9_arr)

    print("\neval_battery self-test PASSED (A0/A1/A2/A8/A9).")
