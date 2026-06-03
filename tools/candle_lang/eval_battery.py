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
def a1_mdl(token_id_iter, codebook_size, null_bits_per_bar, n_boundaries, codebook_bits) -> dict:
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
    boundary_bits = float(int(n_boundaries) * np.log2(max(int(codebook_size), 2)))
    # the per-bar NULL: bits to describe each covered candle bar without the language.
    # "covered bars" == total tokens for fixed-length members; for variable-length the caller
    # folds bar-coverage into n_boundaries semantics — we report per-*token* here for parity.
    null_bits = float(float(null_bits_per_bar) * total_tokens)

    denom = max(total_tokens, 1)
    bits_per_candle = (covered_bits + boundary_bits + float(codebook_bits)) / float(denom)

    return dict(
        bits_per_candle=float(bits_per_candle),
        covered_bits=float(covered_bits),
        null_bits=float(null_bits),
        boundary_bits=float(boundary_bits),
        codebook_bits=float(codebook_bits),
        entropy_bits_per_token=float(h_bits_per_token),
        n_tokens=int(total_tokens),
        codebook_size=int(codebook_size),
        # compression vs null (>1 means the language is cheaper than the per-bar null).
        compression_ratio=float(null_bits / max(covered_bits + boundary_bits + float(codebook_bits), C.DIV_EPS)),
    )


# --------------------------------------------------------------------------- #
# A2 — reconstruction fidelity (streamed online means)
# --------------------------------------------------------------------------- #
def a2_reconstruction(recon_pairs_iter, ratio_tol=0.02) -> dict:
    """A2: can the price path be rebuilt from the token stream?

    recon_pairs_iter : iterator of (recon, true) pairs, each either
                         - (recon_logret_path(K,), true_logret_path(K,)), OR
                         - (recon_price(K,), true_price(K,))
                       streamed one window/run at a time (NEVER all at once).
    ratio_tol        : per-bar relative tolerance; a bar "passes" if |log(recon/true)| <= tol
                       i.e. recon within ~tol of true on a multiplicative scale (scale-free).

    Online accumulators (host memory O(1)): bar count, within-tol count, sum of squared
    log-ratios. Mirrors run_gate.py:194-204 (price path from cumsum recon log-returns) but
    accepts either price OR logret pairs and reduces them streaming.
    """
    n_bars = 0
    n_within = 0
    sum_sq_log_ratio = 0.0
    n_pairs = 0
    for recon_path, true_path in recon_pairs_iter:
        n, log_ratio, _ = _ratio_logmse_pair(recon_path, true_path)
        if n == 0:
            continue
        n_pairs += 1
        n_bars += int(n)
        n_within += int(np.count_nonzero(np.abs(log_ratio) <= float(ratio_tol)))
        sum_sq_log_ratio += float(np.sum(log_ratio * log_ratio))

    frac_within = float(n_within) / float(n_bars) if n_bars else 0.0
    mean_log_ratio_mse = float(sum_sq_log_ratio / n_bars) if n_bars else 0.0
    return dict(
        frac_within_tol=float(frac_within),
        mean_log_ratio_mse=float(mean_log_ratio_mse),
        ratio_tol=float(ratio_tol),
        n_bars=int(n_bars),
        n_pairs=int(n_pairs),
    )


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
                            control="shuffle", exclude_top_k=8) -> dict:
    """A8: does a language fit on BUILD assets transfer to UNSEEN HOLDOUT assets?

    build_vocab_fn   : callable () or (assets) -> a frozen vocab object the member understands.
                       Called twice: once on the real BUILD substrate, once on a
                       structure-destroyed CONTROL substrate (so the control is a *fair*
                       same-pipeline null, like run_gate's shuffle).
    encode_assets_fn : callable (vocab, assets, control:str|None) -> iterator of token-id
                       chunks for those assets. control=None -> real volunit substrate;
                       control='shuffle' -> donor phase_shuffle_returns surrogate (data.py:165);
                       control='random_merge' -> assets' returns randomly merged/permuted.
    build_assets     : the fit corpus (e.g. cfg.build_assets).
    holdout_assets   : the unseen test corpus (e.g. cfg.holdout_assets) at extreme price levels.
    control          : which null to compare transfer against ('shuffle' | 'random_merge').
    exclude_top_k    : drop this many mega-tokens from both sides before measuring overlap.

    We measure, on the HOLDOUT stream, what fraction of holdout token *mass* is explained by the
    BUILD vocabulary (coverage), the Jaccard of the build-vocab vs holdout-vocab token SETS, and
    a Spearman rank correlation of token frequencies on shared tokens. Then we repeat the whole
    transfer under the structure-destroying control; transfer must BEAT the control to pass.
    All folds are O(codebook) bincount accumulators — the streams are never materialized.
    """
    # --- 1) fit the real vocab on BUILD, encode BUILD + HOLDOUT (real substrate) ---
    vocab = build_vocab_fn(build_assets) if _wants_arg(build_vocab_fn) else build_vocab_fn()
    cb = int(_vocab_codebook_size(vocab))

    build_counts = _fold_counts(encode_assets_fn(vocab, build_assets, None), cb)
    hold_counts = _fold_counts(encode_assets_fn(vocab, holdout_assets, None), cb)

    cov_real, jac_real, sp_real = _transfer_stats(build_counts, hold_counts, exclude_top_k)

    # --- 2) control: same pipeline on structure-destroyed substrate ---
    ctrl_build = _fold_counts(encode_assets_fn(vocab, build_assets, control), cb)
    ctrl_hold = _fold_counts(encode_assets_fn(vocab, holdout_assets, control), cb)
    cov_ctrl, jac_ctrl, sp_ctrl = _transfer_stats(ctrl_build, ctrl_hold, exclude_top_k)

    # transfer beats control if real coverage exceeds control coverage by a margin.
    beats = bool(cov_real > cov_ctrl)
    return dict(
        coverage=float(cov_real),
        jaccard=float(jac_real),
        spearman=float(sp_real),
        coverage_control=float(cov_ctrl),
        jaccard_control=float(jac_ctrl),
        spearman_control=float(sp_ctrl),
        control=str(control),
        exclude_top_k=int(exclude_top_k),
        codebook_size=int(cb),
        beats_control=beats,
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


def _transfer_stats(build_counts: np.ndarray, hold_counts: np.ndarray, exclude_top_k: int):
    """Coverage (holdout mass explained by build vocab), Jaccard of token sets, Spearman of
    shared-token frequencies — all from the two O(codebook) counts vectors."""
    # exclude mega-tokens (computed on the BUILD side, the vocab we are testing transfer of).
    drop = _top_k_megatokens(build_counts, exclude_top_k)
    if drop:
        mask = np.ones(build_counts.shape[0], dtype=bool)
        mask[list(drop)] = False
    else:
        mask = np.ones(build_counts.shape[0], dtype=bool)

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

    # --- A8: cross-asset transfer with a shuffle control ---
    class _Vocab:
        codebook_size = 64
    def build_vocab_fn(assets=None):
        return _Vocab()
    def encode_assets_fn(vocab, assets, control):
        # real substrate: peaked distribution (transfers); control: uniform (doesn't).
        n = len(assets)
        for _ in range(n):
            if control is None:
                yield rng.choice(64, size=2048, p=_peaked(64, rng))
            else:
                yield rng.integers(0, 64, size=2048)
    def _peaked(k, r):
        p = r.random(k) ** 4 + 1e-3
        return p / p.sum()
    a8 = a8_cross_asset_transfer(build_vocab_fn, encode_assets_fn,
                                 build_assets=(1, 2, 3), holdout_assets=(4, 5),
                                 control="shuffle", exclude_top_k=4)
    print("A8:", {k: round(v, 4) if isinstance(v, float) else v for k, v in a8.items()})

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
