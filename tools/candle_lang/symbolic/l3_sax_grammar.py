"""
L3 — SAX + grammar-induction panel member (design 01 §2.3 / §6, research digest §SAX, §Sequitur).

The L3 language is the *hierarchical-motif* member of the candle-language panel. Pipeline:

    minute OHLC  -> vol-unit substrate (volunit.features_for_run, shared with every member)
                 -> SAX letters on the MOVE channel (mv, idx 0) using the base-alphabet move-axis
                    quantile edges as the SAX alphabet (empirical-quantile, NOT Gaussian — minute
                    returns are fat-tailed; design §2.2 / digest §SAX)
                 -> numerosity reduction (collapse identical-letter runs to one letter; the SAX
                    flat-window noise filter) over each WITHIN-RUN window
                 -> RePair / Sequitur-style grammar induction: replace the most frequent digram
                    by a fresh non-terminal, repeat while utility >= 2 (the built-in "viable token"
                    definition: a rule used >= 2 times earns a word; digest §Sequitur)
                 -> greedy minimum-cardinality SET-COVER prune over the grammar rules on
                    fraction-of-bars-explained (design §2.6 prune step b)
                 -> FROZEN grammar vocab -> encode any stream to a token-id stream (terminals +
                    surviving non-terminals; longest-rule-first replacement against the frozen vocab).

`mv_t` is rescale-invariant by construction (volunit / eval A0), and the SAX edges + grammar are
fitted ONCE on the pooled BUILD corpus, so the SAME relative-shape run lands on the SAME token across
assets ($0.007 vs $105,747). The member is judged by the shared NON-predictive battery:
A0 (rescale-invariance, must be byte-identical), A1 (MDL bits-per-candle), A2 (reconstruction),
A8 (cross-asset transfer vs the phase_shuffle control), A9 (length non-degeneracy).

CPU / numpy ONLY — no GPU, no JAX execution. Streaming-bounded: the grammar is fit on a bounded,
most-recent window cap per asset (db_stream's max_windows_per_asset ring is the RAM cap), and every
battery reduction is the eval_battery online O(codebook) fold (never O(windows)).

CLI:
    python -m tools.candle_lang.symbolic.l3_sax_grammar \
        --db /path/trading_data.db --out runs/789_token_c0/L3_RESULT.md \
        --max-windows-per-asset 25000

Writes <out> (RESULT.md) + the .json sidecar via result_writer.write_result.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter

import numpy as np

# Package-relative imports (symbolic/ is a subpackage -> two dots to the package root).
from .. import config as C
from ..base_alphabet import fit_alphabet
from .. import db_stream
from .. import eval_battery
from ..result_writer import write_result


# --------------------------------------------------------------------------- #
# constants / tunables (kept private; never a contracted signature)
# --------------------------------------------------------------------------- #
# RePair utility floor: a digram must occur at least this many times to earn a rule (the
# "rule-utility >= 2" viable-token definition from the GrammarViz / Sequitur literature, digest §20).
# TASK_796: this is the HARD floor (a digram seen once can never earn a rule); the pooled-frequency
# floor `_MIN_DIGRAM_COUNT` below is the PERFORMANCE floor that stops minting rare rules early.
_MIN_RULE_UTILITY = 2
# TASK_796 PERFORMANCE: a digram must occur at least this many times across the POOLED corpus to earn
# a rule. Mirrors L1's --p-min=50 viable-token floor: stops the RePair loop minting thousands of
# rare rules (each of which cost a full-corpus pass in v1). The induction stops the moment the top
# digram count drops below this floor.
_MIN_DIGRAM_COUNT = 50
# Cap on the number of grammar non-terminals we ever mint (bounds the codebook + RePair passes).
# TASK_796: lowered 4096 -> 512 (mirror L1 --max-merges=512). 4096 rules x full-corpus rescan was the
# 134-min v1 pathology; combined with the incremental pair-count and p_min floor this is minutes.
_MAX_RULES = 512
# Set-cover target: keep the smallest rule set explaining this fraction of covered (non-terminal) bars.
_SET_COVER_TARGET = 0.95
# Numerosity-reduction coarse length-tag buckets (design §2.3 F5 short/med/long). Lengths in [1,2] ->
# short, [3,7] -> med, >=8 -> long. The tag rides on the SAX letter so FLATxshort != FLATxlong, but we
# keep it OUT of the grammar terminal id to avoid blowing the alphabet up (it is recorded for A9 only).
_LEN_TAG_EDGES = (3, 8)  # bucket boundaries


# --------------------------------------------------------------------------- #
# SAX over the move channel (base-alphabet move-axis edges as the SAX alphabet)
# --------------------------------------------------------------------------- #
def _move_axis_edges(table: dict) -> np.ndarray:
    """The move-axis interior quantile edges from a fitted base-alphabet table — these ARE the SAX
    breakpoints (empirical-quantile, fat-tail-aware; design §2.2). card = len(edges)+1 letters."""
    edges = np.asarray(table["edges"]["move"], dtype=np.float64)
    return edges


def _sax_letters(mv: np.ndarray, edges: np.ndarray, card: int) -> np.ndarray:
    """Digitize the (already vol-unit, rescale-invariant) mv channel into SAX letters in [0, card).

    Mirrors base_alphabet.encode_symbols' move-axis digitize EXACTLY (right=False, clip to range) so
    the L3 SAX alphabet is byte-identical to the shared move axis -> A0 holds by construction."""
    bins = np.digitize(np.asarray(mv, dtype=np.float64), edges, right=False)
    return np.clip(bins, 0, card - 1).astype(np.int64)


def _len_tag(run_len: int) -> int:
    """Coarse 3-bucket length tag (design §2.3 F5): 0=short, 1=med, 2=long."""
    if run_len < _LEN_TAG_EDGES[0]:
        return 0
    if run_len < _LEN_TAG_EDGES[1]:
        return 1
    return 2


def _numerosity_reduce(letters: np.ndarray):
    """Collapse maximal runs of identical SAX letters to one letter (the SAX flat-window noise
    filter; design §2.3 step 1). Returns (reduced_letters(int64,), run_lengths(int64,)).
    run_lengths[k] = number of original bars collapsed into reduced_letters[k] (>=1)."""
    letters = np.asarray(letters, dtype=np.int64)
    if letters.size == 0:
        return letters, np.zeros(0, dtype=np.int64)
    # boundaries where the letter changes
    change = np.nonzero(np.diff(letters))[0] + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [letters.size]])
    reduced = letters[starts]
    lengths = (ends - starts).astype(np.int64)
    return reduced.astype(np.int64), lengths


# --------------------------------------------------------------------------- #
# RePair grammar induction (frequency-greedy digram replacement)
# --------------------------------------------------------------------------- #
_SENTINEL = -1  # window-seam separator id (never merged; never a member of a counted pair)


def _replace_digram(seq: list, digram, new_sym: int) -> list:
    """Non-overlapping left-to-right replacement of every `digram` occurrence in `seq` by new_sym.
    Retained for the frozen-vocab encode path (L3Grammar._apply_rules / _apply_rules_static), which
    is O(rules x window) on SHORT per-window sequences — cheap. The HOT fit-time induction uses the
    incremental in-place merge in `_induce_grammar` instead (TASK_796)."""
    a, b = digram
    out = []
    i = 0
    n = len(seq)
    while i < n:
        if i < n - 1 and seq[i] == a and seq[i + 1] == b:
            out.append(new_sym)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def _count_initial_pairs(seq: list) -> Counter:
    """Initial adjacent-pair counts over the POOLED sequence, skipping any pair that touches a
    sentinel (a window seam — digrams must never span the seam, design §fit). One linear pass."""
    cnt = Counter()
    n = len(seq)
    for i in range(n - 1):
        a = seq[i]
        b = seq[i + 1]
        if a == _SENTINEL or b == _SENTINEL:
            continue
        cnt[(a, b)] += 1
    return cnt


def _induce_grammar(seq: list, base_card: int):
    """RePair grammar induction over the POOLED reduced-symbol sequence (TASK_796: incremental).

    v1 rescanned the whole corpus per merge (O(rules x corpus) — the 134-min pathology). This port of
    L1's incremental pair-count idiom maintains a GLOBAL adjacent-pair Counter and updates only the
    neighbours touched by each merge, so each round is O(occurrences of the merged pair), not a full
    rescan. Mechanism is otherwise identical: replace the most frequent valid adjacent digram with a
    fresh non-terminal, record the rule, repeat while the top count >= _MIN_DIGRAM_COUNT (the
    performance floor) and >= _MIN_RULE_UTILITY (the hard viability floor), until _MAX_RULES.

    Sentinels (_SENTINEL) separate window runs; a pair touching a sentinel is never counted, so
    digrams never span a seam. Merges are applied IN PLACE on a single working list.

    Returns (rules, next_id):
        rules   : dict{non_terminal_id -> (left_sym, right_sym)}  production table (each rule a digram).
        next_id : the first unused symbol id (== base_card + len(rules)).
    Rules may reference earlier non-terminals (hierarchical) — the Sequitur/RePair "rules for free"
    property (digest §20). LOSSLESS: encode->expand round-trips exactly (verified by A2)."""
    rules: dict = {}
    next_id = base_card
    work = list(seq)
    pair_counts = _count_initial_pairs(work)
    floor = max(int(_MIN_DIGRAM_COUNT), int(_MIN_RULE_UTILITY))

    while len(rules) < _MAX_RULES and pair_counts:
        best_pair, best_n = max(pair_counts.items(), key=lambda kv: kv[1])
        if best_n < floor:
            break
        new_sym = next_id
        rules[new_sym] = (int(best_pair[0]), int(best_pair[1]))
        next_id += 1
        _apply_merge_inplace(work, best_pair, new_sym, pair_counts)

    return rules, next_id


def _apply_merge_inplace(work: list, pair, new_sym: int, pair_counts: Counter) -> None:
    """In-place, non-overlapping left-to-right replacement of every adjacent `pair` in `work` by
    `new_sym`, incrementally updating `pair_counts` for the left/right neighbours touched (the
    classic BPE/RePair incremental-count idiom, ported from l1_bpe._apply_merge). Sentinel neighbours
    are never (de/in)cremented, so seam-spanning pairs stay out of the count. Mutates `work` and
    `pair_counts` in place; counting stays O(merged-pair occurrences), not a full rescan."""
    a, b = pair
    pair_counts.pop(pair, None)
    out = []
    i = 0
    n = len(work)
    while i < n:
        if i < n - 1 and work[i] == a and work[i + 1] == b:
            # left neighbour (from already-emitted output): (left,a) -> (left,new_sym)
            if out:
                left = out[-1]
                if left != _SENTINEL:
                    pair_counts[(left, a)] -= 1
                    if pair_counts[(left, a)] <= 0:
                        pair_counts.pop((left, a), None)
                    pair_counts[(left, new_sym)] += 1
            # right neighbour (next symbol after the merged pair): (b,right) -> (new_sym,right)
            if i + 2 < n:
                right = work[i + 2]
                if right != _SENTINEL:
                    pair_counts[(b, right)] -= 1
                    if pair_counts[(b, right)] <= 0:
                        pair_counts.pop((b, right), None)
                    pair_counts[(new_sym, right)] += 1
            out.append(new_sym)
            i += 2
        else:
            out.append(work[i])
            i += 1
    work[:] = out


# --------------------------------------------------------------------------- #
# rule expansion + set-cover prune
# --------------------------------------------------------------------------- #
def _expand_lengths(rules: dict, base_card: int) -> dict:
    """Bar-length of each symbol id = number of TERMINAL (base) symbols it expands to.
    Terminals (< base_card) have length 1; a non-terminal is the sum of its children's lengths.
    Computed in id order (rules reference only smaller ids -> single forward pass)."""
    lengths: dict = {}
    for sym in range(base_card):
        lengths[sym] = 1
    for nt in sorted(rules.keys()):
        a, b = rules[nt]
        lengths[nt] = lengths.get(a, 1) + lengths.get(b, 1)
    return lengths


def _rule_bar_coverage(rules: dict, base_card: int, encoded_seqs: list, lengths: dict):
    """For each surviving non-terminal, the number of CANDLE BARS it covers across the encoded
    corpus (= occurrences * its bar-length). Used as the set-cover utility. Returns dict{nt -> bars}."""
    occ: dict = {}
    for seq in encoded_seqs:
        for s in seq:
            if s >= base_card:
                occ[s] = occ.get(s, 0) + 1
    cover: dict = {}
    for nt, n in occ.items():
        cover[nt] = int(n) * int(lengths.get(nt, 1))
    return cover


def _set_cover_prune(rules: dict, base_card: int, encoded_seqs: list, lengths: dict,
                     target: float = _SET_COVER_TARGET):
    """Greedy minimum-cardinality set-cover (design §2.6 prune b): keep the smallest set of
    non-terminals that together explain `target` fraction of all bars covered by ANY non-terminal.

    We rank non-terminals by descending bar-coverage and keep adding until the cumulative covered
    fraction reaches `target`. Dropped non-terminals are simply not in the kept set; at encode time
    a dropped rule's digram is left as its two children (it just doesn't get a word). Returns the
    SET of kept non-terminal ids (terminals are always implicitly kept)."""
    cover = _rule_bar_coverage(rules, base_card, encoded_seqs, lengths)
    total = float(sum(cover.values()))
    if total <= 0.0:
        return set()
    order = sorted(cover.items(), key=lambda kv: kv[1], reverse=True)
    kept = set()
    acc = 0.0
    for nt, bars in order:
        kept.add(nt)
        acc += float(bars)
        if acc / total >= float(target):
            break
    return kept


# --------------------------------------------------------------------------- #
# frozen grammar vocab + encode
# --------------------------------------------------------------------------- #
class L3Grammar:
    """Frozen L3 grammar vocabulary: the SAX base alphabet + the surviving (set-cover) grammar
    rules, plus the contiguous id remap eval_battery expects (codebook_size == #ids in use).

    Public attributes used by the eval battery / A8:
        codebook_size : int   contiguous token-id alphabet size
        base_card     : int   number of SAX letters (terminals)
    Public methods:
        encode_symbols(reduced_letters(int64,)) -> token-ids(int64,)   apply frozen rules then remap
        expand(token_id) -> list[terminal letters]                     for A2 reconstruction
    """

    def __init__(self, table: dict, rules: dict, kept: set, base_card: int):
        self.table = table
        self.base_card = int(base_card)
        # keep only surviving rules; renumber to a dense id space for eval_battery's bincount.
        kept_nts = sorted(int(nt) for nt in kept)
        self.rules = {nt: rules[nt] for nt in kept_nts}  # original-id-keyed production table
        self._lengths_orig = _expand_lengths(rules, base_card)  # bar-length on ORIGINAL ids

        # dense remap: terminals [0, base_card) keep their ids; surviving non-terminals get the next ids.
        self._remap = {s: s for s in range(base_card)}
        nxt = base_card
        for nt in kept_nts:
            self._remap[nt] = nxt
            nxt += 1
        self.codebook_size = int(nxt)

        # reverse remap (dense id -> original id) + dense-id bar-length, for A2 expansion.
        self._inv = {v: k for k, v in self._remap.items()}
        self.token_bar_length = np.ones(self.codebook_size, dtype=np.int64)
        for dense_id, orig in self._inv.items():
            self.token_bar_length[dense_id] = int(self._lengths_orig.get(orig, 1))

        # precompute, for each surviving non-terminal (dense id), its full TERMINAL expansion (the
        # SAX-letter sequence it stands for) so A2 reconstruction is a dict lookup, not recursion.
        self._terminal_expansion = {}
        for dense_id, orig in self._inv.items():
            self._terminal_expansion[dense_id] = self._expand_terminals(orig, rules, base_card)

    # ----- frozen-vocab encode (greedy longest-rule-first replacement) ----- #
    def _apply_rules(self, seq: list) -> list:
        """Apply the surviving grammar rules to a reduced-letter sequence in ID-DESCENDING order
        (longest/highest non-terminals first), replacing each rule's digram by its non-terminal.
        Deterministic + frozen -> same input gives byte-identical output (A0)."""
        work = list(seq)
        for nt in sorted(self.rules.keys(), reverse=True):
            digram = self.rules[nt]
            work = _replace_digram(work, digram, nt)
        return work

    def encode_symbols(self, reduced_letters: np.ndarray) -> np.ndarray:
        """Encode a reduced SAX-letter sequence to dense token ids in [0, codebook_size)."""
        seq = [int(x) for x in np.asarray(reduced_letters, dtype=np.int64).tolist()]
        coded = self._apply_rules(seq)
        out = np.fromiter((self._remap[s] for s in coded), dtype=np.int64, count=len(coded))
        return out

    def _expand_terminals(self, sym, rules: dict, base_card: int) -> list:
        """Fully expand an ORIGINAL symbol id to its list of terminal SAX letters (iterative
        stack expansion; rules reference only smaller ids so this always terminates)."""
        out = []
        stack = [int(sym)]
        while stack:
            s = stack.pop()
            if s < base_card:
                out.append(s)
            else:
                a, b = rules[s]
                # push right then left so left expands first (preserve order)
                stack.append(int(b))
                stack.append(int(a))
        return out

    def expand(self, token_id: int) -> list:
        """Dense token id -> list of terminal SAX letters it represents (A2 reconstruction)."""
        tid = int(token_id)
        if tid < self.base_card:
            return [tid]
        return list(self._terminal_expansion.get(tid, [tid]))


# --------------------------------------------------------------------------- #
# substrate streaming -> reduced SAX letters per window
# --------------------------------------------------------------------------- #
def _stream_window_feats(db: str, assets, cfg: "C.CandleConfig", max_windows_per_asset,
                         control: str | None = None, rng_seed: int = 0):
    """Yield per-window vol-unit feature arrays (W,6) for the given assets, WITHIN-run only.

    control:
        None        -> real vol-unit substrate.
        'shuffle'   -> donor phase_shuffle_returns surrogate applied to the window's feature rows
                       (the A8 structure-destroying null; design §4 / data.py:165).
        'random_merge' -> rows of the window randomly permuted in time (same marginals, no order).
    """
    rng = np.random.default_rng(rng_seed)
    for aid in assets:
        for chunk in db_stream.iter_windows(db, int(aid), cfg, max_windows=max_windows_per_asset):
            X = chunk["X"]  # (n, W, 6) f32
            for w in range(X.shape[0]):
                feats = X[w]  # (W, 6)
                if control == "shuffle":
                    valid = np.ones(feats.shape[0], dtype=bool)
                    feats = _phase_shuffle(feats, valid, rng)
                elif control == "random_merge":
                    perm = rng.permutation(feats.shape[0])
                    feats = feats[perm]
                yield feats


def _phase_shuffle(feats: np.ndarray, valid: np.ndarray, rng) -> np.ndarray:
    """Local mirror of the donor phase_shuffle_returns (data.py:165-203) on a (W,6) feature window:
    phase-randomize the move channel (idx 0), time-permute the geometry channels (idx 1..5). Keeps
    per-bar marginals, destroys temporal order — the proper A8 'shuffle' null."""
    feats = np.asarray(feats, dtype=np.float64).copy()
    T = feats.shape[0]
    if T >= 8:
        x = feats[:, 0]
        Xf = np.fft.rfft(x)
        mag = np.abs(Xf)
        ph = np.angle(Xf)
        rand_ph = rng.uniform(-np.pi, np.pi, size=ph.shape)
        rand_ph[0] = ph[0]
        if T % 2 == 0:
            rand_ph[-1] = ph[-1]
        Xnew = mag * np.exp(1j * rand_ph)
        feats[:, 0] = np.fft.irfft(Xnew, n=T)
        perm = rng.permutation(T)
        feats[:, 1:] = feats[:, 1:][perm]
    return feats.astype(np.float32)


def _window_to_reduced_letters(feats: np.ndarray, edges: np.ndarray, card: int):
    """One (W,6) window -> (reduced_letters, run_lengths) via SAX on mv + numerosity reduction."""
    mv = np.asarray(feats, dtype=np.float64)[:, 0]  # move channel
    letters = _sax_letters(mv, edges, card)
    return _numerosity_reduce(letters)


# --------------------------------------------------------------------------- #
# fit (pooled BUILD corpus) -> frozen grammar
# --------------------------------------------------------------------------- #
def fit_grammar(db: str, cfg: "C.CandleConfig", max_windows_per_asset,
                fit_window_cap: int = 4000):
    """Fit the L3 grammar on the pooled BUILD corpus.

    1) Pool vol-unit feature rows across BUILD assets (bounded by fit_window_cap windows/asset for
       the alphabet fit), fit the shared base alphabet (we only consume its move-axis edges).
    2) Build the reduced-SAX-letter sequences per window, induce a RePair grammar over the pooled
       sequence stream, set-cover-prune the rules, freeze into an L3Grammar.

    Returns (grammar:L3Grammar, table:dict, edges, card, n_fit_windows).
    Bounded RAM: the alphabet fit pools at most fit_window_cap windows/asset; grammar induction
    operates on the reduced (numerosity-collapsed) letter sequences which are << raw bars."""
    build_assets = list(cfg.build_assets)

    # ---- 1) pool vol-unit rows for the base-alphabet fit (move-axis edges) ----
    pooled_rows = []
    pooled_windows = 0
    for aid in build_assets:
        held = 0
        for chunk in db_stream.iter_windows(db, int(aid), cfg, max_windows=fit_window_cap):
            X = chunk["X"]  # (n, W, 6)
            # subsample rows: one feature row per window is enough to fit quantile edges cheaply,
            # but pooling the full window rows gives a better empirical distribution; cap by budget.
            rows = X.reshape(-1, X.shape[-1])
            pooled_rows.append(rows)
            held += X.shape[0]
            pooled_windows += X.shape[0]
            if held >= fit_window_cap:
                break
    if not pooled_rows:
        raise RuntimeError("L3 fit: no BUILD windows streamed — check --db / build_assets.")
    feats_pooled = np.concatenate(pooled_rows, axis=0)
    table = fit_alphabet(feats_pooled, cfg)
    edges = _move_axis_edges(table)
    card = int(table["cardinalities"]["move"])  # SAX alphabet size (== ALPHABET['move'] = 8)
    del pooled_rows, feats_pooled  # free the pooled matrix before grammar induction

    # ---- 2) reduced-letter sequences across BUILD, then RePair grammar ----
    reduced_seqs = []
    for feats in _stream_window_feats(db, build_assets, cfg, max_windows_per_asset):
        red, _ = _window_to_reduced_letters(feats, edges, card)
        if red.size >= 2:
            reduced_seqs.append([int(x) for x in red.tolist()])
    if not reduced_seqs:
        raise RuntimeError("L3 fit: no reduced sequences (all windows collapsed to <2 symbols).")

    # RePair over the POOLED stream: concatenate with a sentinel separator so digrams never span the
    # seam between two windows (sentinels are negative ids the grammar never merges).
    pooled = []
    for s in reduced_seqs:
        pooled.extend(s)
        pooled.append(_SENTINEL)
    rules, _next = _induce_grammar(pooled, card)

    # encode each window's reduced seq with the FULL rule set (pre-prune) for set-cover bar coverage.
    lengths = _expand_lengths(rules, card)
    encoded_pre = [_apply_rules_static(s, rules) for s in reduced_seqs]
    kept = _set_cover_prune(rules, card, encoded_pre, lengths)

    grammar = L3Grammar(table, rules, kept, card)
    return grammar, table, edges, card, len(reduced_seqs)


def _apply_rules_static(seq: list, rules: dict) -> list:
    """Apply ALL rules (id-descending) to a sequence — used during fit for set-cover coverage
    accounting (before we know which rules survive pruning)."""
    work = list(seq)
    for nt in sorted(rules.keys(), reverse=True):
        work = _replace_digram(work, rules[nt], nt)
    return work


# --------------------------------------------------------------------------- #
# encode streams for the battery (token-id chunks, reconstruction pairs, A0 tokenizer)
# --------------------------------------------------------------------------- #
def _token_id_chunks(grammar: L3Grammar, db: str, assets, cfg, max_windows_per_asset,
                     control: str | None = None):
    """Iterator of token-id CHUNKS (1-D int64 arrays), ONE per window — the A1/A8 stream fold input.
    Online: never concatenates the whole stream; each chunk is a single window's token ids."""
    for feats in _stream_window_feats(db, assets, cfg, max_windows_per_asset, control=control):
        red, _ = _window_to_reduced_letters(feats, grammar.edges_card[0], grammar.edges_card[1])
        if red.size == 0:
            continue
        yield grammar.encode_symbols(red)


def _length_counts(grammar: L3Grammar, db: str, assets, cfg, max_windows_per_asset) -> dict:
    """Accumulate token BAR-LENGTH counts {length -> n_tokens} over the stream (A9 input). Online —
    only the small {length->count} dict grows (host memory O(distinct lengths), not O(windows))."""
    counts: dict = {}
    for feats in _stream_window_feats(db, assets, cfg, max_windows_per_asset):
        red, _ = _window_to_reduced_letters(feats, grammar.edges_card[0], grammar.edges_card[1])
        if red.size == 0:
            continue
        ids = grammar.encode_symbols(red)
        for tid in ids:
            L = int(grammar.token_bar_length[int(tid)])
            counts[L] = counts.get(L, 0) + 1
    return counts


def _recon_pairs(grammar: L3Grammar, db: str, assets, cfg, max_windows_per_asset):
    """Iterator of (recon_letters, true_letters) PER-BAR pairs for A2 reconstruction (TASK_796).

    L3 is a SYMBOLIC member on the SIGNED SAX-letter axis (the quantized mv channel). The token
    stream expands back to terminal SAX letters; we compare them PER BAR to the original
    numerosity-reduced letter sequence with the shared additive battery (mode='additive',
    abs_tol=0.25). NO base=offset hack — the letters are the signed per-bar quantity directly, and
    the grammar round-trip (encode -> expand) is LOSSLESS so |recon - true| is ~0 everywhere (any
    deviation is a real bug, not a zero-crossing artifact). Streamed one window at a time."""
    edges, card = grammar.edges_card
    for feats in _stream_window_feats(db, assets, cfg, max_windows_per_asset):
        red, _ = _window_to_reduced_letters(feats, edges, card)
        if red.size == 0:
            continue
        ids = grammar.encode_symbols(red)
        recon_letters = []
        for tid in ids:
            recon_letters.extend(grammar.expand(int(tid)))
        recon = np.asarray(recon_letters, dtype=np.float64)   # signed SAX letters, per bar
        true = red.astype(np.float64)                          # signed SAX letters, per bar
        yield recon, true


# --------------------------------------------------------------------------- #
# A0 single-window tokenizer (rescale-invariance check)
# --------------------------------------------------------------------------- #
def _make_a0_tokenizer(grammar: L3Grammar):
    """Return tokenize_window_fn(ohlc(W,4)) -> int for eval_battery.a0_rescale_invariance.

    Reconstructs the vol-unit features for the single window (window-local, gap-free), SAX-encodes
    the mv channel, numerosity-reduces, applies the frozen grammar, and returns the FIRST token id
    (a stable scalar fingerprint of the window's token sequence). Because the substrate is
    rescale-invariant and the grammar is frozen, x and lam*x yield identical ids -> A0 PASS."""
    edges, card = grammar.edges_card

    def tokenize_window(ohlc: np.ndarray) -> int:
        ohlc = np.asarray(ohlc, dtype=np.float64)
        o, h, l, c = ohlc[:, 0], ohlc[:, 1], ohlc[:, 2], ohlc[:, 3]
        # window-local vol-unit mv (mirror of volunit.features_for_run, no day re-anchor: a window
        # is << 1 day). Causal EWMA of |logret| seeded at the first value.
        prev_c = c[:-1]
        cur_c = c[1:]
        logret = np.log(np.maximum(cur_c, C.EPS) / np.maximum(prev_c, C.EPS))
        absr = np.abs(logret)
        alpha = 1.0 - 2.0 ** (-1.0 / float(grammar.halflife))
        sigma = np.empty_like(absr)
        if absr.size:
            sigma[0] = absr[0]
            for k in range(1, absr.size):
                sigma[k] = alpha * absr[k] + (1.0 - alpha) * sigma[k - 1]
        sigma = np.maximum(sigma, grammar.sigma_floor)
        mv = logret / np.maximum(sigma, C.DIV_EPS)
        letters = _sax_letters(mv, edges, card)
        red, _ = _numerosity_reduce(letters)
        if red.size == 0:
            return 0
        ids = grammar.encode_symbols(red)
        return int(ids[0]) if ids.size else 0

    return tokenize_window


def _sample_a0_windows(db: str, cfg: "C.CandleConfig", n: int = 16):
    """A small sample of raw-price OHLC windows (W,4) for the A0 unit check. Pulls a few BUILD
    windows' raw close path and synthesizes a plausible OHLC around it (we only need self-consistent
    rescale behavior). We reconstruct raw OHLC from the streamed close path + the window features is
    not available here, so we draw real close paths and build OHLC bands around them."""
    out = []
    W = int(cfg.window)
    for aid in cfg.build_assets:
        for chunk in db_stream.iter_windows(db, int(aid), cfg, max_windows=64):
            Cp = chunk["C"]  # (n, W) raw close path
            for w in range(min(Cp.shape[0], 4)):
                close = np.asarray(Cp[w], dtype=np.float64)
                if close.size != W or not np.all(np.isfinite(close)) or np.any(close <= 0):
                    continue
                # build a plausible OHLC: open = previous close, hi/lo bracket open/close.
                op = np.concatenate([[close[0]], close[:-1]])
                hi = np.maximum(op, close) * 1.0005
                lo = np.minimum(op, close) * 0.9995
                out.append(np.stack([op, hi, lo, close], axis=1))
                if len(out) >= n:
                    return out
        if len(out) >= n:
            break
    return out


# --------------------------------------------------------------------------- #
# CLI / main
# --------------------------------------------------------------------------- #
def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="L3 SAX + grammar candle-language member (CPU).")
    p.add_argument("--db", type=str, default=C.db_path(), help="trading_data.db path (or $CANDLE_DB).")
    p.add_argument("--out", type=str,
                   default="runs/789_token_c0/L3_RESULT.md",
                   help="RESULT.md output path (.json sidecar written alongside).")
    p.add_argument("--max-windows-per-asset", type=int, default=25000,
                   help="cap windows kept per asset (most-recent ring; RAM bound).")
    p.add_argument("--fit-window-cap", type=int, default=4000,
                   help="windows/asset pooled for the base-alphabet (SAX edges) fit.")
    p.add_argument("--window", type=int, default=C.DEFAULT_WINDOW)
    # TASK_796 scale-up: explicit UNIFORM asset selection + A-by-B transfer matrix.
    p.add_argument("--build-assets", nargs="+", type=int, default=None,
                   help="explicit BUILD asset ids (default: CandleConfig build_assets).")
    p.add_argument("--holdout-assets", nargs="+", type=int, default=None,
                   help="explicit HOLDOUT asset ids (default: CandleConfig holdout_assets).")
    p.add_argument("--transfer-matrix", action="store_true",
                   help="compute the per-asset A8 A-by-B transfer matrix + per-holdout breakdown.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    t0 = time.time()
    # TASK_796: frozen dataclass -> pass explicit asset lists into the constructor; bare invocation
    # keeps the existing CandleConfig defaults (build_assets / holdout_assets) untouched.
    cfg = C.CandleConfig(
        window=int(args.window), db=args.db,
        build_assets=tuple(args.build_assets) if args.build_assets else C.DEFAULT_BUILD_ASSETS,
        holdout_assets=tuple(args.holdout_assets) if args.holdout_assets is not None
        else C.DEFAULT_HOLDOUT_ASSETS,
    )
    mwpa = int(args.max_windows_per_asset)

    print(f"[L3] SAX+grammar member | db={cfg.db} window={cfg.window} "
          f"max_windows/asset={mwpa}", flush=True)

    # DB sanity (non-fatal warning, mirrors BatchStreamer contract).
    if not db_stream.quick_check(cfg.db):
        print(f"[L3] WARNING: PRAGMA quick_check did not return 'ok' for {cfg.db}", flush=True)

    # ---- fit frozen grammar on pooled BUILD ----
    grammar, table, edges, card, n_fit = fit_grammar(
        cfg.db, cfg, max_windows_per_asset=mwpa, fit_window_cap=int(args.fit_window_cap)
    )
    # stash the static encode params + substrate constants on the grammar (so the battery closures
    # need only the grammar object). edges_card is a (edges, card) tuple consumed everywhere.
    grammar.edges_card = (edges, card)
    grammar.halflife = cfg.ewma_halflife
    grammar.sigma_floor = cfg.sigma_floor

    n_rules = len(grammar.rules)
    print(f"[L3] fit done: SAX card={card}, grammar rules kept={n_rules}, "
          f"codebook_size={grammar.codebook_size}, fit windows={n_fit}", flush=True)

    build_assets = list(cfg.build_assets)
    holdout_assets = [a for a in cfg.holdout_assets]

    # ---- A0 rescale-invariance (must be byte-identical) ----
    a0_windows = _sample_a0_windows(cfg.db, cfg, n=16)
    if a0_windows:
        a0 = eval_battery.a0_rescale_invariance(_make_a0_tokenizer(grammar), a0_windows,
                                                lambdas=(0.01, 100.0))
    else:
        a0 = dict(agree=float("nan"), **{"pass": False}, n_trials=0, note="no A0 windows streamed")
    print(f"[L3] A0: {a0}", flush=True)

    # ---- A1 MDL over the full BUILD stream ----
    null_bits_per_bar = float(np.log2(max(card, 2)))             # base SAX-symbol entropy per bar
    codebook_bits = float(grammar.codebook_size) * float(np.log2(max(grammar.codebook_size, 2)))
    # boundary count == total tokens emitted (each token is a boundary); fold once. TASK_796: also
    # accumulate the TRUE candle-BAR coverage = sum of each token's bar-length (grammar.token_bar_length),
    # so the per-bar null is charged over covered BARS not token count (the v1 accounting artifact).
    n_tokens_total = 0
    n_bars_covered = 0
    for chunk in _token_id_chunks(grammar, cfg.db, build_assets, cfg, mwpa):
        ids = np.asarray(chunk, dtype=np.int64)
        n_tokens_total += int(ids.size)
        n_bars_covered += int(grammar.token_bar_length[ids].sum())
    a1 = eval_battery.a1_mdl(
        _token_id_chunks(grammar, cfg.db, build_assets, cfg, mwpa),
        codebook_size=grammar.codebook_size,
        null_bits_per_bar=null_bits_per_bar,
        n_boundaries=n_tokens_total,
        codebook_bits=codebook_bits,
        n_bars_covered=n_bars_covered,
    )
    print(f"[L3] A1: bits/candle={a1['bits_per_candle']:.4f} "
          f"compression={a1['compression_ratio']:.3f} n_tokens={a1['n_tokens']} "
          f"bars_covered={a1['n_bars_covered']}", flush=True)

    # ---- A2 reconstruction (SAX-letter path fidelity; TASK_796 additive on signed letters) ----
    a2 = eval_battery.a2_reconstruction(
        _recon_pairs(grammar, cfg.db, build_assets, cfg, mwpa), mode="additive", abs_tol=0.25
    )
    print(f"[L3] A2: frac_within_tol={a2['frac_within_tol']:.4f} "
          f"mean_abs_err={a2['mean_abs_err']:.5f} (mode={a2['mode']})", flush=True)

    # ---- A9 length non-degeneracy ----
    a9 = eval_battery.a9_length_nondegeneracy(
        _length_counts(grammar, cfg.db, build_assets, cfg, mwpa)
    )
    print(f"[L3] A9: median_len={a9['median_len']} max_share={a9['max_token_share']:.4f} "
          f"pass={a9['pass']}", flush=True)

    # ---- A8 cross-asset transfer (only if holdout assets present) ----
    a8 = None
    if holdout_assets:
        def build_vocab_fn(assets=None):
            return grammar  # frozen vocab already fit on BUILD

        def encode_assets_fn(vocab, assets, control):
            return _token_id_chunks(vocab, cfg.db, list(assets), cfg, mwpa, control=control)

        a8 = eval_battery.a8_cross_asset_transfer(
            build_vocab_fn, encode_assets_fn,
            build_assets=build_assets, holdout_assets=holdout_assets,
            controls=("shuffle", "random_merge"), exclude_top_k=8,
            per_asset_matrix=args.transfer_matrix,
        )
        print(f"[L3] A8: transfer_bits={a8['transfer_bits']:.4f} "
              f"perm_min={a8['perm_min_bits']:.4f} beats_random_codebook={a8['beats_random_codebook']} "
              f"beats_substrate={a8['beats_substrate']} beats_all={a8['beats_all']}", flush=True)

    # ---- verdict (design §6 KILL/PROCEED, A8 + A2 + A1 + A9) ----
    a8_ok = bool(a8 is None or a8["beats_control"])
    a2_ok = bool(a2["frac_within_tol"] >= 0.90)
    a1_ok = bool(a1["compression_ratio"] > 1.0)
    a9_ok = bool(a9["pass"])
    a0_ok = bool(a0.get("pass", False))
    proceed = a0_ok and a8_ok and a2_ok and a1_ok and a9_ok
    verdict = "PROCEED" if proceed else "KILL"

    # ---- write RESULT.md + .json ----
    sections = [
        ("Member", [
            "- **L3 — SAX + RePair grammar** (hierarchical-motif candle language; CPU/numpy).",
            "- SAX on the vol-unit `mv` channel using the base-alphabet move-axis quantile edges",
            "  (empirical-quantile, fat-tail-aware) -> numerosity reduction -> RePair grammar",
            "  induction (rule utility >= 2) -> set-cover prune -> frozen vocab.",
        ]),
        ("Configuration", [
            f"- DB: `{cfg.db}`",
            f"- Window {cfg.window} bars | SAX alphabet card {card} | max windows/asset {mwpa}",
            f"- Build assets: {build_assets}",
            f"- Holdout assets: {holdout_assets}",
            f"- Grammar rules kept (set-cover {_SET_COVER_TARGET}): {n_rules} | "
            f"codebook size {grammar.codebook_size}",
        ]),
        ("A0 — rescale invariance (hard pre-gate)", [
            f"- agreement {a0.get('agree')} over {a0.get('n_trials', 0)} trials "
            f"(lambdas {a0.get('lambdas', '(0.01,100)')}) -> pass={a0.get('pass')}",
            "- Must be byte-identical: SAX letters come from the rescale-invariant vol-unit substrate.",
        ]),
        ("A1 — MDL (bits per candle, full BUILD stream)", [
            f"- bits/candle {a1['bits_per_candle']:.4f} | entropy/token "
            f"{a1['entropy_bits_per_token']:.4f}",
            f"- null bits/bar {null_bits_per_bar:.4f} | compression ratio "
            f"{a1['compression_ratio']:.4f} (>1 = cheaper than per-bar null)",
            f"- tokens {a1['n_tokens']} | codebook {a1['codebook_size']}",
        ]),
        ("A2 — reconstruction (SAX-letter path fidelity)", [
            f"- mode `{a2['mode']}` (signed per-bar SAX letters) | frac within tol "
            f"(±{a2['abs_tol']}) {a2['frac_within_tol']:.4f} | mean abs err {a2['mean_abs_err']:.5f}",
            "- grammar round-trip is LOSSLESS -> frac_within_tol should be ~1.0 "
            "(any miss is a real encode/expand bug, not a zero-crossing artifact).",
            f"- bars compared {a2['n_bars']} over {a2['n_pairs']} windows.",
        ]),
        ("A9 — length non-degeneracy", [
            f"- median token bar-length {a9['median_len']} | max single-length share "
            f"{a9['max_token_share']:.4f}",
            f"- all-length-1 {a9['all_len1']} | distinct lengths {a9.get('n_lengths')} | "
            f"pass {a9['pass']}",
        ]),
    ]
    if a8 is not None:
        ctl_lines = [
            f"  - `{name}`: transfer_bits {d['transfer_bits']:.4f} | coverage {d['coverage']:.4f} "
            f"| jaccard {d['jaccard']:.4f} | spearman {d['spearman']:.4f} | beats {d['beats']}"
            for name, d in a8.get("per_control", {}).items()
        ]
        sections.append((
            "A8 — cross-asset transfer (headline = non-saturating transfer codelength)", [
                f"- **transfer_bits {a8['transfer_bits']:.4f}** bits/in-band-token (HEADLINE; "
                f"lower = holdout mass concentrates where build put its mass) over "
                f"{a8['n_band_tokens']} band tokens",
                f"- freq-matched random-codebook null: perm_min {a8['perm_min_bits']:.4f} | "
                f"perm_mean {a8['perm_mean_bits']:.4f} ({a8['n_perm_seeds']} seeds) -> "
                f"beats_random_codebook {a8['beats_random_codebook']}",
                f"- substrate controls {a8['controls']} -> beats_substrate {a8['beats_substrate']}",
                *ctl_lines,
                f"- secondary (reported, band-desaturated): coverage {a8['coverage']:.4f} | "
                f"jaccard {a8['jaccard']:.4f} | spearman {a8['spearman']:.4f}",
                f"- band {a8['band']} | excludes top-{a8['exclude_top_k']} mega-tokens | "
                f"**beats_all {a8['beats_all']}**",
            ]))
        if a8.get("holdout_transfer_rate") is not None:
            sections[-1][1].append(
                f"- holdout assets with real transfer (beat random codebook): "
                f"{a8['holdout_transfer_rate']}"
            )

    meta = dict(
        member="L3_sax_grammar",
        window=int(cfg.window),
        sax_card=int(card),
        codebook_size=int(grammar.codebook_size),
        rules_kept=int(n_rules),
        fit_windows=int(n_fit),
        max_windows_per_asset=int(mwpa),
        a0=a0, a1=a1, a2=a2, a9=a9,
        a8=(a8 if a8 is not None else {}),
        gates=dict(a0=a0_ok, a8=a8_ok, a2=a2_ok, a1=a1_ok, a9=a9_ok),
        wallclock_s=float(time.time() - t0),
        db=cfg.db,
    )

    write_result(args.out, "TASK_789 L3 — SAX + Grammar Candle Language",
                 sections, verdict, meta)
    print(f"[L3] VERDICT {verdict}  wall {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
