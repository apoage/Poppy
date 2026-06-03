# Stage-1 design — The Candle Language (revised after critique)

> Revised synthesis from workflow `w6fwtir2h`, after the adversarial critique
> (`02_adversarial_critique.md`) caught the first draft skipping the resolved F1 (vol-unit) decision
> and proposing a circular eval gate. Honors F1 = vol-unit magnitude, surfaces F2–F5 as forks.
> **Note:** the 2026-06-02 operator decision is to run F2/F3 as a *panel* (see `README.md`); the
> single-method framing below is the per-language design each panel member instantiates.

## 1. What this is, in your terms

We take **one asset's minute OHLC stream** and turn it into a sequence of discrete **tokens**, where
each token is a **recurring, variable-length run of candle shapes** described by **relative moves
measured in local-volatility units** — so a token carries its own magnitude (a big version and a small
version of the same proportional shape are *different* tokens, per resolved F1), and is invariant to
absolute price, scale, position, and location. We are **not predicting anything**: the tokenizer only
builds a **vocabulary** — a language of price-action shapes — by finding which relative-shape runs
*recur often enough to be worth a word*. Transfer across the 64 assets ($225/unit vs $0.45/unit → same
token) is a **structural** consequence of (a) encoding everything as vol-unit ratios and (b) fitting
**one shared codebook** on the pooled corpus — never learned cross-asset attention. "Good" is judged
the way you'd judge a language/codebook (compresses, covers, recurs, transfers, reconstructs), **with
no downstream predictor and no prediction framing anywhere**.

## 2. Tokenizer design

### 2.1 Per-bar substrate — vol-unit relative moves (F1: corrected)

Every magnitude feature is divided by a **causal local-vol estimate** before binning. Let
`σ_t = EWMA(|log(c/c₋₁)|, halflife=H)` computed **causally** (only bars ≤ t), re-anchored on the
cadence in §2.5. Per bar:

| Feature | Definition | Role |
|---|---|---|
| `mv_t` | `log(c_t / c_{t-1}) / σ_t` | **vol-unit inter-bar move** — token-identity magnitude axis |
| `gap_t` | `log(o_t / c_{t-1}) / σ_t` | **inter-bar gap** (minute bars gap; gap-up-then-flat ≠ grind-up) |
| `body_t` | `log(c_t / o_t) / σ_t` | intrabar direction/strength, vol-unit |
| `range_t`| `log(h_t / l_t) / σ_t` | bar extent, vol-unit |
| `uw_t` | `(h_t − max(o,c)) / (h_t − l_t + ε)` | upper-wick fraction ∈[0,1] (scale-free; no vol norm) |
| `lw_t` | `(min(o,c) − l_t) / (h_t − l_t + ε)` | lower-wick fraction ∈[0,1] |

`mv_t ≈ gap_t + body_t`; keep gap and body as independent axes, derive mv for the move-bin. Guard every
denominator with `maximum(·, ε)`; clamp `σ_t` to a floor so flat regimes don't blow up vol-units (the
SAX flat-window pathology, relocated to the vol estimate). **$225/$0.45 demo:** a pure proportional
rescale leaves both the log-ratio and `σ_t` unchanged → identical vol-units → identical token. F1 gain:
`log(1.25)` on a placid vs frantic asset now land in **different** move-bins.

### 2.2 Base alphabet — data-adaptive product quantization

Map each bar's vol-unit tuple to a base symbol via a product alphabet (1d-SAX generalized): move axis
(quantile/Lloyd-Max bins on the **pooled** vol-unit distribution — fat-tailed, never Gaussian), gap
axis (coarse signed), body/range (joint quantile), wick (coarse 3-way). Breakpoints fitted **once on
the pooled 64-asset corpus** → the shared bin table is what makes the same shape land on the same
symbol across assets. Initial cardinality ≈ move 8 / gap 3 / body·range 4 / wick 3; final by MDL (§4).

### 2.3 Variable-length segmentation (F3 leans variable)

1. **Numerosity reduction + run-length tag** — collapse runs of identical base symbols to one symbol +
   coarse length tag (`FLAT×short` ≠ `FLAT×long`). Symbol-level noise filter.
2. **Frequency-greedy merge (BPE/RePair)** over the reduced stream — composes variable-length tokens;
   boundaries **frequency-derived, never reward-learned**. Token bar-length is emergent.

### 2.4 "Viable" vs "null" — defined end-to-end

A merge is a **viable token** iff it recurs **≥ `p_min`** across the pooled corpus and survives pruning
(§2.6). Else **null**, represented three ways for three consumers (reconciles the drop/code/placeholder
contradiction):

| Consumer | Null representation |
|---|---|
| MDL / gate (§4) | null bars **coded at full base-symbol entropy cost** (like-for-like vs raw baseline) |
| Stage-2 embedding | a single reserved `<NULL>` token (keeps bar-position alignment) |
| Trader (§5) | `<NULL>` runs **dropped** from the planned sequence (the denoising payoff) |

### 2.5 Drift control for a 24/7 corpus

`σ_t` and the increment anchor re-anchor on a **fixed minute cadence (every 1440 min / UTC-day), not
"per session"** — most of the 64 assets are crypto/FX with no session boundary.

### 2.6 Build → prune → freeze (one shared codebook)

- **Build:** BPE merge counts **pooled across all 64 assets** → single shared vocabulary; applied
  per-asset independently at encode time (no cross-asset coupling).
- **Prune:** (a) near-duplicate cluster under length-normalized z-normed Euclidean (÷√len), keep one
  canonical; (b) greedy minimum-cardinality **set-cover** on fraction-of-stream explained.
- **Artifact audit (offline, O(n²)):** Matrix-Profile/VALMOD on raw vol-unit windows confirm each
  token's occurrences are tight — kills binning-coincidence "tokens".
- **Serving = exact BPE merge ranking, NOT greedy longest-match** (longest-match would silently break
  rescale-invariance at serving; add an A0 check at serving).

## 3. Stage-2 embedding sketch (non-predictive)

**Objective: masked-token recovery (BERT/word2vec cloze), bidirectional, never causal/forecasting**
(TimeMAE template). Mask random token IDs; bidirectional encoder classifies the masked token from
two-sided context; hidden states → contextual embeddings. **Init** the embedding table from each
token's canonical vol-unit template (not random). **Optional** TS2Vec-style contrastive auxiliary.
**No cross-asset smuggle:** masking/attention must not cross asset-sequence seams; no asset-ID
conditioning. (Out of scope for the Stage-1 gate; flagged so 793 doesn't drift.)

## 4. Non-predictive eval battery (replaces the next-bar-sign probe)

Frame: **lossy compression + discrete coding.** Every metric computable from raw token streams, zero
labels, zero prediction. Thresholds are starting points, calibrated against baselines + the MDL knee.

- **Tier 0 — A0 rescale-invariance** (cheap pre-gate): re-tokenize a window at a different price scale →
  token sequence **byte-identical, PASS = 100%**, at training *and* serving. Hard gate.
- **Tier 1 — A8 cross-asset transfer (THE headline):** build vocab on a subset, tokenize held-out
  assets at extreme price levels with the same codebook; report mass-weighted coverage of B by tokens
  that also fire on A, active-vocab Jaccard, **mid-frequency Spearman frequency-rank correlation**.
  **Exclude top-k mega-tokens** (flat/doji fakes overlap). **Two controls, both required:** beat (i) a
  per-asset returns-shuffled stream AND (ii) a **random-merge same-size vocabulary** (ii isolates *good
  merges* from *any merges*). PASS = high held-out mid-freq coverage + rank-correlation, beats both.
- **Tier 2 — A1 two-part MDL, bits-per-candle, FULL stream:**
  `bits = Σ_covered log₂(1/p(token)) + Σ_null (base-symbol entropy of raw bars) + L(boundaries) +
  L(codebook)`. Null bars **and** boundary bits are charged. PASS = strictly below (a) a raw
  base-symbol entropy coder over the *same full stream* and (b) a **random-merge same-size vocab**.
  Vocab size `k` chosen by the **A8-transfer-stable region**, not an eyeballed knee.
- **Tier 3 — A2 reconstruction** within ratio-tolerance: PASS ≥ 90–95% of occurrences within tolerance,
  bounded mean log-ratio MSE. Always paired with A1 (recon alone improves monotonically with alphabet).
- **Tier 4 — A9 length non-degeneracy:** median token length in ~[3, 30]; no single token covers >X%
  of bars; not all length-1. The only direct test that variable-length shapes emerged. **A4 coverage
  band:** 60–70% ≤ coverage, null < 25–40% (deliberately *not* ~100% — total coverage means tokenizing
  noise, denoising payoff gone).
- **Tier 5 — health bands (NEVER maximized):** A5 active-code fraction >80%, perplexity not collapsed;
  A6 Zipf slope / Rényi efficiency reported as a band, *not* a target (gameable); A7 segmentation
  stability under sub-tolerance jitter, boundary-F1 > 0.85.

**DEMOTED — A3 persist-vs-time-shuffle:** circular (numerosity+BPE manufacture self-similarity and
minute returns are autocorrelated, so *any* tokenizer beats its own time-shuffle). Tests "does the
data have temporal structure" (known), not "is the vocabulary good." Retained only as the cheap
shuffled-control input to A8, never as a standalone proof.

**Verdict rule:** ship to Stage-2 iff **A8 transfer (mid-freq, mega-tokens excluded, beats BOTH
controls) AND A2 fidelity AND A1 corrected-full-stream MDL beats the random-merge same-size vocab AND
A9 non-degenerate length distribution.** A0 is the cheap correctness pre-gate; Tiers 4-band/5 advisory.
*Necessary-not-sufficient:* proves the language is well-formed/recurring/transferable/reconstructible —
**not** that tokens carry tradeable meaning (that is strictly Stage-2/3 by design).

## 5. Pipeline slot-in + efficiency

- **Online path (O(n), in the loop):** candles → vol-unit ratio features → base symbols (frozen bin
  table) → numerosity-reduce → exact-BPE-merge-order encode against frozen vocab → token IDs.
  (Matrix-Profile/VALMOD audits stay offline.)
- **Embedding:** token ID → Stage-2 contextual embedding → fed where raw candle features used to go;
  `<NULL>` positions carry the reserved embedding for bar-position alignment.
- **Denoising payoff:** `<NULL>` runs dropped from the trader's planned sequence; MCTS plans over a
  shorter, cleaner stream. ~2–4× token compression is the reported ballpark for grammar/BPE compression.
- **Safety:** a pure representation swap, MCTS planning untouched; each asset tokenized independently
  with the shared codebook → no cross-asset coupling introduced.

## 6. Cheapest first experiment (CPU-first)

Prove the **language**, not the trader. CPU/RAM under `tools/safe_run.sh --cpu --mem 64G` (no GPU on
workstation). No Stage-2, no MCTS, no forecast.

- **E0 data:** minute OHLC for 64 assets from `trading_data.db` to RAM once. ~6 assets spanning the
  widest price range ($225↔$0.45) = **held-out transfer set**; build on the rest.
- **E1 substrate:** vol-unit ratio features (causal `σ_t`, fixed-cadence anchor); pooled quantile/
  Lloyd-Max bins + coarse wick code.
- **E2 tokenize:** numerosity-reduce (run-length tag) → pooled BPE with floor `p_min`; sweep vocab
  size; prune (near-dup cluster + set-cover).
- **E3 eval in cost order:** A0 (100%, train *and* serving) → A8 transfer vs both controls → A1
  corrected MDL vs random-merge → A2 fidelity → A9 length → A4/A5/A6 bands. (A7 + VALMOD audit fast-follow.)

**Kill / proceed (NOT a predictor):** PROCEED iff A8 beats both controls at mid-freq AND A1-corrected
beats random-merge AND A2 ≥ 90% AND A9 non-degenerate. KILL if A8 collapses to a control, or
A1-corrected can't beat random-merge once null+boundary bits are charged, or the vocab degenerates
(A9). None is a forecast. **Sweep order on soft-fail:** K (bins) → p_min/vocab size → run-length tag
granularity → wick resolution. Set alphabet-size gates before judging coverage.

## 7. Open design forks

- **F2 discovery** — *(operator 2026-06-02: run the panel — BPE + ABBA + SAX-grammar + VQ/FSQ +
  Matrix-Profile, isolated and simultaneous; see `README.md`.)* Per-language default for the
  statistical member: BPE/RePair (CPU-first, prediction-free, deterministic serving).
- **F3 segmentation** — variable-length emergent (the reference token is variable-length).
- **F4 vocab size `k`** — driven by the A8 transfer-stable region; MDL reported as a curve.
- **F5 run-length tag** — coarse 3-bucket (short/med/long).
