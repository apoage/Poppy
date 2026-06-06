# TASK_797 — Asset fingerprints from candle-language tokens (model-unbound token use)

**Status**: Stage 1 + Stage 2 BUILT + first result (2026-06-06). CPU-only. Committed `d78b01e`.
- **Stage 1 DONE** — `tokenizer_api.py` (`CandleTokenizer`, `save_vocab`/`load_vocab`/`encode_asset`);
  round-trip gate byte-identical (frozen == in-process); `l1_bpe.py` untouched.
- **Stage 2 DONE + VALIDATED** — `fingerprint.py`; N×N over 50 price-stratified assets → `runs/797_fingerprint/`.
  bigram carries structure beyond marginals (independent structure 0.26 at 50 assets); tight small clusters
  (306↔892, 421↔747, 25↔860 0.76) + one dominant cluster (all same asset class) + outliers (360, 969).
  **Shuffle-collapse control PASSED** (`collapsed=True`): destroying token order (keep marginals) halves
  the bigram's independent structure (0.26→0.12) — so it's genuine ORDER-dependent transition structure,
  not an artifact. First evidence the tokens carry recognizable real-asset structure, outside the A8 harness.
- **Next**: validate tight pairs against asset metadata (need a symbol map); rerun on the (denser) server DB
  once the 150/150 finishes; thicker vocab; then Stage-3 roadmap (regime clustering / motif mining) on this N×N.

**Parent**: TASK_789 (candle-language). **Sibling-of, not-blocked-by**: TASK_793/794 (those are GPU +
trained-model downstream stages; this one is deliberately **unbound to any trained model**).

---

## Why this exists

The candle-language panel ([[TASK_796]]) proved its central claim: the tokens carry **real cross-asset
structure** — every member beats the fair frequency-matched random-codebook null on both token marginals
*and* transitions (lore `notes_2026-06-06.md` §183/§186). But that signal currently dead-ends in a
PROCEED/REVIEW/KILL gate verdict. **No consumer touches the tokens**, and the two designed consumers
(793 embedding, 794 trader-on-tokens) are both GPU-bound and gated behind a trained model.

This task gives the tokens their **first real, model-unbound use today, on the workstation CPU**: turn
each asset into a **token fingerprint** and the panel of fingerprints into an **asset-vs-asset similarity
surface**. It reuses the exact, already-validated A8 machinery (`_fold_with_bigram` + `_bigram_codelength`),
so it is low-risk and high-signal — and it validates the transfer claim *outside* the synthetic
power-harness, on real asset relationships.

A similarity surface is directly useful: **asset selection, regime-matched backtesting, cross-asset
transfer screening, dedup of near-identical listings**. And the N×N matrix is the substrate the other two
token-use directions (regime clustering, motif mining) both build on — so it is the right first cut.

---

## Stage 1 (enabler) — frozen tokenizer encode API

Closes the one piece of forward-looking debt this needs (survey item 4). Today each member re-fits its
vocab in-process per run and never saves it; the tokenizer exists only as private eval-battery plumbing.

- **L1 (production path) first**: `save_vocab(path)` (serialize the fitted base alphabet + exact
  BPE merge-rank table to JSON/npz) + `load_vocab(path)` + a public
  `CandleTokenizer.encode(asset_ohlc_or_units) -> token_id_stream` implementing the online O(n) path
  (candles → vol-unit → base symbols → numerosity-reduce → exact BPE-merge → ids) with **no re-fit**.
- Round-trip unit test (CPU, synthetic — no DB): fit → save → reload → encode, assert ids identical to
  the in-process encode.
- This also de-risks 793/794 (they need exactly this frozen "asset → token-id stream").

## Stage 2 (first deliverable) — N×N asset fingerprint similarity

- For each asset: reuse the frozen-encode path + `eval_battery._fold_with_bigram` to accumulate an
  **online O(codebook) unigram histogram + bigram transition table** (never O(windows)). That pair is the
  asset's **fingerprint**.
- **Distance**: pairwise transfer codelength `eval_battery._bigram_codelength(asset_b under asset_a's
  Laplace-smoothed model)` — the exact, validated, entropy-aware A8 metric — symmetrized; and/or a
  symmetric Jensen–Shannon over the token distributions as a cheap cross-check.
- Aggregate into an **N×N matrix** over the panel assets → a **nearest-asset table** + an **agglomerative
  dendrogram**. Write `runs/797_fingerprint/RESULT.md` + JSON sidecar.
- **No neural model, no GPU, no trained embedding** — pure token-statistics retrieval.

## Validation (does the surface mean anything?)
- Sanity: does token-similarity recover *known* relationships (same-sector / wrapped-vs-underlying /
  stablecoin cluster / obvious near-duplicate listings)? If yes, the fingerprint carries real structure.
- Control: shuffle each asset's token stream (phase-shuffle the substrate) and confirm the similarity
  structure **collapses** — the surface must depend on real transition structure, not marginals alone.

---

## Roadmap (built on the Stage-2 N×N, all CPU + model-unbound)
- **Regime clustering across assets** — cluster token *windows* pooled from all assets into K shared
  regimes (TF-IDF over token ids + a vendored k-means); per-window labels + population table. "What are
  the recurring market states the language sees, across the whole universe?"
- **Motif / recurring-phrase mining** — token n-grams that recur across *many* assets, ranked by
  document-frequency × lift (streaming n-gram hash + per-asset bitset). "What are the candle 'phrases'
  the market reuses everywhere?"
- **Token-distribution drift / change-point** — per-asset fingerprint over a rolling window → an
  unsupervised regime-shift / anomaly score. (Later.)

## Scope
- IN: Stage 1 (L1 frozen encode API + round-trip test) + Stage 2 (N×N fingerprint similarity, nearest
  table, dendrogram, validation), CPU-only, workstation.
- OUT (for the first cut): any trained model / embedding (that is 793); GPU; the multilingual shared
  embedding (trains a new model — a separate, later, GPU task); the trader integration (794).

## Deliverable
`runs/797_fingerprint/RESULT.md` + JSON: an N×N token-fingerprint similarity matrix over the panel
assets, a nearest-asset table, and a dendrogram — produced on the workstation CPU, reusing the frozen L1
encode path + `eval_battery._fold_with_bigram` / `_bigram_codelength`. First evidence that the candle
language's transfer claim shows up as **recognizable real-asset structure**, not just a synthetic gate pass.

## References
- Metric + fingerprint machinery: `tools/candle_lang/eval_battery.py` (`_fold_with_bigram`,
  `_bigram_codelength`, `a8_*`). Tokenizers: `tools/candle_lang/symbolic/`, `train_l4.py`.
- Transfer result this builds on: `lore/notes_2026-06-06.md` §183/§186; [[TASK_796]].
- Survey that scoped this: workflow `w0d5uzcaq`.
