# TASK_791 — Stage ①: Adaptive minute-candle tokenizer (the language)

**Status**: DESIGN (method pending research workflow `w6fwtir2h`)
**Created**: 2026-06-02 · **Parent**: TASK_789 (umbrella) · **Gate**: TASK_792 · **Next**: TASK_793

## Goal

Turn ONE asset's **minute** OHLC stream into a sequence of discrete **tokens**, where a token is a
recurring, **variable-length** sequence of candle shapes defined by **relative moves**, so the same
shape transfers across assets at any price level/scale. Build the **vocabulary** (the "shapes
database"). **No prediction anywhere in this stage.**

## Hard constraints (from TASK_789 reframe)

- **Minute** granularity. No aggregation to hourly/daily.
- **Single-asset, adaptive.** One tokenizer, asset-agnostic; transferability is a *property of the
  relative encoding*, not learned cross-asset coupling.
- **Relative / scale-free encoding.** A token is invariant to absolute price, overall scale, position,
  and location. `$225/unit` and `$0.45/unit` versions of the same shape → same token.
- **Variable-length tokens.** The example token spans a variable number of bars; the unit is a *shape*,
  not a fixed window.
- **"Viable token" must be defined.** The tokenizer must distinguish a recurring, vocabulary-worthy
  shape from noise that should be dropped/mapped to a null token (this feeds the stage-③ filtering
  payoff).

## Reference: the example token, to be encoded end-to-end

`3× up ×1.25, 1× down ×1.8, 1.4, 1.2, 1.2, up ×1.85` — the design must show exactly how this candle
run maps to a token, and how the same proportional run at a different price/scale maps to the **same**
token id. (This is the acceptance demo for the encoder.)

## DESIGN — LOCKED (full detail: `docs/candle_language/01_design_stage1_candle_language.md`)

**Shared substrate (all panel members use this):** per-bar **vol-unit relative moves** — divide every
magnitude feature by a causal local-vol `σ_t = EWMA(|log(c/c₋₁)|)`: `mv`, `gap=log(o/c₋₁)/σ`,
`body`, `range` (vol-unit) + wick fractions. Pooled quantile/Lloyd-Max bins (fat-tailed, never
Gaussian). Fixed-cadence (UTC-day) re-anchor for 24/7 assets. This satisfies F1=(A): a rescale leaves
vol-units unchanged (transfer) while `×1.25`-placid ≠ `×1.25`-frantic (magnitude is identity).

**The PANEL (F2 resolved — build several, isolated, simultaneous; each a language over the substrate):**
- **L1 BPE-delta** — numerosity-reduce (+run-length tag) → pooled BPE merge, shared codebook, `p_min`
  floor = "viable token" (CPU, cheap; the primary statistical member).
- **L2 ABBA** — adaptive polygonal segmentation by recon tolerance → cluster `(len, inc)` (CPU).
- **L3 SAX+grammar** — Sequitur/RePair → rules-as-tokens, set-cover prune (CPU).
- **L4 learned VQ/FSQ** — 1D-conv VQ-VAE w/ FSQ codebook, reconstruction-only (small GPU/CPU).
- **L5 Matrix-Profile/VALMOD** — symbolization-free variable-length motifs; also the offline validator
  that L1–L4 tokens are real shapes, not binning coincidences.

Variable-length (F3) via numerosity reduction + merging. "Viable" = recurs ≥ `p_min` and survives
near-duplicate clustering + greedy set-cover; everything else = null (`<NULL>`: coded for MDL,
placeholder for embedding, dropped for the trader). **Stage by cost:** L1/L2/L3/L5 first (CPU), add L4.

## Open forks (operator)

- ~~F1 amplitude~~ ✅ (A) vol-unit · ~~F2 discovery~~ ✅ panel (L1–L5) · ~~F3 segmentation~~ ✅
  variable-length · ~~F4 `k`~~ ✅ transfer-driven · ~~F5 run-length~~ ✅ coarse 3-bucket. All resolved
  2026-06-02 — see TASK_789 + `docs/candle_language/`.

## Acceptance (this stage, NOT a predictor)

1. Encoder maps the reference example token correctly, and a rescaled/repositioned copy to the **same**
   token id (invariance demo passes).
2. Produces a finite vocabulary over real minute data for ≥3 assets, with a working encode/decode and
   a "viable vs null" partition.
3. Hands TASK_792 a token stream + vocabulary to judge. **No accuracy/prediction metric here.**

## Out of scope

Embedding (793), trading (794), cross-asset anything, any forecasting objective.
