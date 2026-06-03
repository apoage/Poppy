# TASK_792 — Stage ① gate: Non-predictive language-quality evaluation

**Status**: DESIGN (metric battery + thresholds pending research workflow `w6fwtir2h`)
**Created**: 2026-06-02 · **Parent**: TASK_789 · **Depends on**: TASK_791 · **Kill-switch for branch C**

## Why this task exists (the C0 lesson)

The original C0 gate judged the language by **"predict next-bar sign"** — exactly the thing we are
*not* doing — on **hourly** data with a 5-asset toy. Verdict was meaningless and is thrown out. This
task replaces it with a battery that judges **"is the language good?"** using **no downstream
predictor and no prediction framing.** This is the real proceed/kill gate before any embedding or
trader is built.

## Principle

A good *language* is judged like a language/codebook, not like a forecaster:
- **Compression / MDL** — does tokenizing the minute stream reduce description length vs raw candles?
  A real vocabulary compresses; noise-fitting does not.
- **Coverage** — what fraction of the stream is explained by *viable* tokens (vs null/noise)?
- **Vocabulary shape** — is the token-frequency distribution **Zipfian** (natural languages are), with
  healthy utilization (no collapse to a few codes, no blow-up to all-singletons)?
- **Reconstruction fidelity** — can the (de-normalized) price path be rebuilt from tokens within the
  relative-shape tolerance? (Reward-critical for stage ③ — phantom decode error → phantom alpha.)
- **Cross-asset transfer** — do the **same** tokens recur across assets (vocabulary overlap / reuse
  rate)? This is the operationalization of "tokens are cross-asset transferable" — and it needs **no**
  predictor: measure how much of asset B's stream is covered by a vocabulary built on asset A.
- **Stability** — segmentation/token assignment is stable under small perturbation (re-seed, jitter,
  rescale → same tokens).

## DESIGN — LOCKED battery (full detail: `docs/candle_language/01_*` §4, critique §1-2 for *why*)

Run per-language (each panel member scored independently, same battery):
- **A0 rescale-invariance** (pre-gate): re-tokenize at a different price scale → byte-identical, 100%,
  at train AND serving (catches the longest-match bug).
- **A8 cross-asset transfer (HEADLINE):** build on a subset, score held-out assets at extreme price
  levels; mid-frequency coverage + rank-correlation, **mega-tokens excluded**; must beat **BOTH**
  controls — returns-shuffled stream AND a **random-merge same-size vocab** (the latter isolates *good*
  merges from *any* merges).
- **A1 corrected MDL** (full stream — charges null bars + boundary bits + codebook): beat raw entropy
  AND the random-merge vocab. Vocab size `k` from the A8-transfer-stable region, not the MDL knee.
- **A2 reconstruction** within ratio-tolerance ≥90-95%. **A9 length non-degeneracy** (median ∈ ~[3,30],
  no mega-token, not all length-1). **A4 coverage band** 60-70% (NOT ~100% — total coverage = noise).
- Bands (never maximized): A5 utilization >80%, A6 Zipf/Rényi as a health band, A7 stability >0.85.
- **A3 persist-vs-time-shuffle is DEMOTED** — circular (any tokenizer beats its own time-shuffle on
  autocorrelated minute data); kept only as the shuffled-control input to A8.

**Verdict:** ship a language iff **A8 (both controls, mega-excluded) AND A2 AND A1-corrected-vs-random-
merge AND A9**. Necessary-not-sufficient (proves well-formed/transferable, NOT tradeable — that's 2/3).

## Acceptance / verdict

PASS (→ build TASK_793) requires, at minimum: compression beats raw, coverage above threshold, a
Zipfian/healthy vocabulary, reconstruction within tolerance, **and** demonstrable cross-asset token
reuse above a returns-shuffled / per-asset-shuffled control. KILL otherwise — and unlike C0, the KILL
will mean the *language* is bad, not that a strawman probe failed.

## Explicit non-goals

No next-bar prediction, no directional accuracy, no trading reward, no cross-asset model. Those are
later stages. This gate only asks: *is the vocabulary a real, compressive, transferable language?*

## CPU-first

The battery is information-theoretic / counting / reconstruction — CPU-friendly. Run under
`tools/safe_run.sh --cpu --mem 64G` (no GPU on the workstation; server CPU fine).
