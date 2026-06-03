# TASK_793 — Stage ②: Token-embedding model (self-supervised, non-predictive)

**Status**: BLOCKED on TASK_792 PASS
**Created**: 2026-06-02 · **Parent**: TASK_789 · **Depends on**: TASK_791/792 · **Next**: TASK_794

## Goal

Train a model **on the token sequences** that learns a **contextual embedding** for each token — its
"meaning" given the company it keeps. This is the **word2vec / masked-token** stage, not a forecaster.
The output is an embedding table / contextual encoder that stage ③ (the trader) consumes instead of
raw candles.

## Multilingual: ONE model speaks all panel languages (operator 2026-06-02)

The Stage-1 panel (TASK_791) emits **several** token languages (L1 BPE, L2 ABBA, L3 grammar, L4 VQ, …),
each seeing different structure. They are **not** embedded by N separate models — exactly the way an
LLM speaks many human languages in one model, **a single multilingual embedding model ingests all
panel languages into one shared space**:
- each language keeps its own vocabulary (own token-id range), tagged by a language marker;
- masked-token recovery runs with **cross-language context** (recover a masked L1 token given
  surrounding L1 + L2 + L3 tokens), so the model learns **cross-language alignment** — a BPE-word and
  an ABBA-word for the same underlying shape land **near each other** in the shared space (the
  *dog≈perro≈chien* effect). Template: TimeMAE masked-codeword classification (see
  `docs/candle_language/03_*` §E).
- Benefit: the diversity of views coexists *and* reinforces in one model; alignment is the mechanism
  that fuses what each language sees, rather than concatenating N stovepipes.
- Constraint preserved: still **non-predictive** (masked recovery, not forecasting); masking/attention
  must **not** cross asset-sequence seams; no asset-ID conditioning (a *language* marker is fine, an
  *asset* marker is not — that would smuggle cross-asset coupling).

## Constraint — still no forecasting

The objective is **self-supervised contextual**, not next-step prediction:
- masked-token reconstruction (BERT-style over the token stream), and/or
- skip-gram / CBOW co-occurrence (word2vec) over the token sequence, and/or
- contrastive (a token's neighborhood vs negatives).

A token "may signal something may happen" — but that association is left for stage ③ to exploit; this
stage only places tokens in an embedding space where **similar/co-occurring shapes are near each
other**. We do not train this stage to predict returns.

## Why an embedding stage at all

Tokens out of stage ① are discrete ids with no notion of similarity (token 4001 vs 4002 are unrelated
integers). The embedding gives the trader a *metric* over shapes — near-synonyms, context families —
so the downstream policy generalizes across related shapes instead of memorizing ids.

## Acceptance (non-predictive)

- Embedding neighborhoods are **coherent**: nearest neighbors of a token are shapes a human/heuristic
  would call similar (or that co-occur in similar contexts) — measured by neighborhood purity /
  analogy-style structure, **not** by return prediction.
- Stable training (the discrete regime is the point — should be far more stable than the continuous
  dynamics-consistency losses that kept collapsing).
- Hands stage ③ a frozen embedding (or fine-tunable, fork below).

## Open forks

- **embedding objective** (masked vs skip-gram vs contrastive) — set after 792, informed by the
  vocabulary's statistics (Zipfian → word2vec-style works well).
- **frozen vs fine-tunable** into the trader (default: frozen first, mirror of the tokenizer's
  frozen-first discipline).

## Out of scope

Trading, cross-asset mixing, any reward. Those are TASK_794.
