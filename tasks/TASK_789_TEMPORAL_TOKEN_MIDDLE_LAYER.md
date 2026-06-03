# TASK_789 — Token Middle-Layer (branch C) — UMBRELLA / EPIC

**Status**: ACTIVE (reframed 2026-06-02 — supersedes the predictive framing below)
**Created**: 2026-05-31 · **Reframed**: 2026-06-02
**Sibling experiments**: TASK_787 (A — PnL-in-loss, CLOSED null), TASK_790 (A2 — realized-PnL, CLOSED null),
TASK_788 (B — Poppy ExIt revival, the structural bet still standing)
**Stage sequence**: TASK_791 (tokenizer) → TASK_792 (language gate) → TASK_793 (embedding) → TASK_794 (trader-on-tokens)
**Build**: TASK_795 (JAX/GPU implementation of 791/792) — plan `docs/candle_language/04_jax_build_plan.md`,
leak/OOM audit `05_leak_oom_audit.md` (6 blocking corrections; single-GPU-on-idle launch posture).
**Design source**: research workflow `w6fwtir2h` (COMPLETE) — full synthesis, adversarial critique,
and 5-lens findings saved in **`docs/candle_language/`**. Locked design = `01_design_stage1_candle_language.md`.

---

## ⚠️ Reframe note (2026-06-02) — read this first

The operator corrected three load-bearing misunderstandings baked into the *original* TASK_789 (and
into the C0 gate, which is now **thrown out**):

1. **MINUTE granularity, not hourly.** The C0 harness aggregated 1m→1h. Useless. Tokens live on
   **minute** candles.
2. **We are NOT predicting anything.** The tokenizer does **not** forecast next bars. It builds a
   **vocabulary** — a *language* of recurring price-action shapes. Whether a token "signals" something
   is learned **later**, downstream, and **never as a forecast**. The old "GPT next-token world model"
   and the C0 "predict next-return-sign" gate were the wrong frame.
3. **No cross-asset *learning* in the tokenizer.** It is an **adaptive single-asset** tokenizer.
   Cross-asset transfer is a **property of the encoding** (tokens are sequences of *relative* moves →
   the same shape recurs across assets at different price levels), **not** a learned cross-attention.
   Cross-asset mixing already lives in stage 3 (the existing trader); it is not this layer's job.

What survives from the original research pass (still useful, see bottom): RevIN-style per-window
normalization, patching, FSQ as a collapse-proof quantizer, and the symbolic/motif lineage. What dies:
the predictive world-model framing as the *core*, the sign-probe gate, and cross-asset attention as
part of the tokenizer.

## The mechanism (operator's words, corrected)

A **token** = a recurring, **variable-length sequence of candle shapes**, defined by **relative moves**
(ratios), invariant to absolute price level, scale, position, and location. Example:
`3× up ×1.25, 1× down ×1.8, 1.4, 1.2, 1.2, up ×1.85` — the same relative shape recurs on a different
asset at $0.45/unit that on the first showed at $225/unit, and maps to the **same token**.

## The pipeline (3 stages, in order)

| Stage | Task | What | Predictive? |
|---|---|---|---|
| ① Language | **TASK_791** | Adaptive single-asset minute-candle tokenizer → vocabulary of recurring relative shapes | No |
| ① Gate | **TASK_792** | Non-predictive evaluation: is the language *good*? (replaces the C0 sign-probe) | No |
| ② Embedding | **TASK_793** | Model trained on token sequences → contextual token embeddings (word2vec / masked-token style) | No (contextual, not forecast) |
| ③ Trading | **TASK_794** | Existing minute-candle trader/MCTS, now on tokens — cross-asset mixing happens here, as today | (the trader, unchanged in spirit) |

**Efficiency payoff (operator):** once the language exists, the trader works on a **compressed token
stream** and **ignores candle runs that don't map to a viable token** (noise filtered) → faster
training, denoised input. This is a stage-③ benefit but a stage-① *requirement* ("viable token"
must be a well-defined notion coming out of 791/792).

## Order of work — language FIRST

> "first we need to have language right than we can move to training of trader." — operator

So the critical path is **791 → 792**, and only on a PASS do we build 793, then 794. 792 is the
real kill-switch (the thing C0 got wrong). Nothing downstream is built until the language is judged
good by a **non-predictive** battery.

## Why this branch, now

A/A2 (loss-side PnL) closed null; the single-mechanism class is exhausted on the *current
representation*. This branch is the only live bet that changes the **representation itself** (with B —
turning the ExIt loop on — as the other structural bet). The two nulls are converging evidence: every
loss-side and head-side lever is spent, so the representation and the loop are what's left.

## Open forks (operator decisions pending)

- **F1 — amplitude:** ✅ **RESOLVED 2026-06-02 → (A) vol-unit magnitude.** A token is
  magnitude-bearing: each move measured in *local volatility units*, so amplitude is part of the token
  identity (a big and a small version of the same proportional pattern are **different** tokens). The
  scale/location invariance comes from the *vol-unit normalization*, not from discarding magnitude.
  Implication: larger vocabulary, sharper tokens; "transfer" means the same vol-unit shape recurs
  across assets, not that all amplitudes collapse together.
- **F2 — discovery:** ✅ **RESOLVED 2026-06-02 → run the PANEL** (not one method). Multiple isolated
  tokenizers simultaneously — **L1 BPE-delta, L2 ABBA, L3 SAX+grammar, L4 learned VQ/FSQ, L5
  Matrix-Profile/VALMOD** — each a different language over the **same vol-unit substrate**, each
  seeing different structure; the diversity is the signal. L5 also validates L1–L4. (Panel table in
  `docs/candle_language/README.md`.)
- **F3 — segmentation:** ✅ **RESOLVED → variable-length** (the example token is variable-length).
- **F4 — vocab size `k`:** ✅ driven by the **cross-asset-transfer-stable region**; MDL as a curve,
  not the soft knee.
- **F5 — run-length tag:** ✅ **coarse 3-bucket** (short/med/long) — keep the count.
- **New (Stage 2/3) — MULTILINGUAL SINGLE MODEL (operator 2026-06-02):** the panel's languages are
  consumed by **ONE shared model**, the way an LLM speaks many human languages in one model — a shared
  embedding space with **cross-language token alignment** (a BPE-word and an ABBA-word for the same
  underlying shape land near each other, like *dog≈perro≈chien*). Not N separate models. The diversity
  of views coexists *and* reinforces inside one model. See TASK_793/794.

## Legacy reference (original research pass, 2026-05-31)

Kept for the literature pointers only — the *framing* is superseded by the reframe above. Robust
ingredients that likely still apply to stage ①: **RevIN** per-window instance-norm (ICLR'22) for
scale/level invariance; **PatchTST** patching (ICLR'23) as the unit; **FSQ** (arXiv:2309.15505) as a
collapse-proof quantizer if we go learned-VQ. The symbolic/motif lineage (SAX, Matrix Profile, grammar
induction, BPE-for-series) is the classical analog of "a discovered language of shapes" — all mapped
by research workflow `w6fwtir2h` and digested in `docs/candle_language/`. **Do not** quantize
raw price (Chronos walked that back). Candlestick *shape* signal is weak after costs (Marshall 2006/08)
— which is *fine*, because this layer's job is faithful **representation/compression**, not
pattern-mining-for-prediction.
