# The Candle Language — Stage-1 tokenizer research & design

**What this is:** the research, design, and decisions for the **token middle-layer** (branch C,
`tasks/TASK_789`) — an adaptive, **non-predictive** tokenizer that turns minute OHLC into a vocabulary
of recurring relative-shape "words", so the trader can work on a compressed, denoised token stream.

**Created:** 2026-06-02 from research workflow `w6fwtir2h` (5-lens web research → synthesis →
adversarial critique → revised synthesis; 8 agents, ~400k tokens).

## Files

| File | What |
|---|---|
| `01_design_stage1_candle_language.md` | The design proposal (tokenizer + non-predictive eval battery), revised after critique. |
| `02_adversarial_critique.md` | The adversarial pass that caught the first draft skipping the F1 vol-unit decision and proposing a circular eval gate. Read to see *why* the design is shaped as it is. |
| `03_research_findings_digest.md` | Curated catalog of the borrowable methods across all 5 lenses. |
| `_raw_research_w6fwtir2h.json` | Complete verbatim workflow output (all 5 lenses, full findings, both synthesis passes). Source of truth. |

## The mechanism (operator's framing, corrected 2026-06-02)

A **token** = a recurring, variable-length run of candle shapes in **vol-unit relative moves**
(magnitude measured in local-volatility units → fork **F1 = (A)**, magnitude is part of token
identity). Invariant to absolute price / scale / position / location, so the same shape recurs across
the 64 assets at any price level ($225/unit ↔ $0.45/unit → same token). **No prediction anywhere** —
the tokenizer only builds a vocabulary; "meaning" is learned later (Stage 2), trading later (Stage 3).
Cross-asset transfer is a **property of the relative encoding + one shared codebook**, never learned
cross-asset attention.

Pipeline: ① tokenizer (language) → ② token-embedding (masked-token, non-predictive) → ③ existing
trader on tokens. Critical path is ①, gated by a **non-predictive** quality battery (replaces the
discarded next-bar-sign probe).

## DECISION (2026-06-02) — a PANEL of isolated languages, run simultaneously

The forks (F2 discovery / F3 segmentation) are **not** resolved by picking one method. Operator call:
**run multiple isolated tokenizers at once, each a different technique, each seeing different
structure** — the diversity is the signal. All panel members share the **same vol-unit relative-move
per-bar substrate** (so they're comparable and transfer identically); they differ only in how they
segment / quantize / merge:

| Language | Technique | What it "sees" | Cost |
|---|---|---|---|
| **L1 BPE-delta** | numerosity-reduce → BPE merge, pooled shared codebook (Götz 2025 / Tavabi-Lerman) | frequent adjacency runs → variable-length words | CPU, cheap |
| **L2 ABBA** | adaptive polygonal segmentation by recon tolerance → cluster (len, inc) | piecewise trend pieces | CPU, cheap |
| **L3 SAX+grammar** | symbol string → Sequitur/RePair grammar → rules as tokens, set-cover prune (GrammarViz) | hierarchical motifs | CPU |
| **L4 Learned VQ/FSQ** | 1D-conv VQ-VAE w/ FSQ codebook, reconstruction-only (TOTEM + FSQ) | reconstruction-optimal latent shapes | small GPU/CPU |
| **L5 Matrix-Profile / VALMOD** | symbolization-free variable-length recurring motifs | geometric recurrence (also validator for L1–L4) | offline, O(n²) |

Each is evaluated **independently** by the same non-predictive battery (`TASK_792`). Survivors feed
forward as **multi-view token streams** (Stage 2/3 consume several parallel channels — fusion is a
793/794 design question, not a Stage-1 concern). L5 doubles as the artifact-detector that confirms
L1–L4 tokens are genuinely recurring shapes, not binning coincidences.

## Open forks after this decision

- **F2 / F3** → resolved as "all of them, in parallel" (the panel above).
- **F4** (vocab size `k`): driven by the **cross-asset-transfer-stable region**, MDL reported as a
  curve — *not* the soft MDL knee.
- **F5** (run-length tag): **coarse 3-bucket** (short/med/long) — keep the count, "flat for 8" ≠
  "flat for 40".
- **New (Stage 3):** multi-view fusion — concat embeddings vs per-language heads vs cross-view
  attention at the decision head. Deferred to TASK_794.

## Status

Design locked (panel approach). Next: the cheapest first experiment — build the panel CPU-first on
~58 assets, hold out ~6 spanning the widest price range, run the non-predictive battery per language.
No trader, no GPU on the workstation (`tools/safe_run.sh --cpu --mem 64G`). See `TASK_791`/`TASK_792`.
