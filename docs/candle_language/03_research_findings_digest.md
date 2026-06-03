# Research findings digest — methods catalog

> Curated from the 5-lens web-research pass (`w6fwtir2h`). Full per-lens findings, sources, and the
> complete `borrow`/`warning` lists are in `_raw_research_w6fwtir2h.json`. This is the working catalog.

## A. Symbolic representation / base alphabet (per-bar → symbols)

| Method | One-line | Borrow |
|---|---|---|
| **SAX + PAA** | per-window z-norm → segment-means → equiprobable letters | invariance recipe at bar level; but use **empirical-quantile** bins, not Gaussian (minute returns are fat-tailed) |
| **1d-SAX** | mean + slope per segment, quantized separately | encode ≥2 channels per bar (magnitude × shape) as a **product alphabet** |
| **SFA + MCB** | DFT + per-coefficient data-adaptive bins | take **MCB** (learn breakpoints per channel from data); **drop the DFT** (smears sharp candles) |
| **RevIN** | reversible per-instance norm | borrow the **normalize half** to strip level/scale before a learned encoder |

## B. Variable-length token construction (symbols → words)

| Method | One-line | Borrow |
|---|---|---|
| **ABBA / LLM-ABBA / FAPCA** | adaptive polygonal segmentation → cluster `(len, inc)` tuples; FAPCA = fixed-point re-anchor | **closest off-the-shelf scaffold**: the example token *is* an ABBA `(len, +log-move)`; variable-length by recon tolerance; FAPCA fixes drift on long minute streams |
| **Sequitur / RePair (GrammarViz)** | grammar induction; non-terminals = recurring variable-length motifs; rule-utility ≥2 | hierarchical vocabulary "for free"; **rule-utility = a built-in "viable token" definition**; set-cover pruning |
| **BPE-for-TS (Tavabi-Lerman; Götz 2025)** | discretize → merge frequent adjacent pairs into variable-length tokens; **one shared vocab across series**, 2–4× compression, `p_min` stop | **the primary statistical builder**: pooled-across-64-assets merges = transfer by construction; `p_min` = viable-token floor; AR/delta variation = relative-move encoding |
| **Numerosity reduction** | collapse consecutive identical symbols | the spec's noise filter; lets a token span different bar counts (keep a run-length tag — don't collapse blindly) |

## C. Geometric validators (symbolization-free)

| Method | One-line | Borrow |
|---|---|---|
| **Matrix Profile (STOMP/SCRIMP)** | z-normed nearest-neighbor distance per subsequence → motifs/discords | confirm BPE tokens are tight z-normed shapes, not binning coincidences (offline audit) |
| **VALMOD** | variable-length motifs over `[l_min, l_max]`, length-normalized distance | independent census of what variable-length shapes truly recur; use **÷√len** to compare tokens of different lengths |
| **Uniform-scaling motifs** | match a shape across time-stretch | formalizes time-scale invariance; fallback if numerosity reduction doesn't absorb tempo |

## D. Learned quantizers (the VQ panel member)

| Method | One-line | Borrow |
|---|---|---|
| **TOTEM** | per-channel 1D-conv VQ-VAE, reconstruction-only, one shared codebook per channel | learned-tokenizer skeleton; per-channel-shared codebook = cross-asset transfer; **but** fixed-length (F=4) → add BPE for variable length; **must normalize** (TOTEM doesn't → would key on absolute level) |
| **FSQ** | project to few dims, round to fixed levels; **collapse-proof** | the quantizer to use if learned — no commitment loss, deterministic, reproducible vocab; levels → target vocab size |
| **LFQ (MAGVIT-v2)** | lookup-free binarized dims + entropy loss; very large vocab | scaling lever; borrow the **code-usage entropy loss** to keep any learned vocab broad |

## E. Stage-2 token embedding (non-predictive "meaning")

| Method | One-line | Borrow |
|---|---|---|
| **TimeMAE** | masked-codeword classification (BERT cloze over tokens) + masked-rep regression; **no forecasting** | the Stage-2 template: mask token IDs, recover from bidirectional context → contextual embeddings |
| **TS2Vec** | hierarchical contrastive over augmented views | optional contrastive auxiliary; eval philosophy = clustering/retrieval, not prediction |
| **LLM-ABBA cluster-center embeddings** | seed token embeddings from codebook centroids | init Stage-2 embeddings from canonical vol-unit templates, not random |

## F. Non-predictive vocabulary-quality metrics (the gate)

The replacement for the discarded next-bar-sign probe — all label-free, all in `TASK_792`:

- **Compression / MDL** — model + residual bits; pick alphabet size by total description length (Hu/Keogh).
- **Coverage / grammar-cover** — fraction of bars in a viable token (band, not maximized).
- **Cross-asset token overlap / reuse** — same token fires on multiple assets at different price levels
  (the direct transfer test; the *headline* — A8).
- **Reconstruction fidelity** — decode tokens back to the candle shape within ratio-tolerance.
- **Codebook utilization / perplexity** — most tokens used, no collapse.
- **Token-frequency Zipf / Rényi** — health **band**, never a target (gameable).
- **Persist KL vs shuffle** — *demoted* to a control input only (circular as a standalone — see critique §1).

**Cross-cutting warnings (from the raw findings):** reconstruction alone is deceptive (monotone in
alphabet size — always pair with MDL/utilization); bin-boundary brittleness (near-threshold moves split
across symbols → cluster near-duplicates); flat/illiquid runs auto-compress to a mega-token and skew the
vocab; Matrix-Profile/VALMOD are O(n²) → offline audits only; **reject "Dynamic Tokenization via
Reinforcement Patching"** (arXiv:2603.26097) — its boundaries are learned with a *forecasting* reward,
violating the no-prediction rule.
