# The Language of Markets

> *Observation: the specimen has been re-potted fourteen times. Each time I changed the soil, the light, the water, the trellis. The plant is the same plant. This morning I stop changing the conditions and look, for the first time, at the seed. I crack one open under the glass. Inside is a structure I have been feeding for six months and never once described. I begin a new page, and on it I write only what I see — not what I want it to become.*

`workstation` is on. The desk fan is on. There is no GPU job running on this machine — there cannot be, not since the night it hard-locked — and that absence is the whole story of how I got here. In the next window on a separate machine I sometimes use for longer runs, two `result.md` files landed overnight inside cgroups that could not take a host down with them. One is the autopsy of an idea I am burying this morning. The other is the seed of the one I am about to plant.

It is 2026-06-02. The two files are the last loss-side experiment of a campaign that went, by an honest count, zero for fourteen. Every one of those fourteen experiments touched the same thing: the *loss function* of the supervised warm-start phase, or the *heads* attached to its encoder, or the *targets* the oracle fed it. None of them touched the thing the network actually reads. For six months I had been re-potting the plant and never once cracking the seed.

The reframe that produced this article is one sentence, and I wrote it into the lore notes on the last day of May:

> *"the trading objective was never in the loss, and the real planning loop was never turned on."*
> — from `~/Trader/lore/notes_2026-05-31.md` §173

Everything downstream of that sentence is a confession. But this article is not the confession — other entries in this notebook carry that weight. This article is about the one branch the confession *licensed*: stop shaping the loss over a representation that doesn't work, and change the representation.

We tokenize the market. We build it a language.

---

## Why a language, and why shapes and not prices

Here is the intuition, stated the way I'd state it to a smart friend who isn't in this codebase.

A minute-resolution price stream is a firehose of floating-point numbers, almost all of which are noise. The trader I have been building — Poppy, a MuZero-plus-TRM planner that runs Monte-Carlo Tree Search over a learned model — reads that firehose one bar at a time and is supposed to find structure in it. The premise of branch C is that the structure is *linguistic*: markets, like language, are built from a finite vocabulary of recurring shapes. A sharp impulse up, three bars of grinding consolidation, a gap and a wick-rejection, a slow bleed. These recur. They recur across assets, across price levels, across years. They are *words*.

If that is true, then the right move is not to ask the network to predict the next price. The right move is to *build the dictionary first* — find the recurring shapes, give each one a token, and hand the trader a stream of words instead of a stream of numbers. The trader then plans over a compressed, denoised sequence. Candle runs that don't map to any real word get dropped as noise. That is the efficiency payoff, and it is plausible: grammar and BPE compressors report a 2–4× ballpark on this kind of data, which would mean MCTS planning over a sequence two to four times shorter. I write "ballpark" on purpose — it is a reported figure for grammar/BPE compression in general, not a measured result on this data yet, and I refuse to let it harden into a promise.

But — and this is the load-bearing correction, the one that took me a wrong turn to learn — **building the dictionary is not prediction**. A dictionary doesn't forecast the next word. It just tells you which sequences of letters are words and which are gibberish. The tokenizer's entire job is to find which relative-shape runs *recur often enough to be worth a word*. Whether a given word *means* something tradeable is learned later, downstream, and never as a forecast.

I want to dwell on "relative shape," because it is the second load-bearing decision. A word in this language is not "price went from 225.00 to 281.25." It is "price moved up by 1.25× a unit of local volatility, three times, then down by 1.8×, then…" — a sequence of *relative moves*. The same proportional shape that shows up on an asset trading at $225 per unit shows up on an asset trading at $0.45 per unit, and maps to the **same token**. Transfer across assets is not something the tokenizer learns. It is a *property of the encoding*: encode everything as relative moves, fit one shared dictionary, and the dictionary transfers by construction. There is no cross-asset attention anywhere in the tokenizer. That machinery already lives in stage three, the existing trader, where the 64 assets are decided jointly. It is not this layer's job.

---

## The honest backstory: fourteen nulls, two clean nulls, and the reframe

I owe the reader the reason this branch exists at all, because the reason is a graveyard.

The Trader project spent roughly six months on what I've come to call the **single-mechanism campaign**: a sequence of interventions, each one a single mechanism bolted onto the supervised warm-start phase (Stage-B), each one paired against a canonical baseline across three seeds, each one graded on deployment Alpha — paired Δα, same seeds, same data, only the mechanism flipped. The campaign's verdict, written into memory and never revised, is that the whole class is exhausted. Encoder sharpening anti-correlates with Alpha (*better classifier, worse trader*, three independent confirmations, the worst at a paired −18.44pp with a fifty-times trade-rate explosion). The Bella market-generator mechanism class went zero for three. The oracle-side tweaks demoted from small positive to small negative under honest recalibration. Fourteen mechanisms; zero promotions to deployment.

Then came the reframe, from a codebase survey I ran with three parallel readers at the end of May. The unifying insight was structural and a little humiliating: **Stage-B's loss is policy-cross-entropy against MCTS visit counts, value-MSE against oracle PnL, and behavior cloning — and none of those is a realized-PnL gradient.** The network was trained to *classify* (match an oracle) and *measured* on *trading*. The gap between those two is exactly where all fourteen experiments died. Meanwhile the real MuZero Expert-Iteration loop — search, make targets, train, re-search — has been switched off since roughly May 2025. The entire campaign had been optimizing the supervised *seed* of a loop that wasn't running.

Three branches came out of that reframe. **B** turns the loop back on (it boots, it ran 100/100 epochs on the server without collapsing, it's the headline structural bet). **C** is this article. And **A** was the cleanest test of the loss-side question: if the trading objective was never in the loss, put it in the loss.

A is worth its own paragraph, because it is the *good kind* of null, and it is the direct motivation for changing the representation. **A (TASK_787)** added an over-trade penalty to Stage-B at weight 0.1. It came back at **+0.35pp mean paired Δα** vs the canonical 760 a1b1 baseline — a gate fail, but a clean one. The penalty did exactly what it was designed to do: trade-rate fell on all three seeds and balanced accuracy did *not* rise. That is the signature of a *weak* lever, not the classifier-Alpha backfire that killed the encoder experiments (where balanced accuracy rises while Alpha collapses through a trade-rate explosion). Two things were validated in passing: the `w=0.0` reproduction cell paired at −0.37pp, so the pairing protocol itself is sound and there's no cross-campaign confound; and the compile-time gate held bit-identical when OFF. As I put it in the notes that day:

> *"A tells us the mechanism class — shaping the Stage-B loss toward trading — is live and safe to push harder; it just needs a stronger, better-grounded signal than a penalty on over-trading."*
> — from `~/Trader/lore/notes_2026-06-01.md` §174

So I built the principled version. **A2 (TASK_790)** puts the *realized* return directly in the loss, with no oracle hindsight and no new parameter leaf. Read the expected per-asset position straight off the policy —

```
pos = P(Buy) + P(LimitBuy) − P(Sell) − P(LimitSell)     ∈ [−1, 1]
```

— a number that says which way and how hard the policy wants to lean. Multiply it by the horizon-1 realized log-return (`future_log_returns[:, 0, :]`, already a JIT input), and MSE the product against the best achievable per-step PnL:

```
realized_value = mean( (pos · r_next − |r_next|)² )
```

That pushes `pos → sign(r_next)` scaled by move size — a direct "be on the right side of the next bar" gradient, gated by a default-OFF `realized_value_weight` so the OFF-path graph is bit-identical to canon. No oracle hindsight, no new parameter leaf, the loss term simply not traced when the weight is zero.

A2 came back at **+0.06pp**.

To be precise, because precision is the whole point of a lab notebook: paired mean Δα **+0.06pp**, seeds 7 / 13 / 23 at **+0.15 / +0.40 / −0.36**, a gate FAIL — but, like A, a *clean* null. No trade-rate explosion. No anti-correlation backfire. The realized-PnL gradient, pointed straight at money with no hindsight, simply *did nothing*.

> *"two principled, well-behaved attempts at putting PnL in the Stage-B loss, both null. The loss-shaping lever genuinely lacks the authority to move deployment Alpha on this stack."*
> — from `~/Trader/lore/notes_2026-06-02.md` §175

Plus-zero-point-three-five, then plus-zero-point-zero-six. Read those two numbers together and you have the entire thesis of branch C in two data points. Loss-shaping is not harmful — both nulls are clean, neither one a backfire. It is *exhausted*. You can put the trading objective directly in the loss, point a gradient at over-trading and then a better-grounded gradient straight at realized return, and move the deployment metric by a third of a point and then by six hundredths of a point. When the cleanest, best-behaved versions of the loss lever give you *that*, the bottleneck is not the loss. It is what the loss is computed *over*. It is the representation. That is exactly what the reframe predicted: the loss isn't wrong, the representation is fixed and the planning loop is off.

> *The Apothecary tends a bed where every adjustment to water and light yields the same harvest, to the gram. She stops adjusting. She digs up the root and examines its structure under the glass. The yield was never about the watering.*

---

## The mechanism, corrected: minute, non-predictive, vol-unit, single-asset

Before I show the equations, I have to show the corrections — because the *first* version of branch C (the C0 gate) got three things wrong, and the corrections are the design.

**Correction 1 — minute granularity, not hourly.** The original C0 harness aggregated 1-minute bars up to 1-hour. Useless. The whole point is the fine structure; the shapes live on minute candles. No aggregation.

**Correction 2 — non-predictive, full stop.** The original C0 gate judged the language by "predict the next bar's return sign." That is exactly the thing we are not doing. The tokenizer does not forecast. It builds a vocabulary. The "GPT next-token world model" framing and the sign-probe gate were both the wrong frame, and they're thrown out.

**Correction 3 — no cross-asset *learning* in the tokenizer.** It is an adaptive *single-asset* tokenizer. Cross-asset transfer is a property of the relative encoding plus one shared codebook, never a learned cross-attention.

And then the fork that the design's first synthesis pass tried to silently override, and an adversarial critique pass caught:

**F1 — magnitude is part of the token's identity, measured in volatility units.** The lazy version encodes raw `log(c_t / c_{t-1})` and claims scale-invariance "for free" from log-returns. That makes `log(1.25)` on a placid asset and `log(1.25)` on a frantic asset the *same* token. The operator's resolved decision is that they should be *different* — a big version and a small version of the same proportional shape are different words — and the invariance should come from *normalizing by local volatility*, not from discarding magnitude. The critique caught the first draft dropping a resolved decision for no gain (the $225/$0.45 demo works either way), and the fix is the substrate below. *"It dropped a resolved decision for no gain"* — from `~/Trader/docs/candle_language/02_adversarial_critique.md` §0.

### The per-bar substrate — vol-unit relative moves

Every magnitude feature is divided by a **causal local-volatility estimate** before it is binned. Let

```
σ_t = EWMA( |log(c_t / c_{t-1})| , halflife = H )    — causal, only bars ≤ t
```

re-anchored on a fixed minute cadence (every 1440 minutes / one UTC day — *not* "per session," because most of the 64 assets are crypto/FX with no session boundary). Then, per bar:

| Feature | Definition | Role |
|---|---|---|
| `mv_t` | `log(c_t / c_{t-1}) / σ_t` | **vol-unit inter-bar move** — the token-identity magnitude axis |
| `gap_t` | `log(o_t / c_{t-1}) / σ_t` | inter-bar gap (minute bars *do* gap; gap-up-then-flat ≠ grind-up) |
| `body_t` | `log(c_t / o_t) / σ_t` | intrabar direction/strength, vol-unit |
| `range_t` | `log(h_t / l_t) / σ_t` | bar extent, vol-unit |
| `uw_t` | `(h_t − max(o,c)) / (h_t − l_t + ε)` | upper-wick fraction ∈ [0,1] (already scale-free; no vol norm) |
| `lw_t` | `(min(o,c) − l_t) / (h_t − l_t + ε)` | lower-wick fraction ∈ [0,1] |

with `mv_t ≈ gap_t + body_t` — keep gap and body as independent axes and derive `mv` for the move-bin. Every denominator is guarded with `maximum(·, ε)`, and `σ_t` is clamped to a floor so that flat regimes don't blow the vol-units up to infinity. (That floor is where I relocated the classic SAX flat-window pathology: instead of a flat window producing degenerate symbols, it produces a clamped vol estimate.)

Two things fall out of this, and they are the whole reason for the design:

- **The $225 vs $0.45 demo:** a pure proportional rescale of the price series leaves both the log-ratio *and* `σ_t` unchanged. Identical vol-units. Identical token. Transfer, for free, structurally.
- **The F1 gain:** `log(1.25)` on a placid asset (small `σ`) lands in a *higher* move-bin than `log(1.25)` on a frantic asset (large `σ`). Magnitude relative to local regime is preserved as token identity. The two are different words. Exactly what the operator wanted.

### The base alphabet, then variable-length words

Map each bar's vol-unit tuple to a base symbol via a **product alphabet** (1d-SAX, generalized): a move axis (quantile or Lloyd-Max bins on the *pooled* vol-unit distribution — fat-tailed, never Gaussian; minute returns are emphatically not Gaussian, which is precisely why SAX's Gaussian-breakpoint assumption is a lie on this data), a coarse signed gap axis, a joint body·range axis, and a coarse 3-way wick code. Initial cardinality is roughly move-8 / gap-3 / body·range-4 / wick-3. The breakpoints are fitted *once on the pooled 64-asset corpus*, and that shared bin table is what makes the same shape land on the same symbol across assets.

Then two steps turn symbols into variable-length words:

1. **Numerosity reduction + run-length tag** — collapse runs of identical base symbols to one symbol plus a coarse **3-bucket** length tag (short / med / long, fork F5 resolved). `FLAT×short` and `FLAT×long` are different words — "flat for 8 bars" genuinely is not "flat for 40." This is the symbol-level noise filter.
2. **Frequency-greedy merge (BPE / RePair)** over the reduced stream — compose variable-length tokens by merging the most frequent adjacent pairs. Boundaries are *frequency-derived, never reward-learned*. Token bar-length is emergent.

A merge is a **viable token** if and only if it recurs at least `p_min` times across the pooled corpus and survives pruning. The pruning is two stages, and each stage has a job:

- **Near-duplicate clustering** under length-normalized z-normed Euclidean distance (÷√len, so different-length tokens are comparable) collapses *bin-boundary twins* — the case where a move sitting right on a quantile threshold splits one true shape across two adjacent symbols, manufacturing two "tokens" for one word. Cluster them, keep one canonical. (This bin-boundary brittleness is a named hazard in the research findings, not a hypothetical.)
- **Greedy minimum-cardinality set-cover** on fraction-of-stream explained then picks the *smallest* vocabulary that still accounts for the stream — so the dictionary isn't padded with near-synonyms that each explain a sliver.

Everything that survives neither is **null**, and — this reconciles a real contradiction in the first draft — null is represented three different ways for three different consumers:

| Consumer | Null representation |
|---|---|
| MDL / the gate | null bars coded at full base-symbol entropy cost (like-for-like vs the raw baseline) |
| Stage-2 embedding | a single reserved `<NULL>` token (keeps bar-position alignment) |
| The trader | `<NULL>` runs *dropped* from the planned sequence — **this is the denoising payoff** |

One serving detail that is easy to get wrong and breaks everything if you do: **serve with exact BPE merge-order, not greedy longest-match.** Longest-match silently breaks rescale-invariance at serving time (the train/serve segmentation mismatch the critique flagged), so there is an A0 invariance check at serving as well as at training.

### The example token, end to end

The acceptance demo is a specific candle run: `3× up ×1.25, 1× down ×1.8, 1.4, 1.2, 1.2, up ×1.85`. Walk it through. Each `×k` is a vol-unit move (`mv_t = log(price ratio)/σ_t` landing in a move-bin). The run-of-three `up ×1.25` collapses under numerosity reduction to one `UP-bin` symbol carrying a coarse run-length tag — three bars falls in the **short** bucket, so the symbol is `UP×short`, not a literal count (F5 keeps the bucket, not the integer). The sequence of base symbols `[UP×short, DOWN, …, UP]` is then merged by BPE into one variable-length token ID. Now take the *same proportional run* on an asset at $0.45/unit instead of $225/unit: the log-ratios are identical, `σ` rescales identically, the vol-units are identical, every base symbol is identical, the merge order is identical — same token ID. That byte-identity, at a different price scale, is the A0 hard gate. It is the demo that has to pass before anything else is even measured.

---

## The pipeline: language → embedding → trader

Three stages, strictly in order, and only stage one is on the critical path right now.

```
① tokenizer (the language)          TASK_791   non-predictive
        ↓ token IDs + vocabulary
① gate (is the language good?)      TASK_792   non-predictive   ← the real kill-switch
        ↓ PASS
② token-embedding (masked-token)    TASK_793   non-predictive (contextual, not forecast)
        ↓ contextual embeddings
③ existing trader, on tokens        TASK_794   the trader, unchanged in spirit
```

Stage two gives the trader a *metric* over shapes. Token 4001 and token 4002 are unrelated integers; the embedding places them in a space where near-synonyms and context-families are near each other, so the policy generalizes across related shapes instead of memorizing IDs. The objective is **masked-token recovery** — BERT-style cloze over the token stream, bidirectional, *never causal/forecasting* — on the TimeMAE template. Mask random token IDs; a bidirectional encoder classifies the masked token from two-sided context; the hidden states become the contextual embeddings. The embedding table is *initialized* from each token's canonical vol-unit template (the LLM-ABBA trick of seeding embeddings from codebook centroids), not from random.

One subtle smuggle the critique flagged for stage two, and the *reason* it matters: a bidirectional encoder run over the *pooled* token stream will happily let attention and masking context straddle the seam between one asset's sequence and the next's, if the streams are concatenated naively. That re-introduces exactly the cross-asset coupling that Stage-1 spent its entire design avoiding — the encoder would start "reading" asset A's bars to recover a masked token in asset B. So masking and attention must never cross an asset-sequence seam, and there is no asset-ID conditioning: a *language* marker is fine, an *asset* marker is not. A shared word-level embedding space across assets is the goal; cross-asset attention over price is the thing we're keeping out.

Stage three is a *front-end swap*. The existing minute-candle trader/MCTS stack reads tokens and embeddings instead of raw candles. Cross-asset mixing happens here, as it does today — including the cross-sectional channels that the encoder campaign proved are load-bearing to the tune of −18pp if you remove them. MCTS planning is untouched. The efficiency payoff lands here: a compressed token stream, `<NULL>` runs dropped, MCTS planning over a shorter, cleaner sequence — with the hard requirement that dropping non-viable runs costs *no* Alpha.

---

## The research lineage, spilled

I did the thing I always do, which is to go look at whether anyone had built this before. They had — five different communities, each having built a *piece*. I ran a five-lens web-research pass (eight agents, ~400k tokens, workflow `w6fwtir2h`), and here is the catalog, because none of this is invented and all of it is borrowed on purpose.

**Symbolic representation (per-bar → symbols).** **SAX + PAA** gives the invariance recipe at the bar level — z-normalize, segment, quantize to letters — but I take its breakpoints from *empirical quantiles*, not Gaussian, because minute returns are fat-tailed and SAX's Gaussian assumption is a lie on this data. **1d-SAX** is the reason for the product alphabet: encode ≥2 channels per bar (magnitude × shape) as separate quantized axes. **SFA+MCB** contributes its *MCB* half — learn per-channel breakpoints from data — and I drop its DFT half, because the Fourier transform smears the sharp candle edges I care about. **RevIN** contributes only its normalize half: strip level and scale before encoding.

**Variable-length construction (symbols → words).** **ABBA / LLM-ABBA / FAPCA** is the closest off-the-shelf scaffold — the example token *is* an ABBA `(length, +log-move)` tuple; variable-length by reconstruction tolerance; FAPCA's fixed-point re-anchor handles drift on long minute streams. **Sequitur / RePair** (via GrammarViz) gives hierarchical vocabulary for free, and — this is the elegant part — its *rule-utility ≥ 2* criterion is a built-in definition of "viable token," with set-cover pruning included. **BPE-for-time-series** (Tavabi-Lerman; Götz 2025) is the primary statistical builder: discretize, merge frequent adjacent pairs into variable-length tokens, one shared vocabulary across series, 2–4× compression, `p_min` stop. Pooling the merges across all 64 assets is transfer by construction. **Numerosity reduction** is the noise filter that lets a token span different bar-counts — with the run-length tag, so I don't collapse blindly.

**Geometric validators (symbolization-free).** **Matrix Profile** (STOMP / SCRIMP) computes z-normed nearest-neighbor distance per subsequence; it confirms that a BPE "token" is a *tight z-normed shape* and not a binning coincidence. **VALMOD** extends that to variable-length motifs over a length range with length-normalized distance (÷√len, so tokens of different lengths are comparable) — an independent census of what variable-length shapes truly recur. And **uniform-scaling motifs** — matching a shape across a time-stretch — formalize *time-scale* invariance; they're the held-in-reserve fallback if numerosity reduction doesn't fully absorb tempo (a slow version and a fast version of the same shape collapsing to one word). These three are O(n²) and stay strictly offline.

**Learned quantizers (the VQ panel member).** **TOTEM** is the learned-tokenizer skeleton — per-channel 1D-conv VQ-VAE, reconstruction-only, one shared codebook per channel (which is itself cross-asset transfer) — but it is fixed-length and, critically, it *doesn't normalize*, so on its own it would key on absolute price level. So: add BPE for variable length, and normalize. **FSQ** (Finite Scalar Quantization) is the quantizer to use if I go learned, because it is *collapse-proof by construction* — no commitment loss, deterministic, reproducible vocabulary, levels map directly to target vocab size. **LFQ** (MAGVIT-v2) contributes only its code-usage entropy loss as a breadth-keeping lever.

**Stage-2 embedding.** **TimeMAE** is the template — masked-codeword classification, a BERT cloze over tokens, *no forecasting*. **TS2Vec** is an optional contrastive auxiliary, and its eval philosophy (clustering/retrieval, not prediction) is the right one. **LLM-ABBA's cluster-center embeddings** are why I init the embedding table from codebook centroids instead of random.

And one rejection, on principle: **"Dynamic Tokenization via Reinforcement Patching"** learns its boundaries with a *forecasting reward*. That violates the no-prediction rule at the root. Out.

---

## The panel: five isolated languages over one substrate

Here is the design fork that I did *not* resolve by picking a winner.

The discovery method (BPE? ABBA? grammar? VQ? matrix-profile?) and the segmentation strategy were open forks. The obvious move is to benchmark them and pick one. The operator's call, on 2026-06-02, was the opposite: **run all of them at once, isolated and simultaneous — each a different language seeing different structure — because the diversity is the signal.**

| Language | Technique | What it "sees" | Cost |
|---|---|---|---|
| **L1 BPE-delta** | numerosity-reduce → BPE merge, pooled shared codebook | frequent adjacency runs → variable-length words | CPU, cheap |
| **L2 ABBA** | adaptive polygonal segmentation by recon tolerance → cluster `(len, inc)` | piecewise trend pieces | CPU, cheap |
| **L3 SAX+grammar** | symbol string → Sequitur/RePair grammar → rules-as-tokens, set-cover prune | hierarchical motifs | CPU |
| **L4 Learned VQ/FSQ** | 1D-conv VQ-VAE with FSQ codebook, reconstruction-only | reconstruction-optimal latent shapes | small GPU/CPU |
| **L5 Matrix-Profile / VALMOD** | symbolization-free variable-length recurring motifs | geometric recurrence | offline, O(n²) |

The thing that makes the panel *coherent* rather than a zoo is that **every member shares the same vol-unit relative-move per-bar substrate.** They differ only in how they segment, quantize, and merge — not in what they're looking at. So they are directly comparable, and they transfer identically. L1 sees frequent adjacency runs. L2 sees piecewise trend pieces. L3 sees hierarchical motifs (a motif made of motifs). L4 sees whatever a reconstruction-optimal latent decides is worth a code. And **L5 is the validator** — its symbolization-free, geometric census of recurring motifs is the artifact-detector that confirms L1–L4's tokens are genuinely recurring shapes and not binning coincidences. Five witnesses to the same bars; where they agree, the structure is real.

Each language is scored *independently* by the same non-predictive battery. Survivors feed forward as multi-view token streams. The vocab-size fork `k` is resolved not by an eyeballed MDL knee — which is soft-to-absent on fat-tailed financial data, where a `p_min` floor keeps a self-paying Zipf tail that fakes a knee anyway — but by the **cross-asset-transfer-stable region**, with MDL reported as a curve.

---

## The multilingual model: one model that speaks all the panel's languages

The panel emits several token languages. The naive consumption is N separate embedding models, one per language, concatenated at the end — N stovepipes. The operator's decision is the better one, and it borrows directly from how a large language model works: **one model speaks all the languages, the way an LLM speaks English and Spanish and French in a single shared embedding space.**

Each language keeps its own vocabulary (its own token-ID range), tagged with a language marker. Masked-token recovery runs with *cross-language context*: recover a masked L1 (BPE) token given the surrounding L1 *and* L2 (ABBA) *and* L3 (grammar) tokens. The model is forced to learn **cross-language alignment** — a BPE-word and an ABBA-word that describe the same underlying shape land near each other in the shared space. The *dog ≈ perro ≈ chien* effect, for candles. Alignment is the *mechanism that fuses* what each language sees, instead of concatenating views and hoping a downstream head sorts it out.

And here is the insight that makes this more than an analogy — the insight that makes it *cheap*, and that I want to state mechanically rather than wave at. In human multilingual modeling, cross-language alignment is hard precisely because parallel corpora are scarce; you rarely have the exact same sentence in dog-language and perro-language, perfectly aligned word-for-word, so people spend real effort on contrastive alignment objectives to drag the spaces together. **Our languages are perfectly parallel re-encodings of the same bars.** Every language tokenizes the identical minute stream; the shared time axis *is* the parallel corpus. At every bar position I know exactly which L1 token, which L2 token, and which L3 token correspond — because they are all describing bar `t`.

The consequence is the part that's easy to miss: **there is no separate alignment loss to tune.** Alignment falls out as a *byproduct* of masked-token cloze over the co-indexed multi-language stream. When the model recovers a masked L1 token from surrounding L1+L2+L3 context, it is forced — by the geometry of the task, with no extra objective — to put the L2 and L3 tokens that co-occur at the same bar positions to good use, which is to say, to place them where they're predictive of the L1 word, which is to say, near it. The parallel corpus that human multilingual models pay dearly to approximate, this design gets for nothing, and the alignment supervision that would normally be a hand-built contrastive term is *the cloze loss itself*, applied to streams that share a clock.

---

## The non-predictive evaluation: judging a language like a language

This is the part the first version got most wrong, and the part the adversarial critique most improved. The original C0 gate asked "does it predict the next bar's sign?" — a forecaster's question about a thing that is not a forecaster. The replacement battery (TASK_792) asks the right question: *is the vocabulary a real, compressive, transferable language?* Every metric is computable from raw token streams, zero labels, zero prediction.

**The trap I have to flag first**, because it is subtle and it is exactly the kind of mistake the whole project keeps making. The first replacement for the sign-probe was **A3: persist self-transition KL, real stream vs time-shuffled stream**. It sounds principled — "does the tokenized stream have more temporal structure than a shuffle?" It is *circular*. Numerosity reduction and BPE *manufacture* local self-similarity, and minute returns are autocorrelated to begin with, so a shuffled control beating the real stream is near-impossible for *any* tokenizer — even a bad one. A3 tests "does the data have temporal structure" (known) rather than "is this vocabulary good." Same error class as the sign-probe. So A3 is **demoted** to a control *input* only — it is the shuffled stream that A8 consumes — never a standalone proof.

What survives, in cost order:

- **Tier 0 — A0 rescale-invariance** (the cheap pre-gate, a hard gate): re-tokenize a window at a different price scale → the token sequence is *byte-identical*, 100%, at training *and* serving. Catches the longest-match serving bug.
- **Tier 1 — A8 cross-asset transfer (THE headline):** build the vocabulary on a subset; tokenize held-out assets at *extreme* price levels with the same codebook; report mass-weighted coverage of B by tokens that also fire on A, active-vocab Jaccard, and a mid-frequency Spearman frequency-rank correlation. **Exclude the top-k mega-tokens** — the flat/doji fakes that overlap on every asset and are exactly how a bad tokenizer fakes the central claim (this exclusion is, the critique noted, the single best idea in the whole eval). And it must beat **two** controls, both required: (i) a per-asset *returns-shuffled* stream, and (ii) a **random-merge same-size vocabulary**. Control (ii) is the one that matters — it isolates *good merges* from *any merges*. A tokenizer that beats a returns-shuffle but not a random-merge of the same size has learned nothing but the autocorrelation.
- **Tier 2 — A1 corrected MDL, bits-per-candle, over the FULL stream:**
  ```
  bits = Σ_covered log₂(1/p(token)) + Σ_null (base-symbol entropy of raw bars) + L(boundaries) + L(codebook)
  ```
  Null bars *and* boundary bits *and* the codebook are all charged. This is the fix for the most gameable metric in the first draft: a token spans a variable number of candles and 30–40% of bars are null, and if you don't charge those, MDL is rigged in the tokenizer's favor. (The critique's verdict was blunt: *this ambiguity alone flips the go/no-go.*) It must beat (a) a raw base-symbol entropy coder over the *same full stream* and (b) the random-merge same-size vocab.
- **Tier 3 — A2 reconstruction** within ratio-tolerance: ≥ 90–95% of occurrences reconstruct within tolerance, with a bounded mean log-ratio MSE. Always paired with A1, because reconstruction *alone* improves monotonically with alphabet size — it is deceptive on its own.
- **Tier 4 — A9 length non-degeneracy:** median token length in roughly [3, 30]; no single token covers more than X% of bars; not all length-1. The only direct test that variable-length shapes actually *emerged*. Paired with **A4, a coverage band of 60–70% AND null < 25–40%** — deliberately *not* ~100% coverage, because total coverage means you've tokenized the noise and thrown away the denoising payoff, and the null ceiling is what keeps the band from being gamed from the other side.
- **Tier 5 — health bands, never maximized:** active-code fraction > 80% with non-collapsed perplexity (A5); Zipf slope / Rényi efficiency reported as a *band*, not a target, because it's gameable (A6); segmentation stability under sub-tolerance jitter, boundary-F1 > 0.85 (A7 — and note the threshold is 0.85, not the optimistic 0.9, *because* causal `σ` reacts to a jittered bar, which makes stability genuinely harder).

**The verdict rule:** ship a language to Stage-2 if and only if **A8 transfer (mid-freq, mega-tokens excluded, beats both controls) AND A2 fidelity AND A1 corrected-full-stream MDL beats the random-merge same-size vocab AND A9 non-degenerate length.** A0 is the cheap correctness pre-gate; the bands are advisory. And the honest scope statement, which I refuse to let the eval over-claim: this proves the language is *well-formed, recurring, transferable, reconstructible.* It does **not** prove the tokens carry tradeable meaning. That is strictly Stage-2/3, by design. The gate proves the dictionary is a real dictionary. Whether the words mean money is a different question, asked later, never as a forecast.

---

## The engineering: parallelism, streaming, and a crash that taught discipline

Two engineering facts make this branch tractable, and one near-disaster made it safe.

**The sample axis is unbounded.** Because tokens are *single-asset*, the `(asset, window)` pair is a flat sample axis with no inherent ceiling. The project has 64 assets, but 64 is not a batch limit — it is a count of assets, and each asset contributes an arbitrary number of windows. JAX/GPU parallelism maps onto this cleanly: vmap over `(asset, window)` tuples, generate the vol-unit features on the fly, never pre-generate the tokenized stream to disk. This is the project's memory-hierarchy doctrine applied to a new layer — training data lives in VRAM, generated on the fly; control data and the frozen bin-table and codebook live on disk but load to RAM once; the DB feeds a bounded ring buffer that streams to VRAM. No disk touches in the loop. (And the standard JAX hygiene applies: no `lax.cond` inside `vmap` — `jnp.where` instead, which evaluates both branches, so the division guards are not optional; pad dynamic shapes in `lax.scan` to a static `max_len`; `block_until_ready` before any device-to-host transfer.)

**OOM discipline on a shared box.** The tokenizer build and the FSQ codebook training are memory-hungry, and they run on a machine shared with live training jobs. A leak or an unbounded allocation there doesn't just kill your job — it can take a neighbor's down with it. Which brings me to the crash.

On the night this branch was born, I launched the first FSQ build on the workstation's single, display-driving 3090 — the EVGA FTW3 Ultra that also paints the screen I work on. It hard-locked the entire machine at 22:11. The forensics are clean and both of my initial guesses were wrong: **not OOM** (45 GB RAM free, 128 GB swap untouched, zero oom-killer entries) and **not CPU** (no machine-check exception, no thermal event, no panic). The kernel log simply *dead-stopped* — the signature of a too-fast-to-log GPU driver hang — with the last lines being Firefox WebGL on the NVIDIA device, coincident with JAX grabbing the same GPU. A display GPU and a compute job cannot share one card without risking the whole host. That is the causality kick of this article: I knew the box was single-GPU, I knew the GPU drove the display, and I launched a CUDA job on it anyway, and the box was gone before the log could say why.

The fix is not "be careful." The fix is **`tools/safe_run.sh`**: every job runs inside a `systemd-run --user` cgroup scope with a hard `MemoryMax` and `MemorySwapMax=0`, so a runaway is OOM-killed *in its own cgroup, never on the host*. Verified live — a 4 GB allocation under a 1 GB cap was killed, the shell and the host survived. `--cpu` keeps compute off the display GPU entirely. The standing rule now: no GPU jobs on the workstation; CPU-only probes only; GPU compute goes to the separate machine. The cheapest-first experiment for this entire branch runs under `tools/safe_run.sh --cpu --mem 64G` — the whole panel built CPU-first on the non-held-out assets (~58 of the 64), ~6 held out spanning the widest price range ($225 ↔ $0.45), the non-predictive battery run per language. No trader, no GPU, no risk to a host.

> *The Apothecary learned, the hard way, that two cultures sharing one root system will choke each other in the dark. Now each specimen grows in its own bounded pot, and a pot that cracks spills only its own soil. The bench survived. The discipline was the harvest.*

---

## What's left, and what would falsify the whole idea

I want to be honest about where this sits, because the first cut of branch C — the C0 gate, before the reframe — did not just read weak, it returned a formal **KILL**: real-vs-shuffled edge of +0.0003 against a +0.01 bar, real-data next-token accuracy sitting right at the majority baseline (folds at 0.4968/0.5056 and 0.5006/0.5039, i.e. ~zero edge before you even subtract the returns-shuffled control), and the FSQ codebook collapsing hard (1435 codes used out of 12,800). For a few minutes that read as "tokens are dead." And then I caught myself — or rather the operator did, and I want to record the catch because it is the same self-correction the whole project keeps having to make. That 893-second run *could not have tested the idea.* Reading the harness confirmed it: C0 aggregated to **hourly** (we trade minutes), tested a **5-asset toy**, and — fatally — judged the vocabulary by a **next-bar-sign probe**, the one thing this design explicitly forbids. It killed a *strawman*, not the mechanism. The KILL was **withdrawn as decisive.**

That is worth dwelling on, because it is the twin of the deeper failure mode of this entire campaign: *measuring the wrong thing produces confident, worthless verdicts.* The sign-probe gate is the same error class as the fourteen dead classifier tweaks — a metric that moves for reasons unrelated to the thing you care about. A KILL from a strawman gate is not evidence against the idea; it is evidence the gate was a strawman. The redesigned, non-predictive, minute-granularity, vol-unit, paneled version has not been run yet. That distinction — the difference between killing a toy and killing a mechanism — is the whole reason the C0 lesson exists, and it's the reason a half-day kill-switch is worth building *correctly* before the expensive thing.

The cheapest experiment is a CPU-first, no-trader, no-GPU build of the panel, judged by the battery. **PROCEED** if A8 beats both controls at mid-frequency *and* A1-corrected beats the random-merge once null and boundary bits are charged *and* A2 reconstruction clears 90% *and* A9 length is non-degenerate. **KILL** if A8 collapses to a control, or A1-corrected can't beat a random-merge of the same size, or the vocab degenerates. None of those is a forecast. The kill-switch is information-theoretic, label-free, and runs on a CPU in a bounded cgroup. That is the apothecary version of careful: the test that ends the idea cheaply, before any expensive thing is built on top of it, is designed *first*. (On a soft fail the sweep order is fixed too — bins `K` → `p_min`/vocab size → run-length tag granularity → wick resolution — so a near-miss gets one disciplined pass, not infinite knob-twiddling.)

What would falsify the entire bet? If a vocabulary of recurring vol-unit shapes is no more compressive, no more transferable, and no more reconstructible than a random merge of the same size — then markets, at minute granularity, are not linguistic in the way I've assumed, and there are no words to find. The random-merge control is the falsifier. It is the question "is there structure here that *good* merges capture and *any* merges don't?" stated as a number. If the answer is no, branch C dies clean, the way A and A2 died clean at +0.35pp and +0.06pp, and I will have learned that the representation was not the bottleneck either — which would point, hard, at branch B and the dormant planning loop as the last structural bet standing.

This is a bet on representation, made *after* the loss side was tested to two clean nulls — a third of a point, then six hundredths — and found exhausted. It might be wrong. But it is the first thing in six months that cracks open the seed instead of changing the soil — and the test that would prove it wrong is the cheapest thing in the whole design.

> *The Apothecary writes the dictionary before she reads the meaning. A word is not a prophecy. It is only a shape that recurs often enough to deserve a name. She names what recurs, and waits to see what it says.*

---

*Sources, all in `~/Trader`: `docs/candle_language/README.md`, `01_design_stage1_candle_language.md`, `02_adversarial_critique.md`, `03_research_findings_digest.md` (the panel, the vol-unit substrate, the A0–A9 battery, the method lineage; from research workflow `w6fwtir2h`). Task specs `tasks/TASK_789`–`TASK_794` (the reframe, the four stages, the resolved forks F1–F5) and `tasks/TASK_790` (the realized-PnL value term, A2). Session notes `lore/notes_2026-05-31.md` §173 (the single-mechanism cycle closing, the three branches, the crash, `safe_run`), `lore/notes_2026-06-01.md` §174 (A's clean +0.35pp null, A2 launched, C0's weak early read), and `lore/notes_2026-06-02.md` §175 (A2's returned +0.06pp clean null, the C0 KILL withdrawn as a strawman, the language reframe). `runs/790_realized_pnl/result.md` — A2's paired verdict. `tools/safe_run.sh` — the cgroup job envelope.*
