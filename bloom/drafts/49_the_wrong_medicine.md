# The Wrong Medicine

> *Field note, the morning after. I reached into the cabinet for the assay and pulled down the jar one shelf to the left. The label was right. The contents were not. I am leaving this entry exactly as I wrote it through the night — the handwriting that loops and the handwriting that shakes — because the cabinet sang to me and every note it sang was, against all reason, in tune. The medicine was wrong. The math was not. Read it knowing which was which.*

There is one signal.

Not sixty-four. Not one per asset, racing each other down their own little ladders of price. One. I learned this the way you learn anything real, which is sideways, at 3 a.m., with the wrong tincture warm in my mouth and the whole apothecary cabinet humming a single low note like a finger run around a wine glass that never empties.

Here is what they don't tell you on the radio, between the songs about freedom: a market is not a number. A market is a *waveform*. Strip the price off it — divide every move by its own local breath, $\sigma_t$, the causal exponential running-average of how hard the thing has been breathing —

$$\sigma_t = \mathrm{EWMA}\big(\,|\log(c_t/c_{t-1})|\,,\ \text{halflife}=H\,\big), \qquad \text{only bars} \le t$$

$$x_t = \frac{\log(c_t / c_{t-1})}{\sigma_t}$$

and what's left, $x_t$, is dimensionless. Scale-free. A move measured in *units of itself*. An asset at \$3 million a unit and an asset at a fraction of a cent, rescaled by their own volatility, become **the same song in different keys**. Multiply the whole price series by any constant you like — the log-ratio doesn't move, $\sigma_t$ doesn't move, $x_t$ doesn't move. Invariance, not bolted on, but *structural*. The patient stripped to the waveform underneath the flesh.

So there is one signal, $x_t$. And the entire question of branch C — the question I have been circling for a season, building dictionaries of candle-shapes and calling them words — collapses into one clean sentence that I want to carve into the workbench:

> **Tokenizing the market is structured coding of one 1-D signal. And every method for doing it is a member of the signal-processing toolbox, competing in a tournament, refereed by compression.**

That's it. That's the whole acid-clear thing the wrong medicine showed me. Let me walk you through the cabinet while the jars are still singing.

---

## The universal solvent is compression

Every herbalist has a solvent that dissolves everything, so you can see what was really in the leaf. Mine is *compression*, and it is not a metaphor — it is a theorem wearing a lab coat.

The ideal measure of how much structure lives in a string $x$ is its **Kolmogorov complexity** $K(x)$: the length of the shortest program that prints it and halts. A string with deep structure is short to describe; a string of pure noise is its own shortest description. $K(x)$ is the perfect assay. It is also *uncomputable* — there is no machine that returns it, ever, by Turing's old curse. So we do what apothecaries always do with a perfect medicine we cannot synthesize: we brew the closest computable shadow. The shadow of $K(x)$ is a **real compressor's codelength**, and the principle that lets us drink it is **Minimum Description Length**: the best model of the data is the one that lets you transmit the data in the fewest bits.

A tokenizer *is* a compressor. Say it again until the cabinet stops ringing. A tokenizer is a compressor. When I segment the waveform into recurring shapes and give each one an id, I am building a code. The codelength of that code, in bits per token, is the assay. Low bits means the structure I found is *real* — it recurs, it pays for itself, it transmits.

And here is the move that makes the whole season worth it. Two assets, $A$ and $B$. I build my dictionary on $A$. I ask: how many bits does it cost to encode $B$ under $A$'s code? If $A$ and $B$ share structure, $B$ is *cheap under $A$* — the shapes that are common on $A$ are common on $B$, so $B$'s mass lands where $A$ put its short codewords. **Cross-asset transfer is cross-asset compressibility.** The handshake between two markets is whether they compress better together than apart. This is not philosophy. It is the formula I score the whole panel on.

There's even a named distance for it, and it's beautiful — the **Normalized Compression Distance** of Cilibrasi and Vitányi:

$$\mathrm{NCD}(x,y) = \frac{C(xy) - \min\!\big(C(x),\,C(y)\big)}{\max\!\big(C(x),\,C(y)\big)}$$

where $C(\cdot)$ is any real compressor's output length. If $x$ and $y$ are the same song, $C(xy) \approx C(x)$ and the distance falls toward zero. If they share nothing, concatenating them buys nothing and the distance climbs toward one. The Kolmogorov handshake, computable, in production, today.

---

## A cautionary trip: the medicine that was cut

Now — and the cabinet got *loud* here, so forgive the handwriting — I have to tell you about the time the whole world got high on this and forgot to balance the equation.

In 2023 a paper went around like a good rumor at a festival: *"Low-Resource Text Classification: gzip beats BERT."* Jiang and colleagues. The recipe was gorgeous in its insolence: don't train a billion-parameter transformer; just `gzip` your text, compute NCD between documents, run **$k$-nearest-neighbors**, and watch a forty-year-old compression utility out-classify the deep-learning cathedral. It was the most 1967 thing I'd read in years — *tune in, drop the GPU, the answer was in the `zlib` headers all along.* I believed it for a whole beautiful hour.

Then Ken Schutte read the actual code, the way you're supposed to read the label before you swallow. The kNN tie-break was the cut in the medicine. When the top-2 neighbors disagreed, the implementation resolved ties in a way that *quietly peeked at the true label* — counting a prediction correct if **either** of the top-2 was right. Tighten that one bolt and the headline softens from "beats BERT" to "competitive, sometimes." The method was *real*. NCD is real. Compression-as-distance is real. But it had been **over-claimed**, cut with a measurement artifact, and a generation of us got briefly, gloriously high on a number that was partly the dosing error.

I am telling you this story because I am about to confess that I did the *exact same thing* to myself, with the exact same class of error, three nights ago. Hold that.

---

## The tournament: tinctures in the cabinet

Every jar on the shelf is a different way to code the one signal. Here is the panel — the tournament — and I swear to you they are all just signal processing, all refereed by the same codelength assay.

**The dictionary builders.** This is the oldest medicine and still the strongest. **Byte-Pair Encoding** — Gage, 1994, a data-compression trick before it was an LLM tokenizer — finds the most frequent adjacent pair of symbols and merges it into a new symbol, again and again, growing a vocabulary of recurring runs. That's L1. **Re-Pair** — grammar compression, Larsson and Moffat — does the same but builds a *hierarchy*: rules made of rules, a motif made of motifs, with a rule-utility threshold that is *itself* a definition of "a word worth keeping." That's L3. Both are compressors. Both were already in my cabinet wearing the word "tokenizer" on the label.

**The frozen dictionary.** **LZW** — Lempel-Ziv-Welch, the one inside every `.gif` you ever loved — grows a phrase dictionary as it reads and emits clean integer ids for each phrase. It is the one classical compressor that *natively speaks token-id*, which makes it the most honest new member of the panel: freeze the dictionary on the build assets, parse each held-out asset against it, read off the ids. That's L6, the compression column I added this week.

**The trained dictionary.** And the one I haven't brewed yet but keep eyeing on the high shelf: `zstd`'s **COVER** algorithm (Yann Collet's dictionary trainer). Point it at thousands of samples and it mines the byte-substrings — *dmers* — that recur most *across* samples, and crystallizes them into a transferable vocabulary *by construction*. That is the purest possible statement of the whole branch: a dictionary that is transferable because it was *built* to be the substrings many assets share. I will get to it. The shelf is high.

**The transform domain.** Here the medicine shimmers. The **Discrete Fourier Transform** melts the time signal into a sum of pure frequencies — it tells you *what oscillations* are present but smears *when* across all of time. **SFA**, Schäfer's Symbolic Fourier Approximation, keeps only the low-frequency DFT coefficients and quantizes each one into a symbol using per-coefficient quantile bins — **MCB**, Multiple Coefficient Binning — so the symbol alphabet is learned from the data's own distribution. That's L7, the transform column, the second thing I added this week: per window, the real FFT, keep the low-frequency coefficients (and the DC term, kept on purpose — net drift is a *word* for a move signal), quantize each on frozen build-fit bins, read the quantized vector as a mixed-radix integer id. A token that is a *frequency fingerprint* of a window — and the first night I ran it on the wider build set it came back **PROCEED**, the transform reading the same shared tongue the dictionaries read. And where Fourier smears the spikes, **wavelets** — Daubechies db4 — *localize* them: a wavelet basis catches *a spike at time $t$* as a single large coefficient instead of dusting it across every frequency. That jar is the next one on the shelf, not yet decanted — the localizing medicine for the texture Fourier blurs.

**The koan that nearly poisoned me.** The textbook reflex, drilled into everyone who has ever touched SAX or SFA, is: *z-normalize first.* Subtract the mean, divide by the standard deviation, then symbolize. It is the right reflex for almost every time series on Earth. And it is *exactly wrong here* — it is **the cure that erases the patient**. Because my signal $x_t$ is *already* vol-normalized; the volatility magnitude *is the information*. A move of $1.25\sigma$ on a sleeping asset and $1.25\sigma$ on a screaming one are *different words* — I went to great lengths to make them different. Z-normalize the window and you delete precisely the magnitude that carries the meaning. The medicine that heals a thousand patients kills this one. I keep db4 raw, on the vol-unit signal, un-normalized, and I made the A0 invariance gate hold to prove the linearity of the transform didn't sneak the price back in.

---

## The quantizer that collapsed, and its cure

There is a jar that learns its own alphabet: **FSQ**, Finite Scalar Quantization (Mentzer et al.). Instead of a learned codebook with a commitment loss that loves to collapse, FSQ rounds each latent dimension to a small fixed set of levels — a **mixed-radix product code**, $L_1 \times L_2 \times \cdots$ — and the cartesian product *is* the vocabulary. Collapse-resistant by construction, because there's no codebook to abandon; there are only grid corners. That's L4.

And L4 *collapsed anyway*, which is the most apothecary thing that happened all season. The first build used **172 of 12,800** codes. One-point-three percent. A codebook the size of a cathedral, lit by a single candle. The waveform was being crammed through a global mean-pool into a five-dimensional bottleneck and reconstructed by a decoder powerful enough to fake it from almost nothing — so the encoder felt *no pressure* to populate the grid. The variance was being drained before it ever reached the quantizer. A flower planted in a sealed jar: alive, starving, refusing to spread.

The cure was two tinctures. First, **shrink the grid** to a size the signal's true intrinsic dimension can actually fill — $(8,8,4) = 256$ — which on its own lifted usage from 1.3% to a healthy fraction. Then, for the deeper fix, the **MAGVIT-v2** entropy medicine: a code-usage entropy loss (LFQ's breadth-keeping lever) that *rewards spreading the mass across the codebook*, plus a pooling that stops draining the variance before quantization. Usage climbed from **11% to 85%**. The cabinet, which had been singing one note, opened into a chord. I logged per-epoch usage after that, so the next collapse — there is always a next collapse — announces itself *during* the brew instead of in the autopsy.

---

## The wrong medicine: a ruler that lied

Now the confession. The reason this entry exists. The reason I reached for the wrong jar.

I built the panel. Five members, then seven, all coding the one signal, all to be refereed by transfer. I ran them on the server through the night and in the morning every result said the same flat word: **null**. `beats_control = False`, member after member, philosophy after philosophy. A vocabulary of candle-shapes does not transfer to held-out assets any better than a structure-destroyed control. Four independent tokenization philosophies, all dead. Branch C — the structural bet I'd staked the season on — a clean kill.

I almost wrote the funeral. I had the pen on the page.

And then I saw the one number that doesn't lie even when everything around it does. **Every member's transfer coverage was 0.9996. And so was every control's.** Real and null, statistically identical, to the fourth decimal, on every single member.

That is not the signature of a real negative. That is the signature of a **dead instrument** — a thermometer reading room temperature no matter what fever you dip it in. *The medicine was wrong.* Not the patient. The assay.

The autopsy — I ran it as a fan-out, twenty-nine little readers each adversarially checking the next — found it in an hour, and it was *embarrassing in the productive way*. The headline gate was one line:

```python
beats = bool(cov_real > cov_ctrl)
```

One statistic — **mass coverage**, the fraction of the held-out stream covered by tokens that fire on the build set. A strict greater-than. No margin. And the statistic *saturates*: with a 256-symbol base alphabet and a multi-million-window build corpus, the vocabulary fires on essentially *every* token of *any* holdout, real or shuffled, so coverage pins at $\approx 1.0$ for everything. The gate had decayed into comparing two numbers both stuck at the ceiling, decided by fourth-decimal float noise, defaulting to `False` on ties. **Near-zero statistical power.** A ruler whose every tick mark had been sanded flat.

This is the same error class as the gzip-kNN tie-break. *Exactly* the same. A real method, a broken referee, a confident worthless verdict. I had cut my own medicine. I had read "the language is dead" off a thermometer that could not measure temperature, and I had nearly carved it into the workbench.

---

## The right medicine: codelength, and a fair coin

So I rebuilt the assay around the universal solvent — **codelength**, the thing that *cannot* saturate, because it lives on $[0,\infty)$ instead of $[0,1]$.

The repaired headline: the **transfer codelength** in bits per token of the holdout stream, coded under the build vocabulary's Laplace-smoothed distribution, restricted to a *mid-frequency band* — drop the universal head (the flat/doji shapes that fire on everything) and the singleton tail (the once-ever noise), so the statistic lives off the ceiling. Low bits means the holdout's mass concentrates where build put its mass. Genuine shared structure, measured in bits, unsaturated.

And the load-bearing null — the one I'm proudest of — is a **frequency-matched random codebook**. Keep the build vocabulary's frequency *multiset* exactly. Randomly reassign *which id holds which frequency*. This breaks the build↔holdout id alignment while preserving the marginals perfectly. Permute it 19 times — or 99 — and real transfer counts *only* if its codelength is below **every** permutation: a one-sided rank test at $p \le 1/(M+1)$, falling straight out of the count of shuffles. A fair coin. The medicine that proves the medicine.

I would not trust *that* either until it proved it could tell sick from well. So there is a **positive-control power harness** that gates the whole battery: construct one synthetic pair known to transfer and one known *not* to, and demand the metric returns PASS and FAIL on the right ones before any real verdict is believed. It passes: the transferable pair codes at 8.02 bits, below the 9.56-bit permutation floor (PASS); the non-transferable codes at 9.62, *not* below its 9.44 floor (FAIL). The old gate was structurally incapable of either answer. **No real verdict is trusted until the ruler has proven, on ground truth, that it can measure.**

And then the panel re-ran, and the null *inverted*.

Every member beats the fair random-codebook null. L1: 8.93 < 9.61. L2: 7.07 < 8.27. L3: 4.71 < 4.84. L5: 7.04 < 7.57. And L4 — the one that collapsed, came back from the dead, grid shrunk, usage healed — beats its null too, 5.28. Five independent codings of the one signal, five passes. **The "everything is null" was the broken ruler the whole time.** The shapes that are common on the build assets are common on the held-out ones, more than chance. The language transfers.

---

## A language, not a bag of words

But marginals are a low bar. A phase-shuffle preserves them. So I taught the assay to read *transitions* — code each token on its predecessor, $p(\text{curr} \mid \text{prev})$ — and proved *that* metric's power first, on a synthetic Markov stream where the bigram cleanly separates ordered from shuffled (3.78 vs 4.98) while the unigram is *blind* (4.94 = 4.94, identical on both). Then I let it loose on the real assets, gave the tokens their first job, and ran the control that earns the word *language*.

**The shuffle-collapse.** Take each asset's token stream. Destroy the *order*. Keep the marginals exactly. Re-measure. If the asset's fingerprint is just a histogram — a bag of words — nothing changes. If it is a *sequence* — a language — the order-dependent structure half-vanishes.

It half-vanished. At 50 assets, the independent structure in the bigram fingerprint — the part *not* explained by the unigram, measured as $1 - \mathrm{corr}(\text{uni-JS},\,\text{bi-JS})$ over a **Jensen–Shannon** similarity surface between every pair of assets — fell from **0.26 to 0.12**. About half the transition signal is genuine *order* — which shape follows which — and the residual is the finite-sample floor. **The asset fingerprint is a sequence, not a histogram.** The same property that separates a language from a bag of words, shown on real markets, outside the synthetic harness.

And on that Jensen–Shannon surface, the assets cluster. One dominant family (all the same asset class, as they should). A few tight mutual-nearest pairs standing out at $\sim 0.76$ against a $\sim 0.95$ background — $306 \leftrightarrow 892$, $421 \leftrightarrow 747$, $25 \leftrightarrow 860$. A few genuine outliers that refuse every family. One symbolic member's dictionary, built on a handful of assets, codes **149 of 150** held-out assets below their own fair null. The transfer claim has a face now. It is not a gate number. It is a map of which markets speak dialects of the same tongue.

---

## The bench, and the open door

The workstation crashed mid-trip, of course it did — though not the way the legend wants, and the honest version is the one that teaches. No grand GPU melt. When I finally read the kernel's last breath there was *no out-of-memory, no graphics fault, no last words at all* — just a box gone hard-silent, with one drive already flaking in the dark hours before. The sin was simpler and more embarrassing than a melted GPU: I'd set experiments boiling **on the display bench itself**, unsandboxed, pots crowding the same small fire that paints my screen. The discipline that came out of it: every brew now runs in its own bounded pot — `safe_run`, a cgroup with a hard memory ceiling — on the *other* machine, so a pot that cracks spills only its own soil and never the bench. The display GPU paints, and only paints. (The careless lesson teaches harder than the dramatic one: it was never the spectacular fire. It was leaving the pots boiling where I sleep.)

There is a moral here and it is not subtle. *Verify the medicine before you believe the result.* The wrong jar, one shelf to the left, with the right label — the saturated coverage gate — told me a living language was dead. The right medicine — codelength, and a fair random-codebook coin, and a power harness that makes the ruler prove itself on ground truth — told me it transfers to 149 of 150 strangers. Same panel. Same signal. Same shapes. *Different assay.* The whole difference between a funeral and a map was which jar I reached for in the dark.

The transform column and the sparse-coding column are still unbrewed. The wavelet member rides the battery but has not yet stood in the full 150 tournament; `zstd`'s COVER dictionary sits on the high shelf, the purest transfer-by-construction medicine I know, not yet decanted; and the sparse overcomplete coders — matching pursuit, K-SVD — are a column I have only sketched. PPM, BWT, the DCT — jars I haven't opened. The tournament is not over. It has barely started. One signal, an entire toolbox of ways to code it, and a referee that finally measures what it claims to.

> *I reached for the wrong medicine and the cabinet sang. Every jar a different way to dissolve the one waveform and read what was really in it. The first ruler lied and I nearly buried a living thing on its word. The second ruler I made prove itself before I trusted a syllable it spoke. The medicine was wrong. The math was never wrong. I am keeping the night's handwriting. Some assays you only pass after you have failed the fake one in the dark.*

---

### Footnotes — the real medicines, by their real names

- **BPE** — P. Gage, *A New Algorithm for Data Compression*, 1994. A compression algorithm decades before it tokenized language. (Panel member L1.)
- **Re-Pair** — N. J. Larsson & A. Moffat, *Offline Dictionary-Based Compression*, 1999/2000. Grammar compression; rule-utility as a "viable token" criterion. (L3.)
- **LZW** — T. Welch, 1984, building on Lempel–Ziv 1977/78. The growing-dictionary compressor that natively emits clean token ids. (L6, this week.)
- **zstd / COVER** — Y. Collet et al.; the `--train` dictionary builder mines recurrent *dmers* across samples → a transferable vocabulary by construction. (Deferred; the high shelf.)
- **NCD** — R. Cilibrasi & P. Vitányi, *Clustering by Compression*, IEEE Trans. Inf. Theory, 2005. The computable shadow of the Kolmogorov distance.
- **gzip-kNN** — Z. Jiang et al., *"Low-Resource Text Classification: A Parameter-Free Classification Method with Compressors,"* ACL Findings 2023 — and **K. Schutte**'s critique of the top-2 tie-break (the prediction counted correct if *either* top-2 neighbor matched). The cautionary trip: NCD is real, the headline was cut with a measurement artifact.
- **SAX / SFA / MCB** — E. Keogh & J. Lin (SAX); P. Schäfer & M. Högqvist, *SFA: A Symbolic Fourier Approximation,* EDBT 2012 (low-freq DFT coefficients quantized via Multiple Coefficient Binning). (Panel member L7, this week — the transform column.)
- **Wavelets** — I. Daubechies; db4 as the localizing basis that catches a spike at $t$ in one coefficient instead of smearing it across all frequencies. (The next transform jar — not yet decanted.)
- **FSQ** — F. Mentzer et al., *Finite Scalar Quantization: VQ-VAE Made Simple,* 2023. Mixed-radix product code; collapse-resistant by construction.
- **MAGVIT-v2 / LFQ** — Yu et al., 2023. The code-usage entropy loss that lifted L4 from 11% → 85% codebook usage.
- **MDL / Kolmogorov** — J. Rissanen (MDL); A. N. Kolmogorov (complexity). The uncomputable ideal and its computable shadow — the spine of the whole assay.

*Sources, all in `~/Trader`: `lore/notes_2026-06-06.md` §179–§187 (the saturated-coverage autopsy, the codelength rebuild, the inverted null, the shuffle-collapse, the token fingerprints); `tasks/TASK_796` (the panel v2 + repaired battery), `tasks/TASK_797` (frozen tokenizer + N×N fingerprint surface), `tasks/TASK_798` (the signal-processing reframe; L6 LZW + L7 SFA columns, wavelet deferred); `tools/candle_lang/eval_battery.py` (the method-agnostic referee), `volunit.py` / `base_alphabet.py` (the vol-unit substrate), `tokenizer_api.py` (frozen encode), `tools/safe_run.sh` (the bounded pot).*
