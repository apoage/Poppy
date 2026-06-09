# TASK_798 — Signal-processing tokenizer panel (compression + transform columns)

**Status**: L6+L7+L8 BUILT & SCORED — all three transfer at 150/150. CPU. Extends the candle-language
panel ([[TASK_796]]) / [[TASK_797]].

## Result (150/150, `runs/798_scorecard/{L6,L7,L8}`)
Both new columns clear the §186 gate — beat the frequency-matched random-codebook null on **unigram AND
bigram**. (Stale JSON verdict strings — L6 "KILL", L7 "PROCEED" — are meaningless; re-derive from
`beats_random_codebook[_bigram]`.)

| Member | family | transfer /150 | transfer_bits | A0 | uni | bigram |
|---|---|---|---|---|---|---|
| **L6 LZW** | compression | 78 (0.52) | 7.72 | 1.0 | ✓ | ✓ |
| **L7 SFA** | transform (Fourier) | 72 (0.48) | 10.17 | 1.0 byte-exact, 1536/1536 codes | ✓ | ✓ |
| **L8 Haar** | transform (wavelet) | 69 (0.46) | 10.22 | **1.0 byte-exact**, 1536/1536 codes (100%) | ✓ | ✓ |

**L8 Haar wavelet** built 2026-06-08, scored 2026-06-09 (`l8_wavelet.py`): pure-numpy multilevel Haar
DWT (pywt not installed/needed; window=64=2⁶ is perfect dyadic), n_levels=4, keep coarse approx + 2
detail bands → MCB bins → mixed-radix pack (codebook 1536). Same scale-free discipline as L7 (NO
per-window z-norm → A0 byte-exact 1.0, 1536/1536 codes used at scale). The scale-native / "fractal"
transform the §189 research + the 1.5D article point at. **Clears the §186 gate** — beats the
frequency-matched random-codebook null on unigram AND bigram; transfer_bits=10.22, holdout rate 0.46
(69/150). db4 deferred (needs pywt).

Lands mid-pack in the full L1–L8 tournament (**L2 149 · L5 135 · L4-w50 89 · L6 78 · L7 72 · L1 70 ·
L8 69 · L3 36**), ahead of SAX-grammar; the transform/wavelet column transfers but doesn't beat the
symbolic (L2) or motif (L5) leaders. **The tournament thesis is demonstrated: cross-asset transfer =
cross-asset compressibility, across symbolic / motif / VQ / dictionary / compression / transform.** Full
scorecard + the L4-entropy-fix verdict → lore §190; L8 + grammar-probe verdict → §192.

## The reframe

Tokenizing the candle language = **structured coding of one 1-D signal** (the vol-unit move series).
Every panel member is a *signal-processing method* for finding recurring structure in that signal, and
the eval battery is **method-agnostic**: it only needs `encode → token-id stream` (+ a decode for A2);
everything downstream (transfer-codelength, bigram axis, controls, fingerprints, §186 `a8_verdict`) is
identical regardless of *how* the tokens are made. So this is a fair **tournament of the DSP toolbox**,
refereed by codelength (= compression = MDL = our transfer metric).

| DSP family | members |
|---|---|
| symbolic approximation | SAX (L3), ABBA (L2) |
| vector quantization | FSQ (L4) |
| motif / template matching | matrix-profile (L5) |
| **dictionary / grammar compression** | BPE (L1), Re-Pair (L3) — **+ L6 LZW** (this task) |
| **transform-domain** | **+ L7 wavelet** (this task) |
| sparse / overcomplete coding | — (future: matching-pursuit / K-SVD) |

**Insight:** L1 (BPE) and L3 (Re-Pair) are *already compression algorithms* used as tokenizers — and the
battery's A1 (MDL) and A8 (transfer-codelength) are *already compression measurements*. The purest
statement of the whole branch: **cross-asset transfer = cross-asset compressibility** (train a
compressor/coder on BUILD assets, measure how well it codes HELD-OUT assets vs controls).

## In scope (this pass)
- **L6 — LZW (compression column).** Substrate = the base-symbol stream (vol-unit → base_alphabet, ids
  0–287). Build a frozen LZW phrase dictionary on the pooled BUILD stream (capped — the "word-size
  limit"); encode = LZW-parse each asset's base-symbol stream against the frozen dictionary → token-id
  sequence (dictionary-entry ids). Lossless on symbols → A2 reconstructs mv from the base symbols (like
  L1/L3, additive metric). Rescale-invariant (vol-unit substrate) → A0 must hold.
- **L7 — wavelet (transform column).** Substrate = the raw mv window signal (the L5 view). Per window:
  DWT (Haar/db4, a few levels) → quantize the coefficients on frozen BUILD-fit bins → a token = the
  quantized coefficient vector → integer id (mixed-radix product code, like FSQ). A2 = inverse-DWT from
  the quantized coefficients → reconstruct mv. DWT is linear + mv is vol-unit → A0 must hold (verify).
- Both: ride the **existing eval battery** end-to-end (A0/A1/A2/A8/A9 + bigram), use the shared
  `eval_battery.render_a8_lines` / `render_transfer_matrix` / `a8_verdict` / `run_a8_power`, the
  `--build-assets/--holdout-assets/--transfer-matrix` CLI, absolute `--out` default; CPU-only.

## Deferred / optional
- **zstd trained dictionary** (`zstd --train`, COVER/fastCover) — the strongest "learned transferable
  vocabulary by construction"; scored by *compression ratio* of holdout-under-build-dict. Needs the
  corpus materialized (a base-symbol stream dump) + a way to read dictionary matches → token ids. Add
  after L6/L7 prove the column.
- **Corpus dump** (`corpus.py`): persist per-asset base-symbol byte streams — only needed for members
  that must see the whole stream at once (zstd-dict). L6/L7 stream like the other members, so not
  required yet (answers TASK_797's "token buildup" question: build it when a member needs it).
- PPM (order-k context model), BWT, matching-pursuit / K-SVD (sparse column), DCT/STFT.

## Run plan
CPU members → gpu-server, **sandboxed** (`tools/safe_run.sh --cpu --mem`, tmux) — never bare/local
([[feedback_sandbox_experiments]]). Score on the 150/150 split through the repaired battery; headline =
transfer-codelength + `beats_random_codebook` (uni+bigram) + `holdout_transfer_rate`, vs the existing
L1–L5 columns.

## Companion
Bloom article (`bloom/drafts/`): signal processing as the lens — "the apothecary tasted the wrong
medicine" (60s / freedom / LSD vibe, hardcore math underneath).

## References
`tools/candle_lang/symbolic/l3_sax_grammar.py` + `l5_matrix_profile.py` (member pattern),
`eval_battery.py` (the method-agnostic contract + shared helpers), `volunit.py` + `base_alphabet.py`
(substrate), `tokenizer_api.py` (frozen-encode lib). Frame: `lore/notes_2026-06-06.md`.
