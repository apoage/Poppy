# candle language

> Branch-C candle-language bet — the LANGUAGE is real (grammar + cross-asset transfer confirmed) but as a TRADER mechanism it is NULL (TASK_804 five-arm, 2026-06-19: no Alpha on perception OR planning). Tokenizer/panel stand as measurement infra.

*Migrated from local agent memory 2026-08-31 (`project_candle_language.md`) so subagents and the server agent
can read it — see `lore/adr/0008`. Content verbatim; only frontmatter and wiki-links changed.*
Branch C (the "candle language" / token middle-layer) is the active structural bet alongside B (Poppy
ExIt revival), after the **loss-side closed null** (A over-trade penalty +0.35pp; A2 realized-PnL value
+0.06pp — both CLEAN nulls, no trade-rate explosion; PnL-in-loss is exhausted). C0's token KILL was
**withdrawn as a strawman** — it aggregated to hourly, used a 5-asset toy, and judged by a next-bar
SIGN probe, none of which the real design uses.

Design (locked 2026-06-02): **minute** granularity, **non-predictive** vocabulary of recurring candle
shapes in **vol-unit relative moves** (fork F1 = vol-unit magnitude → magnitude is part of token
identity), **single-asset**, transferable-by-construction via one shared codebook (no cross-asset
attention in the tokenizer). Pipeline: ① tokenizer → ② masked-token embedding (non-predictive) → ③
existing trader on tokens. Operator's two key ideas: a **PANEL** of isolated languages (L1 BPE-delta /
L2 ABBA / L3 SAX-grammar / L4 learned VQ-FSQ / L5 Matrix-Profile) over one shared substrate, each
seeing different structure; and **one multilingual model** speaking all of them (LLM-style shared
embedding space, cross-language alignment with *free supervision* — the languages are parallel
re-encodings of the same bars, so the shared time axis is the parallel corpus).

Non-predictive eval battery (A0 rescale-invariance / A8 cross-asset transfer = headline / A1 corrected
MDL / A2 reconstruction / A9 length non-degeneracy) replaces the sign-probe; the naive time-shuffle
control is circular and demoted.

Artifacts: **`docs/candle_language/`** (README + 01 design + 02 critique + 03 findings + 04 JAX build
plan + 05 leak/OOM audit + raw JSON). Tasks: **789** (umbrella) / **791–794** (stages) / **795** (JAX
build). Bloom article draft was `bloom/drafts/48_language_of_markets.md` — **path dead: `bloom/` pruned 2026-07-07** (`lore/prune_2026-07-07.md`, backup exists).

**BUILD WRITTEN + CPU-VERIFIED 2026-06-03** — `tools/candle_lang/` (15 modules, ~3.5k LOC). Foundation
(config/fsq_core/volunit) authored by hand and proven on CPU: FSQ even-level bug fixed (reference
formulation + reachability gate), vol-unit A0 rescale-invariance byte-identical (max|Δ|=0). Other 13
modules built by verify-gated workflow `wl3uffw6j` (22 agents). All **6 blocking corrections SATISFIED**
(adversarial qa-hunt × crossval, file:line evidence); CPU smoke PASS incl. **A0=1.0 end-to-end through
BPE**, db_stream streaming the real 25GB DB. Smoke-scale gates A1/A2/A8 fail (expected at 3-asset cap) →
the full-corpus server run is the real mechanism test; **A8 cross-asset transfer beating BOTH controls is
the headline number to watch**. DB has 1193 assets (not 64); default corpus spans ~$0.007–$30k.

**RAN on gpu-server 2026-06-03** (tmux `789_panel`, full corpus, cap 20k). First launch fixed 3 bugs:
DB path → repo-root-relative; quick_check → cheap sanity; the **I/O bottleneck** (members re-streamed the
~0.5GB corpus ~9× each → load-once per-process window cache in `db_stream.py` + tail-seek, gated on
max_windows; members 5%→100% CPU). v1 panel COMPLETED: L1 REVIEW, L2 INCONCLUSIVE, L3 KILL (8022s/2h14m),
L5 PROCEED; L4 CRASHED before A8 (172/12800 codes used). All members `A8 beats_control=False`.

**v1 "branch-C null" is a MEASUREMENT ARTIFACT, not a result (diagnosed 2026-06-05, workflow `wheo5htsj`,
29 agents, 17 CONFIRM / 5 DEMOTE).** The A8 gate `bool(cov_real > cov_ctrl)` (`eval_battery.py:272`) gated
on mass coverage, which SATURATES to ~1.0 for both vocab AND control in the small-codebook/large-corpus
regime → strict zero-margin `>` decayed to a float-noise coin flip defaulting to False; the mandated
random-merge control was plumbed but NEVER invoked (all 5 sites hardcoded `control='shuffle'`, the
demoted-circular control); A1 charged null per-TOKEN not per-BAR (compression<1 forced); A2 magnitude-ratio
ill-posed for signed mv paths (L1's 0.071 = missing-offset artifact). Substrate confirmed HEALTHY. **Do NOT
record branch C as null until a repaired, power-verified metric reports False.** Plus real bugs: L4 crash
(`train_l4.py:503-504` class-body scope) + collapse (mean-pool d=5 + recon-only loss), L3 O(rules×corpus)
runtime, L5 tautological motif metric + verdict never reads A8.

**v2 repair underway 2026-06-05 (`tasks/` TASK_796).** New headline A8′ = non-saturating cross-asset transfer
CODELENGTH (bits/in-band-token of holdout under build's Laplace unigram p, mid-freq [p10,p90] band) +
margin'd multi-control (freq-matched random-codebook permutation rank test, 19-seed; phase-shuffle;
random-merge) + a POSITIVE-CONTROL POWER HARNESS (`a8_power_check`) gating all verdicts. eval_battery.py
repaired + **power-proven on CPU** (PASS transferable, FAIL non-transferable). Member adaptation (multi-
control A8, per-bar A1, additive A2) + L4 crash/grid-256/collapse + L3 perf + L5 motif in flight; then
re-run on gpu-server as `796_token_v2` (committed `4c4393c`, tmux `796_panel`). Diagnosis+rebuild story:
`lore/notes_2026-06-06.md` §179–182.

**v2 9/5 BASELINE RESULT (2026-06-06): v1 null OVERTURNED.** All 5 members beat the freq-matched
random-codebook null (real cross-asset transfer of shape-frequencies); **L5 PROCEED (beats every
control)**; **L4 collapse fixed** (grid 12800→256 = 75/256/29% used, was 1.3%). Symbolic members do NOT
beat the marginal-preserving substrate controls (shuffle/random_merge) — because the unigram codelength
only sees marginals, which phase-shuffle preserves. **Fix shipped:** BIGRAM transition codelength
(`89f698d`) codes `p(curr|prev)` — the structure shuffle destroys — power-proven (distinguishes
ordered vs shuffled where unigram is blind). **L5 GPU-accelerated** (matrix-profile matmul → JAX,
parity 9.5e-7). **150/150 scaled run (bigram + GPU-L5) RUNNING** on gpu-server (`runs/796_bigram/`,
tmux `796b_cpu`/`796b_l5`@GPU1/`796b_l4`@GPU3). Watch `beats_substrate_bigram` + `holdout_transfer_rate`.
See `tasks/` TASK_795, `tasks/` TASK_796,
[single_mechanism_exhausted](single_mechanism_exhausted.md), `feedback_server_vram_envelope`.

**TOURNAMENT COMPLETE + GRAMMAR EXISTS (2026-06-09).** Full L1–L8 DSP tournament on the 150/150 split,
every member passing the §186 gate (beat freq-matched random-codebook null on unigram AND bigram):
**L2 ABBA 149 · L5 matrix-profile 135 · L4-w50 FSQ 89 · L6 LZW 78 · L7 SFA-Fourier 72 · L1 BPE 70 ·
L8 Haar-wavelet 69 · L3 SAX-grammar 36**. Thesis demonstrated: cross-asset transfer = cross-asset
compressibility across symbolic/motif/VQ/dictionary/compression/transform families; symbolic (L2) +
motif (L5) are the strongest coders, the two transform columns (Fourier/wavelet) the weakest passers.
§186 verdict rule = re-derive from JSON `beats_random_codebook[_bigram]`, NOT the stale verdict string.
**GRAMMAR PROBE PASS (`tasks/` TASK_799, `token_lm.py`): the candle language has structure beyond order-1.**
First model trained *on* the tokens (L1-BPE, small causal transformer, held-out cross-asset): LM 7.61
bits/tok < bigram 8.19 < unigram 8.36; LM-on-shuffled-holdout 8.99 ≈ unigram → the +0.58 LM-vs-bigram
gain is genuinely order-dependent. **Green light for the tokenised trader** (`tasks/` TASK_796 item-11,
real-vs-random gated, ≥3 seeds). Next: grammar probe on L2/L5 champions + width/context sweep. Memory
discipline (printed VRAM envelope, MEM_FRACTION=0.7, PREALLOCATE untouched, single GPU) held end-to-end.
Lore: `notes_2026-06-06.md` §190 (tournament) + §192 (L8 + grammar verdicts).

**TRADER VERDICT — branch C CLOSED NULL on Alpha (TASK_804 five-arm, 2026-06-19).** The language is real
(grammar + cross-asset transfer above) but it does NOT move deployment Alpha. Five arms × 3 seeds
(7/13/23), L1-BPE frozen vocab, cyclic-shift null control, eval-MCTS sims=16. Mean avg_alpha: raw 0.0640
· real_fe 0.0628 (**Δα −0.12pp**) · shift_fe 0.0692 · real_dyn 0.0651 (**Δα +0.12pp**). PROMOTE bar was
arm4−arm1 ≥ +3pp AND arm4>arm5 — real_dyn's +0.12pp misses by an order of magnitude → **NO PROMOTE**.
Null on BOTH surfaces: perception (real_fe ≈ raw; shift_fe ≥ real_fe 3/3) AND planning (real_dyn ≈ raw;
real_dyn vs shift_dyn mixed — s7 shift-wins, s13 real-wins +0.75pp, s23 control unrecoverable). One faint
flutter (real_dyn edges real_fe +0.24pp; s13 real>shift) sits inside the ≤5pp noise band — null-with-a-
flutter, NOT weak-positive; do NOT chase. Confirms [single_mechanism_exhausted](single_mechanism_exhausted.md) (null since 738).
14/15 cells: **shift_dyn_seed23 unrecoverable after 8 [seed7_stageb_wedge](seed7_stageb_wedge.md) failures** (3 two-wide
+ 5 solo) — the missing cell can't change the locked no-promote. What STANDS: the tokenizer + grammar/lag
cert (TASK_799/800), the **cyclic-shift null** (reusable content-preserving control), and the TASK_802
NextTokenHead (validated-but-null architectural option). TASK_802 dynamics code committed `09d134d`,
campaign runner `experiments/804/dyn_solo.py`, verdict commit `1ec61ad`. Branch C is done as a trader
mechanism; the panel work stands as measurement infrastructure. Lore: `notes_2026-06-19.md` §211–212.
