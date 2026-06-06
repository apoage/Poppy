# TASK_796 — Candle-Language Panel v2: repair the eval battery, fix L4/L3/L5, re-run before any branch-C keep/kill

**Status**: IN PROGRESS. (opened 2026-06-05)
- **v2 9/5 baseline DONE** (`runs/796_token_v2`, committed `4c4393c`/`48790e6`): **v1 null OVERTURNED** —
  all 5 members beat the freq-matched random-codebook null (real cross-asset transfer); **L5 beats every
  control → PROCEED**; **L4 collapse fixed** (grid-256 → 75/256=29% codebook usage, was 1.3%). Symbolic
  members do NOT beat the marginal-preserving substrate controls (shuffle/random_merge) → needs the
  deferred **bigram/transition codelength** to make that axis meaningful.
- **Scale-up LAUNCHED** (committed `c0a3d3c`): **150 build / 150 holdout** stratified split spanning the
  full price range, 99-seed null, **A-by-B transfer matrix**. Running on gpu-server: tmux `796_scaled`
  (CPU L1/L2/L3/L5) + `796_scaled_l4` (GPU 3), `runs/796_scaled/`. Overnight. Awaiting scorecards +
  `holdout_transfer_rate` (fraction of 150 holdout assets that transfer).
**Parent**: TASK_789 (candle-language umbrella) · **Supersedes the v1 verdict** in `runs/789_token_c0/`
**Predecessor**: TASK_795 (panel build) — built + ran v1; v1's "branch-C null" is **not a result**.

---

## TL;DR

The v1 panel reported `A8 beats_control=False` for every member and read as a NULL for branch C.
A 6-way diagnostic workflow (29 agents, 22 findings, **17 CONFIRM / 5 DEMOTE**, adversarially
cross-validated) proved that **null is a measurement artifact, not evidence against the candle
language.** This task repairs the eval harness + the broken members, re-runs on the full corpus,
and only THEN renders a defensible keep/kill.

**Do NOT record branch C as null until a repaired, power-verified metric reports False across members.**

---

## What went wrong (confirmed)

### (a) Measurement artifacts — the A8 headline gate + A1/A2 diagnostics are invalid/insensitive
- **A8 verdict = `bool(cov_real > cov_ctrl)`** (`eval_battery.py:272`): ONE statistic (mass coverage),
  STRICT `>`, NO margin. Coverage saturates to ~1.0 for both vocab AND control in the production
  small-codebook/large-corpus regime (build fires on ~every token → `build_set` = full alphabet).
  `exclude_top_k` cannot de-saturate (saturation is driven by build-set BREADTH, not head mass).
  Jaccard/Spearman are computed but **never gate**. Near-zero statistical power: a positive-control
  analysis shows it can only return True when the build vocab is sparse, which it never is.
- **Only the demoted-circular phase-shuffle control is ever run.** The design-mandated **random-merge
  control #2 is plumbed in every member but NEVER invoked** (all 5 A8 call sites hardcode
  `control='shuffle'`); for L4 the two controls even collapse to the identical phase-shuffle. The
  phase-shuffle preserves per-bar marginals — the very thing a candle vocab keys on — so it is the
  wrong null and makes real≈control by construction.
- **A1 `compression_ratio<1` for all members is an accounting artifact:** null is charged PER-TOKEN
  not per covered BAR (`eval_battery.py:161`), and the A1 PASS gate compares to an internal per-token
  null instead of the spec random-merge / raw-stream baseline. (Keep `boundary_bits` — the "drop it"
  finding was **DEMOTED**; it's spec-mandated and does not flip the ratio.)
- **A2 is ill-posed for signed cumulative-move paths** (magnitude log-ratio explodes at zero-crossings,
  blind to sign flips) and is NOT comparable across members (4 surfaces, per-member offsets, 2 tols).
  L1's 0.071 is the missing-offset artifact, not a BPE serving bug (A0=1.0 byte-identical).

### (b) Implementation bugs — real defects independent of the metrics
- **L4 CRASH** (`train_l4.py:503-504`): `class _Vocab: codebook_size = int(codebook_size)` raises
  `NameError` (class bodies don't close over enclosing locals). A8 never computed for L4.
- **L4 CODEBOOK COLLAPSE** (172/12800 used): time-mean-pool to d=5 bottleneck (`l4_model.py:75/77`) +
  reconstruction-ONLY loss + high-capacity decoder; FSQ has no usage pressure.
- **L3 runtime 134 min**: pure-Python O(rules×corpus) RePair rescan-per-merge (`l3_sax_grammar.py`).
  Grammar is correct/lossless (round-trip verified) → slow, not broken.
- **L5 motif_coverage=0.10 is tautological** (empirical CDF at its own self-quantile → always ≈frac),
  and `_verdict` returns PROCEED without ever consulting A8.

### (c) Genuine null signal: NONE established. The substrate is independently confirmed HEALTHY
(A0=1.0 for the right reason; mv well-conditioned; causal EWMA exact). Failures live in the eval
harness + two members, not the languages.

---

## In scope

### 1. Eval battery (`tools/candle_lang/eval_battery.py`)
- **New headline A8′ = cross-asset held-out transfer COMPRESSION**: bits-per-candle of holdout set B
  under the vocab frozen on build set A, using a **bigram/prequential `p_A`** (rewards shared
  TRANSITIONS, not the marginals a phase-shuffle preserves). Continuous codelength → does NOT
  saturate. Emit the full **A-by-B (build×holdout) matrix**, not a scalar.
- **Mid-frequency band** mask in `_transfer_stats` (keep tokens with build-count in [p10, p90]) so the
  retained Spearman/Jaccard stop pinning at 1.0.
- **Accept a LIST of controls**; require beating ALL: phase-shuffle (kept as one required control, not
  standalone proof), a true cross-asset **random-merge**, a **frequency-matched random codebook**, and
  an **IAAFT surrogate** vocab (lower bound). Margin'd verdict (no bare `>`); reject the null only via
  a **≥19-seed one-sided rank test** (M = 1/α − 1 ⇒ 19 for one-sided p=0.05), beats_all over controls.
- **Positive-control POWER HARNESS** (one known-transferable synthetic + one known-non-transferable);
  **block trusting any verdict until the metric returns PASS/FAIL on these respectively.**
- **A1**: thread `n_bars_covered` into `a1_mdl` (per-BAR null, `eval_battery.py:161` + `train_l4.py:470`);
  re-key the A1 gate to the spec baseline (random-merge same-size vocab + raw per-bar coder). KEEP
  `boundary_bits`.
- **A2**: standardize surface + single `ratio_tol`; for signed vol-unit members use additive per-bar
  `|recon_mv − true_mv|` (incl. sign), well-posed at zero-crossings; label L5's number non-A2.

### 2. Member fixes
- **L4**: crash one-liner (`train_l4.py:503-504` → `cb = int(codebook_size); class _Vocab: codebook_size = cb`)
  + collapse remedy: **grid→256 `levels=[8,6,5]`** (zero-risk first experiment — research: usage jumps
  to ~100% on low-intrinsic-dim data), then MAGVIT-v2/LFQ **entropy loss** + strided/multi-token pool +
  train >8 epochs; per-epoch effective-code/top1/per-dim usage logging; **A9 used_codes-floor gate**;
  distinct `random_merge` transform at `train_l4.py:530`.
- **L3**: incremental pair-count maintenance (port `l1_bpe.py:218-248`) + cap `_MAX_RULES`→512 + p_min
  floor (134 min → minutes).
- **L5**: absolute control-NN-distance recurrence threshold (new control profiling pass) replacing the
  self-quantile; `_verdict` must require A8 beats_control before PROCEED.
- Wire `control='random_merge'` at all 5 A8 call sites.

### 3. Re-run (gpu-server, tmux, single-GPU on idle GPU)
4 symbolic members (CPU, one self-advancing orchestrator) + L4 (single idle GPU, `--gpu-frac 0.7`);
pull scorecards; read the **A-by-B transfer matrix + beats_all + positive-control-harness pass**.

### 4. Downstream non-circular confirmation (only if `beats_all` holds)
Stage-2 trader-on-tokens, real codebook vs random codebook, ≥3 seeds, BEFORE any keep/kill.

## Out of scope
- The vol-unit substrate (CONFIRMED healthy; day-start sigma re-seed micro-caveat is low-priority/optional).

---

## Priority order
1. ✅ L4 crash one-liner (`train_l4.py:559-562`) — done.
2. ✅ Repair A8 (non-saturating transfer-codelength + mid-freq [p10,p90] band + margin'd composite + controls list).
3. ✅ Positive-control power harness (`a8_power_check`) gating all verdicts — **power-proven on CPU** (PASS transferable / FAIL not).
4. ✅ Wire `controls=("shuffle","random_merge")` at all 5 sites + L4 distinct transform (id-permutation).
5. ✅ A1 per-bar null (`n_bars_covered`) + KEEP boundary_bits.
6. ✅ A2 additive signed metric (L1/L2/L3 per-bar mv, abs_tol=0.25); L5 number labeled non-A2 (motif). (No base-offset hack.)
7. ✅ L3 incremental RePair (in-place pair-counts) + `_MAX_RULES`→512 + p_min — full fix, not just the cap.
8. ⏳ L4 collapse remedy: **grid→256 (8,8,4) done** (zero-risk first experiment). DEFERRED if grid-256 insufficient:
   MAGVIT-v2 entropy loss + strided/multi-token pool + masked-token objective (Phase 1).
9. ✅ L5 absolute control-reservoir motif threshold + `_verdict` consults A8.
10. ⏳ **RUNNING** — `796_token_v2` on gpu-server; read transfer_bits + beats_random_codebook + beats_all + power-harness pass.
11. ☐ Only if beats_all: downstream trader-on-tokens real-vs-random, ≥3 seeds.

### Deferred to a follow-up (documented, not blocking the v2 verdict)
- A-by-B per-asset transfer matrix (v2 ships the aggregate B-under-A headline + per-control breakdown).
- IAAFT surrogate as a 4th control (the freq-matched random-codebook permutation null + shuffle + random_merge are in).
- L4 collapse Phase-1 mechanisms (entropy loss / pool / masked objective) — only if grid-256 doesn't clear the A9 usage floor.

## Deliverable
A defensible PROCEED/REVIEW/KILL per member from a metric that has **passed its own positive control**,
plus the A-by-B transfer-compression matrix as the new headline scorecard.

## References
- Diagnosis + rebuild story: `lore/notes_2026-06-06.md` §179–182 · diagnostic workflow `wheo5htsj` (29 agents).
- Research: transfer-compression headline — arxiv 2505.14411 (BPE-for-TS), TOTEM 2402.16412,
  surrogate-data/IAAFT testing; FSQ fixes — Mentzer 2023 (2309.15505), MAGVIT-v2/LFQ entropy loss
  (2310.05737), Discrete-JEPA masked objective (2506.14373).
- v1 scorecards (historical): `runs/789_token_c0/{L1_BPE,l2_abba,L3,L5}_RESULT.md` + `logs/L4.log`.
