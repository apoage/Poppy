# TASK_861 — Search-teacher imprint (D26 ruled: park the search-leaf lane, test the better-teacher × proven-recipe combination)

> **STATUS (2026-07-28): SHIPPED as `task861-search-teacher-imprint`** (factory run 20).
> Both flags OFF bit-identical (orchestrator-reproduced vs `HEAD`: 0 differing leaves on both
> modules; selftest 12/12). 3-member panel over 2 fix rounds, **5 MAJOR found and closed**:
> ungated teacher operator · no payload integrity · empty-string flags failing OPEN (one
> clobbered the tracked parity baseline) · pre-registered legs mutually un-runnable · and the
> replay guard being a TRAJECTORY not a TARGET check (78.8% of BC targets rewritable with
> net@21 bit-identical — closed by a payload-vs-header action-composition check).
> **REMAINDER (planner-owned) is BLOCKED on DECISIONS D28**: the pre-registered real legs would
> grade a CROSS-UNIVERSE comparison (measured: 59/64 positional slots differ from the
> checkpoint's recorded frame). Do NOT launch legs 1–4 as written — see the "MEASURED
> PRE-REGISTRATION DEFECT" block below and D28.
> Scope limits recorded, deliberately not closed: acceptance (iv) certifies the FIXTURE pipeline
> only (the gate never fires on the real panel); the digest+replay pair is accident-detection,
> not tamper-detection; sanity-replay seed noise spans −8.93 … +12.74 across 10 seeds.

**Status**: READY 2026-07-27 (D26 RULED — operator: "prepare dark factory tasks").
**Size: M. Risk: medium-low** (two flag-gated extensions to shipped modules; canon defaults
bit-identical; zero training in the factory half — real legs are planner-scheduled).

## Why (LEDGER 860/codistill-corridor + 832/E5-inversion3 + 832/E5-oos-real)

The program now holds two proven-but-unmarried facts:

1. **The best teacher ever measured is the search**: the true-env-leaf operator at prior-weight
   0.0 scores **+8.024 net@21** on the canonical panel (LEDGER 832/E5-inversion3) — but its leaf
   is an R-step LOOKAHEAD, so it can never trade live, and two distillation rounds trying to give
   it a deployable learned leaf failed structurally (857 best −4.23, 860 best −4.02 vs floor
   +0.424; lane PARKED per D26).
2. **The best deployable recipe ever measured is the BC imprint**: hard per-step weight-vector
   targets via the 832 weight-native recipe produced +0.839 in-sample, **+0.898 OOS (Δ+2.33)**,
   walk-forward consistent both quarters — but its teacher was only the ~floor-grade trend book
   (+0.542).

The one untested strong combination: **use the +8.02 search as an OFFLINE teacher for the 832
imprint recipe.** Run the search operator ONCE over the panel, record the action vector it
CHOOSES at each decision, and BC-imprint the policy on those hard chosen-vector targets — the
exact recipe that produced +0.839. This is explicitly NOT ExIt2's visit-distribution CE
(LEDGER 832/E5-exit2: distribution CE + per-asset argmax rounds to 100% HOLD); the targets here
are the same semantic object 832 already imprints successfully — one weight vector per decision.

## Teacher operator of record (PIN EXACTLY — do not retune)

The +8.024 configuration, verbatim from `runs/832_e5/results_e5inv854_db_pw0_832_e5__ckpt_imprint_full.json`:
`--operator search --vector-proposal-scheme dithered-book --vector-k 16 --sims 32
--vector-hold-bias 3.0 --vector-dither-scale 1.0 --proposal-temp 1.0 --proposal-explore-slots 2
--prior-weight 0.0 --leaf-eval true-env --leaf-rollout-r 3` (no v̂ tail), gumbel_scale 3.0,
deterministic, weight-rebalance env flags as in that artifact, checkpoint
`runs/832_e5/ckpt_imprint_full`. Any deviation is a different experiment and must fail loud.

## Scope (two files, both flag-gated OFF = bit-identical)

1. **Collection hook** — `experiments/835_stage4_eval/eval_vector.py` gains
   `--collect-chosen-vectors <out>`: during a search-operator walk, record per decision step
   (index/timestamp) the CHOSEN action vector — the same semantic object the env executes and the
   832 weight-native BC loss consumes (record the realized book too if cheap, diagnostics only) —
   plus a provenance header: checkpoint path, full operator block (as in the results JSON), db
   path, window, panel-frame provenance, decision count. **Pure recording**: with the flag ON the
   walk numbers must be IDENTICAL to the flag-OFF run (no RNG consumption, no behavior change);
   flag absent = byte-identical module behavior.
2. **Imprint target source** — `experiments/832_linear_imprint/imprint.py` gains
   `--teacher-vectors <file>`: replaces the replay-teacher target stream with the collected
   chosen vectors; EVERYTHING else is the exact shipped recipe (weight-native mapping, same BC
   loss/optimizer/iters/sanity-replay). **Fail-loud provenance gate**: window / panel frame /
   env-flag block in the collection file must match the imprint run's config — refuse on any
   mismatch. Default OFF = shipped 832 bit-identical.
3. **Fixture smoke, end-to-end**: collect on the fixture (tiny window) → imprint (small iters) →
   sanity replay runs. Include a falsification hook (shuffled/permuted teacher vectors must FAIL
   the smoke gate). **860 lesson applies**: the gate must be a high-SNR quantity (train-set
   agreement/fit vs the never-shuffled targets), NOT a small-n held-out metric that inverts
   per-seed at fixture scale.

## Acceptance (gates)

- (i) Both flags OFF = bit-identical (canonical pins / selftests for both modules green).
- (ii) Fixture end-to-end: collection → imprint → sanity replay completes; the collection walk's
  results are IDENTICAL with and without the flag (recording perturbs nothing).
- (iii) Provenance fail-loud demonstrated (mismatched window and mismatched frame both refused).
- (iv) Falsification: imprint on shuffled vectors fails its gate; legit run passes; seed-swept
  (the 860 fail-closed pattern).

**SCOPE LIMIT of (iv) — recorded from verification, do not over-read it.** The gate is
`nll_reduction = 1 − NLL/H(marginal)`, an entropy RATIO whose denominator is the *target*
distribution's entropy, and it is calibrated on the fixture (76.75 % HOLD, H = 0.8632). The real
teacher of record is **94.2 % HOLD (H ≈ 0.30)**, where both arms compress toward zero — so the
0.05 floor is a fixture-local number, not a panel-invariant one. Additionally `smoke_assert`
defaults to `is_fixture`, so **on the real panel the gate is computed and printed but never
enforced**. Acceptance (iv) therefore certifies the fixture pipeline, NOT the real leg.
Verified headroom: the strongest shuffle-invariant degenerate (the per-asset marginal — under a
row permutation `p(a|asset j)` is the exact ceiling of the sabotage arm) measures **+0.0347** on
the shipped collection, i.e. 1.4× under the 0.05 floor (it was 2.9× under the old 0.10). Nothing
measures that ceiling at runtime; the artifact's `modal_action_rate` is the GLOBAL modal rate and
is not the null the shuffle preserves. If this gate is ever re-tuned, measure the per-asset
ceiling on the actual target distribution first.

## Real bars (planner post-ship legs, PRE-REGISTERED here)

> **D30 ADDENDUM (2026-07-30, planner ruled opt (i); operator may override) — fourth and, per a
> requested fresh read of this whole registration, hopefully final framing touch-up.** Stale-tail
> semantics on the minute pin = FORWARD-FILL + disclosure (D30; the zeroed tail fabricated up to
> ~12.5% of PV into leg 3's exact grading scalar — slots 14 `MATIC/USD` and 41 `FTM/USD` both die
> 2025-03-31, 186/1734 rows). Consequences for the legs, all mechanical:
> 1. **Leg 1 gains HARD checks** (the soft "≈+8.02" stands unchanged): zero bankruptcy; the
>    artifact's stale-tail disclosure lists EXACTLY the 2 migration slots; the inert-cell report
>    reads ≈9.6% of the decision grid (22 late-listing heads + the 2 tails — the earlier "≈3%"
>    was slot-level, corrected in D30). Materially different on any of the three = STOP.
> 2. **Every teacher-floor read is masked** (D30 rider (a)): unmasked, `compute_teacher_floor`
>    goes bankrupt on a pinned panel (−15.83/PV 0.00 on the fixture) while `--teacher-floor` is
>    hardwired ON in the read-of-record operator. Applies to leg 3(b)'s on-window book-floor
>    option and leg 4's on-window teacher floor.
> 3. **Leg 4 is additionally gated on D30 rider (c)**: `scorecard.py` must carry the frame
>    SOURCE through the manifest + `OPERATOR_PIN` (or refuse) — as shipped it would reuse the
>    frame but silently read `ohlcv_daily` (33/64), reintroducing between legs 3 and 4 both
>    confounds D29 removed.
> Fresh-read verdict on the rest: the bars, seeds (3-seed median > +0.839, ≥2/3 above), window
> (`--end 2025-10-04`, 1673 decisions), co-runnability exemptions, and floor-referent correction
> below all stand as written. The recurring failure class across all three blocks was "the
> grading basis differs from the bar's basis in an unstated dimension" (universe → source/breadth
> → valuation semantics); the PV-conservation smoke leg and the cell-level inert accounting now
> assert the last unstated dimension this desk can name.
>
> **⚠ SUPERSEDED IN PART BY D29 (2026-07-29): point 1 below is re-amended — the frame pin must go
> through the MINUTE path (TASK_865), NOT the shipped daily-only pin.** TASK_863's flag is sound
> but its only available loader is daily, which covers 33/64 frame slots on the leg-1 window
> (31 forced HOLD) while the +0.839 bar is minute-panel-native with a 64-asset book — pinning
> through daily would substitute a source confound (D20: cross-source forbidden) plus a 64→33
> breadth confound (first-order, ST3 2.76×) for the universe drift. **The legs are now gated on
> TASK_865 (minute-framed loader, 62/64 coverage; the 2 inactive slots = the FTM/MATIC ticker
> migrations, disclosed).** Points 2 (3-imprint-seed battery, median > +0.839, ≥2/3 above) and
> 3 (soft +8.02 check) stand unchanged.
>
> **D28 RESOLUTION ADDENDUM (2026-07-28, planner ruled opt (a); operator may override) — the
> legs below are AMENDED, gated on TASK_863 shipping:**
> 1. **Frame pin, both legs**: leg 1 adds `--panel-frame-from-checkpoint
>    runs/832_e5/frame_provenance_imprint.json`; leg 2 adds the same via TASK_863's new flag.
>    This restores direct comparability to +0.839 / +0.898 / the OOS+WF battery; symbols that
>    lost coverage ride the 853 inactive-slot mechanics (dual-disclosed). Leg 4 reuses
>    `frame_provenance_imprint.json` unchanged; TASK_862's canonical manifest stays valid.
> 2. **Leg 2/3 is a 3-imprint-seed battery** (`--seed 0/1/2`), graded per-seed + median vs
>    +0.839: the banner's recorded sanity-replay seed noise (−8.93…+12.74 across 10 seeds) makes
>    any single-seed read against a +0.839 bar uninterpretable. Standing ≥3-seed discipline
>    applies; the bar is met only if the MEDIAN clears +0.839 with ≥2/3 seeds above it.
> 3. **Leg 1's "≈ +8.02" check stays SOFT and is now doubly off-reference** (the inversion read
>    used a different window AND the unpinned pre-backfill universe): expect "large positive,
>    same behavior class (hold ~94%, narrow book)", not numeric reproduction. A negative or
>    near-zero walk = STOP and diagnose; a ±2-point delta from +8.02 alone does not. The two recipes as originally written were
not co-runnable and leg 1→2 would have hard-refused on first launch:
(a) the teacher of record records `position_size_pct = 0.1` while `jobs_imprint_full.manifest`
uses `0.5/64 = 0.0078125`, and `position_size_pct` is one of the 33 gated EnvConfig fields. The
field is now a **conditional** exemption — exempt iff `vector_weight_rebalance` is true on BOTH
sides (verified inert: the pinned operator walked at 0.1 vs 0.0625 gives 1.000 chosen-vector
agreement and identical net@21 / final_pv; with weight-rebalance OFF the same pair diverges, so
the exemption does not apply there and the run is refused as before).
(b) `--teacher-vectors` + `--imprint-mapping weight-native` used to be an argparse error, but
weight-native IS in the pinned manifest; the combination is now accepted (`imprint_mapping` is
only a label on this path) with the 845 `--vector-weight-rebalance` fail-closed assert kept live.
(c) the window must be **identical** on both legs (the provenance gate compares it), so leg 2
runs `--end 2025-10-04` too — not the manifest's original `2025-09-01`.
Both legs run WITHOUT `--allow-off-pin-teacher`: the collection is recorded under the pinned
operator + `runs/832_e5/ckpt_imprint_full`, and the imprint now REFUSES any off-pin teacher.

1. **Teacher pass** on the real panel, window pinned `--end 2025-10-04` (post-858-backfill this
   yields **1673 decisions, not the canonical 1670** — flagged in LEDGER 860/codistill-corridor;
   floor references nominal). One walk, deterministic; sanity check: walk net@21 reproduces
   ≈ +8.02 (it is the same command + a recorder).

   ```
   python experiments/835_stage4_eval/eval_vector.py \
     --db trading_data.db --start 2021-01-01 --end 2025-10-04 \
     --bar-minutes 1440 --n-assets 64 --position-size-pct 0.1 \
     --operator search --vector-proposal-scheme dithered-book --vector-k 16 --sims 32 \
     --vector-hold-bias 3.0 --vector-dither-scale 1.0 --vector-proposal-temp 1.0 \
     --vector-proposal-explore-slots 2 --prior-weight 0.0 \
     --leaf-eval true-env --leaf-rollout-r 3 --no-leaf-vhat-tail --gumbel-scale 3.0 \
     --vector-weight-rebalance \
     --checkpoint runs/832_e5/ckpt_imprint_full \
     --out-dir runs/861_teacher --tag 861teacher \
     --collect-chosen-vectors runs/861_teacher/chosen_vectors_real.json
   ```
   The recorder preflights the output path before the walk starts and stamps the walk's own
   `walk_net21` / `final_pv` / `bankrupt_at` + a payload sha256 into the collection header.

   **⚠ MEASURED PRE-REGISTRATION DEFECT — UNIVERSE DRIFT (blocks leg 3/4 grading as written).**
   The check demanded below was RUN (read-only, 2026-07-28). **Result: the frame does NOT match.**
   `frame_symbols` in `runs/832_e5/frame_provenance_imprint.json` vs the leg-1-window top-64:
   **59 of 64 positional slots differ**, and membership differs (≥10 symbols each way — e.g.
   frame-only `AKT/USD, ARB/USD, FET/USD, GALA/USD, GRT/USD, ICP/USD`; leg-1-only `/EURGBP,
   /USDCAD, AVAX/USD, DAI/USD, ATOM/EUR`). The frame does not even reproduce from its OWN recorded
   `train_window` (2021-01-01→2025-09-01) any more — the 858 backfill moved the coverage ranking
   under it.
   **What this does NOT break:** the teacher walk. The pinned operator is `prior_weight 0.0` +
   `dithered-book` + `true-env` leaf + no v̂ tail ⇒ `operator_net_dependent(...) == False`
   (`eval_vector.py:253-272`) — the checkpoint is NOT read, so misaligned slots cannot corrupt the
   teacher. Legs 1 and 2 also stay consistent with each other (same db/window/n_assets ⇒ same
   derived universe ⇒ the provenance gate passes legitimately).
   **What it DOES break — the grading:** leg 2 trains on a 64-asset universe that is ~92%
   positionally different from the one `ckpt_imprint_full` was trained on. Therefore
   (a) **leg 3's ">+0.839" bar is a CROSS-UNIVERSE comparison** — +0.839 was measured on the old
   universe; a delta against it confounds "better teacher" with "different assets", which is the
   single question this task exists to answer. (b) The ≈+8.02 teacher reference is likewise from a
   different universe AND window. (c) **Leg 4 cannot reuse `frame_provenance_imprint.json`** — the
   OOS/WF reads must pin a NEW frame provenance recorded from leg 2's own panel, or they will
   grade the new net through the old net's slots. (d) TASK_862's canonical battery manifest pins
   the OLD frame + OLD checkpoint and remains valid for THAT artifact only; benchmarking the 861
   imprint needs its own manifest.
   **Resolution required before GPU spend — see DECISIONS D28.** The cheap option is to pin both
   legs to the frame (needs `imprint.py` taught the 853 flag); the honest-comparison option is to
   re-measure the +0.839 reference on the leg-1 universe.

   _Original hazard note (the check that produced the result above):_
   **⚠ OPEN PRE-REGISTRATION HAZARD — POSITIONAL FRAME (resolve before spending the GPU).**
   Neither leg passes `--panel-frame-from-checkpoint` (default OFF), and `imprint.py` does not
   support it at all (it only WARNS if the collection used one — `imprint.py:620`, "imprint =
   coverage-ranked"). **[CORRECTION 2026-07-29, verified against the tag: the "only WARNs" claim
   is WRONG — shipped `imprint.py:627–631` is a hard `RuntimeError`. The universe-drift finding
   and the rest of this block stand; only the WARN claim was false. See the TASK_863 banner.]**
   So both legs derive their universe from
   `ORDER BY COUNT(*) DESC LIMIT n_assets` over the given window (`vectorbt_env.py:579-590`,
   minute `ohlcv` table). That makes the two legs agree with EACH OTHER — the provenance gate
   compares `loaded_symbols`, so it passes — but it does NOT make either agree with the frame
   `runs/832_e5/ckpt_imprint_full` was trained on. That frame was coverage-ranked over
   **2021-01-01 → 2025-09-01 BEFORE the 858 backfill**; leg 1 ranks over
   **2021-01-01 → 2025-10-04 AFTER it** (+8.28M minute rows across 62/64 symbols). Because
   **slot k IS asset k positionally** and the encoder kernel is 27 feats × n_assets slots, any
   change in either membership or ORDER silently misaligns every slot — this is precisely the
   failure TASK_853 was built to prevent, and the gate cannot see it (both sides are wrong
   identically). The only detector as written is leg 1's soft "≈ +8.02" sanity check, which is
   itself compared against a number measured on a THIRD window (`..2025-12-01`) — so a scrambled
   frame could read as an ordinary miss. **Before launching: verify `frame_symbols` in
   `runs/832_e5/frame_provenance_imprint.json` equals the leg-1-window top-64 ORDER-EXACTLY.**
   If it does not, either pin the window to the frame's own `train_window` end (`2025-09-01`) on
   both legs, or teach `imprint.py` the 853 frame flag so both legs pin explicitly. Do not
   proceed on a membership-only match — the order is load-bearing.

2. **Imprint** per the `jobs_imprint_full.manifest` recipe (fresh init, iters 2000,
   weight-native) with `--teacher-vectors` from leg 1, on the SAME window:

   ```
   python experiments/832_linear_imprint/imprint.py \
     --db trading_data.db --start 2021-01-01 --end 2025-10-04 \
     --bar-minutes 1440 --n-assets 64 --iters 2000 --batch-size 256 \
     --vector-weight-rebalance --imprint-mapping weight-native \
     --teacher-vectors runs/861_teacher/chosen_vectors_real.json \
     --ckpt-out runs/861_imprint/ckpt_imprint_full \
     --tag 861_imprint_full --out runs/861_imprint/results.json
   ```
   `position_size_pct` defaults to `0.5/64` here (the manifest value) and is conditionally
   exempted with a recorded WARN. `reward_attributable` is also exempt (imprint defaults True,
   835 always records False): trajectory-inert, but it defines THIS run's reward stream, which
   `inject_teacher` turns into MC value targets — so the student's value/reward heads are fit
   under a reward definition the search teacher never optimised (the POLICY/BC targets, the
   object of this experiment, are unaffected). Pass `--no-reward-attributable` if the operator
   wants the two definitions aligned; that is a deliberate choice, not the default.

3. **In-sample bar**: sanity replay (policy-argmax) net@21 **> +0.839** (the 832 imprint
   reference), zero bankruptcies. Expectation management: the argmax policy will NOT reproduce
   the search's +8.02 (timing lives in the lookahead); the question is whether a better teacher
   lifts the proven recipe above its current mark.

   **FLOOR REFERENT CORRECTION (fix pass — read this before grading).** On the `--teacher-vectors`
   path the 819-book teacher walk is deliberately SKIPPED (it was 4-7 discarded full-panel walks),
   so `teacher.floor_inenv`, `sanity.corridor_position` and `ledger_row.*_corridor_position_net21`
   in this run's artifact are all measured against the **SEARCH** teacher (≈ +8.02), NOT the
   ~floor-grade 819 book (+0.542) those fields carried on the 832 path. Consequence: the phrase
   "above floor both tiers" is **unreachable by construction here** and has been removed from
   this bar — an argmax policy cannot clear its own lookahead teacher, and reading a ≈ −7
   `corridor_position` as a failure would be a grading error. Grade leg 3 as:
   (a) **primary** — net@21 vs the +0.839 832-imprint reference (same window, same recipe, better
   teacher = the actual question this task asks);
   (b) **corridor** — vs the +0.424 official floor / +0.751 ref from LEDGER 830/E0-REAL, or a
   dedicated on-window book-floor read; do NOT use this artifact's `floor_inenv` for it.
4. **OOS** (only if leg 3 clears): the 858-extended window 2025-10-05 → 2026-03-31, policy-argmax
   + on-window teacher floor, then the walk-forward quarters per
   `runs/858_oos/WALKFORWARD_PREREG.md`. Per D17: benchmark reads only — NO gate status.

## Out of scope

ExIt/self-play of any kind; further search-lane distillation (lane PARKED per D26); retuning the
teacher operator; changes to the OOS/floor mechanics; any training spend in the factory half.
