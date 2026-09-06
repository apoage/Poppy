# TASK_919 phase A — the rebalance grid (H × θ × accounting)

**Spec of record**: `tasks/TASK_919_EXECUTION_TURNOVER_MODULE.md`. **Implementation brief**
(factory run 55, orchestrator) RESOLVES the under-specified parts and CORRECTS four of the task
file's own claims. Every correction is independently re-derived by this pass, not transcribed.

> **THIS IS A CYCLE-3 ARTIFACT PLUS A D76 CLOSING PASS.** A four-lens panel FAILED cycle 1 (two
> BLOCKERs, three MAJORs); cycle 2's fixes were themselves failed by a cycle-3 panel; cycle 3's
> own headline was then failed by the cycle-3 final panel. **The closing pass DELIBERATELY
> EXCEEDS this run's 3-cycle limit under the single named exception in
> `.claude/rules/30-agent-conduct.md` (D76), and this sentence is the disclosure that exception
> requires — not a footnote.** The exception's conditions and how this pass meets them:
> (a) it replaces framing THIS RUN proved defective — chiefly a leverage verdict stated as
> *closed* which is in fact *construction-dependent and unresolved*; (b) it only TRANSCRIBES
> evidence judges already collected and confirmed — it made **no new solution attempt**, built
> **no new arm or computed column**, and gathered **no new evidence of its own**; (c) three
> items it could not fix without new computation are named in **"What the closing pass left
> undone"** rather than quietly dropped. Every *number* below is unchanged; the *claims wrapped
> around them* are what moved.
>
> Three consequences a reader must carry into every number on this page:
>
> 1. **The pre-registered ±0.20 band question is REFUSED FOR WANT OF POWER.** This design cannot
>    resolve ±0.20 — the MDE is 1.6×–13× the band, and 0 of 48 cells clear the band at the
>    family's own pre-registered α on any of three windows. That conclusion is **prior to and
>    independent of** everything else in this pass. See **"The band question: REFUSED FOR WANT
>    OF POWER"**.
> 2. **Every table on this page is generated from `result_919a.json`** by `render_tables.py` and
>    lives inside `<!-- AUTOGEN:… -->` markers. Cycle 1 hand-wrote its headline table and 26 of
>    48 rows disagreed with the artifact of record by up to +102%. No table here is hand-written,
>    and `test_919_grid.py::test_results_tables_match_json` fails if one drifts again. Since
>    cycle 3 a second gate also reaches the **prose** — but it is a *partial* gate, not the
>    complete accounting cycle 3 claimed for it; its measured reach is stated in **"What the
>    prose gate actually catches"** and the reader must not treat an unflagged prose number as
>    checked.
> 3. **The hold-shares mode's full-window Sharpe advantage is UNRESOLVED, and its leverage
>    component is CONSTRUCTION-DEPENDENT.** De-levering removes ~85% of the full-window
>    hold-shares−vs−target-weight edge and leaves a residue an order of magnitude inside this
>    design's SE. Cycle 3 wrote that up as "the entire advantage is a leverage artifact, closed
>    against the mode". It is not closed: the shipped construction (`ret/G`, a rescaling of the
>    *realised return series*) leaves −0.035…+0.032 with no consistent sign, while a genuinely
>    `gross_cap=1.0`-constrained book — built and validated by a cycle-3 lens — leaves
>    +0.0039…+0.0571, positive in all 16 cells. **Neither clears zero, and neither does the raw
>    advantage.** That is the finding. The grid's "best cell" and its one raw-95% band-clearing
>    cell remain withdrawn (they do not clear on *any* construction), but they are withdrawn for
>    want of power, not refuted as leverage. CORRECTION 2's **BEAR22 direction survives**. See
>    **"De-levering the hold-shares book: what it settles and what it does not"**.

Harness: `grid_harness.py` (the 48-cell grid + all scoring), `hour_of_day.py` (section 7
mechanism), `render_tables.py` (every table below). Tests: `test_919_grid.py`. Raw result:
`result_919a.json`.

Read-only against `fixtures/trading_fixture.db`, CPU-only. Three windows × 48 cells, with a
2000-draw block bootstrap per cell **and** a 2000-draw bootstrap on each cell's uplift:
**~49 s wall clock**.

---

## What cycle 1 got wrong (and what this pass did about it)

| finding | what was wrong | fix |
|---|---|---|
| **BLOCKER-C1** | `hold-shares` was not a share-carrying path at all — all three accounting modes scored the **same** target-weight position array, and the mode's only difference was log-vs-simple compounding (TASK_920's separate, already-shipped leg 2). `true_bh_series` was never called; no share count was ever carried. So "accounting" was a post-hoc scoring lens, not a treatment arm, and the effective grid was **16** position paths, not 48. | `grid_harness.hold_shares_walk` — a real share-carrying book (shares struck at the decision price, carried across decisions, only the signal change traded, gross re-derived from the equity path), modelled on `trader/backtest.py:525 true_bh_series` and reusing its `_book_value` marking convention, its "liquidate at last print" delisting convention and its fail-loud refusals. **Correction 2's CONFIRMED status was retracted and re-derived from scratch**, which changed the best cell (H=3→**H=2**, θ=0.25, hold-shares) and split BEAR22's **−0.602** gap into **−0.569** compounding convention and **−0.033** share carry. *Cycle 3 amendment:* those components now carry paired CIs and three of the four are **UNRESOLVED**; the BEAR22 *direction* survives, the *split* does not (MAJOR-E2B). |
| **BLOCKER-F1** | `RESULTS.md`'s 48-cell table disagreed with `result_919a.json` on 26 of 48 rows (up to **+102%** on `gross`), independent of C1 — it corrupted `target-weight` rows too. The JSON was independently confirmed correct; the hand-written table was not. | `render_tables.py` emits every table from the JSON into `AUTOGEN` markers. `test_results_tables_match_json` re-renders and fails on any diff. No hand-written table remains. |
| **MAJOR-E1** | The design's power was never computed, and an `exceeds_band: YES` table was readable before any caveat. | MDE re-derived two ways plus a **direct** bootstrap CI on the uplift itself; the band verdict is now stated as **REFUSED FOR WANT OF POWER**, and the power table sits *above* the band table, not below it. |
| **MAJOR-E2** | SECONDARY was implied to be an out-of-sample check. It is a strict **superset** of PRIMARY. | The nesting is stated wherever the windows are compared, and a genuinely **DISJOINT** window (2018-01-01..2021-01-01 = SECONDARY minus PRIMARY, n=1096) was added as the only actually-out-of-sample read. |
| **MAJOR-F2** | The three accounting modes charged on three **non-comparable cost bases**, and nothing said so; the band's cost-only reading leaned on cross-mode Δcost_drag. | Each mode's basis is named in every grid row and in `COST_BASIS`; all four bases are measured on every cell; **cross-mode Δcost_drag is forbidden** and the cost-only reading is computed WITHIN each mode. |
| **MINOR-C2** | G1 "bit-identical" asserted on 3 scalar reductions. | Asserted **element-wise** on the full 1704-length gross/turn/net series. |
| **MINOR-C3** | A code comment cited a borrow-indexing equivalence test that did not exist. | `test_borrow_indexing_equivalence` now exists and the comment points at it. |
| **MINOR-R1** | `test_g2_no_trader_edits` imported `trader` and returned a string — structurally unable to reach its claim. | It now asserts `git diff HEAD -- trader/`, `git diff --cached -- trader/` and `git status --porcelain trader/` are all empty, and pins `git log -1 --format=%H -- trader/`. |
| **MINOR-E3/O2** | The analytic residual was compared against TOTAL turnover; the model is a pure sign-flip model. | Both comparands reported; the flip-only comparand is the load-bearing one and the candidate-mechanism discussion is corrected. |
| **MINOR-E4** | The θ=0.5-vs-"40–50%" contrast was missing. | Stated as **CORRECTION 4** with the full θ × H table. |

Preserved unchanged because the panel verified them clean: the θ hysteresis carry/reset/seeding,
leak-freedom, the exact decomposition closure, reproducibility and seeding, the window
discipline, the four degrade paths, the hour-of-day mechanism proof, and G1's identity (whose
numbers are **byte-for-byte the same** after the fix — see G1 below).

## What cycle 2 got wrong (and what cycle 3 did about it)

| finding | what was wrong | fix |
|---|---|---|
| **BLOCKER-E2A** | Cycle 2 disclosed the hold-shares book's leverage drift, then declared the leverage-vs-carry separation out of scope — "this pass did not separate the two … isolating those two needs a gross-renormalised hold-shares variant, which is a **new arm**, not a fix". Isolating the leverage *component* is not a new arm. | A per-decision leverage-neutral rescaling of the realised return series, ~10 lines reusing the `held_w` `hold_shares_walk` already returns; the rescaling of the series is exact because per-decision return is linear in the weights. Reported per cell, with the BEAR22 counter-check that bounds it. **Cycle 3 then over-claimed on it** — see the closing-pass row below. |
| **MAJOR-C2A** | `--check` diffed only the bytes *inside* the AUTOGEN markers, so prose passed through identically in both sides of the comparison and could never trigger a diff. A judge planted `0.078` → `9.9999 WRONG NUMBER PLANTED` in prose and `--check` reported success, exit 0. Dozens of hand-typed literals sat outside any block, many of them data-derived. | `render_tables.check_prose_numbers` — a prose literal must round-match a value in `result_919a.json` at its printed precision or sit in an enumerated whitelist with a written reason. `test_prose_number_gate` plants the judge's exact poison in **prose** and proves `--check` fails on it (and that the old AUTOGEN-only comparison does not). **Cycle 3 then over-stated its reach** — see the closing-pass row below. |
| **MAJOR-E2B** | Correction 2's attribution split (−0.5691 / −0.0328 on BEAR22, −0.0049 / +0.2842 on the full window) carried **no CI or SE anywhere**, in a pass that refused the ±0.20 band question for want of power. The −0.0328 component is **11.4%** of the paired bootstrap SD (0.2886) the same cycle derives for it. | Every component now carries a **paired** block-bootstrap 95% CI, and three of the four are marked **UNRESOLVED**. The BEAR22 *direction* is kept — it is supported independently — but the *split* no longer reads as confirmed. |
| **MINOR-E2C** | `expected_false_positives_at_this_alpha` used `0.05·n`, conflating a two-sided-CI-lower-bound test (null rate α/2) with a one-sided α=0.05 test — 2× too high. | `α/2·n` (2.4 → **≤1.2**), with the derivation stated in the artifact. It is an **upper bound**, not the expectation — see the closing-pass row below. |
| **MINOR-E2D** | `uplift_ci`'s paired branch keyed pairing on post-filter sizes alone, so two series dropping non-finite entries at different positions but in equal counts would have been silently mis-paired. Latent, not live. | The positional-alignment invariant the "paired, assumption-free" claim rests on is now **asserted**: when candidate and frozen share H *and* raw length, a difference in their finite-masks **raises** instead of mis-aligning. Same-H series of different raw length fall through to the *unpaired* branch with no raise — conservative, and correct, but it is a narrower guard than "raises rather than mis-aligning" implies. `test_uplift_ci_pairing_alignment` plants the misalignment. |
| **MINOR/HUNCH-C2B** | Held-vs-target divergence was explained purely as **price** drift; a same-signed asset's *target* also drifts when other assets enter/leave the panel. | Both sources named in `hold_shares_walk`'s docstring and in "Deviations". The retrim rule is **unchanged** — disclosure only. |

## What cycle 3 got wrong (and what the D76 closing pass did about it)

Every row here is a **framing** correction. No number in this artifact changed, no arm was built,
no column was computed. Where a fix would have required new computation, the row says so and the
gap is carried into **"What the closing pass left undone"**.

| finding | what was wrong | what the closing pass did |
|---|---|---|
| **The leverage headline was CONSTRUCTION-DEPENDENT and stated as closed** | Cycle 3 wrote that `ret/G` **is** "the same book run at gross 1", that "the entire advantage is a leverage artifact", and that the measurement "does not leave the question open — it closes it against the mode". `ret/G` rescales a decision's *realised return*; it never produces a book at gross 1 and never changes the weights carried into the next decision. A book actually constrained to `gross_cap=1.0` renormalises its **holdings** each decision, after which the retrim rule mixes an unscaled target with a scaled drift — proportionality breaks and the k+1 position path diverges. A cycle-3 lens built that book (validated Σ\|w\|=1±4e-16) and got the **opposite sign pattern**. | Demoted to **UNRESOLVED** everywhere, with both constructions and all three paired CIs stated side by side. "No consistent sign" is demoted from a finding to a property of the shipped proxy. The verdict now reads as the artifact's own cycle-3 text already did in "What could not be completed": de-levering removes ~85% and leaves a residue an order of magnitude inside this design's SE. **The constrained book was NOT rebuilt here** — its figures are transcribed. |
| **The candidate retraction named a false comparator and the wrong mechanism** | It said the cell "was clearing the band against the same-cell target-weight rule". It was not: `attach_uplift_cis` fixes the comparator to the **frozen** cell (H=1, θ=0, target-weight). And it called the clearance "a leverage differential, not an execution or accounting effect" — false on the point estimate, since **half** of the +0.6523 uplift is the θ hysteresis effect, which is available under canon target-weight accounting and survives de-levering entirely. | Retraction of candidate status **kept** (the de-levered cell no longer clears). Mechanism rewritten as leverage-**assisted**, not leverage-constituted, against the actual comparator, with the θ half given its due as the row's most interesting surviving result. |
| **The de-levering conclusion was never stated in the pass's own gate quantity (D81(c))** | `renormalised_to_gross1` emits gross + net8/12/21 and **no borrow**, while every band / uplift / significance / candidate statement here is denominated in `net21_borrow_low`. | Disclosed plainly, and the flow-verifier's already-computed re-denominated arm transcribed: the conclusion **holds** in the gate quantity, it was simply never demonstrated there. **The borrow column was NOT added** — that is new computation. |
| **A false universal in the cost caveat, contradicted by this artifact's own table 80 lines above it** | "mean Σ\|w\| < 1 at **every** cell here, so the net columns **flatter** the renormalised arm … survives its own most favourable treatment". False at 3 of 16 cells — mean Σ\|w\| = 1.0142, 1.0104, 1.0316 — which this page prints twice and states in words at the exposure table. Per-cell, the approximation **penalises** in 4 cells (those three plus H=1, θ=1.0 at mean 0.9931, because Sharpe depends on G's whole distribution, not its mean). | Corrected at all four sites to "at 12 of 16 cells", with the four exceptions named. Magnitude ≤0.002 Sharpe, so nothing flips. Note the prose gate **cannot** catch this class: it matches literals, and `1` matches trivially. |
| **The prose gate's reach was over-stated** | Cycle 3 wrote that "every prose numeric literal is accounted for" and "must match a value in the artifact". Measured, it rejects **50.9%** of last-digit typos, and **1%** at two decimals. | Replaced with the measured reach, the demonstrated value-keyed holes, and the previously undisclosed `_SCRUBS` class exemption — see **"What the prose gate actually catches"**. **The gate was NOT strengthened**; that is a redesign. |
| **`α/2·n` was presented as the expectation** | It is an **upper bound**: it assumes all 48 cells sit exactly at the boundary null, when most sit far below and the cells share position paths. The true chance baseline is *below* 1.2, which makes "1 observed ≈ chance" easier to invoke than warranted — the opposite of the conservative reading claimed. | Relabelled **≤1.2, an upper bound**, in the artifact and here. Immaterial to the disposition: Bonferroni 0/48 and DISJOINT 0/48 carry the refusal. |
| **Several mechanical mis-statements** | The banner said cycle 2; "three comparability warnings" preceded four; `grid_harness`'s module docstring said **two** windows while the artifact ships **three** (omitting DISJOINT, the only out-of-sample read); `score_cell`'s docstring documented a returned key that has never existed; "the search's maximum lands on the cell that drifts to the most leverage" was unsupported; the exposure range `0.35–2.10` was labelled as if it covered all windows; and a share-carry component was measured against an SE this pass does not derive at that n. | All corrected in place. |

## Data provenance (measured, re-verified by this pass)

- `fixtures/trading_fixture.db` sha256 = `b0f8de8b108b50521f301df1d0856aed72bc760c57b67a842fa62fa
  918206cf9` (mtime 2026-08-05 07:12) — **matches** the recorded baseline; this run did not
  regenerate the fixture (`grid_harness.print_fixture_sha()` reprints it every run).
- `ohlcv_daily` (the table `trader.data.load_db_backtest_panel` reads, hence this whole grid) is
  **byte-identical** between the fixture and the 24 GB `trading_data.db` source of record
  (499,430 rows, 411 assets, 2013-10-06..2026-07-06, sha256 `90d67f40…` both sides — reproduces
  TASK_917's own changelog hash). **The grid below scores production numbers, not a fixture
  proxy.**
- `ohlcv` (the 1-**minute** table) is **not** at parity: fixture holds 11 assets / 2025-09-09..
  2025-12-12 (474,343 rows) vs `trading_data.db`'s 1,130 assets / 2013-09-10..2026-03-31
  (254,904,234 rows). This is why section 7 implements the hour-of-day mechanism but does not run
  it at panel scale.

## Windows — and why only ONE of them is out-of-sample (MAJOR-E2)

<!-- AUTOGEN:windows BEGIN -->
| window | start | end | n (H=1) | relation to PRIMARY |
|---|---|---|---|---|
| **PRIMARY** | 2021-01-01 | 2025-09-01 | 1704 | — (the grid's own denominator; the cell was SELECTED here) |
| **SECONDARY** | 2018-01-01 | 2025-09-01 | 2800 | **strict SUPERSET of PRIMARY** — NOT out-of-sample |
| **DISJOINT** | 2018-01-01 | 2021-01-01 | 1096 | **disjoint from PRIMARY** — the only genuinely held-out read |
<!-- AUTOGEN:windows END -->

**SECONDARY CONTAINS PRIMARY. It is not independent evidence and must not be read as a
validation.** The grid's best cell is SELECTED on PRIMARY; scoring it on a window that *contains*
PRIMARY is dilution-by-extra-years, and the shrinkage that produces is exactly what pure
overfitting predicts whether or not the effect is real. Cycle 1's write-up implied otherwise, and
so did the orchestrator's own cycle-1 reasoning (it had treated the two windows landing on
opposite sides of the band as strong evidence); that argument is **withdrawn** — the real
argument is the power analysis below.

**This pass performed no permutation test, no max-of-correlated-normals correction, and no
pre-registered holdout.** DISJOINT (2018-01-01..2021-01-01, n=1096 at H=1) is the only genuinely
out-of-sample read available and it was added by this fix pass; it is a *post hoc* holdout, not a
pre-registered one, and it is reported as such.

## The band question: REFUSED FOR WANT OF POWER (MAJOR-E1)

**Read this before any number in the grid below.** The task pre-registers a ±0.20 net@21 band on
"max uplift over the frozen rule". Three independent instruments say this design cannot resolve
it.

### Instrument 1 — the MDE

<!-- AUTOGEN:mde BEGIN -->
**SE of one cell's annualized `net21_borrow_low` Sharpe — two independent derivations:**

| derivation | median | min | max |
|---|---|---|---|
| (A) empirical, from each cell's own 95% block-bootstrap CI half-width / 1.96 | 0.4609 | 0.4472 | 0.5157 |
| (B) closed form, √(365/H)·√((1+SR_period²/2)/n) | 0.4632 | 0.4628 | 0.4648 |

The two agree to 0.0023; SE used = **0.4631**, and it is essentially INVARIANT ACROSS H (n·H ≈ 1704 is held constant by the grid, which exactly cancels the √(365/H) annualizer).

**MDE at 80% power, one-sided, vs the pre-registered ±0.20 band:**

| SE_diff assumption | SE_diff | MDE @ raw α=0.05 | × band | MDE @ Bonferroni α=0.05/48 | × band |
|---|---|---|---|---|---|
| frozen treated as a NOISELESS reference (SE_diff = SE) | 0.4631 | **1.152** | 5.8× | **1.815** | 9.1× |
| rho=0 (series independent; SE_diff = sqrt(2)*SE) | 0.6550 | **1.629** | 8.1× | **2.567** | 12.8× |
| rho=0.7 (SE_diff = sqrt(2(1-rho))*SE) | 0.3587 | **0.892** | 4.5× | **1.406** | 7.0× |
| rho=0.9 (SE_diff = sqrt(2(1-rho))*SE) | 0.2071 | **0.515** | 2.6× | **0.812** | 4.1× |

**ρ is MEASURED, not assumed**, on the 11 H=1 candidate cells (same 1704 decision rows as the frozen cell, so the two net series can be paired directly): ρ min 0.5499, median 0.9102, max 1.0000. The most favourable NON-DEGENERATE pair (ρ<0.999 — at ρ≈1 the two cells are the same position path differing by a cost constant, so there is nothing to detect) is `H1_th0.25_drift-inclusive` at ρ=0.9837: SE_diff 0.0837 → MDE **0.208** raw (1.0× band) / **0.328** Bonferroni (1.6× band). Cross-H cells cannot be paired and are less correlated in every measurable case here, so this is a **LOWER BOUND** on the MDE of the cells the grid actually nominates.
<!-- AUTOGEN:mde END -->

Two derivations, agreeing to 0.002 on a quantity of 0.46, both re-derived from this pass's own
arrays. **Every** row of the MDE table exceeds the ±0.20 band at the family's own pre-registered
Bonferroni α. So does the best case obtainable from a *measured* correlation: at ρ=0.9837 —
a candidate whose per-decision series is 98.4% correlated with the frozen rule's, i.e. one with
very little independent variation left to detect — the MDE is still **0.328** at the family's α,
**1.6× the band**. (The one pair with ρ≈1.0, θ=0 `drift-inclusive`, is excluded as degenerate: it
is the *same* position path as the frozen cell differing only by a deterministic cost constant,
so there is nothing there to detect at all.) The candidates the grid actually nominates sit at
ρ ≈ 0.82–0.91, or are cross-H and cannot be paired at all.

### Instrument 2 — a direct bootstrap CI on the uplift itself

The MDE is a summary statistic. The direct test is: **does the uplift's own confidence interval
fall wholly on one side of +0.20?** Where candidate and frozen share their decision rows (same H)
the bootstrap is **paired** — one set of block indices applied to both series — so this needs no
correlation assumption at all. Where H differs the rows cannot be paired and the series are
resampled independently, which over-states the variance and is therefore conservative *in the
direction of refusing a claim*; that is stated, not hidden.

<!-- AUTOGEN:uplift_ci_summary BEGIN -->
| window | cells | raw 95%: CI lower bound > +0.20 | raw 95%: CI wholly BELOW +0.20 | raw 95%: UNDECIDED (straddles +0.20) | **Bonferroni α=0.05/48: clears band** | Bonferroni: UNDECIDED | median CI half-width |
|---|---|---|---|---|---|---|---|
| PRIMARY | 48 | 1 | 3 | 44 | **0** | 46 | 1.271 |
| SECONDARY | 48 | 1 | 4 | 43 | **0** | 46 | 0.985 |
| DISJOINT | 48 | 0 | 2 | 46 | **0** | 46 | 1.547 |

At raw α=0.05 over a 48-cell family, chance alone produces **at most** 1.2 clearances (α/2·n is an UPPER BOUND — it assumes every cell sits exactly at the boundary null, and the cells share position paths, so the true baseline is lower); the measured count is 1.

Every cell clearing the band at raw 95%, and what happens to it under the family's own pre-registered α:

| window | cell | uplift | raw 95% CI | Bonferroni CI (α=0.05/48) | paired? | clears band at Bonferroni? |
|---|---|---|---|---|---|---|
| PRIMARY | H=1, θ=0.25, hold-shares | +0.6523 | [+0.2270, +1.0828] | [-0.0778, +1.3823] | yes | **no** |
| SECONDARY | H=1, θ=0.25, hold-shares | +0.5611 | [+0.2451, +0.8871] | [+0.0183, +1.1039] | yes | **no** |
<!-- AUTOGEN:uplift_ci_summary END -->

**0 of 48 cells clear the +0.20 band at the pre-registered α=0.05/48, on any of the three
windows. 46 of 48 are UNDECIDED — their CI straddles the band.** The grid's own max-uplift cell
(H=2, θ=0.25, hold-shares, point estimate +0.6774) is among the undecided: its raw 95% CI is
[−0.625, +2.048]. A cell whose CI spans 2.7 Sharpe cannot answer a question about 0.20.

One cell — H=1, θ=0.25, hold-shares — clears the band at **raw** 95% on PRIMARY *and* on
SECONDARY. That is worth recording, and it is **not** a result: the clearance test is a two-sided
95% interval's lower bound clearing the band, whose null rate is **at most** α/2 per cell, so raw
α=0.05 over a 48-cell family produces **≤1.2** such clearances by chance and produced 1. That
bound is *not* the expectation and the closing pass relabels it: it is attained only if every one
of the 48 cells sits exactly at the +0.20 boundary null, when most sit far below it and the cells
share position paths, so the true chance baseline is **below** 1.2 — which makes "1 observed ≈
chance" easier to invoke than the data warrant, the opposite of the conservative reading cycle 3
claimed for the correction. It is immaterial to the disposition, which rests on 0/48 at Bonferroni
and 0/48 on DISJOINT. Under
the family's own pre-registered α its interval widens to [−0.078, +1.382] and no longer clears the
band; and on the genuinely disjoint holdout its uplift falls to +0.3096 with CI [−0.110, +0.794],
which excludes neither zero nor the band.

**Cycle 3 withdrew even the candidate status, and the closing pass corrects why.** The retraction
stands; the mechanism cycle 3 gave for it does not.

*The comparator, named correctly.* Cycle 3 wrote that this cell "was clearing the band against the
same-cell target-weight rule". It was not. `attach_uplift_cis` (`grid_harness.py`) fixes the
comparator for **every** uplift on this page to the **frozen** cell — H=1, θ=0, target-weight —
scored in `net21_borrow_low`. Measured against that actual comparator:

| arm | uplift over the frozen cell (`net21_borrow_low`) | raw 95% CI | clears +0.20? |
|---|---|---|---|
| raw candidate (H=1, θ=0.25, hold-shares) | **+0.6523** | [+0.227, +1.083] | yes |
| the same cell de-levered | **+0.3745** | [+0.058, +0.697] | **no** |
| pure-θ effect (H=1, θ=0.25, **target-weight**) | **+0.3263** | [+0.159, +0.504] | no |

*The mechanism, named correctly.* Of the +0.6523, **+0.3263 — half of it — is the θ HYSTERESIS
effect**, which is available under plain canon target-weight accounting and survives de-levering
entirely. Only +0.2778 (43%) is leverage. So cycle 3's "what cleared at raw 95% was a leverage
differential, **not** an execution or accounting effect" is **false on the point estimate**: θ *is*
this family's own execution treatment, and it is the larger of the two halves. The honest statement
is that the clearance was **leverage-ASSISTED**, not leverage-constituted.

*What that leaves.* The retraction is still correct — the de-levered cell no longer clears the
band, so it is a **retracted candidate**, not a surviving one, and its raw clearance sits inside
the ≤1.2 chance baseline for a 48-cell family anyway. But the surviving θ half is the most
interesting result this row produced: a pure target-weight, canon-accounted, gross-1 treatment
worth +0.3263 over the frozen rule with a CI that excludes zero. It does not clear the
pre-registered band and this design cannot resolve whether it would, so it is **not** promoted
here — it is recorded as the one component of the headline that de-levering does not touch.

### Instrument 3 — the disjoint holdout

<!-- AUTOGEN:significance BEGIN -->
| window | cells scored | Bonferroni-significant (α=0.05/48) | raw p<0.05 | 95% CI lower bound > 0 |
|---|---|---|---|---|
| PRIMARY | 48 | **0** | 13 | **5** |
| SECONDARY | 48 | **0** | 10 | **7** |
| DISJOINT | 48 | **0** | 0 | **0** |
<!-- AUTOGEN:significance END -->

On the only genuinely out-of-sample window, **0 of 48 cells' uplift CI even excludes zero**, let
alone clears +0.20 — and the frozen rule itself is *negative* there (net21_borrow_low −0.0364).

### The verdict, stated plainly

> **The pre-registered ±0.20 band question is REFUSED FOR WANT OF POWER.** This design cannot
> resolve a ±0.20 uplift at any plausible correlation assumption, and the refusal is **prior to
> and independent of** the hold-shares fix — no re-implementation rescues it. Same disposition
> run 54 gave TASK_917's P1 (kill refused for want of power → D84).

What *would* resolve it: shrinking a 0.46 SE to the ~0.05 needed for ±0.20 at Bonferroni α needs
on the order of **85× the sample** (SE ∝ 1/√n); or a design that shares decision rows across the
whole grid, which the H axis structurally cannot; or pre-registering a **single** cell instead of
searching 48. Planner decision, not this pass's.

## The 48-cell grid (PRIMARY window)

Fee tiers 8/12/21 bp/side + `net21_borrow_low` (the primary gate quantity, per D81(c) brief
section 2). `p` is the one-sided normal p-value on `net21_borrow_low`; `sig` marks Bonferroni
significance at α=0.05/48=0.0010417. 916's 40/80 bp ladder and the `none`/`high` borrow scenarios
are in `result_919a.json` (reported, not searched — they do not multiply the 48-cell family).

Four comparability warnings, all load-bearing:

1. **The `cost basis` column is not decoration** — the three modes' `turn` columns are three
   different quantities on two different books ("Cost bases are not comparable across accounting
   modes", below).
2. **`gross` is not the same estimator across modes.** `target-weight` and `drift-inclusive`
   report the canon weighted **log** return; `hold-shares` reports the weighted **simple** return
   its equity path actually realises. Part of every hold-shares−vs−target-weight gap is therefore
   a compounding convention, not a share carry — split explicitly in CORRECTION 2, and material
   (it is *most* of the BEAR22 gap).
3. **Every `p` and every CI here is a point-estimate instrument on an underpowered design** — see
   the previous section before acting on any of them.
4. **The `hold-shares` rows do not run at the canon's gross exposure.** Read a `hold-shares`
   `gross`/`net` column as *"a levered book"*, not as *"the same book, accounted differently"*.
   De-levering removes ~85% of the advantage over the same-cell `target-weight` row, but the
   sign of what remains is **construction-dependent** and no construction's CI clears zero, so
   the advantage is **UNRESOLVED**, not shown to be leverage (BLOCKER-E2A, below).

<!-- AUTOGEN:grid_primary BEGIN -->
| H | theta | accounting | cost basis | n | gross | turn/dec | turn/day | net8 | net12 | net21 | net21_borrow_low | p | CI95 lo..hi | sig |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.0 | target-weight | target-change only | 1704 | +1.0339 | 0.1561 | 0.1561 | +0.9584 | +0.9207 | +0.8358 | +0.4264 | 0.1784 | -0.457..+1.296 |  |
| 1 | 0.0 | drift-inclusive | upper bound (unnetted) | 1704 | +1.0339 | 0.1998 | 0.1998 | +0.9373 | +0.8890 | +0.7803 | +0.3710 | 0.2114 | -0.513..+1.241 |  |
| 1 | 0.0 | hold-shares | actual trade, OWN book | 1704 | +1.3132 | 0.1555 | 0.1555 | +1.2330 | +1.1928 | +1.1025 | +0.7447 | 0.0538 | -0.120..+1.642 |  |
| 1 | 0.25 | target-weight | target-change only | 1704 | +1.2476 | 0.0642 | 0.0642 | +1.2164 | +1.2008 | +1.1657 | +0.7527 | 0.0519 | -0.126..+1.631 |  |
| 1 | 0.25 | drift-inclusive | upper bound (unnetted) | 1704 | +1.2476 | 0.1080 | 0.1080 | +1.1952 | +1.1689 | +1.1099 | +0.6969 | 0.0661 | -0.184..+1.576 |  |
| 1 | 0.25 | hold-shares | actual trade, OWN book | 1704 | +1.5178 | 0.0633 | 0.0633 | +1.4850 | +1.4686 | +1.4318 | +1.0787 | 0.0099 | +0.199..+1.974 |  |
| 1 | 0.5 | target-weight | target-change only | 1704 | +1.0112 | 0.0438 | 0.0438 | +0.9897 | +0.9790 | +0.9548 | +0.5374 | 0.1228 | -0.367..+1.418 |  |
| 1 | 0.5 | drift-inclusive | upper bound (unnetted) | 1704 | +1.0112 | 0.0876 | 0.0876 | +0.9683 | +0.9469 | +0.8987 | +0.4812 | 0.1492 | -0.423..+1.364 |  |
| 1 | 0.5 | hold-shares | actual trade, OWN book | 1704 | +1.2355 | 0.0434 | 0.0434 | +1.2134 | +1.2023 | +1.1775 | +0.8302 | 0.0364 | -0.066..+1.727 |  |
| 1 | 1.0 | target-weight | target-change only | 1704 | +0.5576 | 0.0280 | 0.0280 | +0.5437 | +0.5367 | +0.5211 | +0.0866 | 0.4258 | -0.820..+1.016 |  |
| 1 | 1.0 | drift-inclusive | upper bound (unnetted) | 1704 | +0.5576 | 0.0717 | 0.0717 | +0.5220 | +0.5042 | +0.4641 | +0.0296 | 0.4745 | -0.875..+0.960 |  |
| 1 | 1.0 | hold-shares | actual trade, OWN book | 1704 | +0.6895 | 0.0284 | 0.0284 | +0.6762 | +0.6695 | +0.6545 | +0.3263 | 0.2404 | -0.584..+1.266 |  |
| 2 | 0.0 | target-weight | target-change only | 852 | +1.2305 | 0.2194 | 0.1097 | +1.1774 | +1.1508 | +1.0910 | +0.6808 | 0.0706 | -0.228..+1.531 |  |
| 2 | 0.0 | drift-inclusive | upper bound (unnetted) | 852 | +1.2305 | 0.2801 | 0.1400 | +1.1627 | +1.1288 | +1.0524 | +0.6423 | 0.0826 | -0.269..+1.493 |  |
| 2 | 0.0 | hold-shares | actual trade, OWN book | 852 | +1.5228 | 0.2178 | 0.1089 | +1.4649 | +1.4360 | +1.3709 | +1.0054 | 0.0149 | +0.035..+1.942 |  |
| 2 | 0.25 | target-weight | target-change only | 852 | +1.2642 | 0.1111 | 0.0556 | +1.2371 | +1.2235 | +1.1930 | +0.7780 | 0.0464 | -0.140..+1.617 |  |
| 2 | 0.25 | drift-inclusive | upper bound (unnetted) | 852 | +1.2642 | 0.1718 | 0.0859 | +1.2222 | +1.2013 | +1.1541 | +0.7391 | 0.0551 | -0.181..+1.577 |  |
| 2 | 0.25 | hold-shares | actual trade, OWN book | 852 | +1.5372 | 0.1094 | 0.0547 | +1.5084 | +1.4940 | +1.4616 | +1.1038 | 0.0085 | +0.147..+2.030 |  |
| 2 | 0.5 | target-weight | target-change only | 852 | +0.9887 | 0.0809 | 0.0404 | +0.9689 | +0.9590 | +0.9366 | +0.5177 | 0.1317 | -0.444..+1.388 |  |
| 2 | 0.5 | drift-inclusive | upper bound (unnetted) | 852 | +0.9887 | 0.1416 | 0.0708 | +0.9540 | +0.9366 | +0.8975 | +0.4786 | 0.1506 | -0.488..+1.349 |  |
| 2 | 0.5 | hold-shares | actual trade, OWN book | 852 | +1.2179 | 0.0800 | 0.0400 | +1.1973 | +1.1871 | +1.1640 | +0.8135 | 0.0394 | -0.184..+1.736 |  |
| 2 | 1.0 | target-weight | target-change only | 852 | +0.4575 | 0.0513 | 0.0257 | +0.4445 | +0.4380 | +0.4235 | -0.0159 | 0.5137 | -1.008..+0.910 |  |
| 2 | 1.0 | drift-inclusive | upper bound (unnetted) | 852 | +0.4575 | 0.1120 | 0.0560 | +0.4292 | +0.4151 | +0.3832 | -0.0562 | 0.5483 | -1.053..+0.873 |  |
| 2 | 1.0 | hold-shares | actual trade, OWN book | 852 | +0.5822 | 0.0527 | 0.0263 | +0.5695 | +0.5632 | +0.5490 | +0.2123 | 0.3232 | -0.841..+1.181 |  |
| 3 | 0.0 | target-weight | target-change only | 568 | +1.3315 | 0.2648 | 0.0883 | +1.2858 | +1.2629 | +1.2113 | +0.7698 | 0.0481 | -0.182..+1.608 |  |
| 3 | 0.0 | drift-inclusive | upper bound (unnetted) | 568 | +1.3315 | 0.3391 | 0.1130 | +1.2729 | +1.2436 | +1.1777 | +0.7361 | 0.0559 | -0.221..+1.574 |  |
| 3 | 0.0 | hold-shares | actual trade, OWN book | 568 | +1.5261 | 0.2624 | 0.0875 | +1.4790 | +1.4555 | +1.4025 | +1.0313 | 0.0129 | +0.060..+1.930 |  |
| 3 | 0.25 | target-weight | target-change only | 568 | +1.2936 | 0.1505 | 0.0502 | +1.2678 | +1.2549 | +1.2258 | +0.7863 | 0.0447 | -0.188..+1.615 |  |
| 3 | 0.25 | drift-inclusive | upper bound (unnetted) | 568 | +1.2936 | 0.2250 | 0.0750 | +1.2551 | +1.2358 | +1.1925 | +0.7529 | 0.0519 | -0.223..+1.584 |  |
| 3 | 0.25 | hold-shares | actual trade, OWN book | 568 | +1.4842 | 0.1483 | 0.0494 | +1.4579 | +1.4448 | +1.4153 | +1.0552 | 0.0113 | +0.102..+1.978 |  |
| 3 | 0.5 | target-weight | target-change only | 568 | +1.0137 | 0.1153 | 0.0384 | +0.9941 | +0.9843 | +0.9623 | +0.5237 | 0.1289 | -0.449..+1.388 |  |
| 3 | 0.5 | drift-inclusive | upper bound (unnetted) | 568 | +1.0137 | 0.1897 | 0.0632 | +0.9815 | +0.9655 | +0.9293 | +0.4907 | 0.1445 | -0.485..+1.356 |  |
| 3 | 0.5 | hold-shares | actual trade, OWN book | 568 | +1.1988 | 0.1141 | 0.0380 | +1.1789 | +1.1689 | +1.1465 | +0.7913 | 0.0436 | -0.170..+1.728 |  |
| 3 | 1.0 | target-weight | target-change only | 568 | +0.5305 | 0.0733 | 0.0244 | +0.5177 | +0.5113 | +0.4969 | +0.0448 | 0.4615 | -0.932..+0.957 |  |
| 3 | 1.0 | drift-inclusive | upper bound (unnetted) | 568 | +0.5305 | 0.1477 | 0.0492 | +0.5047 | +0.4918 | +0.4628 | +0.0107 | 0.4908 | -0.969..+0.923 |  |
| 3 | 1.0 | hold-shares | actual trade, OWN book | 568 | +0.6914 | 0.0743 | 0.0248 | +0.6800 | +0.6743 | +0.6614 | +0.3449 | 0.2281 | -0.563..+1.284 |  |
| 5 | 0.0 | target-weight | target-change only | 340 | +1.2014 | 0.3464 | 0.0693 | +1.1662 | +1.1486 | +1.1090 | +0.6793 | 0.0713 | -0.221..+1.542 |  |
| 5 | 0.0 | drift-inclusive | upper bound (unnetted) | 340 | +1.2014 | 0.4440 | 0.0888 | +1.1563 | +1.1338 | +1.0830 | +0.6534 | 0.0793 | -0.250..+1.519 |  |
| 5 | 0.0 | hold-shares | actual trade, OWN book | 340 | +1.3849 | 0.3429 | 0.0686 | +1.3503 | +1.3329 | +1.2939 | +0.9474 | 0.0204 | -0.024..+1.841 |  |
| 5 | 0.25 | target-weight | target-change only | 340 | +1.1224 | 0.2265 | 0.0453 | +1.0993 | +1.0878 | +1.0618 | +0.6268 | 0.0881 | -0.270..+1.494 |  |
| 5 | 0.25 | drift-inclusive | upper bound (unnetted) | 340 | +1.1224 | 0.3242 | 0.0648 | +1.0894 | +1.0729 | +1.0358 | +0.6008 | 0.0974 | -0.296..+1.467 |  |
| 5 | 0.25 | hold-shares | actual trade, OWN book | 340 | +1.3121 | 0.2234 | 0.0447 | +1.2895 | +1.2782 | +1.2528 | +0.9079 | 0.0250 | -0.079..+1.783 |  |
| 5 | 0.5 | target-weight | target-change only | 340 | +0.9004 | 0.1748 | 0.0350 | +0.8829 | +0.8742 | +0.8545 | +0.4246 | 0.1797 | -0.479..+1.294 |  |
| 5 | 0.5 | drift-inclusive | upper bound (unnetted) | 340 | +0.9004 | 0.2725 | 0.0545 | +0.8732 | +0.8595 | +0.8289 | +0.3990 | 0.1946 | -0.506..+1.270 |  |
| 5 | 0.5 | hold-shares | actual trade, OWN book | 340 | +1.1194 | 0.1726 | 0.0345 | +1.1022 | +1.0936 | +1.0742 | +0.7358 | 0.0562 | -0.231..+1.646 |  |
| 5 | 1.0 | target-weight | target-change only | 340 | +0.5380 | 0.1067 | 0.0213 | +0.5272 | +0.5217 | +0.5095 | +0.0681 | 0.4416 | -0.844..+0.962 |  |
| 5 | 1.0 | drift-inclusive | upper bound (unnetted) | 340 | +0.5380 | 0.2044 | 0.0409 | +0.5172 | +0.5068 | +0.4834 | +0.0419 | 0.4640 | -0.870..+0.935 |  |
| 5 | 1.0 | hold-shares | actual trade, OWN book | 340 | +0.6393 | 0.1095 | 0.0219 | +0.6295 | +0.6246 | +0.6136 | +0.3060 | 0.2545 | -0.659..+1.211 |  |
<!-- AUTOGEN:grid_primary END -->

### The same 48 cells, expressed as the quantity the band is stated in

<!-- AUTOGEN:uplift_grid_primary BEGIN -->
| H | theta | accounting | uplift Δ vs frozen | raw 95% CI | Bonferroni CI | paired? | band verdict for this cell |
|---|---|---|---|---|---|---|---|
| 1 | 0.0 | target-weight | +0.0000 | [+0.000, +0.000] | [+0.000, +0.000] | yes | inside band |
| 1 | 0.0 | drift-inclusive | -0.0554 | [-0.060, -0.051] | [-0.063, -0.048] | yes | inside band |
| 1 | 0.0 | hold-shares | +0.3183 | [-0.065, +0.693] | [-0.323, +0.960] | yes | UNDECIDED |
| 1 | 0.25 | target-weight | +0.3263 | [+0.159, +0.504] | [+0.037, +0.615] | yes | UNDECIDED |
| 1 | 0.25 | drift-inclusive | +0.2705 | [+0.102, +0.447] | [-0.017, +0.558] | yes | UNDECIDED |
| 1 | 0.25 | hold-shares | +0.6523 | [+0.227, +1.083] | [-0.078, +1.382] | yes | UNDECIDED |
| 1 | 0.5 | target-weight | +0.1110 | [-0.193, +0.397] | [-0.390, +0.612] | yes | UNDECIDED |
| 1 | 0.5 | drift-inclusive | +0.0548 | [-0.249, +0.340] | [-0.447, +0.556] | yes | UNDECIDED |
| 1 | 0.5 | hold-shares | +0.4037 | [-0.101, +0.920] | [-0.465, +1.272] | yes | UNDECIDED |
| 1 | 1.0 | target-weight | -0.3398 | [-0.950, +0.234] | [-1.350, +0.670] | yes | UNDECIDED |
| 1 | 1.0 | drift-inclusive | -0.3969 | [-1.007, +0.176] | [-1.407, +0.613] | yes | UNDECIDED |
| 1 | 1.0 | hold-shares | -0.1001 | [-0.888, +0.699] | [-1.472, +1.271] | yes | UNDECIDED |
| 2 | 0.0 | target-weight | +0.2544 | [-0.999, +1.550] | [-1.876, +2.385] | no | UNDECIDED |
| 2 | 0.0 | drift-inclusive | +0.2159 | [-1.041, +1.508] | [-1.916, +2.348] | no | UNDECIDED |
| 2 | 0.0 | hold-shares | +0.5790 | [-0.741, +1.934] | [-1.649, +2.807] | no | UNDECIDED |
| 2 | 0.25 | target-weight | +0.3515 | [-0.882, +1.634] | [-1.769, +2.472] | no | UNDECIDED |
| 2 | 0.25 | drift-inclusive | +0.3127 | [-0.925, +1.597] | [-1.809, +2.435] | no | UNDECIDED |
| 2 | 0.25 | hold-shares | +0.6774 | [-0.625, +2.048] | [-1.533, +2.888] | no | UNDECIDED |
| 2 | 0.5 | target-weight | +0.0913 | [-1.139, +1.407] | [-2.063, +2.245] | no | UNDECIDED |
| 2 | 0.5 | drift-inclusive | +0.0522 | [-1.181, +1.373] | [-2.104, +2.208] | no | UNDECIDED |
| 2 | 0.5 | hold-shares | +0.3871 | [-0.920, +1.745] | [-1.842, +2.616] | no | UNDECIDED |
| 2 | 1.0 | target-weight | -0.4423 | [-1.727, +0.894] | [-2.671, +1.786] | no | UNDECIDED |
| 2 | 1.0 | drift-inclusive | -0.4826 | [-1.767, +0.855] | [-2.713, +1.748] | no | UNDECIDED |
| 2 | 1.0 | hold-shares | -0.2141 | [-1.559, +1.126] | [-2.493, +2.065] | no | UNDECIDED |
| 3 | 0.0 | target-weight | +0.3434 | [-0.946, +1.621] | [-1.779, +2.466] | no | UNDECIDED |
| 3 | 0.0 | drift-inclusive | +0.3097 | [-0.979, +1.588] | [-1.814, +2.434] | no | UNDECIDED |
| 3 | 0.0 | hold-shares | +0.6049 | [-0.725, +1.871] | [-1.585, +2.794] | no | UNDECIDED |
| 3 | 0.25 | target-weight | +0.3598 | [-0.936, +1.615] | [-1.779, +2.499] | no | UNDECIDED |
| 3 | 0.25 | drift-inclusive | +0.3265 | [-0.972, +1.584] | [-1.814, +2.467] | no | UNDECIDED |
| 3 | 0.25 | hold-shares | +0.6288 | [-0.695, +1.886] | [-1.569, +2.826] | no | UNDECIDED |
| 3 | 0.5 | target-weight | +0.0973 | [-1.224, +1.375] | [-2.070, +2.265] | no | UNDECIDED |
| 3 | 0.5 | drift-inclusive | +0.0643 | [-1.257, +1.341] | [-2.105, +2.234] | no | UNDECIDED |
| 3 | 0.5 | hold-shares | +0.3649 | [-0.983, +1.620] | [-1.838, +2.568] | no | UNDECIDED |
| 3 | 1.0 | target-weight | -0.3816 | [-1.720, +0.910] | [-2.607, +1.844] | no | UNDECIDED |
| 3 | 1.0 | drift-inclusive | -0.4157 | [-1.755, +0.876] | [-2.643, +1.811] | no | UNDECIDED |
| 3 | 1.0 | hold-shares | -0.0815 | [-1.422, +1.186] | [-2.277, +2.114] | no | UNDECIDED |
| 5 | 0.0 | target-weight | +0.2529 | [-1.033, +1.464] | [-1.828, +2.333] | no | UNDECIDED |
| 5 | 0.0 | drift-inclusive | +0.2270 | [-1.060, +1.441] | [-1.855, +2.309] | no | UNDECIDED |
| 5 | 0.0 | hold-shares | +0.5210 | [-0.799, +1.738] | [-1.668, +2.710] | no | UNDECIDED |
| 5 | 0.25 | target-weight | +0.2004 | [-1.112, +1.397] | [-1.888, +2.289] | no | UNDECIDED |
| 5 | 0.25 | drift-inclusive | +0.1744 | [-1.138, +1.374] | [-1.916, +2.265] | no | UNDECIDED |
| 5 | 0.25 | hold-shares | +0.4814 | [-0.885, +1.694] | [-1.702, +2.665] | no | UNDECIDED |
| 5 | 0.5 | target-weight | -0.0018 | [-1.297, +1.179] | [-2.116, +2.112] | no | UNDECIDED |
| 5 | 0.5 | drift-inclusive | -0.0274 | [-1.323, +1.155] | [-2.143, +2.088] | no | UNDECIDED |
| 5 | 0.5 | hold-shares | +0.3093 | [-1.063, +1.533] | [-1.878, +2.497] | no | UNDECIDED |
| 5 | 1.0 | target-weight | -0.3584 | [-1.681, +0.845] | [-2.489, +1.772] | no | UNDECIDED |
| 5 | 1.0 | drift-inclusive | -0.3845 | [-1.708, +0.819] | [-2.516, +1.747] | no | UNDECIDED |
| 5 | 1.0 | hold-shares | -0.1204 | [-1.488, +1.034] | [-2.307, +2.066] | no | UNDECIDED |
<!-- AUTOGEN:uplift_grid_primary END -->

**Best cell across the three windows** (selected on PRIMARY, so only DISJOINT is out-of-sample):

<!-- AUTOGEN:best_cell BEGIN -->
| window | relation to PRIMARY | n | best cell's `net21_borrow_low` | its window's frozen cell | uplift Δ | inside ±0.20 band? |
|---|---|---|---|---|---|---|
| PRIMARY | SELECTED here | 852 | +1.1038 | +0.4264 | **+0.6774** | no |
| SECONDARY | SUPERSET (not OOS) | 1400 | +0.8121 | +0.2223 | **+0.5898** | no |
| DISJOINT | DISJOINT (OOS) | 548 | +0.2847 | -0.0364 | **+0.3211** | no |

Best cell (selected on PRIMARY by `net21_borrow_low`): **H=2, θ=0.25, hold-shares**.
<!-- AUTOGEN:best_cell END -->

**Did the hold-shares fix change the best cell? Yes.** Cycle 1 reported H=3, θ=0.25,
*mislabeled* hold-shares; with a real share-carrying path the best cell is **H=2, θ=0.25,
hold-shares** at +1.1038 (uplift +0.6774 vs +0.3643 before). **Did it change the band picture?
No.** Every one of those uplifts is a point estimate on an interval that straddles +0.20 at the
family's own α — the fix changed the label, the mechanism and the point estimates, but not the
conclusion, which is exactly what the power analysis predicts.

> **And the "best cell" is not a strategy finding (BLOCKER-E2A, cycle 3; scope corrected in the
> closing pass).** It is a hold-shares cell, so a large part of its edge over the same-cell
> target-weight rule is gross exposure the canon book is not permitted to carry: the
> leverage-neutral rescaling takes this cell's gross edge from **+0.2730 to −0.0049** and its
> net@21 edge from +0.2686 to −0.0081. What that does **not** establish is that the edge *is*
> leverage — the residue's sign flips with the construction and no construction's CI clears
> zero. See **"De-levering the hold-shares book: what it settles and what it does not"**. The row
> above is retained because the grid's arithmetic is what it is; it should be read as *"the
> search's maximum lands on a cell whose book is not gross-1 constrained"*, not as a nominated
> rule. (Cycle 3 wrote *"the cell that drifts to the most leverage"*, which the exposure table
> refutes: H=2, θ=0.25 has mean Σ\|w\| 0.967, the 4th-**lowest** of the 16.)

The full SECONDARY and DISJOINT grids:

<details><summary>SECONDARY window, 48 cells (superset of PRIMARY — NOT out-of-sample)</summary>

<!-- AUTOGEN:grid_secondary BEGIN -->
| H | theta | accounting | cost basis | n | gross | turn/dec | turn/day | net8 | net12 | net21 | net21_borrow_low | p | CI95 lo..hi | sig |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.0 | target-weight | target-change only | 2800 | +0.7626 | 0.1525 | 0.1525 | +0.6949 | +0.6610 | +0.5849 | +0.2223 | 0.2691 | -0.467..+0.897 |  |
| 1 | 0.0 | drift-inclusive | upper bound (unnetted) | 2800 | +0.7626 | 0.1947 | 0.1947 | +0.6762 | +0.6329 | +0.5357 | +0.1731 | 0.3158 | -0.520..+0.847 |  |
| 1 | 0.0 | hold-shares | actual trade, OWN book | 2800 | +1.0898 | 0.1519 | 0.1519 | +1.0172 | +0.9809 | +0.8992 | +0.5793 | 0.0543 | -0.123..+1.232 |  |
| 1 | 0.25 | target-weight | target-change only | 2800 | +0.8591 | 0.0656 | 0.0656 | +0.8298 | +0.8151 | +0.7821 | +0.4156 | 0.1248 | -0.286..+1.080 |  |
| 1 | 0.25 | drift-inclusive | upper bound (unnetted) | 2800 | +0.8591 | 0.1078 | 0.1078 | +0.8109 | +0.7868 | +0.7325 | +0.3660 | 0.1553 | -0.337..+1.032 |  |
| 1 | 0.25 | hold-shares | actual trade, OWN book | 2800 | +1.1832 | 0.0647 | 0.0647 | +1.1522 | +1.1367 | +1.1017 | +0.7833 | 0.0150 | +0.072..+1.455 |  |
| 1 | 0.5 | target-weight | target-change only | 2800 | +0.6591 | 0.0453 | 0.0453 | +0.6383 | +0.6278 | +0.6044 | +0.2247 | 0.2668 | -0.479..+0.908 |  |
| 1 | 0.5 | drift-inclusive | upper bound (unnetted) | 2800 | +0.6591 | 0.0876 | 0.0876 | +0.6189 | +0.5987 | +0.5535 | +0.1738 | 0.3151 | -0.530..+0.858 |  |
| 1 | 0.5 | hold-shares | actual trade, OWN book | 2800 | +0.9872 | 0.0449 | 0.0449 | +0.9662 | +0.9556 | +0.9320 | +0.6138 | 0.0446 | -0.094..+1.295 |  |
| 1 | 1.0 | target-weight | target-change only | 2800 | +0.3215 | 0.0290 | 0.0290 | +0.3079 | +0.3010 | +0.2857 | -0.1190 | 0.6292 | -0.837..+0.604 |  |
| 1 | 1.0 | drift-inclusive | upper bound (unnetted) | 2800 | +0.3215 | 0.0713 | 0.0713 | +0.2880 | +0.2712 | +0.2336 | -0.1711 | 0.6822 | -0.890..+0.551 |  |
| 1 | 1.0 | hold-shares | actual trade, OWN book | 2800 | +0.5488 | 0.0294 | 0.0294 | +0.5356 | +0.5290 | +0.5142 | +0.1960 | 0.2936 | -0.529..+0.930 |  |
| 2 | 0.0 | target-weight | target-change only | 1400 | +0.9567 | 0.2122 | 0.1061 | +0.9065 | +0.8814 | +0.8249 | +0.4380 | 0.1126 | -0.276..+1.154 |  |
| 2 | 0.0 | drift-inclusive | upper bound (unnetted) | 1400 | +0.9567 | 0.2702 | 0.1351 | +0.8928 | +0.8608 | +0.7889 | +0.4019 | 0.1328 | -0.314..+1.118 |  |
| 2 | 0.0 | hold-shares | actual trade, OWN book | 1400 | +1.2706 | 0.2104 | 0.1052 | +1.2173 | +1.1907 | +1.1308 | +0.7935 | 0.0140 | +0.057..+1.555 |  |
| 2 | 0.25 | target-weight | target-change only | 1400 | +0.9233 | 0.1143 | 0.0571 | +0.8961 | +0.8825 | +0.8518 | +0.4603 | 0.1012 | -0.262..+1.190 |  |
| 2 | 0.25 | drift-inclusive | upper bound (unnetted) | 1400 | +0.9233 | 0.1724 | 0.0862 | +0.8822 | +0.8617 | +0.8154 | +0.4240 | 0.1201 | -0.298..+1.156 |  |
| 2 | 0.25 | hold-shares | actual trade, OWN book | 1400 | +1.2187 | 0.1126 | 0.0563 | +1.1905 | +1.1765 | +1.1448 | +0.8121 | 0.0123 | +0.074..+1.572 |  |
| 2 | 0.5 | target-weight | target-change only | 1400 | +0.7306 | 0.0836 | 0.0418 | +0.7104 | +0.7004 | +0.6777 | +0.2786 | 0.2201 | -0.448..+1.012 |  |
| 2 | 0.5 | drift-inclusive | upper bound (unnetted) | 1400 | +0.7306 | 0.1417 | 0.0709 | +0.6965 | +0.6794 | +0.6411 | +0.2420 | 0.2514 | -0.486..+0.976 |  |
| 2 | 0.5 | hold-shares | actual trade, OWN book | 1400 | +1.0600 | 0.0824 | 0.0412 | +1.0400 | +1.0301 | +1.0076 | +0.6816 | 0.0295 | -0.042..+1.460 |  |
| 2 | 1.0 | target-weight | target-change only | 1400 | +0.3901 | 0.0521 | 0.0261 | +0.3773 | +0.3708 | +0.3564 | -0.0664 | 0.5729 | -0.840..+0.694 |  |
| 2 | 1.0 | drift-inclusive | upper bound (unnetted) | 1400 | +0.3901 | 0.1102 | 0.0551 | +0.3630 | +0.3494 | +0.3189 | -0.1039 | 0.6132 | -0.876..+0.655 |  |
| 2 | 1.0 | hold-shares | actual trade, OWN book | 1400 | +0.5489 | 0.0528 | 0.0264 | +0.5363 | +0.5299 | +0.5156 | +0.1825 | 0.3066 | -0.590..+0.963 |  |
| 3 | 0.0 | target-weight | target-change only | 933 | +1.0174 | 0.2569 | 0.0856 | +0.9775 | +0.9576 | +0.9128 | +0.5335 | 0.0698 | -0.185..+1.212 |  |
| 3 | 0.0 | drift-inclusive | upper bound (unnetted) | 933 | +1.0174 | 0.3296 | 0.1099 | +0.9663 | +0.9407 | +0.8832 | +0.5040 | 0.0814 | -0.216..+1.184 |  |
| 3 | 0.0 | hold-shares | actual trade, OWN book | 933 | +1.3386 | 0.2538 | 0.0846 | +1.2968 | +1.2760 | +1.2290 | +0.9002 | 0.0063 | +0.165..+1.594 |  |
| 3 | 0.25 | target-weight | target-change only | 933 | +0.8704 | 0.1551 | 0.0517 | +0.8465 | +0.8346 | +0.8077 | +0.4281 | 0.1179 | -0.288..+1.097 |  |
| 3 | 0.25 | drift-inclusive | upper bound (unnetted) | 933 | +0.8704 | 0.2279 | 0.0760 | +0.8353 | +0.8178 | +0.7783 | +0.3987 | 0.1348 | -0.319..+1.070 |  |
| 3 | 0.25 | hold-shares | actual trade, OWN book | 933 | +1.1830 | 0.1526 | 0.0509 | +1.1583 | +1.1460 | +1.1181 | +0.7938 | 0.0140 | +0.051..+1.475 |  |
| 3 | 0.5 | target-weight | target-change only | 933 | +0.6502 | 0.1169 | 0.0390 | +0.6322 | +0.6232 | +0.6030 | +0.2211 | 0.2702 | -0.497..+0.905 |  |
| 3 | 0.5 | drift-inclusive | upper bound (unnetted) | 933 | +0.6502 | 0.1898 | 0.0633 | +0.6210 | +0.6065 | +0.5737 | +0.1918 | 0.2977 | -0.526..+0.877 |  |
| 3 | 0.5 | hold-shares | actual trade, OWN book | 933 | +0.9592 | 0.1155 | 0.0385 | +0.9412 | +0.9322 | +0.9120 | +0.5955 | 0.0496 | -0.123..+1.306 |  |
| 3 | 1.0 | target-weight | target-change only | 933 | +0.2866 | 0.0744 | 0.0248 | +0.2750 | +0.2692 | +0.2561 | -0.1465 | 0.6575 | -0.833..+0.590 |  |
| 3 | 1.0 | drift-inclusive | upper bound (unnetted) | 933 | +0.2866 | 0.1473 | 0.0491 | +0.2636 | +0.2521 | +0.2262 | -0.1763 | 0.6873 | -0.862..+0.561 |  |
| 3 | 1.0 | hold-shares | actual trade, OWN book | 933 | +0.4967 | 0.0759 | 0.0253 | +0.4851 | +0.4793 | +0.4663 | +0.1450 | 0.3440 | -0.620..+0.898 |  |
| 5 | 0.0 | target-weight | target-change only | 560 | +1.0425 | 0.3379 | 0.0676 | +1.0101 | +0.9938 | +0.9573 | +0.5624 | 0.0597 | -0.063..+1.180 |  |
| 5 | 0.0 | drift-inclusive | upper bound (unnetted) | 560 | +1.0425 | 0.4322 | 0.0864 | +1.0010 | +0.9802 | +0.9335 | +0.5386 | 0.0679 | -0.086..+1.158 |  |
| 5 | 0.0 | hold-shares | actual trade, OWN book | 560 | +1.3201 | 0.3334 | 0.0667 | +1.2855 | +1.2682 | +1.2293 | +0.8844 | 0.0072 | +0.213..+1.571 |  |
| 5 | 0.25 | target-weight | target-change only | 560 | +0.9445 | 0.2282 | 0.0456 | +0.9227 | +0.9117 | +0.8871 | +0.4894 | 0.0876 | -0.152..+1.120 |  |
| 5 | 0.25 | drift-inclusive | upper bound (unnetted) | 560 | +0.9445 | 0.3226 | 0.0645 | +0.9136 | +0.8981 | +0.8633 | +0.4656 | 0.0986 | -0.175..+1.097 |  |
| 5 | 0.25 | hold-shares | actual trade, OWN book | 560 | +1.2312 | 0.2240 | 0.0448 | +1.2079 | +1.1963 | +1.1700 | +0.8270 | 0.0110 | +0.135..+1.516 |  |
| 5 | 0.5 | target-weight | target-change only | 560 | +0.6589 | 0.1813 | 0.0363 | +0.6413 | +0.6325 | +0.6127 | +0.2074 | 0.2828 | -0.472..+0.880 |  |
| 5 | 0.5 | drift-inclusive | upper bound (unnetted) | 560 | +0.6589 | 0.2757 | 0.0551 | +0.6322 | +0.6188 | +0.5886 | +0.1834 | 0.3058 | -0.497..+0.857 |  |
| 5 | 0.5 | hold-shares | actual trade, OWN book | 560 | +0.9041 | 0.1791 | 0.0358 | +0.8863 | +0.8773 | +0.8572 | +0.5180 | 0.0757 | -0.228..+1.270 |  |
| 5 | 1.0 | target-weight | target-change only | 560 | +0.1905 | 0.1152 | 0.0230 | +0.1790 | +0.1732 | +0.1603 | -0.2660 | 0.7693 | -0.966..+0.406 |  |
| 5 | 1.0 | drift-inclusive | upper bound (unnetted) | 560 | +0.1905 | 0.2096 | 0.0419 | +0.1696 | +0.1591 | +0.1356 | -0.2907 | 0.7896 | -0.991..+0.382 |  |
| 5 | 1.0 | hold-shares | actual trade, OWN book | 560 | +0.4513 | 0.1182 | 0.0236 | +0.4407 | +0.4354 | +0.4235 | +0.1006 | 0.3902 | -0.697..+0.847 |  |
<!-- AUTOGEN:grid_secondary END -->

</details>

<details><summary>DISJOINT window, 48 cells (2018-2020 — the only out-of-sample read)</summary>

<!-- AUTOGEN:grid_disjoint BEGIN -->
| H | theta | accounting | cost basis | n | gross | turn/dec | turn/day | net8 | net12 | net21 | net21_borrow_low | p | CI95 lo..hi | sig |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.0 | target-weight | target-change only | 1096 | +0.4240 | 0.1470 | 0.1470 | +0.3656 | +0.3363 | +0.2705 | -0.0364 | 0.5251 | -1.113..+0.958 |  |
| 1 | 0.0 | drift-inclusive | upper bound (unnetted) | 1096 | +0.4240 | 0.1869 | 0.1869 | +0.3497 | +0.3125 | +0.2289 | -0.0780 | 0.5538 | -1.158..+0.914 |  |
| 1 | 0.0 | hold-shares | actual trade, OWN book | 1096 | +0.7353 | 0.1462 | 0.1462 | +0.6694 | +0.6364 | +0.5623 | +0.2768 | 0.3157 | -0.801..+1.324 |  |
| 1 | 0.25 | target-weight | target-change only | 1096 | +0.3620 | 0.0677 | 0.0677 | +0.3347 | +0.3211 | +0.2904 | -0.0202 | 0.5139 | -1.116..+0.983 |  |
| 1 | 0.25 | drift-inclusive | upper bound (unnetted) | 1096 | +0.3620 | 0.1076 | 0.1076 | +0.3187 | +0.2970 | +0.2483 | -0.0623 | 0.5430 | -1.157..+0.946 |  |
| 1 | 0.25 | hold-shares | actual trade, OWN book | 1096 | +0.6443 | 0.0669 | 0.0669 | +0.6137 | +0.5984 | +0.5640 | +0.2732 | 0.3180 | -0.808..+1.330 |  |
| 1 | 0.5 | target-weight | target-change only | 1096 | +0.1940 | 0.0476 | 0.0476 | +0.1739 | +0.1639 | +0.1413 | -0.1903 | 0.6292 | -1.266..+0.849 |  |
| 1 | 0.5 | drift-inclusive | upper bound (unnetted) | 1096 | +0.1940 | 0.0875 | 0.0875 | +0.1571 | +0.1386 | +0.0972 | -0.2344 | 0.6577 | -1.309..+0.805 |  |
| 1 | 0.5 | hold-shares | actual trade, OWN book | 1096 | +0.5516 | 0.0470 | 0.0470 | +0.5305 | +0.5200 | +0.4963 | +0.1944 | 0.3681 | -0.890..+1.268 |  |
| 1 | 1.0 | target-weight | target-change only | 1096 | -0.1207 | 0.0310 | 0.0310 | -0.1343 | -0.1412 | -0.1565 | -0.5285 | 0.8201 | -1.618..+0.517 |  |
| 1 | 1.0 | drift-inclusive | upper bound (unnetted) | 1096 | -0.1207 | 0.0709 | 0.0709 | -0.1519 | -0.1675 | -0.2025 | -0.5744 | 0.8402 | -1.666..+0.471 |  |
| 1 | 1.0 | hold-shares | actual trade, OWN book | 1096 | +0.1149 | 0.0316 | 0.0316 | +0.1011 | +0.0942 | +0.0787 | -0.2441 | 0.6638 | -1.312..+0.818 |  |
| 2 | 0.0 | target-weight | target-change only | 548 | +0.5566 | 0.2009 | 0.1005 | +0.5107 | +0.4877 | +0.4360 | +0.0821 | 0.4434 | -1.140..+1.256 |  |
| 2 | 0.0 | drift-inclusive | upper bound (unnetted) | 548 | +0.5566 | 0.2549 | 0.1274 | +0.4983 | +0.4692 | +0.4036 | +0.0497 | 0.4657 | -1.174..+1.223 |  |
| 2 | 0.0 | hold-shares | actual trade, OWN book | 548 | +0.8786 | 0.1989 | 0.0994 | +0.8292 | +0.8045 | +0.7489 | +0.4328 | 0.2267 | -0.711..+1.615 |  |
| 2 | 0.25 | target-weight | target-change only | 548 | +0.4166 | 0.1193 | 0.0596 | +0.3891 | +0.3753 | +0.3444 | -0.0134 | 0.5093 | -1.194..+1.171 |  |
| 2 | 0.25 | drift-inclusive | upper bound (unnetted) | 548 | +0.4166 | 0.1732 | 0.0866 | +0.3766 | +0.3566 | +0.3117 | -0.0461 | 0.5318 | -1.228..+1.141 |  |
| 2 | 0.25 | hold-shares | actual trade, OWN book | 548 | +0.6820 | 0.1177 | 0.0589 | +0.6527 | +0.6381 | +0.6051 | +0.2847 | 0.3109 | -0.868..+1.454 |  |
| 2 | 0.5 | target-weight | target-change only | 548 | +0.3511 | 0.0876 | 0.0438 | +0.3307 | +0.3205 | +0.2974 | -0.0727 | 0.5501 | -1.267..+1.128 |  |
| 2 | 0.5 | drift-inclusive | upper bound (unnetted) | 548 | +0.3511 | 0.1416 | 0.0708 | +0.3181 | +0.3015 | +0.2643 | -0.1058 | 0.5727 | -1.300..+1.096 |  |
| 2 | 0.5 | hold-shares | actual trade, OWN book | 548 | +0.7367 | 0.0858 | 0.0429 | +0.7159 | +0.7056 | +0.6823 | +0.3632 | 0.2646 | -0.821..+1.577 |  |
| 2 | 1.0 | target-weight | target-change only | 548 | +0.0558 | 0.0545 | 0.0272 | +0.0426 | +0.0360 | +0.0212 | -0.3931 | 0.7521 | -1.560..+0.818 |  |
| 2 | 1.0 | drift-inclusive | upper bound (unnetted) | 548 | +0.0558 | 0.1084 | 0.0542 | +0.0295 | +0.0164 | -0.0131 | -0.4274 | 0.7705 | -1.594..+0.786 |  |
| 2 | 1.0 | hold-shares | actual trade, OWN book | 548 | +0.1941 | 0.0552 | 0.0276 | +0.1805 | +0.1737 | +0.1584 | -0.1910 | 0.6297 | -1.339..+1.060 |  |
| 3 | 0.0 | target-weight | target-change only | 365 | +0.6704 | 0.2465 | 0.0822 | +0.6354 | +0.6180 | +0.5786 | +0.2527 | 0.3308 | -0.899..+1.372 |  |
| 3 | 0.0 | drift-inclusive | upper bound (unnetted) | 365 | +0.6704 | 0.3148 | 0.1049 | +0.6258 | +0.6034 | +0.5532 | +0.2273 | 0.3469 | -0.927..+1.347 |  |
| 3 | 0.0 | hold-shares | actual trade, OWN book | 365 | +1.0203 | 0.2427 | 0.0809 | +0.9802 | +0.9601 | +0.9149 | +0.6004 | 0.1492 | -0.505..+1.774 |  |
| 3 | 0.25 | target-weight | target-change only | 365 | +0.3603 | 0.1639 | 0.0546 | +0.3372 | +0.3256 | +0.2996 | -0.0290 | 0.5200 | -1.148..+1.143 |  |
| 3 | 0.25 | drift-inclusive | upper bound (unnetted) | 365 | +0.3603 | 0.2323 | 0.0774 | +0.3276 | +0.3112 | +0.2743 | -0.0543 | 0.5375 | -1.175..+1.117 |  |
| 3 | 0.25 | hold-shares | actual trade, OWN book | 365 | +0.6688 | 0.1612 | 0.0537 | +0.6426 | +0.6295 | +0.6001 | +0.2818 | 0.3127 | -0.843..+1.449 |  |
| 3 | 0.5 | target-weight | target-change only | 365 | +0.2658 | 0.1195 | 0.0398 | +0.2485 | +0.2398 | +0.2203 | -0.1247 | 0.5855 | -1.257..+1.080 |  |
| 3 | 0.5 | drift-inclusive | upper bound (unnetted) | 365 | +0.2658 | 0.1879 | 0.0626 | +0.2385 | +0.2249 | +0.1941 | -0.1508 | 0.6030 | -1.283..+1.052 |  |
| 3 | 0.5 | hold-shares | actual trade, OWN book | 365 | +0.6208 | 0.1174 | 0.0391 | +0.6025 | +0.5934 | +0.5728 | +0.2593 | 0.3267 | -0.901..+1.447 |  |
| 3 | 1.0 | target-weight | target-change only | 365 | -0.1179 | 0.0783 | 0.0261 | -0.1295 | -0.1353 | -0.1483 | -0.5230 | 0.8175 | -1.662..+0.733 |  |
| 3 | 1.0 | drift-inclusive | upper bound (unnetted) | 365 | -0.1179 | 0.1467 | 0.0489 | -0.1396 | -0.1505 | -0.1749 | -0.5495 | 0.8294 | -1.688..+0.707 |  |
| 3 | 1.0 | hold-shares | actual trade, OWN book | 365 | +0.0362 | 0.0803 | 0.0268 | +0.0240 | +0.0180 | +0.0043 | -0.3256 | 0.7136 | -1.516..+0.935 |  |
| 5 | 0.0 | target-weight | target-change only | 219 | +0.5408 | 0.3499 | 0.0700 | +0.5119 | +0.4974 | +0.4649 | +0.1460 | 0.4002 | -0.846..+1.096 |  |
| 5 | 0.0 | drift-inclusive | upper bound (unnetted) | 219 | +0.5408 | 0.4417 | 0.0883 | +0.5043 | +0.4860 | +0.4449 | +0.1260 | 0.4136 | -0.869..+1.078 |  |
| 5 | 0.0 | hold-shares | actual trade, OWN book | 219 | +0.8143 | 0.3451 | 0.0690 | +0.7805 | +0.7636 | +0.7256 | +0.4139 | 0.2367 | -0.689..+1.478 |  |
| 5 | 0.25 | target-weight | target-change only | 219 | +0.5477 | 0.2397 | 0.0479 | +0.5277 | +0.5176 | +0.4951 | +0.1669 | 0.3863 | -0.836..+1.179 |  |
| 5 | 0.25 | drift-inclusive | upper bound (unnetted) | 219 | +0.5477 | 0.3315 | 0.0663 | +0.5199 | +0.5060 | +0.4748 | +0.1467 | 0.3997 | -0.857..+1.157 |  |
| 5 | 0.25 | hold-shares | actual trade, OWN book | 219 | +0.8496 | 0.2340 | 0.0468 | +0.8262 | +0.8145 | +0.7881 | +0.4731 | 0.2063 | -0.610..+1.590 |  |
| 5 | 0.5 | target-weight | target-change only | 219 | +0.3333 | 0.1910 | 0.0382 | +0.3165 | +0.3082 | +0.2894 | -0.0560 | 0.5386 | -1.102..+1.056 |  |
| 5 | 0.5 | drift-inclusive | upper bound (unnetted) | 219 | +0.3333 | 0.2829 | 0.0566 | +0.3085 | +0.2961 | +0.2682 | -0.0770 | 0.5531 | -1.123..+1.034 |  |
| 5 | 0.5 | hold-shares | actual trade, OWN book | 219 | +0.6594 | 0.1871 | 0.0374 | +0.6421 | +0.6334 | +0.6139 | +0.3009 | 0.3011 | -0.847..+1.504 |  |
| 5 | 1.0 | target-weight | target-change only | 219 | -0.1969 | 0.1275 | 0.0255 | -0.2086 | -0.2144 | -0.2275 | -0.6268 | 0.8612 | -1.746..+0.538 |  |
| 5 | 1.0 | drift-inclusive | upper bound (unnetted) | 219 | -0.1969 | 0.2194 | 0.0439 | -0.2169 | -0.2270 | -0.2495 | -0.6487 | 0.8694 | -1.767..+0.517 |  |
| 5 | 1.0 | hold-shares | actual trade, OWN book | 219 | -0.1156 | 0.1337 | 0.0267 | -0.1276 | -0.1336 | -0.1471 | -0.5091 | 0.8110 | -1.708..+0.764 |  |
<!-- AUTOGEN:grid_disjoint END -->

</details>

## Cost bases are not comparable across accounting modes (MAJOR-F2)

The three modes' `turn` columns are **three different quantities**, and since this pass's
hold-shares fix, two different **books**. Measured on every (H, θ) cell of the PRIMARY window:

<!-- AUTOGEN:turnover_bases BEGIN -->
| H | theta | target_change_only | upper_bound_unnetted | netted_combined | hold_shares_actual |
|---|---|---|---|---|---|
| 1 | 0.0 | 0.1561 | 0.1998 | 0.1952 | 0.1555 |
| 1 | 0.25 | 0.0642 | 0.1080 | 0.1055 | 0.0633 |
| 1 | 0.5 | 0.0438 | 0.0876 | 0.0856 | 0.0434 |
| 1 | 1.0 | 0.0280 | 0.0717 | 0.0703 | 0.0284 |
| 2 | 0.0 | 0.2194 | 0.2801 | 0.2715 | 0.2178 |
| 2 | 0.25 | 0.1111 | 0.1718 | 0.1663 | 0.1094 |
| 2 | 0.5 | 0.0809 | 0.1416 | 0.1372 | 0.0800 |
| 2 | 1.0 | 0.0513 | 0.1120 | 0.1086 | 0.0527 |
| 3 | 0.0 | 0.2648 | 0.3391 | 0.3258 | 0.2624 |
| 3 | 0.25 | 0.1505 | 0.2250 | 0.2158 | 0.1483 |
| 3 | 0.5 | 0.1153 | 0.1897 | 0.1823 | 0.1141 |
| 3 | 1.0 | 0.0733 | 0.1477 | 0.1420 | 0.0743 |
| 5 | 0.0 | 0.3464 | 0.4440 | 0.4222 | 0.3429 |
| 5 | 0.25 | 0.2265 | 0.3242 | 0.3077 | 0.2234 |
| 5 | 0.5 | 0.1748 | 0.2725 | 0.2589 | 0.1726 |
| 5 | 1.0 | 0.1067 | 0.2044 | 0.1939 | 0.1095 |
<!-- AUTOGEN:turnover_bases END -->

| basis | definition | which mode charges it |
|---|---|---|
| `target_change_only` | `Σ\|Δ target\|` — drift assumed free | **target-weight** (canon) |
| `upper_bound_unnetted` | `Σ\|Δ target\| + Σ\|drift\|` — a triangle-inequality **UPPER BOUND** on the single combined trade, per `drift_retrim_turnover`'s own docstring | **drift-inclusive** |
| `netted_combined` | `Σ\|Δ target − drift\|` — the one combined trade a desk actually sends, still on the target-weight book | reported as a diagnostic only; charged by no mode |
| `hold_shares_actual` | `Σ\|w_new − w_drifted\|` — the actual trade at the decision, on the hold-shares book's **own** drifted weights | **hold-shares** |

**Consequence, enforced in code:** `cost_drag = gross − net(fee)` is denominated in whichever
basis its mode charges, so a **cross-mode Δcost_drag subtracts two numbers that were never
measured the same way.** `band_verdict` refuses that comparison
(`cross_mode_cost_drag_comparison: FORBIDDEN`) and computes the cost-only reading **within** each
mode, against that mode's own H=1/θ=0 cell. Cycle 1 leaned the band's cost-only reading on a
cross-mode difference; that reading is withdrawn.

Note the ordering the table makes visible and cycle 1 did not.
`target_change_only < netted_combined < upper_bound_unnetted` holds at **every** cell; the last
two are on the same book and their inequality is the triangle bound `drift_retrim_turnover`
proves pointwise. `hold_shares_actual` sits **below** `target_change_only` at θ ∈ {0, 0.25, 0.5}
— a share-carrying book does not pay to correct drift at all, it only trades the signal change —
but **crosses above it at θ = 1.0 at all four H** (e.g. H=1: 0.0284 vs 0.0280; H=5: 0.1095 vs
0.1067). That crossover is a real property, not noise: at high hysteresis flips are rare, so the
carried weights drift much further from target between trades, and when a flip finally comes the
trade from the *drifted* weight is larger than the target-to-target change. A share-carrying book
is therefore not uniformly cheaper — it is cheaper when it trades often and dearer when it
trades rarely.

## The hold-shares book is not a gross-1 book (disclosure)

A share-carrying book that does not pay to re-trim drift **cannot** honour
`StrategyConfig.gross_cap = 1.0` — un-corrected drift is precisely the mechanism. So this mode
trades a book whose leverage varies, which is **not** a like-for-like comparison against the
other two modes. Measured, never assumed (PRIMARY window):

<!-- AUTOGEN:hold_shares_exposure BEGIN -->
| H | theta | Σ\|w\| mean | Σ\|w\| min | Σ\|w\| max | equity final | equity min |
|---|---|---|---|---|---|---|
| 1 | 0.0 | 0.981 | 0.357 | 1.387 | 15.100 | 0.932 |
| 1 | 0.25 | 0.968 | 0.362 | 1.394 | 25.773 | 0.894 |
| 1 | 0.5 | 0.969 | 0.372 | 1.555 | 12.661 | 0.876 |
| 1 | 1.0 | 0.993 | 0.347 | 1.677 | 2.965 | 0.846 |
| 2 | 0.0 | 0.974 | 0.358 | 1.401 | 24.989 | 0.978 |
| 2 | 0.25 | 0.967 | 0.368 | 1.433 | 26.698 | 0.928 |
| 2 | 0.5 | 0.969 | 0.372 | 1.484 | 12.158 | 0.910 |
| 2 | 1.0 | 1.014 | 0.358 | 1.791 | 2.221 | 0.827 |
| 3 | 0.0 | 0.972 | 0.409 | 1.416 | 24.695 | 1.000 |
| 3 | 0.25 | 0.962 | 0.405 | 1.421 | 23.091 | 1.000 |
| 3 | 0.5 | 0.961 | 0.373 | 1.435 | 11.399 | 1.000 |
| 3 | 1.0 | 1.010 | 0.353 | 1.660 | 3.158 | 0.917 |
| 5 | 0.0 | 0.968 | 0.407 | 1.308 | 20.853 | 0.920 |
| 5 | 0.25 | 0.963 | 0.440 | 1.371 | 16.982 | 0.975 |
| 5 | 0.5 | 0.967 | 0.420 | 1.514 | 10.257 | 0.990 |
| 5 | 1.0 | 1.032 | 0.386 | 2.099 | 2.816 | 1.000 |
<!-- AUTOGEN:hold_shares_exposure END -->

Gross exposure runs **0.35–2.10** across the PRIMARY grid above (and **0.2865–2.9185** once the
other two windows are included) against the canon book's exact 1.0, and it is
above 1.0 on **52%** of decisions at the frozen (H=1, θ=0) cell. The drift is worst exactly where
trading is rarest: at θ=1.0 the mean exposure exceeds 1.0 (up to 1.032 at H=5) and the maximum
reaches **2.099** — i.e. the book runs **2.1× the canon leverage** at its worst decision — for
the same reason the cost basis crosses over there.

### De-levering the hold-shares book: what it settles and what it does not (BLOCKER-E2A)

> **Cycle 2 wrote here that "part of the hold-shares mode's Sharpe advantage is a leverage/timing
> effect, not an accounting one", that this pass "did not separate the two", and that isolating
> them "needs a gross-renormalised hold-shares variant, which is a new arm, not a fix".** The
> "new arm" half was wrong — `hold_shares_walk` already returns the weights actually held.
> **Cycle 3 then over-corrected**, writing that the measurement "does not leave the question
> open — it closes it against the mode". It does not close it. **The verdict is UNRESOLVED**, and
> the rest of this section says why.

**What the comparator is.** A decision's simple portfolio return is *linear* in the weight vector,
`ret[k] = Σᵢ w_heldᵢ·(P_new/P_old − 1)`, so dividing it by that decision's own gross exposure
`G[k] = Σᵢ|w_heldᵢ|` gives exactly the return this book would have realised over **that decision**
at gross 1. The rescaling of the *series* is exact, and it is **not** a uniform rescale — which
would leave Sharpe untouched — for exactly the reason that makes it worth doing: `G` varies
decision to decision (0.35–2.10 on PRIMARY; 0.2865–2.9185 across all three windows) and its
variation is *correlated with returns*, because a winning book's notional grows before the next
decision gets to trade it. That correlation is the leverage/timing component being removed.

**What the comparator is NOT — and this is the correction.** It is **not** "the same book run at
gross 1". It rescales a *realised return*; it never renormalises the **holdings**, so the drifted
weights carried into decision k+1 are unchanged. A book actually constrained to `gross_cap = 1.0`
renormalises its holdings each decision, and its retrim rule then mixes an **unscaled** target with
a **scaled** drift — proportionality breaks, and its k+1 position path diverges from this proxy's.
The two are different objects, and they answer differently.

<!-- AUTOGEN:leverage_renorm BEGIN -->
| H | θ | mean Σ\|w\| | HS gross RAW | HS gross RENORM | target-weight gross | RAW − TW | **RENORM − TW** | HS net21 RAW | HS net21 RENORM | TW net21 | RENORM − TW (net21) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.0 | 0.981 | +1.3132 | +1.0013 | +1.0339 | +0.2794 | **-0.0326** | +1.1025 | +0.7928 | +0.8358 | -0.0429 |
| 1 | 0.25 | 0.968 | +1.5178 | +1.2326 | +1.2476 | +0.2702 | **-0.0150** | +1.4318 | +1.1480 | +1.1657 | -0.0177 |
| 1 | 0.5 | 0.969 | +1.2355 | +1.0305 | +1.0112 | +0.2242 | **+0.0192** | +1.1775 | +0.9723 | +0.9548 | +0.0175 |
| 1 | 1.0 | 0.993 | +0.6895 | +0.5519 | +0.5576 | +0.1319 | **-0.0057** | +0.6545 | +0.5148 | +0.5211 | -0.0063 |
| 2 | 0.0 | 0.974 | +1.5228 | +1.2035 | +1.2305 | +0.2923 | **-0.0271** | +1.3709 | +1.0542 | +1.0910 | -0.0368 |
| 2 | 0.25 | 0.967 | +1.5372 | +1.2593 | +1.2642 | +0.2730 | **-0.0049** | +1.4616 | +1.1849 | +1.1930 | -0.0081 |
| 2 | 0.5 | 0.969 | +1.2179 | +0.9946 | +0.9887 | +0.2291 | **+0.0059** | +1.1640 | +0.9405 | +0.9366 | +0.0038 |
| 2 | 1.0 | 1.014 | +0.5822 | +0.4229 | +0.4575 | +0.1247 | **-0.0346** | +0.5490 | +0.3869 | +0.4235 | -0.0365 |
| 3 | 0.0 | 0.972 | +1.5261 | +1.3212 | +1.3315 | +0.1946 | **-0.0104** | +1.4025 | +1.1966 | +1.2113 | -0.0147 |
| 3 | 0.25 | 0.962 | +1.4842 | +1.3163 | +1.2936 | +0.1906 | **+0.0227** | +1.4153 | +1.2471 | +1.2258 | +0.0212 |
| 3 | 0.5 | 0.961 | +1.1988 | +1.0457 | +1.0137 | +0.1851 | **+0.0320** | +1.1465 | +0.9932 | +0.9623 | +0.0309 |
| 3 | 1.0 | 1.010 | +0.6914 | +0.5242 | +0.5305 | +0.1609 | **-0.0062** | +0.6614 | +0.4908 | +0.4969 | -0.0060 |
| 5 | 0.0 | 0.968 | +1.3849 | +1.1839 | +1.2014 | +0.1835 | **-0.0175** | +1.2939 | +1.0894 | +1.1090 | -0.0196 |
| 5 | 0.25 | 0.963 | +1.3121 | +1.1468 | +1.1224 | +0.1897 | **+0.0243** | +1.2528 | +1.0848 | +1.0618 | +0.0230 |
| 5 | 0.5 | 0.967 | +1.1194 | +0.9259 | +0.9004 | +0.2189 | **+0.0255** | +1.0742 | +0.8787 | +0.8545 | +0.0242 |
| 5 | 1.0 | 1.032 | +0.6393 | +0.5086 | +0.5380 | +0.1012 | **-0.0294** | +0.6136 | +0.4791 | +0.5095 | -0.0304 |

**Before the rescaling** the hold-shares gross advantage over target-weight is +0.1012…+0.2923 and is **positive in all 16 cells**: `true`. **After it** the residue is -0.0346…+0.0320, with **no consistent sign under THIS construction** (6 of 16 cells positive; consistent-sign check `false`).

**The sign pattern is CONSTRUCTION-DEPENDENT, so the residue is UNRESOLVED, not zero.** The column above is the proxy `ret/G` — an exact rescaling of the *realised return series*, holdings untouched. A cycle-3 estimator lens built the other construction, a book genuinely constrained to `gross_cap=1.0` that renormalises its **holdings** each decision, and got the opposite pattern (+0.0039 .. +0.0571, 16 of 16 cells positive (consistent POSITIVE sign)). Neither clears zero. Paired 95% CIs at the frozen cell — raw +0.2794 [-0.088, +0.638], proxy -0.0326 [-0.282, +0.214], constrained gross-1 +0.0039 [-0.186, +0.204] — **all straddle zero**. Those figures are TRANSCRIBED from the lens; this pass did not rebuild the constrained book.

**BEAR22 counter-check** (the finding's own disconfirming test): target-weight +1.1595, hold-shares RAW +0.5576, hold-shares RESCALED +0.3845 — a de-levered gap of **-0.7750**. De-levering does **not** rescue hold-shares in BEAR22, so CORRECTION 2's BEAR22 direction is a genuine compositional effect and **survives**; the full-window *advantage* is left **UNRESOLVED**, not killed.
<!-- AUTOGEN:leverage_renorm END -->

**Reading, stated as bluntly as the numbers permit.** Before the rescaling the hold-shares book
beats target-weight on gross Sharpe in **all 16** (H, θ) cells, by +0.1012 to +0.2923 — a
consistent, large-looking edge. After it the residue runs −0.0346 to +0.0320 (6 of 16 positive).
The headline cell (H=2, θ=0.25) goes from +0.2730 to **−0.0049**; the frozen-θ cell (H=1, θ=0)
from +0.2794 to **−0.0326**. The net@21 columns say the same thing (raw +0.1041…+0.2799, rescaled
−0.0429…+0.0309), and the gross and net deltas track each other closely, so **costs are not
driving this either.** De-levering removes roughly **85%** of the advantage.

**But 85% removed is not 100%, and what remains is UNRESOLVED — not zero, and not shown to be
leverage.** Two things stop the stronger claim:

1. **The residue's sign is construction-dependent.** The proxy above gives −0.0346…+0.0320,
   6 of 16 positive. The genuinely `gross_cap=1.0`-constrained book — same walk, holdings
   renormalised after the retrim mix, built and validated by a cycle-3 estimator lens
   (Σ|w| = 1 ± 4e-16, `r = Σ wᵢrᵢ` to 4.4e-16, identical asset support) — gives
   **+0.0039…+0.0571, positive in 16 of 16 cells**. Opposite pattern, same underlying book.
   *Provenance, stated exactly:* the **proxy** column above was reproduced independently by the
   orchestrator (−0.0346…+0.0320, 6 of 16 positive). The **constrained-book** column was not:
   it rests on that one lens's construction, and neither the orchestrator nor this closing pass
   rebuilt it. Read it as a demonstration that the sign is construction-sensitive, not as a
   second independently-verified measurement.
2. **Neither construction clears zero, and neither does the raw advantage.** Paired 95% CIs at
   the frozen cell: raw **+0.2794 [−0.088, +0.638]**; proxy **−0.0326 [−0.282, +0.214]**;
   constrained gross-1 **+0.0039 [−0.186, +0.204]**. All three straddle zero.

**So the honest verdict is the one this artifact's own "What could not be completed" section
already reached: de-levering removes ~85% of the advantage and leaves a residue an order of
magnitude inside this design's SE. The carry is UNRESOLVED, not measured-as-zero and not refuted.**
"No consistent sign" is a property of the shipped proxy, not a property of the accounting mode.

What *is* established, and does not depend on the construction: the hold-shares book runs at an
exposure the canon book is not permitted to carry (Σ|w| above 1.0 on 52% of decisions, up to
2.099), it structurally cannot honour `StrategyConfig.gross_cap = 1.0`, and most of its apparent
edge goes away when that exposure is taken out. It is therefore **not a candidate for promotion**
— but for want of power, not because it has been refuted. A planner who wants the leverage is
buying leverage and should price it as leverage, against a book not permitted to carry it.

**The denomination gap, stated in the gate's own quantity (D81(c)).** `renormalised_to_gross1`
emits gross + net8/12/21 and **no borrow column**, while every band, uplift, significance and
candidate statement on this page is denominated in `net21_borrow_low`. So everything above is
argued in **gross and net21, not in the gate quantity** — a real defect in how the conclusion was
demonstrated. A cycle-3 flow verifier re-denominated it properly (weights `held_w/G`, turnover
`turn/G`, borrow charged on the **de-levered** short notional) and got renorm−TW of
**−0.0380…+0.0497 with 9 of 16 positive** — the same absence of a consistent sign, the same
order-of-magnitude collapse. **The conclusion holds in the gate quantity; it was simply never
demonstrated there.** Note also that the *raw* advantage is **larger** in the gate quantity
(+0.2282…+0.3260 against the +0.1012…+0.2923 quoted on gross), because the hold-shares book carries
less short notional and so pays less borrow. Adding the borrow column is new computation and is
**not** done here; it is rowed as follow-up work below.

**Bounding the blast radius — what this does NOT kill.** The BEAR22 counter-check in the table
above is the finding's own disconfirming test and it comes back the other way: de-levered
hold-shares in BEAR22 is +0.3845 against target-weight's +1.1595, a gap of **−0.7750** — worse,
not better, than the levered book's −0.6019. **CORRECTION 2's BEAR22 direction therefore survives
as a genuine compositional effect**, and it survives *strengthened*, not merely intact.

The two windows are **not the same mechanism read twice**, and it is worth being precise about
the asymmetry rather than reaching for a symmetric story. In the full window the exposure drift
accounts for ~85% of the effect — remove it and what is left cannot be told from zero. In BEAR22
the exposure drift was mildly *cushioning* a deficit that comes from somewhere else entirely
(mostly the compounding convention — the only attribution component below that clears its own
noise floor), and that deficit **survives** the drift's removal. So removing the leverage leaves
the full-window claim unresolved while reinforcing the BEAR22 one; a reading that had drifting
exposure "helping in a rising window and hurting in a bear window" would be tidier and would not
match these numbers.

**Cost caveat, disclosed rather than buried — and its direction corrected.** The GROSS columns
above involve no cost model at all and are an exact rescaling of the realised series. The NET
columns charge the *levered* book's own turnover against the *de-levered* return series — a book
genuinely run at gross 1 would trade different notionals — so they are an approximation. Cycle 3
stated its direction as a universal: *"mean Σ|w| < 1 at every cell here, so the net columns flatter
the renormalised arm … the conclusion survives its own most favourable treatment."* **That
universal is false, and this page's own exposure table two sections above prints the
counterexamples**: mean Σ|w| = **1.0142** (H=2, θ=1.0), **1.0104** (H=3, θ=1.0) and **1.0316**
(H=5, θ=1.0). Recomputed per cell, the approximation **penalises** rather than flatters in **4**
cells — those three, plus H=1, θ=1.0 at mean 0.9931, because Sharpe depends on `G`'s whole
*distribution*, not on its mean. Corrected statement: **the approximation flatters this arm at 12
of 16 cells and penalises it at 4**, by at most 0.002 Sharpe (max RENORM−TW +0.0309 → +0.0303, min
−0.0429 → −0.0442), so nothing flips; the gross-only comparison carries the reading regardless.
Note that the prose gate below **cannot** catch an error of this class: it matches numeric
literals, and `1` matches trivially.

## Turnover decomposition (target-weight accounting)

**Definitions implemented** (per-asset, then summed/averaged per decision):
- **flip** — asset present at both t−H and t, sign flips: `Σ|Δpos_i|` over those assets.
- **renormalisation** — asset present at both, sign unchanged: `Σ|Δpos_i|` over those (pure
  weight churn from `N` changing while the asset itself never left the book).
- **entry_exit** (the brief's allowed "named fourth term") — asset present at exactly one of
  t−H, t. This is the *direct* turnover of the asset entering/leaving, as distinct from the
  *indirect* renorm pressure it puts on everyone else's weight (already inside the renorm bucket,
  since those other assets ARE present at both times).
- **drift** — `drift_retrim_turnover`'s own term (920, reused unmodified).

**Adjudication of the brief's "the three must sum to the charged turnover" instruction**: the
brief names three components (flip/renorm/drift) but drift is not part of the SAME charged
quantity as flip+renorm under target-weight accounting. Rather than force a 3-way split that
cannot close, this pass reports **two exact closing identities**:

    flip + renorm + entry_exit  ==  target-weight charged turnover              (exact, every cell)
    (flip + renorm + entry_exit) + drift  ==  drift-inclusive charged turnover   (exact, every cell)

<!-- AUTOGEN:decomposition BEGIN -->
| H | theta | flip | renorm | entry_exit | drift | flip+renorm+entry_exit | measured turn (target-weight) | closes |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.0 | 0.1537 | 0.0012 | 0.0012 | 0.0437 | 0.156087 | 0.156087 | yes |
| 1 | 0.25 | 0.0618 | 0.0012 | 0.0012 | 0.0438 | 0.064208 | 0.064208 | yes |
| 1 | 0.5 | 0.0414 | 0.0012 | 0.0012 | 0.0438 | 0.043804 | 0.043804 | yes |
| 1 | 1.0 | 0.0255 | 0.0012 | 0.0012 | 0.0438 | 0.027976 | 0.027976 | yes |
| 2 | 0.0 | 0.2147 | 0.0022 | 0.0025 | 0.0606 | 0.219428 | 0.219428 | yes |
| 2 | 0.25 | 0.1063 | 0.0023 | 0.0025 | 0.0607 | 0.111122 | 0.111122 | yes |
| 2 | 0.5 | 0.0761 | 0.0024 | 0.0025 | 0.0607 | 0.080876 | 0.080876 | yes |
| 2 | 1.0 | 0.0465 | 0.0024 | 0.0025 | 0.0607 | 0.051322 | 0.051322 | yes |
| 3 | 0.0 | 0.2578 | 0.0033 | 0.0037 | 0.0743 | 0.264795 | 0.264795 | yes |
| 3 | 0.25 | 0.1434 | 0.0034 | 0.0037 | 0.0744 | 0.150534 | 0.150534 | yes |
| 3 | 0.5 | 0.1082 | 0.0035 | 0.0037 | 0.0744 | 0.115302 | 0.115302 | yes |
| 3 | 1.0 | 0.0661 | 0.0035 | 0.0037 | 0.0744 | 0.073316 | 0.073316 | yes |
| 5 | 0.0 | 0.3351 | 0.0052 | 0.0061 | 0.0975 | 0.346441 | 0.346441 | yes |
| 5 | 0.25 | 0.2149 | 0.0055 | 0.0061 | 0.0977 | 0.226503 | 0.226503 | yes |
| 5 | 0.5 | 0.1632 | 0.0055 | 0.0061 | 0.0977 | 0.174831 | 0.174831 | yes |
| 5 | 1.0 | 0.0949 | 0.0057 | 0.0061 | 0.0977 | 0.106713 | 0.106713 | yes |
<!-- AUTOGEN:decomposition END -->

**Reading**: flip dominates the charged turnover at every H (renorm + entry_exit together are
1–2 orders of magnitude smaller — the 363-asset universe's present-count churns very little day
to day). θ's whole effect is on the flip term; renorm/entry_exit are ~θ-invariant, as expected
(they never depended on the trend statistic). Drift is comparable in size to flip at θ≥0.5 — at
high hysteresis the UNCHARGED drift-retrim cost becomes the dominant turnover component
target-weight accounting is *not* paying, which is exactly what drift-inclusive charges and what
hold-shares avoids by not correcting drift at all.

## G1 — no-op bit-identity (PASS, and unchanged by the fix)

At θ=0, H=1, accounting=target-weight, borrow=none, on the PRIMARY window:

<!-- AUTOGEN:g1 BEGIN -->
| quantity | harness | pinned reference | match |
|---|---|---|---|
| n | 1704 | 1704 | exact |
| gross Sharpe | 1.0338583098255947 | 1.0338583098255947 | bit-identical |
| turn (mean/decision) | 0.1560866077894317 | 0.1560866077894317 | bit-identical |
| net@21 Sharpe | 0.8357541266542514 | 0.8357541266542514 | bit-identical |
<!-- AUTOGEN:g1 END -->

**These are byte-for-byte the cycle-1 numbers** — the hold-shares fix touches only the
hold-shares branch, so the no-op identity is untouched, which is itself the check that the fix
did not leak into the canon path. **MINOR-C2 fix**: the identity is now asserted **element-wise**
on the full 1704-length gross, turn AND net@21 series (`np.array_equal`), not on three scalar
reductions — cycle 1's test could not have ruled out a compensating per-decision error. Also
checked element-wise at H∈{2,3,5}, θ=0 against a hand-built `regime_series` call on the canon
`positions_from_config` output. **Non-vacuity**: a one-cell perturbation moves the mean turnover
away from canon. `test_919_grid.py::test_g1_noop_bit_identical`.

## CORRECTION 1 — the analytic anchor's formula AND denomination (CONFIRMED; comparand corrected)

Re-derived independently: for jointly-Gaussian zero-mean trend statistics at correlation
ρ=(L−H)/L, P(sign flip)=arccos(ρ)/π; a flip changes `sign_i` by 2 and `pos_i=sign_i/N` on an
equal-weight gross-1 book, so `E Σ_i|Δpos_i| = 2·P(flip) = (2/π)·arccos(ρ)` **per decision** —
dividing by H gives the corrected per-day model `turnover_per_day(H) = (2/π)·arccos((L−H)/L)/H`.

**MINOR-E3/O2 fix**: the model describes **only sign flips**, so the correct comparand is the
decomposition's **flip** term, not total turnover. Cycle 1 compared against total. Both are now
reported; **the flip-only ratio is the load-bearing one**:

<!-- AUTOGEN:correction1 BEGIN -->
| H | n | turn/day measured | flip/day measured | model (corrected) | ratio vs TOTAL turn | ratio vs FLIP only | model (task as-written) | ratio vs uncorrected |
|---|---|---|---|---|---|---|---|---|
| 1 | 1704 | 0.1561 | 0.1537 | 0.1648 | 0.9469 | **0.9325** | 0.0824 | 1.894 |
| 2 | 852 | 0.1097 | 0.1074 | 0.1169 | 0.9386 | **0.9186** | 0.1169 | 0.939 |
| 3 | 568 | 0.0883 | 0.0859 | 0.0957 | 0.9222 | **0.8980** | 0.1436 | 0.615 |
| 5 | 340 | 0.0693 | 0.0670 | 0.0746 | 0.9291 | **0.8988** | 0.1864 | 0.372 |
<!-- AUTOGEN:correction1 END -->

**CONFIRMED, independently.** The corrected formula reproduces the brief's quoted
0.165/0.117/0.0957/0.0746 per day exactly; the task file's as-written `arccos(ρ)/π` is right only
at H=2 (algebraic coincidence: `(2/π)arccos(ρ)/2 == arccos(ρ)/π`) and off by 1.9×/1.6×/2.7× at
H=1/3/5.

**The residual, on the CORRECT comparand.** Against flip only, measured turnover runs 6.7%–10.2%
below the model, and — unlike the total-turn comparison, which looks flat at 0.92–0.95 — the
flip-only ratio is **not flat**: it declines ~3.5 pp from H=1 (0.9325) to H=3 (0.8980) and then
plateaus (H=5: 0.8988). **This refutes cycle 1's first candidate mechanism.** Cycle 1 proposed
cross-sectional correlation via the renorm channel; but renorm + entry_exit together are only
0.0024 at H=1 (~1.5% of turnover) and are **additive**, so they cannot close a gap that lives
*inside* the flip term. Nor is a fixed-scale distributional artifact consistent with an
H-dependent ratio. The surviving reading is **genuine trend persistence beyond the ρ=(L−H)/L
assumption**: the model derives ρ from a pure random walk, and a persistent trend statistic
flips less often than a random walk does, by an amount that grows with the horizon — which is the
shape observed. Not isolated quantitatively here, so it is recorded as a **narrowed** residual,
not a resolved cause.

**Non-load-bearing**: the anchor gates G1/G5 only; it never feeds a Sharpe or a cost.
**G5 non-vacuity**: the check band [0.85, 0.98] REJECTS the uncorrected formula at 3/4 H values
(only H=2 passes, by the coincidence above). `test_919_grid.py::test_g5_anchor_nonvacuous`.

## CORRECTION 2 — the hold-shares BEAR22 attribution (RE-DERIVED; direction CONFIRMED, split UNRESOLVED)

> **Cycle 1's CONFIRMED status on this correction was RETRACTED.** That pass compared
> `gross_simple_series` against `gross_log` on the *same* target-weight positions — a
> compounding-convention swap — and never carried a share, so it did not test the claim at all.
> Independently verified: on the frozen positions, BEAR22, n=416, H=1, `gross_log` =
> 1.1595401838210024 and `gross_simple` = 0.5903959074315116, bit-identical to cycle 1's
> target_weight/hold_shares pair. Everything below is measured through the new
> `hold_shares_walk`.

<!-- AUTOGEN:correction2 BEGIN -->
| window | n | target-weight gross | hold-shares gross | Δ (hold − target) |
|---|---|---|---|---|
| **BEAR22** (2021-11-10..2022-12-31) | 416 | +1.1595 | +0.5576 | **-0.6019** |
| PRIMARY full (2021-01-01..2025-09-01) | 1704 | +1.0339 | +1.3132 | **+0.2794** |

Hold-shares book gross exposure Σ|w| in BEAR22: mean 0.869, min 0.357, max 1.191 (a share-carrying book is NOT pinned at gross 1 — measured, not assumed).
Direction confirms the 830_corridor mechanism in BEAR22: **True**. Sign reverses outside BEAR22: **True**.
<!-- AUTOGEN:correction2 END -->

**The direction claim is CONFIRMED under a real share-carrying path.** Hold-shares gross in
BEAR22 is +0.5576 against target-weight's +1.1595 — it **hurts by −0.6019**, the direction
`830_corridor/SPEC.md:95-98` and LEDGER row `830/E0` name, for the mechanism they name: a winning
short's notional `s·P` shrinks as price falls, so a fixed-shares book de-leverages into its own
winning short while a constant-dollar book rebalances back up to full weight and keeps compounding
the gain. The task file's own figures ("hold-shares BEAR22 gross +1.218 vs +0.225") were
**swapped**; +1.218 is the constant-dollar side, and this pass's constant-dollar BEAR22 gross
(+1.1595) reproduces it closely on a different panel.

**But the ATTRIBUTION of that −0.60 is not what it looks like, and this refines the finding
against cycle 1 in BOTH directions.** A carried book differs from the canon book in two ways at
once: it realises a weighted **SIMPLE** return where the canon gross is a weighted **LOG** return
(a pure compounding convention — TASK_920's separate leg 2, and the only thing cycle 1 actually
measured), *and* it holds **drifted** weights. Splitting the gap:

<!-- AUTOGEN:correction2_attribution BEGIN -->
| window | canon gross (log-sum on target) | + compounding convention → simple-sum on the SAME target | + share carry → the carried book | Δ from CONVENTION | Δ from SHARE CARRY |
|---|---|---|---|---|---|
| **BEAR22** | +1.1595 | +0.5904 | +0.5576 | **-0.5691** | **-0.0328** |
| PRIMARY full | +1.0339 | +1.0290 | +1.3132 | **-0.0049** | **+0.2842** |

**Each component with its own 95% PAIRED block-bootstrap CI (MAJOR-E2B).** Both series in each difference live on the same decision rows, so the bootstrap is paired and needs no correlation assumption. A component whose interval contains 0 is **UNRESOLVED at this design's power** — the same standard this pass applies to the ±0.20 band, now applied to its own attribution.

| window | component | point | 95% CI (paired) | bootstrap SD | verdict |
|---|---|---|---|---|---|
| **BEAR22** | Δ from CONVENTION | **-0.5691** | [-0.8075, -0.3217] | 0.1272 | **RESOLVED** |
| **BEAR22** | Δ from SHARE CARRY | **-0.0328** | [-0.6065, +0.5132] | 0.2886 | **UNRESOLVED** |
| PRIMARY full | Δ from CONVENTION | **-0.0049** | [-0.1515, +0.1465] | 0.0760 | **UNRESOLVED** |
| PRIMARY full | Δ from SHARE CARRY | **+0.2842** | [-0.0364, +0.6237] | 0.1726 | **UNRESOLVED** |
<!-- AUTOGEN:correction2_attribution END -->

**In BEAR22, −0.5691 of the −0.6019 gap is the compounding convention and −0.0328 is the share
carry itself.** The convention component is the only one of the four that clears its own noise
floor: its paired 95% CI is [−0.8075, −0.3217], excluding zero. What dominates BEAR22 is therefore
an accounting fact rather than the 830 mechanism: `exp(x)−1 ≥ x`, so a *short* position's simple
return is worse than its log return, and BEAR22's book is heavily short, so the log-sum convention
**flatters a short book** and the simple-return convention (which is what a real book realises)
does not. Cycle 1's number was approximately right for approximately the wrong reason.

**The rest of the split is UNRESOLVED, and cycle 3 says so rather than reporting it (MAJOR-E2B).**
Cycle 2 shipped all four components as bare point estimates while the *same* pass refused the
±0.20 band question for want of power — an internal double standard, since BEAR22's −0.0328 share
carry is **11.4%** of the paired bootstrap SD (**0.2886**) that cycle 3 derives for that very
component. (Cycle 3 wrote "about 3.5% of the single-cell Sharpe SE this pass derives at that n";
the closing pass corrects the denominator, because this pass derives per-cell SEs **only** on
PRIMARY — n=1704/852/568/340 — and derives **no** SE at BEAR22's n=416. Quoting a denominator the
artifact does not contain made the double standard look ~3× worse than the numbers support; the
double standard is real either way.) Given error bars:

- BEAR22 **share carry −0.0328**, CI [−0.6065, +0.5132] → **UNRESOLVED**. The 830 mechanism is
  not distinguishable from noise *as a magnitude* on this window at this n. Cycle 2's "it is
  small — a twentieth of the gap" was reading a number the design cannot resolve.
- PRIMARY-full **convention −0.0049**, CI [−0.1515, +0.1465] → **UNRESOLVED**.
- PRIMARY-full **share carry +0.2842**, CI [−0.0364, +0.6237] → **UNRESOLVED** — and separately
  bounded by the de-levering above, which removes ~85% of the full-window carry advantage. Cycle 3
  wrote that this "kills it as a mechanism"; it does not. The residue's sign is
  construction-dependent and no construction's CI clears zero, so this component is **unresolved
  twice over**, not refuted.

**What survives, and why it does not depend on the split.** The *direction* of the BEAR22 gap —
hold-shares hurts there — is supported independently of any attribution: by the pre-existing
record (`830_corridor/SPEC.md:95-98`, LEDGER `830/E0`) and by the **de-levered** counter-check,
which strips the exposure and finds the book *still* loses in BEAR22 (+0.3845 vs +1.1595) by
**−0.7750**, paired 95% CI **[−1.109, −0.452]**, stable across 6 bootstrap seeds. Direction:
**CONFIRMED**. Split: **carries error bars, and three of its four components do not clear zero**.

**And the direction claim does NOT rest on the total gap, which is seed-fragile (closing pass).**
Cycle 3 listed "the sign of the total gap" among the direction's supports without ever error-barring
it. Error-barred, the total gap **−0.6019** has paired 95% CI **[−1.158, −0.037]** at the shipped
seed but **[−1.115, +0.009]** at seed 3 — i.e. it **straddles zero on another draw**. (Both CIs
transcribed from the cycle-3 estimator lens; not recomputed here.) The un-error-barred total is
therefore struck from the support list above: the direction claim stands on the renormalised leg,
whose CI is stable across seeds, plus the 830 prior. Keeping the direction while downgrading the
split is defensible — but on the renormalised leg, not on the total.

**The sign-reversal reading is likewise downgraded.** Cycle 2 called the accounting choice a
"regime bet" because the carry component flips sign between BEAR22 (−0.0328) and the full window
(+0.2842). Both of those components are UNRESOLVED, and the full-window one is additionally
~85% attributable to leverage, so the reversal is stated here as a **hypothesis consistent with the point
estimates**, not a finding: un-corrected drift lets a winning long's notional grow as well as
letting a winning short's shrink, so a rising window would favour the carry and a bear window
punish it. Testing that needs a design with power this one does not have.

## CORRECTION 3 — the pre-registered band needs a gross/cost split (CONFIRMED)

**H=3 vs H=1, target-weight, θ=0, net21, PRIMARY window:**

<!-- AUTOGEN:correction3 BEGIN -->
| | H=1 | H=3 | Δ |
|---|---|---|---|
| net@21 Sharpe | +0.8358 | +1.2113 | **+0.3756** |
| gross Sharpe | +1.0339 | +1.3315 | **+0.2977** |
| cost_drag (gross−net21) | 0.1981 | 0.1202 | **-0.0779** (a SAVING) |

Fraction of Δnet that is a gross/signal effect: **79%**; fraction that is a genuine cost saving: **21%**.
<!-- AUTOGEN:correction3 END -->

This independently
reproduces the task file's own anchor ("H=3d survivor +1.35 vs +0.97, ~0.07 of it cost") to
within rounding (0.078 here), on a **different** panel — convergent evidence for the mechanism,
not a repeated number.

**Grid-wide band readings (PRIMARY window), with the cost-only reading now computed WITHIN each
mode (MAJOR-F2) and every row's decidability stated (MAJOR-E1):**

<!-- AUTOGEN:band_primary BEGIN -->
| reading | value | cell | point estimate > +0.20? | DECIDABLE at this design's power? |
|---|---|---|---|---|
| max **Δnet** (the band's literal, unqualified "uplift") | **+0.6774** | H=2, θ=0.25, hold-shares | yes | **NO** |
| max **Δcost_drag saving**, WITHIN `target-weight` (vs that mode's own H=1/θ=0 cell) | +0.1696 | H=5, θ=1.0 | no | **NO** |
| max **Δcost_drag saving**, WITHIN `drift-inclusive` (vs that mode's own H=1/θ=0 cell) | +0.1988 | H=5, θ=1.0 | no | **NO** |
| max **Δcost_drag saving**, WITHIN `hold-shares` (vs that mode's own H=1/θ=0 cell) | +0.1852 | H=5, θ=1.0 | no | **NO** |
<!-- AUTOGEN:band_primary END -->

The band's own text ("max uplift over the frozen rule ≤ +0.20 net@21 ⇒ the family is capped as
cost-only") reads most naturally as the Δnet row. But the H axis is provably not cost-only
(above), so that row conflates a signal/holding-period effect with an execution saving. The rows
that *do* isolate the cost-only component all land **inside** the band on their point estimate —
and all at H=5, θ=1.0, a cell whose Δnet is strongly **negative** (−0.3584 within target-weight).
**No cell in this grid delivers both a large cost saving and a large net uplift from the same
mechanism.**

**And none of it is decidable.** Every row above is a point estimate whose interval straddles
+0.20 at the family's own α. The conflict cycle 1 flagged for the planner is real but *moot at
this power*: the band question cannot be answered either way by this design. That is the verdict,
stated in "The band question" above, and it is the planner's to record.

Same readings on the other two windows:

<details><summary>SECONDARY (superset — not out-of-sample) and DISJOINT (out-of-sample)</summary>

<!-- AUTOGEN:band_secondary BEGIN -->
| reading | value | cell | point estimate > +0.20? | DECIDABLE at this design's power? |
|---|---|---|---|---|
| max **Δnet** (the band's literal, unqualified "uplift") | **+0.6779** | H=3, θ=0.0, hold-shares | yes | **NO** |
| max **Δcost_drag saving**, WITHIN `target-weight` (vs that mode's own H=1/θ=0 cell) | +0.1476 | H=5, θ=1.0 | no | **NO** |
| max **Δcost_drag saving**, WITHIN `drift-inclusive` (vs that mode's own H=1/θ=0 cell) | +0.1720 | H=5, θ=1.0 | no | **NO** |
| max **Δcost_drag saving**, WITHIN `hold-shares` (vs that mode's own H=1/θ=0 cell) | +0.1629 | H=5, θ=1.0 | no | **NO** |
<!-- AUTOGEN:band_secondary END -->

<!-- AUTOGEN:band_disjoint BEGIN -->
| reading | value | cell | point estimate > +0.20? | DECIDABLE at this design's power? |
|---|---|---|---|---|
| max **Δnet** (the band's literal, unqualified "uplift") | **+0.6368** | H=3, θ=0.0, hold-shares | yes | **NO** |
| max **Δcost_drag saving**, WITHIN `target-weight` (vs that mode's own H=1/θ=0 cell) | +0.1231 | H=3, θ=1.0 | no | **NO** |
| max **Δcost_drag saving**, WITHIN `drift-inclusive` (vs that mode's own H=1/θ=0 cell) | +0.1425 | H=5, θ=1.0 | no | **NO** |
| max **Δcost_drag saving**, WITHIN `hold-shares` (vs that mode's own H=1/θ=0 cell) | +0.1415 | H=5, θ=1.0 | no | **NO** |
<!-- AUTOGEN:band_disjoint END -->

</details>

## CORRECTION 4 — θ's flip reduction is much larger than the record claims (MINOR-E4)

The task file (via `lore/harvest_6b32_appendix/tactics_gaps.md:43`, self-labelled "heuristic,
must be measured") claims **"θ = 0.5σ̂ kills 40–50% of flips"** and "~70–75%" at θ=1σ̂. Measured
on the PRIMARY window's flip term:

<!-- AUTOGEN:correction4 BEGIN -->
| theta | flip turn H=1 | reduction vs θ=0 | flip H=2 | reduction | flip H=3 | reduction | flip H=5 | reduction |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.1537 | — | 0.2147 | — | 0.2578 | — | 0.3351 | — |
| 0.25 | 0.0618 | 59.8% | 0.1063 | 50.5% | 0.1434 | 44.4% | 0.2149 | 35.9% |
| 0.5 | 0.0414 | 73.1% | 0.0761 | 64.6% | 0.1082 | 58.0% | 0.1632 | 51.3% |
| 1.0 | 0.0255 | 83.4% | 0.0465 | 78.4% | 0.0661 | 74.4% | 0.0949 | 71.7% |
<!-- AUTOGEN:correction4 END -->

**θ=0.5 removes 73.1% of flip turnover at H=1, not 40–50%** — the record is wrong in the
*permissive* direction, and by a wide margin (θ=0.25 alone already removes 59.8%, more than the
record attributes to twice that band). θ=1.0 removes 83.4%, above the record's "~70–75%".

**Interpretation, which is the point of the correction**: hysteresis is far more aggressive than
the record implies, so θ is **not** the cheap free lunch "40–50%" suggests — a band that removes
three quarters of the flips is cutting deep enough to destroy signal, not just cost. That is
exactly what the grid shows: every θ=1.0 cell has a strongly negative Δnet (target-weight H=1:
−0.3398; H=5: −0.3584) despite having the lowest cost of any cell in the grid. The reduction also
falls monotonically with H (73.1% → 51.3% at θ=0.5), because a longer horizon produces a larger
trend statistic that clears a fixed θ·σ̂_S band more often.

## Section 7 — hour-of-day decision boundary (mechanism implemented; NOT run at panel scale)

**Data-scale facts** (measured): the fixture's `ohlcv` (1-minute) table holds **11 assets**
(8 real + 3 synthetic test fixtures), **2025-09-09..2025-12-12** (474,343 rows) — no BEAR22, no
cross-sectional breadth. `trading_data.db` holds **1,130 assets, 2013-09-10..2026-03-31**
(254,904,234 rows) — the only source wide enough to answer the task's question. A full-timeline
minute pass over that table is "expensive real-data validation," which is SCHEDULED, never polled
inside the factory loop (project invariant).

**What was built** (`hour_of_day.py`): `build_shifted_daily_panel(db_path, asset_ids,
day_boundary_utc, ...)` — a `--day-boundary-utc`-capable daily-close panel built by streaming the
minute table (same O(1)-RAM aggregation pattern as `trader.data.build_daily_panel_from_db`,
attributed) and bucketing by a boundary shifted `day_boundary_utc` hours from UTC midnight.

**Proof of correctness on the 11-asset fixture slice** (`verify_hour0_matches_ohlcv_daily`,
`test_919_grid.py::test_hour_of_day_mechanism`): `day_boundary_utc=0` reproduces `ohlcv_daily`'s
own `agg1m`-sourced rows **exactly** (max abs close diff = 0.0, 143 (asset, day) pairs compared
across the 8 real assets) for every day OUTSIDE a disclosed 5-day trailing window. That trailing
window (2025-12-08..2025-12-12, 4 of 8 assets, 20 (asset,day) pairs) shows small mismatches (max
diff 1909.6 on XBT, ~2% of price) — measured and diagnosed as a **snapshot-staleness signature**
in the fixture itself (the 4 affected assets are exactly the ones whose minute coverage extends
that far) — disclosed explicitly, not silently excluded by widening a tolerance.

**Not run at panel scale.** An hour-of-day trading result from an 11-asset, 3-month, no-BEAR22
slice would not answer the task's question and is not reported as one. Planner: the scheduled
invocation (naming `trading_data.db`, the 363-asset PIT universe via
`trader.data.build_pit_universe`, and `day_boundary_utc=15.0`) is in
`hour_of_day.SCHEDULED_RUN_INVOCATION` and printed by `python hour_of_day.py`.

## How many independent things were actually tested

48 cells rest on **32 distinct position paths**: `target-weight` and `drift-inclusive` share one
path per (H, θ) — they differ only in whether the drift re-trim is *charged* — while `hold-shares`
is now its own path. (Before this fix pass, all three shared one path, i.e. **16** paths behind 48
cells; the estimator lens established this and it is why the accounting axis was not a treatment
arm.) **Bonferroni-over-48 is kept**: it controls FWER without any independence assumption, and
is conservative here precisely because it re-tests each trajectory more than once. A
correlation-aware correction would be less conservative but changes nothing qualitatively — 0/48
clear even the *unadjusted* CI-excludes-band test on the out-of-sample window.

## What the prose gate actually catches (closing pass — cycle 3 over-stated this)

Cycle 3 wrote that "every numeric literal outside the markers must match a value in the artifact",
that "every prose numeric literal is accounted for" (the `--check` success message), and left a
reader to infer that an unflagged prose number has been verified. **It has not.** A cycle-3 flow
verifier swept a last-digit typo over all **144** distinct prose literals in this file — 1170
mutations — and counted rejections. Transcribed, not re-measured here:

| printed precision | mutations caught |
|---|---|
| 0 dp | 61% |
| 1 dp | 43% |
| **2 dp** | **1% — effectively unguarded** |
| 3 dp | 34% |
| 4 dp | 72% |
| **overall** | **596 of 1170 = 50.9%** |

**Cause.** `result_919a.json` holds **3804** distinct absolute values, so a random 2-decimal
literal in [0, 2] round-matches *something* with probability ≈99.5%. The gate has real teeth only
on 4-decimal Sharpes — which is exactly where the cycle-2 judge's demonstrated poison sat, so `G5`'s
non-vacuity proof is honest but narrow.

**Two demonstrated value-keyed holes on real sentences in this file.** The whitelist is keyed by
VALUE, so a whitelisted number is admitted anywhere: substituting the whitelisted `70.0` into
"θ=1.0 removes 83.4%, above the record's '~70–75%'" is **accepted**, and silently inverts
CORRECTION 4's verdict into a self-refuting sentence; substituting the whitelisted `20` into the
share-carry SE sentence is likewise accepted. This is not hypothetical — whitelist entry `3.5`
was already doing duty at a **second, never-audited site** while carrying one written reason. That
entry's audit is recorded in `render_tables.PROSE_NUMBER_WHITELIST` and the second site has been
rewritten.

**A larger, previously undisclosed exemption: `_SCRUBS`.** Before any literal is extracted,
`render_tables._SCRUBS` strips every inline code span *and* any bare 3-digit number followed by a
space, paren, comma or period. That exempts `n=852`, `n=568`, `n=340`, `n=416` and `n=143` **as a
class** — a wider hole than the whitelist, and one no reason string covered.

**Not fixed here.** Keying the match by (value, context) and refusing sub-3-decimal matches is a
redesign — a new solution attempt — and is out of scope for a D76 closing pass. Rowed as follow-up
work below. What the closing pass did do is stop the artifact from claiming a reach the gate does
not have.

## What the closing pass left undone (and why)

The D76 closing-pass exception permits transcription and reframing, **not new computation**. Three
fixes were identified and deliberately **not** made, because each would have required building
something new. They are follow-up work for the planner, not defects hidden in a footnote:

1. **The borrow column on `renormalised_to_gross1`.** The de-levering conclusion is argued in gross
   and net21, not in the pass's own gate quantity `net21_borrow_low` (D81(c)). The flow verifier's
   already-computed re-denominated arm is transcribed above and the conclusion holds there, but
   this harness still does not emit it. Adding it means charging borrow on the de-levered short
   notional — a new computed column.
2. **Rebuilding the `gross_cap=1.0`-constrained book inside this harness.** The construction that
   gives the opposite sign pattern was built by a cycle-3 lens and is transcribed here on that
   lens's authority. Reproducing it in `grid_harness.py` — and giving it its own adversarial pass —
   is a new arm. Until then the leverage question stays UNRESOLVED **and** partly second-hand.
3. **Strengthening the prose gate.** Keying by (value, context) and refusing sub-3-decimal matches
   would close the 2-dp blind spot and the value-keyed whitelist holes. That is a gate redesign.

One further thing this pass **added** rather than removed: the whitelist grew, because transcribing
judge-collected figures puts numbers in the prose that are — by construction — absent from
`result_919a.json`, and this pass is forbidden from computing them into it. Each such entry says so
in its reason. That is a genuine, disclosed weakening of a gate this same pass criticises for being
weaker than advertised.

## Gates G1–G5 — evidence

| Gate | Claim | Evidence |
|---|---|---|
| **G1** (no-op) | θ=0,H=1,target-weight,borrow=none bit-identical to `compute_full_cell` | Section above; asserted **element-wise** on the full series (MINOR-C2), also at H∈{2,3,5}. `test_g1_noop_bit_identical` PASS. |
| **G2** (flag OFF) | `trader/` untouched; selftest 17/17 | `test_g2_no_trader_edits` now **asserts** `git diff HEAD -- trader/`, `git diff --cached -- trader/` and `git status --porcelain trader/` are all empty and pins `git log -1 -- trader/` (MINOR-R1: cycle 1's version imported `trader` and returned a string). `python -m trader selftest` → **17/17 PASS**. |
| **G3** (θ path-dependence) | retained-sign scan carries across DECISION points, resets on presence-mask entry | Planted panel: a single-day countertrend blip strictly BETWEEN two H=3 decisions. A negative-control per-DAY scan DISAGREES with the harness's per-DECISION scan. `test_g3_theta_path_dependence` PASS. Unchanged by this fix pass. |
| **G4** (leak-freedom) | σ̂_S uses rows ≤ t only | ST6-pattern wrecking test: corrupt `logc` rows > t0, assert hysteresis positions AND retained-sign state at decisions ≤ t0 are byte-identical. Extended by this pass to the **hold-shares walk** (`test_g4_hold_shares_leak_free`): wrecking closes > t0 leaves the carried book's returns/turnover/held-weights at decisions ≤ t0 byte-identical. PASS. |
| **G5** (non-vacuity) | every check must FAIL on a planted defect at the mechanism's WEAKEST point | Anchor check: band [0.85,0.98] rejects the uncorrected formula at 3/4 H. G1: one-cell sign perturbation moves mean turnover off canon. **New for the hold-shares path** (`test_g5_hold_shares_poison`): a planted variant that RE-STRIKES shares to the target at every decision — i.e. exactly the cycle-1 defect — is shown to produce different returns/turnover, so the new tests would have FAILED against the old implementation. Proven, not asserted. `test_results_tables_match_json` likewise fails on any hand-edit of a table. **New for the PROSE gate** (`test_prose_number_gate`, cycle 3): the judge's own demonstrated mutation — `0.078` → `9.9999 WRONG NUMBER PLANTED`, planted in prose, outside every marker — is shown in one test to be ACCEPTED by the cycle-2 AUTOGEN-only comparison and REJECTED by the cycle-3 gate. The same test asserts no whitelist entry is dead, so the whitelist cannot be padded to manufacture a pass. **Scope, corrected in the closing pass:** this proves the gate is *non-vacuous on a 4-decimal literal*. It does **not** establish that every prose number is guarded — measured reach is 50.9% overall and ~1% at two decimals. See "What the prose gate actually catches". |

`python -m trader selftest`: **17/17 PASS**, unchanged in count and content (this task makes zero
`trader/` changes).

## Deviations from the brief, disclosed

- **This artifact carries a FOURTH pass past a three-cycle limit, deliberately.** The run's
  `halt_after_failures` is 3 and cycles 1–3 are spent. The closing pass runs under the one named
  exception in `.claude/rules/30-agent-conduct.md` (D76): it replaces framing this same run proved
  defective, it only transcribes evidence judges already collected and confirmed, and it discloses
  the deviation in the banner at the top of this file rather than here. It made **no new solution
  attempt**, gathered **no new evidence**, and built **no new arm or computed column** — the three
  fixes that would have required one are named in "What the closing pass left undone". Every number
  in `result_919a.json` is unchanged by it; only prose, docstrings and JSON *description strings*
  moved, and the artifact was regenerated and re-verified byte-identical across two consecutive
  runs afterwards.
- **The hold-shares re-strike convention is this pass's own choice, not inherited.**
  `true_bh_series` supplies the share walk, the marking convention, the delisting convention and
  the fail-loud refusals, but not *when* to re-strike (it re-forms once per calendar year, which
  makes no sense for a signal-driven book). A hold-shares book that re-trimmed to target every
  decision would have the target-weight position path exactly, i.e. would not be a treatment arm
  — the cycle-1 defect in a different costume. So shares are re-struck **iff the used sign
  changes** (which covers flips, entries, and exits/delistings, since an absent asset's used sign
  is 0). Stated in `hold_shares_walk`'s docstring and here rather than buried.
- **The held-vs-target divergence that rule creates has TWO sources, not the one previously
  named** (MINOR/HUNCH-C2B, cycle 3). The first is the one the docstring and CORRECTION 2 tell:
  **price drift on the held side** — a winning long's notional grows, a winning short's shrinks.
  The second was uncredited: **the TARGET itself drifts on the un-traded side.** The canon target
  is risk-weighted across the assets *present* at each decision, so a same-signed asset's target
  weight moves whenever the sizing denominator shifts because *other* assets enter or leave the
  panel, or their trailing vol moves — and the retrim rule, gated on this asset's own sign alone,
  never re-strikes for a target that moved underneath it. Both sources are inside every measured
  hold-shares number and neither is separated here. **The rule itself is unchanged** — this is a
  disclosure correction, not a mechanism change; a cycle-2 lens raised it as a hunch and could
  not produce a materially different live number for it.
- **The hold-shares mode's borrow charge is denominated on the weights actually HELD**, not on the
  target weights, because that is the short notional the book actually carries. This makes the
  borrow column non-comparable across modes in the same way the fee column is; disclosed with the
  cost bases.
- **The gross-renormalised comparator's NET columns are an approximation; its GROSS columns are
  an exact rescaling of the realised return series.** The rescaling is exact because per-decision
  return is linear in the weights. The net columns charge the *levered* book's own turnover against
  the *de-levered* return series, because re-deriving the trade sizes a genuinely gross-1 book
  would send is a different arm (a leverage-managed book that also pays to correct its exposure).
  Direction, **corrected in the closing pass**: the approximation flatters the renormalised arm at
  **12 of 16** cells and **penalises** it at 4 (H=2/3/5 at θ=1.0, where mean Σ|w| exceeds 1, plus
  H=1 at θ=1.0), by ≤0.002 Sharpe. The conclusion is drawn from the gross-only comparison, which
  involves no cost model at all.
- **The comparator is a rescaling of the return SERIES, not a book re-run under a gross-1
  constraint.** It never renormalises the holdings, so the weights carried into the next decision
  are unchanged; a genuinely `gross_cap=1.0`-constrained book has a different position path from
  decision k+1 onward. The constrained book is **not built here**.
- **The comparator emits no borrow column,** so the de-levering conclusion is argued in gross and
  net21 rather than in this pass's own gate quantity `net21_borrow_low` (D81(c)). Disclosed and
  rowed as follow-up work; the column is **not** added here.
- **A Bonferroni interval on the uplift is a normal approximation** on the bootstrap SD, not an
  empirical quantile: α=0.05/48 puts ~1 of 2000 draws in each tail, which no percentile can
  resolve. Labelled `ci_bonferroni_normal_approx: true` in the artifact.
- **DISJOINT is a post-hoc holdout**, added by this fix pass and not pre-registered. It is
  reported as the only genuinely out-of-sample read available, with that caveat attached.
- The brief's §6 names three decomposition components; this pass reports a **named fourth term**
  (entry_exit) as the brief explicitly permits, producing two exact closing identities instead of
  one forced 3-way split that cannot close.
- Bootstrap CI + Bonferroni p-value computed ONLY at the primary gate quantity, per the brief's
  own instruction that fee tiers/borrow scenarios are "REPORTED, not searched".

## What could not be completed

- **A resolved band verdict** — not by design this time but by *demonstration*: the design's power
  forecloses it. See "The band question". The pre-registered ±0.20 question needs a different
  design, and that is a planner decision.
- **Separating the hold-shares mode's SHARE-CARRY effect from its LEVERAGE-DRIFT effect —
  PARTIALLY done, and NOT resolved.** Cycle 2 called this "a new arm"; it is not, for the leverage
  *component*: the comparator reuses the held weights `hold_shares_walk` already returns and the
  rescaling of the series is exact. Cycle 3 then reported the answer as "the full-window advantage
  is *all* leverage", which the closing pass withdraws. What is established: de-levering removes
  ~85% of the advantage. What is **not**: the sign of the residue, which flips between the shipped
  proxy (−0.0346…+0.0320) and a genuinely `gross_cap=1.0`-constrained book (+0.0039…+0.0571), with
  no construction's paired CI clearing zero. The carry is **UNRESOLVED** — not measured-as-zero
  and not refuted. See "De-levering the hold-shares book: what it settles and what it does not".
- **A pre-registered out-of-sample window.** DISJOINT was added by this fix pass, so it is a
  post-hoc holdout. It answers the same way as everything else (0/48), but it is not the
  pre-registration a promotion claim would need.
- **An hour-of-day trading result** — by design (data-scale invariant); the mechanism is proven
  correct and handed to the planner as a scheduled invocation.
- **A quantitative attribution of Correction 1's residual** to trend persistence specifically. The
  candidate list is now narrowed to one surviving mechanism (the renorm-channel candidate is
  refuted), and the residual's H-dependence is measured, but the magnitude is not derived.

## Files

- `grid_harness.py` — the grid harness: panel loading, base arrays, θ-hysteresis,
  `hold_shares_walk` (the share-carrying book), the leverage-neutral rescaling of the realised
  return series (`renormalised_to_gross1`, with its `construction_caveat` and
  `gate_quantity_caveat`), three accounting modes, scoring, MDE and uplift CIs, Correction 2's
  component CIs, corrections 1–4, band verdict, significance summary, report assembly.
- `render_tables.py` — emits every table in this file from `result_919a.json`, and since cycle 3
  also checks **prose** numeric literals against the artifact. `--check` gates both. Its reach is
  partial and measured: `PROSE_GATE_MEASURED_REACH` in that module carries the numbers, and
  `_SCRUBS` / `PROSE_NUMBER_WHITELIST` carry the exemptions.
- `hour_of_day.py` — section 7 mechanism + fixture-scale proof + the scheduled real-leg invocation.
- `test_919_grid.py` — **16 checks, 16 PASS** (`919A_SUITE_EXIT: 0`): gates G1–G5 (G1 now
  element-wise, G2 now actually runs git, G4 extended to the share walk, G5 given a hold-shares
  poison *and* a prose poison), decomposition closure, borrow-indexing equivalence, the
  hold-shares return identity and fail-loud contract (both signs), the table-vs-JSON drift gate,
  the prose-number gate, the gross-renormalisation exactness/non-vacuity check, the uplift-CI
  pairing-alignment check, and the hour-of-day mechanism proof.
- `result_919a.json` — full raw result (three windows, 48 cells each, per-cell uplift CIs,
  corrections, MDE, band verdicts, significance summaries).
