# Pathfinder audit — 2026-08-19

**Provenance:** independent deep audit by a general-purpose subagent ("pathfinder"), commissioned
by the operator 2026-08-19: *"find what older agents messed up … find gaps, things that were
tried that should work but were blinded, blocked or misrepresented … show a path to a profitable
system."* Read-only; ~218k tokens over 84 tool calls; headline numbers re-derived from raw
artifacts where they exist locally (`runs/861_ref`, `861_ref_scorecard`, `861_imprint`,
`861_teacher`, `858_oos`, `856_rulegrid`, `878_ablation`, `882_decisive`, `884r1_smoke`,
`885_screen`, `887_confirm`). `runs/875_wf` and `runs/879_rung3` artifacts live only on
gpu-server; those rows were graded on internal consistency and disclosed caveats.

**Operator disposition 2026-08-19 (rulings via planner):** all four §D amendments to
`experiments/892_action_ev/SPEC.md` ADOPTED (to land as SPEC Amendment 1 after factory run 36
wraps); new rows to be specced from §B G1/G3/G6 (corrected 809 re-measure; trend-decay +
spot-legal variant; data backfill to present); record corrections ADOPTED (809 retraction at
source, +0.97 survivorship caveat, 887 alpha-ordering correction). Operator annotations: the
MuZero line is NOT abandoned ("I still like MuZero as an algorithm" — the audit kills the
current line's evidence, not the algorithm); compute placement guidance — idle GPUs are a
failure mode, schedule measurements to keep server GPUs fed, no CPU-heavy work on the
workstation; on backfill — data only accrues forward in time, 899 is catch-up-to-present plus
a pulling habit.

---

# PATHFINDER AUDIT — Trader repo, 2026-08-19 (verbatim report)

Method note: I read `experiments/LEDGER.md` (all 100 lines), `docs/pathway.md`,
`tasks/DECISIONS.md` + archive, `tasks/QUEUE.md` + `QUEUE_ARCHIVE.md`, ~20 `lore/` notes/handoffs
spanning 2026-06-20 → 2026-08-17, the 892/887/885/879 SPECs, and re-derived headline numbers from
the raw artifacts wherever they exist locally.

## A. Verdict audit

| Verdict | Grade | Reason | Evidence |
|---|---|---|---|
| 875/wf-generalization — no OOS edge, 1/5 folds | SOLID | Frame-pinned throughout, each fold graded vs its own same-run floor; f5 confound disclosed and verdict doesn't rest on it. Fold JSONs are srv-only — not re-derived here, but the grader tooling defect was caught and disclosed in-row | LEDGER:97; `runs/875_wf/grade_correct.py` (srv) |
| 861/search-teacher-imprint — stronger teacher FAIL 0/3 | SOLID | Numbers re-verified locally: teacher +8.1180, students 0.5718/0.6923/0.7802 vs ref median 0.9914. Interpretation (hard-target BC can't carry lookahead timing) is almost tautological given the teacher is an R=3 true-env (lookahead) operator — the test only ever measured "is the teacher's non-lookahead residue better", and it answered: worse | `runs/861_teacher/results_861teacher_*.json`; `runs/861_imprint/results_s{0,1,2}.json` |
| 879/rung3 — distill strictly worse; search stack subtractive | SOLID (internal) | The argmax-vs-search control (−0.564 vs −2.166, Δ1.602) is a same-universe internal comparison, valid as stated; absolute levels unpinned by design and disclosed. Its exposure of 857/860 as bankrupt-at-step-1-27 artifacts is the best measurement-integrity catch in the record | LEDGER:91; `experiments/879_rerun/SPEC.md` Amendments 1-2 |
| 887/collapse-confirm — collapse is ADAPTIVE | C1/C2 SOLID; C3 number SOFT; C3 interpretation OVERREACHES | C1 verified from raw CSVs (A 99.072/98.976/98.756; D 91.234/91.412/91.357). C2 slopes all ≤+0.02 pp/ep (OLS: +0.0026/−0.0071/+0.0049 — verdict holds, quoted slopes −0.0189/+0.0031/−0.0040 not reproducible as OLS, cosmetic). C3 return monotonicity verified (pooled final-50 medians: A −2.97%, E −9.11%, D −13.63%). **But "alpha ordered identically (−10.05/−15.27/−23.68)" is wrong as stated**: cross-seed median alpha for A is −17.33 (per-seed −10.05/−17.33/−19.64), so E (−15.27) beats A on alpha — the claimed identical ordering fails. Verdict unaffected (C3 was never a gate) | `runs/887_confirm/training_887_*.csv`; LEDGER:95 |
| 887's *interpretation* ("no edge to protect") | SOFT | Two confounds: (1) reward = absolute return while median per-episode `market_return` ≈ +10.3% and the active books are net short — part of "actions have negative EV" is directional bleed in an up market, not timing; (2) the trend teacher's actions demonstrably have positive in-window EV at fee0 (tier-0 floor +1.19), so the environment contains findable edge — the from-scratch loop's failure to find the trend is a learning/search failure, not proof of absent edge. The SPEC's own framing ("a policy with no edge to protect") is about the learned policy and is fine; the handoff's stronger reading is not supported | `runs/887_confirm/*` (market_return column); LEDGER:95; `lore/handoff_2026-08-11_adaptive-collapse-guard-class-planner.md` |
| 878/ablation — NOISE | SOLID | All 9 arms + control re-verified: medians A +0.9735 / B +0.9698 / C +0.9690; flag-absent control == arm A to 7 digits (0.9698144772760452 both). Affected-cell share 0.006–0.03%. The −0.022 level shift vs the 882 pin is disclosed as the 877 alias fold | `runs/878_ablation/results_878*.json`; commit 63dad5c |
| 861/ref-rebase +0.9914 | SOLID | Re-verified: `runs/861_ref/results_s0.json` sanity `policy_inenv` = 0.9913782688863609; 882 decisive leg cross-instrument +0.9915919/+0.9935519/+0.9521222, Δ2.1e-4, PV within 0.060% | `runs/861_ref/results_s{0,1,2}.json`; `runs/882_decisive/` |
| 861/ref-oos-rebase — no OOS edge under clean semantics | SOLID | Re-verified from `runs/861_ref_scorecard/s{0,1,2}`: regimes +0.59..+1.59 all above floor (PASS 3/3); oos_full +0.107/−0.833/+0.064; wf_q4 median −1.767; wf_q1 median −1.041. The in-row correction (withdrawing "contamination-fabricated") is honest and its self-refutation is correctly argued | `runs/861_ref_scorecard/` |
| 882/instrument-disagreement + decisive-legA | SOLID | Pin mechanism proven by single-variable swaps to float32-ulp agreement; the old-ckpt anomaly correctly left OPEN as an epoch confound, not swept under the pin fix | `runs/882_decisive/`; `tasks/TASK_882_INSTRUMENT_RECONCILIATION.md` |
| 832/E5-oos-real +0.898 (withdrawn) | Handled correctly | Artifact exists (+0.8982 vs floor −1.435, verified); withdrawal is the right call (unreproducible under clean semantics, cause unisolated). Loose end below (G4) | `runs/858_oos/results_oos858_imprint_*.json`; LEDGER:89 |
| 857/860 corridor rows | Correctly VOID-stamped | "Best = survived longest" (28 of ~1670 decisions) — a weight-native ckpt walked through the order env. D26 parked the lane on those numbers; 879 re-parked it on valid ones | LEDGER:80,84,91 |
| **809 maker-gate PASS (+5.08)** | **SUSPECT — retracted in lore, never corrected at the source** | `runs/809_decision_rule_rescue/result.md` still asserts "2 of 864 realistic cells pass" + a deploy plan. `lore/notes_2026-06-25.md:58` records the retraction: the +5.08 was a per-bar-Sharpe artifact (per-bar returns ×√525600 on overlapping holds — the "809 trap" 819's discipline section cites). The corrected per-decision number was never computed; `recompute_best.py` never ran; the second passing cell was never named anywhere durable; the OPT-flagged table is the only data in the artifact | `runs/809_decision_rule_rescue/result.md`; `lore/notes_2026-06-25.md`; `lore/notes_2026-06-22.md` |
| **The +0.97 bar itself** | **SOFT — known-inflated, still cited uncaveated** | `tasks/TASK_819_SINGLE_ASSET_BOTTOM_UP.md` Phase 3 (2026-06-28): on the point-in-time 276-asset universe the same config reads net@21 **+0.56 CI[−0.13,+1.23] — straddles 0**; "Phase-2's [+0.21,+1.95] was survivorship-inflated; survivorship ≈ +0.42 Sharpe". `docs/pathway.md:13` still defines working as beating +0.97 with no survivorship caveat. Every corridor floor/ceiling in the vector line is measured on a coverage-ranked (survivor-tilted) universe | `tasks/TASK_819_*.md`; `docs/pathway.md` |
| 836/E4-p1b — value > momentum (ρ 0.21/0.30) | SOFT | The only positive learned-model signal in the record: 1/5 windows evaluable, n=95, p=0.04/0.0035, pre-E-c framework, on checkpoints (p1b) later abandoned. Never re-measured post-fix; p1c's full-coverage re-read was weak-mixed | LEDGER:60,63 |
| 812 — planner abstains (scalar, minute, fee16) | SOLID and *pre-figures 887* | A month before 887, the scalar stack already measured "on a cost-bound signal under a post-cost objective the optimal learnable policy is don't-trade" — with the epsilon containment working there too. 887's new content is the fee0 control, not the phenomenon | `lore/notes_2026-06-26.md` |

## B. The gaps (ranked by decision value)

**G1. The only gate-passing strategy in repo history was never honestly measured, then abandoned
for non-falsification reasons.** The 806→811 campaign measured a real minute-scale cross-sectional
signal (OOS IC 0.06–0.15, BH-FDR-surviving, n≈217k strided — `runs/806_predictability_probe/result.md`),
proved it cost-bound at taker (808: 0/72), then 809 found conviction-gating + maker execution
passing the post-cost gate (+5.08, CI[+2.77,+7.63], turnover 0.087, on untouched OOS
2024-11→2025-04). That PASS was an estimator artifact (per-bar annualization) — but the
*corrected* number was never computed, so the cell was never actually falsified either; the
closure rests on (a) the artifact retraction, (b) the operator's Fork-C venue ruling (stay Kraken,
no rebate venue → zero maker fee commercially out of reach), (c) D0/D3 killing deployment and the
paper loop. The maker-fill model (OHLC-touch + flat 5bp adverse selection) — flagged in the notes
themselves as "the load-bearing assumption and the known weak point" — was never validated against
a single real order. The idea died of venue economics and a broken estimator, not of evidence.
Re-measurement: re-run `runs/809_decision_rule_rescue/decide.py`'s best cells on the H-strided
estimator, on the original test window AND the 858-backfilled span (2025-04→2026-03 is a full year
of untouched OOS the minute table now carries), with the fee-sweep recorded per-cell this time.
CPU-only, hours.

**G2. The bar is survivorship-tilted and nothing in the vector line ever ran point-in-time.** 819
Phase 3 quantified the inflation (+0.97 → +0.56 straddling 0 on a dead-coin-inclusive 276-asset
universe). The 64-asset frame anchoring every 828+ measurement is coverage-ranked —
survivor-selected by construction (868's alias union literally recovered dead-twin rows). The PIT
machinery exists (819 POWERED run; TASK_837's module-side PIT) but 837 is declared KNOWN-STALE and
no vector-line verdict ever used a PIT universe. Direction of bias: it flatters floors AND
ceilings; the policies failing to clear a flattered floor is doubly bad, but the "no OOS edge"
verdicts' absolute levels inherit the tilt. Re-measurement: one 875-fold floor+policy pair re-run
on a PIT universe would price the bias directly.

**G3. The fallback rule's edge is measured dead in the most recent data — and no row aggregates
that.** Scatter the record already contains: 875 fold floors +3.90 → +1.43 → +1.40 → +0.26 → +0.42
(2023H2→2025H2, monotone-ish decay); 861 scorecard OOS floors (verified locally): oos_full
**−0.818**, wf_q1 **−2.806** (wf_q4 +1.794); 820 gate-2 on truly unseen 2025-12-13→2026-07-04:
net@8 **−1.31** CI[−4.45,+1.29], absolute-negative (beats B&H −2.61 only by avoiding the bear).
"Fixed rule stays the fallback shippable at every rung" (pathway.md:106) is currently false on
recent data. A long-flat, no-short, spot-legal variant (the rule's edge is concentrated in crash
avoidance — "beats B&H by shorting the 2022 crash") has never been pre-registered or measured, and
shorting was never spot-possible on this venue anyway. This is a CPU-hour measurement on the
existing daily table.

**G4. The 861 attribution ladder was abandoned mid-isolation.** Leg D exonerated the mask (5/7
windows bit-identical). Legs B (`task867…` tag) and C (`task868…` tag) — isolating alias-union vs
UTC — never ran, though both are eval-only against existing tags. Decision value is low (875
supersedes the question), but record integrity ("can ANY pre-868 number be trusted even
directionally") stays open, and the program cites the old-ckpt anomaly as UNRESOLVED in two LEDGER
rows. Hours of CPU.

**G5. The one positive learned-model signal was never re-measured on the fixed framework.**
836/E4-p1b: value net ranks states above momentum (ρ 0.21–0.30, p ≤ 0.04), pre-E-c, 1/5 windows,
n=95. The lane was abandoned with the order-env checkpoints, and the ref imprint ckpts likely have
untrained value heads (BC-only) — so this needs a subject check first. If it ever re-runs, it
should be on 887's D-arm checkpoints (broad, non-collapsed, value-trained), which exist on srv.

**G6. Data rot is now the binding constraint on any future OOS claim.** Canon minute panel ends
2026-03-31; the daily table reached 2026-07-04 (820's status log); today is 2026-08-19. Every
"recent" window in the record is ≥4.5 months stale on minute bars. The 858 backfill machinery
exists and is proven; nobody has pulled 2026-Q2+ dumps. Any verdict taken from now on ages in dog
years.

**G7. 892's pre-registration contains two factual miscalibrations** (details in D): the "low
hundreds of non-HOLD decisions per seed" thin-n disclosure is off by ~2 orders of magnitude for
its primary subjects (the ref ckpts' own artifacts show ~89.6% non-HOLD action cells —
51,354+44,607 of 107,072 in `runs/861_ref/results_s0.json`'s `policy_action_hist` — the book
re-marks almost every slot daily), and it measures the *imprint* policies on the *training*
window, while the question it exists to answer was raised by *self-play* policies and is only
decision-relevant OOS.

**G8. Smaller instrument notes.** (a) The 809 result table's only durable rows are the OPT corner
it excludes from its own verdict — the realistic sweep's per-cell data is gone. (b) The 875
grader's silent-fake-null (`WF_SUMMARY.json` = `[]`) was caught; the class-rule (fail-loud summary
extraction) is recorded but not enforced anywhere. (c) D37's class — no historical artifact
records whether its checkpoint loaded — is fixed forward by 888R/889 but remains retrospectively
unfalsifiable for all pre-888R numbers except +0.9914. (d) 884's S1 gate was written against a
percent column as if fractional — caught pre-launch by luck, owned in-row.

## C. The path

Honest answer: **the audited record points away from profitability on the current line, and it
mostly did before the line started.** The daily-bar surface was already measured (2026-06-28,
before the 828 pivot): the only daily signal is slow trend; trend is survivorship-marginal (+0.56,
CI straddling 0, PIT); every nonlinear/sequence model scored *below the shuffled-label noise
floor* OOS (819 CAPACITY: ridge 0.011 → GRU 0.033 vs noise floor |0.059|); and 811 proved the
minute signal linear-optimal. The vector/MuZero program then spent seven weeks re-measuring, with
far better instruments, exactly that: BC reproduces trend in-window (+0.9914,
instrument-independent), generalizes not at all (875), search subtracts (879), distillation
subtracts (879), better teachers don't transfer (861), and from-scratch self-play rationally
refuses its own negative-EV actions (887 — which 812 had already measured on the scalar stack in
June). Nothing in the audit suggests any of these verdicts is an instrument artifact; the two that
were (857/860) were caught and re-measured, and the lane still failed.

The ordered measurement sequence, ranked by information-per-GPU-hour:

1. **Corrected 809 re-measurement** (G1). H-strided estimator, original test window + the
   untouched 2025-04→2026-03 span, full fee×adverse-selection grid with every cell recorded.
   Decision it seeds: if a cell clears at a maker-fee tier the operator can plausibly reach, the
   path to profit is *execution engineering* (fill-model validation via small live limit orders,
   then paper) and the MuZero line's bar becomes that cell, not +0.97. If it fails at every
   attainable tier, the minute surface is closed for real, on an honest estimator, and the
   venue/cost question is forced to the operator with numbers.
2. **Trend-decay aggregation + spot-legal variant** (G3). Re-run the 819 PIT config forward
   through 2026-07 (daily table), plus a pre-registered long-flat/no-short variant. Decision: does
   the program's only positive-expectancy artifact still exist, and is there a version deployable
   without perps?
3. **892 as amended** (G7) — with the OOS leg as the binding read. Decision: does ANY learned
   policy's action stream carry timing edge against a matched null OOS? This is the cheapest
   falsification of *my* bearish read of the trained line: if the ref policies' actions beat the
   matched null OOS at fee16 with a CI excluding zero, there is extractable edge on the daily
   surface and my read is wrong.
4. **Data backfill to present** (G6) before any further OOS claim; otherwise step 3's OOS window
   is the last one that will ever exist.
5. **861 B/C isolation legs** (G4) — cheap, record-integrity only; run when the pool is idle.

What I would *not* spend on: more policy machinery on the daily surface (819 CAPACITY + 875 + 887
have it surrounded), more distill rounds (879), and any training battery before 892 reports
(already the queue's position — correct).

## D. What 892 should change

The SPEC (`experiments/892_action_ev/SPEC.md`) is well-built in mechanics (matched-activity
per-asset-marginal null, within-regime matching, block bootstrap, pre-registered three-band
verdict with an explicit instrument-fail escape) and the partially-built instrument (`trace.py`,
untracked) has an honest event-study contract. Four amendments, in priority order:

1. **Add the OOS leg and make it binding; demote the training window to a sanity leg.** As
   specced, the only window is 2021-01-01..2025-10-04 — the ref ckpts' own training window. An
   EDGE verdict there is near-foregone and nearly information-free: the ckpts are 97.9%-agreement
   clones of a teacher whose in-window floor is +0.66/+1.19, so the policy's actions will show
   "edge" that is just the trend's in-window presence. The decision the SPEC says this feeds (the
   D47 pivot) needs the OOS read: run the identical audit on 2025-10-05→2026-03-31 (the 861
   scorecard's oos_full window, 116 decisions, 62/64 slots — artifacts prove it runnable).
   In-window NULL/ANTI-EDGE would be devastating and is worth having as the sanity leg; but only
   OOS EDGE should be allowed to re-open the line.
2. **Add the subjects 887 was actually about.** The primaries are the BC-imprint ref ckpts; 887's
   claim was about from-scratch self-play policies. The 887 A-arm and D-arm ep300 snapshots exist
   (runs/887_confirm, srv). Adding one A-arm and one D-arm ckpt as a second subject class (never
   pooled with the imprint primaries) directly tests "the collapsed policy's rare actions" (A) and
   "the un-collapsed policy's frequent actions" (D) — the exact quantities 887's interpretation
   rests on. Eval-only, cheap.
3. **Pre-register a position-CHANGE-only co-primary.** The primary subjects' action streams are
   ~90% non-HOLD cells that are mostly deadband re-trims of a standing book; the event-study
   contract in `trace.py` attributes every non-zero post-decision direction, so the primary Δ will
   mostly measure "the book was long/short the right things over the next 21 bars" (book quality,
   ≈ the teacher's signal) rather than decision quality. Report separately: entries/exits (sign
   change or 0↔nonzero) vs maintenance re-trims. Also record per-decision net exposure and forward
   *market* return alongside each event, so a directional artifact (long book in a rally) is
   separable from timing — this is the confound that 887's C3 could not exclude.
4. **Fix the thin-n disclosure and relabel two legs.** The "low hundreds of non-HOLD decisions per
   seed" sentence is wrong for the primary subjects (~95k non-HOLD action cells/seed pre-blocking;
   the binding constraint is the ~80 blocks of 21 decisions, which is fine) — as written it will
   read as an excuse if the CI is tight, and the block count is the honest number to pre-disclose.
   The secondary search leg uses a true-env leaf = R-step lookahead: label it a capacity ceiling,
   not a deployable comparison (the SPEC implies this but doesn't say it). The old-ckpt contrast
   leg inherits the unresolved 882 epoch confound (post-hoc reconstructed frame, mismatched
   window) — keep it explicitly out of the D47 evidence base, not just "reported separately".

One non-amendment: the null design itself (subject's own per-asset marginals, seeded,
path-independent, within-regime) is right, and the "no fourth silent outcome" instrument-fail
clause is exactly the discipline the 875 grader bug argued for. The core measurement stands as
specced; it just needs to be pointed at the window and the policies the question is actually
about.
