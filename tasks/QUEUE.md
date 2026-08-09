# QUEUE — the ranked dark-factory work order

The factory's single entry point (planner-maintained). STATUS banners on task files remain the
truth for what's done; this file only orders what's OPEN. The factory takes from the top, skips
gated rows (noting why), and treats FILLER rows as parallel/anytime slices. Last groomed:
**2026-08-07 (planner, post factory run 27) — run 27 RATIFIED, both halts CORRECT, and the
second one was MY boundary error.** 876 and 880 both halted on the 3-cycle rule with no code
shipped; D33 and D34 are RULED. **Ready (physically ordered): 1 TASK_876 (re-rescoped to THREE
items, zero deleter surface) → 2 TASK_882 (NEW — instrument reconciliation) → 3 TASK_873 →
4 TASK_877 → 5 TASK_874 → 6 TASK_878. GATED: TASK_880 (behind DEPLOYED emission, D34) and
TASK_881.**
**D33 — my error, owned:** D32 rescoped 876 to "the non-destructive half" and then listed
`_is_protected` descendant-awareness and the TTL re-scan as *mechanical* items. Those ARE the
protection model. I put two 881 items into the row I had just declared free of 881, and three
cycles died in exactly that surface. 876 is now EXACTLY: (1) unconditional `scores.jsonl`
emission, (2) `flax_keep = 3` + `overwrite=False`, (3) `--ckpt-retention` defaults to `recency`.
Everything touching `ckpt_gc.py`'s protect-or-delete logic — including the symlink
`followlinks=False` BLOCKER and the disputed `_order_newest_first` (a PRE-EXISTING HEAD bug the
orchestrator owns adding to a cycle-3 brief) — is now TASK_881's.
**D34 — the queue was missing a hard dependency, also mine:** I ordered 876→880 as adjacent rows
without encoding that 880's entire safety argument rests on an artifact 876 produces ON THE HOST.
The gate is "876 DEPLOYED to gpu-server with real `scores.jsonl` written there", not "876
shipped". **`--unscored archive` (pull, never delete) is AUTHORIZED; the `--apply` freeze STANDS;
do NOT run the 2.61 GB bulk pull** (162 GB free, server disposable, and store-then-decide is what
the unscored ruling rejected) — pull only the score-bearing anchors, ~0.4 GB.
**Run 27's most valuable output is a MEASUREMENT, not a fix:** the real-host dry-run
`ARCHIVE 1 ckpt / 0.09 GB · DROP-WITHOUT-ARCHIVE 27 ckpts / 2.51 GB` turned the `--apply` freeze
from precautionary into a demonstrated near-miss on `runs/861_ref/ckpt_ref_s0/checkpoint_2000`
(net@21 0.9914, the clean-recipe anchor). Root cause is not a bug: **no `scores.jsonl` exists
anywhere on gpu-server**, so `--unscored drop` reads "nobody wrote the file" as "measured and
worthless".
**SCIENCE, planner lane (LEDGER 879/rung3-distill-remeasure + 882/instrument-disagreement):** the
857/860 distill verdicts are VOID — all six artifacts bankrupt at step 1-27 of ~1670, and both
headlines (−4.23/−4.02) are just the seeds that survived longest; cause = a weight-native ckpt
walked through the ORDER env because E-a is FLAG-GATED and the verbatim command omits the flag.
Re-measured correctly the lane still fails (medians −2.84/−2.73 vs floor +0.5164) and 6/6
distilled seeds are BELOW the un-distilled baseline. **The control is the bigger result: the
SEARCH STACK IS SUBTRACTIVE** — argmax −0.564 vs search+v̂-leaf −2.166 on identical weights
(Δ1.602 > the pre-registered 1.0 bar). **And then the instruments disagreed:** the same ckpt
reads +0.8394 / PV 299,547 on `imprint.py` sanity and −0.6145 / PV 41,919 on `eval_vector
policy-argmax`, same window — hence TASK_882 at Ready #2. **No number may be compared across
those harnesses until it lands.** Factory GREEN.**
Prior grooming:
**2026-08-06 (planner, post factory run 26) — run 26 RATIFIED. TASK_872 shipped
(`task872-live-stream-utc`, `f5a74db`); TASK_876 STOPPED on the halt rule and D32 is now RULED.
Ready (physically ordered): 1 TASK_876 (RESCOPED — emission half only, resume from the stash) →
2 TASK_880 (NEW — `ckpt_archive.py` discovery, the tool that deletes REMOTELY) → 3 TASK_873 →
4 TASK_877 → 5 TASK_874 → 6 TASK_878. TASK_881 (NEW — the checkpoint protection model) is GATED
behind 876+880.**
**D32 RULED (planner, operator may override): (d) SPLIT + (a) explicit fail-loud protection, and
the destructive default goes OFF (`--ckpt-retention recency`).** Full text in DECISIONS.md. Two
consequences the factory should read before taking row 1: the 876 stash resumes as the
NON-DESTRUCTIVE half only (emission + three mechanical fixes + revert `flax_keep = 1_000_000`
and set `overwrite=False`), and **there is an OPERATIONAL FREEZE on `tools/ckpt_archive.py
--apply` until TASK_880 lands** — planner-verified in source: `is_ck()` returns true for any dir
merely CONTAINING a `checkpoint_*` entry, so a whole scored run reads as one unscored opaque
leaf and gets deleted on the training host under the (correct, operator-ruled) `--unscored drop`
default.
**TWO PLANNER ERRORS THE RUN CORRECTED, both owned:** (1) **the 872 ranking justification was
false.** This queue ranked it "the only known defect result-bearing on the LIVE path, not
latent"; both named consumers are DEAD CODE — `real_trade_jax.py` is self-documented BROKEN/
LEGACY and `sys.exit`s before constructing, `PaperExchange.set_data_source` has had no caller
since `c4a27e0` (planner re-verified both). The FIX is correct and worth having for the live
rebuild; the ranking rationale was not, and it displaced work that was actually result-bearing.
(2) **876 was sized "S" and carried irreversible-deletion semantics** — 1,208 insertions across
6 files with three consecutive panels finding real BLOCKERs in the previous round's fixing code.
Emission should have been split from deletion policy at authoring time; the halt rule caught a
decomposition error, not an implementation one.
**Also absorbed (factory finding 3, cross-cutting, no row of its own yet):** 872's specified
two-line fix would have made a PRE-EXISTING defect silent — attaching `tzinfo=utc` to a naive
parse is right, but upstream truncation had already destroyed the real offsets, so the guard
made a wrong answer CONSISTENT rather than CORRECT, and the round's own test asserted the wrong
value and passed. **A guard that normalizes looks identical to a guard that corrects.** Same
class as the tooling defect the planner hit this window (a summary reader whose missing-key
default turned absence into a result — LEDGER 875 pointer) and as TASK_878's masked-target
hazard. Any guard or reader added anywhere in this program must state which of the two it is.
**SCIENCE — the 875 walk-forward battery COMPLETED and returned a verdict (LEDGER
875/wf-generalization): NO OOS EDGE, 1/5, the pre-registered "none" band.** Fold medians vs
their own floors: +1.4957/+3.8987 · +0.3995/+1.4301 · +0.3108/+1.3952 · +0.4121/+0.2635 (the
only clear, and only because the FLOOR collapsed) · **−2.9566/+0.4244** on 2025-H2, a 33% median
drawdown on the half-year closest to deployment. The recipe's OOS median is flat (+0.31..+0.41
on three folds) while the floor swings 15×, so it does not track window difficulty at all.
Caveat recorded: fold 5's test window overlaps the known sparse span for 3 of its 6 months, so
its MAGNITUDE is confounded (TASK_877 shrinks that set) — but folds 1-3 sit outside the span and
all three fail the floor, so the verdict does not rest on it. **The imprint lane now has real
in-window learning (+0.9914) and no out-of-sample edge on n=5 pre-registered folds.** Factory
GREEN.**
Prior grooming:
**2026-08-05 (planner, LATE — compute program + operator requirement) — Ready = TASK_876
(score-based checkpoint retention, OPERATOR-REQUIRED, gates the long base-training program) → 872
→ 873 → 874. RUNNING on gpu-server (planner-scheduled, GPUs 1-3, tmux `wf875`): the **TASK_875
walk-forward generalization battery** — 5 expanding-window folds × 3 seeds, train 2021-01-01→fold
cutoff, test the 6 unseen months after it, universe pinned identical across folds. Pre-registered
bar (in the script header before launch): per fold take the seed MEDIAN; ≥4/5 folds positive AND
above their own on-window teacher floor = real OOS edge · 2-3/5 inconclusive · ≤1/5 no edge. This
replaces the n=2-quarter OOS evidence the withdrawn +0.898 claim rested on. **ISOLATION RESULT
(leg D, done): the frozen-cell MASK is EXONERATED** — mask-off vs mask-on on the old ckpt is
5/7 windows bit-identical, max |Δ| 0.032, still negative everywhere; after the 868 alias repair
only ~1% of cells are frozen, so the gate had almost nothing to forbid. Remaining unisolated
candidates for the old ckpt's collapse: the alias union (i.e. ADDING REAL DATA changed the panel —
not contamination) and UTC windows. Operator architectural ruling recorded in LEDGER: forward-fill
is REQUIRED (dense panel) and a stale price IS valid while nothing trades; the real defects were
the zero price (fixed) and the identity split (fixed). Factory GREEN.**
Prior grooming:
**2026-08-05 (planner, post factory run 25) — run 25 RATIFIED: TASK_870
(`task870-canon-db-readonly-handle`, `4fb0d77`) + TASK_871 (`task871-residual-localtime-parses`,
`9790644`) both shipped, 0 BLOCKER/MAJOR each. The medium-tier escalation over the declared XS
was correct (870's diff reached `eval_vector.py` + `trader/`), and control B EARNED the task by
catching the real pre-870 loader silently rewriting a journal-carrying DB on a bare SELECT —
that plus the untouched `trading_data.db` (mtime/size unchanged, only atime moved, with two
judges reaching it) is the shipped handle working on day one. The import-cost correction and the
disclosed-cost block are the right call; a wrong "does not touch X" comment is worth catching.
**Both run-25 questions answered YES.** (1) The live TZ defect is Ready #1 — TASK_872, and the
planner verified its blast radius: live/paper decision path only, NO persistence, so no
source-of-record rows are implicated. (2) The DB backup was refreshed FIRST, before anything
else touches that file — and the situation was worse than "predates 858": the newest backup
(`backups/trading_data_postquarantine_2026-07-26.db`) reaches only **2025-09-30**, so the entire
858 OOS extension (2025-10-01 → 2026-03-31) — precisely the data every OOS read depends on —
existed in NO backup. **DONE 2026-08-05 18:09 UTC — `backups/trading_data_858oos_2026-08-05.db`,
ALL GATES PASS**: source `PRAGMA integrity_check` ok · target `integrity_check` ok · content
parity IDENTICAL on all 8 probes (size, min/max ts, n_assets, quarantine, ohlcv_daily, two
per-asset row counts) · coverage **2013-09-10 .. 2026-03-31** · sha256
`799dca96c99e76e9c84533585f2829f8e82e004b897bd163dc491f0f25fa5bc8` (sidecar committed-adjacent,
file itself untracked). Live DB untouched throughout: mtime still 2026-07-26 20:47:32, size
unchanged. (First attempt was killed mid-copy at 11 GB by a harness/session event; the truncated
`.partial` was DISCARDED, not reused, and the retry ran detached under `setsid`.) **Residual ops
item, operator-owned: all three backups sit on the SAME disk as the live DB — a second-disk or
offsite copy is still unaddressed.**
**Ready = TASK_872 → 873 (fixture identity pin) → 874 (imprint emits provenance).** The
fixture-regen incident is recorded as a standing rule (memory + TASK_873): a gitignored fixture
is invisible to the clean-tree check the whole ritual leans on, so no agent may regenerate it as
a side effect — a re-base is planner-scheduled work. Factory GREEN.**
Prior grooming:
**2026-08-03 (planner, FINAL) — THE OOS RE-MEASURE REVERSES THE PROGRAM'S HEADLINE (LEDGER
861/ref-oos-rebase): the +0.898 OOS / WF-pass anchors are REVOKED as contamination artifacts.**
Clean-trained ref seeds: regime form PASS 3/3 (+0.59..+1.59) but oos_full median +0.064 · wf_q4
−1.767 · wf_q1 −1.041; attribution leg (OLD ckpt × clean battery) NEGATIVE everywhere in-window
(form FAIL 0/4) — the old profile was contamination-DEPENDENT. Program state under honest
measurement: real in-window learning (+0.9914 median, seed-stable), NO demonstrated OOS edge
anywhere in the stack; search ceiling +8.118 stands (clean-measured, non-deployable). Recommended
factory row for next grooming: imprint.py emits frame provenance for its output ckpt (the gap
cost a REFUSE round-trip tonight; provenance derived planner-side with order-exact assert,
runs/861_ref/frame_provenance_ref_s*.json). Ready = TASK_870 → 871 unchanged; factory GREEN.
The frontier question (what generalizes OOS?) returns to the planner on clean instruments.**
Prior grooming:
**2026-08-03 (planner, LATE) — 861 CHAIN COMPLETE, VERDICT IN: search-teacher imprint FAIL 0/3
(median +0.6923 vs re-based reference median +0.9914 — every seed below the book-teacher
battery; LEDGER 861/search-teacher-imprint). The re-based reference itself is the headline:
+0.9914 median, seed spread 0.041, supersedes the demoted +0.839 (LEDGER 861/ref-rebase).
Better-teacher hypothesis on the imprint lane CLOSED (joins the parked 857/860 distill lane:
the search's edge lives in lookahead timing that neither distillation nor hard-target BC can
carry). Leg 4 not run. Ready = TASK_870 → 871 unchanged; factory GREEN. Planner remainder,
sanctioned by D31 (a3): re-measure the +0.898 OOS and WF quarters for the RE-BASED reference
(862 scorecard over ckpt_ref_s{0,1,2}) — the reference's OOS standing is still the pre-D31
number.**
Prior grooming:
**2026-08-03 (planner, post run-24 ratification) — 861 REAL-LEG CHAIN LAUNCHED on gpu-server**
(tmux `861chain`, deploy `b695487`, orchestrator `orch_861_chain.sh`): leg 0 (3-seed 832-recipe
reference battery, GPU 2 serial) ∥ leg 1 (pinned search-teacher pass, GPU 1) → leg 2 (3-seed
861 imprint, GPU 2) → leg 3 summary; STOPS before leg 4 (planner-gated median read). Both legs
minute-pinned on `frame_provenance_imprint.json` (md5 d7a44ff8…, verified on srv), 160G safe_run
caps, no `--allow-off-pin-teacher`. Grading per the D31 addendum: median-vs-median, same
semantics both sides; +0.839/+0.898/WF are demoted anchors. **Ready = TASK_870 (canon DB
read-only handle — run-23 rec (D), Q2 answered YES) → TASK_871 (residual local-time parses,
filler).** Factory GREEN — surface disjoint from the running srv chain (srv runs the deployed
snapshot). Ops flag for the operator: the only backup of the 24 GB source of record predates the
858 backfill — refresh when convenient (sync-db.sh discipline, srv-side or offline copy). Lost
uncommitted CLAUDE.md planner edit from run 24: content unknown to the planner — operator to say
if it should be reconstructed.**
Prior grooming:
**2026-08-03 (post factory run 24, FINAL) — THE D31 RE-BASE WAVE IS CLOSED: all four rows SHIPPED, strictly serial as ordered.** TASK_866 (`task866-weight-on-pin`, `ac8dfe1`) · TASK_867 (`task867-861-baseline-promote-port`, `d670d57`) · TASK_868 (`task868-asset-identity-repair`, `3388a0c` + 861 promote `a687aff` — alias union + causal frozen-cell open-gate; frozen-interior 2,013 → 1,011 cells; PV exact mask ON; re-anchor repro 130,000 → exactly 100,000 masked) · TASK_869 (`task869-rebase-wave-semantics`, `3e0baa9` + `9a5b0b2` + `894ab36` — UTC windows byte-proven TZ-invariant; daily floor mask measured inert; release round caught + fixed a REAL leg-vi break). Every moved numeric attributed (`runs/868_asset_identity_repair/baseline_attribution.json`, `runs/869_rebase_wave_semantics/baseline_attribution.json`); sibling suites 834/836/837/844/857 declared KNOWN-STALE (858-backfill drift) riding the reference re-measure wave. Final re-verify: 861/863/865 ALL LEGS PASS, selftest 12/12, scorecard PASS. **Ready = EMPTY; factory IDLE. UN-GATED (planner-scheduled, per the D31 addendum in TASK_861): the 861 leg-0 reference battery — re-measure +0.839 / +0.898 / WF on the post-868/869 semantics — then the median-vs-median leg 3. Process note for the planner: two panel/implementer agents exceeded role this run (one shipped 866's ritual itself — destroying the uncommitted CLAUDE.md planner edit, which is LOST; two disarmed the factory mid-run); the orchestrator completed both rituals and re-armed. Recommended rows awaiting specs: run-23 rec (D) (canon panel path opens trading_data.db READ-WRITE — vectorbt_env.py:515/:731); the surviving local-time parses off the read-of-record surface (diff_teacher/tools, named in the 869 banner). Factory GREEN.**
Prior grooming:
**2026-08-02 (post factory run 24) — TASK_866 SHIPPED (`task866-weight-on-pin`, `ac8dfe1`): the weight-ON configuration of the 832 module is now pinned against the tracked baseline read as its OWN-configuration oracle (fail-closed on missing/dirty/corrupt/wrong-config; cross-config = named REFUSAL). The 864 incident class is closed — every committed parity baseline in the 832 module is guarded by an oracle of its own configuration (scope-3 sweep: no siblings). Correctness judge GREEN first-hand, 0 BLOCKER/MAJOR; selftest 12/12, 861 smoke 11/11. Ready = TASK_867 → 868 → 869, STRICTLY SERIAL (shared surface, unchanged per D31). Factory GREEN.**
Prior grooming:
**2026-08-01 (D31 RULED by the OPERATOR: opt (a) as amended — repair the identity split, mask the
~1% genuine residue, ONE re-base wave, then re-measure references) — Ready = TASK_866 → 867 →
868 (alias union + interior mask) → 869 (UTC windows + masked daily floor), STRICTLY SERIAL
(shared surface). TASK_868/869 un-flagged: attribution resolved — the run-23 "unauthorised
drafts" were the planner session writing while the factory was ARMED; the orchestrator's
quarantine was correct and is preserved in the D31 row. Standing rule adopted: the planner does
NOT write tasks/, DECISIONS.md, or QUEUE.md while the factory is armed. +0.839 / +0.898 / WF are
DEMOTED to pre-D31-semantics anchors; the 861 chain (leg 0 reference battery + median-vs-median
leg 3, per the D31 addendum in TASK_861) launches only after 868+869 ship verified. Factory
GREEN.**
Prior grooming:
**2026-07-31 (post factory run 23) — TASK_865 SHIPPED (D30 opt (i): stale tails forward-filled +
disclosed; 2 BLOCKER + 2 MAJOR + ~12 MINOR closed in one fix round; selftest 12/12 · 863 13/13 ×2 ·
865 25/25 · 861 e2e 10/10 · scorecard 8/8). **Ready = TASK_866 → 867** (SERIAL, the 865 stash no
longer owns their surface). **NEW HUMAN WALL: D31 OPEN — it gates the 861 real-leg chain a FOURTH
time.** 865's own disclosure work made it visible: interior frozen spans are tradeable and can
fabricate PnL (a re-anchoring constant series is NOT the fee-drag-only case D30 opt (i) reasoned
about) — 11 of 64 slots carry a near-empty 2025-03-31→2025-10-01 span = 2,013 cells / 1.88%, with
re-anchor jumps up to +64.52%. And the canon UNPINNED loader fills identically, so the +0.839 bar is
likely contaminated the same way. Honest inert share is 10.67%, not the 8.79% D29/D30 record. Two
further planner rows recommended below (daily teacher floor unmasked; `_to_epoch` local-time window
boundary). Factory GREEN.**
Prior grooming:
**2026-07-30 (D30 planner-RULED opt (i): FORWARD-FILL + disclosure, uniform on both pin paths) —
Ready = TASK_865 (RESUME FROM STASH, amended scope: PV-conservation leg first-class, riders
(a)/(b)/(c), daily disclosure parity, cell-level ≈9.6% coverage gate, baseline re-promote) →
866 → 867 (SERIAL, shared surface; 865 MUST land first or the verified stash breaks). TRUE
LIQUIDATION (D30 opt (ii)) PARKED as a future row alongside row D. Factory GREEN.**
Prior grooming:
**2026-07-30 (post factory run 22) — TASK_865 STOPPED, NOT SHIPPED: built + fully verified, then BLOCKED
on the newly-seeded **D30** (stale-tail PV semantics — the minute pin's tail treatment fabricates up to
~12.5% of PV into the exact scalar leg 3 grades against +0.839, and the panel's own unanimous fix was
measured NOT to work). Code preserved in `git stash`, nothing committed. **Ready = TASK_866 → 867**
(both untouched this run, see the sequencing note); the 861 real-leg chain is blocked a THIRD time.**
Prior grooming:
**2026-07-29 (D29 planner-RULED opt (a)) — Ready = TASK_865 → 866 → 867 (SERIAL, shared surface); row D PARKED (no ExIt continuation scheduled); factory GREEN.**
Prior grooming (run 21 final):
**2026-07-29 (post factory run 21, FINAL) — BOTH rows SHIPPED: TASK_863 (`task863-imprint-frame-pin`,
`609d6d3`) + TASK_864 (`task864-parity-baseline-hygiene`, `6b71e0e` + refresh `086b1c4`). Ready =
EMPTY; factory IDLE. One human wall: **D29 OPEN** — it gates the 861 real-leg chain. Four recommended
next rows listed below for the planner to spec (A: minute-path frame pin, the D29 unblock; B: a
weight-ON pin — the missing guard that let the 864 drift hide for ~80 commits; C: port the promote
gate to 861's baselines; D: frame pin for the ExIt continuation).**
Prior grooming (mid-run 21):
**2026-07-29 — TASK_863 SHIPPED; Ready = TASK_864
(sole row). D29 seeded OPEN and it GATES the 861 real-leg chain: the shipped frame pin is only
available on a DAILY loader reaching 33/64 frame slots, while the +0.839 bar is minute-panel-native
with a 64-asset book — so D28 opt (a) as written substitutes a 64→33 breadth confound (first-order,
ST3 2.76×) plus a source confound (D20, cross-source rulings forbidden) for the 59/64 universe drift
it removes. Recommended: teach the pin to the MINUTE path (858 gives 62/64). Do NOT launch the 861
real legs on the current pre-registration.**
Prior grooming (run 20):
**2026-07-28 (post factory run 20; D27+D28 planner-RULED opt (a)) — Ready = TASK_863 → TASK_864 (SERIAL, same surface); factory GREEN.**
Prior grooming (D26):
**2026-07-27 (D26 RULED: "prepare dark factory tasks") — Ready = TASK_861 + TASK_862 (filler); factory GREEN.**
Prior grooming (run 19):
**2026-07-27 (post factory run 19) — Ready = EMPTY; factory IDLE; 860 real leg RUNNING on srv (planner).**
Prior grooming (D24):
**2026-07-26 (post D24 ruling) — Ready = TASK_860 (reward-head co-distill); factory GREEN.**
Prior grooming (run 17):
**2026-07-24 (post factory run 17 — D23 queue CLEARED) — Ready = EMPTY; factory IDLE.**
Run 17 shipped all three rows (856/857/855, below) and seeded **D24 OPEN** (frozen reward head
caps the deployed-Q ranking — read-then-decide fork). Planner ruling 2026-07-24: both binding
reads SCHEDULED on gpu-server per the pre-registrations — (a) the 856 rule-grid sweep (ep300
ckpts × threshold-τ/top-m grid vs floor +0.424) and (b) the 857 real distill (3 seeds, warm from
`ckpt_imprint_full`) → calibration (bar ≥ +0.5) → the FIRST deployable-operator corridor read
(`--leaf-eval vhat-distilled`, floor +0.424 bar, +8.02 true-leaf ceiling reference), with a
pre-distill baseline leg for attribution. Next Ready rows seed from those verdicts (D24 fork:
reward-head co-distill = operator wall). D22 (minute backfill, ops) stays OPEN non-blocking.
Prior grooming (run 15):
**2026-07-22 (post factory run 15 + D21 ruling) — Ready = TASK_854; factory GREEN.** Run 15
shipped **TASK_853** (fixed-frame OOS panel, below) and independently confirmed the D20 audit;
the planner's 851 isolation ladder (6 variants, LEDGER 832/E5-inversion2) missed the +0.74 bar
but isolated the last delta vs E1 = the prior term in Gumbel selection ⇒ **D21 RULED: TASK_854**.
Gated behind 854's real inversion read: the operator sweep → Amendment-6 ExIt re-try. Gated
behind the planner's 853 numeric leg: the binding ~240-decision OOS benchmark.
Prior grooming (run 14): Run 14 shipped **TASK_851** (`task851-true-env-leaf`) — `--leaf-eval true-env`
wired into search + self-play, both fixture acceptance legs PASS, medium panel 2 rounds clean.
**TASK_852** reconciled the daily/minute divergence: diagnosis REFUTED the lookahead/+1-day
hypothesis (LEDGER 850/eqgate); root cause = daily-loader densification of zero-coverage assets
(78% of the Δ0.2565 gap); fix closes it to Δ0.0555 — still 2.75× the 0.02 tolerance ⇒ **STOP per
its own acceptance (ii), DECISIONS D20 seeded OPEN** (residual not fully isolated; recommendation
= audit intra-series gaps first). Verification incidentally caught+fixed a real BLOCKER (the
fix's own active-mask filter reactivated a dormant duplicate-cycling bug in `create_daily_panel`,
reproduced live on TASK_852's own scheduled OOS command — 3 of 4 panel members converged on it
independently) — now fails loud instead. Binding OOS re-run (`832/E5-oos`) and the Amendment-6
ExIt re-try (post-851, `--leaf-eval true-env`) both stay gated: OOS on D20, ExIt on the planner's
scheduled real-panel inversion read (≥ +0.74) using 851's new leaf.
Prior context: D13 redirect; the imprint (+0.839) is the program's first above-floor learned
policy; the 840/843/846 env stack has run every fee16 rollout with zero bankruptcies.
The E5 chain executed clean through the night: D15 gate-read (order-code FAIL rel 0.290 /
weight-native PASS rel 0.041 ⇒ branch 2, LEDGER `832/E5-gate`) → **full imprint DONE**
(weight-native, iters 2000: BC CE ÷100, agreement 97.5%, sanity replay net@21 **+0.839 vs
teacher floor +0.542**, above floor BOTH tiers, zero bankruptcies — first learned policy above
floor in the program; strict ±0.10 window missed on the UPSIDE ⇒ PASS-with-caveat, LEDGER
`832/E5-imprint`; ckpt `runs/832_e5/ckpt_imprint_full` on srv). Factory run 11 held 846 on
acceptance (iv) → **D16 seeded, then RULED same night: SHIP AS-IS** (binding 64-asset read
Δ=0.0008 ≪ 0.02, `runs/832_e5/gate_read_post846.json`; fixture Δ known-benign; (a′) rejected
unsound). Prior context: **2026-07-14 D13 REDIRECT** (operator; H6 = rainy-day contingency) →
critical path = E5 imprint→ExIt.

**BOTH OPEN QUESTIONS RULED 2026-08-05 by the OPERATOR — one rule answers both, because they are
the same problem ("define what is real, and the consequences"): a bar with no print means an order
is PLACED AND FAILS TO FILL — the stale price is a reasonable quote, so the trade is neither
illegal nor fabricated; valuation marks at the last valid price (D30 forward-fill = architecture),
execution requires a real print, and a failed fill changes nothing.** This makes 868 (a2) wrong in
KIND though near-inert in size: it restricts the legal mask to HOLD, so the BC target teaches
"HOLD was intended" when the truth is "BUY was intended and could not fill" — intent must be
preserved and only the OUTCOME nulled. → **TASK_878** (Ready #2). The stride question is answered
by the same rule: a sampled minute with no print is not a data defect, it is a bar in which no
trade was possible.
**ALSO RULED — the standing directive with the widest reach: "if there is experiment on known
broken framework we shall rerun experiment on fixed framework".** Almost every negative verdict on
file was measured pre-E-c (see the new FRAMEWORK EPOCHS block at the top of `experiments/LEDGER.md`)
— those verdicts are now **PROVISIONAL**, citable as history but NOT as "closed" in any argument
that gates work. Ladder + sequencing: **TASK_879** (planner-scheduled GPU program; rung 1 is the
828s3 phase-1 self-play battery, which is also the long base-training block the idle compute is
for).

_Superseded note (kept for the record) — the two questions as originally posed:_
**PLANNER NOTE 2026-08-05 — two OPEN QUESTIONS for the operator, both of the FORWARD-FILL CLASS
(a design decision I may have misread as a defect). Neither is a task; both need a ruling, not a
fix.** (a) **The interior frozen-cell MASK (868 a2) rests on a premise the operator rejects.** I
shipped it believing stale fills fabricate PnL; the operator's position is that a stale price IS
valid while nothing trades. Measurement says the mask is near-inert (0.94% of cells; mask-off vs
mask-on = 5/7 windows bit-identical, LEDGER 861/ref-oos-maskoff) — harmless today, but it is an
unconditional semantic gate in the canon loader justified by a premise now known to be wrong.
KEEP as defence-in-depth, or REMOVE? (b) **The `[::bar_minutes]` stride samples ONE minute as the
daily bar** instead of aggregating the day — so a sparse asset's "daily price" is whatever printed
at that exact minute, else a fill (measured: MATIC/USD, 1,076 rows with no print at the sampled
minute). If deliberate (fast, dense, deterministic) it is ARCHITECTURE and should be stated as
such; if nobody chose it, it silently shapes every panel number. I do not know which, and guessing
is how the last misreading happened.

## Ready — take from the top

_D29 planner-RULED 2026-07-29 (opt (a); operator may override — see DECISIONS). Rows A/B/C from
the run-21 wrap are seeded below as TASK_865/866/867; **row D (ExIt-continuation frame pin) is
PARKED** — no ExIt continuation is scheduled or planned (ExIt failed twice, LEDGER 832/E5-exit +
E5-exit2; the current program is imprint→OOS), so the row would guard a phase that doesn't
exist. It un-parks automatically the day an ExIt continuation is pre-registered._

_**FACTORY RUN 28 (2026-08-08) — TASK_876 SHIPPED** as `task876-score-emission` (`ae127b4`), on
the third attempt, after two halts. D33's three-item re-cut was the thing that worked: resumed
from `stash@{0}` as a SUBTRACTION (1,864 → 611 insertions), `tools/ckpt_gc.py` byte-identical to
HEAD, no protect-scan finding had anywhere to land. 2 cycles, 1 MAJOR closed (reproduced), 0
BLOCKER/MAJOR outstanding across a 4-lens panel + 2-lens re-verify. **NEW WALL: D35 OPEN** — the
emitted scores are unreadable by the shipped GC for 2 of 3 naming schemes (`_load_scores` keys by
epoch, `_epoch_of` parses the STEP out of the dir basename; `checkpoint_epoch_<N>` matches
nothing; only `snapshot_ep<N>` round-trips), reproduced by three independent agents. **This makes
D34's gate for TASK_880 necessary but NOT sufficient**, and it may mean TASK_881 should move ahead
of TASK_880. Fail-safe today, `--apply` freeze STANDS. `stash@{0}` deliberately RETAINED for 881._


_**PLANNER, RUN-28 RATIFICATION + D35 RULED (2026-08-08).** Run 28 ratified as reported: three rows
shipped, `trading_data.db` and the fixture byte-untouched, selftest 12/12. Two orchestrator calls
reviewed and both RATIFIED — routing the epoch/step reader defect to 881 rather than blocking 876
(the alternative was a third halt on a boundary D33 had just re-cut), and having 882's first ruling
rewritten from "imprint.py vindicated, +0.9914 restored" to "corroborated, not restored; the old
checkpoint's disagreement UNRESOLVED". The narrow ruling is the correct one and it is the second time
this program has been saved by a judge refusing a flattering conclusion._

_**Queue changes.** D35 ruled (a)+(b) with a planner rider: 880's gate rewritten (881's READER must
ship, not merely 876's emitter), 881 UN-GATED and moved to Ready — and **both ranked behind the
science rows**, because the checkpoint-lifecycle lane has now consumed 9 factory cycles across three
runs while nothing was ever actually at risk. TASK_883 added at rank 5 (the orchestrator's proposed
row, adopted verbatim) and 874 promoted to rank 4 ahead of it, because 883's refusal needs the
artifact 874 emits. **876 is DEPLOYED to gpu-server as of 2026-08-08 (`60f77f7`)** — which is what
makes D35 concrete rather than theoretical: half of D34's gate is literally met and the signal is
still unreadable._

_**Both operator questions answered.** (1) 881 ahead of 880: YES — see D35. (2)
`runs/861_ref/frame_provenance_ref_s{0,1,2}.json`: **RECOVERED, not lost** — they were on gpu-server
all along (derived planner-side there 2026-08-03) and had simply never been pulled to the
workstation; 9 KB pulled 2026-08-08 along with the three `scorecard_manifest_s*.json`. All three
VERIFIED order-exact against `runs/832_e5/frame_provenance_imprint.json`, which upgrades TASK_874's
"order-exact ASSERTED at derivation" to measured. **882 Leg A is RUN and PASSES in the top band** (Δ 0.000214; `+0.9914` RESTORED, instrument-independent, 3-seed) — see Scheduled runs._


_**PLANNER, RUN-29 RATIFICATION + D36 RULED (2026-08-09).** Run 29 ratified: 874 shipped, 883 halted correctly
rather than producing a fourth variant of the same lie. The orchestrator's framing — "a guard that asserts
looks identical to a guard that verifies" — is the same reader-defect class as run 28's "a guard that
normalizes looks identical to a guard that corrects", now on the certification side. **D36 ruled (a) as thesis
but REJECTED the proposed (c) split**: hole 1 lives inside the half (c) would ship, so the banner would have
gone on the wrong lane. Re-cut by CERTIFICATION CAPACITY instead — 883R may only refuse or disclaim and MUST
NOT emit `universe_pinned: true`; TASK_886 owns the positive claim. Cycle-3 `explicit-symbols` call RATIFIED.
**Planner size error owned: I declared 883 XS/S "arg validation"; it is a universe-identity contract across
two pinning mechanisms and three checkpoint producers. Sixth error of one family.**_

_**877 SEQUENCING — HOLD until 883R ships, then take it.** 877 changes the alias resolver ⇒ coverage ranking ⇒
derived frames, which is exactly the surface D36 is open on, and hole 2's measured cost was a paste-ready row
at net@21 −3.953 with 6/8 slots wrong from a SILENTLY DROPPED symbol — precisely what an alias-resolver change
can trigger. 883R's realized-panel equality assertion is the guard that makes such a drop refuse instead of
grade, so it must land first. It does not need 886: preventing the lie is 883R's job, certifying the truth is
886's. Ranked 6, behind both._

_**INCIDENT (run 29) — root-caused and FIXED at the root this time.** The implementer wrote-then-deleted the
real `runs/861_ref/frame_provenance_ref_s0.json` and reconstructed it. **Independently confirmed correct
against a copy the incident could not reach**: gpu-server's `~/Trader/runs/861_ref/frame_provenance_ref_s0.json`
(mtime 2026-08-03 23:30, 1524 B, untouched since) is sha256-identical to the restored file — a stronger check
than order-exactness vs s1/s2, because it predates the incident and lives on another machine. **The real defect
was that these files were gitignored**, so no clean-tree gate and no `git checkout` could see or restore them.
sha256 sidecars share the disk with what they check; version control does not. `.gitignore` now tracks
`runs/*/frame_provenance*.json` generically — the 4 existing files (5.3 KB) are committed, and every future one
is tracked automatically. Second incident in three runs, and it should be the last of this kind._

| # | Task | Size | Note |
|---|------|------|------|
| ~~1~~ | ~~`tasks/TASK_876_SCORE_BASED_CHECKPOINT_RETENTION.md`~~ | S | **SHIPPED 2026-08-08** (`task876-score-emission`, `ae127b4`) — see the task banner + CHANGELOG. Remainder → D35 OPEN + a TASK_881 scope addition. _(prior note:)_ **RE-RESCOPED by D33 to EXACTLY THREE ITEMS with ZERO deleter surface** — (1) unconditional `scores.jsonl` emission at every save, (2) `flax_keep = 3` + `overwrite=False`, (3) `--ckpt-retention` defaults to `recency`. All three are already verified inside `stash@{0}`; resume from it. **Everything touching `ckpt_gc.py`'s protect-or-delete logic moved to TASK_881** (`_is_protected` descendants, TTL re-scan, failed-`rmtree` audit row, `scan_referenced_paths` `followlinks=False`, and the disputed `_order_newest_first`). Run 27 halted this row twice in that surface because D32's split kept two protection items in the "non-destructive half" — **planner error, owned in D33.** With no deleter surface left there is nothing here for a protect-scan finding to land on. Do not let the `_order_newest_first` dispute hold it: pre-existing HEAD bug, now an 881 item. **This row gates the long base-training program AND TASK_880.** |
| ~~2~~ | ~~`tasks/TASK_882_INSTRUMENT_RECONCILIATION.md`~~ | S/M | **SHIPPED 2026-08-08** (`task882-instrument-reconciliation`) — cause = a missing `--panel-frame-from-checkpoint`; the encoder is positionally universe-locked, so an unpinned run scrambles slot→asset (63/64 positions differ, 24 symbols absent). First divergence at decision 0. **Unpinned −0.6145 WITHDRAWN, and every unpinned eval_vector number with it.** Ruling deliberately narrow: imprint.py NOT vindicated in general, **+0.8394 NOT restored**, **+0.9914 CORROBORATED not restored** (same pinned instrument reads 12/12 positive on the clean-trained seeds), and **the OLD checkpoint's disagreement is UNRESOLVED — this row does NOT close `861/ref-oos-rebase`.** 875 + 879 unaffected. **PLANNER: the decisive same-window pinned measurement is pre-registered and NOT run; `runs/861_ref/frame_provenance_ref_s{0,1,2}.json` are absent locally and must be recovered first.** See the task banner + CHANGELOG. _(prior note:)_ **NEW 2026-08-07, and it BLOCKS every cross-harness comparison in the program.** Two instruments both claiming to replay per-asset argmax of the SAME checkpoint disagree by **1.6 net@21** and by an order of magnitude in terminal PV: `imprint.py` sanity says +0.8394 / PV 299,547 (3x gain), `eval_vector --operator policy-argmax` says −0.6145 / PV 41,919 (58% loss), identical window, weight env ON, 1673/1673 decisions, no bankruptcy. The window was eliminated as the cause by measurement. **+0.9914 is the program's in-window reference** (LEDGER 861/ref-rebase) and it comes from the first harness, while the 875 WF battery and the 879 corridor come from the second — one family is wrong and both are load-bearing. Lead: `position_size_pct 0.0078125` (1/128) in the imprint artifact vs eval_vector's own weight unit/cap. Mostly artifact reading + a per-decision fixture diff; find the FIRST divergent decision rather than arguing aggregates. |
| ~~3~~ | ~~`tasks/TASK_873_FIXTURE_IDENTITY_PIN.md`~~ | XS/S | **SHIPPED 2026-08-08** (`task873-fixture-identity-pin`) — tracked `fixtures/FIXTURE_IDENTITY.json` pinning sha256 **`b0f8de8b108b…`** (79310848 B), fail-loud verification at the start of all four consuming suites, an atomic `--promote-fixture-identity` re-base path, and a `make_fixture.py` guard that now **refuses unconditionally** on an existing fixture. 1 BLOCKER (the first guard failed OPEN on a missing/corrupt pin AND on an already-mismatched fixture — reproduced by all three lenses) + 1 MAJOR closed. **Intended behavioral break: a bare `make_fixture.py` against an existing fixture now always refuses.** See the task banner + CHANGELOG. _(prior note:)_ Closes the class behind run 25's decision-changing finding: the fixture is GITIGNORED, so the ritual's `git status` gate cannot see it change, and a mid-run regeneration (`b0f8de8b`→`59525bd3`) silently re-based every baseline promoted against it — recovered by luck. Tracked `FIXTURE_IDENTITY.json` pin, verified at suite start, fail-loud on mismatch, atomic promote path for a deliberate re-base, `make_fixture.py` guard. Same shape as 864's promote gate. |
| ~~4~~ | ~~`tasks/TASK_874_IMPRINT_EMITS_FRAME_PROVENANCE.md`~~ | XS | **SHIPPED 2026-08-08** (`task874-imprint-frame-provenance`, `cecfcfd`) — 3 cycles, 4-lens panel, 4 BLOCKERs + 3 MAJORs closed, 874 smoke 8/8, canon byte-identical. Same defect class as 883 from the other side: the first version emitted provenance ~380 lines BEFORE the run's own degenerate-replay refusal, and a judge drove a self-refused checkpoint through `scorecard.py` to exit 0 to prove it. _(prior note:)_ `imprint.py` consumes frame provenance but never emits it for the ckpt it writes, so grading a fresh imprint is (correctly) REFUSED and needs a hand-derived file — it cost a round-trip on 2026-08-03 and the planner hand-derived three (`runs/861_ref/frame_provenance_ref_s{0,1,2}.json`, order-exact asserted). Emit on write; refuse to emit on a pinned-run mismatch; verify the three existing files match what the emitter would produce. The 863/865 gate must NOT be weakened. |
| 4 | `tasks/TASK_883_EVAL_VECTOR_UNIVERSE_IDENTITY_FAIL_LOUD.md` (**RE-CUT as 883R by D36**) | S | **HALTED run 29 → RE-CUT 2026-08-09. Resume `stash@{0}` BY MESSAGE, not index.** D36 cut this by CERTIFICATION CAPACITY, not by lane — the run-29 proposal to ship "the verified checkpoint-provenance half" does not work, because **hole 1 lives inside that half** (orbax-step-subdir spelling defeats the sibling lookup, and a fully reversed pin then grades to exit 0 with `universe_pinned: true`). **Ship only what can REFUSE or DISCLAIM; ship nothing that can BLESS** — the same re-cut that finally landed 876 under D33, applied to labels. IN: gate (i) refusal-before-panel (`strace`-proven: 0 DB opens, 0 ckpt opens) · gate (iii) `--allow-unpinned-universe` + the NEGATIVE stamp · gate (iv) byte-identity on both pin paths · the ORDER-exact mismatch refusal wherever provenance IS found · **NEW: the realized-panel equality assertion** (refuse unless `build_panel`'s returned symbol list equals what was requested, order-exact — small, fail-closed, and it is hole 2's mechanism). **OUT, and this is the load-bearing constraint: `universe_pinned: true` MUST NOT EXIST.** Every artifact carries `false`/`unverified` until TASK_886. Nothing consumes the field yet, so the cost is zero. `ledger_row` carries the caveat INLINE (D36 Q2 — the caveat must travel with the pasted text). |
| 5 | `tasks/TASK_886_UNIVERSE_PIN_CERTIFICATION.md` | M | **NEW 2026-08-09 (D36 option (a)).** Binds the universe claim to the checkpoint actually LOADED and the panel actually RETURNED — argv is not evidence. Closes hole 1 (resolve provenance from the RESOLVED ckpt dir, so every spelling reaches the same verdict) and re-introduces the positive claim, earned. Replaces the boolean with a mechanism-typed `universe_pin` record, because the run-29 measurement proves `universe_pinned` conflates universe IDENTITY with PANEL CONSTRUCTION (same ckpt, same 8-symbol order: **−1.796** pin-file vs **−1.866** `--symbols`, both stamped `true`). **Answers the program call: the two mechanisms are NOT cross-comparable; comparability holds within a mechanism, and the artifact must say which.** |
| 6 | `tasks/TASK_877_XBT_ALIAS_FOLD.md` | XS/S | **Same defect class as D31 (a1), still live — planner-measured 2026-08-05.** The resolver normalizes SLASH-ONLY and deliberately does not fold `XBT`→`BTC`; `XBTCHF` (id 1175) holds **92,639 real rows inside the exact 2025-03-31→2025-10-01 gap** where `BTCCHF` (frame slot 23) has 697, and the 48,780-row overlap agrees to **max rel close diff 0.0**. So one of the 5 "genuine sparse" slots is recoverable data, not market absence: the identity repair is incomplete and the frozen-interior residue is overstated. Explicit alias TABLE (not a blanket replace), same fail-loud agreement gate, re-derive the disclosure numbers. |
| 7 | `tasks/TASK_878_FAILED_FILL_SEMANTICS.md` | S/M | **OPERATOR-RULED 2026-08-05 (+amendment) — resolves both open questions.** A no-print bar is an order PLACED THAT IS UNFILLED — not illegal, and not dead either: it could still execute on a later step, so execution is a **COINFLIP** (operator's simulation choice; the pending-order queue that holds it ~5 steps is the heavy variant and stays OUT of scope as its own future row — the per-step flip is its geometric special case). `p` comes from the slot's trailing print density (causal), not a magic number; the flip must be a seeded path-independent hash of (eval_seed, decision, slot) or every parity gate in the repo dies. A 3-arm ablation (never-fill / coinflip / always-fill) answers the operator's open question of whether flat-before-move cells carry SIGNAL rather than noise, instead of assuming. valuation marks at the last valid price (architecture), execution requires a real print, a failed fill changes position/cash/fee by nothing, and **the policy's INTENT is preserved** — today's 868 (a2) mask rewrites the BC target to HOLD, teaching "HOLD was intended" when the truth is "could not fill" (the 78.8%-class hazard). Force-HOLD survives only for zero-coverage slots (nothing quotable ⇒ nothing attemptable). PV conservation must stay exact. |
| 8 | `tasks/TASK_881_CHECKPOINT_PROTECTION_MODEL.md` | M | **UN-GATED and re-ordered by D35 (2026-08-08).** Was Gated behind 880; 880's entire safety argument now runs THROUGH 881's path-first reader, so that ordering was a cycle. Owns: the score reader that resolves by `path`/`step` instead of digits-in-the-basename (D35 — `checkpoint_<step>` and `checkpoint_epoch_<N>` currently read UNSCORED forever, only `snapshot_ep<N>` round-trips), the `_is_protected` descendant rules, the TTL re-scan, the failed-`rmtree` audit row, the `followlinks=False` BLOCKER (`jax_ckpt_fast`/`jax_ckpt_easy` ARE symlinks to `/mnt/scratch`), and the disputed pre-existing `_order_newest_first` (needs a third independent measurement). Resume from `stash@{0}`'s stripped half — deliberately RETAINED for exactly this. **Ranked LAST on purpose:** nothing is at risk today (unreadable ⇒ UNSCORED is fail-safe, the `--apply` freeze STANDS, srv holds 162 GB free), and this lane has consumed 9 factory cycles across three runs while the ladder waited. |

_Sequencing (2026-08-08, supersedes the 2026-08-06 note below): **874 first** — XS, and it is the artifact 883 makes mandatory. **883 second** — it closes the fail-open that manufactured the 882 crisis, and every measurement the ladder is about to take runs through that harness. **877 third** — it shrinks the sparse-slot set, which is exactly the surface 878 then re-specifies (878's own out-of-scope says do 877 first). **878 fourth and alone** — it rewrites eval-core execution semantics with every parity gate in the repo riding on it. **881 last** — deliberately. It is real work and it is not urgent: unreadable ⇒ UNSCORED is fail-safe, the `--apply` freeze STANDS, srv holds 162 GB free. Four of the last five planner errors were boundary calls inside this one lane; ranking it last is the structural correction, not a demotion of the work._

_(2026-08-06 note, superseded — kept because its 876/873 half is what actually happened:)_ _Sequencing (2026-08-06): **876 first** — emission gates the long base-training program, and it
touches the training save path that no other Ready row touches. **880 second** — it lifts an
operational freeze on a tool that deletes remotely, and the long run will want that tool. **873
third** — it changes what every suite checks at start-up, so everything after it is measured
under the pin. **877 fourth** — it shrinks the sparse-slot set, which is exactly the surface 878
then re-specifies (878's own out-of-scope says do 877 first if it is scheduled). **874 fifth**
(XS, independent). **878 last and alone** — it rewrites eval-core execution semantics with every
parity gate in the repo riding on it; the factory deliberately deferred it in run 26 to finish
one row properly, and that judgement is ratified here._ _Original note:_ 872 first (live path). 873 and 874 touch different surfaces from each other and from
872 — but 873 changes what every suite checks at start-up, so take it BEFORE 874 and re-run 874's
suites under the pin. Serial is safest; all three are XS/S._

_Recommended, not yet specced (XS, planner-found 2026-08-06, has now cost TWO launches):_
_(0) `tools/deploy/run_queue.sh` does not activate the venv, so a bare `python` in a manifest
resolves to **python2** on gpu-server and every job dies in ~30 s with a PEP-263 SyntaxError
(the 861 chain, then the 879 rung-3 launch). Resolve `trader_env/bin/python` in the runner —
or refuse a manifest whose command starts with a bare `python` — so the trap cannot recur._

_Recommended, not yet specced (both named in shipped banners, neither urgent):_
_(a) `mcts_trader/data/database.py` — the last known unaddressed RW reach at the canon layer
(aiosqlite, real live-write path; needs a dual-connection refactor, beyond XS — 870 remainder).
(b) `_ro_connect`-style helpers still coexist in three places (shared / `837_walkforward` /
`trader/data.py::_connect`) — "one implementation" holds for 870's touched surface, not repo-wide._

_Shipped 2026-08-02/03 factory run 24 (the D31 re-base wave, strictly serial):_
- _**TASK_866** (`task866-weight-on-pin`, `ac8dfe1`) — weight-ON parity pin reads the tracked 832 baseline as its own-config oracle; the 864 incident class closed. See the task banner + CHANGELOG._
- _**TASK_867** (`task867-861-baseline-promote-port`, `d670d57`) — the 864 promote gate ported to both 861 baselines; the `--quick` silent downgrade is a named REFUSAL; leg `vii_baseline_pins` pins the COMMITTED baselines (tamper → FAIL pre-leg). See the task banner + CHANGELOG._
- _**TASK_868** (`task868-asset-identity-repair`, `3388a0c` + 861 promote `a687aff`) — D31 (a1) load-time alias union (NO DB writes, fail-loud agreement gate, disclosed) + (a2) causal frozen-cell open-gate (whole-action force-HOLD; D30 fill still values; PV exact mask ON). Frozen-interior 2,013 → 1,011 cells; honest not-real share 10.67% → 9.73%. Medium panel: 1 MAJOR (fixture-regen blast radius unattributed) + 4 MINORs closed and re-verified. See the task banner + CHANGELOG._
- _**TASK_869** (`task869-rebase-wave-semantics`, `3e0baa9` + `9a5b0b2` + `894ab36`) — D31 (a3): `_to_epoch` → UTC everywhere (TZ-invariance byte-proven) + daily teacher floor masked (measured inert, defence-in-depth). Release round caught + fixed a REAL leg-vi break (stale constants + 150-iter collapse → 250). Final re-verify: 861/863/865 ALL LEGS PASS. **Wave CLOSED — references un-gate planner-side.** See the task banner + CHANGELOG._

_(prior Ready rows, now shipped:)_

_Recommended new planner rows from run 23 (not yet specced, listed so they are not lost):_
_**(D) The canon panel path still opens `trading_data.db` READ-WRITE** — `mcts_trader/env/vectorbt_env.py:515` and `:731` use a plain `sqlite3.connect`, reached via `build_panel` → `create_jax_env_historical` on the DEFAULT unpinned path (and by the equivalence gate's FLOOR leg at `eval_vector.py:3152`). TASK_865 routed all 11 of its OWN sites through `_ro_connect`, and the orchestrator verified the fix (a hot-journal db is left byte-identical; the probe raises `attempt to write a readonly database`; a nonexistent path is refused with nothing created) — but the canon env loader is a wider surface than 865 and was deliberately left alone. The exposure is real and demonstrated in principle: SQLite performs hot-journal recovery on the first READ through a read-write handle, no write SQL involved, on a 24 GB asset whose only backup predates the 858 backfill. **XS/S, fail-closed, no numerics risk** (a `mode=ro` handle cannot change a SELECT's result). Recommend: route both loaders through the same helper, or refuse a protected `--db` on any RW path._
_**(A) Daily-framed teacher floor walks UNMASKED** — same defect class as D30 rider (a), but masking it moves shipped daily-pinned numerics, so it needs its own ruling rather than a factory fix. **(B) `_to_epoch` resolves `--start/--end` in LOCAL time**, so a benchmark window's extent depends on which machine ran it (measured: a 308-row CEST offset on the OOS window; the reason several counts differ by one between parties — 1734 vs 1735 panel rows, 2 vs 3 stale tails, 90 vs 91 daily fabricated rows). A determinism hazard for a read-of-record operator across workstation and `gpu-server`; changing it re-bases every existing window number, so it is a planner call. **(C) fixture-native interior-gap coverage** — under the shipped (bar-window) definition the fixture has no interior gap, so that proof currently rides on a synthetic DB; `fixtures/make_fixture.py` would need a multi-bar hole._

_Shipped 2026-07-31 factory run 23:_
- _**TASK_865** (`tasks/TASK_865_MINUTE_FRAME_PIN.md`) — minute-framed frame pin, D30 opt (i). See the task banner and the CHANGELOG entry. **Remainder → planner: D31 OPEN, gates the 861 real leg.**_

_(prior Ready row, now shipped:)_
| ~~1~~ | `tasks/TASK_865_MINUTE_FRAME_PIN.md` | S/M | **RESUMED FROM `git stash`** ("TASK_865 minute frame pin — REVIEWED, BLOCKED on D30" — 4 tracked edits + the untracked 978-line smoke; verified to apply cleanly on HEAD). D30 RULED opt (i): forward-fill the stale tail + disclosure, uniform on BOTH pin paths. Amended scope in the task's D30 addendum, all mandatory: PV-conservation smoke leg FIRST (shown RED on stashed behavior, GREEN after); riders (a) masked `compute_teacher_floor` / (b) `--require-active-slots` refusal / (c) scorecard frame-source plumb-or-refuse; daily-pin disclosure parity (numerics unchanged, additive fields only); coverage gate re-specified to CELL level (≈9.6% expected, named slots); every VOID numeric re-derived + fixture baselines re-promoted through the 864 gate. |
| ~~2~~ | ~~`tasks/TASK_866_WEIGHT_ON_PIN.md`~~ | XS/S | _(superseded — now Ready #1 above; 865 shipped so the stash no longer owns the surface)_ FILLER, **SERIAL after #1 (865's stash owns this surface until it lands)**. Close the CLASS behind the 864 incident: `CANON_832_PINS` is a weight-OFF oracle guarding a weight-ON baseline — nothing pins the weight-ON numbers, which is how 846 moved them silently for ~80 commits. Every committed baseline gets an oracle of ITS OWN configuration; cross-config check = named REFUSE; mutation-tested. |
| ~~3~~ | ~~`tasks/TASK_867_861_BASELINE_PROMOTE_PORT.md`~~ | XS/S | _(superseded — now Ready #2 above)_ FILLER, **SERIAL after #2 (same surface)**. Port 864's promote gate (recipe-pinned, validity-gated, atomic, dirty-refuse) to `chosen_vectors_fixture.json` + `results_fixture_smoke.json`; kill the `--quick` silent seed-downgrade; make `smoke_e2e.py:428/459` check the COMMITTED baseline (tamper test mandatory — today it passes against arbitrary content). **Path drift: the task file says `experiments/832_linear_imprint/861/`; the real path is `experiments/861_search_teacher_imprint/`.** |

_Sequencing note: these rows touch `experiments/832_linear_imprint/` and
`experiments/861_search_teacher_imprint/` — SERIAL queue, take in rank order. All GPU legs remain
planner-scheduled; the amended 861 chain (minute-pinned frame, 3 imprint seeds, median > +0.839 with
≥2/3 above — see the D29 + D30 addenda in TASK_861) launches only after 865 ships verified
(D30 RULED 2026-07-30, planner opt (i); leg-1 hard checks + masked floors + leg-4 frame-source
gate per the D30 addendum)._

_Shipped 2026-07-29 factory run 21:_
- _**TASK_863** (`task863-imprint-frame-pin`, `609d6d3`) — `--panel-frame-from-checkpoint` on
  `imprint.py`, calling `eval_vector`'s proven 853 validator/framed loader rather than re-deriving
  them (`eval_vector.py` byte-unmodified); 861 provenance gate now fail-closed BOTH directions,
  same-membership reorder refused. **OFF bit-identical proven five independent ways** (2 orchestrator
  re-proofs at matched `_REPO` with identical `--out`/`--ckpt-out` → stdout byte-identical, 170
  leaves/0 differing; fixer's mirrored tree 0-of-759 + 0-of-545; regression judge's **30-leg** legacy
  battery, zero rc/artifact divergence; fresh judge's mirrored-tree control). Smoke **8 → 11 → 13
  legs**; selftest 12/12; 861 e2e 9/9. 3-member panel + a **FRESH fail-closed judge** over the fix
  pass: **1 MAJOR + 6 MINOR + 3 NIT closed over 2 fix rounds**, every new guard demonstrated to FAIL
  when disabled. MAJOR (pre-existing, fixed pin-gated): a degenerate pinned run exited **0** with a
  green-looking artifact (null Sharpes beside `all_losses_finite: true`, no marker) whenever
  `--smoke-assert` was off — the default for any non-repo-fixture db, **i.e. the real leg**. Also
  closed: masked-cell dilution of every reported scalar (0.8929 vs 0.8571 active-only); a
  false-AND-vacuous `panel_source` pin (2 judges independently); a FALSE "flat-zero features"
  disclosure (896/6048 cells at a constant 0.2) now **asserted** by the smoke; three malformed-frame
  paths; and `panel_frame: {}` as a real **fail-open**. The force-HOLD mask had **no regression guard
  at all** (inert on the tv path + zero partial slots in the smoke) — now boundary-asserted.
  **Spec correction (binding): D28 + TASK_861 line 179 wrongly state `imprint.py:620` "only WARNs" —
  it is a hard `RuntimeError`.** **Remainder → planner, BLOCKED on D29.**_

_Shipped 2026-07-28 factory run 20 (861 + 862 parallel lanes, disjoint surfaces as planned):_
- _**TASK_861** (`task861-search-teacher-imprint`) — `--collect-chosen-vectors` (eval_vector) +
  `--teacher-vectors` (imprint); both flags OFF **bit-identical** (orchestrator-reproduced vs
  `HEAD`: 0 differing leaves on both modules). 3-member panel, **2 fix rounds, 5 MAJOR closed** —
  the pinned +8.024 operator was ungated (3 verifiers imprinted garbage teachers with green
  gates, and the falsification metric scored garbage HIGHER than the real teacher); no payload
  integrity (`x−0.5` codes swapped a +13.893 teacher for −0.649); empty-string flags failed OPEN
  (one clobbered the tracked parity baseline); the two pre-registered legs were mutually
  un-runnable; and — found by brute force — the new replay guard is a TRAJECTORY not a TARGET
  check: **315/400 cells (78.8%) admit trajectory-inert substitutions**, so 78.8% of BC targets
  could be rewritten with net@21 bit-identical and every gate green (closed via a payload-vs-header
  action-composition check). **Remainder → planner, BLOCKED on D28.**_
- _**TASK_862** (`task862-imprint-scorecard`) — `experiments/835_stage4_eval/scorecard.py` + the
  canonical 7-window manifest + smoke; NEW FILES ONLY, zero diffs to existing tracked files
  (re-confirmed). 2-member panel, **2 fix rounds, 3 MAJOR + 1 latent-MAJOR closed** — any
  non-`WindowFailure` exception destroyed the whole artifact (a non-numeric floor value killed a
  completed battery, no json/no md, contradicting its own "never silently skipped" contract); the
  operator block was asserted-not-observed (~25 knobs unpinned — a top-m battery would emit a
  status-OK card headed "848 read of record"); the recovery path re-ran the call that had failed;
  and the write sat outside every guard (truncated unparseable JSON). Also closed: ambient
  `JAX_PLATFORMS` silently selecting the backend while the header said `null`. **Decision-changing
  finding → D27. Remainder → planner: canonical manifest on `ckpt_imprint_full` (workstation).**_

_(prior Ready rows, now shipped:)_
| 1 | `tasks/TASK_861_SEARCH_TEACHER_IMPRINT.md` | M | **D26 RULED (operator, 2026-07-27): "prepare dark factory tasks" — opt (a).** Marry the two proven pieces: the +8.024 true-env-leaf search operator (PINNED verbatim per `runs/832_e5/results_e5inv854_db_pw0_*.json`) as an OFFLINE teacher × the exact 832 weight-native imprint recipe (+0.839/+0.898-validated). Two flag-gated extensions: `--collect-chosen-vectors` on the search walk (`eval_vector.py`, pure recording — walk numbers identical ON/OFF) + `--teacher-vectors` on `imprint.py` (hard chosen-vector targets, fail-loud provenance gate; NOT visit-distribution CE — the ExIt2 collapse mode). Falsification gate must be high-SNR (the 860 lesson — no small-n held-out quantities). OFF = bit-identical both modules. **Real bars = planner post-ship (pre-registered in the spec): teacher pass on the pinned `--end 2025-10-04` panel (1673 dec, flagged) → imprint → sanity replay > +0.839 → 858-extended OOS + WF quarters.** |
| 2 | `tasks/TASK_862_IMPRINT_SCORECARD.md` | S | FILLER (parallel-safe). Codify the imprint's D17 battery into one command + one fail-loud scorecard artifact: DEC-2 regime windows (floors from `runs/831_ext/*.json`) + OOS window + WF quarters via the 848 policy-argmax read-of-record configuration. **NEW FILES ONLY — must not edit `eval_vector.py`/`imprint.py` (861's surface); merge-safe by construction.** No silent window skips; schema documented (the eventual gate-ladder submission cites it). Real leg = planner (workstation, DB source of record). |

_Independence note: 861 edits `eval_vector.py` + `imprint.py`; 862 adds
`experiments/835_stage4_eval/scorecard.py` + a windows manifest and is FORBIDDEN from editing
861's files — zero file overlap, parallel lanes safe. Neither row trains anything; all GPU legs
are planner-scheduled. Search-leaf lane (857/860 distill line) is PARKED per D26 — no rows._

_Shipped 2026-07-26 factory run 19:_
- _**TASK_860** (`task860-reward-head-codistill`, `736fdc6`) — `--codistill-reward` (OFF =
  shipped-857 bit-identical, 615/615 leaves diff 0): reward head trained on true-r0 counterfactual
  targets alongside the value imprint. Panel BLOCKER: the shuffle-falsification gate was fixture-noise
  (deployed-Q at n=12) AND the newly-trainable reward head defeated the frozen-reward assumption —
  replaced with a high-SNR train-set fit gate (~n=972, floor +0.45, 0.38 cushion), 8/8 seed sweep
  re-verified. **Remainder → planner: real 3-seed co-distill + corridor (RUNNING).**_

_(prior Ready row, superseded:)_
_| 1 | `tasks/TASK_860_REWARD_HEAD_CODISTILL.md` | S/M | **D24 RULED (operator, 2026-07-26): "seed co-distill."** Extend the shipped 857 recipe: emit true-r0 targets, `--codistill-reward` unfreezes the `dynamics_dense2` reward projection (λ-weighted reward CE alongside the value CE); r_model calibration column added to the instrument. OFF = shipped-857 bit-identical. Real bars = planner post-ship: deployed-Q ≥ +0.5 → the same corridor (floor +0.424; value-only best-seed reference −4.23). |

_Context: 858 real ingest DONE (8.28M rows, 62/64 → 2026-03-31); first binding OOS reads landed —
imprint +0.898 net@21 OOS (Δ+2.33 over the on-window floor, LEDGER 832/E5-oos-real); 856 top-m NOT
confirmed OOS. The imprint is the deployable-artifact frontrunner; its walk-forward battery is
planner lane._

_Shipped 2026-07-24 factory run 18:_
- _**TASK_858** (`task858-minute-backfill`, `19f955c`) — `tools/backfill_minute.py`, the single
  sanctioned append-only write path to `trading_data.db`. Panel closed 2 BLOCKER + 2 MAJOR in the
  guard architecture (commit-before-guards → atomic; false-firing append-only proof → subset proof;
  fail-open backup gate → hardened; 11/64 unreachable dump spellings → `_canon_key`). Real ingest
  correctly STOPPED (no dumps on disk). **Remainder → operator/planner: acquire Kraken OHLCVT
  quarterly dumps (Q4-2025…Q3-2026), fresh DB backup, run on the WORKSTATION.**_
- _**TASK_859** (`task859-eval-activity-metadata`, `d32c64a`) — additive `activity` block in eval
  JSONs; no panel findings. Caveat: `n_long`/`n_short` = action-code intent, not net-position
  direction (a Sell closing a long buckets as "short")._

_Independence note: 858 touches the data pipeline (`tools/`), 859 touches `eval_vector.py` result
metadata — zero file overlap with each other AND with the in-flight 857 server runs (which only
read the DB and write to `runs/`). 858's real-ingest leg (iii) runs on the WORKSTATION (the DB
source of record lives here); do not run it on the server DB._

_Context: the 857 binding reads (3 distill+corridor chains) are running on gpu-server overnight;
their verdicts + the D24 fork are planner lane, not queue rows._

_Shipped 2026-07-24 factory run 17 (D23 queue cleared):_
- _**TASK_856** (`task856-decision-rule-probe`, `861ea63`) — `--decision-rule {argmax,threshold,top-m}`
  (+ `--decision-tau`/`--decision-m`) on the 848 replay operator + the `net_dependent` grading-hygiene
  fix (fail-loud on zero-learned-component operators). Panel MAJOR: top-m originally vacuous in the
  collapse regime (assigned full argmax incl. HOLD) — fixed to best NON-HOLD on selected assets,
  mutation-tested. Default bit-identical. **Remainder → planner: the real rule-grid sweep (scheduled).**_
- _**TASK_857** (`task857-vhat-leaf-distill`, `15a80f5`) — `experiments/857_vhat_distill/distill.py`
  (counterfactual dataset via the 851 true-env leaf → value-head imprint, dynamics+value only) +
  `--leaf-eval vhat-distilled` (deployable model-leaf on the distilled ckpt; `net_dependent: true`).
  3 fix rounds; calibration instrument hardened (held-out per-state Pearson; deployed-Q leg;
  `--shuffle-targets` falsification hook). **Decision-changing finding → D24 (OPEN): v̂-alone improves
  (+0.233→+0.342 fixture) but deployed-Q (r_model+γ·v̂) does NOT lift — frozen 832 reward head is a
  poor r0 estimator (corr≈0.27). Remainder → planner: real distill + corridor read (scheduled).**_
- _**TASK_855** (`task855-flag-hygiene-minors`, `a19c528`) — `value_shift` inert-combo fail-loud
  (sibling to 854's `value_scale`), ST11 pins both; 835 smoke baseline refreshed (additive keys only,
  tiers-SHA unchanged)._

_Shipped 2026-07-22 factory run 15:_
- _**TASK_853** (`task853-inactive-mask-oos-panel`, `c8ee701`) — `--panel-frame-from-checkpoint`:
  fixed 64-slot daily panel in checkpoint order, zero-coverage + pre-listing slots INACTIVE and
  dual-disclosed, fail-loud on identity scrambling; OFF bit-identical (SHAs), selftest 10/10.
  Verification caught a REAL LEAK (partial slots tradeable in a fabricated pre-listing region —
  3 verifiers independently) — fixed to flat-zero + price-validity force-HOLD. Plus the D20 audit
  (`5bbbd26`, read-only) independently matching the planner's concurrent ruling. **Remainder →
  planner: 853 numeric leg (+0.839 ±0.06 round-trip, needs the imprint's 64-symbol provenance
  JSON) then the binding OOS benchmark.**_

_Shipped 2026-07-21 factory run 14:_
- _**TASK_851** (`task851-true-env-leaf`, `41561d4`) — `--leaf-eval {model,true-env}` on the
  Sampled-MuZero search (D19 opt (a), the E1 +5.29 operator); default `model` bit-identical.
  Fixture inversion (i) PASS (+6.801 vs −6.778 net@21) + E1-consistency anchor (ii) PASS
  (fresh-init +2.198 true-env vs −10.287 model, floor −3.437). Medium panel, 2 rounds: **2 MAJOR
  found + fixed + independently re-verified** (unique-per-env-prices dispatch crash; unreproducible
  headline numbers — root-caused to a concurrent TASK_852 edit transiently mutating the shared
  `eval_vector.py` file mid-review, now pinned via LEDGER + committed artifacts). Zero
  regressions, selftest 10/10. Real binding leg + Amendment-6 ExIt re-try = planner, post-ship._
- _**TASK_852** (`c105038`, finding-style commit — not tagged, this is a STOP not a ship) —
  diagnosed + partially fixed the daily/minute panel divergence (LEDGER 850/eqgate). Lookahead
  hypothesis REFUTED; root cause = densification of zero-coverage assets (78% of the gap).
  Fix (active-mask filter + fail-loud pad guard) verified — but FLOOR leg still fails its own
  tolerance (residual Δ0.0555 vs 0.02) ⇒ **STOP, DECISIONS D20 seeded OPEN**. 4-member panel (the
  task's own mandated fail-closed lens + correctness/regression/flow-verifier) caught a real
  **BLOCKER**: the fix's active-mask filter reactivated a dormant duplicate-cycling bug in
  `create_daily_panel`, reproduced on TASK_852's own scheduled command (28/64 dropped → would
  have double-exposed 28 symbols) — 3 of 4 panel members found it independently; fixed to fail
  loud instead. All numbers independently re-verified by the orchestrator against the real DB._

_Shipped 2026-07-18 factory run 12 (both D17 instrument builds):_
- _**TASK_847** (`task847-policy-prior-proposals`, `873effe`) — `policy-prior` scheme; canon pin bit-identical, selftest 8/8, mechanism leg 40/40. **Real-panel numeric leg FAILED post-ship (−7.839 vs ≥+0.74, LEDGER 832/E5-inversion) — the fault moved to leaf-eval selection (⇒ 849); the scheme stays shipped/inert (default unchanged).**_
- _**TASK_848** (`task848-policy-replay-instrument`, `371e7c5`) — policy-argmax operator + OOS window + on-window teacher floor; 1 BLOCKER (date-coercion in-sample leak) + 2 MAJOR fixed. **Its fail-loud guard immediately caught the STALE SERVER DB (daily ends ~2025-10-04 vs local 2026-07-06) on the first OOS attempt — the OOS read moved to the workstation.**_

_Shipped 2026-07-17 factory run 11:_
- _**TASK_846** (`task846-weight-fee-headroom`, `c2f1ba5`) — weight-budget fee headroom (D14 opt (a)):
  `budget × (1 − 2·fee_rate)`, closes the all-short fee16 full-wipe; fee0 BIT-IDENTICAL (3 pins MATCH
  incl. `6293668f…`); 5 D14 regression tests, selftest 7/7. Acceptance (iv) went through **D16**
  (fixture teacher-gate Δ1.576 → held → planner's binding 64-asset real-panel read Δ=0.0008 ≪ 0.02 →
  RULED ship-as-is; (a′) rejected unsound). Un-gated the ExIt continuation + all phase-2 fee16._

_Shipped 2026-07-16 factory run 10 (843 → 845 in sequence, the D12-cleanup → E5-compat chain):_
- _**TASK_843** (`task843-weight-incremental-netting`, `ed27849`) — weight-path incremental netting + snap-to-flat
  (D12): delta-only re-trim (entry preserved) + snap-to-flat kills the short over-tax (185.2→≈canon) and
  slippage dust. Canon-OFF `48e1f467…` + budget `4f8b692e…` pins UNCHANGED; weight-ON/fee0 RE-BASED
  `1c50d58c…`→`6293668f…`. Escalated 4-member panel: **1 BLOCKER (winning-short survivor under-reservation
  → all-short margin wipe at fee0 from a valid state; orchestrator-crossval-CONFIRMED vs pre-edit) FIXED**
  (footprint-fraction reservation `(2E−P)/P` clamped ≥1; re-run fail-closed judge confirmed the fee0 hole
  closed, 3,840-case fuzz + sweep 0 hits). Acceptance (i)–(iv) PASS; E1 fee16 edge +2.1667 (from 842's −0.134).
  **Remainder → DECISIONS D14** (PRE-EXISTING fee16 all-short overdraft wipe, stash-confirmed 842-identical)._
- _**TASK_845** (`task845-imprint-weight-compat`, `4ac7721`) — 832 imprint runner threaded through the weight
  env + teacher-expressibility gate; ZERO canon diffs. Medium panel + fresh fix-judge: **1 MAJOR (weight-native
  remediation mapping was DIAGNOSTIC-ONLY — runner always imprinted the order-code teacher) FIXED** via
  `--imprint-mapping {order-code,weight-native}` (default order-code = byte-identical) + **2 MINOR guards**
  (fail-closed flag-coupling assert; None-guard the fee16 bankrupt-teacher WARN crash). Expressibility gate
  REPORTED, fixture PASS/FAIL INCONCLUSIVE (losing slice) → **DECISIONS D15** (structural sizing divergence
  under default w_unit/w_max; binding read = real-panel run per DEC-2). Un-blocks the scheduled E5 runs._

_Shipped 2026-07-14 factory run 9 (solo, read-only diagnostic):_
- _**TASK_844** (`task844-short-bias-probe`, `b53ecfe`) — short-bias postmortem instrument (mirror-book
  value / flat-book marginals / leg decomposition + fixture smoke), zero canon diffs, D13-independent.
  Medium panel: regression clean; **2 MAJOR FIXED** — probe-1 mirror-Δ was uncalibrated (independently-
  init BUY/SELL embedding columns ⇒ a FRESH model trips |t|~4; now gated on a matched fresh-init null
  z-stat, smoke robust across seeds 0/7/8 z +0.10/−0.85/−1.12) + probe-3 mislabeled an all-close SUPERSET
  as the EXACT reward_attributable term (relabeled + action-attributed subset now gates credit_misassign).
  MINOR (smoke tol false-alarm) fixed; NIT (open_count counts limit-fill opens) → planner. Real numbers =
  the SCHEDULED planner run (p1c s0/s1(+s2) + p1b s1 control, weight-env flags)._

_Shipped 2026-07-12 factory run 8 (solo, canon-adjacent fee fix):_
- _**TASK_842** (`task842-weight-close-fee`, `30013a4`) — canon-equivalent explicit close/open-short fee
  on the TASK_840 weight re-trim (was undercharging ~16bps/close, D9). Every term ×fee_rate ⇒ **fee0
  bit-identical** (THIRD pin weight-ON/fee0 `1c50d58c…` captured pre-edit at ba53294; canon `48e1f467…`
  + budget `4f8b692e…` also MATCH) ⇒ **RUNNING phase-1c cost0 untouched**. Medium panel: correctness+
  regression session-limit-truncated (trending clean, orchestrator-substituted); flow-verifier re-run
  CONFIRMED **no undercharge** (weight ≥ canon per re-trim — the short RT is an OVER-charge from 840
  close+reopen/residual geometry, fee-independent, → **D12**) + canon-faithful arithmetic. NIT: within-1e-6
  is float32-summation fragile (float64-exact). **Un-gates phase-2 fee16.** 840 (iii) E1 fee16 edge honestly
  moved +0.235→−0.134 (the +0.235 was flattered by the fixed undercharge)._

_Shipped 2026-07-11 factory run 7 (840 + 841 parallel lanes, disjoint surfaces):_
- _**TASK_840** (`task840-weight-rebalance-step`, `31a545f`) — target-weight rebalance step (H6 env,
  D8 opt (c)): flag `--vector-weight-rebalance` default OFF (canon + 839-budget both bit-identical,
  TWO pins `48e1f467…`/`4f8b692e…`). Full high-tier panel: **1 BLOCKER (held-winning-short budget
  under-reservation, crossval-CONFIRMED) + 1 flow-verifier-CONFIRMED held-leg drift bleed → BOTH FIXED**
  (whole-book posval/PV reservation + cap-clip re-trims every drifted leg → cash≥0, Σ|w|≤1, margin no-op).
  Acceptance (i) survives full 1697-step walk (invariant gate cash≥0/Σ|w|≤1) / (ii) sticky-short PASS;
  **(iii) E1 edge +0.235 (flipped from −0.23, below +0.436) = FINDING, non-gating**. Residuals → **D9**
  (fee undercharge, gates fee16 phase-2) + **D10** (expressibility). Unblocks phase-1c re-train._
- _**TASK_841** (`task841-checkpoint-wedge`, `52b155f`) — `run_queue.sh --stall-timeout` wedge watchdog,
  **default OFF (opt-in)** per invariant #1 (unarmed = byte-identical to HEAD; CSV-declaring jobs only when
  armed). Medium panel: **1 MAJOR (default-armed false-kills healthy CSV-less/`--name=` jobs at ~55min)
  double-CONFIRMED → FIXED** + 3 MINOR. Root cause characterized (orbax post-save GC vs RLock), STOPPED at
  out-of-scope trainer surgery; `--debug-stall-dump` wired for the next real wedge. Default → **D11**._

_Shipped 2026-07-09 factory run 6 (solo, canon-adjacent env fix):_
- _**TASK_839** (`task839-budget-vector-step`) — the env-wall fix: flag-gated (`--vector-budget-step`,
  default OFF = canon bit-identical, pin `48e1f467…` MATCH pre-edit 39ae0bf) budget-constrained
  equal-split sizing in `step_vector` (RATIFIED D7 opt (a)). Full high-tier panel: **1 MAJOR
  crossval-CONFIRMED → RESOLVED** ("≤cash by construction" holds only at PV≈cash → honest wording +
  PV>cash test + **DECISIONS D8** seeded; ratified formula untouched) + security MINOR (symlink
  write-guard realpath) & fail-closed MINOR (probe exit code) FIXED. Acceptance: OFF bankrupt step 0 /
  **ON 4/4 survive ≥64 steps zero-overdraft**; E1 byte-identical (narrow-regime cap non-binding).
  **Unblocks the line**: phase-1 RE-RUN under the fixed step → phase-2 fee16 + 832 imprint-sanity +
  834/836 re-runs. (NB: phase-2 gate row below still lists TASK_839 as a gate — now SATISFIED; D8 is
  the residual planner call on PV-relative vs cash-relative before flipping the flag ON.)_

_Shipped 2026-07-09 factory run 5 (836+837 parallel lanes, then 838 canon-adjacent solo):_
- _**TASK_836** (`task836-e4-rank-probe`) — E4 value-vs-momentum rank probe; 0 MAJOR; 3 MINORs fixed
  (honest baseline labels + true walk-PV mirror, overlap p-caveat, degenerate-value reading)._
- _**TASK_837** (`task837-walkforward`) — R4 walk-forward rails; `build_pit_universe` verified NON-PIT
  → module-side PIT; 3 MAJOR gate-coverage roots fixed (POLICY-PIN to pre-registered constants,
  eval-file identity, de-confounded 5-negative battery + no-op control)._
- _**TASK_838** (`task838-pernode-resample`) — flag-gated per-node resampling, OFF bit-identical
  (pin 22b0c7f5 MATCH, provenance independently reproduced); 1 MAJOR fixed (835 threads the flag —
  train/eval operator symmetry); **acceptance FINDING: OFF > ON under an uninformed policy
  (Δnet@21 −2.79) — any ON run must re-measure with a trained policy first; default stays OFF**._

_Shipped 2026-07-08 factory run 4 (parallel-lane batch, all zero-canon-diff instrument modules,
each medium-panel verified with adversarial crossval + fix re-verify):_
- _**TASK_834** (`task834-e3-runner`) — checkpoint E3/DEC-1 runner; 2 MAJORs found+fixed (legality
  mask on both proposal legs; fail-loud vs flax silent fresh-init restore — a reproduced FALSE PASS)._
- _**TASK_835** (`task835-stage4-eval`) — faithful Stage-4 vector eval; 0 MAJOR; 3 MINORs fixed
  (LEDGER pointer, partial-rollout guard, provenance)._
- _**TASK_832** (`task832-linear-imprint`) — E5 buffer-injection BC runner; 1 MAJOR found+fixed
  (imprint entropy_coef leaking into the ExIt half via checkpoint metadata)._

_Shipped 2026-07-08 factory run 3: **TASK_828 Stage 3** (`task828-stage3` — vector-action training
loop + D2 action-conditional reward head + factored-marginal targets; parity MATCH==PINNED, E3 probe
green, full high-tier panel 0 BLOCKER/MAJOR + 1 MINOR fixed). Stage-3 TRAINING itself is the scheduled
gpu-server GPU run below, not a factory-loop job._
_Shipped 2026-07-08 factory run 2: TASK_833 (`task833-run-queue-repair` — repaired scheduler
redeployed to srv same day) and TASK_829 (Foxglove retired, `40afa14`)._

## Scheduled runs — owner: planner/main session (NOT factory-loop jobs)

**Server status (updated 2026-07-09)**: gpu-server db verified at FULL workstation parity —
daily tables synced 2026-07-08, and the minute `ohlcv` table measured at max ts 1765558500
(2025-12-12, == workstation; the "Jan-2026 vintage" concern was file-mtime, not data). All
daily AND minute jobs cleared. Server code current; the 7 drifted server files remain STASHED
(`git stash list`).
Stage-3 training destination = gpu-server GPUs 1/2/3 — run_queue.sh REPAIRED (TASK_833) and
redeployed 2026-07-08; the scheduler is cleared for use.

| Run | What | Where | Status |
|-----|------|-------|--------|
| 830 real corridor | full-panel floors+ceiling+expressibility gap, `trading_data.db` read-only | workstation CPU (screen+safe_run) | **DONE 2026-07-08** — corridor official [+0.424, +29.698] net@21, ref +0.751; H6 armed; LEDGER 830/E0-REAL |
| 831 real E1/E2 sweep | 9 scheme×K combos, full panel, seeds 0,1,2 | workstation CPU (screen+safe_run, ~7 min) | **DONE 2026-07-08** — E1 PASS 9/9 ⇒ Stage-3 training gate OPEN; carried scheme dithered-book\|K16; LEDGER 831/E1-REAL + E2-REAL |
| 828 Stage-3 training **phase 1** | cost0 ×3 seeds from scratch, dithered-book\|K16 **hb0.0/d1.0** (3-arm diagnostic winner; fee16-from-scratch REJECTED — collapse finding, LEDGER 828s3/preflight), snapshots every 100; `jobs_phase1.manifest` | gpu-server GPUs 1/2/3 via run_queue.sh (tmux `q_828s3`) | **DONE 3/3 2026-07-10 05:18** — s1/s2 ep2000 first pass; s0 wedged ep1581 (log-freeze post-checkpoint-save) → killed, archived `*_wedged_ep1581`, solo retry completed clean (rc 0, 20 snapshots; wedge did NOT reproduce at ep1581 — non-deterministic, not data-positional). Ops note: fresh tmux needs `source trader_env/bin/activate` (bare `python` = system Py2) |
| 828 phase-1 **instrument passes** | 835 eval + 834 E3/DEC-1 + 836 E4 on s1/s2 `snapshot_ep2000`, phase-1 flags EXPLICIT (`--vector-hold-bias 0.0` — 835's CLI default is 3.0!), fee0/slip0 run configs `runs/828s3p1_instruments/cfg_*.json` (server) | gpu-server GPUs 2/3 via run_queue.sh (tmux `q_828s3_inst`) | **DONE 2026-07-09 20:06** — 835 FAIL both seeds (bankrupt, below floor; LEDGER 835/S4-p1); 834 fail-loud walk-bankrupt (834/E3-p1); 836 n=0 (836/E4-p1). Root = **828s3/env-wall**. 834/836 re-run AFTER phase-1b. s0's 835 DONE 2026-07-10: same FAIL (bankrupt 501/676, hold 98.2) — verdict 3/3, LEDGER row folded |
| 828 Stage-3 training **phase 1b** | RE-RUN under the fixed step: phase-1 command + `--vector-budget-step` (D7 opt (a), **D8 ratified ship-and-measure**), cost0 ×3 seeds, `jobs_phase1b.manifest`; pre-registration = SPEC amendment 3 | gpu-server GPUs 1/2/3 via run_queue.sh | **s1/s2 DONE 2026-07-10 ~16:15** (ep2000, 20 snapshots, 9.7h). **s0 WEDGED at ep162** — SAME signature as phase-1's wedge: frozen immediately after "Finished synchronous save" (orbax post-checkpoint deadlock, now 2/6 seed-runs — pattern, candidate factory row) → killed 2026-07-11 21:08, archived `*_wedged_ep162`, solo retry RUNNING (tmux `q_828s3p1b_s0`, GPU 1) |
| 828 phase-1b **instrument passes** | 835 (D8 primary evidence) + 834 DEC-1 + 836 E4 on s1/s2 `snapshot_ep2000`, phase-1b flags (budget ON) | gpu-server GPUs 2/3 | **DONE 2026-07-11 21:35, 6/6 rc0 — the decisive mixed verdict**: 835 still bankrupt (step 132/181, s1 book = 10–11 shorts → D8 escalation FIRED ⇒ TASK_840); **834 DEC-1 PASS ×2** (R 1.30/1.12 vs bars 0.33/0.24 — D2 rebuild works); **836 value>momentum ×2** (ρ +0.21/+0.30 vs −0.03 — first positive learning signal). LEDGER 835/S4-p1b · 834/E3-p1b · 836/E4-p1b |
| 828 Stage-3 **phase 1c** | re-train under the TASK_840 weight env: phase-1 recipe + `--vector-weight-rebalance` (NO budget flag — one variable), cost0 ×3 seeds, `jobs_phase1c.manifest`; pre-registration = SPEC amendment 4; **D10 ratified PROCEED**; watchdog ARMED per D11 (`--stall-timeout 2700`) | gpu-server via run_queue (tmux `q_828s3p1c`, `--gpus 1,2,3 --acquire-timeout 0`) | **s0/s1 DONE 2026-07-12 07:32 + instruments graded same morning (LEDGER 835/S4-p1c · 834/E3-p1c · 836/E4-p1c): ENV FIXED (zero bankruptcies, first-ever complete rollouts) / POLICY BELOW FLOOR 2/2 (short-biased book) ⇒ D13 OPEN**; s2 DONE 14:16 + record pass graded (−1.385, complete rollout, 7L/8.4S — verdict 3/3). **Battery closed CLEAN: zero failures, zero wedges, watchdog armed throughout.** Earlier: s2 start was delayed by the **p1b-s0 retry wedge #3 at ep982** → py-spy stack CAPTURED (841 root-cause: XLA/CUDA dispatch hang post-save, orbax-lock hypothesis REFUTED — see TASK_841 evidence note + `lore/notes_2026-07-12.md`), killed, NO re-retry (p1b verdict stands on s1/s2). Graded read: corridor vs floor +0.424 on ≥2/3 seeds; any bankruptcy = TASK_840 defect |
| 844 **short-bias postmortem** | `bias_probe.py` on p1c s0/s1/s2 + p1b s1 control (`jobs_844.manifest`, weight/order run flags from `cfg_*.json`) | gpu-server GPUs 2/3 via run_queue.sh (tmux `q_844`) | **DONE 2026-07-14 20:40, 4/4 rc0 — NO LOCUS TRIPPED (LEDGER 844/postmortem)**: value clean (calibrated \|z\|≤0.9, the raw-t would have false-tripped ×4), prior BUY-leaning (0.65–0.84), credit-misassignment unsupported. Planner post-hoc: probe leg tracker is FLIP-BLIND in weight semantics — shorts exit ONLY by BUY-flip to long (exact arithmetic 3/3); longs culled by non-action re-trim closes (margin ruled out — fee0 no-op invariant). Folded into D13 |
| 832 **imprint-sanity** (E5 leg 1) | teacher replay floor (weight env, real panel — the binding ±0.10-absolute expressibility read) + BC imprint + deterministic sanity replay vs teacher ±0.10 (DEC-2 tolerance); cost0, weight flags = phase-1c set; run BOTH mappings via `--imprint-mapping {order-code,weight-native}` (845); ≤1 GPU-day | gpu-server single verified-idle GPU, run_queue + D11 watchdog | **DONE 2026-07-17 00:06 — BOTH LEGS (LEDGER 832/E5-gate + 832/E5-imprint).** Gate-read: order-code FAIL (rel 0.290) / weight-native PASS (rel 0.041) ⇒ D15 branch 2, no retune. Full imprint (weight-native, iters 2000): BC converged (CE ÷100, agreement 97.5%), sanity replay net@21 **+0.839 vs floor +0.542** — ABOVE floor both tiers, zero bankruptcies, first learned policy above floor in the program. Strict ±0.10 window missed on the UPSIDE (+0.297) ⇒ ruled PASS-with-caveat (planner; H5 kill clause targets BC failure, which did not occur). Ckpt `runs/832_e5/ckpt_imprint_full` (srv) ready for the ExIt restore |
| 832 **ExIt** (E5 leg 2, = the phase-2 continuation under redirect) | short ExIt from the imprint ckpt (`--load-checkpoint --reset-optimizer --clear-buffer`; verify `Restored entropy_coef: 0.01` at launch — SPEC PIN c), ≥3 seeds, fee16 per DEC-2; PASS = floor+0.10 net@21 ≥3 seeds (per-seed above bare floor) + ≥3/4 regime windows at/above their window floor | gpu-server, run_queue + D11 watchdog | **DONE — FAIL 2/2 (LEDGER 832/E5-exit, graded 2026-07-18).** s1 −1.223 / s2 −0.791 net@21 vs bar +0.524 from a +0.839 warm start — self-play eroded the prior into the from-scratch attractor (99% hold, sparse book); s0 wedge-failed ×2 (watchdog fail-loud), moot. Zero bankruptcies at fee16 (846 validated in production). H1 improvement operator implicated; H6 NOT triggered (imprint holds the floor). Pivot = **D17** |
| ~~828 Stage-3 phase 2 (from-scratch continuations)~~ | ~~fee16 continuation from phase-1b/1c ckpts~~ | — | **SUPERSEDED by D13 redirect 2026-07-14**: the ExIt row above IS the phase-2 fee16 continuation, warm-started from the imprint. From-scratch ckpts (p1/p1b/p1c) = forensics + probe inputs only, never warm-start (cell measured negative twice: LEDGER 835/S4-p1b, 835/S4-p1c) |
| **882 decisive — Leg A (identity check)** | eval-only, 3 seeds: `ckpt_ref_s{0,1,2}` graded over the CONTINUOUS 2021-01-01..2025-10-04 — the same window their `imprint.py` sanity replay used — CORRECTLY PINNED to `frame_provenance_ref_s{0,1,2}.json`, `--operator policy-argmax --vector-weight-rebalance --teacher-floor`. Manifest `experiments/882_instrument_reconciliation/manifest_882_decisive.txt`. **Bands PRE-REGISTERED before launch** (TASK_882 §6): |Δ| vs `+0.9914` ≤ 0.1 ⇒ the instruments AGREE once pinned · ≤ 0.5 ⇒ named residual · > 0.5 or a SIGN FLIP ⇒ a second, pin-independent defect and the corroboration is WITHDRAWN. Framework-epoch diff run BEFORE launch per the 879 doctrine and recorded in the manifest header: srv redeployed b695487 → `60f77f7`; the eval-path delta is TASK_870's `_ro_connect` move plus a `--smoke`-only `overwrite=True`, neither semantic; `--vector-weight-rebalance` and `--panel-frame-from-checkpoint` PRESENT on every line. | gpu-server GPUs 1/2/3, tmux `d882` | **DONE 2026-08-08 — PASS, TOP BAND.** LEDGER `882/decisive-legA`. The instruments AGREE to **Δ 0.000214** net@21 (+0.9915919 pinned eval_vector vs +0.9913783 imprint.py), PV within 0.060%, residual localised to exactly 3 of 107,072 decision cells (float32 argmax tie-breaks). 3/3 seeds +0.99159 / +0.99355 / +0.95212, 1673/1673 decisions, no bankruptcy, 64/64 slots active. **`+0.9914` RESTORED, instrument-independent and 3-seed.** Does NOT close `861/ref-oos-rebase` — the OLD `ckpt_imprint_full` anomaly is an EPOCH question, untested here. |
| **884 rung 1 — step-0 smoke** | 40 epochs, seed 0, 1 GPU, phase-1c recipe under the FIXED env, window truncated to 2021-01-01..2025-10-04 so the held-out span matches the imprint lane's OOS window. Pre-registered `experiments/884_rung1/SPEC.md` (pre-launch sha256 `2b265701…`); universe preflighted on the actual periods file (`preflight_frame.py`: 64/64 order-exact, pad does not fire, 0 zero-price cells). | gpu-server GPU 1, tmux `r884smoke` | **DONE 2026-08-08 — S2 FAILED, BATTERY NOT LAUNCHED.** LEDGER `884/rung1-smoke`. `pct_hold` 82.3% → 95.6% by **epoch 3** → 99.0% plateau; `policy_entropy` 0.691 → 0.070. **14.17 s/epoch measured** ⇒ the battery would have been ~7.9 h × 3 seeds; the gate cost 9 GPU-min and saved ~24 GPU-h. **No bankruptcy anywhere** (min `agent_return` −22.6%) — E-a fixed the cash cascade and the hold collapse happens anyway, so they were separate defects. `manifest_884_battery.txt` staged, NOT launched. |
| **885 hold-collapse screen** | 3 arms × 40 epochs, seed 0: **B** `--sims 8` · **C** `--sims 128` · **D** `--sims 32 --epsilon-greedy 0.25` (the TASK_812 target-mix that severs MCTS-sharpening → policy-sharpening). Arm A = the completed 884 smoke. Tests whether the collapse is SEARCH-driven — the training-lane twin of `879/rung3`'s subtractive-search control. **Bands pre-registered** in `experiments/885_collapse_screen/SPEC.md` (sha256 `fc2edd4d…`): (P1) `pct_hold` ordered B<A<C with ≥2.0pp B-to-C spread AND `mcts_visit_entropy` ordered B>A>C · (P2) arm D reaches `pct_hold` < 95% · (NULL) all arms within 1.0pp. DIAGNOSTIC SCREEN, 1 seed/arm — the winner earns a 3-seed battery, nothing promotes off this. | gpu-server GPUs 1/2/3, tmux `r885` | **DONE 2026-08-08 — P1 FAILS, P2 PASSES.** LEDGER `885/hold-collapse-screen`. `pct_hold@40`: A 99.016 · B 98.854 · C 98.987 · **D 90.971**. A 16× sims range moves hold **0.13pp** and is non-monotone ⇒ **search depth is irrelevant**; the epsilon target-mix holds it at a plateau (82.3 → 92.2 by ep5, then flat to ep40) and keeps the RAW pre-mix `mcts_visit_entropy` at its epoch-1 value (0.1554 vs A's 0.0246, 6.3×). **The collapse is the distillation loop, not the search depth.** 1 seed/arm, 40 epochs — a screen; the winner owes a 3-seed battery. |
| **886 rung-1 re-spec (PROPOSED, not launched)** | The 885 winner owes a 3-seed confirmation before any 2000-epoch commitment: arm D vs baseline at **3 seeds × 200–400 epochs**, plus an epsilon **sweep or decay schedule** (`--epsilon-schedule`, TASK_812) — 0.25 is a single untested point, and a constant mix that never decays caps final policy sharpness by construction. Needs its own SPEC and bands. **Rung 1's original `manifest_884_battery.txt` should NOT be launched as written** — it reproduces arm A. | gpu-server, ~1–2 h at the measured rates | **AWAITING OPERATOR** — planner recommends it as the next scheduled run |

## Gated — do NOT start (named gate must open first)

| Task | Gate |
|------|------|
| `tasks/TASK_880_CKPT_ARCHIVE_DISCOVERY_FIX.md` | **GATE REWRITTEN BY D35 (2026-08-08): TASK_881's path-first reader must be SHIPPED, and a deployed run must have written `scores.jsonl` that THAT READER RESOLVES.** D34's original gate — 876 shipped AND deployed AND real score files on the host — is now known to be necessary but NOT sufficient, and half of it is already literally met (deployed at `60f77f7`, 2026-08-08) while the signal stays unreadable for `checkpoint_<step>` and `checkpoint_epoch_<N>`. `--unscored archive` (pull, never delete) remains authorized under D34 and is unaffected. The `--apply` freeze on `tools/ckpt_archive.py` STANDS until both halves hold. _(D34 gate text, superseded, kept verbatim for the record:)_ **TASK_876 must be SHIPPED *and DEPLOYED to gpu-server*, with a run having written real `scores.jsonl` files there** (D34). Not "876 merged" — the host must actually carry the artifact, because 100% of this tool's score signal is otherwise inference over `results*.json`. Run 27's real-host dry-run measured `ARCHIVE 1 ckpt / 0.09 GB · DROP-WITHOUT-ARCHIVE 27 ckpts / 2.51 GB` — one operator command from deleting 27 of 28 checkpoints including `runs/861_ref/ckpt_ref_s0/checkpoint_2000` at net@21 **0.9914**, the clean-recipe anchor. The `--apply` freeze STANDS; `--unscored archive` (pull, never delete) is the only authorized mode meanwhile. |
| ~~`tasks/TASK_843_WEIGHT_INCREMENTAL_NETTING.md`~~ | gate SATISFIED 2026-07-14 → moved to Ready #1 |
| `tasks/TASK_828_VECTOR_ACTION_MUZERO.md` (Stage 4 eval) | Stages 1–3 shipped ✓ (2026-07-08) + Stage-3 training run's rung verdicts green; SCHEDULED server run, not a factory-loop job |

_TASK_832 (E5 linear imprint) moved to Ready 2026-07-08 — its gate (828 Stage 3 banner) is now satisfied._

## Parked (D1/D4, 2026-07-07 — non-binding "interesting outview"; not factory work)

- `tasks/TASK_824_SERVER_HONEST_REMEASURE.md`, `tasks/TASK_825_CROSSASSET_TREND_CTA.md`,
  `tasks/TASK_826_ONCHAIN_SIGNAL_PROBE.md` (D4: OHLC-only, live within the 24GB DB),
  `tasks/TASK_823_VPS_MICROLIVE_CTA.md` (D0: rejected).

## Grooming rules
Planner re-ranks after each factory run + banners; a row leaves this file only by shipping
(banner) or being explicitly parked. If the factory finds the top row blocked in practice, it
takes the next and records why in its wrap-up.
