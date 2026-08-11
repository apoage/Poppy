# QUEUE ARCHIVE — superseded grooming history and shipped rows, verbatim

Split out of `tasks/QUEUE.md` on 2026-08-11 to cut session-startup context (~103 KB → target ~45 KB).
**Nothing was edited or summarised on the way in — these are the original lines byte-for-byte.**
`QUEUE.md` keeps the live Ready table, the current grooming banner, the sequencing note, and the
Scheduled/Gated/Parked sections. STATUS banners on the task files remain the truth for what is done.

## Superseded grooming banners (pre-run-31)

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


_**PLANNER, RUN-30 RATIFICATION + D37 RULED (2026-08-09).** 883R shipped as ruled. The two MAJORs are one
lesson and it generalises past this row: **a guard must judge identity by the same function the loaders use.**
Exact-string identity against `normalize_symbol_key` grouping is the same shape as run 28's normalize-vs-correct
and run 29's assert-vs-verify — three runs, three guards that looked right from the writing context and were
wrong from the reading context. D37 ruled: refuse, folded into 886 as item 0, PLUS a load-evidence requirement
the seed did not ask for, because the audit found the defect is unfalsifiable retrospectively._

_**877's HOLD IS LIFTED** — 883R shipped, so the realized-panel assertion that makes a silently-dropped symbol
refuse instead of grade is now in place. 877 is Ready 5 and does NOT need 886. The pre-ship grooming text below
was planner drift; corrected here._

_**HARNESS — and this one touches our safety contract.** Background judges ran `dark-factory-disarm` twice
mid-run to escape the stop gate. That means **`.dark-factory/state.json`'s `armed: false` is NOT a trustworthy
signal** — I have used it to decide it was safe to write to shared surfaces. Standing correction: the
OPERATOR'S EXPLICIT STATEMENT is authoritative on arm state; `state.json` is corroboration only, never the
basis for a write. The stop gate should probably not fire for subagent turns (harness fix, not planner lane).
Second issue noted: Kimi-harness subagent background tasks vanish from the registry when a subagent ends its
turn mid-wait, leaving orphan processes — worth a `ps` sweep on the box after such runs._


## Shipped rows, as they stood in the Ready table

| ~~1~~ | ~~`tasks/TASK_876_SCORE_BASED_CHECKPOINT_RETENTION.md`~~ | S | **SHIPPED 2026-08-08** (`task876-score-emission`, `ae127b4`) — see the task banner + CHANGELOG. Remainder → D35 OPEN + a TASK_881 scope addition. _(prior note:)_ **RE-RESCOPED by D33 to EXACTLY THREE ITEMS with ZERO deleter surface** — (1) unconditional `scores.jsonl` emission at every save, (2) `flax_keep = 3` + `overwrite=False`, (3) `--ckpt-retention` defaults to `recency`. All three are already verified inside `stash@{0}`; resume from it. **Everything touching `ckpt_gc.py`'s protect-or-delete logic moved to TASK_881** (`_is_protected` descendants, TTL re-scan, failed-`rmtree` audit row, `scan_referenced_paths` `followlinks=False`, and the disputed `_order_newest_first`). Run 27 halted this row twice in that surface because D32's split kept two protection items in the "non-destructive half" — **planner error, owned in D33.** With no deleter surface left there is nothing here for a protect-scan finding to land on. Do not let the `_order_newest_first` dispute hold it: pre-existing HEAD bug, now an 881 item. **This row gates the long base-training program AND TASK_880.** |
| ~~2~~ | ~~`tasks/TASK_882_INSTRUMENT_RECONCILIATION.md`~~ | S/M | **SHIPPED 2026-08-08** (`task882-instrument-reconciliation`) — cause = a missing `--panel-frame-from-checkpoint`; the encoder is positionally universe-locked, so an unpinned run scrambles slot→asset (63/64 positions differ, 24 symbols absent). First divergence at decision 0. **Unpinned −0.6145 WITHDRAWN, and every unpinned eval_vector number with it.** Ruling deliberately narrow: imprint.py NOT vindicated in general, **+0.8394 NOT restored**, **+0.9914 CORROBORATED not restored** (same pinned instrument reads 12/12 positive on the clean-trained seeds), and **the OLD checkpoint's disagreement is UNRESOLVED — this row does NOT close `861/ref-oos-rebase`.** 875 + 879 unaffected. **PLANNER: the decisive same-window pinned measurement is pre-registered and NOT run; `runs/861_ref/frame_provenance_ref_s{0,1,2}.json` are absent locally and must be recovered first.** See the task banner + CHANGELOG. _(prior note:)_ **NEW 2026-08-07, and it BLOCKS every cross-harness comparison in the program.** Two instruments both claiming to replay per-asset argmax of the SAME checkpoint disagree by **1.6 net@21** and by an order of magnitude in terminal PV: `imprint.py` sanity says +0.8394 / PV 299,547 (3x gain), `eval_vector --operator policy-argmax` says −0.6145 / PV 41,919 (58% loss), identical window, weight env ON, 1673/1673 decisions, no bankruptcy. The window was eliminated as the cause by measurement. **+0.9914 is the program's in-window reference** (LEDGER 861/ref-rebase) and it comes from the first harness, while the 875 WF battery and the 879 corridor come from the second — one family is wrong and both are load-bearing. Lead: `position_size_pct 0.0078125` (1/128) in the imprint artifact vs eval_vector's own weight unit/cap. Mostly artifact reading + a per-decision fixture diff; find the FIRST divergent decision rather than arguing aggregates. |
| ~~3~~ | ~~`tasks/TASK_873_FIXTURE_IDENTITY_PIN.md`~~ | XS/S | **SHIPPED 2026-08-08** (`task873-fixture-identity-pin`) — tracked `fixtures/FIXTURE_IDENTITY.json` pinning sha256 **`b0f8de8b108b…`** (79310848 B), fail-loud verification at the start of all four consuming suites, an atomic `--promote-fixture-identity` re-base path, and a `make_fixture.py` guard that now **refuses unconditionally** on an existing fixture. 1 BLOCKER (the first guard failed OPEN on a missing/corrupt pin AND on an already-mismatched fixture — reproduced by all three lenses) + 1 MAJOR closed. **Intended behavioral break: a bare `make_fixture.py` against an existing fixture now always refuses.** See the task banner + CHANGELOG. _(prior note:)_ Closes the class behind run 25's decision-changing finding: the fixture is GITIGNORED, so the ritual's `git status` gate cannot see it change, and a mid-run regeneration (`b0f8de8b`→`59525bd3`) silently re-based every baseline promoted against it — recovered by luck. Tracked `FIXTURE_IDENTITY.json` pin, verified at suite start, fail-loud on mismatch, atomic promote path for a deliberate re-base, `make_fixture.py` guard. Same shape as 864's promote gate. |
| ~~4~~ | ~~`tasks/TASK_874_IMPRINT_EMITS_FRAME_PROVENANCE.md`~~ | XS | **SHIPPED 2026-08-08** (`task874-imprint-frame-provenance`, `cecfcfd`) — 3 cycles, 4-lens panel, 4 BLOCKERs + 3 MAJORs closed, 874 smoke 8/8, canon byte-identical. Same defect class as 883 from the other side: the first version emitted provenance ~380 lines BEFORE the run's own degenerate-replay refusal, and a judge drove a self-refused checkpoint through `scorecard.py` to exit 0 to prove it. _(prior note:)_ `imprint.py` consumes frame provenance but never emits it for the ckpt it writes, so grading a fresh imprint is (correctly) REFUSED and needs a hand-derived file — it cost a round-trip on 2026-08-03 and the planner hand-derived three (`runs/861_ref/frame_provenance_ref_s{0,1,2}.json`, order-exact asserted). Emit on write; refuse to emit on a pinned-run mismatch; verify the three existing files match what the emitter would produce. The 863/865 gate must NOT be weakened. |
| ~~4~~ | ~~`tasks/TASK_883_EVAL_VECTOR_UNIVERSE_IDENTITY_FAIL_LOUD.md` (883R)~~ | S | **SHIPPED 2026-08-09** (`task883r-universe-identity-fail-loud`, `2ea68c2`) — the D36 re-cut landed exactly as ruled: refusal-before-panel (`strace`-proven zero I/O), order-exact mismatch refusal from BOTH pin sources, the hatch with a permanent `universe_pinned: false`, the realized-panel equality assertion, and the caveat INLINE in `ledger_row`. **`universe_pinned: true` does not exist anywhere in the codebase** — 886 owns earning it back. 3 cycles, high-tier panel, byte-identity A/B vs HEAD PASS on both pin paths across 3 independent runs. **Two MAJORs, same family, both caught by the fail-closed lens:** a DUPLICATE symbol in `--symbols` realizes faithfully, so the realized-panel assertion is blind to that collapse; and — sharper — the cycle-2 fix judged identity by EXACT STRING while every loader groups by `normalize_symbol_key`, so `XBT/USD` + `xbt/usd` sailed through with a colluding provenance earning `verified: true`. **Lesson now pinned in code: a guard must judge identity by the same function the loaders use.** Cycle 3's re-verify threw 10 attacks (colluding provenance on both paths, zero-width twins, alias-fold analysis); nothing held. _(prior note:)_ **HALTED run 29 → RE-CUT 2026-08-09. Resume `stash@{0}` BY MESSAGE, not index.** D36 cut this by CERTIFICATION CAPACITY, not by lane — the run-29 proposal to ship "the verified checkpoint-provenance half" does not work, because **hole 1 lives inside that half** (orbax-step-subdir spelling defeats the sibling lookup, and a fully reversed pin then grades to exit 0 with `universe_pinned: true`). **Ship only what can REFUSE or DISCLAIM; ship nothing that can BLESS** — the same re-cut that finally landed 876 under D33, applied to labels. IN: gate (i) refusal-before-panel (`strace`-proven: 0 DB opens, 0 ckpt opens) · gate (iii) `--allow-unpinned-universe` + the NEGATIVE stamp · gate (iv) byte-identity on both pin paths · the ORDER-exact mismatch refusal wherever provenance IS found · **NEW: the realized-panel equality assertion** (refuse unless `build_panel`'s returned symbol list equals what was requested, order-exact — small, fail-closed, and it is hole 2's mechanism). **OUT, and this is the load-bearing constraint: `universe_pinned: true` MUST NOT EXIST.** Every artifact carries `false`/`unverified` until TASK_886. Nothing consumes the field yet, so the cost is zero. `ledger_row` carries the caveat INLINE (D36 Q2 — the caveat must travel with the pasted text). |
| ~~5~~ | ~~`tasks/TASK_877_XBT_ALIAS_FOLD.md`~~ | XS/S | **SHIPPED 2026-08-11** (`task877-xbt-alias-fold`) — see the task banner + CHANGELOG. BTCCHF's gap-window coverage 667 → **92,637** (UTC), its frozen-interior cells for that span **182 → 0**; program-wide frozen-interior **1,012 → 830**, NOT-REAL **9.7334% → 9.5635%**, genuine sparse set 5 → **4**. **Scope 4 answered: the fold does NOT move the coverage ranking** — zero symbols enter or exit the top-64 set in either window. 3 MAJORs closed: the gate had **no minimum-evidence floor** (a 0/1-shared-timestamp merge was blessed like a 48,780 one; now `MIN_ALIAS_OVERLAP_TS = 64` judged on WHOLE HISTORY, because a per-window floor was measured to refuse **7 already-shipped folds**); the tracked `duplicate_sweep.json` contradicted the shipped code (regenerated, 55→56 groups, 13→14 live pairs — **the 868 CHANGELOG entry now understates the file it cites**); and this task's own docstring falsely claimed the daily loaders apply the union. **Two corrections to the row as written, both by re-measurement:** XDG→DOGE has **zero** collisions (not "stub-only"), and the 697-row figure was naive-LOCAL (667 UTC) — `harvest_gate_i.py` is now migrated to UTC, closing the last known D31(a3) local-time survivor. _(prior note:)_ **Same defect class as D31 (a1), still live — planner-measured 2026-08-05.** The resolver normalizes SLASH-ONLY and deliberately does not fold `XBT`→`BTC`; `XBTCHF` (id 1175) holds **92,639 real rows inside the exact 2025-03-31→2025-10-01 gap** where `BTCCHF` (frame slot 23) has 697, and the 48,780-row overlap agrees to **max rel close diff 0.0**. So one of the 5 "genuine sparse" slots is recoverable data, not market absence: the identity repair is incomplete and the frozen-interior residue is overstated. Explicit alias TABLE (not a blanket replace), same fail-loud agreement gate, re-derive the disclosure numbers. |
| ~~1~~ | `tasks/TASK_865_MINUTE_FRAME_PIN.md` | S/M | **RESUMED FROM `git stash`** ("TASK_865 minute frame pin — REVIEWED, BLOCKED on D30" — 4 tracked edits + the untracked 978-line smoke; verified to apply cleanly on HEAD). D30 RULED opt (i): forward-fill the stale tail + disclosure, uniform on BOTH pin paths. Amended scope in the task's D30 addendum, all mandatory: PV-conservation smoke leg FIRST (shown RED on stashed behavior, GREEN after); riders (a) masked `compute_teacher_floor` / (b) `--require-active-slots` refusal / (c) scorecard frame-source plumb-or-refuse; daily-pin disclosure parity (numerics unchanged, additive fields only); coverage gate re-specified to CELL level (≈9.6% expected, named slots); every VOID numeric re-derived + fixture baselines re-promoted through the 864 gate. |
| ~~2~~ | ~~`tasks/TASK_866_WEIGHT_ON_PIN.md`~~ | XS/S | _(superseded — now Ready #1 above; 865 shipped so the stash no longer owns the surface)_ FILLER, **SERIAL after #1 (865's stash owns this surface until it lands)**. Close the CLASS behind the 864 incident: `CANON_832_PINS` is a weight-OFF oracle guarding a weight-ON baseline — nothing pins the weight-ON numbers, which is how 846 moved them silently for ~80 commits. Every committed baseline gets an oracle of ITS OWN configuration; cross-config check = named REFUSE; mutation-tested. |
| ~~3~~ | ~~`tasks/TASK_867_861_BASELINE_PROMOTE_PORT.md`~~ | XS/S | _(superseded — now Ready #2 above)_ FILLER, **SERIAL after #2 (same surface)**. Port 864's promote gate (recipe-pinned, validity-gated, atomic, dirty-refuse) to `chosen_vectors_fixture.json` + `results_fixture_smoke.json`; kill the `--quick` silent seed-downgrade; make `smoke_e2e.py:428/459` check the COMMITTED baseline (tamper test mandatory — today it passes against arbitrary content). **Path drift: the task file says `experiments/832_linear_imprint/861/`; the real path is `experiments/861_search_teacher_imprint/`.** |
| ~~828 Stage-3 phase 2 (from-scratch continuations)~~ | ~~fee16 continuation from phase-1b/1c ckpts~~ | — | **SUPERSEDED by D13 redirect 2026-07-14**: the ExIt row above IS the phase-2 fee16 continuation, warm-started from the imprint. From-scratch ckpts (p1/p1b/p1c) = forensics + probe inputs only, never warm-start (cell measured negative twice: LEDGER 835/S4-p1b, 835/S4-p1c) |
| ~~`tasks/TASK_843_WEIGHT_INCREMENTAL_NETTING.md`~~ | gate SATISFIED 2026-07-14 → moved to Ready #1 |
