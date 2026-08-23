# Factory run 41 — handoff (2026-08-23 → 2026-08-24)

Planner lane armed the factory at 22:15 on a clean `master` (`c6afec7`). Five queue rows were
touched. This document is the durable record; task banners and `tasks/DECISIONS.md` are the
authority on any point where this file drifts.

**Outcome in one line: one row shipped, one halted, one parked, one corrected and under final
review — and every one of the four had a real defect found by an independent lens rather than by
the agent that wrote it.**

| Row | Outcome | Record |
|---|---|---|
| TASK_897 | **SHIPPED** — `task897-d60-guard-rebase` | `5ee6381`, banner `d1951bc`, correction `2eaa844` |
| TASK_904 | **HALTED** on the 3-cycle rule | D61, `09a6918` |
| TASK_881D | **PARKED** — green, BLOCKER dead, blocked on a ruling | D62 + D63, `1c0cb4f` |
| TASK_901 | corrected after a partly-unsound negative; re-verify running | see Late lanes |
| TASK_902 | four items routed in | `d1951bc`, `2294a11` |
| TASK_905 / TASK_906 | seeded | `d1951bc`, `7dd8d26` |

Two rows ended without shipping. Both preserved their work as an applyable patch (not a stash —
other lanes were live in the tree), left the tree at HEAD, and handed the operator three concrete
options with a recommendation. Neither halt was a failure of the work: 904 solved both of its
original defects and died on the belt it added; 881D killed a severe BLOCKER and stopped at a
scope wall that only the planner can open.

---

## Shipped

### TASK_897 — `task897-d60-guard-rebase` (`5ee6381`), banner + routing (`d1951bc`)

D60 option (a) implemented as ruled. The 806-inherited IC guard no longer asserts June-era pins
against the harness's SWEEP panel; it builds a separate **pin-era panel** (`probe.Config`'s own
split defaults — `test_end=2025-04-01`, the pins' actual basis — exact-spelling read, never the
D56(a) alias union) and asserts the pins there with the **tol UNMOVED at 1e-3**. The sweep-panel
IC becomes a disclosed, non-gated MEASUREMENT.

**The trap the fix turns on, and the reason a naive reading of D60 would have failed:**
`probe.make_splits` **ignores `cfg.test_end`** — the PANEL BUILD is what moves the test span. A
`decide.CFG` scope-patch alone is vacuous; the planner's own first diagnostic died on this.

**Second vacuous-pass root cause, found by the panel and closed.** The lineage check guards with
`if tgt is not None and abs(ic - tgt) > tol`, so for any H outside `{1, 5, 20}` the comparison
short-circuits and a real, **finite, never-compared** IC was recorded `status: "pass"` under a
bullet calling it "the lineage-fidelity proof; GATED". An `isfinite` screen is structurally unable
to catch it. Reproduced independently by two lenses (H=2, H=7). Not reachable from the CLI;
reachable via the public `run_sweep(grid_h=…)` API — the exact reuse surface D60 names for
898-class decay work. **The general rule now lives in TASK_902 item 2: _a check that did not
execute a comparison is not a passed check._**

Verification: 215 tests in the 897 dir (212 baseline + 3 new, each mutation-verified RED→GREEN and
re-checked under `-k` isolation), selftest 12/12, 144-cell fixture smoke green both ways. A/B vs
pre-change: `## Cells` through EOF **byte-identical** (`8468be63…`) — no cell number moved. 809
lineage byte-untouched, md5-verified at four points.

**ACTION FOR THE PLANNER: the srv leg can now relaunch.** Same command as the 2026-08-23 launch
record, after deploying `5ee6381`. Expected: the guard passes on the real pin-era basis at
**+0.151053 / +0.141160 / +0.059755** vs pins ±1e-3 (planner-measured 2026-08-23). Gate (iv) still
needs the operator's attainable Kraken maker-fee tiers recorded BEFORE the verdict is read.

---

## Halted and parked

### TASK_904 — HALTED on the 3-cycle rule (D61, `09a6918`). NOT shipped.

Both ORIGINAL defects were solved; the row died on the **belt it added**, not the landmine it set
out to defuse. Cycle 1's sentinel-string gate was broken in both directions (forgeable AND it
locked out the `SPEC.md:62-64` re-materialization). Cycle 2's redesign — operator flag as the gate,
provenance demoted to disclosure — is **sound and twice re-verified**. Cycle 3 closed four more
holes, then failed its own re-verify on a demonstrated BLOCKER: **the corridor refusal mutates the
tracked fold tree while exiting 1 and printing "Nothing was written"** (`walkforward.py:1185`
writes inside the per-fold loop, before a later fold's refusal at `:1174-1182`).

**Structural lesson worth more than the row:** a per-fold write-then-refuse loop cannot be
fail-closed. The corridor writer needs a two-phase design — decide every fold, then write.

**Why HEAD was left with the original landmine deliberately:** HEAD's defect is latent (smoke
overwrites 15 tracked fold files with content-identical output; it arms when the real
re-materialization lands). Cycle 3's defect fires today. Trading a dormant problem for a live one
is not an improvement, so `experiments/837_walkforward/` was reverted to HEAD and the work
preserved at `lore/handoff_2026-08-24_factory_run_41_904_cycle3.patch` (1838 lines, incl. a
32-test suite).

**Scope 2 IS done and needs no commit.** The `WF_SUMMARY.json` fake-null is dead — regenerated from
an untouched `grade_correct.py` (`e87b080d…`) via `runpy`, atomic write, refusing empty and partial
results, `folds_positive=4 / folds_above_own_floor=1` matching the LEDGER row and reproduced by
three agents independently. It lives under gitignored `runs/875_wf/`.

`folds/real` tree hash `1b31624a2da9984620567afc30df4cbf` — identical before and after every cycle,
judge, and attack. Gate (iv)'s 875 numbers never moved.

---

## The pattern this run kept finding — read this part first

**Seven defects across four rows were the same defect.** Not a coding error: in every case the
guard's logic was correct, and it governed the wrong object.

| Where | The check was right about | What it actually governed |
|---|---|---|
| 897 (D59, prior run) | the raw `--db-path` string | the file sqlite opened after URI stripping |
| 897 (this run) | `abs(ic - tgt) > tol` | an H where `tgt` was `None` — nothing was compared |
| 881D | the protection set it was handed | a set cached up to 300 s ago from a **different tree**, feeding an irreversible deleter |
| 904 | which directory it was authorised to write | a per-fold loop that had **already written** before the refusal fired |
| 904 (cycle 1) | a sentinel string inside the manifest | a credential copy-pasteable from git history |
| 901 | "has the VR curve levelled off?" | a one-step delta, on a curve still climbing to 6× |
| **the orchestrator** | 901's measured negative | a conclusion the evidence did not support (corrected in `2eaa844`) |

**A check that reasons about a stale, different, absent, or already-mutated object is not a check.**
Three specific shapes worth naming, because each has now cost a cycle:

1. **A cache in front of an irreversible action.** The protection set is the one input that may
   never be served from cache. Freshness cost: 1.694 s, against saves minutes apart.
2. **Write-then-refuse loops.** A per-item loop that writes as it goes cannot be fail-closed: decide
   every item, then write. 904's refusal printed "Nothing was written" while having written.
3. **A pass recorded for a comparison that never executed.** Now the rule in TASK_902 item 2:
   *a check that did not execute a comparison is not a passed check.* Two independent root causes
   were found for that one shape in a single file.

**Suggest the planner stand this up as a standing review lens** — "what object does this guard
actually hold, at the moment it acts?" — rather than seven separate fixes. Note also that **none of
these were caught by the agent that wrote the code**; every one came from an independent lens
briefed to disbelieve the claim.

## Remainders routed (all committed)

- **TASK_905** (seeded, `d1951bc`) — `validate --fold-root <copy>` resolves periods-file paths via
  `os.path.join(_REPO, …)`, the manifest's own baked-in path, so it re-reads the ORIGINAL tree and
  reports `ALL 36 checks PASS` on a tampered copy. Smoke's internal `_clone_fold_tree` already
  fixes this for its own use; the public CLI never got it.
- **TASK_902 item 2 amended** — now names BOTH root causes of the inherited-IC vacuity class.
- **TASK_902 item 2b added** — an absent/unreadable `--db-path` crashes uncaught at
  `probe.report_bar_frequency` before any artifact is written, and D58's contract is that
  orchestration parses the ARTIFACT, never the exit code. A refusal the orchestrator cannot read is
  indistinguishable from a run that never started.
- **TASK_906** (seeded, `7dd8d26`) — the 809/897 per-decision series is measurably NON-STATIONARY
  across its OOS window: between-chunk sd 0.09491 vs within-chunk SE 0.03538 (variance ratio 7.19),
  χ²₂₅ = 196.5, p = 1.4e-28 at H=1, replicated at H=5 and H=20, with a monotone downward drift. The
  two facts that rule out serial dependence as the cause: the bootstrap SE is FLAT in block length
  (+6.7% across a 30× change at H=1, +1.0% at H=20) and the series is ANTI-persistent (full-series
  VR 0.273 at q=2048). **This bears on every whole-window Sharpe claim on that series, not on one
  instrument** — and it may reframe D57's parking of 897's `CI lo > 0`, which assumed the missing
  ingredient was a DEPENDENCE instrument. Scope item 4 is the honest guard: a drifting INSTRUMENT
  looks identical to a drifting MARKET in this measurement, so alias/identity work, forward-filled
  stale prices, the frozen-cell mask and universe turnover must be excluded first.
- **Residual bare-`open` readers in 837** — the cycle-3 re-verify (MINOR 4) found the accepted
  disclosure named 3 of ~10. Full list is in that report; belongs with whichever 904 successor
  lands.

## Queue drift (planner to groom)

`tasks/QUEUE.md` still ranks 897 at row 1 and 904 at row 5. 897 is shipped; 904 is halted pending
D61's re-cut decision. The factory did not re-order the queue — banners and DECISIONS are the
truth.

## Process notes, disclosed not buried

- **A `safe_run`-wrapped job shows NO `safe_run` parent in the process tree**, because
  `tools/safe_run.sh:50` is `exec systemd-run --user --scope …`. The orchestrator misread an absent
  parent as an unwrapped run and said so to an implementer, which was wrong; the implementer
  correctly refused the claim. Every judge brief after that point carried the correction.
- **901's real-data leg ran on the workstation, deliberately**, under D54's data-local exception:
  the 11 GB extract cannot cross a ~5 Mbps uplink. It was `safe_run`-wrapped and the exception is
  recorded in the row's own artifact. The compute-shaped MC sweep ran on srv as normal.
- Several implementer lanes ran srv jobs `safe_run`-wrapped but **not inside tmux** — capped and
  short, but they would not have survived an SSH drop. Flagged by the agents themselves.
- **Orchestrator error, corrected in the open (`5e6f2ff`): a commit swept in another lane's files.**
  `git commit -F <file>` with no pathspec commits the WHOLE INDEX, and a concurrently-running lane
  had staged its own work — so TASK_904's halt record silently carried TASK_881D's `PROTECTED.txt`
  corpus, making an un-shipped, currently-red row look committed. The first correction attempt then
  failed a second way: `git commit -- <paths>` commits the WORKING-TREE state of those paths, which
  silently overrode a staged `git rm --cached`, producing a commit whose message announced a removal
  it did not perform. **Rule for a shared tree with live lanes: use an explicit pathspec AND verify
  `git diff --cached --name-status` before committing — neither habit alone is sufficient.** Both
  mistakes are recorded in the commit messages rather than quietly amended away.
- A test-stub contract drift was found in 897's suite: `fake_run_decision` returned an array where
  `decide._run_decision` returns a scalar `fill_frac`. Invisible for four cycles because no prior
  D60 test rendered an artifact. **A stub whose contract drifts from the real function stops
  testing the thing it names.**

---

## Late lanes (resolved after the first draft)

### TASK_881D — PARKED (D63, `1c0cb4f`). Green, BLOCKER dead, NOT shipped.

Cycle 1 shipped the protection model + deleter and authored the first real `PROTECTED.txt` corpus
the repo has ever had — 0 of 56 candidate files previously carried a `protected:` block, so the
protection model had been protecting **nothing**.

**The BLOCKER, and it was severe: a manifest-protected checkpoint WAS being deleted.** The in-run
pruner consumed a protection set cached up to 300 s (`_DEFAULT_PROTECTION_TTL_SECONDS`); the `stale`
flag fires only when a scan FAILS, so a scan that SUCCEEDED 299 s ago against a *different* tree was
authoritative for an irreversible deleter. Measured in the same second on the same tree: offline
`ckpt_gc.py` said `keeping 4` while the in-run pruner deleted the protected snapshot with
`stale=False` and no warning. **FIXED** — `get_protection(..., force_rescan)` with
`prune_run_dir_by_score` passing `force_rescan=apply`, keyed off `apply` itself so the delete branch
cannot be reached with a cached set; a failed fresh scan still refuses. Mutation-proven at the real
drop site (reverted → `1 failed`, restored → `1 passed`) at the DEFAULT TTL with `apply=True` and no
`sleep`. A full fresh scan costs 1.694 s against saves minutes apart.

Also closed in cycle 2: the two `--protect-file` readers UNIFIED (FIFO exit 2 where it hung at 124);
the falsified "the in-run deleter and the offline tool can never disagree about the same tree"
claims corrected to what actually holds; ingestion drift back to **EMPTY** (AST entity-diff over all
43 `INGESTION_NAMES` = `[]` after `load_manifests` / `_run_manifest_check` were reverted verbatim);
and D62's measured consequence closed by an ADDITIVE `UNTOUCHED_GUARD_NAMES` pin — `_safe_target`
×2, `_du`, `free_gb`, `_WALK_PRUNED_DIRNAMES` and four `_WHY` constants all mutate RED again.
**376 passed / 2 skipped** (baseline 351), selftest 12/12 both hosts, all 7 load-bearing checkpoints
byte-identical throughout, no `--apply` ever outside a scratch copy.

**Why it did not ship.** Two items have their ONLY drop site inside ingestion pinned by
`INGESTION_NAMES`, which this row's own scope item 2 forbids reopening: (i) MAJOR —
`--verify-manifests` exits 2 **permanently**, caused by this row's own corpus (3 acknowledged
dangling 828/scratch disk entries), so the one pre-flight signal the delete path does not repeat
carries zero information; `--known-dangling` was built, measured working on the real tree, and
reverted at the wall. (ii) MINOR — the `experiments/__pycache__` silent withdrawal, same wall.
A verdict cannot be green with an open MAJOR. And **at HEAD, `--ckpt-retention score` is refused at
parse time** — this row is what makes an unsupervised `apply=True` per-save deleter reachable at
all, so shipping it with a permanently-red pre-flight gate is the "always-red gate gets ignored"
hazard on exactly the path whose refusals the deleter does not repeat.

Work preserved: `lore/handoff_2026-08-24_factory_run_41_881D_cycle2.patch` (4321 lines, incl. the
1341-line suite and the corpus). Corpus un-tracked meanwhile (`5e6f2ff`); the three files now carry
a `TRACKING` block recording why they must be tracked once the row ships (untracked files never
reach `gpu-server` under the deploy contract, and that is where `ckpt_gc.py` runs).

**D63 recommends a narrow, WRITTEN scope amendment** permitting those two ingestion functions to
change, with `INGESTION_NAMES` re-pinned. Both walled changes are already built and measured — the
wall is procedural, not technical.

### TASK_901 — cycle 2 complete, corrections landed; re-verify running at time of writing.

Cycle 1 reported a total negative ("no candidate certifies out-of-family"). An opus correctness
judge, briefed to attack the negative as hard as a positive because a wrong negative closes a live
research direction, found it **only PARTIALLY sound**. Cycle 2 corrected it. Both cycles are worth
recording, because the row was wrong in a way that would have been acted on.

**What was wrong in cycle 1**

- **`variance_ratio` was condemned by its read-off rule, not the statistic.** The shipped rule
  returned at the first grid q whose ONE-STEP change was <10%; the docstring described "the smallest
  q at which the curve has plateaued". On the POPULATION VR of `0.2·AR1(φ=.99)` — zero sampling
  noise — it declared plateau at q=4 while the curve ran to 6.062 at q=256 (asymptote 8.92). So the
  shipped verdict's "sampling noise … mistaken for a genuine plateau" was factually wrong: the false
  plateau was deterministic.
- **`acf_band`'s Bonferroni band was applied in the fail-open direction**, making it blinder than the
  τ guard it replaces (0.0677 vs τ's 0.0365, against a population ρ(1)=0.0396 sitting between them).
- **Gate (iii) measured NON-STATIONARITY, not dependence** — see TASK_906, which this produced.

**What cycle 2 measured (1000 trials/cell, up from 200)**

- `variance_ratio` under the documented rule: **certify_rate 0.000** on the persistent component —
  gate (ii)'s REFUSE branch, a pass as worded. But it refuses **80.6% of i.i.d. draws at n=3000**
  too. The row states it plainly: **never correctly tested in cycle 1; measured correctly it is safe
  but unusably conservative** — not that it works. A φ=.999 leak is reported honestly: 8 certified
  draws of 1000, covering 0/8.
- `acf_band` with the corrected 1.96/√n band: readout on the persistent component **1.0 → 18.0**,
  coverage 0.7243 ± 0.0154 and 0.2625 ± 0.0153. **It genuinely fails** (13.4σ, 43.6σ). The restated
  mechanism: the crossing statistic DOES read the structure; what fails is converting a
  roughly-correct range into a covering interval at block ≤ 60.
- **A discrimination test nobody had run** (`paired_check.py`): restricted to chunks BOTH estimators
  certified, **8 of 9 estimator pairs contain the grand Sharpe on identical chunks, chunk for
  chunk**, at blocks differing up to 19×. Containment is retained as a description of the series and
  **withdrawn as evidence against any estimator, including retroactively against τ** — τ's synthetic
  0.805/0.376 is a different measurement and is left standing.
- Gate (iii)'s H shortfall closed (all of H ∈ {1,5,20} run). The `k=16 → 0 decisions` finding is
  kept and re-verified, and its justification corrected BY MEASUREMENT rather than assertion:
  `present_count_hist={7: 217438}`, `n_elig_hist={1: 217438}` — τ=0.9 binds on every bar, so the
  eligible pool is exactly 1 per side and `k` never binds. Universe restated as four underlyings
  with duplicate quote-currency listings plus a stablecoin FX pair — *a* real per-decision series,
  not "the 7 majors".

**Cycle 2 was then re-verified, and its REASONS were wrong again (top-line survived).** An opus
re-verify confirmed the negative — `acf_band`/`spectral_zero` fail conclusively (13–44σ) reproduces
exactly — and found four MAJORs, none reversing it:

- **The cycle-2 read-off rule is ITSELF an artifact.** It compares against a FIXED `plateau_tol ≈
  0.10` while the sampling sd of VR(q) at the tail reaches **0.319** — 3× the tolerance. 91% of
  refusals are driven by the last two grid points. Make the tolerance SE-aware and the SAME draws
  yield a plateau **100%** of the time, correct at q=2 in **97.7%**. And when the shipped rule does
  return on white noise it reads **the grid, not the data**: median 4/16/64 at n=200/1000/3000,
  exactly ¼ of `max_q = n//10` every time. So "unusably conservative" is STILL a rule-and-grid
  property, not a property of the statistic — cycle 1's mis-attribution, one level down.
- **Gate (ii)'s "pass" is the block ceiling, not the estimator.** On the persistent component the
  estimator returns a finite plateau in **16.4%** of draws; **163 of those 164 are killed by
  `MAX_BLOCK_CAP=60`**. A ceiling that rejects any reading >60 would refuse τ too.
- **"Never over-certifies" is contradicted by the row's own cell.** 0/8 against a 0.93 bar is
  p = 5.8e-10; the judge's independent reproduction gives 13 certified at coverage 0.308; pooled
  **4/21 = 0.19 [0.054, 0.419], p = 1.1e-16.** The row's own n<20 "unreadable" caveat is right for
  establishing a pass and wrong for refuting one, and as written it shielded the only conclusive
  evidence against the candidate.
- **"Monotone drift" is not in the data.** Slope p = 0.122 / 0.273 / 0.660, R² 0.097 / 0.052 /
  0.009; at H=20 the second-half mean is HIGHER than the first-half; the "last three chunks" line was
  a selected 3-vs-3 extract; detrending removes essentially none of the dispersion (7.19 → 6.77).
  **The load-bearing finding is untouched** — the pseudo-truth is not a constant — but the mechanism
  claim is struck. TASK_906 was corrected downstream (`d7b0d3f`) before anyone could act on it.

**Cycle 3 corrected the reasons and PASSED independent verification** — no BLOCKER, no MAJOR;
verbatim: *"the row's stated reasons now match its measurements … This ships as a measured
negative."* Cycle 3's own contributions are worth recording, because two of them corrected the
people checking it:

- **It refused the tempting fix.** On the read-off artifact it chose to DISCLOSE rather than adopt
  the SE-aware tolerance, reasoning that the SE-aware rule is *a different instrument* whose refusal
  behaviour would need re-certifying across all 21 cells — and that adopting it would be a parameter
  change made after seeing it improves a number. That is exactly the failure mode this row exists to
  prevent, declined at the moment it was tempting.
- **It overturned an opus judge, with measurement.** The re-verify had cleared `calibrate_k` as "not
  a hidden refusal channel". It is one: exactly **one stratum of 27** (`block=3, n_dec=3000`) fails
  validation at **0.925 vs 0.930** on a 200-trial batch (z = −0.27 — a coin flip), and that single
  stratum drives up to **32%** of some cells' non-certification. Reproduced bit-identically from a
  cleared cache by the final judge.
- **It reversed its own claim rather than defend it.** Instead of arguing the 0/8 threshold it re-ran
  the cell at 5× trials: 27 certified, coverage 0.185, p = 2.2e-21. "Does not over-certify anywhere"
  is explicitly **WITHDRAWN**. Four independent draws now pool to 15/81 = 0.185.
- **It contradicted the orchestrator's brief and was right to.** My brief claimed the "last three
  chunks" illustration was a selected extract; at H=1 the last three genuinely ARE the three lowest
  (`lowest3_chunks = [23,24,25]`). The row records that fact and rests non-significance on
  t/p/R²/Spearman/halves/non-replication instead. The final judge confirmed the implementer's
  rendering is the accurate one.

**The gate (ii) "pass" is mostly a ceiling, not an estimator.** The refusal split (verified to
partition and reconcile across all 63 cells) shows `MAX_BLOCK_CAP=60` doing the work: on the
persistent component the estimator returns a finite plateau in 16.4% of draws and **163 of 164 are
killed by the cap** — a ceiling that would refuse any estimator reading >60, τ included.

**What the three cycles actually bought:** the headline never moved, but the REASON attached to it
changed three times, and each version implied a different next experiment — (1) "the statistic can't
see the structure" → abandon; (2) "unusably conservative" → abandon differently; (3) "our read-off
rule and block ceiling reject it; the statistic's own behaviour is UNSEPARATED" → **the question is
open**. Only the third is what was measured.

**Headline as it now stands:** no candidate is certified as a general range instrument, but the
three fail DIFFERENTLY and the direction is **not closed**. `acf_band` and `spectral_zero` fail
conclusively; `variance_ratio` never over-certifies but refuses too much for ≥93% to be established;
gate (iii) as specified cannot discriminate between estimators at all.

**Consequence for TASK_897, and a correction to this document's own earlier draft:** 897's
`CI lo > 0` conjunct is **"not yet established", NOT "structurally unavailable"**. An earlier draft
of this file and of 897's banner said the latter, on a total negative that did not survive review.
897's banner was corrected in `2eaa844`.

**Two open questions handed forward:** (1) `acf_band` reads range 18 and *still* under-covers, which
points at the `MAX_BLOCK_CAP=60` ceiling and the matched-dependence calibration convention rather
than at the range read; (2) the non-stationarity — now **TASK_906**.

**Verified clean by the first judge:** 809 lineage byte-untouched; 897 machinery imported unmodified
with correct contracts; real leg reproduces bit-identically; no ACF off-by-one; `spectral_zero` is
Andrews (1991) verbatim and its failure genuine; refusal accounting honest; `gate_iv_hook.py`
fabricates nothing; `trading_data.db` never opened.

**Self-flagged risk worth the planner's eye:** `variance_ratio`'s gate-(ii) "pass" IS a refusal, and
read without the certify-rate column it will mislead. The implementer stated it three ways and still
called the shape inherently misreadable — that judgement is worth respecting when the row is read.
