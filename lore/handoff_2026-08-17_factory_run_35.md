# Factory run 35 — handoff (2026-08-16/17)

_Final. Run closed 2026-08-17 ~04:10._

Four lanes taken in parallel, fenced by file surface: **878** (eval-core failed-fill semantics),
**881** (checkpoint protection model), **891R** (run_queue venv shim), **895** (D41 docstring +
819 rider). The Ready queue was fully committed — nothing takeable was left unstarted.

| Row | Outcome | Cycles | Decision seeded |
|-----|---------|--------|-----------------|
| **895** | SHIPPED `task895-d41-docstring` (item 1 of 2) | 1 | **D44** — gate (iii) unsatisfiable |
| **878** | **SHIPPED** `task878-failed-fill-semantics` | 3 | — |
| **881** | **HALTED** (3-cycle rule) | 3 | **D45** — re-cut by capacity |
| **891R** | **HALTED** (3-cycle rule) | 3 | **D46** — delete the shim dir |

**Two halts is the rule working, not the run failing.** In both cases the MECHANISM is verified and
the SCOPE BOUNDARY was wrong, and in both cases the third cycle's own fix introduced the defect
that stopped it. D38 and D39 each refused a fourth attempt in exactly this position, and all four
re-cuts that followed those refusals — 876, 883R, 886, 888R/889 — shipped on the next try.

Twenty-one implementer/judge agents ran, on `sonnet` for first attempts and `opus` for every fix
and final verification, per `.dark-factory/config.json`'s escalation policy.

---

## The through-line of this run

**Every row that failed, failed at the VERIFICATION layer, not the mechanism layer.**

- **891R** built the correct shim — `exec` wrappers, not symlinks, exactly as D42 measured — and
  then probed it by EXIT CODE. `/bin/true` passes that probe. The mechanism was right; the check
  that it was the right mechanism was not.
- **881** correctly adopted `followlinks=False` at all four walk sites — and then dropped what it
  skipped without recording it. Worse, at `:490`/`:646` the skip is EXPLICIT (`if
  os.path.islink(fp): continue`) and never appended to `errors`.

Both are the same shape, and it is this repo's signature defect written twice in one night:
**a guard that establishes a guarantee ONCE and then lets it be consumed arbitrarily later, or a
path the code decides it cannot read being silently treated as a path with nothing in it.**
Absence read as a value.

It is also the D33/D36/D38 pattern — each cycle's fixing code containing the next defect one
layer down — appearing on two independent lanes simultaneously. Worth noting for the planner:
891R's suite passed **6/6 with 10 of 11 mutations killed** and still shipped two false-label
BLOCKERs, because the mutation set tested the mechanism it had just built and never the two
scenarios in which that mechanism's guarantee expires. **A strong mutation score is coverage of
the implementation, not of the guarantee.**

### The generalisation worth keeping

Both fix passes converged on the SAME corrective principle without being told to:

> **Make the verified thing and the executed thing identical.**

891R stopped delivering the interpreter via `PATH` and now execs the exact wrapper it probed —
and removed PATH-delivery entirely, because with both paths present each makes the other
unkillable by mutation. 881 stopped declining to look at paths and now resolves them, refusing
only where resolution would let an unreviewed file authorise a deletion.

In neither case was the original defect a MISSING check. In both, the check existed and was
correct — there was simply a gap between **what was checked** and **what was used**. That is the
same defect as D38's c3 (`realpath` guard vs `abspath` loader — a verdict about directory A
stamped on a run that graded directory B) and the same as D40's unpack-and-discard (five values
taken, three used).

**By the end of the run it had appeared EIGHT times, on THREE independent lanes, in one night:**

| # | Lane | The guard | What changed under it |
|---|---|---|---|
| 1 | 891R | shim identity probe | verified at parse time, consumed at a launch deferred by hours |
| 2 | 881 | `os.path.islink(fp): continue` | the path was declined, and the declining was never recorded |
| 3 | 878 | `gate_dec` | one conditional covering two distinct populations, so a decision about frozen cells silently applied to zero-coverage ones |
| 4 | 878 | `if hold_mask is not None:` | it meant "is anything masked at all" — true until `fill_gate` arrived and quietly made it false |
| 5 | 878 | the same guard, WIDENED to fix #4 | the condition learned the new state; one statement inside the body did not (`imprint.py:2753` still calls `hold_mask.mean()`) |
| 6 | 878 | `decision_cell_honesty.rule` | artifact PROSE asserting frozen cells are "force-HELD by the 868 gate" — false under `--fill-prob`; the same claim was corrected in two neighbouring places and missed here |
| 7 | 878 | branch (B), narrowed BECAUSE (A) was unconditional | the orchestrator's own instruction to gate (A) would have left (B) as sole cover while (B) excluded exactly the cells needing cover — caught by the implementer |
| 8 | 878 | `teacher_floor.hold_mask.rule` | asserted the operator masks "UNCONDITIONALLY on every path" — falsified 230 lines away by the gating fix requested to satisfy invariant #1 |

**Instances 6-8 are the dangerous group for this program specifically: artifact PROSE asserting a
rule the code no longer follows.** Those strings are what a reader trusts when they cannot
re-derive the number — the same failure `_ffill_bfill_cols` shipped for years and that TASK_895
corrected in this very run. A sweep for the CLAIM (26 hits across three files) found 4 false and 2
misleading; a sweep for the line numbers a judge happened to name would have found one.

Instance 4 is the purest form: the guard was correct when written, nothing touched that line, and
it became wrong because a NEW masking mechanism appeared beside the one it tested. No diff shows
that defect. Only running the thing does.

**Instance 5 is the most useful, because of HOW it was found.** After four occurrences the
orchestrator briefed the final judge to *"assume a fifth instance exists and go find it"* — and one
did, inside the code written that same cycle to fix instance 4. `imprint.py:2745` was widened to
`hold_mask is not None or fill_gate is not None`, `:2746` learned the `hold_mask is None`
fallback, and `:2753` did not: `float(hold_mask.mean())` → `AttributeError`, exit 1, on the
`--teacher-vectors` BC-target lane that gate (ii) exists for, after training completes and the
checkpoint is written. **Widening a guard is not one edit; it is one edit plus an audit of every
statement in the body.** Treating a known defect class as a base rate rather than a set of closed
incidents is what turned it up.

Each was initially filed as its own incident. It is ONE defect class, and it deserves naming in
the program's standing lessons rather than a fifth rediscovery. A useful review question falls
straight out of it: **"what does this condition MEAN, and is that still the only way to be in that
state?"**

---

## SHIPPED

### TASK_895 — `task895-d41-docstring` (`bfa8f97`), 1 cycle, item 1 of 2

The `_ffill_bfill_cols` docstring claimed *"Uses only same-series past/adjacent bars (no
cross-time leak)"*. Disproven by 893's value anchor and now replaced with a statement of the real
mechanism, the protection that holds today (`present_mask` warm=30), and the honest limit (a raw
`close_d`/`logc` reader gets none of it). D41(a) ruled the BEHAVIOUR stays; the diff is
comment-only. selftest 12/12, parity GATE 1a PASS, 893 positive control +0.000749.

The correctness judge re-derived SOL's 59,760 s and XDG's 420 s from the live 26 GB DB rather
than restating the ledger, and specifically hunted the D41 overclaim — the "structural corollary,
not a general fence" caveat survives intact.

**Item 2 did not ship. Gate (iii) is internally unsatisfiable → `D44` (OPEN).**

---

## DECISION-CHANGING FINDINGS

### D44 (NEW, OPEN) — TASK_895 gate (iii) contradicts itself, and it corrects the D41 record

Gate (iii) demands the prefix count move 4933 → 4917 **and** the bar stay bit-frozen at
`+0.9673047900277524`. Neither half is reachable, both verified first-hand:

- **The count cannot move from inside the row's surface.** `tools/measure_leading_prefix.py:118`
  hardcodes `_ADMIT`, a verbatim copy of the OLD `build_close_aligned` admission SQL, and Paths
  A/B (`:163`, `:178`) plus the value anchor (`:202`, `:218`) all re-query raw `ohlcv` through it.
  The script's own header says it "never executes or edits that prototype". The fix was applied,
  the panel regenerated, the tool re-run: **4933, bit-for-bit unchanged.** Moving the number means
  editing the measurement instrument — which is not an implementer's unilateral call, especially
  for an instrument backing a SHIPPED ledger row.
- **A genuine fix must move the bar.** Row 0's fabricated value EQUALS the first admitted close,
  which is exactly why it is invisible to log-difference signals (`logc[t]−logc[0] ==
  logc[t]−logc[1]`). Admitting the real earlier price breaks that identity for every lookback
  (L up to 90) reaching row 0 from a scored row (warm=31). Measured **+0.9673047900277524 →
  +0.9672495154672209**. Masking does not escape it either: masking suppresses POSITIONS while the
  fabricated PRICE stays in the signal input.

**The correction this forces on D41:** 893's Δ = 0.000000 held because the fill is VALUE-NEUTRAL,
not because row 0 is unreached. **Row 0 IS read, by every lookback spanning it.** "No scored
number moves" must never be re-quoted as "these cells are unreachable".

---

## VERIFICATION FINDINGS BY LANE

### 891R — **HALTED on the 3-cycle rule → `D46` (OPEN)**

All three cycles died in the same place, and naming it makes the fix obvious: **the row keeps
trying to make a JOB-WRITABLE DIRECTORY authoritative.**

Cycle 3 restored the PATH prepend (correctly, to close a coverage-shrink MAJOR) and that
**re-opened BLOCKER 2 one indirection out.** The header claims "verified, then executed, with no
lookup in between" — but two lookups sit in between and both use the already-prepended PATH:
`env … bash -c …` execvp's **`bash`** itself (`:386`), and the verified wrapper's own
`#!/usr/bin/env bash` shebang (`:766`) resolves at exec time. A job that plants `$SHIM_DIR/bash`
hijacks the whole job shell while `python`/`python3` stay **byte-identical** — so the `cmp` passes,
the probe passes (it runs from the QUEUE's PATH, which has no shim dir), and python2 runs under
`interpreter=venv-shim` at **rc 0 with a clean summary row**. Worse than HEAD, which claimed
nothing. Proven causal by an independent M15 rebuild. The precondition is exactly the threat model
the `cmp` was bought for — and the mitigation covers 2 filenames out of N.

MAJOR: `cmp` (`:781`) is the ONE unbounded call in a guard advertising boundedness.
`mkfifo $SHIM_DIR/python` wedges the launch-time guard, so later passthrough rows never launch and
no summary is written — **a hang HEAD never had**, and precisely what the timeout-free probe design
at `:133-136` promises to prevent.

**What is verified and should survive the re-cut:** both original kills DEAD (including a wrapper
that FORGES the exact expected `RQPROBE|3|…` line for `-c` and execs a stub otherwise — caught by
the byte-compare); gate (iv) clean; **six mutations independently rebuilt at their real drop sites,
6/6 red**; suites 16/16 + 16/16 + 18/18 + selftest 12/12. **The three-tier honest contract —
CLAIMED / BEST-EFFORT-EXPLICITLY-NOT-CLAIMED / NOT COVERED, limits re-derived from the mechanism
with real manifests cited inline — is the best-written artifact of the run.**

**D46(a) recommends deleting the shim directory entirely** and delivering the leading token by
absolute-path substitution (`python foo.py` → `<repo>/trader_env/bin/python foo.py`). Three cycles
of increasingly sophisticated defenses — exec wrappers, identity probes, byte-compares,
launch-time re-verification — were all spent protecting a directory that jobs can write to and
that **does not need to exist**. Removing it deletes the attack surface in one line; `trader_env/`
is already PROTECTED. Cost, stated honestly: chained/child `python`s become a documented
NOT-COVERED limit, so it ships with a one-time manifest sweep. **Needs OPERATOR ratification —
D42(a) ruled the mechanism.**

_(cycle-1 findings that produced the above:)_

1. **BLOCKER — the probe validates an exit code, not an identity** (`run_queue.sh:596-603`).
   Output is sent to `/dev/null`; only `rc` is read. Symlink a sandbox `trader_env/bin/python` to
   `/bin/true` and the queue prints `venv: … (python3 probe OK)` and labels jobs
   `interpreter=venv-shim` for something provably not Python.
2. **BLOCKER — the guarantee is established once and consumed arbitrarily later.** The shim
   self-probe runs once at `:684`; `:312` injects `PATH="$SHIM_DIR:$PATH"` at every launch with no
   re-check, including rows the scheduler DEFERS waiting for a slot — the queue's whole purpose.
   Strip `+x` mid-wait: PATH silently skips the shim, ambient python runs, exit 0, clean summary,
   and the line still says `venv-shim`. **This is strictly worse than the pre-891 baseline, which
   claimed nothing and was therefore honestly silent.**
3. **MAJOR — the shim outlives an interrupt during construction.** Built and self-probed before
   `trap cleanup INT TERM EXIT` is armed (`:722`). SIGINT during a slow probe → exit 130, dir
   survives, contradicting the code's own "on ANY exit path" claim.
4. MINOR: the no-`timeout` fallback runs a possibly-wedged interpreter unbounded at parse time,
   hanging the whole manifest including passthrough rows. MINOR: metacharacter-glued spellings
   (`python;true`) are one whitespace token and label `passthrough` — correct per the contract's
   literal wording and never a dangerous mislabel, but absent from the documented NOT-COVERED
   list. NIT: `LINE[]` is not rebuilt when refused rows are filtered (dormant).

**Upheld, do not change:** the `interpreter=` label on passthrough rows is literally what D42
item 2 ruled — an all-passthrough A/B differs only in that token plus timestamps, with
`queue_summary.md` and per-job logs byte-identical. `reap()`'s PID-hash summary ordering is
genuinely pre-existing (4 distinct orderings across 4 runs at HEAD). Scheduler 16/16 and watchdog
18/18 on BOTH HEAD and candidate — the known load-sensitive watchdog flake did not fire.

### 881 — **HALTED on the 3-cycle rule → `D45` (OPEN)**

Cycle 3's re-verify found a BLOCKER that **cycle 3 itself introduced**. `_parse_spec_block`
(`ckpt_gc.py:618`) consumes only the FIRST `protected:`/`covers:` block (`re.search`), while
`_strip_spec_manifest_blocks` (`:646`, ADDED in cycle 3) strips EVERY block (`while True`). The
strip removes strictly more than the parse consumes, so every block after the first is neither
manifest-protected nor advisory-rescued — a `SPEC.md` naming a checkpoint under a second
`protected:` block has it **DELETED at exit 0, zero diagnostic, by BOTH deleters**. Proven a
regression by monkeypatching the strip to identity (0 deletions, 3 kept). **It inverts ruled model
item 3 — "prose citations can only ADD protection, never subtract it" — using the code added to
honour it, and it is D38-c3's shape at the TEXT level: two functions disagreeing about what the
same bytes mean.** Plus two MAJORs (a malformed `protect:` line is reported and then ignored, so an
inline trailing comment gets the named checkpoint deleted at exit 0; and `_root_escape` receives an
empty allowed-list when `was_resolved` is False, so the in-run pruner deletes outside every
configured root for an out-of-tree non-symlinked run dir, contradicting its own docstring).

**The halt is the right outcome, not a failure.** Six opus judges across three cycles killed six
real defects — including one that would have deleted the clean-recipe anchor — and the protection
MODEL now holds under repeated attack. What kept failing was one unscoped layer: **manifest
INGESTION** (what counts as a manifest, what the scan can see, what a line MEANS). D45 recommends
the cut that has now worked four times (D33→876, D36→883R, D38→886, D39→888R/889): split ingestion
from the deleter, with ONE function owning both parse and strip so the strip can only ever remove
exactly what the parse consumed. Nothing is at risk meanwhile — the `--apply` freeze STANDS,
`--ckpt-retention` stays `recency`, and the real-tree dry run keeps all 7 checkpoints.

_(what the three cycles BUILT and VERIFIED — resume from this, per D45(a):)_

### 881 — cycle-2 fix (opus) detail

**What the fix did.** A new `_walk_scan_dirs()` replaces `os.walk` at all four drop sites. The
policy is now deliberately ASYMMETRIC, and the asymmetry is the interesting part:

- **Manifests (`PROTECTED.txt`/`SPEC.md`) → REFUSED when symlinked.** A manifest is the
  AUTHORITATIVE input — a `covers:` line is what UN-refuses a run dir and therefore ENABLES
  deletion. Model item 1 requires manifests be committed to the repo; a symlink's target may be
  untracked or outside the tree, so following one lets an unreviewed file authorise an
  irreversible deleter. Same rule for a real manifest reached only THROUGH a symlinked directory.
- **Advisory prose artifacts → READ (followed).** Add-only by construction, can never satisfy
  gate (i), so nothing read here can authorise a deletion. Refusing would block `--apply` on a
  whole tree for one benign doc symlink and make the dry run report "would delete" for
  checkpoints that ARE cited. A symlink that cannot be read at all is still an incompleteness.
- **Directories → FOLLOWED with realpath cycle detection.** A deliberate deviation from the
  orchestrator's instruction ("treat them as incompleteness"), and the right call: the repo really
  ships a self-referential symlink at `experiments/832_linear_imprint/ckpt_smoke/ckpt_smoke` →
  its own parent (**orchestrator-verified**), so the literal instruction would have produced a
  PERMANENT false refusal on the live tree. Following answers the question instead of guessing.

Torn `scores.jsonl` now REFUSES the run dir rather than warning — the file's own stated policy is
"missing data never causes a deletion decision", and an unreadable row is missing data whose
absence provably flips a keep into a delete. Extended to a corrupt `checkpoints_index.json` and an
unreadable score log (same failure direction, one rule instead of three).

**Evidence quality:** 40 tests (was 23), of which **13 were proven RED by reverting each fix in
place at its real drop site** and re-running, with all four files then restored byte-identically
(md5-verified against pre-revert backups). Four further tests are deliberate controls that pass
either way, including one guarding against an over-aggressive fix. Suites: 881 40/40, 876 20/20
(untouched this cycle), selftest 12/12. Orchestrator-reproduced real-tree dry run: 7 checkpoints
across 3 run dirs, **0 refused, 0 deletions, exit 0**.

**OPEN DESIGN QUESTION handed to the re-verify panel, not decided unilaterally.** The fix makes a
symlinked ROOT refuse. `ckpt_gc.py`'s default root `jax_ckpt_fast` IS a symlink (dangling today —
scratch disk unmounted), so once that mount returns the DEFAULT invocation refuses `--apply` and
labels dry runs INCOMPLETE until the operator passes the resolved path. Fail-safe and loudly
named, which suits an irreversible deleter — but **D38/TASK_888R ruled the opposite shape**: ONE
realpath-based resolver feeding both consumers, resolving honest spellings and refusing only
escapes, and a shipped documented symlink is arguably an honest spelling. Whichever way this
lands, it should be consistent with 888R rather than decided per-row.

_(Cycle-1 findings that produced the above:)_

1. **BLOCKER — a path the scan cannot read is silently dropped, and a protected checkpoint is
   deleted.** Two variants, both reproduced end-to-end under `--apply` with real deletion, exit 0,
   zero stderr. **(A)** symlinked FILE: `ckpt_gc.py:490`/`:646` do `if os.path.islink(fp):
   continue` and never append to `errors`, so a symlinked `SPEC.md`/`PROTECTED.txt` is treated as
   nonexistent. **(B)** symlinked DIRECTORY: `onerror` fires only on real OS errors, so
   `os.walk(..., followlinks=False)` omits a symlinked `experiments/<id>` with no signal at all.
   `followlinks=False` does NOT cover variant A. **This is the row's own named historical
   BLOCKER, recreated** — and neither test file contains the word "symlink" anywhere.
2. **MAJOR — offline and in-run deleters disagree about whether a symlinked run dir exists.**
   `collect()`'s `islink` exclusion fires only for directories DISCOVERED as a `dirname`, never
   for the ROOT argument. So `runs/my_run` as a symlink is invisible to `--roots runs` (`0
   checkpoints`, silent) while `prune_run_dir_by_score` — which passes the run dir AS a root —
   finds and plans deletion. D38 c3's disease one layer BELOW `plan()`, which is where the
   "cannot disagree by construction" argument was scoped. Dormant today; one ops action away.
3. **MAJOR — a torn `scores.jsonl` row is swallowed** (`:316-326`, bare `except Exception:
   continue`, no diagnostic, unlike every other path in the file). `scores.jsonl` is appended by
   concurrent writers with NO locking. Reproduced: the run's true best checkpoint (score 42.0)
   read as UNSCORED and recency-pruned, exit 0, no trace.
4. **MAJOR (orchestrator) — the conditional refusal has no test.** `cli.py:495-502` turned an
   UNCONDITIONAL refusal of `--ckpt-retention score` into one conditional on
   `retention_policy_available()`. The implementation is good (`parser.error`, explicitly refusing
   "rather than silently falling back"), but the test file has ZERO references to it and the
   replacement test exercises only the ACCEPT branch. An untested guard can be deleted invisibly.
5. MINOR: `_order_newest_first`'s docstring overclaims — within the same rounded second the
   epoch tiebreak precedes sub-second mtime. MINOR: `_maybe_prune_by_score`
   (`train_jax_fast.py:738-756`) is not wrapped in try/except although the startup check's comment
   promises a mid-run unavailability must not crash a healthy job.

**Confirmed GOOD:** the two edited 876 tests are legitimate supersession — 876's own shipped
banner names TASK_881 as the row that would implement `score` (the orchestrator and the regression
judge reached this independently). The **`_order_newest_first` third measurement CONFIRMS the
disputed D33 bug by execution** (`checkpoint_100`, the true newest, deleted while a 999 s-stale
`checkpoint_200` survived as "newest (resume)") and the fix ranks correctly — the correctness
judge's execution was right, the fail-closed judge's hand-trace missed the number/mtime
decoupling. The SCOPE ADDITION reader bug is genuinely closed: HEAD collapses a 4-checkpoint run
to `1 ckpts, 0 scored`, the candidate resolves `4 ckpts, 4 scored`.

### 878 — cycle 3 (final) RUNNING; cycle 2 failed its panel

**Cycle 2's panel returned a BLOCKER and three MAJORs, two of them caused BY cycle 2's own fix.**

- **BLOCKER, `imprint.py:2753` — the FIFTH instance of the defect class, in code written that same
  cycle.** The guard at `:2745` was widened to `hold_mask is not None or fill_gate is not None` and
  `:2746` learned the `hold_mask is None` fallback; `:2753` still calls `float(hold_mask.mean())`.
  `AttributeError`, exit 1, on the `--teacher-vectors` BC-target lane — THE lane gate (ii) exists
  for — after the walk, after training, after the checkpoint is written. The 121-leg smoke misses
  it because its CLI legs run without `--teacher-vectors`.
- **MAJOR — the disclosure IS forgeable; gate (v) fails.** Counters accumulate BEFORE the applying
  branch (`imprint.py:1425-1428` vs `:1429-1431`; `eval_vector.py:2700-2702` vs `:2703-2705`).
  Severing the application leaves PV reverting EXACTLY to the no-gate walk while the artifact still
  asserts 23 (resp. 33) failed fills and a per-slot breakdown. **An earlier correctness judge had
  concluded this was NOT forgeable — by reading. This one built TASK_890's severance mutation and
  measured it.** Reading found nothing; building the mutation found it.
- **MAJOR — `activity.n_long`/`n_short` are INTENT inside a block the artifact declares OUTCOME.**
  The invariant `n_long + n_short == n_trades_total` held in EVERY pre-878 artifact and is now
  broken by 23/14 in arms A/B. Arm A executes byte-identically to OFF (same PV 97959.0859375, same
  268 trades) yet its OUTCOME long count moves 87 → 108. **It escapes the artifact:**
  `scorecard.py:1126,1586` copy those seven keys verbatim into graded rows and carry NO
  `frozen_fill` companion (verified: zero references) — and per TASK_862 the scorecard is what a
  gate-ladder submission cites, so this is a wrong number in a promotion claim, not just an
  artifact field.
- **MAJOR — cycle 2's fix broke a DOWNSTREAM consumer.** Moving `hold_count` to the executed vector
  was correct, and it silently split `walk_pct_hold_agg` (now OUTCOME, `eval_vector.py:3017`) from
  the payload it exists to describe (the ATTEMPT). Measured header 13.4375% vs payload 9.0625%,
  **gap 4.375 pp against `TV_COMPOSITION_TOL_PCT = 0.5`**, so `imprint.py --teacher-vectors`
  refuses with a **false tampering diagnosis** and no `fill_prob` key in `provenance` to explain
  it. That breaks the exact 861 collect→imprint pipeline gate (ii) serves.

**The pattern in 878 is a SCOPE boundary, same as the other two rows.** Three findings running have
been "another consumer nobody audited" — `sanity[c]`, then `tv_fit`/`sanity_extra`, now
`walk_pct_hold_agg` and `scorecard.py`. Scope 2 says *audit every consumer of the mask*, and each
cycle audited whichever ones the last judge named. Cycle 3 was therefore instructed to **enumerate
the COMPLETE consumer set before writing anything, and to stop and say so if it exceeds one
cycle** — a scoping finding being worth more than a partial fix. If it comes back that way, 878
wants the same cut as 881 and 891R: the SEMANTICS change in one row, the CONSUMER AUDIT in another.

_(cycle-2 detail, still accurate for what it fixed:)_

**Cycle 2 closed everything, each with a revert-proven RED test.** The BLOCKER fix hoists
`_zero_coverage = dec_prices <= 0` out of BOTH conditionals so it is unconditional, leaving the
frozen term as `(~gate_dec) & ~_zero_coverage` — the two concerns now textually separate, with the
OFF-parity algebra (`hold_mask_OFF == zero_coverage | frozen`) stated in-comment. It deliberately
adds no artifact key (that would be a leaf on the OFF path) and prints only when the mask bites,
so canon stdout stays byte-identical too. MAJOR-1 became a REFUSAL (`resolve_fill_prob_mode` raises
when `--fill-prob` meets `open_gate is None`) rather than an honest-recording block, on the TASK_890
argument: an artifact whose every number carries pre-878 semantics under a flag naming post-878
semantics is the hazard, and refusal costs nothing. MAJOR-2 routed `hold_count` to the executed
vector while `action_hist` stays INTENT, with INTENT companions added inside the flag-gated block —
renaming was unavailable because a rename is itself a differing leaf.

**FINDING 4 turned out to have FOUR sites, not one** (root in `walk_env`, plus `sanity[c]`,
`tv_fit.active_cells_only`, `sanity_extra.active_cells_only`), two of them pinned-path-only and
therefore unreachable by the flow-verifier that found the first. **FINDING 5's root fix also closed
a hard crash**: on a zero-frozen panel `eval_vector.py:3363-3372` indexed `["frozen_fill"]`
unguarded — `KeyError`, proven by execution rather than asserted, and it required reverting BOTH
hunks to reproduce (the implementer flagged that its first attempt did not reproduce).

**Verification:** 878 smoke **121 PASS / 0 FAIL**; full 861 recipe **12/12, 1013.1 s, tracked
baseline pins IDENTICAL**; selftest 12/12; 863 all legs (247.8 s); 865 all legs (411.4 s).
**OFF-parity re-measured on all three pairs, 0 keys added and 0 removed**: imprint 953/953 leaves
with 2 differing (`checkpoint`, `db`), eval_vector search 248/248 with 6, eval_vector
**policy-argmax** — the operator whose masking changed — 248/248 with 6. Every differing leaf is in
the documented invocation-volatile set; `ledger_row` differs only in its embedded `out_pointer`.
The unconditional mask is proven INERT on canon: unpinned minute 0/320 zero-price decision cells,
unpinned daily 0/300 with `open_gate=None` (so `--fill-prob` now refuses there), and the daily
loader structurally REFUSES rather than densifying a flat column. Fixture untouched.

**`step_vector` got no guard, deliberately** — not provably inert across self-play, training
rollouts, MCTS leaf, `trader/` and both experiment engines, so proving it exceeds this row while
risking canon bit-identity. It got a CALLER CONTRACT docstring naming the measured corruption plus
an executable characterization leg so the contract cannot rot. `jax_trading_env.py` stays **purely
additive, 205 insertions / 0 deletions**, its only non-878 hunk being 21 docstring lines.

_(cycle-1 panel findings that produced the above:)_

**BLOCKER — `--fill-prob` makes zero-coverage slots TRADEABLE on eval_vector's policy-argmax path.**
Gate (iv) ("nothing quotable ⇒ nothing attemptable") holds in `imprint.py`, where the `price<=0`
term is UNCONDITIONAL (`:2359-2360`) and `include_frozen` toggles only the FROZEN term. It does not
hold in `eval_vector.py`: `:3178`'s zero-price mask fires only when a panel frame is pinned, and
`:3192`'s mask is now gated `if fill_prob_mode is None` — **so enabling the frozen-cell feature
silently switched OFF the zero-coverage guard.** On the canon UNPINNED minute panel with
`--fill-prob` set, neither fires; `compute_frozen_fill_gate` deliberately excludes zero-coverage
cells from `frozen_mask` so the coinflip cannot backstop it; and `step_vector`
(`jax_trading_env.py:1354`) has NO internal price-validity guard — masking is entirely the
caller's job. Reproduced: `prices[:,1]=0`, `action=[HOLD,BUY]` → cash 100000→50000,
`pos_size[1]=5e14`, and PV **2.5e16** once that slot later prints. Breaks gate (iv) outright and
gate (iii) the moment a zero-coverage slot starts printing — a new listing or a healing gap.

**The root shape: two distinct conditions sharing one conditional.** The task correctly separates
"no data at all" (force-HOLD, always) from "data but no print this bar" (attempt, maybe miss), and
the code comment at `:3188-3191` reasons correctly about the FROZEN case — but `gate_dec` covers
both populations, so a decision made about frozen cells silently applied to zero-coverage ones.

**Why the suite could not catch it:** `smoke_failed_fill.py`'s only real-CLI leg invokes
`imprint.py` exclusively, and its gate-(iv) leg unit-tests imprint's helpers in isolation.
**Gate (iv) was proven on the one implementation where the separation already held.** The
`eval_vector.py --operator policy-argmax --fill-prob …` combination on an unpinned panel was never
run by the suite that exists to prove it.

**MAJOR — the activity metrics count ATTEMPTS as OUTCOMES**, i.e. the scope-2 consumer audit was
not done for one family of fields. `:2713` builds `hold_count` from `av` (the attempt) rather than
`exec_vec` (the executed vector), and `trades_per_asset`, `pct_hold_per_asset`, `pct_hold_agg`
(`:2734-2738`) and the whole `activity` block (`:2743-2744`, `:2789-2805`) derive from it. Before
878 the two arrays were always identical, so this was ACCIDENTALLY outcome-correct; 878 created
the divergence and re-routed nothing — while correctly documenting the choice for
`action_hist_pct` and keeping `avg_long`/`avg_short` outcome-based. Measured: 40 attempts, 17
fills, 23 misses, PV unmoved — and the artifact reports `trades_per_asset [40]`, `pct_hold_agg 0.0`.
**This lands on the row's own headline deliverable: in the scope-6 ablation, arm A ("never fill")
would report `n_trades_total`/`active_assets` IDENTICAL to arm C ("always fill"), making the
planner's real-data read uninterpretable in exactly the dimension it exists to measure.**

**MAJOR — an explicit `--fill-prob` can be silently inert.** `eval_vector.py:3143`'s
`if fill_prob_mode is not None and open_gate is not None:` has no `else`. On a gate-free panel the
flag is a complete no-op with NO stdout line (zero `"[878]"` prints in that file, unlike
`imprint.py`) and NO artifact key, against scope 1b's "whatever is used is RECORDED in the
artifact". MINORs: imprint's message conflates "ran, found none" with "could not run"; the
`None`-only OFF sentinel means a hand-built args object using `False`/`0` gets arm A rather than
true OFF (and `test_instrument_agreement.py:236` already builds args via `SimpleNamespace`);
`causal_print_density`'s t=0 empty-window floor yields `p=0` for a fully liquid slot (unreachable
today, both call sites satisfy `start_row >= lookback`, but unenforced).

**What the panel CONFIRMED as sound** — the coinflip is statistically clean (per-slot KS vs
Uniform p≈0.55-0.64, slot-slot and lag-{1,2,3,10} autocorrelation within the √N noise floor,
marginal miss rate tracking 1−p across p∈{0.1..0.9}); PV conservation exact for miss-on-open,
miss-on-close and miss-on-reversal; `parse_fill_prob` refuses `"AUTO"`, `""`, `nan`, `inf`,
`"0.5x"`; every gate uses `is not None` rather than truthiness, so `--fill-prob 0` genuinely
reaches the never-fill path; and the disclosure counters are NOT forgeable — they accumulate from
the same local `fill_miss_mask[k]` that drives the override, so TASK_890's decoupling defect does
not recur here.

_(cycle-1 evidence, still valid for the OFF path:)_

**Full 861 recipe (no `--quick`, canonical seeds): 12/12 legs PASS, 1169.9 s.** Byte-identity vs a
HEAD worktree with the flag OFF: **1 differing leaf each** — `imprint.py` → `db`, `eval_vector.py`
→ `ledger_row`, both environmental path strings, every scored value bit-identical. New smoke
`experiments/878_failed_fill/smoke_failed_fill.py`, 6 legs / 37 checks, **orchestrator-run
independently: ALL LEGS PASS**. selftest 12/12. No tracked baseline touched (orchestrator-verified:
`git status` clean on `experiments/861_search_teacher_imprint/*.json`, tracked baselines still
carrying their Aug 14 19:54 timestamps from `1049152`; the suite writes gitignored `*_local.json`).

**The judgement call that made it green — worth keeping.** The first iteration emitted the new
disclosure keys as present-and-`null` when the flag was OFF. That produced **4 differing leaves**
and turned three 861 legs red (`i_canon_835`, `ib_weight_on_pin`, `vii_baseline_pins`), and the
lane reasonably proposed promoting the baselines to absorb them. **Promotion was refused.** Four
leaves of `null` is not bit-identical — the house standard is ZERO (890 shipped 615/615 at diff 0;
863 proved it five ways) — and emitting `fill_prob: null` when the feature never ran is the
artifact asserting something about a mechanism that was never engaged, the same absence-read-as-a-
value inversion pointed the other way. TASK_883R's precedent already settled the shape: ship the
positive claim STRUCTURALLY ABSENT. Making the keys absent when OFF turned all three legs green on
their own, with **no operator authorization and no oracle re-based**. Had we promoted, three
tracked baselines would now encode "878 added some keys" as canon.

Orchestrator-verified primitives (measured directly, not taken on report):
`fill_coinflip_uniforms` is deterministic, seed-sensitive, in range, and **prefix-stable** —
recomputing with 90 decisions leaves the first 50 rows bit-identical, which a sequential RNG
stream cannot do. `causal_print_density` is exactly causal — corrupting rows ≥45 leaves decisions
0..15 untouched and first perturbs decision 16, and decision 15 IS row 45 with its window
`[25,45)` correctly excluding its own row. `parse_fill_prob` refuses −0.1, 1.1, `"nope"`. The flag
is `default=None` = OFF and OFF preserves 868's force-HOLD.

Gate (ii) — the 78.8%-class hazard — is pinned by a REAL mutation, not a strawman: the test calls
the actual pre-878 guard (`frame_hold_mask(..., include_frozen=True)`) at the true drop site and
asserts the two disagree, so it goes red if the old behaviour is restored. Gate (ia) ships a
correct negative control: a naive sequential CONDITIONAL-draw stream that PASSES the same-seed
rerun and FAILS path-independence — the distinction a same-seed rerun alone cannot make.

**Scope 6 (the 3-arm never-fill / coinflip / always-fill ablation) is PLANNER-SCHEDULED real-data
work and correctly NOT in this deliverable.** The lane built the ability to run the arms and
demonstrated them on the fixture; the binding read is the planner's.

**Intervention worth recording:** the lane was about to certify OFF-parity with `smoke_e2e.py
--quick`, which `:1318`/`:1331`/`:1306` show SKIPS the tracked-baseline comparison entirely — a
documented silent downgrade that TASK_867 shipped a named refusal against. Redirected to the full
recipe plus a direct HEAD-worktree byte-identity diff. Had that stood, 878 would have shipped on a
parity proof that proves only that the run completed.

_(superseded section header:)_ ### 878 — earlier PENDING notes

Orchestrator-verified primitives (measured directly, not taken on report):
`fill_coinflip_uniforms` is determinstic, seed-sensitive, in range, and **prefix-stable** —
recomputing with 90 decisions leaves the first 50 rows bit-identical, which a sequential RNG
stream cannot do. `causal_print_density` is exactly causal — corrupting rows ≥45 leaves decisions
0..15 untouched and first perturbs decision 16, and decision 15 IS row 45 with its window
`[25,45)` correctly excluding its own row. `parse_fill_prob` refuses −0.1, 1.1, `"nope"`. The flag
is `default=None` = OFF and OFF preserves 868's force-HOLD.

**Intervention worth recording:** the lane was about to certify OFF-parity with `smoke_e2e.py
--quick`, which `:1318`/`:1331`/`:1306` show SKIPS the tracked-baseline comparison entirely — a
documented silent downgrade that TASK_867 shipped a named refusal against. Redirected to a full
recipe plus a direct HEAD-worktree byte-identity diff.

---

## OPERATOR ASKS — four OPEN rows, ranked by what unblocks most

1. **`D46` (NEW) — needs OPERATOR ratification specifically, because D42(a) ruled the mechanism.**
   Delete the shim directory; deliver the leading token by absolute-path substitution. Three cycles
   of escalating defenses were spent protecting a job-writable directory that does not need to
   exist. D42's symlink measurement is not contradicted — it simply stops applying when there is no
   shim. **The cost is real and should be priced in: chained/child `python`s lose best-effort
   coverage, so the re-cut ships with a one-time sweep of the tracked manifests** (`runs/860_codistill/manifest_860.txt:6-8`).
2. **`D45` (NEW)** — re-cut 881 by capacity: `881I` (ingestion; one function owns parse AND strip,
   returns `(entries, spans)`; deletes nothing) → `881D` (model + deleter, resumed from verified
   work). Planner-rulable; the same cut has shipped four times.
3. **`D44` (NEW)** — the 819 rider re-cut: one row owning BOTH the prototype and the instrument,
   with a pre-registered tolerance replacing the bit-match, since a genuine fix must move the bar.
4. **`D43`** — unaddressed this run. **No 857 number is quotable from the distill CLI path until it
   answers.**

**Nothing is at risk while these sit.** The `--apply` freeze stands, `--ckpt-retention` stays
`recency`, the halted 891R work is unshipped so the queue keeps its pre-891 behaviour, and the real
tree keeps all 7 checkpoints.
3. ~~**STANDING:** the 861 `--promote-baselines` for `/provenance/alias_union_rule`.~~
   **RETIRED 2026-08-17 — it was already done and the handoff chain lost track of it.** The 878
   lane observed `collection_pin_fresh_vs_committed` reporting OK; rather than accept that on one
   observation, the orchestrator found the commit: **`1049152` (2026-08-14) "861 baseline re-base:
   TASK_877's alias-union description, promoted with per-field attribution"**, touching exactly
   `results_fixture_smoke.json` and `chosen_vectors_fixture.json`. So the authorization D39 asked
   for was granted and executed on 2026-08-14 — but run 34's handoff never recorded it, so the ask
   was still being carried as outstanding two runs later, and this run briefed an agent to expect a
   red that no longer exists. **Process lesson: an executed operator authorization needs to be
   written back into the handoff chain, or the factory keeps reasoning around a phantom red.**

## PRE-EXISTING REDS AT HEAD (measured this run; NOT caused by any lane)

`tools/tests/` is **320 passed / 3 failed** on the working tree, and 881's fix agent attributed two
of them to "another lane's uncommitted `mcts_trader/env/jax_trading_env.py`" — i.e. to 878.
**That attribution is WRONG, and I checked rather than relaying it.** Run in a clean HEAD worktree
at `bfa8f97`, before any uncommitted lane work:

- `test_jax_env.py::TestLimitOrders::test_limit_buy_places_order` — FAILS at HEAD
  (`assert 0.0 > 0` on `state.order_reserved[0]`, `test_jax_env.py:380`)
- `test_jax_env.py::TestLimitOrders::test_limit_order_expiration` — FAILS at HEAD
- `test_physics_core.py::test_physics_signals` — `NameError: name 'flax' is not defined` at
  `test_physics_core.py:24`, a missing import in the TEST file

All three are pre-existing. 878 touched `jax_trading_env.py` **purely additively** (three new
functions at the end plus a `numpy as np` import) and is not responsible. Left uncorrected this
would have sent the next run hunting a regression in 878's lane that does not exist — the second
misattribution tonight that surfaced only because HEAD was measured instead of the report being
believed (the first: a standing operator ask that had already been executed at `1049152`).

Worth an XS row: three tests have been red at HEAD long enough that agents now trip over them and
blame whoever is nearby.

## FOLLOW-UP ROWS SURFACED BY 878 (deliberately NOT built)

1. **Reproducible fill-gated collections.** `--collect-chosen-vectors` + `--fill-prob` is now
   REFUSED at argparse, and the reasoning is worth preserving: it is not a disclosure problem, it
   is an **unsatisfiable contract**. A collection artifact promises that replaying its stream
   reproduces the score its own walk recorded (`provenance.walk_net21`), and
   `imprint._tv_replay_check` enforces that at `TV_WALK_SCORE_REL_TOL`. Under `--fill-prob` that
   score is coinflip-dependent while the payload is the deterministic ATTEMPT stream (gate ii), so
   **a consumer replaying the payload cannot reproduce it BY CONSTRUCTION, not by corruption** —
   measured: the provenance and composition gates both PASS and the run still dies in the replay
   check. Lifting the refusal means letting a consumer rebuild the exact gate (seed, mode, rows) in
   a second engine — a real cross-engine feature. **The header work is retained and is the
   prerequisite for that row.** Note what was NOT done: weakening the replay check to accommodate a
   combination nothing asks for yet. When a new feature cannot satisfy an existing gate, the gate
   is not the thing that should move.
2. **Arm-matched teacher floor.** Threading the fill gate through `compute_teacher_floor` changes
   what `corridor_position` measures, so it needs a decision rather than a fix. Today the floor
   keeps pre-878 semantics under `--fill-prob` and the exemption is disclosed.
3. **`scorecard.py`'s missing `frozen_fill` passthrough** (another lane). `:1108-1130` copies the
   activity tuple verbatim into graded rows with no `frozen_fill` companion (verified: zero
   references in that file). Making `n_long`/`n_short` OUTCOME restores
   `n_long + n_short == n_trades_total` in graded rows, so the inconsistency is gone — but a
   scorecard reader still cannot see the fill disclosure, and per TASK_862 the scorecard is what a
   gate-ladder submission cites.

## RECOMMENDED PLANNER ROW (new, surfaced by 878, quantified by the orchestrator)

**Two legs of the same suite are both called parity pins, and only one of them can catch a schema
change.** `i_canon_835` does a full exact-object comparison (`canon_json(strip(got)) ==
canon_json(strip(want))`), so it rejects any added key. `i_canon_832` compares
**`CANON_832_PINS` — exactly EIGHT scalar values** (`policy_net21`, `floor_net21`, `floor_gross`,
`teacher_action_agreement`, `policy_ce_first`, `policy_ce_last`, `gate_rel_diff`,
`final_pv_net21`). It therefore cannot detect an added key, a removed key, or any movement in any
field outside those eight.

878 demonstrated this live: its additive disclosure keys passed `i_canon_832` and were caught only
by `i_canon_835`. **The hazard is the natural misreading — "the 832 canon pin is green, so the
change is canon-safe"** — which is exactly backwards for anything structural. A future row that
adds or renames a field gets one green and one red from two legs that present as equivalent.

Not fixed here (out of 878's scope, and re-scoping a pin mid-row is how a baseline stops meaning
anything). Recommend an XS row that either widens the 832 leg to a full-object comparison with an
explicit volatile-key strip list, or renames it so it cannot be read as a parity guarantee.

## WHERE THE HALTED WORK LIVES

Both halted rows are preserved as **path-limited** `git stash` entries (path-limited so the 878
lane, which shipped, was never touched). **Select them by MESSAGE, not by index — indices shift,
and there are now TWO 891 stashes:**

- `"TASK_891R run_queue venv interpreter — HALTED on the 3-cycle rule (D46)"` — **this run.**
  Contains the identity probe, the byte-compare, exec-through, the launch-time re-verify, the
  restored PATH prepend, the trap fix, the GPU/relaunch coverage, and the three-tier honest
  contract. Files: `tools/deploy/run_queue.sh`, `tools/tests/test_run_queue_interpreter.sh`.
- `"TASK_891 run_queue venv interpreter — cycles 1-3, HALTED on the 3-…"` — **run 34's earlier
  halt**, also reachable at tag `stash-891-halted-d42`. Do not resume this one by mistake.
- `"TASK_881 checkpoint protection model — HALTED on the 3-cycle rule (D45)"` — this run. Files:
  `tools/ckpt_gc.py`, `mcts_trader/training/{safe_checkpoint,cli}.py`, `train_jax_fast.py`,
  `tools/tests/test_ckpt_{876,881}*`, and the three `experiments/*/PROTECTED.txt` manifests.
- `"TASK_876 score-based checkpoint retention — BUILT + VERIFIED, STOP"` — the
  `stash-876-rescoped-half` half that 881 was resumed from; still needed by the 881 re-cut.

Each stash message states what is VERIFIED and what HALTED it, so a resumer does not have to
reconstruct that from the decision rows. Selftest is 12/12 with both stashed (verified after
stashing, not assumed).

## SAFETY STATE AT END OF RUN (orchestrator-measured, not taken on report)

Twelve-plus agents ran this session, several building sandboxed deleters and venv stubs. The
protected surface is intact:

- **`fixtures/trading_fixture.db` — identity pin HOLDS.** `b0f8de8b108b5052…`, **79,310,848 bytes**,
  mtime still 2026-08-05 — exactly TASK_873's pin, the generation every committed baseline in the
  repo was promoted against. No lane regenerated it. This is the run-25 incident class (an agent
  regenerated it mid-run, `b0f8de8b` → `59525bd3`, while the clean-tree gate stayed green because
  the fixture is gitignored, and it was recovered by luck from a 47-second-old copy).
- **`trading_data.db` untouched** — mtime still 2026-07-26 20:47, 26,245,337,088 bytes.
- **No tracked baseline modified** — `git status` clean on `experiments/861_search_teacher_imprint/*.json`;
  the tracked `results_fixture_smoke.json` / `chosen_vectors_fixture.json` still carry their
  Aug 14 19:54 timestamps from `1049152`. `--promote-baselines` was never run.
- **`runs/819_single_asset/_close_broad_daily.npz`** (gitignored frozen panel) verified by sha256
  AND mtime (still 2026-06-28) after a regenerate-then-revert in the 895 lane.
- **`runs/893_prefix_cells/prefix_cells_real.json`** was rewritten by a judge's default `--out` —
  sha256 recomputed as `6bede1e9…`, EXACTLY the LEDGER pin, so no pin moved.

## PROCESS NOTES

- **A verification agent wrote a pinned artifact.** The 895 correctness judge ran
  `tools/measure_leading_prefix.py` with its default `--out` and rewrote
  `runs/893_prefix_cells/prefix_cells_real.json` — gitignored, therefore invisible to the
  clean-tree gate. No harm: sha256 recomputed as `6bede1e9…`, EXACTLY the LEDGER pin, so the pin
  holds and 893's determinism claim is incidentally re-proven. But the brief said "edit nothing"
  and a default argument still wrote. **Fencing must name paths, not merely forbid editing** —
  this is run 34's lesson recurring despite being briefed.
- The frozen 819 panel `_close_broad_daily.npz` is gitignored under `runs/*/*`. It was
  regenerated and reverted during 895; verified untouched by sha256 AND mtime (still 2026-06-28).
  mtime is the stronger check — a regenerate-then-restore would move it.
- A bare git worktree does NOT contain `trader_env/`; without symlinking it in, several 891R
  gates fail spuriously. Rig artifact, not a code defect (matches `feedback_head_control_worktree_rig`).
- **Pre-existing, for the planner, NOT fixed here:** `tools/ckpt_gc.py`'s module docstring (`:72`)
  advertises `.db` as a forbidden path component, but `FORBIDDEN_PARTS` (`:101`) is
  `(".git", "trading_data", "fixtures", "backups")` — no `.db`, identical at HEAD. Narrow in
  practice since `trading_data` already covers `trading_data.db`, but it is a code/comment
  mismatch inside a deleter.
- Stale worktrees left by earlier runs: `/var/tmp/judge865/{BASE,CAND}`.
- **SECOND ORCHESTRATOR ERROR, caught by the implementer — and it would have been instance SEVEN.**
  On 878's MINOR-3 I instructed: gate guard (A) on
  `(fill_prob_mode is not None) or (panel_frame is not None)`, so invariant #1 holds literally.
  Followed as written, that OPENS A HOLE. Cycle 2 had narrowed branch (B) to
  `(~gate_dec) & ~_zero_coverage` **precisely because (A) was unconditional**; gating (A) off on the
  flag-absent unpinned path leaves (B) as the sole cover while (B) excludes exactly the cells that
  need covering. The implementer caught it, restored (B) to the full `~gate_dec` union, and pinned
  all five paths in a new leg. **The coupling was invisible in the line I asked to change** — which
  is the defect class itself, arriving in an instruction rather than in code: I changed what a
  guard MEANS without auditing what else had been written to depend on that meaning.
  **Standing clause: when narrowing or gating a guard, first find every other guard that was
  relaxed BECAUSE of it.** A brief that names one site is a brief that has not done that search.
- **ORCHESTRATOR ERROR, caught before it cost anything — worth a standing clause.** On receiving
  878's fail-closed BLOCKER I immediately spawned a fix implementer onto `eval_vector.py` /
  `imprint.py` while THREE verification agents were still measuring those exact files — a
  regression judge ~20 minutes into a full 861 run and a byte-identity diff, a correctness judge
  running semantics probes, and a flow-verifier driving both CLIs. That is **TASK_851's incident
  class exactly**: its headline numbers proved unreproducible because a concurrent TASK_852 edit
  transiently mutated the shared `eval_vector.py` mid-review, and root-causing it cost a whole
  investigation. Caught before any write landed; the fixer was held to read-only analysis until
  the panel cleared. **Standing clause for future runs: a fix pass must not be spawned onto files
  a panel is still measuring — either wait for the panel, or give the fixer a worktree.** The
  parallel-lane FILE fences worked all night; what I missed is that a fence also has a TIME
  dimension, because a judge and a fixer can be disjoint in intent and still collide in the tree.
