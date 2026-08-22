# TASK_903 — constant-provenance audit: 892's calibration table, 875's walk-forward numbers

**Status**: audit complete. **No verdict changed** (out of scope per TASK_903 and the SPEC.md
pre-registration discipline both lanes carry) — and both load-bearing verdicts now rest on
numbers that were actually re-derived: 892's ANTI-EDGE band survives the corrected calibration
(892-C4), and 875's "no OOS edge" reproduces exactly (875-C1).

**This audit is HOST-DEPENDENT and says so.** `runs/875_wf/` and `runs/892_action_ev/` are
gitignored and live on gpu-server, so the honest classification of 875-C1 and 892-C4 differs by
where the gate runs. An earlier revision hardcoded "unreachable from the workstation" as the
875 finding; running on gpu-server it FAILED loudly and correctly, and 875-C1 is now
**REPRODUCES** (re-derived read-only via `grade_correct.py` — no re-run of the 8h57m battery).
Run this gate on gpu-server for the complete picture; from the workstation it SKIPS the two
srv-only gates and exits 2, claiming nothing about them.

**Checkable enumeration (gate i)**: `test_903_constant_provenance.py`, `MANIFEST` (8 entries,
9 declared gates — 892-C2 and 892-C5 share one function). Run:
`trader_env/bin/python experiments/892_action_ev/test_903_constant_provenance.py`
(~2m16s, CPU only, fixture db only). On **gpu-server** (all 9 gates run): 4 REPRODUCE, 2
DRIFT-with-named-cause (both non-material), 2 PROVENANCE-UNRECOVERABLE (non-material,
disclosed), 1 REPRODUCES with a disclosed fake-null caveat (875 — `WF_SUMMARY.json` is `[]`).
On the **workstation**: 2 gates SKIP fail-closed and the run exits **2**, INCOMPLETE by
construction — a skip never exits 0.

**Exit-code contract, and why the summary is not a fixed sentence** (cycle-3 MAJOR 1/2 fix —
the earlier version of this instrument could certify a finding it had not made):

| exit | meaning |
|---|---|
| `0` | every declared gate RAN and every check passed |
| `1` | at least one check FAILED |
| `2` | nothing failed, but ≥1 gate was **SKIPPED** fail-closed — the run is **INCOMPLETE** and makes **no** provenance claim about the skipped gates |

The closing summary is assembled at run time from `Audit.findings`, entries each gate appends
*as it concludes*; there is no hardcoded sentence listing the classifications. A gate that
skipped, failed, or crashed contributes nothing to it, and every declared gate that reaches
none of {finding, skip, failure} is itself reported as a failure. Two gates can legitimately
skip on a checkout other than this one — **892-C4** (its input `runs/892_action_ev/` is
gitignored, so it is absent on a fresh clone, on CI, on the server and in any git worktree) and
**837-C1** (it refuses if `experiments/837_walkforward/` is dirty). Both now exit `2`, name
themselves, and are excluded from the findings list; before the cycle-3 fix the first was
invisible and the second still exited `0` under a banner asserting both of their
classifications. Demonstrated by re-running the superseded instrument and this one side by side
in a disposable worktree (old: `EXIT=0`, "1 gate(s) SKIPPED" with two actually skipped, banner
claiming `837-C1 REPRODUCE` and the 892-C4 DRIFT; new: `EXIT=2`, "2 gate(s) SKIPPED: 892-C4,
837-C1", findings list 7 of 9 with neither named).

The isolation of the one side-effecting gate is demonstrated separately by
`test_903_worktree_isolation_demo.py` (~35s) — see 837-C1.

## Classification table

| id | constant | advertised command/claim | classification | material? |
|----|----------|---------------------------|-----------------|-----------|
| 892-C1 | `disclosures.CALIBRATION_TABLE` | `python calibration.py --n-trials-calib 200 --n-trials-valid 200 --n-boot 400 --seed 31337 --calibration-margin 0.03 --block-counts 2 3 4 5 6 10 20 80` | **REPRODUCES** — bit-identical, every field, all 8 rows | n/a |
| 892-C3 | SPEC.md Amendment-1 body sha256 prefix `8408970d…` | `sha256sum experiments/892_action_ev/SPEC.md` | **REPRODUCES** — full digest `8408970d27f9668853e06d9991299b596689db6d04f3bdadcf4bee74e20dc358` | n/a |
| 837-C1 | `results_fixture_smoke.json` (fold/corridor/negative-test doc) | `python walkforward.py smoke` | **DRIFTS-with-named-cause** | NO |
| 892-C4 | the higher-precision recalibration `CALIBRATION_PRECISION_DISCLOSURE` calls "REQUIRED before the real leg launches" | `python calibration.py --n-trials-calib 2000 --n-trials-valid 2000 --n-boot 2000 --out runs/892_action_ev/calibration_real_leg.json` (manifest_892_p1_policy.txt:29) | **DRIFTS-with-named-cause** (the file exists; it was never wired into the constant the shipped reports read) | NO — verified by direct recomputation, see below |
| 892-C6 | `report.ZERO_WIDTH_FLOOR = 1e-4` + the MEASURED MARGIN claims in its comment (`report.py:214-251`) | in-comment reproduction claim, no CLI ("Measured directly (reproduced here and cross-checked against an independent probe…)"); existing executable coverage at `test_action_ev_892.py:1474` | **the constant and its load-bearing claim REPRODUCE; two stated margins DRIFT-with-named-cause** | NO |
| 892-C2 | `NULL_COVERAGE_TABLE` / `NULL_COVERAGE_TABLE_OVERLAPPING` | "block=21, n_boot=2000, 1000 trials/row, seed varied per trial" — no concrete seed given | **PROVENANCE-UNRECOVERABLE** | NO |
| 892-C5 | "independent cross-check across three dependence models" (k*~3.25/2.0/1.5) | none — no script, no seed, in-tree | **PROVENANCE-UNRECOVERABLE** | NO |
| 875-C1 | every per-fold/per-seed net@21 + teacher-floor number in LEDGER row `875/wf-generalization` | `runs/875_wf/grade_correct.py` over the raw per-seed JSONs (read-only; the `orch_wf875.sh` battery itself is 8h57m, LEDGER:97, and was NOT re-run) | **REPRODUCES** — all 5 folds' medians + floors + the 1-of-5-above-floor tally, exact, re-derived on gpu-server | None — the "no OOS edge" verdict rests on numbers that reproduce. Caveat: `WF_SUMMARY.json` ships as `[]`, a documented fake-null; the source is the raw JSONs + `grade_correct.py`. Superseded classification kept below per D39. |

## 892-C1 — CALIBRATION_TABLE: REPRODUCES

Re-ran the exact command disclosures.py:314-316 advertises. Output (excerpt):

```
blocks=   2 k=3.494949538093774 achieved=True cov_overlap=0.94  cov_iid=1.0
blocks=  80 k=1.37601365581144  achieved=True cov_overlap=0.96  cov_iid=0.985
```

Every field (`k`, `achieved`, both validation coverages, both validation `n`) at all 8 block
counts matches `disclosures.CALIBRATION_TABLE` exactly — no rounding, no tolerance needed.
Wall time ~1m48s (comment claims "1m35s" — machine variance, not a drift in the *constant*).
Checked by `gate_892c1_calibration_table_reproduces` in the test file.

## 892-C3 — SPEC.md body sha256: REPRODUCES

`sha256sum experiments/892_action_ev/SPEC.md` → `8408970d27f9668853e06d9991299b596689db6d04f3bdadcf4bee74e20dc358`,
which starts with the advertised `8408970d…`. Only one commit (`0f3a5aa`) has ever touched
`SPEC.md`, and it is HEAD's content — there is no ambiguity about which revision the hash
refers to.

## 837-C1 — results_fixture_smoke.json: DRIFTS-with-named-cause, NON-MATERIAL

Re-ran `python experiments/837_walkforward/walkforward.py smoke` (the harness's own advertised
self-test) against the tracked, committed `results_fixture_smoke.json` (generated 2026-07-08).
After normalizing the two known-volatile fields (the top-level `generated` timestamp, and a
second run-date embedded inside `smoke_aggregate.ledger_row`'s prose string), the documents
differ at exactly one further location:

```
- "[smokefold1] ASSERT-1 ... vs eval start 2025-11-01 (>= 1761951600); ... (vectorbt_env.py:584)"
+ "[smokefold1] ASSERT-1 ... vs eval start 2025-11-01 (>= 1761955200); ... (vectorbt_env.py:660)"
```

`1761955200` is the correct UTC epoch for 2025-11-01T00:00:00Z; `1761951600` is one hour
earlier (2025-10-31 23:00 UTC) — the checked-in artifact carries a now-known, now-fixed
timezone/DST bug. Confirmed by `git log -p` on `walkforward.py`: **TASK_869 (2026-08-03,
"UTC windows")** changed this exact computation and the line reference moved from
`vectorbt_env.py:584` to `:660` in the same commit family. The fixture-smoke artifact predates
that fix (2026-07-08) and was never regenerated after it landed — a stale-artifact drift, not a
live bug.

**Sensitivity**: NOT MATERIAL. The document's own `verdict` and `ledger_row` fields say so
explicitly — *"FIXTURE-SMOKE with STUB scorecards — plumbing proof only, no verdict"* — the
`net21`/`gross`/`pct_hold` numbers quoted in that row are hardcoded stub inputs
(`_stub_scorecard(..., net21=0.30, ...)`), not measurements, and the drift is confined to a
negative-test's own diagnostic message text. No LEDGER verdict rests on this artifact.
D39 disclosure: the stale, pre-869 value is recorded here rather than silently overwritten;
the tracked `results_fixture_smoke.json` was left byte-for-byte as committed.

**How the re-run is isolated (cycle-2 rebuild — the cycle-1 mechanism here was a BLOCKER).**
Cycle 1 ran the smoke in the live working tree and then "self-cleaned" by computing
`git diff --name-only -- experiments/837_walkforward` *after* the subprocess and passing
whatever came back to `git checkout --`. That claim — "confined to this directory… the exact 8
files touched" — was **true only from a clean start**. From a dirty start the sweep cannot
distinguish "dirtied by this run" from "was already dirty for an unrelated reason", so it
reverted *whatever it found*: a pre-existing uncommitted edit to `experiments/837_walkforward/
SPEC.md` — a file `walkforward.py` never writes — was silently destroyed at exit 0, and the
`finally` block fired the revert even on runs where the smoke subprocess itself failed rc=1 and
wrote nothing at all. There was no pre-run cleanliness check. That is the run-34 incident class
(a stray `git stash` stripping a concurrent lane) committed into a test every future run would
execute.

It is rebuilt so the question does not arise: the smoke re-run happens inside a **throwaway
detached `git worktree`** checked out at HEAD in a temp directory. Every file the smoke rewrites
is rewritten there; the live tree is never dirtied, so **nothing is reverted and there is no
restore step that can fail half-way**. No `git checkout`/`stash`/`reset`/`add` is issued
anywhere in `test_903_constant_provenance.py`; against the live repo it runs only
`worktree add` / `worktree remove --force` (administrative, and both name the throwaway path
explicitly) and read-only `status`/`log`. The gate **refuses fail-closed** — skipping loudly,
naming the offending paths, and now *registering* the skip so the run exits `2` and claims
nothing — if `experiments/837_walkforward/` is not clean before it starts, and it asserts the
directory is *still* clean afterwards. **On SIGKILL** (or SIGTERM, crash, or a smoke subprocess
that dies) the live working tree is byte-for-byte unchanged, because no code path writes
outside the throwaway directory; what can leak is the temp directory plus a `.git/worktrees/`
bookkeeping entry — harmless, and deliberately **not** reclaimed by this code (below).

**Two cycle-3 corrections to this isolation story.**

1. **`git worktree prune` is gone, and the claim that went with it is withdrawn.** The teardown
   used to fall back to `git worktree prune` if `worktree remove` failed, under the comment
   *"prune deletes admin records whose directory is already gone — it can never remove another
   lane's LIVE worktree."* **That is false**, and it was demonstrated in a throwaway probe
   repo: `prune` takes no path filter, and a live sibling worktree whose directory was
   momentarily absent had its record, index, HEAD and reflog deleted. This repo has exactly
   such a sibling — `/home/lukas/Trader/.git-training-worktree`, unlocked, carrying a live index
   and recent commits. The fallback bought nothing: `git worktree remove --force <path>` is
   path-scoped and already returns 0 when the directory has vanished, and a stale bookkeeping
   record is harmless. Both deliverables now run **no `prune` at all** and log a warning
   instead. (`git worktree lock` was *not* applied to the training worktree: mutating another
   lane's state is not this row's to do.)
2. **The throwaway directory is checked to be outside the repo.** `tempfile.mkdtemp()` honours
   `$TMPDIR`; with `TMPDIR` set somewhere inside the working tree, the "throwaway" worktree
   would materialise *inside* the tree this module promises not to write to, and a SIGKILL
   would leak a ~23 MB nested checkout that the 837-scoped post-condition cannot see.
   `_throwaway_dir` now resolves the path and refuses (→ a named SKIP, no run) if it falls
   under the repo root. Verified by pointing `TMPDIR` inside a disposable worktree: the gate
   skipped with *"resolved to …/tmpdir_inside_repo/… which is INSIDE the repo … Refusing"*, and
   left that directory empty.

**Known-drift normalization is single-locus (cycle-3 MINOR 5).** The TASK_869 substrings below
were previously normalized with a whole-document `str.replace`, which would silently rewrite a
*second*, never-inspected occurrence — in a detector whose whole job is to not do that. The
gate now asserts each known pair occurs exactly once in each document and **fails loudly,
without normalizing**, if it does not. Verified by substituting a substring that occurs 44×:
*"known-drift substring 'fold' is not single-locus (44x shipped / 44x fresh) — refusing a
blanket replace"*, recorded as a failure with no finding.

**Demonstrated, not asserted** — `test_903_worktree_isolation_demo.py`, four scenarios, each in
its own disposable worktree (no decoy is ever planted in the live tree). Scenarios A1/A2 drive
a **reconstruction** of the cycle-1 mechanism, not a verbatim copy: cycle 1's file was never
committed and was overwritten in place, so no artifact of it survives to diff against and no
verbatim claim is checkable. The reconstruction is vendored inline in the demo (ten lines) so a
reader can judge it; the three properties the comparison turns on — smoke runs in the caller's
tree, dirty set computed *after* the subprocess by a blanket `git diff --name-only`, result
passed to `git checkout --` from a `finally` — are all visible there:

| scenario | mechanism | result |
|---|---|---|
| A1 | reconstructed cycle-1, smoke succeeds (rc=0) | decoy edit to `837_walkforward/SPEC.md` **DESTROYED**, silently |
| A2 | reconstructed cycle-1, smoke fails (rc=2, wrote nothing) | decoy **DESTROYED anyway** — the `finally` fires on every exit path |
| B | rebuilt gate, 837 dirty | gate **SKIPS fail-closed** and **registers** the skip (`skipped=['837-C1']`, `findings=[]` — so the summary can make no claim); decoy **survives byte-identical**; no other file modified |
| C | rebuilt gate, 837 clean, unrelated edit elsewhere | gate **runs to completion** and records **exactly one** finding; host tree ends with **0 modifications** under `experiments/837_walkforward/`; the unrelated edit **survives byte-identical** |

No file under `experiments/897_corrected_809/` is read, listed, staged, or touched by either
file: the only paths they name are under `experiments/837_walkforward/`,
`experiments/892_action_ev/`, `runs/892_action_ev/`, and their own temp directories.

## 892-C4 — the required-but-unwired higher-precision recalibration

`disclosures.CALIBRATION_PRECISION_DISCLOSURE` (shipped verbatim in every 892 real-leg report,
including the ones that carry the ANTI-EDGE verdict) states: *"CALIBRATION_TABLE was measured
at CHEAP, IN-LOOP precision... A HIGHER-PRECISION RE-RUN is REQUIRED before the real leg
launches... that re-run is planner-scheduled on gpu-server, not implied by this table's
presence."*

`experiments/892_action_ev/manifest_892_p1_policy.txt:29` schedules exactly that re-run:
```
892cal cpu 16G trader_env/bin/python experiments/892_action_ev/calibration.py \
  --n-trials-calib 2000 --n-trials-valid 2000 --n-boot 2000 \
  --out runs/892_action_ev/calibration_real_leg.json
```
and `runs/892_action_ev/calibration_real_leg.json` **exists on this workstation** — the
higher-precision recalibration WAS computed, at the default 17 block counts, seed 0 (no
`--seed` given → the argparse default).

**Where this finding does and does not hold.** `runs/` is gitignored, so that input is absent
on a fresh clone, on CI, on gpu-server, and inside any `git worktree`. On such a checkout the
gate **skips, names itself, and the run exits `2`** — it does *not* assert the drift below.
(Before the cycle-3 MAJOR-2 fix the skip was a bare `return`: registered nowhere, counted
nowhere, while the closing banner asserted this gate's DRIFT anyway. Reproduced and then
refused — see the exit-code section at the top.) Everything below was established on the
workstation checkout, where the file is present and all 9 gates ran.

**But the shipped reports never read it.** `report.py:1082` embeds `disclosures.CALIBRATION_
TABLE` (the source-baked, cheap 200/200/400-trial, 8-row table) into every report's
`calibration_table` field — never `calibration_real_leg.json`. Verified directly: every
primary-subject binding-leg report (`report_s{0,1,2}_bind_t0.json`, the reports the D47
ANTI-EDGE verdict is read from) carries `calibration_source_n_blocks=80` and
`calibration_k=1.37601365581144` — the CHEAP table's k at block count 80 — while
`calibration_real_leg.json["80"]["k"] = 1.2772196308464874`, a ~7.2% relative difference, sits
unused in the same output directory. `CALIBRATION_PRECISION_DISCLOSURE` is carried verbatim,
unedited, into those same shipped reports, so the reports self-report a requirement they do not
in fact satisfy with the higher-precision number, even though the higher-precision number was
computed. (mtimes cannot establish whether the higher-precision run predates or postdates the
policy walks that produced the reports — both are gitignored `runs/` artifacts, most likely
bulk-copied from gpu-server together; this audit does not claim a launch-order violation, only
a wiring one.)

**Materiality — checked, not assumed.** The widening is
`ci_{lo,hi} = delta + k · (ci_{lo,hi}_raw − delta)`, so for a fixed `delta` and fixed raw
bounds it is **monotone in k**: shrinking k moves *both* bounds toward `delta`, and a bound that
was already on `delta`'s side of zero cannot cross zero when it moves toward `delta`. The
argument therefore applies to **every calibrated field in the report**, not only the top-level
`delta` — so all of them were recomputed, not just the headline. Every field with
`calibration_source_n_blocks = 80` shares the shipped k=1.37601365581144 (`delta`,
`maintenance_re_trim`); the `entry_exit` split leg sits at `calibration_source_n_blocks = 20`
(k=1.463321901624038 → real_leg 1.3784834511295248) and is checked at *its* k too:

| seed | field | k (shipped → real_leg) | delta | ci_hi shipped → real_leg | band unchanged? |
|------|-------|------------------------|-------|---------------------------|------|
| s0 | `delta` | 1.376014 → 1.277220 | −0.05031 | −0.03485 → −0.03596 | YES (ANTI-EDGE) |
| s1 | `delta` | 1.376014 → 1.277220 | −0.05591 | −0.04086 → −0.04194 | YES (ANTI-EDGE) |
| s2 | `delta` | 1.376014 → 1.277220 | −0.04284 | −0.02861 → −0.02963 | YES (ANTI-EDGE) |
| s0 | `maintenance_re_trim` | 1.376014 → 1.277220 | −0.05991 | −0.04201 → −0.04330 | YES (ANTI-EDGE) |
| s1 | `maintenance_re_trim` | 1.376014 → 1.277220 | −0.06186 | −0.04471 → −0.04594 | YES (ANTI-EDGE) |
| s2 | `maintenance_re_trim` | 1.376014 → 1.277220 | −0.04964 | −0.03356 → −0.03472 | YES (ANTI-EDGE) |
| s0 | `entry_exit` | 1.463322 → 1.378483 | −0.00416 | ci [−0.04702, +0.03982] → [−0.04454, +0.03727] | YES (straddles 0, non-significant) |
| s1 | `entry_exit` | 1.463322 → 1.378483 | −0.02234 | ci [−0.06865, +0.02013] → [−0.06597, +0.01767] | YES (straddles 0, non-significant) |
| s2 | `entry_exit` | 1.463322 → 1.378483 | −0.00460 | ci [−0.05269, +0.04200] → [−0.04990, +0.03930] | YES (straddles 0, non-significant) |

The narrower (higher-precision) k makes each CI slightly *tighter around its own point
estimate* (both bounds shrink toward delta), so no sign, no zero-crossing, and no band changes
anywhere — if anything the negative fields read marginally more decisively negative.
**NOT MATERIAL to the D47 ANTI-EDGE verdict, on any field.** This is
disclosed, not re-graded, per TASK_903's scope; it is reported to the orchestrator as a defect
to fix in TASK_892's own instrument (wire `calibration_real_leg.json` in, or regenerate
`disclosures.CALIBRATION_TABLE` from it) — that is instrument hardening, not a verdict change.

## 892-C6 — ZERO_WIDTH_FLOOR: the constant REPRODUCES, two of its stated margins DRIFT

**Why this entry exists at all (cycle-2 correction to gate (i)).** Cycle 1's `MANIFEST` claimed
to be *the* complete checkable enumeration of every shipped constant whose comment advertises a
generating command, seed, or reproduction claim — and missed this one. `report.py:214-251`'s
`ZERO_WIDTH_FLOOR = 1e-4` carries an explicit first-person reproduction claim
(*"Measured directly (reproduced here and cross-checked against an independent probe from the
factory orchestrator)…"*, and a *"MEASURED MARGIN:"* paragraph), which is exactly the shape
scope item 1 names. It was not a coverage gap — the lane already exercises the guard
executably at `test_action_ev_892.py:1474` (`gate_zero_width_ci_never_bands`, jitter 1e-9 and
1e-6 at n_blocks 33/35) — it was an *enumeration completeness* defect: gate (i) exists so that
"we asked" is itself checkable, and this constant had not been asked. It is now entry 892-C6,
with its own re-run in `gate_892c6_zero_width_floor_margins`.

**Asking it turned up a small, non-material drift.** Re-run with the module's own shipped
near-duplicate construction (`test_action_ev_892.py::_near_duplicate`, verbatim — the only
reconstruction the tree supports), block=21, n_boot=2000, bootstrap seed 0:

| claim in the comment | measured | verdict |
|---|---|---|
| `ZERO_WIDTH_FLOOR = 1e-4` (1bp, the anchor argument) | `1e-4` | **REPRODUCES** |
| "ci_width / jitter_std is CONSTANT to within Monte-Carlo noise" across 1e-9 / 1e-6 / 1e-4 | ratio = 1.058603 / 1.058603 / 1.058603 (spread 2.7e-09) | **REPRODUCES** — exactly, to ~7 s.f. |
| 1e-9 jitter → width "O(1e-9 to 1e-10)" | 1.06e-09 | REPRODUCES (at the O(1e-9) end) |
| 1e-6 jitter → width "O(1e-7) — three orders below" | **1.06e-06** (O(1e-6), two orders below) | **DRIFTS** |
| 1e-4 jitter → width "O(1e-5), below this floor, so it reads ZERO-WIDTH-CI here — deliberately" | **1.06e-04 > floor**, bands **EDGE**; and it is a coin flip in the *data* seed (4 of 8 seeds land above the floor, 4 below) | **DRIFTS** |
| 1e-9 and 1e-6 jitter at n_blocks 33 and 35 read ZERO-WIDTH-CI and are excluded from banding | all four cases do | **REPRODUCES** |

**Named cause.** The two failing margins are arithmetically incompatible with the comment's
*own* homogeneity claim, which reproduces exactly: with `ci_width/jitter_std` pinned at ≈1.06,
a 1e-6 jitter cannot emit O(1e-7) and a 1e-4 jitter cannot emit O(1e-5). The margins were
transcribed an order of magnitude too small; the code is self-consistent throughout. The 1e-4
parenthetical is additionally wrong in its *conclusion*, not just its number — that case is a
knife-edge at the floor, not a deliberate exclusion.

**Sensitivity: NOT MATERIAL.** No code path reads this prose. The floor's value, the anchor
argument it rests on (which needs only the homogeneity result, and that reproduces), and the
guard's behaviour at every scale the lane covers are all unaffected; no 892 band and nothing in
the D47 ANTI-EDGE reading depends on it. Recorded rather than silently corrected in the comment
(D39): the gate pins the measured numbers so the mis-statement cannot be quietly reworded
without this note being revisited. Reported to the orchestrator as instrument hardening for
892's own lane (correct the two margins in `report.py`'s comment, and add the 1e-4 knife-edge
case to `gate_zero_width_ci_never_bands`), **not** as a verdict change — TASK_903 measures
provenance.

## 892-C2 — NULL_COVERAGE_TABLE / _OVERLAPPING: PROVENANCE-UNRECOVERABLE, non-material

`disclosures.py:230-237`'s "HOW (reproducible)" note gives block=21, n_boot=2000, 1000
trials/row, and states plainly that "the data seed and the bootstrap seed both varied per
trial" — no starting seed, no seed-derivation rule, no script name. There is nothing to run:
even a faithful re-implementation of "vary a seed per trial" cannot reproduce a specific,
unstated seed sequence. `test_action_ev_892.py::_measure_null_coverage` (the module's own
regression tripwire) explicitly disclaims re-derivation ("Cheap settings — enough to separate
78% from 93%, not to re-derive the table") and checks with a wide tolerance (±3pp), which is
consistent with treating this table as approximate from the start.

**Sensitivity: NOT MATERIAL.** This table is not consulted by `calibrated_k_for` (that reads
only `disclosures.CALIBRATION_TABLE`, 892-C1/C4 above) — it is carried purely as descriptive
disclosure (`_coverage_table_text`) and as a loose regression tripwire. No band in any shipped
892 report is a function of `NULL_COVERAGE_TABLE`'s exact values.

**Attempted cost check**: re-measuring even a *loose* 1000-trial/row version of this table at
n_boot=2000 was not attempted at full scale — the sibling command (892-C4, n_boot=2000,
n_trials=2000, ONE block count) did not finish in >128s of wall time on this workstation before
being killed; 1000-trial rows across ~13 block counts × 2 scenarios would be of the same order
or worse. Recorded as **DEFERRED-EXPENSIVE** if an exact-seed version is ever specified and a
tight re-derivation is wanted; not run here regardless, because no command exists to run yet
(the seed is the missing piece, not the compute budget).

## 892-C5 — "independent cross-check across three dependence models": PROVENANCE-UNRECOVERABLE, non-material

`disclosures.py:324-326` cites a corroborating check (Gaussian rolling-sum / AR(1) φ=0.909 /
fat-tailed t3, "k*~3.25 at 2 blocks, ~2.0 at 3-5, ~1.5 at 30") with no script or seed anywhere
in the tree (`grep -rl "AR(1)\|phi=0.909\|fat-tailed" experiments/892_action_ev/` finds nothing
but `disclosures.py` itself). The figures are stated with `~` (approximate) throughout, which
reads as intentionally non-exact prose corroboration rather than a pinned constant — but the
comment does present it as an independently-measured check, so it is enumerated. **Sensitivity:
NOT MATERIAL** — never consulted by any banding code path, used only as narrative support in
the disclosure text.

## 875-C1 — the walk-forward real numbers: **REPRODUCES** (re-derived on gpu-server)

**Superseded classification, kept visible per D39.** This section previously read
"PROVENANCE-UNRECOVERABLE-FROM-WORKSTATION — materiality UNASSESSABLE, and that is itself the
finding." **That was true only of the workstation, and it was stated as though it were true of
the numbers.** The artifacts were never unreachable; they were simply on the other host. The
gate that asserted the absence FAILED the first time it ran on gpu-server — correctly, and the
correction below is the result. The lesson is recorded in the gate itself: a provenance gate
must classify what it can SEE, never assert a fixed verdict about where evidence lives.

**Re-derived 2026-08-22 on gpu-server** (`safe_run --gpu 3`, read-only, **no training and no
re-run of the 8h57m battery**):

```
cd ~/Trader && python3 runs/875_wf/grade_correct.py
```

Every number in the LEDGER row reproduces **exactly**:

| fold | test window | LEDGER median vs floor | re-grade | seeds |
|------|-------------|------------------------|----------|-------|
| f1 | 2023-07-01..2024-01-01 | +1.4957 vs +3.8987 | ✅ exact | +2.3236 / +0.3305 / +1.4957 |
| f2 | 2024-01-01..2024-07-01 | +0.3995 vs +1.4301 | ✅ exact | +1.3726 / +0.3995 / +0.3534 |
| f3 | 2024-07-01..2025-01-01 | +0.3108 vs +1.3952 | ✅ exact | +0.1575 / +0.3915 / +0.3108 |
| f4 | 2025-01-01..2025-07-01 | +0.4121 vs +0.2635 | ✅ exact, `above_floor=True` | +1.6120 / +0.4121 / +0.1931 |
| f5 | 2025-07-01..2026-01-01 | −2.9566 vs +0.4244 | ✅ exact | −3.6469 / −1.2491 / −2.9566 |

Also confirmed: `folds graded: 5/5 | positive: 4 | above own floor: 1` (the pre-registered
"none" band), all 15 runs at full step count (123/121/123/120/123), `bankrupt_at=-1` on every
seed, and f5's terminal PVs 59,860.21 / 83,720.83 / 67,055.84 — the 33% median drawdown the row
quotes. **The "no OOS edge" verdict rests on numbers that reproduce.** No drift, nothing to
escalate; D47 and the imprint-lane closure are untouched by this row.

**A real provenance hazard found while doing it — the canonical-looking artifact is empty.**
`runs/875_wf/WF_SUMMARY.json` ships as `[]`. It is a documented silent fake-null: per
`runs/875_wf/NOTE_summary_key_mismatch.md`, `orch_wf875.sh`'s summary block read
`tiers["21"]["sharpe"]` and `teacher.floor_inenv`, while the schema keys are `sharpe_full` and
`tiers["21"]["teacher_floor_inenv"]`, so every seed was skipped and the summary was written
empty mid-run. It was deliberately NOT fixed in place (bash reads a running script by byte
offset; editing it mid-run can corrupt the remainder), and `grade_correct.py` exists precisely
to grade around it. **Consequence for anyone auditing later: the file named like the summary is
not the source of the numbers.** The provenance chain is the raw per-seed
`f{1..5}_s{0,1,2}/results_wf875_*.json` plus `grade_correct.py` — and that chain is intact.

**Still true, and worth keeping:** none of `runs/875_wf/`, `grade_correct.py` or `orch_wf875.sh`
has ever been committed (`git log --all --oneline -- runs/875_wf '*orch_wf875*'` returns
nothing). The evidence for a load-bearing LEDGER verdict lives on one host, gitignored, on the
same disk class as everything else there — D48's cited-but-untracked class. Corroborated by
`docs/pathfinder_audit_2026-08-19.md:9,36`, which hit the same wall three days earlier. That is
a durability question for the planner, not a provenance defect: the numbers are right, they are
just not backed up anywhere git can see.

## Out-of-scope finding, for the planner: `walkforward.py smoke` overwrites the tracked real folds

Surfaced while rebuilding 837-C1's isolation. **Not fixed here** — `experiments/837_walkforward/`
is out of this row's edit scope. Reported so a DECISIONS row can be seeded. Everything below is
confirmed by direct inspection and by an observed run, not inferred.

**Exact call path.** `run_smoke` (`experiments/837_walkforward/walkforward.py:1080-1147`) opens
with `db = FIXTURE_DB` (line 1086) and, at step 5, calls:

```python
real_root = os.path.join(EXP_DIR, "folds", "real")
materialize(db, REAL_FOLDS, "real", real_root,
            n_assets=PIT_N_ASSETS, min_days=PIT_MIN_DAYS, jobs_db="trading_data.db")
```

i.e. `materialize(FIXTURE_DB, REAL_FOLDS, …)` — the **real** fold definitions, derived from the
**fixture** db. `materialize` (line 331) writes unconditionally: `os.makedirs(fdir,
exist_ok=True)` then `_write_json` per artifact. There is **no guard of any kind** — no
`--dry-run`, no out-root override for the real set, no clean-tree check, no prompt, no flag to
skip it. `--skip-real-corridor` (line 1223) skips only the *corridor* recomputation; the
universe/periods/manifest re-materialization always runs.

**Exact files rewritten** (observed, from a real smoke run in an isolated worktree — 8 tracked
files, all `git ls-files`-tracked):

```
experiments/837_walkforward/results_fixture_smoke.json          (the intended output)
experiments/837_walkforward/folds/real/fold_manifest.json
experiments/837_walkforward/folds/real/train_jobs.manifest
experiments/837_walkforward/folds/real/eval_jobs.manifest
experiments/837_walkforward/folds/real/fold1/corridor.json
experiments/837_walkforward/folds/real/fold1/universe_pit.json
experiments/837_walkforward/folds/real/fold2/corridor.json
experiments/837_walkforward/folds/real/fold2/universe_pit.json
```

**What the damage is *today*, stated precisely (do not overstate it).** On the current tree the
overwrite is **content-idempotent**: the diff after a smoke run is *only* the `generated`
timestamp in each file (`2026-07-08T22:23:0x+00:00` → the run time). The tracked `folds/real/*`
were themselves materialized from the fixture db on 2026-07-08, so re-materializing from the
fixture reproduces them byte-for-byte apart from that stamp. The four `period_*.json` per fold
are not rewritten at all in a way git notices.

**Why it is still a landmine.** `SPEC.md:60-72` pre-registers a *scheduled cross-check*: the
server re-materializes `folds/real/` against the real `trading_data.db` and the result must
match on the derivation fields. The moment anyone commits a real-db-derived `folds/real/`
(the natural outcome of that scheduled run, or of ever fixing a fixture/real divergence), **any
subsequent `walkforward.py smoke` from the repo root silently reverts those definitions to
fixture-derived content** — and these are the files that gate real training and eval launches
(`train_jobs.manifest` / `eval_jobs.manifest` reference `trading_data.db` as `jobs_db` while
the universes they pin would be fixture-derived). The failure is silent: smoke prints
`SMOKE PASS`, exits 0, and says nothing about having touched the real folds. It also means the
"regenerate the fixture-smoke artifact" operation cannot be performed without dirtying eight
tracked files, which is what pushed cycle 1 of this row into its destructive revert in the
first place.

**Cheapest plausible fixes** (planner's call, not implemented): give `run_smoke` an out-root for
the real-fold rails (e.g. a temp dir, or `folds/real_smokecheck/`) and diff instead of
overwrite; or gate step 5 behind an explicit `--materialize-real` flag; or make `materialize`
refuse to write into a directory whose existing content it did not generate.

## Deferred-expensive items (for planner scheduling)

| item | command | cost evidence |
|------|---------|----------------|
| 892-C4 full-precision recalibration re-run (independent verification, not just the wiring check above) | `python calibration.py --n-trials-calib 2000 --n-trials-valid 2000 --n-boot 2000 --out runs/892_action_ev/calibration_real_leg.json` (default 17 block counts) | measured: a single block count alone (`--block-counts 2`) did not finish in >128s wall/CPU time on this workstation before being killed; the full command runs 17 block counts → likely tens of minutes to a few hours, single-core CPU, no GPU/DB involved |
| 875 raw artifact retrieval | `rsync gpu-server:~/Trader/runs/875_wf/ <workstation>/runs/875_wf/` | cheap (no compute) but requires cross-host access unavailable in this session — scheduling item, not a cost item |
| 875 full walk-forward re-run | `orch_wf875.sh` (5 folds × 3 seeds) | measured 8h57m wall on gpu-server GPUs 1-3 (LEDGER:97) — explicitly the class of run the cost-discipline invariant reserves for scheduled, non-factory-loop execution |

## What this row does NOT do

No verdict changed. 892's ANTI-EDGE and 875's NO-OOS-EDGE readings stand exactly as the LEDGER
records them. No estimator or calibration METHOD was touched (TASK_901's lane). No file under
`experiments/897_corrected_809/` was read, opened, staged, or touched.

**No `git checkout`, `git stash`, `git reset`, `git add`, or `git worktree prune` was run
against the live working tree at any point, by either deliverable.** (`prune` is repo-wide and
can delete a live sibling worktree's record — see the two cycle-3 corrections under 837-C1.)
The one side-effecting command this audit needs
(`walkforward.py smoke`, which rewrites 8 tracked files under `experiments/837_walkforward/`)
runs inside a throwaway detached `git worktree`, so the live tree is never dirtied and nothing
needs restoring — see the 837-C1 section above for the full failure/SIGKILL argument and the
four-scenario demonstration. The cycle-1 revert-based mechanism, and the "confined to… the exact
8 files touched" claim that went with it, are **superseded and recorded here rather than quietly
deleted** (D39): they were true only from a clean start, and destructive from a dirty one.
