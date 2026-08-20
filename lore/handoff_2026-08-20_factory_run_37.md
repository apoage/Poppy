# Factory run 37 — 2026-08-20

Implementer lane. Resumed from run 36's two preserved halts. Queue taken: **892** (rank 1) and
**881I** (rank 5) in parallel — chosen because their file sets are strictly disjoint
(`experiments/` vs `tools/`), which is the only condition under which this repo's history permits
two lanes in one tree.

## Shipped

| version | item | verification outcome |
|---|---|---|
| `task881i-manifest-ingestion` (`bf1f8d5`) | **TASK_881I** — manifest ingestion, refuse AND accept paths | 3 cycles (implement=sonnet, fix=opus, fix2=opus). Medium panel + a fresh fail-closed lens on opus. Suite **162 passed / 1 skipped twice back-to-back** (86+1 → 114+1 → 162+1, +76 tests); with the 876 suite 182+1 twice; selftest 12/12. Diff a **pure insertion 1430 / 0**, 8 canon functions and `main()`'s post-guard body byte-identical to HEAD. **10 mutation-reverts** at real drop sites. Fence re-proven by contrast. Real corpus still **56 manifest files read, exit 0**. Every finding reproduced first-hand by the orchestrator with a control before briefing, and every fix re-verified first-hand after. |
| `task892-action-ev-instrument` (`ea0b354`) | **TASK_892** — the action-EV INSTRUMENT (not a verdict) | 2 cycles (build=sonnet, fix=opus), three-lens panel. Suite **130/130 PASS twice back-to-back** (25 gates; 21 → 93 → 120 → 130 across the re-cut), re-run twice independently by me; selftest 12/12; gate (i) default-OFF byte-identity re-proven in a real HEAD worktree (0 diffs); 10 mutation-reverts at real drop sites. Gates (vii)/(viii)/(ix) verified first-hand by me. |

**Both queue rows taken this run shipped.** Nothing is left dirty; the remaining work is decisions,
not code — see "Remainders → planner".

## Startup verification (measured, not taken from run 36's report)

- Both run-36 preservation patches **reverse-apply cleanly** ⇒ the tree was exactly the documented
  resume state.
- 892 acceptance suite: **21/21 gates PASS**, exit 0.
- `python -m trader selftest`: **12/12 PASS**.
- Only **D43** OPEN, and it gates 857 — not this queue.
- HEAD `eval_vector.py` sha256 `407e17ad4493b80a1848b61d73fc5559b6063d5e3b4b1808faa17595537ed84c`
  (pinned for gate-(i) A/B; reference worktree `/var/tmp/wt-orch892`).
- **Verified the run-36 preservation mechanism itself is faithful**: the part-2 inlined copies of
  the untracked 892 files are byte-identical to the working tree (after one trailing blank line my
  extraction regex added). Two rows now depend on that mechanism, so this was worth confirming.

## TASK_881I — cycle 1 → panel → cycle 2 → cycle 3 → SHIPPED

Cycle 1 (sonnet) delivered; the **medium panel** (correctness / regression / flow-verifier) plus my
own checks found:

- **BLOCKER** (correctness lens): `parse_spec_manifest()` silently TRUNCATED a `protected:` list at
  the first non-`- <path>` line. I reproduced it with a control: identical 2-item list, one mid-block
  `#` comment ⇒ **`1 protect`, "OK", exit 0**; comment removed ⇒ **`2 protect`**. A comment line
  silently cost a protection. The judge's argument was the good one: the function's own docstring
  already reasons that an EMPTY block is "indistinguishable from a truncated write, so it WITHDRAWS
  rather than quietly granting nothing" — and never applied that to a PARTIALLY consumed block.
- **MAJOR**, found independently by three parties (flow-verifier, regression judge, me):
  `ckpt_gc.py:1048` `scan_dirs = list(a.experiments_dir or [])` — `--experiments-dir` with zero
  values scanned **nothing** and reported `OK`/exit 0. My baseline control set the severity: on a
  tree with 2 real protections the default reports `2 protect`, the bare flag reports `0 protect`
  **and success**. A degenerate flag silently WITHDREW REAL COVERAGE while reporting OK — precisely
  what the tool's own dangling-symlink message says must never happen.
- **MINOR**: `--experiments-dir <regular file>` silently treated as absent
  (`_unresolvable_link_on_path()` only walks symlink components).
- **MINOR**: no scan count, so "read 56 manifests, none grant coverage" was indistinguishable from
  "read nothing".
- **MINOR** (regression lens): the task file's "86 (+1 skip) / 106 tests" was real but undecodable —
  87 alone; **106** when run with `tools/tests/test_ckpt_876_score_emission.py`. Reconciled exactly.

The regression lens cleared canon parity hard: all 8 canon functions AST-diffed byte-identical,
`main()`'s only delta the two `add_argument` calls plus a 3-line early return, and a full
HEAD-vs-candidate invocation matrix **including a real `--apply` pass** producing identical stdout and
identical resulting trees (including which snapshots were deleted).

**Cycle 2 (escalated to opus) closed all five.** Verified by me, independently:

| check | result |
|---|---|
| comment / blank-split / wrong-bullet in a `protected:` list | **REFUSED rc=2**, three DISTINCT messages |
| clean 2-item list | `2 protect`, rc=0 |
| **over-refusal controls**: list+blank+heading+prose; prose both sides | `2 protect`, rc=0 — the real SPEC.md shape still parses |
| bare `--experiments-dir` | rc=2 "expected at least one argument" |
| `--experiments-dir <file>` | rc=2 "is NOT A DIRECTORY" |
| `--experiments-dir no_such_dir` (**the deliberate control at test:559**) | rc=0 OK — **preserved** |
| real tree | `2 of 2 scan dir(s) visited, 1071 dir(s) walked, 56 manifest file(s) read` — matches my independent count of 56 |
| suite ×2 back-to-back | **114 passed, 1 skipped** both runs (86+1 → +28 tests) |
| `python -m trader selftest` | 12/12 |
| canon parity | **1088 insertions / 0 deletions** |

Its **7 mutation-reverts** were each at a real drop site with a green-and-wrong mutant — including
`nargs="+"`→`"*"` reproducing `0 of 0 scan dir(s) visited` / OK / exit 0, and `n_read += 1`→`+= 0`
reporting `0 manifest file(s) read` on the tree that read 56.

**Its design call on `#` is better than what I proposed.** It REFUSES `#` inside a manifest block
rather than treating it as a comment: `PROTECTED.txt` is plain text whose grammar defines `#` as a
comment, but `SPEC.md` is **Markdown**, where `#` is a HEADING — real content. Supporting it would
mean putting it in `spans`, and `spans` is the channel `strip_spans` deletes through, so an in-block
comment would buy a parser that can silently ERASE A HEADING — a fresh instance of the
strip-removes-more-than-the-parse-consumed defect this row was re-cut over.

### Why cycle 2 was not enough — two new defects I found in it

- **S1 (MAJOR).** `--experiments-dir ''` (empty string) silently scans nothing and reports OK — on a
  tree with 2 real protections it yields `0 protect` / OK / rc=0. Same class as the item-2 MAJOR, new
  vector: `nargs="+"` requires ≥1 value and `''` **is** one value. Mechanism pinned:
  `classify_entry('')` → `kind='unstatable'`, and `classify_entry('no_such')` → **also
  `'unstatable'`**, so `''` inherits exit 0 from the deliberate decision that an absent scan dir is a
  silent no-op. It is separable at zero risk, because the control test passes a NON-EMPTY name.
  **Repo precedent makes this concrete:** run 36 recorded that in TASK_861 "empty-string flags failed
  OPEN (one clobbered the tracked parity baseline)", and a caller writing `--experiments-dir "$DIR"`
  with `DIR` unset produces exactly `''`.
- **S2 (MINOR).** Nested scan dirs over-count. On a tree with **1** manifest file:
  `--experiments-dir . experiments` reports **`2 manifest file(s) read`**. The protect SET dedupes
  correctly, so no wrong protection decision follows — but item 4 existed to make that count
  trustworthy, so an overstated count is the same false-assurance class.

Cycle accounting: cycle 1 (BLOCKER+MAJORs) → cycle 2 (S1+S2) → cycle 3, still inside
`halt_after_failures: 3`. Shipping a known NEW instance of the exact silent-OK class this row exists to
close would have been the wrong call, so cycle 3 was spent rather than shipping dirty.

### Cycle 3 — a fresh fail-closed lens on opus, then SHIPPED

That lens cleared the cycle-2 fix itself as sound (block-level guards, blank-line straddle logic, shape
space, stats channel and the fence all held under a 35-case parse battery and a re-attack of the fence)
and then found **four silent-OK holes in the ACCEPT path, all PRE-EXISTING** — the layer D50 had
declared settled. That is the run's most transferable finding: **"settled after five lens-runs" was
scoped to the REFUSE path.** Nothing had ever exercised a manifest that PARSES, because 0 of 56 real
manifest files carries a block. The four:

1. **The item regex was UNANCHORED** (`^\s*-\s*(\S+)`), so `- runs/a runs/b` granted ONE protection at
   exit 0 while `strip_spans` erased the token the parse never consumed — falsifying the file's own
   docstring claims. **The identical shape in `PROTECTED.txt` REFUSED**, because that regex was
   `$`-anchored: the two parsers disagreed and the anchored one was right.
2. **Near-miss headers** (indented, UTF-8 BOM, UTF-16, `PROTECTED:`) produced no entries and no
   refusal — and item 4's new counts made the false green *more* convincing.
3. **`--roots` of the wrong shape** returned a full green on a tree whose score file was provably torn
   — the surviving half of the defect cycle 2 closed for `--experiments-dir`.
4. **`--experiments-dir` blank** (`''`, `=`, `' '`) scanned nothing and reported OK.

Cycle 3 closed all four plus the false symlink diagnosis, and I re-verified every one myself at the
final state (1430/0): two-paths-on-one-line refuses; all four near-miss headers refuse; `--roots`
file/fifo refuse with a new `score scan:` count line; all three blank spellings refuse with "BLANK
value"; a live symlink→file no longer claims "BROKEN SYMLINK" while a genuinely dangling link keeps
that message; both over-refusal controls still pass; the deliberate `no_such_dir` control still exits
0; and the real corpus still reads 56 manifest files at exit 0.

Two decisions in cycle 3 worth carrying forward, both of which I would not have thought of:
- **The blank-line straddle look-ahead deliberately kept the LOOSE recognizer.** Anchoring it too
  would have moved the subset hole one layer up, so `protected:\n- a\n\n- b c\n` would have granted
  `a` alone. Pinned by its own control test.
- **`--roots ''` is refused inside `_find_score_dirs`, not at argparse** — because `--roots` is a HEAD
  flag read by canon `main()`, and a parse-time rejection would have changed `--apply` behaviour for
  `--roots ''` (HEAD filters it out, "nothing to do", exit 0) and broken the byte-identity gate.
  `--experiments-dir`, added by this row and never read by canon, got the parse-time check instead.

## TASK_892 — SHIPPED as the INSTRUMENT (the D47 verdict remains the planner's)

**The cycle-3 BLOCKER is FIXED and I verified it myself.** Against the live tree, all near-degenerate
cases now read `ZERO-WIDTH-CI` and are excluded from banding where cycle 3 read **EDGE**:
bit-identical (width 0.0), +N(0,1e-9) (2.28e-10), +N(0,1e-6) (2.28e-07). That is gate (vii)'s hard
requirement met on measurement, not on report.

**The design they chose is sound**: `block_bootstrap_delta_ci` now ALWAYS widens by a measured factor
from `disclosures.calibrated_k_for`, built by `calibration.py`'s two-scenario Monte Carlo;
`REPORT_SCHEMA` → v3; a `calibration_capped` / `k_cap` guard whose stated rationale is "inflated until
it can't fail, not calibrated"; and `calibrated_k_for` takes the **max k among achieved entries at or
below n_blocks** because "the measured table is not perfectly monotonic under Monte Carlo noise" — the
fail-safe direction. `DEFAULT_BLOCK_COUNTS` spans exactly the strata Amendment 1 names.

**The calibration table was the decisive item, and it is now RESOLVED.** It is a literal pasted from a
`calibration.py` run; while empty, every stratum read LOW-POWER and the instrument could band nothing.
En route I caught that the lane had launched a **~10-hour** Monte Carlo *inside the factory loop*
(`--n-trials-calib 450 --n-trials-valid 450 --n-boot 1100`, 17 block counts × 2 scenarios,
single-threaded, ~2% done at 13 minutes) — which breaks both the factory's own "expensive validation is
SCHEDULED, never polled in-loop" rule and the operator's no-CPU-heavy-work-on-the-workstation guidance.
I told it to kill that and rebuild at in-loop precision over only the strata Amendment 1 names, which
it did.

**Final table: 8 entries, all `achieved=True`**, and every k landed **≥ mine**, exactly as predicted
from its calibrate-with-margin-then-validate-on-fresh-trials design:

| blocks | their k | my independent k* | their val cov (overlapping) |
|---|---|---|---|
| 2 | 3.495 | 3.25 | 0.940 |
| 3 | 2.402 | 2.0 | 0.960 |
| 4 | 2.276 | 2.0 | 0.975 |
| 5 | 2.064 | 1.75–2.0 | 0.970 |
| 6 | 2.055 | ~2.0 | 0.955 |
| 10 | 1.532 | 1.5 | 0.955 |
| 20 | 1.463 | — | 0.975 |
| 80 | 1.376 | 1.5 (at 30) | 0.960 |

**Instrument behaviour, my own measurements against the live tree:** healthy signal → **EDGE** at 20
and 80 blocks; the same real effect at **5 blocks → NULL** (the honest under-powered read I forecast in
writing before it existed); a **true null → NULL** with no false EDGE; and all near-degenerate cases
(bit-identical, 1e-9, 1e-6) excluded as ZERO-WIDTH-CI. **Gates (vii) and (viii) satisfied on
measurement, not on report.** So the OOS binding leg (5 blocks) will most likely return a wide NULL —
publishable per SPEC, not a failure.

**Suite 120/120 PASS twice** back-to-back (93 → 120: 4 new gates + 2 rewritten). Gate (i) re-proven by
the lane in a real HEAD worktree, 0 diffs — and it surfaced a reusable setup fact: **a fresh worktree
lacks the gitignored `runs/831_ext/` regime data and must have it symlinked**, or the A/B reports
phantom differences.

**Amendment-1 surface: complete** — `net_exposure`, forward H=21 `market_return`, the entry/exit vs
re-trim co-primary split, the OOS leg structurally binding via `--leg` + `validate_leg_window`, fee16
forced to `BAND_FEE_TIER_DIAGNOSTIC` in code from trace provenance (never caller-settable), 887 A/D
never-pooled enforcement, and `D47_EXCLUDED_CLASSES` excluding the 832 contrast.

**The zero-width rebuild deviates from D51's LETTER, with a sound argument — a planner call.** It is now
an ABSOLUTE floor, `(ci_hi - ci_lo) <= 1e-4` (1 basis point, the instrument's own smallest cost-relevant
unit since `cost = (fee_bps+slip_bps)/1e4`), not a tolerance relative to the data's own scale. The
justification: `ci_width` is **homogeneous of degree 1** in the input scale, so a test built from the
sample's own dispersion cannot separate a genuinely-tiny-real effect from an artificially-shrunk one.
That independently reproduces my own measurement (`ci_width/jitter ≈ 0.215`, constant across four
orders of magnitude) and draws the right inference from it. So it satisfies D51's spirit — a guard that
fires on the NEAR case, tested on the near case — while departing from its wording. **D51 is a planner
ruling, so this is flagged for ratification, not treated as either compliance or a defect.** My
verification found no over-exclusion.

**Honest gaps the lane disclosed unprompted** (all recorded, none hidden): the table is a restricted
cheap-precision pass (200+200 trials, n_boot=400, ~1m35s) per my own time-boxing, with block counts
8/12/15/25/30/40/50/65/100 unprobed and a higher-precision re-run **required before the real leg** and
stated in `CALIBRATION_PRECISION_DISCLOSURE`; an earlier exploratory run had found genuine borderline
failures (~0.90–0.93), which is why a calibration margin was added — **applied uniformly, not
per-stratum**; the "no achieved calibration" → LOW-POWER branch is unreachable via any in-range query
against the shipped table (proven to fire anyway through `calibrated_k_for`'s explicit `table=`
input, not monkeypatching); `capped`/`k_cap` is effectively dead code (banding unaffected, only the
diagnostic message loses precision); and **`report.py --leg` / `trace.py --subject-class` were not run
through a class-closing abbreviation sweep** — a live precedent in this repo, since adding
`--check-manifests` once silently made `--c` ambiguous.

### The three-lens panel (this row's verdict seeds D47, so it earned three)

**Correctness: NO FINDINGS**, and earned rather than asserted — it proved the co-primary split
partitions exactly (194 == 131+63, zero overlap, on a synthetic 60×6 trace), proved `market_return` is
genuinely undirected (a short cell with `forward_return=+0.0134` pairing with `market_return=-0.0134`),
showed `ZERO_WIDTH_FLOOR=1e-4` does not over-exclude at realistic scale (80 blocks, std=0.01 → width
1.8e-3, three orders above the floor), brute-forced the closed-form k against **50,000 triples with 0
mismatches**, and confirmed the calibrate/validate batches are disjoint sequential slices, not
circular. It diffed both rewritten gates against the cycle-3 baseline and found every
previously-passing assertion preserved, with the two that flipped being exactly what Amendment 1
mandates — and the new assertions checking the MECHANISM (`calibration_achieved`, `k > 1.0`,
widened > raw), not merely the outcome.

**Regression: no BLOCKER-level regression**, and it CLOSED the implementer's self-reported
abbreviation gap (prefix sets computed before/after against the real parsers — `--leg` and
`--subject-class` collide with nothing). It independently re-confirmed 120/120 twice in a clean
worktree, the v2→v3 schema refusal being loud, `market_return`/`net_exposure` as hard subscript reads,
`calibrated_k_for` conservative at every boundary and degenerate input, no circularity in the
calibration, `calibration.py` having no default write path, and the tracked-dir gate correctly
snapshotting after `calibration.py` exists.

**Fail-closed: the one substantive finding**, reproduced independently by the regression lens too and
confirmed by me. `report.py:812-813` reads `int(doc["provenance"].get("tier_bp", 0))` while `tier_bp`
is absent from `REQUIRED_TRACE_PROVENANCE` — so deleting one JSON key from a genuine fee16/slip5 trace
flips identical numbers (`delta 0.020790652301311996`) from mandatory `DIAGNOSTIC-ONLY` to a citable
**EDGE** with `instrument_fail: []`, defeating D49(a). **The contrast inside the same file is the
damning part:** `budget_conservation_violations` is read by hard subscript AND hard-required, because
its soft `.get(..., 0)` was the cycle-2 BLOCKER and `_load_trace`'s docstring explains at length why.
`tier_bp` is mechanically identical and is the sole gate on a D47-seeding guarantee.
**Severity reconciled to MAJOR, not BLOCKER**, because the regression lens established that
`trace.py:415` always writes the field unconditionally, so it cannot fire through the documented
producer — the *claim* ("enforced in CODE … regardless of caller discipline") is what is false.
Second finding: **`assert_never_pooled` is called from no live code path** (only the test suite), so
SPEC's "structurally impossible" pooling is today a library function someone must remember — the
convention-not-structure pattern SPEC rejects.

Both are small; one consolidated fix cycle was dispatched on opus. 892 is one short cycle from
shippable.

## My independent calibration measurements — reusable, and they settle gate (viii)

All measured through the REAL code path in an isolated byte-exact copy of the cycle-3 modules
(`report.block_bootstrap_delta_ci`, never a re-implementation), true Δ = 0 from two independent sides.

1. **The pre-registered method clears the 93% bar at NO stratum.** Overlapping-scenario coverage:
   0.767 at 5 blocks, 0.853–0.877 at 30 (two independent runs), against a plateau that maxes ~88.8%
   and is flat 30→40. Raising the floor cannot fix it — the block length EQUALS the dependence length.
2. **A calibrated-width correction clears it at EVERY stratum.** k*(≥95%): **1.5 at 30 blocks**
   (cov 0.974), **2.0 at 5** (0.970), **2.0 at 4**, **2.0 at 3**, **3.25 at 2** (0.960). Below 2
   blocks the shipped `MIN_BOOTSTRAP_BLOCKS` floor already refuses outright — gate (viii)'s flagged
   path, fail-safe.
3. **k* is INVARIANT across dependence models.** Gaussian rolling-sum, AR(1) φ=0.909, and fat-tailed
   t3 rolling-sum all give k*=2.0 at 5 blocks and k*=1.5 at 30, even though median raw interval width
   varies ~3.5× between them (3.35 → 11.63). So the correction is far more robust than the
   single-scenario caveat implied, and the honest residual shrinks to: "robust across the dependence
   models tested, which exclude regime shifts and cross-asset correlation".
4. **SPEC's premise verified at source**: `runs/861_ref_scorecard/s0/…json`, window `oos_full` →
   `n_decisions: 116`, 62/64 active slots, panel 177 rows. 116/21 = **5 full blocks** on the binding
   leg. Blocks run over DECISIONS (the overlap unit), not the ~6.4k cells — pooling cells would
   inflate n while cross-sectional correlation leaves the honest unit at the decision.

**Consequence: there is no configuration in which 892 must fudge a number or leave a stratum
bare-CI'd.** Expect a WIDE NULL on thin strata (a 3.25× inflation will span 0) — a publishable band
per SPEC, not a failure.

**Note for whoever checks the lane's harness:** their k should come out **equal to or LARGER** than
mine. They calibrate to a margin (`--calibration-margin` 0.03) and VALIDATE on fresh trials; I
calibrated and measured on the same trials, which biases k optimistically low. Larger is correct.
Only a materially SMALLER k, or an `achieved=True` whose validation coverage is below 0.93, is a
finding. Treat one-grid-step differences as noise.

## Corrections I made to my own work (all caught before reaching an implementer)

1. Called the absent-`--experiments-dir` exit 0 a "fail-open" — **wrong**. There is a deliberate,
   documented control (`test_absent_scan_root_is_not_a_refusal`) whose rationale is "a host with no
   `experiments/` has nothing to say". The distinction it draws is principled: a dangling symlink
   actively WITHDRAWS coverage; an absent path never claimed any.
2. Proposed "refuse a non-default nonexistent `--experiments-dir`" — **withdrew it** on reading the
   control properly: it passes an EXPLICIT `no_such_dir` and still requires exit 0, so my remedy would
   have inverted a tested decision. Re-filed as a planner question (below).
3. Claimed TASK_896's specified `.gitignore` carve doesn't work — **retracted in full**. My sandbox
   omitted `.gitignore:53 !runs/*/`, the very line whose comment explains that trap. Re-tested with
   the real rule sequence: the specified one-liner **works**. D48(a) and the QUEUE are correct.
   *Lesson: I built a test that did not reproduce the system under test and drew a spec-defect
   conclusion from it — the same error class this run was auditing in others.*

4. **Praised `calibrated_k_for`'s "max k among achieved entries ≤ n_blocks" as "a correct fail-safe
   choice" — it was a BUG**, and the 892 lane found it. Because k decreases monotonically with block
   count, max-k always returns the SMALLEST probed count's k: `calibrated_k_for(80)` was handing back
   the k measured at 2 blocks (3.495), throwing away all power at every stratum. Fixed to nearest-below
   (the codebase's existing `measured_null_coverage` convention), still conservative in the right
   direction since a smaller block count needs a larger k. My own measurement confirms the fix is live
   (k=1.463 at 20 blocks, 1.376 at 80 — not 3.495 everywhere). *Lesson: I endorsed a mechanism from its
   stated rationale without checking it against the table's actual monotonicity.*

Also corrected: I described `nargs="*"` as letting `--experiments-dir` "swallow" a following
`--apply`. It does not — argparse stops at a defined option token and `apply=True` is set correctly.
The only defect was the empty list.

## Post-ship state

Both rows are committed and tagged; the working tree is clean. Nothing needs resuming.

- `bf1f8d5` / `task881i-manifest-ingestion` — `tools/ckpt_gc.py` (+1430/0),
  `tools/tests/test_ckpt_881i_manifest_ingestion.py`, its task banner, CHANGELOG.
- `ea0b354` / `task892-action-ev-instrument` — the six `experiments/892_action_ev/` modules,
  `experiments/835_stage4_eval/eval_vector.py` (the null operator, both flags OFF by default and
  byte-identical when off), its task banner, CHANGELOG.

**Both agents were interrupted repeatedly by transient API errors** — the 892 lane three times
(`401 OAuth access token has expired`, `Request timed out`, `The response stopped arriving`) and the
881I fix once. **The first masqueraded as an auth failure and I wrongly told the operator to
`/login`;** they corrected me. Every one was resumed with context intact via SendMessage and lost no
work. Recorded as a durable lesson in
`~/.claude/.../memory/feedback_transient_api_error_masquerades_as_auth.md`: check whether your own
tool calls still work, assess the tree, then RESUME rather than re-spawn.

## Remainders → planner

- **D-question (new)**: should an explicitly-passed NONEXISTENT `--experiments-dir` refuse, given that
  an explicitly-passed nonexistent `--protect-file` DOES refuse (exit 2, "could not be read")? The two
  are the same operator error with opposite outcomes. Resolving it means changing a deliberate tested
  decision, so the factory must not take it.
- **881I residual, self-reported and accepted**: a top-level markdown bullet immediately after a
  manifest block's blank line now refuses (loud exit 2). Harmless today (0 of 56 files carry a block)
  but it will bite when 881D authors the corpus.
- **881I residual, pre-existing**: `_SPEC_ITEM_RE` treats a markdown horizontal rule `---` as an item
  with path `--` → dangling → loud exit 2. Fail-closed and safe, but a WRONG DIAGNOSIS.
- **881D precondition is worse than "unmet"**: I confirmed **0 of 56** manifest-basename files carry a
  `protected:`/`covers:` line, AND a manifest `PROTECTED.txt` refuses both plausible syntaxes (bare
  paths and `- path`) — so **the grammar it DOES accept is undocumented anywhere an author would
  look.** Authoring the corpus needs that written down first.
- **Repo health (new, XS)**: `tools/tests/` carries **3 standing RED tests** unrelated to any queue row
  (`test_jax_env.py::TestLimitOrders` ×2, `test_physics_core.py::test_physics_signals`, the last
  failing on `NameError: name 'flax' is not defined` — a missing import in the test file). Confirmed
  identical at HEAD and on the candidate. A suite with standing reds trains everyone to ignore reds.
- **TASK_897 placement conflict**: its header says "Runs on gpu-server", but its gates need the pinned
  MINUTE table, which lives in the 24GB `trading_data.db` that infra rules say must never cross the
  ~5 Mbps uplink — while operator guidance says no CPU-heavy work on the workstation. Three-way bind;
  needs an operator ruling, not a factory decision.
- **TASK_898** is the same shape: eval-only and genuinely light CPU, but its verdict rewrites
  `docs/pathway.md:106` and its own header schedules the real leg to gpu-server.
- **TASK_894** preconditions VERIFIED and ready (the 7 load-bearing checkpoints resolve exactly, all
  with `param_fingerprint` present and `frame_symbols_sha256` ABSENT — the byte-identity baseline gate
  (ii) needs). **Spec correction, binding**: 894 names the record `trainer_metadata_<step>.json`; the
  real filename is **`trainer_metadata.json`** (97 exist; the only `_<step>` instance anywhere is one
  `trainer_metadata_0.json` in a smoke dir). Same error class as run 36's
  `ckpt_gc`-vs-`ckpt_archive`.

## Housekeeping

- Removed run 36's stale worktrees `/var/tmp/judge865/{BASE,CAND}` (pinned 086b1c4 / aef1d98) and
  pruned — done only after the 881I panel finished, so no live agent could be using that path.
- `/var/tmp/wt-orch892` is the orchestrator's gate-(i) HEAD reference; remove at wrap-up.
- **Invariants held all run**, re-checked with four agents live: every write inside its lane fence;
  `fixtures/trading_fixture.db` byte-unchanged (79310848, 2026-08-05 07:12:59);
  `trading_data.db` untouched (26245337088, 2026-07-26 20:47:32).
- An environmental confounder worth knowing: a `backintime backup-job` ran during the panel, adding
  disk I/O. The repo has a known load-sensitive watchdog flake, so any timing-shaped failure this run
  should be re-run before being read as a regression. (None fired.)
