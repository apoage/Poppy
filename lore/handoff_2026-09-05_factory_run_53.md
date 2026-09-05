# Factory run 53 — TASK_914 streaming reader: HALTED at the cycle limit, reader verified clean

Date: 2026-09-05. Base HEAD `5010a1b`. Nothing pushed.

## Startup: the tree was dirty and the wall was the operator's to clear

Run 52 left its HALTED 907-B2 round-4 work in the live tree. Row 2 (TASK_914) rewrites
`read_capture`, one of the dirty files, and 914's central gate is "byte-identical to the CURRENT
reader" — undefined on a base carrying four open BLOCKERs. `.claude/rules/00` §1 forbids the
orchestrator from clearing it, so it went to the operator as a blocking question.

Verified BEFORE asking, so the question was decidable rather than speculative:
- regenerated the live 907 diff and compared byte-for-byte against
  `lore/handoff_2026-09-05_factory_run_52_907B2_round4.patch` — content identical, differing only
  in the patch's internal file ORDER (an artifact of `git add -N` on the untracked file);
  confirmed by `LC_ALL=C sort | sha256sum` matching on both sides.
- confirmed `5010a1b` had made that patch a TRACKED file in git (it landed after this session's
  startup snapshot, so the initial `git status` was stale).
- `git apply --check` of the patch against HEAD in a detached worktree: OK. Recovery proven.
OPERATOR RULED: revert and take 914. Reverted; fixture `b0f8de8b…` unchanged; 10 halt stashes intact.

## What the row is

`read_capture` held the whole capture in memory. The spec (37 lines) named ONE materialization.
There were **three**, and two were alive simultaneously — which is what actually set the peak:
1. `_iter_lines_tolerant` (`capture.py:504`) — a generator IN SHAPE ONLY: `fh.readlines()` inside.
2. `capture.py:1317` `lines = list(_iter_lines_tolerant(path))` — every line again as a parsed dict.
3. `_file_has_truncated_tail` (`capture.py:1727`) — a THIRD `readlines()`, called at `:1318`
   while `lines` was still live.

## Two orchestrator errors, both caught by the panel, both recorded

**(1) A baseline carried in from the wrong tree.** The brief told the implementer "HEAD is 362
test nodes, per the run-52 record". 362 was measured on the ROUND-4 tree; that work had been
reverted at the start of this very run, so HEAD is **302** — which the task spec's own acceptance
(iii) also says, and which `CHANGELOG.md`'s B1 entry ("Suite 271 → 302") independently
corroborates. The implementer caught it by collecting on HEAD itself instead of trusting either
number. Same class as D81(c): a quantity stated against the wrong base.

**(2) Single-point evidence presented next to a scaling claim.** The brief set the gate at peak
RSS < 4 GB on ONE 1.44 GB file, while the RISK the task exists to retire is denominated in
FILE-SIZE SCALING ("a 7-day file implies ~24 GB"). The regression lens reported this against the
orchestrator per D81(c); the flow-verifier then MEASURED it (below). The brief should have called
for a multi-size sweep on a realistic mix. This is the D81(c) lens working as designed on its
first run after being ruled.

## The scaling answer — LINEAR, not flat (flow-verifier, 5 points, real capture)

| lines | file bytes | BEFORE RSS | AFTER RSS | AFTER RSS/file-byte |
|---|---|---|---|---|
| 200,000 | 24,422,767 | 223,850,496 | 39,215,104 | 1.61 |
| 1,000,000 | 122,414,646 | 1,062,141,952 | 137,748,480 | 1.13 |
| 3,000,000 | 368,611,010 | 3,169,742,848 | 375,660,544 | 1.02 |
| 6,000,000 | 731,832,595 | 6,280,445,952 | 721,350,656 | 0.99 |
| 11,764,364 | 1,440,479,339 | 12,337,229,824 | 1,457,827,840 | 1.01 |

Ratio dead flat at ~1.0 from 3M lines on = genuine per-event linear growth, not a memory floor.
Only the SLOPE moved (~8.5x). **The <4 GB gate is one point on a line that recrosses it** at a
~3.7-5.6 GB capture (~9-14 days at this rate vs the closed capture's 3.586 days). The honest and
still-valuable claim: the safe_run-cap boundary moves from ~2.7 GB out to ~30 GB. "Risk retired
for good" would be a FALSE reading and must not appear in the banner.

## The BEFORE number did not reproduce

implementer (1 run) **15,190,589,440 B** vs flow-verifier (2 runs, <0.01% jitter)
**12,337,229,824 B** — a ~19% spread, cause unknown. AFTER reproduced to 0.17%
(1,460,252,672 vs 1,457,827,840). Reduction is ~8.5x or ~10.4x depending on which BEFORE.
Cycle 2 was told to re-measure and to leave all figures on the record rather than quietly
resolving the spread in the flattering direction.

## Panel (medium tier; task self-declared LOW, orchestrator OVERRODE to medium)

Override rationale: declared risk describes the CHANGE's nature (pure refactor), not its BLAST
RADIUS — `read_capture` has ~110 call sites across 8 test modules and is the entry point every
other 907 row depends on.

| lens | model | verdict |
|---|---|---|
| correctness | sonnet | NO FINDINGS — 16 resume-path fixtures, 11 read_capture fixtures, 400 fuzz trials, sentinel-safety check |
| regression | sonnet | 1 MAJOR + 1 finding against the orchestrator's brief; no BLOCKER |
| flow-verifier | sonnet | green on flows exercised; scaling caveat; mandate limits stated explicitly |

Two panel behaviours worth preserving as precedent:
- The regression lens **caught its own first fixture being vacuous** — its in-gap timestamps were
  monotonic in loop index, so file order == sorted order and the sort-at-insertion poison produced
  an identical result. It rebuilt with genuinely out-of-file-order data within one channel, and
  only then did all three poisons go RED. Recorded because it is easy to get wrong: a non-vacuous
  T3 corpus needs NON-MONOTONIC-IN-FILE-ORDER data inside a single (pair, channel), not merely
  "a gap with several events in it".
- The flow-verifier **refused to inherit** the implementer's scoping argument on G2 and ran the
  full CLI end-to-end on the actual 1.44 GB capture: byte-identical 27,221 B artifacts (modulo the
  `Generated:` wall-clock line). It also disproved the implementer's "JAX unavailable" condition —
  selftest is **12/12 PASS, none skipped**; the SKIPs were an artifact of the minimal copy.

## The MAJOR (cycle 2's scope)

`test_streaming_reader.py:57-64` builds 1,000,000 HEARTBEATS and zero book/trade events.
Heartbeats populate only `all_ts` (`capture.py:1569`); they never reach `data_ts`, `all_data_ts`
(`:1627`), `_pair_coverage`, `_cadence_stats` or the in-gap bisect — the terms the brief itself
called dominant. Measured: heartbeat path **46.8 B/event**, book path **188.2 B/event** (~4x).
So the only committed guard for the scaling claim cannot reach the code carrying most of the
memory, and would stay green if an O(file) defect were reintroduced there.

## Remainder found, deliberately NOT taken (scope discipline)

`health_probe.py:125` defines its OWN `_file_has_truncated_tail` and `:313` runs it against a copy
of the capture — the same whole-file `readlines()` materialization 914 just removed from the
reader, in the probe TASK_912 shipped. Verified independent (health_probe does not import
capture.py's; the deleted function has zero surviving callers). Out of 914's scope. Candidate XS row.

## Open for the operator
- D81(a)/(b) still OPEN — 907 B2 lane remains closed to patching; no fifth round attempted.
- TASK_916 carries a DENOMINATION ERROR in its own "Why" — see
  `scratchpad/finding_916_denomination.md`. Not fixed in-place: `tasks/` is a shared surface the
  planner owns while ARMED (rules/00 §4).

---------------------------------------------------------------------------------------------
## Cycle 2 (opus) — closed the MAJOR, and EXPLAINED the RSS discrepancy rather than picking a side

Non-vacuity was **demonstrated, not asserted**: two poisons planted in scratch copies of
`capture.py` (a duplicated PARSED full-file pass; a duplicated RAW-LINE pass, both gated on
`data_ts` non-empty) drove the new book/trade test RED (1163.17 and 260.43 B/event vs a 160 bound)
while the heartbeat test stayed GREEN — the cycle-1 finding reproduced as a controlled experiment.

**The 19% BEFORE spread was the INTERPRETER, and both prior measurements were right:**

| interpreter | pre-914 | post-914 | reduction |
|---|---|---|---|
| CPython 3.10.14 (`trader_env`) | 15,190,888,448 | 1,458,266,112 | 10.42x |
| CPython 3.14.4 | 12,342,190,080 | 1,462,108,160 | 8.44x |

3.11+ shrank exactly the `str`/`dict` materialization this task deletes, while the surviving float
lists cost the same — which is also why AFTER agreed all along (0.17%). The implementer's
15,190,589,440 reproduces to 0.002%; the flow-verifier's 12,337,229,824 to 0.04%. **Quote 8.44x**
(conservative). A spread that looked like sloppiness was a real, explicable effect.

## Cycle 2 also found a SECOND latent vacuity — and got the fix wrong (cycle 3's BLOCKER)

It correctly observed that `ru_maxrss` is a process-wide high-water mark, so cycle 1's in-process
before/after silently subtracted an earlier test's peak (a poison cost 304 B/event alone, 212
behind a hungry neighbour). It moved measurement into a fresh CHILD PROCESS and asserted
(`test_streaming_reader.py:176-177`) that "a child process starts from its own floor, so the
number is absolute and independent of test order".

**That is false on Linux, and the regression lens refuted it from first principles.**
`ru_maxrss` is `signal->maxrss`, NOT reset by `execve`; `/proc/self/status` `VmHWM` is
`mm->hiwater_rss`, which IS fresh after exec. Orchestrator-verified first-hand:

    parent ru_maxrss after allocating 400 MB = 421,564 KB
    freshly exec'd child, having allocated NOTHING:
        child ru_maxrss = 421,564 KB   <-- parent's peak, verbatim
        child VmHWM     =   9,928 KB   <-- the true figure

Materiality on the REAL test file: one 300 MB test earlier in the same pytest session makes a
POISONED `capture.py` **PASS at 0.00 B/event**. The guard SATURATES AT ZERO — an unconditional
mask of a defect of any severity, strictly worse than the cycle-1 flaw it replaced (which merely
deflated). A bare `import numpy` in the parent (~41 MB) already pulls a clean ~96 B/event to 67.4.
Not exposed today only because nothing in the 907 directory imports numpy/pandas/JAX — incidental
and undocumented.

**The evidence was already in cycle 2's own docstring and was misdiagnosed**: it recorded 260.43
"under pytest" vs 275.99-281.32 bare-interpreter for the same defect and attributed the gap to
"a slightly different allocator state" (`:300-301,307-308`). Same mechanism, seen small.

Lesson worth carrying: **a measurement instrument needs its own adversarial pass.** Two successive
cycles built guards whose INSTRUMENT was the defect, and both passed their author's own review.
This is the `.claude/rules/30-agent-conduct.md` "a refutation's replacement needs its own
adversarial pass" rule, applied to measurement rather than to diagnosis.

## The regression lens corrected ITSELF, on the record

It withdrew its own cycle-1 figures (46.8 vs 188.2 B/event) as "directionally indicative only" —
a different fixture shape, never the same quantity as cycle 2's 92/5/3 three-pair mix — while
noting that order-contamination alone cannot explain the direction (contamination deflates; 188.2
is higher than the clean ~96). The qualitative claim it was hired to make (the book/trade path
costs several times more per event; the heartbeat-only guard was structurally blind to it) stands
independent of either side's absolute numbers. Cycle 3 was told NOT to calibrate against them.

## A THIRD orchestrator provenance error (recorded — the run's clearest pattern)

The cycle-2 brief said the closed capture spans "3.586 days". Wrong twice over: the capture's
wall-clock span is **4.1198 days** (`end_ts - start_ts`); 3.6932 days is that span MINUS its
36,860.652 s of recorded gap time — the right denominator for COVERAGE, the wrong one for "how
long until a file gets this big". And 3.586 was itself lifted from `CHANGELOG.md`'s B1 entry,
which describes the **1.248 GB SNAPSHOT capture, not the 1.44 GB FINAL one**. Corrected on the
wall-clock rate (3.4965e8 B/day): the 4 GiB gate recurs at **~12.1 days**, the 32 GiB safe_run cap
at **~97 days** post-914 vs **~9.3 days** pre-914.

**The pattern across all three orchestrator errors this run is identical and is NOT quite D81(c):**
362 nodes (from the round-4 tree), single-point evidence for a scaling claim, 3.586 days (from a
different capture file). Each is a number carried across from a different base — tree, file, or
basis — without re-derivation. D81(c) governs DENOMINATIONS (the quantity a gate consumes);
these are PROVENANCE failures (which artifact a number was measured on). Adjacent, not the same.
Candidate refinement for the operator: **a brief must state, for every number it asserts, the
artifact and revision it was measured on — and a number whose provenance is not stated may not
be used as a gate baseline.** All three would have been caught by that rule; none by D81(c) alone.

## Config gap found (planner lane — NOT edited while ARMED)

`.dark-factory/config.json` `protected` lists `~/trader-live/`, which **does not exist on this
host** (verified), while `runs/907_live_fill_probe/` — holding the irreplaceable 1.44 GB closed
capture — is **not protected**. Worth correcting.

---------------------------------------------------------------------------------------------
## Cycle 3 (opus) — closed the BLOCKER, shipped BOTH remedies, and re-derived the bound

R1: instrument replaced with `/proc/self/status` `VmHWM` read in the child. R2: judged workable
WITHOUT touching `capture.py` (the reader opens through `builtins.open` at `_iter_lines_tolerant:537`),
so the child wraps `builtins.open` for the duration of the call and asserts the capture is opened
exactly once and `chars_read == file_chars`.

**It re-derived the bound and found cycle 2's margin claim was itself wrong.** Honest cost under the
correct instrument is **120.69-121.21 B/event**, not ~96 — so the old 160 bound sat only **1.32x**
above honest, not the 1.67x cycle 2 claimed. The contaminated instrument had been flattering the
margin too. New bound 190.

Its `P_RETAIN` poison is the experiment that proves both guards load-bearing: a retained parsed
list built from a SINGLE pass gives `traversals=1.000` (structural guard passes) but 1177.99
B/event (RSS guard fires). Neither guard alone is sufficient; the file documents the split.

A1 reproduced: poisoned `capture.py` + a 300 MB/`import numpy` test collected FIRST in the same
session -> **RED** (306.18 B/event vs bound 190, traversals 2.000), where cycle 2's instrument
read **0.00 and passed** the identical configuration.

## The fail-closed lens (opus, FRESH — deliberately not the lens that proposed the remedy)

The regression lens PROPOSED `VmHWM`, so it could not bless cycle 3 — that is self-review one
level up, and `.claude/rules/30-agent-conduct.md` requires a refutation's REPLACEMENT get its own
adversarial pass. A fresh fail-closed lens ran it instead.

Verdict: **SHIPPABLE, no BLOCKER** — cycle 3's central claim survived every attack (VmHWM
order-independent: 258.65 vs 257.74 B/event with and without a 300 MB + numpy parent hog; both
guards load-bearing, independently reproduced). **But 2 MAJOR + 3 MINOR + 1 NIT**, and both MAJORs
are places where the file makes an explicit fail-closed claim that MEASUREMENT REFUTES:

- **MAJOR-1** (`:162`): `MAX_CHILD_BASELINE_BYTES = 64 MiB` is ~5x above the point where baseline
  deflation eats the bound. `incremental = VmHWM(after) - VmHWM(baseline)`, so a baseline of B
  deflates the figure by `(B - 12.8 MB)`; at the ceiling that is **108.6 B/event = 158% of the
  entire headroom** between honest cost and bound. Measured end-to-end: a duplicated raw-line
  full-file pass reports **166.35 / 158.62 / 154.47 B/event and PASSES** at child baselines of
  58.2 / 62.3 / 64.4 MB. The "loud refusal" is silent by design across that whole band.
- **MAJOR-2** (`:332-342`): the structural guard patches only `builtins.open`. **`io.open` IS
  `builtins.open`** (same object, two names) and `Path.open` holds a direct reference to it, so a
  bypassing second traversal leaves `n_opens == 1` and the test **affirmatively certifies
  "traversals=1.000" for a genuinely double-traversing reader**. Measured: `rig_stream_io` (second
  full traversal via `io.open`) -> 120.95 B/event, n_opens=1, traversals=1.000, **`2 passed`**.
  Orchestrator-verified first-hand: rebinding `builtins.open` intercepts **0** `io.open` calls.

MINORs worth carrying: the cheapest reintroducible full-file materialization is **228 B/event**
(contiguous `bytes` blob / `mmap`), not the 304.62 the file derives — real margin 1.20x, not 1.60x.
The heartbeat test's RSS half is vacuous against any single duplicated pass (its fixture is 42.0
file-bytes/event, 2.4x below the >100 threshold its own sibling test states as the vacuity
criterion). And `skipif` deletes the DETERMINISTIC half too, though that half needs no `/proc` —
a skip reads green in CI.

## Why the row HALTED rather than taking a cycle 4

Cycle budget spent 3/3: MAJOR -> BLOCKER -> 2 MAJOR. `green: true` requires every BLOCKER/MAJOR
fixed and re-verified, so the verdict is **green: false** and the commit gate correctly refuses.

**D76's closing-pass exception was considered and does NOT apply.** It permits exceeding the limit
only for a pass that "only transcribes or reassembles evidence a judge has already collected — no
new solution attempt". Closing MAJOR-2 requires a NEW MECHANISM (intercept at the `os.open`/fd
layer, or compare the fd's own `lseek` position); that is a new solution attempt, and D76 states
the exception "is never a fourth try". Halting is the pre-committed behaviour and it was not
drifted past.

**The distinction that matters for the ruling:** unlike runs 50/51/52, this halt is NOT a verdict
on the production code. `capture.py` is verified clean by three independent lenses and is
byte-identical to what cycle 1 shipped. Both MAJORs live in `test_streaming_reader.py`, a file that
did not exist before this run. Seeded as **D82** with three scope options; (b) — ship the reader,
correct the guard's overclaiming prose, row the two mechanism fixes — is the recommendation, on
the grounds that correcting a false claim IS transcription of collected evidence while changing a
mechanism is not.

## The run's transferable lesson: a MEASUREMENT INSTRUMENT needs its own adversarial pass

Two consecutive cycles shipped a guard whose INSTRUMENT was the defect, and both passed their
author's own review. Cycle 1 measured in-process (`ru_maxrss` is a process high-water mark, so an
earlier test's peak was silently subtracted). Cycle 2 moved to a child process and asserted the
number was "absolute and independent of test order" — false, because `ru_maxrss` is
`signal->maxrss` and survives `execve`, so the guard SATURATED AT ZERO under a memory-hungry
predecessor. Cycle 3's replacement instrument is sound, but its two FAIL-CLOSED CLAIMS were also
refuted by measurement.

This is `.claude/rules/30-agent-conduct.md`'s "a refutation's replacement needs its own adversarial
pass", applied to MEASUREMENT rather than to diagnosis. Each cycle was right about the negative
half and wrong about the positive half, exactly as the rule predicts. Recommend generalising the
rule's wording to name instruments explicitly.

Corollary that paid off twice: never let the lens that PROPOSED a remedy be the lens that blesses it.

## Panel scorecard

| lens | model | cycle | verdict |
|---|---|---|---|
| correctness | sonnet | 1 | NO FINDINGS (16 + 11 fixtures, 400 fuzz trials, sentinel check) |
| regression | sonnet | 1 | 1 MAJOR + 1 finding against the orchestrator's brief |
| flow-verifier | sonnet | 1 | green; byte-identity on the REAL capture; scaling measured LINEAR; selftest 12/12 |
| regression (re-verify) | sonnet | 2 | **1 BLOCKER** (ru_maxrss survives execve; guard saturates at zero) |
| fail-closed (fresh) | opus | 3 | SHIPPABLE, no BLOCKER, **2 MAJOR** + 3 MINOR + 1 NIT |

## State at disarm

Tree carries the candidate (`capture.py` 71/32, md5 `eeb7e6175c2bfab40c98aa69fcfca0c6`;
`test_streaming_reader.py` untracked). Nothing committed, tagged, deployed or pushed. Fixture
`b0f8de8b...` unchanged and never regenerated. 10 halt stashes intact. Suite 304 passed,
selftest 12/12.

=============================================================================================
# TASK_920 — scorer honesty pack (second row of run 53; started after 914 halted)

Taken because it was genuinely unblocked (no OPEN-decision gate) and file-disjoint from the
halted 914 candidate: `trader/backtest.py` + `trader/cli.py` vs `experiments/907_live_fill_probe/`.
The queue was not exhausted, so stopping after the 914 halt would have scaled the run down
without asking.

## Scope split made explicit at brief time

Legs 1-4 are code and were implemented. **G3 and the four "Record-fix riders" edit `lore/` and
`experiments/LEDGER.md`, which `.dark-factory/config.json` assigns to `lanes.planner_owns`** — the
lane_guard hook denies those writes. They were NOT attempted and are listed as planner remainders.
This is a deliberate split, not an omission.

## Cycle 1 (sonnet) — G1 holds; TWO findings contradict the task file

**G1 (OFF = bit-identical) PASS**, denominated in bytes of the pinned artifact. Independently
re-derived by the regression lens in a detached worktree (both pins SYMLINKED, never regenerated —
verification-rig trap #1): content-sha256 identical on all five commands
(`93c7d2f7` parity / `e5c47eee` backtest / `9b281243` windowed / `0f5a055d` overlay / `b3ee01fb`
selftest, the last derived independently). Design: every new helper is additive and unreachable
from `compute_full_cell` / `cell_metrics` / `window_metrics` / `regime_series` /
`run_overlay_backtest`; `_print_honesty_pack` returns early with all flags at default, and
`cmd_parity`/`cmd_selftest` never call it at all.

**Non-vacuity PROVEN — two poisons, both RED:** (1) the isolation claim's single point of failure —
a new leg-1/2/3 helper wired directly into `cell_metrics`'s FULL branch, nudging gross +0.05
(`e5c47eee` -> `53cdb51c`); (2) a 1% fee-arithmetic nudge in `net_series` (`93c7d2f7` -> `42a150f5`).

### Leg 1 — `--bh-book {legacy-label, true-bh}`
The task's premise is REAL and orchestrator-verified at the cited lines: `backtest.py:219`
sums LOG returns; `:226-228` recomputes `wbh = pr/sum(pr)` every bar and charges `turn` only on
`pos`, never on `wbh` — so the "B&H" column is a per-bar re-weighted, ZERO-TURNOVER book.
Measured: legacy B&H Sharpe **-0.15** vs true-bh **+0.29** (equity x2.84 over 4.58 yr).

### Leg 2 — the task's quoted magnitude DOES NOT HOLD on this book
Task: "≈ +0.3 Sharpe-equivalent (≈5 bp/day)". Measured on FULL net@8: **-1.02 bp/day, Sharpe delta
-0.043** (net@21: -1.02 bp/day, -0.045). **Opposite sign, ~7x smaller.** On a 2022 bear window the
gap is large (-14.5 bp/day, -0.71), so the effect is real and REGIME-DEPENDENT, but the quoted
figure most plausibly describes a long-only 1/N book's rebalancing premium, not this long/short
TSMOM book's traded gross/net. Flagged by the implementer as a first-class finding rather than
built to.

### Leg 3 — a DENOMINATION AMBIGUITY in the task file (D81(c)-shaped, in the spec not the brief)
"+9-12x/yr" is unresolvable between two quantities and the implementer refused to pick the
flattering reading: as a **turnover MULTIPLE** the measurement is 1.24x (56.6 -> 70.2 ann_turn);
as **ADDITIVE turns/yr** it is +13.6. Sharpe side: net@21 **+0.97 -> +0.92 (delta -0.046)**,
decomposition **81% target-weight-change / 19% drift-retrim**.
**G2 tripwire NOT triggered** (-0.046 inside the -0.10 pre-registered threshold), so no disclosure
obligation attaches to prior 819/898 rows.

### Leg 4 — only ONE of three overlays actually moves the number
Baseline net@21 +0.97. Cap-only **+0.97** (capped_steps=0), DD-throttle-only **+0.61 (-0.36**,
half=378 deep=53), band-only **+0.97**, all-three **+0.61** (= DD alone). So the task's premise
that all three are "expected to FAIL on Sharpe" holds for exactly ONE. This is the real TASK_820
gate-2 overlay number and nothing was tuned to force a fail.
**The adversarial question the flow-verifier was pointed at: "the cap never binds" and "the cap
was never applied" produce an IDENTICAL table, and only one is a finding.**

## Repo-level finding worth carrying beyond this row (regression lens, orchestrator-verified)

**"`parity` GATE 1a: PASS" is a materially WEAKER check than byte-identity, and must never be
substituted for it.** Under BOTH poisons above the parity gate still printed PASS.
Verified first-hand: `PARITY_TARGETS` (`backtest.py:455-461`) checks exactly six metrics —
FULL {net8, net21, turn, mult8, maxdd8} + BEAR22 net8 — and **`gross` is not among them**; the
Sharpe tolerance of 0.02 comfortably absorbs a 1% fee-arithmetic error at 8 bp. A cost-constant
regression can therefore pass the parity gate green. Not a 920 defect (G1's byte hash caught both
poisons), but a standing trap for anyone treating parity as the safety net.

## A FOURTH orchestrator error, corrected by the implementer

The 920 brief instructed that the suite and scoring runs go on `gpu-server`. **Wrong**:
`.claude/rules/10-compute-allocation.md`'s own `paths:` frontmatter scopes it to
`experiments/** · tools/** · mcts_trader/** · diff_teacher/** · train_*.py · real_trade_jax.py` —
**`trader/` is not in that list** — and shipping uncommitted edits to the server would require a
deploy, which implementers never do. The implementer ran wrapped on the workstation
(`safe_run --cpu`, sub-second numpy on a 547 KB npz) and DISCLOSED the deviation with its
reasoning rather than silently complying or silently deviating. Orchestrator verified the
frontmatter: the implementer is right. Same family as the run's other three: a rule/number
asserted against the wrong scope without checking the source.

## TASK_920 OUTCOME: SHIPPED (`f896b86`, tag `task920-scorer-honesty`)

Three cycles (MAJOR -> MAJOR -> close), five lens-passes. Deciding verdict: **SHIPPABLE, no
BLOCKER, no MAJOR** — from a lens deliberately FRESH to both remedies (correctness proposed the
cycle-2 fix, fail-closed proposed the cycle-3 fix, so neither could bless its own suggestion).

**Cycle 2's MAJOR (fail-closed lens):** `--overlay` combined with any honesty flag SILENTLY DROPPED
the request and printed the mislabeled legacy `B&H` column at **exit 0 with empty stderr** —
`-0.15`, the per-bar reweight column this task exists to declare "not buy-and-hold", **opposite in
sign** to the +0.29 requested. `--bh-book`'s own help promised a refusal that the overlay branch
never reached. Second instance: `--overlay-only` accepted WITHOUT `--overlay`, printing the
overlay-OFF report — which would have recorded +1.09 as leg 4's cap-only number, corrupting the
very figure the row exists to produce. Both now refuse with **exit 2 and byte-empty stdout**
(orchestrator-reproduced first-hand).

**Scoping calls made deliberately on a last cycle**, to stop a fix becoming a feature:
- **REFUSE, don't wire.** Making `--overlay` honour the honesty flags is a new feature with its own
  risk surface; refusing closes the defect without building anything. Wiring seeded as a row.
- **DISCLOSE, don't pick a convention.** Whether a delisted holding is a total loss or is
  liquidated at last print is a MODELLING decision belonging to the operator. So the phantom share
  is now reported on every true-B&H line (3.55% terminal, 3.77% max at 2025-08-02) rather than a
  convention being silently chosen.

**The deciding pass measured rather than argued.** It attacked the phantom-disclosure judgement by
claiming it was denominated wrongly (a dollar share, against a Sharpe and an equity multiple) —
then recomputed the whole book under the alternative convention: Sharpe 0.2859 -> 0.2757, and
**`1 - 0.96452 = 0.03548` is EXACTLY `phantom_share_final`**. The disclosed number IS the precise
relative shift under the alternative convention. Attack failed on evidence; judgement accepted.

Verification weight: **360-invocation exhaustive flag-combination enumeration** (248 refusals /
112 accepted; **0** surviving silent-drops, **0** over-broad refusals; a strict pass confirmed 124
refusals at exit 2 with `p.stdout == b""`), **15 poisons** re-planted, and every published figure
re-derived from scratch (true-B&H matched to `maxabs 0.0`).

**A poison that FAILED to fire, reported rather than hidden.** Cycle 3's first additivity poison —
a CONSTANT `+1e-9` on `logV` — stayed GREEN, because a constant cancels identically in
`logV[idx1] - logV[idx0]`. It diagnosed this as a BAD POISON rather than a weak test, re-planted
row-dependent, got RED, and disclosed the whole episode "because a poison that passes is otherwise
indistinguishable from a vacuous assertion." The deciding lens independently confirmed the
diagnosis and found four further mutations that DO land inside the same test. **This is the
correct epistemics for non-vacuity work and should be the standard.**

Four MINOR/NIT survive, named in the banner, none changing shipped behaviour — the most notable
being that the delisting-disclosure PRINT PATH itself has no regression guard (planting a no-op
keeps selftest green while the convention silently vanishes): the run's recurring shape, one more
time, caught prospectively.

=============================================================================================
# RUN 53 SUMMARY

| row | outcome |
|---|---|
| TASK_914 streaming reader | **HALTED 3/3** (D82) — reader VERIFIED CLEAN, both MAJORs confined to the new test file; 12.34 GB -> 1.46 GB, artifacts byte-identical on the real capture |
| TASK_920 scorer honesty pack | **SHIPPED** `f896b86`, tag `task920-scorer-honesty`, selftest 12 -> 16 |

## The run's transferable lesson (candidate conduct rule, operator's to rule)

**A VERIFICATION INSTRUMENT NEEDS ITS OWN ADVERSARIAL PASS.** Four times in two unrelated rows,
new code shipped with a guard that structurally could not reach its own defect, and each passed
its author's own review:
1. 914 c1 — the guard built only heartbeats, never reaching the book/trade path carrying the memory.
2. 914 c2 — the guard's INSTRUMENT (`ru_maxrss`) inherited the parent's peak across `execve`,
   saturating at zero.
3. 920 c1 — four new helpers with NO tests at all, which is why the `true_bh_series` fixed point
   survived to the panel.
4. 920 c3 — the delisting-disclosure print path still has no guard (caught prospectively).

`.claude/rules/30-agent-conduct.md` already says a refutation's REPLACEMENT needs its own
adversarial pass. That rule is about DIAGNOSES. This is about APPARATUS, and the wording does not
currently reach it. **Candidate: "A guard, harness, or measurement instrument is itself an artifact
requiring adversarial verification. Never accept a guard that has not been SEEN to fail: plant the
defect it exists to catch, watch it go red, then revert. A poison that passes is a finding about
the poison OR the guard — resolve which, and disclose it either way."**

Corollary that paid off twice: **never let the lens that PROPOSED a remedy be the lens that
blesses it.** Both deciding passes used a lens fresh to the remedy under review, and both found
things the proposing lens had not.

## Orchestrator errors this run (five, one family) — a SECOND candidate rule

1. "HEAD is 362 test nodes" — measured on the ROUND-4 tree, reverted at the start of this run.
   HEAD is 302. Caught by the implementer collecting on HEAD itself.
2. Single-point RSS evidence offered against a claim denominated in FILE-SIZE SCALING. Caught by
   the regression lens under D81(c); the flow-verifier then measured the curve LINEAR.
3. "3.586 days" — lifted from `CHANGELOG.md`'s B1 entry, which describes the **1.248 GB SNAPSHOT
   capture, not the 1.44 GB FINAL one**; and it is span-minus-gaps, the COVERAGE denominator, not
   wall-clock. True: 4.1198 days. Caught by the cycle-3 implementer.
4. Instructed 920's suite onto `gpu-server` — but `10-compute-allocation.md`'s own `paths:`
   frontmatter does not list `trader/`. Caught by the implementer, which disclosed the deviation
   with its reasoning rather than silently complying.
5. Cited `selftest b3ee01fb` as a G1 gate reference — it does not reproduce from HEAD (`bc70f355`),
   having been produced under another lens's capture convention; and a selftest hash cannot survive
   adding tests anyway. Caught by the cycle-2 implementer.

**All five are PROVENANCE failures, and D81(c) does not cover them.** D81(c) governs DENOMINATIONS
(the quantity a gate consumes). These are about WHICH ARTIFACT AND REVISION a number was measured
on. **Candidate: "A brief must state, for every number it asserts, the artifact and revision it was
measured on. A number whose provenance is not stated may not serve as a gate baseline."** All five
would have been caught by that; none by D81(c) alone.

What worked: every one was caught because the briefs instructed implementers to treat orchestrator
numbers as hypotheses to re-measure and to report brief errors as first-class findings. Keep that.

## State at disarm

- **Committed:** `f896b86` (TASK_920 + changelog + banner), tag `task920-scorer-honesty`; plus the
  914 halt records. Nothing pushed. Nothing deployed.
- **Deliberately UNCOMMITTED:** the TASK_914 candidate itself
  (`experiments/907_live_fill_probe/{capture.py,test_streaming_reader.py}`) — D82 decides whether
  it ships. Preserved at `lore/handoff_2026-09-05_factory_run_53_914_halt.patch` (768 lines,
  sha256 `a2577008…`, `git apply --check` OK vs `5010a1b`). This matches the runs 50-52 halt
  pattern; the tree is therefore NOT clean at disarm, by design.
- Fixture `b0f8de8b…` and npz pin `7017e6c3…` unchanged all run, never regenerated.
- 10 halt-preservation stashes intact.
