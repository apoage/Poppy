# Factory run 38 — handoff (2026-08-22)

Seven tasks shipped, two decision rows seeded, tree clean, nothing pushed.
Base `730c629` → head `a84cb18`.

## Shipped

| Tag | Row | Cycles | Verification outcome |
|---|---|---|---|
| `task900-standing-red-tests` | TASK_900 | 1 | `tools/tests` 420p/3f → **423p/0f**. All three FIXED (spec's first choice). Correctness lens attacked the "assertions weakened to buy a green" hypothesis and it did not survive: `1702a18` verified first-hand, `fill_limit_orders` confirmed to pair no-reservation with a real afford-check. |
| `task894-frame-symbols-digest` | TASK_894 | 2 | Run-34 forgery now REFUSES for a stamped ckpt, certifies-with-`NOT PROVEN` for an unstamped one — rebuilt through the real CLI by three independent agents. Byte-identity vs a real HEAD worktree: zero differences. |
| `task897-corrected-809-estimator` | TASK_897 | 3 | **Instrument only.** 2 BLOCKERs closed. |
| `task881s-score-source-enumeration` | TASK_881S | 3 | D50 closed. Final gate: 960 differential CLI pairs, shuffled-`os.scandir` determinism probe, reconstructed prior build for a byte-diff. |
| `task898-trend-decay-instrument` | TASK_898 | 2 | **Instrument only.** Caught a confound that would have driven a wrong ruling. |
| `task881g-manifest-grammar` | TASK_881G | 3 | Gate (i) executed and FAILED the doc. 1 BLOCKER + 2 MAJOR closed. **Unblocks 881D.** |
| `task896-819-fill-comment` | TASK_896 | 1 | Re-pin verified by an independent 1,218-key diff of the OLD instrument. |

## The three findings that change what you do next

**1. TASK_898 found a confound that would likely have produced a wrong verdict.** The long-flat
(spot-legal) variant **silently de-leverages** — `weights()` is signal-sign-agnostic, so removing
shorts sends their budget to *implicit cash*, not to the longs. Gross exposure is **43% of baseline
on FULL and 27.6% in BEAR22**, the exact regime the crash-avoidance thesis turns on. A
leverage-matched third series was added: in **every** regime the variant's Sharpe was flattered by
cash drag, and FULL goes **+0.321 → −1.308**. Reading the un-matched variant alone would plausibly
have said *VARIANT LIVES* where the long-only signal by itself scores −1.31.

**2. TASK_897's own pre-registered band is not readable as written (D55).** Its CI is
measured-uncalibrated: coverage **0.530 at n_dec=100** under overlapping dependence vs nominal 0.95,
and block length is capped at 20 so longer dependence is uncapturable **at any n**. The CELL CLEARS
band requires `CI lo > 0`. Rule D55 before the srv leg is worth launching.

**3. Gate (i) is a real instrument, and it failed the doc it was pointed at.** An isolated agent
authored 46 cases from 881G's grammar doc alone; 40 agreed, **3 SURPRISE + 3 UNDERSPECIFIED**, every
one a doc defect. The failures were at the *first realistic shapes an author types* — a dangling
entry actually exits 2, and a fenced example whose closing fence follows the last item withdraws the
whole file. Worth reusing this shape (author-blind corpus → run it) wherever a doc is load-bearing.

## Decision rows seeded

- **D55** — 897's CI is uncalibrated; the pre-registered `CI lo > 0` conjunct cannot be read.
  Recommendation (a): calibrate *this* path (892's method, not its table) before launching.
- **D56** — 897's gate (iii) requires the 877 alias fold applied; its own Out-of-scope forbids
  touching the 809 lineage that would carry it, and the fold is verifiably absent there.
  Recommendation (a): measure the blast radius first (a cheap query — how many 809-universe slots
  have twin spelling-variant rows over 2024-11→2026-03?), then choose.

## State of the queue

**Next Ready row: TASK_881D** — unblocked by 881G, whose grammar doc is now in its corpus
precondition. Note its `--apply` freeze also needs **880's D35** gate, which is a human call.
**TASK_899** was skipped deliberately: the queue itself marks it planner/server ops, not factory
work. **TASK_880** stays GATED (D34+D35).

## Standing remainders, by row

- **881S**: MAJOR-A — one shared absolute `--log-glob` is reported once per co-located score dir.
  Pre-existing at `bf1f8d5`, *measured* so by a test that runs the old code. Needs a row permitted
  to touch `_load_scores_strict`. Plus 3 fail-closed count-inflation MINORs.
- **894**: gate (ii) shipped on an **accepted proxy**, recorded as such — the literal 7-checkpoint
  A/B was not run and needs authorizing as a scheduled job if wanted.
- **897**: `recompute_best.py`'s repair lives under gitignored `runs/`, so staging it to srv is a
  manual step the clean-tree gate cannot see. Do **not** pass `--skip-inherited-ic-check` on the real
  launch.
- **898**: the leverage-matched series is concentration-unbounded (scale hit 221× on a thin fixture
  row) — diagnostic only. `SPEC.md` was authored *and* amended untracked, so its pre-amendment prose
  cannot be byte-diffed; commit SPECs before amending.
- **896**: one number in the 819 comment is not recomputable from committed artifacts; the comment
  says so and the verification took it on trust and said so.

## Process notes worth keeping

- **A guard written as a syntactic test for one spelling of a property is not a guard on that
  property.** 881S burned two cycles on this: an `os.path.isabs` check standing in for
  "directory-independent pattern", broken by a relative `../shared.log`. The fix was to implement the
  invariant (resolved-path collision).
- **A repro test that asserts a substring and an exit code but never a COUNT will ship a
  per-directory inflation under a passing test.** That is exactly how 881S's MAJOR got in.
- Escalation to `opus` was used on 897, 898, 881S (×2) and 881G. Every one of those rows had a
  BLOCKER or MAJOR that the sonnet cycle produced and the opus cycle closed.
