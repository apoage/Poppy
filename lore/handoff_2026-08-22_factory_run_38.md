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

---

# Run 38b — the D55/D56 adaptation (2026-08-22, later same day)

Planner ruled D55(a) and D56(a); the factory took the follow-up row. **HALTED on the 3-cycle
rule.** Work is green and stashed at `stash@{0}`; **D57** seeded. Tree clean, nothing pushed.

## What is DONE and worth keeping (in the stash, 116 tests + selftest 12/12)

**D56(a) — the alias read-layer is VERIFIED CLEAN and independently reusable.**
`alias_read_layer.py` applies the canon 868/877 union (`resolve_requested_symbols` +
`alias_agreement_gate` + evidence floor — not hand-rolled) as a READ-TIME layer for **both** the
universe ranking and the panel build, with the 809 lineage byte-untouched. A flow-verifier ran
`rank_universe_with_alias_union` over the **full 291-candidate pool** and measured **zero rank
changes** — the 6 material twins carry zero rows in the train window, so the identity split lands
in OOS only and cannot move the top-64. Per-slot row counts reproduced against the real DB exactly
(window a: 0% missing; window b: `/USDTEUR` 56.1%, `/USDTAUD` 56.0%, `/USDC/USDT` 55.2%,
`/USDTCHF` 53.4%, `/USDCGBP` 51.3%, `/USDC/USD` 37.7%). A genuinely disagreeing overlap raises
`AliasDisagreementError` rather than averaging.

**D55(a) — the calibration re-keying is a strict tightening**, three cycles deep: k keyed to the
real `(block, n_dec)` pairing with 67 strata generated from `block_for`'s own step function (one at
every block transition); coverage re-measured **at the borrowed pairings**, not only at the strata;
a certified ceiling that refuses rather than extrapolates; `apply_ci_calibration` hardened with a
superset invariant plus a zero-width refusal and 15 dedicated tests; and
`CI_UNCALIBRATED_DISCLOSURE` made an f-string over `MAX_BLOCK_DEFAULT`, pinned by a test asserting
the **live** cap so it cannot go stale again. Three false shipped strings retracted with their
arithmetic recomputed.

## What halted it (D57)

Cycle 3 added a dependence guard — τ = integrated autocorrelation time, refuse when τ > block,
fail closed on unmeasurable — and the artifact claims "the condition is now MEASURED per cell
rather than assumed". **It is measured only inside the ACF family it was calibrated on.** A block
bootstrap needs `block ≳ correlation RANGE`; **τ mixes range with amplitude**, so the comparison
quantity is wrong out-of-family:

- `0.2·AR1(φ=.99)` — a persistent component worth **4% of variance**, population τ = 8.6, genuinely
  below block 60 — reads **τ=1**, certifies **100%**, covers **0.805**. At φ=.999: **0.376**, below
  the 0.530 that raised D55.
- On **real** minute data, every per-decision series at H ∈ {1,5,20}, long-only and long/short,
  reads **τ = 1–2 against block 60**. A non-degenerate sweep refused **0/864**; an adversarial
  out-of-family fixture still certified **90%**.
- Containment check on the real long/short H=1 series: 97 disjoint 1000-decision chunks contained
  the grand Sharpe in **0.639**, while the guard certified **100%**.

So the guard is not conservative — it is **inert exactly where it will be used**, and would switch
D55(a)'s protection off for the whole srv leg under a paragraph the artifact renders as legitimacy.
That is a question about what the interval may *claim* — the same kind of question D55 itself was —
so it went to the planner rather than a fourth cycle.

**D57 recommends (a):** narrow the claim, ship the rest, keep the `CI lo > 0` qualifier until a
range-based probe exists. (b) build that probe as its own row, validated **out**-of-family. (c)
refuse certification wholesale and publish every cell qualified, leaving `Sharpe >= 1.0` and
`PnL > 0` readable.

## Process note worth keeping

Three separate defects this run were of one shape: **a claim in a shipped string that the code does
not back** — "the matched-dependence worst case" (it is the easiest point), "500 was not a stratum
at all" (it is one), and now "the condition is MEASURED" (only in-family). Each shipped verbatim
into the artifact at the exact spot a reader checks legitimacy. The 881G gate-(i) shape — have an
isolated agent author from the doc alone, then run it — is the cheap general test for this class.
