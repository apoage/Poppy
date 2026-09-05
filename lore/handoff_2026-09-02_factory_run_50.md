# Factory run 50 — TASK_907 Addendum B2: HALTED at cycle 2 of 3

Date: 2026-09-02. Base HEAD `96b59eb`. Nothing committed, nothing deployed, nothing pushed.

## Outcome

**HALTED.** Two cycles produced five BLOCKERs. Cycle 3 was available and deliberately not
spent: the row is blocked on operator decisions, not on implementation.

Work preserved two ways (per `.claude/rules/30-agent-conduct.md`): left in the live tree, and
copied to `lore/handoff_2026-09-02_factory_run_50_907B2_halt.patch` (2858 lines, sha256
`b11c1bbe0c06d5df6133cd47eb9e1907114c1af894ce8aae4ce3a548b34ff909`, `git apply --check` OK
against HEAD). Suite 302 → 342, selftest 12/12, fixture `b0f8de8b…` unchanged, 10
halt-preservation stashes intact.

**The tree is NOT shippable.** Two BLOCKERs are open in it: a contaminated limit price leaks a
~4000 bp markout past the run-exclusion, and a cross-capture over-refusal discards healthy pairs.

## Why it halted, in one sentence

Every one of B2's three disqualifiers is denominated in a quantity that is not the one it needs
to gate — the same error that halted run 46 and forced the D73 split — and the row keeps
reproducing it because the question it implements (*what makes a book-mid estimand trustworthy?*)
has a ruled qualitative shape and three never-ruled quantitative bounds.

## The two decision-changing findings

### 1. D65-amend's governing premise is refuted by the final capture

Run 49 recorded "20m and 1h are STRUCTURALLY 0-SAMPLE on all four pairs", reasoning that the
longest gap-free stretch was 561.8 s while `covers()` requires the whole `[t, t+h]` clean.
D65-amend ruled "two horizons suffice" on that basis.

Measured on both files with the same code:

| capture | span | gap records | longest gap-free stretch |
|---|---|---|---|
| snapshot (3.586 d) | 309,784 s | 1688 | **561.8 s** |
| FINAL (4.117 d) | 355,963 s | 1727 | **9,670.9 s** |

Top five stretches in the final capture: 9671, 7198, 6668, 5658, 4131 s. **Run 49's measurement
was correct for the file it had** — 561.8 s reproduces exactly. The final capture added only 39
gaps across its last 46,179 s: the stall regime very nearly stopped near the end, consistent with
run 48's pending `systemctl mask --now udisks2` action.

At HEAD, on the final capture, all four pre-registered horizons produce samples on all four
pairs (BTC/USD: 1m 8557, 5m 4536, 20m 962, 1h 467). D65 pre-registered four horizons and this
capture supports four.

### 2. The probe's verdict turns entirely on three never-ruled bounds

The real capture REFUSES under the shipped constants. Every pair is excluded for exactly one
reason — unexplained book silence against a bound of `0.0000%` ("any violation disqualifies"):

| pair | unexplained silence | % of accounted | worst interval | intervals |
|---|---|---|---|---|
| BTC/USD | 303.8 s | 0.0952% | 15.6 s | 26 |
| ETH/USD | 561.1 s | 0.1758% | 17.2 s | 47 |
| SOL/USD | 97.8 s | 0.0307% | 15.2 s | 8 |
| XRP/USD | 488.8 s | 0.1532% | 19.6 s | 40 |

At any bound ≥ 0.2% all four pairs publish. A clearly-labelled counterfactual render at 0.2%,
run in a scratch copy of the lane (live tree constant verified unchanged at `0.0` afterwards),
produced a full 350 KB artifact with all four pairs and all four horizons. The factory did not
pick this number: it decides D65's answer.

## What cycle 2 got right, verified on the real capture

`book_uncovered_fraction` now reads **0.0019–0.0082%** across all four pairs, where cycle 1 read
~46% for BTC/USD. The wire-evidence denominator — a pure keepalive echo is an event whose FULL
payload `(bid, ask, bid_sz, ask_sz)` repeats — correctly separates "the market was still" from
"we could not see the market". BTC's 138,449 s of frozen mid stays disclosed via `stale_book_s`
without being counted as un-observed.

Correctly denominated, item 2b became a real signal: it flags 10/1222, 12/773, 24/600, 10/441
frozen runs per pair, worst runs 66.7–94.0% outside against baselines of 16.9–30.9%,
`venue_ts`-restricted.

## BTC/USD's frozen mid is real market stillness

Established against an attempt to falsify it (a stale book must catch up to where trades already
went when it recovers):

| pair | median \|un-freeze jump\| | median \|in-run trade drift\| | corr |
|---|---|---|---|
| BTC/USD | 0.257 bp | 0.006 bp | 0.58 |
| ETH/USD | 0.320 bp | 0.020 bp | 0.95 |
| SOL/USD | 0.483 bp | 0.482 bp (tick floor) | 0.82 |
| XRP/USD | 0.144 bp | 0.037 bp | 0.92 |

The catch-up signature is absent. Frozen runs are quieter than average (trades ~4.5× sparser) and
print outside the touch *less* than baseline (BTC 10.0% vs 23.3% capture-wide). ETH's book freezes
precisely when locked at a 1-tick spread (median spread 0.01 frozen vs 0.11 overall).

Retracted from an earlier pass in this run: per-tick bp conversions quoted for ETH and XRP were
derived from a `min_tick` of 4.5e-13, which is float-subtraction noise, not a tick. The spread
medians stand; those two conversions do not.

## Open BLOCKERs in the preserved tree

1. **Contaminated limit price** (`shadow_engine.py:355-397`). The flagged-run membership test
   checks only the fill/anchor trade's timestamp, never whether the posted limit (`bar.close`,
   `:371`) came from a bar composed of flagged-run trades. The cycle-1 repro still publishes
   `median 4000.52 bp` at rc=0, now dressed with an honest-looking "8 of 16 candidates excluded"
   disclosure that gives false confidence. The shipped test asserts the same anchor-timestamp
   criterion the implementation uses — test and implementation share one blind spot.
2. **Cross-capture over-refusal** (`cli.py:118-134`). A pair disqualified in *every* contributing
   capture now triggers a B4 collision refusal, discarding every healthy pair in both captures.
   B4's stated concern ("which capture's number silently wins") cannot arise when no number is
   produced anywhere. This is the D73 over-refusal shape one layer up.

Also open: FIX 6's decline rationale is mechanistically false and ships in the rendered artifact
(`render.py:685`) — `_build_bars_and_mids` buckets on each trade's own local receipt `ts` and never
consults `venue_ts`, so re-anchoring would not re-bucket prints; the pre-registration-amendment
half of the argument is sound on its own. And echo detection is gated on `declares_keepalive`, so
a capture declaring no emission policy disables the wire-evidence detector entirely (bounded: the
production recorder always declares both keys).

## Process notes

- The five-lens panel earned its cost: two lenses found the cycle-1 outside-book BLOCKER
  independently and convergently, from different angles.
- The security lens proved the TASK_910 credential fence covers a brand-new lane file **by
  mutation**, not by reading — run 47's recurring "a file that should be scanned wasn't" hole does
  not recur.
- Cycle 2's own report claimed the 4000 bp repro fixed; a judge disproved it by re-running the
  repro verbatim. A fix pass's self-report is not evidence.
- `CaptureWriter` **appends**: reusing a fixture path across runs silently produces a resume
  session and a spurious gate-(iv) refusal. It cost cycle 2 one false failure. Worth a fresh
  tmpdir in the flow-verifier's `run_scenarios.py`.
- The stop gate fired repeatedly against legitimately-running background agents (rule 5 in
  `00-shared-tree-concurrency.md`). No queue row was taken to satisfy it and the factory was not
  disarmed early.

## Next

The row resumes only after the operator rules the bounds and re-rules D65-amend. Resume mechanics:
re-apply the preserved patch, close the two open BLOCKERs, then re-render the real capture
(14.8 GB peak RSS, 3:29 — TASK_914 would reduce this and is queue row 2).

## Addendum — cycle-2 regression lens (returned after disarm)

**CLEAN: no BLOCKER, no MAJOR.** Six independently-built fixtures, including the exact
200s-silence / 150s-gap / 50s-residual shape that broke cycle 1, rendered through a detached HEAD
worktree vs the live tree. **B-iv passes**: every pre-existing number byte-identical; only the
permitted new blocks differ. FIX 1 confirmed restored at both code and render level
(`worst 200.0s` byte-identical). FIX 3's two coverage lists proven separate by construction
(`covered_s` 1.0 vs `book_covered_s` 7.0 on the adversarial full-payload case). The removed test's
replacements cover strictly more, including a render-level pin. Node IDs vs HEAD: 0 removed, 40
added (report said 39 — one off, the known parametrization/collection discrepancy).

This does not change the halt: the lens answers "did a number move", while the two open BLOCKERs
are a fail-closed leak (contaminated limit price) and a correctness over-refusal (cross-capture).

Two things worth carrying forward:

- **Verification gap.** The lens could not verify the "vs cycle 1" claims because no cycle-1
  source artifact exists — the preserved patch captures the cumulative cycle-2 state vs HEAD, not
  the cycle-1 intermediate. Preserving a **per-cycle** patch would close this.
- **Process warning.** The lens nearly reported a false BLOCKER from a stale render left in its
  scratchpad while the live tree was mid-edit. It checked mtimes, confirmed `git diff --stat` was
  stable, rebuilt every fixture and re-rendered before concluding. This is the shared-tree
  concurrency hazard in its documented form: a cached artifact from a flickering tree. Reviewers
  must not trust cached renders.
