# Factory run 54 — handoff (2026-09-06)

Implementer lane, armed 07:46, disarmed at the end of this doc. Two rows shipped, one queue
contradiction found, one DECISIONS row seeded. Nothing pushed.

## Shipped

| Row | Tag / commit | Verification | Outcome |
|---|---|---|---|
| TASK_916 cost truth pack | `task916-cost-truth-pack` · `18d42d3` (+ banner `017cfdb`) | 4 lenses → fix pass → delta lens. No BLOCKER, no MAJOR at any point. 2 cycles, both sonnet | SHIPPED |
| TASK_917 instrument pack | `task917-instrument-pack` · `8f411d1` (+ banner/ledger/D84 `13385b3`) | 4 lenses (sonnet) → BLOCKER+MAJOR → fix (opus) → 2 lenses (opus) → 6 MAJOR on arguments → cycle-3 D76 closing pass (opus). 3 cycles | SHIPPED |

Suite went 16 → 17 checks (916's ST17) and stayed green throughout. Fixture sha256
`b0f8de8b…` unchanged across the whole run, verified before and after every pass.

## The numbers that matter

**916.** At the **verified** Kraken base taker tier (80 bp) with borrow charged, PIT FULL goes
`+0.279 → −0.132`. The record's cost assumptions (21 bp legacy; the `{8,16}` maker set) are
2–5× optimistic below ~$100k/30d volume, so a book that reads mildly positive under them reads
**negative** under the honest tier. Registered-window PIT anchor reproduces **n=2800 exactly**
at uncharged net21 +0.5849, charged **+0.2223 (low) / −0.1402 (high)** vs the recorded +0.56 —
so the task's pre-committed consequence **fires** and the borrow-inclusive bar becomes TASK_915's
reference. **Carry the caveat with the number:** that is a POINT ESTIMATE; the bootstrap CI is
[−0.47,+0.90] low and [−0.86,+0.56] high (upper bound exactly at the bar), and the recorded
reference is equally wide. Not a statistically demonstrated shortfall.

**916 leg 2 discharged D80(i)'s re-verify instruction.** The Kraken+ waiver **excludes Pro/API
trades** (app/web Buy-Sell-Convert only) — confirmed against primary sources. An API-traded book
gets nothing from the subscription; the honest tier is the Pro ladder. Three intermediate tiers
the harvest omitted were found ($2.5k 0.30/0.60, $10k 0.22/0.38, $25k 0.20/0.35), and tier can
now qualify on "Assets on Platform" as an alternative to 30d volume. **Not modelled, disclosed:**
the one-time margin opening fee and the 2% liquidation fee (no margin-call hook in a backtest).

**917.** P2 **KILL FIRED, robust** — the event-conditional liquidity-provision family is closed;
κ₃ = +0.0361 against a 0.18–0.20 line, never approached anywhere in a threshold sweep 8→40, and
the hedge round trip is 338–356 bp against +62.7 bp capture. P1 **INCONCLUSIVE** — regime-gating
is **unmeasured, not closed**. P3 **SURVIVES** — age-conditional post-listing IC +0.0304,
NW-t +4.15, placebo-clean under an independent i.i.d. null (p = 0.0050, z ≈ 5.5), leakage-audited.
Recorded as an instrument reading, not a promotion candidate.

## Decision-changing findings

1. **D84 SEEDED — P1's regime-gating question is unmeasured.** The 0.10 threshold was inherited
   from `mech_B.md`'s *different* statistic and never calibrated to pooled VR(30); the anchor's
   power to separate ρ̄=0.10 from 0 is **~9–10%**, and observed dispersion is indistinguishable
   from estimator noise at all 11 block lengths. **No block length could have fired a defensible
   kill.** Recommendation in the row: build the powered instrument (average over tiling phases,
   sample-match the denominator, k-invariant bias correction) and re-calibrate the threshold, or
   drop the line explicitly. Three book families otherwise sit in limbo.

2. **`tasks/QUEUE.md` contradicts itself on row 3 — planner's to resolve.** Line 26 calls
   write-hardening "**NEXT TAKEABLE**; pure code, gate open"; line 100 fences the only file it can
   edit ("reads yes, writes no, until D82 rules"). The spec's whole design (R1–R6) lands in
   `CaptureWriter._write` at `experiments/907_live_fill_probe/capture.py:303-306`. The 914
   candidate's hunks are reader-side (504–1780), so they are line-disjoint — but the fence is
   stated on the path, not a line range. I treated the fence as binding and skipped to row 4.
   **If you rule that the writer-side region is fair game, row 3 becomes immediately takeable.**

3. **Every placebo in TASK_917 was defective, in three different ways.** P1's circular-shift null
   contained **exact copies of the statistic under test** (`c(1) == ρ̄` bit-identically), which
   bounded min achievable p near 0.08 and made "none Bonferroni-significant at α=0.00625" a
   structural certainty rather than a datum. P2's relocated events onto dates where the asset had
   no print, silently discarding them (median draw **2124 of 7360** events; the "~0.5% null mass at
   the kill line" claim came from a 426-event draw and is withdrawn). P3's was a deterministic
   `linspace` over one bounded scalar, so `max|placebo|` **saturates** and its p-value was a pure
   function of grid density. **Proposed rule for `.claude/rules/` (planner's lane, not written by
   me): a null must be sample-matched to the statistic under test, must exclude the draws that
   reconstruct the true assignment, and must report its minimum achievable p-value.** All three
   defects would have been caught by it.

4. **A third worktree trap belongs in `.claude/rules/20-verification-rig.md`** (currently documents
   two). `python -m <pkg>` prepends the **current working directory** to `sys.path` ahead of
   `PYTHONPATH`, so running the HEAD side of a byte-identity A/B from the live tree silently
   imports the LIVE package and reports a spurious match. It produced a false-positive G1 inside
   this run's own panel; the verifier caught it because `--borrow-model` "worked" on code that
   predates the flag. **Rig sanity check: the old tree must ERROR with
   `unrecognized arguments`.** Three later agents applied it correctly, one also confirming
   `trader.__file__` resolved into the worktree before trusting the diff.

## Process findings against myself

- **I ran an in-tree mutation-testing lens concurrently with three readers**, and it corrupted two
  of their readings — one judge saw a benchmark number it could not reproduce (`+0.568` vs the true
  `+0.197`), another saw a one-off ST17 failure it could not reproduce across 239 further trials.
  Both diagnosed the tree correctly and neither implicated the shipped code. **Fix applied
  mid-run:** later panels were told to mutate only copies outside the repo, and the delta lens ran
  alone. **Residual gap in my own brief:** "read-only" review that *executes* an instrument still
  writes, because these scripts emit `__file__`-relative artifacts — two judges tripped that. The
  closing pass now makes each script announce the artifact it is about to overwrite.
- **I relayed an unchecked label to the operator** (the worst-moving 916 cell is the *high*
  scenario, not low) and **passed a miscounted figure into a brief** ("9 of 11 violate ≤0.15";
  the list shows 8). The closing pass caught the second and shipped the computed 8 with the
  criterion stated.
- **I recommended the wrong D84 answer before the evidence was in** — "fire the kill on consistent
  CI standards" — which the estimator lens refuted: that standard is a lottery on an undisclosed
  tiling origin (15.6% of origins land above the line). Recorded because the reversal is the useful
  part.

## Where it stands

- Queue rows shipped: 4 (TASK_916) and 8 (TASK_917). Row 3 blocked on the contradiction above;
  rows 1/2/5/7/9(B) still gated on D43/D81/D82/D80.
- **Next takeable: TASK_919 phase A** (rebalance grid: H × hysteresis θ × accounting × 916's fee
  tiers incl. borrow). Unblocked by 916 shipping. Carries its own verification arithmetic
  (analytic flip rates 0.165/0.117/0.0957/0.0746 per day at H=1/2/3/5) and flags its two anchors
  as n=1 readings to re-verify rather than re-trust.
- **P4 (917) is prepared-not-run** with a recorded pre-`--full` blocker: its local
  `discover_universe()` omits the `source_filter="kraken_rest"` that P1–P3 use, so it carries 8
  real assets as a 10-column panel (`TWX/USD`, `HOLE/USD` admitted). Reconcile before invoking.
- **Fenced and untouched all run:** `experiments/907_live_fill_probe/capture.py` (M, 71/32) and
  `test_streaming_reader.py` (untracked) — the TASK_914 candidate awaiting D82. Verified `71 32`
  at every hand-off between agents.
- Housekeeping: a stale detached worktree `wt914` at `e146d5a` sits under a dead session's
  scratchpad (`git worktree list`). Harmless litter from run 53; left alone rather than removed
  mid-run.

## The two questions that unblock the most

1. **D82 — TASK_914's scope** (deferred 2026-09-06). It also gates row 3 indirectly, since the
   fence is what makes write-hardening untakeable. One ruling frees two rows.
2. **D84 — build the powered VR instrument, or drop the regime-gating line?** Until this is
   answered, gated trend/reversal, CPD switches and XS-AUTOCORR books are neither closed nor live,
   and TASK_917 P1 will keep being mis-citable as a closure it explicitly refuses to be.

Secondary but cheap: **your actual Kraken 30-day volume tier** is still unrecorded, and it alone
decides which ladder row is honest for every restated column 916 published.
