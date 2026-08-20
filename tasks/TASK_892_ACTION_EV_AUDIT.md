> **STATUS (2026-08-20): SHIPPED as `task892-action-ev-instrument`** — factory run 37, 2 cycles
> (build=sonnet, fix=opus), three-lens panel (correctness / fail-closed / regression).
> **This shipped the INSTRUMENT, not a verdict.** The EDGE / ANTI-EDGE / NULL ruling is the planner's
> **D47**, seeded by a real leg that remains planner-scheduled on gpu-server.
> Suite **130/130 PASS twice back-to-back** (25 gates; 21 → 93 → 120 → 130 across the re-cut),
> selftest 12/12, gate (i) default-OFF byte-identity re-proven in a real HEAD worktree (0 diffs),
> 10 mutation-reverts at real drop sites.
> **Gate (viii) settled by measurement:** the pre-registered block bootstrap clears 93% at NO stratum
> (block length equals the dependence length, so no sample size fixes it); the calibrated-width
> remedy clears it at EVERY stratum down to 2 blocks. `CALIBRATION_TABLE` = 8 entries, all achieved,
> validated overlapping coverage 0.940–0.975, k falling 3.495 (2 blocks) → 1.376 (80). Verified
> end-to-end: EDGE at 20/80 blocks, **NULL on a real effect at 5 blocks** (honest under-powered read),
> NULL on a true null with no false EDGE, all near-degenerate cases excluded. **So the OOS binding leg
> (116 decisions ≈ 5 blocks) will most likely read a WIDE NULL — publishable per SPEC, not a failure.**
> **DECISIONS NEEDED FROM THE PLANNER:**
> 1. **Ratify or reject the ABSOLUTE 1bp zero-width floor** against D51's "tolerance relative to the
>    data's own scale". The deviation is deliberate: `ci_width` is homogeneous of degree 1 in the input
>    scale (measured independently twice), so a relative-to-self test is mathematically unable to
>    separate a genuinely-tiny-real effect from an artificially-shrunk one. D51 is a planner ruling.
> 2. **A higher-precision calibration re-run is REQUIRED before the real leg** — the shipped table is a
>    cheap in-loop pass (200+200 trials, n_boot=400, 8 counts, MC SE ~1.5–2.5pp), disclosed in
>    `CALIBRATION_PRECISION_DISCLOSURE`, not implied complete by the table's presence.
> 3. **`REPORT_SCHEMA` was NOT bumped** despite five additive report keys (additive only, no external
>    consumer) — planner call if v4 is wanted.
> **OPEN ITEMS shipped as disclosed, not closed:** `assert_never_pooled` has no live caller (no
> aggregator exists) and every report says so, including that an untagged/MISTAGGED subject would not
> trip it; `capped`/`k_cap` is effectively dead code though the refusal is intact; and
> `runs/887_confirm/` holds **no checkpoints on this workstation**, so the 887 A/D subject class is
> wired and fixture-smoked but has never been exercised against a real subject.

# TASK_892 — action-EV audit: does any policy on this recipe have positive-EV actions?

**Status**: READY 2026-08-19 (**RE-CUT per D51**, planner, after factory run 36's 3-cycle halt).
**SPEC Amendment 1 (2026-08-19) is landed and binding** — body sha256 `8408970d…` (supersedes the
2026-08-16 body; the launch note records this hash before the real leg runs). Amendment 1 carries
D49(a) (event-study estimator accepted at cost0; fee16 demoted to non-binding diagnostic),
D49's interval-method escalation (calibration-harness gate before the real leg), and the four
operator-adopted pathfinder amendments (`docs/pathfinder_audit_2026-08-19.md` §D).
**Size: S. Risk: LOW to build, HIGH in consequence** — its verdict seeds **D47**, the pivot ruling
for the whole post-887 line. (The header formerly said D42 and the queue note D44 — both stale;
D47 is the live number.)
**Sequencing from the original spec is DISCHARGED** (886/890 run 34, 878 run 35). **The
pre-registered experiment design is `experiments/892_action_ev/SPEC.md` as amended — it is the
contract; this file is the build row.**
**Resume from the tree state** — the working tree CARRIES run 36's work; durable copy
`lore/handoff_2026-08-19_factory_run_36_892_cycle3.patch` (sha256 `e8bc83b0…`).

## Thesis

887 proved the hold collapse is adaptive: at zero costs, more activity = monotonically more loss.
What 887 could not distinguish: whether the policy's non-HOLD actions carry **no edge** (the
collapse is a correct refusal of worthless actions) or carry edge the policy fails to **find**
(the collapse is a search/selection failure). The two readings demand opposite pivots — reward/
data/edge structure vs policy machinery — and the distinguishing measurement is cheap: replay the
policies, and compare realized forward returns after their non-HOLD decisions against a
**matched-activity random null** that sees the same states at the same costs. The pathfinder
audit (`docs/pathfinder_audit_2026-08-19.md` §A) sharpens the stakes: 887's environment
demonstrably CONTAINS findable edge (the trend teacher is positive-EV at fee0), so an OOS
ANTI-EDGE/NULL here is a learning/search failure with a price tag, not a property of the world.

## What is settled and must NOT be re-opened (D51)

The TRACE/instrument layer: the realized-allocation budget check, `universe_pin` consumed via
`ev.enforce_universe_pin` (886's split inherited, never synthesized), the NaN guard, the
`walk_kind` enum, the per-invocation smoke checkpoint path, atomic writes, the
`--marginals-sidecar` rename with its 470-prefix class-closing gate, and the operator-coherence
refusal. Gate (i) byte-identity re-proven (0 scored-block diffs, canonical JSON identical);
suite 93/0 twice back-to-back; 8/8 mutation reverts; selftest 12/12.

## Scope of this re-cut (the layer all three cycles died in, plus the Amendment-1 surface)

1. **`report.py`'s banding/exclusion layer, rebuilt under D51's transferable rule:** every
   exclusion guard is a **tolerance relative to the data's own scale, never an exact-equality or
   bit-identity test** — and its test constructs the **NEAR case**, not the exact one (cycle 3's
   killer: bit-identical floats caught, 1e-9-jittered near-duplicates reading EDGE at CI width
   6.7e-10).
2. **The Amendment-1 surface:** the position-CHANGE-only co-primary split (entries/exits vs
   re-trims) with per-decision net exposure and forward market return recorded; the OOS leg
   (2025-10-05→2026-03-31) wired as the binding read; the fee16 leg emitted as disclosed
   diagnostic (estimator, deviation, bias direction all stated in the artifact); the 887 A/D
   subject class, never pooled.
3. **The calibration gate (pre-registered in SPEC [A1]):** the two-scenario calibration harness
   must certify the chosen interval method at ≥93% two-sided coverage under realistic dependence
   at EACH reported stratum's block count — including the OOS leg's ~6-block case — before the
   real leg runs. The choice, measured coverage, and any uncalibratable stratum (flagged, never
   bare-CI'd) go in the artifact.
4. Fixture smoke stands as built; extend it so the planted-sign case is exercised through the
   NEAR-duplicate guard, not only the exact one.

## Acceptance (gates)

- (i)–(vi) as originally written (default-OFF byte-identity; matched-null activity and
  long/short-mix deltas ≤0.5pp; path-independence bit-for-bit per seed; fixture planted-sign
  recovered + shuffled-trace falsification reads non-EDGE; no test monkeypatches the function
  under test; regression tests, mutations independently rebuilt) — several already proven on the
  preserved tree; re-prove, don't rebuild.
- (vii) Every banding guard fires on its NEAR case; a bit-identity-only guard is a BLOCKER.
- (viii) The calibration harness certifies ≥93% coverage per stratum, or the stratum ships
  flagged insufficient-n.
- (ix) The artifact carries the D49(a) disclosure triplet (estimator, deviation, bias direction).

## Out of scope

The trace layer (settled — D51). The real leg (planner-scheduled on gpu-server GPUs, per the
operator's 2026-08-19 compute-placement guidance). Any verdict past INSTRUMENT GREEN: the
EDGE/ANTI-EDGE/NULL ruling is the planner's, seeded as **D47** with the LEDGER row. The
env-accounting/PV-exact estimator (D49(b)) — not required unless a future ruling makes fee16
load-bearing. Training anything.
