# Factory run 49 — 2026-09-01

## Shipped

**TASK_907 Addendum B1** (`1360250`, tag `task907b1-declared-vs-observed`) — the D73 clean half.
Boundaries B1/B3/B4/B5, the B2 span fixes, the over-refusal asymmetric fix, terminal reconnect,
all six in-gate defects. The cycle-3 per-pair stale-book exclusion is carved out and stays B2's.

Suite 271 → 302. selftest 12/12. `fixtures/trading_fixture.db` sha256 `b0f8de8b…` unchanged.
B-iv byte-identical on six independently-built captures across two lenses. Cycles: 3, exactly at
`halt_after_failures`, cycle 3 pre-declared as the last and bounded to text plus one pinning test.

## B1-vii: the live capture renders

Pre-fix it refused: `SELF-CONTRADICTORY GAP LEDGER: 598 data-bearing event(s) … STRICTLY INSIDE a
recorded gap span`. The refusal named a true condition and a **false cause**.

Measured (orchestrator streaming pass over all 10,196,132 lines; reproduced exactly by an opus
refutation judge with its own parse):

| class | n | mechanism |
|---|---|---|
| inside an individual `clock_jump` gap | 584 | `recorder.py` stamps a frame's events from one `now`; `_record_gap` sampled a **second, later** clock as `end_ts`. The event that *terminated* the outage lands inside its own gap. min 5.007 µs / median 12.159 µs / **max 32.902 µs**, 584/584 on the HIGH edge. |
| inside **no** individual gap | 14 | Each sits exactly on a `_close_dark_window` gap's `end_ts` *and* the next `clock_jump`'s `start_ts`. `_merge_gap_spans` unions the two legal endpoint contacts; the seam read as containment. Depth from the merged low edge 2.537–4.396 s. |

The refusal text asserted this shape "is not one it can produce under a monotonic clock" and sent
the reader to look for a hand-built ledger or a backward clock step. All three clauses false.

Fixed at both ends: `_record_gap` now takes the caller's already-sampled `now` (new recorder test
drives a `clock_jump` on a µs-resolution injected clock and pins `gap.end_ts == event.ts`), and the
reader tests containment against **individual** gaps with a 1 ms high-edge-only tolerance
(`GAP_EDGE_CLOCK_SAMPLE_TOLERANCE_S`), justified in its own comment from the mechanism and the
measured bound, with a do-not-widen instruction. A judge's three-way ablation on the real capture
forced the split: **598 → 584 → 0**.

Result: `rc=0`, 214-line artifact, four pairs, both estimands, `closed_cleanly=False` correctly
stated (the file is a copy of a still-running capture).

## D57, fourth instance — closed at the claim, not only the code

The staleness run key was `(bid, ask, bid_sz, ask_sz)` — strictly finer than the mid the estimand
consumes (`shadow_engine.py`, `(bid+ask)/2`). A frozen price with jittering sizes therefore read
`stale_book_s = 0.0` and published `1m median 2.48 bp`, deciles −4.00…+9.00, `mid age p50 0.500s`,
on a mid that had not moved for two hours. The 60 s `MAX_MID_STALENESS_S` guard does not catch it:
the mid is fresh by *timestamp*, frozen only in *value*.

Re-keyed to `(bid, ask)`. **Re-keying does not make the gate fire** — measured in a scratch copy
before assigning the fix: `stale_book_s` 0.0 → 7195.0 but `assert_complete()` still passes, because
`uncovered_fraction` is denominated over the **union** of book and trade timestamps and live prints
keep the union dense (0.00132 against a 2.00% bound). Both known stale-book shapes need the
book-mid-only denominator D73 assigned to B2, so the claim could not be made true inside B1.

Every shipped string was narrowed instead, and now prints the falsifying measurement beside itself.
Also retired: "keepalive-driven" was itself unsupported — the recorder emits on a size-only change
too, and the record carries no flag distinguishing a timer firing from a real frame — so the line
is now FROZEN-MID BOOK TIME and asserts only that the mid carried no new information, never that
the wire was unobserved.

## Deliberately un-closed, pinned

Two shapes reach the estimand table unflagged, each while trades on the same pair keep flowing:
a book that goes **silent**, and a mid that **oscillates** over a small value set (period ≥ 2
breaks every literal repeat run). Both pre-existing, both B2's by D73.

The oscillation case was raised as a cycle-2 BLOCKER and **downgraded to MAJOR on measurement**:

```
period=1 trades=NO   stale 7199.0  uncovered 0.9979  REFUSED
period=2 trades=NO   stale    0.0  uncovered 0.0000  REFUSED
period=3 trades=NO   stale    0.0  uncovered 0.0000  REFUSED
period=1 trades=yes  stale 7199.0  uncovered 0.0019  PASSED
period=2 trades=yes  stale    0.0  uncovered 0.0000  PASSED
period=3 trades=yes  stale    0.0  uncovered 0.0000  PASSED
```

The no-trades rows refuse on an independent capture-level gate (`subscribed channel(s) ['trade']
recorded ZERO events`), not on staleness. The with-trades rows are exactly the class B1's narrowed
text already disclaims. The real residue is that the FROZEN-MID line **under-reports** — `0.0 s`
for a two-valued dead mid — fixed as disclosure in cycle 3 and pinned by tests whose docstrings
state they assert a hole, not a feature.

Widening the run key to chase period-N was refused deliberately. That is TASK_891's halt pattern
verbatim (`stash@{3}`): "each cycle's escaping spelling got more exotic … the row chases total
coverage over an unbounded input space." The denomination-correct instrument is B2's freshness
term, which is now load-bearing rather than polish.

## For the operator — the run's real output

**The capture can answer only half its question.**

- Longest gap-free stretch in the entire 3.586-day capture: **561.8 s**. `covers()` requires the
  whole `[t, t+h]` free of recorded gaps, so the pre-registered **20m and 1h horizons are
  structurally 0-sample on all four pairs** — rendered as explicit `not-evaluable` rows (B3
  holding), never omitted. Only 1m and 5m carry numbers.
- Gap-straddling candidate drop: BTC 25,613/36,264 = **70.63%**, ETH 71.20%, SOL 70.91%, XRP 71.30%.
- **BTC/USD reads 0.126% un-observed and 127,071 s of 273,233 s accounted (46.5%) FROZEN MID on the
  same page.** 2,016,348 of its 2,279,026 book events repeat the previous `(bid, ask)`; longest
  frozen run 779 s. Its 1m decile spine `p40 −0.01 / p50 0.01 / p60 0.05` has the shape a frozen
  mid produces, against XRP's `−0.18 / 1.19 / 2.77`. ETH/SOL/XRP do not show it.

D65 authorised this probe on four horizons, "just to know". It supports two, dropping ~71% of
candidates, on a flagship pair whose mid did not move for nearly half its accounted time. Whether
that still answers the question is the operator's call — **DECISIONS row to be seeded (planner
lane, at disarm).**

Stall regime confirmed **from the capture** rather than from dmesg: 566 `clock_jump` gaps, median
inter-start delta **600.0 s exactly** (p25 599.7 / p75 600.1), median duration 61.53 s, 33,743 s
total = 10.9% of span. Independent corroboration of run 48's udisksd SMART-poll finding.

## Not closed, named — needs its own row

`read_capture` materialises the whole capture (`lines = list(_iter_lines_tolerant(path))`) although
the helper is already an Iterator: **11.8–12.6 GB RSS for a 1.248 GB / 10.2M-line file (~9.5×)**,
and `run_shadow` reads it twice; full CLI wall time ~152 s. The capture is still running and still
growing — a 7-day file implies ~24 GB against a 32 G single-job cap. Out of B1's scope; a streaming
rewrite needs its own B-iv proof.

Minor, noticed in cycle 3: `render_report` emits a `Generated: <unix>` line, so artifact
byte-identity A/B requires filtering it. An injectable clock would close it.

## Process notes

- **The orchestrator's own diagnosis was sent to an opus judge briefed to break it**, per
  `.claude/rules/30-agent-conduct.md`. It reproduced every number, proved both mechanisms from
  code, found the 561.8 s horizon consequence, and corrected the framing twice: "none is a corrupt
  ledger" overstated (for the 584 the ledger *is* self-contradictory — a real defect, immaterial in
  magnitude; the defensible form is the bound), and "fix the epsilon and it analyses" understated
  the fix by one independent change (the 14 need individual-gap containment, which no epsilon can
  reach).
- **An error in the orchestrator's own first pass, recorded:** "depth inside a gap" was measured
  against the **merged** span rather than the individual gap, manufacturing 51 false "genuine
  contradictions" until re-denominated. The same wrong-denominator error this task exists to fix.
- **An inverted instruction in the orchestrator's fix brief, caught by the implementer:** the brief
  said not to choose a tolerance in the 33 µs – 2.5 s range, which is exactly the range it must sit
  in; as written it had an empty solution set. The orchestrator's own notes had it right and the
  brief contradicted them.
- **Cycle discipline held at 3.** Cycle 3 was pre-declared as the last and bounded to text plus one
  pinning test, specifically so this run would not repeat run 48's deviation while **D76 is still
  open in front of the operator on that rule**.

## Tree state at disarm — deliberately not clean

`docs/infrastructure.md` and `lore/notes_2026-08-31.md` carry the **operator's own parked
[redacted: VPS ops detail]
dark-factory git gate is ARMED (apply at disarm)"). This run's commit was **path-scoped** to
`experiments/907_live_fill_probe/` + `CHANGELOG.md` + the task banner so that work was neither
swept into a TASK_907 commit nor reverted. Disarming releases the git gate for it.

10 halt-preservation stashes verified intact throughout. No `git stash`/`checkout`/`restore` in the
live tree; all A/B done in detached worktrees. No deploy. gpu-server and the live capture process
untouched.

## Next

Queue after this row: **907 Addendum B2** (now load-bearing — the book-mid freshness term, the
per-pair stale exclusion, the true mid-age disclosure, D74 receipt-lag denomination), then 913's
`WRITE_HARDENING_SPEC.md` residue (post-capture), 881E, 904B, 908, 906. Open decisions: D43, D76,
plus the new horizon/frozen-mid row to be seeded.
