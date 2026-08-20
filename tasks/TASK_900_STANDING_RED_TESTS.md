# TASK_900 — the three standing red tests in `tools/tests`

**Status**: READY 2026-08-20 (planner; flagged by factory run 37 — 3 failures in the wider
`tools/tests` directory are PRE-EXISTING, identical failure set at HEAD and on every candidate,
unrelated to any queue row).
**Size: XS. Risk: LOW.** **FILLER (parallel-safe).**

## Why this row exists

Standing reds erode the signal of every suite run — every future "suite green" claim carries an
unspoken "except the three we all know about", which is exactly how a real red gets waved
through. This program's rule is that a suite count is not evidence until someone reads the
output; a permanently-red suite trains readers to stop reading.

## Scope

Diagnose the three failing tests (identical set at HEAD; see 881I's shipped banner for the
pointer). For each: **fix it, or delete it with the reason recorded, or quarantine it
(`pytest.mark.skip` with a named reason and an owner row)** — in that order of preference. No
fourth outcome: a red test that nobody owns is deleted, not kept.

## Acceptance

- (i) `tools/tests` runs clean — zero failures, and every skip carries a reason string.
- (ii) The diagnosis of each of the three is one line in the commit message or this file's
  banner: what it was testing, why it was red, why the chosen disposition is right.

## Out of scope

Re-speccing the tools the tests cover. Any test that turns out to be RED FOR A REAL REASON
(the tool regressed, not the test) stops this row and seeds its own decision row instead.
