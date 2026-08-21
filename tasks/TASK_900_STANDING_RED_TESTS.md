> **STATUS (2026-08-21): SHIPPED as `task900-standing-red-tests` (ed19a03).**
> `tools/tests` goes **420 passed / 3 failed → 423 passed / 1 skipped / 0 failed**; the one skip
> carries a reason string (acceptance (i) met). All three FIXED — the spec's first-choice
> disposition; nothing deleted, nothing quarantined. Acceptance (ii), one line each:
> **(1)** `test_physics_core.py::test_physics_signals` tested the physics signal shapes and was red
> because the module imported `flax` only under its `if __name__ == "__main__":` guard, which pytest
> never runs — a module-scope import is right because the tool was never implicated.
> **(2)/(3)** `test_jax_env.py::TestLimitOrders::test_limit_buy_places_order` and
> `::test_limit_order_expiration` tested that placing a limit order reserves cash, and were red
> because TASK_124 (`1702a18`, 2026-01-30) deliberately moved the env to LAZY EXECUTION — the commit
> replaces the `total_reserve` write with `0.0`, deletes the cash debit, and documents the new
> contract in-code; `== 0.0` is right because the tests were stale, not the tool.
> **No STOP condition fired** — none of the three was a tool regression. That was not assumed: an
> independent correctness lens attacked the "assertion weakened to buy a green" hypothesis, verified
> `1702a18` first-hand, and confirmed `fill_limit_orders` pairs no-reservation-at-placement with a
> real cumulative afford-check at fill time (`can_afford_cumulative` / `can_afford_short`).
> **Remainder (NIT, no row opened):** `compute_portfolio_value` still adds `order_reserved` back for
> active buy orders — now provably always `0.0`, so vestigial, not a wrong consumer. Separately, an
> unregistered `@pytest.mark.slow` at `test_jax_env.py:583` warns on every run.

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
