# TASK_873 — Pin the fixture's identity: close the "gitignored baseline" class the run-25 incident exposed

**Status**: READY 2026-08-05 (seeded from the run-25 wrap's decision-changing finding; the wrap
itself asked for a standing rule — this is the mechanical half of it).
**Size: XS/S. Risk: low.** Surface: `fixtures/` + the suites' start-up checks.

## Why (measured, run 25)

`fixtures/trading_fixture.db` is GITIGNORED. Every committed baseline in the repo is promoted
against a specific fixture, and the ship ritual's central safety check is `git status --short` —
which **cannot see the fixture change at all**. In run 25 a review agent regenerated it mid-run
(`b0f8de8b` → `59525bd3`) while every committed baseline had been promoted against `b0f8de8b`;
the clean-tree gate stayed green throughout, and 869's banner had already warned that the next
regeneration re-bases several 865 expectations. It was restored from a reference corroborated
across the 869 run plus a copy taken 47 s before the regeneration — i.e. recovered by luck, not
by design. The instruction set had a hole: an agent told "never write `trading_data.db`" had no
rule about the fixture DERIVED from it.

This is the same class as the 864 incident (a baseline with no oracle of its own configuration)
and closes the same way: make the invisible thing loud.

## Scope

1. **Tracked identity pin**: `fixtures/FIXTURE_IDENTITY.json` (committed) records the fixture's
   `sha256`, byte size, row counts for the tables the suites read, the `make_fixture.py` git
   revision + its own sha256, and a human note naming which baseline generation it belongs to.
2. **Verify at suite start, fail loud**: the smoke/e2e suites that consume the fixture (861,
   863, 865, scorecard smoke, selftest where applicable) check the live fixture against the pin
   BEFORE doing work, and REFUSE with a named mismatch (expected vs actual sha256, plus "a
   fixture re-base is planner-scheduled work — see TASK_873") rather than running and quietly
   producing re-based numbers. Missing fixture ⇒ existing generate-path behavior, unchanged;
   missing/unreadable PIN ⇒ refuse (no fail-open on a deleted oracle — the 866 rule).
3. **Deliberate re-base path**: one explicit flag/target (`--promote-fixture-identity` or a
   documented `make_fixture.py` mode) that regenerates AND re-pins atomically, prints the full
   per-field diff, and refuses on a dirty tree — mirroring 864's promote gate. Regenerating
   without it must not silently succeed.
4. **`make_fixture.py` guard**: refuse to overwrite an existing fixture whose sha256 matches the
   pin unless the promote flag is passed (so an incidental invocation cannot re-base).

## Acceptance (gates)

- (i) Tamper test: mutate one byte / regenerate the fixture → every consuming suite REFUSES with
  the mismatch named; restore → all green. (This is the run-25 incident, replayed and now caught.)
- (ii) Promote path works: deliberate re-base updates fixture + pin atomically, prints the diff,
  refuses on a dirty tree.
- (iii) Deleted/corrupt pin ⇒ refuse (not fail-open); missing fixture ⇒ unchanged behavior.
- (iv) All existing suites green against the current fixture with the pin in place, and the
  pinned sha256 is stated in the wrap.

## Out of scope

Re-basing any baseline or the fixture itself (guard, don't move — if the current fixture no
longer matches what baselines were promoted against, that is a STOP and a planner finding);
`trading_data.db` (read-only source of record, unrelated); the plugin-level role guards the
run-25 wrap also recommended (upstream `apothecary`, operator-owned).
