# TASK_908 — the 904A destination fence's pin-trust dependency: verify the pin against the fixture

**Status**: READY 2026-08-27 (planner; seeded from run-43's open item 3).
**Size: S. Risk: MEDIUM** — small mechanism, but it sits under a write-authorization guarantee.

## Why this row exists

904A's destination-identity fence guarantees "a renamed byte-identical copy cannot author this
tree" — and that guarantee rests entirely on `fixtures/FIXTURE_IDENTITY.json` being CORRECT.
With a stale or corrupt sha256 there, a renamed fixture copy classifies as the real tree and IS
authorized to write. Run 43 closed the honesty half (the dependency is disclosed in the design
note and SPEC.md). The mechanism half is open — and noted: `walkforward.py` never calls
`fixtures/fixture_identity.py::verify_or_die`, unlike five other experiment smoke scripts.

## Scope

1. **Cross-verify the pin against a locally-present fixture when one exists.** At the point the
   fence trusts `FIXTURE_IDENTITY.json`, if the pinned fixture file is present locally, verify
   the live hash matches the pin (and refuse the WRITE path on mismatch — stale pin must fail
   closed toward refusal, never toward authorization).
2. **Wire `verify_or_die` into `walkforward.py`'s startup** the way the five other experiment
   smoke scripts already do, or state exactly why this one differs.
3. **The pin's own freshness story gets a test:** a tampered/corrupt pin file makes the fence
   refuse (demonstrated), and a rotated-after-incident pin is re-derivable through the existing
   `--promote-fixture-identity` path only.

## Acceptance (gates)

- (i) A renamed byte-identical fixture copy with a STALE pin is refused (the D64-class repro
  for this row).
- (ii) `walkforward.py` verifies fixture identity at startup or documents the exception.
- (iii) 904A's gates stay green; `folds/real` tree hash `1b31624a…` unmoved.

## Out of scope

Changing the pin's canonical content absent a demonstrated rotation need. The 904B writer
redesign (two-phase), which is its own row.
