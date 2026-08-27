# TASK_911 — identity pins for load-bearing gitignored artifacts (D69)

**Status**: READY 2026-08-27 (planner; seeded by D69's ruling, option (a) — after a verified-
deployed, gitignored repair (`recompute_best.py`) silently regressed to its June revision on
this workstation, and the only in-repo detector was a red test the factory nearly deleted).
**Size: S. Risk: LOW — additive checking; writes only a tracked manifest.**

## Why this row exists

`runs/` is gitignored by design, and some files under it (or beside it) gate suites and
launches: the 809 lineage (`probe.py` lineage copy, `decide.py`, `recompute_best.py`),
`runs/875_wf/grade_correct.py` + `orch_wf875.sh`, and their kin. Git cannot protect them, and a
backup restore/sync with an mtime-preserving copy can silently write stale content over a
deployed repair — measured 2026-08-25 (D69's forensics). The repo already owns this pattern
twice (`fixtures/FIXTURE_IDENTITY.json`, TASK_894's `frame_symbols` sha baked into
checkpoints); this row applies the existing mechanism, it does not invent one.

## Scope

1. **A small tracked manifest** (e.g. `config/gitignored_pins.json`) recording
   `{path, sha256, provenance, why-load-bearing}` for each gitignored file any suite or launch
   depends on. Seeded with the D69 casualty list: the 809 lineage's three files (with the
   repaired `recompute_best.py`'s sha256 `783b2ea3…`), `runs/875_wf/grade_correct.py`,
   `runs/875_wf/orch_wf875.sh`. Discovery rule for what belongs: a file is in the manifest iff
   a suite, gate, or launch note names it.
2. **A startup check** in the relevant suites (the 897 suite first): verify each pinned file's
   sha256, REFUSE LOUD on divergence (naming expected vs actual), and skip-with-silence
   nothing.
3. **A promote path** (`--promote-pin <path>` style, deliberate-only, like the fixture pin's)
   for when a gitignored file is legitimately re-derived — with the re-derivation evidence
   named in the commit.

## Acceptance (gates)

- (i) Mutation proof: revert `recompute_best.py` to its June bytes → the 897 suite refuses at
  startup, loudly; restore → green.
- (ii) The manifest's seeded entries verify GREEN against both hosts' current copies (or any
  divergence is surfaced as a finding, never silently re-pinned).
- (iii) The manifest is checked by the factory's own startup gate (the clean-tree/checksum
  ritual), so a divergence fails every lane, not just one suite.

## Out of scope

Tracking `runs/` content in git (D69's (b) — the operator's policy call, not taken). Re-running
any measurement.
