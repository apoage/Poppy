> **STATUS (2026-08-27, factory run 45): SCOPE 1-3 + GATES (i)+(ii) SHIPPED. GATE (iii) IS PLANNER WORK.**
>
> `config/gitignored_pins.json` (6 pins) · `tools/verify_gitignored_pins.py` (verify + promote +
> `__main__`) · `experiments/897_corrected_809/conftest.py` (a `pytest_sessionstart` gate that aborts
> the session before collection on divergence).
>
> **Gate (iii) entry point, for the planner to wire in one line:**
> `python -m tools.verify_gitignored_pins` → exit **0** all OK · **1** DIVERGED/MISSING · **2** UNPINNED.
> Programmatic: `from tools.verify_gitignored_pins import verify_all, refuse_on_divergence`
> (`verify_all()` never raises per-file; `refuse_on_divergence(results)` raises on DIVERGED/MISSING).
> Gate (iii) is excluded because it requires writing under `.dark-factory/`, which `lanes.planner_owns`
> and the `lane_guard` hook correctly deny to the implementer lane — see the LANE NOTE below.
>
> **FINDING, unresolved → D71.** `runs/875_wf/orch_wf875.sh` is absent from the workstation AND from
> `gpu-server` (orchestrator checked both), and **no sha256 of it was ever recorded anywhere** —
> `lore/notes_2026-08-19.md:200-203` says it was "sha256-verified identical" on 2026-08-19 but never
> wrote the value down. The entry is left `sha256: null` / MISSING and was **NOT guessed**. So
> `python -m tools.verify_gitignored_pins` exits **1** on this host until D71 is ruled — correct
> fail-closed behaviour per "missing must be as loud as diverged", not a defect. **This is the D69
> mechanism working on its first day**, and it is the second confirmed load-bearing gitignored file to
> go missing after the 2026-08-25 restore sweep.
>
> **Panel (correctness): 1 MAJOR + 3 MINOR, all closed.** The MAJOR was in the escape hatch:
> `--promote-pin` accepted a re-pin with no `--provenance`/`--why`, writing nulls that then passed
> `load_manifest`'s **key-presence** check forever after — the one path built to preserve an audit
> trail could silently discard it. Now refused on WRITE (before hashing and before the manifest is
> opened, so a refusal leaves it byte-identical) **and** caught on READ (truthiness, not key presence),
> with `sha256: null` deliberately exempt because it is a documented load-bearing value. MINORs:
> `OSError` on a directory/unreadable file raised uncaught (would surface as a pytest `INTERNALERROR`
> instead of the clean refusal — now a structured result); manifest paths had no containment check
> (now `realpath` on both sides, matching TASK_905's `_require_within_fold_root` shipped the same
> night); and `--noconftest` bypasses the gate — **documented, not fought**, since it is structural to
> any conftest-hosted hook and shared with the `FIXTURE_IDENTITY` precedent this row follows.
>
> **What the panel CLEARED, under direct attack:** the startup gate fires identically under
> full-suite, single-file, `-k`, `--collect-only`, `-p no:cacheprovider`, and a foreign working
> directory — `pytest.exit`, exit 2, before any test ran. And the null-hash entry does **not** bless
> later-appearing content: arbitrary content at that path reports `UNPINNED`, never `OK`. An
> independent repo-wide sweep found no manifest gaps, confirming the judgement call to add
> `runs/808_linear_backtest/backtest.py` beyond the task's literal three-file list.
>
> **Verification:** 897 suite **253 → 256 passed**; module suite **30 → 54 passed**; selftest 12/12.
> Gate (i) mutation-proven (June-stale bytes → `pytest.exit`, exit 2, before collection; restore →
> green). Integrity held throughout: `recompute_best.py` `783b2ea3…` and `runs/875_wf` (md5
> `e87b080d…`, 269 files) unchanged. Fix pass at opus.
>
> **Remainder → planner:** gate (iii) wiring (after D71, or scoped with `--path`); and
> `promotion_evidence` is required at write time but not truthiness-checked on read, because
> back-filling the four seeded pins would be a manifest edit the implementer deliberately did not make.

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

---

## LANE NOTE (2026-08-27, factory run 45 orchestrator — not a scope change)

**Gate (iii) is not completable by the implementer lane while the factory is armed.** It requires
the manifest to be checked by "the factory's own startup gate", which lives under
`.dark-factory/` — `lanes.planner_owns` in `.dark-factory/config.json`. The `lane_guard`
PreToolUse hook denies implementer writes there, correctly and in both directions.

So this row splits cleanly and should be taken that way:

- **(i) and (ii) are implementer work** — the tracked manifest, the suite startup check, the
  promote path, and the mutation proof. That is the whole mechanism and it is where the value is.
- **(iii) is a planner action**: wiring the manifest into the factory startup ritual. The
  implementer should deliver the check as a callable entry point (a module function plus a
  `python -m ...` invocation returning a non-zero exit on divergence) and NAME it in the STATUS
  banner, so the planner can wire it in one line without re-deriving the interface.

Taking the row without this split produces either a hook denial mid-run or, worse, a green
verdict for a row whose third gate was quietly skipped.
