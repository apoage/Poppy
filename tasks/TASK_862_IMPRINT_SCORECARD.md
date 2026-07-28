# TASK_862 — Imprint benchmark scorecard harness (battery codification; FILLER)

> **STATUS (2026-07-28): SHIPPED as `task862-imprint-scorecard`** (factory run 20).
> New files only — acceptance (i) zero diffs to any existing tracked file, re-confirmed after
> both fix rounds; selftest 12/12; smoke byte-identical on re-run (json + md) and under a
> hostile ambient env. 2-member panel over 2 fix rounds, **3 MAJOR + 1 latent-MAJOR closed**:
> any non-`WindowFailure` exception destroyed the whole artifact (contradicting this module's own
> "never silently skipped" contract) · the operator block was ASSERTED not OBSERVED (~25 knobs
> unpinned; a top-m battery would have emitted a status-OK card headed "848 read of record") ·
> the cross-config `meets_form` · and the recovery path re-running the call that had failed,
> plus an unguarded in-place write that could leave truncated unparseable JSON.
> **DECISION-CHANGING FINDING → DECISIONS D27 (OPEN)**: the pinned `runs/831_ext` floors this
> task's scope 1 mandates were measured on a materially different configuration (n_assets
> 95/204/214/273, psp 0.0018–0.0053, warm 31, 10-seed search) than the battery (64 slots, 0.1,
> warm 60, 1-seed policy-argmax). The factory shipped the DISCLOSURE and promoted the on-window
> teacher floor to PRIMARY — a deviation from scope 1 and from the ratified DEC-2 row that needs
> an operator ruling before the consistency read is cited.
> **Remainder → planner**: canonical manifest on `ckpt_imprint_full` (workstation, DB source of
> record). A GPU run still needs `safe_run` — `MEM_FRACTION` is not a cap.

**Status**: READY 2026-07-27 (seeded with D26 grooming — filler row, rank below TASK_861).
**Size: S. Risk: low** (one NEW file; zero edits to shipped modules; zero training).

## Why (D17 sequence — the frontrunner needs its full battery as one honest artifact)

The imprint (`runs/832_e5/ckpt_imprint_full`) is the validated frontrunner (+0.839 in-sample,
+0.898 OOS Δ+2.33, walk-forward consistent). Per the D17 correction, before the 4-gate ladder it
owes the FULL benchmark battery: the DEC-2 regime windows (bear22w / bull23w / y2024 / recent,
each with its own per-window floor from `runs/831_ext/*.json`), the full OOS window, and the
walk-forward quarters. Today each read is a hand-assembled `eval_vector.py` invocation and the
numbers live scattered across `runs/` JSONs. This task codifies the battery: one command, one
scorecard artifact, a stable schema the eventual gate-ladder submission can cite.

## Scope (one new file — INDEPENDENCE CONSTRAINT: must NOT edit `eval_vector.py`)

`experiments/835_stage4_eval/scorecard.py` — imports/invokes the shipped eval machinery
read-only (TASK_861 edits `eval_vector.py` concurrently; this row stays merge-safe by adding a
file only):

1. **Battery manifest**: a windows spec (JSON) listing named windows — the 4 DEC-2 regime
   windows with their `runs/831_ext` floor artifact paths, the OOS window
   (2025-10-05 → 2026-03-31), and the two walk-forward quarters per
   `runs/858_oos/WALKFORWARD_PREREG.md`. Ship the canonical manifest alongside the script.
2. **Runner**: for each window, run the policy-argmax replay operator with
   `--panel-frame-from-checkpoint` + on-window teacher floor + `--vector-weight-rebalance`
   (the 848 read-of-record configuration), tiers fee0 + fee21.
3. **Scorecard artifact** (JSON + MD): per window — net@21 both tiers, on-window floor, delta,
   decision count, activity block (859), bankruptcies; header — checkpoint path, frame
   provenance, db path, git describe. Deterministic across re-runs on the same inputs.
4. **Fail-loud**: refuse on frame/provenance mismatch or a missing floor artifact; NO silent
   window skips — a window that cannot run is reported as FAILED in the artifact and the exit
   code is non-zero.

## Acceptance (gates)

- (i) Zero diffs to any existing file (new files only); shipped selftests untouched and green.
- (ii) Fixture smoke: a 2-window toy manifest runs end-to-end on the fixture DB and emits a
  schema-valid scorecard; re-run is byte-identical.
- (iii) Fail-loud demonstrated: missing floor artifact and mismatched frame both produce a
  FAILED row + non-zero exit (no partial-success masquerade).
- (iv) Schema documented in the module docstring (the gate-ladder submission will cite it).

## Real leg (planner post-ship, pre-registered)

Run the canonical manifest on `ckpt_imprint_full` (workstation, DB source of record). Read of
record: per-window delta vs floor; the DEC-2 consistency form (≥3 of 4 regime windows at/above
their window floor) is the reporting frame — per D17, a pass confers NO gate status; it
completes the battery prerequisite.

## Out of scope

Any edit to `eval_vector.py` / `imprint.py` (TASK_861's surface); new metrics beyond what eval
JSONs already carry; paper-forward mechanics; any training.
