# TASK_922 — PROTECTED.txt self-declaration convention (881E's forward item)

**Status**: PROPOSED 2026-09-06 (planner; from run 56's 881E ship — zero run dirs eligible
for deletion is D72's pre-registered OUTCOME, not a defect; the forward item is keeping that
property cheap as new runs appear).
**Size XS. Risk: LOW — one small writer hook + one reader clause; the deleter is under the
standing `--apply` freeze (D34/D35) which this row does NOT touch.**

## What

New-run convention: every run dir writes its own `PROTECTED.txt` at creation (name, task,
date, one-line reason), and the 881E deleter reads it as `covers:`-grade authority —
self-declared protection instead of grandparented manifests. As runs age, the zero-eligible
property then holds by construction rather than by 881i-era manifests that eventually rot.

## Gates

- **G1**: a run dir with `PROTECTED.txt` is refused by the deleter's dry-run; without it,
  normal `covers:` rules apply. Demonstrated both ways (the D83(i) lens: plant a dir whose
  PROTECTED.txt is malformed — the reader must fail LOUD, not silently treat as unprotected).
- **G2**: existing run dirs untouched (their protection rides the shipped 881E manifests);
  no `--apply` semantics change; freeze untouched.
- **G3**: the writer hook is one call in the run-dir creation path — no behavior change to
  any existing run (flag-off default where a hook cannot be inert).
