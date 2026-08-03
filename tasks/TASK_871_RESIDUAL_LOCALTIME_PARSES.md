# TASK_871 — Residual local-time parses off the read-of-record surface (869 remainder) — FILLER

**Status**: READY 2026-08-03 (seeded from the 869 banner remainder). **Size: XS/S. Risk: low.**
Surface: `diff_teacher/`, `tools/` (NOT the read-of-record operators — 869 already fixed those).

## Why

TASK_869 made every window resolution on the read-of-record surface UTC (byte-proven
TZ-invariant, 4 TZs + DST edge). Its banner names surviving LOCAL-TIME parses off that surface
(`diff_teacher/`, `tools/`): not result-bearing today, but each is a latent 308-row-class
machine dependency waiting for a future task to promote it into a measurement path.

## Scope

1. Sweep `diff_teacher/` + `tools/` for date-string → epoch conversions that resolve in local
   time (the `_to_epoch` pattern 869 replaced); convert to the shared UTC helper where XS,
   report-only where larger or where local time is semantically intended (name the intent).
2. No numeric artifact on the read-of-record surface may move — if one does, STOP (that would
   mean the site was result-bearing after all, which is a finding).

## Acceptance

- (i) Sweep report (sites found / converted / intentionally-local, one line each).
- (ii) TZ probe on converted sites (same call, two TZs, identical output).
- (iii) All suites green; read-of-record artifacts byte-identical.

## Out of scope

The read-of-record surface (done, 869); window semantics beyond timezone.
