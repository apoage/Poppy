> **STATUS (2026-08-04): SHIPPED as `task871-residual-localtime-parses`** — `diff_teacher/data_loader.py::_parse_dates`
> and `tools/map_training_data.py::load_benchmark_periods` resolve `YYYY-MM-DD` at UTC midnight
> via the canonical `trader.backtest._to_epoch`; the same file's two read-only SQLite handles moved
> onto TASK_870's `_ro_connect`. Medium panel, **zero BLOCKER/MAJOR, one cycle, no escalation**.
> Gate (ii): byte-identical across UTC/Prague/Kathmandu/Chatham (DST interior + non-hour offset)
> **against a HEAD control that differs in all four** — not a vacuous probe. Gate (iii): selftest
> 12/12; the STOP condition is NOT triggered, established by consumer trace (`benchmark_easy.py`
> parses the same config via its own independent path; nothing read-of-record imports
> `diff_teacher.data_loader`; `training_schedule.json` is unpinned/non-canon). **Gate (i)'s sweep
> report — every site found / converted / reported-only / intentionally-local — is recorded in full
> in the CHANGELOG entry**, which is this repo's durable home for it.
>
> **Disclosed cost:** importing `trader.backtest` for the date parse now transitively pulls
> `mcts_trader.utils.db` (via TASK_870, shipped hours earlier) and scipy — zero → three
> `mcts_trader` modules, 0.49s → 0.71s. The implementer's comment claiming otherwise was measured
> false and replaced with an explicit disclosed-cost block. No new JAX (already in the graph).
>
> **Remainders (planner), highest first:** `mcts_trader/data/stream.py:157,162` — same defect class
> but on the LIVE WebSocket candle path (strips a confirmed `Z`, then `.timestamp()` on a naive
> string), so **result-bearing today, not latent**; fix looks XS; left alone as outside this task's
> declared surface — recommend its own row. Then: `tools/orchestration/train_manager.py:321,330`
> (same class, untouched because `tools/orchestration/CLAUDE.md:19` marks it "Deprecated — DO NOT
> EDIT"); a neutral home for the canonical date parse if `diff_teacher`'s documented independence
> from `mcts_trader` is meant strictly; and the `fromisoformat` → strict `%Y-%m-%d` narrowing in
> `load_benchmark_periods` (a full-ISO config value now raises uncaught, unreachable today).

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
