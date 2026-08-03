# TASK_870 — Canon panel path opens `trading_data.db` READ-WRITE (run-23 rec (D)) — close the last RW handle

**Status**: READY 2026-08-03 (seeded from the run-23/24 wraps; planner Q2 answered YES by the
operator flow). **Size: XS/S. Risk: low** (fail-closed change; zero numerics risk).
Surface: `mcts_trader/env/vectorbt_env.py` (+ any sibling canon connect sites found in scope 2).

## Why (measured in run 23, still open after the wave)

TASK_865's BLOCKER (1): a read-write SQLite handle on the 24 GB source of record lets
hot-journal recovery REWRITE the file on a pure SELECT — measured: a copy's sha256 and size
changed with no write SQL involved. 865/868 fixed all 11 sites on the experiments surface via
`_ro_connect` (`file:...?mode=ro`, timeout 60, realpath, refuses a nonexistent path instead of
creating a 0-byte db). **The CANON panel path still opens RW: `vectorbt_env.py:515` and
`:731`.** Risk is amplified by an ops fact the run-24 wrap flags: the only backup of the
source of record predates the 858 backfill — an accidental rewrite today is not fully
recoverable.

## Scope

1. Port the shipped `_ro_connect` pattern (or import it from where 865 landed it — one
   implementation, not a copy) to `vectorbt_env.py:515`/`:731` and every other canon-path
   `sqlite3.connect` on `trading_data.db`.
2. One read-only sweep for sibling RW connect sites across `mcts_trader/` + top-level entry
   points (`train_jax_fast.py`, `real_trade_jax.py`, `trader/`); fix XS ones, report the rest.
   The single sanctioned WRITE path (`tools/backfill_minute.py`) is explicitly OUT of scope and
   must keep working.
3. The 865 probe test ported: opening the canon path against a hot-journal fixture copy leaves
   it byte-identical; a write attempt raises `attempt to write a readonly database`.

## Acceptance (gates)

- (i) Hot-journal probe: canon loader on a journal-carrying fixture copy → file byte-identical
  (sha256 before/after shown); write probe refused.
- (ii) All suites green, canon numerics bit-identical (this is a handle-mode change only).
- (iii) `tools/backfill_minute.py` still writes (its own smoke or a fixture append proves it).

## Out of scope

Backup strategy for the DB (operator/ops — flagged separately); the experiments surface
(already RO per 865/868); any loader semantics.
