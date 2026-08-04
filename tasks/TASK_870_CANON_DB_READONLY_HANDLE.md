> **STATUS (2026-08-04): SHIPPED as `task870-canon-db-readonly-handle`** — canon panel loader
> (`vectorbt_env.py`, both `_load_database_prices*`) + 6 sibling sites routed through one shared
> `_ro_connect` (`mcts_trader/utils/db.py`); `eval_vector.py` delegates to it instead of its own
> copy. All three acceptance gates PASS first-hand: (i) hot-journal probe byte-identical with the
> journal intact, and **CONTROL B showed the real pre-870 code path silently rewriting the file on
> a bare SELECT** — the gate is not vacuous; (ii) canon numerics bit-identical, proven three times
> against independent `git worktree`s at HEAD; (iii) `backfill_minute.py` still writes (17/17 + an
> independent row-count probe). Medium panel, **zero BLOCKER/MAJOR, one cycle, no escalation**.
> Selftest 12/12 · 863 smoke 13/13 (254.6s).
>
> **Remainders (planner):** `mcts_trader/data/database.py` is a real live-write path (aiosqlite,
> needs a dual-connection refactor — beyond XS, deliberately untouched) and is now the only known
> unaddressed RW reach at the canon layer. Three `_ro_connect`-style helpers still coexist
> (shared / `837_walkforward` / `trader/data.py::_connect`) — "one implementation" holds for the
> touched surface, not repo-wide. `trader/backtest.py:build_aligned_ohlcv` has zero live callers,
> so its new call has no automated coverage (hand-verified). DB backup remains an ops item: the
> only backup still predates the 858 backfill.
>
> **Operator note:** a hot journal now STOPS a canon run (fail-loud, db untouched) instead of
> being silently rolled back — intended; see the docstring in `mcts_trader/utils/db.py` for how to
> recover it deliberately.

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
