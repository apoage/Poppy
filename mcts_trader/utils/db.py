"""Shared SQLite read-only connection helper.

TASK_865/868 shipped ``_ro_connect`` on the experiments surface (11 sites — originally
defined in ``experiments/835_stage4_eval/eval_vector.py``) to close a measured hazard: a
plain READ-WRITE ``sqlite3.connect`` on a db carrying a stale hot journal plays that journal
back (and unlinks it) on the FIRST read — no write SQL involved. Measured on a copy of
``trading_data.db`` (24 GB, protected, the only backup predates the 858 backfill): the
copy's sha256 AND size changed after nothing but a SELECT through a read-write handle, while
a ``mode=ro`` handle left it byte-identical. A read-write handle also silently CREATES a
0-byte db at a typo'd/symlinked path instead of failing loud.

TASK_870 ports the fix to the CANON panel loader (``mcts_trader/env/vectorbt_env.py``),
which reached the same 24 GB file through a plain read-write handle on the default (unpinned)
eval/training path. It lives here rather than being imported from
``experiments/835_stage4_eval/eval_vector.py`` because that module imports
``mcts_trader.training.env_factory`` -> ``mcts_trader.env.vectorbt_env``, so the canon env
loader importing back from it would be circular. ``eval_vector.py`` now delegates to this
one implementation instead of carrying its own copy (``experiments/837_walkforward/
walkforward.py`` has its own, independently-defined, weaker ``_ro_connect`` — pre-existing,
out of TASK_870's scope, left alone here).

OPERATOR NOTE — this changes what a hot journal DOES on the canon path (measured, TASK_870
verification). Before: a training/eval run whose db carried a stale ``-journal`` sidecar
silently rolled it back and proceeded. Now SQLite refuses to serve reads it cannot recover
under a read-only handle, so the run fails loud with
``sqlite3.OperationalError: attempt to write a readonly database`` and the db is left
byte-identical. That is the intended trade: on a 24 GB read-only source of record whose only
backup predates the 858 backfill, a stopped run is strictly cheaper than a silent rewrite. If
you hit it, the journal is a real signal — an earlier writer died mid-transaction. Recover it
deliberately through the sanctioned write path (``tools/backfill_minute.py``, which still
opens read-write by design) or a deliberate read-write session, after taking a backup; do NOT
"fix" it by reverting a call site here to a plain ``sqlite3.connect``.
"""
import os
import sqlite3

# Python's sqlite3 default busy timeout is 5s; the repo documents 60s (this was originally
# trader/data.py:29's SQLITE_TIMEOUT_S) so a transient lock from the fetcher
# (tools/fetch_daily_kraken.py writes to the same file) cannot kill a long-running read.
SQLITE_TIMEOUT_S = 60.0


def _ro_connect(db_path, **connect_kwargs):
    """Open ``db_path`` STRICTLY read-only (``file:...?mode=ro``, repo's 60s busy timeout).

    Refuses a path that does not exist (rather than creating a 0-byte db, which a read-write
    handle would do), naming the resolved target so a symlink cannot hide what was opened.
    Extra keyword args (e.g. ``check_same_thread=False``) are forwarded to
    ``sqlite3.connect``; ``timeout`` defaults to ``SQLITE_TIMEOUT_S`` but callers may override
    it."""
    p = os.path.abspath(db_path)
    rp = os.path.realpath(p)
    if not os.path.isfile(rp):
        raise RuntimeError(
            f"--db {db_path}: no such SQLite file (resolved: {rp}). Refusing to open it — a "
            f"read-write handle would CREATE a 0-byte database here and then fail with a "
            f"misleading 'no such table'. Point --db at an existing database.")
    kwargs = {"uri": True, "timeout": SQLITE_TIMEOUT_S}
    kwargs.update(connect_kwargs)
    return sqlite3.connect(f"file:{p}?mode=ro", **kwargs)
