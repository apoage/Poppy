"""TASK_873 — pin + verify the identity of ``fixtures/trading_fixture.db``.

``trading_fixture.db`` is GITIGNORED, so the ship ritual's clean-tree gate (``git status``)
cannot see it change at all. Every committed baseline in the repo is promoted against a
SPECIFIC fixture; a silent regeneration re-bases every one of them while every gate stays
green (run-25 incident: ``b0f8de8b`` -> ``59525bd3``, recovered by luck, not by design). This
module makes that invisible thing loud: it computes the fixture's identity (sha256, size, row
counts of the tables the suites actually read) and checks it against the tracked pin
(``fixtures/FIXTURE_IDENTITY.json``) BEFORE a consuming suite does any work.

Same class as the 864/866 incident: a baseline (here, the fixture itself) must never be
guarded by no oracle, or by an oracle that can be deleted without consequence — a missing or
corrupt pin REFUSES (fail-closed), it never falls back to "no check happened" (the 866 rule).
A missing FIXTURE (not pin) is unchanged: existing generate-path behavior, decided by the
caller.

Two things this module records but deliberately does NOT enforce: ``make_fixture_git_rev`` and
``make_fixture_sha256`` are ADVISORY provenance only. They are excluded from
``_REQUIRED_PIN_FIELDS`` on purpose — enforcing them would refuse on any unrelated edit to
make_fixture.py (a comment fix would "invalidate" a fixture whose bytes never moved), which is
too brittle to be a gate. They answer "what cut this?", never "may this run?".

Import hazard (noted, not fixed): the five consuming suites reach this module as
``import fixtures.fixture_identity``, which resolves ``fixtures/`` as a PEP 420 namespace
package off the repo root. If the venv ever installs the PyPI distribution literally named
``fixtures``, that REGULAR package shadows the namespace one and all five imports break at
once (loud ImportError, not a silent skip). Not installed today; restructuring into a real
package is out of scope here.
"""
import hashlib
import json
import os
import sqlite3
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FIXTURE = os.path.join(_HERE, "trading_fixture.db")
DEFAULT_PIN = os.path.join(_HERE, "FIXTURE_IDENTITY.json")

# The tables the verification suites actually read (make_fixture.py's own copy list, plus the
# minute table). Order is fixed so a diff print is stable.
ROW_COUNT_TABLES = ("assets", "ohlcv_daily", "fetch_log", "ohlcv")

REBASE_HINT = "a fixture re-base is planner-scheduled work — see TASK_873."


class FixtureIdentityError(RuntimeError):
    """Live fixture doesn't match its pin, or the pin can't be read at all."""


def sha256_of(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def row_counts(db_path):
    """Row counts for ROW_COUNT_TABLES, read-only. A missing TABLE -> None (schema drift is
    itself a mismatch the caller will report, not a crash).

    A corrupt FILE is not schema drift and must not escape as a bare traceback: a truncated /
    bit-rotted / half-rsynced fixture (the ~5 Mbps uplink is a real source of these) raises
    ``sqlite3.DatabaseError`` ("file is not a database", "database disk image is malformed"),
    which is a SIBLING of OperationalError, not a subclass — so the old OperationalError-only
    catch let it propagate through compute_identity() -> verify() and straight past
    verify_or_die()'s ``except FixtureIdentityError``. Still fail-closed (the process died),
    but as a stack trace instead of the named refusal. Every sqlite failure other than "no such
    table" is now re-raised as FixtureIdentityError, naming the DB and the sqlite message.
    """
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error as e:
        raise FixtureIdentityError(
            f"fixture DB cannot be opened as a database: {db_path} ({type(e).__name__}: {e}) "
            f"— refusing. {REBASE_HINT}") from e
    try:
        counts = {}
        for table in ROW_COUNT_TABLES:
            try:
                counts[table] = conn.execute(f"select count(*) from {table}").fetchone()[0]
            except sqlite3.Error as e:
                if isinstance(e, sqlite3.OperationalError) and "no such table" in str(e).lower():
                    counts[table] = None          # schema drift -> reported as a mismatch
                    continue
                raise FixtureIdentityError(
                    f"fixture DB unreadable (corrupt/truncated?): {db_path} table {table!r} "
                    f"({type(e).__name__}: {e}) — refusing. {REBASE_HINT}") from e
        return counts
    finally:
        conn.close()


def compute_identity(db_path):
    """The measurable identity of a fixture DB at ``db_path`` — everything the pin records
    except provenance (make_fixture.py revision + note), which the caller attaches."""
    st = os.stat(db_path)
    return {
        "sha256": sha256_of(db_path),
        "size_bytes": st.st_size,
        "row_counts": row_counts(db_path),
    }


def make_fixture_provenance(make_fixture_path):
    """git revision (full hash, empty tree -> None) + sha256 of make_fixture.py itself."""
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", make_fixture_path],
            cwd=os.path.dirname(make_fixture_path) or ".",
            capture_output=True, text=True, check=False)
        rev = out.stdout.strip() or None
    except OSError:
        rev = None
    return {
        "make_fixture_git_rev": rev,
        "make_fixture_sha256": sha256_of(make_fixture_path),
    }


_REQUIRED_PIN_FIELDS = ("sha256", "size_bytes", "row_counts")


def load_pin(pin_path=DEFAULT_PIN):
    """Load + validate the tracked pin. Raises FixtureIdentityError (never returns a partial
    or default pin) if it is missing, unreadable, or missing a required field — no fail-open
    on a deleted oracle (the 866 rule)."""
    if not os.path.exists(pin_path):
        raise FixtureIdentityError(
            f"fixture identity pin missing: {pin_path} — refusing (no fail-open on a deleted "
            f"oracle). {REBASE_HINT}")
    try:
        with open(pin_path) as fh:
            pin = json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        raise FixtureIdentityError(
            f"fixture identity pin unreadable/corrupt: {pin_path} ({e}) — refusing. "
            f"{REBASE_HINT}") from e
    missing = [f for f in _REQUIRED_PIN_FIELDS if f not in pin]
    if missing:
        raise FixtureIdentityError(
            f"fixture identity pin {pin_path} is missing field(s) {missing} — refusing. "
            f"{REBASE_HINT}")
    return pin


def diff_identity(old, new):
    """Per-field diff between two identity dicts (as loaded from a pin, or compute_identity()
    output merged with provenance). Returns a list of human-readable "field: old -> new"
    strings for fields present in either side and differing."""
    lines = []
    keys = sorted(set(old) | set(new))
    for k in keys:
        ov, nv = old.get(k, "<absent>"), new.get(k, "<absent>")
        if ov != nv:
            lines.append(f"  {k}: {ov} -> {nv}")
    return lines


def verify(fixture_path=DEFAULT_FIXTURE, pin_path=DEFAULT_PIN):
    """Verify the live fixture at ``fixture_path`` matches the tracked pin at ``pin_path``.

    Returns "missing" if the fixture file itself doesn't exist — the caller's existing
    generate-path behavior is unchanged, this function does no further work in that case.
    Returns "ok" if it exists and matches. Raises FixtureIdentityError otherwise (missing/
    corrupt pin, or any field mismatch — named expected vs actual).
    """
    if not os.path.exists(fixture_path):
        return "missing"
    pin = load_pin(pin_path)          # raises loud on missing/corrupt pin
    live = compute_identity(fixture_path)
    mismatches = []
    for field in _REQUIRED_PIN_FIELDS:
        if live.get(field) != pin.get(field):
            mismatches.append(f"{field}: expected {pin.get(field)!r}, got {live.get(field)!r}")
    if mismatches:
        raise FixtureIdentityError(
            "fixture identity MISMATCH vs " + pin_path + ":\n  "
            + "\n  ".join(mismatches)
            + f"\n{REBASE_HINT}")
    return "ok"


def verify_or_die(fixture_path=DEFAULT_FIXTURE, pin_path=DEFAULT_PIN, label="[fixture-identity]"):
    """Convenience for a suite's main(): call BEFORE any real work. Turns a mismatch/missing-pin
    into a loud SystemExit; a missing fixture prints a note and returns "missing" so the
    caller's own existing missing-fixture handling still runs (unchanged behavior)."""
    try:
        status = verify(fixture_path, pin_path)
    except FixtureIdentityError as e:
        raise SystemExit(f"{label} REFUSED:\n{e}")
    if status == "missing":
        print(f"{label} fixture DB missing: {fixture_path} "
              f"(regenerate: fixtures/make_fixture.py)")
    else:
        print(f"{label} fixture matches pin ({pin_path}): sha256 "
              f"{load_pin(pin_path)['sha256'][:12]}...")
    return status


if __name__ == "__main__":
    import sys
    fx = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FIXTURE
    pn = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_PIN
    try:
        result = verify(fx, pn)
    except FixtureIdentityError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    print(result)
    sys.exit(0)
