#!/usr/bin/env python
"""Regenerate fixtures/trading_fixture.db from trading_data.db (read-only source).

Contents: full `assets` + `ohlcv_daily` panel + `fetch_log`, plus a 90-day 1-min
`ohlcv` slice for the 8 assets with the longest minute coverage. ~50MB, instant-load.
This is the dark-factory verification fixture (`.dark-factory/config.json paths.fixture`).
The fixture db itself is gitignored (*.db rule) — regenerate with this script.

TASK_868 (D31 scope 4): three SYNTHETIC minute symbols are appended after the copy — the
twin-identity pair TWX/USD (90001) + TWXUSD (90002) with an orphaned span, and HOLE/USD
(90003) with a multi-bar interior hole — all on 2025-09-10..2025-09-12, i.e. BEFORE the
copied slice and outside every smoke window, so no existing panel or universe ranking moves.

TASK_873: a bare invocation used to unconditionally remove+rebuild the fixture — the exact
mechanism of the run-25 incident (an agent regenerated it mid-run, silently re-basing every
baseline promoted against the old one, while the gitignored fixture kept `git status` green
throughout). Now: a bare invocation REFUSES to overwrite an EXISTING fixture, full stop. The
guard is a refusal guard, not a normalizer — it never repairs or re-pins anything, and it
refuses whenever it cannot positively establish that overwriting is safe: pin matches, pin
mismatches, pin missing, pin corrupt all end in the same refusal with the reason named. Only
two paths reach a destructive rebuild: the fixture does not exist yet (fresh clone — nothing to
destroy), or the caller explicitly passed `--promote-fixture-identity`. That flag requires a
clean git tree, regenerates + re-pins (build to temp paths; `.db` and pin `os.replace()`d only
after the build succeeds — see the guarantee note on those calls), and prints the full
per-field diff between the old and new pin.
"""
import argparse
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import fixtures.fixture_identity as fixture_identity  # noqa: E402

DST = os.path.join(_HERE, "trading_fixture.db")
PIN = fixture_identity.DEFAULT_PIN
DAYS = 90
TOP_N = 8
# TASK_868: the slice window is PINNED, not max-relative. trading_data.db keeps accruing
# minute rows (its max had already moved ~3.5 months past 2025-12-12 by 2026-08), so the old
# `max(timestamp) - DAYS*86400` cutoff silently re-based every fixture panel on each regen.
# The verification suites pin their windows inside 2025-09-13..2025-12-12; re-pin the slice
# deliberately here, never implicitly.
SLICE_END = "2025-12-12"            # inclusive UTC-calendar end day of the minute slice
# TASK_869: UTC, not local — a regen on any machine resolves the same bounds.

# TASK_868 phenomena anchor, UTC (TASK_869: was local-time; see the REGEN TRAP note below).
_T868 = int(datetime.strptime("2025-09-10", "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def _t868_rows(aid, minutes, price):
    return [(aid, _T868 + 60 * m, price, price, price, price, 1.0, 1) for m in minutes]


def _insert_synthetic_rows(dst):
    """TASK_868 (D31 scope 4): synthetic identity/hole phenomena, minute table only. Both
    mechanisms the task repairs must be fixture-provable forever. All synthetic rows land
    BEFORE the copied slice (2025-09-10..2025-09-12), so every smoke window's coverage-ranked
    universe and every existing panel is untouched (zero synthetic rows in any smoke window).
    The ohlcv_daily table is NOT touched (daily paths stay bit-identical).

    *** REGEN TRAP (TASK_869, run-25 panel NIT) *** The COMMITTED fixture's synthetic rows are
    anchored at the LOCAL-TIME epoch the fixture was cut with (CEST); _T868 above now anchors
    UTC, and SLICE_END likewise. A REGEN therefore shifts every synthetic row (and the slice
    bounds) by the old local offset, which SILENTLY re-bases several 865 smoke expectations
    (the T868-window twin per-id counts {90001: 1320, 90002: 2160}, the window-aware hole-bar
    derivation, and anything measured off the synthetic anchor) — plus the minute-slice panels
    themselves. After ANY regen, re-derive those constants: the full list of what moves and why
    is in runs/869_rebase_wave_semantics/baseline_attribution.json. (TASK_873: this is exactly
    the class of silent re-base the identity pin now catches loud, at every consuming suite's
    startup, instead of relying on someone reading this comment.)
    """
    # (i) a TWIN-IDENTITY pair with an orphaned span (the D31 ingest-split pattern):
    # original prints minutes 0..1439 then goes dark; the twin (spelling variant, added
    # "later") prints minutes 720..2879 — agreeing in the 720..1439 overlap, and carrying the
    # ORPHANED span 1440..2879 the original lacks. The loader's alias union must read one
    # continuous 2880-minute series; the agreement gate must pass (and refuse a tampered
    # overlap).
    twin_a_id, twin_b_id = 90001, 90002
    dst.execute("insert into assets (id, symbol) values (?, ?)", (twin_a_id, "TWX/USD"))
    dst.execute("insert into assets (id, symbol) values (?, ?)", (twin_b_id, "TWXUSD"))
    dst.executemany("insert into ohlcv values (?,?,?,?,?,?,?,?)",
                    _t868_rows(twin_a_id, range(0, 1440), 100.0))
    dst.executemany("insert into ohlcv values (?,?,?,?,?,?,?,?)",
                    _t868_rows(twin_b_id, range(720, 2880), 100.0))

    # (ii) a genuine MULTI-BAR INTERIOR HOLE (run-23 rec (C)): prints every minute of the same
    # two days EXCEPT minutes 1000..1599 (a 600-minute hole), re-anchoring +60% on resumption —
    # the cell class D31's causal open-gate mask force-HOLDs.
    hole_id = 90003
    dst.execute("insert into assets (id, symbol) values (?, ?)", (hole_id, "HOLE/USD"))
    dst.executemany("insert into ohlcv values (?,?,?,?,?,?,?,?)",
                    _t868_rows(hole_id, range(0, 1000), 100.0))
    dst.executemany("insert into ohlcv values (?,?,?,?,?,?,?,?)",
                    _t868_rows(hole_id, range(1600, 2880), 160.0))


def build_fixture(src_path, dst_path, days=DAYS, top_n=TOP_N, slice_end=SLICE_END):
    """Build one fixture DB at ``dst_path`` from ``src_path`` (read-only). Overwrites
    ``dst_path`` if present. Returns (top_asset_ids, minute_row_count, daily_row_count).
    Pure generation logic, unchanged from the pre-873 script — only factored into a function
    so both the default and --promote-fixture-identity paths (and tests, on scratch paths)
    call the exact same code."""
    if os.path.exists(dst_path):
        os.remove(dst_path)
    src = sqlite3.connect(f"file:{src_path}?mode=ro", uri=True)

    top = [r[0] for r in src.execute(
        "select asset_id from fetch_log where data_type='ohlcv_1' "
        "order by (newest_timestamp-oldest_timestamp) desc limit ?", (top_n,))]
    if not top:
        top = [r[0] for r in src.execute(
            "select asset_id from ohlcv_daily group by asset_id order by count(*) desc limit ?",
            (top_n,))]
    end_ts = (int(datetime.strptime(slice_end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                  .timestamp()) + 86399)
    cutoff = end_ts - days * 86400 + 1

    dst = sqlite3.connect(dst_path)
    for (sql,) in src.execute(
            "select sql from sqlite_master where type='table' and sql is not null"):
        dst.execute(sql)

    for t in ("assets", "ohlcv_daily", "fetch_log"):
        ncols = len(src.execute(f"pragma table_info({t})").fetchall())
        dst.executemany(f"insert into {t} values ({','.join('?' * ncols)})",
                        src.execute(f"select * from {t}"))

    cur = src.execute(
        f"select * from ohlcv where timestamp>=? and timestamp<=? "
        f"and asset_id in ({','.join('?' * len(top))})",
        [cutoff, end_ts] + top)
    ncols = len(cur.description)
    dst.executemany(f"insert into ohlcv values ({','.join('?' * ncols)})", cur)

    _insert_synthetic_rows(dst)

    dst.commit()
    n_min = dst.execute("select count(*) from ohlcv").fetchone()[0]
    n_day = dst.execute("select count(*) from ohlcv_daily").fetchone()[0]
    dst.execute("vacuum")
    dst.close()
    src.close()
    return top, n_min, n_day


def _git_dirty(repo_root=_REPO_ROOT):
    """True if the working tree has any uncommitted changes. Mirrors the 864/861 promote
    gate's dirty-tree refusal. (The fixture itself is gitignored and never shows up here —
    that gap is exactly what the pin file, which IS tracked, exists to close.)"""
    out = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root,
                          capture_output=True, text=True, check=False)
    return bool(out.stdout.strip())


def _build_pin(dst_path, top, note):
    """Assemble a pin: enforced identity + ADVISORY provenance. `advisory_fields` is recorded
    explicitly so nobody later reads make_fixture_git_rev/make_fixture_sha256 as a gate — they
    are recorded, never compared (fixture_identity._REQUIRED_PIN_FIELDS excludes them). Making
    them enforced would refuse on any unrelated edit to this file, including a comment."""
    identity = fixture_identity.compute_identity(dst_path)
    provenance = fixture_identity.make_fixture_provenance(os.path.abspath(__file__))
    return {**identity, **provenance,
            "advisory_fields": ["make_fixture_git_rev", "make_fixture_sha256"],
            "note": note}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", nargs="?", default="trading_data.db",
                    help="source DB (default: trading_data.db, opened read-only)")
    ap.add_argument("--promote-fixture-identity", action="store_true", dest="promote",
                    help="TASK_873: deliberately re-base the fixture AND its tracked pin. "
                         "Requires a clean git tree; builds to temp paths and only then "
                         "os.replace()s each into place (each replace atomic, the pair not — "
                         "the one visible interleaving fails closed); prints the full "
                         "per-field diff between the old and new pin. Without this flag an "
                         "EXISTING fixture is never overwritten — matching, mismatching, or "
                         "unpinned, the run refuses — so an incidental invocation cannot "
                         "silently re-base every baseline promoted against it.")
    args = ap.parse_args()

    # ---- write guard: a REFUSAL guard, not a normalizer -------------------------------------
    # It never repairs, re-pins or "resyncs" anything. If DST exists and the caller did not ask
    # to promote, this exits non-zero — whatever the pin says. Assuming the worst is the point:
    # the earlier version only refused on a CLEAN match and fell through to build_fixture()'s
    # unconditional os.remove() when the pin was missing, corrupt, or already mismatched. That
    # inverted the 866 rule on the one side where the damage is irreversible, and it made the
    # post-incident recovery state (fixture != pin) rebuild silently on every bare re-run,
    # forever. Only two paths may reach a destructive rebuild: (a) DST does not exist — fresh
    # clone, nothing to destroy, behavior unchanged; (b) --promote-fixture-identity was passed.
    if os.path.exists(DST) and not args.promote:
        try:
            pin = fixture_identity.load_pin(PIN)
        except fixture_identity.FixtureIdentityError as e:
            reason = (f"its tracked pin cannot be read, so overwriting cannot be shown to be "
                      f"safe ({e})")
        else:
            try:
                live_sha = fixture_identity.sha256_of(DST)
            except OSError as e:
                reason = f"the existing fixture itself cannot be read ({e})"
            else:
                if live_sha == pin.get("sha256"):
                    reason = (f"it already matches the tracked pin ({PIN}) — an incidental "
                              f"invocation cannot re-base a fixture every committed baseline "
                              f"was promoted against (the run-25 incident)")
                else:
                    reason = (f"it MISMATCHES the tracked pin ({PIN}): pin sha256 "
                              f"{pin.get('sha256')}, on-disk sha256 {live_sha}. This is the "
                              f"post-incident state — overwriting would destroy the only "
                              f"evidence of what the baselines actually ran against")
        raise SystemExit(
            f"[make_fixture] REFUSED to overwrite {DST}: {reason}. Pass "
            f"--promote-fixture-identity to deliberately re-base: it regenerates + re-pins, "
            f"prints the full per-field diff, and refuses on a dirty tree.")

    if args.promote:
        if _git_dirty():
            raise SystemExit(
                "[make_fixture] --promote-fixture-identity REFUSED: working tree is dirty "
                "(git status --porcelain is non-empty). Commit or stash first — mirrors the "
                "864/861 promote gate.")
        old_pin = None
        if os.path.exists(PIN):
            try:
                old_pin = fixture_identity.load_pin(PIN)
            except fixture_identity.FixtureIdentityError:
                old_pin = None  # corrupt/missing old pin: nothing to diff against, proceed

        tmp_db = DST + ".tmp"
        if os.path.exists(tmp_db):
            os.remove(tmp_db)
        top, n_min, n_day = build_fixture(args.src, tmp_db)
        new_pin = _build_pin(tmp_db, top,
                             note=f"TASK_873 promote: re-based from {args.src} "
                                  f"(top{TOP_N}={top}).")
        tmp_pin = PIN + ".tmp"
        with open(tmp_pin, "w") as fh:
            json.dump(new_pin, fh, indent=2)
            fh.write("\n")

        print("[make_fixture] --promote-fixture-identity: per-field diff (old pin -> new pin)")
        diff_lines = fixture_identity.diff_identity(old_pin or {}, new_pin)
        for line in (diff_lines or ["  (no fields differ)"]):
            print(line)

        # Each os.replace is individually atomic; the PAIR is not. What IS guaranteed: neither
        # lands until the build fully succeeded, and the only observable interleaving (new .db,
        # old pin) FAILS CLOSED — the next verify_or_die() in any suite reports a mismatch and
        # refuses, which is the safe direction. Not redesigned into a single transaction on
        # purpose: two files on one filesystem cannot be replaced atomically together, and the
        # failure mode is already a loud refusal, not a silent re-base.
        os.replace(tmp_db, DST)
        try:
            os.replace(tmp_pin, PIN)
        except OSError as e:
            raise SystemExit(
                f"[make_fixture] promote INTERRUPTED after the .db landed but before the pin "
                f"did ({e}). State: {DST} is the NEW fixture, {PIN} is the OLD pin — every "
                f"suite will now refuse with a mismatch (fail-closed, as designed). Recover by "
                f"re-running --promote-fixture-identity on a clean tree, or restore the old "
                f"fixture; the new pin was staged at {tmp_pin}.") from e
        print(f"[make_fixture] promoted: top{TOP_N}={top} minute_rows={n_min} "
              f"daily_rows={n_day} size={os.path.getsize(DST) / 1e6:.1f}MB -> {DST}")
        print(f"[make_fixture] pin updated -> {PIN} (sha256 {new_pin['sha256'][:12]}...)")
        return 0

    # Default invocation (no --promote): existing generate-path behavior, unchanged. Never
    # touches the pin.
    top, n_min, n_day = build_fixture(args.src, DST)
    print(f"fixture: top{TOP_N}={top} minute_rows={n_min} daily_rows={n_day} "
          f"size={os.path.getsize(DST) / 1e6:.1f}MB -> {DST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
