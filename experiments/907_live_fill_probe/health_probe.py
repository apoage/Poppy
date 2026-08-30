"""TASK_912 — the capture health probe (the T+1h span check, made into tooling).

Answers one question, from OUTSIDE the recorder process: "is this capture healthy right now?"
Deliberately narrow — see the Out-of-scope section of tasks/TASK_912_CAPTURE_HEALTH_PROBE.md:
no markouts, no fill modeling, no coverage fractions, no mid-age statistics beyond the
split-feed detector below. Those are 907B's (`estimands.py`/`shadow_engine.py`) job.

Read-only by construction
--------------------------
`copy_capture()` is the ONLY function in this module that ever opens the LIVE capture path, and
it only ever byte-copies it (local `cp`, or `ssh <host> cat ...` into a local temp) — never
parses it. Every other function in this module (`analyze_capture` and everything it calls)
operates exclusively on that copy. The CLI (`main`) always copies before analyzing; there is no
flag that skips the copy step, because the copy step is part of the tool, never the caller's job.

The probe's only write targets are `runs/907_live_fill_probe/health/` (the `--watch` log) and
its own explicit `--out` path. The temp copy itself lives in the system temp directory
(`tempfile`), not under `runs/`, and is removed after each analysis unless `--keep-copy` is
passed for debugging.

Format this module reads (restated here, not imported — this probe is stdlib-only with no
project imports; see `capture.py`'s module docstring for the authoritative format definition)
---------------------------------------------------------------------------------------------
An append-only JSONL file, one JSON object per line, every line carrying a `"type"` field:
`meta_start` (once, first line: `capture_id`, `pairs`, ...), `book` (`pair`, `ts`, `bid`, `ask`,
`bid_sz`, `ask_sz`), `trade` (`pair`, `ts`, `price`, `volume`, `side`, optional `venue_ts`),
`heartbeat` (`ts`), `counters` (`ts`, `json_decode_errors`, `schema_mismatch`,
`unrecognized_frames` — CUMULATIVE, so the reader takes the LAST one), `gap` (`start_ts`,
`end_ts`, `duration_s`, `reason`, `detail`), `meta_end` (`ts`, written once on clean shutdown
only). Clock basis for every `ts`/`start_ts`/`end_ts` is `local_receipt` (the recorder's own
wall clock) — see `capture.py` for the full rationale.

Crash safety (from `capture.py`): every line is flushed+fsynced individually, so a kill
mid-write (or a copy taken mid-append) can only ever truncate the LAST line, never corrupt an
earlier one. This probe tolerates a corrupt/truncated TRAILING line without counting it as a
schema mismatch or raising; a corrupt NON-trailing line is real corruption and raises.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


class CaptureParseError(ValueError):
    """A capture COPY line violates the append-only capture contract (see `capture.py`'s module
    docstring). Raised only for a non-trailing corrupt line — a corrupt/truncated TRAILING line
    is expected (the copy was taken mid-append) and is tolerated instead, matching
    `capture.py::_iter_lines_tolerant`'s contract exactly.
    """


# --- verdict thresholds (fail-closed, conservative; TASK_912 Scope 3) -----------------------
#: A pair's book feed silent this long is dead, not a quiet market (Kraken book updates on any
#: real liquidity far more often than this).
BOOK_STALE_THRESHOLD_S = 300.0
#: The capture's newest event of ANY type under this old is "still alive" — the signal that
#: distinguishes one dead book subscription (split-feed) from the whole recorder being dead.
FRESH_THRESHOLD_S = 60.0
WARN_MAX_GAP_S = 600.0
WARN_TRADE_STALE_S = 600.0
#: Gap buckets reported in `gap_histogram` (cumulative thresholds, not exclusive bins).
GAP_BUCKET_THRESHOLDS_S = (5.0, 60.0, 600.0)
#: `counters` records exist from this format version on (`capture.py`: format_version 1 captures
#: have no counters record, so their absence THERE is a shape the writer really produces).
COUNTERS_FORMAT_VERSION = 2

#: The probe's ONLY write target besides an explicit `--out`/`--log-dir` override.
HEALTH_LOG_RELPATH = ("runs", "907_live_fill_probe", "health")


def _repo_root() -> Path:
    # experiments/907_live_fill_probe/health_probe.py -> parents[2] is the repo root.
    return Path(__file__).resolve().parents[2]


# ============================= copy step (the ONLY read of the LIVE file) ===================

def copy_capture(source_path: str, dest_path: str, host: Optional[str] = None) -> None:
    """Byte-copy the capture into `dest_path`. Never parses it.

    `host`, if given, copies over ssh (`ssh <host> cat -- <source_path>`) instead of a local
    `cp`/`shutil.copyfile` — both are a straight byte copy, nothing more. This is the ONLY
    function in this module that ever touches the live capture path.
    """
    if host:
        with open(dest_path, "wb") as out:
            subprocess.run(["ssh", host, "cat", "--", source_path], stdout=out, check=True)
    else:
        shutil.copyfile(source_path, dest_path)


# ============================= tolerant JSONL reader (copy only) ============================

def _iter_lines_tolerant(path: str) -> Iterator[Dict[str, Any]]:
    """Yield parsed JSON lines, silently dropping ONLY a truncated/corrupt trailing line.

    Mirrors `capture.py::_iter_lines_tolerant` exactly, restated here (not imported — this
    module is stdlib-only, no project imports). A kill mid-write can only ever truncate the LAST
    line (capture.py's fsync-per-line crash-safety guarantee), so a non-trailing corrupt line is
    real corruption, not an artifact of copying mid-append, and is not tolerated.
    """
    with open(path, "r") as fh:
        lines = fh.readlines()
    for i, line in enumerate(lines):
        line = line.rstrip("\n")
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                continue  # truncated trailing write from a copy taken mid-append — tolerated
            raise CaptureParseError(f"{path}: corrupt non-trailing line {i} is not valid JSON")


def _file_has_truncated_tail(path: str) -> bool:
    with open(path, "r") as fh:
        lines = fh.readlines()
    if not lines:
        return False
    last = lines[-1].rstrip("\n")
    if not last:
        return False
    try:
        json.loads(last)
        return False
    except json.JSONDecodeError:
        return True


def _finite_ts(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def _counter_value(value: Any) -> Optional[int]:
    """Guard for a cumulative `counters` field: returns the count, or None if UNREADABLE.

    The same defensive style every other field in this module already uses (`_finite_ts`, the
    `pairs` shape-check, the `pair`/`ts` guards) — restated for the counters because this field
    used to go through a bare `int(...)`, which had two ways to be wrong and no way to be
    honest: `0.9` truncated to a VERIFIED ZERO while a mismatch had in fact occurred, and
    `"corrupt"`/`null` raised straight out of `analyze_capture`. An unreadable count is UNKNOWN;
    the caller records it in `malformed_records` and the verdict says so, because a broken
    instrument must never read the same as a clean measurement.

    Accepted: a non-negative int, or a float that is an exact non-negative integer (JSON has one
    number type, so `2.0` is how some encoders write 2). Rejected: `bool` (True would count as a
    mismatch of 1 and False as a verified zero), a fractional/non-finite/negative number, and
    anything that is not a number at all (str/None/dict/list — including an absent field, which
    `CaptureWriter.write_counters` never produces).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float):
        if not math.isfinite(value) or value != int(value):
            return None
        value = int(value)
    return value if value >= 0 else None


# ============================= the core analysis (copy-only, no estimands) ===================

def analyze_capture(copy_path: str, wall_now: Optional[float] = None) -> Dict[str, Any]:
    """Parse the capture COPY at `copy_path` and report its current health.

    `copy_path` must already be a copy — this function has no way to enforce that, but the CLI
    in this module (`run_once` / `run_watch`) never calls it on anything else; `copy_capture`
    always runs first.

    Deliberately narrow (TASK_912 Out-of-scope): no markouts, no fill modeling, no coverage
    fractions, no mid-age statistics beyond the split-feed detector. Everything returned here is
    a liveness/count fact observed directly in the copy, not a claim about the estimand.

    DECLARED vs OBSERVED (run-47 BLOCKER). `per_pair` is seeded from `meta_start`'s declared
    pair set UP FRONT and then unioned with whatever pairs are actually observed. Populating it
    lazily from observed records alone made the one failure this row exists to catch invisible:
    a pair whose feed died before the copy window (or never connected at all) was absent from
    `per_pair`, absent from every reason string, and could not FAIL — the capture read PASS
    while a declared pair contributed nothing. Same defect family, same answer as
    `test_declared_pairs.py`: an absent age is not a young age.
    """
    if wall_now is None:
        wall_now = time.time()
    file_bytes = os.path.getsize(copy_path)

    capture_id: Optional[str] = None
    format_version: Optional[int] = None
    declared_pairs: Optional[List[str]] = None
    declared_pairs_malformed = False
    events_total = 0
    last_counters: Optional[Dict[str, Any]] = None
    #: Only the NEWEST timestamp of any type is needed (the freshness signal that separates a
    #: split feed from a dead recorder), so this is a running max rather than a list — the live
    #: capture is 744MB / millions of records and the merged list existed only for the old
    #: inter-event gap histogram, which `_gap_histogram` no longer uses.
    newest_ts: Optional[float] = None
    gap_records: List[Dict[str, Any]] = []
    #: Records the probe could not interpret (non-finite `ts`, missing/non-string `pair`,
    #: unusable gap duration). Counted and REPORTED rather than silently dropped: dropping them
    #: while still counting them in `events_total` makes the totals disagree with the per-pair
    #: table with nothing reconciling the two.
    malformed: Dict[str, int] = {}
    per_pair: Dict[str, Dict[str, Any]] = {}

    def _bad(kind: str) -> None:
        malformed[kind] = malformed.get(kind, 0) + 1

    def _pair(name: str) -> Dict[str, Any]:
        return per_pair.setdefault(
            name, {"n_book": 0, "n_trade": 0, "_last_book_ts": None, "_last_trade_ts": None})

    def _note_ts(ts: Optional[float]) -> None:
        nonlocal newest_ts
        if ts is not None and (newest_ts is None or ts > newest_ts):
            newest_ts = ts

    for obj in _iter_lines_tolerant(copy_path):
        events_total += 1
        t = obj.get("type")
        if t == "meta_start":
            # `CaptureWriter` emits exactly one meta_start per file (a resume does NOT re-emit
            # it), so first-wins and last-wins coincide in practice; the declared set is unioned
            # across any further ones because fail-closed means never shrinking it.
            if capture_id is None:
                cid = obj.get("capture_id")
                if isinstance(cid, str) and cid:
                    capture_id = cid
            if format_version is None:
                fv = obj.get("format_version", 1)
                format_version = fv if isinstance(fv, int) and not isinstance(fv, bool) else None
            declared = obj.get("pairs")
            if declared is not None:
                # Shape-check before use, exactly as `capture.py` does at the same field: a bare
                # string would be iterated character-by-character into a declared set of single
                # letters that no observed pair can ever match.
                if (not isinstance(declared, list)
                        or not all(isinstance(p, str) and p for p in declared)):
                    declared_pairs_malformed = True
                else:
                    declared_pairs = sorted(set(declared) | set(declared_pairs or []))
        elif t == "book":
            ts = _finite_ts(obj.get("ts"))
            pair = obj.get("pair")
            if ts is None:
                _bad("book:ts")
            elif not (isinstance(pair, str) and pair):
                _note_ts(ts)
                _bad("book:pair")
            else:
                _note_ts(ts)
                p = _pair(pair)
                p["n_book"] += 1
                if p["_last_book_ts"] is None or ts > p["_last_book_ts"]:
                    p["_last_book_ts"] = ts
        elif t == "trade":
            ts = _finite_ts(obj.get("ts"))
            pair = obj.get("pair")
            if ts is None:
                _bad("trade:ts")
            elif not (isinstance(pair, str) and pair):
                _note_ts(ts)
                _bad("trade:pair")
            else:
                _note_ts(ts)
                p = _pair(pair)
                p["n_trade"] += 1
                if p["_last_trade_ts"] is None or ts > p["_last_trade_ts"]:
                    p["_last_trade_ts"] = ts
        elif t == "heartbeat":
            ts = _finite_ts(obj.get("ts"))
            _note_ts(ts)
            if ts is None:
                _bad("heartbeat:ts")
        elif t == "counters":
            # Cumulative — capture.py's own docstring says the reader takes the LAST one.
            last_counters = obj
            ts = _finite_ts(obj.get("ts"))
            _note_ts(ts)
            if ts is None:
                _bad("counters:ts")
        elif t == "gap":
            end_ts = _finite_ts(obj.get("end_ts"))
            start_ts = _finite_ts(obj.get("start_ts"))
            _note_ts(end_ts)
            duration_s = _finite_ts(obj.get("duration_s"))
            if duration_s is None and start_ts is not None and end_ts is not None:
                duration_s = end_ts - start_ts
            if duration_s is None or duration_s < 0:
                _bad("gap:duration_s")
            else:
                gap_records.append({"duration_s": duration_s, "end_ts": end_ts,
                                     "reason": obj.get("reason")})
        elif t == "meta_end":
            ts = _finite_ts(obj.get("ts"))
            _note_ts(ts)
            if ts is None:
                _bad("meta_end:ts")
        # Any other/unknown record type: still counted in events_total, otherwise ignored — a
        # future record type must not crash unattended monitoring.

    truncated_tail = _file_has_truncated_tail(copy_path)

    # `None` here always means UNKNOWN, never zero — either no counters record was seen at all
    # (`counters_present` distinguishes that) or the record's value was unreadable, in which case
    # it is counted as a malformed record like every other uninterpretable field in this module.
    schema_mismatch_count: Optional[int] = None
    if last_counters is not None:
        schema_mismatch_count = _counter_value(last_counters.get("schema_mismatch"))
        if schema_mismatch_count is None:
            _bad("counters:schema_mismatch")

    # The declared seed. Union, never replacement: a pair OBSERVED but never DECLARED must stay
    # visible too (`test_declared_pairs.py::test_the_union_is_blind_where_the_declared_set_is_not`).
    for name in (declared_pairs or []):
        _pair(name)

    gap_histogram = _gap_histogram(gap_records)
    newest_event_age_s = (wall_now - newest_ts) if newest_ts is not None else None

    per_pair_out: Dict[str, Dict[str, Any]] = {}
    for pair, rec in sorted(per_pair.items()):
        last_book_ts = rec["_last_book_ts"]
        last_trade_ts = rec["_last_trade_ts"]
        per_pair_out[pair] = {
            "n_book": rec["n_book"],
            "n_trade": rec["n_trade"],
            "last_book_age_s": (wall_now - last_book_ts) if last_book_ts is not None else None,
            "last_trade_age_s": (wall_now - last_trade_ts) if last_trade_ts is not None else None,
        }

    verdict = _compute_verdict(
        per_pair_out, schema_mismatch_count, newest_event_age_s, gap_histogram,
        declared_pairs=declared_pairs, declared_pairs_malformed=declared_pairs_malformed,
        counters_present=last_counters is not None, format_version=format_version,
        malformed=malformed)

    return {
        "wall_now": wall_now,
        "file_bytes": file_bytes,
        "capture_id": capture_id,
        "events_total": events_total,
        "schema_mismatch_count": schema_mismatch_count,
        "gap_histogram": gap_histogram,
        "per_pair": per_pair_out,
        "verdict": verdict,
        "truncated_tail": truncated_tail,
        # --- provenance for the three judgements above, so a reader of the log line can tell
        # "verified" from "unknown" without re-reading the capture ---
        "format_version": format_version,
        "declared_pairs": declared_pairs,
        "counters_present": last_counters is not None,
        "malformed_record_count": sum(malformed.values()),
        "malformed_records": dict(sorted(malformed.items())),
    }


def _gap_histogram(gap_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Count of RECORDED capture gaps exceeding each of `GAP_BUCKET_THRESHOLDS_S` (cumulative
    and strict, i.e. a gap counted in `gt_600s` is also counted in `gt_60s` and `gt_5s`, and a
    gap of exactly 5.0s is in none of them), plus the single largest gap, the wall-clock
    timestamp at which it ended, and its reason.

    WHICH OF THE TWO READINGS THIS IS, AND WHY (run-47 MAJOR)
    --------------------------------------------------------
    There are two things "gap histogram" could mean, and they are not close:

      (a) deltas between consecutive events on ONE merged timeline (all pairs, all record
          types, sorted together), and
      (b) the capture's OWN `gap` records — the ledger `capture.py` writes on every
          disconnect / sequence discontinuity / heartbeat timeout / clock jump, and which that
          module guarantees is "never silently interpolated".

    This function implements (b). Reading (a) is structurally blind to exactly the failure this
    probe exists to catch: because the timeline merges every pair AND the heartbeat, it never
    goes quiet as long as ANY other channel is alive. Measured on a fixture with a genuine 150s
    single-pair outage and 5s heartbeats, reading (a) returned
    `{"gt_5s": 0, "gt_60s": 0, "gt_600s": 0, "max_gap_s": 4.0}` and a PASS verdict — the outage
    was invisible. On the live 907 capture, (b) reads >5s=431, >60s=362, >600s=0, max=95.3s
    against (a)'s near-zero; `test_gap_histogram_reads_the_captures_own_gap_records` pins it.

    A gap record whose duration cannot be read is NOT silently skipped — `analyze_capture`
    counts it under `malformed_records["gap:duration_s"]` and the verdict says so, because a
    broken instrument must not read the same as a quiet one.
    """
    counts: Dict[str, Any] = {f"gt_{int(t)}s": 0 for t in GAP_BUCKET_THRESHOLDS_S}
    max_gap_s: Optional[float] = None
    max_gap_wall_ts: Optional[float] = None
    max_gap_reason: Optional[str] = None
    for rec in gap_records:
        duration_s = rec["duration_s"]
        for t in GAP_BUCKET_THRESHOLDS_S:
            if duration_s > t:
                counts[f"gt_{int(t)}s"] += 1
        if max_gap_s is None or duration_s > max_gap_s:
            max_gap_s = duration_s
            max_gap_wall_ts = rec["end_ts"]      # TASK_912 Scope 2: the max gap WITH its timestamp
            reason = rec.get("reason")
            max_gap_reason = reason if isinstance(reason, str) and reason else None
    counts["max_gap_s"] = max_gap_s
    counts["max_gap_wall_ts"] = max_gap_wall_ts
    counts["max_gap_reason"] = max_gap_reason
    counts["n_gap_records"] = len(gap_records)
    return counts


def _compute_verdict(per_pair: Dict[str, Dict[str, Any]], schema_mismatch_count: Optional[int],
                      newest_event_age_s: Optional[float],
                      gap_histogram: Dict[str, Any], *,
                      declared_pairs: Optional[List[str]] = None,
                      declared_pairs_malformed: bool = False,
                      counters_present: bool = True,
                      format_version: Optional[int] = None,
                      malformed: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """TASK_912 Scope 3, in the order given there. Fail-closed and conservative: an unknown
    (`None`) age is treated as stale, never as healthy, and an UNKNOWN value never borrows the
    reason string of a VERIFIED one.

      1. FAIL, full stop, if `schema_mismatch_count` > 0 (Addendum A's invariant).
      2. Counters provenance: `schema_mismatch_count is None` means the count is UNKNOWN — and
         the mechanism that would have told us may itself be dead. Two distinct shapes, two
         distinct reason strings, neither of them the "verified 0" string: the counters record
         was present but its value UNREADABLE (always at least WARN), or no `counters` record was
         seen at all — at `format_version >= COUNTERS_FORMAT_VERSION` at minimum a WARN, at
         format_version 1 tolerated because the record does not exist in that format
         (`capture.py`) — but it still never reads as "verified clean".
      3. FAIL if the capture declares no pair set, or a malformed one: the declared set is the
         only thing that makes (4) meaningful, and "declared := observed" is a completeness
         claim about nothing.
      4. FAIL (split-feed death) if any pair's `last_book_age_s` exceeds
         `BOOK_STALE_THRESHOLD_S` — or is unknown, i.e. the pair was DECLARED and never carried
         a single book event — while the capture's newest event of ANY type is under
         `FRESH_THRESHOLD_S` old.
      5. FAIL (recorder-dead), the SAME book-staleness condition but the newest event of any
         type is ALSO stale — nothing is fresh, not just one pair's book. Distinguished from (4)
         in the reason string on purpose: they are different failures with different remedies.
      6. WARN if the max RECORDED capture gap exceeds `WARN_MAX_GAP_S`, any pair's
         `last_trade_age_s` exceeds `WARN_TRADE_STALE_S`, or any record was malformed.
      7. PASS otherwise.

    Rules 6 and 7 decide the STATUS only. Every rule's reasons are appended unconditionally: the
    reasons list is the full account of what was observed, not just the account of whichever
    condition set the final status.
    """
    reasons: List[str] = []
    status = "PASS"
    malformed = malformed or {}

    def _escalate(to: str) -> None:
        nonlocal status
        if to == "FAIL" or (to == "WARN" and status == "PASS"):
            status = to

    if schema_mismatch_count:  # a VERIFIED nonzero count
        reasons.append(
            f"schema_mismatch_count={schema_mismatch_count} > 0 — Addendum A invariant violated "
            f"(emitted top-of-book did not reconcile with the CRC-verified state). FAIL, full "
            f"stop.")
        _escalate("FAIL")
    elif counters_present and schema_mismatch_count is None:
        # A counters record WAS written but its value could not be read (fractional, negative,
        # bool, non-finite, a string, null, or absent). Its own reason string, distinct from both
        # "verified 0" and "counters record ABSENT": three different facts, three different
        # sentences. See the docstring of this function, rule 2.
        reasons.append(
            "schema_mismatch UNREADABLE: a `counters` record is present but its "
            "`schema_mismatch` value is not an int-shaped non-negative number, so the count is "
            "UNKNOWN, not zero. Reading it as zero would report a mismatch that DID occur as "
            "verified clean — the exact inversion Addendum A's \"FAIL, full stop\" rule exists "
            "to prevent. See malformed_records['counters:schema_mismatch'].")
        _escalate("WARN")
    elif not counters_present:
        # NOT the same fact as a verified zero, and never the same words. See the docstring of
        # this function, rule 2.
        if format_version is None or format_version >= COUNTERS_FORMAT_VERSION:
            reasons.append(
                f"counters record ABSENT (format_version={format_version}) — schema_mismatch is "
                f"UNKNOWN, not zero. This format writes a cumulative `counters` record; its "
                f"absence means either the copy predates the first one or the counter mechanism "
                f"itself is dead, and a dead instrument must not read as a clean measurement.")
            _escalate("WARN")
        else:
            reasons.append(
                f"counters unknown: format_version {format_version} has no `counters` record, so "
                f"schema_mismatch is not verified here (tolerated — absence is a shape this "
                f"format really produces — but not reported as clean).")

    if declared_pairs_malformed:
        reasons.append(
            "MALFORMED PAIR SET: meta_start `pairs` is not a list of non-empty strings. Refusing "
            "to read it (a bare string would be iterated character-by-character into a declared "
            "set of single letters no observed pair can ever match) — with no readable "
            "declaration the split-feed detector below cannot mean what it says.")
        _escalate("FAIL")
    elif not declared_pairs:
        reasons.append(
            "DECLARES NO PAIR SET: meta_start carries no `pairs` field. `CaptureWriter` writes it "
            "unconditionally at every format version, so this is not a shape this recorder "
            "produces. Fail-CLOSED rather than falling back to \"declared := observed\": that "
            "fallback IS the bypass — a capture that declares nothing reconciles perfectly "
            "against whatever it happens to contain, and the split-feed detector becomes a "
            "completeness claim about nothing.")
        _escalate("FAIL")

    if not per_pair or newest_event_age_s is None:
        reasons.append(
            "recorder-dead: no book/trade events observed in this capture copy at all")
        return {"status": "FAIL", "reasons": reasons}

    never_book_pairs = sorted(
        pair for pair, rec in per_pair.items() if rec["last_book_age_s"] is None)
    stale_book_pairs = sorted(
        pair for pair, rec in per_pair.items()
        if rec["last_book_age_s"] is not None
        and rec["last_book_age_s"] > BOOK_STALE_THRESHOLD_S)
    affected = sorted(set(never_book_pairs) | set(stale_book_pairs))

    if never_book_pairs:
        reasons.append(
            f"declared-but-silent: pair(s) {never_book_pairs} carry ZERO book events anywhere in "
            f"this capture copy — declared in meta_start `pairs` and never observed (or dead "
            f"before this copy window). An absent age is not a young age: treated as maximally "
            f"stale.")

    if affected:
        if newest_event_age_s < FRESH_THRESHOLD_S:
            reasons.append(
                f"split-feed death: pair(s) {affected} book feed stale (> "
                f"{BOOK_STALE_THRESHOLD_S:g}s or never seen) while the capture's newest event of "
                f"ANY type is only {newest_event_age_s:.1f}s old — the rest of the capture is "
                f"alive, this pair's book subscription is dead")
        else:
            reasons.append(
                f"recorder-dead: pair(s) {affected} book feed stale (> "
                f"{BOOK_STALE_THRESHOLD_S:g}s or never seen) AND the capture's newest event of "
                f"ANY type is {newest_event_age_s:.1f}s old (>= {FRESH_THRESHOLD_S:g}s) — "
                f"nothing in this capture is fresh, not just one pair's book")
        _escalate("FAIL")

    # UNGATED, exactly like the malformed-records check below it (run-48 MAJOR). These checks
    # used to sit behind `if status != "FAIL":`, which truncated the account at the first
    # FAIL-tier condition: a capture with a schema mismatch AND a 900s gap AND a pair whose
    # trades stopped 15 minutes ago reported only the mismatch, with the other two facts sitting
    # in `gap_histogram`/`per_pair` and nothing in `reasons` pointing at them. Only the STATUS
    # obeys FAIL-dominates-WARN (that is `_escalate`'s job, and it is a no-op once status is
    # FAIL); `reasons` is a complete account of everything the probe saw.
    stale_trade_pairs = sorted(
        pair for pair, rec in per_pair.items()
        if rec["last_trade_age_s"] is not None
        and rec["last_trade_age_s"] > WARN_TRADE_STALE_S)
    max_gap_s = gap_histogram.get("max_gap_s")
    warn_reasons = []
    if max_gap_s is not None and max_gap_s > WARN_MAX_GAP_S:
        warn_reasons.append(
            f"max recorded capture gap {max_gap_s:.1f}s > {WARN_MAX_GAP_S:g}s "
            f"(reason={gap_histogram.get('max_gap_reason')!r}, "
            f"ended {gap_histogram.get('max_gap_wall_ts')})")
    if stale_trade_pairs:
        warn_reasons.append(
            f"pair(s) {stale_trade_pairs} last_trade_age_s > {WARN_TRADE_STALE_S:g}s")
    if warn_reasons:
        _escalate("WARN")
        reasons.extend(warn_reasons)

    if malformed:
        # Visible, not silently dropped: these records counted toward `events_total` but could
        # not be attributed to a pair / measured as a gap.
        reasons.append(
            f"{sum(malformed.values())} malformed record(s) could not be interpreted and are "
            f"excluded from the per-pair and gap tables: "
            f"{dict(sorted(malformed.items()))}")
        _escalate("WARN")

    if not reasons:
        reasons.append(
            "all checks passed: no stale book feeds, every declared pair observed, "
            "schema_mismatch verified 0, gaps within bounds")

    return {"status": status, "reasons": reasons}


# ============================= CLI ===========================================================

def format_human_summary(record: Dict[str, Any]) -> str:
    v = record["verdict"]
    lines = [
        f"CAPTURE HEALTH: {v['status']}  (capture_id={record['capture_id']!r}, "
        f"wall_now={record['wall_now']:.0f}, file_bytes={record['file_bytes']})",
        f"  events_total={record['events_total']}  "
        f"schema_mismatch_count={record['schema_mismatch_count']}"
        f"{'' if record.get('counters_present', True) else ' (UNKNOWN — no counters record)'}  "
        f"malformed={record.get('malformed_record_count', 0)}",
        f"  declared_pairs={record.get('declared_pairs')}",
    ]
    gh = record["gap_histogram"]
    lines.append(
        f"  gap_histogram (from the capture's own {gh.get('n_gap_records')} `gap` records): "
        f">5s={gh.get('gt_5s')} >60s={gh.get('gt_60s')} >600s={gh.get('gt_600s')} "
        f"max={gh.get('max_gap_s')} @ {gh.get('max_gap_wall_ts')} "
        f"({gh.get('max_gap_reason')})")
    for pair, rec in sorted(record["per_pair"].items()):
        lines.append(
            f"  {pair}: n_book={rec['n_book']} n_trade={rec['n_trade']} "
            f"last_book_age_s={rec['last_book_age_s']} last_trade_age_s={rec['last_trade_age_s']}")
    for reason in v["reasons"]:
        lines.append(f"  reason: {reason}")
    return "\n".join(lines)


def _analyze_via_copy(capture_path: str, host: Optional[str], keep_copy: bool = False,
                       ) -> Dict[str, Any]:
    """Copy-then-analyze, always. No caller of this module ever gets to skip the copy step."""
    fd, tmp_path = tempfile.mkstemp(prefix="health_probe_", suffix=".jsonl")
    os.close(fd)
    try:
        copy_capture(capture_path, tmp_path, host=host)
        return analyze_capture(tmp_path)
    finally:
        if not keep_copy:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _error_record(exc: BaseException) -> Dict[str, Any]:
    """The fail-closed floor, shared by both modes so they cannot drift apart: a probe that
    cannot analyse the capture reports FAIL in the normal record shape, never silence."""
    return {
        "wall_now": time.time(), "error": str(exc),
        "verdict": {"status": "FAIL", "reasons": [f"probe error: {exc}"]},
    }


def run_once(capture_path: str, host: Optional[str], out_path: Optional[str],
             keep_copy: bool = False) -> Dict[str, Any]:
    """One-shot: copy, analyze, optionally write JSON to `out_path`, always print both the JSON
    (stdout) and a human summary (stderr).

    Has the SAME fail-closed floor as `run_watch` (run-48 MAJOR). Without it, the malformed input
    that `--watch` degrades on gracefully killed one-shot mode with a bare traceback: no JSON on
    stdout, no `--out` file, nothing structured for the caller that is Acceptance criterion (i).
    The failure is still loud (stderr, and `main` exits non-zero) — swallowing it would be the
    opposite mistake.
    """
    try:
        record = _analyze_via_copy(capture_path, host, keep_copy=keep_copy)
    except Exception as exc:  # noqa: BLE001 — fail CLOSED and structured, never a bare traceback
        record = _error_record(exc)
    if out_path:
        with open(out_path, "w") as fh:
            json.dump(record, fh, separators=(",", ":"))
    print(json.dumps(record, separators=(",", ":")))
    if "per_pair" in record:
        print(format_human_summary(record), file=sys.stderr)
    else:
        print(f"PROBE ERROR: {record.get('error')} — verdict FAIL (the capture could not be "
              f"analysed; this is not a healthy capture reading clean)", file=sys.stderr)
    return record


def run_watch(capture_path: str, host: Optional[str], interval_s: float,
              log_dir: Optional[str] = None, iterations: Optional[int] = None) -> Path:
    """Loop runner for unattended use. Appends ONE JSON line per interval to a health log under
    `runs/907_live_fill_probe/health/` (the probe's ONLY write target besides `--out`), so the
    whole capture's health history is one `grep FAIL` away. Returns the log path.

    `iterations`, if given, bounds the loop (used by tests and manual smoke checks); omitted or
    `None` loops forever at `interval_s`. A single bad copy/parse must never kill an unattended
    loop, so per-iteration errors are caught, logged as a FAIL record, and the loop continues —
    and neither must a single bad WRITE: the log append sits inside its own guard, because a
    transient disk-full/permission hiccup used to terminate the whole monitoring loop and leave
    the capture unwatched for the rest of its multi-day run. It is loud (stderr, every time) but
    never fatal.
    """
    log_dir_path = Path(log_dir) if log_dir else (_repo_root() / Path(*HEALTH_LOG_RELPATH))
    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_path = log_dir_path / f"{Path(capture_path).stem}.health.jsonl"

    n = 0
    while iterations is None or n < iterations:
        try:
            record = _analyze_via_copy(capture_path, host)
        except Exception as exc:  # noqa: BLE001 — unattended: one bad iteration must not kill the loop
            record = _error_record(exc)
        try:
            with open(log_path, "a") as fh:
                fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        except Exception as exc:  # noqa: BLE001 — a write hiccup must not end the watch
            print(f"HEALTH LOG WRITE FAILED ({log_path}): {exc!r} — this iteration's line is "
                  f"LOST from the log; the loop continues", file=sys.stderr)
        if "per_pair" in record:
            print(format_human_summary(record))
        else:
            print(f"PROBE ERROR: {record.get('error')}")
        n += 1
        if iterations is None or n < iterations:
            time.sleep(interval_s)
    return log_path


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="TASK_912 read-only capture health probe (stdlib only).")
    p.add_argument("--capture", required=True, help="path to the (live) capture JSONL")
    p.add_argument("--host", default=None,
                    help="ssh host to copy the capture FROM (e.g. gpu-server); "
                         "omit to copy a local file")
    p.add_argument("--out", default=None,
                    help="one-shot mode: write the JSON result to this path")
    p.add_argument("--watch", action="store_true",
                    help="loop forever, appending one JSON line every --interval seconds to "
                         "runs/907_live_fill_probe/health/")
    p.add_argument("--interval", type=float, default=900.0, help="--watch loop interval, seconds")
    p.add_argument("--log-dir", default=None,
                    help="override the --watch log directory "
                         "(default: runs/907_live_fill_probe/health/)")
    p.add_argument("--iterations", type=int, default=None,
                    help="--watch: stop after this many iterations (default: forever)")
    p.add_argument("--keep-copy", action="store_true",
                    help="one-shot mode only: keep the temp copy after the run (debugging)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.watch:
        run_watch(args.capture, args.host, args.interval, log_dir=args.log_dir,
                   iterations=args.iterations)
    else:
        record = run_once(args.capture, args.host, args.out, keep_copy=args.keep_copy)
        # Non-zero ONLY when the probe itself could not analyse the capture. A capture that
        # analyses cleanly and reads FAIL/WARN still exits 0 — that is a verdict, reported in the
        # JSON; this is the tool failing to produce one at all.
        if "error" in record:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
