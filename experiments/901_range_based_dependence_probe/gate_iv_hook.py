#!/usr/bin/env python3
"""TASK_901 gate (iv) — DEFERRED. Do not fill this in speculatively.

TASK_901's own scope note: "(iv) When this row ships: 897's `CI lo > 0` conjunct is re-read from
the EXISTING srv-leg artifacts (same traces, new interval) — recorded as a follow-up read, not a
new run." The build brief for THIS implementation explicitly defers gate (iv): 897's srv leg was
mid-relaunch at the time this row was implemented (guard re-based to the pins' own basis, tiers
{8,16} pre-registered — see `tasks/QUEUE.md` / DECISIONS D55/D57/D60 and the "897 SRV LEG
LAUNCHED" commit) and its output artifacts DO NOT YET EXIST. There is nothing to re-read.

THIS MODULE IS THE HOOK, NOT THE ANSWER. It intentionally contains NO 897 numbers, fabricated,
approximated, or otherwise — filling in a plausible-looking verdict here would be exactly the
"claim in a shipped string the code doesn't back" defect DECISIONS D57 already named and ruled
against once this run.

WHEN 897's SRV-LEG ARTIFACT EXISTS (a CellRecord table with `ci_lo_ADVISORY`/`ci_hi_ADVISORY`,
`measured_dependence_len`, `block_used`, per real cell — see
`experiments/897_corrected_809/sweep_gated_maker.py::write_artifact` / `CellRecord`), the
follow-up read this hook exists for is:

  1. Load that artifact (path is that lane's to name at launch time — check `tasks/QUEUE.md` /
     `DECISIONS.md` for the srv leg's actual output location before writing this).
  2. For each cell, take its ALREADY-RECORDED per-decision return series basis (n_decisions,
     block_used) — TASK_901 does not re-run the sweep or re-fit any signal.
  3. Re-estimate the dependence RANGE on that cell's own series with THIS module's certified
     estimator (`range_probe.ESTIMATORS[<the one `measured_findings.MEASURED_COVERAGE_TABLE` certified>]`),
     not tau.
  4. Feed the result through `range_probe.block_rule` to get a (possibly different) block length
     and calibrated k, and re-derive `ci_lo`/`ci_hi` from the cell's OWN raw bootstrap draws (or,
     if those are not retained in the artifact, from a fresh `honest_estimator.
     block_bootstrap_sharpe_ci` call at the SAME seed the srv leg recorded, on the SAME series —
     "same traces, new interval", never a new fit/predict/decision-engine run).
  5. `CI lo > 0` reads off `ci_lo` from step 4, per-cell, replacing the "UNAVAILABLE-because-
     uncalibratable" status `honest_estimator.CI_UNCALIBRATED_DISCLOSURE` currently assigns.

`reread_897_conjunct` below is the typed stub for that procedure. Calling it before 897's
artifact exists REFUSES loudly (raises) rather than returning a plausible-looking dict — refusal
is the correct behavior here, not a bug to silence.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict


class Gate4NotYetRunnable(RuntimeError):
    """Raised by `reread_897_conjunct` whenever the srv-leg artifact it needs does not exist yet.
    This is the EXPECTED exception at the time TASK_901 was implemented — 897's srv leg was
    mid-relaunch, per the task brief's explicit instruction to defer gate (iv) rather than
    fabricate, stub, or approximate its numbers."""


def reread_897_conjunct(artifact_path: str, selected_estimator: str) -> Dict:
    """DEFERRED. Raises `Gate4NotYetRunnable` unless `artifact_path` exists on disk — this
    function does not know where the srv leg will land its output and must not guess one into
    existence. `selected_estimator` must name one of `range_probe.ESTIMATORS`'s keys — the one
    THIS row's own `measured_findings.MEASURED_COVERAGE_TABLE` certified, never chosen ad hoc at
    read time."""
    p = Path(artifact_path)
    if not p.exists():
        raise Gate4NotYetRunnable(
            f"897 srv-leg artifact not found at {artifact_path!r} — gate (iv) is DEFERRED until "
            "that leg completes (see this module's docstring for the exact re-read procedure "
            "once it does). This is not a bug: TASK_901's own build explicitly scoped gates "
            "(i)-(iii) only and left this hook unfilled rather than fabricate a number."
        )
    raise NotImplementedError(
        "897's artifact now exists, but the re-read procedure (this module's docstring, steps "
        "1-5) has not been implemented yet — implementing it against a real artifact schema "
        "sight-unseen would itself be a guess. Read the artifact's actual CellRecord schema "
        "before writing the loop, then replace this stub."
    )
