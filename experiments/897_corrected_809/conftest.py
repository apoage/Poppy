"""TASK_911 (D69) -- session-level startup gate for the 897 suite.

Verifies the 806/808/809 gitignored lineage files this suite imports/reads (see
``config/gitignored_pins.json``) BEFORE any test in this directory collects or runs, and
refuses the ENTIRE session, loudly, on divergence -- not just one failing test among many. This
is what makes gate (i) literal: "the 897 suite refuses at startup, loudly."

``test_gitignored_pins.py`` in this same directory exercises the identical check as an ordinary
pytest test too (so its pass/fail shows up in normal test accounting/CI reporting when nothing
diverges) -- the two layers agree by construction because both call
``tools.verify_gitignored_pins.verify_all`` the same way. When something DOES diverge, this
hook fires first and the session never reaches collection, so that test never even runs; that's
intentional, not a gap -- the loud refusal happens strictly earlier.

Wholesale absence (a fresh checkout that never staged the gitignored ``runs/`` tree at all) is
NOT the D69 defect and is not refused here -- it is the same, already-sanctioned condition
test_recompute_gate.py / test_sweep_guards.py already skip on individually. Only a
present-but-diverged or present-but-partially-missing lineage refuses.

KNOWN BYPASS -- ``pytest --noconftest`` (stated, not fought)
-----------------------------------------------------------
``pytest experiments/897_corrected_809/test_recompute_gate.py --noconftest -q`` against a
mutated ``recompute_best.py`` reports ``3 passed`` and never mentions the divergence: the flag
tells pytest not to load conftest files at all, so ``pytest_sessionstart`` below never runs.
This is structural to ANY conftest-hosted startup gate and is shared with the
``fixtures/FIXTURE_IDENTITY.json`` precedent this row deliberately follows -- there is no
in-conftest defense against not loading the conftest.

It is documented rather than patched because every available "fix" is worse than the honest
limit: a defense that pretends to be total invites exactly the trust it cannot earn. What
closes it is a check that does not live in a conftest at all:

    python -m tools.verify_gitignored_pins          # exit 0 = all OK, 1 = DIVERGED/MISSING,
                                                    #        2 = UNPINNED (incomplete)

That invocation cannot be bypassed by a pytest flag, verifies the WHOLE manifest (not just this
suite's four lineage files), and is the entry point TASK_911's gate (iii) wires into the
factory's own startup ritual, so a divergence fails every lane rather than one suite. Anyone
reaching for ``--noconftest`` on this directory should run it.
"""
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.verify_gitignored_pins import verify_all, refuse_on_divergence, PinVerificationError

# The 806/808/809 lineage files THIS suite depends on -- see config/gitignored_pins.json for
# provenance/why_load_bearing on each. runs/875_wf/* is a separate lineage this suite never
# imports or reads -- out of scope for this suite's startup check per TASK_911's scope fence.
_LINEAGE_PATHS = (
    "runs/806_predictability_probe/probe.py",
    "runs/808_linear_backtest/backtest.py",
    "runs/809_decision_rule_rescue/decide.py",
    "runs/809_decision_rule_rescue/recompute_best.py",
)


def pytest_sessionstart(session):
    results = verify_all(paths=list(_LINEAGE_PATHS), repo_root=_REPO)
    present = [r for r in results if r["status"] != "MISSING"]
    if not present:
        print(
            f"[897 startup check] SKIP: the gitignored 806/808/809 lineage is not staged on "
            f"this host at all (0/{len(results)} pinned files present) -- nothing to "
            f"gate-test, same condition test_recompute_gate.py / test_sweep_guards.py already "
            f"skip on individually. Not the D69 defect (a present-but-stale file), so this run "
            f"proceeds; lineage-dependent tests will skip cleanly on their own.")
        return
    try:
        refuse_on_divergence(results, label="[897 startup check]")
    except PinVerificationError as e:
        pytest.exit(str(e), returncode=2)
    unpinned = [r for r in results if r["status"] == "UNPINNED"]
    if unpinned:
        pytest.exit(
            f"[897 startup check] REFUSED -- lineage file(s) present with no recorded pin in "
            f"config/gitignored_pins.json (should never happen; the 806/808/809 lineage is "
            f"seeded in full): {[r['path'] for r in unpinned]}", returncode=2)
    print(f"[897 startup check] OK -- {len(present)}/{len(results)} pinned 806/808/809 "
          f"lineage file(s) verified against config/gitignored_pins.json")
