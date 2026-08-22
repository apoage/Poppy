#!/usr/bin/env python3
"""TASK_897 — the single-cell mechanical gate in `runs/809_decision_rule_rescue/recompute_best.py`
must apply ALL THREE pre-registered criteria and must never emit a harvestable bare "PASS".

Two defects found by the fix-pass review are pinned here:

  1. The gate checked only `sharpe_decision >= 1.0 AND ci_lo > 0`, omitting the band's third
     criterion `PnL > 0` (TASK_897 spec line 45) — so it printed PASS on a constructed cell that
     had lost ALL its capital (volatility drag: per-decision returns alternating +20% / -19% have
     a positive arithmetic mean and a strongly positive Sharpe, but 1.20*0.81 = 0.972 per pair
     compounds to ruin).
  2. The PASS/FAIL token sat on a line that read as a verdict, with the "this is one cell, not the
     TASK_897 verdict" qualifier on SEPARATE following lines — i.e. greppable away. The qualifier
     now shares the token's line.

NOTE ON LOCATION: the code under test lives under gitignored `runs/`, while this suite lives in
tracked `experiments/`. The tests skip cleanly where the 809 lineage is not present (e.g. a fresh
checkout, or gpu-server before the lineage is staged) rather than failing for the wrong reason.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
_LINEAGE = _REPO / "runs" / "809_decision_rule_rescue" / "recompute_best.py"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import honest_estimator as hest  # noqa: E402

pytestmark = pytest.mark.skipif(
    not _LINEAGE.exists(),
    reason=f"809 lineage not present at {_LINEAGE} (gitignored runs/); nothing to gate-test",
)


def _load_recompute():
    for sub in ("806_predictability_probe", "808_linear_backtest", "809_decision_rule_rescue"):
        p = str(_REPO / "runs" / sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    import recompute_best  # noqa: E402  (import is side-effect-free: no DB touch at import time)
    return recompute_best


def _volatility_drag_metrics():
    """The review's counter-example cell, built through the REAL estimator: per-decision returns
    alternating +20% / -19%, one decision per bar. Positive arithmetic mean, negative geometric."""
    n = 400
    gross = np.empty(n, dtype=np.float64)
    gross[0::2] = 0.20
    gross[1::2] = -0.19
    turn = np.full(n, 0.1, dtype=np.float64)
    return hest.compute_honest_metrics(gross, turn, cost_per_side=0.0, bar_sec=60, seed=0,
                                        n_boot=500)


def test_volatility_drag_cell_is_failed_by_the_pnl_criterion():
    """The cell that used to print PASS: Sharpe and CI-lo both clear, PnL is -1.0 (total ruin).
    With the PnL>0 criterion restored the gate must FAIL it."""
    rb = _load_recompute()
    m = _volatility_drag_metrics()
    # the two criteria the old gate looked at both PASS...
    assert m.sharpe_decision >= 1.0
    assert m.ci_lo_UNCALIBRATED > 0.0
    # ...while the cell has destroyed capital.
    assert m.pnl < 0.0
    lines = rb.single_cell_gate_lines(m, "constructed volatility-drag cell")
    text = "\n".join(lines)
    assert "PnL>0 False" in text
    assert re.search(r"mechanical gate \(.*\): FAIL\b", text), text
    assert "PASS" not in text


def test_all_three_criteria_must_hold_for_the_token_to_read_pass():
    """Each criterion is individually load-bearing: flip any one and the token flips to FAIL."""
    rb = _load_recompute()

    class M:                       # a minimal stand-in with exactly the fields the gate reads
        def __init__(self, sharpe_decision, pnl, ci_lo):
            self.sharpe_decision = sharpe_decision
            self.pnl = pnl
            self.ci_lo_UNCALIBRATED = ci_lo

    def token(m):
        return "PASS" if "): PASS" in "\n".join(rb.single_cell_gate_lines(m, "x")) else "FAIL"

    assert token(M(1.5, 0.2, 0.3)) == "PASS"        # all three hold
    assert token(M(0.9, 0.2, 0.3)) == "FAIL"        # Sharpe below 1.0
    assert token(M(1.5, -0.2, 0.3)) == "FAIL"       # PnL <= 0   <- the criterion that was missing
    assert token(M(1.5, 0.2, -0.3)) == "FAIL"       # CI lo <= 0
    for bad in (float("nan"),):                      # NaN never reads as a pass on any criterion
        assert token(M(bad, 0.2, 0.3)) == "FAIL"
        assert token(M(1.5, bad, 0.3)) == "FAIL"
        assert token(M(1.5, 0.2, bad)) == "FAIL"


def test_pass_token_is_never_harvestable_as_a_bare_verdict():
    """Grepping any line that carries the token must also return the qualifier — the token is
    never alone on its line, and the line names both 'NOT A VERDICT' and the UNCALIBRATED CI."""
    rb = _load_recompute()

    class M:
        sharpe_decision, pnl, ci_lo_UNCALIBRATED = 1.5, 0.2, 0.3

    lines = rb.single_cell_gate_lines(M(), "fee=0+adv=5bp")
    hits = [ln for ln in lines if "PASS" in ln or "FAIL" in ln]
    assert hits, "the gate must emit a token line at all"
    for ln in hits:
        assert "NOT A VERDICT" in ln
        assert "UNCALIBRATED" in ln
        assert "ONE cell" in ln
        # and the token is never the whole line, nor a lone trailing word after a bare label
        assert ln.strip() not in {"PASS", "FAIL"}
        assert len(ln) > 200, "the qualifier must travel with the token on the same line"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
