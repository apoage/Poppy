#!/usr/bin/env python3
"""Integration smoke of run_grid.run() end-to-end, with loader.load_panel/apply_liquidity_
filter MONKEYPATCHED to a synthetic panel (no fixture DB dependency -- catches wiring bugs
between loader/grid/gates/run_grid before the real fixture smoke on gpu-server)."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import loader     # noqa: E402
import run_grid   # noqa: E402
from test_grid_smoke import _synthetic_panel  # noqa: E402


def test_run_grid_end_to_end_synthetic(monkeypatch):
    panel, day_ts = _synthetic_panel(T=800, A=10, seed=7, day0="2020-06-01")
    syms = [f"SYN{i}" for i in range(10)]

    def fake_load_panel(*a, **k):
        return {**panel, "day_ts": day_ts, "ids": list(range(10)), "syms": syms,
                "loader_info": {}, "seam_block": {"mismatches_n": 0, "overlap_cells_total": 0},
                "fixture_sha256": "deadbeef", "loader_sha256": loader.module_sha256()}

    def fake_apply_liquidity_filter(p, **k):
        return p

    monkeypatch.setattr(loader, "load_panel", fake_load_panel)
    monkeypatch.setattr(loader, "apply_liquidity_filter", fake_apply_liquidity_filter)

    with tempfile.TemporaryDirectory() as td:
        summary = run_grid.run(td, n_cells=4, first_test_date="2022-01-01",
                                last_date="2022-06-30")
        assert summary["n_cells_run"] == 4
        assert (Path(td) / "DONE").exists()
        assert (Path(td) / "grid_results.json").exists()
        assert (Path(td) / "mde_table.json").exists()
        payload = json.loads((Path(td) / "grid_results.json").read_text())
        assert len(payload["cells"]) == 4
        for row in payload["cells"]:
            assert "ga_stats" in row and "ga_pass" in row


if __name__ == "__main__":
    import types
    class _MP:
        def setattr(self, obj, name, val):
            setattr(obj, name, val)
    test_run_grid_end_to_end_synthetic(_MP())
    print("PASS test_run_grid_end_to_end_synthetic")
