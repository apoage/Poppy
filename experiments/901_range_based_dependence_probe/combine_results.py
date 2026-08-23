#!/usr/bin/env python3
"""Merge the gpu-server synthetic-sweep output (`RESULTS_srv.json`) with the workstation
real-data leg (`REAL_LEG.json`, see `run_real_leg.py`'s docstring for why gate (iii) runs there
instead) into one `RESULTS.json` — the single machine-readable artifact `measured_findings.py`'s
pinned table is curated from. (There is no `REPORT.md` in this row: the prose artifact IS
`measured_findings.py`, per this repo's convention of keeping measured evidence as pinned code.)"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()


def main() -> int:
    srv_path = _THIS.parent / "RESULTS_srv.json"
    real_path = _THIS.parent / "REAL_LEG.json"
    out_path = _THIS.parent / "RESULTS.json"
    srv = json.loads(srv_path.read_text())
    real = json.loads(real_path.read_text())
    srv["real_leg"] = real["real_leg"]
    srv["real_leg_runtime_s"] = real["runtime_s"]
    out_path.write_text(json.dumps(srv, indent=2, default=str))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
