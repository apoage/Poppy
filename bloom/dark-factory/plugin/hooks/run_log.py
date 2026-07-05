#!/usr/bin/env python3
"""dark-factory run log — PostToolUse on Write|Edit|MultiEdit|NotebookEdit|Bash.

While armed, appends one JSONL line per mutation/command to
.dark-factory/run-YYYYMMDD.jsonl — the machine-readable trail the planner session
reconciles next morning without re-reading a whole transcript. Silent otherwise.
"""
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from df_common import read_input, find_root, get_state


def main():
    data = read_input()
    root = find_root(data.get("cwd"))
    if not root:
        return
    state = get_state(root)
    if not state.get("armed"):
        return

    tool = data.get("tool_name") or "?"
    ti = data.get("tool_input") or {}
    if tool == "Bash":
        target = (ti.get("command") or "")[:160]
    else:
        target = ti.get("file_path") or ti.get("notebook_path") or ""

    now = datetime.datetime.now()
    entry = {
        "ts": now.isoformat(timespec="seconds"),
        "role": "implementer" if data.get("session_id") == state.get("implementer_session") else "other",
        "tool": tool,
        "target": target,
    }
    path = os.path.join(root, ".dark-factory", "run-%s.jsonl" % now.strftime("%Y%m%d"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
    sys.exit(0)
