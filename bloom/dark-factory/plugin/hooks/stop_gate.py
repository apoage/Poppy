#!/usr/bin/env python3
"""dark-factory stop gate — Stop hook.

While the factory is armed, the implementer session cannot simply end its turn:
either there is queue work left (keep going), or only human-decision walls remain —
then the wrap-up ritual must run (banners, handoff doc, clean tree, wrap-up message)
and the factory must DISARM before stopping. Turns the exit ritual from a convention
into a mechanism.

Respects stop_hook_active (never loops). Other sessions stop freely.
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from df_common import read_input, find_root, load_json, config_path, get_state


def main():
    data = read_input()
    if data.get("stop_hook_active"):
        return
    root = find_root(data.get("cwd"))
    if not root:
        return
    state = get_state(root)
    if not state.get("armed"):
        return
    if data.get("session_id") != state.get("implementer_session"):
        return

    cfg = load_json(config_path(root), {}) or {}
    queue = (cfg.get("paths") or {}).get("queue", "tasks/QUEUE.md")

    dirty = ""
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                             capture_output=True, text=True, timeout=10)
        lines = [l for l in out.stdout.splitlines() if l.strip()]
        if lines:
            shown = ", ".join(l.split()[-1] for l in lines[:8])
            dirty = " Tree is DIRTY (%d files: %s%s)." % (
                len(lines), shown, ", …" if len(lines) > 8 else "")
    except Exception:
        pass

    reason = (
        "dark-factory stop gate: the factory is still ARMED. Either (a) take the next "
        "unblocked task from %s, or (b) if only human-decision walls / missing real data "
        "remain, run the wrap-up ritual — banner every shipped task, write the dated "
        "handoff doc, leave a clean tree, post the wrap-up message —%s then disarm by "
        "running: true # dark-factory-disarm — and stop." % (queue, dirty)
    )
    print(json.dumps({"decision": "block", "reason": reason}))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass  # fail open — never trap a session in a broken gate
    sys.exit(0)
