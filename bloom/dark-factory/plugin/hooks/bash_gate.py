#!/usr/bin/env python3
"""dark-factory bash gate — PreToolUse on Bash.

Three jobs:
  1. Arm / disarm. `true # dark-factory-arm` records THIS session as the implementer
     and arms every guard; `true # dark-factory-disarm` releases them. The hook owns
     the state file — the model can only ask, never write the state itself.
  2. No-push wall. While armed, `git push` is denied outright (commits/tags stay
     local) unless guards.allow_push is explicitly true in config.
  3. Commit gate. While armed, `git commit` / `git tag` require a FRESH, GREEN
     verdict file (.dark-factory/verdicts/current.json) written after fan-out
     verification. Missing / stale / not-green → deny, fail closed.

Disarmed projects and projects without a dark-factory config are untouched.
"""
import datetime
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from df_common import (read_input, find_root, load_json, save_json, config_path,
                       state_path, verdict_path, get_state, deny, allow)

ARM = "dark-factory-arm"
DISARM = "dark-factory-disarm"


def main():
    data = read_input()
    cmd = (data.get("tool_input") or {}).get("command") or ""
    root = find_root(data.get("cwd"))

    if DISARM in cmd:
        if root:
            state = get_state(root)
            state.update({"armed": False, "implementer_session": None,
                          "disarmed_at": datetime.datetime.now().isoformat(timespec="seconds")})
            save_json(state_path(root), state)
        allow()

    if ARM in cmd:
        if not root:
            deny("dark-factory: no .dark-factory/config.json found in or above %r — "
                 "run the skill's setup step first (it writes the config from the "
                 "template after the owner confirms)." % (data.get("cwd") or "?"))
        save_json(state_path(root), {
            "armed": True,
            "implementer_session": data.get("session_id"),
            "armed_at": datetime.datetime.now().isoformat(timespec="seconds"),
        })
        allow()

    if not root:
        allow()
    state = get_state(root)
    if not state.get("armed"):
        allow()

    cfg = load_json(config_path(root), {}) or {}
    guards = cfg.get("guards", {})

    if re.search(r"\bgit\b[^\n;|&]*\bpush\b", cmd) and not guards.get("allow_push", False):
        deny("dark-factory git gate: PUSH IS FORBIDDEN while the factory is armed — "
             "commits and tags stay local; the owner pushes after morning review. "
             "(Override only via guards.allow_push in .dark-factory/config.json.)")

    if re.search(r"\bgit\b[^\n;|&]*\b(commit|tag)\b", cmd) and guards.get("require_verdict", True):
        if data.get("session_id") != state.get("implementer_session"):
            deny("dark-factory git gate: the factory is armed — commits/tags belong to "
                 "the implementer session. If you are the planner, document instead; "
                 "if the factory is done, disarm first: true # dark-factory-disarm")
        vp = verdict_path(root)
        verdict = load_json(vp)
        if verdict is None:
            deny("dark-factory git gate: no verdict file at .dark-factory/verdicts/"
                 "current.json — run the fan-out verification for this task and write "
                 "the verdict file (schema in the dark-factory skill), then commit.")
        if verdict.get("green") is not True:
            deny("dark-factory git gate: verdict for %r is NOT green — fix the findings, "
                 "re-verify what changed, and rewrite the verdict file before committing."
                 % verdict.get("task", "?"))
        max_age_min = guards.get("verdict_max_age_minutes", 60)
        try:
            age_min = (datetime.datetime.now().timestamp() - os.path.getmtime(vp)) / 60.0
        except OSError:
            age_min = None
        if age_min is None or age_min > max_age_min:
            deny("dark-factory git gate: verdict file is STALE (%s min old, max %s) — "
                 "verification must be fresher than the code it approves. Re-run the "
                 "panel and rewrite the verdict file."
                 % ("?" if age_min is None else int(age_min), max_age_min))
    allow()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        sys.exit(0)  # fail open on unexpected errors — arming/gating bugs must not brick the shell
