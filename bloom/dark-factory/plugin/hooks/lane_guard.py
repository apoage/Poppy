#!/usr/bin/env python3
"""dark-factory lane guard — PreToolUse on Write|Edit|MultiEdit|NotebookEdit.

Enforces single-writer file lanes MECHANICALLY while the factory is armed.
Fast no-op when the project has no dark-factory config or the factory is disarmed,
so having the plugin enabled costs nothing outside factory runs.

While armed:
  - the implementer session may not write planner-owned areas (docs/, prototype/, ...),
    except files whose name marks them as handoff/validation docs;
  - every OTHER session is held to planner lanes (planner-owned + shared) — this is
    the guard against the classic edit race: a planner session "just fixing one typo"
    in code the factory is mid-flight on.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from df_common import read_input, find_root, load_json, config_path, get_state, deny, allow


def main():
    data = read_input()
    root = find_root(data.get("cwd"))
    if not root:
        allow()
    state = get_state(root)
    if not state.get("armed"):
        allow()

    tool_input = data.get("tool_input") or {}
    fp = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
    if not fp:
        allow()
    if not os.path.isabs(fp):
        fp = os.path.join(data.get("cwd") or root, fp)
    fp = os.path.abspath(fp)
    if not fp.startswith(root + os.sep):
        allow()  # outside the project (scratchpad etc.) — not our lane to police
    rel = os.path.relpath(fp, root).replace(os.sep, "/")

    cfg = load_json(config_path(root), {}) or {}
    lanes = cfg.get("lanes", {})
    planner_owns = lanes.get("planner_owns", ["docs/", "prototype/", ".claude/"])
    shared = lanes.get("shared", ["tasks/"])
    doc_exceptions = lanes.get("implementer_doc_exceptions", ["handoff", "validation"])

    is_implementer = data.get("session_id") == state.get("implementer_session")

    if rel.startswith(".dark-factory/"):
        if rel == ".dark-factory/config.json" and is_implementer:
            deny("dark-factory lane guard: config.json is planner-owned — the factory "
                 "reads it, never edits it. Seed a DECISIONS row if a setting is wrong.")
        allow()  # runtime state/verdicts/logs are fair game

    def under(prefixes):
        return any(rel.startswith(p.rstrip("/") + "/") or rel == p.rstrip("/")
                   for p in prefixes)

    if is_implementer:
        if under(planner_owns):
            name = os.path.basename(rel).lower()
            if not any(k in name for k in doc_exceptions):
                deny("dark-factory lane guard: '%s' is PLANNER lane (%s). The factory "
                     "documents instead of editing here — write a dated handoff/validation "
                     "doc (filename containing one of %s) or seed a DECISIONS row."
                     % (rel, ", ".join(planner_owns), doc_exceptions))
    else:
        if not (under(planner_owns) or under(shared)):
            deny("dark-factory lane guard: the factory is ARMED and owns the code lanes. "
                 "'%s' is outside the planner lanes (%s). Document the change as a task "
                 "file instead — or, if this is deliberate, disarm first by running: "
                 "true # dark-factory-disarm" % (rel, ", ".join(planner_owns + shared)))
    allow()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        sys.exit(0)  # fail open — a broken guard must not brick the session
