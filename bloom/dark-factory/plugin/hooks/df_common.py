"""Shared helpers for dark-factory hooks.

stdlib only. Guards fail OPEN on unexpected errors (a broken guard must not brick
the session) — except the git gate's verdict check, where failing closed is the point.

State model:
  <root>/.dark-factory/config.json   committed, planner-owned, per-project settings
  <root>/.dark-factory/state.json    runtime: {armed, implementer_session, armed_at}
  <root>/.dark-factory/verdicts/current.json   written after fan-out verification
"""
import json
import os
import sys


def read_input():
    try:
        return json.load(sys.stdin)
    except Exception:
        return {}


def find_root(start):
    """Walk up from `start` to the first dir containing .dark-factory/config.json."""
    d = os.path.abspath(start or os.getcwd())
    while True:
        if os.path.isfile(os.path.join(d, ".dark-factory", "config.json")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def load_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def config_path(root):
    return os.path.join(root, ".dark-factory", "config.json")


def state_path(root):
    return os.path.join(root, ".dark-factory", "state.json")


def verdict_path(root):
    return os.path.join(root, ".dark-factory", "verdicts", "current.json")


def get_state(root):
    return load_json(state_path(root), {}) or {}


def deny(reason):
    """PreToolUse deny — the reason is fed back to the model."""
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }))
    sys.exit(0)


def allow():
    sys.exit(0)
