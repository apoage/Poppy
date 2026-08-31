---
description: Hard prohibitions for any agent working in the shared ~/Trader tree. Deliberately NOT path-scoped — these must be loaded whenever any agent edits anything.
---

# Shared-tree concurrency — hard prohibitions

`~/Trader` is a **shared** working tree. The dark factory routinely runs 3–8 subagents in it
concurrently. Every rule below exists because it was broken once and cost real work.
**These bind implementers, judges, and verifiers alike — not just reviewers.**

## 1. Never `git stash` / `checkout` / `restore` in the live tree

For any A/B against HEAD (byte-identity gates, "did my change move a number"), build a detached
worktree: `git worktree add --detach <scratch> HEAD`. Never mutate HEAD in `/home/lukas/Trader`.

*Incident (factory run 45, 2026-08-27):* a TASK_881E **implementer** ran `git stash` on the live
tree with five other agents mid-edit. No damage resulted only because it happened to scope the
stash to its own two paths; a blanket `git stash` would have stripped a concurrent fix pass
mid-write. An earlier run corrupted a live review the same way, with five agents reading a file
that flickered between HEAD and candidate. This is a **concurrency** hazard, not a reviewer
hazard.

Recovery facts, because the incident is survivable:
- A scoped `git stash push <paths>` still prints the generic `WIP on master: <sha>` message, so
  the message does **not** tell you the blast radius. Use `git stash show --stat stash@{0}`.
- The tree carries ~10 long-lived **halt-preservation stashes** holding the only record of
  several HALTED tasks (897/891R/881/891/888/886/883/876/880). They are irreplaceable. An
  accidental stash pushes them all down one index, so any agent told to "pop the stash" by index
  can destroy one. Fence a recovering agent to `stash@{0}` explicitly and say the others are off
  limits.

## 2. Never regenerate the verification fixture as a side effect

`fixtures/trading_fixture.db` is **gitignored**, so `git status --short` — the check the whole
ship ritual leans on — cannot see it change. Regenerating it is therefore a **silent re-base of
every baseline** promoted against it. No agent may regenerate it as a side effect of any task; a
deliberate re-base is planner-scheduled work with its own queue row.

*Incident (run 25, 2026-08-05):* a review agent regenerated the fixture mid-run
(`b0f8de8b` → `59525bd3`) while every committed baseline had been promoted against `b0f8de8b`.
Restored only by luck. The clean-tree gate reported green throughout.

If you touched fixture-adjacent code, state the fixture's sha256 explicitly rather than relying
on the clean-tree check.

## 3. Sub-agents never deploy

Only the main agent runs `tools/deploy/deploy.sh`. Sub-agents may commit locally and write
reports; they must never deploy, and never write server-side `runs/<id>/` via ssh/scp.

**Why:** `deploy.sh` builds the training branch from the **workstation working tree, uncommitted
state included**. If two sub-agents are mid-edit and one deploys, the other's in-flight edits
ship unreviewed. This happened 2026-05-23: a Batch 4 sub-agent deployed while the TASK_768
sub-agent was still editing `jax_encoder.py`. The bit-identity gate caught it — luck, not design.

Spawn at most one code-modifying sub-agent at a time on overlapping paths. Disjoint file scopes
may run in parallel provided none of them deploys.

## 4. Planner does not write shared surfaces while the factory is ARMED

While armed, the planner session does not write `tasks/*.md`, `tasks/DECISIONS.md`, or
`tasks/QUEUE.md`. It documents in its own surfaces (`lore/`, the operator reply) and applies the
edits **at disarm**.

**Why:** in run 23 (2026-07-31) the planner ruled D31 and authored TASK_868/869 mid-run. The
orchestrator saw uncommitted, unattributed writes on the decision surface, could not attribute
them, and correctly **quarantined** them — reverting D31 to OPEN and flagging both specs
"UNAUTHORISED DRAFT". Mid-run unattributed writes to the decision surface are indistinguishable
from tampering. The quarantine was the system working.

## 5. The stop gate is blind to background subagents

The `stop_gate` (PreToolUse on Stop) sees only "armed" and "tree dirty". It cannot see background
subagents, so "armed, idle" and "armed, five agents mid-flight, tree dirty from their work" are
indistinguishable to it and both get blocked.

When it fires with agents genuinely running: do bounded, read-only, non-conflicting work
(baselines, corpus measurements, load checks) and say plainly that the gate is misreading an
async pattern. **Do not** take an extra queue row to satisfy it — that breaches the deliberate
concurrency cap. **Do not** disarm early to escape it; disarming is the wrap-up's last step.

Related: the `bash_gate` also matches `git tag` on the command string alone, so a read-only
`git tag --sort=... | head` is refused when the verdict is stale. That is conservatism, not a
defect — get tag conventions from `git log` or the queue instead of routing around it.
