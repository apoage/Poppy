# Yarrow Execution — Mapping to Claude Code Primitives

This document is the *how-it-runs* layer on top of the conceptual design in
`architecture.md`, `agents.md`, and `workflow.md`. Those docs define the roles
and the loop; this one says which Claude Code primitives execute each part.

## Status

Conceptual design: end of January 2026. By April 2026 the Claude Code
primitives below (sub-agents, skills, MCP, hooks) are all available and fit
yarrow's design cleanly. The architecture is not invalidated — it's executable
with less custom glue than originally planned.

## Role → Primitive Mapping

| Yarrow role | Runs as | Location | Notes |
|---|---|---|---|
| Apothecary (Orchestrator) | Main Claude Code thread | N/A — it's you + main model | Dispatches sub-agents, reads workspace, writes prescriptions |
| Apprentice (Assistant) | Sub-agent | `.claude/agents/apprentice.md` | Haiku, Read-only, deterministic cluster output |
| Compounder (Executor) | Sub-agent | `.claude/agents/compounder.md` | Opus, Edit+Bash+Read, scoped to one task spec |
| Harvester: Claude | Sub-agent, parallel-instanced | `.claude/agents/harvester-claude.md` | Multiple concurrent invocations in one dispatch message |
| Harvester: Gemini / GLM / Kimi / DeepSeek / OpenAI / MiniMax | MCP tools | `yarrow-harvesters` MCP server | One tool per external model, API-backed |
| The loop itself | Skill | `.claude/skills/yarrow/SKILL.md` | Invoked by `/yarrow scan <path>` |
| Prescription format | Existing `tasks/TASK_XXX_*.md` | `tasks/` | Reuse project task system — don't invent a parallel one |

## Why sub-agents fit

Each sub-agent invocation gets:
- Isolated context window (no pollution of main thread)
- Own system prompt (yarrow's role prompts transplant directly into frontmatter + body)
- Own tool allowlist (enforce the constraints in `agents.md` — e.g., Apprentice never gets Edit)
- Fresh state each call (stateless role execution matches yarrow's disk-as-bus design)

**Parallel fan-out is free.** One dispatch message with N `Agent` calls to
`harvester-claude` runs N concurrent scans. No queue, no wrapper script.

## Why external harvesters need MCP

Claude Code sub-agents run on Claude models only. For the cross-model ensemble
(Gemini, GLM, Kimi, DeepSeek, OpenAI, MiniMax) there are two options:

1. **MCP server** (`yarrow-harvesters`) — each model exposed as a tool:
   `harvest_gemini(file_range, focus) -> finding[]`. The Apothecary calls them
   in parallel alongside native Claude harvesters in the same dispatch.
2. **Bash wrappers** (`yarrow/tools/harvest_<model>.sh`) invoked via Bash tool.
   Simpler; less uniform with native harvesters.

MCP is preferred because it keeps the dispatch uniform: all harvesters look
like tools from the Apothecary's POV.

## Loop execution (Skill)

`.claude/skills/yarrow/SKILL.md` encodes the workflow from `workflow.md` as
executable instructions for the main model. Invocation:

```
/yarrow scan mcts_trader/jax_agent/
```

Phases, in order (each phase is a dispatch from the main thread):

1. **Scope** — Apothecary (main) reads target, decides focus areas per harvester
2. **Harvest** — one message, parallel `Agent` calls to all harvesters (native + MCP). Output → `yarrow/workspace/findings_<scan_id>.json`
3. **Cluster** — `Agent` call to Apprentice. Output → `yarrow/workspace/clusters_<scan_id>.json`
4. **Validate + Prescribe** — Apothecary (main) reads clusters, validates against source, writes `tasks/TASK_XXX_*.md` files
5. **Gear shift** — user approves (plan → approval → yolo)
6. **Execute** — `Agent` call to Compounder, one task at a time
7. **Verify** — Apothecary reads compounder's diff + test output, marks task done or sends back

## What's not native

- **No workflow engine.** The skill is instructions; the main model is the
  executor. If the main model drops the ball mid-loop (compaction, user
  interrupt), workspace files on disk are the only recovery state. This
  matches yarrow's original assumption — findings DB is the bus.
- **No persistent memory between phases.** Each sub-agent invocation is fresh.
  All context transfer happens through disk artifacts under `yarrow/workspace/`.
- **Sub-agents can spawn sub-agents, but keep the hierarchy flat.** Main =
  Apothecary; everyone else is one level down. Don't let harvesters dispatch
  harvesters.
- **Dispatch is synchronous from the main thread.** Apothecary waits on the
  parallel harvest to complete, then moves to clustering. No true async /
  event-driven execution. Fine for yarrow's scale.

## Precision tracking

`yarrow/workspace/agent_stats.json` persists precision calibration across
scans. The Apprentice reads it for ensemble scoring; the Apothecary updates
`confirmed` counts after human validation of prescriptions (or after executor
confirms a fix actually resolved the issue).

This is plain JSON — no primitive needed. Just a file the Apprentice and
Apothecary both know how to read/write.

## Session integration

Yarrow scans are session events worth capturing in `lore/notes_*.md`:
- What was scanned, with scan_id
- How many findings, how many clusters, how many prescriptions
- Which tasks the scan generated
- Harvester precision updates

The session-note hook (PreCompact / manual `/note`) can pull scan_ids from
`yarrow/workspace/` modified-since-session-start and include them in the
materials block. The interpretation (was the scan useful? which harvester
over-fired?) stays human-authored.

## Minimum viable path

Phase 1 — get the loop running Claude-only:
- `.claude/agents/apprentice.md`
- `.claude/agents/compounder.md`
- `.claude/agents/harvester-claude.md`
- `.claude/skills/yarrow/SKILL.md`
- `/yarrow scan <path>` works end-to-end with one Claude harvester

Phase 2 — diversity:
- MCP server with two external models (start with DeepSeek + Gemini —
  highest-precision non-Claude per `agents.md` priors)
- Apprentice uses `agent_stats.json` for weighted scoring

Phase 3 — full ensemble:
- Remaining external harvesters via MCP
- Precision tracking loop: Apothecary writes confirmed-count updates after
  prescriptions land

---

*Conceptual design: `architecture.md`, `agents.md`, `workflow.md`, `vision.md`.
Communication protocol between roles: `communication.md`. This document:
execution on Claude Code primitives.*
