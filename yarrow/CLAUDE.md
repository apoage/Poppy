# yarrow — Multi-model Code Review

Design-stage side project. **No executable code yet** beyond `tools/` scaffolding and
`workspace/` for run artifacts. Everything in the top level is documentation.

## What goes where

| File | Purpose |
|---|---|
| `vision.md` | Why yarrow exists |
| `architecture.md` | System diagram + data flow |
| `agents.md` | Role definitions (Apothecary, Apprentice, Compounder, 7 Harvesters), precision priors |
| `workflow.md` | The loop: scope → harvest → cluster → prescribe → approve → execute → verify |
| `communication.md` | Inter-role message protocol |
| `execution.md` | **Runtime mapping** — how roles run as Claude Code sub-agents / MCP / skill. Read this before implementing. |
| `orchestrator_prompt.md`, `assistant_prompt.md`, `executor_prompt.md` | Full role prompts (future sub-agent bodies) |
| `harvester_{claude,gemini,glm,kimi,deepseek,openai,minimax}_prompt.md` | Per-model harvester prompts |
| `backends.md` | Which models available, how to call them |
| `README.md` | Project summary |
| `tools/` | Future harvester invocation scripts (bash or MCP backing) |
| `workspace/` | Run artifacts: `findings_*.json`, `clusters_*.json`, `agent_stats.json` |

## Before editing

- **Role prompts** (`*_prompt.md`) are the canonical definitions. If implementation diverges, fix implementation to match prompt — not vice versa. These are the spec.
- **`agents.md` precision priors** (e.g. `claude: 0.73`, `gemini: 0.30`) are starting estimates. Updated empirically via `workspace/agent_stats.json` once runs happen.
- **Do not invent a parallel task system.** Prescriptions are regular `tasks/TASK_XXX_*.md` files in the project root, by design. See `execution.md`.

## Implementation path (not started)

Per `execution.md` phasing:
1. `.claude/agents/apprentice.md`, `compounder.md`, `harvester-claude.md` — transplant role prompts into sub-agent frontmatter
2. `.claude/skills/yarrow/SKILL.md` — encodes the loop, invokable as `/yarrow scan <path>`
3. MCP server `yarrow-harvesters` — exposes Gemini / GLM / Kimi / DeepSeek / OpenAI / MiniMax as tools

Main Claude thread plays the Apothecary role; sub-agents/MCP tools cover the rest.

## Status

- Conceptual design: complete (end of January 2026)
- Runtime mapping: complete (`execution.md`, April 2026)
- Implementation: not started. Parked pending bandwidth; pipeline (Foxglove → Poppy)
  research takes priority.
