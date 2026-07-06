# AGENTS.md

This project's primary configuration is in **[CLAUDE.md](CLAUDE.md)**. Read that first — it is the source of truth for project identity, components, layout, and rules.

This file exists so that non-Claude AI coding tools (Codex CLI, Cursor, Aider, etc.) can find a recognizable entry point. The conventions are identical; only the filename differs.

## Quick orientation

1. **Read `CLAUDE.md`** at the repo root — components, top-level layout, cross-cutting rules, naming theme.
2. **Read the per-component CLAUDE.md** in the subtree you're working in (`mcts_trader/`, `diff_teacher/`, `tools/orchestration/`, `yarrow/`). These are auto-loaded by Claude when you're inside the subtree; for other tools, load explicitly.
3. **Read `lore/notes_YYYY-MM-DD.md`** — most recent first. The numbered sections continue across files. The latest "State at end of session" block tells you what's running, what's next, and what NOT to do.
4. **Read relevant `tasks/TASK_NNN_*.md`** — feature specs with binary decision gates. Status field at top: `PROPOSED` / `IMPLEMENTED` / `BENCHMARKED` / `KILLED`.
5. **Architectural decisions live in `lore/adr/`** (Markdown ADR format). Read these before proposing major changes.
6. **`docs/`** has long-form references (research, designs, deployment). `docs/conventions_template.md` describes the project's working conventions in detail.

## Activate environment

```bash
cd /home/lukas/Trader
source trader_env/bin/activate
```

## Hard rules (do not violate)

- Pinned: `optax==0.2.6` (semi-private API used).
- `numpy<2` (TF / numba compat).
- JAX env: keep `XLA_PYTHON_CLIENT_PREALLOCATE=true` (default), only set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.7`. Do **not** force `=false` or `XLA_PYTHON_CLIENT_ALLOCATOR=platform`.
- Use `threading.RLock()` not `Lock()` (reentrant contexts).
- Do not silently delete failed runs — preserve with a suffixed directory.
- Do not benchmark a `*_cheat/` checkpoint as production — safety rail in `tools/benchmark_easy.py` will refuse it; do not bypass.

## Where to put new work

| Kind of work | Goes in |
|---|---|
| Code | the relevant component (`mcts_trader/`, `diff_teacher/`, etc.) |
| Launcher script for a task | `tools/run_NNN_tag.sh` |
| Feature spec / experiment plan | `tasks/TASK_NNN_TITLE.md` |
| Session note / decision capture | `lore/notes_YYYY-MM-DD.md` (continue numbering from last file) |
| Major architectural decision | `lore/adr/NNNN-title.md` (MADR format) |
| Reusable agent prompt (manually loaded) | `prompt-lib/` |
| Long-form reference / research | `docs/` |
| Per-task run artifacts | `<component>_ckpt_<TASK>_<TAG>/` (best.pkl, configs, logs, benchmark.json) |

## Tool portability note

Anthropic's Claude Code reads `CLAUDE.md` natively and does not natively read `AGENTS.md` (as of 2026). Other tools (Codex CLI, Cursor, Aider, Augment, Continue) read `AGENTS.md` per the open standard. Both files exist intentionally — `CLAUDE.md` is the primary source, this file is a thin compatibility layer pointing at it.

If conventions ever diverge between this file and `CLAUDE.md`, **`CLAUDE.md` wins**.
