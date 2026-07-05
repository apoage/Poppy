# Project Conventions Template

Drop-in conventions for a new codebase. Distilled from how the Trader project ended up working after ~6 months.

This file is meta-documentation: read once, copy structure into a new project, then delete (or keep at `docs/conventions.md` as the project's own spec).

---

## 0. The naming theme

Pick one *coherent, constraining* theme up front, before you have anything to name. The constraint is what makes future references readable.

Trader uses **apothecary / medicinal plants**: Poppy (planner), Bella (generator), Foxglove (pre-train), Yarrow (review), Mint (architectural fix), Aconite (MEV), Iris (data pipeline), Bloom (publishing). The unwritten rule: "careful work with powerful tools, never edgy/dark."

Other workable themes — pick by aesthetic:
- **Constellations** (Orion, Lyra, Vega, …) — works for distributed/parallel systems.
- **Winds** (Mistral, Sirocco, Boreas, …) — works for streaming/realtime.
- **Lighthouses**, **rivers**, **alpine huts**, **bridges**, **gemstones**.

Avoid: Greek gods (overused), military codenames (impersonal), clever puns (age badly).

Each component gets exactly one name. Reuse it everywhere — chat, commits, file names, dashboards.

---

## 1. Directory layout

```
Project/
├── CLAUDE.md                       # router (see §2)
├── AGENTS.md                       # thin portability shim → CLAUDE.md (see §2)
├── README.md                       # public-facing
├── requirements.txt                # loose ranges
├── requirements.lock.txt           # frozen pip output
├── .python-version
├── lore/                           # session notes + ADRs (§3, §4)
│   ├── notes_YYYY-MM-DD.md
│   └── adr/                        # architectural decision records (§4)
│       ├── README.md               # index + MADR template
│       └── NNNN-title.md
├── tasks/                          # feature specs / experiment plans (§5)
│   └── TASK_001_NAME.md
├── docs/                           # reference, designs, setup (§6)
├── prompt-lib/                     # manually-loaded agent prompts (§7)
├── tools/                          # launchers + dashboards (§8)
│   ├── run_001_name.sh
│   └── benchmarks/
├── <component_a>/                  # named after the theme
│   └── CLAUDE.md                   # component-specific facts
├── <component_b>/
│   └── CLAUDE.md
└── <component>_ckpt/ or runs/      # per-task subdirs (§9)
```

`config/`, `tests/`, `scripts/` add as needed but stay out of `tools/`.

---

## 2. `CLAUDE.md` (+ `AGENTS.md`) — router, not manual

Top-level `CLAUDE.md` is loaded on every session by Claude Code. Keep it lean. It should answer:

1. **Identity** — one-sentence statement of what the project is and what the *user* expects of themselves (e.g. "I am an expert ML engineer. I read files before editing.").
2. **Components** — table: Name | Role | Location | Subtree CLAUDE.md.
3. **Top-level layout** — annotated tree of important paths.
4. **Cross-cutting rules** — efficiency philosophy, performance principles, hard constraints, pitfalls. Things that apply everywhere.
5. **Pointers** — where session notes live, where tasks live, where ADRs live, what to read at session start.
6. **Theme statement** — one paragraph on the naming convention so new components fit.

Per-component `CLAUDE.md` files live in their subtree and are auto-loaded by Claude when working in that directory. They hold:
- Component-specific facts (what this module does, key files, invariants).
- Rules unique to this component (e.g. "no attention in reasoning").
- Don't duplicate top-level CLAUDE.md.

**Rule**: top-level CLAUDE.md says "check `lore/` and `tasks/` at session start — do not rely on stale CLAUDE.md or memory." Actively distrust your own documentation.

### `AGENTS.md` — portability shim

`AGENTS.md` at the repo root is a thin pointer at `CLAUDE.md`. It exists so non-Claude tools (Codex CLI, Cursor, Aider, Augment, Continue) — which read `AGENTS.md` per the open standard — can find a recognizable entry point. As of 2026, Anthropic's Claude Code reads `CLAUDE.md` natively but does not natively read `AGENTS.md`; conversely most other tools are the inverse. Maintaining both is one ~30-line file and makes the project tool-portable.

**`CLAUDE.md` is the source of truth.** If conventions ever drift between the two, `CLAUDE.md` wins, and `AGENTS.md` should be updated to match.

---

## 3. `lore/` — session notes

Filename: `notes_YYYY-MM-DD.md` (ISO date, sortable).

### Structure
- **Numbered sections** that continue across files. If yesterday ended at §57, today starts at §58. Reads like a captain's log — you can scroll through history without losing thread.
- Each section has a heading like `## 58. Overnight run outcome (YYYY-MM-DD HH:MM)` — date-stamped, action-oriented.
- Mid-session captures and pre-compact checkpoints, **not** end-of-day summaries.

### What goes in
- Decisions made and *why*.
- Run outcomes with tables (see §10).
- Bugs found, root cause, fix.
- Pivots, dead ends, things you killed and why.
- Pre-launch system state ("RAM 39 GB free, GPU idle, no other processes") — your future-self thanks you.

### What stays out
- Code (it's in git).
- Long task specs (they go in `tasks/`).
- Reference material (it goes in `docs/`).

### End each notes file with a "State at end of session" block

```markdown
## State at end of session

**Nothing running.** GPU/CPU/etc. status.

**Active checkpoints / artifacts**:
| Path | Notable | Note |
| ...

**Recommended next** (priority order):
1. ...
2. ...

**Hard rules — DO NOT**:
- ...

**Files added this session**:
- ...
```

### Session continuation pointer
At the bottom: a short "**Read order when you return**" list with file refs and section numbers (`notes_2026-04-24.md §65–§67`). So a future-you (or a fresh agent) can rejoin cold in 5 minutes.

---

## 4. `lore/adr/` — architectural decision records

Major architectural decisions live here as standalone, atomic, immutable records. Format: **MADR (Markdown Architectural Decision Records)**.

ADRs sit *under* `lore/` (not `docs/adr/` as the mainstream convention suggests) on purpose: the decisions emerge from session narrative, and grouping them with the lore makes their genesis traceable. Cross-references to `lore/notes_*.md §NN` are mandatory inside each ADR.

### Filename
`lore/adr/NNNN-short-kebab-title.md`, zero-padded 4-digit (`0001-foo.md`, `0002-bar.md`). Numbers are monotonic and never reused.

### Required sections
```yaml
---
status: accepted | superseded | deprecated
date: YYYY-MM-DD                  # when the decision was originally made
deciders: <name(s)>
superseded-by: NNNN               # only if status=superseded
---
```

```markdown
# NNNN. Title

## Status
One line.

## Context
What situation forced a decision; what evidence drove it. Cite lore.

## Decision
Declarative voice. "We use FiLM injection post-temporal-pool because…"

## Consequences
What we gain. What we give up. What becomes harder.

## Alternatives considered
What other paths were on the table; why rejected.

## References
Lore sections, task files, papers, other ADRs.
```

### Bar for inclusion
A decision belongs in an ADR only if it meets at least one of:
- **Contested** — alternatives were considered.
- **Non-obvious** — the rationale isn't visible from the code.
- **Load-bearing** — reversing it would force a major rewrite.

Routine implementation choices DO NOT belong in ADRs. Aim for ~5–10 ADRs over the project's life. If you're past 30, you're over-recording.

### Decision uncertainty section
If you write an ADR for a decision whose rationale is genuinely ambiguous from the lore (e.g. the original choice predates the notes), include `## Decision uncertainty` and say so explicitly. Do NOT fabricate rationale.

### `lore/adr/README.md`
An index file at the top of `lore/adr/` with: one-line description of MADR, the template above, and a table of contents listing each ADR by id + title + status.

### Promote, don't duplicate
When you make an architectural decision in a session note, the note keeps the chronological story; the ADR captures the standalone decision with full rationale. The ADR cites the note. Do NOT delete from lore — the narrative is part of memory.

---

## 5. `tasks/` — feature specs

Filename: `TASK_NNN_SHORT_TITLE.md`. Numbers are a global monotonic counter (TASK_001, TASK_002, …, TASK_741). Not per-component, not per-day. Easier to reference in chat.

### Required fields
- **Status** at the top: `PROPOSED` / `IMPLEMENTED` / `BENCHMARKED` / `KILLED`.
- **One-line summary**.
- **Rationale** — why now, what hypothesis, what evidence motivates it.
- **Design** — concrete: which files, which functions, which lines, what changes.
- **Implementation footprint** — file count + LoC estimate. Calibrates effort.
- **Validation gates** — see below.
- **Decision criteria** — what result means "ship it" vs "kill it" vs "escalate."

### Decision gates are binary or near-binary
"`dir_acc > 0.62` → label-side reformulation alive; `≤ 0.58` → escalate to TASK_741." Pre-commit before the run. Avoids motivated reasoning when the number lands ambiguous.

### Specs may include "spec-killer" gates
A binary falsifier means the task can be killed cleanly without ego. "If the cheat re-run still caps at 87%, the FiLM hypothesis is wrong, abandon the design." Saves you from sunk-cost-coding for weeks.

---

## 6. `docs/` — reference and design

Long-form, doesn't churn:
- Research papers (PDFs or summaries).
- Architecture designs.
- Setup guides.
- Deployment specs.
- External-system pointers.

Subfolders by area (`MEV/`, `imprinting/`, `research/`). Files named with `UPPER_SNAKE.md` for first-class docs (`HYBRID_LLM_REASONING_DESIGN.md`), `lower_snake.md` for notes-style.

If something belongs in a session note OR a task spec OR docs, ask: *will this still matter in 3 months?* If yes → docs. If no → notes/tasks.

---

## 7. `prompt-lib/` — manually-loaded agent prompts

Templates for agent invocations you've written carefully and don't want to re-invent: adversarial code reviews, research prompts, audits, ablation drivers.

Each file is a **complete prompt**, ready to paste. Header explains when to use it. Reusable across tasks.

Examples from Trader: `adversarial_bughunt_easy_mode.md`, `encoder_direction_audit.md`, `ceiling_research_prompt.md`.

### `prompt-lib/` vs Claude "skills"
Claude Code's *skill* system (`.claude/skills/*.md`) is similar but auto-injects when keyword/path triggers match. `prompt-lib/` is **manually selected** — you grab the file when you spawn a research agent. Different trigger model:

- **Skill** = "every time the agent touches `/api/`, automatically prepend these 200 lines."
- **prompt-lib entry** = "I'm about to spawn an adversarial reviewer — let me grab `adversarial_bughunt_easy_mode.md` and pass it in."

For research projects where agents are spun up *deliberately* (audit, bug-hunt, deep research), manual selection is correct — auto-injection just bloats context. For product codebases where you want consistent style enforcement everywhere, auto-skills win. Pick the model that matches how you actually work; don't migrate to skills just because the docs push them.

---

## 8. `tools/` — launchers + dashboards

### Launchers: `tools/run_NNN_tag.sh`
One launcher per task. Never paste a 400-char python invocation into chat — it'll mutate silently and you'll lose the audit trail.

Skeleton:
```bash
#!/bin/bash
# TASK_NNN: One-line summary.
# Reference spec:     tasks/TASK_NNN_TITLE.md
# Reference baseline: TASK_MMM Alpha +X.YY% / metric / beats N/M

set -eu -o pipefail
cd "$(dirname "$0")/.."
source <env>/bin/activate
export <ENV_VARS>

# --- Sanity checks ---
if [[ ! -f "${INPUT}" ]]; then echo "ERROR: missing"; exit 2; fi

# --- Args ---
COMMON_ARGS=( ... )

# --- Summary header ---
{ echo "# TASK_NNN Result: ..."; echo "..."; } > "${SUMMARY}"

# --- Train / run ---
python <entry> "${COMMON_ARGS[@]}" > "${OUT_DIR}/run.log" 2>&1 || { ...; exit 1; }

# --- Bench / verify ---
python tools/benchmark.py ... > "${OUT_DIR}/benchmark.log" 2>&1 || { ...; exit 1; }

# --- Parse + write summary table row ---
metric=$(grep ... | awk ...)
echo "| ${TAG} | ${metric} | ... |" >> "${SUMMARY}"
```

Result row goes into `task_NNN_result.md`. One row per launch — easy to diff.

### Dashboards
If you build local dashboards, fix ports per dashboard (`tools/DS24/` → :8050, `tools/DSe/` → :8052) and document them in CLAUDE.md.

### Benchmarks
`tools/benchmarks/` for perf tests, separate from launchers.

---

## 9. Run artifacts

### Per-run directory
`<component>_ckpt_<TASKNUM>_<TAG>/` or `runs/<TASKNUM>_<TAG>/`. Inside:

```
<TASK>_<TAG>/
├── best.pkl                # current best checkpoint
├── ckpt_e{50,100,150}.pkl  # periodic
├── final.pkl               # last epoch
├── config.json             # what we ran
├── metrics.csv             # per-epoch numbers
├── run.log                 # full stdout/stderr (nohup style)
├── train.log               # structured log
├── benchmark.json          # post-run eval
└── benchmark.log
```

Same layout every run. Future-you can grep across runs without writing a parser.

### Failed runs preserved
Suffix the dir (`_dense_sign.eps1e-4/`) and keep it. Never silently delete a failed run — the failure mode is part of the project's memory. Move to a sibling `Project_archive/` only when you've stopped referencing it.

### Result tables — same shape every time
```markdown
| Tag | Metric A | Metric B | Metric C | Sanity | Δ vs ref |
|---|---|---|---|---|---|
| ref | ... | ... | ... | ... | — |
| variant_1 | ... | ... | ... | ... | ±X.Ypp |
```

Picking a fixed shape lets you eyeball deltas across weeks of work.

---

## 10. Memory / persistent notes (if using Claude)

If you have a memory system (Claude's `MEMORY.md` index + per-fact files):

- **Types**: `user`, `feedback`, `project`, `reference`. Never duplicate code or git history.
- **Structure feedback as**: rule → **Why:** (the reason the user gave, often a past incident) → **How to apply:** (when this kicks in).
- **One file per fact**, semantically named (`feedback_jax_vram_preallocate.md`).
- **`MEMORY.md` is an index, not a memory** — one-line pointers, ~150 chars max each.
- **Update or remove** stale memories. Don't let them accumulate.

If not using Claude memory: a `NOTES.md` at the root with the same idea works fine.

---

## 11. Hard rules — "DO NOT" lists

Each notes file ends with explicit prohibitions. Examples that have actually fired:

- Do NOT silently remove component X based on a single-run result within the noise band.
- Do NOT benchmark a leaked-input checkpoint as production.
- Do NOT bump pinned packages without a task spec explaining why.
- Do NOT bypass safety rails with `--no-verify` or equivalent.

The point is: future-you reads these before acting. Forces a re-read of context.

---

## 12. Reproducibility hygiene

- **`.python-version`** — pinned (project root).
- **`requirements.txt`** — loose ranges, human-edited.
- **`requirements.lock.txt`** — `pip freeze` output. Refresh before snapshots/deployments.
- **`START_AFTER_REBOOT.txt`** — for system-state recovery (which services to start, where logs go).
- **One env per project** — `<project>_env/` next to source. Never share venvs across projects.

---

## 13. Anti-patterns called out by name

Codify the things you keep almost-doing. From Trader CLAUDE.md:

- Pre-generating training data to disk then loading (memory hierarchy).
- Mocking the database in integration tests (incident-derived).
- Silently bumping pinned dependencies.
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` (degrades JIT).

Each anti-pattern in CLAUDE.md should have a *why* — usually a past failure. New developers (and Claude) read these and don't re-step on the rake.

---

## 14. Decision-driven, not feature-driven

Trader's tasks aren't "add feature X" — they're "answer question Y."

- TASK_733 = "is the dir_acc ceiling architectural or informational?" (binary diagnostic)
- TASK_741 = "does FiLM injection break the 87% architectural ceiling?" (binary falsifier)
- TASK_737 = "which heads matter?" (4-way ablation)

Frames work as *experiments with pre-committed gates* rather than *implementations*. You finish a task with an answer, not just code.

---

## 15. Six-month sanity checks (advisory)

Once a quarter, sweep:

- `lore/notes_*.md` count > 50? Consider quarterly archive (`lore/2026Q1/`).
- Tasks all `IMPLEMENTED` but never `BENCHMARKED`? You stopped reading your own gates.
- More than ~5 stale checkpoint dirs? Move to `Project_archive/`.
- Memory index > 200 entries? Trim.
- ADR list > 30? You're recording decisions that don't meet the bar — review and demote some to lore-only.
- CLAUDE.md hasn't been edited in 3 months but the project has changed? Update or delete sections.
- AGENTS.md drifted from CLAUDE.md? Re-sync (CLAUDE.md wins).

---

## 16. The cultural rule under all of this

**Write so a stranger (or future-you, post-vacation) can pick up cold.** Every notes file, every task spec, every CLAUDE.md is read by someone with no context. Date-stamped numbered sections, explicit gates, "read order when you return" pointers, "DO NOT" lists — all of it exists to compress months of context into a 10-minute orientation.

The cost of writing it down is small. The cost of *not* writing it down compounds quickly once the project is more than a few weeks old.

---

## Quick start for a new project

```bash
mkdir NewProject && cd NewProject
mkdir -p lore/adr tasks docs prompt-lib tools
touch CLAUDE.md AGENTS.md README.md requirements.txt
echo "3.10.14" > .python-version

# CLAUDE.md: identity, components (table), layout, cross-cutting rules, theme statement.
# AGENTS.md: 10-line shim pointing at CLAUDE.md + build/run/test commands + hard rules.
# lore/notes_$(date +%F).md: write §1.
# lore/adr/README.md: MADR template + empty ToC. ADRs come later, when load-bearing
#                     decisions are actually made.
# tasks/TASK_001_<first_thing>.md: binary decision gate.
# tools/run_001_<tag>.sh: skeleton from §8.

git init && git add -A && git commit -m "Initial scaffold"
```

Then write code.
