# Research prompt — autonomous code-loop & agent-roster frameworks vs "dark-factory"

*Paste this into a deep-research agent (e.g. Compass / deep-research). It is self-contained — the
agent has no prior context. Tune the Tier-1 list if you want to add/drop frameworks. Output should
be a critical, source-grounded report, not a survey of marketing copy.*

---

You are researching frameworks, skills, and tools that solve the same problems as **dark-factory**,
so its author can match ecosystem conventions, steal good patterns, and avoid known anti-patterns
when rebuilding it as a clean, portable skill.

## Context — what dark-factory is

dark-factory is an autonomous implementer loop packaged as a Claude Code skill. Core design:

- **Two-thread contract** — an interactive PLANNER session writes tasks/specs/docs/prototypes; a
  separate autonomous IMPLEMENTER session writes code/tests/version/commits. **Single-writer file
  areas** prevent edit races between the two concurrent sessions.
- **Task-file-as-contract** — the implementer builds one task file's spec faithfully; it does not
  re-derive scope or gold-plate.
- **Fan-out adversarial verification, never self-review** — 2–3 subagent "judges" each with ONE
  named lens (correctness / fail-closed / regression / security), refutation-first; a flow-verifier
  runs the real code path and inspects artifacts; a visual-judge does headless + Playwright +
  screenshot judgment.
- **DECISIONS.md as a wall** — the loop stops cleanly at human-decision boundaries rather than
  guessing product decisions.
- **Release ritual** — version bump, changelog, commit, tag, build; STATUS banners on task files
  are the source of truth over the queue index.
- **Validation tiers** — instant-load fixture first; expensive real-data runs are scheduled
  (overnight one-shots), never polled in-loop.

It runs on Claude Code (a skill + a `.claude/agents/` roster) but is currently overfit to one
project. The goal is a project-neutral version.

## What to research (priority order)

**Tier 1 — agent-roster / autonomous-loop frameworks (cover all):**
- BMAD-METHOD (`bmad-code-org/BMAD-METHOD`)
- Aider architect mode (`Aider-AI/aider`)
- Roo Code custom modes (`RooCodeInc/Roo-Code`) and Cline
- SuperClaude
- OpenHands (`All-Hands-AI/OpenHands`, ex-OpenDevin)
- SWE-agent (`princeton-nlp/SWE-agent`)
- Devin / Cognition
- Cursor and Windsurf agent modes

**Tier 2 — distribution & ecosystem:**
- Community `.claude/agents/` collections and "awesome-claude-code" lists — what rosters and
  autonomous-loop skills exist in the wild?
- spec-kit (`github/spec-kit`) — re-verify the template-override precedence claim.
- Anthropic's published multi-agent / subagent / Agent-Skills guidance — include for completeness
  and correction.

**Tier 3 — discover on your own:** anything else you find that does autonomous code loops with
verification gates or plan/implement splits.

## Questions to answer for each framework

1. **Roster / agent structure** — what agents/roles/modes does it define, and how (separate files?
   config? prompts)? How are they scoped (tools, permissions)?
2. **Plan-vs-implement split** — does it separate planning from editing? How (sessions, models,
   modes)? What's the claimed or measured benefit?
3. **Verification mechanism** — how does it verify autonomous changes? LLM judges, test suites,
   formal gates, diffs, human-in-loop? Is there eval/benchmark evidence?
4. **Stop conditions / safety** — how does it decide to stop or escalate to a human? Does it guard
   against guessing decisions, bad edits, or pushing code?
5. **Portability across project types** — how does it stay project-agnostic? (config file, env
   detection, stack-agnosticism, template overrides, something else?) **This is the most important
   question.**
6. **Distribution** — how is it installed/shared? Claude Code skill, VS Code extension, CLI,
   standalone runtime?
7. **One concrete recommendation** for dark-factory from this framework — adopt or avoid — with
   reasoning.

## Two cross-cutting questions (answer with evidence, not opinion)

**A. Portability convention.** Across all frameworks, what is the dominant pattern for staying
project-agnostic? Is there convergence on a config/override layer vs other approaches? What does
each framework actually assume — or refuse to assume — about the target project?

**B. Verification state-of-practice.** For autonomous code changes, what works — LLM judges,
objective test suites, formal/spec checks, or hybrids? Is there *quantitative* evidence (benchmarks,
ablations) on: (i) multi-judge vs single-judge-with-rubric, (ii) self-review vs independent review,
(iii) LLM-judge vs test-suite-as-ground-truth? When does each approach win?

## Verify, confirm, or refute these provisional claims

*(From prior knowledge; possibly stale. These projects move fast — cite the commit/doc date.)*

- BMAD uses a per-project config (`.bmad-core/core-config.yaml`), defines personas as separate agent
  files, and ships installers for Cursor/Roo/Claude Code.
- Aider architect mode splits a strong "architect" model (plans in natural language) from a cheaper
  "editor" model (applies SEARCH/REPLACE); the finding is that separating planning from editing
  improves results.
- Roo Code's `Architect` mode is read-only by config (cannot write files), enforced at the platform
  level not the prompt; modes live in a `.roomodes` JSON file, shareable per project.
- OpenHands and SWE-agent verify via the repo's own test suite / SWE-bench hidden tests (objective),
  not LLM judges; SWE-agent's load-bearing insight is "agent-computer interface (ACI) design," and
  they found self-review weak.
- spec-kit's template resolution precedence is project-local overrides > presets > extensions > core.
- Anthropic's multi-agent engineering post reports that a single LLM judge with a 0–1 rubric beat
  multiple judges on consistency; recommends filesystem artifact handoff to avoid the "game of
  telephone"; saves the plan to Memory to survive context truncation.

For each: **CONFIRM** (with source) · **REFUTE** (with source + the truth) · **PARTIALLY CORRECT**
(with correction).

## Output format

1. **Per-framework table** — name · repo URL · what it does (1–2 sentences) ·
   roster/plan-split/verification/stop/portability mechanisms · one adopt-or-avoid recommendation ·
   confidence · key source URL with access date.
2. **Cross-cutting findings** — portability convention (A) and verification state-of-practice (B),
   each with the evidence behind it.
3. **Provisional-claims scorecard** — confirm / refute / partial per claim above.
4. **Ranked recommendations for dark-factory** — the 5–10 highest-leverage things to adopt, avoid,
   or validate — each tied to a specific source.
5. **Sources** — full URL list with access dates; flag any you could not access directly.

## Rules

- Read **primary sources** (GitHub repos, official docs, engineering blogs). Prefer these over
  secondary listicles.
- Cite **versions / commit dates** — these projects change fast and a stale mechanism misleads.
- Distinguish a **marketing claim** from a **verified mechanism**. If a capability is asserted but
  not shown, say so.
- Be **critical**. Note where dark-factory's design is *better* than the comparison, not only where
  it's worse.
- **Flag uncertainty** explicitly rather than confabulating. "Could not verify" is a valid finding.
- **Access dates** on all sources.
