# Rebuilding dark-factory as a Portable Claude Code Skill: A Critical Comparison of Autonomous-Loop Frameworks

## TL;DR
- **Adopt spec-kit's runtime template-resolution stack and BMAD's typed per-project config as your portability layer** — these are the two clearest, verified conventions for staying project-agnostic; dark-factory's current overfitting to one project is the single biggest gap, and a config/override layer is the industry-convergent fix.
- **dark-factory's fan-out adversarial verification is defensible but partly contradicts Anthropic's own finding** that a *single* LLM judge with a 0–1 rubric was more consistent than multiple specialized judges; keep multi-lens judging ONLY where you also have an objective oracle (flow-verifier running real code, test suites), because the strongest quantitative evidence (OpenHands, SWE-bench) shows test-execution-as-ground-truth beats LLM judgment.
- **dark-factory is genuinely ahead of most comparators on stop-conditions and single-writer discipline** — its DECISIONS.md "wall" and single-writer file areas are more principled than the Ralph loop's "never exit" hooks and stronger than most roster collections, which have no autonomous-loop safety at all.

## Key Findings

**The field has converged on config/override layers for portability, and dark-factory currently lacks one.** Every serious framework externalizes project-specific assumptions: BMAD via a typed `config.yaml`/`config.toml` with a `project.type` field that switches which knowledge bases agents load; spec-kit via a runtime template-resolution stack (project overrides → presets → extensions → core); Roo Code via `.roomodes`; Claude Code itself via `.claude/agents/` markdown + `CLAUDE.md`. dark-factory is described as "overfit to one project" — closing that gap is the primary task, and the patterns to copy are well-established.

**Plan/implement separation is now an industry default, and dark-factory's two-thread contract is a strong version of it.** Aider proved with a benchmark that splitting a reasoning "architect" model from an "editor" model improves code-editing scores. Roo Code enforces the split at the platform level (Architect mode is read-only). dark-factory's PLANNER/IMPLEMENTER split with single-writer file areas is architecturally similar but adds concurrency-safety that the others lack.

**Verification state-of-practice strongly favors objective oracles over LLM judgment where a ground truth exists.** OpenHands, SWE-agent, and Devin all verify against the repo's own test suite. The strongest quantitative signal available is OpenHands's critic-model result. LLM-judge approaches win only where there is no programmatic ground truth (research outputs, UX/visual judgment).

**Self-review is measurably weak; independent review is measurably better — but multiple judges are not obviously better than one good rubric.** This is the key tension for dark-factory's design.

## Details

### Tier 1 — Per-framework analysis

**BMAD-METHOD** (`github.com/bmad-code-org/BMAD-METHOD`)
- *Roster:* Specialized agents (analyst/Mary, PM, architect/Winston, Scrum Master/Bob, dev/Amelia, QA/Quinn, Test Architect/Murat) each defined as a self-contained Markdown file with embedded YAML persona, commands, and dependencies. In v6 the source format is `.agent.yaml` compiled into `.md` with XML activation blocks.
- *Plan/implement split:* Strong, phase-based: Analysis → Planning (PRD) → Solutioning (architecture + ADRs) → Implementation (SM creates story files, Dev implements, QA reviews). Each phase produces a versioned artifact that feeds the next.
- *Verification:* Quality gates at every stage transition. The Dev agent HALTs on "3 failures attempting to implement or fix something repeatedly," missing config, unapproved dependencies, or ambiguity. Gate ownership stays with QA; Dev only signals readiness via Status. TEA module adds P0–P3 risk-based gates with go/no-go release decisions.
- *Stop conditions:* Explicit HALT conditions in the Dev agent contract; the "3 repeated failures → HALT" rule is a directly transferable pattern.
- *Portability:* Per-project `core-config.yaml` (v4) / `config.toml` (v6) with a `project.type` field (e.g., `python_django`, `elixir_phoenix`) that determines which implicit knowledge bases agents load; customizations live in `_bmad/_config/` and survive updates. Ships installers for Cursor, Roo, Claude Code, and other IDEs.
- *Distribution:* npm/npx (`npx bmad-method install`); compiles agents into IDE-specific directories.
- *Recommendation:* **ADOPT the typed `project.type` config pattern and the explicit HALT-on-N-failures rule.** These map almost directly onto dark-factory's needs for project-neutrality and clean stop conditions.
- *Confidence:* High (primary GitHub + official docs).

**Aider architect mode** (`github.com/Aider-AI/aider`)
- *Roster:* Not a roster; two roles — an "architect" model that proposes changes in natural language and an "editor" model that translates them into SEARCH/REPLACE edits. Also `/ask` and `/help` modes.
- *Plan/implement split:* This IS the product. Per Aider's post "Separating code reasoning and editing" (Sept 26, 2024): "Using o1-preview as the Architect with either DeepSeek or o1-mini as the Editor produced the SOTA score of 85%" on aider's 133-exercise code-editing benchmark, using the slower "whole" edit format.
- *Verification:* None intrinsic beyond the edit-format round trip; relies on the user and the editor's diff application. No LLM-judge layer.
- *Stop conditions:* Interactive; human approves edits.
- *Portability:* Stack-agnostic by construction (operates on any repo via edit formats); model-agnostic pairing.
- *Distribution:* CLI (pip), terminal-native.
- *Recommendation:* **ADOPT the framed insight (separate reasoning from editing) as validation of dark-factory's two-thread contract; do NOT adopt its verification approach** — it has essentially none.
- *Confidence:* High (official benchmark post + docs).

**Roo Code custom modes** (`github.com/RooCodeInc/Roo-Code`) and Cline
- *Roster:* Built-in modes — Code (full access), Architect (read + Markdown-only edit), Ask (read-only), Debug, Orchestrator (delegates via `new_task`). Each mode is a `ModeConfig` (slug, name, roleDefinition, groups, customInstructions).
- *Plan/implement split:* Architect mode is **read-only on project code, enforced at the platform level** via tool-group filtering and a `fileRegex` validation layer that catches violations before tool execution — not merely a prompt instruction. Sticky per-mode models (e.g., o3 for Architect, Sonnet for Code).
- *Verification:* No built-in adversarial verifier; relies on mode restrictions + human review.
- *Stop conditions:* Human-in-loop; mode restrictions prevent destructive edits during planning.
- *Portability:* `.roomodes` file (YAML or JSON) at project root; project modes override global ones completely for a given slug. Supports `AGENTS.md` standard.
- *Distribution:* VS Code extension (also works in Cursor/Windsurf).
- *Recommendation:* **ADOPT platform-level (not prompt-level) enforcement of write restrictions.** This is the strongest available precedent for dark-factory's single-writer file areas — enforce them mechanically, not by asking the model nicely.
- *Confidence:* High (official docs + source-referenced DeepWiki).

**SuperClaude** (`github.com/SuperClaude-Org/SuperClaude_Framework`)
- *Roster:* 22,100 stars / 1,900 forks. The current v4 release ships 30 slash commands, 16 domain agents, and 7 behavioral modes (the official `agents.md` doc states 16 domain specialist agents; a community review counts "30 slash commands, 20 agents, and 7 behaviour modes"; the netlify site lists an earlier "22 slash commands, 14 agents, and 6 behavioral modes" for v4.0). Injected as behavioral instructions into Claude Code's config.
- *Plan/implement split:* Workflow commands (`/sc:brainstorm` → `/sc:design` → `/sc:workflow` → `/sc:implement` → `/sc:test`) but no hard platform enforcement; "auto-activation" is explicitly behavioral instruction, not system logic.
- *Verification:* None objective; relies on personas.
- *Stop conditions:* None autonomous.
- *Portability:* Config-framework layered on CLAUDE.md; installs to `~/.claude/`. Uses Serena MCP for code memory. A DEV Community reviewer notes the tradeoff: "it's designed for someone else's workflow" — the same overfitting risk dark-factory faces.
- *Distribution:* pip (`pip install SuperClaude`) / install.sh; TypeScript plugin system planned but not yet available.
- *Recommendation:* **AVOID copying its scale** (16–20 agents, 30 commands is bloat for a focused loop) but note its honest framing that pre-built configs overfit — a caution for dark-factory's neutralization.
- *Confidence:* Medium-High (primary GitHub + docs; agent/command counts vary by version).

**OpenHands** (`github.com/OpenHands/OpenHands`, ex-OpenDevin)
- *Roster:* AgentHub registry with 10+ agents: CodeActAgent (generalist), BrowsingAgent, plus micro-agents (npm, github) instantiated from natural language. Controller-agent-runtime architecture; AgentController enforces iteration/budget limits and lifecycle (start/stop/pause).
- *Plan/implement split:* Plan–act loop within a unified code action space (CodeAct); delegation via `AgentDelegateAction`.
- *Verification:* **Objective — the repo's own test suite / SWE-bench hidden tests.** Their April 17, 2025 result is the strongest quantitative datapoint in this report: a trained **critic model** (Qwen 2.5 Coder Instruct 32B, fine-tuned with veRL) plus best-of-5 inference-time scaling raised SWE-bench Verified from **60.6% (single trajectory) to 66.4% (five attempts)** using Claude 3.7 Sonnet. The critic's terminal reward `r*T` is **1 for passing all unit tests, 0 for failing**, propagated backward with γ=0.99 — i.e., test execution is the ground truth, NOT LLM judgment. (Note: the officially reproduced leaderboard figure was 65.8%, slightly below the blog's 66.4%.)
- *Stop conditions:* AgentFinishAction, resource/iteration limits, or error. In SWE-bench eval the user-response function explicitly instructs "NEVER ASK FOR HUMAN HELP" — an autonomous-benchmark posture, not a production-safety one.
- *Portability:* Sandboxed Docker runtime; model-agnostic; config.toml.
- *Distribution:* Open-source platform + SDK. The newer OpenHands Software Agent SDK (arXiv 2511.03690, Nov 12, 2025) reports a 72% resolution rate on SWE-bench Verified "using Claude Sonnet 4.5 with extended thinking" (and 67.9% on GAIA val); a 15-day production comparison showed the V1 SDK "reduces system-attributable failures by 61% relative to V0."
- *Recommendation:* **ADOPT test-execution-as-ground-truth as the top verification tier and the "generate N candidates, select by objective score" pattern** for dark-factory's scheduled real-data runs. This is stronger evidence than any LLM-judge scheme.
- *Confidence:* High (primary blog verified + arXiv papers).

**SWE-agent** (`github.com/princeton-nlp/SWE-agent`)
- *Roster:* Single agent; the load-bearing insight is **Agent-Computer Interface (ACI) design**, not roles.
- *Plan/implement split:* None; single interactive loop.
- *Verification:* **Objective — SWE-bench hidden unit tests.** A linter runs on every edit and blocks syntactically invalid edits (a fail-closed gate). The ACI ablation showed +10.7 points over a bare-shell baseline on SWE-bench Lite; original SWE-bench resolve rate 12.5% (GPT-4 Turbo) vs 3.8% prior RAG SOTA.
- *Stop conditions:* Turn/cost limits.
- *Portability:* ACI is portable across models (GPT-4 Turbo → Claude 3 Opus 10.5%); mini-SWE-agent is 100 lines achieving 65% on SWE-bench Verified.
- *Distribution:* Python repo/CLI.
- *Recommendation:* **ADOPT the fail-closed edit gate (reject invalid edits before they land) and the ACI principle** — give dark-factory's implementer a tightly designed edit/verify interface with automatic confirmation feedback. The paper found LMs waste actions when edits "succeed silently."
- *Confidence:* High (arXiv 2405.15793 + docs).

**Devin / Cognition**
- *Roster:* Single autonomous agent with shell/editor/browser in a sandbox.
- *Plan/implement split:* Proposes an editable plan, then executes; dynamic re-planning (DAG).
- *Verification:* Self-verification via sandbox test execution and, since Devin 2.2 (Feb 2026), computer-use/desktop testing with screen recordings; browser/Playwright-style testing of web apps.
- *Stop conditions:* **Human-in-the-loop governance with two non-negotiable checkpoints: the Planning Checkpoint and the Pull Request Checkpoint.** On unexpected test failures Devin chooses among investigate / fix / note-and-continue / escalate.
- *Portability:* Cloud sandbox; repo-scoped least-privilege permissions, ACU budget.
- *Distribution:* SaaS; Slack + VS Code integrations.
- *Recommendation:* **ADOPT the two-checkpoint model (plan approval + PR approval) as a named pattern** — it closely matches dark-factory's DECISIONS.md wall and release ritual, and validates gating at exactly those two boundaries. Note benchmark caveat: Cognition's own SWE-bench figure was ~13.9% (79/570), so treat autonomy claims cautiously.
- *Confidence:* Medium (vendor blogs + secondary; Cognition primary for 2.2).

**Cursor and Windsurf agent modes**
- *Roster:* Cursor: Composer/Agent mode. Windsurf: Cascade ("Flows" = plan → execute → verify).
- *Plan/implement split:* Cursor plans then shows a diff for approval at each step (control-first). Windsurf Cascade is more autonomous, integrating terminal/build/test output into its loop; asks approval before applying.
- *Verification:* Cascade folds build errors and test results into its loop; Windsurf added Cascade Hooks (pre/post-action triggers for linters/standards). Neither replaces code review.
- *Stop conditions:* Diff-and-approve (Cursor); approval-before-apply (Windsurf).
- *Portability:* `.windsurfrules` / Cursor rules files; memory scoping.
- *Distribution:* Proprietary IDEs.
- *Recommendation:* **ADOPT Cascade's plan→execute→verify Flow framing and hooks-for-standards; note the Cursor lesson that diff-and-approve is a feature on production systems** — relevant to how dark-factory surfaces implementer output.
- *Confidence:* Medium (secondary comparison sources; vendor docs).

### Tier 2 — Distribution & ecosystem

**Community `.claude/agents/` collections & awesome lists.** Large rosters exist: VoltAgent/awesome-claude-code-subagents (over 130 subagents organized into 10 hierarchical categories), wshobson/agents, gstack (Garry Tan's roles for product/design/eng/release/docs/QA), Claude-Flow (enterprise orchestration). Format is standardized: Markdown + YAML frontmatter (`name`, `description`, `tools`, `model`) in `.claude/agents/`; project-local `.claude/agents/` overrides `~/.claude/agents/` on name conflict; subagents run in isolated context and don't inherit parent skills (must preload). Crucially, **most of these are role libraries with no autonomous-loop safety** — they are rosters, not loops.

**Autonomous-loop skills in the wild.** The closest analogs to dark-factory are the **Ralph / Ralph Wiggum** family (Geoffrey Huntley, mid-2025; formalized by Anthropic as a Claude Code plugin using Stop hooks). Ralph re-injects the same prompt against accumulating filesystem/git state until a completion promise + tests pass. frankbria/ralph-claude-code adds a **dual-condition exit gate (completion indicators AND explicit EXIT_SIGNAL), rate limiting, and a circuit breaker.** Other loops: Continuous Claude, claude_code_agent_farm, ClaudeNightsWatch. Huntley himself warned the official plugin "risks being misunderstood if people treat it as a hands-off product rather than a technique requiring supervision," and practitioners flag "overbaking"/"sycophancy loops" where an agent can't exit because it must satisfy a completion promise.

**spec-kit** (`github.com/github/spec-kit`). Re-verified: template resolution is a **runtime stack walked top-down, first match wins**: `.specify/templates/overrides/` (project-local) → `.specify/presets/<id>/templates/` (by priority) → `.specify/extensions/<id>/templates/` → `.specify/templates/` (core). Presets are priority-ordered (lower number = higher precedence) and **override by default, not merge** (though composition strategies exist). Command overrides are applied at install time, not runtime. `constitution.md` lives in `.specify/memory/` as project-specific non-negotiable principles; there is a known bug where `--force` reinit overwrites a customized constitution.

**Anthropic's published guidance.** Subagents = Markdown in `.claude/agents/` with YAML frontmatter; isolated context; tool restrictions; built-in Explore/Plan agents skip CLAUDE.md for speed. Agent Skills = portable folders (SKILL.md + progressive disclosure) following an open standard, "build once, use across Claude apps, Claude Code, and API." The multi-agent research post (Jun 13, 2025) reports the orchestrator saves its plan to Memory to survive the 200K-token truncation, recommends filesystem artifact handoff, and found **a single LLM judge with a 0.0–1.0 rubric + pass/fail was more consistent than multiple specialized judges.** Multi-agent used ~15× the tokens of chat and beat single-agent Opus 4 by 90.2% on their internal research eval (a research task, not coding).

## Cross-cutting findings

### A. Portability convention
There is clear convergence on an **externalized config/override layer**, but two distinct flavors:
1. **Typed config that switches behavior** (BMAD `project.type` → loads stack-specific knowledge; quality.pre_commit gate commands). Best when the framework needs to reason differently per stack.
2. **Layered template/override resolution** (spec-kit's overrides → presets → extensions → core, walked at runtime, first-match-wins). Best when the framework ships defaults that projects selectively override without forking.

What each framework *refuses* to assume: SWE-agent and Aider assume almost nothing about the stack (they operate through edit formats / ACI on any repo). BMAD assumes an agile artifact chain but externalizes the stack. spec-kit assumes an SDD workflow but is "cross-agent by default." Roo Code assumes VS Code. The dominant portable-primitive is **Markdown + YAML frontmatter agent/mode definitions** plus a **project-root config file** (`.roomodes`, `config.toml`, `.specify/`, `.claude/`). For dark-factory the recommended synthesis: a small project-root config declaring stack/test/build/validation commands (BMAD-style) PLUS a template-override directory so the neutral core ships defaults that a project overrides without forking (spec-kit-style).

### B. Verification state-of-practice
The evidence hierarchy, strongest to weakest:
1. **Objective test-suite / hidden-test execution is the gold standard where ground truth exists.** OpenHands's critic uses unit-test pass/fail (1/0) as terminal reward; SWE-agent, Devin, and SWE-bench Verified (500 human-validated tasks, deterministic pass/fail oracle in Docker) all rely on this. OpenHands quantified the payoff: 60.6%→66.4% via best-of-5 with a test-reward-trained critic.
2. **Independent review beats self-review — measurably.** Per OpenAI's CriticGPT paper ("LLM Critics Help Catch LLM Bugs," arXiv 2407.00215, McAleese et al., 2024): "On code containing naturally occurring LLM errors model-written critiques are preferred over human critiques in 63% of cases, and human evaluation finds that models catch more bugs than human contractors paid for code review"; on tampered/inserted-bug code, model critiques were preferred "more than 80% of the time," and human+critic teams hallucinated less than critic-alone. Conversely, self-review is weak: one study found LLMs self-identify 78.7% of their own bugs on review yet still emit them during generation; Snorkel's "self-critique paradox" found a critique loop *dropped* accuracy 10% overall and was "devastating" on tasks the model already did well (≥75% accuracy) because the critic hallucinated errors — but was a "miracle worker" on tasks where the model failed badly (<35%).
3. **Multiple judges are NOT clearly better than one good rubric.** Anthropic found a single 0–1 rubric judge more consistent than multiple specialized judges. This directly challenges dark-factory's 2–3 named-lens fan-out. HOWEVER, Anthropic's own "Demystifying evals" post recommends grading each dimension with an isolated judge rather than one judge for all dimensions — so the evidence is genuinely mixed and task-dependent.
4. **LLM judges work only without ground truth** (research, UX, visual). They fail on creative/subjective tasks and can be gamed by weak tests (SWE-bench+ showed weak tests inflate resolution rates by 27–36 points).

The synthesis for autonomous *code* changes: **objective oracles first (tests, real-code-path execution, linters/fail-closed edit gates), independent review second, single-rubric LLM judging third, multi-judge only where dimensions are genuinely orthogonal and you accept the token cost.**

## Provisional-claims scorecard

1. **BMAD uses per-project config, persona agent files, ships Cursor/Roo/Claude Code installers** — **CONFIRMED.** `core-config.yaml` (v4) / `config.toml` (v6); agents are self-contained Markdown+YAML files; installers target multiple IDEs. Correction/nuance: v6 renamed to `config.toml` and moved customizations to `_bmad/_config/`; the exact filename `.bmad-core/core-config.yaml` is version-specific.

2. **Aider architect (strong architect plans NL, cheap editor applies SEARCH/REPLACE; separating improves results)** — **CONFIRMED.** Official post (Sept 26, 2024): SOTA 85% on aider's 133-exercise code-editing benchmark; o1-preview architect + DeepSeek/o1-mini editor.

3. **Roo Architect mode is read-only by config, enforced at platform level; modes in `.roomodes`, shareable per project** — **CONFIRMED.** Tool-group filtering + fileRegex validation layer enforces read-only/Markdown-only before tool execution; `.roomodes` (YAML or JSON) at project root; project overrides global by slug.

4. **OpenHands/SWE-agent verify via test suite / SWE-bench hidden tests (objective), not LLM judges; SWE-agent's insight is ACI; self-review weak** — **CONFIRMED with nuance.** Both use objective tests; ACI is SWE-agent's thesis. Nuance: OpenHands *added* a trained critic model — but that critic is trained on test-execution rewards (1/0 unit-test pass), not free LLM judgment, so the claim holds. Self-review weakness confirmed by multiple sources.

5. **spec-kit precedence: project-local overrides > presets > extensions > core** — **CONFIRMED.** Exactly the documented runtime resolution stack.

6. **Anthropic multi-agent post: single 0–1 rubric judge beat multiple judges; filesystem artifact handoff; plan saved to Memory to survive truncation** — **CONFIRMED.** All three verified verbatim in the primary source. Caveat: their separate "Demystifying evals" post recommends isolated per-dimension judges, so Anthropic's overall guidance is nuanced, not a blanket "one judge always wins."

## Ranked recommendations for dark-factory

1. **Build a two-part portability layer: a typed project-root config (stack, test/build/validation commands) + a template-override directory.** Adopt BMAD's `project.type`-style typed config and spec-kit's runtime first-match override stack. This is the highest-leverage change and directly fixes the "overfit to one project" problem. *(Sources: BMAD config docs; spec-kit presets/README.)*

2. **Enforce single-writer file areas mechanically, not by prompt.** Follow Roo Code's platform-level model: a validation layer that rejects out-of-lane writes before they execute (analogous to Roo's fileRegex gate). Prompt-only single-writer discipline will drift. *(Source: Roo Code custom-modes docs.)*

3. **Make objective oracles your top verification tier; demote LLM judges.** Adopt OpenHands's test-execution-as-ground-truth and SWE-agent's fail-closed edit gate (reject syntactically/lint-invalid edits before they land). dark-factory's flow-verifier (runs real code, inspects artifacts) is exactly right and should outrank the LLM judges. *(Sources: OpenHands critic blog; SWE-agent ACI paper.)*

4. **Re-scope the fan-out judges in light of Anthropic's finding.** Anthropic found one 0–1 rubric judge more consistent than many. Keep multiple lenses ONLY where dimensions are genuinely orthogonal (security vs regression) and pair each with an objective check; otherwise consolidate to a single rubric judge to cut cost and inconsistency. Validate empirically on ~20 representative tasks (Anthropic's recommended starting eval size). *(Sources: Anthropic multi-agent post; Demystifying evals post.)*

5. **Keep and formalize the DECISIONS.md wall as a named two-checkpoint pattern.** This matches Devin's Planning + PR checkpoints and is a genuine strength over Ralph-style "never exit" loops. Make the two boundaries explicit: plan/spec approval and release/PR approval. *(Source: Devin human-in-the-loop governance.)*

6. **Adopt BMAD's explicit HALT-on-N-repeated-failures rule and a circuit breaker.** BMAD HALTs after 3 repeated fix attempts; Ralph implementations add rate limiting + circuit breakers to prevent runaway loops and "overbaking." dark-factory's scheduled (non-polled) expensive runs already avoid one failure mode; add explicit failure-count and cost ceilings. *(Sources: BMAD dev-agent contract; frankbria/ralph-claude-code.)*

7. **Persist the plan and STATUS as files, and prefer filesystem artifact handoff.** Anthropic saves the plan to Memory to survive context truncation and recommends filesystem handoff to avoid the "game of telephone." dark-factory's STATUS-banners-as-source-of-truth and task-file-as-contract already embody this — keep it and document it as the canonical handoff. *(Source: Anthropic multi-agent post.)*

8. **Distribute as an Agent Skill following the open standard, optionally bundled as a plugin.** Skills are portable "build once, use across Claude apps, Claude Code, and API"; a plugin can bundle the skill + `.claude/agents/` roster + hooks. Keep SKILL.md under ~500 lines with progressive disclosure; split detail into referenced files. *(Sources: Anthropic Agent Skills post; Claude Code skills/subagents docs.)*

9. **Avoid roster bloat.** SuperClaude (16–20 agents, 30 commands) and the 130+ community collections show a common anti-pattern: sprawling rosters that overfit someone else's workflow. dark-factory's tight 2–3-judge + flow/visual-verifier roster is a strength; keep it minimal. *(Sources: SuperClaude docs; awesome-claude-code lists.)*

10. **Validate the visual-judge against an objective anchor.** LLM visual judgment is subject to the self-critique paradox (hallucinated 2-pixel "discrepancies"). Anchor the visual-judge to screenshot diffs / Playwright assertions where possible rather than free judgment. *(Source: Snorkel self-critique paradox; Windsurf/Devin computer-use testing.)*

## Caveats
- **Fast-moving targets.** BMAD v6 is alpha; SuperClaude's plugin system is unreleased; Windsurf changed ownership (Codeium team acquired by OpenAI in 2025, then some 2025–2026 sources place the Windsurf team under Cognition — ownership reporting is inconsistent and I could not fully reconcile it). Cite dates accordingly.
- **Vendor vs verified.** Devin's autonomy and "self-verification" claims are largely vendor-sourced; its own SWE-bench figure (~13.9%) is modest. Cursor/Windsurf internals rely on secondary comparison articles, not primary docs, for some claims.
- **Benchmark integrity.** SWE-bench resolution rates are inflated by weak tests (SWE-bench+ found 27–36-point drops after removing weak tests); OpenHands's 66.4% was reproduced at 65.8% officially. Treat all leaderboard numbers as directional.
- **The multi-judge question is not settled.** Anthropic's "one rubric judge" finding is from a *research* task; per-dimension isolated judges are recommended elsewhere by Anthropic for *agent* evals. dark-factory should treat this as an empirical question to test, not a solved one.
- **Could not verify:** exact current commit dates for each repo's cited features (I relied on doc/post publication dates and DeepWiki source references); the precise internal enforcement mechanism of dark-factory's own single-writer areas (described only in the task prompt); SuperClaude's exact current agent count (official doc says 16, community reviews say 20, netlify site says 14 — version-dependent).

## Sources
Primary sources (accessed July 5, 2026):
- BMAD-METHOD: github.com/bmad-code-org/BMAD-METHOD (agent-customization-guide.md); docs.bmad-method.org (how-to/customize-bmad, reference/testing, reference/agents); DeepWiki BMAD Core Architecture and TEA module; redreamality.com BMAD guide (config.yaml `project.type`, pre_commit gate); diegorodrigo.dev (implementation-loop flowchart, development_status file); scribd Dev agent contract (HALT conditions).
- Aider: aider.chat/2024/09/26/architect.html (SOTA 85%, Sept 26 2024); aider.chat/docs/usage/modes.html.
- Roo Code: docs.roocode.com/features/custom-modes; docs.roocode.com/update-notes/v3.3; DeepWiki Roo-Code mode selection; GitHub Roo-Code issue #5020.
- SuperClaude: github.com/SuperClaude-Org/SuperClaude_Framework (README, docs/user-guide/agents.md, commands.md); superclaude.netlify.app; pypi.org/project/superclaude; dev.to/stevengonsalvez SuperClaude review (22.1k stars/1.9k forks).
- OpenHands: openhands.dev/blog/sota-on-swe-bench-verified-with-inference-time-scaling-and-critic-model (Apr 17 2025; 60.6%→66.4%, critic reward r*T, γ=0.99, Claude 3.7 Sonnet); arXiv 2511.03690 (SDK, 72% SWE-bench Verified, Nov 12 2025); arXiv 2407.16741 (OpenHands platform); github.com/OpenHands/OpenHands codeact_agent README; docs.openhands.dev evaluation-harness.
- SWE-agent: arXiv 2405.15793 (ACI, 12.5% vs 3.8%, +10.7 ablation, Claude 3 Opus 10.5%); swe-agent.com/1.0/background/aci (linter fail-closed gate, mini-SWE-agent 65%).
- Devin/Cognition: cognition.com/blog/introducing-devin-2-2 (Feb 2026 desktop/self-verify); wwt.com Devin strategic view (two checkpoints); skywork.ai Devin explainers (~13.9% SWE-bench, ACU budget); medium.com Devin planning/DAG.
- Cursor/Windsurf: devtoolsacademy.com, verdent.ai, mindstudio.ai, digitalapplied.com (Cascade Flows plan→execute→verify, hooks, diff-and-approve).
- Ralph loop: github.com/frankbria/ralph-claude-code (dual-condition exit gate, circuit breaker, v0.11.5); prg.sh/notes/Ralph-Wiggum-Loop; tessl.io, ralph-wiggum.ai, thegoodprogrammer.medium.com (overbaking/sycophancy loop, COMPLETE token).
- spec-kit: github.com/github/spec-kit (README, presets/README, docs/upgrade.md, issue #1541); developer.microsoft.com/blog/spec-driven-development-spec-kit.
- Anthropic: anthropic.com/engineering/multi-agent-research-system (Jun 13 2025; single-rubric judge, Memory plan, 90.2%, 15× tokens); anthropic.com/engineering/demystifying-evals-for-ai-agents (isolated per-dimension judges); anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills; code.claude.com/docs/en/sub-agents; claude.com/blog/steering-claude-code.
- Verification research: arXiv 2407.00215 CriticGPT (63% / >80% preferred, human+critic hallucinate less); snorkel.ai self-critique paradox (10% drop, ≥75%/<35% split); dev.to LLM self-review 78.7% self-identification; openreview SWE-bench+ (27–36-point weak-test inflation).
- Community rosters: github.com/VoltAgent/awesome-claude-code-subagents (130+ subagents); github.com/hesreallyhim/awesome-claude-code; scriptbyai.com Claude Code resource list; awesomeclaude.ai (Ralph collection).

Flagged access limitations: Cursor/Windsurf and Devin claims lean on secondary comparison articles and vendor blogs rather than primary engineering docs; DeepWiki entries are third-party summaries of primary repos (used where they cite specific source files). All access dates July 5, 2026.