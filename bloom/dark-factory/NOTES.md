> **STATUS (2026-07-05): acted on — v0.2.0 shipped to apoage/apothecary as a plugin (hooks + config variables). This file is the design record; see README.md + plugin/.**

# dark-factory — notes toward a clean, portable skill

Working notes. The current artifact (skill + 4 agents + sauce/) is kept as-is; this file is the
raw material for building the real, project-neutral version later. Three sections the owner asked
for: **weak places**, **overfitting inventory**, **research output** (pending → filled in when the
two research agents return), plus a recommendations stub.

---

## 1. What dark-factory is (one paragraph)

An autonomous-implementer *loop* packaged as a Claude Code skill. A ranked task queue feeds a
build subagent; every load-bearing change faces fan-out adversarial verification (never self-
review); a release ritual versions/commits/tags/builds; STATUS banners keep the queue truthful; the
loop stops cleanly at human-decision walls. It is the *implementer half* of a two-session model
(planner writes specs, implementer writes code) where single-writer file areas prevent edit races
between the two concurrent sessions.

---

## 2. What's genuinely good — KEEP these in the real skill

The substance is correct. Don't lose it while stripping the videoED paint.

- **Three load-bearing rules.** (a) Single-writer areas — planner owns tasks/docs/prototypes,
  implementer owns code/version/commits. (b) Never self-review — the writing context is the weakest
  reviewer; fan out to separate agents. (c) Decisions-are-walls — never build on a guessed product
  decision; seed a recommendation-first DECISIONS row and stop. These three are the whole game.
- **Refutation-first judges with named lenses.** correctness / fail-closed / regression / security.
  The rule "*an empty verdict must list the attacks attempted, or it's worthless*" is the anti-
  sycophancy guard that makes the judges actually bite. Keep it verbatim.
- **Claims in task files are test cases for the flow verifier.** "byte-identical when off" → diff
  it. This turns the spec into executable verification. Strong pattern.
- **Banner > queue truth hierarchy.** The queue is an index; the dated STATUS banner on the task
  file is the truth; on conflict trust the banner and log the drift. Small, correct.
- **Scheduled-not-polled expensive validation.** Long real-data runs go overnight as one-shots,
  never in-loop polling. They learned this burns context for nothing — preserve it.
- **Visual judge returns findings, not pass/fail.** Screenshot judgment is noisy; evidence travels
  with every finding; severity-ranked. Honest about its own weakness.
- **Minimum-viable-factory admission** in PORTING.md (1 implementer + 1 correctness judge = 80%).
  Keep this escape hatch — it lets people adopt incrementally.

---

## 3. Weak places — honest critique

Things to fix or harden when building the real skill. Ranked roughly by how much they bite.

### 3.1 Single-writer is enforced by prompt, not by the filesystem  *(high)*
The whole two-thread contract rests on each session staying in its lane. There is **no pre-commit
hook, no permission boundary, no path allowlist** enforcing it. A tired planner session that "just
fixes one typo" in code breaks the contract, and nothing catches it until the next factory run
notices drift. This is the trick's soft underbelly.
→ **Fix idea:** ship an optional git `pre-commit` hook (or a `.claude/settings.json` `PreToolUse`
hook) that fails writes outside the owning thread's allowlist. Convention + a soft guardrail.

### 3.2 Auto-commit / tag / build without per-action confirmation  *(high)*
The release ritual commits, tags, and builds on its own. Normal Claude Code guidance is "commit or
push only when the user asks." Defensible *inside* the factory contract (the user opted in by
running it), but it is a meaningfully different risk posture, and the skill never makes the user
affirm it.
→ **Fix idea:** an explicit opt-in line in the skill ("running /dark-factory authorizes the release
ritual; tags stay local; never pushes"), and a hard rule that **nothing is pushed to a remote**
without a separate explicit confirmation.

### 3.3 The skill is still deeply videoED-flavored  *(high — this is the overfit; see §4)*
Despite PORTING.md's swap-list, SKILL.md and the four agents hardcode fixture paths, version files,
build commands, design-system principles, a port number, and a real-footage volume. "Drop in and
go" is optimistic; you'd be *rewriting* the skill, not configuring it.

### 3.4 Dangling memory references  *(medium)*
SKILL.md leans on `videoed-two-thread-contract`, `videoed-blue-access`, `videoed-nondestructive`,
`videoed-versioning-scheme`, `videoed-git-local-no-github`. None ship with the package. On a fresh
install those are dead pointers — the substance behind them is lost **silently** (the skill reads
as if they'll resolve; they won't). PORTING.md says "delete or repoint" but SKILL.md still cites
them as live.
→ **Fix idea:** inline the one-line substance of each rule into the skill body; drop the slug
references entirely. Memory is project-local; a portable skill can't depend on it.

### 3.5 Visual judge is the most fragile agent  *(medium)*
Headless boot + Playwright + screenshot judgment is real but noisy (the doc admits it). Failure
modes that produce false findings: wrong port pick, stale fixture, webkit-vs-chromium mismatch
with the real runtime, a flaky `pytest -m ui`. A false BLOCKER stalls the queue under a human.
→ **Fix idea:** make the engine explicit and matched to the real runtime (pywebview→webkit is a
videoED fact, not universal); require the judge to attach the console-error log + the exact command
it ran, not just screenshots; document the retry/noise floor.

### 3.6 Fan-out verification cost  *(medium)*
2–3 judges + flow verifier + visual judge per load-bearing change is a lot of subagent invocations.
On a long unattended run that's real token spend and wall-clock. The minimum-viable admission (§2)
partially mitigates it, but the skill doesn't give the loop a way to *scale verification to risk*
(e.g. XS filler task → single correctness judge; strategic change → full panel).
→ **Fix idea:** a verification tier keyed off task size/risk, not one-size-fits-all.

### 3.7 `app/version.py` letter-suffix + PyInstaller is presented as if universal  *(low–medium)*
The release ritual is videoED's. A node project, a rust crate, a library — none of this maps. The
*shape* (bump version → changelog → commit → tag → build → smoke) is universal; the *mechanics*
aren't.
→ **Fix idea:** parameterize the ritual behind a named step list the owner fills in per project
(like the invariants block should be).

### 3.8 No telemetry / no "what did the factory actually do" artifact beyond the wrap-up  *(low)*
The wrap-up message is good but conversational. For unattended runs you want a machine-readable
run log (shipped versions, findings counts, decisions seeded) so the planner session can reconcile
next morning without re-reading the whole transcript.

---

## 4. Overfitting inventory — what to strip, file by file

`vd-*` = the agent name prefix (videoED). Recommend renaming to `df-*` (dark-factory) in the real
skill and fixing every cross-reference.

### `skills/dark-factory/SKILL.md` — *heavy rewrite*
- Description: "videoED dark factory", hardcoded owner "Lukas".
- 5 dangling memory slugs (see §3.4).
- Hard-invariants block: `uv venv 3.12`, `pipeline/tools.py`, "EDL times = seconds in original
  clip timeline", "WebM VP9 yuva420p alpha", "fail-closed creative gates", "Blue Tier-B",
  `fixtures/antminer_s21`.
- Validation tiers: `fixtures/antminer_s21` (instant-load), "Blue Tier-B" real-footage,
  `/Volumes/Blue`, `blue_sample/`.
- Fan-out section: `pytest -m ui` (webkit), `tests/ui_smoke_playbook.md`, `fixtures/antminer_s21`.
- Release ritual: `app/version.py` letter-suffix, `pyinstaller videoED.spec`, frozen-bundle smoke.
- Stop conditions: "Lukas-decision", `docs/<date>-session-handoff`.

### `agents/vd-implementer.md` → `df-implementer.md` — *medium*
- Name + description say "videoED".
- `source .venv/bin/activate && python -m pytest -q`.
- The `node --check` + grep-new-top-level-JS-identifier check (a videoED-specific classic-script-
  shared-global-scope bug — not universal).
- uv venv 3.12 / `pipeline/tools.py` / never bare ffmpeg/node.
- EDL/greenlit times, `.videoED/` write dir, fail-closed creative gates, ledger locks, synthetic-
  always-marked, schema-validate-or-degrade-on-every-LLM-call.
- The "don't bump version / don't edit tasks/docs" line — **keep**, it's universal.

### `agents/vd-judge.md` → `df-judge.md` — *medium*
- Name + description say "videoED".
- correctness lens: EDL timebase invariant.
- fail-closed lens: "rung below", ledger lock, confirmed beat, unmarked synthetic — all videoED
  domain language. Generalize to "can any input make the path ship a worse result than the safe
  fallback / violate a hard lock / drop a confirmed state".
- `source .venv/bin/activate && python -m pytest -q`.
- The refutation-first method + "list the 3 hardest attacks" rule — **keep verbatim**.

### `agents/vd-visual-judge.md` → `df-visual-judge.md` — *heavy*
- Name + description say "videoED".
- `tests/ui_smoke_playbook.md` (hardcoded playbook path).
- `python -m pytest -m ui` webkit layer.
- `python -m uvicorn app.server:app --port <free port>` + "NOT 8765 — Thunderbird squats it" (a
  videoED-machine fact).
- `fixtures/antminer_s21`.
- pywebview = WKWebView → webkit (runtime coupling).
- Grading-suite principles cited from `tasks/2026-06-30-ui-workspace.md`: "neutral graphite; color
  = signal only; mono for data; footage is the hero" — **these are videoED's design system, not a
  universal standard**. Generalize to "judge against the project's own design principles doc".
- Ledger three-indicator lockstep — videoED-specific coherence rule.

### `agents/vd-flow-verifier.md` → `df-flow-verifier.md` — *medium*
- Name + description say "videoED".
- `app/passes.py` `writes`.
- `fixtures/antminer_s21` + copy `.videoED` to scratchpad.
- `pipeline/tools.py`.
- EDL timebase, `cut_start` monotone.
- Marking rules (synthetic/inserted content carries marks).
- `python -m pytest -q`.
- "/Volumes/Blue" (the real-footage volume) — "never run against /Volumes/Blue".
- The "claims in the task file are test cases" method + VERIFIED/REFUTED/NOT-TESTABLE report —
  **keep verbatim**.

### Already clean — leave alone
- `sauce/templates/QUEUE.template.md`, `DECISIONS.template.md` — project-neutral.
- `sauce/TWO-THREAD-CONTRACT.md` — uses generic "planner/implementer" throughout; nearly clean
  (one or two passing videoED mentions).
- `sauce/examples/*` — *supposed* to be videoED live files; that's their purpose.
- `sauce/PORTING.md` — already the porting guide; minor touch-ups only.

---

## 5. Portability approach — the open design question

Three ways the real skill could stay project-agnostic. Research (§6) should settle which is
conventional.

- **A. Porting checklist + inline placeholders** (current approach via PORTING.md). Owner greps the
  installed skill for `YOUR_*` markers and rewrites. Simple, no runtime magic, but every install is
  a manual edit pass and drifts.
- **B. A single swap-file / config layer** — one `factory.config.md` (or YAML) per project that
  names the fixture path, version file, build command, invariants, design-principles doc, UI
  playbook. The skill reads it at startup. More upfront design; near-zero per-install editing.
- **C. Env detection** — the skill probes for common shapes (is there a `package.json`? `pyproject
  .toml`? a `version.py`?) and fills defaults. Convenient but magic, and wrong guesses are silent.

**Resolved by research (§6.5):** approach **B**, modeled on spec-kit's template-override precedence
(project-local overrides > presets > extensions > core), AND move all project specifics (owner
routing, channels, heat scoring, invariants, fixture path, build command) into `CLAUDE.md` per
Agent Skills best-practice — the skill ships a generic harness and reads `CLAUDE.md` for the rest.
Not env-detection. My pre-research lean (B) held; the CLAUDE.md split is the addition the research
added.

---

## 6. Research — similar skills & patterns

Two threads researched: **6.1 Anthropic-native** (complete, below) and **6.6 Community / analogous**
(recovering — the agent delegated to sub-agents and ended before synthesizing; being resumed).

### 6.1 Summary — dark-factory pillar vs. native analog

| Dark-factory pillar | Native analog | Match |
|---|---|---|
| Two-thread contract (planner↔implementer, single-writer) | spec-kit phase pipeline + multi-agent "filesystem handoff" | **Strong** |
| Fan-out adversarial verify (2–3 named-lens judges) | Subagents "fresh perspective" + LLM-as-judge | **Mixed** — pattern endorsed, but Anthropic's data says single-judge-with-rubric beat multi-judge on consistency |
| Flow-verifier (run real pass, inspect artifacts) | Skills' plan-validate-execute + Hooks (Stop/PreToolUse) | **Strong** |
| Visual-judge (Playwright + screenshots) | Playwright MCP + native `webapp-testing` skill | **Strong** — official path |
| DECISIONS.md wall / STATUS banners | spec-kit `constitution.md` + Stop-hook block + plan-in-Memory | **Strong** — pick the right mechanism |
| Skill portability | Agent Skills best-practices doc | **Strong** — explicit guidance |

### 6.2 Anthropic-native findings

**1. Multi-agent research system** ([source](https://www.anthropic.com/engineering/multi-agent-research-system)).
Orchestrator + 3–5 parallel subagents.
- **Token use = ~80% of performance variance.** Fan-out is justified by spending more tokens in
  parallel contexts. Dark-factory's 2–3 judge fan-out is defensible only if task value covers the
  ~15× token multiplier vs. single chat.
- **"Subagent output to filesystem to minimize the game of telephone."** Direct validation of the
  two-thread contract — subagents write durable artifacts, pass lightweight refs to the coordinator.
- **LeadResearcher saves plan to Memory** to survive context truncation → the native analog of
  DECISIONS.md-as-wall.
- **Teach the orchestrator how to delegate**: each subagent needs objective + output format + tools +
  task boundaries. Implication: judges need a *named lens* AND a *strict output format*.
- **Adopt: explicit scaling rules in the orchestrator prompt.** Anthropic hard-codes
  "simple=1 agent/3–10 calls; comparison=2–4/10–15 each; complex=10+." Dark-factory's judge count
  should be a function of task complexity, **not a constant 3**.
- ⚠️ **Honest caveat.** Anthropic's eval found a *single* LLM judge with a 0–1 rubric + pass/fail
  beat *multiple* judges on consistency/alignment. Do NOT assume 2–3 named-lens judges > 1
  structured-rubric judge on consistency. Defend multi-judge on **coverage** grounds (different
  lenses catch different failure modes) and validate with evals — or collapse to one rubric judge.

**2. Native subagents** (`.claude/agents/*.md`) ([blog](https://claude.com/blog/subagents-in-claude-code), [SDK docs](https://code.claude.com/docs/en/agent-sdk/subagents)).
- One agent file per lens (`judge-correctness.md`, `judge-failclosed.md`, …). `description` is the
  **auto-routing trigger** — write as a trigger condition, not a noun ("Refutes correctness of a
  change before commit. Use proactively after implementation.").
- **Read-only enforced at platform level** via `tools: [Read, Grep, Glob]`, not by instruction.
- **"Fresh perspective" is explicitly endorsed** — invoking a subagent *is* the never-self-review
  mechanism (context isolation is automatic).
- **Anti-pattern named directly:** "Same-file edits. Two subagents editing the same file in parallel
  is a recipe for conflict." → validates single-writer.
- **New fields worth using:** `effort` (security judge `high`/`max`, correctness `medium`), `maxTurns`
  (bound each judge), `permissionMode`, `skills: []` (preload e.g. `secure-coding` into the security
  judge), `model` override (security on Opus, correctness on Sonnet).
- `Workflow` tool only worth it if judge count ever scales to dozens; plain subagents are correct
  here. Agent Teams = over-engineering (judges are independent, shouldn't communicate).

**3. Agent Skills** ([engineering post](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills), [best practices](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices)).
Portability is the owner's biggest concern — this is the most actionable section.
- **"Default assumption: Claude is already very smart."** Only add context Claude doesn't have.
  Challenge each paragraph: does it justify its token cost? Dark-factory over-explains patterns
  Claude already knows (refutation, isolation). Cut.
- **Body < 500 lines.** Split to `references/<topic>.md` when approaching. **References one level
  deep** (no `a.md → b.md` chains).
- **Set freedom by fragility** (3 tiers): autonomous loop / adversarial-verify / visual-judge =
  **LOW freedom** (specific executable scripts). Planner = high freedom (text).
- **Naming:** gerund form (`verifying-changes`), description in third person with *what + when*.
- **Project-agnostic by design.** Skills = portable expertise; project specifics go in `CLAUDE.md`.
  Concretely: dark-factory should ship a generic harness and **read CLAUDE.md** for owner routing,
  channel conventions, heat scoring. The loop mechanics are portable; the routing table is not.
- **Plan-validate-execute** pattern: Claude writes a machine-verifiable plan file → script validates
  → execute. This is the native shape for the flow-verifier.

**4. spec-kit** ([github/spec-kit](https://github.com/github/spec-kit)).
The canonical planner/implementer split, made literal.
- **Artifact-per-phase pipeline:** `constitution.md → spec.md → plan.md → tasks.md → implement`.
  Each phase writes a durable artifact; artifacts ARE the inter-phase contract.
- **`/speckit.analyze`** — cross-artifact consistency/coverage check after `tasks`, before
  `implement`. A *pre-implement* fan-out verify over the spec stack (dark-factory's judges run on
  code; this runs on specs — steal as a second verify layer).
- **`/speckit.converge`** — reconcile codebase-vs-spec drift by appending remaining work as tasks,
  not restarting. The positive complement to DECISIONS.md-as-wall (DECISIONS = blocked-on-human;
  converge = negotiable-with-evidence scope drift).
- **Template-resolution priority (steal for portability):** project-local overrides > presets >
  extensions > core. → a project can override the judge roster / scripts / banner format **without
  forking the skill**. Ship defaults; let projects override. *(This is the answer to §5's open
  question — approach B, with spec-kit's precedence as the model.)*
- **`/speckit.clarify`** — structured clarification before planning; don't implement until the
  checklist is complete.

**5. Agent SDK hooks** ([docs](https://code.claude.com/docs/en/agent-sdk/hooks)).
The enforcement layer dark-factory currently lacks.
- **Stop hook `decision: "block"` + `reason`** = the native runtime form of DECISIONS.md-as-wall.
  When Claude tries to finish, the hook checks; on fail it returns block + reason and Claude Code
  feeds the reason back as instruction and **does not let Claude stop**. `stop_hook_active` guards
  infinite loops.
- **PreToolUse hook can deny/modify/redirect tool calls** before execution → gate `git commit` /
  `git push` on judge sign-off. Read a verdict file; fail closed.
- Other events: `PostToolUse` (audit trail), `SubagentStart/Stop` (judge lifecycle),
  `PreCompact` (archive transcript before summarizing — important for long runs),
  `UserPromptSubmit` (inject current DECISIONS state each turn).
- **Checkpointing** = native undo for autonomous runs — critical safety net.
- **Sessions: resume + fork.** Two-thread contract could be two sessions (long-lived planner;
  forked implementer at decision points) rather than two processes — more native.

### 6.3 Native primitives already in this environment (don't reinvent)
- `loop` skill — primitive autonomous loop. Dark-factory = opinionated `loop` + verify gates +
  decision-wall. Build on it, don't replace it.
- `verify` skill — *is* the flow-verifier ("exercise the affected flow, not just tests"). The
  flow-verifier agent should **call `verify`**, not reimplement it.
- `code-review` skill — diff review at low/med/high effort. Natural input to one judge.
- `webapp-testing` skill + Playwright MCP — canonical visual-judge harness. The visual-judge agent
  should **be `webapp-testing` invoked by a subagent**, not a bespoke Playwright script.

### 6.4 Ranked adoption recommendations (from the native research)
1. Express each judge lens as `.claude/agents/<lens>-judge.md` — read-only tools, model/effort
   matched to stakes, description as trigger condition.
2. Replace DECISIONS.md-as-wall *convention* with a **Stop hook** + PreToolUse gate on
   `git commit`/`git push`. Convention can be ignored by an autonomous loop; a hook cannot.
3. Cut SKILL.md < 500 lines; push rubrics/templates/banner-formats into `references/*.md` (one
   level deep). Apply "Claude is already smart" to every paragraph.
4. Move project-specific routing (owner table, channels, heat) into **CLAUDE.md**; skill reads it.
   **This is the portability fix.**
5. Borrow spec-kit's artifact-per-phase convention (spec/plan/tasks as durable judge inputs).
6. Steal spec-kit's template-override precedence (project overrides without forking).
7. Express flow-verifier + visual-judge as **LOW-freedom executable scripts** Claude runs via bash
   (script code stays out of context; only output enters).
8. Make judge count + turn budget a function of task complexity, not a constant 3.
9. Re-evaluate multi-judge vs single-judge-with-rubric (Anthropic's data favors single on
   consistency; defend multi on coverage + eval it).
10. Consider sessions (fork/resume) instead of two processes for the two-thread split.

### 6.5 What this settles for dark-factory
- **Portability (§5 open question):** answered — approach **B (config/override layer)**, modeled on
  spec-kit's precedence (project overrides > core defaults), AND move all project specifics to
  `CLAUDE.md` per Skills best-practice. Not env-detection.
- **Single-writer softness (§3.1):** answered — back it with a PreToolUse/Stop hook.
- **Auto-commit risk (§3.2):** answered — PreToolUse gate on `git commit`/`git push`, fail closed
  without judge sign-off; push stays forbidden.
- **Visual-judge fragility (§3.5):** answered — make it `webapp-testing` + Playwright MCP, not
  bespoke; require console log + exact command per finding.
- **Flow-verifier (§3.8):** answered — have it call the native `verify` skill; emit a
  machine-readable verdict file the commit-gate reads.

### 6.6 Community / analogous tooling — *from knowledge (verify after 2026-07-18)*

⚠️ **Provenance:** live web research is blocked this session — the Anthropic 5h usage cap killed
the sub-agents (resets 2026-07-05 07:18) AND the web-search service's monthly quota is exhausted
(resets **2026-07-18 19:30**). The entries below are from training knowledge (cutoff Jan 2026),
**not verified against current sources this session**. Confidence flagged per row. Re-verify the
repo slugs and current mechanics before relying on any of it. Treat as a research plan + provisional
findings, not ground truth.

| Framework | What it does | Mechanism relevant to dark-factory | Portability approach | Confidence |
|---|---|---|---|---|
| **BMAD-METHOD** (`bmad-code-org/BMAD-METHOD`) | Multi-agent "agile AI" methodology: personas (Analyst, Architect, PM, Scrum Master, Dev, QA, UI/UX, Data Eng…) each as their own agent file; workflow brainstorms → PRD → architecture brief → stories w/ file paths → implement → QA verify | **Closest roster analog.** Persona-per-agent-file, named roles, installers for Cursor/Roo/Claude Code. Leaner than BMAD is the lesson — it over-processes; dark-factory's 4-agent roster is better calibrated. | **Config file** (`.bmad-core/core-config.yaml`): models per agent, dev-story settings. → approach **B**. | Med-high on pattern, med on specifics |
| **Aider architect mode** (`Aider-AI/aider`) | Two-model: a strong *architect* model reasons/plans in plain English; a cheaper *editor* model applies SEARCH/REPLACE blocks to implement | **Validates the two-thread / plan-edit split.** Aider's whole finding: separating the *planning* context from the *editing* context improves results — and lets you spend model budget where it matters (plan strong, edit cheap). Same thesis as dark-factory's planner/implementer, but split by model not by session. | **Stack-agnostic + git-aware.** No project-type config; it just edits files in any repo. Lesson for dark-factory: **assume less about the stack.** | High (stable feature) |
| **Roo Code custom modes** (`RooCodeInc/Roo-Code`) | VS Code agent; modes incl. `Architect`, `Code`, `Ask`, `Debug`, user-defined; each mode has its own system prompt + tool access + approval mode | **Closest ACI analog + read-only-by-config.** `Architect` mode is *read-only* (plans, can't write) — enforced by mode config, not prompt. This is exactly dark-factory's read-only-judges idea, done at the platform level. | **`.roomodes` JSON per project** (shareable). → approach **B**. | High on pattern, med on current names |
| **SuperClaude** (`obra/superclaude`?) | A Claude Code "distro": bundled personas + slash commands + MCP configs + subagents | **Breadth, not depth.** Structurally different — it ships *many* commands; dark-factory is *one* deep autonomous loop. Relevant mainly as a packaging/distribution reference, not a mechanism one. | Global Claude-Code install, not project-typed. | Med (evolved fast; may conflate versions) |
| **OpenHands** (`All-Hands-AI/OpenHands`, ex-OpenDevin) | Open-source autonomous SWE runtime; observe-think-act loop over bash/file/browser tools; ships an eval harness | The loop shape (universal-tool bash, eval as ground truth). It's a full **runtime**, not a skill — different layer. | Repo-agnostic runtime. Verification = **the repo's own test suite + SWE-bench-style hidden tests.** | High on shape, med on current specifics |
| **SWE-agent** (`princeton-nlp/SWE-agent`) | Princeton agent; core idea is **agent-computer interface (ACI)** design — the tools/feedback the agent sees matter more than the model | **The ACI lesson + objective verification.** They found self-review weak; ground truth = passing the hidden test suite, not an LLM judge. dark-factory's tool choices (read-only judges, flow-verifier) *are* its ACI — design them deliberately. | Repo-agnostic. | High on ACI insight |

### 6.7 Two cross-cutting findings from the community thread

**Finding A — portability: both threads converge on approach B.**
The Anthropic-native thread (§6.2 spec-kit) and the community thread (BMAD `core-config.yaml`,
Roo `.roomodes`) **agree**: a per-project config/override file is the convention for staying
project-agnostic. This raises confidence in §5's resolution (approach B + move specifics to
CLAUDE.md) from "my lean" to "the established pattern across ecosystems." Aider is the lone
exception (stack-agnostic-by-assuming-nothing) — a useful secondary principle: **the less the skill
assumes about the stack, the less config it needs.**

**Finding B — verification: a real tension between LLM-judges and objective tests.**
dark-factory verifies with **LLM judges** (noisy, semantic). SWE-agent / OpenHands verify with the
**repo's test suite / SWE-bench** (objective, deterministic). SWE-agent explicitly found self-review
weak and outsourced truth to tests. Combined with Anthropic's single-judge-beats-multi caveat
(§6.2.1), this sharpens the recommendation:
- **Prefer objective verification wherever it can reach** — tests, byte-diffs, schema validation,
  exit codes. dark-factory's flow-verifier already does some of this ("byte-identical when off" →
  diff it); promote it to the *primary* gate.
- **Use LLM judges only where objective checks can't reach** — design/UX, spec-conformance,
  "does this match intent." That is exactly the visual-judge's job (findings, not pass/fail).
- **Don't stack LLM judges for consistency** (Anthropic data says it doesn't help); stack them for
  *coverage* (different lenses) only where the change is high-stakes and semantically rich.

Net: dark-factory's "never self-review" is right and well-supported; its "verify with 2–3 LLM
judges by default" is the part most worth re-examining against objective-test-first verification.

### Community sources (verify slugs — repos move)
- BMAD-METHOD: `github.com/bmad-code-org/BMAD-METHOD`
- Aider: `github.com/Aider-AI/aider` (formerly `paul-gauthier/aider`)
- Roo Code: `github.com/RooCodeInc/Roo-Code`
- SuperClaude: `github.com/obra/superclaude` (slug uncertain — verify)
- OpenHands: `github.com/All-Hands-AI/OpenHands`
- SWE-agent: `github.com/princeton-nlp/SWE-agent`

### Sources (native thread)
- [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)
- [Equipping agents for the real world with Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)
- [How and when to use subagents in Claude Code](https://claude.com/blog/subagents-in-claude-code)
- [Subagents in the SDK](https://code.claude.com/docs/en/agent-sdk/subagents)
- [Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview) · [Hooks](https://code.claude.com/docs/en/agent-sdk/hooks)
- [Agent Skills overview](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) · [Best practices](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices)
- [github/spec-kit](https://github.com/github/spec-kit) · [GitHub blog post](https://github.blog/ai-and-ml/generative-ai/spec-driven-development-with-ai-get-started-with-a-new-open-source-toolkit/)

---

## 7. Recommendations for the actual skill

*Distilled advice. The authoritative ranked list is §6.4 (native research); below is the subset to
act on first, merged with the pre-research weak-points work. Community findings (§6.6) are
provisional (from knowledge, verify after 2026-07-18) and add one actionable point — §6.7 Finding B:
verify **objective-test-first**, reserve LLM judges for semantic ground they alone cover.*

**Top 5, ordered by leverage (research-weighted):**
1. **Split project specifics into `CLAUDE.md`; skill reads them.** The portability fix. Owner
   routing, invariants, fixture path, build command, design-principles doc — all leave the skill.
2. **Back the rules with hooks, not just prompts.** Stop hook enforces DECISIONS-as-wall;
   PreToolUse gates `git commit`/`git push` on judge sign-off (fail closed). This cures §3.1 and
   §3.2 — the two highest-severity weak points — with native mechanism.
3. **One `.claude/agents/<lens>-judge.md` per lens**, read-only by config, `effort`/`model` matched
   to stakes, description written as a trigger. The roster IS the native pattern.
4. **Cut SKILL.md < 500 lines; push rubrics/templates/banner-format into `references/*.md` (one
   level deep).** Apply "Claude is already smart" to every paragraph.
5. **Flow-verifier calls native `verify`; visual-judge is `webapp-testing` + Playwright MCP.** Don't
   reimplement shipped primitives. Emit a machine-readable verdict file the commit-gate reads.

### Keep
- The three rules (single-writer, never self-review, decisions-as-walls) — verbatim in spirit.
- Refutation-first judges + "list the attacks you tried" rule.
- Flow verifier's "claims are test cases" method.
- Banner > queue truth hierarchy; scheduled-not-polled validation.
- Visual judge returns findings-not-pass/fail.
- Minimum-viable-factory escape hatch.

### Change
- Strip every videoED-specific identifier (see §4). Rename `vd-*` → `df-*`.
- Replace the dangling `videoed-*` memory slugs with inlined one-line rules.
- Move the release-ritual mechanics, the invariants block, the fixture path, the UI playbook, and
  the design-principles reference behind a per-project config (approach B in §5, pending research).
- Generalize the fail-closed lens and the visual-judge principles away from videoED domain language.

### Harden
- Add a soft guardrail for single-writer (pre-commit or `PreToolUse` hook) — convention alone is too
  soft (§3.1).
- Make the no-push rule explicit and absolute; make the release-ritual opt-in affirmative (§3.2).
- Match the visual-judge engine to the project's real runtime; require console logs + exact command
  with every finding (§3.5).
- Key verification effort to task size/risk, not one-size-fits-all (§3.6).
- Emit a machine-readable run log alongside the wrap-up message (§3.8).

---

## 8. Hook enforcement map — the guardrail layer

dark-factory's load-bearing rules currently rely on **prompt discipline**. Native Claude Code hooks
turn each one into a **runtime guardrail an autonomous loop cannot ignore**. This is the single
biggest upgrade available: it converts the contract from "conventions a tired model can violate"
into "a runtime the model can't violate." Each hook cures a specific weak point from §3.

| dark-factory rule | Hook event | Mechanism | Cures |
|---|---|---|---|
| **Single-writer areas** (planner vs implementer file ownership) | `PreToolUse` on `Edit`/`Write`/`MultiEdit` | read `file_path` from the tool input; check against the current thread's allowlist; deny with a reason if out of lane | §3.1 |
| **DECISIONS-as-wall** (stop cleanly at human decisions) | `Stop` | scan `tasks/DECISIONS.md` for OPEN rows blocking in-flight work; return `{"decision":"block","reason":"OPEN D<n>: …"}`; the loop cannot declare done past one | §3.1 / stop-conditions |
| **No commit without judge sign-off** | `PreToolUse` on `Bash` matching `git commit`/`git tag` | require a fresh `.dark-factory/verdicts/<task>.json` (all lenses green) to exist and be newer than the last edit; fail closed | §3.2 |
| **Never push** (tags stay local) | `PreToolUse` on `Bash` matching `git push` | always deny in factory context | §3.2 |
| **Run log / telemetry** | `PostToolUse` | append every file change + every verdict to `.dark-factory/run-<date>.jsonl` | §3.8 |
| **Aggregate judge verdicts** | `SubagentStart` / `SubagentStop` | collect each judge's structured output into the verdict file the commit-gate reads | §3.6 / §3.8 |
| **Long-run context survival** | `PreCompact` | archive transcript + current QUEUE/DECISIONS state before summarization | (new — unattended runs) |
| **Inject state every turn** | `UserPromptSubmit` | prepend current QUEUE rank / blocked task / OPEN decisions into each prompt so the loop never loses the plot | (new) |

### Two hard cautions from the SDK docs
- **Scope every hook to top-level only** (check for subagent context inside the hook). Anthropic
  warns a `UserPromptSubmit` hook that itself spawns subagents can recurse infinitely.
- **Respect `stop_hook_active`.** The Stop hook must no-op when the guard is set, or it loops.

### Pair with checkpointing
Native **checkpointing** ("rewind file changes") is the safety net for the case a hook didn't catch
— undo the last autonomous burst if a wrap-up review finds a bad change slipped through. Hooks =
prevention; checkpoints = recovery. Ship both before trusting unattended runs.

### Why this is the headline recommendation
The native research's #2 ranked adoption item is "replace the DECISIONS-as-wall *convention* with a
Stop hook + PreToolUse gate." That single change retires the two highest-severity weak points
(§3.1 single-writer-by-prompt, §3.2 auto-commit-without-sign-off) using mechanism the platform
already gives you — no new discipline required from the model, no honor system.

---

## 9. Naming — DECIDED 2026-07-05

**The name stays `dark-factory`; `moonflower` is the codename.**

- **Name** (commands, agents, folders, marketplace entry): `dark-factory`. The agent
  naming convention stays — `dark-factory-judge` reads like a badass cyberpunk villain,
  and that's a feature.
- **Codename** (apothecary theme, garden ledger, lore/Bloom contexts): **moonflower** —
  night-blooming datura: powerful, toxic if mishandled, and it only opens in the dark.
  Reconciles the shelf's "medicinal plants, not dark" tagline with a tool that literally
  runs while the lights are off: the factory doesn't break the garden theme, it's the
  night shift of it.
- If a full rename ever happens anyway: one folder + `name` fields in plugin.json /
  marketplace.json / SKILL.md frontmatter; agents keep `dark-factory-*` regardless.
