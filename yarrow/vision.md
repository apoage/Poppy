# Yarrow — Vision

**Status: Architecture complete. Implementation pending.**

Named for Achillea — a compound flower where dozens of tiny blooms work as one.
Used in the I Ching for decision-making. Stops bleeding — controls the response.

---

## The Insight

One agent is a monologue. Multiple agents with no structure is noise.
Yarrow is **structured disagreement** — diverse perspectives, forced independence,
noise reduction, then decisive action.

```
Diverse perspectives → noise reduction → validated decision → precise execution
```

The architecture is not a code review tool. It's a **decision-quality pattern**
that generalizes to any domain where:
- Multiple valid interpretations exist
- No single perspective is reliably correct
- The cost of a wrong action exceeds the cost of deliberation
- Trust must be earned, not assumed

Code review was the first domain. Trading decisions, architecture design, research
analysis — all fit the same pipeline.

## Why Not Just "Throw N Agents At It"

Every multi-agent system degrades toward one of two failure modes:
1. **Groupthink** — agents see each other's work, converge on the same answer
2. **Chaos** — agents contradict each other, no mechanism to resolve

Yarrow prevents both through structural constraints:

| Rule | Prevents |
|------|----------|
| Harvesters can't edit code | Drive-by fixes that skip review |
| Harvesters can't see each other | Groupthink, confirmation bias |
| Orchestrator can't implement | "Just quickly fix this" scope creep |
| Executor can't scope | Gold-plating, unauthorized expansion |
| Assistant can't discard | Over-filtering before the brain sees it |
| User controls gear shifts | System never self-promotes to autonomous |

Remove any boundary and the system degrades toward "one agent doing everything" —
which is what every other system already is.

---

## The Apothecary's Workshop

The metaphor isn't decoration — it IS the architecture. An apothecary works with
powerful, dangerous materials. Belladonna heals eyes or kills. Poppy stops pain
or stops breathing. The entire profession exists because these substances demand protocol.

LLMs are the dangerous materials — powerful, unpredictable, capable of brilliance or
hallucination. A swarm throws them at the problem and hopes. An apothecary shop
handles them with the respect they require: every role constrained, every handoff
inspectable, trust earned by track record.

### The Roles

```
                         ┌─────────┐
                         │   YOU   │
                         │ (gear)  │
                         └────┬────┘
                              │ plan / approve / yolo / stop
                              ▼
                    ┌──────────────────┐
                    │  THE APOTHECARY  │  Reads clusters. Validates. Prescribes.
                    │  (Claude Opus)   │  CANNOT act. CANNOT implement.
                    └──┬───────────┬───┘
                  read │           │ dispatch
            ┌─────────┘           └──────────┐
            ▼                                ▼
   ┌──────────────────┐            ┌──────────────────┐
   │  THE APPRENTICE  │            │  THE COMPOUNDER   │
   │  (Haiku/Sonnet)  │            │  (Opus/Gemini)    │
   │                  │            │                   │
   │  Deduplicates    │            │  Follows the      │
   │  Clusters        │            │  prescription     │
   │  Scores          │            │  exactly           │
   └────────┬─────────┘            └───────────────────┘
            │ reads
            ▼
   ┌──────────────┐
   │  FINDINGS DB  │
   └──────────────┘
      ▲ ▲ ▲ ▲ ▲ ▲ ▲   independent, isolated, never see each other
      │ │ │ │ │ │ │
   ┌──┘ │ │ │ │ │ └──┐
   ▼    ▼ ▼ ▼ ▼ ▼    ▼
  7 HARVESTERS — different models, different providers, different eyes
```

### The Seven Harvesters

Each harvester is named for their method of gathering, and tuned to a specific
model's strengths:

| Name | Model | Territory | Nature |
|------|-------|-----------|--------|
| **Wildcrafter** | Gemini | Data flow, race conditions, edge cases | Forages the untamed margins. Creative chaos — finds what garden-keepers miss. |
| **Simpler** | Claude | Logic errors, JAX pitfalls, math | Knows each plant by name. Reasons through mechanism, not pattern. Most trusted. |
| **Root-Digger** | GLM | Types, signatures, API misuse | Works below the surface. Structural precision — provable mismatches. |
| **Herb-Walker** | Kimi K2 | Duplication, dead code, patterns | Surveys the whole garden from above. Reports health, not individual ailments. |
| **Distiller** | DeepSeek | Algorithms, numerics, invariants | Patient prover. Traces code with concrete values until contamination concentrates and becomes visible. |
| **Physick** | OpenAI | Security, resilience, error handling | The physician. Examines the living system — checks pulse, tests reflexes, looks for systemic illness. |
| **Chronicler** | MiniMax | Cross-file consistency, drift | Reads the long scrolls. Finds where the records disagree — what was written vs what exists. |

**Why seven different models instead of seven instances of the best one?**

Same model twice gives you parallelism — faster coverage of the same perspective.
Different models give you **actual diversity of failure modes**. Gemini hallucinates
differently from Claude. DeepSeek reasons about numerics differently from GPT-4.
GLM's training data gives it different structural intuitions than Kimi.

The ensemble isn't faster. It's **less wrong**.

### Why These Particular Territories

The seven territories were chosen to minimize overlap and maximize coverage:

```
                        CORRECTNESS AXIS
                              │
          Logic ──────────────┼────────────── Structure
                              │
    Simpler (reasoning)       │       Root-Digger (types/API)
    Distiller (proofs)        │       Chronicler (cross-file)
                              │
          ────────────────────┼─────────────────────
                              │
    Wildcrafter (edge cases)  │       Herb-Walker (patterns)
    Physick (resilience)      │
                              │
          Dynamic ────────────┼────────────── Static
                              │
                        ANALYSIS AXIS
```

- **Left side** (Logic/Dynamic): finds bugs that require *reasoning* — tracing values, understanding flow, proving incorrectness
- **Right side** (Structure/Static): finds bugs that require *checking* — matching signatures, comparing files, measuring health
- **Top** (Correctness): "is this code right?"
- **Bottom** (Robustness): "will this code survive?"

No single model covers all four quadrants. The ensemble does.

---

## The Pipeline (Domain-Agnostic)

```
1. DIRECT    — Apothecary tells harvesters WHERE to look (not what to find)
2. HARVEST   — N models independently scan, write structured findings
3. REDUCE    — Apprentice deduplicates, clusters, computes ensemble scores
4. DECIDE    — Apothecary validates clusters, writes prescriptions (task specs)
5. APPROVE   — User reviews, approves/rejects/modifies
6. EXECUTE   — Compounder implements approved specs, reports results
7. VERIFY    — Apothecary checks results, marks done or sends back
```

Steps 1-7 are the same regardless of domain. What changes per domain:
- The **directive** (what to scan)
- The **finding schema** (what a finding looks like)
- The **action spec** (what the executor does)

## Domain Instantiation

### Code Review (current — fully specified)
```
Directive:    "Scan mcts_trader/jax_agent/ for bugs"
Finding:      { file, line, type: bug|perf|smell, confidence, evidence, suggested_fix }
Action spec:  Task spec with problem, evidence, specification, acceptance criteria
Executor:     Edits source code, runs tests, reports diff + test results
Harvesters:   All 7 (or subset based on target type)
```

### Architecture Design
```
Directive:    "Propose approaches for distributed training"
Finding:      { approach, tradeoffs, complexity, evidence, risks }
Action spec:  "Implement approach B: parameter server with async updates"
Executor:     Writes design doc or prototype code
Harvesters:   Simpler (reasoning), Physick (robustness), Chronicler (consistency)
```

### Research Analysis
```
Directive:    "Analyze this paper's claims against our codebase"
Finding:      { claim, evidence_for, evidence_against, applicability, effort }
Action spec:  "Implement technique X from section 3.2, adapted for JAX"
Executor:     Implements the technique
Harvesters:   Simpler (logic), Distiller (math), Chronicler (fit with existing)
```

### Trading Decision
```
Directive:    "Evaluate current portfolio given market state"
Finding:      { asset, signal, confidence, timeframe, reasoning }
Action spec:  "Rebalance: reduce BTC 5%, increase ETH 3%, hold rest"
Executor:     Executes trades via exchange API
Harvesters:   Distiller (numerics), Wildcrafter (risk), Physick (operational safety)
```

### Incident Response
```
Directive:    "Training crashed at step 45000. Diagnose."
Finding:      { component, failure_mode, evidence, root_cause_candidate }
Action spec:  "Fix the memory leak in JIT cache accumulation"
Executor:     Implements fix, runs stress test
Harvesters:   Wildcrafter (trace failure cascade), Simpler (root cause), Physick (recovery)
```

---

## Ensemble Voting

The key differentiator. Not "union of all findings" but **weighted consensus**:

```
1 harvester flags it          → low confidence, investigate
2 harvesters flag same thing  → medium, likely real
3+ harvesters agree           → high, fast-track

Weight adjusted by precision history:
  Simpler alone at 0.8        → probably real (~73% historical precision)
  Wildcrafter alone at 0.8    → maybe real (~30% historical precision)
  Wildcrafter + Root-Digger   → unusual pair, strong signal (rarely overlap)
  Simpler + Distiller         → two provers agree, near certain
```

Precision tracked per agent, updated after every orchestrator validation cycle.
Over time, the system learns which harvesters to trust for which types of findings.

---

## Communication: Files, Not Wires

All inter-agent communication through the filesystem:

```
yarrow/workspace/
├── directives/       # Apothecary → Harvesters (where to look)
├── findings/         # Harvesters → DB (structured JSON per scan)
├── clusters/         # Apprentice → Apothecary (deduplicated, scored)
├── tasks/            # Apothecary → Compounder (prescriptions)
├── executions/       # Compounder → Apothecary (diffs, reports, outcomes)
├── state.json        # Current mode, active tasks, agent status
└── agent_stats.json  # Precision tracking per harvester
```

Why files:
- **Inspectable** — you can read any message at any time
- **Git-trackable** — version control the decision process itself
- **Crash-proof** — recovery = re-read files
- **Model-agnostic** — any model that reads/writes JSON works
- **Debuggable** — when something goes wrong, the evidence trail exists

SQLite mirrors the filesystem for queries. Either rebuilds from the other.

---

## Mode State Machine

```
        user: "plan"           user: "approved"          user: "yolo"
IDLE ────────────→ PLAN MODE ────────────────→ APPROVAL ──────────→ YOLO MODE
  ▲                  │   ▲                       │                     │
  │                  │   │ needs more research    │ user: "reject"     │
  │                  │   └───────────────────────-┘                    │
  │                  │                                                 │
  │                  └─── user: "stop" ───────────────────────────────→│
  │                                                                   │
  └──────────── cycle complete OR user: "stop" OR uncertainty ────────┘
```

User controls gear shifts. The system never self-promotes to autonomous mode.
Uncertainty auto-stops execution and returns to PLAN.

---

## What This Is Not

- Not a chatbot swarm (OpenClaw) — those are personas, not specialists
- Not same-model parallelism (Claude Code Swarms) — same blind spots, just faster
- Not a wrapper around one model's context window
- Not an agent that "can do everything"

It's a **decision-quality amplifier**. Multiple diverse lenses, forced independence,
noise reduction, then action. The architecture encodes distrust — no single agent
is trusted to both diagnose and treat. That constraint is the feature.

---

## The Naming Philosophy

Yarrow exists in the apothecary's garden alongside:
- **Poppy** — dreams via MCTS planning (the trading agent)
- **Bella** — vision of unseen markets (belladonna, the market generator)
- **Foxglove** — careful dosage heals the heart (training orchestration)

All medicinal plants. All heal or kill depending on how carefully you work with them.
Not a theme — a philosophy. The names encode the same worldview as the architecture:
**powerful tools handled with protocol, not permission**.

Yarrow itself — the wound-healing herb that stops bleeding. It doesn't cause action,
it **controls the response**. A compound flower made of many tiny blooms working as
one organism. The name IS the architecture.

---

## Implementation Path

### Phase 1: Prompt Architecture (now)
Harvesters are subagents or separate terminal sessions. Orchestrator is the main
Claude Code session. Filesystem is the message bus. Works today, no new code needed.
Validate the pattern on real code review cycles.

### Phase 2: Script Orchestrator
Python script that calls APIs, manages state, routes findings, dispatches harvesters.
More control over scheduling, parallel execution, and retry logic. Build when the
filesystem bus or manual terminal management becomes a bottleneck.

### Phase 3: Runtime
WebSocket/MCP bus, queue-based dispatch, live dashboard, automatic harvester scaling.
Build when this is a product, not a tool. Premature until the pattern is validated
across 3+ domains.

### Adding a Harvester
1. Write a prompt (use existing prompts as template)
2. Define its territory (what it looks for, what it doesn't)
3. Give it read access to context and write access to `findings/`
4. Register in `state.json` and `agent_stats.json`
5. Apprentice auto-picks up new findings

### Adding a Domain
1. Define the finding schema
2. Define the action spec format
3. Write harvester directives for the domain
4. Executor needs domain-appropriate tools
5. Pipeline stays identical

---

## Metrics That Matter

Once running, track these to measure whether Yarrow actually improves decision quality:

```
Finding precision per harvester    — are we wasting tokens on noise?
Ensemble gain                      — do multi-harvester clusters outperform singles?
Time to validated decision         — is the pipeline faster than manual review?
False negative rate                — are real bugs getting through?
Orchestrator context efficiency    — how much context does the Apprentice save?
Cost per validated finding         — API tokens per real bug found
```

The system should pay for itself: more bugs found per hour of human attention,
with a clear audit trail of how each decision was reached.
