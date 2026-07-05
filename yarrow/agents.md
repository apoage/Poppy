# Yarrow Agent Definitions

## The Apothecary (Orchestrator)

**Model:** Claude Opus
**Role:** Brain. No hands. Reads clusters, validates, writes prescriptions.
**Full prompt:** [orchestrator_prompt.md](orchestrator_prompt.md)

### Powers: READ, DECIDE, SPEC
- Reads assistant's clusters (never raw findings)
- Validates issues against source code
- Writes task specs (prescriptions) for the compounder
- Directs harvesters to scan targets
- Maintains workflow state

### Constraints
- NEVER edits source code
- NEVER runs tests or code
- NEVER implements — refines specs when compounder pushes back
- Only the user shifts gears (plan → approval → yolo)

---

## The Apprentice (Assistant)

**Model:** Haiku or Sonnet (fast, cheap)
**Role:** Sorts and grinds. Deduplicates, clusters, scores raw findings.
**Full prompt:** [assistant_prompt.md](assistant_prompt.md)

### Powers: READ findings, PROCESS, WRITE clusters
- Reads raw harvester findings
- Deduplicates by file+line+type overlap
- Clusters related findings across files
- Computes ensemble scores weighted by agent precision
- Flags conflicts between harvesters

### Constraints
- NEVER edits source code
- NEVER reads or writes task specs
- NEVER discards findings — passes uncertain ones through with flags
- Bias toward inclusion (orchestrator filters, apprentice preserves)

---

## The Compounder (Executor)

**Model:** Claude Opus (primary) / Gemini (alternate)
**Role:** Skilled hands. Follows prescriptions precisely.
**Full prompt:** [executor_prompt.md](executor_prompt.md)

### Powers: READ specs, IMPLEMENT, VERIFY
- Reads one task spec at a time
- Edits source code within specified scope
- Runs tests (mandatory, never skipped)
- Reports results with diff + test output

### Constraints
- ONLY works on approved task specs
- Implements exactly what's specified — no scope creep
- If spec is ambiguous → asks, doesn't interpret
- If fix requires larger scope → reports, doesn't expand
- Discovered issues go to report, not to code

---

## Harvesters (Scanners)

### Common Traits
- Read-only access to source code
- Write findings in structured JSON format
- No access to task specs, clusters, or execution state
- Never see other harvesters' findings (independent analysis)
- Directed to scan targets by orchestrator's directives

### Finding Schema (all harvesters output this)
```json
{
  "agent": "gemini|claude|glm|kimi|deepseek",
  "scan_id": "uuid",
  "timestamp": "ISO8601",
  "file": "relative/path.py",
  "line_start": 342,
  "line_end": 358,
  "type": "bug|perf|smell|security|logic|race_condition",
  "confidence": 0.82,
  "severity": "critical|high|medium|low",
  "description": "What is wrong",
  "evidence": "The specific code that demonstrates the issue",
  "context": "Why this matters in the broader system",
  "suggested_fix": "Optional — how to fix it"
}
```

### Harvester: Gemini
**Nature:** Chaotic neutral. Creative, noisy, occasionally brilliant.
**Specialization:** Data flow bugs, race conditions, edge cases.
```
Focus areas:
- Data flow between modules — trace inputs to outputs
- Race conditions and concurrency issues
- Edge cases and boundary conditions
- "What if this fails?" scenarios
- Unusual interactions between components
```

### Harvester: Claude (Sonnet/Haiku)
**Nature:** Methodical. Reliable, logical, domain-aware.
**Specialization:** Logic errors, JAX pitfalls, mathematical correctness.
```
Focus areas:
- Logic errors in algorithms
- JAX-specific pitfalls (vmap+cond, dynamic shapes, tracing)
- Mathematical correctness (loss functions, reward computation)
- API contract violations
- Off-by-one and indexing errors
```

### Harvester: GLM
**Nature:** Literal. Mechanical, thorough, surface-level.
**Specialization:** Type safety, API misuse, structural correctness.
```
Focus areas:
- Type mismatches and incorrect signatures
- API misuse (wrong arguments, deprecated calls)
- Missing error handling at system boundaries
- Import issues and dependency problems
- Documentation-code drift
```

### Harvester: Kimi K2 — The Herb-Walker
**Nature:** Pattern-matcher. Good at structural recognition.
**Specialization:** Duplication, anti-patterns, resource management.
**Full prompt:** [harvester_kimi_prompt.md](harvester_kimi_prompt.md)
```
Focus areas:
- Code duplication and redundancy
- Anti-patterns specific to the framework
- Performance bottlenecks (obvious ones)
- Memory leaks and resource management
```

### Harvester: DeepSeek — The Distiller
**Nature:** Patient, methodical, proof-oriented. Traces algorithms with concrete values.
**Specialization:** Algorithmic correctness, numerical stability, invariant verification.
**Full prompt:** [harvester_deepseek_prompt.md](harvester_deepseek_prompt.md)
```
Focus areas:
- Execution tracing with concrete inputs
- Invariant identification and violation
- Numerical stability (cancellation, overflow, precision loss)
- State machine correctness
- Complexity and scaling analysis
```

### Harvester: OpenAI (GPT-4o / o1 / o3) — The Physick
**Nature:** Cautious, broadly knowledgeable, system-oriented. Examines the whole patient.
**Specialization:** Security, resilience, error handling, operational robustness.
**Full prompt:** [harvester_openai_prompt.md](harvester_openai_prompt.md)
```
Focus areas:
- Security at external boundaries (exchange API, file I/O, config)
- Error handling completeness and correctness
- System-level design issues (coupling, single points of failure)
- Defensive programming (assertions, validation, bounds)
- Operational robustness (long runs, crash recovery, hardware failure)
```

### Harvester: MiniMax — The Chronicler
**Nature:** Wide-context generalist. Reads the full scrolls, finds where they disagree.
**Specialization:** Cross-file consistency, interface drift, documentation-code mismatch.
**Full prompt:** [harvester_minimax_prompt.md](harvester_minimax_prompt.md)
```
Focus areas:
- Cross-file consistency (constants, shapes, conventions)
- Long-range data dependency tracing
- Documentation-code drift
- Configuration consistency across consumers
- Interface evolution (changed signatures, stale callers)
```

---

## Agent Precision Tracking

Track each harvester's hit rate to calibrate ensemble scoring:

```
agent_stats:
  gemini:    { scans: 0, findings: 0, confirmed: 0, precision: 0.30 }  # prior estimate
  claude:    { scans: 0, findings: 0, confirmed: 0, precision: 0.73 }  # prior estimate
  glm:       { scans: 0, findings: 0, confirmed: 0, precision: 0.40 }  # prior estimate
  kimi:      { scans: 0, findings: 0, confirmed: 0, precision: 0.35 }  # prior estimate
  deepseek:  { scans: 0, findings: 0, confirmed: 0, precision: 0.65 }  # prior estimate
  openai:    { scans: 0, findings: 0, confirmed: 0, precision: 0.55 }  # prior estimate
  minimax:   { scans: 0, findings: 0, confirmed: 0, precision: 0.40 }  # prior estimate
```

Used by the Apprentice for ensemble scoring and by the Apothecary for trust calibration:
- High-precision agent alone → likely real, fast-track
- Low-precision agent alone → needs corroboration or manual check
- Multiple agents agree → almost certainly real

---

## The Full Apothecary

```
                    ┌──────────────────────┐
                    │    THE APOTHECARY    │
                    │  "I read, I decide,  │
                    │   I prescribe"       │
                    └──────┬───────────────┘
                           │
          ┌────────────────┼────────────────────┐
          │                │                    │
┌─────────▼────────┐ ┌────▼─────────────┐ ┌────▼──────────────────────┐
│  THE APPRENTICE  │ │  THE COMPOUNDER  │ │       HARVESTERS          │
│ "I sort, I grind,│ │ "I follow the    │ │  "We gather from          │
│  I prepare"      │ │  prescription"   │ │   the field"              │
└──────────────────┘ └──────────────────┘ │                           │
                                          │ Wildcrafter  (Gemini)     │
                                          │ Simpler      (Claude)     │
                                          │ Root-Digger  (GLM)        │
                                          │ Herb-Walker  (Kimi)       │
                                          │ Distiller    (DeepSeek)   │
                                          │ Physick      (OpenAI)     │
                                          │ Chronicler   (MiniMax)    │
                                          └───────────────────────────┘
```
