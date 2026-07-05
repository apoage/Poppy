# Yarrow Architecture

## System Overview

```
                         ┌─────────┐
                         │   YOU   │
                         │ (gear)  │
                         └────┬────┘
                              │ plan / approve / yolo / stop
                              ▼
                     ┌────────────────┐
                     │  ORCHESTRATOR  │
                     │   (Claude)     │
                     │                │
                     │ - validates    │
                     │ - prioritizes  │
                     │ - writes specs │
                     └───┬───────┬───┘
                    read │       │ dispatch
              ┌─────────┘       └──────────┐
              ▼                             ▼
     ┌─────────────────┐          ┌─────────────┐
     │   ASSISTANT      │          │  EXECUTOR   │
     │  (Haiku/Sonnet)  │          │ (Claude/Gem) │
     │                  │          │              │
     │ - deduplicates   │          │ - implements │
     │ - clusters       │          │ - tests      │
     │ - summarizes     │          │ - reports    │
     └────────┬────────┘          └─────────────┘
              │ reads
              ▼
     ┌──────────────┐
     │  FINDINGS DB  │
     └──────────────┘
           ▲ ▲ ▲
           │ │ │  write findings
     ┌─────┘ │ └─────┐
     ▼       ▼       ▼
  ┌──────┐┌──────┐┌──────┐
  │Harvester││Harvester││Harvester│
  │Gemini││Claude││ GLM  │
  └──────┘└──────┘└──────┘
```

## Data Flow

```
1. Harvesters scan code → write findings to findings DB (structured JSON)
2. Assistant reads raw findings → deduplicates, clusters, summarizes
3. Orchestrator reads clusters → validates, prioritizes, writes task specs
4. User approves task specs
5. Executor reads approved tasks → implements → reports result
6. Orchestrator verifies result → marks done or sends back
```

## Separation of Concerns

### Hard Boundaries (NEVER cross)

| Agent | Can Read | Can Write | Cannot |
|-------|----------|-----------|--------|
| Harvester | Source code, docs | Findings DB only | Edit source, write tasks |
| Orchestrator | Everything | Task specs, workflow state | Edit source code |
| Executor | Task specs, source code | Source code, execution results | Write task specs, modify findings |

### Why This Matters

- **Orchestrator can't implement** → no "just quickly fix this" scope creep
- **Harvesters can't edit** → no drive-by fixes that skip review
- **Executor can't scope** → implements exactly what's specified, nothing more

## Storage Layer

### Files (simple, inspectable, git-trackable)

```
yarrow/
  workspace/
    findings/          # Harvester outputs (JSON per scan)
    tasks/             # Task specs (MD per task)
    executions/        # Executor results (diff + report)
    state.json         # Current workflow state
```

### Database (queryable, persistent across sessions)

```sql
-- Core tables
findings     (id, agent, file, line_start, line_end, type, confidence,
              description, evidence, suggested_fix, status, created_at)

tasks        (id, source_finding_ids, spec, priority, assigned_to,
              status, created_at, updated_at)

executions   (id, task_id, agent, diff_path, test_result,
              validation_status, created_at)

-- Tracking
agent_stats  (agent, total_findings, confirmed, false_positive, precision)
```

SQLite is sufficient. Single file, no server, portable.

## Deduplication Strategy

Findings from multiple harvesters often describe the same issue differently.
The **assistant** (cheap model — Haiku/Sonnet) handles deduplication so the
orchestrator's context stays clean. Deduplication uses:

1. **File + line range overlap** — findings within 10 lines of each other
   on the same file are candidate duplicates
2. **Type match** — same type (bug, perf, smell) raises duplicate probability
3. **Semantic check** — orchestrator reads both descriptions, confirms same root cause
4. **Merge** — keep highest-confidence version, note all contributing agents

## Ensemble Voting

When multiple harvesters flag the same area:

```
1 harvester  → low confidence, orchestrator investigates before tasking
2 harvesters → medium confidence, likely real
3 harvesters → high confidence, fast-track to task spec
```

Per-agent precision tracking adjusts weight over time:
- High-precision agent solo finding → trust it
- Low-precision agent solo finding → needs corroboration
