# Yarrow Workflow & Operational Modes

## Mode State Machine

```
          user: "plan"              user: "approved"           user: "yolo"
IDLE ──────────────→ PLAN MODE ──────────────────→ APPROVAL ────────────→ YOLO MODE
  ▲                    │    ▲                        │                      │
  │                    │    │ needs more research     │ user: "reject"      │
  │                    │    └────────────────────────-┘                     │
  │                    │                                                    │
  │                    └──── user: "stop" ─────────────────────────────────→│
  │                                                                        │
  └──────────── cycle complete OR user: "stop" OR uncertainty ─────────────┘
```

## Mode Definitions

### PLAN MODE (Research Phase)

**Orchestrator behavior:**
1. Dispatches harvesters to scan specified files/modules
2. Reads incoming findings from findings DB
3. Assistant deduplicates and clusters findings
4. Orchestrator validates clusters, assigns severity
5. Drafts task specs for confirmed issues
6. Presents summary to user

**What's happening:**
- Harvesters are active, writing findings
- Assistant is processing findings
- Orchestrator is reading and planning
- Executor is idle
- NO code changes

**Ends when:** Orchestrator presents task specs → transitions to APPROVAL

### APPROVAL MODE (User Checkpoint)

**User behavior:**
1. Reviews task specs drafted by orchestrator
2. Adjusts priority, scope, or approach
3. Approves, modifies, or rejects each task
4. Sets execution order if needed

**What's happening:**
- All agents paused waiting for user
- User has full control

**Ends when:** User approves tasks → transitions to YOLO (or rejects → back to PLAN)

### YOLO MODE (Execution Phase)

**Orchestrator behavior:**
1. Dispatches approved tasks to executor in priority order
2. Monitors executor progress
3. Validates execution results (runs tests or delegates validation)
4. Marks tasks done or sends back for revision
5. Optionally dispatches harvesters to scan changed files

**What's happening:**
- Executor is implementing approved tasks
- Orchestrator is monitoring and validating
- Harvesters may be scanning next area (pipelined)
- User watches from dashboard (optional)

**Auto-stops when:**
- All approved tasks completed
- Executor reports unresolvable conflict
- Test failures that don't match expectations
- Orchestrator encounters uncertainty

**Ends when:** Cycle complete → back to PLAN, or user: "stop"

---

## Typical Session Flow

```
Session Start
│
├─ 1. User: "plan — review mcts_trader/jax_agent/"
│     Orchestrator dispatches all harvesters to that module
│     Harvesters scan independently, write findings
│     Assistant deduplicates findings
│     Orchestrator validates, drafts 5 task specs
│
├─ 2. Orchestrator: "Found 5 issues. 2 critical, 2 medium, 1 low."
│     User reviews task specs
│     User: "approve tasks 1-3, reject 4, defer 5"
│
├─ 3. User: "yolo"
│     Executor implements task 1 → tests pass → done
│     Executor implements task 2 → tests pass → done
│     Executor implements task 3 → test fails → reports back
│     Orchestrator: "Task 3 failed. Stopping for review."
│     → Back to PLAN mode
│
├─ 4. User reviews task 3 failure
│     Adjusts spec → approves
│     User: "yolo"
│     Executor re-implements → tests pass → done
│
└─ Session End
    Orchestrator writes session summary
```

---

## Communication Protocol

### Hunter → Findings DB
- JSON files in `yarrow/workspace/findings/`
- Filename: `{agent}_{scan_id}_{timestamp}.json`
- One file per scan (may contain multiple findings)

### Findings DB → Assistant → Orchestrator
- Assistant reads raw findings, deduplicates, clusters
- Writes processed findings for orchestrator review

### Orchestrator → Task DB
- Markdown files in `yarrow/workspace/tasks/`
- Filename: `TASK_{id}_{short_name}.md`
- Contains: source findings, spec, acceptance criteria, test plan

### Orchestrator → Executor
- Points executor at specific task file
- Executor reads spec, implements, writes result

### Executor → Results DB
- JSON + diff in `yarrow/workspace/executions/`
- Contains: task_id, diff, test output, status

### State Tracking
- `yarrow/workspace/state.json` — current mode, active tasks, agent status
- Updated by orchestrator on every state transition
- User can inspect at any time

---

## Error Handling

| Situation | Response |
|-----------|----------|
| Hunter produces malformed finding | Assistant flags, orchestrator skips |
| Executor can't implement spec | Reports back, orchestrator refines |
| Executor's fix breaks tests | Auto-stop YOLO, back to PLAN |
| Conflicting findings between harvesters | Orchestrator investigates, user decides |
| Orchestrator context fills up | Save state to DB, restart with summary |
| Model API goes down | Continue with remaining agents, note gap |
