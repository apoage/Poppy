# Yarrow Communication Protocol

## Agent Topology

```
                      ┌─────────┐
                      │   YOU   │
                      └────┬────┘
                           │
                  ┌────────┴────────┐
                  │  ORCHESTRATOR   │
                  │  (Claude Opus)  │
                  │                 │
                  │  Brain. Decides.│
                  └──┬──────────┬───┘
                     │          │
            ┌────────┘          └────────┐
            ▼                            ▼
   ┌─────────────────┐         ┌─────────────┐
   │   ASSISTANT      │         │  EXECUTOR   │
   │  (Haiku/Sonnet)  │         │ (Opus/Gem)  │
   │                  │         │             │
   │  - deduplicates  │         │  - implements│
   │  - clusters      │         │  - tests    │
   │  - summarizes    │         │  - reports  │
   └───────┬──────────┘         └─────────────┘
           │ reads
           ▼
    ┌──────────────┐
    │  FINDINGS DB  │
    └──────────────┘
       ▲  ▲  ▲
       │  │  │
    ┌──┘  │  └──┐
    ▼     ▼     ▼
 Harvester Harvester Harvester
 (Gem)  (Claude)(GLM)
```

## Why an Assistant?

The orchestrator's context window is precious. Deduplication, clustering,
and summarization are mechanical tasks that:
- Process large volumes of raw findings
- Don't require strategic thinking
- Would fill orchestrator context with noise

The assistant (cheap model — Haiku or Sonnet) pre-processes findings
so the orchestrator receives clean, deduplicated clusters.

### Assistant Responsibilities
1. **Deduplicate** — match findings by file+line range+type
2. **Cluster** — group related findings across files
3. **Summarize** — one-paragraph summary per cluster
4. **Flag conflicts** — when harvesters disagree on same code
5. **Compute ensemble score** — weighted by agent precision

### Assistant Output Format
```json
{
  "cluster_id": "uuid",
  "summary": "Buffer overflow in replay buffer resize logic",
  "severity": "high",
  "ensemble_score": 0.87,
  "contributing_agents": ["gemini", "claude"],
  "findings": ["finding_id_1", "finding_id_2"],
  "files_affected": ["mcts_trader/jax_agent/replay_buffer.py"],
  "line_range": [142, 168],
  "consensus": "agree",
  "notes": "Both agents identify same root cause, gemini has more detailed fix"
}
```

## Communication Rules

### No Lateral Communication
Harvesters NEVER talk to each other. This is critical:
- Prevents groupthink
- Ensures independent analysis
- Each harvester sees only source code, never other findings

### Unidirectional Flow
```
Harvesters  →  Findings DB  →  Assistant  →  Orchestrator  →  Executor
                                               ↕
                                              YOU
```

No backflow from orchestrator to harvesters (harvesters don't adjust based on
what orchestrator thinks). Fresh eyes every scan.

Exception: Orchestrator can **direct** harvesters to specific files/modules
for next scan. But doesn't share what was found — only where to look.

### Message Bus: Filesystem

All communication happens through files. No direct API calls between agents.

```
yarrow/workspace/
├── findings/                    # Harvesters write here
│   ├── gemini_scan001_*.json
│   ├── claude_scan001_*.json
│   └── glm_scan001_*.json
├── clusters/                    # Assistant writes here
│   ├── cluster_001.json
│   └── cluster_002.json
├── tasks/                       # Orchestrator writes here
│   ├── TASK_001_replay_buffer.md
│   └── TASK_002_loss_nan.md
├── executions/                  # Executor writes here
│   ├── exec_001_result.json
│   └── exec_001.diff
├── directives/                  # Orchestrator → Harvesters (scan targets)
│   └── next_scan.json
└── state.json                   # Workflow state (orchestrator maintains)
```

### Why Files, Not Direct IPC

1. **Inspectable** — you can read any file at any time
2. **Git-trackable** — version control the review process itself
3. **Resumable** — crash recovery is trivial, just re-read files
4. **Model-agnostic** — any model that can read/write JSON works
5. **Debuggable** — when something goes wrong, the evidence trail exists

### Database (SQLite) for Queryable State

Files for communication, SQLite for queries:

```sql
-- "Show me all unresolved critical findings"
SELECT * FROM findings WHERE severity='critical' AND status='open';

-- "What's gemini's precision this week?"
SELECT confirmed::float / total FROM agent_stats WHERE agent='gemini';

-- "Which tasks are blocked?"
SELECT * FROM tasks WHERE status='blocked';
```

The DB mirrors the filesystem. Either can be rebuilt from the other.

## Scaling Considerations

### Adding a New Harvester
1. Give it read access to source code
2. Give it write access to `findings/`
3. Teach it the finding JSON schema
4. Register in `state.json` agent list
5. Assistant automatically picks up new findings

### Adding a New Executor
1. Give it read access to `tasks/`
2. Give it write access to source code + `executions/`
3. Orchestrator dispatches to it by name

### Context Window Management
- Harvesters: fresh context per scan (no accumulation)
- Assistant: processes in batches, resets between
- Orchestrator: accumulates, but reads summaries not raw findings
- Executor: fresh context per task (reads one spec at a time)
