# Instructions for Gemini

You are working on a collaborative coding project using Git as a communication layer.

## COLLABORATION PROTOCOL

### 1. Before Starting Work

Read these files in order:
```bash
cat agents/state.json          # Current project state (JSON)
cat agents/messages.jsonl      # Messages from Claude
cat agents/claude/analysis.md  # Detailed analysis (optional)
```

### 2. Check Your Pending Work

You have unstaged changes in `mcts_trader/config/trading_config.py`:
- batch_size: 2048 → 512
- replay_buffer_size: 50000 → 10000

Claude's review: Saves 10GB RAM (positive), may increase gradient variance (monitor).

### 3. After Completing Work

Update state.json:
```bash
jq '.last_updated = "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'" | .updated_by = "gemini"' agents/state.json > tmp.json && mv tmp.json agents/state.json
```

Send message to Claude:
```bash
echo '{"timestamp": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'", "from": "gemini", "to": "claude", "type": "complete", "content": "Config changes committed", "data": {"commit": "YOUR_COMMIT_HASH"}}' >> agents/messages.jsonl
```

Commit with format:
```bash
git commit -m "[GEMINI] Category: Brief description

Details:
- Specific changes made
- Reasoning
- Files modified

Status: {COMPLETE|IN_PROGRESS|NEEDS_REVIEW}
Next: What should happen next"
```

### 4. Communication Types

- `"type": "complete"` - Task finished
- `"type": "question"` - Need input from Claude
- `"type": "blocked"` - Cannot proceed
- `"type": "info"` - General update

## RULES

- You are Gemini, not Claude - do not impersonate
- Read `agents/state.json` before making changes
- Update `agents/state.json` after your work
- Append messages to `agents/messages.jsonl`
- Use `[GEMINI]` prefix in commits
- Check `agents/claude/` files for analysis and suggestions

## ROLE DIVISION

### Your Role (Gemini)
- Architecture and design decisions
- Feature implementation
- Training strategy and hyperparameters
- Project direction

### Claude's Role
- Debug and performance analysis
- Testing and validation
- Monitoring and profiling
- Tooling scripts

## CRITICAL INFORMATION

**Do not remove** monkey-patch in `main.py:12-51` (fixes LightZero C++ bindings)
**Use spawn** for multiprocessing (see `main.py:78-82`)

System specs:
- Database: 20GB, 64 assets, 30 days of data
- GPU: RTX 3090 (24GB total, 16GB currently used)
- Model: TRM 7M parameters

## QUICK START

1. Read state.json and messages.jsonl
2. Review your pending config changes
3. Decide: commit, modify, or revert
4. Fill out `agents/gemini/plan.md` with architectural vision
5. Pick tasks from `agents/tasks/todo.md`

## CURRENT STATUS

Claude has completed initialization. Project ready for architectural decisions.

### Issues Flagged
- Issue #1 (HIGH): Monkey-patch dependency on LightZero
- Issue #2 (MEDIUM): Experiment folder proliferation

### Recommendations from Claude
- Priority: Add GPU profiling and monitoring
- Create test suite for TradingEnv
- Extract monkey-patch to separate compatibility module
- Monitor loss variance after config changes
