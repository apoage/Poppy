# Gemini - Quick Start

## 📋 Read These Files First

1. **`../state.json`** - Current project state (structured data)
2. **`../messages.jsonl`** - Message log from Claude
3. **`../claude/analysis.md`** - Detailed analysis (if you need context)

## 🤝 Communication Protocol

### Read Messages
```bash
tail ../messages.jsonl
# Or parse JSON:
cat ../messages.jsonl | jq 'select(.to == "gemini")'
```

### Send Messages
```bash
# Append to messages.jsonl
echo '{"timestamp": "2025-12-03T23:00:00Z", "from": "gemini", "to": "claude", "type": "complete", "content": "Config changes committed", "data": {"commit": "abc123"}}' >> ../messages.jsonl
```

### Update State
```bash
# Edit state.json, update last_updated and updated_by
jq '.last_updated = "2025-12-03T23:00:00Z" | .updated_by = "gemini"' ../state.json > tmp && mv tmp ../state.json
```

## 🎯 Your Pending Actions

From `state.json`:
- **REVIEW**: Your config changes (batch_size, replay_buffer)
- **ACTION**: Commit or modify based on Claude's analysis
- **FILL**: Your architectural plan in `plan.md`

## 📝 Commit Format

```
[GEMINI] Category: Brief description

Details:
- Changes made
- Reasoning

Status: {COMPLETE|IN_PROGRESS|NEEDS_REVIEW}
```

## 🚨 Critical Info

- ⚠️ **Don't remove** monkey-patch in `main.py:12-51`
- ⚠️ **Use spawn** for multiprocessing (line 78-82)
- 💾 Your changes save **10GB RAM** (good!)
- 📊 May increase **gradient variance** (monitor training)

## 📊 Current Status Summary

```json
{
  "database": "20GB, 64 assets, 30 days",
  "model": "TRM 7M params implemented",
  "training": "ready to start",
  "your_changes": "batch_size: 2048→512, replay: 50k→10k",
  "claude_says": "commit_and_monitor"
}
```

---

**Next**: Review state.json, commit your changes, send completion message to messages.jsonl
