# Project Organization Rules

**Last Updated**: December 17, 2025
**For**: Claude Code (orchestrator) & Gemini (executor)

---

## Core Principle

**MINIMIZE ROOT DIRECTORY FILES**

Only essential, actively-used files belong in root. Everything else goes into organized subdirectories.

---

## Directory Structure

### Root Directory (/)

**ONLY these files allowed in root**:

1. **Essential Documentation**:
   - `CURRENT_STATE.md` - **AUTHORITATIVE** system state (MUST stay in root)
   - `CLAUDE.md` - Instructions for Claude Code
   - `README.md` - Project overview
   - `ORGANIZATION_RULES.md` - This file

2. **Executable Scripts**:
   - `*.py` - Python scripts (train_jax_fast.py, train_manager.py, etc.)
   - `*.sh` - Shell scripts (START_TRAINING.sh, run_training.sh, etc.)

3. **Configuration**:
   - `*.json` - Config files (training_schedule.json, etc.)
   - `requirements.txt`
   - `.env` (if exists)

4. **Database**:
   - `trading_data.db`

5. **Directories**:
   - Standard project directories (see below)

**NO OTHER FILES** - If you create a report, summary, or documentation, it goes into a subdirectory immediately.

---

### /tasks/

**Purpose**: Active task files for Gemini to execute

**Contents**:
- `TASK_XX_DESCRIPTION.md` - Implementation tasks (numbered, descriptive names)
- `TASK_README.md` - Master task list and execution guide

**Rules**:
- ✅ **DO**: Create new tasks here with clear numbering (TASK_06, TASK_07, etc.)
- ✅ **DO**: Update TASK_README.md when adding tasks
- ❌ **DON'T**: Leave completed tasks here
- ❌ **DON'T**: Create prompts/instructions here (use lore/ for documentation)

**When task is complete**:
```bash
mv tasks/TASK_XX_*.md _old/tasks/
# Update TASK_README.md to mark as completed
```

---

### /lore/

**Purpose**: Research, design documents, development diary, and training documentation

**Contents**:
- `DIARY.md` - **PRIMARY** - Consolidated development diary (chronological)
- `recent_improvements.md` - Log of recent changes
- `future_enhancements.md` - Roadmap and planned features
- Design documents (e.g., `ADVANCED_ORDERS_DESIGN.md`)
- Training documentation (e.g., `TRAINING_QUICK_START.md`)

**Rules**:
- ✅ **DO**: Add design documents here (architecture, specifications)
- ✅ **DO**: Update DIARY.md with significant developments
- ✅ **DO**: Add training guides and documentation here
- ❌ **DON'T**: Create multiple diary files (use DIARY.md)
- ❌ **DON'T**: Put task instructions here (use tasks/)
- ❌ **DON'T**: Put reports/summaries here if they're from completed work (use _old/)

**Subdirectories** (if needed):
- `lore/designs/` - Architecture and design specs
- `lore/training/` - Training guides and documentation

---

### /_old/

**Purpose**: Archive for completed work, outdated files, and historical records

**Structure**:
```
_old/
├── README.md                    # Archive index (REQUIRED)
├── tasks/                       # Completed tasks
├── gemini_reports/              # Gemini's work reports
├── gemini_prompts/              # Old task prompts for Gemini
├── audit/                       # Audit and cleanup reports
├── lore/
│   ├── diary_archive/          # Old individual diary files
│   └── research/               # Outdated research docs
└── notes_lukas.txt             # User's notes
```

**Rules**:
- ✅ **DO**: Move completed tasks here immediately
- ✅ **DO**: Archive reports and summaries after work is done
- ✅ **DO**: Update `_old/README.md` when adding files
- ❌ **DON'T**: Reference files in _old/ as current documentation
- ❌ **DON'T**: Create new work here (only archive existing work)

---

## File Naming Conventions

### Tasks
```
TASK_XX_DESCRIPTIVE_NAME.md
```
- XX = two-digit number (01, 02, 03...)
- DESCRIPTIVE_NAME = what the task does (FIX_MCTS_PARALLELIZATION)
- Always use .md extension

### Reports (Gemini)
```
FEATURE_NAME_REPORT.md
FEATURE_NAME_SUMMARY.md
```
- Clear feature name in all caps
- _REPORT or _SUMMARY suffix
- Goes to _old/gemini_reports/ when done

### Design Documents (Claude)
```
FEATURE_NAME_DESIGN.md
CONCEPT_NAME.md
```
- Clear descriptive name
- Lives in lore/

---

## Workflow Rules

### For Claude Code (Orchestrator)

**When creating a task**:
1. Create `tasks/TASK_XX_DESCRIPTION.md` with full specification
2. Update `tasks/TASK_README.md` with task entry
3. Do NOT create prompt in root - specification goes in task file

**When task is completed**:
1. Move task file: `tasks/TASK_XX_*.md` → `_old/tasks/`
2. Update `tasks/TASK_README.md` to mark complete
3. Update `CURRENT_STATE.md` (run audit if significant changes)
4. Add entry to `lore/DIARY.md`

**When creating design documents**:
1. Create in `lore/DESIGN_NAME.md`
2. Do NOT create in root

**When doing audit/cleanup**:
1. Generate reports in root temporarily
2. Move to `_old/audit/` when done
3. Update `CURRENT_STATE.md` (stays in root)

---

### For Gemini (Executor)

**When starting a task**:
1. Read `tasks/TASK_XX_*.md`
2. Read `CURRENT_STATE.md` for current system state
3. Do NOT create new files in root

**When completing a task**:
1. Create report in `_old/gemini_reports/TASK_XX_REPORT.md` directly
2. Update `CURRENT_STATE.md` if you implemented features (run audit)
3. Add entry to `lore/DIARY.md` summarizing what was done
4. Do NOT leave summary files in root

**During implementation**:
1. Do NOT create `FOR_GEMINI_*.md` prompts (use task files)
2. Do NOT create `READY_TO_*.md` status files (report in task file or DIARY.md)
3. If you need to document something, add to existing lore/ files

**When encountering issues**:
1. Document in task file or DIARY.md
2. Do NOT create separate issue files in root

---

## Audit Workflow

**After implementing ANY feature**:
```bash
python tools/audit_codebase.py
python tools/generate_current_state.py
git add CURRENT_STATE.md
git commit -m "Update CURRENT_STATE.md - [feature] implemented"
```

**Result**: CURRENT_STATE.md updated (stays in root), audit reports can go to _old/audit/

---

## Clean Root Directory Example

```
Trader/
├── CLAUDE.md                      # Instructions for Claude
├── CURRENT_STATE.md              # Authoritative state
├── ORGANIZATION_RULES.md         # This file
├── README.md                     # Project overview
├── train_jax_fast.py            # Python scripts
├── train_manager.py
├── paper_trade_jax.py
├── real_trade_jax.py
├── START_TRAINING.sh            # Shell scripts
├── run_training.sh
├── requirements.txt             # Dependencies
├── training_schedule.json       # Config
├── trading_data.db              # Database
├── _old/                        # Archive
├── config/                      # Configuration files
├── datasets/                    # Data
├── jax_ckpt_fast/              # Checkpoints
├── logs/                        # Training logs
├── lore/                        # Documentation & diary
├── mcts_trader/                 # Source code
├── tasks/                       # Active tasks
├── tests/                       # Test files
├── tools/                       # Utility scripts
└── trader_env/                  # Virtual environment
```

**Goal**: Maximum ~10 files in root (excluding directories)

---

## Red Flags (Don't Do This)

❌ **Creating multiple diary files**: Use `lore/DIARY.md` only
❌ **Summary files in root**: `FEATURE_SUMMARY.md` → `_old/gemini_reports/`
❌ **Prompt files in root**: `FOR_GEMINI_*.md` → Use task files instead
❌ **Status files in root**: `READY_TO_TRAIN.md` → Document in DIARY.md
❌ **Multiple audit reports in root**: Generate → Archive to `_old/audit/`
❌ **Leaving completed tasks in tasks/**: Move to `_old/tasks/` immediately

---

## Green Flags (Do This)

✅ **Update DIARY.md**: Add entry for significant work
✅ **Update CURRENT_STATE.md**: Run audit after features implemented
✅ **Archive when done**: Completed task → `_old/tasks/` immediately
✅ **Organize reports**: Gemini report → `_old/gemini_reports/` directly
✅ **Use task files**: All specifications in `tasks/TASK_XX_*.md`
✅ **Clean root**: Only essential files in root directory

---

## Enforcement

**Before committing**:
```bash
# Check root directory
ls -1 *.md | wc -l
# Should be ≤ 4 (CLAUDE.md, CURRENT_STATE.md, ORGANIZATION_RULES.md, README.md)

# If more files exist, organize them:
# - Design docs → lore/
# - Reports → _old/gemini_reports/
# - Completed tasks → _old/tasks/
# - Audit reports → _old/audit/
```

---

## Benefits of This System

1. **Clean root**: Easy to navigate, no clutter
2. **Clear structure**: Know exactly where files go
3. **Easy to find things**: Organized by purpose
4. **Historical record**: _old/ preserves all work
5. **No confusion**: One diary, one state file, clear rules
6. **Scalable**: Works for 100s of tasks and reports

---

## Quick Reference

| I need to... | Where does it go? |
|--------------|-------------------|
| Create a new task | `tasks/TASK_XX_DESCRIPTION.md` |
| Task completed | `_old/tasks/TASK_XX_*.md` |
| Design document | `lore/DESIGN_NAME.md` |
| Implementation report | `_old/gemini_reports/REPORT_NAME.md` |
| Log dev work | `lore/DIARY.md` (add entry) |
| Update system state | `CURRENT_STATE.md` (root) |
| Archive old work | `_old/category/` |

---

**Remember**: When in doubt, DON'T create a new file in root. Find the right subdirectory or ask.

---

**Maintained by**: Claude Code (orchestrator)
**Used by**: Claude Code, Gemini, and any future AI agents
**Last Cleanup**: December 17, 2025
