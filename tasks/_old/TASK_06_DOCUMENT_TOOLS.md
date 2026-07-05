# TASK 06: Document Tools Directory

## Objective

Create markdown documentation files for each tool in `tools/` directory to explain purpose, usage, and examples.

---

## Current State

**Tools Directory**: 36 Python scripts in `tools/`
**Documentation**: None - no README files explaining what each tool does

**Problem**:
- Hard to know which tool to use for a specific task
- No usage examples or parameter documentation
- New users (or AI agents) can't discover available tools

---

## Scope

### Tools to Document (36 total)

**Audit & Validation**:
- `audit_codebase.py`
- `audit_documentation.py`
- `generate_current_state.py`
- `validate_database.py`
- `validate_fx_rates.py`

**Data Management**:
- `build_fx_database.py`
- `download_data.py`
- `update_dataset.py`
- `map_training_data.py`
- `add_time_features.py`

**Training & Scheduling**:
- `run_scheduled_training.py`
- `generate_mcts_data.py`
- `profile_training.py`
- `log_parser.py`
- `replay_trades.py`

**Testing & Debugging** (21 tools):
- `check_*.py` (assets, wsname)
- `debug_*.py` (currency, db_stats, detailed, env, kraken_pairs, rewards, spot_mask, step)
- `diagnose_*.py` (gpu_mcts, jax_training, portfolio)
- `test_*.py` (env_init, episode_length, kraken_connection, mctx_jax, model_init, system, trm_jax)
- `verify_timestamps.py`
- `profile_memory.py`

---

## Implementation

### Step 1: Create Tools README

**File**: `tools/README.md`

```markdown
# Tools Directory

Utility scripts for development, testing, and maintenance.

## Categories

### Audit & Validation
- [audit_codebase.py](./audit_codebase.md) - Scan codebase for implemented features
- [audit_documentation.py](./audit_documentation.md) - Find documentation drift
- [generate_current_state.py](./generate_current_state.md) - Generate CURRENT_STATE.md
- [validate_database.py](./validate_database.md) - Validate assets against Kraken API
- [validate_fx_rates.py](./validate_fx_rates.md) - Validate FX rate data

### Data Management
- [build_fx_database.py](./build_fx_database.md) - Build FX rates table
- [download_data.py](./download_data.md) - Download historical OHLCV data
- [update_dataset.py](./update_dataset.md) - Update existing dataset
- [map_training_data.py](./map_training_data.md) - Map training data coverage
- [add_time_features.py](./add_time_features.md) - Add time-based features

### Training & Scheduling
- [run_scheduled_training.py](./run_scheduled_training.md) - Rainbow table training scheduler
- [generate_mcts_data.py](./generate_mcts_data.md) - Generate MCTS training data
- [profile_training.py](./profile_training.md) - Profile training performance
- [log_parser.py](./log_parser.md) - Parse training logs
- [replay_trades.py](./replay_trades.md) - Replay historical trades

### Testing & Debugging
See [TESTING.md](./TESTING.md) for complete list of test/debug tools.

## Quick Start

### Audit System State
\```bash
python tools/audit_codebase.py
python tools/generate_current_state.py
\```

### Validate Database
\```bash
python tools/validate_database.py
\```

### Build FX Database
\```bash
python tools/build_fx_database.py
\```

### Run Scheduled Training
\```bash
python tools/run_scheduled_training.py --config config/schedule.json
\```
```

---

### Step 2: Create Individual Tool Documentation

**Template**: `tools/TOOL_NAME.md`

```markdown
# TOOL_NAME.py

## Purpose
Brief description of what this tool does.

## Usage

\```bash
python tools/TOOL_NAME.py [arguments]
\```

## Arguments

- `--arg1` - Description
- `--arg2` - Description (optional, default: value)

## Examples

### Example 1: Basic Usage
\```bash
python tools/TOOL_NAME.py --arg1 value
\```

### Example 2: Advanced Usage
\```bash
python tools/TOOL_NAME.py --arg1 value --arg2 value
\```

## Output

Description of what the tool produces (files created, console output, etc.)

## Dependencies

- `dependency1` - Why it's needed
- `dependency2` - Why it's needed

## Notes

- Important caveats
- Known limitations
- When to use this tool

## See Also

- Related tools
- Related documentation
```

---

### Step 3: High-Priority Tools to Document First

**Tier 1** (Most commonly used):
1. `run_scheduled_training.py` - Rainbow table scheduler (actively used)
2. `build_fx_database.py` - FX database builder
3. `validate_database.py` - Database validation
4. `audit_codebase.py` - Audit system
5. `generate_current_state.py` - State generator

**Tier 2** (Development tools):
6. `download_data.py` - Data fetching
7. `validate_fx_rates.py` - FX validation
8. `replay_trades.py` - Trade replay
9. `log_parser.py` - Log analysis
10. `profile_training.py` - Performance profiling

**Tier 3** (Testing/Debugging):
- All test_*.py, debug_*.py, diagnose_*.py tools
- Create single `TESTING.md` with table listing all test tools

---

## Acceptance Criteria

✅ **PASS if ALL met**:
1. `tools/README.md` created with categorized tool list
2. Documentation exists for Tier 1 tools (5 files)
3. Documentation exists for Tier 2 tools (5 files)
4. `tools/TESTING.md` created listing all test/debug tools (21 tools)
5. Each doc has: Purpose, Usage, Arguments, Examples, Output
6. All docs follow consistent template
7. Links work between README and individual docs

---

## Example: audit_codebase.py Documentation

**File**: `tools/audit_codebase.md`

```markdown
# audit_codebase.py

## Purpose
Scans the codebase to detect which features are actually implemented by checking for files, methods, and code patterns.

## Usage

\```bash
python tools/audit_codebase.py
\```

## Arguments

None. This tool requires no arguments.

## Output

Creates `AUDIT_IMPLEMENTATION_STATUS.md` in the root directory with a feature matrix:

\```
# Codebase Audit Report

**Generated**: 2025-12-17 00:15:53

## Implementation Status

### Data Infrastructure
- **Fx Rates Table**: ✅ IMPLEMENTED
- **Ram Caching**: ✅ IMPLEMENTED
- **Fx Matrix**: ✅ IMPLEMENTED

### Training Systems
- **Rainbow Table**: ✅ IMPLEMENTED
...
\```

## How It Works

1. Scans specific files for existence (`os.path.exists`)
2. Checks for methods (`def method_name(` in file)
3. Checks for attributes (`self.attr_name` in file)
4. Checks for text patterns (e.g., `--load-checkpoint`)
5. Generates markdown report

## Dependencies

- Python standard library only (os, re, pathlib)

## When to Use

- After implementing a new feature
- During documentation cleanup
- When creating CURRENT_STATE.md
- To verify what's actually implemented vs documented

## See Also

- [audit_documentation.py](./audit_documentation.md) - Find documentation drift
- [generate_current_state.py](./generate_current_state.md) - Generate authoritative state
- [ORGANIZATION_RULES.md](../ORGANIZATION_RULES.md) - Audit workflow
```

---

## Directory Structure After Completion

```
tools/
├── README.md                          # Main tools index
├── TESTING.md                         # All test/debug tools
│
├── audit_codebase.md                  # Tier 1
├── generate_current_state.md
├── run_scheduled_training.md
├── build_fx_database.md
├── validate_database.md
│
├── download_data.md                   # Tier 2
├── validate_fx_rates.md
├── replay_trades.md
├── log_parser.md
├── profile_training.md
│
├── *.py                               # Python tools (36 files)
└── [other tool docs as needed]
```

---

## Benefits

1. **Discoverability**: Easy to find the right tool
2. **Onboarding**: New users/agents understand tool ecosystem
3. **Self-Documenting**: Each tool has usage examples
4. **Maintenance**: Clear documentation for future updates
5. **Consistency**: Standard template for all tools

---

## Implementation Notes

### For Gemini (Executor)

**Workflow**:
1. Read this task file
2. Create `tools/README.md` with categorized list
3. For each Tier 1 tool:
   - Read the Python file to understand arguments and functionality
   - Create `tools/TOOL_NAME.md` using template
   - Add usage examples
   - Document output format
4. Repeat for Tier 2
5. Create `tools/TESTING.md` with table of all test tools
6. Update `lore/DIARY.md` with completion entry

**Example workflow for one tool**:
```bash
# 1. Read the tool
cat tools/audit_codebase.py

# 2. Create documentation
# [Create tools/audit_codebase.md with template]

# 3. Verify links work
cat tools/README.md | grep audit_codebase
```

**Report format**:
```
TASK_06 COMPLETED ✅

Created:
- tools/README.md (main index)
- tools/TESTING.md (21 test tools)
- 10 tool documentation files (Tier 1 + Tier 2)

Documentation includes:
- Purpose and usage for each tool
- Command-line arguments
- Examples
- Output format
- Dependencies and notes

All links verified, template consistent across all docs.
```

---

## Priority

**Priority**: LOW (documentation improvement, not blocking)

**Estimated Effort**: 3-4 hours
- README creation: 30 min
- Tier 1 tools (5): 1.5 hours (15-20 min each)
- Tier 2 tools (5): 1.5 hours (15-20 min each)
- TESTING.md: 30 min
- Review and link verification: 30 min

---

## Next Task After Completion

Continue with curriculum implementation tasks (TASK_02-TASK_05) or cross-pair trading (TASK_07).

This is a parallel task that can be done anytime without blocking other work.
