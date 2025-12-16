# Task Execution Guide for Gemini

## Overview

These tasks implement a complete synthetic curriculum training system for the portfolio manager. Execute them **in order**.

## Task Sequence

### ✅ TASK_00: Codebase Audit & Cleanup - COMPLETED (Dec 17)
**Priority**: HIGHEST (DO FIRST)
**Status**: ✅ DONE
**Completed by**: Claude Code
**Why**: Documentation is out of sync - need single source of truth

**Outcome**:
- ✅ CURRENT_STATE.md created (single source of truth)
- ✅ Audit scripts created (audit_codebase.py, audit_documentation.py, generate_current_state.py)
- ✅ Stale documentation marked
- ✅ **DISCOVERY**: MCTS Worker Pool already implemented (173 lines, Dec 16)

---

### ⚠️ TASK_01: Verify MCTS Parallelization - NEEDS VERIFICATION
**Priority**: CRITICAL
**Status**: ⚠️ Code exists, needs verification
**Updated**: Dec 17 (discovered existing implementation)
**Why**: Worker pool code exists but unknown if active during training

**Outcome**: Verify 3-4× speedup, GPU utilization 85%+, all CPU cores utilized

---

### ✅ TASK_02: Synthetic Market Generators
**Priority**: HIGH
**Estimated effort**: 6 hours
**Why**: Enable training on controlled environments (Levels 0-3)

**Outcome**: Can train on pure sine waves, validate architecture works

---

### ✅ TASK_03: Scale Model to 25M
**Priority**: MEDIUM
**Estimated effort**: 3 hours
**Why**: 7M params might be too small for complex portfolio tactics

**Outcome**: Configurable model sizes (7M/15M/25M)

---

### ✅ TASK_04: Portfolio Curriculum
**Priority**: HIGH
**Estimated effort**: 8 hours
**Why**: Levels 4-6 teach portfolio tactics, not just trading

**Outcome**: Rotation tactics (Level 4), chaos survival (Level 6)

---

### ✅ TASK_05: Prediction Heads
**Priority**: MEDIUM
**Estimated effort**: 6 hours
**Why**: Force latent space to encode "what will happen" not just "what to do"

**Outcome**: Multi-task learning, better generalization, interpretable predictions

---

### 🔜 TASK_06: Cross-Pair Trading (NOT YET CREATED)
**Priority**: MEDIUM
**Estimated effort**: 6 hours
**Why**: Enable BTC→ETH trades (avoid 2× fees)

**Outcome**: Direct pair trading, 50% fee savings

---

### 🔜 TASK_07: Adversarial Training (NOT YET CREATED)
**Priority**: HIGH
**Estimated effort**: 12 hours
**Why**: AlphaGo-style self-play for breakthrough tactics

**Outcome**: Robust agent that survives adversarial attacks

---

## Execution Instructions

### For Each Task:

1. **Read the task file completely**
2. **Check prerequisites** (previous tasks must pass)
3. **Implement step-by-step** (don't skip steps)
4. **Run ALL tests** (including acceptance criteria)
5. **Document failures** (what broke, why, how you fixed it)
6. **Commit with message from task file**

### Example Workflow:

```bash
# Read task
cat tasks/TASK_01_FIX_MCTS_PARALLELIZATION.md

# Implement
# (follow steps in task file)

# Test
python tests/benchmark_mcts_parallelization.py

# Verify acceptance criteria
# - htop shows 16 cores utilized ✅
# - Speedup > 3x ✅
# - Training converges ✅

# Commit
git add -A
git commit -m "Fix MCTS parallelization - 3x speedup"

# Move to next task
cat tasks/TASK_02_SYNTHETIC_MARKET_GENERATORS.md
```

## Critical Rules

### ⚠️ STOP if:
1. **Tests fail** - Don't proceed to next task
2. **Acceptance criteria not met** - Debug until they pass
3. **Architecture breaks** - If Level 0 stops converging, STOP

### ✅ Proceed if:
1. All tests pass
2. Acceptance criteria met
3. No regressions (previous levels still work)

## Current System State

**Hardware**:
- RTX 3090 (24GB VRAM)
- 16-core CPU
- 46GB RAM

**Model**:
- 7M params (TRM)
- Batch size: 512
- MCTS sims: 32

**Bottlenecks**:
1. ❌ MCTS parallelization (1 core only) ← **FIX THIS FIRST**
2. ❌ Model size (might be too small) ← **TASK_03**
3. ⚠️ MCTS depth (32 sims might not be enough for breakthroughs)

## Progress Tracking

After completing each task, update this file:

```markdown
## Completion Status

- [ ] TASK_01: MCTS Parallelization
- [ ] TASK_02: Synthetic Generators
- [ ] TASK_03: Model Scaling
- [ ] TASK_04: Portfolio Curriculum
- [ ] TASK_05: Prediction Heads
- [ ] TASK_06: Cross-Pair Trading
- [ ] TASK_07: Adversarial Training
```

## Expected Outcomes

**After TASK_01**: Training 3× faster
**After TASK_02**: Can validate architecture on simple data
**After TASK_03**: Have 25M model for complex tasks
**After TASK_04**: Agent learns portfolio tactics
**After TASK_05**: Multi-task learning, better latent space
**After TASK_06**: 50% lower trading fees
**After TASK_07**: Robust agent with inhuman tactics

## Communication Protocol

**After each task**:
1. Report completion status
2. Share test results (paste logs)
3. Document any issues encountered
4. Ask clarifying questions if stuck

**Example Report**:
```
TASK_01 COMPLETED ✅

Results:
- Speedup: 3.7× (expected 3.0×+) ✅
- CPU util: All 16 cores at 75%+ ✅
- GPU util: 87% (was 60%) ✅
- Training still converges ✅

Issues encountered:
- Had to add CUDA_VISIBLE_DEVICES='' to worker processes
- Multiprocessing 'spawn' required (not 'fork')

Ready for TASK_02.
```

## Notes

- No timelines/deadlines - work at your own pace
- Break fast, document, fix
- Test rigorously before proceeding
- When in doubt, ask questions

Let's build this. 🚀
