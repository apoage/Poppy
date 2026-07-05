# TASK_028: Tech Debt Cleanup

## Overview

Collection of small cleanup items identified in codebase audit.

## Items

### 1. Dead Code: `mcts_worker_pool.py`

**File**: `mcts_trader/jax_agent/mcts_worker_pool.py`

**Issue**: CPU-based MCTS worker pool that hides GPU from workers. Not used by `train_jax_fast.py` which uses GPU MCTS via `mctx`.

**Action**: Delete file or move to `_old/`

**Risk**: Low - grep for imports first

```bash
grep -r "mcts_worker_pool" --include="*.py" .
```

---

### 2. Hardcoded Default: `n_actions=192`

**Files**:
- `mcts_trader/jax_agent/trm_flax.py` lines 62, 218, 244

**Issue**: Default is 192 (64×3) but actual training uses 320 (64×5). Booby trap for direct instantiation.

**Action**: Change default to 320 or remove default entirely (require explicit arg)

```python
# Before
n_actions: int = 192

# After (option A - correct default)
n_actions: int = 320  # 64 assets × 5 actions (S/H/B/LS/LB)

# After (option B - no default, force explicit)
n_actions: int  # Must be provided
```

---

### 3. Magic Number: `hidden_dim = state_dim * 16`

**File**: `mcts_trader/jax_agent/trm_flax.py` lines 71, 108

**Issue**: Critical architectural parameter buried in code, not exposed in config.

**Action**: Extract to class attribute or config

```python
# Before (inside __call__)
hidden_dim = self.state_dim * 16

# After (class attribute)
class TRMReasoning(nn.Module):
    state_dim: int = 64
    hidden_multiplier: int = 16  # hidden_dim = state_dim * hidden_multiplier

    def __call__(self, ...):
        hidden_dim = self.state_dim * self.hidden_multiplier
```

---

### 4. Legacy Directory: `_old/`

**Path**: `/home/lukas/Trader/_old/`

**Issue**: Contains deprecated agents, envs, docs. Adds noise to searches.

**Action**:
- Review contents for any valuable items
- Archive to separate branch or delete
- Or rename to `.archive/` (hidden from most tools)

---

### 5. Price History RAM Loading

**File**: `mcts_trader/env/vectorbt_env.py` - `_load_database_prices()`

**Issue**: Loads entire price history into RAM. OOM risk as data scales.

**Action**:
- Add memory mapping option (`np.memmap`)
- Or lazy loading with sliding window

**Priority**: Low until dataset grows beyond 32GB

---

### 6. Dojo Subprocess Overhead

**File**: `dojo_dynamic_curriculum.py`

**Issue**: Executes `train_jax_fast.py` as subprocess for every phase. This forces:
- Full reload of price data each phase
- Cold JIT compilation (mitigated by persistent cache)
- Loss of in-memory state between curriculum stages

**Current**:
```python
cmd = ['python', 'train_jax_fast.py', '--synthetic-level', str(level), ...]
result = subprocess.run(cmd, capture_output=False)
```

**Action**: Refactor Dojo to import trainer and execute in-process:
```python
from train_jax_fast import create_trainer, run_training_loop

# Keep trainer/buffer in memory across phases
trainer = create_trainer(model_size='small', ...)
for phase in phases:
    run_training_loop(trainer, phase_config)
```

**Benefits**:
- No data reload between phases
- Warm JIT for all phases after first
- Buffer can optionally persist across phases

**Priority**: Low - subprocess approach works, just slower startup per phase

---

## Execution Order

1. **n_actions default** (5 min) - Quick fix, high value
2. **hidden_multiplier** (10 min) - Quick fix, better readability
3. **mcts_worker_pool.py** (5 min) - Delete after grep check
4. **_old/ directory** (15 min) - Review and archive
5. **RAM loading** (1-2 hours) - Defer until needed
6. **Dojo refactoring** (2-4 hours) - Defer, subprocess works fine

## Verification

```bash
# After cleanup, verify training still works
python train_jax_fast.py --synthetic-level 0 --epochs 5 --steps 50 --name cleanup_test
```

## Priority

Low - Cleanup items, no functional impact. Do when convenient.
