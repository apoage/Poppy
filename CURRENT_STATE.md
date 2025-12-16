# CURRENT STATE - Portfolio Manager System

**Last Updated**: 2025-12-17 00:15:53
**Authoritative**: This is the SINGLE SOURCE OF TRUTH

---

## ✅ What's IMPLEMENTED (Verified)

### Data Infrastructure
- [x] **FX Rates Table**: `fx_rates` table in DB (~860k records)
- [x] **RAM Caching**: O(1) price lookups, no SQL queries during episode
- [x] **FX Matrix**: Pre-loaded FX rates for portfolio valuation

### Training Systems
- [x] **Rainbow Table Continuity**: Checkpoint transfer between training runs
- [x] **Adaptive Curriculum**: Dynamic fee/slippage scaling based on performance
- [x] **Maniac MCTS**: 32 simulations, dirichlet_alpha=0.5

### Model Components
- [x] **TRM Model**: 7M params (d_model=64, H=3, L=4)
- [x] **MuZero Wrapper**: JAX implementation with MCTX
- [x] **Fast Encoder**: CPU-based state encoder (~10ms)

### Environment
- [x] **FastTradingEnv**: JAX-compatible gym environment
- [x] **PortfolioManager**: Multi-asset position tracking
- [x] **Action Space**: 192 actions (64 assets × 3)

### Optimizations
- [x] **MCTS Worker Pool**: Multi-process parallelization (4 workers, CPU-only)

---

## ❌ What's NOT Implemented (Verified)

### Curriculum
- [ ] **Synthetic Market Generators**: Levels 0-6 (designed, not coded)
- [ ] **Pass Criteria Module**: Validation system (designed, not coded)
- [ ] **Curriculum Orchestrator**: Automated level progression (designed)

### Optimizations
- [ ] **Model Scaling**: Fixed at 7M, no small/medium/large options

### Advanced Features
- [ ] **Prediction Heads**: Price/volatility/regime predictors (designed)
- [ ] **Cross-Pair Trading**: Direct BTC→ETH trades (designed)
- [ ] **Adversarial Training**: Protagonist vs antagonist (designed)
- [ ] **Advanced Orders**: Limit/stop orders (designed)

---

## 🎯 Priority Fixes

### CRITICAL (Do First)
1. **Verify MCTS Parallelization** (TASK_01_VERIFY)
   - Code exists but needs testing
   - Measure: CPU utilization across cores, training speedup
   - Expected: 3-4× speedup if working

### HIGH (Do Next)
2. **Synthetic Curriculum** (TASK_02)
   - Validate architecture on Level 0 (sine waves)
   - Before proceeding to real data

3. **Model Scaling** (TASK_03)
   - Enable 25M params for adversarial training

---

## 📊 Current Performance

### Hardware
- GPU: RTX 3090 (24GB VRAM)
- CPU: 16 cores (only 1 utilized - BUG)
- RAM: 46GB

### Training Speed
- Batch size: 512
- MCTS sims: 32
- Epochs/hour: ~225 (CPU-bound)
- GPU utilization: 60% (starved)

### Bottlenecks
1. **CPU**: 1 core bottleneck (⚠️ NEEDS VERIFICATION - worker pool exists but may not be active)
2. **Model size**: 7M might be too small
3. **MCTS depth**: 32 sims might not be enough

---

## 🗂️ File Locations

### Core Training
```
train_jax_fast.py              # Main training script
tools/run_scheduled_training.py  # Rainbow table scheduler
```

### Model
```
mcts_trader/agent/trm.py                     # TRM architecture
mcts_trader/jax_agent/muzero_model.py        # MuZero wrapper
mcts_trader/jax_agent/mctx_search.py         # MCTS search
mcts_trader/jax_agent/mcts_worker_pool.py    # Multi-process MCTS parallelization
```

### Data
```
mcts_trader/data/history.py      # Historical market (RAM cached)
tools/build_fx_database.py       # FX rate builder
```

### Environment
```
mcts_trader/env/trading_env.py   # Legacy TradingEnv
train_jax_fast.py:FastTradingEnv # Current JAX env
```

---

## 🚫 Known Issues

### CRITICAL
- **MCTS Parallelization Status UNKNOWN**: Worker pool code exists (173 lines), imported in train_jax_fast.py, but needs verification if actually active during training

### HIGH
- **No Synthetic Validation**: Can't test architecture before real data
- **Fixed Model Size**: Can't scale to 25M for adversarial

### MEDIUM
- **Documentation Drift**: Notes out of sync with code (THIS AUDIT FIXES IT)

---

## 📝 How to Update This File

**After implementing any feature**:
```bash
python tools/audit_codebase.py
python tools/generate_current_state.py
```

**Manual edits**: Update timestamp and run audit to verify.

---

## 🎓 Lessons Learned

1. **Documentation drift is real** - Multiple AI agents = conflicting notes
2. **Code is ground truth** - Scan codebase, don't trust docs
3. **Single source of truth** - This file (CURRENT_STATE.md)
4. **Audit regularly** - After every major feature, re-run audit
