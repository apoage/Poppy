# Portfolio Manager Development Diary

**Project**: MCTS Trading Agent → Portfolio Manager
**Started**: December 2025
**Last Updated**: December 17, 2025

---

## December 17, 2025 - Documentation Organization & Rules

**Completed by**: Claude Code (orchestrator)

### Root Directory Cleanup

**Problem**: Root directory cluttered with 40+ markdown files - task prompts, reports, design docs, summaries all mixed together.

**Solution**: Organized into clear structure:

**Moved to lore/** (design docs and training guides):
- ADVANCED_ORDERS_DESIGN.md
- PORTFOLIO_MANAGER_TRAINING.md
- PREDICTION_LATENT_SPACE.md
- SYNTHETIC_CURRICULUM_DESIGN.md
- TRAINING_COMMANDS.md
- TRAINING_QUICK_START.md
- TRAINING_STRATEGY_NOTES.md

**Moved to _old/gemini_reports/** (completed work reports):
- GEMINI.md, GEMINI_TASK_CHECKLIST.md
- FX_DB_REPORT.md, FX_FINAL_REPORT.md, FX_INTEGRATION_REPORT.md
- RAINBOW_TABLE_READY.md, RAINBOW_TABLE_SUMMARY.md
- READY_FOR_TRAINING_SUMMARY.md, READY_TO_TRAIN.md
- And 5 other summary/report files

**Moved to _old/gemini_prompts/** (old task prompts):
- FOR_GEMINI.md, FOR_GEMINI_TRAINING_MAPPER.md
- DYNAMIC_ASSET_SELECTION_PROMPT.md
- PRE_TRAINING_FIXES_PROMPT.md
- INFRASTRUCTURE_AUDIT_PROMPT.md
- And 2 other prompt files

**Moved to _old/audit/** (audit reports):
- SESSION_SUMMARY.md
- AUDIT_SUMMARY.md, AUDIT_IMPLEMENTATION_STATUS.md
- AUDIT_DOCUMENTATION_DRIFT.md
- CLEANUP_SUMMARY.md

**Created**: ORGANIZATION_RULES.md
- Clear rules for where files go
- Workflow for Claude Code and Gemini
- Prevents future mess
- Enforces clean root directory (max 10 files)

**Result**:
- Root directory: 40+ files → 4 markdown files (CLAUDE.md, CURRENT_STATE.md, ORGANIZATION_RULES.md, README.md)
- Clean structure: tasks/, lore/, _old/ with clear purposes
- Rules documented for future work
- Both AI agents have clear guidelines

---

### Codebase Audit & Documentation Cleanup

**Completed by**: Claude Code (orchestrator)

### TASK_00: Establish Single Source of Truth

**Problem**: Documentation drift - features implemented but still marked as "future", conflicting notes from multiple AI agents.

**Solution**: Created automated audit system:
- `tools/audit_codebase.py` - Scans actual code
- `tools/audit_documentation.py` - Cross-references docs
- `tools/generate_current_state.py` - Generates authoritative state
- `CURRENT_STATE.md` - Single source of truth

**Critical Discovery**: MCTS Worker Pool already implemented (173 lines, Dec 16 23:58)
- File: `mcts_trader/jax_agent/mcts_worker_pool.py`
- Multi-process parallelization, CPU-only workers
- Imported in `train_jax_fast.py` but needs verification if active

**Impact**:
- Fixed documentation drift
- TASK_01 changed from implementation → verification
- Established audit workflow for future features

**Files Created**:
- CURRENT_STATE.md (authoritative)
- AUDIT_SUMMARY.md
- AUDIT_IMPLEMENTATION_STATUS.md
- AUDIT_DOCUMENTATION_DRIFT.md

---

## December 16, 2025 - Architecture Design Session

**Completed by**: Claude Code

### Paradigm Shift: Trading Agent → Portfolio Manager

**Key Realization**: System is not a trading agent, it's a portfolio manager for 64 assets.

**New Focus**:
- Portfolio rebalancing tactics (rotation, hedging, risk parity)
- Cross-pair trading (BTC→ETH direct, not BTC→USD→ETH)
- Multi-action per step (rapid rebalancing)
- Adversarial robustness

### Extended Synthetic Curriculum (Levels 0-6)

**Levels 0-3** (Basic Mechanics):
- Level 0: Sine waves (validate architecture)
- Level 1: Lagged pairs (test correlation encoder)
- Level 2: Dip crashes (test patience)
- Level 3: Noisy trends (test fee awareness)

**Levels 4-6** (Portfolio Tactics):
- Level 4: Portfolio rotation (capital rebalancing)
- Level 5: Adversarial market (robustness training)
- Level 6: Chaos mode (black swan survival)

**Philosophy**: "If a small TRM cannot solve a noiseless sine wave, it will never solve Bitcoin."

### Multi-Task Learning: Prediction Heads

**Gap Identified**: Model learns VALUE but not PRICE TRAJECTORIES.

**Solution**: Add prediction heads to latent space:
- Price predictor (next 10 steps)
- Volatility predictor (risk level)
- Regime classifier (bull/bear/sideways)

### MCTS Parallelization Critical Issue

**Bottleneck Identified**:
- CPU: 1 core at 90%, 15 cores idle
- GPU: 60% util (starved by CPU)

**Solution Designed**: Multi-process worker pool (4 workers × 4 cores)
- Expected: 3-4× speedup
- Priority: CRITICAL

### Model Scaling Analysis

**Current**: 7M params, 32 MCTS sims
**Required for AlphaGo-level breakthroughs**: 25M params, 256-512 sims
**Hardware**: RTX 3090 (24GB) can handle 25M + batch 256 + 512 sims

### Task Files Created

5 detailed implementation tasks for Gemini:
1. TASK_01: MCTS Parallelization (CRITICAL, 3× speedup)
2. TASK_02: Synthetic Generators (Levels 0-3)
3. TASK_03: Model Scaling (7M → 25M)
4. TASK_04: Portfolio Curriculum (Levels 4-6)
5. TASK_05: Prediction Heads (multi-task learning)

### Design Documents Created

- SYNTHETIC_CURRICULUM_DESIGN.md: Complete Level 0-3 spec
- ADVANCED_ORDERS_DESIGN.md: Limit/stop orders (6 actions per asset)
- PORTFOLIO_MANAGER_TRAINING.md: Master design for portfolio tactics
- PREDICTION_LATENT_SPACE.md: Multi-task learning architecture

**Key Insights**:
- Validation First: Level 0 (sine) is gatekeeper
- No Timelines: Focus on correctness, iterate fast
- Goal: Discover inhuman tactics (like AlphaGo move 37)

---

## December 14, 2025 - Gemini Improvements

### 1. Rainbow Table Training Continuity

**Problem**: Training schedule was launching independent processes for each period, causing model reset and shape mismatches (61 assets vs 59 assets).

**Solution**:
- Checkpoint Transfer: Track latest checkpoint, pass via `--load-checkpoint`
- Shape Standardization: Pad asset lists to TARGET_ASSETS=64 using duplicates
- Updated `run_scheduled_training.py` and `FastTradingEnv` patches

### 2. Adaptive Curriculum & "Maniac Mode"

**Problem**: Model converging to "Turtle Mode" (Reward 0.00, holding cash) because 26bps fee + 5bps slippage barrier too high.

**Solution**:
- Physics-First Curriculum:
  - Level 0: Fee 0, Slippage 0 (Frictionless Sandbox)
  - Progression: If avg_reward > 5.0%, ramp slippage then fees
- Maniac MCTS Settings:
  - Increased num_simulations: 16 → 32
  - Increased dirichlet_alpha: 0.5 (high exploration)

### 3. High-Performance Infrastructure

**Problem**: Single-core CPU bottleneck (SQLite queries) starving GPU (67% load).

**Solution**:
- RAM Caching: Preload entire episode's OHLCV data into numpy matrix
- O(1) array lookups instead of SQL queries
- Reduced CPU load, GPU can focus on increased MCTS simulations

**Modified Files**:
- tools/run_scheduled_training.py
- train_jax_fast.py
- mcts_trader/data/history.py
- mcts_trader/jax_agent/mctx_search.py
- mcts_trader/env/trading_env.py

---

## December 12, 2025 - Pre-Training Strategy Redesign

**Completed by**: Gemini

### Two-Phase Training Strategy

**Phase 1: Large-Scale Pre-Training**
- Assets: Random 64 non-forex pairs
- Time Window: 2024-2025
- Episode Length: 1440 steps = 24 hours
- Training: 10,000 epochs (~4 hours)

**Phase 2: Fine-Tuning**
- Time Window: Oct-Dec 2025
- Episode Length: 200 steps
- Training: 2,000 epochs (~50 minutes)

### Technical Improvements

**Enhanced Asset Selection**:
- `--random-subset N`: Select N random assets
- `--exclude-forex`: Filter out fiat/fiat pairs
- `--date-start`, `--date-end`: Time window filtering
- `--min-candles N`: Minimum candle count requirement

**Forex Pair Detection**:
- Implemented `is_forex_pair()` to identify and exclude fiat pairs
- Prevents model from learning forex-specific behaviors

**Status**: ✅ Implementation complete and tested
- JAX acceleration: ~1.4s per epoch (200 steps)
- Rewards: Positive (+11.79 SPOT, +0.49 MARGIN)

---

## December 11, 2025 - Critical Bug Fixes

**Completed by**: Gemini

### 1. "Unit Confusion" & The -7713% Bug

**Issue**: Catastrophic trade (ETHBTC) sized at $1.4M on $100 account.
**Root Cause**: Old logic treated Quote Currency Price (0.035 BTC) as USD.
**Fix**: `portfolio.py` now correctly converts all values using live FX rates.
**Verification**: New paper trading shows realistic PnL (-0.36%).

### 2. Spot Mode & Negative Cash

**Issue**: Paper trader accumulated negative cash ($-37).
**Root Cause**: `execute_actions` calculated size based on Equity instead of Cash.
**Fix**: Strict cash clamping - buy orders capped at `max_spendable_cash`.

### 3. "No Sell Signal" / Masking Bug

**Issue**: Model never issued Sell commands in Spot Mode.
**Root Cause**: `get_valid_actions` called before `update_prices`.
**Fix**: Update portfolio state immediately after receiving market data.

### Infrastructure Upgrades

**Real Trading Engine** (`real_trade_jax.py`):
- Private API support (Authentication, Signing, Order Management)
- `RealPortfolioManager` syncs state from Kraken
- Safety: Dry Run mode, configurable limits, kill switch

**Robustness & Resilience**:
- WebSocket reconnects indefinitely (was capped at 10)
- Fixed session lifecycle bugs
- Graceful exit with performance report

**Replay Tooling**:
- `replay_trades.py` supports async FX rate fetching

**Status**: Paper trading logic mathematically sound, real trading ready for dry run.

---

## December 09, 2025 - Financial Realism Update

**Completed by**: Gemini

### Changes Implemented

1. **Resolved "NaN Loss"**: Using `int32` for saving values in JAX pipeline.

2. **Fixed Paper Trading Validation**:
   - Implemented `ActionLogger` recording every trade
   - Logs: timestamp, symbol, action, price, size, fee, confidence, portfolio value, cash
   - Output: `trade_log.csv`

3. **Realistic Market Simulation**:
   - Trading Fees: 0.26% taker fee deduction
   - Slippage: 0.05% price adjustment
   - Correct Equity Calculation: Subtract short liabilities

4. **Training Integration**:
   - `FastTradingEnv` uses realistic fees (26 bps fee, 5 bps slippage)
   - Ensures training reflects real-world costs

5. **Inference Integration**:
   - `PaperTrader` uses same realistic parameters
   - Consistency between training and paper trading

6. **Verification**:
   - Unit tests (`test_portfolio_math.py`) verified correctness
   - All tests passed

**Impact**: More robust and realistic simulation for training and evaluation.

---

## December 08, 2025 - First Successful Training

**Milestone**: Completed 49,750 training iterations with working model.

### Problems Solved

**Problem 1: LightZero Memory Issues**
- Massive memory leaks, C++ binding tensor shape issues
- **Solution**: Migrated to pure JAX with mctx library

**Problem 2: PyTorch/JAX GPU Conflict**
- System would hang (100% CPU, 3% GPU)
- **Solution**: Keep encoder on CPU, JAX handles GPU training

**Problem 3: Database Filtering Performance**
- Each step took ~500ms (GROUP BY on every step)
- **Solution**: Pre-compute valid timestamps at init
- Steps now use O(log n) binary search - instant

**Problem 4: Episodes Ending After 1 Step**
- SQL placeholder `?` not working properly
- **Solution**: Direct string substitution

**Problem 5: Constant Zero Rewards**
- Wrong API usage and price mapping
- **Solution**: Fixed `execute_actions()` call and price dict conversion

**Problem 6: SQL Query Empty**
- `id_to_symbol` mapping mismatch
- **Solution**: Rebuild mapping from actual query results

### Architecture Decisions

**Why JAX over PyTorch?**
- mctx library is JAX-native and blazing fast
- Full JIT compilation of training loop
- Better GPU memory management

**Why Fast Encoder?**
- Full correlation matrix: O(n²) = 4096 ops per step
- Fast encoder: 27 feature channels, vectorized
- Speedup: 10-20x faster

**27 Channels**:
- 0-3: Normalized OHLC
- 4-7: Returns (1/5/15/60 min)
- 8-11: Volume patterns
- 12-15: Price momentum
- 16-19: Volatility measures
- 20-23: Price percentiles
- 24-26: VWAP and bands

### Performance Achieved

| Metric | Before | After |
|--------|--------|-------|
| Self-play per episode | 9 minutes | 1.5-2.5 seconds |
| Steps per episode | 1 | 200 |
| Database step time | 500ms | 0ms (pre-computed) |
| Encoding time | ~200ms | ~10ms |

### Known Issues (v1.0)

1. NaN Loss (but model still learns)
2. Paper Trading WebSocket timestamp parsing
3. No correlation features (removed for speed)

---

## Key Learnings

1. **Documentation drift is real** - Multiple AI agents = conflicting notes
2. **Code is ground truth** - Scan codebase, don't trust docs
3. **Audit regularly** - After every major feature
4. **JAX is fast** - ~1.4s per epoch, 10-20x speedup over PyTorch
5. **Financial realism matters** - Fees and slippage change agent behavior
6. **Portfolio management ≠ trading** - Focus on rotation, hedging, risk parity

---

## Current System State

See `CURRENT_STATE.md` for authoritative current state.

**Model**: 7M params TRM (d_model=64, H=3, L=4)
**MCTS**: 32 sims (needs increase to 256-512)
**Batch Size**: 512
**Hardware**: RTX 3090 (24GB), 16-core CPU, 46GB RAM

**Bottlenecks**:
1. CPU parallelization (needs verification)
2. Model capacity (7M might be too small)
3. MCTS depth (32 sims insufficient for breakthroughs)

**Next**: Verify MCTS worker pool, implement synthetic curriculum.
