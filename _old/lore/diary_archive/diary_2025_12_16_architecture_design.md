# Development Diary - December 16, 2025
## Architecture Design Session: Portfolio Manager Revolution

### Session Overview
**Duration**: Extended design session (Claude Code)
**Focus**: Complete architectural redesign from "Trading Agent" to "Portfolio Manager"
**Outcome**: 5 detailed task files + 3 design documents for implementation

---

## Key Insights

### 1. Fundamental Shift: Not a Trader, a Portfolio Manager

**Old Paradigm**: Agent trades individual assets reactively
**New Paradigm**: Agent manages 64-asset portfolio strategically

**Critical Realization**:
> "This is NOT a trading agent. This is a PORTFOLIO MANAGER. Goals: manage 64 assets simultaneously with complex correlation awareness, trade cross-pairs directly, develop inhuman tactics (like AlphaGo's move 37), face adversarial opponents."

**Implications**:
- Action space should enable portfolio rebalancing, not just single trades
- Training must teach rotation tactics, not just trend following
- Success metric is portfolio Sharpe ratio, not individual trade win rate

---

### 2. AlphaGo-Style Self-Play is Possible (But Needs Scaling)

**Question**: Can our 7M TRM + 32 MCTS sims achieve AlphaGo-level breakthroughs?

**Analysis**:
- AlphaGo Zero: 20M params, 1600 sims, 362 actions
- Our setup: 7M params, 32 sims, 384-16K actions
- **Verdict**: Close, but need 2-3× scaling

**Required Upgrades**:
1. Model: 7M → 25M params (more capacity)
2. MCTS: 32 → 256-512 sims (deeper search)
3. Action space: Simplify to 20-50 portfolio templates for adversarial training

**Bottleneck Identified**: Not RAM/VRAM (plenty of headroom on 24GB GPU), but **CPU MCTS tree traversal** (currently 1 core, needs 16 cores).

---

### 3. Synthetic Curriculum is Critical Validation

**Philosophy**: "If a small TRM cannot solve a noiseless sine wave, it will never solve Bitcoin."

**Curriculum Designed** (Levels 0-6):

**Level 0**: Pure sine waves (validate architecture)
- Test: Basic buy low / sell high
- Pass: Reward > +5%
- If fails: Architecture is broken - STOP and debug

**Level 1**: Lagged pairs (test correlation encoder)
- Test: Correlation matrix encoder works
- Pass: Reward > +4%

**Level 2**: Dip crashes (test patience)
- Test: Wait for rare opportunities
- Pass: Sharpe > 1.5

**Level 3**: Noisy trends (test fee awareness)
- Test: Only trade when signal > cost
- Pass: Reward > +1%, trades < 20%

**Level 4**: Portfolio rotation (NEW - portfolio tactics)
- Test: Rotate capital between winners
- Pass: Sharpe > 2.0, turnover > 50%

**Level 5**: Adversarial (NEW - robustness)
- Test: Survive opponent trying to break you
- Pass: Positive reward vs adversary

**Level 6**: Chaos mode (NEW - black swans)
- Test: Survive flash crashes, delistings
- Pass: 0% bankruptcy, drawdown < 30%

**Key Insight**: Levels 4-6 are about **portfolio management**, not trading.

---

### 4. Missing Piece: Explicit Prediction

**Discovery**: Model learns VALUE (expected reward) but not PRICE TRAJECTORIES.

**Problem**: Latent space only encodes "what to do" not "what will happen"

**Solution**: Multi-task learning with prediction heads:
- Price predictor (next 10 steps)
- Volatility predictor (risk level)
- Regime classifier (bull/bear/sideways)

**Benefits**:
1. Richer latent representations (multi-task proven to improve RL)
2. Interpretable (can check if model understands markets)
3. Useful for risk management (position sizing, stops)

**Test**: On Level 0 (sine waves), price prediction MSE should → 0 (perfect).

---

### 5. Current System Bottleneck: MCTS Parallelization

**Critical Issue**:
- CPU: 1 core at 90%, 15 cores idle
- GPU: 60% util (starved by CPU)
- Batch size: 512 (already optimized)

**Root Cause**: MCTS tree traversal is sequential in Python, doesn't utilize all cores.

**Solution**: Multi-process MCTS worker pool
- 4 workers × 4 cores = 16 cores utilized
- Split batch: 512 → 4×128 per worker
- **Expected: 3-4× speedup**

**Priority**: CRITICAL - this is the biggest performance win available.

---

## Design Documents Created

### 1. SYNTHETIC_CURRICULUM_DESIGN.md
Complete specification for Levels 0-3:
- Data generators (sine, lag, dip, trends)
- Pass criteria per level
- Augmented reality (mirror, warp, noise)
- Implementation architecture

### 2. ADVANCED_ORDERS_DESIGN.md
Extension to 6 actions per asset:
- Limit orders (buy/sell at target price)
- Stop orders (breakout/breakdown triggers)
- Price offset mechanism (% from current)
- Pending order book system
- Action space: 192 → 384

### 3. PORTFOLIO_MANAGER_TRAINING.md
**The Master Design**:
- Extended curriculum (Levels 4-6)
- Cross-pair trading (BTC→ETH direct, 50% fee savings)
- Multi-action per step (rapid rebalancing)
- Adversarial training architecture (protagonist vs antagonist)
- AlphaGo-style self-play for breakthrough tactics

### 4. PREDICTION_LATENT_SPACE.md
Multi-task learning design:
- Prediction heads architecture
- Target computation algorithms
- Integration with existing RL
- Ablation study methodology

---

## Task Files Created for Gemini

### TASK_01_FIX_MCTS_PARALLELIZATION.md
**Priority**: CRITICAL
**Impact**: 3-4× speedup
**Effort**: 4 hours

Implementation:
- `mcts_worker_pool.py` (multi-process MCTS)
- Integration with training loop
- Benchmark tests (speedup > 3×)

**Acceptance**: All 16 cores utilized, GPU 85%+

---

### TASK_02_SYNTHETIC_MARKET_GENERATORS.md
**Priority**: HIGH
**Impact**: Validate architecture before real data
**Effort**: 6 hours

Implementation:
- `synthetic_market.py` (Levels 0-3)
- Integration with `FastTradingEnv`
- Visualization tools
- Pass criteria validation

**Acceptance**: Level 0 converges to +5% reward

---

### TASK_03_SCALE_MODEL_TO_25M.md
**Priority**: MEDIUM
**Impact**: Enable adversarial training
**Effort**: 3 hours

Implementation:
- Configurable model sizes (7M/15M/25M)
- Factory method pattern
- CLI argument `--model-size`
- Parameter count tests

**Acceptance**: All 3 sizes fit in 24GB VRAM

---

### TASK_04_PORTFOLIO_CURRICULUM.md
**Priority**: HIGH
**Impact**: Teach portfolio tactics
**Effort**: 8 hours

Implementation:
- Level 4 generator (correlated rotation)
- Level 6 generator (chaos mode)
- Pass criteria module
- Curriculum orchestrator

**Acceptance**: Model learns to rotate capital (turnover > 50%)

---

### TASK_05_PREDICTION_HEADS.md
**Priority**: MEDIUM
**Impact**: Multi-task learning, better generalization
**Effort**: 6 hours

Implementation:
- Prediction heads in MuZeroNetworks
- Target computation module
- Loss functions for predictions
- Ablation study

**Acceptance**: Price MSE < 0.01 on sine waves, regime accuracy > 60%

---

## Strategic Decisions

### 1. Execution Order
1. **TASK_01** first (biggest speedup, unblocks everything)
2. **TASK_02** next (validate architecture)
3. **TASK_03** after validation (scale model)
4. **TASK_04** for advanced training (portfolio tactics)
5. **TASK_05** optional enhancement (predictions)

### 2. Hardware Strategy
**Current**: RTX 3090 (24GB VRAM)
**Future**: 4×5090s available for a week (128GB total)

**Multi-GPU Plan** (when available):
- Data parallelism: 4× speedup (simple)
- Async self-play: 3 GPUs generate, 1 trains (AlphaGo style)
- Deep search: 1024+ MCTS sims (pipeline parallelism)

### 3. Model Size Philosophy
**No timelines** - focus on correctness and iteration speed.

**Scaling approach**:
- Small (7M): Fast iteration, testing
- Medium (15M): Curriculum training (Levels 0-4)
- Large (25M): Adversarial training (Level 5), production

**Beyond 25M**: Diminishing returns for this problem size.

---

## Technical Insights

### Memory Analysis (RTX 3090)
```
Model (25M):              ~100MB
Optimizer state:          ~200MB
Batch (256):              ~1.5GB
MCTS trees (512 sims):    ~6GB
──────────────────────────────
Total:                    ~8GB / 24GB

Headroom: 16GB unused
```

**Bottleneck**: Not memory, but CPU tree traversal speed.

### MCTS RAM Breakdown
```
32 sims:   ~1.8MB per search
128 sims:  ~7.2MB per search
256 sims:  ~14.4MB per search
1600 sims: ~90MB per search

Batched (256): 256 × 90MB = 23GB for 1600 sims
```

**Conclusion**: Can run 256+ sims without RAM issues (have 46GB system RAM).

### Action Space Complexity
**Current**: 64 × 3 = 192 actions (Sell/Hold/Buy)
**Advanced orders**: 64 × 6 = 384 actions (+limit/stop)
**Cross-pair**: 64 × 64 = 4096 pairs (full portfolio rebalancing)

**For adversarial training**: Simplify to 20-50 portfolio templates (not 4096 pairs) - MCTS works better with discrete, bounded action spaces.

---

## Open Questions & Future Work

### 1. Cross-Pair Trading Implementation
Not yet designed in detail. Needs:
- Action space redesign (source_asset, target_asset, size)
- Cross-rate calculation (BTC/USD ÷ ETH/USD = BTC/ETH)
- Single fee (not 2×)
- **Estimated**: 6 hours (TASK_06)

### 2. Adversarial Training Architecture
Designed conceptually, needs implementation:
- Dual model (protagonist + antagonist)
- Self-play loop (both improve by exploiting weaknesses)
- Price manipulation constraints (antagonist budget)
- **Estimated**: 12 hours (TASK_07)

### 3. Hierarchical Planning
For cross-pair (4096 actions), consider:
- High-level: Choose target portfolio state (20 templates)
- Low-level: Plan cross-pair trades to reach target
- Similar to MuZero's hierarchical approach

### 4. Dynamic Asset Universe
Currently fixed 64 assets. Future:
- Transformer backbone (asset-as-token)
- Train on 64, deploy on 500+ assets
- Zero-shot generalization

---

## Risk Mitigation

### If Level 0 Fails
**Symptom**: Model can't learn sine waves after 1000 epochs

**Diagnosis**:
1. Check observation encoder (is price data reaching model?)
2. Check action execution (are trades happening?)
3. Check reward calculation (is PnL correct?)
4. Manual trading test (force buy at troughs)

**Action**: STOP all work, debug architecture before proceeding.

### If MCTS Parallelization Fails
**Rollback**: Simple flag to disable worker pool, use serial MCTS.

### If Large Model OOMs
**Solution**: Reduce batch size (512 → 256 → 128).

---

## Metrics to Track

### Per-Level Pass Criteria
- Level 0: Reward > +5%
- Level 1: Reward > +4%
- Level 2: Sharpe > 1.5
- Level 3: Reward > +1%, trades < 20%
- Level 4: Sharpe > 2.0, turnover > 50%
- Level 5: Reward > 0 vs adversary
- Level 6: Bankruptcy 0%, drawdown < 30%

### System Metrics
- Training speed (epochs/hour)
- CPU utilization (% cores active)
- GPU utilization (% compute)
- VRAM usage
- Prediction accuracy (MSE, regime accuracy)

---

## Lessons Learned

### 1. Design Before Code
This session was pure design - no code written. Result:
- 5 detailed task files (ready for implementation)
- 3 architecture docs (reference material)
- Clear critical path (TASK_01 → TASK_02 → TASK_03)

**Benefit**: Gemini can execute tasks without ambiguity.

### 2. Validate Early
Synthetic Level 0 (sine waves) is the gatekeeper:
- If model can't learn sine → architecture broken
- If model masters sine → proceed to complexity

**Saves time**: Hours of debugging, not weeks of failed training.

### 3. Multi-Task Learning is Underutilized
Adding prediction heads:
- +10% training time
- Better generalization (proven in research)
- Interpretable diagnostics

**Low-risk, high-reward** enhancement.

### 4. AlphaGo Breakthroughs Need Scale
7M params + 32 sims is not enough for emergent tactics.

**Required**: 25M params + 256-512 sims + adversarial training.

**Then** we might see "move 37" moments in portfolio management.

---

## Next Steps

### Immediate (This Week)
1. Hand off task files to Gemini
2. Gemini executes TASK_01 (MCTS parallelization)
3. Validate 3× speedup
4. Proceed to TASK_02 (synthetic data)

### Short-Term (Next Week)
5. Complete curriculum Levels 0-3
6. Scale model to 25M
7. Implement Levels 4-6

### Medium-Term (Next Month)
8. Add prediction heads
9. Implement cross-pair trading
10. Begin adversarial training

### Long-Term (When 4×5090 Available)
11. Multi-GPU async self-play
12. Deep search (1024+ sims)
13. Hunt for breakthrough tactics

---

## Philosophical Notes

### Speed > Perfection
> "No timelines. Break fast, document, fix, iterate."

This is RL research, not production engineering. Rapid iteration beats perfect planning.

### Trust But Verify
> "If Level 0 fails, STOP and debug."

Synthetic curriculum is the gatekeeper. Don't proceed without validation.

### Inhuman Tactics are the Goal
> "Develop tactics that are unthinkable for humans, similar results like with Go."

This is why we're doing this. Not to mimic human traders, but to discover strategies humans can't conceive.

If the model learns to:
- Exploit correlation cascades
- Bait adversaries into wasting manipulation budget
- Harvest volatility through negative correlation pairs

...we've succeeded, even if we don't understand WHY it works.

---

## Summary

**Today's Output**:
- 5 task files (20+ hours of implementation work scoped)
- 3 design documents (architecture reference)
- 1 master plan (portfolio manager revolution)

**Critical Path**:
1. Fix MCTS parallelization (3× speedup)
2. Validate on synthetic Level 0 (sine waves)
3. Scale to 25M params
4. Train on portfolio curriculum
5. Adversarial training for breakthroughs

**Next Session**: Gemini executes TASK_01, reports results.

**Vision**: A portfolio manager that thinks like AlphaGo, manages like a quant fund, and discovers tactics humans can't imagine.

Let's build it. 🚀
