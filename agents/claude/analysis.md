# Claude Code Analysis Log

## Session: 2025-12-03

### Initial Codebase Analysis

**Summary**: Analyzed TRM+MCTS trading agent codebase for documentation creation.

**Architecture Overview**:
- **Data Pipeline**: Kraken API → SQLite (20GB) → HistoricalMarket → Encoder
- **State Encoding**: 64 assets × 60min lookback → 27×64×64 tensor (HSV + correlation)
- **Model**: TinyRecursiveTradingModel (7M params) with H_cycles=3, L_cycles=4
- **MCTS**: LightZero MuZero with 100 simulations, 10-step lookahead
- **Action Space**: 192 discrete (64 assets × 3 actions)

**Key Findings**:
1. **Critical Monkey-Patch** (main.py:12-51): Required to fix C++ binding error in LightZero
   - Issue: C++ MCTS expects reward as Python list, receives tensors
   - Risk: Removing this patch causes runtime crashes

2. **Multiprocessing**: Uses 'spawn' method (required for PyTorch + LightZero)

3. **Memory Budget** (RTX 3090):
   - Model: ~14MB
   - Activations (batch=2048): ~3GB
   - MCTS trees: ~2GB
   - Replay buffer: ~10GB
   - Total: ~16GB used, 8GB headroom

4. **Environment Warmup**: Requires 60 steps to populate encoder buffer before first observation

**Performance Characteristics** (from docs):
- Target: 200+ episodes/hour
- MCTS inference: <100ms per decision
- Data collection: 4-8 hours for 64 assets × 30 days

### Configuration Review

**Current Settings** (trading_config.py):
- `num_simulations: 100` - Reasonable for speed/quality balance
- `batch_size: 2048` (Gemini changed to 512)
- `replay_buffer_size: 50000` (Gemini changed to 10000)
- `learning_rate: 0.0002` - Standard for Adam
- `collector_env_num: 4` - Parallel environments

**Gemini's Changes Analysis**:
- Batch size reduction: 2048→512 (75% reduction)
  - Pro: Reduces memory by ~2.25GB
  - Con: May impact training stability (larger batches = more stable gradients)
  - Recommendation: Monitor loss variance after change

- Replay buffer reduction: 50000→10000 (80% reduction)
  - Pro: Reduces memory by ~8GB
  - Con: Less diverse experience replay (may hurt sample efficiency)
  - Recommendation: Track evaluation performance to detect regression

---

*Status: Initial analysis complete*
*Next: Awaiting task assignment for deeper profiling*
