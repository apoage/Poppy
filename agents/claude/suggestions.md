# Claude Code Suggestions

## Immediate Recommendations

### 1. Profiling & Monitoring
- [ ] Add GPU memory profiling to training loop (track peak usage per batch)
- [ ] Instrument MCTS simulation time (identify if bottleneck is in tree search or NN inference)
- [ ] Add tensorboard logging for loss curves, episode rewards, GPU utilization
- [ ] Create checkpoint cleanup script (many `default_experiment_*` folders accumulate)

### 2. Testing Infrastructure
- [ ] Unit tests for `MultiAssetStateEncoder` (verify 27×64×64 output)
- [ ] Integration test for `TradingEnv` reset/step cycle
- [ ] Validation script for database integrity (check for gaps in OHLCV data)
- [ ] Smoke test for training loop (ensure 10 episodes run without crash)

### 3. Code Quality
- [ ] Extract monkey-patch to separate module (`mcts_trader/utils/lightzero_compat.py`)
- [ ] Add docstrings to TRM forward pass (document H_cycles, L_cycles meaning)
- [ ] Type hints for PortfolioManager methods
- [ ] Consider using `logging` instead of `print(".", end="")` in TradingEnv

### 4. Memory Optimization (given Gemini's changes)
```python
# Add to trading_config.py for dynamic batch sizing
if torch.cuda.get_device_properties(0).total_memory < 16e9:  # <16GB
    batch_size = 256
elif torch.cuda.get_device_properties(0).total_memory < 24e9:  # <24GB
    batch_size = 512
else:
    batch_size = 2048
```

### 5. Debug Tooling
- [ ] Script to analyze training logs for anomalies (NaN losses, reward spikes)
- [ ] Visualization for state encoder output (save sample 27-channel images)
- [ ] Portfolio state inspector (track position changes over episode)

## Performance Optimizations (Lower Priority)

### Encoder Pipeline
- Consider caching correlation matrices (recompute only when needed)
- Profile HSV conversion (potential NumPy→Torch bottleneck)

### MCTS
- Reduce simulations to 50 during training, increase to 100 for evaluation
- Experiment with virtual loss in parallel search (if LightZero supports)

### Data Loading
- Async data prefetching (HistoricalMarket could preload next batch)
- In-memory caching of frequently accessed assets

---

*Priority: Focus on monitoring/profiling first to establish baseline metrics*
