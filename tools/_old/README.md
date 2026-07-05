# Archived Tools (`_old/`)

These are ad-hoc diagnostic and utility scripts that are no longer part of the main codebase.

## Files Here

### HistoricalMarket Debug Tools (5 files)
- `debug_step.py` - Debugs old replay engine step behavior
- `debug_detailed.py` - Database analysis for old replay engine
- `add_time_features.py` - Monkey-patch for PyTorch encoder via HistoricalMarket
- `verify_timestamps.py` - Temporal validation for old replay engine
- `test_system.py` - Performance benchmarks for old replay engine

**Status**: HistoricalMarket is not in current training pipeline. These tools are not needed.

### Trade Replay Tool
- `replay_trades.py` - Trade log replay for backtesting

**Status**: Will be replaced by VectorBT-based backtesting. Recreate as needed.

### Documentation
- `generate_current_state.py` - Old documentation generator

**Status**: Outdated. Current architecture uses different systems (JAX instead of PyTorch, etc).

## Why These Are Archived

These tools are:
1. Ad-hoc diagnostic scripts, not core training code
2. Dependent on systems that are being replaced (HistoricalMarket, trade replay)
3. Not part of the main training pipeline
4. Can be easily recreated if needed

## Important Notes

- **All custom market code will be managed by VectorBT** for training environment and paper/replay trading
- Main codebase focus: clean working trader model with clear interfaces
- These tools are handy ad-hoc utilities, not critical infrastructure

## Future

When VectorBT-based backtesting is implemented, recreate:
- `tools/backtest_vectorbt.py` (to replace `replay_trades.py`)
