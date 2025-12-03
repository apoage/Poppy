# Project Context

**Last Updated**: 2025-12-03 22:35 UTC

## Current State

**Project**: TRM+MCTS Trading Agent for Kraken cryptocurrency trading
**Framework**: LightZero (MuZero) + PyTorch
**Status**: Training pipeline implemented, debugging phase

### Active Components
- ✅ Data Infrastructure (Kraken API + SQLite cache)
- ✅ Multi-Asset State Encoder (27-channel HSV encoding)
- ✅ TRM Model (7M params, recursive reasoning)
- ✅ LightZero Integration (MuZero wrapper)
- ✅ Trading Environment (Gymnasium interface)
- ✅ Training Loop (main.py with monkey-patch for C++ bindings)

### Database
- **File**: `trading_data.db` (20GB)
- **Coverage**: 64 volatile assets, 30 days of 1-minute OHLCV data
- **Status**: Populated and ready

### Recent Changes
- CLAUDE.md documentation created (already exists in repo)
- Git user set to claude@workstation for Claude Code
- Gemini has unstaged changes to trading_config.py (batch_size: 2048→512, replay_buffer: 50000→10000)

## Known Issues
- Training process creates many experiment folders (`default_experiment_*`)
- Requires monkey-patch in main.py for C++ binding compatibility
- GPU memory needs monitoring (RTX 3090, 24GB)

## Next Steps
See `agents/tasks/todo.md` for task assignments
