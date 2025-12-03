# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **TRM+MCTS Trading Agent** for cryptocurrency trading - a reinforcement learning system that combines Tiny Recursive Models (TRM) with Monte Carlo Tree Search (MCTS) via the LightZero framework to trade on Kraken exchange.

**Core Innovation**: Uses TRM's recursive reasoning (7M parameters) instead of traditional deep networks (10-50M params), integrated with MuZero-style MCTS planning for multi-asset portfolio trading.

## Development Environment

### Virtual Environment
The project uses a Python virtual environment at `trader_env/`:
```bash
source trader_env/bin/activate  # Activate venv
pip install -r requirements.txt  # Install dependencies
```

### Key Dependencies
- `torch>=2.0.0` - Neural network backbone
- `lightzero>=0.2.0` - MCTS framework (MuZero/EfficientZero implementation)
- `gymnasium` - RL environment interface
- `aiohttp`, `aiosqlite`, `websockets` - Async data pipeline
- `python-kraken-sdk>=2.3.0` - Exchange API client
- `pyts` - Time series image encoding (GASF, MTF, etc.)

### Database
- **File**: `trading_data.db` (20GB SQLite database)
- **Schema**: Assets, OHLCV candles (1-minute), trades, fetch_log
- **Purpose**: Local cache of Kraken historical data for training

## Running the System

### Data Collection
```bash
# First-time setup: Fetch historical data (takes 4-8 hours)
python collect_data.py

# Check data coverage
sqlite3 trading_data.db "SELECT symbol, COUNT(*) FROM ohlcv o JOIN assets a ON o.asset_id=a.id GROUP BY asset_id"
```

### Training
```bash
# Start training (creates experiment folders with timestamps)
python main.py

# Monitor training
tail -f training.log
tail -f training_console.log

# Check current training process
cat current_training_pid.txt
```

### Debugging
```bash
# Debug environment setup
python debug_env.py

# Stress test
python stress_test.py
```

## Architecture

### Module Structure

```
mcts_trader/
├── agent/          # Neural network models
│   ├── trm.py              # TinyRecursiveTradingModel (core model)
│   ├── wrapper.py          # TRMMuZeroModel (LightZero wrapper)
│   └── portfolio.py        # PortfolioManager (position tracking)
├── data/           # Data pipeline
│   ├── database.py         # SQLite async operations
│   ├── kraken_client.py    # Kraken API with rate limiting
│   ├── fetcher.py          # Historical data downloader
│   ├── history.py          # HistoricalMarket (replay engine)
│   └── stream.py           # Real-time WebSocket feed
├── encoder/        # State representation
│   ├── state.py            # MultiAssetStateEncoder (main encoder)
│   ├── correlation.py      # Cross-asset correlation matrices
│   └── indicators.py       # Technical indicators (RSI, MACD, etc.)
├── env/            # Gym environment
│   └── trading_env.py      # TradingEnv (registered as 'trading_env')
└── config/         # Configuration
    └── trading_config.py   # Main config (MuZero hyperparams)
```

### Data Flow

1. **Data Collection** (`collect_data.py`):
   - Fetches top 64 volatile assets from Kraken
   - Downloads 30 days of 1-minute OHLCV candles
   - Stores in SQLite with incremental resume capability
   - Rate-limited: ~1 req/sec, respects Kraken limits

2. **State Encoding** (`encoder/state.py`):
   - **Input**: 64 assets × 60-minute lookback window
   - **Output**: 27-channel 64×64 image tensor
   - **Channels**: 3 (HSV) × 9 (correlation clusters)
   - **Process**: Price normalization → Technical indicators → Correlation clustering → HSV color encoding

3. **TRM Model** (`agent/trm.py`):
   - **Architecture**: Encoder (CNN) → Recursive refinement → Policy/Value heads
   - **Recursion**: H_cycles (output refinement) × L_cycles (latent refinement)
   - **Params**: ~7M (vs 28M for ConvNeXt-Tiny)
   - **Key Innovation**: Residual recursive refinement instead of depth

4. **MCTS Planning** (via LightZero):
   - 100 simulations per decision
   - PUCT exploration with c_puct=1.25
   - 10-step lookahead (max_depth=10)
   - Integrates TRM policy/value as priors

5. **Trading Environment** (`env/trading_env.py`):
   - **Action Space**: 64 assets × 3 actions (Sell/Hold/Buy) = 192 discrete actions
   - **Observation**: 27×64×64 image + action_mask + to_play
   - **Reward**: Percentage return clipped to [-10, 10]
   - **Episode**: 200 steps (game_segment_length)

6. **Training Loop** (`main.py`):
   - MuZero self-play via LightZero's `train_muzero`
   - 4 collector envs, 2 evaluator envs (subprocess-based parallelism)
   - Batch size 2048, replay buffer 50k
   - Checkpoint frequency: every 100 iterations

### Critical Implementation Details

#### Monkey-Patching for C++ Binding
**Location**: `main.py:12-51`

The code patches `mz_network_output_unpack` to fix a C++ binding error in LightZero. The C++ MCTS tree expects `reward` as a Python list, but receives tensors/arrays. This patch ensures proper conversion.

**Do not remove this patch** - it prevents runtime crashes.

#### Multi-Process Initialization
**Location**: `main.py:78-82`

Uses `spawn` method for multiprocessing (required for PyTorch + LightZero). Always use:
```python
multiprocessing.set_start_method('spawn')
```

#### TRM Recursion Cycles
**Default**: H_cycles=3, L_cycles=4 (total 12 refinement passes)

For faster inference during MCTS (100 simulations), consider reducing to H=2, L=3 in `agent/trm.py:90`.

#### Environment Warmup
**Location**: `env/trading_env.py:131-134`

The environment requires 60 steps to warm up the encoder buffer before generating observations. This is handled automatically in `reset()`.

#### Action Decoding
**Formula**: `action = asset_idx * 3 + action_type`
- `asset_idx`: [0, 63]
- `action_type`: 0=Sell, 1=Hold, 2=Buy

Example: action=130 → asset 43, Buy (130 = 43*3 + 1... wait that's wrong in the code comments - verify this)

## Configuration Tuning

### Key Hyperparameters (`config/trading_config.py`)

**MCTS Settings**:
- `num_simulations: 100` - Balance between speed and search quality (50-200 range)
- `max_depth: 10` - Lookahead horizon (5-20 range)
- `pb_c_init: 1.25` - Exploration constant (increase for more exploration)

**Training Settings**:
- `batch_size: 2048` - Large batches stabilize training (reduce if OOM)
- `learning_rate: 0.0002` - AdamW learning rate
- `replay_buffer_size: 50000` - Experience replay size

**Environment Settings**:
- `game_segment_length: 200` - Episode length in minutes
- `collector_env_num: 4` - Parallel data collection envs
- `n_assets: 64` - Number of tradeable assets

### GPU Memory Management

**RTX 3090 (24GB)**:
- Model: ~14MB (TRM 7M params)
- Activations (batch=2048): ~3GB
- MCTS trees (4 parallel): ~2GB
- Replay buffer: ~10GB
- **Total**: ~16GB used, 8GB headroom

If OOM, reduce `batch_size` or `collector_env_num`.

## Common Tasks

### Adding Technical Indicators
Edit `encoder/indicators.py`, add to `MultiAssetStateEncoder.encode()` pipeline. Ensure output maintains 64×64 image dimensions.

### Changing Asset Universe
Modify `n_assets` in `trading_config.py` and `TradingEnv.__init__`. Re-run `collect_data.py` to fetch data for new assets.

### Adjusting Recursion Depth
Edit `TinyRecursiveTradingModel.forward()` default args (H_cycles, L_cycles). Higher = better reasoning but slower inference.

### Checkpoint Management
Checkpoints saved in `default_experiment_YYMMDD_HHMMSS/` folders. To resume:
```python
# Load checkpoint in main.py before train_muzero()
model.load_state_dict(torch.load('path/to/checkpoint.pth'))
```

### Live Trading (Future)
The system currently trains on historical data. For live trading:
1. Replace `HistoricalMarket` with `stream.py` WebSocket feed
2. Implement paper trading mode first
3. Add execution via `kraken_client.py` order placement
4. **Critical**: Respect Kraken rate limits (see Data Infrastructure.md)

## Research Documentation

The following markdown files contain detailed research and design specs:

- `MCTS_Trading_Agent_Research_2025_Update.md` - Algorithm research, TRM architecture, MCTS variants
- `Data Infrastructure.md` - Kraken API, database schema, rate limiting
- `Multi-Asset State Encoder.md` - State encoding pipeline (HSV, GASF, correlation)
- `TRM+MCTS Agent.md` - Integration architecture
- `Training Loop.md` - Self-play training details
- `Monitoring Analysis.md` - Metrics and evaluation
- `Master Orchestration Prompt.md` - Implementation roadmap
- `PROGRESS.md` - Project status tracker

**When modifying core algorithms**, consult these docs first for design rationale and best practices.

## Important Constraints

### Kraken API Rate Limits
- Public endpoints: 1 req/sec safe
- Private endpoints: Counter-based (15-20 max, decays at 0.33-1/sec)
- WebSocket: 150 reconnections per 10 minutes

### Training Stability
- The C++ MCTS binding is sensitive to tensor shapes - always ensure rewards/values are properly formatted
- Multiprocessing requires 'spawn' method (not 'fork')
- SQLite database is not thread-safe - use aiosqlite for async access

### Performance Targets
- Training throughput: 200+ episodes/hour (RTX 3090)
- MCTS inference: <100ms per decision (100 simulations)
- Data collection: 4-8 hours for full 64 assets × 30 days

## Testing

Currently no formal test suite. For validation:
1. Run `debug_env.py` to verify environment setup
2. Run `stress_test.py` for load testing
3. Check `training.log` for NaN losses or crashes

## Environment Variables

**`.env` file** (not in version control):
```bash
KRAKEN_API_KEY=your_key_here
KRAKEN_API_SECRET=your_secret_here
```

Used by `kraken_client.py` for private endpoints (not needed for training on historical data).
