# MCTS Trader v1.0 - "First Light"

JAX MuZero Trading Agent with Tiny Recursive Model architecture.

## Quick Start

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run training
python train_jax_fast.py

# Run paper trading
python paper_trade_jax.py
```

## Directory Structure

```
v1.0/
├── mcts_trader/           # Core library
│   ├── jax_agent/         # JAX MuZero implementation
│   │   ├── muzero_model.py
│   │   ├── trm_flax.py    # TRM architecture
│   │   ├── mctx_search.py # MCTS with mctx
│   │   ├── trainer.py
│   │   └── replay_buffer.py
│   ├── encoder/
│   │   └── state_fast.py  # Fast 27-channel encoder
│   ├── data/
│   │   ├── history.py     # Historical data replay
│   │   ├── stream.py      # WebSocket streaming
│   │   └── kraken_client.py
│   └── agent/
│       └── portfolio.py   # Position management
├── checkpoints/           # Trained model weights
├── train_jax_fast.py      # Training script
├── paper_trade_jax.py     # Paper trading script
└── requirements.txt
```

## Model Architecture

- **Encoder**: Lightweight CNN (27 channels → 64-dim state)
- **TRM Reasoning**: Recursive refinement (H=3, L=4 cycles)
- **MCTS**: Gumbel MuZero with 16-32 simulations
- **Action Space**: 192 actions (64 assets × 3: sell/hold/buy)

## Configuration

Key parameters in `train_jax_fast.py`:
- `N_ASSETS = 64`
- `LOOKBACK = 60` (minutes)
- `SIMULATIONS = 16` (MCTS)
- `BATCH_SIZE = 256`

## Requirements

- Python 3.10+
- JAX with CUDA support
- ~20GB disk for trading_data.db
- GPU with 8GB+ VRAM recommended
