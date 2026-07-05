# MCTS Trader - Compact Generation Prompt (Quick Start)

## Use This When You Want Fast Results

Copy this entire prompt to Gemini CLI for a minimal working system.

---

```
Generate a minimal but complete MCTS trading agent for cryptocurrency trading.

CONSTRAINTS:
- Hardware: 1x RTX 3090 (24GB), 48GB RAM, Ryzen 5800X
- Exchange: Kraken API
- Language: Python 3.10+
- Framework: PyTorch 2.0+

Generate these 5 core files with complete, runnable code:

=== FILE 1: config.py ===
Dataclass configurations for:
- ModelConfig: backbone_channels=[64,128,256], num_actions=5, dropout=0.2
- MCTSConfig: num_simulations=100, c_puct=1.5, discount=0.99
- TrainingConfig: batch_size=128, lr=2e-4, mixed_precision=True
- TradingConfig: initial_balance=10000, transaction_cost=0.001

=== FILE 2: model.py ===
TradingNetwork class with:
- ResNet-style CNN backbone (3 res blocks)
- Policy head: outputs 5 action probabilities
- Value head: outputs scalar in [-1, 1]
- Mixed precision support
- forward() returns (policy, value) tuple

=== FILE 3: mcts.py ===
MCTSSearch class with:
- MCTSNode dataclass (state, prior, visits, value_sum, children)
- PUCT selection formula
- Dirichlet noise at root
- search() method returning action probabilities dict
- Batched neural network evaluation

=== FILE 4: environment.py ===
TradingEnv (Gymnasium compatible) with:
- State: HSV-encoded market data (64x64x3)
- Actions: [STRONG_SELL, SELL, HOLD, BUY, STRONG_BUY]
- step() with transaction costs and slippage
- Reward: log return * position - costs
- Episode ends at data end or 50% drawdown

=== FILE 5: train.py ===
Complete training script with:
- Data loading from Kraken CSV
- HSV state encoding function
- Self-play episode generation
- Replay buffer with random sampling
- Training loop with mixed precision
- Checkpoint saving every 50 epochs
- Basic logging to console

CRITICAL REQUIREMENTS:
1. Use torch.cuda.amp for mixed precision
2. Batch size 128 for GPU efficiency
3. Include docstrings and type hints
4. Handle edge cases (empty data, NaN values)
5. Print GPU memory usage during training

The code must run with:
```bash
python train.py --data data/XXBTZUSD_5m.csv --epochs 100
```

Start generating now. Output each file completely before moving to the next.
```

---

## After Initial Generation

Once you have the 5 core files working, use these follow-up prompts:

### Add Kraken Data Fetcher
```
Add a kraken_data.py file that:
1. Downloads historical trades from Kraken REST API
2. Resamples to OHLCV at any timeframe
3. Handles pagination (since parameter)
4. Saves to Parquet format
5. Rate limiting (1 req/sec)

Include async support with aiohttp.
```

### Add Paper Trading
```
Add a paper_trader.py file that:
1. Connects to Kraken WebSocket for real-time OHLC
2. Maintains rolling window of candles
3. Runs MCTS search on each new candle
4. Tracks paper portfolio value
5. Logs all trades to CSV

Use asyncio for the main loop.
```

### Add Backtesting
```
Add a backtest.py file that:
1. Loads trained model checkpoint
2. Runs inference (no MCTS) on test data
3. Calculates: Sharpe, MaxDD, Win Rate
4. Plots equity curve with matplotlib
5. Compares to buy-and-hold baseline
```

### Add Distributed Training (4 GPU)
```
Modify train.py to support multi-GPU training:
1. Use PyTorch DistributedDataParallel
2. Spawn 4 processes (one per GPU)
3. Gradient synchronization with NCCL
4. Proper checkpoint saving from rank 0
5. Scale batch size by world_size

Command: torchrun --nproc_per_node=4 train.py
```

---

## Testing Checklist

After generation, verify:

- [ ] `python -c "from config import *; print('Config OK')"` works
- [ ] `python -c "from model import TradingNetwork; m = TradingNetwork(); print(m)"` works
- [ ] `python -c "import torch; x = torch.randn(1,3,64,64); m = TradingNetwork(); print(m(x))"` works
- [ ] `python train.py --data sample.csv --epochs 1` completes
- [ ] GPU utilization > 50% during training (check with `nvidia-smi`)
- [ ] No CUDA out of memory errors with batch_size=128

---

## Sample Data for Testing

Create a minimal CSV for testing:

```python
# generate_test_data.py
import pandas as pd
import numpy as np

dates = pd.date_range('2024-01-01', periods=10000, freq='5min')
price = 50000 + np.cumsum(np.random.randn(10000) * 100)

df = pd.DataFrame({
    'timestamp': dates,
    'open': price,
    'high': price * 1.001,
    'low': price * 0.999,
    'close': price,
    'volume': np.random.lognormal(10, 1, 10000)
})

df.to_csv('data/test_btc_5m.csv', index=False)
print(f"Generated {len(df)} candles")
```
