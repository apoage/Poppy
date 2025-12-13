# JAX MuZero Trading Agent

A high-performance cryptocurrency trading agent powered by **JAX**, **MuZero**, and a **Tiny Recursive Model (TRM)** architecture. It supports multi-asset trading on Kraken (Spot & Margin) with sub-second inference times.

---

## 🚀 Setup & Installation

### 1. Prerequisites
- Linux Environment
- Python 3.10+
- Kraken API Keys (for live data/trading)

### 2. Installation
```bash
# Create and activate virtual environment
python3 -m venv trader_env
source trader_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. API Key Configuration
To enable live data fetching and trading, you must configure your Kraken API keys.

1.  Open `config/real_trade_config.py`.
2.  Edit the `API_KEY` and `API_SECRET` fields, or better yet, set them as environment variables:
    ```bash
    export KRAKEN_API_KEY="your_api_key"
    export KRAKEN_API_SECRET="your_api_secret"
    ```

---

## 🧠 Training Models

We support training specialized models for different market conditions (Spot vs. Margin, High Fee vs. Low Fee).

### Option A: The Manager (Recommended)
Run a curated curriculum of experiments sequentially:
```bash
python train_manager.py
```
*   Logs: `logs/train_manager.log` + `logs/training_fast_{name}.log`
*   Checkpoints: `jax_ckpt_fast/{experiment_name}/`

### Option B: Manual Training
Train a specific model configuration manually:
```bash
# Spot Mode (No Shorting), 26bps Fees
python train_jax_fast.py --name spot_v1 --mode SPOT --fee 26

# Margin Mode, Zero Fees (Research)
python train_jax_fast.py --name research_v1 --mode MARGIN --fee 0
```

---

## 📉 Paper Trading

Simulate trading with live Kraken data without risking real funds.

```bash
# Standard Spot Mode (Default)
python paper_trade_jax.py

# Margin Mode (Allows Shorting)
python paper_trade_jax.py --margin
```
*   **Logs:** `logs/paper_trading.log`
*   **Trade History:** `logs/trade_log.csv`
*   **Features:**
    *   Uses live WebSocket data.
    *   Simulates execution lag and fees.
    *   `Ctrl+C` triggers a graceful shutdown and performance report.

---

## 💸 Real Trading

**⚠️ WARNING: This script trades REAL MONEY.**

### Safety Features
The `RealPortfolioManager` includes several critical safety mechanisms:
1.  **Exchange-Side Stop-Loss:** Every entry order is immediately followed by a Stop-Loss order on Kraken.
2.  **Cash Reserve:** 50% of capital is strictly reserved (hardcoded safety buffer).
3.  **Liquidity Sizing:** Position sizes are calculated based on *available* capital, not total equity.
4.  **Dry Run Mode:** Simulates API calls without executing them (enabled by default in config).

### running Live
1.  **Verify Config:** Check `config/real_trade_config.py`.
    *   Set `DRY_RUN = True` for a safe test.
    *   Set `DRY_RUN = False` for actual trading.
    *   Set `TRADING_MODE` ('SPOT' or 'MARGIN').
2.  **Launch:**
    ```bash
    python real_trade_jax.py
    ```
3.  **Emergency Stop:** Press `Ctrl+C` to stop the bot. It will attempt to close open positions (configurable).

---

## 📊 Analysis Tools

### Replay Trades
Re-calculate PnL from logs with accurate historical FX rates (crucial for cross-currency pairs like ETH/BTC or LINK/EUR).

```bash
# Replay the default log
python tools/replay_trades.py

# Replay a specific log with custom initial capital
python tools/replay_trades.py logs/my_trade_log.csv 1000.0
```
*   **Legacy Logs:** Use `--legacy` flag for logs created before Dec 11, 2025.

---

## 📂 Project Structure

```
Trader/
├── train_manager.py       # Orchestrator for multiple training experiments
├── train_jax_fast.py      # Core training script (JAX/MCTS)
├── paper_trade_jax.py     # Paper trading simulator
├── real_trade_jax.py      # Live trading bot
├── requirements.txt       # Python dependencies
├── logs/                  # Centralized log directory
├── jax_ckpt_fast/         # Model checkpoints
├── config/                # Configuration files
│   └── real_trade_config.py
├── mcts_trader/           # Core Library
│   ├── jax_agent/         # MuZero implementation (Network, Trainer, MCTS)
│   ├── agent/             # Portfolio logic (Sim & Real)
│   ├── data/              # Kraken API, WebSocket, Database
│   └── encoder/           # State representation
└── tools/                 # Utilities (Replay, Debug, Profiling)
```

## 📜 License
Private / Proprietary.