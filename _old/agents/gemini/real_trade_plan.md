# Real Trading Implementation Plan (`real_trade_jax.py`)

## Core Philosophy
- **Source of Truth**: Kraken API. Local state is a cache, refreshed frequently.
- **Execution**: Smart routing based on model confidence (Market vs. Limit).
- **Safety**: Robust error handling, Kill Switch (Ctrl+C), Configurable Limits.
- **Mode**: Spot by default, Margin optional (required for shorts).

## Architecture

### 1. Configuration (`config/real_trade_config.py` or `.yaml`)
Instead of hardcoding, we centralize settings:
- **API Keys**: Loaded from env vars `KRAKEN_API_KEY`, `KRAKEN_API_SECRET`.
- **Trading Mode**: `SPOT` vs `MARGIN`.
- **Execution**:
    - `market_order_threshold`: Confidence > 0.8 (example).
    - `limit_order_timeout`: 60 seconds (cancel if not filled).
- **Risk**:
    - `max_position_pct`: 10% (default).
    - `hard_stop_drawdown`: None (default).
    - `time_limit_hours`: 6 (default).
- **Assets**: List of tradeable pairs.

### 2. `RealPortfolioManager` (Adapter Pattern)
Replaces `PortfolioManager` but adapts to Kraken API.

#### Methods:
- **`sync()`**:
    - Fetches `Balance` (Cash/Equity).
    - Fetches `OpenPositions` (Margin).
    - Fetches `OpenOrders` (Working limits).
    - Updates local `self.state`.
- **`get_valid_actions()`**:
    - Checks `Balance` > `MinOrderSize`.
    - Checks `MarginLevel` (if margin).
- **`execute_actions(actions, confidences)`**:
    - **Logic**:
        - `if action == BUY and confidence > threshold`: Send `Market Buy`.
        - `if action == BUY and confidence < threshold`: Send `Limit Buy @ Best Bid`.
    - **Margin Handling**:
        - If `mode=MARGIN`: Add `leverage=...` to API call.
        - If `mode=SPOT` and `action=SELL`: Check if we hold the asset. If not -> Ignore (or error).
- **`close_all_positions()`**:
    - Cancels all open orders (`CancelAll`).
    - Iterates open positions/balances -> Sends `Market Close` / `Market Sell`.

### 3. Smart Execution Logic
- **Market Orders**: Immediate execution.
- **Limit Orders**:
    - Place at `Best Bid` (Buy) or `Best Ask` (Sell).
    - Track `order_id`.
    - **Supervisory Loop**: Check status. If not filled in `N` seconds -> Cancel & Ignore (or Retry as Market?).
- **Rejection Handling**:
    - `InsufficientFunds`: Log error, maybe reduce size and retry once?
    - `RateLimitExceeded`: exponential backoff is already in `KrakenClient`, but we need to handle it at app level too.

### 4. `real_trade_jax.py` (Main Loop)
Structure similar to `paper_trade_jax.py` but async is mandatory.

```python
async def trade_step():
    # 1. Sync State (Critical!)
    await portfolio.sync()

    # 2. Inference
    action, confidence = model.predict(obs)

    # 3. Execute
    await portfolio.execute(action, confidence)
```

### 5. Kill Switch (Ctrl+C)
- Catches `KeyboardInterrupt`.
- **Action**:
    1. `client.cancel_all_orders()`
    2. `portfolio.close_all_positions()` (Sell all spot, Close all margin).
    3. `print_report()`
    4. `exit()`

## Todo List

1. [ ] **Audit `kraken_client.py`**: Ensure support for:
    - `add_order` (market, limit, leverage flags).
    - `cancel_all_orders`.
    - `get_open_positions`.
    - `get_account_balance`.
2. [ ] **Create `RealPortfolioManager`**: The heavy lifter.
3. [ ] **Create `real_trade_config.py`**: Configuration structure.
4. [ ] **Create `real_trade_jax.py`**: The main script.
5. [ ] **Dry Run Test**: Run with `validate=True` (Kraken API flag for testing orders without execution).

## questions for User
- **Limit Order Management**: If a limit order isn't filled after timeout, should we:
  a) Cancel it and miss the trade?
  b) Convert it to Market Order? (Chase the price)
  c) Update the limit price? (Chase the bid/ask)
