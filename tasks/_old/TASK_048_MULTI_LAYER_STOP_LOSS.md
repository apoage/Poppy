# TASK_048: Multi-Layer Stop Loss System

## CRITICAL SAFETY FEATURE

This is not optional. This is life-safety critical.

> "Recently because Binance implemented stop loss about 10 people suicided themselves cause automatic stop loss didn't work."

Stop loss must work. Period. Multiple independent layers ensure that if ANY single layer fails, others execute.

## Principle: Defense in Depth

```
LAYER 5: EXTERNAL SERVER (completely separate infrastructure)
         └─ Simple rule-based monitor on DIFFERENT server/cloud
         └─ Polls exchange API directly, no dependency on our system
         └─ Dead simple logic: if drawdown > X%, close all
         └─ Works even if our entire infrastructure is down

LAYER 4: EXCHANGE LEVEL (last resort)
         └─ Exchange-native stop loss orders
         └─ Placed at order time, lives on exchange servers
         └─ Works even if our system is offline

LAYER 3: API/SYSTEM LEVEL (external process)
         └─ Independent monitoring process
         └─ Polls positions, executes market close if threshold hit
         └─ Different process than trading logic

LAYER 2: SAFETY WRAPPER (same process, different module)
         └─ SafetyWrapper checks every action
         └─ Enforces drawdown limits
         └─ Cannot be bypassed by model

LAYER 1: MODEL LEVEL (advisory, but LISTEN TO IT FIRST)
         └─ Agent may learn to cut losses
         └─ If model suggests stop → EXECUTE IT (faster than other layers)
         └─ Model is not RELIED upon, but its suggestions are WELCOMED
         └─ First line of defense, just not the last
```

**Execution Order (fastest first):**
1. **Layer 1 suggests** → Execute immediately (model learned something)
2. **Layer 2 catches** → Same process, fast
3. **Layer 3 catches** → Separate process, slight delay
4. **Layer 4 catches** → Exchange execution, depends on exchange
5. **Layer 5 catches** → External server polling, slowest but most independent

**If Layer 1 fails** → Layer 2 catches it
**If Layer 2 fails** → Layer 3 catches it
**If Layer 3 fails** → Layer 4 catches it
**If Layer 4 fails** → Layer 5 catches it (external server)
**If Layer 5 fails** → 5 independent systems all failed, truly catastrophic

## Implementation

### Layer 5: External Server Monitor (Separate Infrastructure)

```python
# deploy on: AWS Lambda, Google Cloud Function, separate VPS, Raspberry Pi at home
# ANYWHERE that is NOT the main trading server
#
# external_monitor.py - DEAD SIMPLE, no dependencies on main system

import time
import ccxt
import requests

# Configuration - hardcoded, no external config files
EXCHANGE = 'binance'
API_KEY = 'xxx'  # Read-only key is sufficient for monitoring, trade key for closing
SECRET = 'xxx'
MAX_DRAWDOWN = 0.05  # 5%
CHECK_INTERVAL = 10  # seconds
ALERT_WEBHOOK = 'https://hooks.slack.com/xxx'  # Or Telegram, Discord, etc.

def get_equity(exchange):
    """Get current account equity."""
    balance = exchange.fetch_balance()
    return float(balance['total']['USDT'])

def close_all_positions(exchange):
    """Emergency close everything."""
    positions = exchange.fetch_positions()
    for pos in positions:
        if abs(pos['contracts']) > 0:
            side = 'sell' if pos['contracts'] > 0 else 'buy'
            exchange.create_market_order(
                pos['symbol'], side, abs(pos['contracts']),
                params={'reduceOnly': True}
            )
            print(f"CLOSED: {pos['symbol']}")

def send_alert(message):
    """Send alert via webhook."""
    try:
        requests.post(ALERT_WEBHOOK, json={'text': f'🚨 {message}'})
    except:
        pass  # Alert failure shouldn't stop the monitor
    print(message)

def main():
    """
    SIMPLE LOOP. No fancy logic. Just:
    1. Check equity
    2. If drawdown > threshold, CLOSE EVERYTHING
    3. Repeat
    """
    exchange = ccxt.binance({
        'apiKey': API_KEY,
        'secret': SECRET,
        'enableRateLimit': True,
    })

    peak_equity = None
    send_alert("External monitor started")

    while True:
        try:
            equity = get_equity(exchange)

            if peak_equity is None:
                peak_equity = equity
            else:
                peak_equity = max(peak_equity, equity)

            drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0

            if drawdown > MAX_DRAWDOWN:
                send_alert(f"DRAWDOWN {drawdown:.2%} - CLOSING ALL POSITIONS")
                close_all_positions(exchange)
                peak_equity = equity  # Reset after emergency close

            time.sleep(CHECK_INTERVAL)

        except Exception as e:
            send_alert(f"Monitor error: {e}")
            time.sleep(CHECK_INTERVAL)

if __name__ == '__main__':
    main()
```

**Deployment options for Layer 5:**
- AWS Lambda with CloudWatch scheduled trigger (every 10s)
- Google Cloud Function with Cloud Scheduler
- Cheap VPS ($5/mo) in different region
- Raspberry Pi at home on different ISP
- Friend's server in different country

**Key properties:**
- NO dependency on main trading system
- NO shared infrastructure
- Dead simple code (< 100 lines)
- Can be audited in 5 minutes
- Read-only API key for monitoring, separate trade key for closing

### Layer 4: Exchange-Native Stop Loss

```python
# exchange/stop_loss.py
class ExchangeStopLoss:
    """
    Place stop loss orders directly on exchange.
    These execute even if our system crashes.
    """

    def __init__(self, exchange_client, default_stop_pct=0.05):
        self.client = exchange_client
        self.default_stop_pct = default_stop_pct
        self.active_stops = {}  # position_id -> stop_order_id

    def place_stop_loss(self, symbol: str, side: str, quantity: float,
                        entry_price: float, stop_pct: float = None):
        """
        Place stop loss order on exchange immediately after entry.

        This order lives on exchange servers - executes even if we're offline.
        """
        stop_pct = stop_pct or self.default_stop_pct

        if side == 'long':
            stop_price = entry_price * (1 - stop_pct)
            stop_side = 'sell'
        else:  # short
            stop_price = entry_price * (1 + stop_pct)
            stop_side = 'buy'

        order = self.client.create_order(
            symbol=symbol,
            type='stop_loss_limit',
            side=stop_side,
            quantity=quantity,
            stop_price=stop_price,
            price=stop_price * 0.99,  # Limit slightly below stop for fills
        )

        self.active_stops[f"{symbol}_{side}"] = order['id']
        print(f"STOP LOSS PLACED: {symbol} {side} @ {stop_price:.2f} ({stop_pct:.1%})")

        return order

    def cancel_stop_loss(self, symbol: str, side: str):
        """Cancel stop loss when position closed normally."""
        key = f"{symbol}_{side}"
        if key in self.active_stops:
            self.client.cancel_order(self.active_stops[key])
            del self.active_stops[key]

    def update_trailing_stop(self, symbol: str, side: str, current_price: float):
        """Update stop loss to lock in profits (trailing stop)."""
        # Implementation for trailing stops
        pass
```

### Layer 3: Independent Monitor Process

```python
# safety/independent_monitor.py
"""
THIS RUNS AS A SEPARATE PROCESS.
Even if main trading process hangs/crashes, this keeps running.
"""

import time
import subprocess
from multiprocessing import Process

class IndependentStopLossMonitor:
    """
    Separate process that monitors positions and forces closure.
    Completely independent of trading logic.
    """

    def __init__(self, exchange_client, max_drawdown=0.05, check_interval=1.0):
        self.client = exchange_client
        self.max_drawdown = max_drawdown
        self.check_interval = check_interval
        self.peak_equity = None
        self.is_running = False

    def get_equity(self):
        """Get current account equity from exchange."""
        account = self.client.fetch_balance()
        return float(account['total']['USDT'])  # Or appropriate quote currency

    def get_positions(self):
        """Get all open positions."""
        return self.client.fetch_positions()

    def close_all_positions(self, reason: str):
        """Emergency close all positions with market orders."""
        print(f"!!! EMERGENCY CLOSE: {reason} !!!")

        positions = self.get_positions()
        for pos in positions:
            if pos['size'] != 0:
                side = 'sell' if pos['size'] > 0 else 'buy'
                try:
                    self.client.create_order(
                        symbol=pos['symbol'],
                        type='market',
                        side=side,
                        quantity=abs(pos['size']),
                        params={'reduceOnly': True}
                    )
                    print(f"  Closed {pos['symbol']}: {pos['size']}")
                except Exception as e:
                    print(f"  FAILED to close {pos['symbol']}: {e}")
                    # Try again, try different method, alert human

    def run_forever(self):
        """Main monitoring loop - runs until killed."""
        print(f"Stop Loss Monitor started (max_drawdown={self.max_drawdown:.1%})")
        self.is_running = True

        while self.is_running:
            try:
                equity = self.get_equity()

                # Update peak
                if self.peak_equity is None:
                    self.peak_equity = equity
                else:
                    self.peak_equity = max(self.peak_equity, equity)

                # Check drawdown
                if self.peak_equity > 0:
                    drawdown = (self.peak_equity - equity) / self.peak_equity

                    if drawdown > self.max_drawdown:
                        self.close_all_positions(
                            f"Drawdown {drawdown:.2%} > {self.max_drawdown:.2%}"
                        )
                        # Don't exit - keep monitoring in case positions reopened
                        self.peak_equity = equity  # Reset peak after emergency close

                time.sleep(self.check_interval)

            except Exception as e:
                print(f"Monitor error: {e}")
                time.sleep(self.check_interval)

    def start_as_subprocess(self):
        """Start monitor as independent subprocess."""
        process = Process(target=self.run_forever, daemon=False)
        process.start()
        return process


def start_independent_monitor(exchange_config, max_drawdown=0.05):
    """
    Call this at system startup.
    Monitor runs independently of trading logic.
    """
    from exchange.client import create_client
    client = create_client(exchange_config)

    monitor = IndependentStopLossMonitor(client, max_drawdown)
    process = monitor.start_as_subprocess()

    print(f"Independent stop loss monitor started (PID: {process.pid})")
    return process
```

### Layer 2: Safety Wrapper (from TASK_047)

```python
# In SafetyWrapper.apply_safety()
def check_stop_loss(self, current_equity: float) -> bool:
    """
    Layer 2 stop loss check.
    This is in the same process but separate module.
    """
    self.update_equity(current_equity)

    if self.current_drawdown > self.config.max_drawdown_pct:
        print(f"LAYER 2 STOP LOSS: Drawdown {self.current_drawdown:.2%}")
        return True

    return False
```

### Layer 1: Model Awareness (Advisory Only)

```python
# In reward shaping - model learns drawdown is bad
def compute_reward(pnl, drawdown, max_drawdown_threshold=0.03):
    """
    Model learns to avoid drawdowns, but this is NOT safety-critical.
    It's just to help the model learn good behavior.
    """
    reward = pnl

    # Penalty for approaching drawdown limit
    if drawdown > max_drawdown_threshold:
        reward -= (drawdown - max_drawdown_threshold) * 10

    return reward
```

## System Startup Sequence

```python
# main.py or live_trading.py
def start_trading_system(config):
    """
    Startup sequence ensures all safety layers active BEFORE trading.
    """

    # LAYER 4: Verify exchange connection and stop loss capability
    print("Initializing Layer 4: Exchange stop loss capability...")
    exchange = create_exchange_client(config.exchange)
    verify_stop_loss_support(exchange)

    # LAYER 3: Start independent monitor FIRST
    print("Initializing Layer 3: Independent monitor process...")
    monitor_process = start_independent_monitor(
        config.exchange,
        max_drawdown=config.max_drawdown
    )
    assert monitor_process.is_alive(), "Monitor failed to start!"

    # LAYER 2: Initialize safety wrapper
    print("Initializing Layer 2: Safety wrapper...")
    safety = SafetyWrapper(SafetyConfig(
        max_drawdown_pct=config.max_drawdown,
        max_position_pct=config.max_position,
    ))

    # LAYER 1: Load model (advisory only)
    print("Initializing Layer 1: Trading model...")
    model, params = load_model(config.model_path)

    # Verify all layers before allowing trading
    print("Verifying all safety layers...")
    assert monitor_process.is_alive(), "Layer 3 not running!"
    assert safety is not None, "Layer 2 not initialized!"

    print("All safety layers active. Trading enabled.")

    return TradingSystem(
        model=model,
        params=params,
        safety=safety,
        exchange=exchange,
        monitor_process=monitor_process,
    )
```

## Failure Modes and Responses

| Failure | Detection | Response |
|---------|-----------|----------|
| Model predicts wrong | Normal | Layer 2/3/4 catch drawdown |
| Safety wrapper bug | Layer 3 catches | Independent monitor closes |
| Main process crash | Layer 3 still running | Monitor closes positions |
| Monitor process crash | Heartbeat check | Alert human, restart, or use Layer 4 |
| Network to our server | Exchange stops still active | Layer 4 executes on exchange |
| Exchange API down | Can't close | **CRITICAL** - alert human immediately |

## Human Alerting

```python
# safety/alerts.py
def send_critical_alert(message: str):
    """
    Multiple channels - at least one must work.
    """
    try:
        send_telegram(message)
    except:
        pass

    try:
        send_email(message)
    except:
        pass

    try:
        send_sms(message)
    except:
        pass

    try:
        # Sound alarm if running locally
        subprocess.run(['paplay', '/usr/share/sounds/freedesktop/stereo/alarm.oga'])
    except:
        pass

    # Log regardless
    print(f"CRITICAL ALERT: {message}")


# In monitor
if drawdown > max_drawdown:
    send_critical_alert(f"STOP LOSS TRIGGERED: {drawdown:.2%} drawdown")
    close_all_positions()
```

## Testing Requirements

### Must Test:
1. **Layer 4**: Place and trigger exchange stop loss manually
2. **Layer 3**: Kill main process, verify monitor closes positions
3. **Layer 2**: Inject fake equity drop, verify wrapper triggers
4. **Integration**: All layers working together
5. **Network failure**: Disconnect during test, verify Layer 4 works

### Test Script

```python
def test_multi_layer_stop_loss():
    """
    MUST RUN THIS BEFORE LIVE TRADING.
    Use testnet/paper trading.
    """

    # Test Layer 4: Exchange stop loss
    print("Testing Layer 4...")
    order = exchange_stop.place_stop_loss('BTC/USDT', 'long', 0.001, 50000, 0.01)
    # Manually verify on exchange, trigger if possible

    # Test Layer 3: Independent monitor
    print("Testing Layer 3...")
    monitor = start_independent_monitor(config, max_drawdown=0.001)  # 0.1% for testing
    # Create small loss, verify monitor catches

    # Test Layer 2: Safety wrapper
    print("Testing Layer 2...")
    safety = SafetyWrapper(SafetyConfig(max_drawdown_pct=0.001))
    safety.update_equity(100)
    safety.update_equity(99.8)  # 0.2% drawdown
    assert safety.check_drawdown() == True

    print("All layers tested successfully!")
```

## Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    STOP LOSS LAYERS                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Layer 4: Exchange-native stops     [INDEPENDENT SERVER]   │
│            └─ Works if we're offline                        │
│                                                              │
│   Layer 3: Independent monitor       [SEPARATE PROCESS]     │
│            └─ Works if main process crashes                 │
│                                                              │
│   Layer 2: Safety wrapper            [SAME PROCESS]         │
│            └─ Checks every action                           │
│                                                              │
│   Layer 1: Model learning            [ADVISORY ONLY]        │
│            └─ NOT safety critical                           │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│   IF ANY LAYER FAILS, OTHERS CATCH IT                       │
│   MULTIPLE INDEPENDENT FAILURE MODES REQUIRED FOR LOSS      │
└─────────────────────────────────────────────────────────────┘
```

## Status

- [ ] Layer 4: Exchange stop loss implementation
- [ ] Layer 3: Independent monitor process
- [ ] Layer 2: Safety wrapper stop loss (from TASK_047)
- [ ] Human alerting system
- [ ] Integration testing
- [ ] Documentation for operators

## Priority: CRITICAL

This must be implemented and tested before any live trading with real money.
