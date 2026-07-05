# Double Safeguard Stop Loss System

**Created**: Dec 20, 2025
**Priority**: 🚨 CRITICAL - Safety First
**Status**: Design Spec (Not Yet Implemented)

---

## Core Principle: Never Trust AI Alone with Capital

**Problem**: AI agents can fail, make mistakes, or learn degenerate strategies.

**Solution**: **TWO independent stop loss systems** - whichever triggers first saves capital.

```
Agent Stop Loss  ──┐
                   ├──> FIRST TO TRIGGER WINS ──> Position Closed
Algorithmic Stop ──┘
```

**Result**: If either system detects danger, position is closed. **Double safety.**

---

## The Two Systems

### System 1: Agent-Learned Stop Loss (Trained)

**What**: Agent learns to set its own stop losses during training.

**How it works**:
```python
# Agent decides: "I'll exit BTC if it drops below $48k"
position = Position(
    symbol='BTC/USD',
    entry_price=50000,
    stop_loss=48000,  # Agent's decision (-4%)
    ...
)

# Every step, check if triggered
if position.should_stop_loss():
    close_position(position)
```

**Advantages**:
- Adapts to market conditions (volatile markets = wider stops)
- Learns from experience (what worked in past)
- Can use complex signals (correlations, momentum, etc.)

**Risks**:
- ❌ Might set stop too wide (90% loss before triggering!)
- ❌ Might never set stops (degenerate policy)
- ❌ Might set stop too tight (stop loss hunting itself)
- ❌ Model bugs, NaN outputs, etc.

**Verdict**: **Good for optimization, BAD as sole safety mechanism.**

---

### System 2: Algorithmic Stop Loss (Hard Limit)

**What**: Rule-based safety net that **always** runs in production.

**How it works**:
```python
# HARD LIMITS (not learned, hardcoded)
MAX_POSITION_LOSS_PERCENT = 5.0  # Never lose more than 5% on one position
MAX_PORTFOLIO_LOSS_PERCENT = 10.0  # Never lose more than 10% total
MAX_POSITION_AGE_HOURS = 24  # Force close stale positions

class AlgorithmicStopLoss:
    def check_position(self, position: Position) -> bool:
        """Returns True if position should be force-closed"""

        # Stop 1: Position loss exceeds 5%
        if position.pnl_percent <= -MAX_POSITION_LOSS_PERCENT:
            log_critical(f"HARD STOP: {position.symbol} lost {position.pnl_percent:.1f}%")
            return True

        # Stop 2: Position held too long (might be stuck/broken)
        age_hours = (current_time - position.entry_time) / 3600
        if age_hours > MAX_POSITION_AGE_HOURS:
            log_warning(f"STALE POSITION: {position.symbol} held {age_hours:.1f}h")
            return True

        # Stop 3: Extreme volatility (flash crash protection)
        recent_volatility = calculate_volatility(position.symbol, window=10)
        if recent_volatility > 0.50:  # 50% intraday swing
            log_critical(f"FLASH CRASH: {position.symbol} volatility {recent_volatility:.1%}")
            return True

        return False

    def check_portfolio(self, portfolio: PortfolioState) -> List[Position]:
        """Returns positions to force-close for portfolio protection"""
        total_pnl_percent = (portfolio.total_value - portfolio.initial_capital) / portfolio.initial_capital * 100

        # Portfolio-level stop: Down 10%
        if total_pnl_percent <= -MAX_PORTFOLIO_LOSS_PERCENT:
            log_critical(f"PORTFOLIO STOP: Down {total_pnl_percent:.1f}%, closing ALL positions")
            return list(portfolio.positions.values())  # Close everything

        return []
```

**Advantages**:
- ✅ Always active (can't be disabled by model)
- ✅ Simple, testable, auditable
- ✅ Protects against catastrophic failures
- ✅ Works even if model outputs NaNs or crashes

**Disadvantages**:
- Less adaptive (same 5% stop in calm and volatile markets)
- Might trigger unnecessarily if model would have recovered

**Verdict**: **Essential safety net. Non-negotiable.**

---

## Implementation Architecture

### Two-Layer Check (Every Step)

```python
class SafePortfolioManager(PortfolioManager):
    def __init__(self, ...):
        super().__init__(...)
        self.algorithmic_stop = AlgorithmicStopLoss()

    def update_from_market(self, market_data: Dict):
        """Called every step to update positions"""
        super().update_from_market(market_data)

        # === LAYER 1: Agent's Learned Stops ===
        agent_triggered = []
        for idx, pos in list(self.state.positions.items()):
            if pos.should_stop_loss() or pos.should_take_profit():
                agent_triggered.append(idx)
                log_info(f"AGENT STOP: Closing {pos.symbol} (agent's stop @ {pos.stop_loss})")

        # === LAYER 2: Algorithmic Hard Stops ===
        algo_triggered = []
        for idx, pos in list(self.state.positions.items()):
            if self.algorithmic_stop.check_position(pos):
                algo_triggered.append(idx)
                log_critical(f"ALGO STOP: Force closing {pos.symbol} (safety override)")

        # === LAYER 3: Portfolio-Level Stops ===
        portfolio_triggered = self.algorithmic_stop.check_portfolio(self.state)
        if portfolio_triggered:
            log_critical("PORTFOLIO CIRCUIT BREAKER: Closing ALL positions")
            algo_triggered.extend([p.asset_idx for p in portfolio_triggered])

        # Close all triggered positions (union of agent + algo)
        all_triggered = set(agent_triggered + algo_triggered)
        for idx in all_triggered:
            self._force_close_position(idx, reason="stop_loss")

        # Log which system saved us
        if algo_triggered and not agent_triggered:
            log_critical("⚠️ ALGO STOP TRIGGERED (Agent failed to protect capital!)")
        elif agent_triggered and not algo_triggered:
            log_info("✅ Agent stop triggered (before hard limit)")
        elif agent_triggered and algo_triggered:
            log_warning("⚠️ Both stops triggered (position deteriorated rapidly)")
```

---

## Safety Scenarios

### Scenario 1: Normal Operation (Agent Wins)

```
Entry: BTC @ $50,000
Agent sets stop_loss: $48,000 (-4%)
Hard limit: $47,500 (-5%)

Price drops to $48,100 → Agent closes position (-3.8% loss)
Hard limit never triggers → ✅ Agent did its job
```

### Scenario 2: Agent Failure (Algo Saves)

```
Entry: BTC @ $50,000
Agent sets stop_loss: $40,000 (-20% 😱 BAD AGENT!)
Hard limit: $47,500 (-5%)

Price drops to $47,600 → Hard limit triggers → Position closed (-4.8% loss)
Agent's stop never reached → 🚨 ALGO SAVED US from -20% loss
```

### Scenario 3: Flash Crash (Algo Saves)

```
Entry: BTC @ $50,000
Agent sets stop_loss: $48,000 (-4%)
Hard limit: $47,500 (-5%)

Flash crash: BTC drops to $40,000 in 1 minute (volatility spike)
Algo detects extreme volatility → Force close at $49,500 (-1% loss)
Agent's stop would have triggered at $48k (-4% loss)
Algo was faster → 🚨 SAVED 3% by detecting volatility spike early
```

### Scenario 4: Agent Goes Degenerate (Algo Saves)

```
Training bug: Agent learns to never set stop losses (stop_loss=0 always)
Agent opens 50 positions, none have stops
One position: ETH @ $2000 → drops to $1600 (-20%)

Agent does nothing (degenerate policy)
Hard limit triggers at -5% → Position closed at $1900 (-5% loss)
🚨 ALGO SAVED US from -20% loss
```

### Scenario 5: Portfolio Meltdown (Circuit Breaker)

```
Flash crash affects entire portfolio
BTC: -8%
ETH: -12%
SOL: -15%
Portfolio: -10.5% total

Portfolio-level circuit breaker triggers → CLOSE ALL POSITIONS
Agent doesn't even get a chance to react
🚨 CIRCUIT BREAKER prevents further losses
```

---

## Configuration (Production Settings)

```python
# config/safety_limits.py

# === POSITION-LEVEL LIMITS ===
MAX_POSITION_LOSS_PERCENT = 5.0      # Never lose >5% on one position
MAX_POSITION_LOSS_USD = 500.0        # Never lose >$500 on one position
MAX_POSITION_AGE_HOURS = 24          # Force close stale positions
MAX_POSITION_SIZE_PERCENT = 10.0     # Max 10% of capital in one asset

# === PORTFOLIO-LEVEL LIMITS ===
MAX_PORTFOLIO_LOSS_PERCENT = 10.0    # Daily loss limit (circuit breaker)
MAX_PORTFOLIO_DRAWDOWN_PERCENT = 15.0  # Peak-to-valley limit
MAX_OPEN_POSITIONS = 32              # Max 32/64 assets held simultaneously

# === VOLATILITY PROTECTION ===
MAX_INTRADAY_VOLATILITY = 0.50       # 50% swing triggers emergency exit
FLASH_CRASH_THRESHOLD = 0.10         # 10% drop in 1 minute = flash crash
CORRELATION_BREAKDOWN_THRESHOLD = 0.3  # If correlations go haywire, exit

# === EXECUTION LIMITS ===
MAX_SLIPPAGE_PERCENT = 2.0           # Cancel order if slippage >2%
MAX_ORDER_RETRIES = 3                # Retry failed orders max 3 times
ORDER_TIMEOUT_SECONDS = 30           # Cancel if order doesn't fill in 30s

# === OPERATIONAL LIMITS ===
TRADING_HOURS = (9, 17)              # Only trade 9 AM - 5 PM (prevent overnight gaps)
PAUSE_ON_EXCHANGE_ERROR = True       # Stop trading if exchange API errors
PAUSE_ON_MODEL_NAN = True            # Stop if model outputs NaN
```

---

## Implementation Checklist

### Phase 1: Core Safety (Critical)
- [ ] Implement `AlgorithmicStopLoss` class
- [ ] Add position-level loss check (-5% hard stop)
- [ ] Add portfolio-level circuit breaker (-10% total)
- [ ] Add position age check (24h stale protection)
- [ ] Test all scenarios with unit tests

### Phase 2: Advanced Protections
- [ ] Add volatility spike detection
- [ ] Add flash crash detection (>10% drop in 1 min)
- [ ] Add correlation breakdown detection
- [ ] Add slippage monitoring

### Phase 3: Integration
- [ ] Wrap `PortfolioManager` with safety layer
- [ ] Add logging for all triggers (agent vs algo)
- [ ] Add metrics dashboard (trigger frequency, saved losses)
- [ ] Add alerts (email/SMS when hard stop triggers)

### Phase 4: Training Integration
- [ ] Reward agent for setting good stops (before hard limit)
- [ ] Penalize agent when hard limit triggers (failed to protect)
- [ ] Add stop loss setting to action space (Level 5+)

---

## Monitoring & Alerts

### Metrics to Track

```python
# Daily safety report
{
    'agent_stops_triggered': 15,      # Agent closed 15 positions
    'algo_stops_triggered': 2,        # Hard limit saved us 2 times (🚨)
    'circuit_breaker_triggered': 0,   # No portfolio meltdowns (✅)
    'avg_agent_stop_loss': -3.2,      # Agent exits at -3.2% avg
    'avg_algo_stop_loss': -4.9,       # Hard limit at -4.9% when triggered
    'positions_saved_by_algo': 2,     # Algo prevented worse losses
    'estimated_losses_prevented': 450 # Algo saved $450 today
}
```

### Alert Triggers

**🚨 CRITICAL (Immediate SMS)**:
- Hard stop triggered >3 times in 1 hour (agent is broken)
- Circuit breaker triggered (portfolio meltdown)
- Agent never setting stops (degenerate policy detected)

**⚠️ WARNING (Email)**:
- Hard stop triggered (agent failed once)
- Volatility spike detected
- Position held >12 hours without action

**ℹ️ INFO (Dashboard)**:
- Agent stop triggered (normal operation)
- Position closed at profit target

---

## Testing Strategy

### Unit Tests

```python
def test_algo_stop_position_loss():
    """Hard limit triggers at -5% loss"""
    pos = Position(entry_price=100, current_price=95, ...)
    algo = AlgorithmicStopLoss()
    assert algo.check_position(pos) == True  # Should trigger

def test_algo_stop_portfolio_loss():
    """Circuit breaker at -10% portfolio loss"""
    portfolio = PortfolioState(initial_capital=10000, total_value=8900)
    algo = AlgorithmicStopLoss()
    triggered = algo.check_portfolio(portfolio)
    assert len(triggered) > 0  # Should close positions

def test_agent_wins_normal():
    """Agent stop triggers before hard limit"""
    pos = Position(entry_price=100, current_price=96, stop_loss=97, ...)
    assert pos.should_stop_loss() == True  # Agent triggers at -3%
    algo = AlgorithmicStopLoss()
    assert algo.check_position(pos) == False  # Hard limit doesn't trigger (>-5%)

def test_algo_wins_agent_failure():
    """Hard limit saves when agent fails to set stop"""
    pos = Position(entry_price=100, current_price=94, stop_loss=0, ...)
    assert pos.should_stop_loss() == False  # Agent didn't set stop
    algo = AlgorithmicStopLoss()
    assert algo.check_position(pos) == True  # Hard limit saves us
```

### Integration Tests

```python
def test_double_safeguard_scenario():
    """Full workflow: Agent failure → Algo saves"""
    portfolio = SafePortfolioManager(...)

    # Agent opens position without stop loss (bug)
    portfolio.buy(asset_idx=0, size=100, price=100)
    pos = portfolio.state.positions[0]
    assert pos.stop_loss == 0  # Agent forgot to set stop

    # Market crashes
    portfolio.update_from_market({'BTC': {'close': 94}})  # -6% loss

    # Algo should have force-closed position
    assert 0 not in portfolio.state.positions  # Position closed
    assert len(portfolio.get_recent_logs()) > 0  # Logged the event
    assert "ALGO STOP" in portfolio.get_recent_logs()[0]  # Algo triggered
```

---

## Training Implications

### Reward Shaping for Safety

```python
def calculate_reward(self, portfolio_change, stop_triggered_by):
    """Reward agent for safe trading"""

    base_reward = portfolio_change  # Normal % return

    # Bonus: Agent set good stop that triggered (protected capital)
    if stop_triggered_by == 'agent':
        safety_bonus = +0.5  # Small bonus for good risk management

    # Penalty: Hard limit triggered (agent failed to protect)
    elif stop_triggered_by == 'algo':
        safety_penalty = -1.0  # Strong penalty for safety failure
        base_reward += safety_penalty

    # Major penalty: Circuit breaker (catastrophic failure)
    elif stop_triggered_by == 'circuit_breaker':
        catastrophic_penalty = -10.0
        base_reward += catastrophic_penalty

    return base_reward
```

**Result**: Agent learns that:
1. Setting good stops → Small bonus
2. Forcing hard limit to trigger → Big penalty
3. Causing circuit breaker → Catastrophic penalty

Over time, agent learns to set stops conservatively to avoid penalties.

---

## Production Deployment Checklist

**Before enabling real money trading**:

- [ ] ✅ Algorithmic stop loss tested on 1000+ scenarios
- [ ] ✅ Circuit breaker tested (mock portfolio crash)
- [ ] ✅ All safety limits tuned for production capital
- [ ] ✅ Monitoring dashboard shows all triggers in real-time
- [ ] ✅ SMS alerts configured for critical events
- [ ] ✅ Backtested on 6 months of historical data
- [ ] ✅ Paper trading for 30 days without safety violations
- [ ] ✅ Legal review of risk management system
- [ ] ✅ Manual kill switch implemented (human override)
- [ ] ✅ Insurance/capital reserves for worst-case scenario

**Only proceed when ALL boxes checked. No exceptions.**

---

## Philosophy: Defense in Depth

**Layer 1**: Agent learns good strategies (optimal)
**Layer 2**: Agent sets stops (learned risk management)
**Layer 3**: Algorithmic hard stops (rule-based safety net)
**Layer 4**: Portfolio circuit breaker (last resort)
**Layer 5**: Human kill switch (manual override)

**Any single layer can fail. All layers together = robust system.**

```
┌─────────────────────────────────────┐
│ Layer 5: Human Kill Switch         │ ← Manual override
├─────────────────────────────────────┤
│ Layer 4: Circuit Breaker           │ ← -10% portfolio stop
├─────────────────────────────────────┤
│ Layer 3: Algorithmic Stop Loss     │ ← -5% position stop
├─────────────────────────────────────┤
│ Layer 2: Agent-Learned Stops        │ ← Model decides
├─────────────────────────────────────┤
│ Layer 1: Optimal Trading Strategy   │ ← Model trades
└─────────────────────────────────────┘
```

**Failure tolerance**: Any **4 of 5 layers can fail**, system still protects capital.

---

## Summary

**🚨 CRITICAL SAFETY REQUIREMENT**

**Two independent stop loss systems**:
1. **Agent-learned** (trained, adaptive, optimizes for profit)
2. **Algorithmic hard limit** (rule-based, always active, protects from catastrophe)

**Whichever triggers first wins → Position closed → Capital saved**

**Implementation**: Wrap `PortfolioManager` with `AlgorithmicStopLoss` checks every step

**Status**: **NOT YET IMPLEMENTED** - Design spec complete, awaiting implementation

**Priority**: **Implement before ANY real money trading** (paper trading OK without this)

**Testing**: Must pass 1000+ failure scenarios before production

**Philosophy**: **Never trust AI alone with capital. Always have a safety net.**

---

## References

- `lore/ADVANCED_ORDERS_DESIGN.md` - Limit/stop order implementation
- `mcts_trader/agent/portfolio.py` - Position class with stop_loss fields
- `config/trading_config.py` - Current config (needs safety_limits.py)

**Next Step**: Implement `AlgorithmicStopLoss` class before live trading.
