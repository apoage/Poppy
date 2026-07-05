# TASK_047: External Safety Wrapper (Execution Layer)

## Overview

Implement safety mechanisms **outside the learned model** at the API/execution layer. These are hard limits that cannot be overridden by the agent's decisions.

> "AlphaZero-style reasoning is exactly what you want — as long as reality always gets veto power."

## Motivation

The agent proposes, external limits dispose. No matter how confident the model is, certain constraints must be enforced externally:

- **Position limits**: Never exceed X% in single asset
- **Drawdown limits**: Stop trading if equity drops Y%
- **Exposure limits**: Total portfolio exposure capped
- **Rate limits**: Max trades per hour

These are NOT learned. They are hard-coded safety rails.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    EXTERNAL SAFETY LAYER                     │
│  (Not learned, not bypassable, reality has veto power)       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│   │   Position   │    │   Drawdown   │    │    Rate      │  │
│   │    Limits    │    │    Limits    │    │   Limits     │  │
│   └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                              │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│   │   Exposure   │    │  Uncertainty │    │   Manual     │  │
│   │    Limits    │    │   Override   │    │  Kill Switch │  │
│   └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                     ┌────────────────┐
                     │  MuZero Agent  │
                     │   (proposes)   │
                     └────────────────┘
```

## Implementation

### Core Safety Wrapper

```python
# safety/wrapper.py
from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np

@dataclass
class SafetyConfig:
    """Hard limits - cannot be changed by agent."""

    # Position limits
    max_position_pct: float = 0.10          # Max 10% in single asset
    max_total_exposure: float = 1.0          # Max 100% total exposure

    # Drawdown limits
    max_drawdown_pct: float = 0.05           # Stop at 5% drawdown
    drawdown_recovery_pct: float = 0.02      # Resume at 2% recovery

    # Rate limits
    max_trades_per_hour: int = 100           # Prevent overtrading
    min_hold_time_steps: int = 5             # Min hold before exit

    # Uncertainty override
    uncertainty_flat_threshold: float = 0.1  # Go flat if uncertainty > 10%

    # Manual override
    manual_flat: bool = False                # Human kill switch


class SafetyWrapper:
    """External safety layer with veto power over agent decisions."""

    def __init__(self, config: SafetyConfig):
        self.config = config
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.is_stopped = False
        self.trade_timestamps = []
        self.position_entry_times = {}

    def update_equity(self, equity: float):
        """Update equity tracking for drawdown calculation."""
        self.current_equity = equity
        self.peak_equity = max(self.peak_equity, equity)

    @property
    def current_drawdown(self) -> float:
        """Current drawdown from peak."""
        if self.peak_equity == 0:
            return 0.0
        return (self.peak_equity - self.current_equity) / self.peak_equity

    def check_drawdown(self) -> bool:
        """Check if drawdown limit exceeded."""
        if self.current_drawdown > self.config.max_drawdown_pct:
            if not self.is_stopped:
                print(f"SAFETY: Drawdown limit hit ({self.current_drawdown:.2%})")
            self.is_stopped = True
            return True

        # Recovery check
        if self.is_stopped and self.current_drawdown < self.config.drawdown_recovery_pct:
            print(f"SAFETY: Recovered from drawdown, resuming")
            self.is_stopped = False

        return self.is_stopped

    def check_position_limits(self, proposed_action: int, current_positions: np.ndarray,
                             prices: np.ndarray, equity: float) -> int:
        """Enforce position size limits. Returns potentially modified action."""
        asset_idx = proposed_action // 5
        action_type = proposed_action % 5

        # Calculate position value after action
        if action_type in [2, 4]:  # Buy or LimitBuy
            new_position = current_positions[asset_idx] + 1  # Simplified
            position_value = abs(new_position * prices[asset_idx])
            position_pct = position_value / equity

            if position_pct > self.config.max_position_pct:
                print(f"SAFETY: Position limit blocked buy on asset {asset_idx}")
                return asset_idx * 5 + 1  # Force HOLD instead

        # Check total exposure
        total_exposure = np.sum(np.abs(current_positions) * prices) / equity
        if total_exposure > self.config.max_total_exposure:
            if action_type in [2, 4]:  # Trying to add exposure
                print(f"SAFETY: Exposure limit blocked trade")
                return asset_idx * 5 + 1  # Force HOLD

        return proposed_action

    def check_rate_limits(self, timestamp: float) -> bool:
        """Check if trading rate limit exceeded."""
        # Remove old timestamps
        one_hour_ago = timestamp - 3600
        self.trade_timestamps = [t for t in self.trade_timestamps if t > one_hour_ago]

        if len(self.trade_timestamps) >= self.config.max_trades_per_hour:
            print(f"SAFETY: Rate limit hit ({len(self.trade_timestamps)} trades/hour)")
            return True

        return False

    def record_trade(self, timestamp: float):
        """Record a trade for rate limiting."""
        self.trade_timestamps.append(timestamp)

    def check_uncertainty_override(self, uncertainty: float) -> bool:
        """Check if uncertainty requires going flat."""
        if uncertainty > self.config.uncertainty_flat_threshold:
            print(f"SAFETY: High uncertainty ({uncertainty:.3f}), forcing flat")
            return True
        return False

    def apply_safety(self, proposed_action: int, state: Dict) -> int:
        """
        Main entry point. Apply all safety checks.

        Args:
            proposed_action: Agent's proposed action
            state: Dict with 'positions', 'prices', 'equity', 'timestamp', 'uncertainty'

        Returns:
            Final action (possibly overridden to HOLD or CLOSE)
        """
        # Manual kill switch - highest priority
        if self.config.manual_flat:
            return self._close_all_action(state['positions'])

        # Update equity
        self.update_equity(state['equity'])

        # Drawdown check
        if self.check_drawdown():
            return self._close_all_action(state['positions'])

        # Uncertainty override
        if 'uncertainty' in state and self.check_uncertainty_override(state['uncertainty']):
            return self._close_all_action(state['positions'])

        # Rate limit check
        if self.check_rate_limits(state['timestamp']):
            asset_idx = proposed_action // 5
            return asset_idx * 5 + 1  # Force HOLD

        # Position limits
        proposed_action = self.check_position_limits(
            proposed_action,
            state['positions'],
            state['prices'],
            state['equity']
        )

        return proposed_action

    def _close_all_action(self, positions: np.ndarray) -> int:
        """Generate action to close largest position."""
        # Find asset with largest position
        abs_positions = np.abs(positions)
        if np.max(abs_positions) == 0:
            return 1  # Already flat, HOLD

        largest_idx = np.argmax(abs_positions)
        if positions[largest_idx] > 0:
            return largest_idx * 5 + 0  # Sell
        else:
            return largest_idx * 5 + 2  # Buy to close short

    def get_status(self) -> Dict:
        """Get current safety status for logging."""
        return {
            'safety_stopped': self.is_stopped,
            'current_drawdown': self.current_drawdown,
            'trades_this_hour': len(self.trade_timestamps),
            'manual_override': self.config.manual_flat,
        }
```

### Integration with Training

```python
# In train_jax_fast.py or self_play
from safety.wrapper import SafetyWrapper, SafetyConfig

# Initialize safety wrapper (OUTSIDE model)
safety_config = SafetyConfig(
    max_position_pct=0.10,
    max_drawdown_pct=0.05,
    max_trades_per_hour=100,
)
safety = SafetyWrapper(safety_config)

# In action selection
def get_safe_action(model, params, obs, safety_wrapper, env_state):
    # Agent proposes
    action, policy, stats = mcts_search(params, rng, obs)

    # Safety disposes
    safe_action = safety_wrapper.apply_safety(action, {
        'positions': env_state.positions,
        'prices': env_state.prices,
        'equity': env_state.equity,
        'timestamp': env_state.timestamp,
        'uncertainty': stats.get('uncertainty', 0.0),
    })

    if safe_action != action:
        log_info(f"Safety override: {action} -> {safe_action}")

    return safe_action, policy, stats
```

### API Integration (Future)

```python
# For live trading API
class TradingAPI:
    def __init__(self, agent, safety_config: SafetyConfig):
        self.agent = agent
        self.safety = SafetyWrapper(safety_config)

    def execute_step(self, market_data):
        # Get agent decision
        obs = self.prepare_observation(market_data)
        action = self.agent.act(obs)

        # Safety check
        safe_action = self.safety.apply_safety(action, self.get_state())

        # Execute
        if safe_action != action:
            self.log_override(action, safe_action)

        return self.exchange.execute(safe_action)

    def emergency_stop(self):
        """Human kill switch."""
        self.safety.config.manual_flat = True
        self.close_all_positions()
```

## CLI Arguments (Training)

```python
# Safety args apply to training environment too
parser.add_argument('--safety-max-position', type=float, default=0.10,
                   help='Max position size as fraction of equity')
parser.add_argument('--safety-max-drawdown', type=float, default=0.05,
                   help='Max drawdown before stopping')
parser.add_argument('--safety-max-trades-per-hour', type=int, default=100,
                   help='Max trades per hour')
parser.add_argument('--safety-enabled', action='store_true',
                   help='Enable safety wrapper during training')
```

## Logging

```python
# Safety events logged separately
'safety_overrides': count,
'safety_drawdown_stops': count,
'safety_rate_limit_hits': count,
'safety_position_limit_hits': count,
'safety_uncertainty_flats': count,
```

## Directory Structure

```
safety/
├── __init__.py
├── wrapper.py              # Main SafetyWrapper class
├── config.py               # SafetyConfig dataclass
├── limits.py               # Individual limit checkers
└── api_integration.py      # Future: live trading hooks
```

## Key Principle

**The safety wrapper is NOT part of the model.** It cannot be:
- Trained away
- Overridden by agent confidence
- Bypassed by any learned component

It is a hard external constraint, like a circuit breaker.

## Testing

```python
def test_drawdown_stop():
    safety = SafetyWrapper(SafetyConfig(max_drawdown_pct=0.05))
    safety.update_equity(100.0)  # Peak
    safety.update_equity(94.0)   # 6% drawdown

    assert safety.check_drawdown() == True
    assert safety.is_stopped == True

def test_position_limit():
    safety = SafetyWrapper(SafetyConfig(max_position_pct=0.10))
    # ... test that large positions are blocked
```

## Success Criteria

1. Safety overrides logged correctly
2. No way to bypass limits from agent
3. Training can proceed with safety enabled
4. Clear separation from learned components

## Status

- [ ] Core SafetyWrapper implementation
- [ ] Integration with training loop
- [ ] CLI arguments
- [ ] Logging and metrics
- [ ] API integration (future, for live trading)
