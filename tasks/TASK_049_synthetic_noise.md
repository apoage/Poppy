# TASK_049: Add Noise to Synthetic Data Generators

## Status: COMPLETED (2026-01-23)

**Implementation**: `--synthetic-noise` CLI flag
- Located in: `cli.py`, `config.py`, `env_factory.py`, `adaptive_noise.py`
- Also supports `--noise-schedule` for progressive ramping

## Problem

All synthetic generators produce deterministic, memorizable patterns:
- Level 0 (sine): Fixed period, amplitude, phase per asset
- Level 1 (lagged): Exact 10-step lag, 1% noise
- Level 2 (dip): Instant V-shaped recovery
- Level 4 (rotation): Exact 50-step rotation period

Agent memorizes patterns instead of learning trading skills.

## Solution

Add stochastic elements to each generator.

### Level 0: Sine Waves

```python
def generate_sine_prices(...):
    # Current (deterministic)
    prices[:, i] = base * (1 + amp * np.sin(2*np.pi*t/period + phase))

    # Fixed (stochastic)
    period_drift = np.cumsum(np.random.normal(0, 0.02, n_steps))  # Period varies
    phase_drift = np.cumsum(np.random.normal(0, 0.01, n_steps))   # Phase wanders
    amp_noise = 1 + np.random.normal(0, 0.1, n_steps)             # Amplitude varies

    effective_t = t + period_drift
    prices[:, i] = base * (1 + amp * amp_noise * np.sin(2*np.pi*effective_t/period + phase + phase_drift))

    # Add random walk component
    random_walk = np.cumsum(np.random.normal(0, 0.002, n_steps))
    prices[:, i] *= (1 + random_walk)
```

### Level 1: Lagged Pairs

```python
def generate_lagged_prices(...):
    # Current
    LAG = 10
    NOISE = 0.01

    # Fixed
    lag = np.random.randint(5, 15)  # Variable lag 5-15 steps
    noise = 0.05  # Higher noise (5%)

    # Lag can vary over time
    lag_variation = np.random.randint(-2, 3, n_steps)  # ±2 steps

    # Correlation decay
    correlation = 0.9 - 0.001 * np.arange(n_steps)  # Decay over episode
```

### Level 2: Dips

```python
def generate_dip_prices(...):
    # Current: Instant V-recovery

    # Fixed: Gradual recovery
    recovery_steps = np.random.randint(10, 50)  # 10-50 steps to recover
    recovery_pct = np.random.uniform(0.8, 1.0)  # 80-100% recovery

    # Add double-dip probability
    double_dip_prob = 0.3

    # Recovery curve: exponential, not instant
    recovery = 1 - (1 - recovery_pct) * np.exp(-np.arange(recovery_steps) / (recovery_steps/3))
```

### Level 4: Rotation

```python
def generate_rotation_prices(...):
    # Current
    ROTATION_PERIOD = 50  # Fixed

    # Fixed
    rotation_period = 50 + np.random.randint(-15, 15)  # 35-65 steps

    # Momentum-based rotation (not clock-based)
    rotation_trigger = momentum > threshold  # Based on price action

    # Incomplete rotations
    rotation_strength = np.random.uniform(0.5, 1.0)  # Partial sector dominance
```

## Implementation

### File: `mcts_trader/data/synthetic_market.py`

Add noise parameters to each generator:

```python
def generate_sine_prices(
    n_assets: int,
    n_steps: int,
    period_noise: float = 0.02,      # NEW
    phase_drift: float = 0.01,        # NEW
    amplitude_noise: float = 0.1,     # NEW
    random_walk_std: float = 0.002,   # NEW
    seed: int = None,
) -> np.ndarray:
```

### CLI Arguments

```python
parser.add_argument('--synthetic-noise', type=float, default=0.1,
                   help='Noise level for synthetic generators (0=deterministic)')
```

## Testing

```python
def test_sine_not_memorizable():
    """Same network should not perfectly predict different seeds."""
    prices_seed1 = generate_sine_prices(seed=1, noise=0.1)
    prices_seed2 = generate_sine_prices(seed=2, noise=0.1)

    # Correlation between seeds should be low
    corr = np.corrcoef(prices_seed1.flatten(), prices_seed2.flatten())[0,1]
    assert corr < 0.5, "Patterns too similar across seeds"

def test_lag_not_exact():
    """Lagged pairs should have variable, imperfect correlation."""
    leader, follower = generate_lagged_prices(noise=0.05)

    # Test multiple lag values
    correlations = [np.corrcoef(leader[:-lag], follower[lag:])[0,1] for lag in range(5, 15)]

    # No single lag should give > 0.95 correlation
    assert max(correlations) < 0.95
```

## Expected Impact

- Agent learns pattern detection, not memorization
- Better generalization to real markets
- Curriculum still teaches intended skills (momentum, correlation, dip-buying)
- Harder training = more robust policies

## Priority

**CRITICAL** - Without this fix, curriculum is fundamentally broken.
