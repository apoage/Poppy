# TASK_046: Dynamics Head Desynchronization (Reality Check)

## Overview

Periodically validate that the prediction head's learned dynamics match actual market behavior. When prediction error exceeds threshold, trigger re-calibration or reduce planning confidence.

## Motivation

> "If the learned dynamics are slightly wrong under a regime shift, the planner can plan very confidently, reduce entropy, generate internally consistent but externally false plans."

The prediction head can drift from reality silently. Unlike value loss (which shows in rewards), dynamics error only manifests when plans fail. We need explicit reality checks.

## The Problem

```
Training Phase:
  Prediction head learns: "momentum = +0.05, reversion = 0.03"
  Works great on synthetic Level 0

Deployment/New Regime:
  Reality: "momentum = -0.02, reversion = 0.15" (bearish, high reversion)
  Prediction head still confident in old dynamics
  MCTS plans based on wrong dynamics
  Agent confidently loses money
```

## Approaches

### Option A: Online Prediction Monitoring (Recommended)

Track prediction error in real-time during episodes.

```python
class DynamicsRealityChecker:
    """Track prediction head accuracy against realized prices."""

    def __init__(self, window_size=100, alert_threshold=2.0):
        self.window_size = window_size
        self.alert_threshold = alert_threshold  # Multiples of baseline error
        self.prediction_errors = deque(maxlen=window_size)
        self.baseline_error = None  # Calibrated during stable period

    def record_prediction(self, predicted_log_price, actual_log_price):
        """Called each step after price is realized."""
        error = jnp.abs(predicted_log_price - actual_log_price)
        self.prediction_errors.append(error)

    def get_current_error(self):
        """Rolling mean prediction error."""
        if len(self.prediction_errors) < 10:
            return 0.0
        return jnp.mean(jnp.array(list(self.prediction_errors)))

    def is_desynchronized(self):
        """True if prediction error significantly above baseline."""
        if self.baseline_error is None:
            return False
        return self.get_current_error() > self.alert_threshold * self.baseline_error

    def calibrate_baseline(self):
        """Call during known-good period to set baseline."""
        self.baseline_error = self.get_current_error()
        print(f"Dynamics baseline calibrated: {self.baseline_error:.6f}")
```

### Option B: Physics Parameter Monitoring

Track learned physics parameters for drift.

```python
def check_physics_parameters(params, acceptable_ranges):
    """Alert if physics parameters drift outside acceptable bounds."""
    physics = params['price_predictor']['physics_core']

    alerts = []
    for name, (low, high) in acceptable_ranges.items():
        if name in physics:
            val = float(physics[name])
            if val < low or val > high:
                alerts.append(f"{name}={val:.4f} outside [{low}, {high}]")

    return alerts

# Example acceptable ranges
PHYSICS_BOUNDS = {
    'alpha': (-0.1, 0.3),      # Momentum
    'beta': (0.0, 0.2),         # Mean reversion
    'sigma': (0.001, 0.1),      # Volatility
}
```

### Option C: Holdout Validation

Periodically test dynamics on held-out data.

```python
def dynamics_holdout_test(model, params, holdout_data, threshold=0.02):
    """Test prediction head on reserved validation data."""
    total_error = 0
    for obs, next_prices in holdout_data:
        latent = model.apply(params, obs, method=model.representation)
        pred_prices, _ = model.apply(params, latent, method=model.predict_log_prices)
        total_error += jnp.mean((pred_prices - next_prices) ** 2)

    avg_error = total_error / len(holdout_data)

    if avg_error > threshold:
        print(f"DYNAMICS DESYNC: holdout error {avg_error:.6f} > {threshold}")
        return True, avg_error

    return False, avg_error
```

### Option D: Gradient-Based Drift Detection

Monitor if dynamics gradients suddenly change direction.

```python
def gradient_drift_detector(recent_grads, historical_grads):
    """Detect if gradient direction has shifted (regime change)."""
    # Cosine similarity between recent and historical gradient directions
    recent_flat = flatten_grads(recent_grads)
    historical_flat = flatten_grads(historical_grads)

    cosine_sim = jnp.dot(recent_flat, historical_flat) / (
        jnp.linalg.norm(recent_flat) * jnp.linalg.norm(historical_flat)
    )

    # Negative cosine = gradients pointing opposite directions = regime shift
    return cosine_sim < 0.0
```

## Response Actions

When desynchronization detected:

### Level 1: Alert Only
```python
if reality_checker.is_desynchronized():
    log_warning(f"Dynamics desync detected: error={reality_checker.get_current_error():.6f}")
```

### Level 2: Reduce Planning Depth
```python
if reality_checker.is_desynchronized():
    # Don't trust deep planning with wrong dynamics
    trainer.update_simulations(num_simulations=8)  # Down from 32
```

### Level 3: Increase Exploration
```python
if reality_checker.is_desynchronized():
    # Don't exploit wrong model, explore to learn correct dynamics
    trainer.entropy_coef = 0.5  # Aggressive exploration
```

### Level 4: Force Conservative Position
```python
if reality_checker.is_desynchronized():
    # External override - reduce exposure
    return scale_position(action, factor=0.25)  # 25% normal size
```

### Level 5: Go Flat (Nuclear Option)
```python
if reality_checker.is_desynchronized() and error > CRITICAL_THRESHOLD:
    # Completely untrust the model
    return CLOSE_ALL_POSITIONS
```

## Integration with Training

```python
# In train_jax_fast.py main loop
reality_checker = DynamicsRealityChecker(window_size=100)

for epoch in range(epochs):
    # Run self-play, collect predictions
    for step, (obs, action, next_obs) in enumerate(episode):
        # Get prediction for this step
        pred_log_price = get_prediction(model, params, obs, action)
        actual_log_price = extract_log_price(next_obs)

        reality_checker.record_prediction(pred_log_price, actual_log_price)

    # Check at end of episode
    if reality_checker.is_desynchronized():
        handle_desync(level=config.desync_response_level)

    # Log
    log_metrics({
        'dynamics_error': reality_checker.get_current_error(),
        'dynamics_desync': reality_checker.is_desynchronized(),
    })
```

## CLI Arguments

```python
parser.add_argument('--dynamics-check', action='store_true',
                   help='Enable dynamics reality checking')
parser.add_argument('--dynamics-window', type=int, default=100,
                   help='Rolling window for prediction error')
parser.add_argument('--dynamics-alert-threshold', type=float, default=2.0,
                   help='Alert when error > N × baseline')
parser.add_argument('--desync-response', choices=['alert', 'reduce-sims', 'explore', 'conservative', 'flat'],
                   default='alert', help='Response to dynamics desync')
```

## Metrics

| Metric | Description |
|--------|-------------|
| `dynamics_error_1step` | 1-step prediction MSE (rolling) |
| `dynamics_error_5step` | 5-step prediction MSE (rolling) |
| `dynamics_baseline` | Calibrated baseline error |
| `dynamics_desync_ratio` | Current error / baseline |
| `dynamics_desync_events` | Count of desync triggers |
| `physics_alpha` | Current momentum parameter |
| `physics_beta` | Current reversion parameter |

## Files to Modify/Create

- [ ] `mcts_trader/jax_agent/reality_check.py` (NEW)
- [ ] `train_jax_fast.py` (integration, CLI args)
- [ ] `mcts_trader/jax_agent/trainer.py` (response actions)

## Success Criteria

1. Desync detection triggers on synthetic regime changes
2. False positive rate < 5% during stable periods
3. Response actions measurably reduce losses during regime shifts
4. No performance impact during normal operation

## Status

- [ ] Phase 1: Online prediction monitoring
- [ ] Phase 2: Physics parameter bounds checking
- [ ] Phase 3: Response action integration
- [ ] Phase 4: Testing with synthetic regime shifts
