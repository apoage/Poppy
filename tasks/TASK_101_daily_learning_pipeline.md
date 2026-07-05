# TASK_101: Daily Learning Pipeline (Continual Adaptation)

## Status: PENDING
## Priority: HIGH

## Context

Base model training:
- Synthetic curriculum (learn trading mechanics)
- 420M historical steps (learn real market patterns)

This task implements the **deployment phase** - keeping the model adapted to current market conditions through daily offline learning.

## Concept

```
┌─────────────────────────────────────────────────────────┐
│                    DEPLOYMENT PIPELINE                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   Base Model (synthetic + 420M historical)              │
│                        ↓                                │
│   Initial Fine-tune: 30 days recent data                │
│                        ↓                                │
│   Daily Job: Encode yesterday → light fine-tune         │
│                        ↓                                │
│   Live Trading (with adapted model)                     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Why Offline Learning for Deployment?

| Approach | Pros | Cons |
|----------|------|------|
| Online RL (live) | Learns from real feedback | Expensive mistakes, slow |
| Offline/Supervised | No risk, fast adaptation | No exploration |

**Decision**: Use offline learning for deployment. The base model already learned exploration strategies from 420M steps. Deployment phase is about **adaptation**, not discovery.

## Architecture: What to Update

### Update (Offline Learning)
```python
# These learn "what the market looks like now"
representation_fn   # h(obs) → state
dynamics_fn         # g(state, action) → next_state, reward
prediction_head     # Market forecasts (price, volatility, etc.)
```

### Freeze or Update Slowly
```python
# These encode "how to trade" - learned from curriculum + history
policy_head         # Action selection - FREEZE or very slow LR
value_head          # State evaluation - slow LR (0.1x)
```

**Intuition**: The model knows HOW to trade. Daily learning teaches it WHAT the current market looks like.

## Daily Learning Pipeline

### Step 1: Data Collection (Nightly)

```python
def collect_daily_data(date: datetime, assets: List[str]) -> DailyDataset:
    """Collect yesterday's market data for all tracked assets."""

    data = {
        'timestamps': [],      # Synchronized across assets
        'ohlcv': [],           # (n_assets, n_steps, 5)
        'returns': [],         # Step-by-step returns
        'volatility': [],      # Rolling volatility estimates
        'correlations': [],    # Cross-asset correlations
    }

    for asset in assets:
        candles = fetch_ohlcv(asset, date, interval='1m')
        data['ohlcv'].append(candles)
        data['returns'].append(compute_returns(candles))

    # Compute cross-asset features
    data['correlations'] = compute_rolling_correlations(data['ohlcv'])
    data['volatility'] = compute_realized_volatility(data['returns'])

    return DailyDataset(data)
```

### Step 2: Create Supervised Targets

```python
def create_training_targets(dataset: DailyDataset) -> TrainingBatch:
    """Create supervised learning targets from market data."""

    targets = []

    for t in range(len(dataset) - lookahead):
        obs_t = encode_observation(dataset, t)
        obs_t1 = encode_observation(dataset, t + 1)

        target = {
            # Dynamics targets
            'next_state': obs_t1,
            'realized_return': dataset.returns[t],

            # Prediction targets (multi-horizon)
            'return_1step': dataset.returns[t],
            'return_5step': dataset.returns[t:t+5].sum(),
            'return_20step': dataset.returns[t:t+20].sum(),

            # Volatility targets
            'volatility_realized': dataset.volatility[t],

            # Correlation targets (for multi-asset)
            'correlation_matrix': dataset.correlations[t],
        }

        targets.append((obs_t, target))

    return TrainingBatch(targets)
```

### Step 3: Offline Fine-tune

```python
@dataclass
class DailyLearningConfig:
    # Learning rates (much lower than base training)
    lr_representation: float = 1e-5
    lr_dynamics: float = 1e-5
    lr_prediction: float = 1e-5
    lr_value: float = 1e-6       # Very slow
    lr_policy: float = 0.0       # Frozen

    # Training config
    batch_size: int = 256
    steps_per_day: int = 500     # Light fine-tune
    gradient_clip: float = 0.5

    # Regularization (prevent catastrophic forgetting)
    weight_decay: float = 0.01
    ewc_lambda: float = 100.0    # Elastic Weight Consolidation

    # Data window
    lookback_days: int = 30      # Rolling window for context


def daily_finetune(
    model: MuZeroModel,
    dataset: DailyDataset,
    config: DailyLearningConfig,
    fisher_matrix: Optional[Dict] = None,  # For EWC
) -> Tuple[MuZeroModel, Dict]:
    """Run daily fine-tuning on yesterday's data."""

    # Create separate optimizers per component
    optimizers = {
        'representation': optax.adam(config.lr_representation),
        'dynamics': optax.adam(config.lr_dynamics),
        'prediction': optax.adam(config.lr_prediction),
        'value': optax.adam(config.lr_value),
        # Policy frozen - no optimizer
    }

    # Create training batch
    batch = create_training_targets(dataset)

    metrics = defaultdict(list)

    for step in range(config.steps_per_day):
        # Sample mini-batch
        mini_batch = batch.sample(config.batch_size)

        # Compute losses
        loss, loss_components = compute_offline_loss(
            model,
            mini_batch,
            fisher_matrix=fisher_matrix,
            ewc_lambda=config.ewc_lambda,
        )

        # Update (policy head frozen)
        grads = jax.grad(loss)(model.params)
        grads = freeze_policy_grads(grads)  # Zero out policy gradients

        model = apply_gradients(model, grads, optimizers)

        metrics['loss'].append(loss)
        for k, v in loss_components.items():
            metrics[k].append(v)

    # Update Fisher matrix for next day's EWC
    new_fisher = update_fisher_matrix(model, batch)

    return model, metrics, new_fisher
```

### Step 4: Elastic Weight Consolidation (Prevent Forgetting)

```python
def compute_ewc_loss(
    params: Dict,
    fisher_matrix: Dict,
    original_params: Dict,
    ewc_lambda: float,
) -> float:
    """Penalize changes to important weights."""

    ewc_loss = 0.0

    for name, param in params.items():
        if name in fisher_matrix:
            # Fisher diagonal approximates parameter importance
            importance = fisher_matrix[name]
            param_diff = param - original_params[name]
            ewc_loss += (importance * param_diff ** 2).sum()

    return ewc_lambda * ewc_loss


def update_fisher_matrix(
    model: MuZeroModel,
    dataset: TrainingBatch,
    n_samples: int = 1000,
) -> Dict:
    """Estimate Fisher information matrix (diagonal approximation)."""

    fisher = {name: jnp.zeros_like(param)
              for name, param in model.params.items()}

    for _ in range(n_samples):
        sample = dataset.sample(1)
        grads = jax.grad(compute_offline_loss)(model.params, sample)

        for name, grad in grads.items():
            fisher[name] += grad ** 2

    # Normalize
    fisher = {name: f / n_samples for name, f in fisher.items()}

    return fisher
```

## Offline Loss Function

```python
def compute_offline_loss(
    model: MuZeroModel,
    batch: TrainingBatch,
    fisher_matrix: Optional[Dict] = None,
    ewc_lambda: float = 100.0,
) -> Tuple[float, Dict]:
    """Compute supervised loss for offline learning."""

    obs, targets = batch

    # Forward pass
    state = model.representation(obs)

    losses = {}

    # 1. Dynamics consistency loss
    # Does our dynamics model predict correct next state?
    for action in range(n_actions):
        pred_next_state, pred_reward = model.dynamics(state, action)

        # We don't know what action was taken, so use contrastive loss
        # Real next state should be close to SOME action's prediction

    # 2. Prediction head losses (main learning signal)
    predictions = model.prediction_head(state)

    # Return predictions
    losses['return_1step'] = mse(predictions.return_1step, targets['return_1step'])
    losses['return_5step'] = mse(predictions.return_5step, targets['return_5step'])
    losses['return_20step'] = mse(predictions.return_20step, targets['return_20step'])

    # Volatility prediction
    losses['volatility'] = mse(predictions.volatility, targets['volatility_realized'])

    # 3. Representation consistency
    # Next observation should map to similar state as dynamics prediction
    next_obs = targets['next_obs']
    next_state_real = model.representation(next_obs)
    next_state_pred, _ = model.dynamics(state, inferred_action)
    losses['state_consistency'] = cosine_distance(next_state_real, next_state_pred)

    # 4. Value sanity check (optional, light weight)
    # Value shouldn't drift too far from realized returns
    value_pred = model.value(state)
    realized_value = targets['return_20step']  # Proxy for value
    losses['value_anchor'] = 0.1 * mse(value_pred, realized_value)

    # Total loss
    total_loss = sum(losses.values())

    # 5. EWC regularization (prevent forgetting)
    if fisher_matrix is not None:
        ewc_loss = compute_ewc_loss(
            model.params,
            fisher_matrix,
            model.original_params,
            ewc_lambda,
        )
        losses['ewc'] = ewc_loss
        total_loss += ewc_loss

    return total_loss, losses
```

## Regime Drift Detection

```python
@dataclass
class RegimeMetrics:
    volatility_zscore: float      # Current vs historical volatility
    correlation_shift: float      # Correlation matrix change
    return_distribution: float    # KL divergence from historical
    prediction_error: float       # Model's prediction accuracy


def detect_regime_drift(
    model: MuZeroModel,
    recent_data: DailyDataset,
    historical_stats: Dict,
    threshold: float = 2.0,
) -> Tuple[bool, RegimeMetrics]:
    """Detect if market regime has shifted significantly."""

    metrics = RegimeMetrics(
        volatility_zscore=compute_volatility_zscore(recent_data, historical_stats),
        correlation_shift=compute_correlation_shift(recent_data, historical_stats),
        return_distribution=compute_return_kl(recent_data, historical_stats),
        prediction_error=compute_model_error(model, recent_data),
    )

    # Drift if any metric exceeds threshold
    drift_detected = (
        abs(metrics.volatility_zscore) > threshold or
        metrics.correlation_shift > threshold or
        metrics.prediction_error > threshold * 2
    )

    return drift_detected, metrics


def handle_regime_drift(
    model: MuZeroModel,
    drift_metrics: RegimeMetrics,
    config: DailyLearningConfig,
) -> DailyLearningConfig:
    """Adjust learning when regime drift detected."""

    # Increase learning rate during drift (adapt faster)
    drift_multiplier = min(3.0, 1.0 + drift_metrics.prediction_error)

    new_config = config.copy()
    new_config.lr_representation *= drift_multiplier
    new_config.lr_dynamics *= drift_multiplier
    new_config.lr_prediction *= drift_multiplier
    new_config.steps_per_day = int(config.steps_per_day * drift_multiplier)

    # Reduce EWC constraint (allow more change)
    new_config.ewc_lambda *= 0.5

    logging.warning(f"Regime drift detected! Increasing adaptation rate by {drift_multiplier:.1f}x")

    return new_config
```

## Daily Job Orchestration

```python
class DailyLearningJob:
    """Nightly job to adapt model to recent market data."""

    def __init__(
        self,
        model_path: str,
        config: DailyLearningConfig,
        assets: List[str],
    ):
        self.model = load_model(model_path)
        self.config = config
        self.assets = assets
        self.fisher_matrix = None
        self.historical_stats = load_historical_stats()

    def run(self, date: datetime) -> Dict:
        """Run daily learning job."""

        # 1. Collect yesterday's data
        dataset = collect_daily_data(date - timedelta(days=1), self.assets)

        # 2. Check for regime drift
        drift_detected, drift_metrics = detect_regime_drift(
            self.model,
            dataset,
            self.historical_stats,
        )

        # 3. Adjust config if drift detected
        config = self.config
        if drift_detected:
            config = handle_regime_drift(self.model, drift_metrics, config)

        # 4. Run fine-tuning
        self.model, metrics, self.fisher_matrix = daily_finetune(
            self.model,
            dataset,
            config,
            self.fisher_matrix,
        )

        # 5. Validate model (sanity checks)
        validation_passed = self.validate_model(dataset)

        if not validation_passed:
            logging.error("Model validation failed! Rolling back to previous checkpoint.")
            self.model = load_model(self.model_path)  # Rollback
            return {'status': 'rollback', 'metrics': metrics}

        # 6. Save updated model
        save_model(self.model, self.model_path)

        # 7. Update historical stats
        self.update_historical_stats(dataset)

        return {
            'status': 'success',
            'drift_detected': drift_detected,
            'drift_metrics': drift_metrics,
            'training_metrics': metrics,
        }

    def validate_model(self, dataset: DailyDataset) -> bool:
        """Sanity check model after fine-tuning."""

        # Check prediction head outputs are reasonable
        sample = dataset.sample(100)
        predictions = self.model.predict(sample)

        checks = [
            # Predictions not NaN
            not jnp.isnan(predictions).any(),
            # Predictions not degenerate (all same value)
            predictions.std() > 1e-6,
            # Returns predictions in reasonable range
            jnp.abs(predictions.returns).max() < 1.0,  # <100% per step
            # Volatility positive
            (predictions.volatility > 0).all(),
        ]

        return all(checks)
```

## CLI Interface

```bash
# Run daily learning job
python daily_learning.py --date 2025-01-08 --model-path models/prod_model.pt

# Run with custom config
python daily_learning.py \
    --date 2025-01-08 \
    --model-path models/prod_model.pt \
    --lr 1e-5 \
    --steps 500 \
    --ewc-lambda 100

# Dry run (no model save)
python daily_learning.py --date 2025-01-08 --dry-run

# Backfill multiple days
python daily_learning.py --start-date 2025-01-01 --end-date 2025-01-08
```

## Monitoring & Alerts

```python
# Key metrics to track
DAILY_METRICS = {
    'prediction_loss': 'Lower is better, spike = regime change',
    'ewc_loss': 'Higher = model changing more',
    'drift_score': 'Composite regime drift metric',
    'validation_passed': 'Boolean sanity check',
}

# Alerts
ALERTS = {
    'prediction_loss_spike': 'Loss > 2x rolling average',
    'drift_detected': 'Regime shift detected',
    'validation_failed': 'Model sanity check failed',
    'rollback_triggered': 'Model reverted to previous version',
}
```

## Integration Points

- **Data pipeline**: `mcts_trader/data/history.py` - fetch from DB
- **Model loading**: `mcts_trader/jax_agent/trainer.py` - checkpoint loading
- **Scheduler**: Cron job or Airflow DAG

## Files to Create

```
mcts_trader/
├── deployment/
│   ├── __init__.py
│   ├── daily_learning.py       # Main job
│   ├── offline_loss.py         # Loss functions
│   ├── ewc.py                  # Elastic Weight Consolidation
│   ├── regime_detection.py     # Drift detection
│   └── validation.py           # Model sanity checks
```

## Testing

```python
def test_daily_learning_reduces_prediction_error():
    """Fine-tuning should improve predictions on recent data."""
    model = load_base_model()
    dataset = load_test_dataset()

    error_before = compute_prediction_error(model, dataset)
    model, _, _ = daily_finetune(model, dataset, config)
    error_after = compute_prediction_error(model, dataset)

    assert error_after < error_before * 0.9  # At least 10% improvement


def test_ewc_prevents_catastrophic_forgetting():
    """Model should retain base knowledge after fine-tuning."""
    model = load_base_model()
    base_performance = evaluate_on_synthetic(model)

    # Fine-tune on real data
    model, _, _ = daily_finetune(model, real_dataset, config)

    # Should still perform well on synthetic
    new_performance = evaluate_on_synthetic(model)
    assert new_performance > base_performance * 0.8  # Max 20% degradation
```

## References

- Elastic Weight Consolidation: "Overcoming catastrophic forgetting in neural networks" (Kirkpatrick et al., 2017)
- Continual Learning: "Continual Lifelong Learning with Neural Networks: A Review" (Parisi et al., 2019)
- Online Adaptation: "Online Meta-Learning" (Finn et al., 2019)
