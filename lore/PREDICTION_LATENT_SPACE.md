# Prediction in Latent Space - Architecture Design

## Problem Statement

Current model has:
- ✅ **Analytics** (encoder processes state)
- ✅ **Decisions** (policy chooses actions)
- ✅ **Reasoning** (MCTS + TRM evaluates outcomes)
- ❌ **Prediction** (no explicit price forecasting)

**Gap**: Model learns to predict VALUE (expected reward) but not actual PRICE TRAJECTORIES.

**Why this matters**:
1. Latent space doesn't encode "what will happen next" - only "what to do now"
2. Can't evaluate model's market understanding
3. Missing auxiliary learning signal (multi-task learning improves generalization)
4. Can't use predictions for risk management

## Solution: Multi-Task Prediction Heads

### Architecture

```
                    ┌─────────────────────────────────┐
                    │   Observation (27, 64, 64)      │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │   Encoder (CNN + TRM)           │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │   Latent State (256d)           │
                    │   Shared Representation          │
                    └──────────────┬──────────────────┘
                                   │
            ┌──────────────────────┼──────────────────────┐
            │                      │                      │
    ┌───────▼───────┐    ┌────────▼────────┐    ┌───────▼────────┐
    │  RL Heads     │    │ Prediction Heads │    │ Auxiliary Heads│
    │               │    │                  │    │                │
    │ • Policy      │    │ • Price (10-step)│    │ • Volatility   │
    │ • Value       │    │ • Direction      │    │ • Regime       │
    │ • Reward      │    │ • Returns        │    │ • Correlation  │
    └───────────────┘    └──────────────────┘    └────────────────┘
```

**Key Insight**: All heads share the same latent representation. Prediction loss forces latent to encode future information, improving RL performance.

---

## Implementation

### Step 1: Add Prediction Heads to Model

**File**: `mcts_trader/jax_agent/muzero_model.py`

```python
class MuZeroNetworks(nn.Module):
    # ... existing ...
    enable_predictions: bool = True  # NEW: Toggle predictions

    def setup(self):
        # ... existing RL heads ...

        if self.enable_predictions:
            # Price prediction head (predict next 10 steps)
            self.price_predictor = nn.Dense(64 * 10)  # 64 assets × 10 steps

            # Volatility prediction head (predict next-step volatility)
            self.volatility_predictor = nn.Dense(64)  # Per-asset volatility

            # Market regime classifier (3 classes: bull/bear/sideways)
            self.regime_classifier = nn.Dense(3)

    def __call__(self, x, mode='initial_inference'):
        # ... existing encoding ...
        latent = self.encoder(x)
        hidden = self.shared_representation(latent)

        # RL heads (existing)
        policy_logits = self.policy_head(hidden)
        value_logits = self.value_head(hidden)

        output = {
            'policy_logits': policy_logits,
            'value': value_logits,
            'latent_state': latent
        }

        # Prediction heads (NEW)
        if self.enable_predictions and mode == 'initial_inference':
            # Price prediction: (batch, 64*10) → reshape to (batch, 64, 10)
            price_pred = self.price_predictor(hidden)
            price_pred = price_pred.reshape(-1, 64, 10)

            # Volatility prediction: (batch, 64)
            volatility_pred = self.volatility_predictor(hidden)

            # Regime classification: (batch, 3) logits
            regime_logits = self.regime_classifier(hidden)

            output.update({
                'price_prediction': price_pred,      # % change for next 10 steps
                'volatility_prediction': volatility_pred,  # Next-step vol
                'regime_logits': regime_logits       # Bull/bear/sideways
            })

        return output
```

---

### Step 2: Generate Prediction Targets

**File**: `mcts_trader/jax_agent/trainer.py`

Add method to compute ground truth:

```python
def compute_prediction_targets(trajectory):
    """
    Compute ground truth for prediction heads.

    Args:
        trajectory: Dict with 'observations', 'actions', 'rewards', ...

    Returns:
        targets: Dict with 'price_targets', 'volatility_targets', 'regime_targets'
    """
    n_steps = len(trajectory['observations'])

    # Extract price history from observations
    # Assume observation channel 0 is price data
    prices = []
    for obs in trajectory['observations']:
        # obs shape: (27, 64, 64)
        # Extract prices from channel 0 (simplified)
        asset_prices = obs[0].mean(axis=1)  # (64,) - mean across correlation matrix
        prices.append(asset_prices)

    prices = jnp.array(prices)  # (n_steps, 64)

    # Price targets: % change for next 10 steps
    price_targets = []
    for t in range(n_steps):
        # Predict % change from t to t+1, t+2, ..., t+10
        future_steps = min(10, n_steps - t - 1)
        if future_steps > 0:
            future_prices = prices[t+1:t+1+future_steps]  # (future_steps, 64)
            pct_changes = (future_prices - prices[t]) / (prices[t] + 1e-8)  # Avoid div by 0

            # Pad if needed
            if future_steps < 10:
                padding = jnp.zeros((10 - future_steps, 64))
                pct_changes = jnp.concatenate([pct_changes, padding], axis=0)

            price_targets.append(pct_changes.T)  # (64, 10)
        else:
            # Last step - no future data
            price_targets.append(jnp.zeros((64, 10)))

    price_targets = jnp.array(price_targets)  # (n_steps, 64, 10)

    # Volatility targets: std of next step % change
    volatility_targets = []
    for t in range(n_steps - 1):
        next_pct_change = (prices[t+1] - prices[t]) / (prices[t] + 1e-8)
        # Use rolling window std (or just abs value as proxy)
        vol = jnp.abs(next_pct_change)
        volatility_targets.append(vol)

    volatility_targets.append(jnp.zeros(64))  # Last step
    volatility_targets = jnp.array(volatility_targets)  # (n_steps, 64)

    # Regime targets: classify based on next 10-step trend
    regime_targets = []
    for t in range(n_steps):
        future_steps = min(10, n_steps - t - 1)
        if future_steps >= 5:
            future_return = (prices[t+future_steps] - prices[t]) / (prices[t] + 1e-8)
            avg_return = future_return.mean()  # Portfolio-wide

            # Classify: bull (>1%), bear (<-1%), sideways
            if avg_return > 0.01:
                regime = 0  # Bull
            elif avg_return < -0.01:
                regime = 1  # Bear
            else:
                regime = 2  # Sideways
        else:
            regime = 2  # Default sideways

        regime_targets.append(regime)

    regime_targets = jnp.array(regime_targets)  # (n_steps,)

    return {
        'price_targets': price_targets,
        'volatility_targets': volatility_targets,
        'regime_targets': regime_targets
    }
```

---

### Step 3: Add Prediction Loss

**File**: `mcts_trader/jax_agent/trainer.py`

Update training loop:

```python
def train_step(self, batch):
    """Training step with prediction losses"""

    def loss_fn(params):
        # ... existing RL losses (policy, value, reward) ...

        # Prediction losses (NEW)
        if self.model.enable_predictions:
            outputs = self.model.apply({'params': params}, batch['observations'])

            # Price prediction loss (MSE)
            price_pred = outputs['price_prediction']  # (batch, 64, 10)
            price_target = batch['price_targets']
            price_loss = jnp.mean((price_pred - price_target) ** 2)

            # Volatility prediction loss (MSE)
            vol_pred = outputs['volatility_prediction']
            vol_target = batch['volatility_targets']
            volatility_loss = jnp.mean((vol_pred - vol_target) ** 2)

            # Regime classification loss (cross-entropy)
            regime_logits = outputs['regime_logits']
            regime_target = batch['regime_targets']
            regime_loss = -jnp.mean(
                jax.nn.log_softmax(regime_logits)[jnp.arange(len(regime_target)), regime_target]
            )

            # Combine losses
            prediction_loss = price_loss + 0.5 * volatility_loss + 0.5 * regime_loss

        else:
            prediction_loss = 0.0

        # Total loss
        total_loss = (
            policy_loss +
            value_loss +
            reward_loss +
            0.1 * prediction_loss  # Weight prediction lower than RL
        )

        return total_loss, {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'reward_loss': reward_loss,
            'price_loss': price_loss if self.model.enable_predictions else 0.0,
            'volatility_loss': volatility_loss if self.model.enable_predictions else 0.0,
            'regime_loss': regime_loss if self.model.enable_predictions else 0.0
        }

    # ... rest of training step ...
```

---

### Step 4: Store Prediction Targets in Replay Buffer

**File**: `train_jax_fast.py`

Modify self-play loop:

```python
def run_self_play(env, search_fn, model_apply_fn, params, rng_key, steps=200):
    # ... existing self-play ...

    # After episode completes
    trajectory = {
        'observations': np.array(observations),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'values': np.array(values),
        'policy_probs': np.array(policy_probs),
        'dones': np.array(dones)
    }

    # Compute prediction targets
    from mcts_trader.jax_agent.trainer import compute_prediction_targets
    prediction_targets = compute_prediction_targets(trajectory)

    trajectory.update(prediction_targets)  # Add to trajectory

    return trajectory, score
```

---

## Benefits

### 1. Richer Latent Representation

**Before**: Latent encodes "what action is best"
**After**: Latent encodes "what will happen" + "what to do"

**Result**: Better generalization, more interpretable embeddings.

### 2. Model Evaluation

Can now measure:
- **Price prediction accuracy**: Is model understanding market dynamics?
- **Regime classification**: Does it detect bull/bear markets?
- **Volatility prediction**: Does it anticipate risk?

**Example diagnostic**:
```python
# After training
predictions = model.predict(test_observations)
actual_prices = get_actual_next_prices(test_data)

prediction_error = np.mean((predictions['price_prediction'] - actual_prices) ** 2)
print(f"Price prediction MSE: {prediction_error:.4f}")
```

### 3. Multi-Task Learning

Research shows auxiliary tasks improve RL performance:
- [Jaderberg et al. 2016](https://arxiv.org/abs/1611.05397): "Auxiliary tasks improve generalization"
- Prediction forces model to learn world dynamics, not just policy

### 4. Risk Management

Can use predictions for:
- **Position sizing**: If high volatility predicted → reduce size
- **Stop losses**: If regime changes bear → exit longs
- **Opportunity detection**: If price pump predicted → increase allocation

---

## Testing Strategy

### Test 1: Prediction on Synthetic Level 0 (Sine Waves)

**Hypothesis**: Model should predict sine waves perfectly (they're deterministic).

```bash
python train_jax_fast.py \
  --synthetic-level 0 \
  --enable-predictions \
  --epochs 1000 \
  --name test_predictions_l0
```

**Expected**:
- Price prediction MSE → 0 (after 500 epochs)
- Proves prediction head can learn

### Test 2: Regime Classification on Level 3

**Hypothesis**: Model should classify bull/bear trends.

```bash
python train_jax_fast.py \
  --synthetic-level 3 \
  --enable-predictions \
  --epochs 2000
```

**Expected**:
- Regime accuracy > 80%
- Model detects embedded trends

### Test 3: Ablation Study

**Question**: Does prediction help RL performance?

**Setup**:
```bash
# With predictions
python train_jax_fast.py --synthetic-level 4 --enable-predictions --name with_pred

# Without predictions
python train_jax_fast.py --synthetic-level 4 --name no_pred
```

**Hypothesis**: With predictions should converge faster (better latent space).

---

## Implementation as Task File

Create **TASK_05_PREDICTION_HEADS.md** for Gemini:

### Priority: MEDIUM (After TASK_03)
### Estimated effort: 6 hours

**Steps**:
1. Add prediction heads to `MuZeroNetworks`
2. Implement `compute_prediction_targets()`
3. Update loss function with prediction losses
4. Modify replay buffer to store targets
5. Test on Level 0 (sine waves - should predict perfectly)
6. Run ablation study (with vs without predictions)

**Acceptance Criteria**:
- Price prediction MSE < 0.01 on Level 0 (sine waves)
- Regime accuracy > 75% on Level 3 (trends)
- RL performance doesn't degrade (with predictions ≥ without)

---

## Advanced: Latent Dynamics Model (Future Extension)

**Current**: Predict prices directly from latent state.

**Advanced**: Learn dynamics model in latent space (like MuZero).

```
Latent_t → Dynamics Model → Latent_{t+1}
                               ↓
                          Price Decoder → Predicted Prices
```

**Benefits**:
- Can predict arbitrarily far into future
- Can use for model-based planning
- Closer to true MuZero architecture

**Implementation**: This would be TASK_07 (after adversarial training).

---

## Summary

**Current Problem**: Model has no explicit prediction - latent space only encodes "what to do."

**Solution**: Add prediction heads (price, volatility, regime) that share latent space.

**Benefits**:
1. Richer latent representations → better RL
2. Can evaluate market understanding
3. Multi-task learning improves generalization
4. Predictions useful for risk management

**Implementation**: Add as TASK_05 after model scaling (TASK_03).

**Trade-off**: +10% training time, but should improve convergence speed and final performance.

Want me to create the full TASK_05 file for Gemini?
