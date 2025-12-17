# TASK 05: Add Prediction Heads to Latent Space

## Objective
Add explicit prediction heads (price, volatility, regime) to force latent space to encode future market states, not just action values. Enable multi-task learning to improve RL performance and model interpretability.

## Prerequisites
- TASK_02 completed (synthetic data working)
- TASK_03 completed (model scaling available)

## Problem Statement

**Current Architecture**:
```
Observation → Encoder → Latent (256d) → Policy + Value
```

**Gap**: Latent encodes "what to do" but NOT "what will happen"

**Solution**: Add prediction heads that share the latent space:
```
Latent (256d) → Policy + Value + Price Predictor + Volatility + Regime
```

**Benefits**:
1. Richer latent representations (multi-task learning)
2. Can evaluate market understanding (prediction accuracy)
3. Predictions useful for risk management
4. Research shows auxiliary tasks improve RL generalization

## Implementation

### Step 1: Add Prediction Heads to Model

**File**: `mcts_trader/jax_agent/muzero_model.py`

**Modify** MuZeroNetworks class:

```python
# Around line 10
class MuZeroNetworks(nn.Module):
    state_dim: int = 64
    action_space_size: int = 192
    support_size: int = 601
    model_size: str = 'small'
    enable_predictions: bool = False  # NEW: Toggle prediction heads

    def setup(self):
        # ... existing TRM setup ...

        # Existing RL heads
        self.policy_head = nn.Dense(self.action_space_size)
        self.value_head = nn.Dense(self.support_size)
        self.reward_head = nn.Dense(self.support_size)

        # NEW: Prediction heads (only if enabled)
        if self.enable_predictions:
            # Price predictor: predict % change for next 10 steps
            # Output: (batch, 64 * 10) → reshape to (batch, 64, 10)
            self.price_predictor = nn.Dense(self.state_dim * 10)

            # Volatility predictor: predict next-step volatility per asset
            # Output: (batch, 64)
            self.volatility_predictor = nn.Dense(self.state_dim)

            # Regime classifier: classify market regime (bull/bear/sideways)
            # Output: (batch, 3) logits
            self.regime_classifier = nn.Dense(3)

    def initial_inference(self, observation):
        """Initial inference from observation"""
        # Encode observation
        latent = self.trm.encode(observation)

        # Shared representation
        hidden = self.trm.representation(latent)

        # RL heads (existing)
        policy_logits = self.policy_head(hidden)
        value_logits = self.value_head(hidden)

        output = {
            'policy_logits': policy_logits,
            'value': value_logits,
            'latent_state': latent
        }

        # Prediction heads (NEW)
        if self.enable_predictions:
            # Price prediction
            price_flat = self.price_predictor(hidden)  # (batch, 64*10)
            price_pred = price_flat.reshape(-1, self.state_dim, 10)  # (batch, 64, 10)

            # Volatility prediction
            volatility_pred = self.volatility_predictor(hidden)  # (batch, 64)

            # Regime classification
            regime_logits = self.regime_classifier(hidden)  # (batch, 3)

            output.update({
                'price_prediction': price_pred,
                'volatility_prediction': volatility_pred,
                'regime_logits': regime_logits
            })

        return output

    def __call__(self, x, action=None, mode='initial_inference'):
        """Main forward pass"""
        if mode == 'initial_inference':
            return self.initial_inference(x)

        elif mode == 'recurrent_inference':
            # ... existing recurrent logic ...
            # Note: Predictions only added to initial inference (not recurrent)
            pass

        # ... rest of modes ...
```

---

### Step 2: Compute Prediction Targets

**File**: `mcts_trader/jax_agent/prediction_targets.py` (NEW)

```python
"""
Compute ground truth targets for prediction heads.
"""

import jax.numpy as jnp
import numpy as np
from typing import Dict

def compute_prediction_targets(trajectory: Dict, env) -> Dict:
    """
    Compute ground truth for prediction heads from trajectory.

    Args:
        trajectory: Dict with 'observations', 'actions', 'rewards', etc.
        env: Environment instance (to extract price data)

    Returns:
        Dict with 'price_targets', 'volatility_targets', 'regime_targets'
    """
    n_steps = len(trajectory['observations'])

    # Extract prices from environment's market data
    # Assumption: Can reconstruct prices from observation or market state
    prices = extract_prices_from_trajectory(trajectory, env)  # (n_steps, 64)

    # Price targets: % change for next 10 steps
    price_targets = compute_price_targets(prices, n_steps)

    # Volatility targets: std of next step % change
    volatility_targets = compute_volatility_targets(prices, n_steps)

    # Regime targets: classify based on next 10-step trend
    regime_targets = compute_regime_targets(prices, n_steps)

    return {
        'price_targets': price_targets,
        'volatility_targets': volatility_targets,
        'regime_targets': regime_targets
    }


def extract_prices_from_trajectory(trajectory: Dict, env) -> np.ndarray:
    """
    Extract price data from trajectory.

    For synthetic environments, we can get exact prices from market.
    For real data, extract from observation or reconstruct.

    Returns: (n_steps, 64) array of close prices
    """
    n_steps = len(trajectory['observations'])
    n_assets = 64

    prices = np.zeros((n_steps, n_assets))

    # Get prices from environment's market
    # (Assumes we can access market state)
    if hasattr(env, 'market'):
        for t in range(n_steps):
            # Reset market to step t and get prices
            # (This is simplified - actual implementation depends on env)

            # For SyntheticMarket, can directly access data
            if hasattr(env.market, 'data'):
                for i in range(n_assets):
                    symbol = f'SYN_{i:02d}'
                    if symbol in env.market.data:
                        prices[t, i] = env.market.data[symbol]['close'][t]
            else:
                # For HistoricalMarket, extract from current_data
                # This requires storing price history during episode
                # For now, use placeholder
                prices[t, :] = 100.0  # Placeholder

    return prices


def compute_price_targets(prices: np.ndarray, n_steps: int) -> np.ndarray:
    """
    Compute price prediction targets (% change for next 10 steps).

    Args:
        prices: (n_steps, 64) close prices
        n_steps: Number of steps

    Returns:
        (n_steps, 64, 10) array of % changes
    """
    n_assets = prices.shape[1]
    price_targets = np.zeros((n_steps, n_assets, 10))

    for t in range(n_steps):
        # Predict % change from t to t+1, t+2, ..., t+10
        future_steps = min(10, n_steps - t - 1)

        if future_steps > 0:
            for k in range(1, future_steps + 1):
                # % change from t to t+k
                pct_change = (prices[t+k] - prices[t]) / (prices[t] + 1e-8)
                price_targets[t, :, k-1] = pct_change

        # Steps beyond episode end remain 0 (will be masked in loss)

    return price_targets


def compute_volatility_targets(prices: np.ndarray, n_steps: int) -> np.ndarray:
    """
    Compute volatility targets (next-step % change magnitude).

    Args:
        prices: (n_steps, 64)
        n_steps: Number of steps

    Returns:
        (n_steps, 64) array of volatility (abs % change)
    """
    n_assets = prices.shape[1]
    volatility_targets = np.zeros((n_steps, n_assets))

    for t in range(n_steps - 1):
        # Next step % change
        pct_change = (prices[t+1] - prices[t]) / (prices[t] + 1e-8)

        # Use absolute value as volatility proxy
        # (Could also use rolling window std for more accurate vol)
        volatility_targets[t] = np.abs(pct_change)

    # Last step has no future data
    volatility_targets[-1] = volatility_targets[-2] if n_steps > 1 else 0

    return volatility_targets


def compute_regime_targets(prices: np.ndarray, n_steps: int) -> np.ndarray:
    """
    Compute regime classification targets (bull/bear/sideways).

    Args:
        prices: (n_steps, 64)
        n_steps: Number of steps

    Returns:
        (n_steps,) array of regime labels (0=bull, 1=bear, 2=sideways)
    """
    regime_targets = np.zeros(n_steps, dtype=np.int32)

    for t in range(n_steps):
        # Look ahead 10 steps to classify regime
        future_steps = min(10, n_steps - t - 1)

        if future_steps >= 5:
            # Average return across all assets
            future_return = (prices[t+future_steps] - prices[t]) / (prices[t] + 1e-8)
            avg_return = future_return.mean()

            # Classify: bull (>1%), bear (<-1%), sideways
            if avg_return > 0.01:
                regime_targets[t] = 0  # Bull
            elif avg_return < -0.01:
                regime_targets[t] = 1  # Bear
            else:
                regime_targets[t] = 2  # Sideways
        else:
            # Not enough future data - default sideways
            regime_targets[t] = 2

    return regime_targets
```

---

### Step 3: Update Training Loop

**File**: `mcts_trader/jax_agent/trainer.py`

**Add prediction loss** to train_step:

```python
# Around line 150 (in train_step method)

def train_step(self, batch):
    """Single training step with prediction losses"""

    def loss_fn(params):
        # ... existing RL losses (policy, value, reward) ...

        # Compute RL losses (existing code)
        outputs = self.model.apply({'params': params}, batch['observations'], mode='initial_inference')

        policy_logits = outputs['policy_logits']
        value_logits = outputs['value']

        # Policy loss (cross-entropy)
        policy_loss = -jnp.sum(batch['policy_targets'] * jax.nn.log_softmax(policy_logits))

        # Value loss (cross-entropy with support)
        value_loss = self._scalar_to_support_loss(value_logits, batch['value_targets'])

        # Reward loss (if applicable)
        reward_loss = 0.0  # Simplified

        # === NEW: Prediction Losses ===
        prediction_loss = 0.0
        losses = {}

        if self.model.enable_predictions:
            # Price prediction loss (MSE)
            price_pred = outputs['price_prediction']  # (batch, 64, 10)
            price_target = batch['price_targets']      # (batch, 64, 10)

            # Only compute loss where we have valid targets (mask future beyond episode)
            price_mask = (price_target != 0).astype(jnp.float32)  # Simple masking
            price_loss = jnp.sum((price_pred - price_target) ** 2 * price_mask) / (jnp.sum(price_mask) + 1e-8)

            # Volatility prediction loss (MSE)
            vol_pred = outputs['volatility_prediction']  # (batch, 64)
            vol_target = batch['volatility_targets']      # (batch, 64)
            volatility_loss = jnp.mean((vol_pred - vol_target) ** 2)

            # Regime classification loss (cross-entropy)
            regime_logits = outputs['regime_logits']  # (batch, 3)
            regime_target = batch['regime_targets']    # (batch,) integers

            regime_probs = jax.nn.softmax(regime_logits)
            regime_loss = -jnp.mean(
                jnp.log(regime_probs[jnp.arange(len(regime_target)), regime_target] + 1e-8)
            )

            # Combine prediction losses
            prediction_loss = price_loss + 0.5 * volatility_loss + 0.3 * regime_loss

            losses = {
                'price_loss': price_loss,
                'volatility_loss': volatility_loss,
                'regime_loss': regime_loss
            }

        # Total loss (RL + Predictions)
        total_loss = (
            1.0 * policy_loss +
            1.0 * value_loss +
            0.5 * reward_loss +
            0.1 * prediction_loss  # Weight predictions lower than RL
        )

        return total_loss, {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'reward_loss': reward_loss,
            **losses  # Add prediction losses
        }

    # Compute gradients
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(self.state.params)

    # Update parameters
    self.state = self.state.apply_gradients(grads=grads)

    return metrics
```

---

### Step 4: Update Replay Buffer

**File**: `mcts_trader/jax_agent/replay_buffer.py`

**Add storage** for prediction targets:

```python
class JAXReplayBuffer:
    def __init__(self, capacity, observation_shape, action_size, batch_size, sequence_length):
        # ... existing ...

        # NEW: Prediction target storage
        self.price_targets = []
        self.volatility_targets = []
        self.regime_targets = []

    def add_trajectory(self, observations, actions, rewards, values, policy_probs, dones,
                       price_targets=None, volatility_targets=None, regime_targets=None):  # NEW args
        """Add trajectory to buffer"""

        # ... existing storage ...

        # Store prediction targets (if provided)
        if price_targets is not None:
            self.price_targets.append(price_targets)
            self.volatility_targets.append(volatility_targets)
            self.regime_targets.append(regime_targets)

        # Keep buffer size under capacity
        if len(self.observations) > self.capacity:
            self.observations.pop(0)
            self.actions.pop(0)
            # ... existing pops ...

            # NEW: Pop prediction targets
            if len(self.price_targets) > 0:
                self.price_targets.pop(0)
                self.volatility_targets.pop(0)
                self.regime_targets.pop(0)

    def sample(self):
        """Sample batch"""
        # ... existing sampling ...

        batch = {
            'observations': sampled_obs,
            'actions': sampled_actions,
            'rewards': sampled_rewards,
            'values': sampled_values,
            'policy_targets': sampled_policy,
            'value_targets': sampled_values,
        }

        # NEW: Add prediction targets to batch (if available)
        if len(self.price_targets) > 0:
            sampled_price = jnp.array([self.price_targets[i] for i in indices])
            sampled_vol = jnp.array([self.volatility_targets[i] for i in indices])
            sampled_regime = jnp.array([self.regime_targets[i] for i in indices])

            batch.update({
                'price_targets': sampled_price,
                'volatility_targets': sampled_vol,
                'regime_targets': sampled_regime
            })

        return batch
```

---

### Step 5: Integrate into Self-Play

**File**: `train_jax_fast.py`

**Modify run_self_play** to compute prediction targets:

```python
# Around line 340
def run_self_play(env, search_fn, model_apply_fn, params, rng_key, steps=200,
                  confidence_threshold=0.0, enable_predictions=False):  # NEW arg
    """Self-play with optional prediction target computation"""

    # ... existing self-play loop ...

    trajectory = {
        'observations': np.array(observations),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'values': np.array(values),
        'policy_probs': np.array(policy_probs),
        'dones': np.array(dones)
    }

    # NEW: Compute prediction targets
    if enable_predictions:
        from mcts_trader.jax_agent.prediction_targets import compute_prediction_targets

        pred_targets = compute_prediction_targets(trajectory, env)

        # Add to trajectory
        trajectory['price_targets'] = pred_targets['price_targets']
        trajectory['volatility_targets'] = pred_targets['volatility_targets']
        trajectory['regime_targets'] = pred_targets['regime_targets']

    return trajectory, score
```

**Update main training loop**:

```python
# Around line 595
# Pass enable_predictions flag
trajectory, score = run_self_play(
    env,
    search,
    model.apply,
    trainer.state.params,
    search_rng,
    steps=args.steps,
    confidence_threshold=args.confidence_threshold,
    enable_predictions=args.enable_predictions  # NEW
)

# Add to buffer (with or without prediction targets)
if args.enable_predictions:
    replay_buffer.add_trajectory(
        jnp.array(trajectory['observations']),
        jnp.array(trajectory['actions']),
        jnp.array(trajectory['rewards']),
        jnp.array(trajectory['values']),
        jnp.array(trajectory['policy_probs']),
        jnp.array(trajectory['dones']),
        price_targets=jnp.array(trajectory['price_targets']),
        volatility_targets=jnp.array(trajectory['volatility_targets']),
        regime_targets=jnp.array(trajectory['regime_targets'])
    )
else:
    # Existing call (no prediction targets)
    replay_buffer.add_trajectory(
        jnp.array(trajectory['observations']),
        jnp.array(trajectory['actions']),
        jnp.array(trajectory['rewards']),
        jnp.array(trajectory['values']),
        jnp.array(trajectory['policy_probs']),
        jnp.array(trajectory['dones'])
    )
```

---

### Step 6: Add CLI Arguments

**File**: `train_jax_fast.py`

```python
# Around line 435
parser.add_argument('--enable-predictions', action='store_true',
                   help='Enable prediction heads (price, volatility, regime)')
```

**Pass to model**:

```python
# Around line 507
model = MuZeroNetworks(
    state_dim=n_assets,
    action_space_size=n_assets * 3,
    support_size=601,
    model_size=args.model_size,
    enable_predictions=args.enable_predictions  # NEW
)
```

**Log status**:

```python
# After model creation
if args.enable_predictions:
    print("✅ Prediction heads ENABLED (price, volatility, regime)")
else:
    print("ℹ️  Prediction heads disabled (standard RL only)")
```

---

## Testing

### Test 1: Prediction on Level 0 (Sine Waves)

**Hypothesis**: Model should predict sine waves perfectly (deterministic).

**Command**:
```bash
python train_jax_fast.py \
  --synthetic-level 0 \
  --enable-predictions \
  --model-size small \
  --epochs 1000 \
  --batch-size 256 \
  --name test_pred_level0
```

**Monitor**: `logs/training_fast_test_pred_level0.log`

**Look for**:
```
Epoch 500 | ... | price_loss: 0.0023 | volatility_loss: 0.0015 | regime_loss: 0.05
```

**Pass Criteria**:
- Price loss < 0.01 after 500 epochs
- Model is learning to predict

### Test 2: Regime Classification on Level 3

**Hypothesis**: Model should detect bull/bear trends.

**Command**:
```bash
python train_jax_fast.py \
  --synthetic-level 3 \
  --enable-predictions \
  --model-size medium \
  --epochs 2000 \
  --name test_pred_level3
```

**Evaluation Script**: `tools/evaluate_predictions.py` (NEW)

```python
import jax
import jax.numpy as jnp
import numpy as np
from train_jax_fast import FastTradingEnv, MuZeroNetworks

def evaluate_predictions(checkpoint_path, level=3):
    """Evaluate prediction accuracy"""

    # Load model
    model = MuZeroNetworks(
        state_dim=64,
        action_space_size=192,
        enable_predictions=True
    )

    # Load checkpoint
    # (Simplified - actual implementation needs proper checkpoint loading)

    # Create test environment
    env = FastTradingEnv(synthetic_level=level, n_assets=64)

    # Run 100 episodes, collect predictions
    price_errors = []
    regime_accuracy = []

    for ep in range(100):
        obs, _ = env.reset()

        for step in range(200):
            # Get model predictions
            output = model.apply(params, obs['observation'][None, ...])

            if 'price_prediction' in output:
                price_pred = output['price_prediction'][0]  # (64, 10)
                regime_pred = jnp.argmax(output['regime_logits'][0])  # Scalar

                # Get ground truth
                # (Would need to extract from environment)
                # For now, placeholder
                price_actual = np.zeros((64, 10))
                regime_actual = 0

                # Compute error
                price_error = np.mean((price_pred - price_actual) ** 2)
                regime_correct = (regime_pred == regime_actual)

                price_errors.append(price_error)
                regime_accuracy.append(regime_correct)

            obs, _, done, truncated, _ = env.step(env.action_space.sample())
            if done or truncated:
                break

    print(f"Price Prediction MSE: {np.mean(price_errors):.4f}")
    print(f"Regime Accuracy: {np.mean(regime_accuracy)*100:.1f}%")

if __name__ == "__main__":
    evaluate_predictions('jax_ckpt_fast/test_pred_level3/checkpoint_2000', level=3)
```

**Run**:
```bash
python tools/evaluate_predictions.py
```

**Pass Criteria**:
- Regime accuracy > 60% (better than random 33%)

### Test 3: Ablation Study (With vs Without Predictions)

**Question**: Do predictions help RL performance?

**Setup**:
```bash
# Without predictions (baseline)
python train_jax_fast.py \
  --synthetic-level 4 \
  --model-size medium \
  --epochs 2000 \
  --name ablation_no_pred

# With predictions
python train_jax_fast.py \
  --synthetic-level 4 \
  --enable-predictions \
  --model-size medium \
  --epochs 2000 \
  --name ablation_with_pred
```

**Compare**:
```bash
grep "Epoch 2000" logs/training_fast_ablation_*.log
```

**Hypothesis**: With predictions should:
- Converge faster (fewer epochs to pass criteria)
- Higher final reward (better generalization)

**Pass Criteria**: With predictions ≥ without predictions (at minimum, no degradation).

### Test 4: Verify No Regression

**Ensure predictions don't break existing functionality**:

```bash
# Test standard RL (no predictions) still works
python train_jax_fast.py \
  --synthetic-level 0 \
  --epochs 500 \
  --name regression_test
```

**Expected**: Converges same as before (reward > +5%).

---

## Acceptance Criteria

✅ **PASS if ALL met**:
1. Model compiles with `--enable-predictions` flag
2. Price loss decreases during training (not stuck)
3. Prediction MSE < 0.01 on Level 0 (sine waves)
4. Regime accuracy > 60% on Level 3 (trends)
5. RL performance ≥ baseline (no degradation)
6. Standard RL still works (no predictions flag)

## Performance Impact

**Expected Training Speed**:
- With predictions: ~10-15% slower (extra forward passes + loss computation)
- But should converge faster (better latent space)

**Net effect**: Similar wall-clock time to convergence, better final performance.

## Debugging

**If price loss doesn't decrease**:

1. **Check target computation**:
```python
# In run_self_play, print targets
print("Price targets shape:", trajectory['price_targets'].shape)
print("Price targets range:", trajectory['price_targets'].min(), trajectory['price_targets'].max())
```

2. **Check loss masking**:
```python
# Verify mask is correct
print("Price mask sum:", jnp.sum(price_mask))  # Should be > 0
```

3. **Simplify test**:
Run on Level 0 (sine) with just 10 epochs - loss should drop immediately.

**If regime accuracy is random (33%)**:

- Check if regime targets are balanced (not all one class)
- Verify regime classification logic (bull/bear thresholds)

## Integration Checklist

- [ ] Modify `muzero_model.py` to add prediction heads
- [ ] Create `prediction_targets.py` module
- [ ] Update `trainer.py` with prediction losses
- [ ] Update `replay_buffer.py` to store targets
- [ ] Modify `run_self_play()` to compute targets
- [ ] Add `--enable-predictions` CLI flag
- [ ] Test on Level 0 (sine waves)
- [ ] Test on Level 3 (trends)
- [ ] Run ablation study
- [ ] Verify no regression
- [ ] Commit: "Add prediction heads to latent space (multi-task learning)"

## Next Task

Once this works → **TASK_06_CROSS_PAIR_TRADING.md** or **TASK_07_ADVERSARIAL_TRAINING.md** (TBD)

## Notes

- Predictions are **optional** (controlled by flag) - can disable if issues
- Multi-task learning is proven technique (Jaderberg et al. 2016)
- Predictions useful for diagnostics even if RL performance same
- Future: Could add more auxiliary tasks (correlation prediction, volume, etc.)
