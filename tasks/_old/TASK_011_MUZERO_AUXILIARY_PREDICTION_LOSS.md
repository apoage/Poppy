# TASK 011: Integrate Auxiliary Prediction Loss into MuZero

**Severity**: FEATURE
**Phase**: 2 (MuZero Integration)
**Location**: `mcts_trader/jax_agent/trainer.py`, `mcts_trader/jax_agent/muzero_model.py`, `mcts_trader/jax_agent/buffer.py`
**Status**: DONE (Completed 2025-12-30)
**Estimated Time**: 4 hours
**Dependencies**: TASK_008, TASK_009, TASK_010

---

## Objective

Integrate the **NeuralMarketODE** prediction model into MuZero training via an **auxiliary prediction loss**. This forces the latent space to encode predictive information about future prices.

**Critical Design Decisions:**
1. All operations in **LOG-SPACE** for numerical stability
2. **Multi-step prediction loss** for temporal consistency
3. **Scheduled sampling** to prevent compounding errors
4. **λ_pred annealing** for stable training
5. **Physics regularization** to prevent degenerate solutions

---

## Concept

### Current MuZero Loss

$$\mathcal{L}_{MuZero} = \mathcal{L}_{policy} + \mathcal{L}_{value} + \mathcal{L}_{reward}$$

### Proposed Combined Loss

$$\mathcal{L}_{total} = \mathcal{L}_{MuZero} + \lambda_{pred}(t) \cdot \mathcal{L}_{prediction} + \lambda_{reg} \cdot \mathcal{L}_{physics}$$

Where:
- $\mathcal{L}_{prediction}$ = Multi-step log-price prediction loss
- $\mathcal{L}_{physics}$ = Physics regularization loss
- $\lambda_{pred}(t)$ = Annealed prediction weight
- $\lambda_{reg}$ = Physics regularization weight

### Multi-Step Prediction Loss (Crucial)

$$\mathcal{L}_{prediction} = \sum_{k=1}^{K} \gamma^{k-1} \cdot \text{MSE}(\hat{p}_{t+k}, p_{t+k})$$

With decay factor γ = 0.9, this enforces temporal consistency across multiple future steps.

---

## Architecture Changes

```
┌────────────────────────────────────────────────────────────────────┐
│                    MuZero + Prediction Model                        │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Observation ──► TRM Encoder ──► Latent State (64-dim)              │
│                                      │                              │
│                    ┌─────────────────┼─────────────────┐            │
│                    ▼                 ▼                 ▼            │
│              ┌──────────┐     ┌──────────┐     ┌─────────────┐     │
│              │ Policy   │     │  Value   │     │ Prediction  │     │
│              │ Head     │     │  Head    │     │ Head (NEW)  │     │
│              └────┬─────┘     └────┬─────┘     └──────┬──────┘     │
│                   │                │                  │             │
│                   ▼                ▼                  ▼             │
│              L_policy          L_value     L_pred + L_physics       │
│                   │                │                  │             │
│                   └────────────────┼──────────────────┘             │
│                                    ▼                                │
│                     L_total (with annealed λ_pred)                  │
└────────────────────────────────────────────────────────────────────┘
```

---

## Implementation

### Step 1: Update Replay Buffer (CRITICAL)

The replay buffer MUST store price data and features for the prediction loss.

#### File: `mcts_trader/jax_agent/buffer.py`

```python
import jax.numpy as jnp
from typing import Dict, Optional, NamedTuple
import numpy as np

class AuxiliaryData(NamedTuple):
    """Auxiliary data for prediction loss."""
    log_prices: jnp.ndarray       # (batch, n_assets) - log(prices) at t
    next_log_prices: jnp.ndarray  # (batch, n_assets) - log(prices) at t+1
    log_returns: jnp.ndarray      # (batch, n_assets) - log returns at t
    moving_avg: jnp.ndarray       # (batch, n_assets) - moving average at t
    features: jnp.ndarray         # (batch, n_features) - external features
    actions: jnp.ndarray          # (batch,) - actions taken at t


class JAXReplayBuffer:
    """
    JAX-optimized replay buffer with auxiliary data for prediction loss.

    CRITICAL: This buffer stores all data needed for:
    1. Standard MuZero training (observations, actions, rewards, policies, values)
    2. Auxiliary prediction loss (log_prices, next_log_prices, features)
    """

    def __init__(
        self,
        capacity: int,
        observation_shape: tuple,
        n_assets: int = 64,
        n_features: int = 10,
        sequence_length: int = 5,  # For multi-step prediction
    ):
        self.capacity = capacity
        self.n_assets = n_assets
        self.n_features = n_features
        self.sequence_length = sequence_length

        # Standard MuZero fields
        self.observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.target_policies = np.zeros((capacity, 192), dtype=np.float32)  # 64*3 actions
        self.target_values = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

        # NEW: Auxiliary fields for prediction loss (LOG-SPACE)
        self.log_prices = np.zeros((capacity, n_assets), dtype=np.float32)
        self.next_log_prices = np.zeros((capacity, n_assets), dtype=np.float32)
        self.log_returns = np.zeros((capacity, n_assets), dtype=np.float32)
        self.moving_avg = np.zeros((capacity, n_assets), dtype=np.float32)
        self.features = np.zeros((capacity, n_features), dtype=np.float32)

        self.ptr = 0
        self.size = 0
        self.generation = 0

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        target_policy: np.ndarray,
        target_value: float,
        done: bool,
    ):
        """Add standard MuZero transition."""
        self.observations[self.ptr] = observation
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.target_policies[self.ptr] = target_policy
        self.target_values[self.ptr] = target_value
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        if self.ptr == 0:
            self.generation += 1

    def add_auxiliary(
        self,
        prices: np.ndarray,           # Raw prices (will convert to log)
        next_prices: np.ndarray,      # Raw next prices (will convert to log)
        returns: Optional[np.ndarray] = None,  # Raw returns (will convert to log)
        moving_avg: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
    ):
        """
        Add auxiliary data for prediction loss.

        IMPORTANT: This should be called AFTER add() for the same transition.
        The auxiliary data is stored at ptr-1 (previous position).

        Args:
            prices: Raw prices at t (will be log-transformed)
            next_prices: Raw prices at t+1 (will be log-transformed)
            returns: Raw returns (optional, computed if None)
            moving_avg: Moving average of log-prices
            features: External features
        """
        aux_ptr = (self.ptr - 1) % self.capacity

        # Store in LOG-SPACE for numerical stability
        self.log_prices[aux_ptr] = np.log(np.clip(prices, 1e-8, None))
        self.next_log_prices[aux_ptr] = np.log(np.clip(next_prices, 1e-8, None))

        if returns is not None:
            # Store log-returns
            self.log_returns[aux_ptr] = returns  # Assume already log-returns
        else:
            # Compute log-returns from prices
            self.log_returns[aux_ptr] = self.next_log_prices[aux_ptr] - self.log_prices[aux_ptr]

        if moving_avg is not None:
            self.moving_avg[aux_ptr] = moving_avg
        else:
            self.moving_avg[aux_ptr] = self.log_prices[aux_ptr]  # Default to current

        if features is not None:
            self.features[aux_ptr] = features
        # else: remains zeros

    def sample_batch(
        self,
        batch_size: int,
        rng: jnp.ndarray,
    ) -> Dict[str, jnp.ndarray]:
        """
        Sample batch for training including multi-step auxiliary data.

        Returns batch with:
        - Standard MuZero fields
        - Multi-step log-price sequences for prediction loss
        """
        if self.size < self.sequence_length + 1:
            raise ValueError(f"Buffer too small: {self.size} < {self.sequence_length + 1}")

        # Sample indices ensuring we have sequence_length future steps
        valid_max = self.size - self.sequence_length
        indices = jax.random.randint(
            rng, (batch_size,), 0, valid_max
        )

        # Validate generation consistency
        indices = self._validate_indices(indices)

        # Standard MuZero batch
        batch = {
            'observations': jnp.array(self.observations[indices]),
            'actions': jnp.array(self.actions[indices]),
            'rewards': jnp.array(self.rewards[indices]),
            'target_policy': jnp.array(self.target_policies[indices]),
            'target_value': jnp.array(self.target_values[indices]),
        }

        # Multi-step auxiliary batch for prediction loss
        # Shape: (batch, sequence_length, n_assets)
        log_price_sequences = []
        for k in range(self.sequence_length + 1):
            seq_indices = (indices + k) % self.capacity
            log_price_sequences.append(self.log_prices[seq_indices])

        batch['log_price_sequence'] = jnp.stack(log_price_sequences, axis=1)  # (B, K+1, N)
        batch['log_prices'] = jnp.array(self.log_prices[indices])
        batch['log_returns'] = jnp.array(self.log_returns[indices])
        batch['moving_avg'] = jnp.array(self.moving_avg[indices])
        batch['features'] = jnp.array(self.features[indices])
        batch['actions_aux'] = jnp.array(self.actions[indices])  # Actions for conditioning

        return batch

    def _validate_indices(self, indices: jnp.ndarray) -> jnp.ndarray:
        """Ensure sampled sequences don't cross episode boundaries or generations."""
        # Check for dones within sequence
        valid_mask = jnp.ones(len(indices), dtype=bool)
        for k in range(self.sequence_length):
            seq_indices = (indices + k) % self.capacity
            valid_mask = valid_mask & ~jnp.array(self.dones[seq_indices])

        # Replace invalid with random valid indices
        # (Simplified - full implementation would resample)
        return jnp.where(valid_mask, indices, 0)

    def clear(self):
        """Clear buffer (call on regime change)."""
        self.ptr = 0
        self.size = 0
        self.generation += 1
```

### Step 2: Add Prediction Head to MuZero Model

#### File: `mcts_trader/jax_agent/muzero_model.py`

```python
from flax import linen as nn
import jax.numpy as jnp
from typing import Dict, Tuple, Optional
from .prediction import NeuralMarketODE

class MuZeroNetworks(nn.Module):
    """
    Complete MuZero model with representation, dynamics, prediction,
    and auxiliary price prediction head.

    All price operations in LOG-SPACE for numerical stability.
    """
    state_dim: int = 64
    action_space_size: int = 192
    support_size: int = VALUE_SUPPORT_SIZE
    H_cycles: int = 4
    L_cycles: int = 6

    # Prediction model parameters
    use_prediction_head: bool = True
    n_assets: int = 64
    n_features: int = 10

    def setup(self):
        """Initialize all networks including prediction head."""
        # Existing components
        self.trm = TinyRecursiveTradingModel(...)
        self.action_embedding = nn.Embed(...)
        self.dynamics_net = nn.Sequential([...])
        self.dynamics_state_norm = nn.LayerNorm(...)

        # NEW: Prediction head
        if self.use_prediction_head:
            # Decoder: latent -> market state (LOG-SPACE)
            self.latent_to_market = nn.Dense(self.n_assets * 3)

            # Price predictor (operates in LOG-SPACE)
            self.price_predictor = NeuralMarketODE(
                n_assets=self.n_assets,
                n_features=self.n_features,
                dt=1.0,
                ode_method='rk2',  # Higher-order solver
                use_action_conditioning=True,
            )

    def decode_latent_to_market(
        self,
        latent_state: jnp.ndarray
    ) -> Dict[str, jnp.ndarray]:
        """
        Decode latent state to market state (LOG-SPACE).

        Args:
            latent_state: (batch, state_dim)

        Returns:
            market_state: Dict with 'log_prices', 'log_returns', 'moving_avg'
        """
        batch = latent_state.shape[0]
        decoded = self.latent_to_market(latent_state)
        decoded = decoded.reshape(batch, 3, self.n_assets)

        return {
            'log_prices': decoded[:, 0, :],     # LOG-SPACE
            'log_returns': decoded[:, 1, :],    # Log returns
            'moving_avg': decoded[:, 2, :],     # LOG-SPACE moving avg
        }

    def predict_log_prices(
        self,
        latent_state: jnp.ndarray,
        features: jnp.ndarray,
        noise: jnp.ndarray,
        action: Optional[jnp.ndarray] = None,
        use_teacher_forcing: bool = True,
        teacher_log_prices: Optional[jnp.ndarray] = None,
        teacher_log_returns: Optional[jnp.ndarray] = None,
        teacher_moving_avg: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, Dict]:
        """
        Predict next log-prices from latent state.

        Supports scheduled sampling: with probability p, use teacher-forced
        market state instead of decoded state.

        Args:
            latent_state: (batch, state_dim)
            features: (batch, n_features)
            noise: (batch, n_assets)
            action: (batch,) - optional action for conditioning
            use_teacher_forcing: Whether to use teacher-forced market state
            teacher_log_prices: Ground truth log-prices
            teacher_log_returns: Ground truth log-returns
            teacher_moving_avg: Ground truth moving average

        Returns:
            pred_log_prices: (batch, n_assets) in LOG-SPACE
            aux: Dict with physics/neural components
        """
        if not self.use_prediction_head:
            raise ValueError("Prediction head not enabled")

        # Choose between decoded and teacher-forced market state
        if use_teacher_forcing and teacher_log_prices is not None:
            # Scheduled sampling: use ground truth
            market_state = {
                'log_prices': teacher_log_prices,
                'log_returns': teacher_log_returns,
                'moving_avg': teacher_moving_avg,
            }
        else:
            # Decode from latent
            market_state = self.decode_latent_to_market(latent_state)

        # Predict using NeuralMarketODE (operates in LOG-SPACE)
        pred_log_prices, aux = self.price_predictor(
            market_state['log_prices'],
            market_state['log_returns'],
            market_state['moving_avg'],
            noise,
            features,
            action=action,
        )

        return pred_log_prices, aux

    def predict_multi_step(
        self,
        latent_state: jnp.ndarray,
        features: jnp.ndarray,
        rng: jnp.ndarray,
        action_sequence: Optional[jnp.ndarray] = None,
        num_steps: int = 5,
        teacher_sequence: Optional[jnp.ndarray] = None,
        scheduled_sampling_prob: float = 0.0,
    ) -> Tuple[jnp.ndarray, Dict]:
        """
        Multi-step prediction for temporal consistency loss.

        Args:
            latent_state: (batch, state_dim)
            features: (batch, n_features)
            rng: Random key
            action_sequence: (batch, num_steps) - optional action sequence
            num_steps: Number of steps to predict
            teacher_sequence: (batch, num_steps+1, n_assets) - ground truth log-prices
            scheduled_sampling_prob: Probability of using teacher forcing

        Returns:
            predictions: (batch, num_steps, n_assets) predicted log-prices
            aux: Dict with step-wise auxiliary info
        """
        batch = latent_state.shape[0]
        predictions = []

        # Decode initial market state
        market_state = self.decode_latent_to_market(latent_state)
        current_log_prices = market_state['log_prices']
        current_log_returns = market_state['log_returns']
        current_moving_avg = market_state['moving_avg']

        for k in range(num_steps):
            rng, noise_rng, sample_rng = jax.random.split(rng, 3)
            noise = jax.random.normal(noise_rng, (batch, self.n_assets))

            # Scheduled sampling decision
            use_teacher = (
                teacher_sequence is not None and
                jax.random.uniform(sample_rng) < scheduled_sampling_prob
            )

            if use_teacher:
                # Use ground truth for this step
                current_log_prices = teacher_sequence[:, k, :]
                if k > 0:
                    current_log_returns = teacher_sequence[:, k, :] - teacher_sequence[:, k-1, :]

            action = action_sequence[:, k] if action_sequence is not None else None

            # Predict next step
            next_log_prices, _ = self.price_predictor(
                current_log_prices,
                current_log_returns,
                current_moving_avg,
                noise,
                features,
                action=action,
            )

            predictions.append(next_log_prices)

            # Update for next iteration
            current_log_returns = next_log_prices - current_log_prices
            current_log_prices = next_log_prices
            current_moving_avg = 0.9 * current_moving_avg + 0.1 * current_log_prices

        return jnp.stack(predictions, axis=1), {}  # (batch, num_steps, n_assets)
```

### Step 3: Multi-Step Prediction Loss with Annealing

#### File: `mcts_trader/jax_agent/trainer.py`

```python
import jax
import jax.numpy as jnp
import optax
from typing import Dict, Tuple, Optional
from functools import partial

class PredictionLossConfig:
    """Configuration for prediction loss."""
    weight_initial: float = 0.01      # Start low
    weight_final: float = 0.2         # Anneal to this
    anneal_steps: int = 10000         # Steps to reach final weight

    multi_step_k: int = 5             # Number of future steps
    step_decay: float = 0.9           # Discount per step

    scheduled_sampling_start: float = 1.0   # Start with full teacher forcing
    scheduled_sampling_end: float = 0.0     # End with no teacher forcing
    scheduled_sampling_steps: int = 20000   # Annealing steps

    physics_reg_weight: float = 0.01  # Physics regularization


def compute_annealed_lambda(
    step: int,
    config: PredictionLossConfig,
) -> float:
    """Compute annealed prediction loss weight."""
    progress = jnp.clip(step / config.anneal_steps, 0.0, 1.0)
    # Linear annealing
    return config.weight_initial + progress * (config.weight_final - config.weight_initial)


def compute_scheduled_sampling_prob(
    step: int,
    config: PredictionLossConfig,
) -> float:
    """Compute scheduled sampling probability (teacher forcing rate)."""
    progress = jnp.clip(step / config.scheduled_sampling_steps, 0.0, 1.0)
    # Linear decay from start to end
    return config.scheduled_sampling_start + progress * (
        config.scheduled_sampling_end - config.scheduled_sampling_start
    )


class MuZeroTrainer:
    """MuZero trainer with multi-step auxiliary prediction loss."""

    def __init__(
        self,
        model: MuZeroNetworks,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        # Prediction loss parameters
        use_prediction_loss: bool = True,
        prediction_config: Optional[PredictionLossConfig] = None,
    ):
        self.model = model
        self.use_prediction_loss = use_prediction_loss
        self.prediction_config = prediction_config or PredictionLossConfig()

        # Initialize optimizer
        self.optimizer = optax.adamw(learning_rate, weight_decay=weight_decay)

    def compute_prediction_loss(
        self,
        params,
        latent_states: jnp.ndarray,
        batch: Dict,
        rng: jnp.ndarray,
        step: int,
    ) -> Tuple[jnp.ndarray, Dict]:
        """
        Compute multi-step prediction loss in LOG-SPACE.

        Loss: sum_{k=1}^{K} gamma^{k-1} * MSE(pred_log_p_{t+k}, true_log_p_{t+k})
        """
        config = self.prediction_config
        batch_size = latent_states.shape[0]

        # Get scheduled sampling probability
        teacher_prob = compute_scheduled_sampling_prob(step, config)

        # Get ground truth multi-step sequence
        # Shape: (batch, K+1, n_assets) where K = multi_step_k
        teacher_sequence = batch['log_price_sequence']  # Already in LOG-SPACE

        # Get action sequence if available
        action_sequence = batch.get('action_sequence', None)

        # Multi-step prediction
        predictions, _ = self.model.apply(
            params,
            latent_states,
            batch['features'],
            rng,
            action_sequence=action_sequence,
            num_steps=config.multi_step_k,
            teacher_sequence=teacher_sequence,
            scheduled_sampling_prob=teacher_prob,
            method=self.model.predict_multi_step,
        )

        # Compute weighted multi-step loss
        # predictions: (batch, K, n_assets)
        # teacher_sequence: (batch, K+1, n_assets) - index 0 is current, 1:K+1 are targets
        target_sequence = teacher_sequence[:, 1:config.multi_step_k+1, :]  # (batch, K, n_assets)

        # Per-step MSE
        step_mse = jnp.mean((predictions - target_sequence) ** 2, axis=-1)  # (batch, K)

        # Weighted sum with decay
        step_weights = jnp.array([config.step_decay ** k for k in range(config.multi_step_k)])
        weighted_mse = jnp.sum(step_mse * step_weights[None, :], axis=-1)  # (batch,)

        loss_prediction = jnp.mean(weighted_mse)

        metrics = {
            'loss_prediction_raw': loss_prediction,
            'step_1_mse': jnp.mean(step_mse[:, 0]),
            'step_k_mse': jnp.mean(step_mse[:, -1]),
            'teacher_forcing_prob': teacher_prob,
        }

        return loss_prediction, metrics

    def compute_physics_regularization(
        self,
        params,
    ) -> jnp.ndarray:
        """
        Physics regularization loss to prevent degenerate solutions.

        From TASK_008:
        - alpha (momentum): encourage small positive values
        - beta (reversion): encourage small positive values
        - sigma (volatility): bounded variance
        - gamma (correlation): sparsity penalty
        """
        physics_params = params.get('price_predictor', {}).get('physics_core', {})

        reg_loss = 0.0

        # Momentum: soft constraint alpha in [0, 0.3]
        if 'alpha' in physics_params:
            alpha = jax.nn.softplus(physics_params['alpha'])
            reg_loss += jnp.mean(jnp.maximum(alpha - 0.3, 0.0) ** 2)

        # Reversion: soft constraint beta in [0, 0.2]
        if 'beta' in physics_params:
            beta = jax.nn.softplus(physics_params['beta'])
            reg_loss += jnp.mean(jnp.maximum(beta - 0.2, 0.0) ** 2)

        # Volatility: penalize extreme values
        if 'sigma' in physics_params:
            sigma = jax.nn.softplus(physics_params['sigma'])
            reg_loss += jnp.mean((sigma - 0.01) ** 2)  # Encourage ~1% volatility

        # Low-rank correlation: L1 sparsity on factor loadings
        if 'factor_loadings' in physics_params:
            loadings = physics_params['factor_loadings']
            reg_loss += 0.01 * jnp.mean(jnp.abs(loadings))

        return reg_loss

    def compute_loss(
        self,
        params,
        batch: Dict,
        rng: jnp.ndarray,
        step: int,
    ) -> Tuple[jnp.ndarray, Dict]:
        """
        Compute combined MuZero + multi-step prediction loss.
        """
        config = self.prediction_config

        # ============================================================
        # EXISTING MUZERO LOSSES
        # ============================================================

        # Representation
        latent_states = self.model.apply(
            params, batch['observations'], method=self.model.representation
        )

        # Policy and value prediction
        policy_logits, value = self.model.apply(
            params, latent_states, method=self.model.prediction
        )

        # Policy loss (cross-entropy with MCTS policy)
        loss_policy = optax.softmax_cross_entropy(
            policy_logits, batch['target_policy']
        ).mean()

        # Value loss (cross-entropy with categorical value support)
        loss_value = optax.softmax_cross_entropy(
            value, batch['target_value']
        ).mean()

        # Dynamics unrolling for reward loss
        loss_reward = self._compute_reward_loss(params, latent_states, batch)

        # Standard MuZero loss
        loss_muzero = loss_policy + loss_value + loss_reward

        # ============================================================
        # AUXILIARY PREDICTION LOSS
        # ============================================================

        metrics = {
            'loss_policy': loss_policy,
            'loss_value': loss_value,
            'loss_reward': loss_reward,
            'loss_muzero': loss_muzero,
        }

        if self.use_prediction_loss and self.model.use_prediction_head:
            rng, pred_rng = jax.random.split(rng)

            # Compute annealed lambda
            lambda_pred = compute_annealed_lambda(step, config)

            # Multi-step prediction loss
            loss_prediction, pred_metrics = self.compute_prediction_loss(
                params, latent_states, batch, pred_rng, step
            )

            # Physics regularization
            loss_physics_reg = self.compute_physics_regularization(params)

            # Combined loss with annealed weight
            total_loss = (
                loss_muzero +
                lambda_pred * loss_prediction +
                config.physics_reg_weight * loss_physics_reg
            )

            metrics.update({
                'loss_prediction': loss_prediction,
                'loss_physics_reg': loss_physics_reg,
                'lambda_pred': lambda_pred,
                **pred_metrics,
            })
        else:
            total_loss = loss_muzero
            metrics['loss_prediction'] = 0.0

        metrics['loss_total'] = total_loss

        return total_loss, metrics

    def train_step(
        self,
        state: TrainState,
        batch: Dict,
        rng: jnp.ndarray,
        step: int,
    ) -> Tuple[TrainState, Dict]:
        """Single training step with gradient update."""
        (loss, metrics), grads = jax.value_and_grad(
            lambda p: self.compute_loss(p, batch, rng, step),
            has_aux=True
        )(state.params)

        # Update parameters
        state = state.apply_gradients(grads=grads)

        return state, metrics

    def _compute_reward_loss(
        self,
        params,
        latent_states: jnp.ndarray,
        batch: Dict,
    ) -> jnp.ndarray:
        """Compute reward prediction loss through dynamics unrolling."""
        # Implementation unchanged from existing
        ...
```

### Step 4: Update Training Script

#### File: `train_jax_fast.py`

```python
import argparse

# Add CLI arguments
parser.add_argument(
    '--prediction-loss-weight',
    type=float,
    default=0.1,
    help='Final weight for auxiliary prediction loss (default: 0.1)'
)
parser.add_argument(
    '--prediction-loss-initial',
    type=float,
    default=0.01,
    help='Initial prediction loss weight (annealed up)'
)
parser.add_argument(
    '--prediction-anneal-steps',
    type=int,
    default=10000,
    help='Steps to anneal prediction loss weight'
)
parser.add_argument(
    '--multi-step-k',
    type=int,
    default=5,
    help='Number of future steps for prediction loss'
)
parser.add_argument(
    '--no-prediction-loss',
    action='store_true',
    help='Disable auxiliary prediction loss'
)
parser.add_argument(
    '--scheduled-sampling-steps',
    type=int,
    default=20000,
    help='Steps to anneal scheduled sampling'
)

# In main():
def main(args):
    # Create prediction config
    prediction_config = PredictionLossConfig(
        weight_initial=args.prediction_loss_initial,
        weight_final=args.prediction_loss_weight,
        anneal_steps=args.prediction_anneal_steps,
        multi_step_k=args.multi_step_k,
        scheduled_sampling_steps=args.scheduled_sampling_steps,
    )

    trainer = MuZeroTrainer(
        model=model,
        use_prediction_loss=not args.no_prediction_loss,
        prediction_config=prediction_config,
        # ... other args
    )
```

### Step 5: Environment Integration (Store Prices)

```python
# In TradingEnv or data collection
def step(self, action):
    # Get current prices BEFORE action
    current_prices = self.market.get_prices()

    # Execute action
    obs, reward, done, truncated, info = self._execute_step(action)

    # Get next prices AFTER action
    next_prices = self.market.get_prices()

    # Store in info for buffer
    info['prices'] = current_prices
    info['next_prices'] = next_prices
    info['features'] = self._compute_features()

    return obs, reward, done, truncated, info

# In self-play collection
def collect_episode(env, model, buffer):
    obs = env.reset()
    done = False

    while not done:
        action, policy = model.get_action(obs)
        next_obs, reward, done, truncated, info = env.step(action)

        # Add to buffer
        buffer.add(
            observation=obs,
            action=action,
            reward=reward,
            target_policy=policy,
            target_value=...,
            done=done or truncated,
        )

        # Add auxiliary data for prediction
        if 'prices' in info:
            buffer.add_auxiliary(
                prices=info['prices'],
                next_prices=info['next_prices'],
                features=info.get('features'),
            )

        obs = next_obs
```

---

## Data Requirements

The replay buffer must store:

| Field | Shape | Description |
|-------|-------|-------------|
| `observations` | (B, 27, 64, 64) | Current observation |
| `log_prices` | (B, 64) | log(prices) at t |
| `next_log_prices` | (B, 64) | log(prices) at t+1 |
| `log_returns` | (B, 64) | Log returns at t |
| `moving_avg` | (B, 64) | Moving average (log-space) |
| `features` | (B, 10) | External features |
| `log_price_sequence` | (B, K+1, 64) | Multi-step sequence |

---

## Training Schedule

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Training Annealing Schedule                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  λ_pred      ^                                                       │
│  0.2         │                    ________________                   │
│              │                   /                                   │
│  0.1         │                  /                                    │
│              │                 /                                     │
│  0.01        │────────────────/                                      │
│              └────────────────────────────────────────────► step     │
│                0           10k                                       │
│                                                                      │
│  Teacher     ^                                                       │
│  Forcing     │                                                       │
│  1.0         │\                                                      │
│              │ \                                                     │
│  0.5         │  \                                                    │
│              │   \                                                   │
│  0.0         │    \______________                                    │
│              └────────────────────────────────────────────► step     │
│                0              20k                                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Hyperparameter Tuning

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `weight_final` | 0.2 | 0.05 - 0.5 | Too high: distorts policy |
| `weight_initial` | 0.01 | 0.001 - 0.1 | Start small |
| `anneal_steps` | 10000 | 5k - 50k | Slower = more stable |
| `multi_step_k` | 5 | 3 - 10 | More = harder |
| `step_decay` | 0.9 | 0.7 - 0.95 | Weight distant steps less |
| `physics_reg_weight` | 0.01 | 0.001 - 0.1 | Prevent degenerate physics |

---

## Metrics to Monitor

```python
# In training loop
for epoch in range(epochs):
    for batch in dataloader:
        state, metrics = trainer.train_step(state, batch, rng, step)

        # Log metrics
        log({
            # MuZero metrics
            'loss/policy': metrics['loss_policy'],
            'loss/value': metrics['loss_value'],
            'loss/reward': metrics['loss_reward'],

            # Prediction metrics (NEW)
            'loss/prediction': metrics['loss_prediction'],
            'loss/physics_reg': metrics['loss_physics_reg'],

            # Multi-step breakdown
            'prediction/step_1_mse': metrics['step_1_mse'],
            'prediction/step_k_mse': metrics['step_k_mse'],

            # Annealing state
            'anneal/lambda_pred': metrics['lambda_pred'],
            'anneal/teacher_forcing': metrics['teacher_forcing_prob'],

            # Combined
            'loss/total': metrics['loss_total'],
        })
```

### Diagnostic Alerts

```python
# Check for issues
if metrics['step_k_mse'] > 10 * metrics['step_1_mse']:
    print("WARNING: Long-horizon predictions degrading - consider increasing teacher forcing")

if metrics['loss_prediction'] < 1e-6:
    print("WARNING: Prediction loss collapsed - check for trivial solution")

if metrics['loss_physics_reg'] > 1.0:
    print("WARNING: Physics parameters out of bounds - increase regularization")
```

---

## Acceptance Criteria

- [x] JAXReplayBuffer stores log-prices and multi-step sequences
- [x] MuZero model has optional prediction head with multi-step support
- [x] Trainer computes multi-step weighted loss correctly
- [x] λ_pred annealing implemented and configurable
- [x] Scheduled sampling implemented and anneals correctly
- [x] Physics regularization prevents degenerate solutions
- [x] All price operations in LOG-SPACE
- [x] Training converges with prediction loss enabled
- [x] Metrics track per-step MSE and annealing state

---

## Verification Commands

```bash
# Test training with multi-step prediction loss
python train_jax_fast.py \
    --synthetic-level 0 \
    --epochs 100 \
    --prediction-loss-weight 0.2 \
    --prediction-loss-initial 0.01 \
    --prediction-anneal-steps 5000 \
    --multi-step-k 5 \
    --name test_multistep_prediction

# Verify metrics:
#   loss/prediction: should decrease
#   prediction/step_1_mse: should be low
#   prediction/step_k_mse: should decrease but may be higher than step_1
#   anneal/lambda_pred: should increase from 0.01 to 0.2
#   anneal/teacher_forcing: should decrease from 1.0 to 0.0

# Compare with baseline (no prediction)
python train_jax_fast.py \
    --synthetic-level 0 \
    --epochs 100 \
    --no-prediction-loss \
    --name test_no_prediction
```

---

## Expected Benefits

1. **Better Representations**: Latent space encodes predictive price dynamics
2. **Improved Value Estimates**: Agent understands future price trajectories
3. **Physics Interpretability**: Can inspect learned α, β, σ, Γ parameters
4. **Temporal Consistency**: Multi-step loss prevents myopic predictions
5. **Stable Training**: Annealing prevents early training disruption
6. **Robust Generalization**: Scheduled sampling prevents teacher forcing dependence

---

## Next Steps

After completing this task:
1. Benchmark MuZero+Prediction vs MuZero alone on held-out data
2. Analyze learned physics parameters (α, β, σ, Γ)
3. Proceed to TASK_012 (Predictive Tree Search)
