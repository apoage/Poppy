# TASK 010: Implement NeuralMarketODE (ODE Integration)

**Severity**: FEATURE
**Phase**: 1 (Core Prediction Model)
**Location**: `mcts_trader/jax_agent/prediction/market_ode.py` (NEW)
**Status**: OPEN
**Estimated Time**: 2 hours
**Dependencies**: TASK_008, TASK_009

---

## Objective

Implement **NeuralMarketODE** - the complete hybrid physics-neural prediction model that combines:
1. Differentiable Market Physics (TASK_008)
2. Neural Market Corrector (TASK_009)
3. ODE Integration (Forward Euler or RK2)
4. **Action-Conditional Dynamics** (market impact modeling)

---

## Critical Design Decisions

### 1. Log-Space Throughout

All operations happen in log-space for numerical stability:

```python
next_log_price = log_price + dt * G_total
next_price = exp(next_log_price)  # Only at output
```

### 2. Action-Conditional Dynamics (P2)

**Problem**: Prediction ignores agent's action - can't model "if I buy, price moves up"

**Solution**: Add market impact term conditioned on action:

```python
# Action encoding: which asset and direction
asset_idx = action // 3
action_type = action % 3  # 0=sell, 1=hold, 2=buy

# Create impact tensor
impact_sign = jnp.where(action_type == 0, -1.0,
              jnp.where(action_type == 2, +1.0, 0.0))
G_impact = impact_coefficient * impact_sign
```

### 3. Higher-Order ODE Solvers (P3)

**Problem**: Forward Euler accumulates error on long rollouts.

**Solution**: Implement RK2 (Midpoint method):

```python
# RK2: k1 = f(y_n), k2 = f(y_n + dt/2 * k1)
# y_{n+1} = y_n + dt * k2

k1 = G(state_t)
state_mid = state_t + (dt/2) * k1
k2 = G(state_mid)
state_next = state_t + dt * k2
```

---

## Mathematical Foundation

### ODE Integration in Log-Space

$$\log p_{t+\Delta t} = \log p_t + \Delta t \cdot G_{total}$$

Where:
$$G_{total} = \underbrace{G_{phys}}_{\text{Physics}} + \underbrace{G_{neural}}_{\text{Neural}} + \underbrace{G_{impact}}_{\text{Action}}$$

### Complete Model

```
Input: state_t = {log_prices, log_returns, log_ma}, action, features
                          │
         ┌────────────────┼────────────────┬────────────────┐
         ▼                ▼                ▼                ▼
    ┌─────────┐     ┌───────────┐    ┌──────────┐    ┌──────────┐
    │ Physics │     │  Neural   │    │  Market  │    │    ODE   │
    │  Core   │     │ Corrector │    │  Impact  │    │  Solver  │
    └────┬────┘     └─────┬─────┘    └────┬─────┘    └────┬─────┘
         │                │               │                │
         │ G_phys    G_neural        G_impact          method
         │                │               │                │
         └────────────────┼───────────────┘                │
                          │                                │
                          ▼ G_total = G_phys + G_neural + G_impact
                     ┌──────────┐
                     │ ODE Step │  log_p_new = log_p + Δt * G
                     └────┬─────┘       (Euler or RK2)
                          │
                          ▼
Output: state_{t+1} = {log_prices_new, log_returns_new, log_ma_new}
```

---

## Implementation

### File: `mcts_trader/jax_agent/prediction/market_ode.py`

```python
"""
Neural Market ODE.

Hybrid physics-neural prediction model using ODE integration.
Combines learnable physics core with neural corrector.

Key features:
- Log-space operations for numerical stability
- Action-conditional dynamics (market impact)
- RK2 ODE solver option for long-term stability

Based on NeuralOGCM (arXiv:2512.11525v1).
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Tuple, Optional, Literal
from functools import partial

from .physics_core import DifferentiableMarketPhysics
from .neural_corrector import NeuralMarketCorrector, SimpleNeuralCorrector


class MarketImpact(nn.Module):
    """
    Models how agent's actions affect market prices.

    Attributes:
        n_assets: Number of assets
        impact_decay: Decay factor for cumulative impact
    """
    n_assets: int = 64
    impact_decay: float = 0.9

    @nn.compact
    def __call__(
        self,
        action: jnp.ndarray,
        action_history: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Compute market impact from agent's action.

        Args:
            action: (batch,) - Action indices (asset_idx * 3 + action_type)
            action_history: Optional (batch, lookback, n_assets) for cumulative impact

        Returns:
            G_impact: (batch, n_assets) - Market impact tendency
        """
        batch = action.shape[0]

        # Learnable impact coefficient
        impact_coef = nn.softplus(self.param(
            'impact_coefficient',
            nn.initializers.constant(-4.0),  # ~0.02 impact
            (1,)
        ))

        # Decode action
        asset_idx = action // 3
        action_type = action % 3  # 0=sell, 1=hold, 2=buy

        # Impact direction: sell=-1, hold=0, buy=+1
        impact_sign = jnp.where(
            action_type == 0, -1.0,
            jnp.where(action_type == 2, +1.0, 0.0)
        )

        # Create per-asset impact (only affected asset moves)
        G_impact = jnp.zeros((batch, self.n_assets))
        G_impact = G_impact.at[jnp.arange(batch), asset_idx].set(
            impact_coef * impact_sign
        )

        # Add cumulative impact from history if provided
        if action_history is not None:
            lookback = action_history.shape[1]
            weights = self.impact_decay ** jnp.arange(lookback)[::-1]
            cumulative = jnp.sum(action_history * weights[None, :, None], axis=1)
            G_impact = G_impact + 0.1 * cumulative * impact_coef

        return G_impact


class NeuralMarketODE(nn.Module):
    """
    Hybrid Physics-Neural Market Prediction Model.

    Combines:
    - Differentiable physics core (momentum, reversion, volatility, correlation)
    - Neural corrector (residual dynamics)
    - Action-conditional market impact
    - Forward Euler or RK2 ODE integration

    All operations in LOG-SPACE for numerical stability.

    Attributes:
        n_assets: Number of tradeable assets
        n_features: Number of external features
        n_factors: Number of correlation factors
        dt: Time step for ODE integration
        use_simple_corrector: If True, use MLP instead of transformer
        use_action_conditioning: If True, include market impact
        ode_method: 'euler' or 'rk2'
    """
    n_assets: int = 64
    n_features: int = 10
    n_factors: int = 8
    dt: float = 1.0
    use_simple_corrector: bool = False
    use_action_conditioning: bool = True
    ode_method: Literal['euler', 'rk2'] = 'euler'

    def setup(self):
        """Initialize physics and neural components."""
        self.physics_core = DifferentiableMarketPhysics(
            n_assets=self.n_assets,
            n_factors=self.n_factors,
            dt=self.dt
        )

        if self.use_simple_corrector:
            self.neural_corrector = SimpleNeuralCorrector(
                n_assets=self.n_assets,
                n_features=self.n_features
            )
        else:
            self.neural_corrector = NeuralMarketCorrector(
                n_assets=self.n_assets,
                n_features=self.n_features
            )

        if self.use_action_conditioning:
            self.market_impact = MarketImpact(n_assets=self.n_assets)

    def compute_tendency(
        self,
        log_prices: jnp.ndarray,
        log_returns: jnp.ndarray,
        log_moving_avg: jnp.ndarray,
        noise: jnp.ndarray,
        features: Optional[jnp.ndarray] = None,
        action: Optional[jnp.ndarray] = None,
        prev_variance: Optional[jnp.ndarray] = None,
        deterministic: bool = True
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Compute total tendency G = G_phys + G_neural + G_impact.

        Args:
            log_prices: (batch, n_assets) - Current log prices
            log_returns: (batch, n_assets) - Recent log returns
            log_moving_avg: (batch, n_assets) - Log moving average
            noise: (batch, n_assets) - Sampled noise
            features: (batch, n_features) - External features (optional)
            action: (batch,) - Agent's action (optional)
            prev_variance: (batch, n_assets) - Previous variance (for GARCH)
            deterministic: If True, disable dropout

        Returns:
            G_total: (batch, n_assets) - Total tendency
            aux: Dict with individual components
        """
        # Physics-based tendency
        G_phys, phys_aux = self.physics_core(
            log_prices, log_returns, log_moving_avg, noise, prev_variance
        )

        # Neural correction
        G_neural, attn_weights = self.neural_corrector(
            log_prices, log_returns, log_moving_avg, features, deterministic
        )

        # Market impact (if action provided and enabled)
        if self.use_action_conditioning and action is not None:
            G_impact = self.market_impact(action)
        else:
            G_impact = jnp.zeros_like(G_phys)

        # Total tendency
        G_total = G_phys + G_neural + G_impact

        aux = {
            'G_phys': G_phys,
            'G_neural': G_neural,
            'G_impact': G_impact,
            'G_total': G_total,
            'attention_weights': attn_weights,
            'new_variance': phys_aux.get('new_variance'),
            **{f'phys_{k}': v for k, v in phys_aux.items()}
        }

        return G_total, aux

    def __call__(
        self,
        log_prices: jnp.ndarray,
        log_returns: jnp.ndarray,
        log_moving_avg: jnp.ndarray,
        noise: jnp.ndarray,
        features: Optional[jnp.ndarray] = None,
        action: Optional[jnp.ndarray] = None,
        prev_variance: Optional[jnp.ndarray] = None,
        deterministic: bool = True
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        One-step prediction via ODE integration in LOG-SPACE.

        Args:
            log_prices: (batch, n_assets) - Current log prices
            log_returns: (batch, n_assets) - Recent log returns
            log_moving_avg: (batch, n_assets) - Log moving average
            noise: (batch, n_assets) - Sampled noise
            features: (batch, n_features) - External features (optional)
            action: (batch,) - Agent's action (optional)
            prev_variance: (batch, n_assets) - Previous variance (for GARCH)
            deterministic: If True, disable dropout

        Returns:
            next_log_prices: (batch, n_assets) - Predicted log prices at t+dt
            aux: Dict with G_phys, G_neural, G_impact for analysis
        """
        if self.ode_method == 'euler':
            return self._euler_step(
                log_prices, log_returns, log_moving_avg, noise,
                features, action, prev_variance, deterministic
            )
        elif self.ode_method == 'rk2':
            return self._rk2_step(
                log_prices, log_returns, log_moving_avg, noise,
                features, action, prev_variance, deterministic
            )
        else:
            raise ValueError(f"Unknown ODE method: {self.ode_method}")

    def _euler_step(
        self,
        log_prices: jnp.ndarray,
        log_returns: jnp.ndarray,
        log_moving_avg: jnp.ndarray,
        noise: jnp.ndarray,
        features: Optional[jnp.ndarray],
        action: Optional[jnp.ndarray],
        prev_variance: Optional[jnp.ndarray],
        deterministic: bool
    ) -> Tuple[jnp.ndarray, Dict]:
        """Forward Euler: y_{n+1} = y_n + dt * f(y_n)"""
        G_total, aux = self.compute_tendency(
            log_prices, log_returns, log_moving_avg, noise,
            features, action, prev_variance, deterministic
        )
        next_log_prices = log_prices + self.dt * G_total
        return next_log_prices, aux

    def _rk2_step(
        self,
        log_prices: jnp.ndarray,
        log_returns: jnp.ndarray,
        log_moving_avg: jnp.ndarray,
        noise: jnp.ndarray,
        features: Optional[jnp.ndarray],
        action: Optional[jnp.ndarray],
        prev_variance: Optional[jnp.ndarray],
        deterministic: bool
    ) -> Tuple[jnp.ndarray, Dict]:
        """
        RK2 (Midpoint method):
        k1 = f(y_n)
        k2 = f(y_n + dt/2 * k1)
        y_{n+1} = y_n + dt * k2
        """
        # k1: tendency at current state
        k1, aux1 = self.compute_tendency(
            log_prices, log_returns, log_moving_avg, noise,
            features, action, prev_variance, deterministic
        )

        # Midpoint state
        log_prices_mid = log_prices + (self.dt / 2) * k1

        # Update returns and MA for midpoint (simplified)
        log_returns_mid = k1  # Instantaneous return = tendency
        log_ma_mid = log_moving_avg + 0.05 * (log_prices_mid - log_moving_avg)

        # k2: tendency at midpoint
        k2, aux2 = self.compute_tendency(
            log_prices_mid, log_returns_mid, log_ma_mid, noise,
            features, action, aux1.get('new_variance'), deterministic
        )

        # Final step
        next_log_prices = log_prices + self.dt * k2

        # Combine aux from both evaluations
        aux = {
            'G_phys': (aux1['G_phys'] + aux2['G_phys']) / 2,
            'G_neural': (aux1['G_neural'] + aux2['G_neural']) / 2,
            'G_impact': (aux1['G_impact'] + aux2['G_impact']) / 2,
            'G_total': k2,  # Final tendency used
            'new_variance': aux2.get('new_variance'),
            'k1': k1,
            'k2': k2,
        }

        return next_log_prices, aux

    def predict_step(
        self,
        state: Dict[str, jnp.ndarray],
        features: Optional[jnp.ndarray] = None,
        action: Optional[jnp.ndarray] = None,
        rng: Optional[jnp.ndarray] = None
    ) -> Dict[str, jnp.ndarray]:
        """
        Convenience method: predict next state from current state dict.

        Args:
            state: Dict with 'log_prices', 'log_returns', 'log_ma', 'variance'
            features: External features
            action: Agent's action
            rng: Random key for noise generation

        Returns:
            next_state: Dict with updated state
        """
        log_prices = state['log_prices']
        log_returns = state['log_returns']
        log_ma = state['log_ma']
        prev_variance = state.get('variance')

        # Generate noise
        if rng is not None:
            noise = jax.random.normal(rng, log_prices.shape)
        else:
            noise = jnp.zeros_like(log_prices)

        # Predict
        next_log_prices, aux = self(
            log_prices, log_returns, log_ma, noise,
            features, action, prev_variance
        )

        # Update state
        next_log_returns = next_log_prices - log_prices
        ema_alpha = 0.1
        next_log_ma = ema_alpha * next_log_prices + (1 - ema_alpha) * log_ma

        return {
            'log_prices': next_log_prices,
            'log_returns': next_log_returns,
            'log_ma': next_log_ma,
            'variance': aux.get('new_variance'),
        }


def create_rollout_fn(model: NeuralMarketODE, params, n_steps: int = 50):
    """
    Create JIT-compiled rollout function.

    Args:
        model: NeuralMarketODE instance
        params: Model parameters
        n_steps: Number of rollout steps

    Returns:
        rollout function: (init_state, features_seq, actions_seq, rng) -> trajectory
    """

    @jax.jit
    def rollout(
        init_log_prices: jnp.ndarray,
        init_log_returns: jnp.ndarray,
        init_log_ma: jnp.ndarray,
        features_seq: Optional[jnp.ndarray],
        actions_seq: Optional[jnp.ndarray],
        rng: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Multi-step rollout prediction.

        Args:
            init_log_prices: (batch, n_assets)
            init_log_returns: (batch, n_assets)
            init_log_ma: (batch, n_assets)
            features_seq: (n_steps, batch, n_features) or None
            actions_seq: (n_steps, batch) or None
            rng: Random key

        Returns:
            trajectory: (n_steps, batch, n_assets) - Log price trajectory
        """
        def step_fn(carry, inputs):
            log_prices, log_returns, log_ma, variance, step_rng = carry
            step_idx = inputs

            # Get features and action for this step
            features_t = features_seq[step_idx] if features_seq is not None else None
            action_t = actions_seq[step_idx] if actions_seq is not None else None

            # Generate noise
            step_rng, noise_rng = jax.random.split(step_rng)
            noise = jax.random.normal(noise_rng, log_prices.shape)

            # Predict
            next_log_prices, aux = model.apply(
                params, log_prices, log_returns, log_ma, noise,
                features_t, action_t, variance
            )

            # Update state
            next_log_returns = next_log_prices - log_prices
            next_log_ma = 0.1 * next_log_prices + 0.9 * log_ma
            next_variance = aux.get('new_variance', variance)

            carry = (next_log_prices, next_log_returns, next_log_ma, next_variance, step_rng)
            return carry, next_log_prices

        init_variance = None
        init_carry = (init_log_prices, init_log_returns, init_log_ma, init_variance, rng)
        _, trajectory = jax.lax.scan(step_fn, init_carry, jnp.arange(n_steps))

        return trajectory

    return rollout


def create_training_loss_fn(model: NeuralMarketODE, n_steps: int = 1, decay: float = 0.9):
    """
    Create MSE loss function for training.

    Supports multi-step prediction loss for temporal consistency.

    Args:
        model: NeuralMarketODE instance
        n_steps: Number of prediction steps (1 = single-step)
        decay: Weight decay for multi-step loss

    Returns:
        loss_fn: (params, batch) -> (loss, aux)
    """

    def single_step_loss_fn(
        params,
        log_prices: jnp.ndarray,
        log_returns: jnp.ndarray,
        log_ma: jnp.ndarray,
        noise: jnp.ndarray,
        features: jnp.ndarray,
        action: jnp.ndarray,
        target_log_prices: jnp.ndarray
    ):
        """Single-step MSE loss."""
        pred_log_prices, aux = model.apply(
            params, log_prices, log_returns, log_ma, noise, features, action
        )

        loss = jnp.mean((pred_log_prices - target_log_prices) ** 2)

        aux['pred_log_prices'] = pred_log_prices
        aux['target_log_prices'] = target_log_prices

        return loss, aux

    def multi_step_loss_fn(
        params,
        trajectory_log_prices: jnp.ndarray,  # (T+1, batch, n_assets)
        trajectory_features: jnp.ndarray,    # (T, batch, n_features)
        trajectory_actions: jnp.ndarray,     # (T, batch)
        noise_seq: jnp.ndarray              # (T, batch, n_assets)
    ):
        """
        Multi-step MSE loss with decaying weights.

        Enforces temporal consistency.
        """
        T = trajectory_features.shape[0]
        total_loss = 0.0

        # Initialize state from first timestep
        log_prices = trajectory_log_prices[0]
        log_returns = jnp.zeros_like(log_prices)
        log_ma = log_prices

        for t in range(T):
            # Target is ground truth next state
            target = trajectory_log_prices[t + 1]

            # Predict
            pred, aux = model.apply(
                params, log_prices, log_returns, log_ma, noise_seq[t],
                trajectory_features[t], trajectory_actions[t]
            )

            # Weighted loss
            weight = decay ** t
            total_loss += weight * jnp.mean((pred - target) ** 2)

            # Use GROUND TRUTH for next step (teacher forcing)
            # This avoids error accumulation during training
            log_prices = target
            log_returns = target - trajectory_log_prices[t]
            log_ma = 0.1 * target + 0.9 * log_ma

        return total_loss / T, {'n_steps': T}

    if n_steps == 1:
        return single_step_loss_fn
    else:
        return multi_step_loss_fn
```

---

## Module `__init__.py`

### File: `mcts_trader/jax_agent/prediction/__init__.py`

```python
"""
Prediction module for NeuralMarketODE.

Hybrid physics-neural price prediction model.

All operations in LOG-SPACE for numerical stability.
"""

from .physics_core import (
    DifferentiableMarketPhysics,
    LowRankCorrelation,
    GARCHVolatility,
    log_physics_params,
    physics_regularization_loss
)
from .neural_corrector import (
    NeuralMarketCorrector,
    SimpleNeuralCorrector,
    CrossAssetAttention,
    CrossAssetTransformerBlock
)
from .market_ode import (
    NeuralMarketODE,
    MarketImpact,
    create_rollout_fn,
    create_training_loss_fn
)

__all__ = [
    # Physics
    'DifferentiableMarketPhysics',
    'LowRankCorrelation',
    'GARCHVolatility',
    'log_physics_params',
    'physics_regularization_loss',
    # Neural
    'NeuralMarketCorrector',
    'SimpleNeuralCorrector',
    'CrossAssetAttention',
    'CrossAssetTransformerBlock',
    # ODE
    'NeuralMarketODE',
    'MarketImpact',
    'create_rollout_fn',
    'create_training_loss_fn',
]
```

---

## Unit Tests

```python
"""Unit tests for NeuralMarketODE."""

import jax
import jax.numpy as jnp
from mcts_trader.jax_agent.prediction import (
    NeuralMarketODE,
    create_rollout_fn,
    create_training_loss_fn
)


def test_market_ode_prediction():
    """One-step prediction works correctly."""
    model = NeuralMarketODE(n_assets=64, n_features=10)
    rng = jax.random.PRNGKey(0)

    # Work in LOG space
    prices = jax.random.uniform(rng, (32, 64), minval=100, maxval=1000)
    log_prices = jnp.log(prices)
    log_returns = jax.random.normal(rng, (32, 64)) * 0.01
    log_ma = log_prices - 0.01
    noise = jax.random.normal(rng, (32, 64))
    features = jax.random.normal(rng, (32, 10))
    action = jax.random.randint(rng, (32,), 0, 192)

    params = model.init(rng, log_prices, log_returns, log_ma, noise, features, action)
    next_log_prices, aux = model.apply(
        params, log_prices, log_returns, log_ma, noise, features, action
    )

    assert next_log_prices.shape == log_prices.shape
    assert 'G_phys' in aux
    assert 'G_neural' in aux
    assert 'G_impact' in aux


def test_market_ode_rk2():
    """RK2 solver produces different (more stable) results."""
    rng = jax.random.PRNGKey(0)

    model_euler = NeuralMarketODE(n_assets=64, ode_method='euler')
    model_rk2 = NeuralMarketODE(n_assets=64, ode_method='rk2')

    log_prices = jax.random.normal(rng, (8, 64))
    log_returns = jax.random.normal(rng, (8, 64)) * 0.01
    log_ma = log_prices
    noise = jax.random.normal(rng, (8, 64))

    params = model_euler.init(rng, log_prices, log_returns, log_ma, noise)

    pred_euler, _ = model_euler.apply(params, log_prices, log_returns, log_ma, noise)
    pred_rk2, aux = model_rk2.apply(params, log_prices, log_returns, log_ma, noise)

    # RK2 should be different from Euler
    assert not jnp.allclose(pred_euler, pred_rk2, atol=1e-6)
    # RK2 aux should have k1, k2
    assert 'k1' in aux
    assert 'k2' in aux


def test_market_ode_rollout():
    """Multi-step rollout works correctly."""
    model = NeuralMarketODE(n_assets=64, n_features=10)
    rng = jax.random.PRNGKey(0)

    log_prices = jax.random.normal(rng, (8, 64))
    log_returns = jax.random.normal(rng, (8, 64)) * 0.01
    log_ma = log_prices
    features_seq = jax.random.normal(rng, (50, 8, 10))
    actions_seq = jax.random.randint(rng, (50, 8), 0, 192)

    # Initialize
    dummy_noise = jax.random.normal(rng, (8, 64))
    params = model.init(rng, log_prices, log_returns, log_ma, dummy_noise, features_seq[0], actions_seq[0])

    # Create rollout function
    rollout = create_rollout_fn(model, params, n_steps=50)

    # Run rollout
    trajectory = rollout(log_prices, log_returns, log_ma, features_seq, actions_seq, rng)

    assert trajectory.shape == (50, 8, 64)
    assert jnp.all(jnp.isfinite(trajectory))


def test_action_conditioning():
    """Market impact changes predictions based on action."""
    model = NeuralMarketODE(n_assets=64, use_action_conditioning=True)
    rng = jax.random.PRNGKey(0)

    log_prices = jax.random.normal(rng, (8, 64))
    log_returns = jax.random.normal(rng, (8, 64)) * 0.01
    log_ma = log_prices
    noise = jax.random.normal(rng, (8, 64))

    # Buy action for asset 0
    action_buy = jnp.full((8,), 2)  # asset 0, buy
    # Sell action for asset 0
    action_sell = jnp.full((8,), 0)  # asset 0, sell

    params = model.init(rng, log_prices, log_returns, log_ma, noise, None, action_buy)

    pred_buy, aux_buy = model.apply(params, log_prices, log_returns, log_ma, noise, None, action_buy)
    pred_sell, aux_sell = model.apply(params, log_prices, log_returns, log_ma, noise, None, action_sell)

    # Buy should push price up (positive impact)
    assert aux_buy['G_impact'][0, 0] > 0
    # Sell should push price down (negative impact)
    assert aux_sell['G_impact'][0, 0] < 0
```

---

## Acceptance Criteria

- [ ] `NeuralMarketODE` combines physics + neural + impact via ODE
- [ ] Operates entirely in LOG-SPACE
- [ ] `MarketImpact` models action-conditional dynamics
- [ ] `ode_method='rk2'` for higher-order integration
- [ ] `predict_step()` convenience method working
- [ ] `create_rollout_fn()` for multi-step prediction
- [ ] `create_training_loss_fn()` supports multi-step loss
- [ ] Rollout stable over 200+ steps
- [ ] All components have auxiliary outputs for analysis

---

## Parameter Count (Combined)

| Component | Parameters |
|-----------|------------|
| Physics Core | ~580 |
| Neural Corrector | ~70K |
| Market Impact | ~1 |
| **Total** | **~71K** |

Memory: ~285 KB

---

## Verification Commands

```bash
# Run all prediction module tests
pytest tests/test_physics_core.py tests/test_neural_corrector.py tests/test_market_ode.py -v

# Full integration test
python -c "
from mcts_trader.jax_agent.prediction import NeuralMarketODE, create_rollout_fn
import jax
import jax.numpy as jnp

model = NeuralMarketODE(n_assets=64, n_features=10, ode_method='rk2')
rng = jax.random.PRNGKey(0)

# Initialize in LOG space
prices = jax.random.uniform(rng, (8, 64), minval=100, maxval=1000)
log_prices = jnp.log(prices)
log_returns = jax.random.normal(rng, (8, 64)) * 0.01
log_ma = log_prices - 0.01
features = jax.random.normal(rng, (8, 10))
action = jax.random.randint(rng, (8,), 0, 192)
noise = jax.random.normal(rng, (8, 64))

params = model.init(rng, log_prices, log_returns, log_ma, noise, features, action)
print(f'Total params: {sum(x.size for x in jax.tree_util.tree_leaves(params)):,}')

# Test one step with action
next_log_prices, aux = model.apply(params, log_prices, log_returns, log_ma, noise, features, action)
print(f'One-step prediction: {next_log_prices.shape}')
print(f'G_phys magnitude: {jnp.abs(aux[\"G_phys\"]).mean():.6f}')
print(f'G_neural magnitude: {jnp.abs(aux[\"G_neural\"]).mean():.6f}')
print(f'G_impact magnitude: {jnp.abs(aux[\"G_impact\"]).mean():.6f}')

# Test rollout
rollout = create_rollout_fn(model, params, n_steps=200)
features_seq = jax.random.normal(rng, (200, 8, 10))
actions_seq = jax.random.randint(rng, (200, 8), 0, 192)
trajectory = rollout(log_prices, log_returns, log_ma, features_seq, actions_seq, rng)
print(f'200-step rollout: {trajectory.shape}')
print(f'Trajectory stable: {jnp.all(jnp.isfinite(trajectory))}')

# Convert back to prices
final_prices = jnp.exp(trajectory[-1])
print(f'Final prices range: [{final_prices.min():.2f}, {final_prices.max():.2f}]')

print('SUCCESS: NeuralMarketODE fully working!')
"
```

---

## Next Steps

After completing this task:
1. Phase 1 complete! Standalone prediction model ready.
2. Proceed to TASK_011 (MuZero Integration)
