# TASK 008: Implement Differentiable Market Physics Core

**Severity**: FEATURE
**Phase**: 1 (Core Prediction Model)
**Location**: `mcts_trader/jax_agent/prediction/physics_core.py` (NEW)
**Status**: OPEN
**Estimated Time**: 3 hours
**Dependencies**: None

---

## Objective

Implement the **Differentiable Market Physics Core** - the physics-based component of the NeuralMarketODE prediction model. This module encodes known market dynamics with **learnable parameters**.

---

## Critical Design Decisions

### 1. Log-Returns, Not Raw Prices (P0)

**Problem**: Raw prices are non-stationary and can produce negative values.

**Solution**: Work in log-space for numerical stability:

```
log_returns = log(p_t) - log(p_{t-1})
G_phys operates in log-space
next_log_price = log_price + dt * G_phys
next_price = exp(next_log_price)  # Only at output
```

**Benefits**:
- Prevents negative prices (impossible in exp-space)
- Returns are stationary (prices are not)
- Multiplicative dynamics become additive

### 2. Low-Rank Correlation Matrix (P1)

**Problem**: Full N×N matrix = 4096 params for 64 assets, overfits easily.

**Solution**: Factor decomposition: Γ = L @ L.T / k

```
factor_loadings: (n_assets, n_factors)  # e.g., (64, 8) = 512 params
gamma = loadings @ loadings.T / n_factors
```

**Benefits**:
- 8x fewer parameters (512 vs 4096)
- Learns meaningful asset clusters/factors
- Better generalization

### 3. Time-Varying Volatility (P2)

**Problem**: Constant volatility ignores volatility clustering (high vol begets high vol).

**Solution**: GARCH-like learnable volatility dynamics:

```
σ²_t = ω + α * r²_{t-1} + β * σ²_{t-1}
```

### 4. Proper Noise Scaling

**Problem**: If sigma is daily vol and dt < 1, noise must be scaled by sqrt(dt).

**Solution**: `G_vol = sigma * noise * sqrt(dt)` for proper diffusion scaling.

---

## Mathematical Foundation

The physics core computes price tendency in **log-space**:

$$G_{phys}(\log p; \theta_p) = G_{momentum} + G_{reversion} + G_{volatility} + G_{correlation}$$

### Components

| Component | Formula | Parameter | Description |
|-----------|---------|-----------|-------------|
| Momentum | $\alpha \cdot r$ | $\alpha \in \mathbb{R}$ | Log-return inertia |
| Mean Reversion | $-\beta(\log p - \log \bar{p})$ | $\beta \in \mathbb{R}$ | Pull to log-equilibrium |
| Volatility | $\sigma \cdot \epsilon \cdot \sqrt{\Delta t}$ | $\sigma \in \mathbb{R}^N$ | Per-asset diffusion |
| Correlation | $(L L^T) \cdot \log p$ | $L \in \mathbb{R}^{N \times k}$ | Low-rank factor coupling |

All parameters are constrained via **softplus** to ensure physical validity.

---

## Implementation

### File: `mcts_trader/jax_agent/prediction/physics_core.py`

```python
"""
Differentiable Market Physics Core.

Implements learnable market dynamics in LOG-SPACE:
- Momentum (log-return inertia)
- Mean reversion (equilibrium pull)
- Volatility (stochastic diffusion with sqrt(dt) scaling)
- Cross-asset correlation (low-rank factor model)

Based on NeuralOGCM (arXiv:2512.11525v1).
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Tuple, Optional


class LowRankCorrelation(nn.Module):
    """
    Low-rank factor model for cross-asset correlation.

    Instead of full N×N matrix (4096 params for 64 assets),
    use factorization: Γ = L @ L.T / k

    This learns underlying market factors with 8x fewer parameters.

    Attributes:
        n_assets: Number of assets
        n_factors: Number of latent factors (default: 8)
    """
    n_assets: int = 64
    n_factors: int = 8

    @nn.compact
    def __call__(self, log_prices: jnp.ndarray) -> jnp.ndarray:
        """
        Compute correlation coupling in log-space.

        Args:
            log_prices: (batch, n_assets) - Log prices

        Returns:
            G_correlation: (batch, n_assets) - Correlation tendency
        """
        # Factor loadings: which assets belong to which factors
        loadings = self.param(
            'factor_loadings',
            nn.initializers.normal(0.1),
            (self.n_assets, self.n_factors)
        )

        # Reconstruct correlation: Γ = L @ L.T / k (normalized)
        # This is symmetric and positive semi-definite by construction
        gamma = loadings @ loadings.T / self.n_factors

        # Apply softplus for non-negativity, scale by 0.1
        gamma = nn.softplus(gamma) * 0.1

        # Correlation coupling: G_i = Σ_j γ_ij * (log_p_j - log_p_i)
        # This is a graph Laplacian diffusion
        correlation_pull = jnp.einsum('ij,bj->bi', gamma, log_prices)
        row_sums = jnp.sum(gamma, axis=1)
        G_correlation = correlation_pull - row_sums[None, :] * log_prices

        return G_correlation


class GARCHVolatility(nn.Module):
    """
    GARCH-like time-varying volatility.

    Models volatility clustering: high vol begets high vol.
    σ²_t = ω + α * r²_{t-1} + β * σ²_{t-1}

    Attributes:
        n_assets: Number of assets
    """
    n_assets: int = 64

    @nn.compact
    def __call__(
        self,
        log_returns: jnp.ndarray,
        prev_variance: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Compute conditional volatility.

        Args:
            log_returns: (batch, n_assets) - Recent log returns
            prev_variance: (batch, n_assets) - Previous variance estimate

        Returns:
            sigma: (batch, n_assets) - Current volatility estimate
        """
        # GARCH parameters (per-asset)
        omega = nn.softplus(self.param(
            'omega',
            nn.initializers.constant(-4.0),  # ~0.02 base variance
            (self.n_assets,)
        ))

        alpha_garch = nn.softplus(self.param(
            'alpha_garch',
            nn.initializers.constant(-2.0),  # ~0.13 return shock weight
            (self.n_assets,)
        ))

        beta_garch = nn.softplus(self.param(
            'beta_garch',
            nn.initializers.constant(0.5),  # ~0.65 persistence
            (self.n_assets,)
        ))

        # If no previous variance, use unconditional variance
        if prev_variance is None:
            # Unconditional: σ² = ω / (1 - α - β)
            # Clip to prevent division by zero
            denom = jnp.maximum(1.0 - alpha_garch - beta_garch, 0.1)
            prev_variance = omega[None, :] / denom[None, :]

        # GARCH(1,1): σ²_t = ω + α * r²_{t-1} + β * σ²_{t-1}
        next_variance = (
            omega[None, :] +
            alpha_garch[None, :] * log_returns ** 2 +
            beta_garch[None, :] * prev_variance
        )

        # Return standard deviation
        return jnp.sqrt(next_variance + 1e-8)


class DifferentiableMarketPhysics(nn.Module):
    """
    Learnable market physics core operating in LOG-SPACE.

    Computes physics-based price tendency with learnable parameters.
    All parameters are constrained to be positive via softplus.

    Attributes:
        n_assets: Number of tradeable assets (default: 64)
        n_factors: Number of correlation factors (default: 8)
        use_garch: Whether to use GARCH volatility (default: False)
        dt: Time step for diffusion scaling (default: 1.0)
    """
    n_assets: int = 64
    n_factors: int = 8
    use_garch: bool = False
    dt: float = 1.0

    def setup(self):
        """Initialize sub-modules."""
        self.correlation = LowRankCorrelation(
            n_assets=self.n_assets,
            n_factors=self.n_factors
        )

        if self.use_garch:
            self.garch = GARCHVolatility(n_assets=self.n_assets)

    @nn.compact
    def __call__(
        self,
        log_prices: jnp.ndarray,
        log_returns: jnp.ndarray,
        log_moving_avg: jnp.ndarray,
        noise: jnp.ndarray,
        prev_variance: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Compute physics-based tendency in LOG-SPACE.

        Args:
            log_prices: (batch, n_assets) - Current log prices
            log_returns: (batch, n_assets) - Recent log returns (velocity)
            log_moving_avg: (batch, n_assets) - Log moving average (equilibrium)
            noise: (batch, n_assets) - Sampled noise for volatility
            prev_variance: (batch, n_assets) - Previous variance (for GARCH)

        Returns:
            G_phys: (batch, n_assets) - Physics tendency (d(log p)/dt)
            aux: Dict with individual components and new variance
        """
        # ============================================================
        # LEARNABLE PARAMETERS
        # ============================================================

        # Momentum strength: how much past returns persist
        alpha_raw = self.param(
            'alpha',
            nn.initializers.constant(0.0),  # softplus(0) ≈ 0.69
            (1,)
        )
        alpha = nn.softplus(alpha_raw)

        # Mean reversion strength: how fast prices return to MA
        beta_raw = self.param(
            'beta',
            nn.initializers.constant(-2.0),  # softplus(-2) ≈ 0.13
            (1,)
        )
        beta = nn.softplus(beta_raw)

        # Per-asset base volatility (used if not GARCH)
        sigma_raw = self.param(
            'sigma',
            nn.initializers.constant(-3.9),  # softplus(-3.9) ≈ 0.02
            (self.n_assets,)
        )
        sigma_base = nn.softplus(sigma_raw)

        # ============================================================
        # PHYSICS COMPONENTS (all in log-space)
        # ============================================================

        # 1. MOMENTUM: G = α * log_returns
        # Log prices tend to continue in their recent direction
        G_momentum = alpha * log_returns

        # 2. MEAN REVERSION: G = -β * (log_price - log_moving_avg)
        # Log prices are pulled back toward equilibrium
        deviation = log_prices - log_moving_avg
        G_reversion = -beta * deviation

        # 3. VOLATILITY: G = σ * noise * sqrt(dt)
        # Proper diffusion scaling for arbitrary time steps
        if self.use_garch:
            sigma = self.garch(log_returns, prev_variance)
            new_variance = sigma ** 2
        else:
            sigma = sigma_base[None, :]
            new_variance = sigma ** 2

        sqrt_dt = jnp.sqrt(self.dt)
        G_volatility = sigma * noise * sqrt_dt

        # 4. CORRELATION COUPLING (low-rank)
        G_correlation = self.correlation(log_prices)

        # ============================================================
        # COMBINE ALL COMPONENTS
        # ============================================================

        G_phys = G_momentum + G_reversion + G_volatility + G_correlation

        aux = {
            'G_momentum': G_momentum,
            'G_reversion': G_reversion,
            'G_volatility': G_volatility,
            'G_correlation': G_correlation,
            'sigma': sigma,
            'new_variance': new_variance,
            'alpha': alpha,
            'beta': beta,
        }

        return G_phys, aux


def log_physics_params(params: Dict, prefix: str = "") -> Dict[str, float]:
    """
    Extract and log physics parameters from model params.

    Args:
        params: Flax parameter dict
        prefix: Prefix for logging keys

    Returns:
        Dict of interpretable physics values
    """
    physics_params = params.get('params', params)

    alpha = float(nn.softplus(physics_params.get('alpha', jnp.array([0.0]))))
    beta = float(nn.softplus(physics_params.get('beta', jnp.array([-2.0]))))
    sigma = nn.softplus(physics_params.get('sigma', jnp.zeros(64)))

    # Factor loadings for correlation
    loadings = physics_params.get('correlation', {}).get(
        'factor_loadings', jnp.zeros((64, 8))
    )
    gamma = loadings @ loadings.T / loadings.shape[1]
    gamma = nn.softplus(gamma) * 0.1

    return {
        f'{prefix}alpha_momentum': alpha,
        f'{prefix}beta_reversion': beta,
        f'{prefix}sigma_vol_mean': float(sigma.mean()),
        f'{prefix}sigma_vol_std': float(sigma.std()),
        f'{prefix}gamma_corr_mean': float(gamma.mean()),
        f'{prefix}gamma_corr_max': float(gamma.max()),
        f'{prefix}n_factors': loadings.shape[1],
    }


def physics_regularization_loss(params: Dict) -> jnp.ndarray:
    """
    Regularization to prevent physics parameters from collapsing.

    Encourages:
    - alpha around 0.5 (moderate momentum)
    - beta below 0.5 (prevent oscillation)
    - sigma diversity across assets

    Args:
        params: Model parameters

    Returns:
        Scalar regularization loss
    """
    physics_params = params.get('params', params)

    alpha = nn.softplus(physics_params.get('alpha', jnp.array([0.0])))
    beta = nn.softplus(physics_params.get('beta', jnp.array([-2.0])))
    sigma = nn.softplus(physics_params.get('sigma', jnp.zeros(64)))

    # Penalty for alpha too close to 0 or 1
    alpha_reg = jnp.abs(alpha - 0.5) ** 2

    # Penalty for beta too high (would cause price oscillation)
    beta_reg = jnp.maximum(0, beta - 0.5) ** 2

    # Encourage sigma diversity (should vary across assets)
    sigma_variance_reg = -jnp.var(sigma)  # Negative = encourage high variance

    return 0.01 * (alpha_reg + beta_reg + 0.1 * sigma_variance_reg).mean()
```

---

## Unit Tests

### File: `tests/test_physics_core.py`

```python
"""Unit tests for DifferentiableMarketPhysics."""

import jax
import jax.numpy as jnp
from mcts_trader.jax_agent.prediction.physics_core import (
    DifferentiableMarketPhysics,
    LowRankCorrelation,
    GARCHVolatility,
    log_physics_params,
    physics_regularization_loss
)


def test_physics_core_log_space():
    """Works in log-space and outputs correct shape."""
    model = DifferentiableMarketPhysics(n_assets=64, n_factors=8)
    rng = jax.random.PRNGKey(0)

    # Work in LOG space
    log_prices = jnp.log(jax.random.uniform(rng, (32, 64), minval=100, maxval=1000))
    log_returns = jax.random.normal(rng, (32, 64)) * 0.01
    log_moving_avg = log_prices - 0.01
    noise = jax.random.normal(rng, (32, 64))

    params = model.init(rng, log_prices, log_returns, log_moving_avg, noise)
    G_phys, aux = model.apply(params, log_prices, log_returns, log_moving_avg, noise)

    assert G_phys.shape == (32, 64)
    assert 'G_momentum' in aux
    assert 'G_correlation' in aux


def test_low_rank_correlation():
    """Low-rank correlation has correct parameter count."""
    model = LowRankCorrelation(n_assets=64, n_factors=8)
    rng = jax.random.PRNGKey(0)

    log_prices = jax.random.normal(rng, (32, 64))
    params = model.init(rng, log_prices)

    # Should have 64 * 8 = 512 params (not 64 * 64 = 4096)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    assert n_params == 64 * 8, f"Expected 512 params, got {n_params}"

    G_corr = model.apply(params, log_prices)
    assert G_corr.shape == (32, 64)


def test_garch_volatility():
    """GARCH volatility produces time-varying sigma."""
    model = GARCHVolatility(n_assets=64)
    rng = jax.random.PRNGKey(0)

    # High returns should produce high volatility
    low_returns = jnp.zeros((1, 64))
    high_returns = jnp.ones((1, 64)) * 0.1  # 10% return

    params = model.init(rng, low_returns)

    sigma_low = model.apply(params, low_returns)
    sigma_high = model.apply(params, high_returns)

    # High returns should give higher sigma (GARCH effect)
    assert sigma_high.mean() > sigma_low.mean()


def test_physics_gradients():
    """Gradients flow through all physics parameters."""
    model = DifferentiableMarketPhysics(n_assets=64, n_factors=8)
    rng = jax.random.PRNGKey(0)

    log_prices = jax.random.normal(rng, (32, 64))
    log_returns = jax.random.normal(rng, (32, 64)) * 0.01
    log_ma = log_prices - 0.01
    noise = jax.random.normal(rng, (32, 64))

    params = model.init(rng, log_prices, log_returns, log_ma, noise)

    def loss_fn(params):
        G, _ = model.apply(params, log_prices, log_returns, log_ma, noise)
        return jnp.mean(G ** 2)

    grads = jax.grad(loss_fn)(params)

    # Check all params have gradients
    assert grads['params']['alpha'] is not None
    assert grads['params']['beta'] is not None
    assert grads['params']['sigma'] is not None
    assert grads['params']['correlation']['factor_loadings'] is not None

    # Check no NaN gradients
    for key, grad in jax.tree_util.tree_leaves_with_path(grads):
        assert jnp.all(jnp.isfinite(grad)), f"NaN gradient in {key}"


def test_sqrt_dt_scaling():
    """Volatility scales with sqrt(dt)."""
    rng = jax.random.PRNGKey(0)

    model_dt1 = DifferentiableMarketPhysics(n_assets=64, dt=1.0)
    model_dt4 = DifferentiableMarketPhysics(n_assets=64, dt=4.0)

    log_prices = jax.random.normal(rng, (8, 64))
    log_returns = jax.random.normal(rng, (8, 64)) * 0.01
    log_ma = log_prices
    noise = jnp.ones((8, 64))  # Fixed noise for comparison

    params = model_dt1.init(rng, log_prices, log_returns, log_ma, noise)

    _, aux1 = model_dt1.apply(params, log_prices, log_returns, log_ma, noise)
    _, aux4 = model_dt4.apply(params, log_prices, log_returns, log_ma, noise)

    # G_vol should scale by sqrt(dt)
    # dt=4 should have 2x the volatility of dt=1
    ratio = jnp.abs(aux4['G_volatility']).mean() / jnp.abs(aux1['G_volatility']).mean()
    assert jnp.isclose(ratio, 2.0, atol=0.01), f"Expected ratio 2.0, got {ratio}"


def test_regularization_loss():
    """Regularization loss is computed correctly."""
    model = DifferentiableMarketPhysics(n_assets=64)
    rng = jax.random.PRNGKey(0)

    dummy = jax.random.normal(rng, (1, 64))
    params = model.init(rng, dummy, dummy, dummy, dummy)

    reg_loss = physics_regularization_loss(params)

    assert reg_loss.shape == ()
    assert jnp.isfinite(reg_loss)
```

---

## Acceptance Criteria

- [ ] `DifferentiableMarketPhysics` operates in LOG-SPACE
- [ ] `LowRankCorrelation` uses factor decomposition (512 params, not 4096)
- [ ] `GARCHVolatility` for time-varying volatility (optional)
- [ ] Noise scaled by sqrt(dt) for proper diffusion
- [ ] `physics_regularization_loss()` prevents trivial solutions
- [ ] Gradients flow through all learnable parameters
- [ ] Unit tests passing
- [ ] No NaN/Inf in outputs or gradients

---

## Parameter Count

| Component | Shape | Count |
|-----------|-------|-------|
| alpha | (1,) | 1 |
| beta | (1,) | 1 |
| sigma | (64,) | 64 |
| factor_loadings | (64, 8) | 512 |
| **Total** | | **578** |

Memory: 578 × 4 bytes = **~2.3 KB** (was 16.6 KB with full correlation)

With GARCH: +192 params (omega, alpha_garch, beta_garch) = **770 total**

---

## Verification Commands

```bash
# Run unit tests
pytest tests/test_physics_core.py -v

# Quick sanity check
python -c "
from mcts_trader.jax_agent.prediction.physics_core import DifferentiableMarketPhysics, log_physics_params
import jax
import jax.numpy as jnp

model = DifferentiableMarketPhysics(n_assets=64, n_factors=8)
rng = jax.random.PRNGKey(0)

# Work in LOG space
prices = jax.random.uniform(rng, (32, 64), minval=100, maxval=1000)
log_prices = jnp.log(prices)
log_returns = jax.random.normal(rng, (32, 64)) * 0.01
log_ma = log_prices * 0.99
noise = jax.random.normal(rng, (32, 64))

params = model.init(rng, log_prices, log_returns, log_ma, noise)
G_phys, aux = model.apply(params, log_prices, log_returns, log_ma, noise)

print(f'Output shape: {G_phys.shape}')
print(f'Output range: [{G_phys.min():.6f}, {G_phys.max():.6f}]')
print(f'G_momentum magnitude: {jnp.abs(aux[\"G_momentum\"]).mean():.6f}')
print(f'G_correlation magnitude: {jnp.abs(aux[\"G_correlation\"]).mean():.6f}')

# Log parameters
logged = log_physics_params(params)
print(f'Physics params: {logged}')

# Parameter count
n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(f'Total parameters: {n_params} (expected ~578)')

print('SUCCESS: Physics core working in log-space!')
"
```

---

## Next Steps

After completing this task:
1. Proceed to TASK_009 (Neural Market Corrector)
2. The physics core will be combined with neural corrector in TASK_010
