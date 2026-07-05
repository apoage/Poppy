"""
Differentiable Market Physics Core (TASK_008)

Physics-based market dynamics with learnable parameters.
All operations in LOG-SPACE for numerical stability.

Key Components:
- DifferentiableMarketPhysics: Main physics module with G_momentum, G_reversion, G_vol, G_correlation
- LowRankCorrelation: Parameter-efficient correlation using factor decomposition
- GARCHVolatility: Time-varying volatility with GARCH-like dynamics

Based on NeuralOGCM (arXiv:2512.11525v1) adapted for financial markets.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, Tuple, Optional, NamedTuple


class PhysicsOutput(NamedTuple):
    """Output from physics computation."""
    G_momentum: jnp.ndarray    # (batch, n_assets)
    G_reversion: jnp.ndarray   # (batch, n_assets)
    G_volatility: jnp.ndarray  # (batch, n_assets)
    G_correlation: jnp.ndarray # (batch, n_assets)
    G_total: jnp.ndarray       # (batch, n_assets) - combined tendency


class LowRankCorrelation(nn.Module):
    """
    Low-rank correlation matrix via factor decomposition.

    Instead of learning full N×N correlation matrix (4096 params for 64 assets),
    we learn N×K factor loadings where K << N (512 params for K=8).

    Correlation matrix: Γ = L @ L.T / K
    where L is (n_assets, n_factors) factor loadings.

    This naturally produces positive semi-definite correlation structure
    and captures market factors (sectors, risk factors, etc.).
    """
    n_assets: int = 64
    n_factors: int = 8

    @nn.compact
    def __call__(self, log_prices: jnp.ndarray) -> jnp.ndarray:
        """
        Compute correlation-driven tendency using Graph Laplacian diffusion.

        Args:
            log_prices: (batch, n_assets) log prices

        Returns:
            G_correlation: (batch, n_assets) correlation tendency
        """
        # Factor loadings: (n_assets, n_factors)
        # Initialize with small values to start with weak correlations
        loadings = self.param(
            'factor_loadings',
            nn.initializers.normal(0.1),
            (self.n_assets, self.n_factors)
        )

        # Correlation matrix via factor decomposition
        # Γ = L @ L.T / K  (ensures PSD)
        gamma = loadings @ loadings.T / self.n_factors  # (n_assets, n_assets)

        # Apply softplus to ensure positive correlations, scale appropriately
        gamma = nn.softplus(gamma) * 0.1

        # Graph Laplacian diffusion for correlation dynamics
        # G_i = sum_j Γ_ij * (p_j - p_i) = Γ @ p - D @ p
        # where D_ii = sum_j Γ_ij (degree matrix)
        correlation_pull = jnp.einsum('ij,bj->bi', gamma, log_prices)
        row_sums = jnp.sum(gamma, axis=1)  # Degree
        G_correlation = correlation_pull - row_sums[None, :] * log_prices

        return G_correlation

    def get_correlation_matrix(self) -> jnp.ndarray:
        """Return the full correlation matrix for inspection."""
        loadings = self.get_variable('params', 'factor_loadings')
        gamma = loadings @ loadings.T / self.n_factors
        return nn.softplus(gamma) * 0.1


class GARCHVolatility(nn.Module):
    """
    GARCH-like time-varying volatility model.

    Volatility clustering: high volatility tends to persist.

    σ²_t = ω + α * r²_{t-1} + β * σ²_{t-1}

    where:
    - ω: base variance
    - α: reaction to recent returns (news impact)
    - β: persistence (volatility memory)
    """
    n_assets: int = 64

    @nn.compact
    def __call__(
        self,
        log_returns: jnp.ndarray,
        prev_variance: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Compute time-varying volatility.

        Args:
            log_returns: (batch, n_assets) recent log returns
            prev_variance: (batch, n_assets) previous variance estimate
                          If None, uses unconditional variance

        Returns:
            sigma: (batch, n_assets) current volatility estimate
        """
        # GARCH parameters (per-asset for flexibility)
        # Use softplus to ensure positivity

        # Base variance (omega): typical ~0.0001 (1 basis point squared)
        omega = nn.softplus(self.param(
            'omega',
            nn.initializers.constant(-4.0),  # exp(-4) ≈ 0.018
            (self.n_assets,)
        )) * 0.01

        # News impact (alpha): typical ~0.05-0.15
        alpha_garch = nn.softplus(self.param(
            'alpha_garch',
            nn.initializers.constant(-2.0),  # softplus(-2) ≈ 0.13
            (self.n_assets,)
        )) * 0.3

        # Persistence (beta): typical ~0.8-0.95
        beta_garch = nn.softplus(self.param(
            'beta_garch',
            nn.initializers.constant(0.5),  # softplus(0.5) ≈ 0.97
            (self.n_assets,)
        )) * 0.99  # Cap at 0.99 for stationarity

        # Initialize prev_variance if not provided
        if prev_variance is None:
            # Unconditional variance: ω / (1 - α - β)
            # Approximate with larger initial variance
            prev_variance = jnp.ones_like(log_returns) * 0.01

        # GARCH update
        next_variance = (
            omega[None, :] +
            alpha_garch[None, :] * log_returns ** 2 +
            beta_garch[None, :] * prev_variance
        )

        # Return volatility (sqrt of variance)
        return jnp.sqrt(next_variance + 1e-8)


class DifferentiableMarketPhysics(nn.Module):
    """
    Core physics module for market dynamics.

    Computes physics-based tendency: G_phys = G_mom + G_rev + G_vol + G_corr

    All operations in LOG-SPACE:
    - log_price_{t+1} = log_price_t + dt * G_total

    Learnable Parameters:
    - α (alpha): momentum coefficient per asset
    - β (beta): mean-reversion coefficient per asset
    - σ (sigma): base volatility per asset (or GARCH)
    - Γ (gamma): correlation structure via low-rank factors
    """
    n_assets: int = 64
    n_factors: int = 8
    use_garch: bool = True

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
        moving_avg: jnp.ndarray,
        noise: jnp.ndarray,
        dt: float = 1.0,
        prev_variance: Optional[jnp.ndarray] = None,
    ) -> Tuple[PhysicsOutput, jnp.ndarray]:
        """
        Compute physics-based tendency for price evolution.

        Args:
            log_prices: (batch, n_assets) current log prices
            log_returns: (batch, n_assets) recent log returns
            moving_avg: (batch, n_assets) moving average of log prices
            noise: (batch, n_assets) standard normal noise
            dt: time step (default 1.0 for minute-level)
            prev_variance: (batch, n_assets) previous variance for GARCH

        Returns:
            physics_output: NamedTuple with component tendencies
            next_variance: (batch, n_assets) updated variance estimate
        """
        batch_size = log_prices.shape[0]

        # =====================================================================
        # Learnable Physics Parameters
        # =====================================================================

        # Momentum coefficient α: captures trend-following behavior
        # Typical range: [0, 0.3], positive means momentum, negative means reversal
        alpha = nn.softplus(self.param(
            'alpha',
            nn.initializers.constant(-2.0),  # softplus(-2) ≈ 0.13
            (self.n_assets,)
        )) * 0.3  # Scale to [0, ~0.3]

        # Mean-reversion coefficient β: pull toward moving average
        # Typical range: [0, 0.2], higher means faster reversion
        beta = nn.softplus(self.param(
            'beta',
            nn.initializers.constant(-2.5),  # softplus(-2.5) ≈ 0.08
            (self.n_assets,)
        )) * 0.2  # Scale to [0, ~0.2]

        # Base volatility σ (used if not using GARCH)
        sigma_base = nn.softplus(self.param(
            'sigma',
            nn.initializers.constant(-3.0),  # softplus(-3) ≈ 0.05
            (self.n_assets,)
        )) * 0.1  # Scale to reasonable volatility

        # =====================================================================
        # Compute Tendency Components
        # =====================================================================

        # G_momentum: α * r_t (trend continuation)
        G_momentum = alpha[None, :] * log_returns

        # G_reversion: -β * (p - μ) (pull toward mean)
        deviation = log_prices - moving_avg
        G_reversion = -beta[None, :] * deviation

        # G_volatility: σ * ε * sqrt(dt) (stochastic volatility)
        if self.use_garch:
            sigma = self.garch(log_returns, prev_variance)
            next_variance = sigma ** 2
        else:
            sigma = sigma_base[None, :] * jnp.ones((batch_size, 1))
            next_variance = sigma ** 2

        # Proper noise scaling: σ * ε * sqrt(dt)
        G_volatility = sigma * noise * jnp.sqrt(dt)

        # G_correlation: Laplacian diffusion coupling
        G_correlation = self.correlation(log_prices)

        # =====================================================================
        # Total Physics Tendency
        # =====================================================================

        G_total = G_momentum + G_reversion + G_volatility + G_correlation

        output = PhysicsOutput(
            G_momentum=G_momentum,
            G_reversion=G_reversion,
            G_volatility=G_volatility,
            G_correlation=G_correlation,
            G_total=G_total,
        )

        return output, next_variance

    def get_parameters(self) -> Dict[str, jnp.ndarray]:
        """Return physics parameters for inspection."""
        alpha = nn.softplus(self.get_variable('params', 'alpha')) * 0.3
        beta = nn.softplus(self.get_variable('params', 'beta')) * 0.2
        sigma = nn.softplus(self.get_variable('params', 'sigma')) * 0.1

        return {
            'alpha': alpha,
            'beta': beta,
            'sigma': sigma,
            'correlation_matrix': self.correlation.get_correlation_matrix(),
        }

    @staticmethod
    def get_components_for_logging(physics_output: PhysicsOutput) -> Dict[str, jnp.ndarray]:
        """Extract physics components for DS24 logging.

        Args:
            physics_output: Output from forward pass

        Returns:
            Dict with G_momentum, G_reversion, G_volatility, G_correlation
        """
        return {
            'G_momentum': physics_output.G_momentum,
            'G_reversion': physics_output.G_reversion,
            'G_volatility': physics_output.G_volatility,
            'G_correlation': physics_output.G_correlation,
        }

    @staticmethod
    def get_learned_params_from_dict(params: Dict) -> Dict[str, float]:
        """Extract learned physics parameters for DS24 logging.

        Args:
            params: Nested parameter dict from model

        Returns:
            Dict with alpha_mean, beta_mean, garch_persistence
        """
        # Navigate to physics params - handle different param structures
        physics_params = params.get('params', params)

        result = {
            'alpha_mean': 0.0,
            'beta_mean': 0.0,
            'garch_persistence': 0.0,
        }

        # Extract alpha
        if 'alpha' in physics_params:
            alpha = nn.softplus(physics_params['alpha']) * 0.3
            result['alpha_mean'] = float(jnp.mean(alpha))

        # Extract beta
        if 'beta' in physics_params:
            beta = nn.softplus(physics_params['beta']) * 0.2
            result['beta_mean'] = float(jnp.mean(beta))

        # Extract GARCH persistence (alpha + beta)
        if 'garch' in physics_params:
            garch_params = physics_params['garch']
            if 'alpha_garch' in garch_params and 'beta_garch' in garch_params:
                garch_alpha = nn.softplus(garch_params['alpha_garch']) * 0.3
                garch_beta = nn.softplus(garch_params['beta_garch']) * 0.99
                result['garch_persistence'] = float(jnp.mean(garch_alpha + garch_beta))

        return result


def physics_regularization_loss(params: Dict) -> jnp.ndarray:
    """
    Compute regularization loss for physics parameters.

    TASK_035 FIX: Changed from ceiling-only to two-sided TARGET regularization.
    The old ceiling-only approach (max(x - ceiling, 0)²) had ZERO penalty for
    low values, causing α and β to collapse to zero.

    New approach: Pull parameters toward sensible targets with bidirectional penalty.
    - alpha (momentum): target ~0.05
    - beta (reversion): target ~0.03
    - sigma (volatility): target ~0.01
    - GARCH persistence: floor at 0.85, ceiling at 1.0

    Args:
        params: Nested dict of model parameters

    Returns:
        reg_loss: Scalar regularization loss
    """
    reg_loss = jnp.array(0.0)

    # Navigate to physics parameters
    physics_params = params.get('physics_core', params)

    # TASK_035 FIX: Two-sided TARGET regularization for momentum
    # OLD (broken): jnp.maximum(alpha - 0.3, 0.0) ** 2  → α=0 has ZERO penalty
    if 'alpha' in physics_params:
        alpha_raw = physics_params['alpha']
        alpha = nn.softplus(alpha_raw) * 0.3
        alpha_target = 0.05  # Target: small positive momentum
        reg_loss += jnp.mean((alpha - alpha_target) ** 2) * 0.5

    # TASK_035 FIX: Two-sided TARGET regularization for mean-reversion
    if 'beta' in physics_params:
        beta_raw = physics_params['beta']
        beta = nn.softplus(beta_raw) * 0.2
        beta_target = 0.03  # Target: small mean reversion
        reg_loss += jnp.mean((beta - beta_target) ** 2) * 0.5

    # Sigma (volatility): already two-sided toward 0.01
    if 'sigma' in physics_params:
        sigma_raw = physics_params['sigma']
        sigma = nn.softplus(sigma_raw) * 0.1
        reg_loss += jnp.mean((sigma - 0.01) ** 2)

    # Factor loadings: L1 sparsity for interpretability
    if 'correlation' in physics_params:
        corr_params = physics_params['correlation']
        if 'factor_loadings' in corr_params:
            loadings = corr_params['factor_loadings']
            reg_loss += 0.01 * jnp.mean(jnp.abs(loadings))

    # TASK_035 FIX: GARCH persistence floor + ceiling
    # OLD: Only ceiling at 0.99, no floor → persistence collapsed to 0.096
    # NEW: Floor at 0.85, ceiling at 1.0 for realistic volatility clustering
    if 'garch' in physics_params:
        garch_params = physics_params['garch']
        if 'alpha_garch' in garch_params and 'beta_garch' in garch_params:
            alpha_g = nn.softplus(garch_params['alpha_garch']) * 0.3
            beta_g = nn.softplus(garch_params['beta_garch']) * 0.99
            persistence = alpha_g + beta_g

            # Floor: penalize if persistence < 0.85 (too low = no volatility memory)
            floor_penalty = jnp.mean(jnp.maximum(0.85 - persistence, 0.0) ** 2)
            # Ceiling: penalize if persistence > 1.0 (non-stationary)
            ceiling_penalty = jnp.mean(jnp.maximum(persistence - 1.0, 0.0) ** 2)
            # Strong weight to enforce realistic GARCH behavior
            reg_loss += (floor_penalty + ceiling_penalty) * 2.0

    return reg_loss


# Convenience function for creating and initializing
def create_physics_core(
    n_assets: int = 64,
    n_factors: int = 8,
    use_garch: bool = True,
    rng: Optional[jnp.ndarray] = None,
) -> Tuple[DifferentiableMarketPhysics, Dict]:
    """
    Create and initialize DifferentiableMarketPhysics.

    Args:
        n_assets: Number of tradeable assets
        n_factors: Number of latent factors for correlation
        use_garch: Whether to use GARCH volatility
        rng: Random key for initialization

    Returns:
        model: Initialized model
        params: Initial parameters
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    model = DifferentiableMarketPhysics(
        n_assets=n_assets,
        n_factors=n_factors,
        use_garch=use_garch,
    )

    # Initialize with dummy inputs
    dummy_log_prices = jnp.zeros((1, n_assets))
    dummy_log_returns = jnp.zeros((1, n_assets))
    dummy_moving_avg = jnp.zeros((1, n_assets))
    dummy_noise = jnp.zeros((1, n_assets))

    params = model.init(
        rng,
        dummy_log_prices,
        dummy_log_returns,
        dummy_moving_avg,
        dummy_noise,
    )

    return model, params
