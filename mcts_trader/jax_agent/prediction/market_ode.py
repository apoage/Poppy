"""
Neural Market ODE (TASK_010)

Combines physics-based dynamics and neural corrections via ODE integration.
Supports action conditioning for modeling agent's market impact.

Key Components:
- MarketImpact: Models agent's impact on prices
- NeuralMarketODE: Main ODE model with Euler and RK2 solvers
- Multi-step rollout functions

All operations in LOG-SPACE for numerical stability.
ODE: d(log_price)/dt = G_phys + G_neural + G_impact
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, Tuple, Optional, NamedTuple

from .physics_core import DifferentiableMarketPhysics, PhysicsOutput
from .neural_corrector import NeuralMarketCorrector, SimpleNeuralCorrector

# Action space: 5 actions per asset (must match vectorbt_env.py)
# 0=Sell, 1=Hold, 2=Buy, 3=LimitSell, 4=LimitBuy
ACTIONS_PER_ASSET = 5


class ODEOutput(NamedTuple):
    """Output from single ODE step."""
    next_log_prices: jnp.ndarray  # (batch, n_assets)
    G_phys: jnp.ndarray           # (batch, n_assets) physics tendency
    G_neural: jnp.ndarray         # (batch, n_assets) neural correction
    G_impact: jnp.ndarray         # (batch, n_assets) action impact
    G_total: jnp.ndarray          # (batch, n_assets) total tendency
    next_variance: jnp.ndarray    # (batch, n_assets) for GARCH


class MarketImpact(nn.Module):
    """
    Models agent's impact on market prices.

    When the agent takes an action, it affects the market:
    - Buy action on asset i -> temporary price increase
    - Sell action on asset i -> temporary price decrease
    - Hold action -> no impact

    Impact decays over time (not modeled here, but in multi-step rollout).
    """
    n_assets: int = 64

    @nn.compact
    def __call__(
        self,
        action: jnp.ndarray,
        action_history: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Compute market impact from agent's action.

        Args:
            action: (batch,) action indices where:
                   asset_idx = action // 3
                   action_type = action % 3  (0=sell, 1=hold, 2=buy)
            action_history: Optional (batch, history_len) for persistent impact

        Returns:
            G_impact: (batch, n_assets) impact tendency
        """
        batch = action.shape[0]

        # Learnable impact coefficient
        # Typical market impact is 0.01-0.1% per trade
        impact_coef = nn.softplus(self.param(
            'impact_coefficient',
            nn.initializers.constant(-4.0),  # softplus(-4) ≈ 0.018
            (1,)
        )) * 0.01  # Scale to ~0.01%

        # Decode action (5 actions per asset: Sell, Hold, Buy, LimitSell, LimitBuy)
        asset_idx = action // ACTIONS_PER_ASSET  # Which asset
        action_type = action % ACTIONS_PER_ASSET  # 0=sell, 1=hold, 2=buy, 3=limit_sell, 4=limit_buy

        # Impact direction
        # Buy/LimitBuy (+1) pushes price up, Sell/LimitSell (-1) pushes down, Hold (0) no impact
        impact_sign = jnp.where(
            (action_type == 0) | (action_type == 3), -1.0,  # Sell or LimitSell
            jnp.where((action_type == 2) | (action_type == 4), +1.0, 0.0)  # Buy, LimitBuy, or Hold
        )

        # Create impact tensor: only the acted-upon asset is affected
        G_impact = jnp.zeros((batch, self.n_assets))

        # Use advanced indexing to set impact for the acted-upon asset
        batch_indices = jnp.arange(batch)
        G_impact = G_impact.at[batch_indices, asset_idx].set(
            impact_coef[0] * impact_sign
        )

        return G_impact


class NeuralMarketODE(nn.Module):
    """
    Neural Market ODE combining physics and neural dynamics.

    ODE: d(log_price)/dt = G_phys + G_neural + G_impact

    Supports:
    - Euler integration (simple, fast)
    - RK2 integration (more accurate)
    - Action conditioning for market impact
    - Multi-step rollout

    All operations in LOG-SPACE.
    """
    n_assets: int = 64
    n_features: int = 10
    n_factors: int = 8
    dt: float = 1.0
    ode_method: str = 'euler'  # 'euler' or 'rk2'
    use_garch: bool = True
    use_transformer: bool = True
    use_action_conditioning: bool = True

    # Neural corrector parameters
    embed_dim: int = 128
    num_transformer_blocks: int = 2

    def setup(self):
        """Initialize sub-modules."""
        # Physics core
        self.physics_core = DifferentiableMarketPhysics(
            n_assets=self.n_assets,
            n_factors=self.n_factors,
            use_garch=self.use_garch,
        )

        # Neural corrector
        if self.use_transformer:
            self.neural_corrector = NeuralMarketCorrector(
                n_assets=self.n_assets,
                n_features=self.n_features,
                embed_dim=self.embed_dim,
                num_transformer_blocks=self.num_transformer_blocks,
            )
        else:
            self.neural_corrector = SimpleNeuralCorrector(
                n_assets=self.n_assets,
                n_features=self.n_features,
            )

        # Action impact (optional)
        if self.use_action_conditioning:
            self.market_impact = MarketImpact(n_assets=self.n_assets)

    def compute_tendency(
        self,
        log_prices: jnp.ndarray,
        log_returns: jnp.ndarray,
        moving_avg: jnp.ndarray,
        noise: jnp.ndarray,
        features: Optional[jnp.ndarray] = None,
        action: Optional[jnp.ndarray] = None,
        prev_variance: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, Dict]:
        """
        Compute total tendency (RHS of ODE).

        G_total = G_phys + G_neural + G_impact

        Args:
            log_prices: (batch, n_assets) current log prices
            log_returns: (batch, n_assets) recent log returns
            moving_avg: (batch, n_assets) moving average
            noise: (batch, n_assets) standard normal noise
            features: (batch, n_features) external features (optional)
            action: (batch,) action indices (optional)
            prev_variance: (batch, n_assets) previous variance for GARCH
            deterministic: Whether to use deterministic mode

        Returns:
            G_total: (batch, n_assets) total tendency
            aux: Dict with component tendencies
        """
        # Physics tendency
        physics_output, next_variance = self.physics_core(
            log_prices,
            log_returns,
            moving_avg,
            noise,
            self.dt,
            prev_variance,
        )
        G_phys = physics_output.G_total

        # Neural correction
        G_neural = self.neural_corrector(
            log_prices,
            log_returns,
            moving_avg,
            features,
            deterministic=deterministic,
        )

        # Action impact (if enabled and action provided)
        if self.use_action_conditioning and action is not None:
            G_impact = self.market_impact(action)
        else:
            G_impact = jnp.zeros_like(log_prices)

        # Total tendency
        G_total = G_phys + G_neural + G_impact

        aux = {
            'G_phys': G_phys,
            'G_neural': G_neural,
            'G_impact': G_impact,
            'physics_components': physics_output,
            'next_variance': next_variance,
        }

        return G_total, aux

    def _euler_step(
        self,
        log_prices: jnp.ndarray,
        log_returns: jnp.ndarray,
        moving_avg: jnp.ndarray,
        noise: jnp.ndarray,
        features: Optional[jnp.ndarray] = None,
        action: Optional[jnp.ndarray] = None,
        prev_variance: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> ODEOutput:
        """
        Single Euler integration step.

        y_{n+1} = y_n + dt * f(y_n)
        """
        G_total, aux = self.compute_tendency(
            log_prices, log_returns, moving_avg, noise,
            features, action, prev_variance, deterministic
        )

        next_log_prices = log_prices + self.dt * G_total

        return ODEOutput(
            next_log_prices=next_log_prices,
            G_phys=aux['G_phys'],
            G_neural=aux['G_neural'],
            G_impact=aux['G_impact'],
            G_total=G_total,
            next_variance=aux['next_variance'],
        )

    def _rk2_step(
        self,
        log_prices: jnp.ndarray,
        log_returns: jnp.ndarray,
        moving_avg: jnp.ndarray,
        noise: jnp.ndarray,
        features: Optional[jnp.ndarray] = None,
        action: Optional[jnp.ndarray] = None,
        prev_variance: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> ODEOutput:
        """
        Single RK2 (Midpoint method) integration step.

        k1 = f(y_n)
        k2 = f(y_n + dt/2 * k1)
        y_{n+1} = y_n + dt * k2

        More accurate than Euler for smooth tendencies.
        """
        # k1: tendency at current state
        k1, aux1 = self.compute_tendency(
            log_prices, log_returns, moving_avg, noise,
            features, action, prev_variance, deterministic
        )

        # Midpoint state
        log_prices_mid = log_prices + (self.dt / 2) * k1
        log_returns_mid = k1  # Approximate: returns are the change rate

        # k2: tendency at midpoint
        k2, aux2 = self.compute_tendency(
            log_prices_mid, log_returns_mid, moving_avg, noise,
            features, action, aux1['next_variance'], deterministic
        )

        # Final update
        next_log_prices = log_prices + self.dt * k2

        return ODEOutput(
            next_log_prices=next_log_prices,
            G_phys=aux2['G_phys'],
            G_neural=aux2['G_neural'],
            G_impact=aux2['G_impact'],
            G_total=k2,
            next_variance=aux2['next_variance'],
        )

    @nn.compact
    def __call__(
        self,
        log_prices: jnp.ndarray,
        log_returns: jnp.ndarray,
        moving_avg: jnp.ndarray,
        noise: jnp.ndarray,
        features: Optional[jnp.ndarray] = None,
        action: Optional[jnp.ndarray] = None,
        prev_variance: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, Dict]:
        """
        Perform single ODE step.

        Args:
            log_prices: (batch, n_assets) current log prices
            log_returns: (batch, n_assets) recent log returns
            moving_avg: (batch, n_assets) moving average
            noise: (batch, n_assets) standard normal noise
            features: (batch, n_features) external features (optional)
            action: (batch,) action indices (optional)
            prev_variance: (batch, n_assets) previous variance for GARCH
            deterministic: Whether to use deterministic mode

        Returns:
            next_log_prices: (batch, n_assets) predicted log prices
            aux: Dict with component tendencies
        """
        if self.ode_method == 'rk2':
            output = self._rk2_step(
                log_prices, log_returns, moving_avg, noise,
                features, action, prev_variance, deterministic
            )
        else:  # euler
            output = self._euler_step(
                log_prices, log_returns, moving_avg, noise,
                features, action, prev_variance, deterministic
            )

        aux = {
            'G_phys': output.G_phys,
            'G_neural': output.G_neural,
            'G_impact': output.G_impact,
            'G_total': output.G_total,
            'next_variance': output.next_variance,
        }

        return output.next_log_prices, aux

    def rollout(
        self,
        log_prices: jnp.ndarray,
        log_returns: jnp.ndarray,
        moving_avg: jnp.ndarray,
        rng: jnp.ndarray,
        features: Optional[jnp.ndarray] = None,
        action_sequence: Optional[jnp.ndarray] = None,
        num_steps: int = 10,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, Dict]:
        """
        Multi-step rollout using jax.lax.scan.

        Args:
            log_prices: (batch, n_assets) initial log prices
            log_returns: (batch, n_assets) initial log returns
            moving_avg: (batch, n_assets) initial moving average
            rng: Random key
            features: (batch, n_features) external features (constant)
            action_sequence: (batch, num_steps) action sequence (optional)
            num_steps: Number of steps to roll out
            deterministic: Whether to use deterministic mode

        Returns:
            trajectory: (batch, num_steps+1, n_assets) log price trajectory
            aux: Dict with aggregated statistics
        """
        batch = log_prices.shape[0]

        # Split RNG for each step
        rngs = jax.random.split(rng, num_steps)

        def scan_fn(carry, inputs):
            log_p, log_r, ma, var = carry
            step_rng, step_action = inputs

            noise = jax.random.normal(step_rng, (batch, self.n_assets))

            next_log_p, aux = self(
                log_p, log_r, ma, noise,
                features, step_action, var, deterministic
            )

            # Update states
            next_log_r = next_log_p - log_p
            next_ma = 0.9 * ma + 0.1 * next_log_p
            next_var = aux['next_variance']

            return (next_log_p, next_log_r, next_ma, next_var), next_log_p

        # Prepare action sequence
        if action_sequence is None:
            action_seq = jnp.zeros((num_steps, batch), dtype=jnp.int32)
        else:
            action_seq = jnp.transpose(action_sequence, (1, 0))  # (num_steps, batch)

        # Initial variance
        init_variance = jnp.ones_like(log_prices) * 0.01

        # Scan over steps
        _, trajectory = jax.lax.scan(
            scan_fn,
            (log_prices, log_returns, moving_avg, init_variance),
            (rngs, action_seq),
        )

        # trajectory: (num_steps, batch, n_assets)
        # Prepend initial state and transpose
        trajectory = jnp.concatenate([
            log_prices[None, :, :],
            trajectory
        ], axis=0)
        trajectory = jnp.transpose(trajectory, (1, 0, 2))  # (batch, num_steps+1, n_assets)

        return trajectory, {}


def create_training_loss_fn(model: NeuralMarketODE, config: Optional[Dict] = None):
    """
    Create training loss function for NeuralMarketODE.

    Returns a function that computes multi-step prediction loss
    with decaying weights for temporal consistency.

    Args:
        model: NeuralMarketODE instance
        config: Optional config dict with:
            - multi_step_k: number of prediction steps
            - step_decay: weight decay per step
            - physics_reg_weight: physics regularization weight

    Returns:
        loss_fn: Function (params, batch, rng) -> (loss, metrics)
    """
    config = config or {}
    multi_step_k = config.get('multi_step_k', 5)
    step_decay = config.get('step_decay', 0.9)
    physics_reg_weight = config.get('physics_reg_weight', 0.01)

    def loss_fn(params, batch, rng):
        """
        Compute multi-step prediction loss.

        Args:
            params: Model parameters
            batch: Dict with:
                - log_prices: (B, n_assets) initial
                - log_returns: (B, n_assets) initial
                - moving_avg: (B, n_assets) initial
                - features: (B, n_features)
                - log_price_sequence: (B, K+1, n_assets) ground truth
                - action_sequence: (B, K) optional
            rng: Random key

        Returns:
            loss: Scalar loss
            metrics: Dict with component losses
        """
        # Multi-step rollout
        pred_trajectory, _ = model.apply(
            params,
            batch['log_prices'],
            batch['log_returns'],
            batch['moving_avg'],
            rng,
            features=batch.get('features'),
            action_sequence=batch.get('action_sequence'),
            num_steps=multi_step_k,
            method=model.rollout,
        )

        # Ground truth: (B, K+1, n_assets)
        target_trajectory = batch['log_price_sequence']

        # Per-step MSE (skip initial state which matches by construction)
        # pred_trajectory[:, 1:] vs target_trajectory[:, 1:]
        pred_steps = pred_trajectory[:, 1:multi_step_k+1, :]  # (B, K, n_assets)
        target_steps = target_trajectory[:, 1:multi_step_k+1, :]  # (B, K, n_assets)

        step_mse = jnp.mean((pred_steps - target_steps) ** 2, axis=-1)  # (B, K)

        # Weighted sum with decay
        step_weights = jnp.array([step_decay ** k for k in range(multi_step_k)])
        weighted_mse = jnp.sum(step_mse * step_weights[None, :], axis=-1)  # (B,)
        prediction_loss = jnp.mean(weighted_mse)

        # Physics regularization
        from .physics_core import physics_regularization_loss
        physics_loss = physics_regularization_loss(params)

        # Total loss
        total_loss = prediction_loss + physics_reg_weight * physics_loss

        metrics = {
            'prediction_loss': prediction_loss,
            'physics_reg_loss': physics_loss,
            'step_1_mse': jnp.mean(step_mse[:, 0]),
            'step_k_mse': jnp.mean(step_mse[:, -1]),
            'total_loss': total_loss,
        }

        return total_loss, metrics

    return loss_fn


def create_neural_market_ode(
    n_assets: int = 64,
    n_features: int = 10,
    n_factors: int = 8,
    dt: float = 1.0,
    ode_method: str = 'euler',
    use_garch: bool = True,
    use_transformer: bool = True,
    use_action_conditioning: bool = True,
    rng: Optional[jnp.ndarray] = None,
) -> Tuple[NeuralMarketODE, Dict]:
    """
    Create and initialize NeuralMarketODE.

    Args:
        n_assets: Number of tradeable assets
        n_features: Number of external features
        n_factors: Number of latent factors for correlation
        dt: Time step
        ode_method: 'euler' or 'rk2'
        use_garch: Whether to use GARCH volatility
        use_transformer: Whether to use transformer corrector
        use_action_conditioning: Whether to model action impact
        rng: Random key for initialization

    Returns:
        model: Initialized model
        params: Initial parameters
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    model = NeuralMarketODE(
        n_assets=n_assets,
        n_features=n_features,
        n_factors=n_factors,
        dt=dt,
        ode_method=ode_method,
        use_garch=use_garch,
        use_transformer=use_transformer,
        use_action_conditioning=use_action_conditioning,
    )

    # Initialize with dummy inputs
    dummy_log_prices = jnp.zeros((1, n_assets))
    dummy_log_returns = jnp.zeros((1, n_assets))
    dummy_moving_avg = jnp.zeros((1, n_assets))
    dummy_noise = jnp.zeros((1, n_assets))
    dummy_features = jnp.zeros((1, n_features))
    dummy_action = jnp.zeros((1,), dtype=jnp.int32)

    params = model.init(
        rng,
        dummy_log_prices,
        dummy_log_returns,
        dummy_moving_avg,
        dummy_noise,
        dummy_features,
        dummy_action,
    )

    return model, params
