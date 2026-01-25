"""
Flow Matching Generator for Market Data.

Uses Conditional Flow Matching (CFM) for fast single-step generation.
Much faster than DDPM while maintaining quality.

Key idea:
    - Learn vector field v(x_t, t) that transports noise x_0 to data x_1
    - Training: x_t = (1-t)*x_0 + t*x_1, predict v = x_1 - x_0
    - Inference: Single step from noise to data

References:
    - "Flow Matching for Generative Modeling" (Lipman et al., 2023)
    - "Improving and Generalizing Flow-Based Generative Models" (Tong et al., 2023)
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple, Dict
from functools import partial


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for conditioning on t."""

    dim: int

    @nn.compact
    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """
        Create time embeddings.

        Args:
            t: (batch,) time values in [0, 1]

        Returns:
            (batch, dim) time embeddings
        """
        half_dim = self.dim // 2
        emb_scale = jnp.log(10000.0) / jnp.maximum(half_dim - 1, 1)
        emb = jnp.exp(-emb_scale * jnp.arange(half_dim))  # (half_dim,)

        # t: (batch,) -> (batch, 1)
        t = t[:, None]
        emb = t * emb[None, :]  # (batch, half_dim)

        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)  # (batch, dim)
        return emb


class RegimeEmbedding(nn.Module):
    """Embedding for regime conditioning (crash, rally, sideways, volatile)."""

    dim: int
    num_regimes: int = 5  # 4 regimes + 1 for unconditional

    @nn.compact
    def __call__(self, regime_idx: jnp.ndarray) -> jnp.ndarray:
        """
        Get regime embedding.

        Args:
            regime_idx: (batch,) regime indices (0-3, or 4 for unconditional)

        Returns:
            (batch, dim) regime embeddings
        """
        embedding = nn.Embed(num_embeddings=self.num_regimes, features=self.dim)
        return embedding(regime_idx)


class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization.

    Modulates layer norm with time/condition embeddings.
    """

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, cond: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Apply adaptive layer norm.

        Args:
            x: (batch, ..., dim) input
            cond: (batch, cond_dim) conditioning

        Returns:
            Modulated output
        """
        dim = x.shape[-1]

        # Compute scale and shift from conditioning
        scale = nn.Dense(dim)(cond)
        shift = nn.Dense(dim)(cond)

        # Reshape for broadcasting
        for _ in range(x.ndim - 2):
            scale = scale[:, None, ...]
            shift = shift[:, None, ...]

        # Layer norm then modulate
        x = nn.LayerNorm()(x)
        return x * (1 + scale) + shift


class FlowMatchingBlock(nn.Module):
    """
    Transformer block for flow matching with time conditioning.

    Uses AdaLN for time/regime conditioning instead of standard LayerNorm.
    """

    dim: int
    num_heads: int
    ff_dim: int
    dropout: float = 0.0

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        cond: jnp.ndarray,
        training: bool = False,
    ) -> jnp.ndarray:
        """
        Process input through flow matching block.

        Args:
            x: (batch, seq, dim) input
            cond: (batch, cond_dim) time + regime conditioning
            training: Whether in training mode

        Returns:
            (batch, seq, dim) output
        """
        # Self-attention with AdaLN
        x_norm = AdaLN()(x, cond)
        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.dim,
            dropout_rate=self.dropout,
            deterministic=not training,
        )
        x_attn = attn(x_norm, x_norm)
        x = x + x_attn

        # Feed-forward with AdaLN
        x_norm = AdaLN()(x, cond)
        x_ff = nn.Dense(self.ff_dim)(x_norm)
        x_ff = nn.gelu(x_ff)
        x_ff = nn.Dropout(rate=self.dropout, deterministic=not training)(x_ff)
        x_ff = nn.Dense(self.dim)(x_ff)
        x = x + x_ff

        return x


class VelocityNetwork(nn.Module):
    """
    Neural network to predict velocity field v(x_t, t, c).

    Architecture:
        1. Project input to hidden dim
        2. Add time and regime embeddings
        3. Transformer blocks
        4. Project to output dim (same as input)
    """

    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    ff_dim: int = 2048
    dropout: float = 0.0
    cond_dim: int = 256  # Time + regime embedding dim

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,  # (batch, seq, input_dim)
        t: jnp.ndarray,  # (batch,) time in [0, 1]
        regime: Optional[jnp.ndarray] = None,  # (batch,) regime idx
        training: bool = False,
    ) -> jnp.ndarray:
        """
        Predict velocity field.

        Args:
            x: (batch, seq, input_dim) noisy input at time t
            t: (batch,) time values
            regime: (batch,) optional regime conditioning
            training: Whether in training mode

        Returns:
            (batch, seq, input_dim) predicted velocity
        """
        batch, seq, input_dim = x.shape

        # Project input to hidden dimension
        h = nn.Dense(self.hidden_dim, name="input_proj")(x)

        # Time embedding
        time_emb = TimeEmbedding(dim=self.cond_dim // 2)(t)  # (batch, cond_dim/2)
        time_emb = nn.Dense(self.cond_dim)(time_emb)

        # Regime embedding (or zeros for unconditional)
        if regime is not None:
            regime_emb = RegimeEmbedding(dim=self.cond_dim // 2)(regime)
            regime_emb = nn.Dense(self.cond_dim)(regime_emb)
        else:
            regime_emb = jnp.zeros((batch, self.cond_dim))

        # Combine conditioning
        cond = time_emb + regime_emb  # (batch, cond_dim)

        # Transformer blocks
        for i in range(self.num_layers):
            h = FlowMatchingBlock(
                dim=self.hidden_dim,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout=self.dropout,
                name=f"block_{i}",
            )(h, cond, training=training)

        # Final layer norm
        h = nn.LayerNorm()(h)

        # Project back to input dimension
        v = nn.Dense(input_dim, name="output_proj")(h)

        return v


class FlowMatchingGenerator(nn.Module):
    """
    Flow Matching generator for market sequences.

    Training:
        - Sample x_0 ~ N(0, 1), x_1 = data
        - Sample t ~ U[0, 1]
        - Compute x_t = (1-t)*x_0 + t*x_1 + sigma_t * noise (OT-CFM path)
        - Predict v = velocity_net(x_t, t)
        - Loss = ||v - (x_1 - x_0)||^2

    Inference:
        - Sample x_0 ~ N(0, 1)
        - Single step: x_1 = x_0 + velocity_net(x_0, t=0) (approx)
        - Or multi-step: Euler integration
    """

    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    ff_dim: int = 2048
    dropout: float = 0.0
    cond_dim: int = 256
    sigma_min: float = 1e-4  # Minimum noise for stability

    def setup(self):
        self.velocity_net = VelocityNetwork(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            dropout=self.dropout,
            cond_dim=self.cond_dim,
        )

    def get_interpolation(
        self,
        x_0: jnp.ndarray,
        x_1: jnp.ndarray,
        t: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute interpolation and target velocity for CFM.

        Uses optimal transport path: x_t = (1-t)*x_0 + t*x_1

        Args:
            x_0: (batch, ...) source (noise)
            x_1: (batch, ...) target (data)
            t: (batch,) time values

        Returns:
            x_t: Interpolated point
            v_target: Target velocity (x_1 - x_0)
        """
        # Reshape t for broadcasting
        t_broadcast = t
        for _ in range(x_0.ndim - 1):
            t_broadcast = t_broadcast[..., None]

        # Optimal transport interpolation
        x_t = (1 - t_broadcast) * x_0 + t_broadcast * x_1

        # Add small noise for stability (variance preserving)
        # sigma_t = sigma_min
        # x_t = x_t + sigma_t * noise  # Optional, can skip for OT-CFM

        # Target velocity (constant along OT path)
        v_target = x_1 - x_0

        return x_t, v_target

    def __call__(
        self,
        x_1: jnp.ndarray,
        t: jnp.ndarray,
        key: jax.Array,
        regime: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Training forward pass.

        Args:
            x_1: (batch, seq, dim) target data
            t: (batch,) time values
            key: Random key for noise
            regime: (batch,) optional regime conditioning
            training: Whether in training mode

        Returns:
            v_pred: Predicted velocity
            v_target: Target velocity
            x_t: Interpolated point (for visualization)
        """
        # Sample noise
        x_0 = jax.random.normal(key, x_1.shape)

        # Get interpolation and target
        x_t, v_target = self.get_interpolation(x_0, x_1, t)

        # Predict velocity
        v_pred = self.velocity_net(x_t, t, regime, training=training)

        return v_pred, v_target, x_t

    def sample(
        self,
        key: jax.Array,
        shape: Tuple[int, ...],
        regime: Optional[jnp.ndarray] = None,
        num_steps: int = 1,
    ) -> jnp.ndarray:
        """
        Generate samples using Euler integration.

        Args:
            key: Random key
            shape: Output shape (batch, seq, dim)
            regime: (batch,) optional regime conditioning
            num_steps: Number of integration steps (1 = single step)

        Returns:
            (batch, seq, dim) generated samples
        """
        # Start from noise
        x = jax.random.normal(key, shape)

        # Euler integration from t=0 to t=1
        num_steps = max(num_steps, 1)  # Guard against zero
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = jnp.full((shape[0],), i * dt)
            v = self.velocity_net(x, t, regime, training=False)
            x = x + dt * v

        return x

    def sample_single_step(
        self,
        key: jax.Array,
        shape: Tuple[int, ...],
        regime: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Generate samples with single step (fastest).

        Approximation that works well for well-trained models.

        Args:
            key: Random key
            shape: Output shape
            regime: Optional regime conditioning

        Returns:
            Generated samples
        """
        # Start from noise
        x_0 = jax.random.normal(key, shape)

        # Single step at t=0
        t = jnp.zeros((shape[0],))
        v = self.velocity_net(x_0, t, regime, training=False)

        # Move from t=0 to t=1 in one step
        x_1 = x_0 + v

        return x_1


def flow_matching_loss(
    v_pred: jnp.ndarray,
    v_target: jnp.ndarray,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Compute flow matching loss (MSE on velocity).

    Args:
        v_pred: Predicted velocity
        v_target: Target velocity (x_1 - x_0)

    Returns:
        loss: Scalar loss
        metrics: Dictionary with loss components
    """
    # MSE loss
    loss = jnp.mean((v_pred - v_target) ** 2)

    # Per-dimension losses for monitoring
    loss_per_dim = jnp.mean((v_pred - v_target) ** 2, axis=(0, 1))

    metrics = {
        "loss": loss,
        "loss_mean": jnp.mean(loss_per_dim),
        "loss_std": jnp.std(loss_per_dim),
    }

    return loss, metrics


def create_train_step(model: FlowMatchingGenerator, optimizer):
    """
    Create JIT-compiled training step.

    Args:
        model: FlowMatchingGenerator instance
        optimizer: Optax optimizer

    Returns:
        train_step function
    """

    @jax.jit
    def train_step(state, batch, key):
        """
        Single training step.

        Args:
            state: Training state (params, opt_state)
            batch: Dictionary with 'data' and optionally 'regime'
            key: Random key

        Returns:
            new_state, metrics
        """
        x_1 = batch["data"]
        regime = batch.get("regime", None)

        # Split key
        key, t_key, noise_key = jax.random.split(key, 3)

        # Sample time uniformly
        t = jax.random.uniform(t_key, (x_1.shape[0],))

        def loss_fn(params):
            v_pred, v_target, _ = model.apply(
                params, x_1, t, noise_key, regime, training=True
            )
            loss, metrics = flow_matching_loss(v_pred, v_target)
            return loss, metrics

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params
        )

        # Update parameters
        updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)

        new_state = state.replace(params=new_params, opt_state=new_opt_state)

        return new_state, metrics

    return train_step


# =============================================================================
# Regime utilities
# =============================================================================
REGIME_TO_IDX = {
    "crash": 0,
    "rally": 1,
    "sideways": 2,
    "volatile": 3,
    "unconditional": 4,
}

IDX_TO_REGIME = {v: k for k, v in REGIME_TO_IDX.items()}


def regime_to_idx(regime: str) -> int:
    """Convert regime string to index."""
    return REGIME_TO_IDX.get(regime, 4)  # Default to unconditional


def idx_to_regime(idx: int) -> str:
    """Convert index to regime string."""
    return IDX_TO_REGIME.get(idx, "unconditional")


# =============================================================================
# Test
# =============================================================================
if __name__ == "__main__":
    import optax

    key = jax.random.PRNGKey(42)

    # Create model
    model = FlowMatchingGenerator(
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        ff_dim=512,
    )

    # Dummy data: (batch, seq, dim)
    batch_size = 4
    seq_len = 100
    dim = 64 * 5  # 64 assets * 5 features (flattened)

    x_1 = jax.random.normal(key, (batch_size, seq_len, dim))
    t = jax.random.uniform(key, (batch_size,))
    regime = jnp.array([0, 1, 2, 3])  # Different regimes

    # Initialize
    key, init_key, noise_key = jax.random.split(key, 3)
    params = model.init(init_key, x_1, t, noise_key, regime, training=False)

    # Forward pass
    v_pred, v_target, x_t = model.apply(params, x_1, t, noise_key, regime, training=False)

    print(f"Input shape: {x_1.shape}")
    print(f"Velocity pred shape: {v_pred.shape}")
    print(f"Velocity target shape: {v_target.shape}")
    print(f"Interpolated shape: {x_t.shape}")

    # Compute loss
    loss, metrics = flow_matching_loss(v_pred, v_target)
    print(f"\nLoss: {loss:.4f}")
    print(f"Metrics: {metrics}")

    # Test sampling
    print("\n--- Sampling Test ---")
    key, sample_key = jax.random.split(key)

    # Single step
    samples_1step = model.apply(
        params,
        sample_key,
        (batch_size, seq_len, dim),
        regime,
        method=model.sample_single_step,
    )
    print(f"Single-step samples shape: {samples_1step.shape}")
    print(f"Single-step samples range: [{samples_1step.min():.3f}, {samples_1step.max():.3f}]")

    # Multi-step (4 steps)
    samples_4step = model.apply(
        params,
        sample_key,
        (batch_size, seq_len, dim),
        regime,
        4,
        method=model.sample,
    )
    print(f"4-step samples shape: {samples_4step.shape}")
    print(f"4-step samples range: [{samples_4step.min():.3f}, {samples_4step.max():.3f}]")

    # Count parameters
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"\nTotal parameters: {n_params:,} ({n_params / 1e6:.2f}M)")

    # Estimate memory (FP16)
    mem_mb = n_params * 2 / (1024**2)
    print(f"Model memory (FP16): {mem_mb:.1f} MB")
