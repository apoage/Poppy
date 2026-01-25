"""
Conv-Based Flow Matching Generator for Market Data.

TASK_111: Combines conv architecture from fast_generator.py (O(n) memory, 302M params)
with training pipeline from diff_teacher (OT-CFM loss, data loading).

Architecture:
    - Dilated causal 1D convolutions (WaveNet-style)
    - FiLM conditioning for time and regime
    - O(n) memory vs O(n²) for transformer attention
    - Handles 1440+ step sequences in 6GB VRAM

Memory comparison for 1440 steps × 15 envs:
    Transformer: ~18GB (n² attention matrices)
    Conv:        ~5GB (linear in sequence length)

Training uses same OT-CFM loss as transformer version.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple, Dict
from functools import partial


# =============================================================================
# Time and Regime Embeddings
# =============================================================================

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
        emb = jnp.exp(-emb_scale * jnp.arange(half_dim))
        t = t[:, None]
        emb = t * emb[None, :]
        return jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)


class RegimeEmbedding(nn.Module):
    """Embedding for regime conditioning (crash, rally, sideways, volatile, calm)."""

    dim: int
    num_regimes: int = 5  # crash, rally, sideways, volatile, calm

    @nn.compact
    def __call__(self, regime_idx: jnp.ndarray) -> jnp.ndarray:
        """
        Get regime embedding.

        Args:
            regime_idx: (batch,) regime indices (0-4)

        Returns:
            (batch, dim) regime embeddings
        """
        return nn.Embed(num_embeddings=self.num_regimes, features=self.dim)(regime_idx)


# =============================================================================
# Causal Convolution Blocks
# =============================================================================

class CausalConv1D(nn.Module):
    """Causal 1D convolution with dilation for temporal modeling."""

    features: int
    kernel_size: int = 3
    dilation: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (batch, seq, features)
        # Causal padding: pad left side only
        pad_len = (self.kernel_size - 1) * self.dilation
        x_padded = jnp.pad(x, ((0, 0), (pad_len, 0), (0, 0)), mode='edge')

        # 1D conv over time dimension
        conv = nn.Conv(
            features=self.features,
            kernel_size=(self.kernel_size,),
            kernel_dilation=(self.dilation,),
            padding='VALID',
        )
        return conv(x_padded)


class DilatedConvBlock(nn.Module):
    """
    Dilated causal convolution block for temporal flow matching.

    Uses WaveNet-style dilated convolutions: O(n) memory, O(log n) receptive field.
    Much more memory efficient than attention for long sequences.
    """

    dim: int = 256
    ff_dim: int = 512
    kernel_size: int = 3
    dilation: int = 1
    dropout: float = 0.0

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        cond: jnp.ndarray,
        training: bool = False,
    ) -> jnp.ndarray:
        # Conditioning modulation (FiLM: scale/shift)
        scale = nn.Dense(self.dim)(cond)[:, None, :]
        shift = nn.Dense(self.dim)(cond)[:, None, :]

        # Layer norm + modulate
        h = nn.LayerNorm()(x)
        h = h * (1 + scale) + shift

        # Dilated causal convolution (replaces attention)
        h = CausalConv1D(
            features=self.dim,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
        )(h)
        h = nn.gelu(h)

        # Dropout
        if self.dropout > 0.0:
            h = nn.Dropout(rate=self.dropout, deterministic=not training)(h)

        # Residual connection
        x = x + h

        # Feed-forward
        h = nn.LayerNorm()(x)
        h = nn.Dense(self.ff_dim)(h)
        h = nn.gelu(h)
        if self.dropout > 0.0:
            h = nn.Dropout(rate=self.dropout, deterministic=not training)(h)
        h = nn.Dense(self.dim)(h)
        x = x + h

        return x


# =============================================================================
# Conv Velocity Network
# =============================================================================

class ConvVelocityNet(nn.Module):
    """
    Conv-based velocity network for flow matching.

    Uses dilated causal convolutions for O(n) memory scaling.
    Receptive field grows exponentially with dilation pattern 1,2,4,8,16,32.
    """

    hidden_dim: int = 1536
    num_layers: int = 16
    ff_dim: int = 3072
    cond_dim: int = 768
    kernel_size: int = 3
    dropout: float = 0.0

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
        time_emb = TimeEmbedding(dim=self.cond_dim // 2)(t)
        time_emb = nn.Dense(self.cond_dim)(time_emb)

        # Regime embedding (or zeros for unconditional)
        if regime is not None:
            regime_emb = RegimeEmbedding(dim=self.cond_dim // 2)(regime)
            regime_emb = nn.Dense(self.cond_dim)(regime_emb)
        else:
            regime_emb = jnp.zeros((batch, self.cond_dim))

        # Combine conditioning
        cond = time_emb + regime_emb  # (batch, cond_dim)

        # Dilated conv blocks with exponentially growing receptive field
        # Dilation pattern: 1, 2, 4, 8, 16, 32 covers ~64 step receptive field per cycle
        for i in range(self.num_layers):
            dilation = 2 ** (i % 6)  # Cycle dilations for very deep nets
            h = DilatedConvBlock(
                dim=self.hidden_dim,
                ff_dim=self.ff_dim,
                kernel_size=self.kernel_size,
                dilation=dilation,
                dropout=self.dropout,
                name=f"block_{i}",
            )(h, cond, training=training)

        # Final layer norm
        h = nn.LayerNorm()(h)

        # Project back to input dimension
        v = nn.Dense(input_dim, name="output_proj")(h)

        return v


# =============================================================================
# Conv Flow Matching Generator
# =============================================================================

class ConvFlowMatchingGenerator(nn.Module):
    """
    Conv-based Flow Matching generator for market sequences.

    Training:
        - Sample x_0 ~ N(0, 1), x_1 = data
        - Sample t ~ U[0, 1]
        - Compute x_t = (1-t)*x_0 + t*x_1 (OT-CFM path)
        - Predict v = velocity_net(x_t, t)
        - Loss = ||v - (x_1 - x_0)||^2

    Inference:
        - Sample x_0 ~ N(0, 1)
        - Single step: x_1 = x_0 + velocity_net(x_0, t=0) (approx)
        - Or multi-step: Euler integration

    Memory: O(n) vs O(n²) for transformer - handles 1440+ steps in 6GB VRAM.
    """

    hidden_dim: int = 1536
    num_layers: int = 16
    ff_dim: int = 3072
    cond_dim: int = 768
    kernel_size: int = 3
    dropout: float = 0.0
    sigma_min: float = 1e-4  # Minimum noise for stability

    def setup(self):
        self.velocity_net = ConvVelocityNet(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            ff_dim=self.ff_dim,
            cond_dim=self.cond_dim,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
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


# =============================================================================
# Loss Function
# =============================================================================

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


# =============================================================================
# Regime utilities (same as flow_matching.py)
# =============================================================================

REGIME_TO_IDX = {
    "crash": 0,
    "rally": 1,
    "sideways": 2,
    "volatile": 3,
    "calm": 4,
}

IDX_TO_REGIME = {v: k for k, v in REGIME_TO_IDX.items()}


def regime_to_idx(regime: str) -> int:
    """Convert regime string to index."""
    return REGIME_TO_IDX.get(regime, 2)  # Default to sideways


def idx_to_regime(idx: int) -> str:
    """Convert index to regime string."""
    return IDX_TO_REGIME.get(idx, "sideways")


# =============================================================================
# Factory Functions
# =============================================================================

def create_conv_generator(
    hidden_dim: int = 1536,
    num_layers: int = 16,
    ff_dim: Optional[int] = None,
    cond_dim: Optional[int] = None,
    dropout: float = 0.0,
) -> ConvFlowMatchingGenerator:
    """
    Create conv flow matching generator.

    Defaults to 302M param config (1536d, 16L) that fits in 5GB VRAM.

    Args:
        hidden_dim: Model width (1536=302M, 1024=101M, 512=17M params)
        num_layers: Depth (16=deep receptive field, 12=balanced, 8=fast)
        ff_dim: Feed-forward dimension (default: 2 * hidden_dim)
        cond_dim: Conditioning dimension (default: hidden_dim // 2)
        dropout: Dropout rate (default: 0.0)
    """
    if ff_dim is None:
        ff_dim = hidden_dim * 2
    if cond_dim is None:
        cond_dim = hidden_dim // 2

    return ConvFlowMatchingGenerator(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        ff_dim=ff_dim,
        cond_dim=cond_dim,
        dropout=dropout,
    )


def count_parameters(params) -> int:
    """Count total parameters in a pytree."""
    return sum(p.size for p in jax.tree_util.tree_leaves(params))


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Conv Flow Matching Generator Test")
    print("=" * 60)

    key = jax.random.PRNGKey(42)

    # Test different configs
    configs = [
        {"hidden_dim": 512, "num_layers": 8, "name": "Small (17M)"},
        {"hidden_dim": 1024, "num_layers": 12, "name": "Medium (101M)"},
        {"hidden_dim": 1536, "num_layers": 16, "name": "Large (302M)"},
    ]

    for cfg in configs:
        print(f"\n--- {cfg['name']} ---")

        model = create_conv_generator(
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
        )

        # Dummy data
        batch_size = 4
        seq_len = 400
        dim = 64 * 5  # 64 assets * 5 features

        x_1 = jax.random.normal(key, (batch_size, seq_len, dim))
        t = jax.random.uniform(key, (batch_size,))
        regime = jnp.array([0, 1, 2, 3])

        # Initialize
        key, init_key, noise_key = jax.random.split(key, 3)
        params = model.init(init_key, x_1, t, noise_key, regime, training=False)

        n_params = count_parameters(params)
        print(f"Parameters: {n_params:,} ({n_params / 1e6:.2f}M)")

        # Forward pass
        v_pred, v_target, x_t = model.apply(params, x_1, t, noise_key, regime, training=False)
        print(f"Velocity shape: {v_pred.shape}")

        # Loss
        loss, metrics = flow_matching_loss(v_pred, v_target)
        print(f"Loss: {loss:.4f}")

        # Sampling
        key, sample_key = jax.random.split(key)
        samples = model.apply(
            params,
            sample_key,
            (batch_size, seq_len, dim),
            regime,
            method=model.sample_single_step,
        )
        print(f"Sample shape: {samples.shape}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
