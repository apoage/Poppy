"""Mamba-based flow matching generator."""

import jax
import jax.numpy as jnp
from jax import Array
import flax.linen as nn
from typing import Optional, Tuple

from .bimamba import BiMambaBlockWithNorm, SelfAttentionWithNorm


def sinusoidal_embedding(t: Array, dim: int) -> Array:
    """Sinusoidal timestep embedding."""
    half_dim = dim // 2
    emb = jnp.log(10000) / jnp.maximum(half_dim - 1, 1)
    emb = jnp.exp(jnp.arange(half_dim) * -emb)

    if t.ndim == 0:
        t = t[None]

    emb = t[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)

    if dim % 2 == 1:
        emb = jnp.pad(emb, ((0, 0), (0, 1)))

    return emb


class AdaLN(nn.Module):
    """Adaptive Layer Normalization for conditioning."""
    d_model: int

    @nn.compact
    def __call__(self, x: Array, cond: Array) -> Tuple[Array, Array, Array]:
        """
        Args:
            x: Input to modulate
            cond: Conditioning vector (batch, d_cond)

        Returns:
            scale, shift, gate parameters
        """
        # Project conditioning to modulation parameters
        # 3 * d_model for scale, shift, gate
        params = nn.Dense(self.d_model * 3)(cond)
        scale, shift, gate = jnp.split(params, 3, axis=-1)

        # LOW-003 FIX: Gate initialized near zero (AdaLN-Zero style)
        # Subtract 5.0 so at init (gate ~ N(0,1)), sigmoid(-5) ≈ 0.007
        gate = nn.sigmoid(gate - 5.0)

        return scale, shift, gate


class HybridMambaFlow(nn.Module):
    """
    Hybrid Mamba-Attention backbone for flow matching.

    Uses 1:7 attention-to-Mamba ratio (Jamba pattern).
    Attention layers placed after initial Mamba compression.
    """
    d_model: int = 768
    n_layers: int = 24
    d_state: int = 16  # MED-004: Reduced from 64 to prevent OOM
    d_conv: int = 4
    expand: int = 2
    n_heads: int = 8
    attention_every: int = 8  # 1:7 ratio
    dropout: float = 0.0

    @nn.compact
    def __call__(
        self,
        x: Array,
        cond: Array,
        training: bool = False,
    ) -> Array:
        """
        Args:
            x: Input, shape (batch, seq, d_model)
            cond: Conditioning, shape (batch, d_cond)
            training: Training mode

        Returns:
            Output, shape (batch, seq, d_model)
        """
        batch, seq, _ = x.shape

        # AdaLN for each layer
        adaln = AdaLN(self.d_model)

        for i in range(self.n_layers):
            # Get modulation parameters
            scale, shift, gate = adaln(x.mean(axis=1), cond)  # Pool over sequence

            # Expand for broadcasting
            scale = scale[:, None, :]  # (batch, 1, d_model)
            shift = shift[:, None, :]
            gate = gate[:, None, :]

            # Pre-norm with modulation
            h = nn.LayerNorm()(x)
            h = h * (1 + scale) + shift

            # Choose layer type
            if i % self.attention_every == 0 and i > 0:
                # Sparse attention (every 8th layer, not at start)
                h = SelfAttentionWithNorm(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    dropout=self.dropout,
                    name=f'attn_{i}'
                )(h, training)
            else:
                # BiMamba
                h = BiMambaBlockWithNorm(
                    d_model=self.d_model,
                    d_state=self.d_state,
                    d_conv=self.d_conv,
                    expand=self.expand,
                    dropout=self.dropout,
                    name=f'mamba_{i}'
                )(h, (scale.squeeze(1), shift.squeeze(1)), training)

            # Gated residual
            x = x + gate * h

        # Final norm
        x = nn.LayerNorm()(x)

        return x


class MambaFlowGenerator(nn.Module):
    """
    Complete Mamba-based flow matching generator.

    Predicts velocity field for flow matching ODE.
    """
    d_model: int = 768
    n_layers: int = 24
    d_state: int = 16  # MED-004: Reduced from 64 to prevent OOM
    n_heads: int = 8
    attention_every: int = 8
    dropout: float = 0.0
    n_regimes: int = 5

    @nn.compact
    def __call__(
        self,
        x: Array,
        t: Array,
        regime: Optional[Array] = None,
        skeleton_cond: Optional[Tuple[Array, Array, Array]] = None,
        training: bool = False,
    ) -> Array:
        """
        Predict velocity field v(x_t, t).

        Args:
            x: Noisy input, shape (batch, seq, input_dim)
            t: Timestep, shape (batch,) in [0, 1]
            regime: Optional regime indices, shape (batch,)
            skeleton_cond: Optional (gamma, beta, h0) from skeleton encoder
            training: Training mode

        Returns:
            Velocity, shape (batch, seq, input_dim)
        """
        batch, seq, input_dim = x.shape

        # Input projection
        h = nn.Dense(self.d_model, name='input_proj')(x)

        # Time embedding
        t_emb = sinusoidal_embedding(t, self.d_model // 2)
        t_emb = nn.Dense(self.d_model)(t_emb)
        t_emb = nn.silu(t_emb)
        t_emb = nn.Dense(self.d_model)(t_emb)

        # Regime embedding (if provided)
        if regime is not None:
            regime_emb = nn.Embed(
                num_embeddings=self.n_regimes,
                features=self.d_model,
                name='regime_embed'
            )(regime)
            cond = t_emb + regime_emb
        else:
            cond = t_emb

        # Skeleton conditioning (if provided)
        if skeleton_cond is not None:
            gamma_s, beta_s, h0_s = skeleton_cond
            # Include all three components: gamma (scale), beta (shift), h0 (initial state)
            cond = cond + gamma_s + beta_s + h0_s

        # Hybrid Mamba-Attention backbone
        h = HybridMambaFlow(
            d_model=self.d_model,
            n_layers=self.n_layers,
            d_state=self.d_state,
            n_heads=self.n_heads,
            attention_every=self.attention_every,
            dropout=self.dropout,
        )(h, cond, training)

        # Output projection (velocity)
        v = nn.Dense(input_dim, name='output_proj')(h)

        return v


class PureMambaFlowGenerator(nn.Module):
    """
    Pure Mamba generator (no attention layers).

    More efficient but potentially less expressive.
    """
    d_model: int = 768
    n_layers: int = 24
    d_state: int = 16  # MED-004: Reduced from 64 to prevent OOM
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.0
    n_regimes: int = 5

    @nn.compact
    def __call__(
        self,
        x: Array,
        t: Array,
        regime: Optional[Array] = None,
        training: bool = False,
    ) -> Array:
        """Predict velocity field."""
        batch, seq, input_dim = x.shape

        # Input projection
        h = nn.Dense(self.d_model)(x)

        # Time embedding
        t_emb = sinusoidal_embedding(t, self.d_model // 2)
        t_emb = nn.Dense(self.d_model)(nn.silu(nn.Dense(self.d_model)(t_emb)))

        # Regime embedding
        if regime is not None:
            regime_emb = nn.Embed(self.n_regimes, self.d_model)(regime)
            cond = t_emb + regime_emb
        else:
            cond = t_emb

        # FiLM parameters
        film = nn.Dense(self.d_model * 2)(cond)
        gamma, beta = jnp.split(film, 2, axis=-1)

        # Pure Mamba layers
        for i in range(self.n_layers):
            h = BiMambaBlockWithNorm(
                d_model=self.d_model,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                dropout=self.dropout,
            )(h, (gamma, beta), training)

        # Final norm and output
        h = nn.LayerNorm()(h)
        v = nn.Dense(input_dim)(h)

        return v


def flow_matching_loss(
    model,
    params,
    x_1: Array,
    regime: Optional[Array],
    key: Array,
) -> Array:
    """
    Compute flow matching loss.

    OT-CFM with straight paths.

    Args:
        model: MambaFlowGenerator
        params: Model parameters
        x_1: Target data, shape (batch, seq, features)
        regime: Optional regime indices
        key: PRNG key

    Returns:
        Scalar loss
    """
    batch = x_1.shape[0]

    # Sample noise
    key, noise_key = jax.random.split(key)
    x_0 = jax.random.normal(noise_key, x_1.shape)

    # Sample timestep
    key, t_key = jax.random.split(key)
    t = jax.random.uniform(t_key, (batch,))

    # Interpolate (straight path)
    t_expand = t[:, None, None]
    x_t = (1 - t_expand) * x_0 + t_expand * x_1

    # Target velocity
    target_v = x_1 - x_0

    # Predict velocity
    pred_v = model.apply(params, x_t, t, regime)

    # MSE loss
    loss = jnp.mean((pred_v - target_v) ** 2)

    return loss
