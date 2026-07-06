"""Mamba block implementation in Flax."""

import jax
import jax.numpy as jnp
from jax import Array, lax
import flax.linen as nn
from typing import Optional, Tuple

from .ssm import selective_scan


class CausalConv1D(nn.Module):
    """Causal 1D convolution for local context."""
    features: int
    kernel_size: int = 4

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """
        Args:
            x: Input, shape (batch, seq, features)
        Returns:
            Output, shape (batch, seq, features)
        """
        # Left-pad for causal convolution
        padding = self.kernel_size - 1
        x_padded = jnp.pad(x, ((0, 0), (padding, 0), (0, 0)), mode='edge')

        # 1D depthwise convolution (Mamba standard)
        x = nn.Conv(
            features=self.features,
            kernel_size=(self.kernel_size,),
            padding='VALID',
            feature_group_count=self.features,  # Depthwise conv
        )(x_padded)

        return x


class MambaBlock(nn.Module):
    """
    Selective State Space Model block.

    Implements the core Mamba architecture with:
    - Linear expansion/projection
    - Causal convolution for local context
    - Selective SSM for content-aware filtering
    - Gated output
    """
    d_model: int
    d_state: int = 16      # SSM state dimension - MED-004: Reduced from 64 to prevent OOM
    d_conv: int = 4        # Local convolution kernel size
    expand: int = 2        # Expansion factor
    use_parallel_scan: bool = False  # CRIT-004 FIX: Default to sequential to avoid memory explosion

    @nn.compact
    def __call__(
        self,
        x: Array,
        cond: Optional[Tuple[Array, Array]] = None,
    ) -> Array:
        """
        Args:
            x: Input, shape (batch, seq, d_model)
            cond: Optional (gamma, beta) for FiLM conditioning

        Returns:
            Output, shape (batch, seq, d_model)
        """
        batch, seq, _ = x.shape
        d_inner = self.d_model * self.expand

        # 1. Linear projection with gating
        # Project to 2*d_inner: one for main path, one for gate
        x_proj = nn.Dense(d_inner * 2, use_bias=False)(x)
        x_main, x_gate = jnp.split(x_proj, 2, axis=-1)

        # 2. Causal convolution (local context)
        x_conv = CausalConv1D(d_inner, self.d_conv)(x_main)
        x_conv = nn.silu(x_conv)

        # 3. SSM parameters (selective = input-dependent)
        # Compute delta (timestep), B (input gate), C (output gate)
        x_ssm = nn.Dense(self.d_state * 2 + d_inner, use_bias=False)(x_conv)

        # Split: delta (d_inner), B (d_state), C (d_state)
        delta_proj, B, C = jnp.split(
            x_ssm,
            [d_inner, d_inner + self.d_state],
            axis=-1
        )

        # Delta: learned bias + softplus for positivity
        dt_bias = self.param(
            'dt_bias',
            nn.initializers.uniform(scale=1.0),
            (d_inner,)
        )
        delta = nn.softplus(delta_proj + dt_bias)

        # 4. A parameter (learned log scale, not input-dependent)
        # Initialize for stable dynamics: A = -exp(uniform)
        A_log = self.param(
            'A_log',
            lambda key, shape: jax.random.uniform(key, shape, minval=-4, maxval=-1),
            (d_inner, self.d_state)
        )
        A = -jnp.exp(A_log)

        # 5. Selective scan
        y = selective_scan(x_conv, delta, A, B, C, use_parallel=self.use_parallel_scan)

        # 6. Gated output
        y = y * nn.silu(x_gate)

        # 7. Output projection (FIRST - reduces d_inner -> d_model)
        y = nn.Dense(self.d_model, use_bias=False)(y)

        # 8. FiLM conditioning (AFTER projection - CRIT-001 FIX)
        # Now y has shape (batch, seq, d_model) matching gamma/beta dimensions
        if cond is not None:
            gamma, beta = cond
            # Broadcast to sequence if needed
            if gamma.ndim == 2:  # (batch, d_model)
                gamma = gamma[:, None, :]
                beta = beta[:, None, :]
            y = y * (1 + gamma) + beta

        return y


class MambaBlockWithNorm(nn.Module):
    """Mamba block with LayerNorm and residual connection."""
    d_model: int
    d_state: int = 16  # MED-004: Reduced from 64 to prevent OOM
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.0

    @nn.compact
    def __call__(
        self,
        x: Array,
        cond: Optional[Tuple[Array, Array]] = None,
        training: bool = False,
    ) -> Array:
        """
        Args:
            x: Input, shape (batch, seq, d_model)
            cond: Optional (gamma, beta) for FiLM conditioning
            training: Training mode (for dropout)

        Returns:
            Output, shape (batch, seq, d_model)
        """
        # Pre-norm
        h = nn.LayerNorm()(x)

        # Mamba block
        h = MambaBlock(
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
        )(h, cond)

        # Dropout
        if self.dropout > 0 and training:
            h = nn.Dropout(rate=self.dropout, deterministic=not training)(h)

        # Residual
        return x + h
