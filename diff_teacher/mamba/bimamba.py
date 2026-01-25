"""Bidirectional Mamba for generative models."""

import jax.numpy as jnp
from jax import Array
import flax.linen as nn
from typing import Optional, Tuple

from .mamba_block import MambaBlock


class BiMambaBlock(nn.Module):
    """
    Bidirectional Mamba block.

    For generative models (diffusion/flow matching), we need to
    process the sequence in both directions since we're denoising
    the entire sequence at once.
    """
    d_model: int
    d_state: int = 16  # MED-004: Reduced from 64 to prevent OOM
    d_conv: int = 4
    expand: int = 2

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
        # Forward pass (causal: left to right)
        h_fwd = MambaBlock(
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            name='mamba_fwd'
        )(x, cond)

        # Backward pass (anti-causal: right to left)
        x_rev = jnp.flip(x, axis=1)
        # LOW-004 FIX: Flip conditioning for backward pass if seq-varying
        cond_bwd = cond
        if cond is not None:
            gamma, beta = cond
            # Flip along sequence dimension if it exists (ndim >= 2)
            gamma_bwd = jnp.flip(gamma, axis=1) if gamma.ndim >= 2 else gamma
            beta_bwd = jnp.flip(beta, axis=1) if beta.ndim >= 2 else beta
            cond_bwd = (gamma_bwd, beta_bwd)
        h_bwd = MambaBlock(
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            name='mamba_bwd'
        )(x_rev, cond_bwd)
        h_bwd = jnp.flip(h_bwd, axis=1)

        # Combine: concatenate and project
        h_combined = jnp.concatenate([h_fwd, h_bwd], axis=-1)
        h = nn.Dense(self.d_model, name='combine')(h_combined)

        return h


class BiMambaBlockWithNorm(nn.Module):
    """BiMamba with LayerNorm and residual connection."""
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
        # Pre-norm
        h = nn.LayerNorm()(x)

        # BiMamba
        h = BiMambaBlock(
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


class SelfAttention(nn.Module):
    """Standard self-attention for hybrid architecture."""
    d_model: int
    n_heads: int = 8
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x: Array, training: bool = False) -> Array:
        batch, seq, _ = x.shape
        head_dim = self.d_model // self.n_heads

        # QKV projection
        qkv = nn.Dense(self.d_model * 3)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        # Reshape for multi-head attention
        q = q.reshape(batch, seq, self.n_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, seq, self.n_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, seq, self.n_heads, head_dim).transpose(0, 2, 1, 3)

        # Attention
        scale = head_dim ** -0.5
        attn = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
        attn = nn.softmax(attn, axis=-1)

        if self.dropout > 0 and training:
            attn = nn.Dropout(rate=self.dropout, deterministic=not training)(attn)

        # Output
        out = jnp.einsum('bhqk,bhkd->bhqd', attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq, self.d_model)
        out = nn.Dense(self.d_model)(out)

        return out


class SelfAttentionWithNorm(nn.Module):
    """Self-attention with LayerNorm and residual."""
    d_model: int
    n_heads: int = 8
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x: Array, training: bool = False) -> Array:
        h = nn.LayerNorm()(x)
        h = SelfAttention(self.d_model, self.n_heads, self.dropout)(h, training)
        if self.dropout > 0 and training:
            h = nn.Dropout(rate=self.dropout, deterministic=not training)(h)
        return x + h
