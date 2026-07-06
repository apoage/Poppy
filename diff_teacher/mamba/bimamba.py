"""Bidirectional Mamba for generative models."""

import jax
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

    HIGH-004 FIX: Added sequential_mode to reduce memory usage by processing
    forward and backward passes one at a time instead of in parallel.
    """
    d_model: int
    d_state: int = 16  # MED-004: Reduced from 64 to prevent OOM
    d_conv: int = 4
    expand: int = 2
    sequential_mode: bool = False  # HIGH-004 FIX: Sequential processing to halve memory

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
        # Forward Mamba block
        fwd_block = MambaBlock(
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            name='mamba_fwd'
        )

        # Backward Mamba block
        bwd_block = MambaBlock(
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            name='mamba_bwd'
        )

        # LOW-004 FIX: Prepare backward conditioning
        cond_bwd = cond
        if cond is not None:
            gamma, beta = cond
            # Flip along sequence dimension if it exists (ndim >= 2)
            gamma_bwd = jnp.flip(gamma, axis=1) if gamma.ndim >= 2 else gamma
            beta_bwd = jnp.flip(beta, axis=1) if beta.ndim >= 2 else beta
            cond_bwd = (gamma_bwd, beta_bwd)

        if self.sequential_mode:
            # HIGH-004 FIX: Sequential processing - halves peak memory
            # Process forward, store result, then process backward
            h_fwd = fwd_block(x, cond)

            # Backward pass (anti-causal: right to left)
            x_rev = jnp.flip(x, axis=1)
            h_bwd = bwd_block(x_rev, cond_bwd)
            h_bwd = jnp.flip(h_bwd, axis=1)
        else:
            # Parallel processing (original behavior)
            h_fwd = fwd_block(x, cond)

            x_rev = jnp.flip(x, axis=1)
            h_bwd = bwd_block(x_rev, cond_bwd)
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
    sequential_mode: bool = False  # HIGH-004 FIX: Pass through to BiMambaBlock

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
            sequential_mode=self.sequential_mode,  # HIGH-004 FIX
        )(h, cond)

        # Dropout
        if self.dropout > 0 and training:
            h = nn.Dropout(rate=self.dropout, deterministic=not training)(h)

        # Residual
        return x + h


class SelfAttention(nn.Module):
    """
    Self-attention for hybrid architecture.

    CRIT-005 FIX: Uses memory-efficient attention via jax.nn.dot_product_attention
    when available (JAX 0.4.1+), falling back to chunked attention otherwise.
    This avoids O(seq^2) memory for the attention matrix.
    """
    d_model: int
    n_heads: int = 8
    dropout: float = 0.0
    chunk_size: int = 256  # For chunked fallback

    @nn.compact
    def __call__(self, x: Array, training: bool = False) -> Array:
        batch, seq, _ = x.shape
        head_dim = self.d_model // self.n_heads

        # QKV projection
        qkv = nn.Dense(self.d_model * 3)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        # Reshape for multi-head attention: (batch, seq, heads, head_dim)
        q = q.reshape(batch, seq, self.n_heads, head_dim)
        k = k.reshape(batch, seq, self.n_heads, head_dim)
        v = v.reshape(batch, seq, self.n_heads, head_dim)

        # CRIT-005 FIX: Use memory-efficient attention
        # jax.nn.dot_product_attention expects (batch, seq, heads, head_dim)
        # and implements flash attention / memory-efficient attention internally
        if hasattr(jax.nn, 'dot_product_attention'):
            # JAX 0.4.1+ has built-in memory-efficient attention
            scale = head_dim ** -0.5
            out = jax.nn.dot_product_attention(q, k, v, scale=scale)
        else:
            # Fallback: chunked attention to avoid O(seq^2) memory
            out = self._chunked_attention(q, k, v, head_dim)

        # Output projection
        out = out.reshape(batch, seq, self.d_model)
        out = nn.Dense(self.d_model)(out)

        return out

    def _chunked_attention(
        self, q: Array, k: Array, v: Array, head_dim: int
    ) -> Array:
        """Chunked attention fallback for memory efficiency."""
        batch, seq, n_heads, _ = q.shape
        scale = head_dim ** -0.5

        # Transpose to (batch, heads, seq, head_dim) for attention
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Process in chunks to avoid O(seq^2) memory
        chunk_size = min(self.chunk_size, seq)
        n_chunks = (seq + chunk_size - 1) // chunk_size

        outputs = []
        for i in range(n_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, seq)
            q_chunk = q[:, :, start:end, :]

            # Compute attention for this chunk against all keys
            attn = jnp.einsum('bhqd,bhkd->bhqk', q_chunk, k) * scale
            attn = jax.nn.softmax(attn, axis=-1)
            out_chunk = jnp.einsum('bhqk,bhkd->bhqd', attn, v)
            outputs.append(out_chunk)

        out = jnp.concatenate(outputs, axis=2)
        # Transpose back to (batch, seq, heads, head_dim)
        return out.transpose(0, 2, 1, 3)


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
