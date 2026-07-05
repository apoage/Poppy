"""
Neural Market Corrector (TASK_009)

Neural network component that learns corrections to physics-based dynamics.
Uses cross-asset transformer to capture complex inter-asset dependencies.

Key Components:
- CrossAssetAttention: Multi-head attention treating assets as tokens
- CrossAssetTransformerBlock: Full transformer block with residual connections
- NeuralMarketCorrector: Main corrector with transformer architecture
- SimpleNeuralCorrector: Lightweight MLP-based alternative

All operations in LOG-SPACE for numerical stability.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, Optional


class CrossAssetAttention(nn.Module):
    """
    Multi-head attention treating assets as tokens.

    Instead of treating the batch as the sequence, we treat assets as tokens.
    This allows the model to learn cross-asset dependencies dynamically.

    Input: (batch, n_assets, embed_dim)
    - Each of n_assets is a token
    - Attention captures which assets influence each other
    """
    num_heads: int = 4
    head_dim: int = 32
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """
        Apply multi-head self-attention across assets.

        Args:
            x: (batch, n_assets, embed_dim) - assets are tokens
            deterministic: Whether to apply dropout

        Returns:
            out: (batch, n_assets, embed_dim) - attention output
        """
        batch, n_assets, embed_dim = x.shape
        qkv_dim = self.num_heads * self.head_dim

        # Project to Q, K, V
        q = nn.Dense(qkv_dim, name='query')(x)  # (batch, n_assets, qkv_dim)
        k = nn.Dense(qkv_dim, name='key')(x)
        v = nn.Dense(qkv_dim, name='value')(x)

        # Reshape for multi-head attention
        # (batch, n_assets, num_heads, head_dim) -> (batch, num_heads, n_assets, head_dim)
        q = q.reshape(batch, n_assets, self.num_heads, self.head_dim)
        k = k.reshape(batch, n_assets, self.num_heads, self.head_dim)
        v = v.reshape(batch, n_assets, self.num_heads, self.head_dim)

        q = jnp.transpose(q, (0, 2, 1, 3))  # (batch, num_heads, n_assets, head_dim)
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # Scaled dot-product attention
        scale = jnp.sqrt(self.head_dim).astype(x.dtype)
        attn_weights = jnp.einsum('bhqd,bhkd->bhqk', q, k) / scale
        attn_weights = nn.softmax(attn_weights, axis=-1)

        # Apply dropout
        if not deterministic and self.dropout_rate > 0:
            attn_weights = nn.Dropout(rate=self.dropout_rate)(
                attn_weights, deterministic=deterministic
            )

        # Apply attention to values
        attn_out = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, v)

        # Reshape back: (batch, num_heads, n_assets, head_dim) -> (batch, n_assets, qkv_dim)
        attn_out = jnp.transpose(attn_out, (0, 2, 1, 3))
        attn_out = attn_out.reshape(batch, n_assets, qkv_dim)

        # Output projection
        out = nn.Dense(embed_dim, name='output')(attn_out)

        return out


class CrossAssetTransformerBlock(nn.Module):
    """
    Full transformer block with cross-asset attention.

    Structure:
    - LayerNorm -> CrossAssetAttention -> Residual
    - LayerNorm -> MLP -> Residual
    """
    num_heads: int = 4
    head_dim: int = 32
    mlp_dim: int = 256
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """
        Apply transformer block.

        Args:
            x: (batch, n_assets, embed_dim)
            deterministic: Whether to apply dropout

        Returns:
            out: (batch, n_assets, embed_dim)
        """
        embed_dim = x.shape[-1]

        # Pre-norm attention
        y = nn.LayerNorm()(x)
        y = CrossAssetAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout_rate=self.dropout_rate,
        )(y, deterministic=deterministic)

        # Residual connection
        x = x + y

        # Pre-norm MLP
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.mlp_dim)(y)
        y = nn.gelu(y)
        if not deterministic and self.dropout_rate > 0:
            y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)
        y = nn.Dense(embed_dim)(y)
        if not deterministic and self.dropout_rate > 0:
            y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)

        # Residual connection
        out = x + y

        return out


class NeuralMarketCorrector(nn.Module):
    """
    Neural corrector using cross-asset transformer.

    Computes G_neural: learned corrections to physics dynamics.

    Architecture:
    1. Input embedding: (prices, returns, moving_avg, features) -> embed_dim
    2. Cross-asset transformer blocks (assets as tokens)
    3. Output projection -> G_neural per asset

    The transformer allows learning complex cross-asset dependencies
    that may not be captured by the physics correlation model.
    """
    n_assets: int = 64
    n_features: int = 10
    embed_dim: int = 128
    num_transformer_blocks: int = 2
    num_heads: int = 4
    head_dim: int = 32
    mlp_dim: int = 256
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(
        self,
        log_prices: jnp.ndarray,
        log_returns: jnp.ndarray,
        moving_avg: jnp.ndarray,
        features: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """
        Compute neural correction term.

        Args:
            log_prices: (batch, n_assets) current log prices
            log_returns: (batch, n_assets) recent log returns
            moving_avg: (batch, n_assets) moving average of log prices
            features: (batch, n_features) external features (optional)
            deterministic: Whether to apply dropout

        Returns:
            G_neural: (batch, n_assets) neural correction tendency
        """
        batch = log_prices.shape[0]

        # =====================================================================
        # Input Embedding
        # =====================================================================

        # Per-asset features: stack along last dim
        # Shape: (batch, n_assets, 3) for prices, returns, deviation
        deviation = log_prices - moving_avg
        per_asset = jnp.stack([log_prices, log_returns, deviation], axis=-1)

        # Project per-asset features to embed_dim
        # (batch, n_assets, 3) -> (batch, n_assets, embed_dim)
        x = nn.Dense(self.embed_dim, name='asset_embed')(per_asset)

        # Add global features if provided
        if features is not None:
            # Project features and broadcast to all assets
            # (batch, n_features) -> (batch, embed_dim)
            global_embed = nn.Dense(self.embed_dim, name='global_embed')(features)
            # Add to all asset embeddings
            x = x + global_embed[:, None, :]

        # Add learnable positional embeddings for assets
        # This allows the model to distinguish asset identities
        pos_embed = self.param(
            'asset_positions',
            nn.initializers.normal(0.02),
            (self.n_assets, self.embed_dim)
        )
        x = x + pos_embed[None, :, :]

        # =====================================================================
        # Transformer Blocks
        # =====================================================================

        for i in range(self.num_transformer_blocks):
            x = CrossAssetTransformerBlock(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                name=f'transformer_block_{i}',
            )(x, deterministic=deterministic)

        # =====================================================================
        # Output Projection
        # =====================================================================

        # Final layer norm
        x = nn.LayerNorm()(x)

        # Project to single value per asset (the correction)
        # (batch, n_assets, embed_dim) -> (batch, n_assets, 1) -> (batch, n_assets)
        G_neural = nn.Dense(1, name='output_proj')(x)
        G_neural = G_neural.squeeze(-1)

        # Scale output to reasonable range for corrections
        # Corrections should be small relative to physics
        G_neural = G_neural * 0.1

        return G_neural


class SimpleNeuralCorrector(nn.Module):
    """
    Lightweight MLP-based corrector (baseline/fast alternative).

    Uses simple MLPs without cross-asset attention.
    Faster but may miss complex inter-asset dependencies.

    Architecture:
    1. Per-asset MLP: local features -> hidden -> G_neural
    2. Optional: global feature integration
    """
    n_assets: int = 64
    n_features: int = 10
    hidden_dim: int = 128
    num_layers: int = 2

    @nn.compact
    def __call__(
        self,
        log_prices: jnp.ndarray,
        log_returns: jnp.ndarray,
        moving_avg: jnp.ndarray,
        features: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """
        Compute neural correction term using MLP.

        Args:
            log_prices: (batch, n_assets) current log prices
            log_returns: (batch, n_assets) recent log returns
            moving_avg: (batch, n_assets) moving average of log prices
            features: (batch, n_features) external features (optional)
            deterministic: Whether to apply dropout (unused in simple version)

        Returns:
            G_neural: (batch, n_assets) neural correction tendency
        """
        batch = log_prices.shape[0]

        # Stack local features
        deviation = log_prices - moving_avg
        local_features = jnp.stack([log_prices, log_returns, deviation], axis=-1)
        # (batch, n_assets, 3)

        # If global features, tile and concatenate
        if features is not None:
            # (batch, n_features) -> (batch, n_assets, n_features)
            global_tiled = jnp.tile(features[:, None, :], (1, self.n_assets, 1))
            x = jnp.concatenate([local_features, global_tiled], axis=-1)
            # (batch, n_assets, 3 + n_features)
        else:
            x = local_features

        # MLP layers
        for i in range(self.num_layers):
            x = nn.Dense(self.hidden_dim, name=f'mlp_{i}')(x)
            x = nn.gelu(x)

        # Output projection
        G_neural = nn.Dense(1, name='output')(x)
        G_neural = G_neural.squeeze(-1)

        # Scale output
        G_neural = G_neural * 0.1

        return G_neural


# Convenience function for creating corrector
def create_neural_corrector(
    n_assets: int = 64,
    n_features: int = 10,
    use_transformer: bool = True,
    embed_dim: int = 128,
    num_transformer_blocks: int = 2,
    rng: Optional[jnp.ndarray] = None,
) -> Tuple[nn.Module, dict]:
    """
    Create and initialize neural corrector.

    Args:
        n_assets: Number of tradeable assets
        n_features: Number of external features
        use_transformer: Whether to use transformer (True) or MLP (False)
        embed_dim: Embedding dimension for transformer
        num_transformer_blocks: Number of transformer blocks
        rng: Random key for initialization

    Returns:
        model: Initialized model
        params: Initial parameters
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    if use_transformer:
        model = NeuralMarketCorrector(
            n_assets=n_assets,
            n_features=n_features,
            embed_dim=embed_dim,
            num_transformer_blocks=num_transformer_blocks,
        )
    else:
        model = SimpleNeuralCorrector(
            n_assets=n_assets,
            n_features=n_features,
        )

    # Initialize with dummy inputs
    dummy_log_prices = jnp.zeros((1, n_assets))
    dummy_log_returns = jnp.zeros((1, n_assets))
    dummy_moving_avg = jnp.zeros((1, n_assets))
    dummy_features = jnp.zeros((1, n_features))

    params = model.init(
        rng,
        dummy_log_prices,
        dummy_log_returns,
        dummy_moving_avg,
        dummy_features,
    )

    return model, params
