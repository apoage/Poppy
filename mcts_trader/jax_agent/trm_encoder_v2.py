import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Optional

class AssetTransformerEncoder(nn.Module):
    """
    Asset-Identity Preserving Encoder (V2).
    
    Instead of 2D CNNs that blend assets, this model:
    1. Extracts time-series features PER ASSET using a shared MLP or 1D Conv.
    2. Uses Cross-Asset Attention to model dependencies without losing identity.
    3. Projects to a global state while maintaining asset-level information.
    
    Input shape: (batch, channels, window, n_assets)
    - Channels: 27 (OHLCV, technicals, etc.)
    - Window: 60 (time steps)
    - n_assets: 64
    """
    state_dim: int = 64
    n_assets: int = 64
    embed_dim: int = 32  # Per-asset embedding dimension
    num_heads: int = 4
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True,
                 return_per_asset: bool = False):
        """
        Args:
            x: (batch, channels, window, n_assets)
            return_per_asset: If True, returns (global_state, per_asset_feats)
                where per_asset_feats has shape (batch, n_assets, embed_dim).
                TASK_729a: per-asset routing for direction + per-asset policy heads.
        """
        batch, channels, window, n_assets = x.shape
        
        # 1. Transpose to (batch, n_assets, window, channels)
        # Treating each asset as an independent time-series "token"
        x = jnp.transpose(x, (0, 3, 2, 1))
        
        # 2. Per-Asset Feature Extraction (Shared across assets)
        # Flatten window and channels: (batch, n_assets, window * channels)
        x_flat = x.reshape(batch, n_assets, -1)
        
        # Extract features per asset using a shared Dense layer
        # This preserves the identity of which asset is which because we stay in (batch, n_assets, ...)
        x_assets = nn.Dense(
            features=self.embed_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="asset_embedding"
        )(x_flat)
        x_assets = nn.gelu(x_assets)
        
        # Add per-asset position embeddings
        pos_embed = self.param(
            'asset_identity_embed',
            nn.initializers.normal(stddev=0.02),
            (1, self.n_assets, self.embed_dim)
        )
        
        # Slicing if n_assets in input is smaller than n_assets in config (test case)
        # Or just use the first N if they match.
        x_assets = x_assets + pos_embed[:, :n_assets, :]
        
        # 3. Cross-Asset Attention
        # MODEL: How do assets influence each other?
        # Use LayerNorm for stability
        x_norm = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype)(x_assets)
        
        # Multi-head attention across the n_assets dimension
        attn_out = nn.SelfAttention(
            num_heads=self.num_heads,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="cross_asset_attention"
        )(x_norm)
        
        # Residual connection
        x_assets = x_assets + attn_out
        
        # 4. Global Projection to state_dim
        # Combine all asset information into the MuZero latent state
        # We use a final LayerNorm + Dense
        x_final = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype)(x_assets)
        # x_final: (batch, n_assets, embed_dim) — TASK_729a: per-asset feats exposed here

        # Flatten all assets into one global state
        x_global = x_final.reshape(batch, -1)

        state = nn.Dense(
            features=self.state_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="global_state_projection"
        )(x_global)

        if return_per_asset:
            return state, x_final
        return state
