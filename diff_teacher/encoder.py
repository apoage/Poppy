"""
Transformer Encoder for Market Data.

Encodes (window_size, n_assets, features) -> (latent_dim,) latent vector.
Uses cross-asset attention to capture correlations.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple
from functools import partial


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for time steps."""

    dim: int
    max_len: int = 2048

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Add positional encoding to input.

        Args:
            x: (batch, seq_len, dim)

        Returns:
            (batch, seq_len, dim) with positional encoding added
        """
        seq_len = x.shape[1]

        # Create position indices
        position = jnp.arange(seq_len)[:, None]  # (seq_len, 1)
        div_term = jnp.exp(
            jnp.arange(0, self.dim, 2) * (-jnp.log(10000.0) / self.dim)
        )  # (dim/2,)

        # Compute sin/cos
        pe = jnp.zeros((seq_len, self.dim))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))

        return x + pe[None, :, :]  # Broadcast over batch


class CrossAssetAttention(nn.Module):
    """
    Attention over the asset dimension.

    Captures correlations between different assets at each timestep.
    """

    num_heads: int
    dim: int
    dropout: float = 0.0

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, training: bool = False
    ) -> jnp.ndarray:
        """
        Apply cross-asset attention.

        Args:
            x: (batch, time, n_assets, dim)
            training: Whether in training mode

        Returns:
            (batch, time, n_assets, dim)
        """
        batch, time, n_assets, dim = x.shape

        # Reshape to (batch * time, n_assets, dim) for attention
        x_flat = x.reshape(batch * time, n_assets, dim)

        # Multi-head attention over asset dimension
        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=dim,
            dropout_rate=self.dropout,
            deterministic=not training,
        )
        out = attn(x_flat, x_flat)

        # Reshape back
        out = out.reshape(batch, time, n_assets, dim)
        return out


class TemporalAttention(nn.Module):
    """
    Attention over the time dimension.

    Captures temporal patterns within each asset's history.
    """

    num_heads: int
    dim: int
    dropout: float = 0.0

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, training: bool = False
    ) -> jnp.ndarray:
        """
        Apply temporal attention.

        Args:
            x: (batch, time, n_assets, dim)
            training: Whether in training mode

        Returns:
            (batch, time, n_assets, dim)
        """
        batch, time, n_assets, dim = x.shape

        # Reshape to (batch * n_assets, time, dim) for attention
        x_flat = x.transpose(0, 2, 1, 3).reshape(batch * n_assets, time, dim)

        # Multi-head attention over time dimension
        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=dim,
            dropout_rate=self.dropout,
            deterministic=not training,
        )
        out = attn(x_flat, x_flat)

        # Reshape back
        out = out.reshape(batch, n_assets, time, dim).transpose(0, 2, 1, 3)
        return out


class EncoderBlock(nn.Module):
    """
    Single encoder block with cross-asset and temporal attention.

    Architecture:
        1. Cross-asset attention (correlations)
        2. Temporal attention (patterns)
        3. Feed-forward network
    """

    num_heads: int
    dim: int
    ff_dim: int
    dropout: float = 0.1

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, training: bool = False
    ) -> jnp.ndarray:
        """
        Process input through encoder block.

        Args:
            x: (batch, time, n_assets, dim)
            training: Whether in training mode

        Returns:
            (batch, time, n_assets, dim)
        """
        # Cross-asset attention with residual
        x_norm = nn.LayerNorm()(x)
        x_attn = CrossAssetAttention(
            num_heads=self.num_heads, dim=self.dim, dropout=self.dropout
        )(x_norm, training=training)
        x = x + nn.Dropout(rate=self.dropout, deterministic=not training)(x_attn)

        # Temporal attention with residual
        x_norm = nn.LayerNorm()(x)
        x_attn = TemporalAttention(
            num_heads=self.num_heads, dim=self.dim, dropout=self.dropout
        )(x_norm, training=training)
        x = x + nn.Dropout(rate=self.dropout, deterministic=not training)(x_attn)

        # Feed-forward with residual
        x_norm = nn.LayerNorm()(x)
        x_ff = nn.Dense(self.ff_dim)(x_norm)
        x_ff = nn.gelu(x_ff)
        x_ff = nn.Dropout(rate=self.dropout, deterministic=not training)(x_ff)
        x_ff = nn.Dense(self.dim)(x_ff)
        x = x + nn.Dropout(rate=self.dropout, deterministic=not training)(x_ff)

        return x


class MarketEncoder(nn.Module):
    """
    Transformer encoder for market data.

    Encodes (batch, window_size, n_assets, features) -> (batch, latent_dim)

    Architecture:
        1. Linear projection of features
        2. Positional encoding
        3. N transformer blocks (cross-asset + temporal attention)
        4. Global pooling
        5. Projection to latent space
    """

    latent_dim: int = 256
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    ff_dim: int = 512
    dropout: float = 0.1
    n_features: int = 5  # OHLCV

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool = False,
    ) -> jnp.ndarray:
        """
        Encode market window to latent vector.

        Args:
            x: (batch, window_size, n_assets, features) normalized market data
            training: Whether in training mode

        Returns:
            (batch, latent_dim) latent representation
        """
        batch, time, n_assets, features = x.shape

        # Project features to hidden dimension
        x = nn.Dense(self.hidden_dim, name="input_proj")(x)  # (B, T, A, H)

        # Add positional encoding over time
        # Reshape to (B*A, T, H) for temporal encoding
        x = x.transpose(0, 2, 1, 3).reshape(batch * n_assets, time, self.hidden_dim)
        x = SinusoidalPositionalEncoding(dim=self.hidden_dim)(x)
        x = x.reshape(batch, n_assets, time, self.hidden_dim).transpose(0, 2, 1, 3)
        # Back to (B, T, A, H)

        # Encoder blocks
        for i in range(self.num_layers):
            x = EncoderBlock(
                num_heads=self.num_heads,
                dim=self.hidden_dim,
                ff_dim=self.ff_dim,
                dropout=self.dropout,
                name=f"encoder_block_{i}",
            )(x, training=training)

        # Final layer norm
        x = nn.LayerNorm(name="final_norm")(x)

        # Global pooling: mean over time and assets
        x = jnp.mean(x, axis=(1, 2))  # (B, H)

        # Project to latent space
        x = nn.Dense(self.latent_dim, name="latent_proj")(x)

        return x

    def encode_with_intermediates(
        self, x: jnp.ndarray, training: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Encode and return both latent and pre-pooling features.

        Useful for decoder or more complex architectures.

        Returns:
            latent: (batch, latent_dim)
            features: (batch, time, n_assets, hidden_dim)
        """
        batch, time, n_assets, features = x.shape

        # Project features
        h = nn.Dense(self.hidden_dim, name="input_proj")(x)

        # Positional encoding
        h = h.transpose(0, 2, 1, 3).reshape(batch * n_assets, time, self.hidden_dim)
        h = SinusoidalPositionalEncoding(dim=self.hidden_dim)(h)
        h = h.reshape(batch, n_assets, time, self.hidden_dim).transpose(0, 2, 1, 3)

        # Encoder blocks
        for i in range(self.num_layers):
            h = EncoderBlock(
                num_heads=self.num_heads,
                dim=self.hidden_dim,
                ff_dim=self.ff_dim,
                dropout=self.dropout,
                name=f"encoder_block_{i}",
            )(h, training=training)

        h = nn.LayerNorm(name="final_norm")(h)
        features = h

        # Pool and project
        pooled = jnp.mean(h, axis=(1, 2))
        latent = nn.Dense(self.latent_dim, name="latent_proj")(pooled)

        return latent, features


class MarketDecoder(nn.Module):
    """
    Decoder to reconstruct market data from latent.

    Used for VAE-style training or validating reconstruction quality.

    Decodes (batch, latent_dim) -> (batch, window_size, n_assets, features)
    """

    window_size: int = 400
    n_assets: int = 64
    n_features: int = 5
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    ff_dim: int = 512
    dropout: float = 0.1

    @nn.compact
    def __call__(
        self,
        z: jnp.ndarray,
        training: bool = False,
    ) -> jnp.ndarray:
        """
        Decode latent vector to market window.

        Args:
            z: (batch, latent_dim) latent representation
            training: Whether in training mode

        Returns:
            (batch, window_size, n_assets, features) reconstructed market data
        """
        batch = z.shape[0]

        # Project latent to hidden space
        h = nn.Dense(self.hidden_dim, name="latent_proj")(z)  # (B, H)

        # Expand to (B, T, A, H) using learned queries
        time_embed = self.param(
            "time_queries",
            nn.initializers.normal(0.02),
            (1, self.window_size, 1, self.hidden_dim),
        )
        asset_embed = self.param(
            "asset_queries",
            nn.initializers.normal(0.02),
            (1, 1, self.n_assets, self.hidden_dim),
        )

        # Broadcast and add latent
        h = h[:, None, None, :]  # (B, 1, 1, H)
        h = h + time_embed + asset_embed  # (B, T, A, H)

        # Decoder blocks (same architecture as encoder)
        for i in range(self.num_layers):
            h = EncoderBlock(
                num_heads=self.num_heads,
                dim=self.hidden_dim,
                ff_dim=self.ff_dim,
                dropout=self.dropout,
                name=f"decoder_block_{i}",
            )(h, training=training)

        # Final layer norm
        h = nn.LayerNorm(name="final_norm")(h)

        # Project to output features
        out = nn.Dense(self.n_features, name="output_proj")(h)

        return out


class MarketVAE(nn.Module):
    """
    Variational Autoencoder for market data.

    Combines encoder and decoder with KL divergence loss.
    """

    latent_dim: int = 256
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    ff_dim: int = 512
    dropout: float = 0.1
    window_size: int = 400
    n_assets: int = 64
    n_features: int = 5

    def setup(self):
        self.encoder = MarketEncoder(
            latent_dim=self.latent_dim * 2,  # mu and log_var
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            ff_dim=self.ff_dim,
            dropout=self.dropout,
            n_features=self.n_features,
        )
        self.decoder = MarketDecoder(
            window_size=self.window_size,
            n_assets=self.n_assets,
            n_features=self.n_features,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            ff_dim=self.ff_dim,
            dropout=self.dropout,
        )

    def encode(
        self, x: jnp.ndarray, training: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Encode to mean and log variance.

        Returns:
            mu: (batch, latent_dim)
            log_var: (batch, latent_dim)
        """
        h = self.encoder(x, training=training)
        mu, log_var = jnp.split(h, 2, axis=-1)
        return mu, log_var

    def reparameterize(
        self, mu: jnp.ndarray, log_var: jnp.ndarray, key: jax.Array
    ) -> jnp.ndarray:
        """Sample from latent distribution using reparameterization trick."""
        std = jnp.exp(0.5 * log_var)
        eps = jax.random.normal(key, shape=mu.shape)
        return mu + eps * std

    def decode(
        self, z: jnp.ndarray, training: bool = False
    ) -> jnp.ndarray:
        """Decode latent to reconstructed market data."""
        return self.decoder(z, training=training)

    def __call__(
        self,
        x: jnp.ndarray,
        key: jax.Array,
        training: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Forward pass through VAE.

        Args:
            x: (batch, window_size, n_assets, features) input
            key: JAX random key
            training: Whether in training mode

        Returns:
            recon: (batch, window_size, n_assets, features) reconstruction
            mu: (batch, latent_dim) mean
            log_var: (batch, latent_dim) log variance
        """
        mu, log_var = self.encode(x, training=training)
        z = self.reparameterize(mu, log_var, key)
        recon = self.decode(z, training=training)
        return recon, mu, log_var


def vae_loss(
    recon: jnp.ndarray,
    target: jnp.ndarray,
    mu: jnp.ndarray,
    log_var: jnp.ndarray,
    kl_weight: float = 0.001,
) -> Tuple[jnp.ndarray, dict]:
    """
    VAE loss: reconstruction + KL divergence.

    Args:
        recon: Reconstructed output
        target: Original input
        mu: Latent mean
        log_var: Latent log variance
        kl_weight: Weight for KL term (beta-VAE)

    Returns:
        loss: Total loss
        metrics: Dictionary with individual loss components
    """
    # Reconstruction loss (MSE)
    recon_loss = jnp.mean((recon - target) ** 2)

    # KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    kl_loss = -0.5 * jnp.mean(1 + log_var - mu**2 - jnp.exp(log_var))

    total_loss = recon_loss + kl_weight * kl_loss

    metrics = {
        "loss": total_loss,
        "recon_loss": recon_loss,
        "kl_loss": kl_loss,
    }

    return total_loss, metrics


# =============================================================================
# Utility functions
# =============================================================================
def create_encoder(config) -> MarketEncoder:
    """Create encoder from config."""
    return MarketEncoder(
        latent_dim=config.latent_dim,
        hidden_dim=config.encoder_ff_dim // 4,  # Reasonable hidden dim
        num_heads=config.encoder_heads,
        num_layers=config.encoder_layers,
        ff_dim=config.encoder_ff_dim,
        dropout=config.encoder_dropout,
    )


def create_vae(config) -> MarketVAE:
    """Create VAE from config."""
    return MarketVAE(
        latent_dim=config.latent_dim,
        hidden_dim=config.encoder_ff_dim // 4,
        num_heads=config.encoder_heads,
        num_layers=config.encoder_layers,
        ff_dim=config.encoder_ff_dim,
        dropout=config.encoder_dropout,
        window_size=config.window_size,
        n_assets=config.n_assets,
        n_features=len(config.features),
    )


# =============================================================================
# Test
# =============================================================================
if __name__ == "__main__":
    import jax

    # Test encoder
    key = jax.random.PRNGKey(42)
    encoder = MarketEncoder(latent_dim=256, hidden_dim=128, num_heads=4, num_layers=2)

    # Dummy input: (batch, time, assets, features)
    x = jax.random.normal(key, (2, 100, 64, 5))  # Smaller for test

    params = encoder.init(key, x, training=False)
    latent = encoder.apply(params, x, training=False)

    print(f"Input shape: {x.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Params: {jax.tree_util.tree_map(lambda x: x.shape, params)['params'].keys()}")

    # Count parameters
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Total parameters: {n_params:,} ({n_params / 1e6:.2f}M)")

    # Test VAE
    print("\n--- VAE Test ---")
    vae = MarketVAE(
        latent_dim=256, hidden_dim=128, num_heads=4, num_layers=2,
        window_size=100, n_assets=64, n_features=5
    )

    key, subkey = jax.random.split(key)
    vae_params = vae.init(subkey, x, key, training=False)
    recon, mu, log_var = vae.apply(vae_params, x, key, training=False)

    print(f"Reconstruction shape: {recon.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Log var shape: {log_var.shape}")

    # Test loss
    loss, metrics = vae_loss(recon, x, mu, log_var)
    print(f"Loss: {loss:.4f}")
    print(f"Metrics: {metrics}")

    # Count VAE parameters
    n_vae_params = sum(p.size for p in jax.tree_util.tree_leaves(vae_params))
    print(f"VAE parameters: {n_vae_params:,} ({n_vae_params / 1e6:.2f}M)")
