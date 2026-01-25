"""Skeleton encoder for conditioning generation."""

import zlib
import jax
import jax.numpy as jnp
from jax import Array
import flax.linen as nn
from typing import Optional, Tuple

from .types import MarketSkeleton, REGIME_NAMES


class TrendEncoder(nn.Module):
    """Encode trend direction (discrete {-1, 0, +1})."""
    d_model: int

    @nn.compact
    def __call__(self, trend_direction: Array) -> Array:
        """
        Args:
            trend_direction: (n_timesteps, n_assets) or (n_assets,)
        Returns:
            (d_model,) global embedding or (n_timesteps, d_model) sequence
        """
        # LOW-005 FIX: Clip to valid range before embedding
        trend_clipped = jnp.clip(trend_direction, -1, 1)
        # Shift to {0, 1, 2} for embedding
        trend_shifted = trend_clipped + 1  # {-1,0,1} -> {0,1,2}

        # Embed each value
        embed = nn.Embed(num_embeddings=3, features=self.d_model)

        if trend_direction.ndim == 1:
            # Per-asset, aggregate
            embedded = embed(trend_shifted)  # (n_assets, d_model)
            return embedded.mean(axis=0)
        else:
            # Per-timestep, per-asset
            embedded = embed(trend_shifted)  # (n_timesteps, n_assets, d_model)
            return embedded.mean(axis=1)  # (n_timesteps, d_model)


class VolRegimeEncoder(nn.Module):
    """Encode volatility regime (continuous [0, 1])."""
    d_model: int

    @nn.compact
    def __call__(self, vol_regime: Array) -> Array:
        """
        Args:
            vol_regime: (n_timesteps,) continuous values in [0, 1]
        Returns:
            (n_timesteps, d_model) or (d_model,) if scalar
        """
        if vol_regime.ndim == 0 or (vol_regime.ndim == 1 and vol_regime.shape[0] == 1):
            # Scalar
            x = vol_regime.reshape(1)
            x = nn.Dense(self.d_model)(x)
            x = nn.silu(x)
            x = nn.Dense(self.d_model)(x)
            return x.squeeze(0)
        else:
            # Sequence
            x = vol_regime[:, None]  # (n_timesteps, 1)
            x = nn.Dense(self.d_model)(x)
            x = nn.silu(x)
            x = nn.Dense(self.d_model)(x)
            return x


class ClusterEncoder(nn.Module):
    """Encode correlation cluster assignments."""
    d_model: int
    n_clusters: int = 10

    @nn.compact
    def __call__(self, clusters: Array) -> Array:
        """
        Args:
            clusters: (n_assets,) cluster IDs
        Returns:
            (d_model,) global embedding
        """
        # Embed each cluster assignment
        embed = nn.Embed(num_embeddings=self.n_clusters, features=self.d_model)
        embedded = embed(clusters)  # (n_assets, d_model)

        # Aggregate
        return embedded.mean(axis=0)


class AlphaShapeEncoder(nn.Module):
    """Encode alpha opportunity shape using 1D conv."""
    d_model: int
    kernel_size: int = 7

    @nn.compact
    def __call__(self, alpha_shape: Array) -> Array:
        """
        Args:
            alpha_shape: (n_timesteps,) cumulative alpha in [0, 1]
        Returns:
            (n_timesteps, d_model) sequence embedding
        """
        x = alpha_shape[:, None]  # (n_timesteps, 1)

        # 1D conv to capture shape
        x = nn.Conv(
            features=self.d_model,
            kernel_size=(self.kernel_size,),
            padding='SAME'
        )(x)
        x = nn.silu(x)
        x = nn.Conv(
            features=self.d_model,
            kernel_size=(self.kernel_size,),
            padding='SAME'
        )(x)

        return x


class RegimeEncoder(nn.Module):
    """Encode regime sequence."""
    d_model: int
    n_regimes: int = 5

    @nn.compact
    def __call__(self, regime_sequence: Array) -> Array:
        """
        Args:
            regime_sequence: (n_segments,) or (n_timesteps,) regime indices
        Returns:
            (d_model,) if short sequence, else (n_timesteps, d_model)
        """
        embed = nn.Embed(num_embeddings=self.n_regimes, features=self.d_model)
        # Clip to valid range to prevent OOB embedding lookup
        regime_sequence = jnp.clip(regime_sequence, 0, self.n_regimes - 1)
        embedded = embed(regime_sequence)

        if regime_sequence.shape[0] <= 10:
            # Short sequence - return single embedding
            return embedded.mean(axis=0)
        else:
            # Per-timestep
            return embedded


class EventEncoder(nn.Module):
    """Encode sparse events."""
    d_model: int
    n_event_types: int = 10

    @nn.compact
    def __call__(self, events: list, n_timesteps: int) -> Array:
        """
        Args:
            events: List of Event objects
            n_timesteps: Sequence length
        Returns:
            (n_timesteps, d_model) sparse event embeddings
        """
        # Create sparse mask
        event_mask = jnp.zeros((n_timesteps, self.d_model))

        if not events:
            return event_mask

        # Event type embedding
        type_embed = nn.Embed(num_embeddings=self.n_event_types, features=self.d_model // 2)

        # Magnitude projection
        mag_proj = nn.Dense(self.d_model // 2)

        for event in events:
            if event.time < n_timesteps:
                # Deterministic hash (zlib.adler32 is stable across processes)
                type_idx = (zlib.adler32(event.event_type.encode()) & 0xffffffff) % self.n_event_types
                type_emb = type_embed(jnp.array(type_idx))

                # Magnitude embedding
                mag_emb = mag_proj(jnp.array([event.magnitude, event.direction]))

                # Combine
                event_emb = jnp.concatenate([type_emb, mag_emb])

                # Add to mask (using at[] for traced computation)
                event_mask = event_mask.at[event.time].set(event_emb)

        return event_mask


class SkeletonEncoder(nn.Module):
    """
    Full skeleton encoder combining all components.

    Outputs FiLM parameters (gamma, beta) for conditioning.
    """
    d_model: int
    n_clusters: int = 10
    n_regimes: int = 5

    @nn.compact
    def __call__(
        self,
        skeleton: MarketSkeleton,
        return_sequence: bool = False,
    ) -> Tuple[Array, Array, Optional[Array]]:
        """
        Encode skeleton into conditioning vectors.

        Args:
            skeleton: MarketSkeleton object
            return_sequence: If True, return per-timestep conditioning

        Returns:
            gamma: Scale parameter for FiLM (d_model,) or (n_timesteps, d_model)
            beta: Shift parameter for FiLM (d_model,) or (n_timesteps, d_model)
            h0: Initial state for Mamba (d_model,)
        """
        embeddings = []

        # Encode each component if present
        if skeleton.trend_direction is not None:
            trend_emb = TrendEncoder(self.d_model)(skeleton.trend_direction)
            if trend_emb.ndim == 1:
                embeddings.append(trend_emb)
            # For sequence, handle separately below

        if skeleton.vol_regime is not None:
            vol_emb = VolRegimeEncoder(self.d_model)(skeleton.vol_regime)
            if vol_emb.ndim == 1:
                embeddings.append(vol_emb)

        if skeleton.correlation_clusters is not None:
            cluster_emb = ClusterEncoder(self.d_model, self.n_clusters)(
                skeleton.correlation_clusters.astype(jnp.int32)
            )
            embeddings.append(cluster_emb)

        if skeleton.regime_sequence is not None:
            regime_emb = RegimeEncoder(self.d_model, self.n_regimes)(
                skeleton.regime_sequence.astype(jnp.int32)
            )
            if regime_emb.ndim == 1:
                embeddings.append(regime_emb)

        # Combine global embeddings
        if embeddings:
            combined = jnp.stack(embeddings, axis=0).mean(axis=0)
        else:
            combined = jnp.zeros(self.d_model)

        # Generate FiLM parameters
        film_mlp = nn.Sequential([
            nn.Dense(self.d_model * 2),
            nn.silu,
            nn.Dense(self.d_model * 2),
        ])

        film_params = film_mlp(combined)
        gamma, beta = jnp.split(film_params, 2, axis=-1)

        # Initial state for Mamba
        h0 = nn.Dense(self.d_model)(combined)

        return gamma, beta, h0


class SkeletonConditioner(nn.Module):
    """
    Complete conditioning module for skeleton + timestep.

    Used in generation models to combine all conditioning signals.
    """
    d_model: int
    n_clusters: int = 10
    n_regimes: int = 5

    def sinusoidal_embedding(self, t: Array, dim: int) -> Array:
        """Sinusoidal positional embedding for timestep."""
        half_dim = dim // 2
        emb = jnp.log(10000) / jnp.maximum(half_dim - 1, 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return emb

    @nn.compact
    def __call__(
        self,
        t: Array,
        skeleton: Optional[MarketSkeleton] = None,
    ) -> Tuple[Array, Array, Array]:
        """
        Combine timestep and skeleton conditioning.

        Args:
            t: Timestep (batch,) in [0, 1]
            skeleton: Optional MarketSkeleton

        Returns:
            gamma: FiLM scale (batch, d_model)
            beta: FiLM shift (batch, d_model)
            h0: Initial state (batch, d_model)
        """
        batch_size = t.shape[0]

        # Time embedding
        t_emb = self.sinusoidal_embedding(t, self.d_model)
        t_emb = nn.Dense(self.d_model)(t_emb)
        t_emb = nn.silu(t_emb)
        t_emb = nn.Dense(self.d_model)(t_emb)

        # Skeleton embedding
        if skeleton is not None and skeleton.has_conditioning():
            skel_encoder = SkeletonEncoder(self.d_model, self.n_clusters, self.n_regimes)
            gamma_s, beta_s, h0_s = skel_encoder(skeleton)

            # Broadcast to batch
            gamma_s = jnp.broadcast_to(gamma_s, (batch_size, self.d_model))
            beta_s = jnp.broadcast_to(beta_s, (batch_size, self.d_model))
            h0_s = jnp.broadcast_to(h0_s, (batch_size, self.d_model))
        else:
            gamma_s = jnp.zeros((batch_size, self.d_model))
            beta_s = jnp.zeros((batch_size, self.d_model))
            h0_s = jnp.zeros((batch_size, self.d_model))

        # Combine time and skeleton
        combined = t_emb + gamma_s  # Additive combination

        # Final FiLM parameters
        film_out = nn.Dense(self.d_model * 2)(combined)
        gamma, beta = jnp.split(film_out, 2, axis=-1)

        # Initial state
        h0 = nn.Dense(self.d_model)(combined) + h0_s

        return gamma, beta, h0
