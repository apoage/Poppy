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
    """
    Encode sparse events.

    MED-003 FIX: Refactored to use padded arrays instead of Python for loop.
    Events are now passed as padded arrays for JIT compatibility.
    """
    d_model: int
    n_event_types: int = 10
    max_events: int = 32  # Maximum events to support

    @nn.compact
    def __call__(
        self,
        event_times: Array,      # (max_events,) timestep indices, -1 for padding
        event_types: Array,      # (max_events,) type indices
        event_magnitudes: Array, # (max_events,) magnitudes
        event_directions: Array, # (max_events,) directions
        n_timesteps: int,
    ) -> Array:
        """
        MED-003 FIX: JIT-compatible event encoding using padded arrays.

        Args:
            event_times: (max_events,) timestep indices, -1 for invalid/padding
            event_types: (max_events,) pre-hashed type indices in [0, n_event_types)
            event_magnitudes: (max_events,) magnitude values
            event_directions: (max_events,) direction values
            n_timesteps: Sequence length

        Returns:
            (n_timesteps, d_model) sparse event embeddings
        """
        # Event type embedding
        type_embed = nn.Embed(num_embeddings=self.n_event_types, features=self.d_model // 2)

        # Magnitude projection
        mag_proj = nn.Dense(self.d_model // 2)

        # Compute all embeddings in parallel (vectorized)
        type_embs = type_embed(event_types)  # (max_events, d_model // 2)

        # Stack magnitude and direction for projection
        mag_dir = jnp.stack([event_magnitudes, event_directions], axis=-1)  # (max_events, 2)
        mag_embs = mag_proj(mag_dir)  # (max_events, d_model // 2)

        # Combine type and magnitude embeddings
        event_embs = jnp.concatenate([type_embs, mag_embs], axis=-1)  # (max_events, d_model)

        # Create valid mask (times >= 0 and < n_timesteps)
        valid_mask = (event_times >= 0) & (event_times < n_timesteps)  # (max_events,)

        # Create output using scatter-like operation
        # Initialize output
        output = jnp.zeros((n_timesteps, self.d_model))

        # Use segment_sum style: for each event, add its embedding to output at its time
        # This is JIT-compatible using jax.ops.segment_sum
        # But simpler: use at[].add with clipped indices and masking

        # Clip times to valid range (invalid times go to index 0, will be masked)
        safe_times = jnp.clip(event_times, 0, n_timesteps - 1)

        # Mask invalid embeddings to zero
        masked_embs = event_embs * valid_mask[:, None]

        # Scatter-add each embedding to its timestep
        # Using a loop here is fine since max_events is small and static
        def add_event(output, idx):
            time = safe_times[idx]
            emb = masked_embs[idx]
            return output.at[time].add(emb)

        output = jax.lax.fori_loop(0, self.max_events, lambda i, o: add_event(o, i), output)

        return output

    @staticmethod
    def prepare_events(events: list, n_event_types: int = 10, max_events: int = 32):
        """
        Helper to convert Event list to padded arrays for JIT-compatible encoding.

        Call this OUTSIDE of JIT to prepare event data.

        Args:
            events: List of Event objects
            n_event_types: Number of event type buckets
            max_events: Maximum events to encode

        Returns:
            Tuple of (event_times, event_types, event_magnitudes, event_directions)
        """
        import numpy as np

        n = min(len(events), max_events)

        event_times = np.full(max_events, -1, dtype=np.int32)
        event_types = np.zeros(max_events, dtype=np.int32)
        event_magnitudes = np.zeros(max_events, dtype=np.float32)
        event_directions = np.zeros(max_events, dtype=np.float32)

        for i, event in enumerate(events[:n]):
            event_times[i] = event.time
            # Pre-hash event type
            event_types[i] = (zlib.adler32(event.event_type.encode()) & 0xffffffff) % n_event_types
            event_magnitudes[i] = event.magnitude
            event_directions[i] = event.direction

        return (
            jnp.array(event_times),
            jnp.array(event_types),
            jnp.array(event_magnitudes),
            jnp.array(event_directions),
        )


class SkeletonEncoder(nn.Module):
    """
    Full skeleton encoder combining all components.

    Outputs FiLM parameters (gamma, beta) for conditioning.

    HIGH-007 FIX: Now properly handles sequence embeddings instead of discarding them.
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
        global_embeddings = []
        sequence_embeddings = []  # HIGH-007 FIX: Track sequence embeddings separately

        # Encode each component if present
        if skeleton.trend_direction is not None:
            trend_emb = TrendEncoder(self.d_model)(skeleton.trend_direction)
            if trend_emb.ndim == 1:
                global_embeddings.append(trend_emb)
            else:
                # HIGH-007 FIX: Keep sequence embeddings
                sequence_embeddings.append(trend_emb)

        if skeleton.vol_regime is not None:
            vol_emb = VolRegimeEncoder(self.d_model)(skeleton.vol_regime)
            if vol_emb.ndim == 1:
                global_embeddings.append(vol_emb)
            else:
                # HIGH-007 FIX: Keep sequence embeddings
                sequence_embeddings.append(vol_emb)

        if skeleton.correlation_clusters is not None:
            cluster_emb = ClusterEncoder(self.d_model, self.n_clusters)(
                skeleton.correlation_clusters.astype(jnp.int32)
            )
            global_embeddings.append(cluster_emb)

        if skeleton.alpha_shape is not None:
            # HIGH-007 FIX: Include alpha shape (always sequence)
            alpha_emb = AlphaShapeEncoder(self.d_model)(skeleton.alpha_shape)
            sequence_embeddings.append(alpha_emb)

        if skeleton.regime_sequence is not None:
            regime_emb = RegimeEncoder(self.d_model, self.n_regimes)(
                skeleton.regime_sequence.astype(jnp.int32)
            )
            if regime_emb.ndim == 1:
                global_embeddings.append(regime_emb)
            else:
                # HIGH-007 FIX: Keep sequence embeddings
                sequence_embeddings.append(regime_emb)

        # Combine global embeddings
        if global_embeddings:
            global_combined = jnp.stack(global_embeddings, axis=0).mean(axis=0)
        else:
            global_combined = jnp.zeros(self.d_model)

        # HIGH-007 FIX: Combine sequence embeddings if present and return_sequence
        # BUG-007 FIX: Proper per-timestep normalization (not just count of embedding types)
        if return_sequence and sequence_embeddings:
            # Stack and average sequence embeddings
            # Need to handle potentially different lengths - use the skeleton's n_timesteps
            n_timesteps = skeleton.n_timesteps
            seq_combined = jnp.zeros((n_timesteps, self.d_model))
            # BUG-007 FIX: Track contribution count per timestep
            contribution_count = jnp.zeros(n_timesteps)
            for seq_emb in sequence_embeddings:
                # Truncate or pad to match n_timesteps
                seq_len = seq_emb.shape[0]
                if seq_len >= n_timesteps:
                    seq_combined = seq_combined + seq_emb[:n_timesteps]
                    contribution_count = contribution_count + 1.0
                else:
                    seq_combined = seq_combined.at[:seq_len].add(seq_emb)
                    contribution_count = contribution_count.at[:seq_len].add(1.0)
            # BUG-007 FIX: Normalize by actual contributions at each timestep
            seq_combined = seq_combined / jnp.maximum(contribution_count[:, None], 1.0)

            # Combine with global embedding broadcast to sequence
            combined_seq = seq_combined + global_combined[None, :]

            # Generate per-timestep FiLM parameters
            film_mlp = nn.Sequential([
                nn.Dense(self.d_model * 2),
                nn.silu,
                nn.Dense(self.d_model * 2),
            ])
            film_params = jax.vmap(film_mlp)(combined_seq)
            gamma, beta = jnp.split(film_params, 2, axis=-1)
        else:
            # HIGH-007 FIX: If we have sequence data but not returning sequence,
            # pool sequence embeddings into global
            if sequence_embeddings:
                seq_pooled = jnp.stack([s.mean(axis=0) for s in sequence_embeddings], axis=0).mean(axis=0)
                global_combined = (global_combined + seq_pooled) / 2.0

            # Generate global FiLM parameters
            film_mlp = nn.Sequential([
                nn.Dense(self.d_model * 2),
                nn.silu,
                nn.Dense(self.d_model * 2),
            ])
            film_params = film_mlp(global_combined)
            gamma, beta = jnp.split(film_params, 2, axis=-1)

        # Initial state for Mamba (always from global)
        h0 = nn.Dense(self.d_model)(global_combined)

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
