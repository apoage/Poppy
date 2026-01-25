"""Fluent builder for creating custom market skeletons."""

import jax.numpy as jnp
from jax import Array
from typing import List, Optional, Union, Tuple

from .types import MarketSkeleton, Event, RegimeType, REGIME_TO_IDX, REGIME_NAMES


class SkeletonBuilder:
    """
    Fluent builder for creating custom market skeletons.

    Example:
        skeleton = SkeletonBuilder() \\
            .set_timesteps(1440) \\
            .set_assets(64) \\
            .add_regime('sideways', duration=100) \\
            .add_regime('crash', duration=50) \\
            .add_regime('recovery', duration=200) \\
            .set_volatility_percentile(0.8) \\
            .set_correlation_mode('crisis') \\
            .add_event(150, 'flash_crash', magnitude=0.9) \\
            .build()
    """

    def __init__(self):
        self._n_timesteps = 1440
        self._n_assets = 64
        self._regime_segments: List[Tuple[RegimeType, int]] = []
        self._vol_percentile: Optional[float] = None
        self._vol_sequence: Optional[Array] = None
        self._correlation_mode: Optional[str] = None
        self._correlation_clusters: Optional[Array] = None
        self._trend_direction: Optional[Array] = None
        self._alpha_shape: Optional[Array] = None
        self._events: List[Event] = []

    def set_timesteps(self, n: int) -> 'SkeletonBuilder':
        """Set number of timesteps."""
        self._n_timesteps = n
        return self

    def set_assets(self, n: int) -> 'SkeletonBuilder':
        """Set number of assets."""
        self._n_assets = n
        return self

    def add_regime(self, regime: RegimeType, duration: int) -> 'SkeletonBuilder':
        """Add a regime segment."""
        if regime not in REGIME_NAMES:
            raise ValueError(f"Unknown regime: {regime}. Must be one of {REGIME_NAMES}")
        self._regime_segments.append((regime, duration))
        return self

    def set_regime_sequence(self, regimes: List[RegimeType]) -> 'SkeletonBuilder':
        """Set regime sequence (equal duration segments)."""
        if not regimes:
            return self
        duration = self._n_timesteps // len(regimes)
        self._regime_segments = [(r, duration) for r in regimes]
        return self

    def set_single_regime(self, regime: RegimeType) -> 'SkeletonBuilder':
        """Set a single regime for entire sequence."""
        self._regime_segments = [(regime, self._n_timesteps)]
        return self

    def set_volatility_percentile(self, percentile: float) -> 'SkeletonBuilder':
        """Set constant volatility percentile [0, 1]."""
        self._vol_percentile = max(0.0, min(1.0, percentile))
        return self

    def set_volatility_sequence(self, sequence: Array) -> 'SkeletonBuilder':
        """Set custom volatility sequence."""
        self._vol_sequence = sequence
        return self

    def set_correlation_mode(self, mode: str) -> 'SkeletonBuilder':
        """
        Set correlation mode.

        Modes:
            'independent': Each asset in its own cluster
            'crisis': All assets in same cluster (high correlation)
            'sector': Assets grouped into 5 sectors
            'pairs': Assets grouped into pairs
        """
        self._correlation_mode = mode
        return self

    def set_correlation_clusters(self, clusters: Union[Array, List[List[int]]]) -> 'SkeletonBuilder':
        """
        Set custom correlation clusters.

        Args:
            clusters: Either Array of cluster IDs per asset, or
                     List of asset groups [[0,1,2], [3,4,5], ...]
        """
        if isinstance(clusters, list):
            # Convert groups to cluster IDs
            cluster_ids = jnp.zeros(self._n_assets, dtype=jnp.int32)
            for cluster_idx, group in enumerate(clusters):
                for asset_idx in group:
                    if asset_idx < self._n_assets:
                        cluster_ids = cluster_ids.at[asset_idx].set(cluster_idx)
            self._correlation_clusters = cluster_ids
        else:
            self._correlation_clusters = clusters
        return self

    def set_trend_direction(
        self,
        direction: Union[int, List[int], Array],
        per_asset: bool = False
    ) -> 'SkeletonBuilder':
        """
        Set trend direction.

        Args:
            direction: Single value, per-asset list, or full array
            per_asset: If True and direction is list, apply per asset
        """
        if isinstance(direction, int):
            # Single value for all
            self._trend_direction = jnp.full(
                (self._n_timesteps - 1, self._n_assets), direction, dtype=jnp.int32
            )
        elif isinstance(direction, list):
            if per_asset:
                # Per-asset, constant over time
                direction_arr = jnp.array(direction, dtype=jnp.int32)
                self._trend_direction = jnp.broadcast_to(
                    direction_arr[None, :],
                    (self._n_timesteps - 1, len(direction))
                )
            else:
                # Per-timestep, broadcast to assets
                direction_arr = jnp.array(direction, dtype=jnp.int32)
                self._trend_direction = jnp.broadcast_to(
                    direction_arr[:, None],
                    (len(direction), self._n_assets)
                )
        else:
            self._trend_direction = direction
        return self

    def set_alpha_shape(self, shape: Union[str, Array]) -> 'SkeletonBuilder':
        """
        Set alpha opportunity shape.

        Args:
            shape: 'linear', 'early', 'late', 'sparse', or custom array
        """
        n = self._n_timesteps - 1
        t = jnp.linspace(0, 1, n)

        if isinstance(shape, str):
            if shape == 'linear':
                self._alpha_shape = t
            elif shape == 'early':
                # Opportunities concentrated early
                self._alpha_shape = jnp.sqrt(t)
            elif shape == 'late':
                # Opportunities concentrated late
                self._alpha_shape = t ** 2
            elif shape == 'sparse':
                # Few concentrated opportunities
                self._alpha_shape = jnp.where(
                    (t > 0.2) & (t < 0.3), t * 3,
                    jnp.where((t > 0.7) & (t < 0.8), t * 3, t * 0.1)
                )
                # MED-007 FIX: Guard against div/0 when last element is near zero
                self._alpha_shape = self._alpha_shape / jnp.maximum(self._alpha_shape[-1], 1e-6)
            else:
                raise ValueError(f"Unknown alpha shape: {shape}")
        else:
            self._alpha_shape = shape
        return self

    def add_event(
        self,
        time: int,
        event_type: str,
        magnitude: float = 0.5,
        direction: int = 0
    ) -> 'SkeletonBuilder':
        """Add a discrete event."""
        self._events.append(Event(
            time=time,
            event_type=event_type,
            magnitude=max(0.0, min(1.0, magnitude)),
            direction=direction
        ))
        return self

    def add_flash_crash(self, time: int, magnitude: float = 0.8) -> 'SkeletonBuilder':
        """Add flash crash event."""
        return self.add_event(time, 'flash_crash', magnitude, direction=-1)

    def add_rally(self, time: int, magnitude: float = 0.6) -> 'SkeletonBuilder':
        """Add rally event."""
        return self.add_event(time, 'rally', magnitude, direction=1)

    def add_volatility_spike(self, time: int, magnitude: float = 0.7) -> 'SkeletonBuilder':
        """Add volatility spike event."""
        return self.add_event(time, 'vol_spike', magnitude, direction=0)

    def _build_regime_sequence(self) -> Optional[Array]:
        """Build regime sequence from segments."""
        if not self._regime_segments:
            return None

        # Calculate total duration
        total = sum(d for _, d in self._regime_segments)

        # Scale to fit n_timesteps
        scale = (self._n_timesteps - 1) / total if total > 0 else 1

        regimes = []
        for regime, duration in self._regime_segments:
            scaled_duration = int(duration * scale)
            regime_idx = REGIME_TO_IDX[regime]
            regimes.extend([regime_idx] * scaled_duration)

        # Pad or truncate to exact length
        while len(regimes) < self._n_timesteps - 1:
            regimes.append(regimes[-1] if regimes else 2)  # Default: sideways
        regimes = regimes[:self._n_timesteps - 1]

        return jnp.array(regimes, dtype=jnp.int32)

    def _build_vol_regime(self) -> Optional[Array]:
        """Build volatility regime sequence."""
        if self._vol_sequence is not None:
            return self._vol_sequence

        if self._vol_percentile is not None:
            return jnp.full(self._n_timesteps - 1, self._vol_percentile)

        # Derive from regime sequence if available
        regime_seq = self._build_regime_sequence()
        if regime_seq is not None:
            # Map regimes to volatility levels
            vol_map = jnp.array([0.9, 0.4, 0.3, 0.8, 0.2])  # crash, rally, sideways, volatile, calm
            # Clip to valid range to prevent OOB indexing
            regime_seq = jnp.clip(regime_seq, 0, len(vol_map) - 1)
            return vol_map[regime_seq]

        return None

    def _build_correlation_clusters(self) -> Optional[Array]:
        """Build correlation cluster assignments."""
        if self._correlation_clusters is not None:
            return self._correlation_clusters

        if self._correlation_mode is not None:
            if self._correlation_mode == 'independent':
                return jnp.arange(self._n_assets, dtype=jnp.int32)
            elif self._correlation_mode == 'crisis':
                return jnp.zeros(self._n_assets, dtype=jnp.int32)
            elif self._correlation_mode == 'sector':
                # 5 sectors
                return jnp.arange(self._n_assets, dtype=jnp.int32) % 5
            elif self._correlation_mode == 'pairs':
                return jnp.arange(self._n_assets, dtype=jnp.int32) // 2
            else:
                raise ValueError(f"Unknown correlation mode: {self._correlation_mode}")

        return None

    def build(self) -> MarketSkeleton:
        """Build the MarketSkeleton."""
        return MarketSkeleton(
            trend_direction=self._trend_direction,
            vol_regime=self._build_vol_regime(),
            correlation_clusters=self._build_correlation_clusters(),
            alpha_shape=self._alpha_shape,
            regime_sequence=self._build_regime_sequence(),
            events=self._events,
            n_timesteps=self._n_timesteps,
            n_assets=self._n_assets,
        )

    def __repr__(self) -> str:
        parts = [f"SkeletonBuilder(timesteps={self._n_timesteps}, assets={self._n_assets})"]
        if self._regime_segments:
            parts.append(f"  regimes: {self._regime_segments}")
        if self._vol_percentile:
            parts.append(f"  vol_percentile: {self._vol_percentile}")
        if self._correlation_mode:
            parts.append(f"  correlation: {self._correlation_mode}")
        if self._events:
            parts.append(f"  events: {len(self._events)}")
        return "\n".join(parts)
