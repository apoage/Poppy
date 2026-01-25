"""Data structures for market skeleton conditioning."""

from dataclasses import dataclass, field
from typing import List, Optional, Literal
import jax.numpy as jnp
from jax import Array


RegimeType = Literal['crash', 'rally', 'sideways', 'volatile', 'calm']
REGIME_NAMES: List[RegimeType] = ['crash', 'rally', 'sideways', 'volatile', 'calm']
REGIME_TO_IDX = {name: idx for idx, name in enumerate(REGIME_NAMES)}


@dataclass
class Event:
    """Discrete market event (sparse skeleton component)."""
    time: int                    # Timestep index
    event_type: str              # 'earnings', 'fomc', 'flash_crash', 'gap', etc.
    magnitude: float             # Normalized magnitude [0, 1]
    direction: int = 0           # -1, 0, +1


@dataclass
class MarketSkeleton:
    """
    Normalized market structure for conditioning generation.

    All components are normalized/tokenized - the generator learns
    to render them into realistic market data at proper scale.
    """

    # Per-timestep, per-asset: trend direction
    # Shape: (n_timesteps, n_assets) or (n_timesteps,) for global
    # Values: {-1, 0, +1} for bearish/sideways/bullish
    trend_direction: Optional[Array] = None

    # Per-timestep: volatility regime (normalized)
    # Shape: (n_timesteps,)
    # Values: [0, 1] where 0=calm, 1=extreme
    vol_regime: Optional[Array] = None

    # Per-asset: correlation cluster assignment
    # Shape: (n_assets,)
    # Values: cluster IDs 0..K-1
    correlation_clusters: Optional[Array] = None

    # Per-timestep: cumulative alpha opportunity shape
    # Shape: (n_timesteps,)
    # Values: [0, 1] normalized cumulative optimal return
    alpha_shape: Optional[Array] = None

    # Regime sequence (coarse segmentation)
    # Shape: (n_segments,) or single value
    # Values: indices into REGIME_NAMES
    regime_sequence: Optional[Array] = None

    # Sparse events
    events: List[Event] = field(default_factory=list)

    # Metadata
    n_timesteps: int = 1440
    n_assets: int = 64

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'trend_direction': self.trend_direction.tolist() if self.trend_direction is not None else None,
            'vol_regime': self.vol_regime.tolist() if self.vol_regime is not None else None,
            'correlation_clusters': self.correlation_clusters.tolist() if self.correlation_clusters is not None else None,
            'alpha_shape': self.alpha_shape.tolist() if self.alpha_shape is not None else None,
            'regime_sequence': self.regime_sequence.tolist() if self.regime_sequence is not None else None,
            'events': [{'time': e.time, 'type': e.event_type, 'magnitude': e.magnitude, 'direction': e.direction}
                      for e in self.events],
            'n_timesteps': self.n_timesteps,
            'n_assets': self.n_assets,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'MarketSkeleton':
        """Create from dictionary."""
        events = [Event(e['time'], e['type'], e['magnitude'], e.get('direction', 0))
                  for e in d.get('events', [])]
        return cls(
            trend_direction=jnp.array(d['trend_direction']) if d.get('trend_direction') else None,
            vol_regime=jnp.array(d['vol_regime']) if d.get('vol_regime') else None,
            correlation_clusters=jnp.array(d['correlation_clusters']) if d.get('correlation_clusters') else None,
            alpha_shape=jnp.array(d['alpha_shape']) if d.get('alpha_shape') else None,
            regime_sequence=jnp.array(d['regime_sequence']) if d.get('regime_sequence') else None,
            events=events,
            n_timesteps=d.get('n_timesteps', 1440),
            n_assets=d.get('n_assets', 64),
        )

    def get_regime_idx(self) -> int:
        """Get single regime index (for simple conditioning)."""
        if self.regime_sequence is not None and self.regime_sequence.shape[0] > 0:
            # Return most common regime
            return int(jnp.bincount(self.regime_sequence.astype(jnp.int32)).argmax())
        return 2  # Default: sideways

    def has_conditioning(self) -> bool:
        """Check if skeleton has any conditioning signals."""
        return any([
            self.trend_direction is not None,
            self.vol_regime is not None,
            self.correlation_clusters is not None,
            self.alpha_shape is not None,
            self.regime_sequence is not None,
            len(self.events) > 0,
        ])


def empty_skeleton(n_timesteps: int = 1440, n_assets: int = 64) -> MarketSkeleton:
    """Create empty skeleton (unconditional generation)."""
    return MarketSkeleton(n_timesteps=n_timesteps, n_assets=n_assets)


def regime_only_skeleton(regime: RegimeType, n_timesteps: int = 1440, n_assets: int = 64) -> MarketSkeleton:
    """Create skeleton with only regime specified."""
    regime_idx = REGIME_TO_IDX[regime]
    return MarketSkeleton(
        regime_sequence=jnp.array([regime_idx]),
        n_timesteps=n_timesteps,
        n_assets=n_assets,
    )
