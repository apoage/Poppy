"""High-level market generation API."""

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np
from typing import Optional, Union, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import json

import flax.linen as nn
from flax.training import checkpoints

# Import skeleton types (graceful fallback if not available)
try:
    from diff_teacher.skeleton.types import MarketSkeleton
    from diff_teacher.skeleton.encoder import SkeletonConditioner
    from diff_teacher.skeleton.builder import SkeletonBuilder
except ImportError:
    MarketSkeleton = None
    SkeletonConditioner = None
    SkeletonBuilder = None


@dataclass
class GeneratorConfig:
    """Configuration for market generator."""
    # Model architecture
    architecture: str = "conv"  # "conv" or "mamba"
    hidden_dim: int = 768
    num_layers: int = 12
    d_state: int = 64  # For Mamba

    # Data dimensions
    n_assets: int = 64
    n_features: int = 5  # OHLCV
    seq_length: int = 1440

    # Generation
    n_steps: int = 10  # ODE integration steps
    sigma_min: float = 1e-4

    def to_dict(self) -> dict:
        return {
            'architecture': self.architecture,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'd_state': self.d_state,
            'n_assets': self.n_assets,
            'n_features': self.n_features,
            'seq_length': self.seq_length,
            'n_steps': self.n_steps,
            'sigma_min': self.sigma_min,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'GeneratorConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class MarketGenerator:
    """
    High-level API for market generation.

    Example:
        generator = MarketGenerator.from_checkpoint("checkpoints/model")

        # Unconditional generation
        prices = generator.generate(n_samples=10, seed=42)

        # Skeleton-conditioned generation
        skeleton = SkeletonBuilder().set_regime_sequence(['crash', 'recovery']).build()
        prices = generator.generate(n_samples=10, skeleton=skeleton, seed=42)
    """

    def __init__(
        self,
        model: nn.Module,
        params: dict,
        config: GeneratorConfig,
    ):
        self.model = model
        self.params = params
        self.config = config

        # JIT compile generation
        self._generate_jit = jax.jit(self._generate_impl)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config: Optional[GeneratorConfig] = None,
    ) -> 'MarketGenerator':
        """
        Load generator from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory
            config: Optional config (loaded from checkpoint if not provided)

        Returns:
            MarketGenerator instance
        """
        checkpoint_path = Path(checkpoint_path)

        # Load config
        config_path = checkpoint_path / "config.json"
        if config_path.exists() and config is None:
            with open(config_path) as f:
                config = GeneratorConfig.from_dict(json.load(f))
        elif config is None:
            config = GeneratorConfig()

        # Create model based on architecture
        if config.architecture == "mamba":
            from diff_teacher.mamba import MambaFlowGenerator
            model = MambaFlowGenerator(
                d_model=config.hidden_dim,
                n_layers=config.num_layers,
                d_state=config.d_state,
            )
        else:
            # Default to conv
            from diff_teacher.conv_flow_matching import ConvFlowMatchingGenerator
            model = ConvFlowMatchingGenerator(
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
            )

        # Load params - MED-008 FIX: Validate checkpoint loaded successfully
        params = checkpoints.restore_checkpoint(str(checkpoint_path), target=None)
        if params is None:
            raise ValueError(f"Failed to load checkpoint from {checkpoint_path}")
        if 'params' in params:
            params = params['params']
        if params is None:
            raise ValueError(f"Checkpoint at {checkpoint_path} has no 'params' key")

        return cls(model, params, config)

    def generate(
        self,
        n_samples: int = 1,
        seq_length: Optional[int] = None,
        skeleton: Optional[MarketSkeleton] = None,
        regime: Optional[Union[int, str]] = None,
        seed: int = 42,
        n_steps: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate market data.

        Args:
            n_samples: Number of samples to generate
            seq_length: Sequence length (default from config)
            skeleton: Optional MarketSkeleton for conditioning
            regime: Optional regime (index or name)
            seed: Random seed
            n_steps: ODE integration steps (default from config)

        Returns:
            Generated prices, shape (n_samples, seq_length, n_assets, n_features)
        """
        seq_length = seq_length or self.config.seq_length
        n_steps = n_steps or self.config.n_steps

        # Handle regime
        if regime is not None:
            if isinstance(regime, str):
                from diff_teacher.skeleton.types import REGIME_TO_IDX
                regime = REGIME_TO_IDX.get(regime, 2)
            regime = jnp.array([regime] * n_samples)
        else:
            regime = None

        # Generate
        key = jax.random.PRNGKey(seed)
        generated = self._generate_jit(
            self.params,
            key,
            n_samples,
            seq_length,
            regime,
            n_steps,
        )

        # Convert to numpy and reshape
        generated = np.array(generated)

        # Reshape from (batch, seq, assets*features) to (batch, seq, assets, features)
        generated = generated.reshape(
            n_samples,
            seq_length,
            self.config.n_assets,
            self.config.n_features,
        )

        return generated

    def _generate_impl(
        self,
        params: dict,
        key: Array,
        n_samples: int,
        seq_length: int,
        regime: Optional[Array],
        n_steps: int,
    ) -> Array:
        """Internal generation implementation."""
        input_dim = self.config.n_assets * self.config.n_features

        # Sample initial noise
        key, noise_key = jax.random.split(key)
        x = jax.random.normal(noise_key, (n_samples, seq_length, input_dim))

        # ODE integration (Euler method for straight paths)
        dt = 1.0 / n_steps

        for i in range(n_steps):
            t = jnp.array([i * dt] * n_samples)

            # Predict velocity
            v = self.model.apply({'params': params}, x, t, regime)

            # Euler step
            x = x + v * dt

        return x

    def generate_on_device(
        self,
        n_samples: int = 1,
        skeleton: Optional[MarketSkeleton] = None,
        seed: int = 42,
    ) -> Array:
        """
        Generate directly on device (no CPU roundtrip).

        Returns JAX array on GPU for downstream JAX computation.
        """
        key = jax.random.PRNGKey(seed)
        return self._generate_jit(
            self.params,
            key,
            n_samples,
            self.config.seq_length,
            None,
            self.config.n_steps,
        )


def generate_markets(
    checkpoint_path: str,
    n_samples: int = 10,
    regime: Optional[str] = None,
    seed: int = 42,
) -> np.ndarray:
    """
    Convenience function for quick generation.

    Args:
        checkpoint_path: Path to model checkpoint
        n_samples: Number of samples
        regime: Optional regime name
        seed: Random seed

    Returns:
        Generated prices
    """
    generator = MarketGenerator.from_checkpoint(checkpoint_path)
    return generator.generate(n_samples=n_samples, regime=regime, seed=seed)
