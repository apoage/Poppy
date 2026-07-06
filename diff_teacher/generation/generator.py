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

        # JIT compile generation with static shape args
        self._generate_jit = jax.jit(
            self._generate_impl,
            static_argnums=(2, 3, 6)  # n_samples, seq_length, n_steps
        )

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
        # Handle nested params structure from training checkpoints
        # Structure can be: {'params': ...} or {'params': {'params': ...}} or {'opt_state': ..., 'params': ...}
        if 'params' in params:
            params = params['params']
        # Check for double-nested params (from Flax TrainState)
        if isinstance(params, dict) and 'params' in params:
            params = params['params']
        if params is None:
            raise ValueError(f"Checkpoint at {checkpoint_path} has no 'params' key")

        return cls(model, params, config)

    def generate(
        self,
        n_samples: int = 1,
        seq_length: Optional[int] = None,
        skeleton: Optional[MarketSkeleton] = None,
        skeleton_cond: Optional[Tuple[Any, Any, Any]] = None,
        regime: Optional[Union[int, str]] = None,
        seed: int = 42,
        n_steps: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate market data.

        Args:
            n_samples: Number of samples to generate
            seq_length: Sequence length (default from config)
            skeleton: Optional MarketSkeleton for conditioning (requires encoder params)
            skeleton_cond: Optional pre-encoded (gamma, beta, h0) - CRIT-002 FIX
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

        # CRIT-002 FIX: Handle skeleton conditioning
        # If skeleton_cond is pre-encoded, use it directly
        # If skeleton is provided but not skeleton_cond, raise error (fail-safe)
        # BUG-FIX: Changed from warning to ValueError to prevent silent unconditional generation
        if skeleton is not None and skeleton_cond is None:
            raise ValueError(
                "skeleton provided but skeleton_cond is None. "
                "Skeleton encoding requires SkeletonConditioner params. "
                "Either pass pre-encoded skeleton_cond for conditioning, "
                "or remove the skeleton argument for unconditional generation."
            )

        # Generate
        key = jax.random.PRNGKey(seed)
        generated = self._generate_jit(
            self.params,
            key,
            n_samples,
            seq_length,
            regime,
            skeleton_cond,
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
        skeleton_cond: Optional[Tuple[Array, Array, Array]],  # CRIT-002 FIX
        n_steps: int,
    ) -> Array:
        """Internal generation implementation."""
        input_dim = self.config.n_assets * self.config.n_features

        # Sample initial noise
        key, noise_key = jax.random.split(key)
        x = jax.random.normal(noise_key, (n_samples, seq_length, input_dim))

        # ODE integration (Euler method for straight paths)
        # HIGH-005 FIX: Use lax.fori_loop instead of Python for loop
        dt = 1.0 / n_steps

        def euler_step(i, x):
            t = jnp.full((n_samples,), i * dt)
            # Predict velocity using velocity_net submodule
            # Conv model has velocity_net submodule, call it directly
            if 'velocity_net' in params:
                # ConvFlowMatchingGenerator: call velocity_net directly
                from diff_teacher.conv_flow_matching import ConvVelocityNet
                velocity_net = ConvVelocityNet(
                    hidden_dim=self.config.hidden_dim,
                    num_layers=self.config.num_layers,
                )
                v = velocity_net.apply(
                    {'params': params['velocity_net']},
                    x, t, regime, skeleton_cond, False  # training=False
                )
            else:
                # Mamba or other: call model directly
                v = self.model.apply({'params': params}, x, t, regime, skeleton_cond)
            # Euler step
            return x + v * dt

        x = jax.lax.fori_loop(0, n_steps, euler_step, x)

        return x

    def generate_on_device(
        self,
        n_samples: int = 1,
        skeleton: Optional[MarketSkeleton] = None,
        skeleton_cond: Optional[Tuple[Array, Array, Array]] = None,  # CRIT-002 FIX
        seed: int = 42,
        regime: Optional[Union[int, str, Array]] = None,
        invert_regime: bool = False,
    ) -> Array:
        """
        Generate directly on device (no CPU roundtrip).

        Args:
            n_samples: Number of samples to generate
            skeleton: Optional skeleton conditioning
            skeleton_cond: Pre-encoded skeleton conditioning
            seed: Random seed
            regime: Optional regime conditioning (int, str, or array of indices)
            invert_regime: If True, swap crash↔rally for dual-branch diversity

        Returns JAX array on GPU for downstream JAX computation.
        """
        key = jax.random.PRNGKey(seed)

        # Handle regime conditioning
        regime_arr = None
        if regime is not None:
            if isinstance(regime, str):
                from diff_teacher.skeleton.types import REGIME_TO_IDX
                regime_idx = REGIME_TO_IDX.get(regime, 2)
                regime_arr = jnp.array([regime_idx] * n_samples)
            elif isinstance(regime, int):
                regime_arr = jnp.array([regime] * n_samples)
            else:
                regime_arr = regime  # Already an array

            # Invert directional regimes: crash(0)↔rally(1)
            if invert_regime and regime_arr is not None:
                # Swap 0↔1, keep 2,3,4 unchanged
                regime_arr = jnp.where(regime_arr == 0, 1,
                             jnp.where(regime_arr == 1, 0, regime_arr))

        return self._generate_jit(
            self.params,
            key,
            n_samples,
            self.config.seq_length,
            regime_arr,
            skeleton_cond,  # CRIT-002 FIX
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
