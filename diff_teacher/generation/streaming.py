"""VRAM-resident streaming generation for downstream JAX consumption."""

import jax
import jax.numpy as jnp
from jax import Array
import flax.linen as nn
from typing import Optional, Iterator, Tuple, Callable, Any
from dataclasses import dataclass
from functools import partial


@dataclass
class StreamConfig:
    """Configuration for streaming generation."""
    batch_size: int = 8
    seq_length: int = 1440
    n_assets: int = 64
    n_features: int = 5
    n_steps: int = 10
    buffer_size: int = 4  # Number of batches to keep in buffer


class VRAMGenerator:
    """
    Generate market data directly on GPU without CPU roundtrip.

    Designed for integration with downstream JAX training loops.
    Data stays on device for maximum throughput.

    Example:
        vram_gen = VRAMGenerator(model, params, config)

        # Generate single batch on device
        key = jax.random.PRNGKey(42)
        batch, key = vram_gen.generate_batch(key)

        # Use in training loop
        for step in range(1000):
            batch, key = vram_gen.generate_batch(key)
            loss = train_step(batch, ...)
    """

    def __init__(
        self,
        model: nn.Module,
        params: dict,
        config: StreamConfig,
    ):
        self.model = model
        self.params = params
        self.config = config

        # Pre-compute input dimension
        self._input_dim = config.n_assets * config.n_features

        # JIT compile generation functions
        self._generate_batch_jit = jax.jit(
            partial(self._generate_batch_impl, config=config)
        )
        self._generate_conditioned_jit = jax.jit(
            partial(self._generate_conditioned_impl, config=config)
        )

    def generate_batch(
        self,
        key: Array,
        regime: Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        """
        Generate a single batch on device.

        Args:
            key: JAX random key
            regime: Optional regime indices (batch_size,)

        Returns:
            Tuple of (generated_data, new_key)
            Generated data shape: (batch_size, seq_length, n_assets, n_features)
        """
        key, subkey = jax.random.split(key)

        generated = self._generate_batch_jit(
            self.params,
            subkey,
            regime,
        )

        # Reshape to (batch, seq, assets, features)
        generated = generated.reshape(
            self.config.batch_size,
            self.config.seq_length,
            self.config.n_assets,
            self.config.n_features,
        )

        return generated, key

    def _generate_batch_impl(
        self,
        params: dict,
        key: Array,
        regime: Optional[Array],
        config: StreamConfig,
    ) -> Array:
        """Internal batch generation."""
        batch_size = config.batch_size
        seq_length = config.seq_length
        input_dim = config.n_assets * config.n_features
        n_steps = config.n_steps

        # Sample initial noise
        x = jax.random.normal(key, (batch_size, seq_length, input_dim))

        # ODE integration
        n_steps = max(n_steps, 1)  # Guard against zero
        dt = 1.0 / n_steps

        def euler_step(carry, t_idx):
            x = carry
            t = jnp.full((batch_size,), t_idx * dt)
            v = self.model.apply({'params': params}, x, t, regime)
            x_new = x + v * dt
            return x_new, None

        x_final, _ = jax.lax.scan(euler_step, x, jnp.arange(n_steps))

        return x_final

    def generate_conditioned(
        self,
        key: Array,
        skeleton_embedding: Array,
    ) -> Tuple[Array, Array]:
        """
        Generate conditioned on skeleton embedding.

        Args:
            key: JAX random key
            skeleton_embedding: Pre-computed skeleton embedding (batch, dim)

        Returns:
            Tuple of (generated_data, new_key)
        """
        key, subkey = jax.random.split(key)

        generated = self._generate_conditioned_jit(
            self.params,
            subkey,
            skeleton_embedding,
        )

        generated = generated.reshape(
            self.config.batch_size,
            self.config.seq_length,
            self.config.n_assets,
            self.config.n_features,
        )

        return generated, key

    def _generate_conditioned_impl(
        self,
        params: dict,
        key: Array,
        skeleton_embedding: Array,
        config: StreamConfig,
    ) -> Array:
        """Internal conditioned generation."""
        batch_size = config.batch_size
        seq_length = config.seq_length
        input_dim = config.n_assets * config.n_features
        n_steps = config.n_steps

        # Sample initial noise
        x = jax.random.normal(key, (batch_size, seq_length, input_dim))

        # ODE integration with conditioning
        dt = 1.0 / n_steps

        def euler_step(carry, t_idx):
            x = carry
            t = jnp.full((batch_size,), t_idx * dt)
            # Pass skeleton embedding as conditioning
            v = self.model.apply(
                {'params': params},
                x,
                t,
                skeleton_cond=skeleton_embedding
            )
            x_new = x + v * dt
            return x_new, None

        x_final, _ = jax.lax.scan(euler_step, x, jnp.arange(n_steps))

        return x_final

    def stream(
        self,
        key: Array,
        n_batches: int,
        regime: Optional[Array] = None,
    ) -> Iterator[Array]:
        """
        Stream batches as a generator.

        Args:
            key: Initial random key
            n_batches: Number of batches to generate
            regime: Optional regime conditioning

        Yields:
            Generated batches (batch_size, seq_length, n_assets, n_features)
        """
        for _ in range(n_batches):
            batch, key = self.generate_batch(key, regime)
            yield batch

    def prefetch_buffer(
        self,
        key: Array,
        regime: Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        """
        Generate a buffer of batches for prefetching.

        Args:
            key: Random key
            regime: Optional regime conditioning

        Returns:
            Tuple of (buffer, new_key)
            Buffer shape: (buffer_size, batch_size, seq_length, n_assets, n_features)
        """
        buffer_size = self.config.buffer_size

        def generate_one(carry, _):
            key = carry
            batch, key = self.generate_batch(key, regime)
            return key, batch

        key, buffer = jax.lax.scan(generate_one, key, None, length=buffer_size)

        return buffer, key


class InfiniteVRAMStream:
    """
    Infinite stream of generated data with automatic key management.

    Example:
        stream = InfiniteVRAMStream(model, params, config, seed=42)

        for step in range(10000):
            batch = next(stream)
            # batch is (batch_size, seq_length, n_assets, n_features)
            loss = train_step(batch, ...)
    """

    def __init__(
        self,
        model: nn.Module,
        params: dict,
        config: StreamConfig,
        seed: int = 42,
    ):
        self.generator = VRAMGenerator(model, params, config)
        self.key = jax.random.PRNGKey(seed)
        self._regime = None

    def set_regime(self, regime: Optional[Array]):
        """Set regime conditioning for subsequent generations."""
        self._regime = regime

    def __iter__(self):
        return self

    def __next__(self) -> Array:
        batch, self.key = self.generator.generate_batch(self.key, self._regime)
        return batch

    def get_batch(self) -> Array:
        """Explicit method alternative to next()."""
        return next(self)


def create_training_data_source(
    model: nn.Module,
    params: dict,
    batch_size: int = 8,
    seq_length: int = 1440,
    n_assets: int = 64,
    n_features: int = 5,
    seed: int = 42,
) -> Callable[[Array], Tuple[Array, Array]]:
    """
    Create a data source function for integration with training loops.

    Returns a function that takes a key and returns (batch, new_key).
    This is the recommended interface for integration.

    Example:
        data_source = create_training_data_source(model, params)

        key = jax.random.PRNGKey(0)
        for step in range(1000):
            batch, key = data_source(key)
            loss = train_step(batch, ...)
    """
    config = StreamConfig(
        batch_size=batch_size,
        seq_length=seq_length,
        n_assets=n_assets,
        n_features=n_features,
    )

    generator = VRAMGenerator(model, params, config)

    def data_source(key: Array, regime: Optional[Array] = None) -> Tuple[Array, Array]:
        return generator.generate_batch(key, regime)

    return data_source
