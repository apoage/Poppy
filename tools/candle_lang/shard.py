"""
Device sharding helpers for the candle-language panel (TASK_789).

Tiny by design. Single-GPU is the DEFAULT path (workstation + server members run on one
verified-idle GPU per the build contract); multi-device is the batch axis only, never the
device count. The leading (flat) batch dimension is split across `n_dev` local devices, so it
MUST be divisible by `n_dev` — we assert that loudly rather than silently dropping/padding rows
(the tail batch is already padded to cfg.batch upstream, so divisibility is a real invariant).

No GPU execution happens at import time; `n_devices()` just inspects `jax.local_devices()`.
"""
import jax


def n_devices() -> int:
    """Number of local JAX devices (1 on the single-GPU default path)."""
    return len(jax.local_devices())


def assert_divisible(flat_B: int, n_dev: int) -> None:
    """Raise if the flat batch can't be evenly split across `n_dev` devices."""
    if n_dev <= 0:
        raise ValueError(f"n_dev must be positive, got {n_dev}")
    if flat_B % n_dev != 0:
        raise ValueError(
            f"flat batch {flat_B} not divisible by n_dev={n_dev} "
            f"(per-device shard would be {flat_B / n_dev:.3f} rows). "
            f"Pick a batch size that is a multiple of the device count."
        )


def shard_flat(x, n_dev: int):
    """Place a flat-batch array onto the local devices.

    n_dev == 1 -> a plain device_put (single-GPU default, no reshape, no SPMD overhead).
    n_dev  > 1 -> reshape (n_dev, B//n_dev, *rest) and device_put_sharded one slice per device,
                  so the leading axis maps 1:1 onto jax.local_devices().
    """
    if n_dev == 1:
        return jax.device_put(x)
    assert_divisible(x.shape[0], n_dev)
    rest = x.shape[1:]
    x = x.reshape((n_dev, x.shape[0] // n_dev, *rest))
    return jax.device_put_sharded(list(x), jax.local_devices())
