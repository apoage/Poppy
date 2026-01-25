"""Selective State Space Model scan implementations."""

import jax
import jax.numpy as jnp
from jax import Array, lax
from typing import Tuple


def selective_scan_sequential(
    x: Array,
    delta: Array,
    A: Array,
    B: Array,
    C: Array,
) -> Array:
    """
    Sequential selective scan (reference implementation).

    This is slower but correct - use for testing.

    Args:
        x: Input, shape (batch, seq, d_inner)
        delta: Timestep, shape (batch, seq, d_inner)
        A: State transition, shape (d_inner, d_state)
        B: Input projection, shape (batch, seq, d_state)
        C: Output projection, shape (batch, seq, d_state)

    Returns:
        Output, shape (batch, seq, d_inner)
    """
    batch, seq, d_inner = x.shape
    d_state = A.shape[-1]

    def scan_fn(h, inputs):
        x_t, delta_t, B_t, C_t = inputs

        # Discretize A: exp(delta * A)
        delta_A = jnp.exp(delta_t[:, :, None] * A)  # (batch, d_inner, d_state)

        # Discretize B: delta * B
        delta_B = delta_t[:, :, None] * B_t[:, None, :]  # (batch, d_inner, d_state)

        # State update: h = A_bar * h + B_bar * x
        h = delta_A * h + delta_B * x_t[:, :, None]

        # Output: y = C @ h
        y = jnp.sum(h * C_t[:, None, :], axis=-1)  # (batch, d_inner)

        return h, y

    # Initialize state
    h0 = jnp.zeros((batch, d_inner, d_state))

    # Transpose for scan over sequence
    x_t = x.transpose(1, 0, 2)        # (seq, batch, d_inner)
    delta_t = delta.transpose(1, 0, 2)
    B_t = B.transpose(1, 0, 2)        # (seq, batch, d_state)
    C_t = C.transpose(1, 0, 2)

    # Run scan
    _, y = lax.scan(scan_fn, h0, (x_t, delta_t, B_t, C_t))

    # Transpose back
    return y.transpose(1, 0, 2)  # (batch, seq, d_inner)


def selective_scan_parallel(
    x: Array,
    delta: Array,
    A: Array,
    B: Array,
    C: Array,
) -> Array:
    """
    Parallel selective scan using associative scan.

    O(log L) depth instead of O(L) sequential.
    Uses the associativity of the linear recurrence.

    Args:
        x: Input, shape (batch, seq, d_inner)
        delta: Timestep, shape (batch, seq, d_inner)
        A: State transition, shape (d_inner, d_state)
        B: Input projection, shape (batch, seq, d_state)
        C: Output projection, shape (batch, seq, d_state)

    Returns:
        Output, shape (batch, seq, d_inner)
    """
    batch, seq, d_inner = x.shape
    d_state = A.shape[-1]

    # Discretize A: exp(delta * A)
    # delta: (batch, seq, d_inner), A: (d_inner, d_state)
    # Result: (batch, seq, d_inner, d_state)
    # Clamp to prevent exp overflow/underflow
    delta_A = jnp.exp(jnp.clip(jnp.einsum('bsd,dn->bsdn', delta, A), -20.0, 20.0))

    # Discretize B*x: delta * B * x
    # delta: (batch, seq, d_inner), B: (batch, seq, d_state), x: (batch, seq, d_inner)
    # Result: (batch, seq, d_inner, d_state)
    delta_B_x = jnp.einsum('bsd,bsn,bsd->bsdn', delta, B, x)

    # Binary operator for associative scan
    # The recurrence h_t = A_t * h_{t-1} + b_t can be composed as:
    # (a1, b1) * (a2, b2) = (a2 * a1, a2 * b1 + b2)
    def binary_operator(
        carry: Tuple[Array, Array],
        element: Tuple[Array, Array]
    ) -> Tuple[Array, Array]:
        a_prev, h_prev = carry
        a_curr, b_curr = element
        return (a_curr * a_prev, a_curr * h_prev + b_curr)

    # Transpose for scan: (batch, seq, d_inner, d_state) -> (seq, batch, d_inner, d_state)
    delta_A_t = delta_A.transpose(1, 0, 2, 3)
    delta_B_x_t = delta_B_x.transpose(1, 0, 2, 3)

    # Run parallel associative scan
    _, h = lax.associative_scan(
        binary_operator,
        (delta_A_t, delta_B_x_t),
        axis=0
    )

    # Transpose back: (seq, batch, d_inner, d_state) -> (batch, seq, d_inner, d_state)
    h = h.transpose(1, 0, 2, 3)

    # Output: y = sum(h * C)
    # h: (batch, seq, d_inner, d_state), C: (batch, seq, d_state)
    y = jnp.einsum('bsdn,bsn->bsd', h, C)

    return y


def selective_scan(
    x: Array,
    delta: Array,
    A: Array,
    B: Array,
    C: Array,
    use_parallel: bool = True,
) -> Array:
    """
    Selective scan with implementation selection.

    Args:
        x: Input, shape (batch, seq, d_inner)
        delta: Timestep, shape (batch, seq, d_inner)
        A: State transition, shape (d_inner, d_state)
        B: Input projection, shape (batch, seq, d_state)
        C: Output projection, shape (batch, seq, d_state)
        use_parallel: Use parallel associative scan (faster)

    Returns:
        Output, shape (batch, seq, d_inner)
    """
    if use_parallel:
        return selective_scan_parallel(x, delta, A, B, C)
    else:
        return selective_scan_sequential(x, delta, A, B, C)
