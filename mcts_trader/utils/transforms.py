import jax.numpy as jnp
import numpy as np
from typing import Union, Any

# Type alias for JAX or NumPy arrays
Array = Union[np.ndarray, Any]

def symlog(x: Array, threshold: float = 1.0) -> Array:
    """
    Symmetric logarithmic transformation.
    Compresses large values while preserving sign and small values.
    
    y = sign(x) * log(|x| + 1)
    """
    if isinstance(x, np.ndarray):
        return np.sign(x) * np.log1p(np.abs(x))
    else:
        # JAX implementation
        return jnp.sign(x) * jnp.log1p(jnp.abs(x))

def inverse_symlog(y: Array, threshold: float = 1.0) -> Array:
    """
    Inverse symmetric logarithmic transformation.
    
    x = sign(y) * (exp(|y|) - 1)
    """
    if isinstance(y, np.ndarray):
        return np.sign(y) * (np.expm1(np.abs(y)))
    else:
        # JAX implementation
        return jnp.sign(y) * (jnp.expm1(jnp.abs(y)))

def test_transforms():
    """Simple self-test."""
    values = np.array([-100.0, -1.0, 0.0, 0.5, 1.0, 100.0])
    transformed = symlog(values)
    reconstructed = inverse_symlog(transformed)
    
    assert np.allclose(values, reconstructed), f"Reconstruction failed: {values} vs {reconstructed}"
    print("✅ Transforms self-test passed")

if __name__ == "__main__":
    test_transforms()
