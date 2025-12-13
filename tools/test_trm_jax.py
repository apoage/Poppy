import jax
import jax.numpy as jnp
from mcts_trader.jax_agent.trm_flax import create_trm_model

def test_trm_forward():
    """Test TRM forward pass"""
    print("🚀 Testing JAX/Flax TRM Implementation...")
    
    # Create model
    model = create_trm_model(state_dim=64, n_actions=192)

    # Initialize with dummy input
    rng = jax.random.PRNGKey(0)
    dummy_obs = jnp.ones((2, 27, 64, 64))  # batch=2

    print("Initializing model parameters...")
    # Initialize parameters
    variables = model.init(rng, dummy_obs)
    params = variables['params']

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"✅ Parameter count: {param_count:,}")

    # Forward pass
    print("Running forward pass...")
    policy_logits, value, state, latent = model.apply(variables, dummy_obs)

    print(f"  Policy logits shape: {policy_logits.shape}")  # (2, 192)
    print(f"  Value shape: {value.shape}")  # (2, 601)
    print(f"  State shape: {state.shape}")  # (2, 64)
    print(f"  Latent shape: {latent.shape}")  # (2, 64)

    assert policy_logits.shape == (2, 192)
    assert value.shape == (2, 601)
    assert state.shape == (2, 64)

    print("✅ TRM forward pass test passed!")

if __name__ == "__main__":
    test_trm_forward()
