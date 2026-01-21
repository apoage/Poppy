import jax
import jax.numpy as jnp
import flax
from mcts_trader.jax_agent.muzero_model import MuZeroNetworks
from mcts_trader.jax_agent.trainer import MuZeroTrainer, REWARD_SCALE
from mcts_trader.jax_agent.replay_buffer import JAXReplayBuffer

def test_gradient_flow_to_physics():
    """Verify that gradients flow to physics parameters with Task 106 fixes."""
    n_assets = 4
    state_dim = 16
    batch_size = 2
    
    model = MuZeroNetworks(
        state_dim=state_dim,
        action_space_size=n_assets * 5,
        use_prediction_head=True,
        n_assets=n_assets
    )
    
    trainer = MuZeroTrainer(
        model=model,
        use_prediction_loss=True,
        batch_size=batch_size
    )
    
    rng = jax.random.PRNGKey(0)
    dummy_obs = jnp.zeros((1, 27, n_assets, n_assets))
    state = trainer.init_train_state(rng, dummy_obs)
    
    # Create a batch with NON-ZERO teacher returns (to break the trap)
    batch = {
        'observations': jnp.zeros((batch_size, 5, 27, n_assets, n_assets)),
        'actions': jnp.zeros((batch_size, 5), dtype=jnp.int32),
        'rewards': jnp.zeros((batch_size, 5)),
        'values': jnp.zeros((batch_size, 5)),
        'policy_probs': jnp.ones((batch_size, 5, n_assets * 5)) / (n_assets * 5),
        'log_price_sequence': jnp.ones((batch_size, 6, n_assets)) * 1.0,
        'log_returns': jnp.ones((batch_size, n_assets)) * 0.1, # Non-zero!
        'moving_avg': jnp.ones((batch_size, n_assets)) * 1.0,
        'valid': jnp.ones(batch_size, dtype=bool)
    }
    
    # Compute gradients
    grad_fn = jax.value_and_grad(trainer._loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params, batch, rng, step=100)
    
    # Check if physics parameters have non-zero gradients
    # Path: price_predictor / physics_core / alpha
    physics_grads = grads['price_predictor']['physics_core']
    alpha_grad_norm = jnp.linalg.norm(physics_grads['alpha'])
    beta_grad_norm = jnp.linalg.norm(physics_grads['beta'])
    
    print(f"Alpha Grad Norm: {alpha_grad_norm}")
    print(f"Beta Grad Norm: {beta_grad_norm}")
    
    assert alpha_grad_norm > 0, "Zero gradient for alpha! Physics head still trapped."
    assert beta_grad_norm > 0, "Zero gradient for beta! Physics head still trapped."
    print("✅ Gradient flow to physics verified!")

if __name__ == "__main__":
    test_gradient_flow_to_physics()
