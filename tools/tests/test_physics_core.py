import flax
import jax
import jax.numpy as jnp
from mcts_trader.jax_agent.prediction.physics_core import DifferentiableMarketPhysics

def test_physics_signals():
    """Verify that physics components respond correctly to market signals."""
    n_assets = 1
    n_factors = 2
    
    model = DifferentiableMarketPhysics(n_assets=n_assets, n_factors=n_factors)
    
    rng = jax.random.PRNGKey(0)
    
    # 1. Pure Momentum Test
    # Price is at MA, but returns are positive. G_momentum should be positive.
    log_prices = jnp.array([[1.0]])
    log_returns = jnp.array([[0.1]]) # Upward trend
    moving_avg = jnp.array([[1.0]])
    noise = jnp.array([[0.0]])
    
    params = model.init(rng, log_prices, log_returns, moving_avg, noise)
    
    # Set alpha to a known positive value
    params = flax.core.unfreeze(params)
    params['params']['alpha'] = jnp.array([0.0]) # softplus(0)*0.3 approx 0.2
    
    output, _ = model.apply(params, log_prices, log_returns, moving_avg, noise)
    
    print(f"G_momentum (Trend Up): {output.G_momentum}")
    assert output.G_momentum > 0, "Momentum should be positive for upward trend"
    
    # 2. Pure Reversion Test
    # Returns are 0, but price is above MA. G_reversion should be negative (pull down).
    log_prices_high = jnp.array([[1.1]])
    log_returns_zero = jnp.array([[0.0]])
    moving_avg_low = jnp.array([[1.0]])
    
    output_rev, _ = model.apply(params, log_prices_high, log_returns_zero, moving_avg_low, noise)
    
    print(f"G_reversion (Price High): {output_rev.G_reversion}")
    assert output_rev.G_reversion < 0, "Reversion should be negative when price is above MA"
    
    print("✅ Physics signals verified!")

if __name__ == "__main__":
    import flax
    test_physics_signals()
