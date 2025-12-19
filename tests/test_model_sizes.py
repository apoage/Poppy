import jax
import jax.numpy as jnp
from mcts_trader.jax_agent.muzero_model import MuZeroNetworks

def test_model_sizes():
    """Verify each size has expected parameter count"""

    configs = {
        'small': {'state_dim': 64, 'expected': (6.0e6, 8.0e6)},    # ~7M ± 1M
        'medium': {'state_dim': 96, 'expected': (14.0e6, 17.0e6)}, # ~15M ± 1M
        'large': {'state_dim': 128, 'expected': (24.0e6, 28.0e6)}  # ~25M ± 2M
    }

    n_assets = 64
    action_space_size = n_assets * 3
    support_size = 601

    print(f"{'Size':<10} | {'State Dim':<10} | {'Params (M)':<12} | {'Status'}")
    print("-" * 50)

    for size_name, config in configs.items():
        state_dim = config['state_dim']
        min_p, max_p = config['expected']

        model = MuZeroNetworks(
            state_dim=state_dim,
            action_space_size=action_space_size,
            support_size=support_size
        )

        # Initialize
        rng = jax.random.PRNGKey(42)
        dummy_obs = jnp.ones((1, 27, 64, 64))
        
        # Need mode='initialize_parameters' or manually run paths
        variables = model.init(rng, observation=dummy_obs, mode='initialize_parameters')

        # Count params
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(variables['params']))
        param_count_m = param_count / 1e6

        status = "✅ PASS" if min_p <= param_count <= max_p else "❌ FAIL"
        
        print(f"{size_name:<10} | {state_dim:<10} | {param_count_m:<12.2f} | {status}")
        
        if status == "❌ FAIL":
            print(f"   Expected: {min_p/1e6:.1f}M - {max_p/1e6:.1f}M")

if __name__ == "__main__":
    test_model_sizes()
