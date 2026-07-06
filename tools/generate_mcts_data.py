import jax
import jax.numpy as jnp
import mctx
import time
import numpy as np
from mcts_trader.jax_agent.muzero_model import MuZeroNetworks
from mcts_trader.jax_agent.mctx_search import create_mctx_search

def generate_realistic_data(num_trajectories=100):
    """Generate synthetic data with MCTS policy targets"""
    rng = jax.random.PRNGKey(42)

    print("Initializing model for data generation...")
    # Create model
    model = MuZeroNetworks(
        state_dim=64,
        action_space_size=192,
        support_size=601
    )
    
    rng, init_rng = jax.random.split(rng)
    dummy_obs = jnp.ones((1, 27, 64, 64))
    
    # Initialize params using the 'initialize_parameters' mode we defined
    variables = model.init(init_rng, dummy_obs, mode='initialize_parameters')
    params = variables['params']

    print("Initializing MCTS search...")
    search = create_mctx_search(
        model, 
        num_simulations=16, # Fast search for data gen
        use_gumbel=True
    )

    trajectories = []
    
    print(f"Generating {num_trajectories} trajectories...")
    start_time = time.time()

    for traj_idx in range(num_trajectories):
        rng, traj_rng = jax.random.split(rng)

        # Random initial observation
        obs = jax.random.normal(traj_rng, (27, 64, 64))

        observations = []
        actions = []
        rewards = []
        values = []
        policy_probs = []

        # Generate trajectory
        for step in range(20):  # 20 steps per trajectory
            rng, step_rng = jax.random.split(rng)

            # Run MCTS to get policy
            # search returns (actions, policy_probs, stats)
            # input obs needs batch dim
            action, policy, _ = search(params, step_rng, obs[None])

            # Get value from initial inference
            output = model.apply(variables, obs[None], mode='initial_inference')
            # output['value'] is categorical logits
            
            # Store (remove batch dim)
            observations.append(obs)
            actions.append(action[0])
            policy_probs.append(policy[0])  # Non-uniform from MCTS!

            # Simulate reward (random for now, in real env this comes from env.step)
            reward = jax.random.normal(step_rng) * 0.1
            rewards.append(reward)

            # Convert categorical value to scalar for storage
            from mcts_trader.jax_agent import VALUE_SUPPORT_MIN, VALUE_SUPPORT_MAX, VALUE_SUPPORT_SIZE
            support = jnp.linspace(VALUE_SUPPORT_MIN, VALUE_SUPPORT_MAX, VALUE_SUPPORT_SIZE)
            value_probs = jax.nn.softmax(output['value'][0])
            value_scalar = jnp.sum(value_probs * support)
            values.append(value_scalar)

            # Next observation (random walk simulation)
            obs = obs + jax.random.normal(step_rng, obs.shape) * 0.1

        trajectories.append({
            'observations': jnp.stack(observations),
            'actions': jnp.array(actions),
            'rewards': jnp.array(rewards),
            'values': jnp.array(values),
            'policy_probs': jnp.stack(policy_probs)
        })

        if (traj_idx + 1) % 10 == 0:
            print(f"Generated {traj_idx + 1}/{num_trajectories} trajectories")

    print(f"Generation complete in {time.time() - start_time:.2f}s")
    return trajectories

if __name__ == "__main__":
    print("Generating MCTS-based synthetic data...")
    data = generate_realistic_data(100)

    # Save
    import pickle
    filename = 'realistic_synthetic_data.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

    print(f"✅ Saved {len(data)} trajectories to {filename}")
    
    # Check entropy of first step
    p = data[0]['policy_probs'][0]
    entropy = -jnp.sum(p * jnp.log(p + 1e-8))
    print(f"   Policy entropy sample: {entropy:.3f}")
    print(f"   (Uniform entropy is ~5.26. Value < 5.26 confirms non-uniformity)")