#!/usr/bin/env python3
"""
Diagnostic tool for JAX MuZero training issues.
Checks XLA setup, model initialization, and data quality.
"""

import sys
import os

def check_jax_setup():
    """Verify JAX installation and GPU access"""
    print("="*60)
    print("1. Checking JAX Setup")
    print("="*60)

    try:
        import jax
        import jax.numpy as jnp
        print(f"✅ JAX version: {jax.__version__}")

        # Check devices
        devices = jax.devices()
        print(f"✅ Available devices: {devices}")

        if not any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices):
            print("⚠️  No GPU detected! Training will be slow.")
            return False

        # Test GPU computation
        x = jnp.ones((1000, 1000))
        y = jnp.dot(x, x)
        print(f"✅ GPU computation test passed")

        return True

    except ImportError as e:
        print(f"❌ JAX not installed: {e}")
        return False
    except Exception as e:
        print(f"❌ JAX setup error: {e}")
        return False


def check_dependencies():
    """Check all required packages"""
    print("\n" + "="*60)
    print("2. Checking Dependencies")
    print("="*60)

    packages = {
        'jax': 'Core JAX library',
        'flax': 'Neural network library',
        'optax': 'Optimizers',
        'mctx': 'MCTS library',
        'chex': 'Testing utilities',
        'orbax': 'Checkpointing'
    }

    all_ok = True

    for package, description in packages.items():
        try:
            if package == 'orbax':
                # Orbax has subpackages
                import orbax.checkpoint
                version = orbax.checkpoint.__version__
            else:
                mod = __import__(package)
                version = mod.__version__

            print(f"✅ {package:12s} {version:12s} - {description}")
        except ImportError:
            print(f"❌ {package:12s} {'NOT INSTALLED':12s} - {description}")
            all_ok = False

    return all_ok


def check_xla_config():
    """Check XLA environment variables"""
    print("\n" + "="*60)
    print("3. Checking XLA Configuration")
    print("="*60)

    xla_flags = os.environ.get('XLA_FLAGS', '')

    if not xla_flags:
        print("⚠️  XLA_FLAGS not set")
        print("   Recommendation: Run 'source setup_xla.sh' before training")
        return False

    print(f"✅ XLA_FLAGS configured:")
    for flag in xla_flags.split('--'):
        if flag.strip():
            print(f"   --{flag.strip()}")

    # Check cache directory
    cache_dir = os.environ.get('JAX_COMPILATION_CACHE_DIR', '')
    if cache_dir:
        print(f"✅ Compilation cache: {cache_dir}")
    else:
        print("⚠️  JAX_COMPILATION_CACHE_DIR not set (will recompile each run)")

    return True


def check_model_implementation():
    """Verify model can be initialized"""
    print("\n" + "="*60)
    print("4. Checking Model Implementation")
    print("="*60)

    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from mcts_trader.jax_agent.muzero_model import MuZeroNetworks
        import jax
        import jax.numpy as jnp

        print("Creating model...")
        model = MuZeroNetworks(
            state_dim=64,
            action_space_size=192,
            support_size=601
        )

        # Initialize
        rng = jax.random.PRNGKey(0)
        dummy_obs = jnp.ones((2, 27, 64, 64))

        print("Initializing parameters...")
        variables = model.init(rng, dummy_obs)
        params = variables['params']

        param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
        print(f"✅ Model initialized: {param_count:,} parameters")

        # Test forward pass
        print("Testing forward pass...")
        output = model.apply(variables, dummy_obs)

        print(f"✅ Forward pass successful:")
        print(f"   Latent state shape: {output['latent_state'].shape}")
        print(f"   Policy logits shape: {output['policy_logits'].shape}")
        print(f"   Value shape: {output['value'].shape}")
        print(f"   Reward shape: {output['reward'].shape}")

        # Test recurrent inference
        print("Testing recurrent inference...")
        actions = jnp.array([0, 1])
        recurrent_output = model.apply(
            variables,
            output['latent_state'],
            actions,
            method='recurrent_inference'
        )
        print(f"✅ Recurrent inference successful")

        return True

    except Exception as e:
        print(f"❌ Model check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_mctx_integration():
    """Verify MCTS search works"""
    print("\n" + "="*60)
    print("5. Checking MCTS Integration")
    print("="*60)

    try:
        from mcts_trader.jax_agent.muzero_model import MuZeroNetworks
        from mcts_trader.jax_agent.mctx_search import MuZeroMCTXSearch
        import jax
        import jax.numpy as jnp

        # Create model
        model = MuZeroNetworks()
        rng = jax.random.PRNGKey(0)
        dummy_obs = jnp.ones((1, 27, 64, 64))
        variables = model.init(rng, dummy_obs)
        params = variables['params']

        # Create search
        print("Creating MCTS search...")
        search = MuZeroMCTXSearch(
            model_apply_fn=model.apply,
            num_simulations=8,  # Small for fast test
            max_num_considered_actions=16,
            use_gumbel=True
        )

        # Test search
        print("Running MCTS search (with JIT compilation)...")
        obs = jax.random.normal(rng, (2, 27, 64, 64))

        import time
        start = time.time()
        actions, policy, stats = search(params, rng, obs)
        compile_time = time.time() - start

        print(f"✅ First call (JIT compile): {compile_time:.2f}s")

        # Test again (should be fast)
        start = time.time()
        actions, policy, stats = search(params, rng, obs)
        runtime = time.time() - start

        print(f"✅ Second call (compiled): {runtime*1000:.2f}ms")
        print(f"✅ Actions shape: {actions.shape}")
        print(f"✅ Policy shape: {policy.shape}")

        # Check policy is non-uniform
        entropy = -jnp.sum(policy * jnp.log(policy + 1e-8), axis=-1).mean()
        uniform_entropy = jnp.log(192)

        print(f"\nPolicy entropy: {entropy:.3f}")
        print(f"Uniform entropy: {uniform_entropy:.3f}")

        if entropy < uniform_entropy - 0.5:
            print("✅ Policy is NON-UNIFORM (good!)")
        else:
            print("⚠️  Policy is too uniform (may indicate issue)")

        return True

    except Exception as e:
        print(f"❌ MCTS check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_training_data():
    """Check if training data exists and is valid"""
    print("\n" + "="*60)
    print("6. Checking Training Data")
    print("="*60)

    data_file = 'realistic_synthetic_data.pkl'

    if not os.path.exists(data_file):
        print(f"⚠️  Training data not found: {data_file}")
        print("   Run: python generate_mcts_data.py")
        return False

    try:
        import pickle
        import jax.numpy as jnp

        with open(data_file, 'rb') as f:
            data = pickle.load(f)

        print(f"✅ Found training data: {len(data)} trajectories")

        # Analyze first trajectory
        traj = data[0]
        print(f"\nTrajectory structure:")
        for key, value in traj.items():
            print(f"  {key:20s}: {value.shape}")

        # Check policy entropy
        policies = traj['policy_probs']
        entropies = -jnp.sum(policies * jnp.log(policies + 1e-8), axis=-1)

        print(f"\nPolicy statistics:")
        print(f"  Mean entropy: {entropies.mean():.3f}")
        print(f"  Uniform (bad): 5.257")

        if entropies.mean() < 5.0:
            print("✅ Policies are non-uniform (ready for training)")
        else:
            print("⚠️  Policies are too uniform (regenerate with MCTS)")

        return True

    except Exception as e:
        print(f"❌ Data check failed: {e}")
        return False


def performance_benchmark():
    """Quick performance benchmark"""
    print("\n" + "="*60)
    print("7. Performance Benchmark")
    print("="*60)

    try:
        from mcts_trader.jax_agent.muzero_model import MuZeroNetworks
        import jax
        import jax.numpy as jnp
        import time

        model = MuZeroNetworks()
        rng = jax.random.PRNGKey(0)
        dummy_obs = jnp.ones((1, 27, 64, 64))
        variables = model.init(rng, dummy_obs)

        # Benchmark batch sizes
        batch_sizes = [1, 4, 16, 64]

        print("\nInference speed (initial_inference):")
        for batch_size in batch_sizes:
            obs = jax.random.normal(rng, (batch_size, 27, 64, 64))

            # Warmup
            _ = model.apply(variables, obs)

            # Benchmark
            start = time.time()
            for _ in range(100):
                _ = model.apply(variables, obs)
            elapsed = (time.time() - start) / 100

            throughput = batch_size / elapsed

            print(f"  Batch {batch_size:3d}: {elapsed*1000:6.2f}ms ({throughput:7.1f} obs/sec)")

        return True

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return False


def main():
    print("\n" + "="*70)
    print(" "*15 + "JAX MuZero Training Diagnostics")
    print("="*70)

    results = {
        'JAX Setup': check_jax_setup(),
        'Dependencies': check_dependencies(),
        'XLA Config': check_xla_config(),
        'Model': check_model_implementation(),
        'MCTS': check_mctx_integration(),
        'Training Data': check_training_data(),
        'Performance': performance_benchmark(),
    }

    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)

    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check:20s}: {status}")

    all_passed = all(results.values())

    print("\n" + "="*70)
    if all_passed:
        print("✅ All checks passed! Ready to train.")
        print("\nNext steps:")
        print("  1. source setup_xla.sh")
        print("  2. python train_jax.py")
    else:
        print("⚠️  Some checks failed. Fix issues before training.")
        print("\nCommon fixes:")

        if not results['JAX Setup']:
            print("  - Install JAX: pip install jax[cuda12_pip]")

        if not results['Dependencies']:
            print("  - Install packages: pip install flax optax mctx chex orbax-checkpoint")

        if not results['XLA Config']:
            print("  - Run: source setup_xla.sh")

        if not results['Model'] or not results['MCTS']:
            print("  - Check JAX agent implementation files")

        if not results['Training Data']:
            print("  - Generate data: python generate_mcts_data.py")

    print("="*70)


if __name__ == "__main__":
    main()
