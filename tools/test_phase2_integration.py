#!/usr/bin/env python3
"""
TASK_125 Phase 2: Integration Tests for BC Loss and Callbacks.

Tests:
1. ExpertBCConfig initialization
2. BC loss functions (large margin, Q-filter)
3. Episode completion callbacks in self_play
4. DS24 metrics logging
"""

import sys
import numpy as np
import jax
import jax.numpy as jnp
import optax

# Ensure path
sys.path.insert(0, '/home/lukas/Trader')


def test_bc_config():
    """Test ExpertBCConfig dataclass."""
    print("=" * 60)
    print("TEST: ExpertBCConfig")
    print("=" * 60)

    from mcts_trader.jax_agent.trainer import ExpertBCConfig

    # Default config
    config = ExpertBCConfig()
    assert config.enabled == False, "Default should be disabled"
    assert config.bc_weight_initial == 0.1
    assert config.bc_weight_final == 0.01
    assert config.use_q_filter == True
    assert config.use_large_margin == True
    print("  Default config: ✓")

    # Enabled config
    config = ExpertBCConfig(enabled=True, bc_weight_initial=0.2)
    assert config.enabled == True
    assert config.bc_weight_initial == 0.2
    print("  Custom config: ✓")

    print("\n  ExpertBCConfig: ALL TESTS PASSED ✓")
    return True


def test_bc_weight_annealing():
    """Test BC weight decay function."""
    print("\n" + "=" * 60)
    print("TEST: BC Weight Annealing")
    print("=" * 60)

    from mcts_trader.jax_agent.trainer import ExpertBCConfig, compute_bc_weight

    config = ExpertBCConfig(
        bc_weight_initial=0.1,
        bc_weight_final=0.01,
        bc_weight_decay_steps=1000
    )

    # At step 0, should be initial weight
    w0 = compute_bc_weight(0, config)
    assert abs(float(w0) - 0.1) < 1e-5, f"Step 0 weight should be 0.1, got {w0}"
    print(f"  Step 0: {float(w0):.4f} ✓")

    # At step 500, should be midway
    w500 = compute_bc_weight(500, config)
    expected = 0.1 + 0.5 * (0.01 - 0.1)  # 0.055
    assert abs(float(w500) - expected) < 1e-5, f"Step 500 weight should be {expected}, got {w500}"
    print(f"  Step 500: {float(w500):.4f} ✓")

    # At step 1000+, should be final weight
    w1000 = compute_bc_weight(1000, config)
    assert abs(float(w1000) - 0.01) < 1e-5, f"Step 1000 weight should be 0.01, got {w1000}"
    print(f"  Step 1000: {float(w1000):.4f} ✓")

    print("\n  BC Weight Annealing: ALL TESTS PASSED ✓")
    return True


def test_large_margin_loss():
    """Test large margin classification loss."""
    print("\n" + "=" * 60)
    print("TEST: Large Margin Loss")
    print("=" * 60)

    from mcts_trader.jax_agent.trainer import compute_large_margin_loss

    # Create test Q-values
    batch_size = 4
    n_actions = 320  # 5 actions x 64 assets

    # Q-values where expert action has highest Q
    q_values = jnp.zeros((batch_size, n_actions))
    expert_actions = jnp.array([10, 50, 100, 200])  # Expert chose these actions

    # Set expert actions to have highest Q
    for i, a in enumerate(expert_actions):
        q_values = q_values.at[i, a].set(1.0)

    # Compute loss - should be 0 since expert is already best
    loss = compute_large_margin_loss(q_values, expert_actions, margin=0.8)
    assert loss.shape == (batch_size,), f"Loss shape should be (4,), got {loss.shape}"
    # When expert Q > all others + margin, loss should be 0
    print(f"  Loss shape: {loss.shape} ✓")
    print(f"  Loss when expert is best: {float(jnp.mean(loss)):.4f}")

    # Now test when expert action is NOT best
    q_values2 = jnp.ones((batch_size, n_actions))  # All Q = 1.0
    q_values2 = q_values2.at[:, 0].set(2.0)  # Action 0 has Q = 2.0
    expert_actions2 = jnp.array([10, 50, 100, 200])  # Expert didn't choose action 0

    loss2 = compute_large_margin_loss(q_values2, expert_actions2, margin=0.8)
    # Expert Q = 1.0, max(Q + margin) = 2.0 + 0.8 = 2.8 (for action 0)
    # Loss = max(0, 2.8 - 1.0) = 1.8
    expected_loss = 2.0 + 0.8 - 1.0  # 1.8
    assert float(jnp.mean(loss2)) > 1.0, f"Loss should be > 1.0 when expert is suboptimal, got {float(jnp.mean(loss2))}"
    print(f"  Loss when expert is suboptimal: {float(jnp.mean(loss2)):.4f} ✓")

    print("\n  Large Margin Loss: ALL TESTS PASSED ✓")
    return True


def test_q_filtered_bc_loss():
    """Test Q-filtered BC loss."""
    print("\n" + "=" * 60)
    print("TEST: Q-Filtered BC Loss")
    print("=" * 60)

    from mcts_trader.jax_agent.trainer import compute_q_filtered_bc_loss

    batch_size = 4
    n_actions = 320

    # Create policy logits (higher = more likely)
    policy_logits = jnp.zeros((batch_size, n_actions))
    policy_logits = policy_logits.at[:, 0].set(5.0)  # Policy prefers action 0

    expert_actions = jnp.array([10, 0, 100, 0])  # 2 match, 2 don't
    policy_actions = jnp.array([0, 0, 0, 0])  # Policy always chooses 0

    # Q-values: expert sometimes better, sometimes worse
    q_values = jnp.zeros((batch_size, n_actions))
    q_values = q_values.at[0, 10].set(2.0)  # Expert action better (Q=2 > Q=0)
    q_values = q_values.at[1, 0].set(1.0)   # Same action
    q_values = q_values.at[2, 100].set(3.0) # Expert action better
    q_values = q_values.at[3, 0].set(1.0)   # Same action

    loss, mask = compute_q_filtered_bc_loss(
        policy_logits, expert_actions, policy_actions, q_values
    )

    assert loss.shape == (batch_size,), f"Loss shape should be (4,), got {loss.shape}"
    assert mask.shape == (batch_size,), f"Mask shape should be (4,), got {mask.shape}"
    print(f"  Loss shape: {loss.shape} ✓")
    print(f"  Mask shape: {mask.shape} ✓")

    # Mask should be 1 when Q(expert) > Q(policy)
    # Sample 0: Q(10)=2.0 > Q(0)=0.0 -> mask=1
    # Sample 1: Q(0)=1.0 > Q(0)=1.0 -> mask=0 (not strictly greater)
    # Sample 2: Q(100)=3.0 > Q(0)=0.0 -> mask=1
    # Sample 3: Q(0)=1.0 > Q(0)=1.0 -> mask=0
    print(f"  Q-filter mask: {mask}")
    assert float(mask[0]) == 1.0, "Sample 0 should pass Q-filter"
    assert float(mask[2]) == 1.0, "Sample 2 should pass Q-filter"
    print(f"  Q-filter logic: ✓")

    # Loss should be masked (0 where mask=0)
    assert float(loss[1]) == 0.0, "Sample 1 loss should be 0 (masked)"
    assert float(loss[3]) == 0.0, "Sample 3 loss should be 0 (masked)"
    print(f"  Masked loss application: ✓")

    print("\n  Q-Filtered BC Loss: ALL TESTS PASSED ✓")
    return True


def test_ds24_expert_columns():
    """Test DS24 logger has expert buffer columns."""
    print("\n" + "=" * 60)
    print("TEST: DS24 Expert Buffer Columns")
    print("=" * 60)

    from tools.DS24.metrics_logger import DS24MetricsLogger

    logger = DS24MetricsLogger(level=2)
    columns = logger.get_ds24_columns()

    required_columns = [
        'bc_loss', 'bc_weight', 'expert_acceptance_rate',
        'expert_demo_count', 'expert_agent_count', 'expert_buffer_utilization'
    ]

    for col in required_columns:
        assert col in columns, f"Missing column: {col}"
        print(f"  {col}: ✓")

    # Test logging method
    logger.reset_row()
    logger.log_expert_buffer(
        bc_loss=0.5,
        bc_weight=0.08,
        expert_acceptance_rate=0.75,
        demo_count=1000,
        agent_count=5000,
        buffer_capacity=10000
    )

    values = logger.get_current_values()
    assert values.get('bc_loss') == 0.5
    assert values.get('bc_weight') == 0.08
    assert values.get('expert_acceptance_rate') == 0.75
    assert values.get('expert_demo_count') == 1000
    assert values.get('expert_agent_count') == 5000
    assert abs(values.get('expert_buffer_utilization') - 0.6) < 1e-5
    print(f"  Logging method: ✓")

    print("\n  DS24 Expert Buffer Columns: ALL TESTS PASSED ✓")
    return True


def test_self_play_callback_signature():
    """Test that self_play functions accept the callback parameter."""
    print("\n" + "=" * 60)
    print("TEST: Self-Play Callback Signature")
    print("=" * 60)

    import inspect
    from mcts_trader.training.self_play import (
        run_self_play,
        run_self_play_batched,
        run_self_play_jax
    )

    # Check all three functions have the callback parameter
    for func_name, func in [
        ('run_self_play', run_self_play),
        ('run_self_play_batched', run_self_play_batched),
        ('run_self_play_jax', run_self_play_jax)
    ]:
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        assert 'on_episode_complete' in params, f"{func_name} missing on_episode_complete parameter"
        print(f"  {func_name}: ✓")

    print("\n  Self-Play Callback Signature: ALL TESTS PASSED ✓")
    return True


def test_trainer_bc_config_integration():
    """Test MuZeroTrainer accepts ExpertBCConfig."""
    print("\n" + "=" * 60)
    print("TEST: Trainer BC Config Integration")
    print("=" * 60)

    from mcts_trader.jax_agent.trainer import MuZeroTrainer, ExpertBCConfig
    from mcts_trader.jax_agent.muzero_model import MuZeroNetworks

    # Create minimal model for testing
    model = MuZeroNetworks(
        action_space_size=320,
        state_dim=256,
        n_assets=64,
        use_prediction_head=False,
    )

    # Create trainer with BC config
    bc_config = ExpertBCConfig(
        enabled=True,
        bc_weight_initial=0.1,
        bc_weight_final=0.01,
    )

    trainer = MuZeroTrainer(
        model=model,
        expert_bc_config=bc_config,
    )

    assert trainer.expert_bc_config.enabled == True
    assert trainer.expert_bc_config.bc_weight_initial == 0.1
    print(f"  ExpertBCConfig attached to trainer: ✓")

    # Check default (disabled) config
    trainer2 = MuZeroTrainer(model=model)
    assert trainer2.expert_bc_config.enabled == False
    print(f"  Default config (disabled): ✓")

    print("\n  Trainer BC Config Integration: ALL TESTS PASSED ✓")
    return True


def main():
    print("=" * 60)
    print("TASK_125 Phase 2: Integration Tests")
    print("=" * 60)

    results = {}

    results['ExpertBCConfig'] = test_bc_config()
    results['BC Weight Annealing'] = test_bc_weight_annealing()
    results['Large Margin Loss'] = test_large_margin_loss()
    results['Q-Filtered BC Loss'] = test_q_filtered_bc_loss()
    results['DS24 Expert Columns'] = test_ds24_expert_columns()
    results['Self-Play Callback'] = test_self_play_callback_signature()
    results['Trainer BC Config'] = test_trainer_bc_config_integration()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
        sys.exit(1)


if __name__ == '__main__':
    main()
