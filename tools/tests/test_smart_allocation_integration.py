"""
Integration tests for TASK_064: Smart Environment Data Allocation

Tests verify:
1. Stratified allocation prevents catastrophic forgetting
2. Distributed tiling preserves i.i.d. within epoch
3. Shared prices work correctly across parallel envs
4. VectorBTTradingEnv accepts start_offset
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mcts_trader.jax_agent.smart_allocation import (
    compute_env_start_offsets,
    get_allocation_stats,
    get_period_for_epoch,
)


class TestNoForgettingIntegration:
    """Verify stratified allocation prevents catastrophic forgetting."""

    def test_all_periods_revisited_within_window(self):
        """
        Key anti-forgetting property: No period goes more than N epochs without being visited.

        For 4 periods, max gap should be exactly 4 epochs (P0, P1, P2, P3, P0...).
        This prevents the model from "forgetting" a regime during long training.
        """
        period_lengths = [259200] * 4  # 4 × 180 days
        n_envs = 15
        n_steps = 1440
        n_epochs = 60

        # Track epochs when each period is visited
        period_epochs = {0: [], 1: [], 2: [], 3: []}

        for epoch in range(n_epochs):
            period_idx, _ = compute_env_start_offsets(
                period_lengths=period_lengths,
                n_envs=n_envs,
                n_steps=n_steps,
                epoch=epoch,
                jitter_minutes=0,
                seed=42
            )
            period_epochs[period_idx].append(epoch)

        # Check that gaps between visits are exactly 4
        for p, epochs in period_epochs.items():
            for i in range(1, len(epochs)):
                gap = epochs[i] - epochs[i-1]
                assert gap == 4, f"Period {p} gap between visits: {gap}, expected exactly 4"

    def test_balanced_visits_across_training(self):
        """Verify each period gets equal training time."""
        period_lengths = [259200] * 4
        n_epochs = 60

        visit_counts = {0: 0, 1: 0, 2: 0, 3: 0}

        for epoch in range(n_epochs):
            period_idx, _ = compute_env_start_offsets(
                period_lengths=period_lengths,
                n_envs=15,
                n_steps=1440,
                epoch=epoch,
                jitter_minutes=0,
                seed=42
            )
            visit_counts[period_idx] += 1

        # Each period should get exactly 15 visits (60/4)
        for p, count in visit_counts.items():
            assert count == 15, f"Period {p} visited {count} times, expected 15"

    def test_period_sequence_is_deterministic(self):
        """Verify same seed produces same allocation sequence."""
        period_lengths = [259200] * 4

        sequence1 = []
        sequence2 = []

        for epoch in range(20):
            p1, _ = compute_env_start_offsets(
                period_lengths, 15, 1440, epoch, jitter_minutes=0, seed=42
            )
            p2, _ = compute_env_start_offsets(
                period_lengths, 15, 1440, epoch, jitter_minutes=0, seed=42
            )
            sequence1.append(p1)
            sequence2.append(p2)

        assert sequence1 == sequence2, "Allocation should be deterministic with same seed"
        assert sequence1 == [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], \
            "Should cycle through periods in order"


class TestIIDPreservation:
    """Verify distributed tiling preserves i.i.d. within each epoch."""

    def test_no_overlap_between_envs(self):
        """Envs should not overlap within same epoch (i.i.d. requirement)."""
        period_lengths = [259200] * 4
        n_envs = 15
        n_steps = 1440  # 1 day

        for epoch in range(10):
            _, starts = compute_env_start_offsets(
                period_lengths=period_lengths,
                n_envs=n_envs,
                n_steps=n_steps,
                epoch=epoch,
                jitter_minutes=0,
                seed=42
            )

            # Check no overlap between any pair of envs
            for i in range(n_envs):
                for j in range(i + 1, n_envs):
                    range_i = (starts[i], starts[i] + n_steps)
                    range_j = (starts[j], starts[j] + n_steps)

                    overlap = max(0, min(range_i[1], range_j[1]) - max(range_i[0], range_j[0]))
                    assert overlap == 0, \
                        f"Epoch {epoch}: Env {i} and {j} overlap by {overlap} steps"

    def test_envs_sample_different_market_conditions(self):
        """Envs should see different market conditions (stride apart)."""
        period_lengths = [259200]  # Single period
        n_envs = 15
        n_steps = 1440

        _, starts = compute_env_start_offsets(
            period_lengths=period_lengths,
            n_envs=n_envs,
            n_steps=n_steps,
            epoch=0,
            jitter_minutes=0,
            seed=42
        )

        # Convert to days
        days = [s // n_steps for s in starts]

        # Should span most of the 180-day period
        coverage = max(days) - min(days)
        assert coverage >= 160, f"Envs only span {coverage} days, expected >= 160"


class TestVectorBTEnvIntegration:
    """Test VectorBTTradingEnv integration with smart allocation."""

    def test_reset_with_start_offset(self):
        """Test that VectorBTTradingEnv accepts start_offset parameter."""
        from mcts_trader.env.vectorbt_env import VectorBTTradingEnv

        # Create env with synthetic data (simpler for testing)
        env = VectorBTTradingEnv(
            synthetic_level=0,
            n_assets=4,
            initial_cash=10000.0,
            max_steps=100,
            synthetic_n_steps=1000,
        )

        # Reset with specific offset
        obs, info = env.reset(start_offset=500)

        # Verify offset was applied (clamped to valid range)
        assert env.episode_start_idx >= env.lookback, "Start should be >= lookback"
        assert env.episode_start_idx <= len(env.prices) - env.max_steps, "Start should leave room for episode"

    def test_reset_without_offset_uses_random(self):
        """Test that reset without offset uses random start."""
        from mcts_trader.env.vectorbt_env import VectorBTTradingEnv

        env = VectorBTTradingEnv(
            synthetic_level=0,
            n_assets=4,
            initial_cash=10000.0,
            max_steps=100,
            synthetic_n_steps=1000,
        )

        # Reset multiple times, should get different starts
        starts = []
        for i in range(5):
            env.reset(seed=i)
            starts.append(env.episode_start_idx)

        # Should have some variation
        assert len(set(starts)) > 1, "Random resets should produce different start positions"

    def test_shared_prices_across_envs(self):
        """Test that envs can share prices array without issues."""
        from mcts_trader.env.vectorbt_env import VectorBTTradingEnv

        # Create first env
        env1 = VectorBTTradingEnv(
            synthetic_level=0,
            n_assets=4,
            initial_cash=10000.0,
            max_steps=100,
            synthetic_n_steps=1000,
        )
        env1.reset()

        # Extract prices
        shared_prices = env1.prices

        # Create second env with shared prices
        env2 = VectorBTTradingEnv(
            prices=shared_prices,
            n_assets=4,
            initial_cash=10000.0,
            max_steps=100,
        )
        env2.reset()

        # Both should reference same array
        assert env1.prices is env2.prices, "Envs should share same prices array"

        # Both should work independently
        obs1, _ = env1.reset(start_offset=100)
        obs2, _ = env2.reset(start_offset=500)

        assert env1.episode_start_idx != env2.episode_start_idx, \
            "Different offsets should produce different starts"


class TestPhysicsCoherence:
    """Verify physics model sees coherent data within each epoch."""

    def test_single_period_per_epoch(self):
        """Each epoch should use exactly one period (physics coherence)."""
        period_lengths = [259200] * 4
        n_envs = 15
        n_steps = 1440

        for epoch in range(60):
            period_idx, starts = compute_env_start_offsets(
                period_lengths=period_lengths,
                n_envs=n_envs,
                n_steps=n_steps,
                epoch=epoch,
                jitter_minutes=0,
                seed=42
            )

            # All envs should use same period
            # (implicitly true since function returns single period_idx)
            assert isinstance(period_idx, int), "Should return single period index"
            assert 0 <= period_idx < 4, f"Period index {period_idx} out of range"

    def test_starts_within_period_bounds(self):
        """All start offsets should be within selected period."""
        period_lengths = [200000, 250000, 230000, 280000]  # Varying lengths
        n_envs = 15
        n_steps = 1440

        for epoch in range(20):
            period_idx, starts = compute_env_start_offsets(
                period_lengths=period_lengths,
                n_envs=n_envs,
                n_steps=n_steps,
                epoch=epoch,
                jitter_minutes=0,
                seed=42
            )

            period_length = period_lengths[period_idx]
            max_valid_start = period_length - n_steps

            for i, start in enumerate(starts):
                assert 0 <= start <= max_valid_start, \
                    f"Epoch {epoch}, Env {i}: start {start} invalid for period {period_idx} (max {max_valid_start})"


class TestAllocationStats:
    """Test allocation statistics for monitoring."""

    def test_stats_show_balanced_coverage(self):
        """Stats should show balanced coverage across periods."""
        stats = get_allocation_stats(
            period_lengths=[259200] * 4,
            n_envs=15,
            n_steps=1440,
            n_epochs=60,
            jitter_minutes=0,
            seed=42
        )

        # All periods should have same visit count
        visits = list(stats['visits_per_period'].values())
        assert all(v == 15 for v in visits), f"Unbalanced visits: {visits}"

        # Average coverage should be high
        assert stats['avg_coverage'] > 0.9, f"Low coverage: {stats['avg_coverage']}"

        # Average overlap should be low (distributed tiling)
        assert stats['avg_overlap'] < 0.1, f"High overlap: {stats['avg_overlap']}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
