"""
Unit tests for smart_allocation.py (TASK_064)

Tests verify:
1. STRATIFIED allocation (P1→P2→P3→P4→P1...)
2. DISTRIBUTED tiling (stride=12 days within period)
3. No catastrophic forgetting (all periods revisited regularly)
4. Period coverage after full training run
"""

import pytest
import numpy as np
from mcts_trader.jax_agent.smart_allocation import (
    compute_env_start_offsets,
    check_overlap,
    compute_coverage,
    get_allocation_stats,
    get_period_for_epoch,
    SmartAllocationConfig,
)


class TestStratifiedAllocation:
    """Test that epochs are interleaved across periods (P1→P2→P3→P4→P1...)"""

    def test_period_cycling(self):
        """Verify epochs cycle through periods in order."""
        period_lengths = [259200] * 4  # 4 × 180 days (in minutes)
        n_envs = 15
        n_steps = 1440

        for epoch in range(12):
            period_idx, _ = compute_env_start_offsets(
                period_lengths=period_lengths,
                n_envs=n_envs,
                n_steps=n_steps,
                epoch=epoch,
                jitter_minutes=0,
                seed=42
            )
            expected_period = epoch % 4
            assert period_idx == expected_period, \
                f"Epoch {epoch} should use Period {expected_period}, got {period_idx}"

    def test_epoch_in_period_progression(self):
        """Verify epoch_in_period increments correctly."""
        period_lengths = [259200] * 4
        n_envs = 15
        n_steps = 1440

        # Epochs 0, 4, 8, 12 should all use Period 0
        # with epoch_in_period = 0, 1, 2, 3
        period0_epochs = [0, 4, 8, 12, 16]
        for i, epoch in enumerate(period0_epochs):
            period_idx, _ = compute_env_start_offsets(
                period_lengths=period_lengths,
                n_envs=n_envs,
                n_steps=n_steps,
                epoch=epoch,
                jitter_minutes=0,
                seed=42
            )
            assert period_idx == 0, f"Epoch {epoch} should use Period 0"
            assert get_period_for_epoch(epoch, 4) == (0, i), \
                f"Epoch {epoch} should have epoch_in_period={i}"


class TestDistributedTiling:
    """Test that envs are strided apart within each period."""

    def test_stride_spacing(self):
        """Verify stride = period_length / n_envs."""
        period_lengths = [259200] * 4  # 180 days in minutes
        n_envs = 15
        n_steps = 1440

        period_idx, starts = compute_env_start_offsets(
            period_lengths=period_lengths,
            n_envs=n_envs,
            n_steps=n_steps,
            epoch=0,
            jitter_minutes=0,
            seed=42
        )

        # Expected stride: 259200 / 15 = 17280 minutes = 12 days
        expected_stride = 259200 // 15

        # Verify spacing (without jitter, should be exact)
        for i in range(len(starts) - 1):
            gap = starts[i + 1] - starts[i]
            assert gap == expected_stride, \
                f"Gap between env {i} and {i+1} should be {expected_stride}, got {gap}"

    def test_stride_12_days(self):
        """Verify 12-day spacing in day units."""
        period_lengths = [259200] * 4
        n_envs = 15
        n_steps = 1440  # 1 day

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

        for i in range(len(days) - 1):
            gap = days[i + 1] - days[i]
            assert gap == 12, f"Gap between env {i} and {i+1} should be 12 days, got {gap}"


class TestNoForgetting:
    """Test that model revisits all periods regularly."""

    def test_all_periods_visited_equally(self):
        """Verify each period visited 15 times over 60 epochs."""
        period_lengths = [259200] * 4
        n_envs = 15
        n_steps = 1440

        period_visits = {0: 0, 1: 0, 2: 0, 3: 0}

        for epoch in range(60):
            period_idx, _ = compute_env_start_offsets(
                period_lengths=period_lengths,
                n_envs=n_envs,
                n_steps=n_steps,
                epoch=epoch,
                jitter_minutes=0,
                seed=42
            )
            period_visits[period_idx] += 1

        # Each period should be visited 15 times
        for p, count in period_visits.items():
            assert count == 15, f"Period {p} visited {count} times, expected 15"

    def test_max_gap_between_visits(self):
        """Verify max gap between visiting same period is 4 epochs."""
        period_lengths = [259200] * 4
        n_envs = 15
        n_steps = 1440

        period_epochs = {0: [], 1: [], 2: [], 3: []}

        for epoch in range(60):
            period_idx, _ = compute_env_start_offsets(
                period_lengths=period_lengths,
                n_envs=n_envs,
                n_steps=n_steps,
                epoch=epoch,
                jitter_minutes=0,
                seed=42
            )
            period_epochs[period_idx].append(epoch)

        # Max gap should be exactly 4
        for p, epochs in period_epochs.items():
            for i in range(len(epochs) - 1):
                gap = epochs[i + 1] - epochs[i]
                assert gap == 4, f"Period {p} gap between epochs: {gap}, expected 4"


class TestCoverage:
    """Test period coverage after full training run."""

    def test_high_coverage_after_15_visits(self):
        """Verify >95% coverage of each period after full training."""
        period_lengths = [259200] * 4
        n_steps = 1440
        n_envs = 15

        # Collect all starts for Period 0 (epochs 0, 4, 8, ...)
        period0_starts = []
        for epoch in range(0, 60, 4):  # Period 0 epochs
            _, starts = compute_env_start_offsets(
                period_lengths=period_lengths,
                n_envs=n_envs,
                n_steps=n_steps,
                epoch=epoch,
                jitter_minutes=0,
                seed=42
            )
            period0_starts.extend(starts)

        coverage = compute_coverage(period0_starts, n_steps, period_lengths[0])
        assert coverage > 0.95, f"Period 0 coverage: {coverage:.1%}, expected >95%"

    def test_coverage_function(self):
        """Test compute_coverage basic functionality."""
        # Simple case: 3 non-overlapping segments
        starts = [0, 100, 200]
        n_steps = 50
        period_length = 300

        coverage = compute_coverage(starts, n_steps, period_length)
        expected = 150 / 300  # 3 * 50 = 150 touched
        assert abs(coverage - expected) < 0.01, f"Coverage {coverage} != expected {expected}"


class TestOverlap:
    """Test overlap calculation."""

    def test_no_overlap_with_stride(self):
        """Verify distributed tiling has minimal overlap."""
        period_lengths = [259200] * 4
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

        overlap = check_overlap(starts, n_steps)
        # With stride=12 days and episode=1 day, no overlap expected
        assert overlap == 0.0, f"Expected no overlap, got {overlap}"

    def test_overlap_with_close_starts(self):
        """Verify overlap detected when starts are close."""
        starts = [0, 500, 1000]  # 500 apart
        n_steps = 600  # Each episode 600 long

        overlap = check_overlap(starts, n_steps)
        # Overlaps: [0,600] & [500,1100] overlap by 100
        #           [500,1100] & [1000,1600] overlap by 100
        # No overlap between [0,600] and [1000,1600]
        # Total overlap: 200, max possible: 600 * 3 = 1800
        assert overlap > 0, "Expected some overlap"


class TestJitter:
    """Test jitter functionality."""

    def test_jitter_within_bounds(self):
        """Verify jitter stays within ±jitter_minutes (accounting for wrap/clamp)."""
        period_lengths = [259200] * 4
        n_envs = 15
        n_steps = 1440
        jitter_minutes = 360

        # With no jitter, all starts should be deterministic for same seed
        _, starts1 = compute_env_start_offsets(
            period_lengths=period_lengths,
            n_envs=n_envs,
            n_steps=n_steps,
            epoch=0,
            jitter_minutes=0,
            seed=42
        )

        _, starts2 = compute_env_start_offsets(
            period_lengths=period_lengths,
            n_envs=n_envs,
            n_steps=n_steps,
            epoch=0,
            jitter_minutes=0,
            seed=42
        )

        # Same seed, no jitter = identical
        assert starts1 == starts2, "Deterministic without jitter"

        # With jitter, all starts should still be valid
        _, starts_jittered = compute_env_start_offsets(
            period_lengths=period_lengths,
            n_envs=n_envs,
            n_steps=n_steps,
            epoch=0,
            jitter_minutes=jitter_minutes,
            seed=123
        )

        max_start = period_lengths[0] - n_steps
        for s in starts_jittered:
            assert 0 <= s <= max_start, f"Start {s} out of valid range [0, {max_start}]"

    def test_jitter_varies_with_seed(self):
        """Verify different seeds produce different jitter."""
        period_lengths = [259200] * 4
        n_envs = 15
        n_steps = 1440

        _, starts1 = compute_env_start_offsets(
            period_lengths=period_lengths,
            n_envs=n_envs,
            n_steps=n_steps,
            epoch=0,
            jitter_minutes=360,
            seed=1
        )

        _, starts2 = compute_env_start_offsets(
            period_lengths=period_lengths,
            n_envs=n_envs,
            n_steps=n_steps,
            epoch=0,
            jitter_minutes=360,
            seed=2
        )

        # Should not be identical
        assert starts1 != starts2, "Different seeds should produce different jitter"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_period(self):
        """Test with only 1 period available."""
        period_lengths = [259200]  # Single period
        n_envs = 15
        n_steps = 1440

        for epoch in range(10):
            period_idx, starts = compute_env_start_offsets(
                period_lengths=period_lengths,
                n_envs=n_envs,
                n_steps=n_steps,
                epoch=epoch,
                jitter_minutes=0,
                seed=42
            )
            assert period_idx == 0, "Single period should always return period 0"
            assert len(starts) == n_envs

    def test_empty_periods_raises(self):
        """Test empty period list raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_env_start_offsets(
                period_lengths=[],
                n_envs=15,
                n_steps=1440,
                epoch=0
            )

    def test_period_shorter_than_episode_raises(self):
        """Test period shorter than n_steps raises error."""
        with pytest.raises(ValueError, match="< n_steps"):
            compute_env_start_offsets(
                period_lengths=[1000],  # 1000 < 1440
                n_envs=15,
                n_steps=1440,
                epoch=0
            )

    def test_unequal_period_lengths(self):
        """Test with varying period lengths (data gaps)."""
        period_lengths = [259200, 250000, 260000, 255000]  # Varying
        n_envs = 15
        n_steps = 1440

        for epoch in range(8):
            period_idx, starts = compute_env_start_offsets(
                period_lengths=period_lengths,
                n_envs=n_envs,
                n_steps=n_steps,
                epoch=epoch,
                jitter_minutes=0,
                seed=42
            )

            # All starts should be valid for the selected period
            max_start = period_lengths[period_idx] - n_steps
            for s in starts:
                assert 0 <= s <= max_start, \
                    f"Start {s} invalid for period {period_idx} (max {max_start})"


class TestAllocationStats:
    """Test get_allocation_stats function."""

    def test_stats_structure(self):
        """Test stats returns expected structure."""
        stats = get_allocation_stats(
            period_lengths=[259200] * 4,
            n_envs=15,
            n_steps=1440,
            n_epochs=60,
            jitter_minutes=360,
            seed=42
        )

        assert 'n_periods' in stats
        assert 'n_epochs' in stats
        assert 'visits_per_period' in stats
        assert 'coverage_per_period' in stats
        assert 'avg_coverage' in stats
        assert 'avg_overlap' in stats
        assert 'visit_pattern' in stats

        assert stats['n_periods'] == 4
        assert stats['n_epochs'] == 60

    def test_stats_visit_balance(self):
        """Test stats shows balanced visits."""
        stats = get_allocation_stats(
            period_lengths=[259200] * 4,
            n_envs=15,
            n_steps=1440,
            n_epochs=60,
            jitter_minutes=0,
            seed=42
        )

        # Should have 15 visits per period
        for p, count in stats['visits_per_period'].items():
            assert count == 15, f"Period {p} has {count} visits, expected 15"


class TestGetPeriodForEpoch:
    """Test convenience function."""

    def test_period_calculation(self):
        """Test get_period_for_epoch returns correct values."""
        assert get_period_for_epoch(0, 4) == (0, 0)
        assert get_period_for_epoch(1, 4) == (1, 0)
        assert get_period_for_epoch(2, 4) == (2, 0)
        assert get_period_for_epoch(3, 4) == (3, 0)
        assert get_period_for_epoch(4, 4) == (0, 1)
        assert get_period_for_epoch(5, 4) == (1, 1)
        assert get_period_for_epoch(59, 4) == (3, 14)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
