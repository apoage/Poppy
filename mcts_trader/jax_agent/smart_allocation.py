"""
Smart Environment Data Allocation (TASK_064)

STRATIFIED allocation prevents catastrophic forgetting by interleaving periods.
DISTRIBUTED tiling within each period preserves i.i.d. for stable gradients.

Quick Reference:
    STRATIFIED: P1→P2→P3→P4→P1... (revisit all regimes every 4 epochs)
    DISTRIBUTED: stride = period_length / n_envs (e.g., 12 days for 180/15)
    JITTER: ±6h for data augmentation

    period_idx = epoch % n_periods
    epoch_in_period = epoch // n_periods
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class SmartAllocationConfig:
    """Configuration for smart data allocation."""
    n_envs: int = 15
    n_steps: int = 1440  # 1 day at 1-minute resolution
    jitter_minutes: int = 360  # ±6 hours
    seed: Optional[int] = None


def compute_env_start_offsets(
    period_lengths: List[int],
    n_envs: int,
    n_steps: int,
    epoch: int,
    jitter_minutes: int = 360,
    seed: Optional[int] = None
) -> Tuple[int, List[int]]:
    """
    Compute STRATIFIED start indices (prevents catastrophic forgetting).

    STRATIFIED: Interleave periods across epochs.
      - Epoch 0: Period 0,  Epoch 1: Period 1,  Epoch 2: Period 2,  Epoch 3: Period 3
      - Epoch 4: Period 0,  Epoch 5: Period 1,  ...

    This prevents forgetting by revisiting all regimes regularly.
    Physics still sees ONE regime per epoch (coherent).

    DISTRIBUTED within period: envs are strided apart to ensure i.i.d.
      - Stride = period_length / n_envs
      - For 180-day period with 15 envs: stride = 12 days

    Args:
        period_lengths: List of timestamps per period (may vary due to data gaps)
        n_envs: Number of parallel environments (typically 15)
        n_steps: Steps per episode (1440 = 1 day at 1-minute resolution)
        epoch: Current epoch number (0-59 typically)
        jitter_minutes: Max jitter (±360 = ±6 hours for augmentation)
        seed: Optional seed for reproducibility

    Returns:
        Tuple of (period_idx, List of start indices within that period)

    Example:
        >>> period_lengths = [259200] * 4  # 4 × 180 days
        >>> period_idx, starts = compute_env_start_offsets(
        ...     period_lengths, n_envs=15, n_steps=1440, epoch=0, jitter_minutes=0
        ... )
        >>> period_idx
        0
        >>> len(starts)
        15
    """
    rng = np.random.default_rng(seed)
    n_periods = len(period_lengths)

    if n_periods == 0:
        raise ValueError("period_lengths cannot be empty")

    # STRATIFIED: interleave periods
    period_idx = epoch % n_periods
    epoch_in_period = epoch // n_periods  # 0-14 for each period over 60 epochs

    period_length = period_lengths[period_idx]

    if period_length < n_steps:
        raise ValueError(
            f"Period {period_idx} length ({period_length}) < n_steps ({n_steps})"
        )

    # DISTRIBUTED within period: stride = period_length / n_envs
    stride = period_length // n_envs

    # Epoch progression within period (shift by n_steps each visit)
    epoch_shift = (epoch_in_period * n_steps) % period_length

    # Maximum valid start position
    max_start = period_length - n_steps

    starts = []
    for env_idx in range(n_envs):
        # STRIDED offset within period
        env_offset = env_idx * stride

        # Jitter for data augmentation (±jitter_minutes)
        if jitter_minutes > 0:
            jitter = rng.integers(-jitter_minutes, jitter_minutes + 1)
        else:
            jitter = 0

        # Combine: env stride + epoch shift + jitter
        start = env_offset + epoch_shift + jitter

        # Wrap within period (but careful not to go negative)
        start = start % period_length

        # Ensure room for full episode within period
        if start > max_start:
            start = max_start

        starts.append(start)

    return period_idx, starts


def check_overlap(starts: List[int], n_steps: int) -> float:
    """
    Calculate actual overlap percentage between envs.

    Args:
        starts: List of start indices for each env
        n_steps: Episode length

    Returns:
        Overlap ratio (0.0 = no overlap, 1.0 = complete overlap)
    """
    if len(starts) < 2:
        return 0.0

    ranges = [(s, s + n_steps) for s in starts]
    total_overlap = 0

    for i in range(len(ranges)):
        for j in range(i + 1, len(ranges)):
            # Calculate overlap between ranges[i] and ranges[j]
            overlap_start = max(ranges[i][0], ranges[j][0])
            overlap_end = min(ranges[i][1], ranges[j][1])
            overlap = max(0, overlap_end - overlap_start)
            total_overlap += overlap

    # Maximum possible overlap if all ranges perfectly aligned
    n_pairs = len(starts) * (len(starts) - 1) / 2
    max_possible = n_steps * n_pairs

    return total_overlap / max_possible if max_possible > 0 else 0.0


def compute_coverage(all_starts: List[int], n_steps: int, period_length: int) -> float:
    """
    Calculate what % of period has been touched by training.

    Args:
        all_starts: All start indices used (across multiple epochs)
        n_steps: Episode length
        period_length: Total period length

    Returns:
        Coverage ratio (0.0 = nothing touched, 1.0 = fully covered)
    """
    touched = set()
    for start in all_starts:
        for t in range(start, min(start + n_steps, period_length)):
            touched.add(t)

    return len(touched) / period_length if period_length > 0 else 0.0


def get_allocation_stats(
    period_lengths: List[int],
    n_envs: int,
    n_steps: int,
    n_epochs: int,
    jitter_minutes: int = 360,
    seed: Optional[int] = None
) -> dict:
    """
    Compute statistics for full training run allocation.

    Args:
        period_lengths: List of timestamps per period
        n_envs: Number of parallel environments
        n_steps: Steps per episode
        n_epochs: Total epochs in training run
        jitter_minutes: Max jitter
        seed: Optional seed

    Returns:
        Dictionary with allocation statistics
    """
    n_periods = len(period_lengths)

    # Track visits per period
    period_visits = {i: [] for i in range(n_periods)}

    # Track all starts per period for coverage calculation
    period_starts = {i: [] for i in range(n_periods)}

    for epoch in range(n_epochs):
        period_idx, starts = compute_env_start_offsets(
            period_lengths=period_lengths,
            n_envs=n_envs,
            n_steps=n_steps,
            epoch=epoch,
            jitter_minutes=jitter_minutes,
            seed=seed + epoch if seed else None  # Different seed per epoch
        )
        period_visits[period_idx].append(epoch)
        period_starts[period_idx].extend(starts)

    # Compute coverage per period
    coverages = {}
    for i in range(n_periods):
        coverages[i] = compute_coverage(
            period_starts[i], n_steps, period_lengths[i]
        )

    # Compute average overlap per epoch
    overlaps = []
    for epoch in range(min(n_epochs, 10)):  # Sample first 10 epochs
        _, starts = compute_env_start_offsets(
            period_lengths=period_lengths,
            n_envs=n_envs,
            n_steps=n_steps,
            epoch=epoch,
            jitter_minutes=jitter_minutes,
            seed=seed + epoch if seed else None
        )
        overlaps.append(check_overlap(starts, n_steps))

    return {
        'n_periods': n_periods,
        'n_epochs': n_epochs,
        'visits_per_period': {i: len(v) for i, v in period_visits.items()},
        'coverage_per_period': coverages,
        'avg_coverage': np.mean(list(coverages.values())),
        'avg_overlap': np.mean(overlaps) if overlaps else 0.0,
        'visit_pattern': [epoch % n_periods for epoch in range(min(n_epochs, 8))]
    }


# Convenience function for dojo_v3
def get_period_for_epoch(epoch: int, n_periods: int = 4) -> Tuple[int, int]:
    """
    Get which period to load for given epoch.

    Args:
        epoch: Current epoch number
        n_periods: Number of periods available

    Returns:
        Tuple of (period_idx, epoch_within_period)
    """
    return epoch % n_periods, epoch // n_periods
