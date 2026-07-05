"""
Adaptive Value Calibration Module (H3)

Auto-tunes value scale and shift to match predicted values to actual rewards.
Extracted from train_jax_fast.py.bak.
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class AdaptiveCalibrator:
    """
    Tracks predicted values vs actual rewards and computes optimal scale/shift.

    The goal is to make value predictions better match actual episode rewards
    by adjusting the MCTS value_scale and value_shift parameters.
    """
    enabled: bool = False
    interval: int = 25  # Recompute every N epochs
    start_epoch: int = 50  # Don't calibrate until this epoch
    max_history: int = 100  # Keep last N data points

    # State
    values: list = field(default_factory=list)
    rewards: list = field(default_factory=list)

    def update(self, avg_value: float, episode_reward: float) -> None:
        """Record a (value, reward) pair."""
        if not self.enabled:
            return

        self.values.append(avg_value)
        self.rewards.append(episode_reward)

        # Trim to max history
        if len(self.values) > self.max_history:
            self.values.pop(0)
            self.rewards.pop(0)

    def should_calibrate(self, epoch: int) -> bool:
        """Check if we should recalibrate this epoch."""
        if not self.enabled:
            return False
        if epoch < self.start_epoch:
            return False
        if (epoch + 1) % self.interval != 0:
            return False
        if len(self.values) < 25:
            return False
        return True

    def compute_calibration(
        self,
        current_scale: float,
        current_shift: float,
    ) -> Tuple[float, float]:
        """
        Compute optimal scale and shift for value calibration.

        Args:
            current_scale: Current MCTS value_scale
            current_shift: Current MCTS value_shift

        Returns:
            (new_scale, new_shift): Updated calibration parameters
        """
        vals = np.array(self.values)
        rews = np.array(self.rewards)

        val_std = np.std(vals)
        rew_std = np.std(rews)

        if val_std < 0.01:  # Avoid division by zero
            return current_scale, current_shift

        # Compute optimal scale: match std of predictions to std of rewards
        optimal_scale = rew_std / val_std

        # Smooth update (don't jump too fast)
        new_scale = 0.7 * current_scale + 0.3 * optimal_scale

        # Clamp to reasonable range
        new_scale = float(np.clip(new_scale, 0.5, 4.0))

        # Compute optimal shift (center alignment)
        val_mean = np.mean(vals) * new_scale
        rew_mean = np.mean(rews)
        optimal_shift = rew_mean - val_mean

        new_shift = float(0.7 * current_shift + 0.3 * optimal_shift)

        return new_scale, new_shift

    def calibrate_search(self, search, epoch: int) -> Optional[Tuple[float, float]]:
        """
        Apply calibration to search object if conditions are met.

        Args:
            search: MCTS search object with value_scale and value_shift attrs
            epoch: Current epoch number

        Returns:
            (new_scale, new_shift) if calibrated, None otherwise
        """
        if not self.should_calibrate(epoch):
            return None

        new_scale, new_shift = self.compute_calibration(
            search.value_scale,
            search.value_shift,
        )

        search.value_scale = new_scale
        search.value_shift = new_shift

        return new_scale, new_shift


def create_adaptive_calibrator(enabled: bool = False) -> AdaptiveCalibrator:
    """Factory function for creating calibrator."""
    calibrator = AdaptiveCalibrator(enabled=enabled)
    if enabled:
        print(f"📐 Adaptive calibration enabled (starts epoch {calibrator.start_epoch}, interval {calibrator.interval})")
    return calibrator
