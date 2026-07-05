"""
TASK_058: Adaptive Difficulty with Multi-Modal Noise

Adjusts noise levels based on reward to keep model challenged but not overwhelmed.
Target reward range: 10-20 (configurable)

Noise types (applied in priority order):
1. Price noise - Random walk + per-step perturbations
2. Temporal jitter - Observation timing shifts
3. Time dilation - Pattern speed variation
4. Phase shift - Mid-episode pattern phase changes
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional
import logging


@dataclass
class NoiseConfig:
    """Configuration for all noise types."""
    price_noise: float = 0.0        # 0.0-1.0, scales price perturbations
    temporal_jitter: int = 0        # Max steps to shift observations
    time_dilation: float = 0.0      # 0.0-0.3, max stretch/compress factor
    phase_shift_prob: float = 0.0   # 0.0-0.5, probability of mid-episode shift

    def to_dict(self) -> Dict[str, float]:
        return {
            'price_noise': self.price_noise,
            'temporal_jitter': float(self.temporal_jitter),
            'time_dilation': self.time_dilation,
            'phase_shift_prob': self.phase_shift_prob,
        }

    def __str__(self) -> str:
        return (f"NoiseConfig(price={self.price_noise:.2f}, "
                f"jitter={self.temporal_jitter}, "
                f"dilation={self.time_dilation:.2f}, "
                f"phase_shift={self.phase_shift_prob:.2f})")


class AdaptiveNoiseController:
    """
    Dynamically adjusts noise levels to maintain target reward range.

    When reward is too high (model too comfortable):
      → Increase difficulty progressively
    When reward is too low (model struggling):
      → Decrease difficulty progressively

    Noise types are added/removed in priority order to create smooth difficulty curve.
    """

    def __init__(
        self,
        target_reward_low: float = 10.0,
        target_reward_high: float = 20.0,
        adjustment_rate: float = 0.05,
        ema_alpha: float = 0.1,
        # Max values for each noise type
        max_price_noise: float = 1.0,
        max_temporal_jitter: int = 5,
        max_time_dilation: float = 0.3,
        max_phase_shift: float = 0.5,
    ):
        """
        Args:
            target_reward_low: Lower bound of comfortable zone (increase difficulty above this)
            target_reward_high: Upper bound of comfortable zone (decrease difficulty below this)
            adjustment_rate: How fast noise changes per update (0.01-0.1)
            ema_alpha: Smoothing factor for reward tracking (0.05-0.2)
            max_*: Maximum values for each noise type
        """
        self.target_low = target_reward_low
        self.target_high = target_reward_high
        self.adjustment_rate = adjustment_rate
        self.ema_alpha = ema_alpha

        # Limits
        self.max_price_noise = max_price_noise
        self.max_temporal_jitter = max_temporal_jitter
        self.max_time_dilation = max_time_dilation
        self.max_phase_shift = max_phase_shift

        # Current noise levels
        self.config = NoiseConfig()

        # Tracking
        self.reward_ema = (target_reward_low + target_reward_high) / 2  # Start at midpoint
        self.total_updates = 0
        self.difficulty_level = 0.0  # 0.0 = easiest, 1.0 = hardest

        # History for logging
        self.history = []

    def update(self, epoch_reward: float) -> NoiseConfig:
        """
        Update noise levels based on recent reward.

        Args:
            epoch_reward: Average reward from last epoch

        Returns:
            Updated NoiseConfig
        """
        # Update EMA
        old_ema = self.reward_ema
        self.reward_ema = (1 - self.ema_alpha) * self.reward_ema + self.ema_alpha * epoch_reward

        # Determine action
        if self.reward_ema > self.target_high:
            # Too easy - increase difficulty
            self._increase_difficulty()
            action = "increase"
        elif self.reward_ema < self.target_low:
            # Too hard - decrease difficulty
            self._decrease_difficulty()
            action = "decrease"
        else:
            # In zone - maintain
            action = "maintain"

        self.total_updates += 1

        # Log history
        self.history.append({
            'epoch_reward': epoch_reward,
            'reward_ema': self.reward_ema,
            'difficulty': self.difficulty_level,
            'action': action,
            **self.config.to_dict()
        })

        # Logging
        if self.total_updates % 10 == 0:
            logging.info(
                f"[AdaptiveNoise] EMA={self.reward_ema:.1f} ({action}) "
                f"difficulty={self.difficulty_level:.2f} {self.config}"
            )

        return self.config

    def _increase_difficulty(self):
        """
        Progressively add noise in priority order:
        1. Price noise (most important, most impactful)
        2. Temporal jitter (forces reasoning about timing)
        3. Time dilation (forces pattern speed adaptation)
        4. Phase shift (forces re-detection of patterns)
        """
        step = self.adjustment_rate

        # Stage 1: Price noise (0-40% difficulty)
        if self.config.price_noise < self.max_price_noise:
            self.config.price_noise = min(
                self.max_price_noise,
                self.config.price_noise + step
            )

        # Stage 2: Temporal jitter (40-60% difficulty)
        elif self.config.temporal_jitter < self.max_temporal_jitter:
            self.config.temporal_jitter = min(
                self.max_temporal_jitter,
                self.config.temporal_jitter + 1
            )

        # Stage 3: Time dilation (60-80% difficulty)
        elif self.config.time_dilation < self.max_time_dilation:
            self.config.time_dilation = min(
                self.max_time_dilation,
                self.config.time_dilation + step * 0.5
            )

        # Stage 4: Phase shift (80-100% difficulty)
        elif self.config.phase_shift_prob < self.max_phase_shift:
            self.config.phase_shift_prob = min(
                self.max_phase_shift,
                self.config.phase_shift_prob + step * 0.5
            )

        self._update_difficulty_level()

    def _decrease_difficulty(self):
        """Reduce difficulty in reverse priority order."""
        step = self.adjustment_rate

        # Stage 4: Phase shift first (remove last added)
        if self.config.phase_shift_prob > 0:
            self.config.phase_shift_prob = max(
                0.0,
                self.config.phase_shift_prob - step * 0.5
            )

        # Stage 3: Time dilation
        elif self.config.time_dilation > 0:
            self.config.time_dilation = max(
                0.0,
                self.config.time_dilation - step * 0.5
            )

        # Stage 2: Temporal jitter
        elif self.config.temporal_jitter > 0:
            self.config.temporal_jitter = max(
                0,
                self.config.temporal_jitter - 1
            )

        # Stage 1: Price noise (last resort)
        elif self.config.price_noise > 0:
            self.config.price_noise = max(
                0.0,
                self.config.price_noise - step
            )

        self._update_difficulty_level()

    def _update_difficulty_level(self):
        """Compute overall difficulty level (0.0-1.0)."""
        # Weight each component
        price_contrib = self.config.price_noise / self.max_price_noise * 0.4
        jitter_contrib = self.config.temporal_jitter / self.max_temporal_jitter * 0.2 if self.max_temporal_jitter > 0 else 0
        dilation_contrib = self.config.time_dilation / self.max_time_dilation * 0.2 if self.max_time_dilation > 0 else 0
        phase_contrib = self.config.phase_shift_prob / self.max_phase_shift * 0.2 if self.max_phase_shift > 0 else 0

        self.difficulty_level = price_contrib + jitter_contrib + dilation_contrib + phase_contrib

    def get_config(self) -> NoiseConfig:
        """Get current noise configuration."""
        return self.config

    def get_synthetic_noise(self) -> float:
        """Get price noise level for SyntheticMarket."""
        return self.config.price_noise

    def get_state(self) -> Dict:
        """Get full state for checkpointing."""
        return {
            'config': self.config.to_dict(),
            'reward_ema': self.reward_ema,
            'total_updates': self.total_updates,
            'difficulty_level': self.difficulty_level,
        }

    def load_state(self, state: Dict):
        """Restore state from checkpoint."""
        if 'config' in state:
            cfg = state['config']
            self.config = NoiseConfig(
                price_noise=cfg.get('price_noise', 0.0),
                temporal_jitter=int(cfg.get('temporal_jitter', 0)),
                time_dilation=cfg.get('time_dilation', 0.0),
                phase_shift_prob=cfg.get('phase_shift_prob', 0.0),
            )
        self.reward_ema = state.get('reward_ema', 15.0)
        self.total_updates = state.get('total_updates', 0)
        self.difficulty_level = state.get('difficulty_level', 0.0)


def apply_temporal_jitter(
    prices: np.ndarray,
    jitter_steps: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    Apply temporal jitter to price data.

    Each asset's price series is shifted by a random amount within [-jitter_steps, +jitter_steps].
    This simulates observation delays and forces the model to be robust to timing variations.

    Args:
        prices: Price array of shape (n_steps, n_assets) or (n_assets, n_steps)
        jitter_steps: Maximum jitter in either direction
        rng: Random state for reproducibility

    Returns:
        Jittered prices with same shape
    """
    if jitter_steps <= 0:
        return prices

    # Handle both (n_steps, n_assets) and (n_assets, n_steps) layouts
    if prices.ndim == 1:
        # Single asset
        jitter = rng.integers(-jitter_steps, jitter_steps + 1)
        return np.roll(prices, jitter)

    # Assume (n_assets, n_steps) or determine from shape
    n_assets = prices.shape[0] if prices.shape[0] < prices.shape[1] else prices.shape[1]
    axis = 1 if prices.shape[0] < prices.shape[1] else 0

    result = prices.copy()
    for i in range(n_assets):
        jitter = rng.integers(-jitter_steps, jitter_steps + 1)
        if axis == 1:
            result[i, :] = np.roll(prices[i, :], jitter)
        else:
            result[:, i] = np.roll(prices[:, i], jitter)

    return result


def apply_time_dilation(
    prices: np.ndarray,
    dilation_factor: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    Apply time dilation (stretch/compress) to price series.

    Resamples the price series to a dilated time axis, making patterns appear
    faster or slower. Forces model to recognize patterns at different speeds.

    Args:
        prices: Price array of shape (n_steps,)
        dilation_factor: Max dilation (e.g., 0.3 means 0.7x to 1.3x speed)
        rng: Random state

    Returns:
        Dilated prices with same length
    """
    if dilation_factor <= 0:
        return prices

    n_steps = len(prices)

    # Random dilation for this series
    actual_dilation = 1.0 + rng.uniform(-dilation_factor, dilation_factor)

    # Create dilated time axis
    original_time = np.arange(n_steps)
    dilated_time = original_time * actual_dilation

    # Interpolate to original time axis
    dilated_prices = np.interp(original_time, dilated_time, prices)

    return dilated_prices


def apply_phase_shift(
    prices: np.ndarray,
    shift_prob: float,
    rng: np.random.RandomState,
    pattern_period: int = 100,
) -> np.ndarray:
    """
    Apply random phase shifts to price patterns.

    With probability shift_prob, shifts the pattern phase at random points.
    Forces model to re-detect patterns rather than relying on fixed timing.

    Args:
        prices: Price array of shape (n_steps,)
        shift_prob: Probability of phase shift per pattern_period
        rng: Random state
        pattern_period: Approximate pattern period for shift points

    Returns:
        Phase-shifted prices
    """
    if shift_prob <= 0:
        return prices

    n_steps = len(prices)
    result = prices.copy()

    # Determine shift points
    n_periods = n_steps // pattern_period
    for i in range(n_periods):
        if rng.random() < shift_prob:
            shift_point = i * pattern_period + rng.integers(0, pattern_period)
            shift_amount = rng.integers(pattern_period // 4, pattern_period // 2)

            # Shift remainder of series
            if shift_point + shift_amount < n_steps:
                result[shift_point:] = np.roll(result[shift_point:], shift_amount)

    return result
