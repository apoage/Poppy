"""
Adaptive Fee/Slippage Controller for Historical Training

Adjusts trading costs based on performance to create gradual difficulty ramp.
Target reward: 2.5+ triggers fee/slippage increases.

Fee progression: 0 → 8 → 16 bps
Slippage progression: 0 → 2 → 5 bps
"""

import logging
from dataclasses import dataclass
from typing import Dict


@dataclass
class FeeConfig:
    """Configuration for trading costs."""
    fee_bps: int = 0          # Trading fee in basis points
    slippage_bps: int = 0     # Slippage in basis points

    def to_dict(self) -> Dict[str, int]:
        return {
            'fee_bps': self.fee_bps,
            'slippage_bps': self.slippage_bps,
        }

    def __str__(self) -> str:
        return f"FeeConfig(fee={self.fee_bps}bps, slippage={self.slippage_bps}bps)"


class AdaptiveFeeController:
    """
    Dynamically adjusts fees/slippage based on reward performance.

    When reward exceeds threshold:
      → Increase difficulty progressively

    Fee stages:
      Stage 0: 0 bps fee, 0 bps slippage (easiest)
      Stage 1: 8 bps fee, 2 bps slippage
      Stage 2: 16 bps fee, 5 bps slippage (hardest)
    """

    # Fee/slippage progression
    STAGES = [
        FeeConfig(fee_bps=0, slippage_bps=0),    # Stage 0: No costs
        FeeConfig(fee_bps=8, slippage_bps=2),    # Stage 1: Light costs
        FeeConfig(fee_bps=16, slippage_bps=5),   # Stage 2: Full costs
    ]

    def __init__(
        self,
        reward_threshold: float = 2.5,
        ema_alpha: float = 0.1,
        upgrade_after_epochs: int = 10,  # Epochs above threshold before upgrading
    ):
        """
        Args:
            reward_threshold: Reward EMA threshold to trigger stage increase
            ema_alpha: Smoothing factor for reward tracking (0.05-0.2)
            upgrade_after_epochs: Consecutive epochs above threshold needed to upgrade
        """
        self.reward_threshold = reward_threshold
        self.ema_alpha = ema_alpha
        self.upgrade_after_epochs = upgrade_after_epochs

        # Current state
        self.current_stage = 0
        self.config = self.STAGES[0]
        self.reward_ema = 0.0
        self.epochs_above_threshold = 0
        self.total_updates = 0

        # History for logging
        self.history = []

    def update(self, epoch_reward: float) -> FeeConfig:
        """
        Update fee/slippage based on recent reward.

        Args:
            epoch_reward: Average reward from last epoch

        Returns:
            Updated FeeConfig
        """
        # Update EMA
        if self.total_updates == 0:
            self.reward_ema = epoch_reward
        else:
            self.reward_ema = (1 - self.ema_alpha) * self.reward_ema + self.ema_alpha * epoch_reward

        # Check if above threshold
        if self.reward_ema > self.reward_threshold:
            self.epochs_above_threshold += 1
        else:
            self.epochs_above_threshold = 0

        # Check for stage upgrade
        action = "maintain"
        if self.epochs_above_threshold >= self.upgrade_after_epochs:
            if self.current_stage < len(self.STAGES) - 1:
                self.current_stage += 1
                self.config = self.STAGES[self.current_stage]
                self.epochs_above_threshold = 0
                action = "upgrade"
                logging.info(
                    f"[AdaptiveFees] STAGE UPGRADE → Stage {self.current_stage}: {self.config}"
                )

        self.total_updates += 1

        # Log history
        self.history.append({
            'epoch_reward': epoch_reward,
            'reward_ema': self.reward_ema,
            'stage': self.current_stage,
            'action': action,
            **self.config.to_dict()
        })

        # Periodic logging
        if self.total_updates % 10 == 0:
            logging.info(
                f"[AdaptiveFees] EMA={self.reward_ema:.2f} stage={self.current_stage} "
                f"epochs_above={self.epochs_above_threshold}/{self.upgrade_after_epochs} {self.config}"
            )

        return self.config

    @property
    def difficulty_level(self) -> float:
        """Get difficulty as 0.0-1.0 value."""
        return self.current_stage / (len(self.STAGES) - 1)

    def get_config(self) -> FeeConfig:
        """Get current fee configuration."""
        return self.config

    def get_fee_rate(self) -> float:
        """Get fee as decimal (for env)."""
        return self.config.fee_bps / 10000.0

    def get_slippage_rate(self) -> float:
        """Get slippage as decimal (for env)."""
        return self.config.slippage_bps / 10000.0

    def get_state(self) -> Dict:
        """Get full state for checkpointing."""
        return {
            'current_stage': self.current_stage,
            'reward_ema': self.reward_ema,
            'epochs_above_threshold': self.epochs_above_threshold,
            'total_updates': self.total_updates,
        }

    def load_state(self, state: Dict):
        """Restore state from checkpoint."""
        self.current_stage = state.get('current_stage', 0)
        self.config = self.STAGES[min(self.current_stage, len(self.STAGES) - 1)]
        self.reward_ema = state.get('reward_ema', 0.0)
        self.epochs_above_threshold = state.get('epochs_above_threshold', 0)
        self.total_updates = state.get('total_updates', 0)
