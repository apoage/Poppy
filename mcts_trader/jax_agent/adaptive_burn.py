"""
Adaptive Burn Controller - Escalating pressure against excessive holding.

Monitors hold rate over consecutive episodes and progressively increases
burn pressure when the agent refuses to trade.

Stages:
  0: Base burn only (0.005% per step)
  1: + Market-linked burn (volatility scaled)
  2: + Random burn events (3% -> 10% chance)
  3: + Increased base rate (4x)
  4: + Position liquidation events (hardcore)

Escalation: If hold_rate > threshold for N consecutive episodes, advance stage.
De-escalation: If hold_rate < threshold for N episodes, retreat one stage.
"""

from dataclasses import dataclass
from typing import Optional
import logging


@dataclass
class BurnConfig:
    """Current burn configuration."""
    enabled: bool = False
    base_rate: float = 0.00005      # 0.005% per step
    market_linked: bool = True
    random_burn: bool = False
    random_prob: float = 0.03       # 3% chance per step
    liquidation_prob: float = 0.0   # 0% chance of forced liquidation
    stage: int = 0

    def __repr__(self):
        if not self.enabled:
            return "BurnConfig(OFF)"
        parts = [f"stage={self.stage}", f"base={self.base_rate:.5f}"]
        if self.market_linked:
            parts.append("mkt")
        if self.random_burn:
            parts.append(f"rnd={self.random_prob:.0%}")
        if self.liquidation_prob > 0:
            parts.append(f"liq={self.liquidation_prob:.0%}")
        return f"BurnConfig({', '.join(parts)})"


class AdaptiveBurnController:
    """
    Adaptive burn controller that escalates pressure when hold rate is too high.

    Args:
        hold_threshold: Hold rate above which triggers escalation (default: 0.60)
        patience: Consecutive episodes before stage change (default: 5)
        max_stage: Maximum burn stage (default: 4 = hardcore)
    """

    # Stage definitions
    STAGES = {
        0: {
            'name': 'base',
            'base_rate': 0.00005,
            'market_linked': True,
            'random_burn': False,
            'random_prob': 0.0,
            'liquidation_prob': 0.0,
        },
        1: {
            'name': 'market',
            'base_rate': 0.00005,
            'market_linked': True,
            'random_burn': True,
            'random_prob': 0.03,
            'liquidation_prob': 0.0,
        },
        2: {
            'name': 'random',
            'base_rate': 0.0001,      # 2x base
            'market_linked': True,
            'random_burn': True,
            'random_prob': 0.07,      # 7% random
            'liquidation_prob': 0.0,
        },
        3: {
            'name': 'aggressive',
            'base_rate': 0.0002,      # 4x base
            'market_linked': True,
            'random_burn': True,
            'random_prob': 0.10,      # 10% random
            'liquidation_prob': 0.0,
        },
        4: {
            'name': 'hardcore',
            'base_rate': 0.0003,      # 6x base
            'market_linked': True,
            'random_burn': True,
            'random_prob': 0.15,      # 15% random
            'liquidation_prob': 0.02, # 2% forced liquidation
        },
    }

    def __init__(
        self,
        hold_threshold: float = 0.60,
        patience: int = 5,
        max_stage: int = 4,
    ):
        self.hold_threshold = hold_threshold
        self.patience = patience
        self.max_stage = min(max_stage, len(self.STAGES) - 1)

        self.current_stage = 0
        self.consecutive_high = 0
        self.consecutive_low = 0
        self.enabled = False  # Must be set before _make_config
        self.config = self._make_config(0)

        logging.info(f"AdaptiveBurnController initialized: threshold={hold_threshold:.0%}, patience={patience}")

    def _make_config(self, stage: int) -> BurnConfig:
        """Create BurnConfig for given stage."""
        stage_def = self.STAGES[stage]
        return BurnConfig(
            enabled=self.enabled,
            base_rate=stage_def['base_rate'],
            market_linked=stage_def['market_linked'],
            random_burn=stage_def['random_burn'],
            random_prob=stage_def['random_prob'],
            liquidation_prob=stage_def['liquidation_prob'],
            stage=stage,
        )

    def enable(self):
        """Enable burn controller."""
        self.enabled = True
        self.config.enabled = True
        logging.info(f"[BURN] Enabled at stage {self.current_stage} ({self.STAGES[self.current_stage]['name']})")

    def update(self, hold_rate: float) -> BurnConfig:
        """
        Update burn config based on hold rate.

        Args:
            hold_rate: Fraction of actions that were HOLD (0.0 to 1.0)

        Returns:
            Updated BurnConfig
        """
        if not self.enabled:
            return self.config

        old_stage = self.current_stage

        if hold_rate > self.hold_threshold:
            # Too much holding - consider escalation
            self.consecutive_high += 1
            self.consecutive_low = 0

            if self.consecutive_high >= self.patience and self.current_stage < self.max_stage:
                self.current_stage += 1
                self.consecutive_high = 0
                stage_name = self.STAGES[self.current_stage]['name']
                logging.warning(
                    f"[BURN] 🔥 ESCALATING to stage {self.current_stage} ({stage_name}) - "
                    f"hold rate {hold_rate:.1%} > {self.hold_threshold:.0%} for {self.patience} episodes"
                )
        else:
            # Hold rate acceptable - consider de-escalation
            self.consecutive_low += 1
            self.consecutive_high = 0

            if self.consecutive_low >= self.patience and self.current_stage > 0:
                self.current_stage -= 1
                self.consecutive_low = 0
                stage_name = self.STAGES[self.current_stage]['name']
                logging.info(
                    f"[BURN] ✓ De-escalating to stage {self.current_stage} ({stage_name}) - "
                    f"hold rate {hold_rate:.1%} < {self.hold_threshold:.0%}"
                )

        # Update config if stage changed
        if self.current_stage != old_stage:
            self.config = self._make_config(self.current_stage)
            self.config.enabled = self.enabled

        return self.config

    def get_config(self) -> BurnConfig:
        """Get current burn configuration."""
        return self.config

    def get_cli_args(self) -> list:
        """Get CLI arguments for current burn config."""
        if not self.enabled:
            return []

        args = ['--burning-money']
        args.extend(['--burn-base-rate', str(self.config.base_rate)])

        if self.config.market_linked:
            args.append('--burn-market-linked')

        if self.config.random_burn:
            args.append('--burn-random')
            args.extend(['--burn-random-prob', str(self.config.random_prob)])

        return args

    def get_status(self) -> str:
        """Get human-readable status."""
        if not self.enabled:
            return "OFF"

        stage_name = self.STAGES[self.current_stage]['name']
        return (
            f"stage={self.current_stage}/{self.max_stage} ({stage_name}) | "
            f"high={self.consecutive_high}/{self.patience} | "
            f"low={self.consecutive_low}/{self.patience}"
        )
