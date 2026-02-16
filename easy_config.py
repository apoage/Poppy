"""
Easy Mode Configuration - Minimal config for Expert Iteration.

TASK_666: Simplified configuration for supervised learning from oracle.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import json
from pathlib import Path


@dataclass
class EasyConfig:
    """Configuration for Easy Mode training."""

    # === Experiment ===
    name: str = "easy_mode_v1"
    seed: int = 42

    # === Model Architecture (reuse existing TRM) ===
    state_dim: int = 64
    n_assets: int = 64
    action_space_size: int = 320  # 5 × 64
    n_channels: int = 27
    H_cycles: int = 4  # TRM horizontal cycles
    L_cycles: int = 6  # TRM vertical cycles
    hidden_dim: int = 256
    num_heads: int = 8
    dropout: float = 0.0

    # === Training ===
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    grad_clip: float = 1.0

    # === Loss Weights ===
    policy_weight: float = 1.0
    value_weight: float = 0.5
    value_scale: float = 2.0  # TASK_674: Was 1.0, under-utilized 401-bin support. 2.0 uses ~30% of bins, safe for PnL≤5
    # DEPRECATED: trade_loss_weight is no longer used (TASK_149 uses adaptive per-batch weighting)
    # Kept for checkpoint compatibility. Adaptive weight = n_holds / n_trades per batch.
    trade_loss_weight: float = 60.0  # Legacy, ignored
    # TASK_151: Warmup epochs for trade weight ramping (prevents gradient shock on fresh models)
    weight_warmup_epochs: int = 50  # Ramp from 1.0 to full adaptive weight over this many epochs

    # === Oracle Parameters ===
    # TASK_145: k=1 because model executes 1 trade/step. Value head must match.
    # k>1 would train value head on unrealizable 10-asset profits → MCTS collapse.
    oracle_k: int = 1
    # TASK_152: Increased from 5 to 20 - 5 steps is micro-noise, 20 steps is a regime
    oracle_lookahead: int = 20  # Steps to look ahead
    transaction_cost: float = 0.001  # 0.1% per trade
    min_alpha_threshold: float = 0.001  # TASK_157: Lowered from 0.002 to increase trade signal (was 4% threshold, now 2%)
    state_aware_oracle: bool = True  # TASK_141: Use state-aware open/close logic
    invert_returns: bool = False  # Dual-Bella: flip bull/bear for diversity
    # TASK_673: Multipass oracle for soft policy targets
    oracle_multipass: bool = False         # Disabled by default (backwards compat)
    oracle_n_passes: int = 10              # Number of noisy oracle passes
    oracle_noise_scale: float = 0.03       # Noise as fraction of return std (0.1 was 2x min_alpha, too noisy)

    # === Data ===
    # TASK_152: Changed from 60 to 64 for temporal encoder (rows=time, cols=assets)
    window_size: int = 64
    episode_length: int = 400
    synthetic_levels: List[int] = field(default_factory=lambda: list(range(15)))
    episodes_per_level: int = 500
    synthetic_noise: float = 0.1  # Anti-memorization
    validation_split: float = 0.1

    # === Checkpointing ===
    checkpoint_dir: str = "jax_ckpt_easy"
    save_every: int = 5  # epochs
    eval_every: int = 1  # epochs

    # === Success Criteria ===
    # TASK_147: With balanced accuracy, turtle gets 50%. Target 75% is meaningful.
    # (Was 0.995 with broken metric, 0.85 before that)
    target_oracle_agreement: float = 0.75
    target_value_mse: float = 0.1

    # === Hardware ===
    n_workers: int = 8  # For parallel data generation

    def save(self, path: str):
        """Save config to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "EasyConfig":
        """Load config from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

    def __post_init__(self):
        """Validate config."""
        assert self.action_space_size == 5 * self.n_assets, \
            f"action_space_size must be 5 * n_assets"
        assert 0 < self.target_oracle_agreement <= 1.0
        assert self.oracle_k > 0
        assert self.batch_size > 0


# Preset configurations
def get_small_config() -> EasyConfig:
    """Small model for quick experiments."""
    return EasyConfig(
        name="easy_small",
        H_cycles=2,
        L_cycles=3,
        hidden_dim=128,
        num_heads=4,
        epochs=50,
        episodes_per_level=100,
    )


def get_medium_config() -> EasyConfig:
    """Medium model (default)."""
    return EasyConfig(
        name="easy_medium",
        H_cycles=4,
        L_cycles=6,
        hidden_dim=256,
        num_heads=8,
    )


def get_large_config() -> EasyConfig:
    """Large model for maximum performance."""
    return EasyConfig(
        name="easy_large",
        H_cycles=6,
        L_cycles=8,
        hidden_dim=512,
        num_heads=16,
        epochs=200,
        episodes_per_level=1000,
    )
