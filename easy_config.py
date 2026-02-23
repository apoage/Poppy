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
    # TASK_676: 2=V2 (flatten, no temporal), 3=V3 (minGRU, preserves temporal ordering)
    encoder_version: int = 2
    state_dim: int = 64
    n_assets: int = 64
    action_space_size: int = 320  # 5 × 64
    n_channels: int = 34  # TASK_689: Was 27. +7 agent state channels (position, pnl, hold_dur, heat×2, capacity, confidence)
    H_cycles: int = 4  # TRM horizontal cycles
    L_cycles: int = 6  # TRM vertical cycles
    hidden_dim: int = 256
    # TASK_684: TRM reasoning hidden dim. None = state_dim * 16 (backward compat)
    trm_hidden_dim: Optional[int] = None
    num_heads: int = 8
    # TASK_683: V3 encoder width (embed_dim per asset, encoder attention heads)
    embed_dim: int = 32
    encoder_num_heads: int = 4
    dropout: float = 0.0
    # TASK_686: Mixed precision. "bfloat16" for ~25% VRAM reduction + speedup on Ampere GPUs
    compute_dtype: str = "float32"

    # === Training ===
    epochs: int = 100
    batch_size: int = 256
    steps_per_epoch: int = 200  # TASK_675: Explicit (was argparse-only default)
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    grad_clip: float = 5.0  # TASK_685: Was 1.0 — 13-16× over-suppression with observed norms 13-16

    # === Loss Weights ===
    policy_weight: float = 1.0
    value_weight: float = 0.5
    value_scale: float = 2.0  # TASK_674: Was 1.0, under-utilized 401-bin support. 2.0 uses ~30% of bins, safe for PnL≤5
    # DEPRECATED (TASK_149): Replaced by adaptive per-batch weighting. Kept only for checkpoint compat.
    trade_loss_weight: float = 60.0
    # TASK_677: Warmup disabled — full trade weight from step 1 to break turtle trap
    # Was 50 (TASK_151), but warmup lets model entrench all-HOLD before trades get a voice
    weight_warmup_epochs: int = 0

    # === Oracle Parameters ===
    # TASK_666 designed k=4. TASK_143/145 reverted k=10→k=1 (value mismatch).
    # TASK_681: Restore k=4 — 6.25% trade density vs 1.56% at k=1.
    # k=4 PnL overestimation is manageable (2-4× vs k=10's 5-10×).
    oracle_k: int = 4
    # TASK_152: Increased from 5 to 20 - 5 steps is micro-noise, 20 steps is a regime
    oracle_lookahead: int = 20  # Steps to look ahead
    transaction_cost: float = 0.001  # 0.1% per trade
    min_alpha_threshold: float = 0.0005  # TASK_679: Lowered from 0.001 to increase trade density (~1.1% → ~3% trades). Was 0.002 (TASK_157), 0.001 too sparse for balanced accuracy learning
    state_aware_oracle: bool = True  # TASK_141: Use state-aware open/close logic
    invert_returns: bool = False  # Dual-Bella: flip bull/bear for diversity
    # TASK_673: Multipass oracle for soft policy targets
    oracle_multipass: bool = False         # Disabled by default (backwards compat)
    oracle_n_passes: int = 10              # Number of noisy oracle passes
    oracle_noise_scale: float = 0.03       # Noise as fraction of return std (0.1 was 2x min_alpha, too noisy)
    oracle_sharpen_temperature: float = 0.1  # TASK_677: Post-averaging sharpening (0.1 = sharp commands, 1.0 = no sharpening)
    # TASK_691: Oracle entropy — corrupt oracle selections before state channel computation
    # Bridges train→inference gap by simulating imperfect model actions
    # TASK_692: DISABLED by default - model-state rollout is the correct fix
    oracle_entropy: bool = False            # Enable oracle entropy corruption (DISABLED)
    oracle_drop_rate: float = 0.15          # Max prob of dropping oracle trades (trade→hold)
    oracle_phantom_rate: float = 0.05       # Max prob of adding phantom trades (hold→trade)
    entropy_warmup_epochs: int = 50         # Clean oracle for this many epochs, then ramp

    # TASK_695: Oracle mask prediction auxiliary head
    # Trains model to predict oracle's top-k selection mask (RL²F-inspired)
    use_oracle_mask_head: bool = False
    oracle_mask_hidden_dim: int = 256
    oracle_mask_weight: float = 0.1

    # TASK_692: Model-state rollout — state channels from model's own actions
    # TASK_696: DAgger fix applied — AR now uses student-position-aware targets
    # Warmup: oracle state for first N epochs, then AR with correct DAgger targets
    autoregressive_state: bool = True
    ar_warmup_epochs: int = 100            # Oracle state channels for E0..N-1, AR from E_N onward
    state_refresh_fraction: float = 0.2     # Fraction of buffer to refresh per epoch (20%)

    # === Data ===
    # TASK_152: Changed from 60 to 64 for temporal encoder (rows=time, cols=assets)
    window_size: int = 64
    episode_length: int = 400
    synthetic_levels: List[int] = field(default_factory=lambda: list(range(15)))
    episodes_per_level: int = 500
    synthetic_noise: float = 0.1  # Anti-memorization
    # TASK_680: Trade-biased sampling — coin-flip probability of sampling from trade-heavy timesteps
    # 0.0 = uniform (off), 0.15 = 15% of windows are trade-biased. Start conservative.
    trade_sample_prob: float = 0.30  # TASK_687: Was 0.15 — double trade example frequency
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
        state_dim=128,        # TASK_683: Was 64 — wider representation
        H_cycles=6,
        L_cycles=8,
        hidden_dim=512,
        trm_hidden_dim=2048,  # TASK_696: Restored from 1024 — halved FFN killed trade learning
        num_heads=16,
        embed_dim=64,         # TASK_683: Was 32 — richer per-asset features
        encoder_num_heads=8,  # TASK_683: Was 4 — more attention heads for 64-dim
        epochs=200,
        episodes_per_level=1000,
    )
