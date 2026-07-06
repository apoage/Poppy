"""
Configuration for Diffusion Teacher.

All hyperparameters and settings in one place.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import json
from pathlib import Path


@dataclass
class TeacherConfig:
    """Configuration for the Diffusion Teacher system."""

    # ==========================================================================
    # Data Configuration
    # ==========================================================================
    db_path: str = "trading_data.db"
    n_assets: int = 64  # Number of assets to load
    window_size: int = 400  # Steps per market window
    features: Tuple[str, ...] = ("open", "high", "low", "close", "volume")

    # Data splits
    train_start: str = "2020-01-01"
    train_end: str = "2024-12-31"
    val_start: str = "2025-01-01"
    val_end: str = "2025-06-30"

    # ==========================================================================
    # Encoder Configuration
    # ==========================================================================
    latent_dim: int = 256  # Latent space dimension (smaller for 6GB target)
    encoder_heads: int = 4  # Attention heads
    encoder_layers: int = 4  # Transformer layers
    encoder_ff_dim: int = 512  # Feed-forward dimension
    encoder_dropout: float = 0.1

    # ==========================================================================
    # Flow Matching Configuration
    # ==========================================================================
    # Architecture choice: "conv" for O(n) memory, "transformer" for O(n²)
    flow_architecture: str = "conv"  # "conv" (default, 302M) or "transformer" (22M)

    # Shared settings
    flow_hidden_dim: int = 1536  # 1536=302M for conv, 1024=100M for transformer
    flow_layers: int = 16        # 16 layers for conv, 12 for transformer
    flow_dropout: float = 0.0    # No dropout for flow matching

    # Transformer-specific (only used if flow_architecture="transformer")
    flow_heads: int = 16         # Attention heads

    # Conv-specific (only used if flow_architecture="conv")
    flow_kernel_size: int = 3    # Kernel size for dilated convs
    flow_ff_dim: int = 3072      # Feed-forward dim (default: 2 * hidden_dim)
    flow_cond_dim: int = 768     # Conditioning dim (default: hidden_dim // 2)

    # OT-CFM (Optimal Transport Conditional Flow Matching)
    sigma_min: float = 1e-4  # Minimum noise level
    use_ot: bool = True  # Use optimal transport paths

    # ==========================================================================
    # Training Configuration
    # ==========================================================================
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    max_steps: int = 100000

    # Checkpointing
    checkpoint_dir: str = "checkpoints/diff_teacher"
    checkpoint_every: int = 5000
    keep_best_n: int = 3

    # Logging
    log_every: int = 100
    eval_every: int = 1000

    # ==========================================================================
    # Curriculum Configuration
    # ==========================================================================
    # Regret-based difficulty adjustment
    difficulty_init: float = 0.5  # Starting difficulty [0, 1]
    difficulty_min: float = 0.1
    difficulty_max: float = 1.0
    difficulty_step: float = 0.05  # How much to adjust per update

    # Thresholds (as fraction of optimal)
    regret_low_threshold: float = 0.2  # Below this = too easy, increase difficulty
    regret_high_threshold: float = 0.6  # Above this = too hard, decrease difficulty

    # EMA for smoothing regret estimates
    regret_ema_alpha: float = 0.1

    # ==========================================================================
    # Hardware Configuration
    # ==========================================================================
    dtype: str = "float16"  # float16 for memory efficiency
    seed: int = 42

    # Memory optimization
    gradient_checkpointing: bool = True
    max_batch_in_memory: int = 64  # For gradient accumulation

    # ==========================================================================
    # Generation Configuration
    # ==========================================================================
    # How many steps to generate per sequence
    generation_steps: int = 400

    # Conditioning
    use_regime_conditioning: bool = True
    regime_embedding_dim: int = 64

    def save(self, path: str) -> None:
        """Save config to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TeacherConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        # Convert tuple back from list
        if "features" in data:
            data["features"] = tuple(data["features"])
        return cls(**data)

    def __post_init__(self):
        """Validate configuration."""
        assert self.n_assets > 0, "n_assets must be positive"
        assert self.window_size > 0, "window_size must be positive"
        assert self.latent_dim > 0, "latent_dim must be positive"
        assert 0 < self.difficulty_min <= self.difficulty_max <= 1.0
        assert self.regret_low_threshold < self.regret_high_threshold


@dataclass
class OptimalSolverConfig:
    """Configuration for the hindsight optimal solver."""

    # Transaction costs (for realistic optimal calculation)
    fee_bps: float = 0.0  # Basis points fee (0 = perfect hindsight)
    slippage_bps: float = 0.0

    # Position limits
    max_position_per_asset: float = 1.0  # Max fraction of portfolio per asset

    # Algorithm choice
    algorithm: str = "peak_valley"  # "peak_valley" or "dp" (dynamic programming)


@dataclass
class GenerationConfig:
    """Configuration for market sequence generation."""

    # Conditioning
    regime: Optional[str] = None  # "crash", "rally", "sideways", "volatile", None=random

    # Diversity
    temperature: float = 1.0  # Higher = more diverse

    # Output format
    return_latent: bool = False  # Return latent vectors instead of prices

    # Correlation constraints
    preserve_correlations: bool = True
    correlation_weight: float = 0.1  # Loss weight for correlation matching
