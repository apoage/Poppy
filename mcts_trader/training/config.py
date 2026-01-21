from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import jax.numpy as jnp

@dataclass
class TrainingConfig:
    """Central configuration for MuZero training."""
    # Experiment metadata
    name: str = "default"
    mode: str = "MARGIN"
    epochs: int = 1000
    steps: int = 400
    batch_size: int = 256
    parallel_envs: int = 4
    
    # Environment settings
    fee: int = 0
    slippage: int = 0
    probabilistic_fills: bool = False
    borrow_rate: float = 0.0
    action_points: int = 1
    tp_sl_enabled: bool = False
    
    # Data settings
    synthetic_level: Optional[int] = None
    synthetic_steps: int = 10000
    db_path: Optional[str] = None
    cache_path: Optional[str] = None
    assets_file: Optional[str] = None
    periods_file: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    config_file: Optional[str] = None
    
    # Model architecture
    model_size: str = "small"
    state_dim: int = 64
    h_cycles: int = 4
    l_cycles: int = 6
    bf16: bool = False
    
    # MCTS settings
    sims: int = 32
    sims_warmup: int = 0
    sims_min: int = 8
    sims_schedule: str = "linear"
    noise: float = 0.3
    noise_frac: float = 0.25
    gumbel_scale: float = 3.0
    value_scale: float = 1.0
    value_shift: float = 0.0
    epsilon_greedy: float = 0.0
    dirichlet_aggressive: bool = False
    
    # Optimizer settings
    lr: float = 1e-4
    lr_schedule: str = "constant"
    lr_min: float = 1e-6
    lr_warmup_epochs: int = 10
    discount: float = 0.997
    entropy_coef: float = 0.01
    per_head_clipping: bool = False
    clip_grad_value: float = 1.0
    clip_grad_policy: float = 1.0
    clip_grad_dynamics: float = 1.0
    clip_grad_encoder: float = 1.0
    
    # Advanced features
    use_gae: bool = False
    gae_lambda: float = 0.95
    use_popart: bool = True
    popart_beta: float = 0.0001
    use_prediction_loss: bool = False
    physics_only: bool = False
    pred_head_size: str = "small"
    prediction_loss_weight: float = 0.2
    prediction_loss_initial: float = 0.01
    prediction_anneal_steps: int = 10000
    multi_step_k: int = 5
    scheduled_sampling_steps: int = 20000
    physics_reg_weight: float = 0.01
    physics_warmup_steps: int = 5000
    redo: bool = False
    redo_interval: int = 10
    redo_threshold: float = 0.01
    atc_loss: bool = False
    atc_weight: float = 0.1
    atc_k_step: int = 5
    atc_temperature: float = 0.1
    atc_num_negatives: int = 32
    
    # Training interventions
    adaptive_noise: bool = False
    target_reward_low: float = 10.0
    target_reward_high: float = 20.0
    noise_adjustment_rate: float = 0.05
    adaptive_fees: bool = False
    fee_reward_threshold: float = 2.5
    fee_upgrade_epochs: int = 10
    smart_allocation: bool = False
    jitter_minutes: int = 360
    episode_variance: int = 0
    adaptive_entropy: bool = False
    entropy_hold_threshold: float = 0.70
    entropy_boost_low: float = 0.15
    entropy_boost_high: float = 0.30
    sac_entropy: bool = False
    
    # Checkpointing & Logging
    load_checkpoint: Optional[str] = None
    no_checkpoint: bool = False
    safe_checkpoints: bool = False
    ds24_logging: bool = False
    ds24_level: int = 0
    ds24_save_tensors: bool = False
    ds24_grad_norms: bool = False
    ds24_intermediates_every: int = 10
    ds24_boost_every: int = 10
    ds24_snapshot_every: int = 50
    
    def __post_init__(self):
        """Handle derivative configuration."""
        # Map model size to state_dim if not explicitly provided
        if self.state_dim == 64 and self.model_size != "small":
            if self.model_size == 'medium':
                self.state_dim = 96
            elif self.model_size == 'large':
                self.state_dim = 160
        
        # Resolve DS24 logging level
        if self.ds24_logging and self.ds24_level == 0:
            self.ds24_level = 2
            
        # Auto-adjust batch size for model size to prevent OOM
        if self.model_size == 'large' and self.batch_size > 512:
            self.batch_size = 512
        elif self.model_size == 'medium' and self.batch_size > 384:
            self.batch_size = 384
