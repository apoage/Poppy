from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import jax.numpy as jnp


@dataclass
class PhysicsConfig:
    """
    Configuration for physics-based market model parameters.

    These were previously hardcoded "magic numbers" that directly influence
    the agent's behavior. Different asset classes may need different values:
    - Crypto: Higher volatility (sigma_target ~0.02-0.05)
    - Forex: Lower volatility (sigma_target ~0.005)
    - Trending markets: Higher alpha_target
    - Mean-reverting: Higher beta_target
    """
    # Reward scaling for categorical value representation
    # Maps symlog rewards to [-10, +10] support range
    reward_scale: float = 100.0

    # Physics regularization targets (what the model is pulled toward)
    alpha_target: float = 0.05    # Momentum bias (5% default)
    beta_target: float = 0.03     # Mean reversion strength (3% default)
    sigma_target: float = 0.01    # Volatility level (1% daily default)

    # GARCH persistence bounds (volatility clustering)
    garch_floor: float = 0.85     # Minimum persistence
    garch_ceiling: float = 1.0    # Maximum persistence
    garch_penalty_weight: float = 2.0

    # Categorical distribution smoothing
    label_smoothing: float = 0.1  # 10% uniform, 90% target

    # Entropy thresholds for policy collapse detection
    min_entropy: float = 3.0      # Danger zone (log(320) ≈ 5.77)
    entropy_boost_multiplier: float = 5.0  # Adaptive entropy penalty multiplier

    # Self-play constraints
    max_positions: int = 16       # Maximum simultaneous positions
    min_cash_threshold: float = 100.0  # Minimum cash to allow buying
    confidence_threshold: float = 0.0  # Force HOLD if max_prob < threshold

    # Batch transfer settings (GPU->CPU)
    batch_steps: int = 50         # Transfer every N steps

    def for_crypto(self) -> 'PhysicsConfig':
        """Preset for high-volatility crypto markets."""
        return PhysicsConfig(
            reward_scale=100.0,
            alpha_target=0.03,      # Less momentum (noisy)
            beta_target=0.02,       # Less reversion (trending)
            sigma_target=0.03,      # Higher volatility (3%)
            garch_floor=0.80,
            min_entropy=2.5,        # Lower threshold (more concentrated)
        )

    def for_forex(self) -> 'PhysicsConfig':
        """Preset for low-volatility forex markets."""
        return PhysicsConfig(
            reward_scale=200.0,     # Smaller moves need more scaling
            alpha_target=0.02,
            beta_target=0.05,       # More mean reversion
            sigma_target=0.005,     # Lower volatility (0.5%)
            garch_floor=0.90,
            min_entropy=3.5,
        )


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

    # TASK_821: RNG seed override for reproducible/multi-seed training runs.
    # None (default) = EXACTLY current behavior (time-seeded model init + time-seeded
    # historical-env RNG in mcts_trader/training/env_factory.py). When set, seeds
    # Python `random`, numpy's global RNG, and derives the model-init PRNGKey +
    # env-factory rng_key deterministically from it (see train_jax_fast.py::main).
    seed: Optional[int] = None
    
    # Environment settings
    fee: int = 0
    slippage: int = 0
    probabilistic_fills: bool = False
    borrow_rate: float = 0.0
    action_points: int = 1
    tp_sl_enabled: bool = False
    
    # Data settings
    synthetic_level: Optional[int] = None
    weighted_synthetic: bool = False  # Use weighted random level selection (L0-L14)
    synthetic_steps: int = 10000
    synthetic_noise: float = 0.0  # Price noise for anti-memorization (0.0 = clean, 1.0 = full noise)
    noise_schedule: bool = False  # Ramp noise from 0 to synthetic_noise over epochs
    regenerate_data_every: int = 0  # Regenerate synthetic data every N epochs (0=disabled)
    unique_data_per_env: bool = False  # Generate unique prices for each parallel env
    data_gen_workers: int = 8  # Threads for parallel data generation
    db_path: Optional[str] = None
    cache_path: Optional[str] = None
    assets_file: Optional[str] = None
    periods_file: Optional[str] = None
    periods: Optional[str] = None
    periods_assets_file: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    schedule_path: Optional[str] = None
    config_file: Optional[str] = None

    # TASK_110: Adversarial curriculum
    adversarial_data_dir: Optional[str] = None
    adversarial_ratio: float = 0.3
    # On-the-fly VRAM generation (fast conv-based generator)
    # Uses O(n) dilated causal convolutions for memory efficiency
    adversarial_on_the_fly: bool = False
    adversarial_hidden_dim: int = 1536  # 1536=302M params (5GB), 1024=101M, 512=17M
    adversarial_num_layers: int = 16    # 16=deep receptive field, 12=balanced, 8=fast
    adversarial_regime_bias: Optional[str] = None  # 'crash', 'rally', etc.
    # TASK_111: Load trained generator checkpoint
    adversarial_checkpoint: Optional[str] = None  # Path to trained diff_teacher checkpoint

    # TASK_111: Bella (trained CNN diffusion) as data source
    bella: bool = False  # Use Bella for market generation
    bella_checkpoint: str = 'checkpoints/diff_teacher/final/generator'
    bella_samples_per_epoch: int = 1  # Fresh samples per epoch

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
    weight_decay: float = 1e-4  # BUG-011: AdamW weight decay (L2 regularization)
    global_grad_clip: float = 5.0  # BUG-013: Global gradient clip (was 10.0, too permissive)
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
    redo_init_scale: float = 0.01
    atc_loss: bool = False
    atc_weight: float = 0.1
    atc_k_step: int = 5
    atc_temperature: float = 0.1
    atc_num_negatives: int = 32

    # Cycle Balance (TASK_110)
    cycle_balance: bool = False
    cycle_balance_weight: float = 0.1
    cycle_balance_target_cv: float = 0.3
    cycle_atc: bool = False
    cycle_atc_weight: float = 0.05
    cycle_atc_temperature: float = 0.1

    # Reward shaping
    differential_reward: bool = False
    differential_baseline: str = "market"
    differential_weight: float = 1.0
    # TASK_818 Tier B: action-attributable realized-PnL-on-close env reward (default OFF = canon)
    reward_attributable: bool = False
    # TASK_818-freq: resample historical 1-min closes to K-minute bars (1 = canon 1-min)
    bar_minutes: int = 1

    # TASK_828 Stage 3: vector-action MuZero (per-asset action VECTOR each step).
    # Default OFF = today's scalar path, canon bit-identical. The single owner flag;
    # fans out to EnvConfig/MuZeroNetworks/JAXReplayBuffer/MuZeroTrainer + search.
    vector_action: bool = False
    vector_k: int = 16                       # K sampled candidate action-vectors / root
    vector_hold_bias: float = 0.0            # additive HOLD-slot bias in the proposal
    vector_dither_scale: float = 1.0         # Gumbel dither scale on the sampling proposal
    vector_gumbel_scale_eval: float = 0.5    # (reserved) lower gumbel scale for eval
    vector_proposal_scheme: str = 'dithered-book'  # 'dithered-book' | 'iid-beta' | 'policy-prior'
    # TASK_847: policy-prior proposal knobs (used ONLY when scheme='policy-prior';
    # default scheme unchanged ⇒ every existing invocation stays bit-identical).
    vector_proposal_temp: float = 1.0        # temperature on the per-asset policy marginals
    vector_proposal_explore_slots: int = 2   # N dither-machinery slots retained (of K)
    # TASK_838: resample K vectors at each expanded interior node (default OFF =
    # root-reuse, bit-identical to the shipped Stage-2/3 vector search).
    vector_per_node_resample: bool = False
    # TASK_839: budget-constrained vector-step sizing (env-wall fix, D7 opt (a)).
    # Default OFF = today's single-snapshot sizing, canon bit-identical. Threaded to
    # EnvConfig via env_factory; the scalar path never reads it.
    vector_budget_step: bool = False
    vector_budget_pct: float = 0.99
    # TASK_840: target-weight rebalance step (D8 opt (c), the H6 env). Default OFF = canon.
    # Threaded to EnvConfig via env_factory; the scalar path never reads it. w_unit sentinel
    # <=0 => 1/n_assets; deadband sentinel <0 => 0.1*w_unit (resolved inside step_vector).
    vector_weight_rebalance: bool = False
    vector_weight_unit: float = -1.0
    vector_weight_max: float = 4.0
    vector_rebalance_deadband: float = -1.0
    burning_money: bool = False
    burn_market_linked: bool = True
    burn_base_rate: float = 0.00005
    burn_random: bool = True
    burn_random_prob: float = 0.03
    burn_random_max: float = 500.0  # Max random burn amount
    burn_random_limit: int = 10  # Max burns per episode
    adaptive_burn: bool = False
    burn_hold_threshold: float = 0.60
    burn_patience: int = 5

    # Training interventions
    adaptive_noise: bool = False
    target_reward_low: float = 10.0
    target_reward_high: float = 20.0
    noise_adjustment_rate: float = 0.05
    max_temporal_jitter: int = 5
    max_time_dilation: float = 0.3
    max_phase_shift: float = 0.5
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
    sac_target_entropy_ratio: float = 0.5
    sac_init_alpha: float = 0.01
    sac_alpha_lr: float = 3e-4

    # Checkpointing & Logging
    load_checkpoint: Optional[str] = None
    no_checkpoint: bool = False
    reset_optimizer: bool = False
    clear_buffer: bool = False
    safe_checkpoints: bool = False
    snapshot_every: int = 0
    rollback_patience: int = 50
    rollback_threshold: float = 0.5

    # Generalization evaluation
    eval_every: int = 0
    eval_level: int = 4
    eval_steps: int = 200

    ds24_logging: bool = False
    ds24_level: int = 0
    ds24_save_tensors: bool = False
    ds24_grad_norms: bool = False
    ds24_intermediates_every: int = 10
    ds24_boost_every: int = 10
    ds24_snapshot_every: int = 50

    # Physics configuration (TASK_066: extracted magic numbers)
    reward_scale: float = 100.0
    alpha_target: float = 0.05
    beta_target: float = 0.03
    sigma_target: float = 0.01
    label_smoothing: float = 0.1
    min_entropy: float = 3.0
    max_positions: int = 16
    min_cash_threshold: float = 100.0
    confidence_threshold: float = 0.0

    # Tree Search (H1 - Phase 3)
    use_tree_search: bool = False
    tree_depth: int = 5
    tree_branches: int = 3
    risk_lambda: float = 0.1

    # Flow Matching (H2 - TASK_041)
    flow_matching: bool = False
    flow_velocity_weight: float = 0.1

    # Adaptive Calibration (H3)
    adaptive_calibration: bool = False

    # Early Graduation (H4)
    early_graduation: int = 0

    # Value Loss Type (H5)
    value_loss_type: str = 'mse'
    value_huber_delta: float = 1.0

    # Prediction Loss Type
    pred_loss_type: str = 'mse'
    pred_huber_delta: float = 1.0

    # TASK_116: Student-t output for fat-tailed returns
    use_student_t: bool = False      # Enable Student-t loss for prediction
    student_t_weight: float = 0.8    # Weight for Student-t NLL component
    student_t_mse_weight: float = 0.2  # Weight for MSE component (stability)
    student_t_df_min: float = 2.1    # Min degrees of freedom (>2 for finite variance)
    student_t_df_max: float = 100.0  # Max df (large = nearly Gaussian)

    # Additional prediction config (L1)
    prediction_loss_initial: float = 0.01
    prediction_anneal_steps: int = 10000
    multi_step_k: int = 5
    physics_reg_weight: float = 0.01

    # TASK_118: Alpha-relative reward normalization
    alpha_relative_reward: bool = False  # Enable alpha-relative reward
    alpha_reward_blend: float = 0.5      # Blend factor (0=raw, 1=pure alpha-relative)
    alpha_top_k: int = 4                 # Top-k assets for max alpha calc

    # TASK_119: Optimal hindsight path computation
    amax: Optional[str] = None           # 'greedy', 'soft', 'sinkhorn', or None
    amax_regret_weight: float = 0.0      # Regret penalty weight (0=disabled, 0.01-0.1=auxiliary)
    amax_imprint_weight: float = 0.0     # Optimal action imprint weight (0=disabled, 0.01-0.02=stabilization)
    amax_imprint_temp: float = 1.0       # Temperature for imprint targets (lower=more peaked)

    # TASK_125: Causal Expert Imprinting
    imprint: bool = False                # Enable causal expert imprinting
    bc_weight_initial: float = 0.1       # Initial BC loss weight
    bc_weight_final: float = 0.01        # Final BC loss weight after decay
    bc_weight_decay_steps: int = 50000   # Steps to decay BC weight
    expert_demo_capacity: int = 50000    # Max expert demos to store
    expert_confidence_threshold: float = 0.001  # Min alpha capture ratio to store as demo (0.1% of optimal)
    bc_use_q_filter: bool = True         # Apply BC loss only when Q(demo) > Q(policy)
    bc_use_large_margin: bool = True     # Use DQfD-style large margin loss

    # Safe checkpoint config (L1)
    max_rollbacks: int = 3
    keep_best_n: int = 3
    lr_decay_on_rollback: float = 0.5
    
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
