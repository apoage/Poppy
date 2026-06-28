import argparse
from mcts_trader.training.config import TrainingConfig

def parse_args() -> TrainingConfig:
    """Parse CLI arguments into a TrainingConfig object."""
    parser = argparse.ArgumentParser(description='JAX MuZero Training')
    
    # Core settings
    parser.add_argument('--name', type=str, default='default')
    parser.add_argument('--mode', type=str, default='MARGIN', choices=['SPOT', 'MARGIN'])
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--episode-variance', type=int, default=0,
                        help='Episode length variance (±N steps). Prevents end-of-episode exploits')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--parallel-envs', type=int, default=4)
    
    # Environment settings
    parser.add_argument('--fee', type=int, default=0)
    parser.add_argument('--slippage', type=int, default=0)
    parser.add_argument('--probabilistic-fills', action='store_true')
    parser.add_argument('--borrow-rate', type=float, default=0.0)
    parser.add_argument('--action-points', type=int, default=1)
    parser.add_argument('--tp-sl-enabled', action='store_true')
    
    # Data settings
    parser.add_argument('--synthetic-level', type=int, default=None)
    parser.add_argument('--weighted-synthetic', action='store_true',
                        help='Use weighted random level selection (L0-L14) - changes level each data regeneration')
    parser.add_argument('--synthetic-steps', type=int, default=10000)
    parser.add_argument('--synthetic-noise', type=float, default=0.0,
                        help='Price noise level for anti-memorization (0.0=clean, 1.0=full noise)')
    parser.add_argument('--noise-schedule', action='store_true',
                        help='Ramp noise from 0 to synthetic-noise over epochs')
    parser.add_argument('--regenerate-data-every', type=int, default=0,
                        help='Regenerate synthetic data every N epochs (0=disabled)')
    parser.add_argument('--unique-data-per-env', action='store_true',
                        help='Generate unique price data for each parallel env (more diversity)')
    parser.add_argument('--data-gen-workers', type=int, default=8,
                        help='Number of threads for parallel data generation')
    parser.add_argument('--db-path', type=str, default=None)
    parser.add_argument('--cache-path', type=str, default=None)
    parser.add_argument('--assets-file', type=str, default=None)
    parser.add_argument('--periods-file', type=str, default=None)
    parser.add_argument('--periods', type=str, default=None,
                        help='Comma-separated period names to use from periods file')
    parser.add_argument('--periods-assets-file', type=str, default=None,
                        help='Assets file for multi-period stratified training')
    parser.add_argument('--start-date', type=str, default=None)
    parser.add_argument('--end-date', type=str, default=None)
    parser.add_argument('--schedule-path', type=str, default=None,
                        help='Path to training schedule JSON file')
    parser.add_argument('--config', type=str, dest='config_file', default=None)

    # TASK_110: Adversarial curriculum data
    parser.add_argument('--adversarial-data-dir', type=str, default=None,
                        help='Directory with adversarial market data from diff_teacher')
    parser.add_argument('--adversarial-ratio', type=float, default=0.3,
                        help='Fraction of adversarial data to mix in (0.0-1.0)')
    # On-the-fly VRAM generation (fast generator)
    parser.add_argument('--adversarial-on-the-fly', action='store_true',
                        help='Generate adversarial data on-the-fly in VRAM using O(n) convolutions')
    parser.add_argument('--adversarial-hidden-dim', type=int, default=1536,
                        help='Fast generator hidden dim (1536=302M params, 1024=101M, 512=17M)')
    parser.add_argument('--adversarial-num-layers', type=int, default=16,
                        help='Fast generator layers (16=302M@1536d, 12=227M@1536d, 8=88M@512d)')
    parser.add_argument('--adversarial-regime-bias', type=str, default=None,
                        choices=['crash', 'rally', 'sideways', 'volatile', 'calm'],
                        help='Bias generation towards specific regime')
    # TASK_111: Load trained generator checkpoint
    parser.add_argument('--adversarial-checkpoint', type=str, default=None,
                        help='Path to trained diff_teacher generator checkpoint (loads trained weights instead of random)')

    # TASK_111: Bella (trained CNN diffusion) as data source
    parser.add_argument('--bella', action='store_true',
                        help='Use trained Bella (CNN diffusion) for market data generation')
    parser.add_argument('--bella-checkpoint', type=str,
                        default='checkpoints/diff_teacher/final/generator',
                        help='Path to Bella checkpoint directory')
    parser.add_argument('--bella-samples-per-epoch', type=int, default=1,
                        help='Number of fresh market samples to generate per epoch')

    # Model architecture
    parser.add_argument('--model-size', type=str, default='small', choices=['small', 'medium', 'large'])
    parser.add_argument('--state-dim', type=int, default=None)
    parser.add_argument('--h-cycles', type=int, default=4)
    parser.add_argument('--l-cycles', type=int, default=6)
    parser.add_argument('--bf16', action='store_true')
    
    # MCTS settings
    parser.add_argument('--sims', type=int, default=32)
    parser.add_argument('--sims-warmup', type=int, default=0)
    parser.add_argument('--sims-min', type=int, default=8)
    parser.add_argument('--sims-schedule', type=str, default='linear')
    parser.add_argument('--noise', type=float, default=0.3)
    parser.add_argument('--noise-frac', type=float, default=0.25)
    parser.add_argument('--gumbel-scale', type=float, default=3.0)
    parser.add_argument('--value-scale', type=float, default=1.0)
    parser.add_argument('--value-shift', type=float, default=0.0)
    parser.add_argument('--epsilon-greedy', type=float, default=0.0)
    # TASK_812: decaying epsilon-greedy target-mix schedule (death-spiral mitigation).
    # When --epsilon-schedule is OFF, behavior is identical to canon (constant --epsilon-greedy).
    parser.add_argument('--epsilon-schedule', action='store_true',
                        help='TASK_812: decay epsilon-greedy target mixing init->final over decay-epochs (death-spiral mitigation)')
    parser.add_argument('--epsilon-init', type=float, default=0.25)
    parser.add_argument('--epsilon-final', type=float, default=0.05)
    parser.add_argument('--epsilon-decay-epochs', type=int, default=0,
                        help='TASK_812: epochs to decay over; 0 = epochs//2')
    parser.add_argument('--assert-postcost', action='store_true',
                        help='TASK_812: hard-assert post-cost reward active (fee>0 & slip>0) and oracle/prediction-loss OFF before launch')
    parser.add_argument('--dirichlet-aggressive', action='store_true')
    
    # Optimizer settings
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr-schedule', type=str, default='constant')
    parser.add_argument('--lr-min', type=float, default=1e-6)
    parser.add_argument('--lr-warmup-epochs', type=int, default=10)
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='AdamW weight decay (L2 regularization)')  # BUG-011
    parser.add_argument('--global-grad-clip', type=float, default=5.0,
                        help='Global gradient norm clipping (default: 5.0)')  # BUG-013
    parser.add_argument('--discount', type=float, default=0.997)
    parser.add_argument('--entropy-coef', type=float, default=0.01)
    parser.add_argument('--per-head-clipping', action='store_true')
    parser.add_argument('--clip-grad-value', type=float, default=1.0)
    parser.add_argument('--clip-grad-policy', type=float, default=1.0)
    parser.add_argument('--clip-grad-dynamics', type=float, default=1.0)
    parser.add_argument('--clip-grad-encoder', type=float, default=1.0)
    
    # Advanced features
    parser.add_argument('--use-gae', action='store_true')
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--no-popart', action='store_false', dest='use_popart')
    parser.set_defaults(use_popart=True)
    parser.add_argument('--popart-beta', type=float, default=0.0001)
    parser.add_argument('--use-prediction-loss', action='store_true')
    parser.add_argument('--physics-only', action='store_true')
    parser.add_argument('--pred-head-size', type=str, default='small')
    parser.add_argument('--prediction-loss-weight', type=float, default=0.2)
    parser.add_argument('--physics-warmup-steps', type=int, default=5000,
                        help='Steps for physics warmup - neural scale anneals 0→1 (default: 5000, 0=disabled)')
    parser.add_argument('--redo', action='store_true')
    parser.add_argument('--redo-interval', type=int, default=10)
    parser.add_argument('--atc-loss', action='store_true')
    parser.add_argument('--atc-weight', type=float, default=0.1)
    
    parser.add_argument('--redo-threshold', type=float, default=0.01)
    parser.add_argument('--redo-init-scale', type=float, default=0.01,
                        help='Initial scale for recycled dormant neurons')
    parser.add_argument('--scheduled-sampling-steps', type=int, default=20000,
                        help='Steps for scheduled sampling warmup')
    parser.add_argument('--atc-k-step', type=int, default=5)
    parser.add_argument('--atc-temperature', type=float, default=0.1)
    parser.add_argument('--atc-num-negatives', type=int, default=32)

    # Cycle Balance (TASK_110: prevent H0 dominance in TRM)
    parser.add_argument('--cycle-balance', action='store_true',
                        help='Enable cycle balance loss to prevent H0 dominance')
    parser.add_argument('--cycle-balance-weight', type=float, default=0.1,
                        help='Weight for cycle balance loss')
    parser.add_argument('--cycle-balance-target-cv', type=float, default=0.3,
                        help='Target coefficient of variation (0.3 = 30%% imbalance allowed)')
    parser.add_argument('--cycle-atc', action='store_true',
                        help='Enable cycle ATC to force H0/H3 diversity')
    parser.add_argument('--cycle-atc-weight', type=float, default=0.05,
                        help='Weight for cycle ATC loss')
    parser.add_argument('--cycle-atc-temperature', type=float, default=0.1,
                        help='Temperature for cycle ATC (lower = stronger push)')

    # Reward shaping
    parser.add_argument('--differential-reward', action='store_true')
    parser.add_argument('--reward-hybrid', action='store_true',
                        help='Mirror the value head recipe (PopArt-normalized target + '
                             'hybrid Huber/categorical loss) for the reward head so it can '
                             'rank actions. Default OFF = canon pure-CE reward loss.')
    parser.add_argument('--reward-attributable', action='store_true',
                        help='TASK_818 Tier B: add realized log-PnL on position CLOSE to the step '
                             'reward (action-attributable, multi-step). Default OFF = canon.')
    parser.add_argument('--bar-minutes', type=int, default=1,
                        help='Resample historical 1-min closes to K-minute bars (1=canon 1-min)')
    parser.add_argument('--differential-baseline', type=str, default='market', choices=['market', 'zero'])
    parser.add_argument('--differential-weight', type=float, default=1.0)
    parser.add_argument('--burning-money', action='store_true')
    parser.add_argument('--burn-market-linked', action='store_true', default=True)
    parser.add_argument('--burn-base-rate', type=float, default=0.00005)
    parser.add_argument('--burn-random', action='store_true', default=True)
    parser.add_argument('--burn-random-prob', type=float, default=0.03)
    parser.add_argument('--adaptive-burn', action='store_true',
                        help='Enable adaptive burn escalation based on hold rate')
    parser.add_argument('--burn-hold-threshold', type=float, default=0.60,
                        help='Hold rate above which triggers burn escalation')
    parser.add_argument('--burn-patience', type=int, default=5,
                        help='Consecutive episodes before burn stage change')
    
    # Interventions
    parser.add_argument('--adaptive-noise', action='store_true')
    parser.add_argument('--target-reward-low', type=float, default=10.0,
                        help='Lower bound of comfortable reward zone for adaptive noise')
    parser.add_argument('--target-reward-high', type=float, default=20.0,
                        help='Upper bound of comfortable reward zone for adaptive noise')
    parser.add_argument('--noise-adjustment-rate', type=float, default=0.05,
                        help='How fast noise changes per update')
    parser.add_argument('--max-temporal-jitter', type=int, default=5,
                        help='Maximum temporal jitter steps for adaptive noise')
    parser.add_argument('--max-time-dilation', type=float, default=0.3,
                        help='Maximum time dilation factor for adaptive noise')
    parser.add_argument('--max-phase-shift', type=float, default=0.5,
                        help='Maximum phase shift probability for adaptive noise')
    parser.add_argument('--adaptive-fees', action='store_true')
    parser.add_argument('--fee-reward-threshold', type=float, default=2.5)
    parser.add_argument('--fee-upgrade-epochs', type=int, default=10)
    parser.add_argument('--smart-allocation', action='store_true')
    parser.add_argument('--jitter-minutes', type=int, default=360)
    parser.add_argument('--sac-entropy', action='store_true',
                        help='Use SAC-style learnable entropy coefficient')
    parser.add_argument('--sac-target-entropy-ratio', type=float, default=0.5,
                        help='Target entropy as ratio of max entropy (default: 0.5)')
    parser.add_argument('--sac-init-alpha', type=float, default=0.01,
                        help='Initial alpha for SAC entropy (default: 0.01)')
    parser.add_argument('--sac-alpha-lr', type=float, default=3e-4,
                        help='Learning rate for SAC alpha (default: 3e-4)')
    parser.add_argument('--adaptive-entropy', action='store_true')
    parser.add_argument('--entropy-hold-threshold', type=float, default=0.70)
    parser.add_argument('--entropy-boost-low', type=float, default=0.15)
    parser.add_argument('--entropy-boost-high', type=float, default=0.30)
    
    # Checkpointing & Logging
    parser.add_argument('--load-checkpoint', type=str, default=None)
    parser.add_argument('--no-checkpoint', action='store_true')
    parser.add_argument('--reset-optimizer', action='store_true',
                        help='Reset optimizer state when loading checkpoint')
    parser.add_argument('--clear-buffer', action='store_true',
                        help='Clear replay buffer when loading checkpoint')
    parser.add_argument('--safe-checkpoints', action='store_true')
    parser.add_argument('--snapshot-every', type=int, default=0,
                        help='Save checkpoint snapshot every N epochs (0=disabled)')
    parser.add_argument('--rollback-patience', type=int, default=50,
                        help='Epochs to wait before considering rollback')
    parser.add_argument('--rollback-threshold', type=float, default=0.5,
                        help='Performance drop threshold to trigger rollback')

    # Generalization evaluation
    parser.add_argument('--eval-every', type=int, default=0,
                        help='Run generalization eval every N epochs (0=disabled)')
    parser.add_argument('--eval-level', type=int, default=4,
                        help='Synthetic level to evaluate on for generalization')
    parser.add_argument('--eval-steps', type=int, default=200,
                        help='Number of steps for evaluation episode')

    parser.add_argument('--ds24-logging', action='store_true')
    parser.add_argument('--ds24-level', type=int, default=0)
    parser.add_argument('--ds24-intermediates-every', type=int, default=10)
    parser.add_argument('--ds24-grad-norms', action='store_true',
                        help='Enable per-head gradient norm tracking (adds overhead)')
    parser.add_argument('--ds24-save-tensors', action='store_true',
                        help='Save tensor snapshots for DS24 visualization')
    parser.add_argument('--ds24-boost-every', type=int, default=10,
                        help='DS24 boost metrics every N epochs')
    parser.add_argument('--ds24-snapshot-every', type=int, default=50,
                        help='DS24 snapshot every N epochs')

    # Physics configuration (TASK_066: extracted magic numbers)
    parser.add_argument('--reward-scale', type=float, default=100.0,
                        help='Scale factor for rewards in categorical bins')
    parser.add_argument('--alpha-target', type=float, default=0.05,
                        help='Physics momentum regularization target')
    parser.add_argument('--beta-target', type=float, default=0.03,
                        help='Physics mean-reversion regularization target')
    parser.add_argument('--sigma-target', type=float, default=0.01,
                        help='Physics volatility regularization target')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='Categorical distribution label smoothing')
    parser.add_argument('--min-entropy', type=float, default=3.0,
                        help='Minimum policy entropy threshold')
    parser.add_argument('--max-positions', type=int, default=16,
                        help='Maximum simultaneous positions')
    parser.add_argument('--min-cash-threshold', type=float, default=100.0,
                        help='Minimum cash to allow buying')
    parser.add_argument('--confidence-threshold', type=float, default=0.0,
                        help='Force HOLD if max policy prob below threshold')

    # Tree Search (H1 - Phase 3 predictive planning)
    parser.add_argument('--use-tree-search', action='store_true',
                        help='Enable predictive tree search for MCTS reward adjustment')
    parser.add_argument('--tree-depth', type=int, default=5,
                        help='Tree search depth (prediction steps)')
    parser.add_argument('--tree-branches', type=int, default=3,
                        help='Tree search branches per node')
    parser.add_argument('--risk-lambda', type=float, default=0.1,
                        help='Uncertainty penalty weight for risk-adjusted rewards')

    # Flow Matching (H2 - TASK_041)
    parser.add_argument('--flow-matching', action='store_true',
                        help='Enable flow matching to prevent identity collapse')
    parser.add_argument('--flow-velocity-weight', type=float, default=0.1,
                        help='Weight for flow velocity regularization')

    # Adaptive Calibration (H3)
    parser.add_argument('--adaptive-calibration', action='store_true',
                        help='Enable adaptive value calibration (auto-tune scale/shift)')

    # Early Graduation (H4)
    parser.add_argument('--early-graduation', type=int, default=0,
                        help='Graduate after N epochs at max difficulty (0=disabled)')

    # Value Loss Type (H5)
    parser.add_argument('--value-loss-type', type=str, default='mse', choices=['mse', 'huber'],
                        help='Value loss type: mse (standard) or huber (robust to outliers)')
    parser.add_argument('--value-huber-delta', type=float, default=1.0,
                        help='Delta parameter for Huber loss')

    # Prediction Loss Type
    parser.add_argument('--pred-loss-type', type=str, default='mse', choices=['mse', 'huber'],
                        help='Prediction loss type: mse (standard) or huber (robust to outliers)')
    parser.add_argument('--pred-huber-delta', type=float, default=1.0,
                        help='Delta parameter for prediction Huber loss')

    # TASK_116: Student-t output distribution for fat-tailed returns
    parser.add_argument('--use-student-t', action='store_true',
                        help='Use Student-t loss for prediction (better handles fat tails in financial data)')
    parser.add_argument('--student-t-weight', type=float, default=0.8,
                        help='Weight for Student-t NLL component (default: 0.8)')
    parser.add_argument('--student-t-mse-weight', type=float, default=0.2,
                        help='Weight for MSE component alongside Student-t (default: 0.2)')
    parser.add_argument('--student-t-df-min', type=float, default=2.1,
                        help='Minimum degrees of freedom for Student-t (>2 for finite variance)')
    parser.add_argument('--student-t-df-max', type=float, default=100.0,
                        help='Maximum degrees of freedom (large = nearly Gaussian)')

    # TASK_118: Alpha-relative reward normalization
    parser.add_argument('--alpha-relative-reward', action='store_true',
                        help='Normalize rewards by max achievable alpha (consistent across regimes)')
    parser.add_argument('--alpha-reward-blend', type=float, default=0.5,
                        help='Blend factor: 0=raw PnL, 1=pure alpha-relative (default: 0.5)')
    parser.add_argument('--alpha-top-k', type=int, default=4,
                        help='Top-k assets for max alpha calculation (position limit proxy)')

    # TASK_119: Optimal hindsight path for alpha-max comparison
    parser.add_argument('--amax', type=str, default=None, choices=['greedy', 'soft', 'sinkhorn'],
                        help='Compute optimal hindsight path for alpha-max metric (greedy/soft/sinkhorn)')
    parser.add_argument('--amax-regret-weight', type=float, default=0.0,
                        help='Weight for regret penalty loss (0=disabled, 0.01-0.1=auxiliary signal)')
    parser.add_argument('--amax-imprint-weight', type=float, default=0.0,
                        help='Weight for optimal action imprint (0=disabled, 0.01-0.02=stabilization)')
    parser.add_argument('--amax-imprint-temp', type=float, default=1.0,
                        help='Temperature for optimal policy targets (lower=more peaked)')

    # TASK_125: Causal Expert Imprinting
    parser.add_argument('--imprint', action='store_true',
                        help='Enable causal expert imprinting (TASK_125)')
    parser.add_argument('--bc-weight-initial', type=float, default=0.1,
                        help='Initial BC loss weight (decays over training)')
    parser.add_argument('--bc-weight-final', type=float, default=0.01,
                        help='Final BC loss weight after decay')
    parser.add_argument('--bc-weight-decay-steps', type=int, default=50000,
                        help='Steps to decay BC weight from initial to final')
    parser.add_argument('--expert-demo-capacity', type=int, default=50000,
                        help='Max expert demos to store (protected, never overwritten)')
    parser.add_argument('--expert-confidence-threshold', type=float, default=0.001,
                        help='Min alpha capture ratio to store as demo (0.001 = 0.1%% of optimal)')
    parser.add_argument('--bc-use-q-filter', action='store_true', default=True,
                        help='Apply BC loss only when Q(demo) > Q(policy)')
    parser.add_argument('--bc-use-large-margin', action='store_true', default=True,
                        help='Use DQfD-style large margin loss instead of cross-entropy')

    # Additional missing flags (L1)
    parser.add_argument('--prediction-loss-initial', type=float, default=0.01,
                        help='Initial prediction loss weight (before annealing)')
    parser.add_argument('--prediction-anneal-steps', type=int, default=10000,
                        help='Steps to anneal prediction loss weight')
    parser.add_argument('--multi-step-k', type=int, default=5,
                        help='Multi-step prediction K')
    parser.add_argument('--physics-reg-weight', type=float, default=0.01,
                        help='Physics regularization weight')
    parser.add_argument('--max-rollbacks', type=int, default=3,
                        help='Maximum checkpoint rollbacks before failing')
    parser.add_argument('--keep-best-n', type=int, default=3,
                        help='Keep N best checkpoints for safe checkpointing')
    parser.add_argument('--lr-decay-on-rollback', type=float, default=0.5,
                        help='LR decay factor after rollback')

    # Extra JAX flags
    parser.add_argument('--jax-env', action='store_true', dest='use_jax_env')
    parser.set_defaults(use_jax_env=False)

    args = parser.parse_args()
    
    # Create config object from args
    # Filter out args that aren't in TrainingConfig or are None
    config_dict = {k: v for k, v in vars(args).items() if hasattr(TrainingConfig, k) and v is not None}
    config = TrainingConfig(**config_dict)
    
    # Keep track of use_jax_env separately as it's not strictly a config but a runner flag
    config.use_jax_env = args.use_jax_env

    # TASK_812: runner flags (not TrainingConfig dataclass fields) — copy explicitly,
    # same pattern as use_jax_env, else the config_dict filter above drops them.
    config.epsilon_schedule = getattr(args, 'epsilon_schedule', False)
    config.epsilon_init = getattr(args, 'epsilon_init', 0.25)
    config.epsilon_final = getattr(args, 'epsilon_final', 0.05)
    config.epsilon_decay_epochs = getattr(args, 'epsilon_decay_epochs', 0)
    config.assert_postcost = getattr(args, 'assert_postcost', False)
    config.reward_hybrid = getattr(args, 'reward_hybrid', False)

    return config
