import os
import warnings

# =============================================================================
# GPU OPTIMIZATION: Must be set BEFORE importing JAX
# =============================================================================
# XLA Autotuning Level 4: Most thorough optimization (slower first compile, fastest runtime)
# Cache directories persist tuning results across runs
_trader_dir = os.path.dirname(os.path.abspath(__file__))
_xla_cache = os.path.join(_trader_dir, '.xla_cache')
_jax_cache = os.path.join(_trader_dir, '.jax_cache')
os.makedirs(_xla_cache, exist_ok=True)
os.makedirs(_jax_cache, exist_ok=True)

# XLA GPU Autotuning: Level 4 + persistent per-fusion cache
# --xla_gpu_per_fusion_autotune_cache_dir: caches autotune results per fusion, grows across runs
# Ref: https://openxla.org/xla/persisted_autotuning
os.environ['XLA_FLAGS'] = (
    f'--xla_gpu_autotune_level=4 '
    f'--xla_gpu_per_fusion_autotune_cache_dir={_xla_cache}'
)

# CUDA async allocator: reduces memory fragmentation
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Suppress TF/XLA verbose logging (dot_search_space, slow_operation_alarm warnings)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warn, 3=error

# JAX compilation cache (for HLO -> compiled code)
os.environ['JAX_COMPILATION_CACHE_DIR'] = _jax_cache

# Suppress FutureWarning from JAX scatter dtype promotion (bf16)
warnings.filterwarnings('ignore', category=FutureWarning, module='jax')

# Now import JAX (picks up env vars)
import jax

# Enable JAX persistent compilation cache with XLA autotuning integration
jax.config.update("jax_compilation_cache_dir", _jax_cache)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)  # Cache everything
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)  # Cache all compiles
import jax.numpy as jnp
import numpy as np
import time
import logging
import argparse

# Import JAX components
from mcts_trader.jax_agent.muzero_model import MuZeroNetworks
from mcts_trader.jax_agent.trainer import MuZeroTrainer, PredictionLossConfig
from mcts_trader.jax_agent.replay_buffer import JAXReplayBuffer
from mcts_trader.jax_agent import VALUE_SUPPORT_SIZE, get_value_support, get_value_support_np
# NOTE: Not using MCTSWorkerPool - it runs MCTS on CPU (slow!)
# Direct GPU MCTS is ~10× faster

# Import environment - VectorBT-based (clean accounting, no FX bugs)
from mcts_trader.env.vectorbt_env import VectorBTTradingEnv
from mcts_trader.training.safe_checkpoint import SafeCheckpointManager
from mcts_trader.training.self_play import (
    run_self_play,
    run_self_play_batched,
    run_self_play_jax,
    N_ACTIONS_PER_ASSET,
)
from mcts_trader.training.evaluator import run_generalization_eval
from mcts_trader.training.env_factory import create_jax_env_and_prices

# Tree Search (Phase 3) - imported conditionally when --use-tree-search is enabled
# from mcts_trader.jax_agent.prediction import (
#     NeuralMarketODE, create_physics_aware_tree_search_fn, TreeSearchConfig
# )

# DS24 Enhanced Metrics Logging - imported conditionally when --ds24-logging is enabled
# from tools.DS24.metrics_logger import DS24MetricsLogger


def main():
    parser = argparse.ArgumentParser(description='JAX MuZero Training')
    parser.add_argument('--name', type=str, default='default', help='Experiment name')
    parser.add_argument('--mode', type=str, default='MARGIN', choices=['SPOT', 'MARGIN'], help='Trading mode')
    parser.add_argument('--fee', type=int, default=0, help='Trading fee in bps (0 for synthetic testing)')
    parser.add_argument('--slippage', type=int, default=0, help='Slippage in bps (0 for synthetic testing)')
    # TASK_047/052: Market realism
    parser.add_argument('--probabilistic-fills', action='store_true',
                       help='Enable probabilistic limit order fills based on distance from market')
    # TASK_055: Borrowing cost for shorts
    parser.add_argument('--borrow-rate', type=float, default=0.0,
                       help='Annual borrowing rate for short positions (e.g., 0.05 = 5%%)')
    # TASK_049: Synthetic noise for anti-memorization
    parser.add_argument('--synthetic-noise', type=float, default=0.0,
                       help='Noise level for synthetic data (0.0-1.0). Higher = harder to memorize patterns')
    # TASK_053: Episode length variance
    parser.add_argument('--episode-variance', type=int, default=0,
                       help='Episode length variance (±N steps). Prevents end-of-episode exploits')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--steps', type=int, default=400, help='Steps per episode (400=6h, 200=3h)')
    parser.add_argument('--config', type=str, help='Training config JSON (dynamic assets)')
    parser.add_argument('--start-date', type=str, help='Start date filter (YYYY-MM-DD or unix ts)')
    parser.add_argument('--end-date', type=str, help='End date filter (YYYY-MM-DD or unix ts)')
    parser.add_argument('--load-checkpoint', type=str, help='Path to checkpoint to resume from (absolute or relative)')
    
    # MCTS Hyperparameters (Maniac Mode)
    parser.add_argument('--sims', type=int, default=32, help='MCTS simulations (32=better planning, 16=faster)')
    parser.add_argument('--sims-warmup', type=int, default=0,
                       help='Epochs to warm up simulations from min to max (0=disabled)')
    parser.add_argument('--sims-min', type=int, default=8,
                       help='Minimum simulations during warmup (default: 8)')
    parser.add_argument('--sims-schedule', type=str, default='linear',
                       choices=['linear', 'step', 'exponential'],
                       help='Sims warmup schedule type (default: linear)')
    parser.add_argument('--noise', type=float, default=0.3, help='Dirichlet noise alpha (Standard MCTS only)')
    parser.add_argument('--noise-frac', type=float, default=0.25, help='Dirichlet noise fraction (Standard MCTS only)')
    parser.add_argument('--gumbel-scale', type=float, default=3.0, help='Gumbel noise scale (Gumbel MCTS) - higher=more exploration')
    parser.add_argument('--value-scale', type=float, default=1.0, help='Value calibration scale (>1.0 expands prediction range)')
    parser.add_argument('--value-shift', type=float, default=0.0, help='Value calibration shift (adjust prediction center)')
    parser.add_argument('--adaptive-calibration', action='store_true', help='Enable adaptive value calibration (auto-tune scale/shift)')

    # Advanced Training Parameters
    parser.add_argument('--batch-size', type=int, default=256, help='Training batch size')
    parser.add_argument('--parallel-envs', type=int, default=4,
                       help='Number of parallel environments for batched self-play (1=serial, 4=4x throughput)')
    parser.add_argument('--jax-env', action='store_true',
                       help='Use JAX-vectorized environment (3000x faster, GPU-native). Requires --synthetic-level.')
    parser.add_argument('--regenerate-data-every', type=int, default=10,
                       help='Regenerate synthetic data every N epochs for training variety (default: 10, 0=never)')
    parser.add_argument('--model-size', type=str, default='small', choices=['small', 'medium', 'large'],
                       help='Model size: small (~7.3M), medium (~15M), large (~36M)')
    parser.add_argument('--state-dim', type=int, default=None, help='Model state dimension (overrides --model-size)')
    parser.add_argument('--synthetic-level', type=int, default=None,
                       help='Use synthetic data (0=sine, 1=lag, 2=dip, 3=trends, None=real)')
    parser.add_argument('--synthetic-steps', type=int, default=10000,
                       help='Steps per synthetic episode')
    parser.add_argument('--schedule-path', type=str, default=None,
                       help='Load from training_schedule.json (random period/run selection)')
    parser.add_argument('--db-path', type=str, default=None,
                       help='Load data from SQLite database')
    parser.add_argument('--cache-path', type=str, default=None,
                       help='Numpy cache for DB data (7min->5sec loading)')
    parser.add_argument('--h-cycles', type=int, default=4, help='TRM Reasoning outer cycles (H)')
    parser.add_argument('--l-cycles', type=int, default=6, help='TRM Reasoning inner cycles (L)')
    parser.add_argument('--confidence-threshold', type=float, default=0.0, help='Minimum policy confidence to execute trade (force Hold otherwise)')
    parser.add_argument('--entropy-coef', type=float, default=0.01, help='Entropy bonus coefficient (higher=more exploration, try 1.0-10.0 for aggressive)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
    # TASK_024: LR schedule
    parser.add_argument('--lr-schedule', type=str, default='constant',
                       choices=['constant', 'cosine', 'linear'],
                       help='LR schedule type (default: constant)')
    parser.add_argument('--lr-min', type=float, default=1e-6,
                       help='Minimum LR for schedule (default: 1e-6)')
    parser.add_argument('--lr-warmup-epochs', type=int, default=10,
                       help='Warmup epochs for LR schedule (default: 10)')
    parser.add_argument('--discount', type=float, default=0.997, help='Discount factor gamma (default: 0.997)')

    # TASK_022: TD(λ)/GAE returns for variance reduction
    parser.add_argument('--use-gae', action='store_true', default=False,
                       help='Use TD(λ)/GAE returns instead of MC returns for variance reduction')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                       help='Lambda parameter for GAE (0=TD(0), 1=MC). Default: 0.95')

    # Adaptive entropy - boost exploration when policy collapses to HOLD
    parser.add_argument('--adaptive-entropy', action='store_true', default=False,
                       help='Enable adaptive entropy: boost when HOLD%% exceeds threshold')
    parser.add_argument('--entropy-hold-threshold', type=float, default=0.70,
                       help='HOLD%% threshold to trigger entropy boost (default: 0.70)')
    parser.add_argument('--entropy-boost-low', type=float, default=0.15,
                       help='Moderate entropy boost when HOLD%% > threshold (default: 0.15)')
    parser.add_argument('--entropy-boost-high', type=float, default=0.30,
                       help='Aggressive entropy boost when HOLD%% > threshold+10%% (default: 0.30)')

    # =========================================================================
    # Stability Interventions (TASK_037-043) - Prevent dual collapse
    # =========================================================================
    # TASK_037: ε-greedy MCTS training target
    parser.add_argument('--epsilon-greedy', type=float, default=0.0,
                       help='Mix uniform into MCTS training target: (1-ε)*π_mcts + ε*uniform. '
                            'Prevents policy collapse. Try 0.1-0.3 (default: 0.0=disabled)')

    # TASK_039: Dirichlet noise (already have --noise and --noise-frac, add recommended preset)
    parser.add_argument('--dirichlet-aggressive', action='store_true', default=False,
                       help='Use aggressive Dirichlet noise (α=0.5, ε=0.5) for trading. '
                            'Overrides --noise and --noise-frac')

    # TASK_040: SAC-style adaptive entropy
    parser.add_argument('--sac-entropy', action='store_true', default=False,
                       help='Use SAC-style learnable entropy coefficient (replaces threshold-based)')
    parser.add_argument('--sac-target-entropy-ratio', type=float, default=0.5,
                       help='Target entropy as ratio of max (0.5=50%% of log(320)≈2.9 nats)')
    parser.add_argument('--sac-alpha-lr', type=float, default=3e-4,
                       help='Learning rate for entropy coefficient (default: 3e-4)')
    parser.add_argument('--sac-init-alpha', type=float, default=0.01,
                       help='Initial entropy coefficient (default: 0.01)')

    # TASK_041: Flow matching for prediction
    parser.add_argument('--flow-matching', action='store_true', default=False,
                       help='Use velocity (flow) matching for prediction loss. '
                            'Prevents identity solution collapse.')
    parser.add_argument('--flow-velocity-weight', type=float, default=0.8,
                       help='Weight for velocity loss vs state loss (default: 0.8)')

    # TASK_038: Differential reward (opportunity cost)
    parser.add_argument('--differential-reward', action='store_true', default=False,
                       help='Use opportunity cost reward: r = r_agent - r_market. '
                            'Makes HOLD costly when market moves.')
    parser.add_argument('--differential-baseline', type=str, default='market',
                       choices=['market', 'zero'],
                       help='Baseline for differential reward (default: market)')
    parser.add_argument('--differential-weight', type=float, default=1.0,
                       help='Weight for market baseline (0=ignore, 1=full, default: 1.0)')

    # TASK_042: ReDo - Recycling Dormant Neurons
    parser.add_argument('--redo', action='store_true', default=False,
                       help='Enable ReDo: recycle dormant neurons to maintain network capacity')
    parser.add_argument('--redo-interval', type=int, default=10,
                       help='Epochs between dormancy checks (default: 10)')
    parser.add_argument('--redo-threshold', type=float, default=0.01,
                       help='Dormancy threshold (CV below this is dormant). Default: 0.01')
    parser.add_argument('--redo-init-scale', type=float, default=0.01,
                       help='Scale for re-initialized weights (default: 0.01)')

    # TASK_043: ATC - Augmented Temporal Contrast
    parser.add_argument('--atc-loss', action='store_true', default=False,
                       help='Enable ATC: contrastive learning for representation health')
    parser.add_argument('--atc-weight', type=float, default=0.1,
                       help='Weight for ATC loss (default: 0.1)')
    parser.add_argument('--atc-k-step', type=int, default=5,
                       help='Temporal distance for positive pairs (default: 5)')
    parser.add_argument('--atc-temperature', type=float, default=0.1,
                       help='Contrastive loss temperature (default: 0.1)')
    parser.add_argument('--atc-num-negatives', type=int, default=32,
                       help='Number of negative samples (default: 32)')

    # Value loss configuration (controls sensitivity to high-variance targets)
    parser.add_argument('--value-loss-type', type=str, default='mse', choices=['mse', 'huber'],
                       help='Value loss type: mse (standard) or huber (robust to outliers)')
    parser.add_argument('--value-huber-delta', type=float, default=1.0,
                       help='Huber loss delta (smaller = more robust, default: 1.0)')

    # PopArt value normalization (TASK_021 - for curriculum stability)
    parser.add_argument('--use-popart', action='store_true', default=False,
                       help='Enable PopArt adaptive value normalization (recommended for curriculum)')
    parser.add_argument('--popart-beta', type=float, default=0.0001,
                       help='PopArt EMA decay rate (lower = more stable, default: 0.0001)')

    # Multi-Action + Limit Orders
    parser.add_argument('--action-points', type=int, default=1,
                       help='Actions per timestep: 1=single action, 4=multi-action')
    parser.add_argument('--tp-sl-enabled', action='store_true', default=False,
                       help='Enable TP/SL guardrails (5%% SL, 10%% TP)')

    # Curriculum transfer flags
    parser.add_argument('--clear-buffer', action='store_true', default=False,
                       help='Clear replay buffer when loading checkpoint (use for new curriculum levels)')
    parser.add_argument('--reset-optimizer', action='store_true', default=False,
                       help='Reset optimizer state when loading checkpoint (fresh momentum for new regime)')

    # =========================================================================
    # Prediction Loss (Phase 2) - Auxiliary price prediction
    # =========================================================================
    parser.add_argument('--use-prediction-loss', action='store_true', default=False,
                       help='Enable auxiliary prediction loss (requires prediction head)')
    parser.add_argument('--physics-only', action='store_true', default=False,
                       help='Use only physics core in prediction (disable neural corrector)')
    parser.add_argument('--pred-head-size', type=str, default='small',
                       choices=['small', 'medium', 'large'],
                       help='Prediction head size: small=128d/2blk, medium=256d/3blk, large=512d/4blk')
    parser.add_argument('--prediction-loss-weight', type=float, default=0.2,
                       help='Final weight for auxiliary prediction loss (default: 0.2)')
    parser.add_argument('--prediction-loss-initial', type=float, default=0.01,
                       help='Initial prediction loss weight (annealed up, default: 0.01)')
    parser.add_argument('--prediction-anneal-steps', type=int, default=10000,
                       help='Steps to anneal prediction loss weight (default: 10000)')
    parser.add_argument('--multi-step-k', type=int, default=5,
                       help='Number of future steps for prediction loss (default: 5)')
    parser.add_argument('--scheduled-sampling-steps', type=int, default=20000,
                       help='Steps to anneal scheduled sampling (default: 20000)')
    parser.add_argument('--physics-reg-weight', type=float, default=0.01,
                       help='Physics regularization weight (default: 0.01)')
    parser.add_argument('--physics-warmup-steps', type=int, default=5000,
                       help='Steps for physics warmup - neural scale anneals 0→1 (default: 5000, 0=disabled)')

    # =========================================================================
    # Tree Search (Phase 3) - Predictive tree search for risk-adjusted rewards
    # =========================================================================
    parser.add_argument('--use-tree-search', action='store_true', default=False,
                       help='Enable predictive tree search for MCTS reward adjustment')
    parser.add_argument('--tree-depth', type=int, default=5,
                       help='Prediction tree depth (default: 5, gives 3^5=243 leaves)')
    parser.add_argument('--tree-branches', type=int, default=3,
                       help='Branches per node (default: 3)')
    parser.add_argument('--risk-lambda', type=float, default=0.1,
                       help='Uncertainty penalty weight for risk-adjusted rewards (default: 0.1)')

    # =========================================================================
    # Mixed Precision (Performance Optimization)
    # =========================================================================
    parser.add_argument('--bf16', action='store_true', default=False,
                       help='Use bfloat16 mixed precision (2x speedup on RTX 30/40 series)')

    # =========================================================================
    # DS24 Enhanced Metrics Logging
    # =========================================================================
    # Level-based logging system:
    #   0 = OFF (no DS24 logging)
    #   1 = CSV only (basic training metrics)
    #   2 = CSV + TRM intermediates + planning metrics (default when --ds24-logging)
    #   3 = Level 2 + physics forward pass (G_momentum, G_reversion, etc.)
    #   4 = Level 3 + latent space heatmaps + weight monitoring
    #   5 = Level 4 + 3D visualization (future, not implemented)
    #   6 = ShroomEngine - Full model introspection (future)
    parser.add_argument('--ds24-level', type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6],
                       help='DS24 logging level: 0=off, 1=csv, 2=+TRM/planning, 3=+physics, 4=+heatmaps, 5=+3D, 6=ShroomEngine')
    parser.add_argument('--ds24-logging', action='store_true', default=False,
                       help='Enable DS24 logging (shortcut for --ds24-level 2)')
    parser.add_argument('--ds24-intermediates-every', type=int, default=10,
                       help='Log TRM intermediate deltas every N epochs (default: 10)')
    parser.add_argument('--ds24-save-tensors', action='store_true', default=False,
                       help='Save periodic tensor snapshots for analysis (every 100 epochs)')
    parser.add_argument('--ds24-grad-norms', action='store_true', default=False,
                       help='Track per-head gradient norms (slower, ~20%% overhead)')

    # TASK_025: Per-head gradient clipping
    parser.add_argument('--per-head-clipping', action='store_true', default=False,
                       help='Enable per-head gradient clipping (prevents one head dominating)')
    parser.add_argument('--clip-grad-value', type=float, default=1.0,
                       help='Max gradient norm for value head (default: 1.0)')
    parser.add_argument('--clip-grad-policy', type=float, default=1.0,
                       help='Max gradient norm for policy head (default: 1.0)')
    parser.add_argument('--clip-grad-dynamics', type=float, default=1.0,
                       help='Max gradient norm for dynamics network (default: 1.0)')
    parser.add_argument('--clip-grad-encoder', type=float, default=1.0,
                       help='Max gradient norm for encoder (default: 1.0)')

    # Dynamic level boosting (reduces overhead while getting periodic deep insights)
    parser.add_argument('--ds24-boost-every', type=int, default=10,
                       help='Boost to DS24 L4 every N epochs (default: 10, 0=disabled)')
    parser.add_argument('--ds24-snapshot-every', type=int, default=50,
                       help='Full DS24 L6 snapshot every N epochs (default: 50, 0=disabled)')

    # =========================================================================
    # Safe Checkpoint Management (for advanced curriculum levels)
    # =========================================================================
    parser.add_argument('--safe-checkpoints', action='store_true', default=False,
                       help='Enable safe checkpoint management (saves only on improvement, rollback on degradation)')
    parser.add_argument('--rollback-patience', type=int, default=50,
                       help='Epochs to wait before considering rollback (default: 50)')
    parser.add_argument('--rollback-threshold', type=float, default=0.5,
                       help='Performance drop to trigger rollback (default: 0.5)')
    parser.add_argument('--max-rollbacks', type=int, default=3,
                       help='Maximum rollbacks allowed before failing (default: 3)')
    parser.add_argument('--keep-best-n', type=int, default=3,
                       help='Number of best checkpoints to keep (default: 3)')
    parser.add_argument('--lr-decay-on-rollback', type=float, default=0.5,
                       help='Learning rate decay factor after rollback (default: 0.5)')
    parser.add_argument('--no-checkpoint', action='store_true', default=False,
                       help='Disable checkpoint saving (for sweeps/experiments)')

    # =========================================================================
    # Generalization Evaluation (Curriculum Training Metrics)
    # =========================================================================
    # Periodically evaluate on harder levels to track generalization during curriculum
    parser.add_argument('--eval-every', type=int, default=0,
                       help='Run evaluation on --eval-level every N epochs (0=disabled)')
    parser.add_argument('--eval-level', type=int, default=4,
                       help='Synthetic level for periodic evaluation (default: 4=rotation)')
    parser.add_argument('--eval-steps', type=int, default=200,
                       help='Steps per evaluation episode (default: 200)')
    parser.add_argument('--snapshot-every', type=int, default=0,
                       help='Generate comprehensive DS24 L4 snapshot every N epochs (0=disabled)')

    args = parser.parse_args()

    # Resolve DS24 logging level (--ds24-logging is shortcut for --ds24-level 2)
    if args.ds24_logging and args.ds24_level == 0:
        args.ds24_level = 2
    # Create convenient level checks
    ds24_level = args.ds24_level

    # Map model size to state_dim if not explicitly provided
    if args.state_dim is None:
        if args.model_size == 'small':
            args.state_dim = 64
        elif args.model_size == 'medium':
            args.state_dim = 96
        elif args.model_size == 'large':
            args.state_dim = 160
    
    # Auto-adjust batch size for model size to prevent OOM on 24GB VRAM
    if args.model_size == 'large' and args.batch_size > 512:
        logging.warning(f"Large model + batch {args.batch_size} might OOM. Reducing to 512.")
        args.batch_size = 512
    elif args.model_size == 'medium' and args.batch_size > 384:
        logging.warning(f"Medium model + batch {args.batch_size} might OOM. Reducing to 384.")
        args.batch_size = 384

    # Dynamic Asset Configuration
    n_assets = 64
    asset_symbols = None
    min_coverage = 0.1  # Default 10%
    start_date = args.start_date
    end_date = args.end_date
    
    if args.config:
        import json
        with open(args.config) as f:
            config = json.load(f)
            if 'n_assets' in config:
                n_assets = config['n_assets']
            if 'asset_symbols' in config:
                asset_symbols = config['asset_symbols']
            if 'experiment_name' in config:
                args.name = config['experiment_name']
            if 'min_coverage' in config:
                min_coverage = config.get('min_coverage', 0.1)
            # Config overrides CLI args if present
            if 'start_date' in config:
                start_date = config['start_date']
            if 'end_date' in config:
                end_date = config['end_date']
                
            print(f"🔧 Loaded config: {args.config} (n_assets={n_assets}, coverage={min_coverage:.2f})")

    # Calculate min_required_assets
    min_required_assets = max(1, int(n_assets * min_coverage))
    print(f"DEBUG: Min required assets set to {min_required_assets} ({min_coverage*100:.1f}% coverage)")

    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join('logs', f"training_fast_{args.name}.log")), # Log per experiment
            logging.StreamHandler()
        ]
    )

    print("="*70)
    print(f"JAX TRADING AGENT | {args.mode} | Fee: {args.fee}bps | Slip: {args.slippage}bps | {n_assets} Assets")
    print("="*70)
    print(f"GPU Optimization: XLA autotune=4 | cache={_xla_cache}")
    print()

    print("[1/4] Initializing Environment...")

    # VectorBT-based environment (clean accounting, no FX bugs)
    # Data source priority: synthetic > schedule > database > error
    env_kwargs = dict(
        n_assets=n_assets,
        initial_cash=10000.0,
        trading_fee_bps=args.fee,
        slippage_bps=args.slippage,
        max_steps=args.steps,
        # Multi-Action + Limit Orders
        action_points=args.action_points,  # 1=single, 4=multi-action
        tp_sl_enabled=args.tp_sl_enabled,  # TP/SL guardrails
    )

    if args.synthetic_level is not None:
        env_kwargs['synthetic_level'] = args.synthetic_level
        env_kwargs['synthetic_n_steps'] = args.synthetic_steps
        data_source = f"synthetic level {args.synthetic_level}"
    elif hasattr(args, 'schedule_path') and args.schedule_path:
        env_kwargs['schedule_path'] = args.schedule_path
        if hasattr(args, 'db_path') and args.db_path:
            env_kwargs['db_path'] = args.db_path
        data_source = "training schedule"
    elif args.db_path:
        env_kwargs['db_path'] = args.db_path
        if args.cache_path:
            env_kwargs['cache_path'] = args.cache_path
        if args.start_date:
            env_kwargs['start_date'] = args.start_date
        if args.end_date:
            env_kwargs['end_date'] = args.end_date
        data_source = f"database {args.db_path}"
    else:
        # Default to synthetic level 0 if no data source specified
        env_kwargs['synthetic_level'] = 0
        env_kwargs['synthetic_n_steps'] = args.synthetic_steps
        data_source = "synthetic level 0 (default)"

    env = VectorBTTradingEnv(**env_kwargs)

    # Create additional parallel environments if requested
    parallel_envs = []
    if args.parallel_envs > 1:
        parallel_envs = [env]  # First env already created
        for i in range(args.parallel_envs - 1):
            # Create additional envs with different random seeds via reset
            parallel_envs.append(VectorBTTradingEnv(**env_kwargs))
        print(f"   Created {args.parallel_envs} parallel environments for batched self-play")

    # Update n_assets to match what the environment actually loaded
    n_assets = env.n_assets
    action_space_size = n_assets * N_ACTIONS_PER_ASSET  # [Sell, Hold, Buy, Limit Sell, Limit Buy]

    multi_action_mode = f"{args.action_points} action points" if args.action_points > 1 else "single action"
    parallel_mode = f", {args.parallel_envs}x parallel" if args.parallel_envs > 1 else ""
    print(f"✅ Environment ready (VectorBT, {args.mode} mode, {n_assets} assets, {args.steps} steps/episode{parallel_mode})")
    print(f"   5-action space (limit orders), {multi_action_mode}, TP/SL={'ON' if args.tp_sl_enabled else 'OFF'}")
    print(f"DEBUG: n_assets={n_assets}, total_actions={action_space_size}")

    # JAX Environment Setup (Phase 4 - 3000x faster)
    jax_env_states = None
    jax_prices = None
    jax_env_config = None
    use_jax_env = args.jax_env

    if use_jax_env:
        if args.synthetic_level is None:
            logging.warning("--jax-env requires --synthetic-level. Falling back to Python env.")
            use_jax_env = False
        else:
            print(f"   Initializing JAX-vectorized environment ({args.parallel_envs} parallel envs)...")
            jax_env_states, jax_prices, jax_env_config = create_jax_env_and_prices(
                synthetic_level=args.synthetic_level,
                n_envs=args.parallel_envs,
                n_assets=n_assets,
                n_steps=args.synthetic_steps,
                fee_bps=float(args.fee),
                slippage_bps=float(args.slippage),
                max_episode_steps=args.steps,
                window_size=60,
                position_size_pct=0.1,
                rng_key=jax.random.PRNGKey(42),
                # TASK_047/052: Market realism
                probabilistic_fills=args.probabilistic_fills,
                # TASK_055: Borrowing cost
                borrow_rate_annual=args.borrow_rate,
                # TASK_049: Synthetic noise
                synthetic_noise=args.synthetic_noise,
                # TASK_053: Episode variance
                episode_variance=args.episode_variance,
            )
            print(f"   ✅ JAX env ready: {args.parallel_envs} envs, {jax_prices.shape[0]} price steps")

    print("[2/4] Initializing JAX model...")
    rng = jax.random.PRNGKey(42)

    # Mixed precision setup
    compute_dtype = jnp.bfloat16 if args.bf16 else jnp.float32
    dtype_str = "bfloat16" if args.bf16 else "float32"

    state_dim = args.state_dim

    # Prediction head size configuration
    PRED_HEAD_SIZES = {
        'small':  {'embed_dim': 128, 'num_blocks': 2, 'mlp_dim': 256, 'n_factors': 8},
        'medium': {'embed_dim': 256, 'num_blocks': 3, 'mlp_dim': 512, 'n_factors': 16},
        'large':  {'embed_dim': 512, 'num_blocks': 4, 'mlp_dim': 1024, 'n_factors': 32},
    }
    pred_cfg = PRED_HEAD_SIZES[args.pred_head_size]

    model = MuZeroNetworks(
        state_dim=state_dim,
        action_space_size=action_space_size,  # n_assets * N_ACTIONS_PER_ASSET
        support_size=VALUE_SUPPORT_SIZE,  # Central config (751 bins over [-750, 750])
        H_cycles=args.h_cycles,
        L_cycles=args.l_cycles,
        # Prediction head (Phase 2)
        use_prediction_head=args.use_prediction_loss,
        physics_only=args.physics_only,  # Disable neural corrector for pure physics
        n_assets=n_assets,
        n_features=10,  # External features dimension
        # Prediction head size
        pred_embed_dim=pred_cfg['embed_dim'],
        pred_num_blocks=pred_cfg['num_blocks'],
        pred_mlp_dim=pred_cfg['mlp_dim'],
        pred_n_factors=pred_cfg['n_factors'],
        # Mixed precision
        dtype=compute_dtype,
        param_dtype=jnp.float32,  # Always keep params in float32 for stability
    )
    print(f"DEBUG: Model definition: state_dim={model.state_dim}, size={args.model_size}, actions={model.action_space_size}")
    print(f"DEBUG: Compute dtype: {dtype_str} ({'TensorCore enabled' if args.bf16 else 'standard precision'})")
    print(f"DEBUG: Entropy coefficient: {args.entropy_coef} (higher=more exploration)")
    if args.use_prediction_loss:
        mode_str = "PHYSICS-ONLY" if args.physics_only else "HYBRID (physics+neural)"
        print(f"DEBUG: Prediction head ENABLED (λ_pred: {args.prediction_loss_initial} -> {args.prediction_loss_weight}, mode: {mode_str}, size: {args.pred_head_size})")
        if args.physics_warmup_steps > 0:
            print(f"DEBUG: Physics warmup ENABLED - neural scale 0→1 over {args.physics_warmup_steps} steps (TASK_035)")
    if args.use_popart:
        print(f"DEBUG: PopArt normalization ENABLED (β={args.popart_beta})")

    # Create prediction loss config if enabled
    prediction_config = None
    if args.use_prediction_loss:
        prediction_config = PredictionLossConfig(
            weight_initial=args.prediction_loss_initial,
            weight_final=args.prediction_loss_weight,
            anneal_steps=args.prediction_anneal_steps,
            multi_step_k=args.multi_step_k,
            scheduled_sampling_steps=args.scheduled_sampling_steps,
            physics_reg_weight=args.physics_reg_weight,
            physics_warmup_steps=args.physics_warmup_steps,  # TASK_035
        )

    trainer = MuZeroTrainer(
        model=model,
        learning_rate=args.lr,
        num_simulations=args.sims,  # Dynamic sims
        batch_size=args.batch_size,      # Dynamic batch size
        unroll_steps=5,
        td_steps=10,
        discount=args.discount,  # Discount factor for TD targets
        value_loss_weight=2.0,  # Increased but not excessive (was 1.0, tried 5.0)
        policy_loss_weight=1.0,
        reward_loss_weight=1.0,
        entropy_coef=args.entropy_coef,  # Entropy bonus to prevent policy collapse
        # Value loss configuration
        value_loss_type=args.value_loss_type,
        value_huber_delta=args.value_huber_delta,
        # TASK_024: LR schedule
        lr_schedule_type=args.lr_schedule,
        lr_min=args.lr_min,
        lr_warmup_epochs=args.lr_warmup_epochs,
        total_epochs=args.epochs,
        steps_per_epoch=100,  # Approximate; depends on buffer size / batch size
        # Prediction loss (Phase 2)
        use_prediction_loss=args.use_prediction_loss,
        prediction_config=prediction_config,
        # DS24 gradient monitoring
        track_grad_norms=args.ds24_grad_norms,
        # PopArt value normalization (TASK_021)
        use_popart=args.use_popart,
        popart_beta=args.popart_beta,
        # TASK_025: Per-head gradient clipping
        per_head_clipping=args.per_head_clipping,
        clip_grad_value=args.clip_grad_value,
        clip_grad_policy=args.clip_grad_policy,
        clip_grad_dynamics=args.clip_grad_dynamics,
        clip_grad_encoder=args.clip_grad_encoder,
    )

    rng, init_rng = jax.random.split(rng)
    dummy_obs = jnp.ones((1, 27, n_assets, n_assets))
    state = trainer.init_train_state(init_rng, dummy_obs)

    # Resume from checkpoint if provided
    epoch_offset = 0  # Track total epochs for CSV continuity
    if args.load_checkpoint:
        # Resolve to absolute path for Orbax
        abs_ckpt_path = os.path.abspath(args.load_checkpoint)
        print(f"📥 Loading checkpoint from: {args.load_checkpoint}")
        print(f"   (Resolved absolute path: {abs_ckpt_path})")

        try:
            # MuZeroTrainer.restore_checkpoint returns the restored state
            restored_state = trainer.restore_checkpoint(abs_ckpt_path)
            if restored_state:
                trainer.state = restored_state
                state = restored_state # Update local reference
                # Calculate epoch offset from step count
                # Assuming ~50 steps per epoch (200 env steps / 4)
                epoch_offset = state.step_count // 50
                print(f"✅ Checkpoint loaded successfully! (Step: {state.step_count}, Epoch offset: {epoch_offset})")

                # Reset optimizer state if requested (for curriculum transfer)
                if args.reset_optimizer:
                    print("🔄 Resetting optimizer state for new regime...")
                    trainer.state = trainer.reset_optimizer_state(trainer.state)
                    state = trainer.state
                    print("   (Momentum and variance reset, model weights preserved)")
            else:
                print(f"⚠️ Failed to load checkpoint (trainer returned None)")
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            # Do not crash, continue from scratch if load fails? Or crash?
            # Crashing is safer to avoid accidental scratch training
            raise e

    # Create search with Maniac params
    # PARALLEL MCTS INTEGRATION
    
    # Report model size
    if state_dim == 64:
        param_estimate = "~7.3M"
    elif state_dim == 96:
        param_estimate = "~15M"
    elif state_dim == 160:
        param_estimate = "~36M"
    else:
        param_estimate = "custom"
    print(f"Model size: state_dim={state_dim} ({param_estimate} params)")
    
    # TASK_039: Apply aggressive Dirichlet if requested
    dirichlet_alpha = args.noise
    exploration_fraction = args.noise_frac
    if args.dirichlet_aggressive:
        dirichlet_alpha = 0.5
        exploration_fraction = 0.5
        print(f"  Using aggressive Dirichlet noise: α={dirichlet_alpha}, ε={exploration_fraction}")

    search_kwargs = {
        'num_simulations': args.sims,
        'use_gumbel': True,
        'gumbel_scale': args.gumbel_scale,
        'dirichlet_alpha': dirichlet_alpha,
        'exploration_fraction': exploration_fraction,
        'value_scale': args.value_scale,
        'value_shift': args.value_shift,
        'epsilon_greedy': args.epsilon_greedy,  # TASK_037
    }
    print(f"MCTS Search: {args.sims} sims | Gumbel scale: {args.gumbel_scale} | Use Gumbel: True")
    if args.epsilon_greedy > 0:
        print(f"  ε-greedy training target: {args.epsilon_greedy:.0%} uniform mixed in")
    if args.value_scale != 1.0 or args.value_shift != 0.0:
        print(f"Value Calibration: scale={args.value_scale}, shift={args.value_shift}")
    
    print(f"🚀 Initializing GPU-native MCTS...")
    # Use direct GPU MCTS instead of CPU worker pool
    # Worker pool runs MCTS on CPU (slow!), direct call runs on GPU (fast!)
    from mcts_trader.jax_agent.mctx_search import create_mctx_search
    search = create_mctx_search(model, **search_kwargs)
    search = jax.jit(search)  # JIT compile for GPU

    print(f"   Maniac Mode: Sims={args.sims}, Noise={args.noise}, Frac={args.noise_frac}")

    # =========================================================================
    # Tree Search (Phase 3) - Predictive tree search for uncertainty metrics
    # =========================================================================
    tree_search_fn = None
    risk_adjust_fn = None
    tree_search_uses_muzero_prediction = False  # Track if we're using trained model

    if args.use_tree_search:
        tree_mode = "PHYSICS-ONLY" if args.physics_only else "HYBRID"
        print(f"🌳 Initializing Predictive Tree Search ({tree_mode})...")
        from mcts_trader.jax_agent.prediction import (
            NeuralMarketODE,
            create_physics_aware_tree_search_fn,
            create_mcts_reward_adjuster,
            create_dynamic_reward_adjuster,
            TreeSearchConfig,
        )

        tree_config = TreeSearchConfig(
            depth=args.tree_depth,
            branches=args.tree_branches,
            store_trajectory=False,
        )
        n_leaves = args.tree_branches ** args.tree_depth

        # Check if we can use MuZero's TRAINED prediction model
        # Use the config flag instead of hasattr (Flax modules don't expose submodules as attrs)
        if args.use_prediction_loss and model.use_prediction_head:
            # Use MuZero's prediction model (will be trained during training)
            print(f"   Using MuZero's trained prediction model (dynamic params)")
            tree_search_uses_muzero_prediction = True

            # Create a standalone model with same config for tree search
            # (We'll pass MuZero's trained params at runtime)
            tree_pred_model = NeuralMarketODE(
                n_assets=n_assets,
                n_features=10,
                use_transformer=True,
                use_action_conditioning=True,  # Match MuZero's config
                physics_only=args.physics_only,  # Match physics-only mode
            )

            # Create dynamic reward adjuster that takes params at runtime
            risk_adjust_fn = create_dynamic_reward_adjuster(
                tree_pred_model,
                tree_config,
                risk_lambda=args.risk_lambda
            )
            print(f"   Depth={args.tree_depth}, Branches={args.tree_branches} → {n_leaves} leaves")
            print(f"   Risk λ={args.risk_lambda} (uncertainty penalty weight)")
            print(f"   ⚡ Params will update with training!")
        else:
            # Standalone (untrained) prediction model - less effective but still works
            if args.use_prediction_loss:
                print(f"   ⚠️ use_prediction_head not enabled, using standalone model")
            else:
                print(f"   ⚠️ Using untrained prediction model (add --use-prediction-loss for better results)")

            pred_model = NeuralMarketODE(
                n_assets=n_assets,
                n_features=10,
                use_transformer=True,
                use_action_conditioning=False,
                physics_only=args.physics_only,  # Match physics-only mode
            )

            rng, pred_init_rng = jax.random.split(rng)
            dummy_prices = jnp.zeros((1, n_assets))
            pred_params = pred_model.init(
                pred_init_rng,
                dummy_prices,
                dummy_prices,
                dummy_prices,
                dummy_prices,
            )

            tree_search_fn = create_physics_aware_tree_search_fn(pred_model, pred_params, tree_config)
            risk_adjust_fn = create_mcts_reward_adjuster(tree_search_fn, risk_lambda=args.risk_lambda)

            print(f"   Depth={args.tree_depth}, Branches={args.tree_branches} → {n_leaves} leaves")
            print(f"   Risk λ={args.risk_lambda} (uncertainty penalty weight)")
    print()

    print("[3/4] Initializing replay buffer...")
    # Reduce buffer size for multi-instance training (5K per instance = 10K total with 2 instances)
    buffer_capacity = 5000
    replay_buffer = JAXReplayBuffer(
        capacity=buffer_capacity,
        observation_shape=(27, n_assets, n_assets),
        action_size=action_space_size,  # n_assets * N_ACTIONS_PER_ASSET
        batch_size=args.batch_size,
        sequence_length=5,
        # Auxiliary data for prediction loss (Phase 2)
        n_assets=n_assets,
        n_features=10,
    )
    print(f"✅ Buffer ready ({buffer_capacity} capacity, float32)")
    if args.use_prediction_loss:
        print(f"   Prediction loss auxiliary storage enabled (n_assets={n_assets}, n_features=10)")

    # Clear buffer if requested (for curriculum transfer to new level)
    if args.clear_buffer and args.load_checkpoint:
        print("🧹 Clearing replay buffer for new regime...")
        replay_buffer.clear()
        print(f"   Buffer cleared: {len(replay_buffer)} trajectories")
    print()

    print("[4/4] Warmup (JIT compilation)...")
    warmup_start = time.time()
    rng, warmup_rng = jax.random.split(rng)
    _, _ = run_self_play(env, search, model.apply, trainer.state.params, warmup_rng, steps=5, support_size=model.support_size, discount=args.discount)
    print(f"✅ Warmup done ({time.time() - warmup_start:.1f}s)")
    print()

    print("="*70)
    print("TRAINING LOOP")
    print("="*70)
    print()

    checkpoint_dir = os.path.abspath(f"jax_ckpt_fast/{args.name}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Track metrics for scheduler and mastery check
    recent_rewards = []
    all_rewards = []  # Full history for mastery check

    # CSV logging for dashboard
    import csv
    csv_log_path = f'logs/training_{args.name}.csv'

    # DS24 Enhanced Metrics Logger (optional, level-based)
    ds24_hooks = None
    ds24_columns = []
    if ds24_level >= 1:
        from tools.DS24.training_hooks import create_ds24_hooks
        ds24_hooks = create_ds24_hooks(args, model=model)
        if ds24_hooks:
            ds24_columns = ds24_hooks.get_ds24_columns()
            print(f"📈 DS24 {ds24_hooks.get_level_description()}")
        if args.ds24_save_tensors:
            print(f"   Tensor snapshots → tensor_snapshots/{args.name}/")
        if args.ds24_grad_norms:
            print(f"   ⚠️ Gradient norm tracking enabled (~20% overhead)")

    # Base CSV columns
    base_columns = [
        # Core metrics
        'epoch', 'reward', 'reward_ma20', 'fee_bps', 'steps',
        # Market baseline (alpha = agent - market, positive = beating market)
        'alpha', 'market_return', 'agent_return',
        # Action counts (5-action space: S/H/B/LS/LB)
        'n_buy', 'n_hold', 'n_sell', 'n_limit_sell', 'n_limit_buy',
        # Action percentages
        'pct_buy', 'pct_hold', 'pct_sell', 'pct_limit_sell', 'pct_limit_buy',
        # Timing
        'self_play_time', 'train_time',
        # Loss components
        'total_loss', 'value_loss', 'value_mse', 'value_cat', 'policy_loss', 'reward_loss',
        # Prediction loss (Phase 2)
        'pred_loss', 'lambda_pred', 'physics_reg',
        # Policy/Value analytics
        'policy_entropy', 'avg_value', 'value_std',
        # Value calibration
        'value_scale', 'value_shift',
        # TASK_031: PopArt normalization stats (for denormalizing logged values)
        'popart_mean', 'popart_std',
        # Tree search (Phase 3)
        'uncertainty',
        # Config reference
        'synthetic_level', 'model_size', 'run_id',
        # Generalization evaluation (curriculum training)
        'eval_level', 'eval_reward', 'eval_alpha', 'eval_pct_hold',
        # TASK_022: GAE returns config
        'use_gae', 'gae_lambda',
        # TASK_025: Per-head gradient clipping
        'per_head_clipping', 'clip_value', 'clip_policy', 'clip_dynamics', 'clip_encoder',
        # Exploration: epsilon-greedy
        'epsilon_greedy',
    ]

    # Combine base + DS24 columns
    all_csv_columns = base_columns + ds24_columns

    # Append mode if resuming from checkpoint, otherwise create new
    csv_exists = os.path.exists(csv_log_path)
    if args.load_checkpoint and csv_exists:
        csv_file = open(csv_log_path, 'a', newline='')
        csv_writer = csv.writer(csv_file)
        print(f"📊 Appending training metrics to: {csv_log_path} (resumed at epoch {epoch_offset})")
    else:
        csv_file = open(csv_log_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(all_csv_columns)
        csv_file.flush()
        print(f"📊 Logging training metrics to: {csv_log_path}")

    # Adaptive value calibration tracking
    calibration_values = []  # Predicted values
    calibration_rewards = []  # Actual rewards
    adaptive_calibration = args.adaptive_calibration
    calibration_interval = 25  # Recompute scale every N epochs
    calibration_start_epoch = 50  # Start after this many epochs
    if adaptive_calibration:
        print(f"📐 Adaptive calibration enabled (starts epoch {calibration_start_epoch}, interval {calibration_interval})")

    # MCTS sims warmup schedule (TASK_020)
    sims_schedule = None
    if args.sims_warmup > 0:
        from mcts_trader.jax_agent.schedules import SimsSchedule
        sims_schedule = SimsSchedule(
            min_sims=args.sims_min,
            max_sims=args.sims,
            warmup_epochs=args.sims_warmup,
            schedule_type=args.sims_schedule
        )
        print(f"🎯 Sims warmup enabled: {args.sims_min} → {args.sims} over {args.sims_warmup} epochs ({args.sims_schedule})")

    # Safe checkpoint management (for advanced curriculum levels)
    safe_ckpt_manager = None
    rollback_count = 0
    if args.safe_checkpoints:
        safe_ckpt_manager = SafeCheckpointManager(
            checkpoint_dir=os.path.join(checkpoint_dir, 'safe'),
            metric_name='avg_reward',
            patience=args.rollback_patience,
            rollback_threshold=args.rollback_threshold,
            keep_best_n=args.keep_best_n,
            rolling_window=20,
            min_improvement=0.01,
            save_golden=True,
        )
        print(f"🛡️ Safe checkpoint management enabled:")
        print(f"   Patience: {args.rollback_patience}, Threshold: {args.rollback_threshold}")
        print(f"   Max rollbacks: {args.max_rollbacks}, Keep best: {args.keep_best_n}")

    # Generalization evaluation tracking (curriculum training)
    last_eval_result = {'eval_level': '', 'eval_reward': '', 'eval_alpha': '', 'eval_pct_hold': ''}
    if args.eval_every > 0:
        print(f"📊 Generalization eval: Every {args.eval_every} epochs on synthetic L{args.eval_level}")

    # TASK_042: ReDo dormancy monitor
    dormancy_monitor = None
    if args.redo:
        from mcts_trader.jax_agent.redo import DormancyMonitor
        dormancy_monitor = DormancyMonitor(
            dormancy_threshold=args.redo_threshold,
            check_interval=args.redo_interval,
            min_samples=1000,
        )
        print(f"🔄 ReDo enabled: check every {args.redo_interval} epochs, threshold={args.redo_threshold}")

    try:
        for epoch in range(args.epochs):
            # Update MCTS sims if warmup schedule active
            if sims_schedule is not None:
                current_sims = sims_schedule.get_sims(epoch)
                trainer.update_simulations(current_sims)
            else:
                current_sims = args.sims

            # Self-play
            start = time.time()
            rng, search_rng = jax.random.split(rng)

            # Extract prediction model params for dynamic tree search
            pred_params_for_tree = None
            if tree_search_uses_muzero_prediction and risk_adjust_fn is not None:
                # Get prediction model params from MuZero's params
                # Path: trainer.state.params['price_predictor'] (no extra 'params' level)
                try:
                    pred_params_for_tree = {'params': trainer.state.params['price_predictor']}
                except (KeyError, TypeError) as e:
                    # Log the path error once
                    if epoch == 0:
                        print(f"⚠️ Could not extract price_predictor params: {e}")
                        print(f"   Available keys: {list(trainer.state.params.keys())[:10]}...")
                    # Create dummy params for the dynamic adjuster (won't be optimal but won't crash)
                    pred_params_for_tree = {'params': {}}

            # Regenerate synthetic data periodically for training variety (~16ms, essentially free)
            if use_jax_env and args.regenerate_data_every > 0 and epoch > 0 and epoch % args.regenerate_data_every == 0:
                rng, regen_rng = jax.random.split(rng)
                jax_env_states, jax_prices, jax_env_config = create_jax_env_and_prices(
                    synthetic_level=args.synthetic_level,
                    n_envs=args.parallel_envs,
                    n_assets=n_assets,
                    n_steps=args.synthetic_steps,
                    fee_bps=float(args.fee),
                    slippage_bps=float(args.slippage),
                    max_episode_steps=args.steps,
                    window_size=60,
                    position_size_pct=0.1,
                    rng_key=regen_rng,
                    # TASK_047/052: Market realism
                    probabilistic_fills=args.probabilistic_fills,
                    # TASK_055: Borrowing cost
                    borrow_rate_annual=args.borrow_rate,
                    # TASK_049: Synthetic noise
                    synthetic_noise=args.synthetic_noise,
                    # TASK_053: Episode variance
                    episode_variance=args.episode_variance,
                )
                if epoch % 50 == 0:  # Log occasionally
                    print(f"   🔄 Regenerated synthetic data (epoch {epoch})")

            # Use JAX-vectorized env (3000x faster) or batched Python env
            if use_jax_env:
                # Reset JAX env states for new episode
                from mcts_trader.env.jax_trading_env import jit_batch_reset
                reset_keys = jax.random.split(search_rng, args.parallel_envs)
                jax_env_states = jit_batch_reset(reset_keys, jax_prices, jax_env_config)

                # TASK_031: Get PopArt state for value denormalization
                popart_state = trainer.popart.state if trainer.use_popart and trainer.popart else None

                trajectories, scores = run_self_play_jax(
                    jax_env_states,
                    jax_prices,
                    jax_env_config,
                    search,
                    model.apply,
                    trainer.state.params,
                    search_rng,
                    steps=args.steps,
                    confidence_threshold=args.confidence_threshold,
                    support_size=model.support_size,
                    discount=args.discount,
                    popart_state=popart_state,
                    use_gae=args.use_gae,
                    gae_lambda=args.gae_lambda,
                )
                # Use average score for display
                score = np.mean(scores)
                trajectory = trajectories[0]  # Use first for action stats below
            elif args.parallel_envs > 1:
                # TASK_031: Get PopArt state for value denormalization
                popart_state = trainer.popart.state if trainer.use_popart and trainer.popart else None

                trajectories, scores = run_self_play_batched(
                    parallel_envs,
                    search,
                    model.apply,
                    trainer.state.params,
                    search_rng,
                    steps=args.steps,
                    confidence_threshold=args.confidence_threshold,
                    support_size=model.support_size,
                    collect_prices=args.use_prediction_loss,
                    discount=args.discount,
                    popart_state=popart_state,
                    use_gae=args.use_gae,
                    gae_lambda=args.gae_lambda,
                )
                # Use average score for display, add all trajectories
                score = np.mean(scores)
                trajectory = trajectories[0]  # Use first for action stats below
            else:
                # TASK_031: Get PopArt state for value denormalization
                popart_state = trainer.popart.state if trainer.use_popart and trainer.popart else None

                trajectory, score = run_self_play(
                    env,
                    search,
                    model.apply,
                    trainer.state.params,
                    search_rng,
                    steps=args.steps,
                    confidence_threshold=args.confidence_threshold,
                    support_size=model.support_size,
                    collect_prices=args.use_prediction_loss,  # Phase 2
                    risk_adjust_fn=risk_adjust_fn,  # Phase 3
                    pred_params=pred_params_for_tree,  # Dynamic tree search params
                    discount=args.discount,
                    popart_state=popart_state,
                    use_gae=args.use_gae,
                    gae_lambda=args.gae_lambda,
                )
                trajectories = [trajectory]  # Wrap in list for uniform handling
                scores = [score]

            self_play_time = time.time() - start

            # Track reward (use average for batched)
            all_rewards.append(score)  # All rewards for metrics
            recent_rewards.append(score)  # Recent for display
            if len(recent_rewards) > 20:
                recent_rewards.pop(0)

            # Add all trajectories to buffer
            for traj_idx, traj in enumerate(trajectories):
                # Sanity check for NaNs in trajectory (only for numpy arrays)
                for key, arr in traj.items():
                    if isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.floating):
                        if np.isnan(arr).any():
                            print(f"!!! CRITICAL WARNING: NaNs detected in {key} from self-play (env {traj_idx})!")

                replay_buffer.add_trajectory(
                    jnp.array(traj['observations']),
                    jnp.array(traj['actions']),
                    jnp.array(traj['rewards']),
                    jnp.array(traj['values']),
                    jnp.array(traj['policy_probs']),
                    jnp.array(traj['dones'])
                )

                # Add auxiliary price data for prediction loss (Phase 2)
                if args.use_prediction_loss and 'prices' in traj:
                    replay_buffer.add_auxiliary(
                        prices=traj['prices'],
                        next_prices=traj['next_prices'],
                    )

            # Training
            if epoch >= 5:
                train_start = time.time()
                metrics = trainer.train(replay_buffer, num_iterations=50, rng=rng)
                train_time = time.time() - train_start
                loss = metrics['total_loss'][-1]
                val_loss = metrics['value_loss'][-1]
                val_mse = metrics['value_mse'][-1]
                val_cat = metrics['value_cat'][-1]
                pol_loss = metrics['policy_loss'][-1]
                rew_loss = metrics['reward_loss'][-1]

                # Prediction loss metrics (Phase 2 - defaults to 0 if not enabled)
                pred_loss = metrics.get('loss_prediction', 0.0)
                if isinstance(pred_loss, list):
                    pred_loss = pred_loss[-1] if pred_loss else 0.0
                lambda_pred = metrics.get('lambda_pred', 0.0)
                if isinstance(lambda_pred, list):
                    lambda_pred = lambda_pred[-1] if lambda_pred else 0.0
                physics_reg = metrics.get('loss_physics_reg', 0.0)
                if isinstance(physics_reg, list):
                    physics_reg = physics_reg[-1] if physics_reg else 0.0

                # Action Stats
                actions_arr = np.array(trajectory['actions'])
                total_actions = len(actions_arr)
                n_sell = np.sum(actions_arr % N_ACTIONS_PER_ASSET == 0)
                n_hold = np.sum(actions_arr % N_ACTIONS_PER_ASSET == 1)
                n_buy = np.sum(actions_arr % N_ACTIONS_PER_ASSET == 2)
                n_limit_sell = np.sum(actions_arr % N_ACTIONS_PER_ASSET == 3)
                n_limit_buy = np.sum(actions_arr % N_ACTIONS_PER_ASSET == 4)
                pct_sell = 100.0 * n_sell / total_actions
                pct_hold = 100.0 * n_hold / total_actions
                pct_buy = 100.0 * n_buy / total_actions
                pct_limit_sell = 100.0 * n_limit_sell / total_actions
                pct_limit_buy = 100.0 * n_limit_buy / total_actions

                # Adaptive entropy: boost when policy collapses to HOLD
                if args.adaptive_entropy:
                    hold_ratio = n_hold / total_actions
                    if hold_ratio > args.entropy_hold_threshold + 0.10:
                        # Very high HOLD% -> aggressive boost
                        new_entropy_coef = args.entropy_boost_high
                    elif hold_ratio > args.entropy_hold_threshold:
                        # Moderate HOLD% -> moderate boost
                        new_entropy_coef = args.entropy_boost_low
                    else:
                        # Normal -> base entropy
                        new_entropy_coef = args.entropy_coef

                    if trainer.entropy_coef != new_entropy_coef:
                        print(f"  [ADAPTIVE ENTROPY] HOLD={hold_ratio*100:.1f}% -> entropy_coef: {trainer.entropy_coef:.3f} -> {new_entropy_coef:.3f}")
                        trainer.entropy_coef = new_entropy_coef

                # Policy Analytics
                policy_probs_arr = np.array(trajectory['policy_probs'])
                policy_entropy = -np.mean(np.sum(policy_probs_arr * np.log(policy_probs_arr + 1e-10), axis=1))
                avg_value = np.mean(trajectory['values'])
                value_std = np.std(trajectory['values'])

                # Reward MA20
                reward_ma20 = np.mean(all_rewards[-20:]) if len(all_rewards) >= 20 else np.mean(all_rewards)

                # =========================================================================
                # TASK_042: ReDo - Compute Dormancy Stats (before DS24 logging)
                # =========================================================================
                redo_stats = None
                if dormancy_monitor is not None and dormancy_monitor.should_check(epoch + 1):
                    stats = dormancy_monitor.get_dormancy_stats()
                    if stats['total_neurons'] > 0:
                        redo_stats = {
                            'dormant_neurons': stats['dormant_neurons'],
                            'dormancy_percent': stats['dormancy_percent'],
                            'recycled_count': 0,  # Not implementing actual recycling yet
                        }
                        print(f"  🔬 ReDo: {stats['dormant_neurons']}/{stats['total_neurons']} "
                              f"dormant ({stats['dormancy_percent']:.1f}%)")
                        dormancy_monitor.reset()

                # DS24 Enhanced Logging (via hooks module)
                if ds24_hooks is not None:
                    # Extract last training metrics for gradient norms (Level 4+)
                    train_metrics_last = None
                    if args.ds24_grad_norms and metrics:
                        train_metrics_last = {
                            k: v[-1] if isinstance(v, list) and v else v
                            for k, v in metrics.items()
                        }
                    ds24_hooks.on_epoch_end(
                        epoch=epoch,
                        trajectory=trajectory,
                        trainer_state=trainer.state,
                        args=args,
                        train_metrics=train_metrics_last,
                        redo_stats=redo_stats,
                        atc_stats=None,  # ATC not yet implemented in trainer
                    )

                calib_str = f" | Calib: {search.value_scale:.2f}/{search.value_shift:.2f}" if (search.value_scale != 1.0 or search.value_shift != 0.0) else ""

                action_str = f"S/H/B/LS/LB: {n_sell}/{n_hold}/{n_buy}/{n_limit_sell}/{n_limit_buy}"

                # Tree search uncertainty (Phase 3)
                avg_uncertainty = 0.0
                uncertainty_str = ""
                if 'uncertainty' in trajectory and len(trajectory['uncertainty']) > 0:
                    avg_uncertainty = np.mean(trajectory['uncertainty'])
                    uncertainty_str = f" | Unc: {avg_uncertainty:.4f}"

                # Prediction loss string (Phase 2)
                pred_str = ""
                if pred_loss > 0:
                    pred_str = f" | Pred: {pred_loss:.4f} (λ={lambda_pred:.3f})"

                # Market baseline metrics (alpha = agent - market)
                alpha = trajectory.get('alpha', 0.0)
                market_return = trajectory.get('market_return', 0.0)
                agent_return = trajectory.get('agent_return', 0.0)
                alpha_str = f" | α: {alpha:+.2f}% (Mkt: {market_return:+.1f}%)"

                print(f"Epoch {epoch+1:4d} | Reward: {score:+7.2f}{alpha_str} | Fee: {env.trading_fee_bps:2d} | Steps: {len(trajectory['actions']):3d} | "
                      f"{action_str} | "
                      f"Self-play: {self_play_time:5.1f}s | Train: {train_time:4.1f}s | "
                      f"Loss: {loss:.4f} | Val: {val_loss:.4f} | Pol: {pol_loss:.4f} | Rew: {rew_loss:.4f}{pred_str}{calib_str}{uncertainty_str}")

                # Log to CSV (use global epoch count)
                global_epoch = epoch + 1 + epoch_offset
                # TASK_031: Get PopArt stats for logging
                popart_mean = float(trainer.popart.mean) if trainer.use_popart and trainer.popart else 0.0
                popart_std = float(trainer.popart.std) if trainer.use_popart and trainer.popart else 1.0

                base_row = [
                    global_epoch, score, reward_ma20, env.trading_fee_bps, len(trajectory['actions']),
                    alpha, market_return, agent_return,  # Market baseline metrics
                    n_buy, n_hold, n_sell, n_limit_sell, n_limit_buy,
                    pct_buy, pct_hold, pct_sell, pct_limit_sell, pct_limit_buy,
                    self_play_time, train_time,
                    loss, val_loss, val_mse, val_cat, pol_loss, rew_loss,
                    pred_loss, lambda_pred, physics_reg,
                    policy_entropy, avg_value, value_std,
                    search.value_scale, search.value_shift,
                    popart_mean, popart_std,  # TASK_031
                    avg_uncertainty,
                    args.synthetic_level, args.model_size, args.name,
                    # Generalization evaluation (curriculum training)
                    last_eval_result.get('eval_level', ''),
                    last_eval_result.get('eval_reward', ''),
                    last_eval_result.get('eval_alpha', ''),
                    last_eval_result.get('eval_pct_hold', ''),
                    # TASK_022: GAE returns config
                    args.use_gae, args.gae_lambda,
                    # TASK_025: Per-head gradient clipping
                    args.per_head_clipping, args.clip_grad_value, args.clip_grad_policy,
                    args.clip_grad_dynamics, args.clip_grad_encoder,
                    # Exploration: epsilon-greedy
                    args.epsilon_greedy,
                ]

                # Append DS24 values if enabled
                if ds24_hooks is not None:
                    ds24_values = ds24_hooks.get_current_values()
                    ds24_row = [ds24_values.get(col, '') for col in ds24_columns]
                    csv_writer.writerow(base_row + ds24_row)
                else:
                    csv_writer.writerow(base_row)

                csv_file.flush()
                os.fsync(csv_file.fileno())  # Force OS to write to disk

                # Checkpoint management (skip if --no-checkpoint)
                if not args.no_checkpoint:
                    if safe_ckpt_manager is not None:
                        # Safe checkpoint: only save on improvement
                        if safe_ckpt_manager.should_save(score):
                            checkpoint_data = {
                                'params': trainer.state.params,
                                'opt_state': trainer.state.opt_state,
                                'step': trainer.state.step,
                            }
                            safe_ckpt_manager.save_checkpoint(checkpoint_data, epoch, score)
                            print(f"  🛡️ Safe checkpoint saved (metric: {score:.2f})")

                        # Check for rollback
                        if safe_ckpt_manager.should_rollback(score):
                            if rollback_count >= args.max_rollbacks:
                                print(f"  ❌ Max rollbacks ({args.max_rollbacks}) exceeded! Failing.")
                                raise RuntimeError(f"Training failed: max rollbacks exceeded at epoch {epoch}")

                            print(f"  🔄 ROLLBACK #{rollback_count + 1}: Loading best checkpoint...")
                            checkpoint_data, rollback_epoch = safe_ckpt_manager.load_best_checkpoint()

                            # Restore model state
                            trainer.state = trainer.state.replace(
                                params=checkpoint_data['params'],
                                opt_state=checkpoint_data['opt_state'],
                                step=checkpoint_data['step'],
                            )

                            # TODO: Add LR decay on rollback
                            # Currently Optax doesn't easily support runtime LR changes
                            # Would need to recreate optimizer with new LR
                            if args.lr_decay_on_rollback < 1.0:
                                print(f"  ⚠️ LR decay on rollback not yet implemented")

                            rollback_count += 1
                            print(f"  ✓ Rolled back to epoch {rollback_epoch}")
                    elif (epoch + 1) % 10 == 0:
                        # Standard checkpoint save (every 10 epochs)
                        trainer.save_checkpoint(checkpoint_dir)
                        print(f"  ✓ Checkpoint saved to {checkpoint_dir}")

                # Note: TASK_042 ReDo stats computation moved before DS24 logging (line ~1195)

                # =========================================================================
                # Generalization Evaluation (Curriculum Training)
                # =========================================================================
                # Run periodic evaluation on harder levels to track generalization
                if args.eval_every > 0 and (epoch + 1) % args.eval_every == 0:
                    rng, eval_rng = jax.random.split(rng)
                    eval_start = time.time()
                    eval_result = run_generalization_eval(
                        eval_level=args.eval_level,
                        search_fn=search,
                        model_apply_fn=model.apply,
                        params=trainer.state.params,
                        rng_key=eval_rng,
                        steps=args.eval_steps,
                        support_size=model.support_size,
                        n_assets=n_assets,
                        fee=args.fee,
                        slippage=args.slippage,
                    )
                    eval_time = time.time() - eval_start

                    # Update last_eval_result for CSV logging
                    last_eval_result = {
                        'eval_level': args.eval_level,
                        'eval_reward': eval_result['reward'],
                        'eval_alpha': eval_result['alpha'],
                        'eval_pct_hold': eval_result['pct_hold'],
                    }

                    print(f"  📊 EVAL L{args.eval_level}: Reward={eval_result['reward']:+.2f}% | "
                          f"α={eval_result['alpha']:+.2f}% | HOLD={eval_result['pct_hold']:.0f}% | "
                          f"Time: {eval_time:.1f}s")

                # Note: DS24 dynamic level boosting (L4/L6 snapshots) is now handled
                # automatically by ds24_hooks.on_epoch_end() based on --ds24-boost-every
                # and --ds24-snapshot-every settings

                # Adaptive value calibration
                if adaptive_calibration:
                    calibration_values.append(avg_value)
                    calibration_rewards.append(score)

                    # Keep only recent data (last 100 epochs)
                    if len(calibration_values) > 100:
                        calibration_values.pop(0)
                        calibration_rewards.pop(0)

                    # Recompute scale every interval after warmup
                    if epoch >= calibration_start_epoch and (epoch + 1) % calibration_interval == 0:
                        if len(calibration_values) >= 25:
                            vals = np.array(calibration_values)
                            rews = np.array(calibration_rewards)

                            # Compute optimal scale: match std of predictions to std of rewards
                            val_std = np.std(vals)
                            rew_std = np.std(rews)

                            if val_std > 0.01:  # Avoid division by zero
                                optimal_scale = rew_std / val_std
                                # Smooth update (don't jump too fast)
                                new_scale = 0.7 * search.value_scale + 0.3 * optimal_scale
                                # Clamp to reasonable range
                                new_scale = np.clip(new_scale, 0.5, 4.0)
                                search.value_scale = float(new_scale)

                                # Also compute optimal shift (center alignment)
                                val_mean = np.mean(vals) * new_scale
                                rew_mean = np.mean(rews)
                                optimal_shift = rew_mean - val_mean
                                search.value_shift = float(0.7 * search.value_shift + 0.3 * optimal_shift)

                                print(f"  📐 Adaptive calibration: scale={search.value_scale:.3f}, shift={search.value_shift:.3f}")
            else:
                # Action Stats for warmup
                actions_arr = np.array(trajectory['actions'])
                total_actions = len(actions_arr)
                n_sell = np.sum(actions_arr % N_ACTIONS_PER_ASSET == 0)
                n_hold = np.sum(actions_arr % N_ACTIONS_PER_ASSET == 1)
                n_buy = np.sum(actions_arr % N_ACTIONS_PER_ASSET == 2)
                n_limit_sell = np.sum(actions_arr % N_ACTIONS_PER_ASSET == 3)
                n_limit_buy = np.sum(actions_arr % N_ACTIONS_PER_ASSET == 4)
                pct_sell = 100.0 * n_sell / total_actions
                pct_hold = 100.0 * n_hold / total_actions
                pct_buy = 100.0 * n_buy / total_actions
                pct_limit_sell = 100.0 * n_limit_sell / total_actions
                pct_limit_buy = 100.0 * n_limit_buy / total_actions

                # Policy Analytics for warmup
                policy_probs_arr = np.array(trajectory['policy_probs'])
                policy_entropy = -np.mean(np.sum(policy_probs_arr * np.log(policy_probs_arr + 1e-10), axis=1))
                avg_value = np.mean(trajectory['values'])
                value_std = np.std(trajectory['values'])

                # Reward MA20
                reward_ma20 = np.mean(all_rewards[-20:]) if len(all_rewards) >= 20 else np.mean(all_rewards)

                action_str = f"S/H/B/LS/LB: {n_sell}/{n_hold}/{n_buy}/{n_limit_sell}/{n_limit_buy}"

                # Market baseline metrics (alpha = agent - market)
                alpha = trajectory.get('alpha', 0.0)
                market_return = trajectory.get('market_return', 0.0)
                agent_return = trajectory.get('agent_return', 0.0)
                alpha_str = f" | α: {alpha:+.2f}% (Mkt: {market_return:+.1f}%)"

                print(f"Epoch {epoch+1:4d} | Reward: {score:+7.2f}{alpha_str} | Fee: {env.trading_fee_bps:2d} | Steps: {len(trajectory['actions']):3d} | "
                      f"{action_str} | "
                      f"Self-play: {self_play_time:5.1f}s | (warmup)")

                # Tree search uncertainty for warmup (Phase 3)
                avg_uncertainty = 0.0
                if 'uncertainty' in trajectory and len(trajectory['uncertainty']) > 0:
                    avg_uncertainty = np.mean(trajectory['uncertainty'])

                # Log warmup to CSV (no train metrics, use global epoch count)
                global_epoch = epoch + 1 + epoch_offset
                # TASK_031: Get PopArt stats for logging (may be default during warmup)
                popart_mean = float(trainer.popart.mean) if trainer.use_popart and trainer.popart else 0.0
                popart_std = float(trainer.popart.std) if trainer.use_popart and trainer.popart else 1.0

                base_row = [
                    global_epoch, score, reward_ma20, env.trading_fee_bps, len(trajectory['actions']),
                    alpha, market_return, agent_return,  # Market baseline metrics
                    n_buy, n_hold, n_sell, n_limit_sell, n_limit_buy,
                    pct_buy, pct_hold, pct_sell, pct_limit_sell, pct_limit_buy,
                    self_play_time, 0,  # train_time=0
                    0, 0, 0, 0, 0, 0,   # loss components all 0
                    0, 0, 0,            # pred_loss, lambda_pred, physics_reg all 0
                    policy_entropy, avg_value, value_std,
                    search.value_scale, search.value_shift,
                    popart_mean, popart_std,  # TASK_031
                    avg_uncertainty,
                    args.synthetic_level, args.model_size, args.name,
                    # Generalization evaluation (curriculum training) - empty during warmup
                    '', '', '', '',
                    # TASK_022: GAE returns config
                    args.use_gae, args.gae_lambda,
                    # TASK_025: Per-head gradient clipping
                    args.per_head_clipping, args.clip_grad_value, args.clip_grad_policy,
                    args.clip_grad_dynamics, args.clip_grad_encoder,
                    # Exploration: epsilon-greedy
                    args.epsilon_greedy,
                ]

                # Append empty DS24 values if enabled (warmup has no TRM data)
                if ds24_hooks is not None:
                    ds24_row = ['' for _ in ds24_columns]
                    csv_writer.writerow(base_row + ds24_row)
                else:
                    csv_writer.writerow(base_row)

                csv_file.flush()
                      
    finally:
        # Close CSV log
        if 'csv_file' in locals():
            csv_file.close()
            print(f"📊 Training metrics saved to: {csv_log_path}")

        # No cleanup needed for GPU-native MCTS
        print("✅ MCTS cleanup complete.")

    print()
    print("="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    
    # Save metrics for scheduler and mastery check
    avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0
    recent_50_avg = np.mean(all_rewards[-50:]) if len(all_rewards) >= 50 else np.mean(all_rewards) if all_rewards else 0.0

    metrics_data = {
        'avg_reward': float(avg_reward),
        'final_loss': float(loss) if 'loss' in locals() else 0.0,
        'episode_rewards': [float(r) for r in all_rewards],  # Full history for mastery check
        'recent_avg_reward': float(recent_50_avg),  # Last 50 epochs average
        'total_epochs': args.epochs,
        'experiment_name': args.name,
        'rollback_count': rollback_count,  # Safe checkpoint rollbacks
        'safe_checkpoints_enabled': args.safe_checkpoints,
    }
    import json
    with open('last_run_metrics.json', 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"📊 Saved metrics: avg_reward={avg_reward:.2f}, recent_50_avg={recent_50_avg:.2f}")

    print()
    print("="*70)
    print("TRAINING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    # Needed for torch import in encoder
    import torch
    main()