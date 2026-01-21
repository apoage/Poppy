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
    parser.add_argument('--synthetic-steps', type=int, default=10000)
    parser.add_argument('--db-path', type=str, default=None)
    parser.add_argument('--cache-path', type=str, default=None)
    parser.add_argument('--assets-file', type=str, default=None)
    parser.add_argument('--periods-file', type=str, default=None)
    parser.add_argument('--start-date', type=str, default=None)
    parser.add_argument('--end-date', type=str, default=None)
    parser.add_argument('--config', type=str, dest='config_file', default=None)
    
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
    parser.add_argument('--dirichlet-aggressive', action='store_true')
    
    # Optimizer settings
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr-schedule', type=str, default='constant')
    parser.add_argument('--lr-min', type=float, default=1e-6)
    parser.add_argument('--lr-warmup-epochs', type=int, default=10)
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
    parser.add_argument('--redo', action='store_true')
    parser.add_argument('--redo-interval', type=int, default=10)
    parser.add_argument('--atc-loss', action='store_true')
    parser.add_argument('--atc-weight', type=float, default=0.1)
    
    # Interventions
    parser.add_argument('--adaptive-noise', action='store_true')
    parser.add_argument('--adaptive-fees', action='store_true')
    parser.add_argument('--fee-reward-threshold', type=float, default=2.5)
    parser.add_argument('--fee-upgrade-epochs', type=int, default=10)
    parser.add_argument('--smart-allocation', action='store_true')
    parser.add_argument('--jitter-minutes', type=int, default=360)
    parser.add_argument('--sac-entropy', action='store_true')
    parser.add_argument('--adaptive-entropy', action='store_true')
    parser.add_argument('--entropy-hold-threshold', type=float, default=0.70)
    parser.add_argument('--entropy-boost-low', type=float, default=0.15)
    parser.add_argument('--entropy-boost-high', type=float, default=0.30)
    
    # Checkpointing & Logging
    parser.add_argument('--load-checkpoint', type=str, default=None)
    parser.add_argument('--no-checkpoint', action='store_true')
    parser.add_argument('--safe-checkpoints', action='store_true')
    parser.add_argument('--ds24-logging', action='store_true')
    parser.add_argument('--ds24-level', type=int, default=0)
    parser.add_argument('--ds24-intermediates-every', type=int, default=10)
    
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
    
    return config
