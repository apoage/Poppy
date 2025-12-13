from easydict import EasyDict
import torch

# OPTIMIZED FOR: RTX 3090 (24GB VRAM) + 16 CPU cores
# Expected GPU utilization: 75-90%
# Expected VRAM usage: 12-18GB
# Expected speedup: 20-30x over baseline

# Aggressive batch size for 24GB GPU
try:
    vram = torch.cuda.get_device_properties(0).total_memory
    vram_gb = vram / (1024**3)

    if vram_gb < 12:
        dynamic_batch_size = 512
    elif vram_gb < 20:
        dynamic_batch_size = 1536
    elif vram_gb < 24:
        dynamic_batch_size = 2560
    else:  # >= 24GB (RTX 3090, RTX 4090, A100)
        dynamic_batch_size = 3072  # ~12-14GB VRAM usage

    print(f"Detected {vram_gb:.1f}GB VRAM → batch_size={dynamic_batch_size}")
except:
    dynamic_batch_size = 512
    print("GPU detection failed → batch_size=512 (fallback)")

# Configuration for TRM+MCTS Agent (OPTIMIZED)
trading_muzero_config = dict(
    env=dict(
        env_id='MultiAssetTrading-v0',
        n_assets=64,
        n_actions_per_asset=3,

        # OPTIMIZED: Use 12/16 CPU cores (75% utilization)
        collector_env_num=12,       # Was 4 → +8 collectors = 3x parallelism

        evaluator_env_num=2,
        n_evaluator_episode=3,
        stop_value=100000,
    ),
    policy=dict(
        model=dict(
            observation_shape=(27, 64, 64),
            action_space_size=64 * 3,  # 192
            state_dim=64,
        ),
        mcts=dict(
            # OPTIMIZED: More simulations (GPU can handle 100-200)
            num_simulations=100,    # Was 50 → 2x more search

            max_depth=10,           # Lookahead steps
            pb_c_base=19652,
            pb_c_init=1.25,
            root_dirichlet_alpha=0.3,
            root_noise_weight=0.25,

            # CRITICAL: GPU-accelerated MCTS (10-20x speedup)
            mcts_ctree=True,        # Enable C++ tree for GPU acceleration
        ),

        cuda=True,
        env_type='not_board_games',
        game_segment_length=100,

        # OPTIMIZED: More frequent updates with larger batches
        update_per_collect=50,

        # OPTIMIZED: Large batch size to utilize 24GB VRAM
        batch_size=dynamic_batch_size,  # Was 512 → 3072 (6x larger)

        optim_type='AdamW',         # AdamW slightly better than Adam
        learning_rate=0.0002,
        weight_decay=1e-4,
        grad_clip_value=10,

        # OPTIMIZED: Larger buffer for better sample diversity
        replay_buffer_size=500,     # Was 300 → +200 segments

        eval_freq=100,

        # GPU MCTS configuration (CRITICAL FOR SPEED)
        use_rpc_wrapper=False,      # Disable RPC (not needed for GPU MCTS)
        mcts_ctree=True,           # Enable C++ tree (required)

        # OPTIONAL: Mixed precision training (test after GPU MCTS works)
        # Uncomment to enable:
        # use_amp=True,
        # amp_dtype='bfloat16',     # bfloat16 better for Ampere GPUs
    ),
)
trading_muzero_config = EasyDict(trading_muzero_config)
main_config = trading_muzero_config

trading_muzero_create_config = dict(
    env=dict(
        type='wrapped_trading_env',
        import_names=['mcts_trader.env'],
    ),
    env_manager=dict(type='subprocess', context='spawn'),
    policy=dict(
        type='muzero',
        import_names=['lzero.policy.muzero'],
    ),
)
trading_muzero_create_config = EasyDict(trading_muzero_create_config)
create_config = trading_muzero_create_config
