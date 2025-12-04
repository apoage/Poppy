from easydict import EasyDict
import torch

# Dynamic batch size based on GPU memory
try:
    vram = torch.cuda.get_device_properties(0).total_memory
    if vram < 16 * 1024**3: # < 16GB
        dynamic_batch_size = 256
    elif vram < 24 * 1024**3: # < 24GB
        dynamic_batch_size = 512
    else: # >= 24GB
        dynamic_batch_size = 2048
except:
    dynamic_batch_size = 256 # Fallback

# Configuration for TRM+MCTS Agent
trading_muzero_config = dict(
    env=dict(
        env_id='MultiAssetTrading-v0',
        n_assets=64,
        n_actions_per_asset=3,
        collector_env_num=4,
        evaluator_env_num=2,
        n_evaluator_episode=3,
        stop_value=100000, # Unreachable, training continuous
    ),
    policy=dict(
        model=dict(
            observation_shape=(27, 64, 64),
            action_space_size=64 * 3, # 192
            state_dim=64,
            model_type='conv', # Custom model type
        ),
        mcts=dict(
            num_simulations=50,
            max_depth=10, # Lookahead steps
            pb_c_base=19652,
            pb_c_init=1.25,
            root_dirichlet_alpha=0.3,
            root_noise_weight=0.25,
        ),
        cuda=True,
        env_type='not_board_games',
        game_segment_length=200,
        update_per_collect=50,
        batch_size=dynamic_batch_size,
        optim_type='Adam',
        learning_rate=0.0002,
        weight_decay=1e-4,
        grad_clip_value=10,
        replay_buffer_size=10000,
        eval_freq=100,
    ),
)
trading_muzero_config = EasyDict(trading_muzero_config)
main_config = trading_muzero_config

trading_muzero_create_config = dict(
    env=dict(
        type='trading_env',
        import_names=['mcts_trader.env.trading_env'],
    ),
    env_manager=dict(type='subprocess', context='spawn'),
    policy=dict(
        type='muzero',
        import_names=['lzero.policy.muzero'],
    ),
)
trading_muzero_create_config = EasyDict(trading_muzero_create_config)
create_config = trading_muzero_create_config
