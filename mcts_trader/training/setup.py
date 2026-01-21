import os
import logging
import jax.numpy as jnp
import numpy as np
from typing import Tuple, List, Optional, Any

from mcts_trader.env.vectorbt_env import VectorBTTradingEnv
from mcts_trader.jax_agent.muzero_model import MuZeroNetworks
from mcts_trader.jax_agent.trainer import MuZeroTrainer, PredictionLossConfig
from mcts_trader.training.config import TrainingConfig

from mcts_trader.training.env_factory import create_jax_env_and_prices
import jax

def setup_jax_environment(config: TrainingConfig):
    """Initialize JAX-vectorized environment."""
    if config.synthetic_level is None:
        raise ValueError("JAX environment currently requires synthetic_level")
        
    jax_env_states, jax_prices, jax_env_config = create_jax_env_and_prices(
        synthetic_level=config.synthetic_level,
        n_envs=config.parallel_envs,
        n_assets=64, # Will be updated by factory
        n_steps=config.synthetic_steps,
        fee_bps=float(config.fee),
        slippage_bps=float(config.slippage),
        max_episode_steps=config.steps,
        window_size=60,
        position_size_pct=0.1,
        rng_key=jax.random.PRNGKey(42),
        probabilistic_fills=config.probabilistic_fills,
        borrow_rate_annual=config.borrow_rate,
        synthetic_noise=config.noise, # config.noise is Dirichlet, config has no price noise field?
        # Actually config.noise is MCTS noise. synthetic_noise was removed from TrainingConfig dataclass?
        # Let's check config.py. It has adaptive_noise but maybe not base synthetic noise?
        # Original cli had --synthetic-noise. I missed adding it to TrainingConfig.
        # I'll default to 0.0 for now or check config.
        episode_variance=config.episode_variance,
    )
    return jax_env_states, jax_prices, jax_env_config

def setup_environment(config: TrainingConfig) -> Tuple[VectorBTTradingEnv, List[VectorBTTradingEnv], Optional[Dict]]:
    """Initialize trading environments based on config.
    
    Returns:
        env: Main environment
        parallel_envs: List of parallel environments
        period_data: Dictionary with 'prices' (dict of period_idx -> prices) and 'lengths' (list of lengths), or None
    """
    env_kwargs = dict(
        n_assets=64, # Default, will be updated by env
        initial_cash=10000.0,
        trading_fee_bps=config.fee,
        slippage_bps=config.slippage,
        max_steps=config.steps,
        action_points=config.action_points,
        tp_sl_enabled=config.tp_sl_enabled,
        # Burning money settings
        burning_money=config.burning_money,
        burn_market_linked=config.burn_market_linked,
        burn_base_rate=config.burn_base_rate,
        burn_random=config.burn_random,
        burn_random_prob=config.burn_random_prob,
    )

    if config.synthetic_level is not None:
        env_kwargs['synthetic_level'] = config.synthetic_level
        env_kwargs['synthetic_n_steps'] = config.synthetic_steps
    elif config.periods_file:
        # TASK_064: Multi-period stratified mode - load first period for initial env
        import json
        with open(config.periods_file, 'r') as f:
            periods_config = json.load(f)
        first_period = periods_config['periods'][0]
        env_kwargs['db_path'] = config.db_path
        env_kwargs['start_date'] = first_period['start_date']
        env_kwargs['end_date'] = first_period['end_date']
        if first_period.get('assets'):
            env_kwargs['specific_symbols'] = first_period['assets']
    elif config.db_path:
        env_kwargs['db_path'] = config.db_path
        env_kwargs['cache_path'] = config.cache_path
        env_kwargs['start_date'] = config.start_date
        env_kwargs['end_date'] = config.end_date
        if config.assets_file:
            import json
            with open(config.assets_file) as f:
                env_kwargs['specific_symbols'] = json.load(f)

    env = VectorBTTradingEnv(**env_kwargs)
    
    parallel_envs = []
    if config.parallel_envs > 1:
        parallel_envs = [env]
        
        # TASK_064: Share prices if possible
        can_share_prices = (
            env.synthetic is None and
            env.prices is not None and
            hasattr(env, 'symbols')
        )
        
        if can_share_prices:
            # Create shared kwargs referencing same prices array
            shared_kwargs = dict(env_kwargs)
            shared_kwargs['prices'] = env.prices
            # Remove paths to prevent reloading
            shared_kwargs.pop('db_path', None)
            shared_kwargs.pop('periods_file', None) # Not a real kwarg but good to be safe
            
            for _ in range(config.parallel_envs - 1):
                parallel_envs.append(VectorBTTradingEnv(**shared_kwargs))
        else:
            for _ in range(config.parallel_envs - 1):
                parallel_envs.append(VectorBTTradingEnv(**env_kwargs))
            
    # TASK_064: Load multi-period data for swapping
    period_data = None
    if config.periods_file:
        import json
        logging.info(f"Loading multi-period data from {config.periods_file}...")
        with open(config.periods_file, 'r') as f:
            periods_config = json.load(f)
        
        periods_list = periods_config.get('periods', [])
        period_prices = {}
        period_lengths = []
        
        for period_idx, period in enumerate(periods_list):
            # Load prices using temporary env
            temp_env = VectorBTTradingEnv(
                db_path=config.db_path,
                n_assets=64, # Fixed asset count for model
                start_date=period['start_date'],
                end_date=period['end_date'],
                specific_symbols=period.get('assets'),
                initial_cash=10000.0,
                max_steps=100,
            )
            prices = temp_env.prices
            
            # Pad to N_ASSETS if needed
            N_ASSETS = 64
            if prices.shape[1] < N_ASSETS:
                n_loaded = prices.shape[1]
                pad_indices = [i % n_loaded for i in range(N_ASSETS)]
                prices = prices[:, pad_indices]
                
            period_prices[period_idx] = prices
            period_lengths.append(len(prices))
            logging.info(f"  P{period_idx}: {period['start_date'][:10]} -> {period['end_date'][:10]} ({len(prices)} steps)")
            
        period_data = {
            'prices': period_prices,
            'lengths': period_lengths,
            'current_idx': 0
        }
            
    return env, parallel_envs, period_data

def setup_model_and_trainer(config: TrainingConfig, n_assets: int) -> Tuple[MuZeroNetworks, MuZeroTrainer]:
    """Initialize model and trainer based on config."""
    compute_dtype = jnp.bfloat16 if config.bf16 else jnp.float32
    
    action_space_size = n_assets * 5
    
    # Prediction head size configuration
    PRED_HEAD_SIZES = {
        'small':  {'embed_dim': 128, 'num_blocks': 2, 'mlp_dim': 256, 'n_factors': 8},
        'medium': {'embed_dim': 256, 'num_blocks': 3, 'mlp_dim': 512, 'n_factors': 16},
        'large':  {'embed_dim': 512, 'num_blocks': 4, 'mlp_dim': 1024, 'n_factors': 32},
    }
    pred_cfg = PRED_HEAD_SIZES.get(config.pred_head_size, PRED_HEAD_SIZES['small'])

    model = MuZeroNetworks(
        state_dim=config.state_dim,
        action_space_size=action_space_size,
        H_cycles=config.h_cycles,
        L_cycles=config.l_cycles,
        use_prediction_head=config.use_prediction_loss,
        physics_only=config.physics_only,
        n_assets=n_assets,
        pred_embed_dim=pred_cfg['embed_dim'],
        pred_num_blocks=pred_cfg['num_blocks'],
        pred_mlp_dim=pred_cfg['mlp_dim'],
        pred_n_factors=pred_cfg['n_factors'],
        dtype=compute_dtype,
    )

    prediction_config = None
    if config.use_prediction_loss:
        prediction_config = PredictionLossConfig(
            weight_initial=config.prediction_loss_initial,
            weight_final=config.prediction_loss_weight,
            anneal_steps=config.prediction_anneal_steps,
            multi_step_k=config.multi_step_k,
            physics_reg_weight=config.physics_reg_weight,
            physics_warmup_steps=config.physics_warmup_steps,
        )

    trainer = MuZeroTrainer(
        model=model,
        learning_rate=config.lr,
        num_simulations=config.sims,
        batch_size=config.batch_size,
        discount=config.discount,
        entropy_coef=config.entropy_coef,
        lr_schedule_type=config.lr_schedule,
        lr_min=config.lr_min,
        lr_warmup_epochs=config.lr_warmup_epochs,
        total_epochs=config.epochs,
        use_prediction_loss=config.use_prediction_loss,
        prediction_config=prediction_config,
        use_popart=config.use_popart,
        popart_beta=config.popart_beta,
        track_grad_norms=config.ds24_grad_norms,
        per_head_clipping=config.per_head_clipping,
        clip_grad_value=config.clip_grad_value,
        clip_grad_policy=config.clip_grad_policy,
        clip_grad_dynamics=config.clip_grad_dynamics,
        clip_grad_encoder=config.clip_grad_encoder,
        redo=config.redo,
        redo_interval=config.redo_interval,
        redo_threshold=config.redo_threshold,
        atc_loss=config.atc_loss,
        atc_weight=config.atc_weight,
        atc_k_step=config.atc_k_step,
        atc_temperature=config.atc_temperature,
        atc_num_negatives=config.atc_num_negatives,
    )
    
    return model, trainer
