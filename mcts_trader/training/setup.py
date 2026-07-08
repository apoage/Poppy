import os
import logging
import glob
import jax.numpy as jnp
import numpy as np
from typing import Tuple, List, Optional, Any, Dict, Dict

from mcts_trader.env.vectorbt_env import VectorBTTradingEnv
from mcts_trader.jax_agent.muzero_model import MuZeroNetworks
from mcts_trader.jax_agent.trainer import MuZeroTrainer, PredictionLossConfig, ExpertBCConfig
from mcts_trader.training.config import TrainingConfig

from mcts_trader.training.env_factory import create_jax_env_and_prices, create_jax_env_unique_prices
import jax


def load_adversarial_prices(data_dir: str, n_sequences: int = 10) -> Optional[np.ndarray]:
    """Load adversarial market data from curriculum bridge.

    TASK_110: External curriculum loop integration.
    Teacher generates .npz files, trainer loads them here.

    Args:
        data_dir: Directory containing adversarial_batch_*.npz files
        n_sequences: Maximum number of sequences to load

    Returns:
        numpy array of shape (n_sequences, seq_len, n_assets) or None if no data
    """
    if not os.path.exists(data_dir):
        logging.warning(f"[ADVERSARIAL] Data directory not found: {data_dir}")
        return None

    # Find all adversarial data files
    pattern = os.path.join(data_dir, "adversarial_batch_*.npz")
    files = sorted(glob.glob(pattern))

    if not files:
        logging.warning(f"[ADVERSARIAL] No adversarial_batch_*.npz files in {data_dir}")
        return None

    # Load sequences from files
    all_prices = []
    for f in files:
        try:
            data = np.load(f)
            prices = data['prices']  # (batch, seq_len, n_assets)
            all_prices.append(prices)
            if len(all_prices) * prices.shape[0] >= n_sequences:
                break
        except Exception as e:
            logging.warning(f"[ADVERSARIAL] Failed to load {f}: {e}")
            continue

    if not all_prices:
        return None

    # Concatenate and limit
    combined = np.concatenate(all_prices, axis=0)[:n_sequences]
    logging.info(f"[ADVERSARIAL] Loaded {len(combined)} sequences from {len(files)} files")
    return combined


def mix_adversarial_prices(
    synthetic_prices: jnp.ndarray,
    adversarial_prices: np.ndarray,
    mix_ratio: float,
    rng_key: jnp.ndarray,
) -> jnp.ndarray:
    """Mix adversarial prices into synthetic data.

    TASK_110: Replaces some synthetic windows with adversarial data.

    Args:
        synthetic_prices: JAX array (n_steps, n_assets) of synthetic prices
        adversarial_prices: numpy array (n_seqs, seq_len, n_assets) of adversarial data
        mix_ratio: Fraction of episodes to use adversarial data (0.0-1.0)
        rng_key: JAX random key

    Returns:
        Modified prices array with some windows replaced by adversarial data
    """
    if mix_ratio <= 0 or adversarial_prices is None:
        return synthetic_prices

    n_steps, n_assets = synthetic_prices.shape
    n_adv_seqs, adv_len, adv_assets = adversarial_prices.shape

    # How many adversarial segments to inject
    # Approximate: treat ratio as fraction of time coverage
    n_segments = max(1, int((n_steps / adv_len) * mix_ratio))

    # Random positions to inject adversarial data
    positions = jax.random.randint(rng_key, (n_segments,), 0, max(1, n_steps - adv_len))
    seq_indices = jax.random.randint(jax.random.fold_in(rng_key, 1), (n_segments,), 0, n_adv_seqs)

    # Convert to numpy for mutation
    result = np.array(synthetic_prices)
    adv_np = np.array(adversarial_prices)

    for i in range(n_segments):
        pos = int(positions[i])
        seq_idx = int(seq_indices[i])
        end_pos = min(pos + adv_len, n_steps)
        actual_len = end_pos - pos

        # Handle asset count mismatch
        if adv_assets >= n_assets:
            result[pos:end_pos] = adv_np[seq_idx, :actual_len, :n_assets]
        else:
            # Tile adversarial assets to match
            repeats = (n_assets + adv_assets - 1) // adv_assets
            tiled = np.tile(adv_np[seq_idx, :actual_len], (1, repeats))[:, :n_assets]
            result[pos:end_pos] = tiled

    logging.debug(f"[ADVERSARIAL] Injected {n_segments} segments into prices")
    return jnp.array(result)


def setup_fast_generator(
    n_assets: int = 64,
    hidden_dim: int = 512,
    num_layers: int = 8,
    seed: int = 42,
    use_conv: bool = True,
    checkpoint_path: Optional[str] = None,
) -> Tuple[Any, Dict]:
    """
    Initialize fast adversarial generator for on-the-fly VRAM generation.

    TASK_110: Flow matching for real-time adversarial data.
    TASK_111: Supports loading trained checkpoints from diff_teacher.

    Uses O(n) causal convolutions by default to handle long sequences (1440+ steps)
    while fitting in 6GB VRAM.

    Args:
        n_assets: Number of assets (default 64)
        hidden_dim: Model hidden dimension (512 default for good quality)
        num_layers: Number of conv/transformer layers (8 default)
        seed: Random seed for initialization
        use_conv: Use O(n) convolutions (True) or O(n²) attention (False)
        checkpoint_path: Path to trained diff_teacher checkpoint (TASK_111)
            If provided, loads trained weights instead of random init.

    Returns:
        model: FastFlowGenerator instance
        params: Initialized parameters (on GPU) - trained if checkpoint provided
    """
    key = jax.random.PRNGKey(seed)

    if checkpoint_path:
        # TASK_111: Load trained generator from diff_teacher checkpoint
        from mcts_trader.jax_agent.fast_generator import load_or_init_generator

        model, params = load_or_init_generator(
            checkpoint_path=checkpoint_path,
            key=key,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            n_assets=n_assets,
        )
        logging.info(f"[FAST GEN] Loaded trained checkpoint from {checkpoint_path}")
    else:
        # Random initialization
        from mcts_trader.jax_agent.fast_generator import init_fast_generator

        model, params = init_fast_generator(
            key,
            batch_size=4,
            seq_len=100,
            n_assets=n_assets,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            use_conv=use_conv,
        )

    # Move params to GPU
    params = jax.device_put(params)

    # Count parameters
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    logging.info(f"[FAST GEN] Initialized: {n_params/1e6:.2f}M params, {hidden_dim}d, {num_layers}L")

    return model, params


def generate_adversarial_on_the_fly(
    model: Any,
    params: Dict,
    key: jax.Array,
    n_steps: int,
    n_assets: int = 64,
    n_envs: int = 1,
    regime_bias: Optional[str] = None,
) -> jnp.ndarray:
    """
    Generate adversarial prices on-the-fly in VRAM.

    TASK_110: Called each epoch to generate fresh adversarial data.

    Args:
        model: FastFlowGenerator instance
        params: Generator parameters
        key: JAX random key
        n_steps: Number of price steps
        n_assets: Number of assets
        n_envs: Number of environments (for diversity)
        regime_bias: Optional regime to favor ('crash', 'rally', etc.)

    Returns:
        (n_steps, n_assets) price array on GPU
    """
    from mcts_trader.jax_agent.fast_generator import sample_regimes

    # Sample regimes
    regime_key, gen_key = jax.random.split(key)
    regimes = sample_regimes(regime_key, n_envs, bias=regime_bias)

    # Generate prices using the sample method
    prices = model.apply(params, gen_key, n_envs, n_steps, regimes, method=model.sample)

    if n_envs == 1:
        # Single env: return (n_steps, n_assets)
        return prices[0]
    else:
        # Multiple envs: blend sequences for variety
        blend_key = jax.random.fold_in(key, 2)
        indices = jax.random.randint(blend_key, (n_steps,), 0, n_envs)
        blended = jax.vmap(lambda t, i: prices[i, t, :])(
            jnp.arange(n_steps), indices
        )
        return blended


def setup_jax_environment(config: TrainingConfig, seed: int = None):
    """Initialize JAX-vectorized environment.

    Args:
        config: Training configuration
        seed: Random seed for data generation. If None, uses epoch from config
              or falls back to time-based seed. TASK_124: Dynamic seed prevents
              memorization of static data when using imprint_loss.
    """
    if config.synthetic_level is None:
        raise ValueError("JAX environment currently requires synthetic_level")

    # TASK_124: Use dynamic seed to prevent memorization
    # Priority: explicit seed > config epoch > time-based
    if seed is None:
        epoch = getattr(config, '_current_epoch', None)
        if epoch is not None:
            seed = epoch * 1000 + config.synthetic_level
        else:
            import time
            seed = int(time.time() * 1000) % (2**31)

    # Get noise config if adaptive noise is active
    noise_cfg = getattr(config, '_noise_config', None)

    # Common kwargs for both modes
    common_kwargs = dict(
        synthetic_level=config.synthetic_level,
        n_envs=config.parallel_envs,
        n_assets=64,
        fee_bps=float(config.fee),
        slippage_bps=float(config.slippage),
        max_episode_steps=config.steps,
        window_size=60,
        position_size_pct=0.1,
        rng_key=jax.random.PRNGKey(seed),
        probabilistic_fills=config.probabilistic_fills,
        borrow_rate_annual=config.borrow_rate,
        synthetic_noise=config.synthetic_noise,
        episode_variance=config.episode_variance,
        temporal_jitter=noise_cfg.temporal_jitter if noise_cfg else 0,
        time_dilation=noise_cfg.time_dilation if noise_cfg else 0.0,
        phase_shift_prob=noise_cfg.phase_shift_prob if noise_cfg else 0.0,
        burning_money=getattr(config, 'burning_money', False),
        burn_market_linked=getattr(config, 'burn_market_linked', True),
        burn_base_rate=getattr(config, 'burn_base_rate', 0.00005),
        burn_random=getattr(config, 'burn_random', False),
        burn_random_prob=getattr(config, 'burn_random_prob', 0.02),
        burn_random_max=getattr(config, 'burn_random_max', 500.0),
        burn_random_limit=getattr(config, 'burn_random_limit', 10),
        # TASK_818 Tier B (covers both **common_kwargs call sites)
        reward_attributable=getattr(config, 'reward_attributable', False),
        # TASK_828 Stage 3 (covers both **common_kwargs call sites)
        vector_action=getattr(config, 'vector_action', False),
    )

    # Choose between shared prices (faster) or unique prices per env (more diversity)
    if getattr(config, 'unique_data_per_env', False):
        # Each env gets its own independently generated price data
        n_steps_per_env = config.steps + 100  # Episode length + buffer
        jax_env_states, jax_prices, jax_env_config = create_jax_env_unique_prices(
            **common_kwargs,
            n_steps=n_steps_per_env,
            n_workers=getattr(config, 'data_gen_workers', 8),
        )
        logging.info(f"[JAX ENV] Unique prices per env: {config.parallel_envs} envs × {n_steps_per_env} steps")
    else:
        # Shared prices with random windows (original mode)
        # TASK_821: when --seed was given (config.seed is not None), also seed the
        # SyntheticMarket price generator with the per-call `seed` (which already
        # includes the CLI seed offset from the call sites in train_jax_fast.py).
        # Default (config.seed None) → price_seed=None → unseeded RandomState,
        # bit-identical to pre-TASK_821 behavior. The unique-prices branch above is
        # already deterministic given rng_key (per-env seeds derived via fold_in).
        _cli_seed = getattr(config, 'seed', None)
        jax_env_states, jax_prices, jax_env_config = create_jax_env_and_prices(
            **common_kwargs,
            n_steps=config.synthetic_steps,
            price_seed=(seed if _cli_seed is not None else None),
        )
        logging.info(f"[JAX ENV] Shared prices: {config.synthetic_steps} steps, {config.parallel_envs} envs with random windows")

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
        # TASK_828 Stage 3: vector-action embedding (default OFF ⇒ no params,
        # canon scalar path byte-identical). action_space_size stays n_assets*5.
        use_vector_action=getattr(config, 'vector_action', False),
        vector_n_actions_per_asset=5,
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
            scheduled_sampling_steps=config.scheduled_sampling_steps,
            # TASK_116: Student-t output for fat-tailed returns
            use_student_t=getattr(config, 'use_student_t', False),
            student_t_weight=getattr(config, 'student_t_weight', 0.8),
            mse_weight=getattr(config, 'student_t_mse_weight', 0.2),
            df_min=getattr(config, 'student_t_df_min', 2.1),
            df_max=getattr(config, 'student_t_df_max', 100.0),
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
        # BUG-011, BUG-013: Optimizer settings from config
        weight_decay=getattr(config, 'weight_decay', 1e-4),
        global_grad_clip=getattr(config, 'global_grad_clip', 5.0),
        redo=config.redo,
        redo_interval=config.redo_interval,
        redo_threshold=config.redo_threshold,
        atc_loss=config.atc_loss,
        atc_weight=config.atc_weight,
        atc_k_step=config.atc_k_step,
        atc_temperature=config.atc_temperature,
        atc_num_negatives=config.atc_num_negatives,
        # TASK_110: Cycle Balance
        cycle_balance=config.cycle_balance,
        cycle_balance_weight=config.cycle_balance_weight,
        cycle_balance_target_cv=config.cycle_balance_target_cv,
        cycle_atc=config.cycle_atc,
        cycle_atc_weight=config.cycle_atc_weight,
        cycle_atc_temperature=config.cycle_atc_temperature,
        # C1 fix: Match original value_loss_weight from .bak (was 2.0, trainer default is 1.0)
        value_loss_weight=2.0,
        # H5: Value loss type (Huber for outlier robustness)
        value_loss_type=config.value_loss_type,
        value_huber_delta=config.value_huber_delta,
        # Prediction loss type (Huber for outlier robustness on hard levels)
        pred_loss_type=getattr(config, 'pred_loss_type', 'mse'),
        pred_huber_delta=getattr(config, 'pred_huber_delta', 1.0),
        # TASK_066: Physics configuration
        reward_scale=config.reward_scale,
        # --reward-hybrid: PopArt-normalized target + hybrid reward loss (default OFF)
        reward_hybrid=getattr(config, 'reward_hybrid', False),
        # TASK_828 Stage 3: vector-action training path (default OFF = canon).
        vector_action=getattr(config, 'vector_action', False),
        alpha_target=config.alpha_target,
        beta_target=config.beta_target,
        sigma_target=config.sigma_target,
        label_smoothing=config.label_smoothing,
        min_entropy=config.min_entropy,
        # TASK_118: Alpha-relative reward
        alpha_relative_reward=getattr(config, 'alpha_relative_reward', False),
        alpha_reward_blend=getattr(config, 'alpha_reward_blend', 0.5),
        alpha_top_k=getattr(config, 'alpha_top_k', 4),
        # TASK_119: Optimal hindsight path
        amax_method=getattr(config, 'amax', None),
        amax_regret_weight=getattr(config, 'amax_regret_weight', 0.0),
        amax_imprint_weight=getattr(config, 'amax_imprint_weight', 0.0),
        amax_imprint_temp=getattr(config, 'amax_imprint_temp', 1.0),
        # TASK_125: Causal Expert Imprinting
        expert_bc_config=ExpertBCConfig(
            enabled=getattr(config, 'imprint', False),
            bc_weight_initial=getattr(config, 'bc_weight_initial', 0.1),
            bc_weight_final=getattr(config, 'bc_weight_final', 0.01),
            bc_weight_decay_steps=getattr(config, 'bc_weight_decay_steps', 50000),
            use_q_filter=getattr(config, 'bc_use_q_filter', True),
            use_large_margin=getattr(config, 'bc_use_large_margin', True),
            expert_batch_size=128,
            demo_ratio=0.125,
            weight_by_confidence=True,
        ),
    )

    return model, trainer


# =============================================================================
# TASK_111: Bella (CNN Diffusion) Market Generator
# =============================================================================

def setup_bella_generator(
    checkpoint_path: str,
    config_path: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Load trained Bella (CNN diffusion) generator.

    Args:
        checkpoint_path: Path to Bella checkpoint directory
        config_path: Optional path to config.json (uses default if not provided)

    Returns:
        generator: MarketGenerator instance
        config: GeneratorConfig
    """
    import json
    from pathlib import Path
    from diff_teacher.generation import MarketGenerator
    from diff_teacher.generation.generator import GeneratorConfig

    checkpoint_path = Path(checkpoint_path).resolve()  # Make absolute

    # Load config from parent or checkpoint directory
    if config_path:
        cfg_path = Path(config_path)
    else:
        # Try parent directory first (where diff_teacher saves it)
        cfg_path = checkpoint_path.parent / "config.json"
        if not cfg_path.exists():
            cfg_path = checkpoint_path / "config.json"

    if cfg_path.exists():
        with open(cfg_path) as f:
            config_dict = json.load(f)
        config = GeneratorConfig(
            architecture=config_dict.get('flow_architecture', 'conv'),
            hidden_dim=config_dict.get('flow_hidden_dim', 768),
            num_layers=config_dict.get('flow_layers', 12),
            n_assets=config_dict.get('n_assets', 64),
            n_features=len(config_dict.get('features', ['open', 'high', 'low', 'close', 'volume'])),
            seq_length=config_dict.get('window_size', 400),
            n_steps=10,
            sigma_min=config_dict.get('sigma_min', 1e-4),
        )
    else:
        logging.warning(f"[BELLA] Config not found at {cfg_path}, using defaults")
        config = GeneratorConfig()

    # Load generator
    generator = MarketGenerator.from_checkpoint(str(checkpoint_path), config=config)
    logging.info(f"[BELLA] Loaded: {config.architecture}, {config.hidden_dim}d, {config.num_layers}L")
    logging.info(f"[BELLA] Output shape: ({config.seq_length}, {config.n_assets}, {config.n_features})")

    return generator, config


def generate_bella_prices_on_device(
    generator: Any,
    key: jax.Array,
    n_envs: int = 1,
    invert_regime: bool = False,
    ohlcv: bool = True,
) -> jnp.ndarray:
    """
    Generate market prices using Bella directly on GPU (no CPU roundtrip).

    TASK_111: VRAM-direct generation for maximum throughput.
    TASK_703: Returns full OHLCV by default. Set ohlcv=False for close-only (Poppy compat).

    Args:
        generator: MarketGenerator instance
        key: JAX random key
        n_envs: Number of environments (for diversity)
        invert_regime: If True, swap crash↔rally for dual-branch diversity
        ohlcv: If True, return (T, n_assets, 5) OHLCV. If False, return (T, n_assets) close-only.

    Returns:
        OHLCV (n_steps, n_assets, 5) or close-only (n_steps, n_assets) JAX array on GPU
    """
    # Use on-device generation - stays entirely in VRAM
    key, subkey = jax.random.split(key)
    seed = int(jax.random.randint(subkey, (), 0, 2147483647))  # 2^31 - 1

    # Sample random regimes for each environment (0-4: crash, rally, sideways, volatile, calm)
    key, subkey = jax.random.split(key)
    regimes = jax.random.randint(subkey, (n_envs,), 0, 5)

    # Generate on device: returns JAX array (n_samples, seq_len, assets*features)
    generated = generator.generate_on_device(
        n_samples=n_envs,
        seed=seed,
        regime=regimes,
        invert_regime=invert_regime,
    )

    # Reshape to (n_envs, seq_len, n_assets, n_features)
    seq_len = generator.config.seq_length
    n_assets = generator.config.n_assets
    n_features = generator.config.n_features
    generated = generated.reshape(n_envs, seq_len, n_assets, n_features)

    # TASK_703: Convert ALL OHLCV channels to price space (not just close)
    # Bella outputs z-scored/standardized values (std ~1.0) for all 5 channels
    # Channel order: O=0, H=1, L=2, C=3, V=4
    volatility_scale = 0.015  # Target ~1.5% step volatility

    # Close prices (channel 3) — cumulative returns → price levels
    close_raw = generated[:, :, :, 3]
    close_returns = close_raw * volatility_scale
    close_log = jnp.cumsum(close_returns, axis=1)
    close_prices = 100.0 * jnp.exp(close_log)

    # Open prices (channel 0) — previous close (standard OHLCV convention)
    open_prices = jnp.concatenate([close_prices[:, :1], close_prices[:, :-1]], axis=1)

    # High prices (channel 1) — Bella generates realistic offset patterns
    # Use absolute value of Bella's output as offset above close
    high_raw = jnp.abs(generated[:, :, :, 1]) * volatility_scale
    high_prices = close_prices * (1.0 + high_raw)

    # Low prices (channel 2) — offset below close
    low_raw = jnp.abs(generated[:, :, :, 2]) * volatility_scale
    low_prices = close_prices * (1.0 - low_raw)

    # Volume (channel 4) — Bella generates volume patterns in z-scored space
    vol_raw = generated[:, :, :, 4]
    volume = jnp.exp(vol_raw * 0.5 + 10.0)  # Log-normal scale
    volume = jnp.maximum(volume, 1.0)

    # TASK_703: Return close-only for backward compat (Poppy) or full OHLCV
    if not ohlcv:
        # Legacy close-only path
        close_prices = jnp.maximum(close_prices, 1.0)
        if n_envs == 1:
            return close_prices[0]  # (seq_len, n_assets)
        return close_prices  # (n_envs, seq_len, n_assets)

    # Stack OHLCV and enforce constraints
    ohlcv_out = jnp.stack([open_prices, high_prices, low_prices, close_prices, volume], axis=-1)
    # Ensure positive prices (high >= close >= low already by construction, but guard)
    ohlcv_out = ohlcv_out.at[:, :, :, :4].set(jnp.maximum(ohlcv_out[:, :, :, :4], 1.0))

    # Return OHLCV - batched if n_envs > 1
    if n_envs == 1:
        return ohlcv_out[0]  # (seq_len, n_assets, 5)
    else:
        return ohlcv_out  # (n_envs, seq_len, n_assets, 5)
