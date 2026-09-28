import os
import sys
import warnings
import time
import random
import logging
import csv
import traceback
from tqdm import tqdm
import jax
import jax.numpy as jnp
import numpy as np

# GPU Optimization settings
_trader_dir = os.path.dirname(os.path.abspath(__file__))
_xla_cache = os.path.join(_trader_dir, '.xla_cache')
_jax_cache = os.path.join(_trader_dir, '.jax_cache')
os.makedirs(_xla_cache, exist_ok=True)
os.makedirs(_jax_cache, exist_ok=True)

os.environ['XLA_FLAGS'] = f'--xla_gpu_autotune_level=4 --xla_gpu_per_fusion_autotune_cache_dir={_xla_cache}'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['JAX_COMPILATION_CACHE_DIR'] = _jax_cache

warnings.filterwarnings('ignore', category=FutureWarning, module='jax')

# Weighted synthetic level distribution (more weight on harder levels)
# Used with --weighted-synthetic flag for Phase 2 speed training
WEIGHTED_SYNTHETIC_LEVELS = {
    0: 1, 1: 1, 2: 2, 3: 2, 4: 2,      # Easy (L0-L4): ~12%
    5: 3, 6: 3, 7: 4, 8: 4, 9: 4,      # Medium (L5-L9): ~32%
    10: 5, 11: 5, 12: 6, 13: 6, 14: 6,  # Hard (L10-L14): ~56%
}

# JAX Configuration
jax.config.update("jax_compilation_cache_dir", _jax_cache)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

# Import refactored components
from mcts_trader.training.cli import parse_args
from mcts_trader.training.setup import (
    setup_environment, setup_model_and_trainer, setup_jax_environment,
    load_adversarial_prices, mix_adversarial_prices,
    setup_fast_generator, generate_adversarial_on_the_fly,
    setup_bella_generator, generate_bella_prices_on_device,
)
from mcts_trader.env.jax_trading_env import jit_batch_reset
from mcts_trader.jax_agent.mctx_search import create_mctx_search
from mcts_trader.jax_agent.replay_buffer import JAXReplayBuffer
from mcts_trader.jax_agent.smart_allocation import compute_env_start_offsets
from mcts_trader.jax_agent.expert_buffer import UnifiedIndexBuffer, create_expert_buffer
# TASK_135: Removed price_cache import - now stores prices directly in expert_buffer
from mcts_trader.training.self_play import run_self_play, run_self_play_batched, run_self_play_jax, N_ACTIONS_PER_ASSET

# DS24 Enhanced Logging
from tools.DS24.metrics_logger import DS24MetricsLogger
from mcts_trader.jax_agent.prediction.physics_core import DifferentiableMarketPhysics


def apply_runtime_price_noise(prices: jnp.ndarray, noise_level: float, rng_key: jnp.ndarray) -> jnp.ndarray:
    """Apply price noise perturbations to JAX price array at runtime.

    This allows adaptive noise to work with pre-generated data by adding
    random perturbations at self-play time rather than generation time.

    Args:
        prices: JAX array of shape (n_steps, n_assets) with clean prices
        noise_level: Noise intensity (0.0 = no noise, 1.0 = full noise)
        rng_key: JAX random key

    Returns:
        Perturbed prices with same shape
    """
    if noise_level <= 0:
        return prices

    # Scale noise to be proportional to price volatility
    # Use log returns to measure typical movement scale
    log_returns = jnp.diff(jnp.log(prices + 1e-10), axis=0)
    volatility = jnp.std(log_returns, axis=0, keepdims=True)  # (1, n_assets)

    # Generate multiplicative noise: price * (1 + noise_level * volatility * random)
    # This preserves price structure while adding realistic perturbations
    noise = jax.random.normal(rng_key, shape=prices.shape)
    noise_scale = noise_level * volatility * 10.0  # Scale factor for visible effect

    # Apply multiplicative noise
    perturbed = prices * (1.0 + noise_scale * noise)

    # Ensure prices stay positive
    perturbed = jnp.maximum(perturbed, 0.01)

    return perturbed

# RESTORED: Adaptive controllers (zombie flags → alive)
from mcts_trader.jax_agent.adaptive_noise import AdaptiveNoiseController, NoiseConfig
from mcts_trader.jax_agent.adaptive_fees import AdaptiveFeeController, FeeConfig
from mcts_trader.jax_agent.adaptive_burn import AdaptiveBurnController, BurnConfig
from mcts_trader.jax_agent.schedules import SimsSchedule
from mcts_trader.training.run_declaration import create_protected_run_dir
from mcts_trader.training.safe_checkpoint import (
    SafeCheckpointManager, append_checkpoint_score, prune_run_dir_by_score,
    retention_policy_available)
from mcts_trader.jax_agent.trainer import CheckpointVanishedError, CheckpointSidecarError
from mcts_trader.training.evaluator import run_generalization_eval
# TASK_888 (D38): the ONE checkpoint resolver + fail-closed loader, shared with
# eval_vector.py / experiments/857_vhat_distill/distill.py / real_trade_jax.py.
from mcts_trader.jax_agent import checkpoint_io

# TASK_042: ReDo - Recycling Dormant Neurons
from mcts_trader.jax_agent.redo import DormancyMonitor, recycle_dormant_neurons

# TASK_040: SAC-style Adaptive Entropy
from mcts_trader.jax_agent.exploration.sac_entropy import AdaptiveEntropyManager


def _attempt_checkpoint_save(trainer, target_dir, keep, what, epoch, run_dir):
    """TASK_876 fix-2 (B1 + M3): save, then decide what ACTUALLY happened.

    Returns ``(written, msgs)``. ``written`` is a path VERIFIED to exist on disk, or None
    when no usable checkpoint was produced — so the caller may emit a score row (and print a
    success line) if and only if it is non-None.

    RAISES for anything it does not recognise. This helper is a CLASSIFIER, not a
    normalizer and not a corrector: it recognises exactly the three DESIGNED outcomes below
    and converts them to "log and continue"; every other exception is re-raised unchanged so
    it kills the run with a nonzero exit, exactly as the bare ``trainer.save_checkpoint(...)``
    calls it replaced always did. That matters because
    ``tools/orchestration/dojo_manager.py`` gates a curriculum level purely on the
    subprocess ``returncode``: a swallowed disk-full/permission-denied would let a run print
    "TRAINING COMPLETE!", exit 0 with no checkpoints on disk, and advance the orchestrator.

    The three DESIGNED outcomes, deliberately NOT collapsed into one ``except Exception``
    (before this fix a bare handler printed "likely a checkpoint already on disk at or above
    this step" for every failure — the wrong diagnosis for a sidecar ENOSPC — and dropped the
    score of a checkpoint that really was on disk):

    1. :class:`CheckpointSidecarError` — flax dir on disk, a sidecar failed. Score it.
    2. :class:`CheckpointVanishedError` — flax returned a path it then deleted itself.
    3. ``ValueError: Destination ... already exists`` — the exact-step collision that
       ``overwrite=False`` is *designed* to produce on a rollback-resume.
    """
    msgs = []
    try:
        written = trainer.save_checkpoint(target_dir, keep=keep)
    except CheckpointSidecarError as _e:
        # The flax checkpoint IS on disk; only trainer_metadata.json / popart_state.npz
        # failed. Score it (it is a real, loadable artifact) and say what is actually wrong.
        written = _e.path
        msgs.append(
            f"  [CKPT] {what} at epoch {epoch}: the checkpoint {written} IS on disk and is "
            f"still resumable, but writing {_e.what} failed ({_e.__cause__}). A resume from "
            f"it will fall back to defaults for the state that sidecar carries "
            f"(entropy_coef / PopArt normalisation), which changes training behaviour "
            f"silently. Free space or fix permissions under {target_dir}; the score row for "
            f"this checkpoint IS being recorded.")
    except CheckpointVanishedError as _e:
        # The save reported success and left nothing behind (flax's own keep-N cleanup ate
        # it inside the same call). This is the collapse-recovery resume window.
        written = None
        msgs.append(
            f"  [CKPT] {what} VANISHED at epoch {epoch}: flax reported writing {_e.path} but "
            f"nothing is on disk there ({_e.reason}). THIS RUN HAS NO RESUME POINT FOR ITS "
            f"CURRENT TRAJECTORY — {run_dir} still holds only OLDER/higher-step checkpoints "
            f"from a previous trajectory, and every later save at this step range will "
            f"vanish the same way. No score row is being written (an unscored, non-existent "
            f"checkpoint must never enter the audit trail). FIX: restart into a FRESH run "
            f"directory (--name <new>), or move the stale higher-step checkpoints out of "
            f"{run_dir}, before relying on any checkpoint from this run.")
    except ValueError as _e:
        # DESIGNED outcome 3 ONLY: the exact-step collision flax raises under
        # overwrite=False ("Destination <dir>/checkpoint_<step> already exists"), i.e. a
        # rollback-resume re-saving a step already on disk. That is the trade this task chose
        # (a refused save beats a protection-blind mass delete) and it must not kill the run.
        #
        # This is a RECOGNITION guard, NOT a normalizer and NOT a corrector: it does not
        # repair, retry, or rename anything, and it does not fold unrelated ValueErrors into
        # a benign-looking log line. Any ValueError whose message is not that collision is
        # re-raised UNCHANGED — an unrecognised failure must take the run down (nonzero exit)
        # so dojo_manager's returncode gate cannot advance past a run with no checkpoints.
        if 'already exists' not in str(_e):
            raise
        written = None
        msgs.append(
            f"  [CKPT] {what} REFUSED at epoch {epoch}: {type(_e).__name__}: {_e} — no "
            f"checkpoint was written for this step. With overwrite=False this is a "
            f"checkpoint already on disk at exactly this step (e.g. after a "
            f"rollback-resume). Earlier checkpoints in {run_dir} are unaffected and still "
            f"resumable; training continues.")
    return written, msgs


def _checkpoint_load_requested(config):
    """Was ``--load-checkpoint`` GIVEN AT ALL — the gate for the load below, asked once here so
    ``main()`` and its test exercise the same expression rather than two spellings that agree
    today (guard-vs-loader drift is this task's whole thesis; a gate-vs-test copy is the same
    class one layer up).

    ``is not None``, deliberately NOT truthiness. ``--load-checkpoint``'s argparse default IS
    ``None`` (``mcts_trader/training/cli.py:313``), so "omitted" and "given, but empty" are
    genuinely distinguishable — and they must be, because eight orchestration scripts build this
    flag PROGRAMMATICALLY and pass its value unconditionally
    (``tools/run_scheduled_training.py:142``, ``tools/orchestration/dojo_v2.py:392``/``:470``,
    ``dojo_v3_historical.py:791``/``:860``/``:942``, the sweeps). When the shell/Python variable
    holding the checkpoint is unset, what reaches argv is ``--load-checkpoint ''`` and argparse
    stores ``''`` — not None, but FALSY. Under the old truthiness gate that collapsed to "flag
    not given": no print, no exception, no non-zero exit, and a multi-hour resume proceeded on
    fresh ``init_train_state`` weights, invisible in the training curve because a fresh run looks
    like a fresh run. That is precisely the scenario this row ranks FIRST, and it was surviving
    at the site it ranks first, because the gate never let ``checkpoint_io`` see the input.

    An explicitly-empty value now reaches ``load_checkpoint_into``, which refuses it by name (it
    resolves to the cwd, finds no checkpoint there, and raises ``CheckpointRefusalError``).
    Omitting the flag entirely still skips loading, exactly as before. Same idiom, same reason,
    as ``eval_vector.py``'s ``is not None`` gates on its collect flags."""
    return getattr(config, "load_checkpoint", None) is not None


def _attempt_checkpoint_load(trainer, load_checkpoint_path, reset_optimizer):
    """TASK_888 (D38 consequence order item 2 — "a multi-hour resume silently discards its
    checkpoint and starts fresh"): the load-side counterpart of ``_attempt_checkpoint_save``,
    factored out for the same reason — direct unit-testability without the rest of ``main()``.

    Pre-888 this checked ``if loaded_state is not None:`` before printing "Checkpoint loaded
    successfully" — flax's ``restore_checkpoint`` never actually returns ``None`` on a miss (it
    silently aliases ``target`` back UNCHANGED instead, D37), so that ``else`` branch was DEAD
    CODE and the success message printed unconditionally, including for a nonexistent/empty
    ``--load-checkpoint`` dir. A resume that silently discards its checkpoint and starts fresh
    is invisible in the training curve — a fresh run looks like a fresh run — which is exactly
    the failure this task ranks #2 by consequence (the 884/885/887 batteries run in this lane).

    Mutates ``trainer.state`` in place via ``checkpoint_io.load_checkpoint_into`` (resolved via
    the SAME function every other checkpoint guard in this program uses) and RAISES on anything
    that is not a real, complete load — never returns a "keep going" signal for a bad
    checkpoint. Returns the load-evidence dict."""
    checkpoint_load = checkpoint_io.load_checkpoint_into(trainer, load_checkpoint_path)
    print(f"  Checkpoint loaded: resolved={checkpoint_load['resolved_path']} "
          f"step={checkpoint_load['step']} "
          f"fingerprint={checkpoint_load['param_fingerprint'][:12]}...")
    if reset_optimizer:
        print("  Resetting optimizer state for new regime...")
        trainer.state = trainer.reset_optimizer_state(trainer.state)
        print("  (Momentum and variance reset, model weights preserved)")
    return checkpoint_load


def main():
    # 1. Parse configuration
    config = parse_args()

    # TASK_821: --seed opt-in deterministic mode.
    # `cli_seed` is None (default) unless --seed was passed, which is EXACTLY current
    # behavior: model init and the historical-env RNG (env_factory.py) stay time-seeded,
    # and every `+ _seed_offset` below is a no-op (_seed_offset == 0).
    # When --seed IS given: seed Python `random` + numpy's global RNG (covers
    # random.choices in the --weighted-synthetic path and the np.random.* global-state
    # sampling in jax_agent/expert_buffer.py's demo replay), and derive the model-init
    # PRNGKey + env-factory rng_key deterministically from cli_seed (see call sites below).
    cli_seed = getattr(config, 'seed', None)
    _seed_offset = cli_seed if cli_seed is not None else 0
    if cli_seed is not None:
        random.seed(cli_seed)
        np.random.seed(cli_seed)
        print(f"[Config] --seed {cli_seed}: deterministic mode "
              f"(numpy/random seeded; model-init + env-factory RNG derived from seed)")

    # Apply dirichlet_aggressive override (increases exploration noise)
    if getattr(config, 'dirichlet_aggressive', False):
        config.noise = 0.5
        config.noise_frac = 0.5
        print("[Config] Dirichlet aggressive mode: noise=0.5, noise_frac=0.5")

    # 2. Setup logging
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join('logs', f"training_fast_{config.name}.log")),
            logging.StreamHandler()
        ]
    )

    print("="*70)
    print(f"JAX TRADING AGENT | {config.mode} | Refactored Core")
    print("="*70)

    # 3. Initialize Environment
    print("[1/4] Initializing Environment...")

    jax_env_states = None
    jax_prices = None
    jax_env_config = None
    env = None
    parallel_envs = []
    period_data = None
    n_assets = 64
    pregenerated_levels = {}  # Cache for pre-generated level data
    async_regenerator = None  # Background thread executor for async data regeneration
    pending_regenerations = {}  # level -> Future

    # TASK_111: If --bella is enabled, use JAX env with placeholder prices
    # Bella will generate fresh prices each epoch, but we need JAX env initialized
    if getattr(config, 'bella', False) and config.synthetic_level is None:
        config.synthetic_level = 0  # Placeholder, will be replaced by Bella

    # Handle weighted synthetic: pre-generate ALL levels at startup
    use_weighted = getattr(config, 'weighted_synthetic', False)
    if use_weighted:
        print("[Config] Weighted synthetic mode: pre-generating all 15 levels...")
        # (random is imported at module level — TASK_821 needs it seedable from main())
        from tqdm import tqdm as tqdm_init
        from concurrent.futures import ThreadPoolExecutor
        import threading

        # Create background regenerator (single thread to avoid memory issues)
        async_regenerator = ThreadPoolExecutor(max_workers=1, thread_name_prefix='data_regen')
        regen_lock = threading.Lock()

        def regenerate_level_async(level, cfg_copy, epoch=0):
            """Generate new data for a level (runs in background thread)."""
            # Create a copy of config for this thread
            cfg_copy.synthetic_level = level
            # TASK_124: Use epoch-based seed to prevent memorization
            # TASK_821: + _seed_offset (0 unless --seed given) for multi-seed variation
            seed = epoch * 1000 + level + _seed_offset
            states, prices, env_cfg = setup_jax_environment(cfg_copy, seed=seed)
            return level, (states, prices, env_cfg)

        # Pre-generate L0-L14 (use level as seed for initial generation)
        for level in tqdm_init(range(15), desc="   Pre-generating levels", ncols=80):
            config.synthetic_level = level
            # TASK_124: Use level-based seed for initial pre-generation
            # TASK_821: + _seed_offset (0 unless --seed given)
            states, prices, env_cfg = setup_jax_environment(config, seed=level * 1000 + _seed_offset)
            pregenerated_levels[level] = (states, prices, env_cfg)

        # Pick initial level with weights
        levels = list(WEIGHTED_SYNTHETIC_LEVELS.keys())
        weights = list(WEIGHTED_SYNTHETIC_LEVELS.values())
        config.synthetic_level = random.choices(levels, weights=weights, k=1)[0]
        print(f"   Starting with L{config.synthetic_level}")
        print(f"   Async regeneration: enabled (fresh data after each use)")

        # Use pre-generated data for initial level
        jax_env_states, jax_prices, jax_env_config = pregenerated_levels[config.synthetic_level]
        n_assets = jax_env_config.n_assets
        use_jax_env = True
    elif config.synthetic_level is None and getattr(config, 'use_jax_env', False):
        # JAX env with historical data
        if config.db_path:
            from mcts_trader.training.env_factory import create_jax_env_historical

            # Get dates from --periods-file if provided, otherwise use --start-date/--end-date
            start_date = config.start_date
            end_date = config.end_date
            specific_symbols = None

            if getattr(config, 'periods_file', None):
                import json
                with open(config.periods_file, 'r') as f:
                    periods_config = json.load(f)
                first_period = periods_config['periods'][0]
                start_date = first_period['start_date']
                end_date = first_period['end_date']
                specific_symbols = first_period.get('assets')
                print(f"   Using periods file: {config.periods_file}")
                print(f"   First period: {start_date[:10]} -> {end_date[:10]}")
            elif getattr(config, 'assets_file', None):
                import json
                with open(config.assets_file, 'r') as f:
                    specific_symbols = json.load(f)

            print("   Initializing JAX-vectorized environment with HISTORICAL data...")
            jax_env_states, jax_prices, jax_env_config = create_jax_env_historical(
                db_path=config.db_path,
                start_date=start_date,
                end_date=end_date,
                n_envs=config.parallel_envs,
                n_assets=64,
                specific_symbols=specific_symbols,
                fee_bps=float(config.fee),
                slippage_bps=float(config.slippage),
                max_episode_steps=config.steps,
                borrow_rate_annual=config.borrow_rate,
                burning_money=getattr(config, 'burning_money', False),
                burn_market_linked=getattr(config, 'burn_market_linked', True),
                burn_base_rate=getattr(config, 'burn_base_rate', 0.00005),
                burn_random=getattr(config, 'burn_random', False),
                burn_random_prob=getattr(config, 'burn_random_prob', 0.02),
                burn_random_max=getattr(config, 'burn_random_max', 500.0),
                burn_random_limit=getattr(config, 'burn_random_limit', 10),
                reward_attributable=getattr(config, 'reward_attributable', False),
                bar_minutes=getattr(config, 'bar_minutes', 1),
                # TASK_828 Stage 3: vector-action env step (default OFF = canon).
                vector_action=getattr(config, 'vector_action', False),
                # TASK_839: budget-constrained vector-step sizing (default OFF = canon).
                vector_budget_step=getattr(config, 'vector_budget_step', False),
                vector_budget_pct=getattr(config, 'vector_budget_pct', 0.99),
                # TASK_840: target-weight rebalance step (default OFF = canon).
                vector_weight_rebalance=getattr(config, 'vector_weight_rebalance', False),
                vector_weight_unit=getattr(config, 'vector_weight_unit', -1.0),
                vector_weight_max=getattr(config, 'vector_weight_max', 4.0),
                vector_rebalance_deadband=getattr(config, 'vector_rebalance_deadband', -1.0),
                # TASK_821: explicit rng_key when --seed given (deterministic episode-start
                # sampling in jit_batch_reset). None (default) preserves the existing
                # time-seeded fallback in env_factory.py::create_jax_env_historical.
                rng_key=jax.random.PRNGKey(cli_seed) if cli_seed is not None else None,
            )
            n_assets = jax_env_config.n_assets
            use_jax_env = True
            print(f"   JAX historical env: {jax_prices.shape[0]} steps, {n_assets} assets")

            # TASK_812: pre-launch post-cost / leak-free / oracle-off asserts (requirement #1:
            # "verify it's overridden, don't assume" — the zero-cost default IS 810's turnover trap).
            if getattr(config, 'assert_postcost', False):
                _amax = getattr(config, 'amax', None)
                _imprint = getattr(config, 'imprint', False)
                _predloss = getattr(config, 'use_prediction_loss', False)
                print("=" * 72)
                print("[TASK_812] PRE-LAUNCH ASSERTS (post-cost genuine-RL probe)")
                print(f"  fee_bps            = {jax_env_config.fee_bps}  (require > 0; 810-trap default is 0)")
                print(f"  slippage_bps       = {jax_env_config.slippage_bps}  (require > 0)")
                print(f"  use_prediction_loss= {_predloss}  (require False — the 97%-of-loss trap)")
                print(f"  amax / imprint     = {_amax} / {_imprint}  (require None / False — oracle OFF)")
                print(f"  value target       = compute_returns (GAE/MC over env rewards); NOT path_solver/oracle")
                print(f"  leak guard         = obs window [t-window, t-1], execute at t (TASK_124, no look-ahead)")
                print("=" * 72)
                assert jax_env_config.fee_bps > 0.0, "TASK_812: POST-COST fee not active (fee_bps=0 is 810's gross trap)"
                assert jax_env_config.slippage_bps > 0.0, "TASK_812: POST-COST slippage not active (slippage_bps=0)"
                assert not _predloss, "TASK_812: prediction loss must be OFF (the 97%-of-loss trap)"
                assert _amax is None and not _imprint, "TASK_812: oracle (amax/imprint) must be OFF — value must come from compute_returns"
        else:
            logging.warning("--jax-env requires --synthetic-level or --db-path. Falling back to Python env.")
            use_jax_env = False
    else:
        use_jax_env = getattr(config, 'use_jax_env', False)
        if use_jax_env and config.synthetic_level is not None:
            print("   Initializing JAX-vectorized environment...")
            # TASK_124: Use seed=0 for initial epoch
            # TASK_821: + _seed_offset (0 unless --seed given)
            jax_env_states, jax_prices, jax_env_config = setup_jax_environment(config, seed=0 + _seed_offset)
            n_assets = jax_env_config.n_assets

    if not use_jax_env:
        env, parallel_envs, period_data = setup_environment(config)
        n_assets = env.n_assets

    # TASK_110: Adversarial data setup (file-based OR on-the-fly)
    adversarial_prices = None
    fast_gen_model = None
    fast_gen_params = None

    if getattr(config, 'adversarial_on_the_fly', False):
        # On-the-fly VRAM generation (fast generator)
        hidden_dim = getattr(config, 'adversarial_hidden_dim', 512)
        num_layers = getattr(config, 'adversarial_num_layers', 4)
        checkpoint_path = getattr(config, 'adversarial_checkpoint', None)
        if checkpoint_path:
            print(f"   Loading trained generator from {checkpoint_path}...")
        else:
            print(f"   Initializing fast adversarial generator ({hidden_dim}d, {num_layers}L)...")
        fast_gen_model, fast_gen_params = setup_fast_generator(
            n_assets=n_assets,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            checkpoint_path=checkpoint_path,  # TASK_111: Load trained weights
        )
        bias = getattr(config, 'adversarial_regime_bias', None)
        if bias:
            print(f"   Regime bias: {bias}")
        print(f"   On-the-fly generation: enabled (~115ms/epoch)")

    elif getattr(config, 'adversarial_data_dir', None):
        # File-based adversarial data loading
        print(f"   Loading adversarial data from {config.adversarial_data_dir}...")
        adversarial_prices = load_adversarial_prices(
            config.adversarial_data_dir,
            n_sequences=config.parallel_envs * 2,
        )
        if adversarial_prices is not None:
            print(f"   Loaded {len(adversarial_prices)} adversarial sequences")
            print(f"   Mix ratio: {config.adversarial_ratio:.0%}")

            # Mix adversarial data into JAX prices if available
            if jax_prices is not None:
                # TASK_821: derive from --seed when given; else unchanged time-seed.
                mix_rng = (jax.random.PRNGKey(cli_seed) if cli_seed is not None
                           else jax.random.PRNGKey(int(time.time()) % 2**31))
                jax_prices = mix_adversarial_prices(
                    jax_prices, adversarial_prices,
                    config.adversarial_ratio, mix_rng
                )
                print("   Mixed adversarial data into training prices")

    # TASK_111: Bella (CNN diffusion) market generator
    bella_generator = None
    if getattr(config, 'bella', False):
        checkpoint_path = getattr(config, 'bella_checkpoint', 'checkpoints/diff_teacher/final/generator')
        print(f"   Loading Bella generator from {checkpoint_path}...")
        bella_generator, bella_config = setup_bella_generator(checkpoint_path)
        print(f"   Bella: {bella_config.architecture}, {bella_config.hidden_dim}d, {bella_config.num_layers}L")
        print(f"   Output: ({bella_config.seq_length}, {bella_config.n_assets}, {bella_config.n_features})")
        print(f"   On-device generation: enabled (VRAM-direct)")

    action_space_size = n_assets * 5

    # 4. Initialize Model and Trainer
    print("[2/4] Initializing JAX model...")
    model, trainer = setup_model_and_trainer(config, n_assets)

    # 5. Initialize Search and Buffer
    # TASK_828 Stage 3: vector-action → Sampled-MuZero vector search + factored
    # planning; default OFF = the canon single-categorical mctx search VERBATIM.
    vector_action = getattr(config, 'vector_action', False)
    if vector_action:
        from mcts_trader.jax_agent.sampled_search import create_sampled_vector_search
        _vk = getattr(config, 'vector_k', 16)
        search_obj = create_sampled_vector_search(
            model, n_assets=n_assets, n_actions_per_asset=5,
            num_samples=_vk, num_simulations=config.sims,
            max_num_considered_actions=min(_vk, 16), discount=config.discount,
            gumbel_scale=config.gumbel_scale, value_scale=config.value_scale,
            value_shift=config.value_shift,
            hold_bias=getattr(config, 'vector_hold_bias', 0.0),
            dither_scale=getattr(config, 'vector_dither_scale', 1.0),
            proposal_scheme=getattr(config, 'vector_proposal_scheme', 'dithered-book'),
            # TASK_847: policy-prior knobs (ignored unless scheme='policy-prior').
            proposal_temp=getattr(config, 'vector_proposal_temp', 1.0),
            explore_slots=getattr(config, 'vector_proposal_explore_slots', 2),
            # TASK_854 (D21): learned-prior weight in candidate scoring (default 1.0 = canon).
            prior_weight=getattr(config, 'prior_weight', 1.0),
            # TASK_838: per-node action resampling (default OFF = root-reuse).
            per_node_resampling=getattr(config, 'vector_per_node_resample', False),
            # TASK_851: search leaf evaluator (default OFF = model leaf, canon).
            leaf_eval=getattr(config, 'leaf_eval', 'model'),
            leaf_rollout_r=getattr(config, 'leaf_rollout_r', 3),
            leaf_vhat_tail=getattr(config, 'leaf_vhat_tail', True),
        )
    else:
        search_obj = create_mctx_search(model, num_simulations=config.sims, use_gumbel=True)
    search = jax.jit(search_obj)  # JITted for non-JAX paths
    # For JAX self-play with lax.scan, use raw_search to allow XLA fusion
    search_raw = search_obj.raw_search

    replay_buffer = JAXReplayBuffer(
        capacity=5000,
        observation_shape=(27, n_assets, n_assets),
        action_size=action_space_size,
        batch_size=config.batch_size,
        n_assets=n_assets,
        vector_action=vector_action,  # TASK_828 Stage 3
    )

    # TASK_828 Stage 3 startup assert: the single flag must be threaded identically
    # everywhere (env / model / buffer / trainer). Vector mode requires the JAX env.
    if vector_action:
        assert use_jax_env, "TASK_828: --vector-action requires the JAX env (--jax-env)"
        assert jax_env_config.vector_action, "TASK_828: env_config.vector_action mismatch"
        assert model.use_vector_action, "TASK_828: model.use_vector_action mismatch"
        assert replay_buffer.vector_action, "TASK_828: buffer.vector_action mismatch"
        assert trainer.vector_action, "TASK_828: trainer.vector_action mismatch"

    # TASK_125: Initialize Expert Buffer for Causal Expert Imprinting
    # TASK_135: Eliminated disk-based price_cache - stores prices in RAM
    expert_buffer = None
    if getattr(config, 'imprint', False):
        print("[TASK_125] Initializing Causal Expert Imprinting...")
        expert_buffer = create_expert_buffer(
            n_assets=n_assets,
            demo_capacity=getattr(config, 'expert_demo_capacity', 50000),
            agent_capacity=500000,
            window_size=60,
        )
        print(f"   Expert buffer: {expert_buffer.demo_capacity} demos, {expert_buffer.agent_capacity} agent")
        print(f"   TASK_135: Prices stored in RAM (no disk cache)")

    # 6. Initialize parameters
    # TASK_124: Use time-based seed for model initialization to prevent
    # "lucky" random weights that accidentally exploit patterns
    # TASK_821: --seed overrides this for deterministic multi-seed validation runs.
    # `rng` below is the single master key threaded (via jax.random.split/fold_in)
    # through the ENTIRE run: self-play, MCTS noise, ReDo, eval — so fixing model_seed
    # here is sufficient to make everything downstream of it deterministic too.
    # (time is imported at module level; a local `import time` here would shadow it
    # for ALL of main() and crash later uses when the else branch is skipped)
    if cli_seed is not None:
        model_seed = cli_seed
    else:
        model_seed = int(time.time() * 1000) % (2**31)
    rng = jax.random.PRNGKey(model_seed)
    rng, init_rng = jax.random.split(rng)
    print(f"   Model seed: {model_seed}")
    dummy_obs = jnp.ones((1, 27, n_assets, n_assets))
    state = trainer.init_train_state(init_rng, dummy_obs)

    # 6b. Load checkpoint if specified
    if _checkpoint_load_requested(config):
        print(f"[CHECKPOINT] Loading from: {config.load_checkpoint!r}")
        try:
            _attempt_checkpoint_load(trainer, config.load_checkpoint, config.reset_optimizer)
        except Exception as e:
            print(f"  Error loading checkpoint: {e}")
            raise

    # Clear buffer if requested (for curriculum transfer to new level)
    if config.clear_buffer and _checkpoint_load_requested(config):
        print("[BUFFER] Clearing replay buffer for new regime...")
        replay_buffer.clear()

    # 7. Warmup
    print("[4/4] Warmup (JIT compilation)...")
    # Get PopArt state for value denormalization (C2 fix)
    popart_state = trainer.popart.state if hasattr(trainer, 'popart') and trainer.popart else None
    # Need prices for prediction loss or amax optimal path
    needs_prices = config.use_prediction_loss or getattr(config, 'amax', None)
    if jax_env_states is not None:
        # Use raw_search for JAX path to allow XLA fusion inside lax.scan
        _, _ = run_self_play_jax(
            jax_env_states, jax_prices, jax_env_config,
            search_raw, model.apply, trainer.state.params, rng,
            steps=5, collect_prices=needs_prices,
            support_size=model.support_size,
            discount=config.discount,
            popart_state=popart_state,
            use_gae=getattr(config, 'use_gae', False),
            gae_lambda=getattr(config, 'gae_lambda', 0.95),
            max_positions=getattr(config, 'max_positions', 16),
            min_cash_threshold=getattr(config, 'min_cash_threshold', 100.0),
            vector_action=vector_action,  # TASK_828 Stage 3
            leaf_eval=getattr(config, 'leaf_eval', 'model'),  # TASK_851
        )
    else:
        _, _ = run_self_play(
            env, search, model.apply, trainer.state.params, rng, steps=5,
            support_size=model.support_size,
            discount=config.discount,
            popart_state=popart_state,
            use_gae=getattr(config, 'use_gae', False),
            gae_lambda=getattr(config, 'gae_lambda', 0.95),
        )
    print("Warmup done")

    # 8. Training Loop with Full DS24 Logging
    print("="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)

    # Core training params
    print(f"  Epochs: {config.epochs} | Steps/env: {config.steps} | Parallel envs: {config.parallel_envs}")
    print(f"  Batch size: {config.batch_size} | Replay buffer: 5000 | Total steps/epoch: {config.steps * config.parallel_envs}")
    print(f"  Model: {config.model_size} (state_dim={config.state_dim}) | H={config.h_cycles} L={config.l_cycles}")
    print(f"  Assets: {n_assets} | Actions: {action_space_size} (5 per asset)")

    # MCTS params
    print(f"\n  MCTS: sims={config.sims} | noise={config.noise} | noise_frac={config.noise_frac}")
    print(f"  Gumbel scale: {config.gumbel_scale} | Value scale: {config.value_scale} | ε-greedy: {config.epsilon_greedy}")

    # Learning params
    print(f"\n  LR: {config.lr} ({config.lr_schedule}) | Discount: {config.discount}")
    print(f"  Entropy coef: {config.entropy_coef} | Grad clips: val={config.clip_grad_value} pol={config.clip_grad_policy}")

    # Features status
    features = []
    if getattr(config, 'use_popart', True): features.append(f"PopArt(β={config.popart_beta})")
    if getattr(config, 'use_gae', False): features.append(f"GAE(λ={config.gae_lambda})")
    if getattr(config, 'use_prediction_loss', False): features.append(f"PredLoss(w={config.prediction_loss_weight})")
    if getattr(config, 'redo', False): features.append(f"ReDo(int={config.redo_interval})")
    if getattr(config, 'atc_loss', False): features.append(f"ATC(w={config.atc_weight})")
    if getattr(config, 'sac_entropy', False): features.append(f"SAC(α₀={config.sac_init_alpha})")
    if getattr(config, 'adaptive_entropy', False): features.append(f"AdaptEnt(thr={config.entropy_hold_threshold})")
    if getattr(config, 'dirichlet_aggressive', False): features.append("DirichletAgg")
    if getattr(config, 'differential_reward', False): features.append("DiffReward")
    if getattr(config, 'safe_checkpoints', False): features.append("SafeCkpt")

    print(f"\n  Features: {' | '.join(features) if features else 'None'}")

    # Eval & logging
    if getattr(config, 'eval_every', 0) > 0:
        print(f"  Eval: every {config.eval_every} epochs on L{config.eval_level} ({config.eval_steps} steps)")
    print(f"  DS24 logging: level {config.ds24_level}")

    # Data source
    if config.synthetic_level is not None:
        if getattr(config, 'weighted_synthetic', False):
            print(f"\n  Data: Weighted Synthetic (L0-L14, start=L{config.synthetic_level})")
        else:
            print(f"\n  Data: Synthetic L{config.synthetic_level} (noise={config.synthetic_noise})")
    elif config.db_path:
        print(f"\n  Data: Historical ({config.db_path})")

    print("="*70)
    print("TRAINING LOOP")
    print("="*70)

    all_rewards = []
    checkpoint_dir = os.path.abspath(f"jax_ckpt_fast/{config.name}")
    # TASK_922: this checkpoint creation path only, not every ad hoc experiment creator.
    # TrainingConfig has no task metadata; never infer a task id from config.name.
    create_protected_run_dir(checkpoint_dir, name=config.name, task="unspecified",
                             reason="Training checkpoints; retain pending explicit review")

    # ==== TASK_881: checkpoint retention (storage policy + explicit manifest protection) ====
    # 'recency' is the DEFAULT (D32/D35: the destructive score path only flips on after the
    # protection model ships AND gate vi's collapse case is re-proven end-to-end — see
    # tasks/TASK_881_CHECKPOINT_PROTECTION_MODEL.md). 'score' is a real, working, opt-in mode
    # now: it delegates every deletion decision to `prune_run_dir_by_score`, which resolves
    # manifest coverage and refuses (never deletes) a run dir no manifest declares — the same
    # policy `tools/ckpt_gc.py` enforces offline (gate iii: the two deleters share one
    # `plan()`). Kept as a runtime check too (defense in depth for any caller that builds
    # TrainingConfig without going through `cli.py::parse_args`, which already refuses at
    # startup) — this one falls back rather than raising, since discovering an unavailable
    # policy module mid-run should not crash a job that was otherwise healthy under 'recency'.
    ckpt_retention = getattr(config, 'ckpt_retention', 'recency')
    if ckpt_retention == 'score':
        _pol_ok, _pol_where = retention_policy_available()
        if not _pol_ok:
            print(f"  [CKPT] CRITICAL: --ckpt-retention score requested but the retention "
                  f"policy is UNAVAILABLE ({_pol_where}). Falling back to --ckpt-retention "
                  f"recency (bounded flax keep=3) so this run cannot grow without bound.")
            ckpt_retention = 'recency'
    elif ckpt_retention != 'recency':
        raise ValueError(
            f"--ckpt-retention {ckpt_retention!r} is not a supported mode ('recency' or "
            f"'score' only).")
    # flax's `keep` stays 3 in EVERY mode — the pre-876 bound, unchanged — because
    # `--ckpt-retention score`'s manifest protection has not yet been stress-tested combined
    # with a raised flax keep across this file's full feature surface (gate vi is explicitly
    # about that end-to-end proof; until it is re-run here, this build does not attempt it).
    # Score-based retention therefore governs the `snapshot_ep<N>` family (flax's own keep
    # never touches it) and the whole-run disk-floor emergency prune; it does not yet change
    # how many `checkpoint_<step>` dirs flax itself retains. `overwrite=False` (trainer.py,
    # D32/D33) is the other intended delta: it makes a resume at a lower step REFUSE the
    # collision instead of silently mass-deleting every checkpoint above the resume point.
    flax_keep = 3
    # TASK_881 scope 3: an in-run prune only runs under 'score'. One helper, called after every
    # successful save (periodic/snapshot/final) instead of only from one site, so a run under
    # 'score' stays bounded and manifest-protected regardless of which family just grew.
    ckpt_min_free_gb = float(getattr(config, 'ckpt_min_free_gb', 5.0))

    def _maybe_prune_by_score(what: str) -> None:
        if ckpt_retention != 'score':
            return
        try:
            _st = os.statvfs(checkpoint_dir)
            under_floor = (_st.f_bavail * _st.f_frsize / 1e9) < ckpt_min_free_gb
        except OSError:
            under_floor = False
        nonlocal ckpt_prune_failed
        # TASK_881 defense in depth, at the point where it matters. The startup availability
        # check above promises that "discovering an unavailable policy module mid-run should
        # not crash a job that was otherwise healthy", but that promise was only enforced at
        # startup: an unexpected raise from this OPPORTUNISTIC prune would have taken down a
        # multi-hour training run whose checkpoints were already safely on disk. Every
        # downstream path is hardened and no raising input is known — this catches the unknown
        # one. It is LOUD and it sets the same non-zero-exit flag a FAILED prune sets, so the
        # run still ends in failure; it just ends after finishing, not in the middle.
        try:
            msgs, had_failure = prune_run_dir_by_score(
                checkpoint_dir, under_floor=under_floor, apply=True)
        except Exception as e:
            ckpt_prune_failed = True
            print(f"  [CKPT] ERROR: the score-based prune after {what} raised {type(e).__name__}"
                  f": {e} — training CONTINUES (the checkpoints are already saved) and this run "
                  f"will exit non-zero. Nothing was pruned this call.")
            traceback.print_exc()
            return
        for m in msgs:
            print(m)
        if had_failure:
            ckpt_prune_failed = True
            print(f"  [CKPT] a score-based prune FAILED after {what} — see FAILED line above; "
                  f"this run will exit non-zero (TASK_881 gate v: never a silent success).")

    ckpt_prune_failed = False

    # ==== RESTORED: Adaptive Controllers ====
    # Sims warmup schedule (TASK_020)
    sims_schedule = None
    if getattr(config, 'sims_warmup', 0) > 0:
        sims_schedule = SimsSchedule(
            min_sims=getattr(config, 'sims_min', 8),
            max_sims=config.sims,
            warmup_epochs=config.sims_warmup,
            schedule_type=getattr(config, 'sims_schedule', 'linear')
        )
        print(f"  Sims warmup: {config.sims_min} -> {config.sims} over {config.sims_warmup} epochs ({config.sims_schedule})")

    # Adaptive noise controller (TASK_058)
    adaptive_noise_controller = None
    if getattr(config, 'adaptive_noise', False):
        adaptive_noise_controller = AdaptiveNoiseController(
            target_reward_low=getattr(config, 'target_reward_low', 10.0),
            target_reward_high=getattr(config, 'target_reward_high', 20.0),
            adjustment_rate=getattr(config, 'noise_adjustment_rate', 0.05),
            max_temporal_jitter=getattr(config, 'max_temporal_jitter', 5),
            max_time_dilation=getattr(config, 'max_time_dilation', 0.3),
            max_phase_shift=getattr(config, 'max_phase_shift', 0.5),
        )
        print(f"  Adaptive noise: target reward [{config.target_reward_low}, {config.target_reward_high}]")
        print(f"    Max noise: jitter={config.max_temporal_jitter}, dilation={config.max_time_dilation}, phase={config.max_phase_shift}")

    # Adaptive fees controller (historical training)
    adaptive_fee_controller = None
    if getattr(config, 'adaptive_fees', False):
        adaptive_fee_controller = AdaptiveFeeController(
            reward_threshold=getattr(config, 'fee_reward_threshold', 2.5),
            upgrade_after_epochs=getattr(config, 'fee_upgrade_epochs', 10),
        )
        print(f"  Adaptive fees: threshold={config.fee_reward_threshold}, upgrade after {config.fee_upgrade_epochs} epochs")

    # Adaptive burn controller (escalates pressure when hold rate too high)
    adaptive_burn_controller = None
    if getattr(config, 'adaptive_burn', False):
        adaptive_burn_controller = AdaptiveBurnController(
            hold_threshold=getattr(config, 'burn_hold_threshold', 0.60),
            patience=getattr(config, 'burn_patience', 5),
        )
        # Enable if burning_money flag is also set
        if getattr(config, 'burning_money', False):
            adaptive_burn_controller.enable()
        print(f"  Adaptive burn: threshold={config.burn_hold_threshold:.0%}, patience={config.burn_patience}")

    # Safe checkpoint manager
    safe_ckpt_manager = None
    rollback_count = 0
    if getattr(config, 'safe_checkpoints', False):
        safe_ckpt_manager = SafeCheckpointManager(
            checkpoint_dir=os.path.join(checkpoint_dir, 'safe'),
            metric_name='avg_reward',
            patience=getattr(config, 'rollback_patience', 50),
            rollback_threshold=getattr(config, 'rollback_threshold', 0.5),
        )
        print(f"  Safe checkpoints: patience={config.rollback_patience}, threshold={config.rollback_threshold}")

    # TASK_042: ReDo - Dormancy Monitor
    dormancy_monitor = None
    redo_stats = {'dormant_neurons': 0, 'dormant_pct': 0.0, 'recycled_count': 0}
    if getattr(config, 'redo', False):
        dormancy_monitor = DormancyMonitor(
            dormancy_threshold=getattr(config, 'redo_threshold', 0.01),
            check_interval=getattr(config, 'redo_interval', 10),
            min_samples=1000,
        )
        print(f"  ReDo enabled: check every {config.redo_interval} epochs, threshold={config.redo_threshold}")

    # TASK_040: SAC-style Adaptive Entropy Manager
    sac_entropy_manager = None
    if getattr(config, 'sac_entropy', False):
        sac_entropy_manager = AdaptiveEntropyManager(
            num_actions=320,  # 64 assets × 5 actions
            use_sac_entropy=True,
            target_entropy_ratio=getattr(config, 'sac_target_entropy_ratio', 0.5),
            init_alpha=getattr(config, 'sac_init_alpha', 0.01),
            alpha_lr=getattr(config, 'sac_alpha_lr', 3e-4),
        )
        print(f"  SAC entropy: target_ratio={config.sac_target_entropy_ratio}, init_alpha={config.sac_init_alpha}")

    # ==== TASK_125: Causal Expert Imprinting Callback ====
    episode_counter = [0]  # Mutable counter for closure
    expert_confidence_threshold = getattr(config, 'expert_confidence_threshold', 0.001)

    def on_episode_complete_callback(result: dict, score: float):
        """TASK_125: Called when episode completes to populate expert buffer.
        TASK_135: Uses expert_buffer.add_episode() instead of price_cache.
        """
        if expert_buffer is None:
            return

        # Get episode data
        episode_optimal_alpha = result.get('episode_optimal_alpha', 0.0)
        agent_alpha = result.get('alpha', 0.0)
        prices = result.get('prices')

        # BUG FIX (TASK_131c): Count ALL episodes processed (not just stored)
        episode_counter[0] += 1

        if prices is None or episode_optimal_alpha <= 0:
            # Log skip reason periodically
            if episode_counter[0] % 50 == 0:
                reason = "no prices" if prices is None else f"oracle_α={episode_optimal_alpha:.2f}%"
                print(f"[IMPRINT] Skip ep {episode_counter[0]}: {reason}", flush=True)
            return

        # BUG FIX (TASK_131a): Confidence should be based on ORACLE quality, not agent performance
        # Old: confidence = agent_alpha / optimal_alpha (circular dependency - bad agent = no learning)
        # New: confidence = 1.0 for all oracle demos (oracle is ground truth, always valid)
        # The oracle's optimal path is independent of agent performance
        confidence = 1.0

        # BUG FIX (TASK_131a): Remove agent_alpha > 0 filter
        # Old: should_store = agent_alpha > 0 (circular - need demos to learn, need learning to get demos)
        # New: Store ALL episodes where oracle found alpha > 0 (oracle found opportunity)
        # This breaks the circular dependency and lets BC learning start immediately
        should_store = True  # Oracle already filtered by episode_optimal_alpha > 0 above

        # Debug: Log callback invocation
        if episode_counter[0] % 10 == 0:
            print(f"[IMPRINT] Episode {episode_counter[0]}: agent_α={agent_alpha:.2f}%, oracle_α={episode_optimal_alpha:.2f}%", flush=True)

        # TASK_135: Store prices directly in expert_buffer (no disk cache)
        # expert_buffer.add_episode handles both (T, n_assets) and (T, n_assets, 5) formats
        episode_id = expert_buffer.add_episode(prices)

        # TASK_136a: Pre-encode observations (compressed format) at add time
        # This avoids expensive encoding at sample time - 25× speedup
        expert_buffer.preencode_episode(episode_id)

        # Compute optimal actions for each step using path_solver
        from mcts_trader.jax_agent.path_solver import compute_optimal_path, optimal_path_to_policy

        # Get log returns
        log_prices = np.log(prices + 1e-8)
        if len(log_prices.shape) == 2:
            log_returns = log_prices[1:] - log_prices[:-1]  # (T-1, n_assets)
        else:
            # OHLCV - use close (index 3)
            close = prices_ohlcv[:, :, 3]
            log_returns = np.log(close[1:] / np.maximum(close[:-1], 1e-8))

        if len(log_returns) < 2:
            return

        # Compute optimal path
        optimal_path = compute_optimal_path(
            jnp.array(log_returns), k=4, method='greedy', transaction_cost=0.0
        )

        # Get actions from optimal path
        # optimal_path.indices: (T-1, k) - indices of selected assets
        # Convert to action indices: asset_idx * 5 + action_type
        # For simplicity, use: Buy if selected, Hold otherwise
        actions = result.get('actions', np.zeros(len(log_returns), dtype=np.int32))
        rewards = result.get('rewards', np.zeros(len(log_returns), dtype=np.float32))
        episode_return = score

        # Add demos to buffer
        # TASK_132e: Skip steps < window_size to prevent future leakage
        # At step=30 with window_size=60, encoder would see prices[0:60] including future
        window_size = expert_buffer.window_size if expert_buffer else 60
        n_steps = min(len(log_returns), len(actions))
        for step in range(n_steps):
            # Skip early steps where observation window would leak future data
            if step < window_size:
                continue

            # Get optimal action for this step from path
            # optimal_path.indices[step]: (k,) - top-k asset indices selected by oracle
            selected_indices = np.array(optimal_path.indices[step])  # (k,)

            # Get actual returns for selected assets from log_returns
            step_returns = log_returns[step]  # (n_assets,)

            # Compute optimal action: Buy best positive, Sell best negative, or Hold
            # Action encoding: asset_idx * 5 + action_type
            # action_type: 0=Sell, 1=Hold, 2=Buy, 3=LimitSell, 4=LimitBuy
            if len(selected_indices) > 0:
                # Get returns for selected assets
                selected_returns = step_returns[selected_indices]  # (k,)

                # Find best opportunity (highest absolute return)
                best_local_idx = int(np.argmax(np.abs(selected_returns)))
                best_asset = int(selected_indices[best_local_idx])
                best_return = float(selected_returns[best_local_idx])

                # BUG FIX: Remove threshold - oracle selected this asset for a reason
                # Old threshold (0.001 = 0.1%) caused mostly HOLD actions
                if best_return > 0:
                    optimal_action = best_asset * 5 + 2  # Buy
                else:
                    optimal_action = best_asset * 5 + 0  # Sell
                # Never HOLD - if oracle selected asset, there's a directional signal
            else:
                # No clear signal - Hold on asset 0
                optimal_action = 1  # Hold asset 0

            expert_buffer.add_demo(
                episode_id=episode_id,
                step=step,
                action=int(actions[step]),
                optimal_action=optimal_action,
                reward=float(rewards[step]) if step < len(rewards) else 0.0,
                confidence=float(confidence),
                episode_return=float(episode_return),
                regime_tag=getattr(config, 'synthetic_level', 0) or 0,
            )

        # Counter already incremented at start of callback (TASK_131c fix)

    # Set callback to None if imprinting disabled
    imprint_callback = on_episode_complete_callback if getattr(config, 'imprint', False) else None

    # ==== CSV Logging Setup with DS24 Columns ====
    # Base columns (standard training metrics)
    base_columns = [
        'epoch', 'reward', 'reward_ma20', 'fee_bps', 'steps',
        'alpha', 'market_return', 'agent_return', 'episode_optimal_alpha',
        'n_sell', 'n_hold', 'n_buy', 'n_limit_sell', 'n_limit_buy',
        'pct_sell', 'pct_hold', 'pct_buy', 'pct_limit_sell', 'pct_limit_buy',
        'self_play_time', 'train_time',
        'total_loss', 'value_loss', 'policy_loss', 'reward_loss',
        'pred_loss', 'loss_decoder', 'physics_reg', 'lambda_pred',
        'policy_entropy', 'mcts_visit_entropy', 'avg_value', 'value_std', 'value_mse',
        'epsilon_greedy', 'synthetic_level', 'synthetic_noise', 'model_size', 'run_id',
        # Adaptive difficulty metrics (TASK_058)
        'difficulty_level', 'reward_ema',
        'price_noise', 'temporal_jitter', 'time_dilation', 'phase_shift_prob',
        'slippage_bps',
        # PopArt metrics
        'popart_mean', 'popart_std',
        # ReDo metrics (TASK_042)
        'redo_dormant_neurons', 'redo_dormant_pct', 'redo_recycled_count',
        # SAC entropy metrics (TASK_040)
        'sac_alpha', 'sac_entropy_error',
        # TASK_119: Optimal hindsight path metrics
        'amax_regret_loss', 'amax_imprint_loss', 'amax_alpha_capture', 'amax_optimal_gain',
        # TASK_125: Causal Expert Imprinting metrics
        'bc_loss', 'bc_weight', 'expert_action_agreement', 'expert_acceptance_rate', 'expert_demo_count',
        'expert_batch_size', 'bc_ce_loss', 'bc_margin_loss',
        # Generalization eval metrics
        'eval_level', 'eval_reward', 'eval_alpha', 'eval_pct_hold',
    ]

    # Initialize DS24 logger
    ds24_level = getattr(config, 'ds24_level', 0)
    ds24_logger = DS24MetricsLogger(level=ds24_level)

    # Add DS24 columns if logging enabled
    csv_columns = base_columns.copy()
    if ds24_level > 0:
        csv_columns.extend(ds24_logger.get_ds24_columns())

    csv_path = os.path.join('logs', f"training_{config.name}.csv")
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_columns, extrasaction='ignore')
    csv_writer.writeheader()
    print(f"Logging training metrics to: {csv_path}")
    if ds24_level > 0:
        print(f"DS24 Level {ds24_level}: {ds24_logger.LEVEL_DESCRIPTIONS.get(ds24_level, 'Unknown')}")

    # TASK_812: decaying epsilon-greedy target-mix schedule (death-spiral mitigation, from step 0).
    # Mixing is applied in-loop to the policy TARGETS (not the search), so no per-epoch XLA recompile.
    eps_schedule = None
    if getattr(config, 'epsilon_schedule', False):
        from mcts_trader.jax_agent.exploration.epsilon_greedy import EpsilonSchedule
        _decay = getattr(config, 'epsilon_decay_epochs', 0) or max(1, config.epochs // 2)
        eps_schedule = EpsilonSchedule(
            initial=getattr(config, 'epsilon_init', 0.25),
            final=getattr(config, 'epsilon_final', 0.05),
            decay_epochs=_decay,
        )
        print(f"  [TASK_812] e-greedy target-mix schedule: {eps_schedule.initial}->{eps_schedule.final} "
              f"over {_decay} epochs (death-spiral mitigation, applied to policy targets)")

    # TASK_876: pre-initialized so the `finally` block can always record a score,
    # even if training crashes before completing epoch 0.
    epoch = -1
    # TASK_876 cycle 3 / B2: None, NOT 0.0. A run that dies during epoch 0 (OOM on a resume)
    # would otherwise emit {"epoch": 0, "score": 0.0} as an ordinary finite row — and because
    # train_reward_ma20 is routinely NEGATIVE early, that fabricated 0.0 would rank FIRST and
    # a score-driven prune could delete a real, better checkpoint. (Not reachable through the
    # SHIPPED ckpt_gc.py for `checkpoint_<step>` dirs, which it reads as unscored regardless —
    # but it IS reachable for `snapshot_ep<N>`, and it must not become reachable when TASK_881
    # lands the path-first reader.) An unmeasured checkpoint must be UNSCORED (never
    # score-pruned), never top of the ranking.
    ckpt_score, ckpt_metric = None, 'train_reward_ma20'
    # TASK_876 M7: the `finally` save must be stamped with the last epoch that actually
    # FINISHED, not with the epoch training died in.
    last_completed_epoch = 0
    last_completed_score, last_completed_metric = None, ckpt_metric
    # TASK_876 FIX-1: exit-code flag for the FINAL save only. The final save lives inside
    # `finally:`, where re-raising would REPLACE whatever actually stopped training and lose
    # the original error. So an unrecognised failure there is recorded here instead of raised,
    # and turned into `sys.exit(1)` AFTER the try/finally — a point only reached when nothing
    # else is propagating. If something IS propagating, it already exits nonzero on its own.
    # (Chosen over a module-level flag: the exit decision belongs to this one invocation.)
    final_ckpt_save_failed = False

    try:
        pbar = tqdm(range(config.epochs), desc="Training", unit="ep",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        for epoch in pbar:
            # TASK_128 FIX: Periodic cache clearing to prevent XLA cache buildup
            if epoch > 0 and epoch % 50 == 0:
                jax.clear_caches()

            start = time.time()
            rng, search_rng = jax.random.split(rng)

            # ==== RESTORED: Update sims if warmup schedule active ====
            current_sims = config.sims
            if sims_schedule is not None:
                current_sims = sims_schedule.get_sims(epoch)
                trainer.update_simulations(current_sims)

            # ==== Noise Schedule (ramp all 4 components from 0 to target over epochs) ====
            if getattr(config, 'noise_schedule', False) and config.synthetic_level is not None:
                progress = (epoch + 1) / config.epochs  # 0.0 -> 1.0

                # Store original target noise on first epoch (BUG FIX: don't use modified value)
                if not hasattr(config, '_noise_target'):
                    config._noise_target = config.synthetic_noise

                # Calculate all noise components based on progress using ORIGINAL target
                noise_cfg = NoiseConfig(
                    price_noise=config._noise_target * progress,
                    temporal_jitter=int(5 * progress),  # 0 -> 5 steps
                    time_dilation=0.3 * progress,       # 0 -> 0.3
                    phase_shift_prob=0.5 * progress,    # 0 -> 0.5
                )
                config._noise_config = noise_cfg  # Store for display

                # Regenerate data with new noise config every ~2% change
                if epoch == 0 or abs(progress - getattr(config, '_last_progress', -1)) > 0.02:
                    config._last_progress = progress
                    print(f"  [NOISE] {noise_cfg}")
                    if jax_env_states is not None:
                        # Update config temporarily for setup_jax_environment
                        config.synthetic_noise = noise_cfg.price_noise
                        # TASK_124: Use epoch-based seed to prevent memorization
                        # TASK_821: + _seed_offset (0 unless --seed given)
                        jax_env_states, jax_prices, jax_env_config = setup_jax_environment(config, seed=epoch * 1000 + _seed_offset)
                        # TASK_135: Clear expert buffer (prices are stale after noise change)
                        if expert_buffer:
                            expert_buffer.clear()

            # ==== Async Regeneration: Check for completed background tasks ====
            if async_regenerator and pending_regenerations:
                completed = [lvl for lvl, fut in pending_regenerations.items() if fut.done()]
                for lvl in completed:
                    try:
                        result_level, result_data = pending_regenerations[lvl].result()
                        pregenerated_levels[result_level] = result_data
                        # Don't print - too noisy
                    except Exception as e:
                        logging.warning(f"Async regeneration for L{lvl} failed: {e}")
                    del pending_regenerations[lvl]

            # ==== Data Regeneration (anti-memorization) ====
            regenerate_interval = getattr(config, 'regenerate_data_every', 0)
            use_weighted = getattr(config, 'weighted_synthetic', False)
            if regenerate_interval > 0 and (config.synthetic_level is not None or use_weighted):
                if (epoch + 1) % regenerate_interval == 0:
                    # Weighted synthetic: pick random level and use pre-generated data (instant swap)
                    if use_weighted and pregenerated_levels:
                        # (random is imported at module level — TASK_821 needs it seedable from main())
                        import copy
                        levels = list(WEIGHTED_SYNTHETIC_LEVELS.keys())
                        weights = list(WEIGHTED_SYNTHETIC_LEVELS.values())
                        old_level = config.synthetic_level
                        new_level = random.choices(levels, weights=weights, k=1)[0]
                        config.synthetic_level = new_level
                        # Instant swap from pre-generated cache
                        jax_env_states, jax_prices, jax_env_config = pregenerated_levels[new_level]

                        # Queue async regeneration of the OLD level (so it has fresh data next time)
                        if async_regenerator and old_level not in pending_regenerations:
                            cfg_copy = copy.copy(config)
                            cfg_copy.synthetic_level = old_level
                            # TASK_124: Use epoch-based seed to prevent memorization
                            # TASK_821: + _seed_offset (0 unless --seed given)
                            regen_seed = (epoch + 1) * 1000 + old_level + _seed_offset
                            future = async_regenerator.submit(
                                lambda lvl=old_level, cfg=cfg_copy, s=regen_seed: (lvl, setup_jax_environment(cfg, seed=s))
                            )
                            pending_regenerations[old_level] = future

                        # TASK_122: Clear JIT cache to prevent stale price references
                        jax.clear_caches()
                        # TASK_135: Clear expert buffer (prices are stale after level switch)
                        if expert_buffer:
                            expert_buffer.clear()
                        print(f"  [DATA] L{new_level} (regen L{old_level} in bg, JIT cleared)")
                    elif jax_env_states is not None:
                        print(f"  [DATA] Regenerating synthetic data (epoch {epoch + 1})")
                        # Re-initialize JAX env with current adaptive noise
                        # TASK_124: Use epoch-based seed to prevent memorization
                        # TASK_821: + _seed_offset (0 unless --seed given)
                        jax_env_states, jax_prices, jax_env_config = setup_jax_environment(config, seed=epoch * 1000 + _seed_offset)
                        # TASK_122: Clear JIT cache to prevent stale price references
                        jax.clear_caches()
                        # TASK_135: Clear expert buffer (prices are stale after regeneration)
                        if expert_buffer:
                            expert_buffer.clear()
                    else:
                        if env is not None and hasattr(env, 'market') and hasattr(env.market, 'regenerate'):
                            env.market.regenerate()
                        for penv in parallel_envs:
                            if hasattr(penv, 'market') and hasattr(penv.market, 'regenerate'):
                                penv.market.regenerate()

            # TASK_064: Smart allocation & Period Swapping
            start_offsets = None
            if config.smart_allocation and period_data:
                current_period_idx = period_data['current_idx']
                period_lengths = period_data['lengths']

                new_period_idx, start_offsets = compute_env_start_offsets(
                    period_lengths=period_lengths,
                    n_envs=config.parallel_envs,
                    n_steps=config.steps,
                    epoch=epoch,
                    jitter_minutes=config.jitter_minutes,
                    # TASK_821: + _seed_offset (0 unless --seed given)
                    seed=epoch + _seed_offset
                )

                # Check for period switch
                if new_period_idx != current_period_idx:
                    print(f"   [STRATIFIED] Switching to period P{new_period_idx}")
                    new_prices = period_data['prices'][new_period_idx]
                    for penv in parallel_envs:
                        penv.set_prices(new_prices)
                    period_data['current_idx'] = new_period_idx
                    # TASK_135: Clear expert buffer (prices are stale after period switch)
                    if expert_buffer:
                        expert_buffer.clear()

            # Self-play (C2 fix: pass all required parameters)
            # Update PopArt state for value denormalization
            popart_state = trainer.popart.state if hasattr(trainer, 'popart') and trainer.popart else None

            if jax_env_states is not None:
                # ==== TASK_111: Bella on-device generation ====
                # Generate fresh market prices using Bella (VRAM-direct)
                if bella_generator is not None:
                    gen_rng, search_rng = jax.random.split(search_rng)
                    prices_for_play = generate_bella_prices_on_device(
                        bella_generator, gen_rng, n_envs=config.parallel_envs,
                        ohlcv=False,  # TASK_703: Poppy uses close-only
                    )
                    # TASK_135: Clear expert buffer every epoch (Bella generates fresh prices)
                    if expert_buffer:
                        expert_buffer.clear()
                # ==== TASK_110: On-the-fly adversarial generation ====
                # Generate fresh adversarial prices each epoch (~5ms)
                elif fast_gen_model is not None and fast_gen_params is not None:
                    gen_rng, search_rng = jax.random.split(search_rng)
                    prices_for_play = generate_adversarial_on_the_fly(
                        fast_gen_model, fast_gen_params, gen_rng,
                        n_steps=jax_prices.shape[0],
                        n_assets=n_assets,
                        n_envs=config.parallel_envs,
                        regime_bias=getattr(config, 'adversarial_regime_bias', None),
                    )
                else:
                    prices_for_play = jax_prices

                # ==== Runtime noise application for pre-generated data ====
                # Apply price noise at runtime if adaptive noise is active
                if hasattr(config, '_noise_config') and config._noise_config.price_noise > 0:
                    noise_rng, search_rng = jax.random.split(search_rng)
                    prices_for_play = apply_runtime_price_noise(
                        prices_for_play, config._noise_config.price_noise, noise_rng
                    )

                # ==== Runtime fee updates for JAX env config ====
                # Update fees if adaptive fees changed them
                active_jax_config = jax_env_config
                if adaptive_fee_controller is not None:
                    current_fee_cfg = adaptive_fee_controller.get_config()
                    if current_fee_cfg.fee_bps != jax_env_config.fee_bps or current_fee_cfg.slippage_bps != jax_env_config.slippage_bps:
                        active_jax_config = active_jax_config.replace(
                            fee_bps=float(current_fee_cfg.fee_bps),
                            slippage_bps=float(current_fee_cfg.slippage_bps)
                        )

                # ==== Runtime burn updates for JAX env config ====
                # Update burn config if adaptive burn is active
                if adaptive_burn_controller is not None:
                    burn_cfg = adaptive_burn_controller.get_config()
                    if burn_cfg.enabled:
                        active_jax_config = active_jax_config.replace(
                            burning_money=True,
                            burn_base_rate=float(burn_cfg.base_rate),
                            # Random burns (wife with debit card mode) - from initial config
                            burn_random=config.burn_random,
                            burn_random_prob=float(config.burn_random_prob),
                            burn_random_max=float(config.burn_random_max),
                            burn_random_limit=int(config.burn_random_limit),
                        )

                # Use raw_search for JAX path to allow XLA fusion inside lax.scan
                trajectories, scores = run_self_play_jax(
                    jax_env_states, prices_for_play, active_jax_config,
                    search_raw, model.apply, trainer.state.params, search_rng,
                    steps=config.steps,
                    collect_prices=needs_prices or getattr(config, 'imprint', False),  # TASK_125: Need prices for imprinting
                    support_size=model.support_size,
                    discount=config.discount,
                    popart_state=popart_state,
                    use_gae=getattr(config, 'use_gae', False),
                    gae_lambda=getattr(config, 'gae_lambda', 0.95),
                    confidence_threshold=getattr(config, 'confidence_threshold', 0.0),
                    max_positions=getattr(config, 'max_positions', 16),
                    min_cash_threshold=getattr(config, 'min_cash_threshold', 100.0),
                    on_episode_complete=imprint_callback,  # TASK_125: Causal Expert Imprinting
                    vector_action=vector_action,  # TASK_828 Stage 3
                    leaf_eval=getattr(config, 'leaf_eval', 'model'),  # TASK_851
                )
                score = np.mean(scores)

                # TASK_122: Reset env states between epochs to prevent overfitting
                # Without this, agent memorizes the same episode over and over
                reset_rng, search_rng = jax.random.split(search_rng)
                reset_keys = jax.random.split(reset_rng, config.parallel_envs)
                jax_env_states = jit_batch_reset(reset_keys, prices_for_play, active_jax_config)
            elif config.parallel_envs > 1:
                trajectories, scores = run_self_play_batched(
                    parallel_envs, search, model.apply, trainer.state.params, search_rng,
                    steps=config.steps,
                    collect_prices=needs_prices,
                    start_offsets=start_offsets,
                    support_size=model.support_size,
                    discount=config.discount,
                    popart_state=popart_state,
                    use_gae=getattr(config, 'use_gae', False),
                    gae_lambda=getattr(config, 'gae_lambda', 0.95),
                    confidence_threshold=getattr(config, 'confidence_threshold', 0.0),
                )
                score = np.mean(scores)
            else:
                trajectory, score = run_self_play(
                    env, search, model.apply, trainer.state.params, search_rng,
                    steps=config.steps,
                    collect_prices=needs_prices,
                    support_size=model.support_size,
                    discount=config.discount,
                    popart_state=popart_state,
                    use_gae=getattr(config, 'use_gae', False),
                    gae_lambda=getattr(config, 'gae_lambda', 0.95),
                    confidence_threshold=getattr(config, 'confidence_threshold', 0.0),
                )
                trajectories = [trajectory]

            self_play_time = time.time() - start
            all_rewards.append(score)

            # ==== Differential Reward Processing ====
            # TASK_122: Use LAGGED market return (t-1 to t) to avoid future leakage
            # At step t, only prices[0..t] are known - use return that JUST happened
            if getattr(config, 'differential_reward', False):
                for traj in trajectories:
                    if 'prices' in traj and len(traj['prices']) > 1:
                        # Compute per-step market returns (average across assets)
                        prices = np.array(traj['prices'])  # (steps, assets)
                        # price_returns[i] = return from prices[i] to prices[i+1]
                        price_returns = (prices[1:] - prices[:-1]) / (prices[:-1] + 1e-8)
                        market_returns = np.mean(price_returns, axis=1)  # Average across assets
                        # Prepend zero: market_returns[t] = return from t-1 to t (LAGGED)
                        # At step 0: no prior return known, use 0
                        # At step t: use return from t-1 to t (known at decision time)
                        market_returns = np.concatenate([[0.0], market_returns])

                        # Adjust rewards: r_diff = r_agent - w * r_market
                        diff_weight = getattr(config, 'differential_weight', 1.0)
                        original_rewards = np.array(traj['rewards'])
                        traj['rewards'] = list(original_rewards - diff_weight * market_returns[:len(original_rewards)])

            # TASK_812: death-spiral diagnostic + epsilon-greedy target mixing.
            # (1) mcts_visit_entropy = entropy of the RAW MCTS visit distribution (policy targets),
            #     computed BEFORE any mixing so the diagnostic sees true search collapse.
            # (2) if an epsilon schedule is active, mix uniform into the policy TARGETS
            #     pi_train = (1-eps)*pi_mcts + eps*uniform — severs MCTS-sharpening -> policy-sharpening.
            eps_now = float(eps_schedule(epoch)) if eps_schedule is not None else float(getattr(config, 'epsilon_greedy', 0.0))
            _vis_ent = []
            for traj in trajectories:
                if vector_action:
                    # TASK_828 Stage 3: factored policy target (n_steps, n_assets, A).
                    # Entropy = per-asset (over A), meaned over assets+steps; the
                    # epsilon-mix is a per-asset-block uniform over A (replicated).
                    _pm = np.asarray(traj['policy_marginals'], dtype=np.float32)
                    if _pm.ndim == 3 and _pm.shape[0] > 0:
                        _p = np.clip(_pm, 1e-12, 1.0)
                        _vis_ent.append(float(np.mean(-np.sum(_p * np.log(_p), axis=-1))))
                        if eps_now > 0.0:
                            _A = _pm.shape[-1]
                            _mixed = (1.0 - eps_now) * _pm + eps_now * (1.0 / _A)
                            _mixed = _mixed / np.maximum(_mixed.sum(axis=-1, keepdims=True), 1e-12)
                            traj['policy_marginals'] = _mixed.astype(_pm.dtype)
                else:
                    _pp = np.asarray(traj['policy_probs'], dtype=np.float32)  # (n_steps, n_actions)
                    if _pp.ndim == 2 and _pp.shape[0] > 0:
                        _p = np.clip(_pp, 1e-12, 1.0)
                        _vis_ent.append(float(np.mean(-np.sum(_p * np.log(_p), axis=-1))))
                        if eps_now > 0.0:
                            _n = _pp.shape[-1]
                            _mixed = (1.0 - eps_now) * _pp + eps_now * (1.0 / _n)
                            _mixed = _mixed / np.maximum(_mixed.sum(axis=-1, keepdims=True), 1e-12)
                            traj['policy_probs'] = _mixed.astype(_pp.dtype)
            mcts_visit_entropy = float(np.mean(_vis_ent)) if _vis_ent else 0.0

            # Buffer update
            for traj in trajectories:
                # TASK_120: Lazy observation reconstruction (save RAM)
                if 'observations' not in traj and 'window_starts' in traj:
                    from mcts_trader.training.self_play import reconstruct_observations
                    # Reconstruct ON DEMAND - avoids holding all envs' obs in RAM simultaneously
                    # This reduces peak RAM by ~10GB per batch
                    traj['observations'] = reconstruct_observations(
                        traj, n_assets=n_assets, window_size=60
                    )

                if vector_action:
                    # TASK_828 Stage 3: route the per-asset action VECTORS + factored
                    # marginals through the same add_trajectory (buffer stores them in
                    # its vector arrays; positional actions/policy_probs carry vectors).
                    replay_buffer.add_trajectory(
                        jnp.array(traj['observations']), jnp.array(traj['action_vecs']),
                        jnp.array(traj['rewards']), jnp.array(traj['values']),
                        jnp.array(traj['policy_marginals']), jnp.array(traj['dones'])
                    )
                else:
                    replay_buffer.add_trajectory(
                        jnp.array(traj['observations']), jnp.array(traj['actions']),
                        jnp.array(traj['rewards']), jnp.array(traj['values']),
                        jnp.array(traj['policy_probs']), jnp.array(traj['dones'])
                    )
                # Add auxiliary data for prediction loss or amax features
                needs_auxiliary = config.use_prediction_loss or getattr(config, 'amax', None)
                if needs_auxiliary and 'prices' in traj:
                    replay_buffer.add_auxiliary(prices=traj['prices'], next_prices=traj['next_prices'])

            # Train
            train_start = time.time()
            metrics = trainer.train(
                replay_buffer,
                num_iterations=50,
                rng=rng,
                expert_buffer=expert_buffer,  # TASK_125: Causal Expert Imprinting
            )
            train_time = time.time() - train_start

            # === Metric Extraction ===
            # Action stats from trajectories
            all_actions = []
            for traj in trajectories:
                if vector_action:
                    # TASK_828 Stage 3: action_vecs entries ARE per-asset action-types
                    # in {0..4}; flatten so the `% N_ACTIONS_PER_ASSET` histogram below
                    # is meaningful (identity on 0..4). Scalar path uses flat indices.
                    all_actions.extend(np.asarray(traj['action_vecs']).reshape(-1).tolist())
                else:
                    all_actions.extend(traj['actions'])
            actions_arr = np.array(all_actions)
            total_actions = max(len(actions_arr), 1)

            n_sell = int(np.sum(actions_arr % N_ACTIONS_PER_ASSET == 0))
            n_hold = int(np.sum(actions_arr % N_ACTIONS_PER_ASSET == 1))
            n_buy = int(np.sum(actions_arr % N_ACTIONS_PER_ASSET == 2))
            n_limit_sell = int(np.sum(actions_arr % N_ACTIONS_PER_ASSET == 3))
            n_limit_buy = int(np.sum(actions_arr % N_ACTIONS_PER_ASSET == 4))

            pct_sell = 100.0 * n_sell / total_actions
            pct_hold = 100.0 * n_hold / total_actions
            pct_buy = 100.0 * n_buy / total_actions
            pct_limit_sell = 100.0 * n_limit_sell / total_actions
            pct_limit_buy = 100.0 * n_limit_buy / total_actions

            # Market baseline metrics (alpha = agent - market)
            trajectory = trajectories[0] if trajectories else {}
            alpha = trajectory.get('alpha', 0.0)
            market_return = trajectory.get('market_return', 0.0)
            agent_return = trajectory.get('agent_return', 0.0)
            episode_optimal_alpha = trajectory.get('episode_optimal_alpha', 0.0)  # TASK_126

            # Extract root values for value metrics
            root_values_all = []
            for traj in trajectories:
                if 'mcts_root_values' in traj:
                    root_values_all.extend(traj['mcts_root_values'])
            avg_value = np.mean(root_values_all) if root_values_all else 0.0
            value_std = np.std(root_values_all) if root_values_all else 0.0

            # Loss breakdown (get last value from each list)
            def get_last(metric_list, default=0.0):
                if isinstance(metric_list, list) and len(metric_list) > 0:
                    return float(metric_list[-1])
                return default

            loss = get_last(metrics.get('total_loss', [0]))
            val_loss = get_last(metrics.get('value_loss', [0]))
            pol_loss = get_last(metrics.get('policy_loss', [0]))
            rew_loss = get_last(metrics.get('reward_loss', [0]))
            pred_loss = get_last(metrics.get('loss_prediction', [0]))
            loss_decoder = get_last(metrics.get('loss_decoder', [0]))
            physics_reg = get_last(metrics.get('loss_physics_reg', [0]))
            lambda_pred = get_last(metrics.get('lambda_pred', [0]))

            # Policy analytics
            policy_entropy = get_last(metrics.get('policy_entropy', [0]))
            value_mse = get_last(metrics.get('value_mse', [0]))

            # Fee (from env or config)
            fee_bps = getattr(env, 'trading_fee_bps', 0) if env else config.fee
            steps = len(trajectory.get('actions', trajectory.get('action_vecs', [])))  # TASK_828: vector traj has action_vecs
            reward_ma20 = np.mean(all_rewards[-20:]) if len(all_rewards) >= 20 else np.mean(all_rewards) if all_rewards else score

            # TASK_876: default per-checkpoint score = rolling train reward (what
            # SafeCheckpointManager already uses). Overridden below with the periodic
            # in-env eval score for this epoch when one is actually measured — never
            # prune on a metric the run didn't measure.
            ckpt_score, ckpt_metric = float(reward_ma20), 'train_reward_ma20'

            # PopArt stats (if available)
            popart_mean = 0.0
            popart_std = 1.0
            if hasattr(trainer, 'popart') and trainer.popart is not None:
                popart_mean = float(trainer.popart.mean)
                popart_std = float(trainer.popart.std)

            # ==== RESTORED: Adaptive Controllers ====
            # Update adaptive noise controller
            if adaptive_noise_controller is not None:
                noise_config = adaptive_noise_controller.update(score)
                # Apply new noise level to synthetic environment if applicable
                if config.synthetic_level is not None and env is not None:
                    if hasattr(env, 'set_noise'):
                        env.set_noise(noise_config.price_noise)
                    for penv in parallel_envs:
                        if hasattr(penv, 'set_noise'):
                            penv.set_noise(noise_config.price_noise)

            # Update adaptive fees controller
            if adaptive_fee_controller is not None:
                fee_config = adaptive_fee_controller.update(score)
                # Apply new fee/slippage to environments
                if env is not None:
                    env.trading_fee_bps = fee_config.fee_bps
                    env.slippage_bps = fee_config.slippage_bps
                for penv in parallel_envs:
                    penv.trading_fee_bps = fee_config.fee_bps
                    penv.slippage_bps = fee_config.slippage_bps
                fee_bps = fee_config.fee_bps  # Update for logging

            # ==== TASK_040: SAC-style Entropy Adjustment ====
            sac_metrics = {}
            if sac_entropy_manager is not None:
                # Update SAC entropy manager with observed policy entropy
                sac_alpha, sac_metrics = sac_entropy_manager.update(jnp.array(policy_entropy))
                # Update trainer's entropy coefficient
                if hasattr(trainer, 'entropy_coef'):
                    old_entropy_coef = trainer.entropy_coef
                    trainer.entropy_coef = sac_alpha
                    if abs(old_entropy_coef - sac_alpha) > 0.001:
                        print(f"  [SAC] α: {old_entropy_coef:.4f} -> {sac_alpha:.4f} (target entropy error: {sac_metrics.get('sac_entropy_error', 0):.3f})")

            # Adaptive entropy: boost when policy collapses to HOLD
            if getattr(config, 'adaptive_entropy', False):
                hold_ratio = n_hold / total_actions
                entropy_hold_threshold = getattr(config, 'entropy_hold_threshold', 0.70)
                entropy_boost_low = getattr(config, 'entropy_boost_low', 0.15)
                entropy_boost_high = getattr(config, 'entropy_boost_high', 0.30)

                if hold_ratio > entropy_hold_threshold + 0.10:
                    # Very high HOLD% -> aggressive boost
                    new_entropy_coef = entropy_boost_high
                elif hold_ratio > entropy_hold_threshold:
                    # Moderate HOLD% -> moderate boost
                    new_entropy_coef = entropy_boost_low
                else:
                    # Normal -> base entropy
                    new_entropy_coef = getattr(config, 'entropy_coef', 0.01)

                if trainer.entropy_coef != new_entropy_coef:
                    print(f"  [ADAPTIVE ENTROPY] HOLD={hold_ratio*100:.1f}% -> entropy_coef: {trainer.entropy_coef:.3f} -> {new_entropy_coef:.3f}")
                    trainer.entropy_coef = new_entropy_coef

            # ==== Adaptive Burn: Escalate pressure when hold rate too high ====
            if adaptive_burn_controller is not None:
                hold_ratio = n_hold / total_actions if total_actions > 0 else 0.0
                burn_cfg = adaptive_burn_controller.update(hold_ratio)

                # Update environment burn parameters if stage changed
                if burn_cfg.enabled:
                    config.burn_base_rate = burn_cfg.base_rate
                    config.burn_random = burn_cfg.random_burn
                    config.burn_random_prob = burn_cfg.random_prob

                    # Log stage changes
                    if (epoch + 1) % 10 == 0:
                        print(f"  [BURN] {adaptive_burn_controller.get_status()}")

            # ==== TASK_042: ReDo - Recycling Dormant Neurons ====
            if dormancy_monitor is not None:
                # Collect activation stats from model output (use latent states from trajectories)
                for traj in trajectories:
                    if 'latent_states' in traj:
                        dormancy_monitor.update({'latent': np.array(traj['latent_states'])})

                # Check if it's time to recycle
                if dormancy_monitor.should_check(epoch + 1):
                    stats = dormancy_monitor.get_dormancy_stats()
                    redo_stats['dormant_neurons'] = stats.get('dormant_neurons', 0)
                    redo_stats['dormant_pct'] = stats.get('dormancy_percent', 0.0)

                    # Only recycle if dormancy exceeds threshold
                    if redo_stats['dormant_pct'] > 5.0:  # 5% dormancy threshold for recycling
                        rng, redo_rng = jax.random.split(rng)
                        dormant_masks = dormancy_monitor.get_dormant_mask()

                        # Recycle dormant neurons
                        new_params = recycle_dormant_neurons(
                            trainer.state.params,
                            dormant_masks,
                            redo_rng,
                            init_scale=getattr(config, 'redo_init_scale', 0.01),
                        )

                        # Update trainer state with recycled params
                        trainer.state = trainer.state.replace(params=new_params)
                        redo_stats['recycled_count'] = sum(np.sum(m) for m in dormant_masks.values())
                        print(f"  [ReDo] Recycled {redo_stats['recycled_count']} dormant neurons ({redo_stats['dormant_pct']:.1f}% dormancy)")
                    else:
                        print(f"  [ReDo] {redo_stats['dormant_neurons']} dormant neurons ({redo_stats['dormant_pct']:.1f}%) - below threshold")

                    dormancy_monitor.reset()

            # Build console output
            alpha_str = f"α: {alpha:+.2f}% (Mkt: {market_return:+.1f}%)"
            action_str = f"S/H/B/LS/LB: {n_sell}/{n_hold}/{n_buy}/{n_limit_sell}/{n_limit_buy}"

            pred_str = ""
            if config.use_prediction_loss:
                pred_str = f" | Pred: {pred_loss:.4f} (λ={lambda_pred:.3f})"

            warmup_str = " | (warmup)" if epoch < 5 else ""

            # Noise info for noise_schedule mode (all 4 components)
            noise_str = ""
            if getattr(config, 'noise_schedule', False):
                noise_cfg = getattr(config, '_noise_config', None)
                if noise_cfg:
                    noise_str = (f" | Noise[P:{noise_cfg.price_noise:.0%} "
                                f"J:{noise_cfg.temporal_jitter} "
                                f"D:{noise_cfg.time_dilation:.0%} "
                                f"Φ:{noise_cfg.phase_shift_prob:.0%}]")

            # Update progress bar with key metrics
            pbar.set_postfix_str(f"R:{score:+.1f} α:{alpha:+.1f}% H:{pct_hold:.0f}%{noise_str}{warmup_str}")

            # Detailed line (tqdm.write avoids breaking progress bar)
            tqdm.write(f"Ep {epoch+1:4d} | Reward: {score:+7.2f} | {alpha_str} | Fee: {fee_bps:2d} | "
                       f"{action_str}{noise_str} | SP: {self_play_time:5.1f}s | Train: {train_time:4.1f}s{warmup_str}")

            # === Build CSV Row ===
            csv_row = {
                'epoch': epoch + 1,
                'reward': score,
                'reward_ma20': reward_ma20,
                'fee_bps': fee_bps,
                'steps': steps,
                'alpha': alpha,
                'market_return': market_return,
                'agent_return': agent_return,
                'episode_optimal_alpha': episode_optimal_alpha,  # TASK_126: Oracle on same data
                'n_sell': n_sell,
                'n_hold': n_hold,
                'n_buy': n_buy,
                'n_limit_sell': n_limit_sell,
                'n_limit_buy': n_limit_buy,
                'pct_sell': pct_sell,
                'pct_hold': pct_hold,
                'pct_buy': pct_buy,
                'pct_limit_sell': pct_limit_sell,
                'pct_limit_buy': pct_limit_buy,
                'self_play_time': self_play_time,
                'train_time': train_time,
                'total_loss': loss,
                'value_loss': val_loss,
                'policy_loss': pol_loss,
                'reward_loss': rew_loss,
                'pred_loss': pred_loss,
                'loss_decoder': loss_decoder,
                'physics_reg': physics_reg,
                'lambda_pred': lambda_pred,
                'policy_entropy': policy_entropy,
                'mcts_visit_entropy': mcts_visit_entropy,  # TASK_812: raw MCTS visit-distribution entropy (death-spiral triad)
                'avg_value': avg_value,
                'value_std': value_std,
                'value_mse': value_mse,
                'epsilon_greedy': eps_now,  # TASK_812: actual epsilon used this epoch (scheduled if --epsilon-schedule)
                'synthetic_level': getattr(config, 'synthetic_level', None),
                'synthetic_noise': getattr(config, 'synthetic_noise', 0.0),
                'model_size': getattr(config, 'model_size', 'small'),
                'run_id': config.name,
                # Adaptive difficulty metrics (TASK_058)
                'difficulty_level': adaptive_noise_controller.difficulty_level if adaptive_noise_controller else 0.0,
                'reward_ema': adaptive_noise_controller.reward_ema if adaptive_noise_controller else 0.0,
                'price_noise': getattr(config, '_noise_config', NoiseConfig()).price_noise if hasattr(config, '_noise_config') else 0.0,
                'temporal_jitter': getattr(config, '_noise_config', NoiseConfig()).temporal_jitter if hasattr(config, '_noise_config') else 0,
                'time_dilation': getattr(config, '_noise_config', NoiseConfig()).time_dilation if hasattr(config, '_noise_config') else 0.0,
                'phase_shift_prob': getattr(config, '_noise_config', NoiseConfig()).phase_shift_prob if hasattr(config, '_noise_config') else 0.0,
                'slippage_bps': adaptive_fee_controller.get_config().slippage_bps if adaptive_fee_controller is not None else getattr(config, 'slippage', 0),
                'popart_mean': popart_mean,
                'popart_std': popart_std,
                # ReDo metrics
                'redo_dormant_neurons': redo_stats.get('dormant_neurons', 0),
                'redo_dormant_pct': redo_stats.get('dormant_pct', 0.0),
                'redo_recycled_count': redo_stats.get('recycled_count', 0),
                # SAC entropy metrics
                'sac_alpha': sac_metrics.get('sac_alpha', 0.0) if sac_metrics else 0.0,
                'sac_entropy_error': sac_metrics.get('sac_entropy_error', 0.0) if sac_metrics else 0.0,
                # TASK_119: Optimal hindsight path metrics
                'amax_regret_loss': get_last(metrics.get('regret_loss', [0])),
                'amax_imprint_loss': get_last(metrics.get('imprint_loss', [0])),
                'amax_alpha_capture': get_last(metrics.get('alpha_capture', [0])),
                'amax_optimal_gain': get_last(metrics.get('optimal_gain', [0])),
                # TASK_125: Causal Expert Imprinting metrics
                # TASK_132B FIX: Use 'key' in metrics, not metrics.get('key') - falsy check fails on 0.0
                'bc_loss': get_last(metrics.get('bc_loss', [0])) if 'bc_loss' in metrics else None,
                'bc_weight': get_last(metrics.get('bc_weight', [0])) if 'bc_weight' in metrics else None,
                'expert_action_agreement': get_last(metrics.get('expert_action_agreement', [0])) if 'expert_action_agreement' in metrics else None,
                'expert_acceptance_rate': get_last(metrics.get('expert_acceptance_rate', [0])) if 'expert_acceptance_rate' in metrics else None,
                'expert_batch_size': get_last(metrics.get('expert_batch_size', [0])) if 'expert_batch_size' in metrics else 0,
                'bc_ce_loss': get_last(metrics.get('bc_ce_loss', [0])) if 'bc_ce_loss' in metrics else None,
                'bc_margin_loss': get_last(metrics.get('bc_margin_loss', [0])) if 'bc_margin_loss' in metrics else None,
                'expert_demo_count': expert_buffer.demo_count if expert_buffer else 0,
                # Eval metrics (filled later if eval runs this epoch)
                'eval_level': 0,
                'eval_reward': 0.0,
                'eval_alpha': 0.0,
                'eval_pct_hold': 0.0,
            }

            # === DS24 Enhanced Logging ===
            if ds24_level > 0:
                ds24_logger.reset_row()

                # Level 1: Market baseline (already in base columns, but also log to DS24)
                # TASK_119: Include optimal hindsight path metrics
                ds24_logger.log_market_baseline(
                    alpha=alpha,
                    market_return=market_return,
                    agent_return=agent_return,
                    optimal_alpha=get_last(metrics.get('optimal_gain', [0])),
                    alpha_capture=get_last(metrics.get('alpha_capture', [0])),
                )

                # Level 1+: Training config flags
                ds24_logger.log_training_config(
                    use_gae=getattr(config, 'use_gae', False),
                    gae_lambda=getattr(config, 'gae_lambda', 0.95),
                    per_head_clipping=getattr(config, 'per_head_clipping', False),
                    clip_value=getattr(config, 'clip_grad_value', 1.0),
                    clip_policy=getattr(config, 'clip_grad_policy', 1.0),
                    clip_dynamics=getattr(config, 'clip_grad_dynamics', 1.0),
                    clip_encoder=getattr(config, 'clip_grad_encoder', 1.0),
                    epsilon_greedy=getattr(config, 'epsilon_greedy', 0.0),
                )

                # Level 2+: Gradient norms (from trainer metrics)
                ds24_logger.log_gradient_norms(
                    value_grad_norm=get_last(metrics.get('grad_norm_value', [0])),
                    policy_grad_norm=get_last(metrics.get('grad_norm_policy', [0])),
                    dynamics_grad_norm=get_last(metrics.get('grad_norm_dynamics', [0])),
                )

                # Level 2+: ReDo stats (TASK_042)
                if getattr(config, 'redo', False):
                    ds24_logger.log_redo_stats(
                        dormant_neurons=int(redo_stats.get('dormant_neurons', 0)),
                        dormant_pct=float(redo_stats.get('dormant_pct', 0.0)),
                        recycled_count=int(redo_stats.get('recycled_count', 0)),
                    )

                # Level 2+: ATC stats (TASK_043)
                if getattr(config, 'atc_loss', False):
                    ds24_logger.log_atc_stats(
                        loss=get_last(metrics.get('atc_loss', [0])),
                        pos_sim=get_last(metrics.get('atc_pos_sim', [0])),
                        neg_sim=get_last(metrics.get('atc_neg_sim', [0])),
                        temperature=getattr(config, 'atc_temperature', 0.1),
                    )

                # Level 2+: Expert buffer / BC stats (TASK_125)
                if expert_buffer is not None:
                    ds24_logger.log_expert_buffer(
                        bc_loss=get_last(metrics.get('bc_loss', [])) if metrics.get('bc_loss') else None,
                        bc_weight=get_last(metrics.get('bc_weight', [])) if metrics.get('bc_weight') else None,
                        # TASK_132B: Renamed to storage_rate (demos/steps), distinct from action_agreement
                        expert_storage_rate=expert_buffer.demo_count / max(1, episode_counter[0] * config.steps),
                        demo_count=expert_buffer.demo_count,
                        agent_count=expert_buffer.agent_count,
                        buffer_capacity=expert_buffer.demo_capacity + expert_buffer.agent_capacity,
                    )

                # Level 2+: Planning metrics with horizon errors
                if trajectories:
                    # Extract MCTS root values
                    root_values = []
                    horizon_returns_all = {1: [], 5: [], 10: []}

                    for traj in trajectories:
                        # MCTS root values
                        if 'mcts_root_values' in traj:
                            root_values.extend(traj['mcts_root_values'])

                        # Horizon returns for prediction error calculation
                        if 'horizon_returns' in traj:
                            for h in [1, 5, 10]:
                                if h in traj['horizon_returns']:
                                    horizon_returns_all[h].extend(traj['horizon_returns'][h])

                    # Compute visit entropy from policy probs (as proxy for visit distribution)
                    visit_entropy = 0.0
                    policy_count = 0
                    for traj in trajectories:
                        if 'policy_probs' in traj and len(traj['policy_probs']) > 0:
                            probs = np.array(traj['policy_probs'])
                            # Entropy: -sum(p * log(p))
                            ent = -np.sum(probs * np.log(probs + 1e-10), axis=-1)
                            visit_entropy += np.mean(ent)
                            policy_count += 1
                    if policy_count > 0:
                        visit_entropy /= policy_count

                    # Compute horizon prediction errors
                    horizon_predictions = {}
                    horizon_actuals = {}
                    if root_values and any(horizon_returns_all[h] for h in [1, 5, 10]):
                        root_values_arr = np.array(root_values)
                        for h in [1, 5, 10]:
                            if horizon_returns_all[h]:
                                horizon_actuals[h] = np.array(horizon_returns_all[h])
                                # Use root values as horizon predictions (model's estimate of future value)
                                # Truncate to match lengths
                                min_len = min(len(root_values_arr), len(horizon_actuals[h]))
                                horizon_predictions[h] = root_values_arr[:min_len]
                                horizon_actuals[h] = horizon_actuals[h][:min_len]

                    ds24_logger.log_planning_metrics(
                        root_values=np.array(root_values) if root_values else None,
                        visit_counts=None,  # Would need to modify search to return raw counts
                        actual_rewards=None,
                        horizon_predictions=horizon_predictions if horizon_predictions else None,
                        horizon_actuals=horizon_actuals if horizon_actuals else None,
                    )

                    # Set visit entropy directly (computed from policy probs)
                    ds24_logger.current_row['mcts_visit_entropy'] = visit_entropy

                # Level 3+: Physics learned parameters AND G components
                if ds24_level >= 3 and config.use_prediction_loss:
                    try:
                        # Extract physics params from model
                        params = trainer.state.params
                        physics_params = DifferentiableMarketPhysics.get_learned_params_from_dict(
                            params.get('price_predictor', {}).get('physics_core', {})
                        )
                        # Show extracted values on first epoch
                        if epoch == 0:
                            print(f"  [DS24] Physics: α={physics_params.get('alpha_mean', 0.0):.4f}, β={physics_params.get('beta_mean', 0.0):.4f}, GARCH={physics_params.get('garch_persistence', 0.0):.4f}")

                        # Run physics forward pass to get G components
                        G_momentum, G_reversion, G_volatility, G_correlation = None, None, None, None
                        if trajectories and 'prices' in trajectories[0] and len(trajectories[0]['prices']) > 1:
                            try:
                                prices = np.array(trajectories[0]['prices'][:10])  # Sample first 10 steps
                                log_prices = jnp.log(jnp.maximum(prices, 1e-8))
                                log_returns = jnp.diff(log_prices, axis=0)
                                # Use simple moving average
                                moving_avg = log_prices[:-1]  # Simplified
                                noise = jax.random.normal(jax.random.PRNGKey(epoch), log_returns.shape)

                                # Run physics core forward pass
                                physics_core_params = params.get('price_predictor', {}).get('physics_core', {})
                                if physics_core_params:
                                    physics_output, _ = DifferentiableMarketPhysics(
                                        n_assets=64, n_factors=8, use_garch=True
                                    ).apply(
                                        {'params': physics_core_params},
                                        log_prices[:-1], log_returns, moving_avg, noise
                                    )
                                    G_momentum = np.array(physics_output.G_momentum)
                                    G_reversion = np.array(physics_output.G_reversion)
                                    G_volatility = np.array(physics_output.G_volatility)
                                    G_correlation = np.array(physics_output.G_correlation)
                            except Exception as e:
                                if epoch == 0:
                                    print(f"  [DS24] Physics forward pass failed: {e}")

                        ds24_logger.log_physics_components(
                            G_momentum=G_momentum,
                            G_reversion=G_reversion,
                            G_volatility=G_volatility,
                            G_correlation=G_correlation,
                            alpha=np.array([physics_params.get('alpha_mean', 0.0)]),
                            beta=np.array([physics_params.get('beta_mean', 0.0)]),
                            garch_alpha=np.array([physics_params.get('garch_persistence', 0.0) * 0.3]),
                            garch_beta=np.array([physics_params.get('garch_persistence', 0.0) * 0.7]),
                        )
                    except Exception as e:
                        if epoch == 0:
                            print(f"  [DS24] Physics extraction failed: {e}")
                        pass  # Physics params not available

                # Level 2+: TRM cycle intermediates (every ds24_intermediates_every epochs)
                intermediates_every = getattr(config, 'ds24_intermediates_every', 10)
                if ds24_level >= 2 and (epoch + 1) % intermediates_every == 0 and trajectories:
                    try:
                        # Get a sample observation from trajectories
                        sample_obs = jnp.array(trajectories[0]['observations'][:1])

                        # Run inference with intermediates capture
                        output = model.apply(
                            {'params': trainer.state.params},
                            sample_obs,
                            mode='initial_inference_with_intermediates'
                        )

                        if 'trm_intermediates' in output:
                            intermediates = output['trm_intermediates']
                            # Convert to numpy for logging
                            latent_deltas_np = {}
                            output_deltas_np = {}

                            if 'latent_deltas' in intermediates:
                                for key, val in intermediates['latent_deltas'].items():
                                    latent_deltas_np[key] = np.array(val)

                            if 'output_deltas' in intermediates:
                                for key, val in intermediates['output_deltas'].items():
                                    output_deltas_np[key] = np.array(val)

                            latent_final = np.array(intermediates.get('latent_final')) if 'latent_final' in intermediates else None

                            ds24_logger.log_trm_dynamics(
                                latent_deltas=latent_deltas_np if latent_deltas_np else None,
                                output_deltas=output_deltas_np if output_deltas_np else None,
                                latent_final=latent_final,
                            )
                    except Exception as e:
                        pass  # TRM intermediates not available

                # Get DS24 values and merge into row
                ds24_values = ds24_logger.get_current_values()
                csv_row.update(ds24_values)

            # ==== RESTORED: Generalization Evaluation (before CSV write) ====
            eval_every = getattr(config, 'eval_every', 0)
            if eval_every > 0 and (epoch + 1) % eval_every == 0:
                rng, eval_rng = jax.random.split(rng)
                eval_start = time.time()
                eval_result = run_generalization_eval(
                    eval_level=getattr(config, 'eval_level', 4),
                    search_fn=search,
                    model_apply_fn=model.apply,
                    params=trainer.state.params,
                    rng_key=eval_rng,
                    steps=getattr(config, 'eval_steps', 200),
                    support_size=model.support_size,
                    n_assets=n_assets,
                    fee=config.fee,
                    slippage=config.slippage,
                    vector_action=vector_action,  # TASK_828 Stage 3
                )
                eval_time = time.time() - eval_start
                # Update csv_row with eval results
                csv_row['eval_level'] = getattr(config, 'eval_level', 4)
                if eval_result.get('skipped'):
                    # TASK_828 Stage 3: vector-action eval is skipped (see evaluator.py).
                    # Leave eval_reward/eval_alpha/eval_pct_hold ABSENT for this row —
                    # no fabricated value — and print no stale/None eval line. The
                    # one-time skip warning is emitted by run_generalization_eval.
                    pass
                else:
                    csv_row['eval_reward'] = eval_result.get('reward', 0.0)
                    csv_row['eval_alpha'] = eval_result.get('alpha', 0.0)
                    csv_row['eval_pct_hold'] = eval_result.get('pct_hold', 0.0)
                    print(f"  [EVAL L{config.eval_level}] Reward={eval_result['reward']:+.2f}% | "
                          f"alpha={eval_result['alpha']:+.2f}% | HOLD={eval_result['pct_hold']:.0f}% | "
                          f"Time: {eval_time:.1f}s")
                    # TASK_876: this epoch measured a real in-env eval score — prefer it
                    # over the rolling train reward for the checkpoint's recorded score.
                    ckpt_score = float(eval_result['reward'])
                    ckpt_metric = f'eval_reward_L{config.eval_level}'

            # Write row
            csv_writer.writerow(csv_row)
            csv_file.flush()

            # ==== RESTORED: Safe Checkpoint Integration ====
            if safe_ckpt_manager is not None:
                # Check if we should save (performance improved)
                if safe_ckpt_manager.should_save(reward_ma20):
                    # Build checkpoint data
                    ckpt_data = {
                        'params': trainer.state.params,
                        'opt_state': trainer.state.opt_state,
                        'step_count': trainer.state.step_count,
                    }
                    # TASK_876 FIX-2: the manager writes TWO checkpoint dirs on its first
                    # save — `checkpoint_golden` and `checkpoint_epoch_<N>` — from the same
                    # params and the same metric. Emission is supposed to be unconditional at
                    # EVERY save, so capture whether golden gets created by this call and
                    # score it too. (`golden_checkpoint` is None until the first save and is
                    # only ever set there, so this is an exact test for "golden written now".)
                    _golden_before = safe_ckpt_manager.golden_checkpoint
                    safe_ckpt_manager.save_checkpoint(ckpt_data, epoch + 1, reward_ma20)
                    # TASK_876: record the score the manager itself decided on (reward_ma20,
                    # under its own configured metric name), not the eval-preferring
                    # `ckpt_score`/`ckpt_metric` used by the other save paths below — this
                    # is what the manager's own keep_best_n/rollback logic actually measured.
                    # TASK_876 B3/M4: the manager names its dirs `checkpoint_epoch_<N>`, so
                    # record the real path as well as the epoch key. NOTE the shipped
                    # `tools/ckpt_gc.py` cannot resolve either name (see
                    # `append_checkpoint_score`'s docstring); the path-first reader is
                    # TASK_881. Until then both read as UNSCORED, i.e. never score-pruned.
                    _sc_metric = getattr(safe_ckpt_manager, 'metric_name',
                                         'train_reward_ma20')
                    append_checkpoint_score(
                        safe_ckpt_manager.checkpoint_dir, epoch + 1, float(reward_ma20),
                        _sc_metric,
                        path=os.path.join(str(safe_ckpt_manager.checkpoint_dir),
                                          f'checkpoint_epoch_{epoch + 1}'),
                        step=int(trainer.state.step_count))
                    if (_golden_before is None
                            and safe_ckpt_manager.golden_checkpoint is not None):
                        # Same epoch, same params, same metric as the row above — a distinct
                        # on-disk artifact, so a distinct row. The manager never prunes
                        # golden; the row exists so the audit trail is complete rather than
                        # to make it prunable.
                        append_checkpoint_score(
                            safe_ckpt_manager.checkpoint_dir, epoch + 1, float(reward_ma20),
                            _sc_metric,
                            path=str(safe_ckpt_manager.golden_checkpoint),
                            step=int(trainer.state.step_count),
                            note='golden checkpoint (first SafeCheckpointManager save; '
                                 'never pruned by the manager)')

                # Check if we should rollback (performance degraded)
                if safe_ckpt_manager.should_rollback(reward_ma20):
                    print(f"  [SAFE CHECKPOINT] Rolling back to best checkpoint...")
                    ckpt_data, ckpt_epoch = safe_ckpt_manager.load_best_checkpoint()
                    # Restore trainer state
                    trainer.state = trainer.state.replace(
                        params=ckpt_data['params'],
                        opt_state=ckpt_data['opt_state'],
                        step_count=ckpt_data.get('step_count', trainer.state.step_count),
                    )
                    rollback_count += 1
                    print(f"  [SAFE CHECKPOINT] Restored from epoch {ckpt_epoch} (rollback #{rollback_count})")
                    # TASK_876 cycle 3 / M6: the params in `trainer.state` are now from epoch
                    # {ckpt_epoch}, but `ckpt_score` still holds THIS epoch's (degraded)
                    # measurement. The snapshot block below sits outside this if/elif chain and
                    # would attach that score to the rolled-back weights — a measurement of one
                    # model recorded against another. Invalidate it: unscored => never score-pruned.
                    ckpt_score = None
                    ckpt_metric = (f"{ckpt_metric} INVALIDATED (rolled back to epoch "
                                   f"{ckpt_epoch} after this epoch was measured)")

            # Standard checkpoint (fallback if safe not enabled)
            elif (epoch + 1) % 10 == 0 and not config.no_checkpoint:
                # TASK_876 (D33): `overwrite=False` (trainer.py) makes flax refuse instead of
                # silently mass-deleting when this save's step collides with a checkpoint
                # already on disk — e.g. a rollback-resume into a directory that still holds
                # later, worse checkpoints. That is the intended trade (a refused save beats a
                # protection-blind delete), and THAT refusal must not take down the whole run.
                # FIX-1: only that refusal. Any OTHER exception (ENOSPC, EACCES, an Orbax
                # internal) propagates out of this loop and kills the run with a nonzero exit,
                # as the bare save call here always did — dojo_manager gates on returncode.
                # Fix-2 B1: a save behind the max step does NOT refuse on the Orbax backend —
                # it returns a path flax then deletes inside the same call — so the helper
                # verifies the artifact is really on disk before any success is reported or
                # any score row is emitted.
                written, _save_msgs = _attempt_checkpoint_save(
                    trainer, checkpoint_dir, flax_keep, 'PERIODIC SAVE', epoch + 1,
                    checkpoint_dir)
                for _m in _save_msgs:
                    print(_m)
                if written is not None:
                    # TASK_876 B3: flax names the dir by state.step_count, NOT by epoch —
                    # record the path flax actually wrote, so a path-first reader can bind
                    # this score to a real artifact. TASK_881 made tools/ckpt_gc.py's reader
                    # PATH-FIRST (it resolves this row by `path`, never by parsing the dir
                    # basename), so a `checkpoint_<step>` row now resolves correctly instead of
                    # reading as permanently UNSCORED. See append_checkpoint_score's docstring.
                    append_checkpoint_score(checkpoint_dir, epoch + 1, ckpt_score,
                                            ckpt_metric, path=written,
                                            step=int(trainer.state.step_count))
                    _maybe_prune_by_score('the periodic save')

            # Snapshot checkpoint (periodic named snapshots that don't get overwritten)
            snapshot_every = getattr(config, 'snapshot_every', 0)
            if snapshot_every > 0 and (epoch + 1) % snapshot_every == 0 and not config.no_checkpoint:
                snapshot_dir = os.path.join(checkpoint_dir, f'snapshot_ep{epoch + 1}')
                os.makedirs(snapshot_dir, exist_ok=True)
                # See the periodic-save comment above: with overwrite=False this can refuse
                # (step collision) or silently vanish (save behind the max step, self-pruned
                # by flax's keep-N inside the same call). Both are reported loudly and
                # training continues; neither may be reported as a success. Anything else
                # (FIX-1) propagates and kills the run.
                _snap_written, _save_msgs = _attempt_checkpoint_save(
                    trainer, snapshot_dir, flax_keep, 'SNAPSHOT SAVE', epoch + 1,
                    snapshot_dir)
                for _m in _save_msgs:
                    print(_m)
                if _snap_written is not None:
                    # TASK_876: the score belongs to the SNAPSHOT dir (the leaf the GC sees),
                    # written into the PARENT run dir alongside its snapshot_ep<N> siblings.
                    # All three checkpoint naming schemes resolve now (TASK_881's path-first
                    # reader), so this is no longer the one family that happens to round-trip.
                    append_checkpoint_score(checkpoint_dir, epoch + 1, ckpt_score,
                                            ckpt_metric, path=snapshot_dir,
                                            step=int(trainer.state.step_count))
                    print(f"  [SNAPSHOT] Saved to {snapshot_dir}")
                    # TASK_881 scope 3: this is the family flax's own keep=N never bounds —
                    # score-based retention matters most here.
                    _maybe_prune_by_score('the snapshot save')

            # TASK_876 M7: only now is this epoch complete. The `finally` save below must
            # attribute its params to THIS epoch, not to the one that crashed half-way.
            last_completed_epoch = epoch + 1
            last_completed_score, last_completed_metric = ckpt_score, ckpt_metric

    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    finally:
        # Cleanup
        if async_regenerator:
            async_regenerator.shutdown(wait=False)
        csv_file.close()
        print(f"Training metrics saved to: {csv_path}")
        if not config.no_checkpoint:
            print("Saving final checkpoint...")
            # TASK_876 fix-2 B1: the final save reports success ONLY if the checkpoint is
            # really on disk — this is the run's result, and "Checkpoint saved to …" used to
            # print unconditionally, even for a save flax deleted inside the same call.
            # TASK_876 FIX-1: this call CAN raise (the helper only swallows the three
            # designed outcomes) and we are inside `finally:` — re-raising here would
            # REPLACE whatever actually stopped training and lose the original error. So the
            # unrecognised failure is reported in full AND recorded in
            # `final_ckpt_save_failed`, which becomes `sys.exit(1)` after the try/finally.
            # Both properties the fix requires therefore hold: the original exception is
            # preserved and still propagates, and the process exits nonzero either way.
            written, _save_msgs = None, []
            try:
                written, _save_msgs = _attempt_checkpoint_save(
                    trainer, checkpoint_dir, flax_keep, 'FINAL SAVE', last_completed_epoch,
                    checkpoint_dir)
            except Exception as _final_e:
                final_ckpt_save_failed = True
                print(f"  [CKPT] FINAL SAVE FAILED at epoch {last_completed_epoch}: "
                      f"{type(_final_e).__name__}: {_final_e} — NO final checkpoint is on "
                      f"disk for this run. This is an unrecognised failure (not a step "
                      f"collision, not a sidecar, not a vanished save); the run will exit "
                      f"NONZERO so no orchestrator gating on returncode can advance past it.")
                import traceback
                traceback.print_exc()
            for _m in _save_msgs:
                print(_m)
            if written is not None:
                # TASK_876 M7: attribute this save to the last COMPLETED epoch. `epoch + 1`
                # would stamp the crashed/interrupted epoch onto params that never finished
                # it — an audit-trail lie. `last_completed_*` are pre-initialized before the
                # `try:` above, so this is safe even if training died before epoch 0.
                _partial = (epoch + 1) != last_completed_epoch
                if last_completed_score is None:
                    # No epoch ever completed (or a rollback invalidated the last one): there
                    # is no measurement to record. `append_checkpoint_score` writes score:null
                    # for a non-finite/None value, which ckpt_gc reads as UNSCORED (never
                    # score-pruned).
                    _note = (f"final save after training stopped during epoch {epoch + 1} with "
                             f"no completed, validly-measured epoch — params are UNMEASURED")
                else:
                    _note = (f"final save; training stopped during epoch {epoch + 1} — params "
                             f"are as of completed epoch {last_completed_epoch}" if _partial
                             else "final save at end of run")
                append_checkpoint_score(
                    checkpoint_dir, last_completed_epoch, last_completed_score,
                    last_completed_metric, path=written, step=int(trainer.state.step_count),
                    note=_note)
                print(f"  Checkpoint saved to {written}")
                _maybe_prune_by_score('the final save')
        print(f"\n{'='*70}")
        # TASK_876 FIX-1: never claim COMPLETE when the run's own result failed to save.
        print("TRAINING FAILED: FINAL CHECKPOINT NOT SAVED" if final_ckpt_save_failed
              else "TRAINING COMPLETE!")
        print(f"{'='*70}")

    # TASK_876 FIX-1: reached only when nothing propagated out of the try/finally above (a
    # propagating exception already exits nonzero and carries the real cause). Exiting here
    # rather than raising inside `finally:` is what keeps the original error intact.
    # TASK_881 gate v: a FAILED score-based prune (rename ok, rmtree failed — orphaned bytes)
    # must never be a silent success either, so it joins the same nonzero-exit path.
    if final_ckpt_save_failed or ckpt_prune_failed:
        sys.exit(1)

if __name__ == "__main__":
    import torch # Required for encoder
    main()
