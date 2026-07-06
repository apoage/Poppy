"""
Diffusion Teacher - Main orchestration class.

Combines:
    - FlowMatchingGenerator: Generates market sequences
    - OptimalSolver: Computes hindsight optimal PnL
    - RegretCurriculum: Adjusts difficulty based on agent performance
    - MarketEncoder/VAE: Encodes market data to latent space

Provides the interface for the adaptive curriculum loop.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
from flax.training import train_state
from typing import Optional, Dict, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import logging
import time

from .config import TeacherConfig
from .flow_matching import FlowMatchingGenerator, flow_matching_loss, REGIME_TO_IDX
from .conv_flow_matching import ConvFlowMatchingGenerator, flow_matching_loss as conv_flow_loss
from .encoder import MarketEncoder, MarketVAE, vae_loss
from .optimal_solver import OptimalSolver, optimal_pnl_jax, regret_score
from .regret_curriculum import RegretCurriculum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GeneratedSequence:
    """A generated market sequence with metadata."""

    prices: jnp.ndarray  # (window_size, n_assets, 5) OHLCV
    latent: jnp.ndarray  # Latent representation
    regime: str  # Generated regime
    difficulty: float  # Difficulty level used
    optimal_pnl: float  # Hindsight optimal PnL


# Use standard TrainState - store extra state in teacher class
TrainState = train_state.TrainState


class DiffusionTeacher:
    """
    Main teacher class for adaptive curriculum learning.

    Workflow:
        1. Generate market sequence at current difficulty
        2. Compute optimal PnL for sequence
        3. Agent trades on sequence, returns actual PnL
        4. Compute regret, update curriculum
        5. Repeat

    Can operate in two modes:
        - Training mode: Train the generator on historical data
        - Inference mode: Generate sequences for RL training
    """

    def __init__(
        self,
        config: TeacherConfig,
        key: Optional[jax.Array] = None,
    ):
        """
        Initialize the Diffusion Teacher.

        Args:
            config: Teacher configuration
            key: JAX random key (will create one if not provided)
        """
        self.config = config

        if key is None:
            key = jax.random.PRNGKey(config.seed)
        self.key = key

        # Initialize components
        self._init_generator()
        self._init_encoder()
        self._init_solver()
        self._init_curriculum()

        # Training state
        self.generator_state: Optional[TrainState] = None
        self.encoder_state: Optional[TrainState] = None
        self.generator_key: Optional[jax.Array] = None
        self.encoder_key: Optional[jax.Array] = None

        # Metrics
        self.metrics_history: Dict[str, list] = {
            "generator_loss": [],
            "encoder_loss": [],
            "regret": [],
            "difficulty": [],
        }

    def _init_generator(self) -> None:
        """Initialize flow matching generator based on architecture config."""
        architecture = getattr(self.config, 'flow_architecture', 'transformer')

        if architecture == "conv":
            # Conv-based generator: O(n) memory, 302M params at default settings
            self.generator = ConvFlowMatchingGenerator(
                hidden_dim=self.config.flow_hidden_dim,
                num_layers=self.config.flow_layers,
                ff_dim=getattr(self.config, 'flow_ff_dim', self.config.flow_hidden_dim * 2),
                cond_dim=getattr(self.config, 'flow_cond_dim', self.config.flow_hidden_dim // 2),
                dropout=self.config.flow_dropout,
                sigma_min=self.config.sigma_min,
            )
            logger.info(f"Generator initialized: CONV ({self.config.flow_hidden_dim}d, {self.config.flow_layers}L)")
        else:
            # Transformer-based generator: O(n²) memory
            self.generator = FlowMatchingGenerator(
                hidden_dim=self.config.flow_hidden_dim,
                num_layers=self.config.flow_layers,
                num_heads=self.config.flow_heads,
                dropout=self.config.flow_dropout,
                sigma_min=self.config.sigma_min,
            )
            logger.info(f"Generator initialized: TRANSFORMER ({self.config.flow_hidden_dim}d, {self.config.flow_layers}L)")

    def _init_encoder(self) -> None:
        """Initialize market encoder (VAE)."""
        self.encoder = MarketVAE(
            latent_dim=self.config.latent_dim,
            hidden_dim=self.config.encoder_ff_dim // 4,
            num_heads=self.config.encoder_heads,
            num_layers=self.config.encoder_layers,
            ff_dim=self.config.encoder_ff_dim,
            dropout=self.config.encoder_dropout,
            window_size=self.config.window_size,
            n_assets=self.config.n_assets,
            n_features=len(self.config.features),
        )
        logger.info("Encoder (VAE) initialized")

    def _init_solver(self) -> None:
        """Initialize optimal solver."""
        self.solver = OptimalSolver(
            fee_bps=0.0,  # Perfect hindsight for curriculum
            algorithm="peak_valley",
        )
        logger.info("Optimal solver initialized")

    def _init_curriculum(self) -> None:
        """Initialize curriculum controller."""
        self.curriculum = RegretCurriculum(
            difficulty_init=self.config.difficulty_init,
            difficulty_min=self.config.difficulty_min,
            difficulty_max=self.config.difficulty_max,
            difficulty_step=self.config.difficulty_step,
            regret_low_threshold=self.config.regret_low_threshold,
            regret_high_threshold=self.config.regret_high_threshold,
            ema_alpha=self.config.regret_ema_alpha,
        )
        logger.info("Curriculum controller initialized")

    def init_training(self, direct_mode: bool = True) -> None:
        """
        Initialize training states for generator and encoder.

        Creates optimizer states and initializes model parameters.

        Args:
            direct_mode: If True, generator learns on raw prices (n_assets * n_features).
                        If False, generator learns on encoded latents (latent_dim).
                        Default True for conv architecture (recommended).
        """
        # Dummy inputs for initialization
        batch_size = 2
        window_size = self.config.window_size
        n_assets = self.config.n_assets
        n_features = len(self.config.features)
        latent_dim = self.config.latent_dim

        # Split keys
        self.key, gen_key, enc_key = jax.random.split(self.key, 3)

        # Initialize generator
        # For direct mode: use n_assets * n_features (generates prices)
        # For VAE mode: use latent_dim (generates latents)
        if direct_mode:
            input_dim = n_assets * n_features  # 64 * 5 = 320
            logger.info(f"Generator input: {n_assets} assets × {n_features} features = {input_dim}D (direct mode)")
        else:
            input_dim = latent_dim
            logger.info(f"Generator input: {latent_dim}D latent (VAE mode)")

        dummy_input = jnp.zeros((batch_size, window_size, input_dim))
        dummy_t = jnp.zeros((batch_size,))
        dummy_regime = jnp.zeros((batch_size,), dtype=jnp.int32)

        gen_key, noise_key = jax.random.split(gen_key)
        gen_params = self.generator.init(
            gen_key, dummy_input, dummy_t, noise_key, dummy_regime, training=False
        )

        # Generator optimizer
        gen_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            decay_steps=self.config.max_steps,
        )
        gen_optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(gen_schedule, weight_decay=self.config.weight_decay),
        )

        self.generator_state = TrainState.create(
            apply_fn=self.generator.apply,
            params=gen_params,
            tx=gen_optimizer,
        )
        self.generator_key = gen_key

        # Initialize encoder (VAE)
        dummy_prices = jnp.zeros((batch_size, window_size, n_assets, n_features))
        enc_params = self.encoder.init(enc_key, dummy_prices, enc_key, training=False)

        # Encoder optimizer
        enc_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            decay_steps=self.config.max_steps,
        )
        enc_optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(enc_schedule, weight_decay=self.config.weight_decay),
        )

        self.encoder_state = TrainState.create(
            apply_fn=self.encoder.apply,
            params=enc_params,
            tx=enc_optimizer,
        )
        self.encoder_key = enc_key

        # Count parameters
        n_gen_params = sum(p.size for p in jax.tree_util.tree_leaves(gen_params))
        n_enc_params = sum(p.size for p in jax.tree_util.tree_leaves(enc_params))

        logger.info(
            f"Training initialized: "
            f"Generator {n_gen_params / 1e6:.2f}M params, "
            f"Encoder {n_enc_params / 1e6:.2f}M params"
        )

    def train_encoder_step(
        self,
        prices: jnp.ndarray,
    ) -> Dict[str, float]:
        """
        Single training step for encoder (VAE).

        Args:
            prices: (batch, window_size, n_assets, features) normalized prices

        Returns:
            Dictionary of metrics
        """
        if self.encoder_state is None:
            raise RuntimeError("Call init_training() first")

        # Split key - need separate keys for reparameterization and dropout
        self.encoder_key, reparam_key, dropout_key = jax.random.split(self.encoder_key, 3)

        # HIGH-001 FIX: Use JIT-compiled training step
        new_state, metrics = self._encoder_train_step_jit(
            self.encoder_state, prices, reparam_key, dropout_key
        )
        self.encoder_state = new_state

        return {k: float(v) for k, v in metrics.items()}

    @staticmethod
    @jax.jit
    def _encoder_train_step_jit(state, prices, reparam_key, dropout_key):
        """HIGH-001 FIX: JIT-compiled encoder training step."""
        def loss_fn(params):
            recon, mu, log_var = state.apply_fn(
                params, prices, reparam_key, training=True,
                rngs={'dropout': dropout_key}
            )
            loss, metrics = vae_loss(recon, prices, mu, log_var)
            return loss, metrics

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, metrics

    def train_generator_step(
        self,
        latents: jnp.ndarray,
        regimes: Optional[jnp.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Single training step for generator (Flow Matching).

        Args:
            latents: (batch, window_size, latent_dim) encoded market data
            regimes: (batch,) optional regime indices

        Returns:
            Dictionary of metrics
        """
        if self.generator_state is None:
            raise RuntimeError("Call init_training() first")

        self.generator_key, t_key, noise_key, dropout_key = jax.random.split(self.generator_key, 4)

        # Sample time uniformly
        batch_size = latents.shape[0]
        t = jax.random.uniform(t_key, (batch_size,))

        # HIGH-001 FIX: Use JIT-compiled training step
        new_state, metrics = self._generator_train_step_jit(
            self.generator_state, latents, t, noise_key, regimes, dropout_key
        )
        self.generator_state = new_state

        return {k: float(v) for k, v in metrics.items()}

    @staticmethod
    @jax.jit
    def _generator_train_step_jit(state, latents, t, noise_key, regimes, dropout_key):
        """HIGH-001 FIX: JIT-compiled generator training step."""
        def loss_fn(params):
            v_pred, v_target, _ = state.apply_fn(
                params, latents, t, noise_key, regimes, training=True,
                rngs={'dropout': dropout_key}
            )
            loss, metrics = flow_matching_loss(v_pred, v_target)
            return loss, metrics

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, metrics

    def encode(self, prices: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Encode market prices to latent space.

        Args:
            prices: (batch, window_size, n_assets, features)

        Returns:
            mu: Latent mean
            log_var: Latent log variance
        """
        if self.encoder_state is None:
            raise RuntimeError("Call init_training() or load checkpoint first")

        return self.encoder.apply(
            self.encoder_state.params, prices, training=False, method=self.encoder.encode
        )

    def decode(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        Decode latent vectors to market prices.

        Args:
            z: (batch, latent_dim) latent vectors

        Returns:
            (batch, window_size, n_assets, features) reconstructed prices
        """
        if self.encoder_state is None:
            raise RuntimeError("Call init_training() or load checkpoint first")

        return self.encoder.apply(
            self.encoder_state.params, z, training=False, method=self.encoder.decode
        )

    def generate(
        self,
        batch_size: int = 1,
        regime: Optional[str] = None,
        num_steps: int = 1,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Generate market sequences.

        Args:
            batch_size: Number of sequences to generate
            regime: Optional regime conditioning
            num_steps: Number of integration steps (1 = fastest)

        Returns:
            latents: (batch, window_size, latent_dim) generated latent sequences
            prices: (batch, window_size, n_assets, features) decoded prices (if decoder available)
        """
        if self.generator_state is None:
            raise RuntimeError("Call init_training() or load checkpoint first")

        self.key, sample_key = jax.random.split(self.key)

        # Convert regime to index
        if regime is not None:
            regime_idx = jnp.full((batch_size,), REGIME_TO_IDX.get(regime, 4))
        else:
            # Sample regime based on curriculum
            regime_idx = jnp.array(
                [REGIME_TO_IDX[self.curriculum.sample_regime()] for _ in range(batch_size)]
            )

        # Generate latents
        shape = (batch_size, self.config.window_size, self.config.latent_dim)

        if num_steps == 1:
            latents = self.generator.apply(
                self.generator_state.params,
                sample_key,
                shape,
                regime_idx,
                method=self.generator.sample_single_step,
            )
        else:
            latents = self.generator.apply(
                self.generator_state.params,
                sample_key,
                shape,
                regime_idx,
                num_steps,
                method=self.generator.sample,
            )

        # Decode to prices (if encoder is trained)
        if self.encoder_state is not None:
            # For VAE, we need to decode each timestep
            # This is expensive - in practice, might skip decoding
            # and work directly in latent space
            prices = None  # TODO: Implement proper decoding
        else:
            prices = None

        return latents, prices

    def compute_optimal(self, prices: np.ndarray) -> np.ndarray:
        """
        Compute optimal PnL for price sequences.

        Args:
            prices: (batch, window_size, n_assets) or (batch, window_size, n_assets, 5)

        Returns:
            (batch,) optimal PnL values
        """
        # Use JAX for speed
        return np.array(optimal_pnl_jax(jnp.array(prices)))

    def update_curriculum(
        self,
        optimal_pnl: np.ndarray,
        actual_pnl: np.ndarray,
    ) -> Tuple[float, str]:
        """
        Update curriculum based on agent performance.

        Args:
            optimal_pnl: (batch,) optimal PnL
            actual_pnl: (batch,) actual PnL achieved

        Returns:
            new_difficulty: Updated difficulty
            action: "increase", "decrease", or "maintain"
        """
        # Compute regret
        regret = regret_score(jnp.array(optimal_pnl), jnp.array(actual_pnl))
        mean_regret = float(jnp.mean(regret))

        # Update curriculum
        new_diff, action = self.curriculum.update(mean_regret)

        # Track metrics
        self.metrics_history["regret"].append(mean_regret)
        self.metrics_history["difficulty"].append(new_diff)

        return new_diff, action

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        metrics = self.curriculum.get_metrics()

        if self.generator_state is not None:
            metrics["generator_step"] = self.generator_state.step

        if self.encoder_state is not None:
            metrics["encoder_step"] = self.encoder_state.step

        return metrics

    def save_checkpoint(self, path: str) -> None:
        """Save all states to checkpoint."""
        import orbax.checkpoint as ocp
        import shutil

        path = Path(path).resolve()  # Convert to absolute path
        path.mkdir(parents=True, exist_ok=True)

        # Save using orbax
        checkpointer = ocp.PyTreeCheckpointer()

        if self.generator_state is not None:
            gen_path = path / "generator"
            # Remove existing checkpoint to allow overwrite
            if gen_path.exists():
                shutil.rmtree(gen_path)
            checkpointer.save(
                str(gen_path),
                self.generator_state,
            )

        if self.encoder_state is not None:
            enc_path = path / "encoder"
            if enc_path.exists():
                shutil.rmtree(enc_path)
            checkpointer.save(
                str(enc_path),
                self.encoder_state,
            )

        # Save curriculum
        self.curriculum.save(str(path / "curriculum.json"))

        # Save config
        self.config.save(str(path / "config.json"))

        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load checkpoint."""
        import orbax.checkpoint as ocp

        path = Path(path)
        checkpointer = ocp.PyTreeCheckpointer()

        # Initialize states first
        self.init_training()

        # Load generator
        gen_path = path / "generator"
        if gen_path.exists():
            self.generator_state = checkpointer.restore(
                str(gen_path), item=self.generator_state
            )
            logger.info("Generator loaded")

        # Load encoder
        enc_path = path / "encoder"
        if enc_path.exists():
            self.encoder_state = checkpointer.restore(
                str(enc_path), item=self.encoder_state
            )
            logger.info("Encoder loaded")

        # Load curriculum
        curriculum_path = path / "curriculum.json"
        if curriculum_path.exists():
            self.curriculum.load(str(curriculum_path))
            logger.info("Curriculum loaded")

        logger.info(f"Checkpoint loaded from {path}")


# =============================================================================
# Test
# =============================================================================
if __name__ == "__main__":
    from .config import TeacherConfig

    print("=" * 60)
    print("Diffusion Teacher Test")
    print("=" * 60)

    # Create config with smaller sizes for testing
    config = TeacherConfig(
        n_assets=16,  # Smaller for test
        window_size=100,
        latent_dim=64,
        encoder_layers=2,
        encoder_heads=2,
        flow_hidden_dim=128,
        flow_layers=2,
        flow_heads=2,
    )

    # Initialize teacher
    teacher = DiffusionTeacher(config)
    teacher.init_training()

    print("\n--- Encoder Training Step ---")
    # Dummy training data
    dummy_prices = jax.random.normal(
        jax.random.PRNGKey(0),
        (4, config.window_size, config.n_assets, len(config.features)),
    )

    enc_metrics = teacher.train_encoder_step(dummy_prices)
    print(f"Encoder metrics: {enc_metrics}")

    print("\n--- Generator Training Step ---")
    dummy_latents = jax.random.normal(
        jax.random.PRNGKey(1),
        (4, config.window_size, config.latent_dim),
    )
    dummy_regimes = jnp.array([0, 1, 2, 3])

    gen_metrics = teacher.train_generator_step(dummy_latents, dummy_regimes)
    print(f"Generator metrics: {gen_metrics}")

    print("\n--- Generation ---")
    latents, _ = teacher.generate(batch_size=2, regime="volatile")
    print(f"Generated latents shape: {latents.shape}")

    print("\n--- Curriculum Update ---")
    optimal = np.array([0.1, 0.15])
    actual = np.array([0.05, 0.08])
    new_diff, action = teacher.update_curriculum(optimal, actual)
    print(f"New difficulty: {new_diff}, action: {action}")

    print("\n--- Metrics ---")
    metrics = teacher.get_metrics()
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\nTest complete!")
