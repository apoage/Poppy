#!/usr/bin/env python3
"""
Diffusion Teacher Training Script.

Entry point for training the market generator.

Usage:
    python -m diff_teacher.train --db-path trading_data.db --epochs 100

Two-phase training:
    Phase 1: Train VAE encoder on historical data
    Phase 2: Train Flow Matching generator on encoded latents
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Optional
import sys

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from .config import TeacherConfig
from .data_loader import MarketDataLoader, RainbowTableLoader
from .teacher import DiffusionTeacher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Diffusion Teacher market generator"
    )

    # Data
    parser.add_argument(
        "--db-path",
        type=str,
        default="trading_data.db",
        help="Path to trading database",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default=None,
        help="Path to cache file for faster loading",
    )
    parser.add_argument(
        "--n-assets",
        type=int,
        default=64,
        help="Number of assets to load",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=400,
        help="Window size (steps per sequence)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2020-01-01",
        help="Training data start date",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-12-31",
        help="Training data end date",
    )

    # Rainbow Table Mode (recommended for full coverage)
    parser.add_argument(
        "--rainbow",
        action="store_true",
        help="Use rainbow table loader (training_schedule.json) for 96-year coverage",
    )
    parser.add_argument(
        "--schedule-path",
        type=str,
        default="training_schedule.json",
        help="Path to training schedule JSON (for --rainbow mode)",
    )
    parser.add_argument(
        "--windows-per-epoch",
        type=int,
        default=50000,
        help="Windows to sample per epoch (for --rainbow mode)",
    )
    parser.add_argument(
        "--batches-per-period",
        type=int,
        default=100,
        help="Batches per period before switching (for --rainbow mode)",
    )

    # Model
    parser.add_argument(
        "--architecture",
        type=str,
        choices=["conv", "transformer"],
        default="conv",
        help="Flow matching architecture: conv (302M, O(n) memory) or transformer (22M, O(n²))",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=256,
        help="Latent space dimension",
    )
    parser.add_argument(
        "--encoder-layers",
        type=int,
        default=4,
        help="Number of encoder layers",
    )
    parser.add_argument(
        "--flow-layers",
        type=int,
        default=None,
        help="Number of flow matching layers (default: 16 for conv, 6 for transformer)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Flow hidden dimension (default: 1536 for conv, 512 for transformer)",
    )

    # Training
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--encoder-epochs",
        type=int,
        default=50,
        help="Epochs to train encoder (Phase 1)",
    )
    parser.add_argument(
        "--generator-epochs",
        type=int,
        default=100,
        help="Epochs to train generator (Phase 2)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1000,
        help="Warmup steps for learning rate",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/diff_teacher",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint",
    )

    # Logging
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Log every N steps",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1000,
        help="Evaluate every N steps",
    )

    # Hardware
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 training",
    )

    # Mode
    parser.add_argument(
        "--phase",
        type=str,
        choices=["encoder", "generator", "both", "direct"],
        default="direct",
        help="Training phase: direct=train on raw prices (recommended), both=VAE+generator",
    )

    return parser.parse_args()


def create_config(args) -> TeacherConfig:
    """Create config from arguments."""
    # Set architecture-dependent defaults
    if args.architecture == "conv":
        default_hidden_dim = 1536
        default_flow_layers = 16
        default_ff_dim = 3072
        default_cond_dim = 768
    else:  # transformer
        default_hidden_dim = 512
        default_flow_layers = 6
        default_ff_dim = 2048
        default_cond_dim = 256

    hidden_dim = args.hidden_dim if args.hidden_dim else default_hidden_dim
    flow_layers = args.flow_layers if args.flow_layers else default_flow_layers

    return TeacherConfig(
        db_path=args.db_path,
        n_assets=args.n_assets,
        window_size=args.window_size,
        train_start=args.start_date,
        train_end=args.end_date,
        latent_dim=args.latent_dim,
        encoder_layers=args.encoder_layers,
        flow_architecture=args.architecture,
        flow_hidden_dim=hidden_dim,
        flow_layers=flow_layers,
        flow_ff_dim=default_ff_dim if args.architecture == "conv" else hidden_dim * 4,
        flow_cond_dim=default_cond_dim if args.architecture == "conv" else hidden_dim // 2,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        log_every=args.log_every,
        eval_every=args.eval_every,
        dtype="float16" if args.fp16 else "float32",
        seed=args.seed,
    )


def train_encoder(
    teacher: DiffusionTeacher,
    data_loader,  # MarketDataLoader or RainbowTableLoader
    epochs: int,
    batch_size: int,
    log_every: int = 100,
    checkpoint_every: int = 10,
    checkpoint_dir: str = "checkpoints/diff_teacher",
    streaming_mode: bool = False,
    batches_per_period: int = 100,
) -> None:
    """
    Phase 1: Train VAE encoder on historical data.

    Args:
        teacher: DiffusionTeacher instance
        data_loader: Market data loader (MarketDataLoader or RainbowTableLoader)
        epochs: Number of training epochs
        batch_size: Batch size
        log_every: Log every N steps
        checkpoint_every: Save checkpoint every N epochs
        checkpoint_dir: Checkpoint directory
        streaming_mode: Use streaming mode for RainbowTableLoader (faster)
        batches_per_period: Batches per period before switching (streaming mode)
    """
    logger.info("=" * 60)
    logger.info("Phase 1: Training Encoder (VAE)")
    logger.info("=" * 60)

    n_windows = len(data_loader)
    steps_per_epoch = n_windows // batch_size

    if streaming_mode:
        logger.info(f"Data: {n_windows} windows/epoch (streaming), {steps_per_epoch} steps/epoch")
        logger.info(f"Switching periods every {batches_per_period} batches")
    else:
        logger.info(f"Data: {n_windows} windows, {steps_per_epoch} steps/epoch")

    rng = np.random.default_rng(teacher.config.seed)

    total_steps = 0
    best_loss = float("inf")

    for epoch in range(epochs):
        epoch_losses = []
        epoch_start = time.time()

        # Progress bar for epoch
        pbar = tqdm(
            range(steps_per_epoch),
            desc=f"Epoch {epoch + 1}/{epochs}",
            leave=False,
        )

        # For streaming mode, track batches since last period switch
        batches_since_switch = 0

        for step in pbar:
            # Get batch - use streaming for RainbowTableLoader
            if streaming_mode:
                if batches_since_switch == 0 or batches_since_switch >= batches_per_period:
                    data_loader.switch_period(rng)
                    batches_since_switch = 0
                prices, _, regimes = data_loader.get_batch_streaming(batch_size, rng)
                batches_since_switch += 1
            else:
                prices, _, regimes = data_loader.get_batch_array(batch_size, rng)

            # Normalize
            prices_norm = data_loader.normalize(prices)
            prices_jax = jnp.array(prices_norm)

            # Training step
            metrics = teacher.train_encoder_step(prices_jax)
            epoch_losses.append(metrics["loss"])
            total_steps += 1

            # Update progress bar
            pbar.set_postfix(
                loss=f"{metrics['loss']:.4f}",
                recon=f"{metrics['recon_loss']:.4f}",
                kl=f"{metrics['kl_loss']:.4f}",
            )

            # Logging
            if total_steps % log_every == 0:
                logger.info(
                    f"Step {total_steps}: "
                    f"loss={metrics['loss']:.4f}, "
                    f"recon={metrics['recon_loss']:.4f}, "
                    f"kl={metrics['kl_loss']:.4f}"
                )

        # Epoch summary
        epoch_loss = np.mean(epoch_losses)
        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch {epoch + 1}/{epochs} complete: "
            f"loss={epoch_loss:.4f}, "
            f"time={epoch_time:.1f}s"
        )

        # Save checkpoint
        if (epoch + 1) % checkpoint_every == 0:
            ckpt_path = Path(checkpoint_dir) / f"encoder_epoch_{epoch + 1}"
            teacher.save_checkpoint(str(ckpt_path))
            logger.info(f"Checkpoint saved: {ckpt_path}")

        # Best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = Path(checkpoint_dir) / "encoder_best"
            teacher.save_checkpoint(str(best_path))
            logger.info(f"New best model: loss={best_loss:.4f}")

    logger.info("Phase 1 complete!")


def train_generator(
    teacher: DiffusionTeacher,
    data_loader,  # MarketDataLoader or RainbowTableLoader
    epochs: int,
    batch_size: int,
    log_every: int = 100,
    checkpoint_every: int = 10,
    checkpoint_dir: str = "checkpoints/diff_teacher",
    streaming_mode: bool = False,
    batches_per_period: int = 100,
) -> None:
    """
    Phase 2: Train Flow Matching generator on encoded latents.

    Args:
        teacher: DiffusionTeacher instance (with trained encoder)
        data_loader: Market data loader (MarketDataLoader or RainbowTableLoader)
        epochs: Number of training epochs
        batch_size: Batch size
        log_every: Log every N steps
        checkpoint_every: Save checkpoint every N epochs
        checkpoint_dir: Checkpoint directory
        streaming_mode: Use streaming mode for RainbowTableLoader (faster)
        batches_per_period: Batches per period before switching (streaming mode)
    """
    logger.info("=" * 60)
    logger.info("Phase 2: Training Generator (Flow Matching)")
    logger.info("=" * 60)

    n_windows = len(data_loader)
    steps_per_epoch = n_windows // batch_size

    if streaming_mode:
        logger.info(f"Data: {n_windows} windows/epoch (streaming), {steps_per_epoch} steps/epoch")
    else:
        logger.info(f"Data: {n_windows} windows, {steps_per_epoch} steps/epoch")

    rng = np.random.default_rng(teacher.config.seed + 1000)

    total_steps = 0
    best_loss = float("inf")

    # Regime mapping
    from .flow_matching import REGIME_TO_IDX

    for epoch in range(epochs):
        epoch_losses = []
        epoch_start = time.time()

        pbar = tqdm(
            range(steps_per_epoch),
            desc=f"Epoch {epoch + 1}/{epochs}",
            leave=False,
        )

        # For streaming mode, track batches since last period switch
        batches_since_switch = 0

        for step in pbar:
            # Get batch - use streaming for RainbowTableLoader
            if streaming_mode:
                if batches_since_switch == 0 or batches_since_switch >= batches_per_period:
                    data_loader.switch_period(rng)
                    batches_since_switch = 0
                prices, _, regimes = data_loader.get_batch_streaming(batch_size, rng)
                batches_since_switch += 1
            else:
                prices, _, regimes = data_loader.get_batch_array(batch_size, rng)

            # Normalize and encode
            prices_norm = data_loader.normalize(prices)
            prices_jax = jnp.array(prices_norm)

            # Encode to latent (using trained encoder)
            mu, log_var = teacher.encode(prices_jax)

            # Sample from latent (reparameterization)
            teacher.key, sample_key = jax.random.split(teacher.key)
            std = jnp.exp(0.5 * log_var)
            eps = jax.random.normal(sample_key, mu.shape)
            latents = mu + eps * std

            # Expand latents to sequence
            # For now, repeat the same latent for each timestep
            # TODO: More sophisticated temporal latent structure
            window_size = teacher.config.window_size
            latents_seq = jnp.tile(latents[:, None, :], (1, window_size, 1))

            # Convert regimes to indices
            regime_idx = jnp.array([REGIME_TO_IDX.get(r, 4) for r in regimes])

            # Training step
            metrics = teacher.train_generator_step(latents_seq, regime_idx)
            epoch_losses.append(metrics["loss"])
            total_steps += 1

            # Update progress bar
            pbar.set_postfix(loss=f"{metrics['loss']:.4f}")

            # Logging
            if total_steps % log_every == 0:
                logger.info(f"Step {total_steps}: loss={metrics['loss']:.4f}")

        # Epoch summary
        epoch_loss = np.mean(epoch_losses)
        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch {epoch + 1}/{epochs} complete: "
            f"loss={epoch_loss:.4f}, "
            f"time={epoch_time:.1f}s"
        )

        # Save checkpoint
        if (epoch + 1) % checkpoint_every == 0:
            ckpt_path = Path(checkpoint_dir) / f"generator_epoch_{epoch + 1}"
            teacher.save_checkpoint(str(ckpt_path))
            logger.info(f"Checkpoint saved: {ckpt_path}")

        # Best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = Path(checkpoint_dir) / "generator_best"
            teacher.save_checkpoint(str(best_path))
            logger.info(f"New best model: loss={best_loss:.4f}")

    logger.info("Phase 2 complete!")


def train_generator_direct(
    teacher: DiffusionTeacher,
    data_loader,  # MarketDataLoader or RainbowTableLoader
    epochs: int,
    batch_size: int,
    log_every: int = 100,
    checkpoint_every: int = 10,
    checkpoint_dir: str = "checkpoints/diff_teacher",
    streaming_mode: bool = False,
    batches_per_period: int = 100,
) -> None:
    """
    Train Flow Matching generator DIRECTLY on prices (no encoder).

    This is the recommended mode for conv architecture training.
    Skips the VAE encoder (which uses O(n²) attention) and trains
    the generator directly on normalized price data.

    Args:
        teacher: DiffusionTeacher instance
        data_loader: Market data loader
        epochs: Number of training epochs
        batch_size: Batch size
        log_every: Log every N steps
        checkpoint_every: Save checkpoint every N epochs
        checkpoint_dir: Checkpoint directory
        streaming_mode: Use streaming mode for RainbowTableLoader
        batches_per_period: Batches per period before switching
    """
    logger.info("=" * 60)
    logger.info("Direct Training: Generator on Raw Prices")
    logger.info("=" * 60)
    logger.info("(Skipping encoder - using normalized prices directly)")

    n_windows = len(data_loader)
    steps_per_epoch = n_windows // batch_size

    if streaming_mode:
        logger.info(f"Data: {n_windows} windows/epoch (streaming), {steps_per_epoch} steps/epoch")
    else:
        logger.info(f"Data: {n_windows} windows, {steps_per_epoch} steps/epoch")

    rng = np.random.default_rng(teacher.config.seed)

    total_steps = 0
    best_loss = float("inf")

    # Regime mapping
    from .conv_flow_matching import REGIME_TO_IDX

    for epoch in range(epochs):
        epoch_losses = []
        epoch_start = time.time()

        pbar = tqdm(
            range(steps_per_epoch),
            desc=f"Epoch {epoch + 1}/{epochs}",
            leave=False,
        )

        batches_since_switch = 0

        for step in pbar:
            # Get batch
            if streaming_mode:
                if batches_since_switch == 0 or batches_since_switch >= batches_per_period:
                    data_loader.switch_period(rng)
                    batches_since_switch = 0
                prices, _, regimes = data_loader.get_batch_streaming(batch_size, rng)
                batches_since_switch += 1
            else:
                prices, _, regimes = data_loader.get_batch_array(batch_size, rng)

            # Normalize prices: (batch, window, assets, features) -> (batch, window, assets*features)
            prices_norm = data_loader.normalize(prices)
            # Flatten features: (batch, window, assets, 5) -> (batch, window, assets*5)
            batch_size_actual, window_size, n_assets, n_features = prices_norm.shape
            prices_flat = prices_norm.reshape(batch_size_actual, window_size, n_assets * n_features)
            prices_jax = jnp.array(prices_flat)

            # Convert regimes to indices
            regime_idx = jnp.array([REGIME_TO_IDX.get(r, 2) for r in regimes])

            # Training step - generator directly on prices
            metrics = teacher.train_generator_step(prices_jax, regime_idx)
            epoch_losses.append(metrics["loss"])
            total_steps += 1

            # Update progress bar
            pbar.set_postfix(loss=f"{metrics['loss']:.4f}")

            # Logging
            if total_steps % log_every == 0:
                logger.info(f"Step {total_steps}: loss={metrics['loss']:.4f}")

        # Epoch summary
        epoch_loss = np.mean(epoch_losses)
        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch {epoch + 1}/{epochs} complete: "
            f"loss={epoch_loss:.4f}, "
            f"time={epoch_time:.1f}s"
        )

        # Save checkpoint
        if (epoch + 1) % checkpoint_every == 0:
            ckpt_path = Path(checkpoint_dir) / f"generator_epoch_{epoch + 1}"
            teacher.save_checkpoint(str(ckpt_path))
            logger.info(f"Checkpoint saved: {ckpt_path}")

        # Best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = Path(checkpoint_dir) / "generator_best"
            teacher.save_checkpoint(str(best_path))
            logger.info(f"New best model: loss={best_loss:.4f}")

    logger.info("Direct training complete!")


def main():
    """Main training entry point."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Diffusion Teacher Training")
    logger.info("=" * 60)

    # Print configuration
    logger.info(f"Database: {args.db_path}")
    logger.info(f"Assets: {args.n_assets}")
    logger.info(f"Window size: {args.window_size}")
    if args.rainbow:
        logger.info(f"Mode: RAINBOW TABLE (96-year coverage)")
        logger.info(f"Schedule: {args.schedule_path}")
        logger.info(f"Windows/epoch: {args.windows_per_epoch:,}")
    else:
        logger.info(f"Mode: Standard (date filter)")
        logger.info(f"Date range: {args.start_date} to {args.end_date}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Phase: {args.phase}")
    logger.info(f"Architecture: {args.architecture.upper()}")
    if args.architecture == "conv":
        logger.info("  Memory: O(n) - handles 1440+ steps in 5GB VRAM")
    else:
        logger.info("  Memory: O(n²) - for shorter sequences only")

    # Create config
    config = create_config(args)

    # Load data
    logger.info("\nLoading data...")
    if args.rainbow:
        # Rainbow table loader - 96 years of rotated asset coverage
        data_loader = RainbowTableLoader(
            db_path=args.db_path,
            schedule_path=args.schedule_path,
            window_size=args.window_size,
            windows_per_epoch=args.windows_per_epoch,
        )
        logger.info(f"Rainbow table: {data_loader.total_windows:,} total windows")
        logger.info(f"Sampling {args.windows_per_epoch:,} windows per epoch")
        # Store streaming params for training functions
        args._streaming_mode = True
        args._batches_per_period = args.batches_per_period
    else:
        # Standard loader - limited date range
        data_loader = MarketDataLoader(
            db_path=args.db_path,
            n_assets=args.n_assets,
            window_size=args.window_size,
            start_date=args.start_date,
            end_date=args.end_date,
            cache_path=args.cache_path,
            min_required_assets=args.n_assets,
        )
        args._streaming_mode = False
    logger.info(f"Loaded {len(data_loader)} windows per epoch")

    # Initialize teacher
    logger.info("\nInitializing teacher...")
    teacher = DiffusionTeacher(config)

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        teacher.load_checkpoint(args.resume)
    else:
        # direct_mode: True for direct training on prices, False for VAE latent training
        direct_mode = (args.phase == "direct")
        teacher.init_training(direct_mode=direct_mode)

    # Training
    try:
        if args.phase in ["encoder", "both"]:
            train_encoder(
                teacher=teacher,
                data_loader=data_loader,
                epochs=args.encoder_epochs,
                batch_size=args.batch_size,
                log_every=args.log_every,
                checkpoint_every=args.checkpoint_every,
                checkpoint_dir=args.checkpoint_dir,
                streaming_mode=args._streaming_mode,
                batches_per_period=getattr(args, '_batches_per_period', 100),
            )

        if args.phase in ["generator", "both"]:
            # For generator-only, need to load encoder first
            if args.phase == "generator" and args.resume is None:
                encoder_path = Path(args.checkpoint_dir) / "encoder_best"
                if encoder_path.exists():
                    logger.info(f"Loading encoder from {encoder_path}")
                    teacher.load_checkpoint(str(encoder_path))
                else:
                    logger.error("No encoder checkpoint found. Train encoder first.")
                    sys.exit(1)

            train_generator(
                teacher=teacher,
                data_loader=data_loader,
                epochs=args.generator_epochs,
                batch_size=args.batch_size,
                log_every=args.log_every,
                checkpoint_every=args.checkpoint_every,
                checkpoint_dir=args.checkpoint_dir,
                streaming_mode=args._streaming_mode,
                batches_per_period=getattr(args, '_batches_per_period', 100),
            )

        if args.phase == "direct":
            # Direct training on prices (no encoder) - recommended for conv
            train_generator_direct(
                teacher=teacher,
                data_loader=data_loader,
                epochs=args.generator_epochs,
                batch_size=args.batch_size,
                log_every=args.log_every,
                checkpoint_every=args.checkpoint_every,
                checkpoint_dir=args.checkpoint_dir,
                streaming_mode=args._streaming_mode,
                batches_per_period=getattr(args, '_batches_per_period', 100),
            )

        # Save final checkpoint
        final_path = Path(args.checkpoint_dir) / "final"
        teacher.save_checkpoint(str(final_path))
        logger.info(f"Final checkpoint saved: {final_path}")

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        interrupt_path = Path(args.checkpoint_dir) / "interrupted"
        teacher.save_checkpoint(str(interrupt_path))
        logger.info(f"Interrupted checkpoint saved: {interrupt_path}")

    finally:
        data_loader.close()

    logger.info("\nTraining complete!")


if __name__ == "__main__":
    main()
