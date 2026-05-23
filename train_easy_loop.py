"""
train_easy_loop.py — Per-epoch training/validation drivers + benchmark plumbing.

Extracted from train_easy.py (TASK_772.2, 2026-05-23). Holds:
- `train_epoch(trainer, generator, config, epoch)` — one epoch of on-the-fly
  generation + train steps.
- `validate(trainer, generator, config, n_batches)` — evaluation loop on
  freshly-generated batches.
- `_launch_benchmark(ckpt_path, gpu_id, db_path)` — non-blocking subprocess
  launcher for `tools/benchmark_easy.py` on a separate GPU.
- `_check_benchmark_passed(output_dir)` — scan the most-recent benchmark JSON
  in a checkpoint dir, return whether its `summary.passed` flag is set.

Type hints reference `EasyTrainer` / `OnTheFlyGenerator` / `EasyConfig` via
`TYPE_CHECKING` to avoid the circular import (train_easy.py defines
OnTheFlyGenerator and imports from this module).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from easy_config import EasyConfig
    from easy_trainer import EasyTrainer
    from train_easy import OnTheFlyGenerator

logger = logging.getLogger(__name__)


def train_epoch(
    trainer: "EasyTrainer",
    generator: "OnTheFlyGenerator",
    config: "EasyConfig",
    epoch: int,
) -> dict:
    """Train for one epoch with on-the-fly generation."""
    # TASK_691: Update generator's epoch for oracle entropy schedule
    generator._current_epoch = epoch
    # TASK_707: Cycle to next historical period each epoch (legacy, pre-TASK_710)
    # TASK_710: SegmentPool handles rotation internally — skip advance_period
    if generator.segment_pool is None and (generator._period_specs or generator._historical_markets):
        generator.advance_period()

    # TASK_692: Update generator's model params for autoregressive state rollout
    # Call at start of each epoch to refresh partial buffer with new params
    autoregressive_state = getattr(config, 'autoregressive_state', False)
    ar_warmup = getattr(config, 'ar_warmup_epochs', 0)
    ar_blend_epochs = getattr(config, 'ar_blend_epochs', 0)
    if autoregressive_state and epoch >= ar_warmup:
        generator.set_model_params(trainer.model, trainer.params)
        if epoch == ar_warmup:
            if ar_blend_epochs > 0:
                logger.info(f"  ► AR warmup complete — beginning {ar_blend_epochs}-epoch blend (oracle→AR)")
            else:
                logger.info(f"  ► AR warmup complete — switching to model-state rollout with DAgger targets")
        if ar_blend_epochs > 0 and epoch == ar_warmup + ar_blend_epochs:
            ar_max_blend = getattr(config, 'ar_max_blend', 1.0)
            if ar_max_blend < 1.0:
                logger.info(f"  ► AR blend ramp complete — capped at {ar_max_blend:.0%} AR (oracle {1-ar_max_blend:.0%})")
            else:
                logger.info(f"  ► AR blend complete — fully model-state rollout from here")

    epoch_metrics = {}  # Dynamic initialization - immune to trainer metric changes

    steps_per_epoch = config.steps_per_epoch if hasattr(config, 'steps_per_epoch') else 200

    # TASK_151: Calculate warmup progress for trade weight ramping
    warmup_epochs = getattr(config, 'weight_warmup_epochs', 50)
    warmup_progress = min(epoch / warmup_epochs, 1.0) if warmup_epochs > 0 else 1.0

    for step in range(steps_per_epoch):
        # TASK_715: Direct sampling when pool is active (bypasses buffer overhead)
        if generator.segment_pool is not None:
            batch = generator.generate_batch_direct(config.batch_size)
        else:
            batch = generator.generate_batch(config.batch_size)

        # TASK_151: Add warmup progress for trade weight ramping
        batch['warmup_progress'] = jnp.array(warmup_progress)

        # Train step (batch already contains JAX arrays - zero copy)
        metrics = trainer.train_step(batch)

        # Dynamic initialization on first step
        if not epoch_metrics:
            epoch_metrics = {k: [] for k in metrics.keys()}

        for key, value in metrics.items():
            if key in epoch_metrics:
                epoch_metrics[key].append(value)

        # Progress logging every 50 steps
        if (step + 1) % 50 == 0:
            avg_loss = np.mean(epoch_metrics.get('total_loss', [0])[-50:])
            avg_agree = np.mean(epoch_metrics.get('oracle_agreement', [0])[-50:])
            logger.info(f"  Epoch {epoch} Step {step+1}/{steps_per_epoch} | Loss: {avg_loss:.4f} | Agree: {avg_agree:.2%}")

    # Safe averaging (handles empty lists)
    return {k: np.mean(v) if v else 0.0 for k, v in epoch_metrics.items()}


def validate(
    trainer: "EasyTrainer",
    generator: "OnTheFlyGenerator",
    config: "EasyConfig",
    n_batches: int = 20,
) -> dict:
    """Validate with fresh generated data (uses trainer.evaluate_batch)."""
    val_metrics = {}  # Dynamic initialization

    for _ in range(n_batches):
        if generator.segment_pool is not None:
            batch = generator.generate_batch_direct(config.batch_size)
        else:
            batch = generator.generate_batch(config.batch_size)
        metrics = trainer.evaluate_batch(batch)  # Zero-copy, reuses trainer logic

        if not val_metrics:
            val_metrics = {k: [] for k in metrics.keys()}

        for k, v in metrics.items():
            if k in val_metrics:
                val_metrics[k].append(v)

    # Safe averaging
    return {k: np.mean(v) if v else 0.0 for k, v in val_metrics.items()}


# =============================================================================
# TASK_706: Periodic Benchmark (runs on separate GPU, non-blocking)
# =============================================================================

_benchmark_proc = None  # Track running benchmark subprocess


def _launch_benchmark(ckpt_path: str, gpu_id: int, db_path: str = "trading_data.db"):
    """Launch benchmark on separate GPU as non-blocking subprocess."""
    global _benchmark_proc
    logger = logging.getLogger(__name__)

    # Don't launch if previous benchmark still running
    if _benchmark_proc is not None and _benchmark_proc.poll() is None:
        logger.info(f"[BENCH] Previous benchmark still running (pid={_benchmark_proc.pid}), skipping")
        return

    results_path = Path(ckpt_path).parent / f"benchmark_{Path(ckpt_path).stem}.json"
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    cmd = [
        sys.executable, "tools/benchmark_easy.py",
        "--checkpoint", str(ckpt_path),
        "--db", db_path,
        "--json", str(results_path),
        "--quiet",
    ]

    stderr_path = results_path.with_suffix('.stderr')
    logger.info(f"[BENCH] Launching benchmark on GPU {gpu_id}: {Path(ckpt_path).name} → {results_path.name}")
    _benchmark_proc = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.DEVNULL, stderr=open(stderr_path, 'w'),
    )


def _check_benchmark_passed(output_dir: Path) -> bool:
    """Check if any benchmark result in output_dir has passed."""
    import glob
    logger = logging.getLogger(__name__)

    for json_path in sorted(glob.glob(str(output_dir / "benchmark_*.json")), reverse=True):
        try:
            with open(json_path) as f:
                data = json.load(f)
            summary = data.get('summary', {})
            passed = summary.get('passed', False)
            if passed:
                logger.info(f"[BENCH] ★ PASSED: {json_path}")
                return True
            # Log latest result summary
            logger.info(
                f"[BENCH] {Path(json_path).stem}: "
                f"alpha+={summary.get('positive_alpha_count', 0)}/6, "
                f"beats_rnd={summary.get('beats_random_count', 0)}/6, "
                f"bal_acc={summary.get('avg_balanced_accuracy', 0):.1%}, "
                f"vmse={summary.get('avg_value_mse', 0):.4f}"
            )
            return False  # Only check latest
        except (json.JSONDecodeError, OSError):
            continue
    return False
