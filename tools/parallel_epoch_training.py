#!/usr/bin/env python3
"""
Parallel Epoch Training with Parameter Averaging

Runs 4 training instances in parallel for N epochs, then averages their
parameters before starting the next batch. This provides:
- 4× speedup from parallelization
- Knowledge continuity (parameter averaging)
- Exploration diversity (different trajectories)
"""

import subprocess
import json
import os
import time
import argparse
from pathlib import Path
import jax
import jax.numpy as jnp
from flax.training import checkpoints
from flax.traverse_util import flatten_dict, unflatten_dict


def average_checkpoints(checkpoint_dirs, output_dir, step):
    """
    Average parameters from multiple Flax checkpoint directories.

    Args:
        checkpoint_dirs: List of checkpoint directory paths
        output_dir: Where to save averaged checkpoint
        step: Step number for the checkpoint
    """
    print(f"\nAveraging {len(checkpoint_dirs)} checkpoints at step {step}...")

    # Load all checkpoints
    states = []
    for ckpt_dir in checkpoint_dirs:
        if not os.path.exists(ckpt_dir):
            print(f"  ⚠️ Checkpoint dir not found: {ckpt_dir}")
            continue

        print(f"  Loading: {ckpt_dir}")
        try:
            # Use Flax's restore_checkpoint to load
            state = checkpoints.restore_checkpoint(ckpt_dir, target=None, step=step)
            if state is None:
                print(f"  ⚠️ Could not restore checkpoint from {ckpt_dir}")
                continue
            states.append(state)
        except Exception as e:
            print(f"  ⚠️ Error loading {ckpt_dir}: {e}")
            continue

    if len(states) == 0:
        raise ValueError("No valid checkpoints found!")

    if len(states) == 1:
        print("  Only 1 checkpoint found, using as-is")
        checkpoints.save_checkpoint(
            ckpt_dir=output_dir,
            target=states[0],
            step=step,
            keep=3,
            overwrite=True
        )
        return

    print(f"  Averaging {len(states)} checkpoints...")

    # Average only the params, not optimizer state
    # Extract params from each state
    params_list = [state.params for state in states]

    # Flatten all params for easier averaging
    flat_params_list = [flatten_dict(params, sep='/') for params in params_list]

    # Average each parameter
    averaged_params = {}
    for key in flat_params_list[0].keys():
        # Stack parameters from all checkpoints
        params = jnp.stack([p[key] for p in flat_params_list])
        # Average
        averaged_params[key] = jnp.mean(params, axis=0)

    # Unflatten back to nested structure
    averaged_params = unflatten_dict(averaged_params, sep='/')

    # Create new state with averaged params (use first state as template)
    averaged_state = states[0].replace(params=averaged_params)

    # Save averaged checkpoint
    print(f"  Saving averaged checkpoint to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    checkpoints.save_checkpoint(
        ckpt_dir=output_dir,
        target=averaged_state,
        step=step,
        keep=3,
        overwrite=True
    )

    print("  ✅ Checkpoint averaging complete")


def create_instance_config(instance_id, base_config, checkpoint_dir, load_checkpoint=None):
    """Create config file for a training instance"""
    config = base_config.copy()
    config['random_seed'] = config.get('random_seed', 42) + instance_id
    config['checkpoint_dir'] = f"{checkpoint_dir}/instance_{instance_id}"

    if load_checkpoint:
        config['load_checkpoint'] = load_checkpoint

    config_path = f"config/parallel_instance_{instance_id}.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    return config_path


def launch_parallel_training(
    n_instances: int,
    epochs_per_batch: int,
    total_epochs: int,
    base_config: dict,
    checkpoint_dir: str = "jax_ckpt_fast/parallel"
):
    """
    Launch parallel training with parameter averaging.

    Args:
        n_instances: Number of parallel instances (e.g., 4)
        epochs_per_batch: Epochs per batch before averaging (e.g., 50)
        total_epochs: Total training epochs (e.g., 5000)
        base_config: Base training configuration
        checkpoint_dir: Root directory for checkpoints
    """

    print("="*60)
    print("PARALLEL EPOCH TRAINING WITH PARAMETER AVERAGING")
    print("="*60)
    print(f"Instances: {n_instances}")
    print(f"Epochs per batch: {epochs_per_batch}")
    print(f"Total epochs: {total_epochs}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print("="*60)

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Track merged checkpoint directory
    merged_checkpoint_dir = None

    # Calculate number of batches
    n_batches = (total_epochs + epochs_per_batch - 1) // epochs_per_batch

    for batch_idx in range(n_batches):
        epoch_start = batch_idx * epochs_per_batch
        epoch_end = min(epoch_start + epochs_per_batch, total_epochs)
        epochs_this_batch = epoch_end - epoch_start

        print(f"\n{'='*60}")
        print(f"BATCH {batch_idx + 1}/{n_batches}: Epochs {epoch_start}-{epoch_end}")
        print(f"{'='*60}")

        # Create config files for each instance
        config_paths = []
        for i in range(n_instances):
            config_path = create_instance_config(
                instance_id=i,
                base_config=base_config,
                checkpoint_dir=checkpoint_dir,
                load_checkpoint=merged_checkpoint_dir
            )
            config_paths.append(config_path)
            print(f"  Instance {i}: {config_path}")

        # Launch parallel training processes
        processes = []
        start_time = time.time()

        print(f"\nLaunching {n_instances} training instances...")
        for i, config_path in enumerate(config_paths):
            cmd = [
                'python', 'train_jax_fast.py',
                '--config', config_path,
                '--epochs', str(epochs_this_batch),
            ]

            # Stagger launches to avoid VRAM allocation spike
            if i > 0:
                time.sleep(5)

            print(f"  Starting instance {i}...")
            proc = subprocess.Popen(
                cmd,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0'},
                stdout=open(f"logs/parallel_instance_{i}_batch_{batch_idx}.log", 'w'),
                stderr=subprocess.STDOUT
            )
            processes.append((i, proc))

        print(f"\n✅ All {n_instances} instances launched")
        print("Monitor with:")
        print("  nvidia-smi -l 1    # GPU usage")
        print("  htop               # CPU usage")
        print(f"  tail -f logs/parallel_instance_*.log")

        # Wait for all instances to complete
        print(f"\nWaiting for all instances to complete epochs {epoch_start}-{epoch_end}...")
        for i, proc in processes:
            proc.wait()
            if proc.returncode == 0:
                print(f"  ✅ Instance {i} completed successfully")
            else:
                print(f"  ❌ Instance {i} failed with code {proc.returncode}")

        elapsed = time.time() - start_time
        print(f"\n⏱️  Batch completed in {elapsed:.1f}s ({epochs_this_batch * n_instances / elapsed:.1f} epoch-instances/s)")

        # Average checkpoints from all instances
        print(f"\nAveraging checkpoints from batch {batch_idx + 1}...")
        checkpoint_dirs = [
            f"{checkpoint_dir}/instance_{i}"
            for i in range(n_instances)
        ]

        merged_checkpoint_dir = f"{checkpoint_dir}/merged"
        average_checkpoints(checkpoint_dirs, merged_checkpoint_dir, step=epoch_end)

        # Also track which individual instance performed best
        # (User can choose to use best individual instead of averaged)
        print(f"\n📊 Checkpoint options for epoch {epoch_end}:")
        print(f"   Averaged:  {merged_checkpoint_dir}")
        print(f"   Individual instances: {checkpoint_dir}/instance_[0-{n_instances-1}]")
        print(f"   💡 Tip: Compare both in evaluation to see which works better!")

        print(f"\n✅ Batch {batch_idx + 1} complete")

    print(f"\n{'='*60}")
    print("PARALLEL TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Final checkpoint: {merged_checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Parallel epoch training with parameter averaging"
    )
    parser.add_argument('--instances', type=int, default=4,
                       help='Number of parallel instances')
    parser.add_argument('--epochs-per-batch', type=int, default=50,
                       help='Epochs per batch before averaging')
    parser.add_argument('--total-epochs', type=int, default=5000,
                       help='Total training epochs')
    parser.add_argument('--config', type=str, default=None,
                       help='Base config file (optional)')
    parser.add_argument('--checkpoint-dir', type=str, default='jax_ckpt_fast/parallel',
                       help='Root checkpoint directory')

    args = parser.parse_args()

    # Load base config or use defaults
    if args.config:
        with open(args.config, 'r') as f:
            base_config = json.load(f)
    else:
        base_config = {
            'n_assets': 64,
            'min_coverage': 0.75,
            'random_seed': 42,
        }

    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)

    # Launch parallel training
    launch_parallel_training(
        n_instances=args.instances,
        epochs_per_batch=args.epochs_per_batch,
        total_epochs=args.total_epochs,
        base_config=base_config,
        checkpoint_dir=args.checkpoint_dir
    )


if __name__ == "__main__":
    main()
