#!/usr/bin/env python3
"""
Memory Profiler - Calculate optimal config for 40GB or 400GB RAM
Run this BEFORE training to determine safe batch sizes and buffer sizes
"""

import torch
import numpy as np
import sys
from dataclasses import dataclass
from typing import Optional

@dataclass
class MemoryConfig:
    """Memory configuration for training"""
    # Hardware
    total_ram_gb: float
    total_vram_gb: float
    num_gpus: int = 1

    # Training params
    collector_env_num: int = 8
    evaluator_env_num: int = 2
    batch_size: int = 1536
    replay_buffer_size: int = 150
    game_segment_length: int = 100
    n_assets: int = 64

    # Model params
    observation_shape: tuple = (27, 64, 64)
    state_dim: int = 64
    action_space_size: int = 192
    value_support_size: int = 601

    def estimate_observation_size_mb(self):
        """Size of one observation in MB"""
        # float32: 4 bytes
        obs_size = np.prod(self.observation_shape) * 4
        # Add action mask, to_play, etc.
        obs_size += self.action_space_size * 4  # action_mask
        obs_size += 4  # to_play
        return obs_size / 1024 / 1024

    def estimate_trajectory_size_mb(self):
        """Size of one trajectory (episode) in MB"""
        obs_mb = self.estimate_observation_size_mb()
        # game_segment_length observations + actions + rewards + values
        traj_size = obs_mb * self.game_segment_length
        traj_size += self.game_segment_length * 4 / 1024 / 1024  # actions (int32)
        traj_size += self.game_segment_length * 4 / 1024 / 1024  # rewards (float32)
        return traj_size

    def estimate_replay_buffer_ram_mb(self):
        """Replay buffer RAM usage in MB"""
        traj_mb = self.estimate_trajectory_size_mb()
        return traj_mb * self.replay_buffer_size

    def estimate_collector_ram_mb(self):
        """RAM per collector process in MB"""
        # Environment state
        env_state_mb = 20  # Encoder buffers, market data cache

        # Episode buffer (stores current episode)
        episode_buffer_mb = self.estimate_trajectory_size_mb()

        # Model copy (in CPU mode)
        model_params_mb = 28  # ~7M params × 4 bytes

        # Overhead
        overhead_mb = 50

        return env_state_mb + episode_buffer_mb + model_params_mb + overhead_mb

    def estimate_total_collector_ram_mb(self):
        """Total RAM for all collectors"""
        return self.estimate_collector_ram_mb() * self.collector_env_num

    def estimate_model_vram_mb(self):
        """Model VRAM usage in MB"""
        # TRM model: ~7M params
        params_mb = 7 * 4 / 1024 / 1024 * 1000  # 7M × 4 bytes
        # Optimizer state (AdamW: 2x params)
        optimizer_mb = params_mb * 2
        return params_mb + optimizer_mb

    def estimate_batch_vram_mb(self):
        """VRAM for one training batch"""
        obs_mb = self.estimate_observation_size_mb()

        # Forward pass
        batch_obs_mb = obs_mb * self.batch_size
        # Latent states (batch × state_dim × float32)
        latent_mb = self.batch_size * self.state_dim * 4 / 1024 / 1024
        # Policy logits
        policy_mb = self.batch_size * self.action_space_size * 4 / 1024 / 1024
        # Value (categorical)
        value_mb = self.batch_size * self.value_support_size * 4 / 1024 / 1024

        # Gradient buffers (approx equal to forward pass)
        forward_mb = batch_obs_mb + latent_mb + policy_mb + value_mb
        backward_mb = forward_mb * 1.5  # Gradients + activations

        return forward_mb + backward_mb

    def estimate_mcts_vram_mb(self):
        """VRAM for MCTS tree search (GPU MCTS)"""
        # Estimate: 50-100 MB per collector when using GPU MCTS
        # This is because tree nodes are stored on GPU
        return self.collector_env_num * 75  # MB per collector

    def estimate_total_vram_mb(self):
        """Total VRAM usage"""
        model_mb = self.estimate_model_vram_mb()
        batch_mb = self.estimate_batch_vram_mb()
        mcts_mb = self.estimate_mcts_vram_mb()
        overhead_mb = 500  # CUDA context, cuDNN, etc.

        return model_mb + batch_mb + mcts_mb + overhead_mb

    def estimate_total_ram_mb(self):
        """Total RAM usage"""
        replay_mb = self.estimate_replay_buffer_ram_mb()
        collector_mb = self.estimate_total_collector_ram_mb()
        evaluator_mb = self.estimate_collector_ram_mb() * self.evaluator_env_num
        main_process_mb = 2000  # Main process overhead
        system_reserved_mb = 2000  # OS + other processes

        return replay_mb + collector_mb + evaluator_mb + main_process_mb + system_reserved_mb

    def print_breakdown(self):
        """Print detailed memory breakdown"""
        print("\n" + "="*80)
        print("MEMORY CONFIGURATION PROFILE")
        print("="*80)

        print(f"\n📋 Configuration:")
        print(f"   RAM: {self.total_ram_gb:.0f} GB | VRAM: {self.total_vram_gb:.0f} GB × {self.num_gpus} GPU(s)")
        print(f"   Collectors: {self.collector_env_num} | Batch: {self.batch_size}")
        print(f"   Replay buffer: {self.replay_buffer_size} episodes × {self.game_segment_length} steps")

        print(f"\n💾 RAM Breakdown:")
        replay_mb = self.estimate_replay_buffer_ram_mb()
        collector_mb = self.estimate_total_collector_ram_mb()
        evaluator_mb = self.estimate_collector_ram_mb() * self.evaluator_env_num

        print(f"   Replay buffer:     {replay_mb:8.1f} MB ({replay_mb/1024:.1f} GB)")
        print(f"   Collectors (×{self.collector_env_num:2d}):  {collector_mb:8.1f} MB ({collector_mb/1024:.1f} GB)")
        print(f"   Evaluators (×{self.evaluator_env_num:2d}):  {evaluator_mb:8.1f} MB ({evaluator_mb/1024:.1f} GB)")
        print(f"   Main process:      {2000:8.1f} MB ({2000/1024:.1f} GB)")
        print(f"   System reserved:   {2000:8.1f} MB ({2000/1024:.1f} GB)")
        total_ram_mb = self.estimate_total_ram_mb()
        print(f"   {'─'*40}")
        print(f"   TOTAL:            {total_ram_mb:8.1f} MB ({total_ram_mb/1024:.1f} GB)")
        print(f"   Available:         {self.total_ram_gb*1024:8.1f} MB ({self.total_ram_gb:.1f} GB)")
        ram_headroom = self.total_ram_gb*1024 - total_ram_mb
        ram_pct = (total_ram_mb / (self.total_ram_gb*1024)) * 100
        print(f"   Headroom:          {ram_headroom:8.1f} MB ({ram_headroom/1024:.1f} GB) | {ram_pct:.1f}% used")

        print(f"\n🎮 VRAM Breakdown (per GPU):")
        model_mb = self.estimate_model_vram_mb()
        batch_mb = self.estimate_batch_vram_mb()
        mcts_mb = self.estimate_mcts_vram_mb()

        print(f"   Model + optimizer: {model_mb:8.1f} MB ({model_mb/1024:.2f} GB)")
        print(f"   Training batch:    {batch_mb:8.1f} MB ({batch_mb/1024:.2f} GB)")
        print(f"   MCTS trees:        {mcts_mb:8.1f} MB ({mcts_mb/1024:.2f} GB)")
        print(f"   CUDA overhead:     {500:8.1f} MB ({500/1024:.2f} GB)")
        total_vram_mb = self.estimate_total_vram_mb()
        print(f"   {'─'*40}")
        print(f"   TOTAL:            {total_vram_mb:8.1f} MB ({total_vram_mb/1024:.1f} GB)")
        print(f"   Available:         {self.total_vram_gb*1024:8.1f} MB ({self.total_vram_gb:.1f} GB)")
        vram_headroom = self.total_vram_gb*1024 - total_vram_mb
        vram_pct = (total_vram_mb / (self.total_vram_gb*1024)) * 100
        print(f"   Headroom:          {vram_headroom:8.1f} MB ({vram_headroom/1024:.1f} GB) | {vram_pct:.1f}% used")

        # Safety check
        print(f"\n✅ Safety Check:")
        safe = True

        if total_ram_mb > self.total_ram_gb * 1024 * 0.95:
            print(f"   ❌ RAM: UNSAFE - Using {ram_pct:.1f}% (>95%)")
            safe = False
        elif total_ram_mb > self.total_ram_gb * 1024 * 0.85:
            print(f"   ⚠️  RAM: MARGINAL - Using {ram_pct:.1f}% (>85%)")
        else:
            print(f"   ✓ RAM: SAFE - Using {ram_pct:.1f}%")

        if total_vram_mb > self.total_vram_gb * 1024 * 0.95:
            print(f"   ❌ VRAM: UNSAFE - Using {vram_pct:.1f}% (>95%)")
            safe = False
        elif total_vram_mb > self.total_vram_gb * 1024 * 0.85:
            print(f"   ⚠️  VRAM: MARGINAL - Using {vram_pct:.1f}% (>85%)")
        else:
            print(f"   ✓ VRAM: SAFE - Using {vram_pct:.1f}%")

        if not safe:
            print(f"\n⚠️  RECOMMENDATIONS:")
            if total_ram_mb > self.total_ram_gb * 1024 * 0.95:
                print(f"   - Reduce replay_buffer_size to {int(self.replay_buffer_size * 0.7)}")
                print(f"   - OR reduce collector_env_num to {int(self.collector_env_num * 0.7)}")

            if total_vram_mb > self.total_vram_gb * 1024 * 0.95:
                print(f"   - Reduce batch_size to {int(self.batch_size * 0.7)}")
                print(f"   - OR use gradient accumulation instead")

        print("="*80)

        return safe


def optimize_for_hardware(ram_gb, vram_gb, num_gpus=1):
    """Auto-optimize config for given hardware"""
    print(f"\n🔧 Auto-optimizing for: {ram_gb}GB RAM, {vram_gb}GB VRAM × {num_gpus} GPU(s)")

    # Start with conservative config
    config = MemoryConfig(
        total_ram_gb=ram_gb,
        total_vram_gb=vram_gb,
        num_gpus=num_gpus,
        collector_env_num=4,
        batch_size=512,
        replay_buffer_size=100
    )

    # Binary search for max batch size
    min_batch = 256
    max_batch = 8192
    best_batch = min_batch

    while min_batch <= max_batch:
        mid_batch = (min_batch + max_batch) // 2
        config.batch_size = mid_batch

        vram_used = config.estimate_total_vram_mb()
        vram_avail = config.total_vram_gb * 1024

        if vram_used < vram_avail * 0.80:  # Use 80% of VRAM
            best_batch = mid_batch
            min_batch = mid_batch + 256
        else:
            max_batch = mid_batch - 256

    config.batch_size = best_batch

    # Binary search for max collectors
    min_collectors = 4
    max_collectors = 32
    best_collectors = min_collectors

    while min_collectors <= max_collectors:
        mid_collectors = (min_collectors + max_collectors) // 2
        config.collector_env_num = mid_collectors

        ram_used = config.estimate_total_ram_mb()
        ram_avail = config.total_ram_gb * 1024

        if ram_used < ram_avail * 0.80:  # Use 80% of RAM
            best_collectors = mid_collectors
            min_collectors = mid_collectors + 2
        else:
            max_collectors = mid_collectors - 2

    config.collector_env_num = best_collectors

    # Optimize replay buffer
    ram_used = config.estimate_total_ram_mb()
    ram_avail = config.total_ram_gb * 1024
    ram_headroom = ram_avail * 0.80 - ram_used

    # Use remaining RAM for replay buffer
    traj_mb = config.estimate_trajectory_size_mb()
    additional_episodes = int(ram_headroom / traj_mb)
    config.replay_buffer_size = config.replay_buffer_size + additional_episodes

    # Multi-GPU: scale batch size
    if num_gpus > 1:
        config.batch_size = config.batch_size * num_gpus

    return config


if __name__ == "__main__":
    print("Memory Profiler for TRM+MCTS Training")

    # Parse arguments
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python profile_memory.py <ram_gb> [vram_gb] [num_gpus]")
        print("\nExamples:")
        print("  python profile_memory.py 40          # 40GB RAM, auto-detect VRAM")
        print("  python profile_memory.py 40 24       # 40GB RAM, 24GB VRAM")
        print("  python profile_memory.py 400 24 4    # 400GB RAM, 24GB VRAM × 4 GPUs")
        sys.exit(1)

    ram_gb = float(sys.argv[1])

    # Auto-detect VRAM if not provided
    if len(sys.argv) > 2:
        vram_gb = float(sys.argv[2])
    else:
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✓ Auto-detected VRAM: {vram_gb:.1f} GB")
        else:
            vram_gb = 24  # Default
            print(f"⚠️  No GPU detected, using default: {vram_gb} GB")

    # Number of GPUs
    num_gpus = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    # Auto-optimize
    config = optimize_for_hardware(ram_gb, vram_gb, num_gpus)

    # Print breakdown
    safe = config.print_breakdown()

    # Print config export
    print(f"\n📝 Recommended Config (trading_config.py):")
    print(f"""
collector_env_num={config.collector_env_num},
batch_size={config.batch_size},
replay_buffer_size={config.replay_buffer_size},
game_segment_length={config.game_segment_length},
""")

    if not safe:
        print("⚠️  WARNING: Configuration may cause OOM errors!")
        sys.exit(1)
