#!/usr/bin/env python3
# Add project root to path for imports
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))
"""
Level 1 Sweep: Compare best L0 configs on harder synthetic level.

Configs:
1. kitchen_sink_v2 - Best L0 composite score
2. eps030_dirichlet - Best entropy stability + max reward
3. hybrid_v1 - eps030 entropy + kitchen_sink pred weight
4. hybrid_v2 - kitchen_sink base + dirichlet exploration

Timeout: 6 hours per config
"""

import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

# Base settings (shared across all configs)
BASE_CMD = [
    "python", str(PROJECT_ROOT / "train_jax_fast.py"),
    "--synthetic-level", "1",  # LEVEL 1!
    "--synthetic-steps", "10000",
    "--epochs", "500",
    "--batch-size", "384",
    "--model-size", "small",
    "--parallel-envs", "50",
    "--jax-env",
    "--bf16",
    "--popart-beta", "0.0001",
    "--use-prediction-loss",
    "--adaptive-entropy",
    "--ds24-level", "3",
    "--ds24-intermediates-every", "10",
]

# L0 checkpoint - solid base for all L1 configs
L0_CHECKPOINT = str(PROJECT_ROOT / "jax_ckpt_fast" / "eps030_dirichlet_longwarmup"

# Config definitions
CONFIGS = {
    "kitchen_sink_v2": [
        "--load-checkpoint", L0_CHECKPOINT,
        "--clear-buffer", "--reset-optimizer",
        "--sims", "24", "--sims-warmup", "300", "--sims-min", "4",
        "--entropy-coef", "0.2",
        "--entropy-hold-threshold", "0.55",
        "--entropy-boost-low", "0.4", "--entropy-boost-high", "0.6",
        "--prediction-loss-weight", "0.5",
        "--physics-warmup-steps", "10000",
    ],
    "eps030_dirichlet": [
        "--load-checkpoint", L0_CHECKPOINT,
        "--clear-buffer", "--reset-optimizer",
        "--sims", "24", "--sims-warmup", "300", "--sims-min", "4",
        "--epsilon-greedy", "0.3",
        "--dirichlet-aggressive",
        "--entropy-hold-threshold", "0.55",
        "--entropy-boost-low", "0.4", "--entropy-boost-high", "0.6",
        "--prediction-loss-weight", "0.2",
        "--physics-warmup-steps", "20000",
    ],
    "hybrid_v1_eps_entropy_ks_pred": [
        # eps030's entropy stability + kitchen_sink's pred weight
        "--load-checkpoint", L0_CHECKPOINT,
        "--clear-buffer", "--reset-optimizer",
        "--sims", "24", "--sims-warmup", "300", "--sims-min", "4",
        "--epsilon-greedy", "0.3",
        "--dirichlet-aggressive",
        "--entropy-coef", "0.2",  # From kitchen_sink
        "--entropy-hold-threshold", "0.55",
        "--entropy-boost-low", "0.4", "--entropy-boost-high", "0.6",
        "--prediction-loss-weight", "0.5",  # From kitchen_sink (higher)
        "--physics-warmup-steps", "15000",
    ],
    "hybrid_v2_ks_base_dirichlet_explore": [
        # kitchen_sink base + dirichlet exploration
        "--load-checkpoint", L0_CHECKPOINT,
        "--clear-buffer", "--reset-optimizer",
        "--sims", "24", "--sims-warmup", "300", "--sims-min", "4",
        "--epsilon-greedy", "0.2",  # Slightly less than eps030
        "--dirichlet-aggressive",
        "--entropy-coef", "0.2",
        "--entropy-hold-threshold", "0.55",
        "--entropy-boost-low", "0.4", "--entropy-boost-high", "0.6",
        "--prediction-loss-weight", "0.5",
        "--physics-warmup-steps", "10000",
    ],
}

TIMEOUT_SECONDS = 6 * 60 * 60  # 6 hours


def run_config(name: str, config_args: list, sweep_dir: Path):
    """Run a single configuration."""

    full_name = f"L1_{name}"
    cmd = BASE_CMD + config_args + ["--name", full_name]

    print(f"\n{'='*80}")
    print(f"Running: {name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Timeout: {TIMEOUT_SECONDS // 3600}h")
    print(f"{'='*80}\n")

    log_file = sweep_dir / f"{full_name}.log"

    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(
                cmd,
                timeout=TIMEOUT_SECONDS,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=os.getcwd(),
            )

        status = "COMPLETE" if result.returncode == 0 else f"FAILED (code {result.returncode})"

    except subprocess.TimeoutExpired:
        status = "TIMEOUT"
    except Exception as e:
        status = f"ERROR: {e}"

    print(f"\n>>> {name}: {status}")

    # Write status file
    with open(sweep_dir / f"{full_name}.status", 'w') as f:
        f.write(status)

    return status


def main():
    # Create sweep directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(f"logs/level1_sweep_{timestamp}")
    sweep_dir.mkdir(parents=True, exist_ok=True)

    print(f"Level 1 Sweep")
    print(f"Output: {sweep_dir}")
    print(f"Configs: {list(CONFIGS.keys())}")
    print(f"Timeout: {TIMEOUT_SECONDS // 3600}h per config")
    print(f"Total max time: {len(CONFIGS) * TIMEOUT_SECONDS // 3600}h")

    # Save config info
    with open(sweep_dir / "sweep_config.txt", 'w') as f:
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Level: 1\n")
        f.write(f"Timeout: {TIMEOUT_SECONDS}s\n\n")
        for name, args in CONFIGS.items():
            f.write(f"{name}:\n  {' '.join(args)}\n\n")

    results = {}

    for name, config_args in CONFIGS.items():
        status = run_config(name, config_args, sweep_dir)
        results[name] = status

    # Summary
    print(f"\n{'='*80}")
    print("SWEEP COMPLETE")
    print(f"{'='*80}")
    for name, status in results.items():
        print(f"  {name}: {status}")

    print(f"\nResults in: {sweep_dir}")
    print(f"Run analyze_sweep.py {sweep_dir} to compare")


if __name__ == "__main__":
    main()
