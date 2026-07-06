#!/usr/bin/env python3
# Add project root to path for imports
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))
"""
TASK_036: Warmup Configuration Sweep Script

Systematically tests different warmup configurations to find stable training equilibrium.
Designed to run overnight and identify configurations that prevent dual collapse.

Usage:
    python warmup_sweep.py --epochs 50 --output-dir sweep_results
    python warmup_sweep.py --epochs 30 --quick  # Fast subset
"""

import subprocess
import json
import re
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import time


@dataclass
class SweepConfig:
    """Configuration for a single sweep run."""
    name: str
    sims_warmup: int = 100
    sims_min: int = 8
    sims_max: int = 32
    entropy_coef: float = 0.01
    entropy_hold_threshold: float = 0.85  # User's production default
    entropy_boost_low: float = 0.05       # User's production default
    entropy_boost_high: float = 0.30      # User's production default
    prediction_loss_weight: float = 0.2
    physics_warmup_steps: int = 5000
    # Future: epsilon_greedy when TASK_037 is implemented
    epsilon_greedy: float = 0.0


@dataclass
class RunMetrics:
    """Metrics collected from a training run."""
    config_name: str
    config: dict

    # Stability metrics
    epochs_until_hold_70: Optional[int] = None  # First epoch where HOLD% > 70%
    epochs_until_hold_80: Optional[int] = None  # First epoch where HOLD% > 80%
    epochs_until_entropy_35: Optional[int] = None  # First epoch where entropy < 3.5
    epochs_until_entropy_30: Optional[int] = None  # First epoch where entropy < 3.0

    # Performance metrics
    max_reward: float = 0.0
    final_reward: float = 0.0
    mean_reward: float = 0.0

    # Final state
    final_entropy: float = 0.0
    final_hold_pct: float = 0.0
    final_pred_loss: float = 0.0
    final_step1_mse: float = 0.0

    # Computed scores
    stability_score: float = 0.0

    # Run info
    total_epochs: int = 0
    runtime_seconds: float = 0.0
    status: str = "pending"
    error: str = ""


def get_sweep_configs(quick: bool = False) -> List[SweepConfig]:
    """Generate configurations to test."""

    configs = []

    # =========================================================================
    # BASELINE (user's production config)
    # =========================================================================
    configs.append(SweepConfig(
        name="baseline",
        sims_warmup=100, sims_min=8, sims_max=32,
        entropy_coef=0.01,
        entropy_hold_threshold=0.85,
        entropy_boost_low=0.05,
        entropy_boost_high=0.30,
    ))

    # =========================================================================
    # EXTENDED LOW-SIMS PHASE (delay MCTS sharpening)
    # =========================================================================
    configs.append(SweepConfig(
        name="extended_lowsims_200",
        sims_warmup=200, sims_min=4, sims_max=32,
        entropy_coef=0.05,
    ))

    configs.append(SweepConfig(
        name="extended_lowsims_300",
        sims_warmup=300, sims_min=4, sims_max=24,
        entropy_coef=0.05,
    ))

    if not quick:
        configs.append(SweepConfig(
            name="extended_lowsims_500",
            sims_warmup=500, sims_min=4, sims_max=16,
            entropy_coef=0.05,
        ))

    # =========================================================================
    # HIGH ENTROPY COEFFICIENT
    # =========================================================================
    configs.append(SweepConfig(
        name="high_entropy_01",
        sims_warmup=100, sims_min=8, sims_max=32,
        entropy_coef=0.1,
    ))

    configs.append(SweepConfig(
        name="high_entropy_02",
        sims_warmup=100, sims_min=8, sims_max=32,
        entropy_coef=0.2,
    ))

    if not quick:
        configs.append(SweepConfig(
            name="very_high_entropy_05",
            sims_warmup=100, sims_min=8, sims_max=32,
            entropy_coef=0.5,
        ))

    # =========================================================================
    # EARLIER ADAPTIVE ENTROPY TRIGGER
    # =========================================================================
    configs.append(SweepConfig(
        name="early_adaptive_070",
        sims_warmup=100, sims_min=8, sims_max=32,
        entropy_coef=0.01,
        entropy_hold_threshold=0.70,  # Trigger earlier than 0.85
        entropy_boost_low=0.1,
        entropy_boost_high=0.4,
    ))

    configs.append(SweepConfig(
        name="early_adaptive_060",
        sims_warmup=100, sims_min=8, sims_max=32,
        entropy_coef=0.05,
        entropy_hold_threshold=0.60,  # Very early trigger
        entropy_boost_low=0.2,
        entropy_boost_high=0.5,
    ))

    # =========================================================================
    # COMBINED: EXTENDED WARMUP + HIGH ENTROPY
    # =========================================================================
    configs.append(SweepConfig(
        name="combined_v1",
        sims_warmup=200, sims_min=4, sims_max=32,
        entropy_coef=0.1,
        entropy_hold_threshold=0.65,
        entropy_boost_low=0.25,
        entropy_boost_high=0.4,
    ))

    configs.append(SweepConfig(
        name="combined_v2",
        sims_warmup=300, sims_min=4, sims_max=24,
        entropy_coef=0.15,
        entropy_hold_threshold=0.60,
        entropy_boost_low=0.3,
        entropy_boost_high=0.5,
    ))

    if not quick:
        configs.append(SweepConfig(
            name="combined_v3_aggressive",
            sims_warmup=400, sims_min=4, sims_max=16,
            entropy_coef=0.2,
            entropy_hold_threshold=0.55,
            entropy_boost_low=0.4,
            entropy_boost_high=0.6,
        ))

    # =========================================================================
    # PREDICTION LOSS WEIGHT VARIATIONS
    # =========================================================================
    configs.append(SweepConfig(
        name="high_pred_weight",
        sims_warmup=100, sims_min=8, sims_max=32,
        entropy_coef=0.05,
        prediction_loss_weight=0.5,
    ))

    if not quick:
        configs.append(SweepConfig(
            name="very_high_pred_weight",
            sims_warmup=100, sims_min=8, sims_max=32,
            entropy_coef=0.05,
            prediction_loss_weight=1.0,
        ))

    # =========================================================================
    # PHYSICS WARMUP VARIATIONS
    # =========================================================================
    configs.append(SweepConfig(
        name="extended_physics_warmup",
        sims_warmup=100, sims_min=8, sims_max=32,
        entropy_coef=0.05,
        physics_warmup_steps=10000,
    ))

    if not quick:
        configs.append(SweepConfig(
            name="no_physics_warmup",
            sims_warmup=100, sims_min=8, sims_max=32,
            entropy_coef=0.05,
            physics_warmup_steps=0,
        ))

    # =========================================================================
    # MOST AGGRESSIVE (kitchen sink)
    # =========================================================================
    configs.append(SweepConfig(
        name="kitchen_sink",
        sims_warmup=300, sims_min=4, sims_max=24,
        entropy_coef=0.2,
        entropy_hold_threshold=0.55,
        entropy_boost_low=0.4,
        entropy_boost_high=0.6,
        prediction_loss_weight=0.5,
        physics_warmup_steps=10000,
    ))

    return configs


def build_command(config: SweepConfig, epochs: int, run_name: str) -> List[str]:
    """Build the training command for a config.

    Base config matches user's production training setup:
    python train_jax_fast.py --synthetic-level 0 --synthetic-steps 10000 --epochs 500
    --batch-size 384 --model-size small --sims 32 --steps 400 --parallel-envs 50
    --jax-env --popart-beta 0.0001 --sims-warmup 100 --sims-min 8
    --use-prediction-loss --prediction-loss-weight 0.2 --pred-head-size large
    --ds24-level 3 --ds24-intermediates-every 5 --regenerate-data-every 50 --bf16
    --adaptive-entropy --entropy-hold-threshold 0.85 --fee 0 --slippage 0
    """
    cmd = [
        "python", str(PROJECT_ROOT / "train_jax_fast.py"),
        # Data config
        "--synthetic-level", "0",
        "--synthetic-steps", "10000",
        "--regenerate-data-every", "50",
        # Training config
        "--epochs", str(epochs),
        "--steps", "400",
        "--parallel-envs", "50",
        "--batch-size", "384",
        "--model-size", "small",
        "--jax-env",
        "--bf16",
        # MCTS config (from sweep)
        "--sims-warmup", str(config.sims_warmup),
        "--sims-min", str(config.sims_min),
        "--sims", str(config.sims_max),
        # Entropy config (from sweep)
        "--entropy-coef", str(config.entropy_coef),
        "--entropy-hold-threshold", str(config.entropy_hold_threshold),
        "--entropy-boost-low", str(config.entropy_boost_low),
        "--entropy-boost-high", str(config.entropy_boost_high),
        "--adaptive-entropy",
        # Prediction head
        "--use-prediction-loss",
        "--prediction-loss-weight", str(config.prediction_loss_weight),
        "--pred-head-size", "large",
        # PopArt (now default=True)
        "--popart-beta", "0.0001",
        # DS24 logging (reduced for sweep)
        "--ds24-level", "3",
        "--ds24-intermediates-every", "10",
        # Trading costs
        "--fee", "0",
        "--slippage", "0",
        # Sweep-specific
        "--name", run_name,
        "--no-checkpoint",  # Don't save checkpoints for sweep
    ]

    # Physics warmup
    if hasattr(config, 'physics_warmup_steps'):
        cmd.extend(["--physics-warmup-steps", str(config.physics_warmup_steps)])

    return cmd


def parse_output(output: str) -> Tuple[List[dict], dict]:
    """Parse training output to extract per-epoch metrics."""
    epochs_data = []

    # Pattern for epoch lines:
    # Epoch   10 | Reward:  +18.24 | α: +52.79% (Mkt: -0.3%) | Fee:  0 | Steps: 400 | S/H/B/LS/LB: 17/339/15/11/18
    epoch_pattern = re.compile(
        r'Epoch\s+(\d+)\s+\|.*?Reward:\s+([\+\-]?\d+\.?\d*)\s+\|.*?'
        r'S/H/B/LS/LB:\s+(\d+)/(\d+)/(\d+)/(\d+)/(\d+)'
    )

    # Pattern for training metrics:
    # Loss: 8.3332 | ... | Pol: 3.3369 | ... | Ent: 3.35
    # Pred: 0.237527 (λ=0.016) | Step1: 0.156381
    train_pattern = re.compile(
        r'Ent:\s+([\d\.]+)'
    )
    pred_pattern = re.compile(
        r'Pred:\s+([\d\.]+).*?Step1:\s+([\d\.]+)'
    )

    current_epoch = None

    for line in output.split('\n'):
        # Check for epoch line
        epoch_match = epoch_pattern.search(line)
        if epoch_match:
            epoch_num = int(epoch_match.group(1))
            reward = float(epoch_match.group(2))
            n_sell = int(epoch_match.group(3))
            n_hold = int(epoch_match.group(4))
            n_buy = int(epoch_match.group(5))
            n_limit_sell = int(epoch_match.group(6))
            n_limit_buy = int(epoch_match.group(7))
            total = n_sell + n_hold + n_buy + n_limit_sell + n_limit_buy

            current_epoch = {
                'epoch': epoch_num,
                'reward': reward,
                'hold_pct': 100.0 * n_hold / max(total, 1),
                'entropy': None,
                'pred_loss': None,
                'step1_mse': None,
            }
            epochs_data.append(current_epoch)

        # Check for entropy
        if current_epoch is not None:
            ent_match = train_pattern.search(line)
            if ent_match:
                current_epoch['entropy'] = float(ent_match.group(1))

            pred_match = pred_pattern.search(line)
            if pred_match:
                current_epoch['pred_loss'] = float(pred_match.group(1))
                current_epoch['step1_mse'] = float(pred_match.group(2))

    return epochs_data


def compute_metrics(epochs_data: List[dict], config: SweepConfig, runtime: float) -> RunMetrics:
    """Compute summary metrics from epoch data."""
    metrics = RunMetrics(
        config_name=config.name,
        config=asdict(config),
        total_epochs=len(epochs_data),
        runtime_seconds=runtime,
        status="completed",
    )

    if not epochs_data:
        metrics.status = "no_data"
        return metrics

    # Find collapse points
    for ep in epochs_data:
        epoch = ep['epoch']
        hold_pct = ep.get('hold_pct', 0)
        entropy = ep.get('entropy')

        if hold_pct > 70 and metrics.epochs_until_hold_70 is None:
            metrics.epochs_until_hold_70 = epoch
        if hold_pct > 80 and metrics.epochs_until_hold_80 is None:
            metrics.epochs_until_hold_80 = epoch

        if entropy is not None:
            if entropy < 3.5 and metrics.epochs_until_entropy_35 is None:
                metrics.epochs_until_entropy_35 = epoch
            if entropy < 3.0 and metrics.epochs_until_entropy_30 is None:
                metrics.epochs_until_entropy_30 = epoch

    # Performance metrics
    rewards = [ep['reward'] for ep in epochs_data]
    metrics.max_reward = max(rewards)
    metrics.final_reward = rewards[-1]
    metrics.mean_reward = sum(rewards) / len(rewards)

    # Final state
    final = epochs_data[-1]
    metrics.final_hold_pct = final.get('hold_pct', 0)
    metrics.final_entropy = final.get('entropy', 0) or 0
    metrics.final_pred_loss = final.get('pred_loss', 0) or 0
    metrics.final_step1_mse = final.get('step1_mse', 0) or 0

    # Stability score (higher = more stable)
    # Reward for delaying collapse, penalize for early collapse
    total_epochs = len(epochs_data)

    hold_70_score = (metrics.epochs_until_hold_70 or total_epochs + 10) / total_epochs
    hold_80_score = (metrics.epochs_until_hold_80 or total_epochs + 10) / total_epochs
    entropy_35_score = (metrics.epochs_until_entropy_35 or total_epochs + 10) / total_epochs
    entropy_30_score = (metrics.epochs_until_entropy_30 or total_epochs + 10) / total_epochs

    # Combine: 40% HOLD stability, 30% entropy stability, 30% performance
    stability = 0.2 * hold_70_score + 0.2 * hold_80_score + 0.15 * entropy_35_score + 0.15 * entropy_30_score
    performance = metrics.max_reward / 50.0  # Normalize to ~1.0 for good runs

    metrics.stability_score = 0.7 * stability + 0.3 * performance

    return metrics


def run_single_config(
    config: SweepConfig,
    epochs: int,
    output_dir: Path,
    run_id: int,
) -> RunMetrics:
    """Run training with a single configuration."""
    run_name = f"sweep_{run_id:03d}_{config.name}"

    print(f"\n{'='*70}")
    print(f"[{run_id}] Starting: {config.name}")
    print(f"{'='*70}")
    print(f"Config: sims_warmup={config.sims_warmup}, sims={config.sims_min}->{config.sims_max}")
    print(f"        entropy_coef={config.entropy_coef}, threshold={config.entropy_hold_threshold}")
    print(f"        pred_weight={config.prediction_loss_weight}")

    cmd = build_command(config, epochs, run_name)

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout per run
            cwd=str(Path(__file__).parent),
        )

        runtime = time.time() - start_time

        # Save raw output
        output_file = output_dir / f"{run_name}_output.txt"
        with open(output_file, 'w') as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n\nSTDERR:\n")
                f.write(result.stderr)

        # Parse metrics
        epochs_data = parse_output(result.stdout)
        metrics = compute_metrics(epochs_data, config, runtime)

        if result.returncode != 0:
            metrics.status = "error"
            metrics.error = result.stderr[:500] if result.stderr else "Unknown error"

        print(f"[{run_id}] Completed in {runtime/60:.1f} min")
        print(f"        Max reward: {metrics.max_reward:.2f}, Final: {metrics.final_reward:.2f}")
        print(f"        HOLD>70% at epoch: {metrics.epochs_until_hold_70 or 'never'}")
        print(f"        Entropy<3.5 at epoch: {metrics.epochs_until_entropy_35 or 'never'}")
        print(f"        Stability score: {metrics.stability_score:.3f}")

        return metrics

    except subprocess.TimeoutExpired:
        runtime = time.time() - start_time
        metrics = RunMetrics(
            config_name=config.name,
            config=asdict(config),
            runtime_seconds=runtime,
            status="timeout",
            error="Exceeded 2 hour timeout",
        )
        print(f"[{run_id}] TIMEOUT after {runtime/60:.1f} min")
        return metrics

    except Exception as e:
        runtime = time.time() - start_time
        metrics = RunMetrics(
            config_name=config.name,
            config=asdict(config),
            runtime_seconds=runtime,
            status="exception",
            error=str(e)[:500],
        )
        print(f"[{run_id}] EXCEPTION: {e}")
        return metrics


def save_results(results: List[RunMetrics], output_dir: Path):
    """Save results to CSV and JSON."""
    # JSON for full data
    json_file = output_dir / "sweep_results.json"
    with open(json_file, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    # CSV summary
    csv_file = output_dir / "sweep_summary.csv"
    with open(csv_file, 'w') as f:
        headers = [
            'config_name', 'stability_score', 'max_reward', 'final_reward',
            'epochs_until_hold_70', 'epochs_until_hold_80',
            'epochs_until_entropy_35', 'epochs_until_entropy_30',
            'final_entropy', 'final_hold_pct', 'final_pred_loss',
            'runtime_min', 'status'
        ]
        f.write(','.join(headers) + '\n')

        for r in sorted(results, key=lambda x: -x.stability_score):
            row = [
                r.config_name,
                f"{r.stability_score:.4f}",
                f"{r.max_reward:.2f}",
                f"{r.final_reward:.2f}",
                str(r.epochs_until_hold_70 or ''),
                str(r.epochs_until_hold_80 or ''),
                str(r.epochs_until_entropy_35 or ''),
                str(r.epochs_until_entropy_30 or ''),
                f"{r.final_entropy:.3f}",
                f"{r.final_hold_pct:.1f}",
                f"{r.final_pred_loss:.4f}",
                f"{r.runtime_seconds/60:.1f}",
                r.status,
            ]
            f.write(','.join(row) + '\n')

    print(f"\nResults saved to:")
    print(f"  {json_file}")
    print(f"  {csv_file}")


def print_final_report(results: List[RunMetrics]):
    """Print summary report."""
    print("\n" + "="*70)
    print("SWEEP RESULTS SUMMARY")
    print("="*70)

    # Sort by stability score
    sorted_results = sorted(results, key=lambda x: -x.stability_score)

    print("\nTOP 5 CONFIGURATIONS (by stability score):\n")
    print(f"{'Rank':<5} {'Config':<25} {'Score':<8} {'MaxRwd':<8} {'HOLD>70':<10} {'Ent<3.5':<10}")
    print("-"*70)

    for i, r in enumerate(sorted_results[:5], 1):
        hold_70 = str(r.epochs_until_hold_70) if r.epochs_until_hold_70 else "never"
        ent_35 = str(r.epochs_until_entropy_35) if r.epochs_until_entropy_35 else "never"
        print(f"{i:<5} {r.config_name:<25} {r.stability_score:<8.4f} {r.max_reward:<8.2f} {hold_70:<10} {ent_35:<10}")

    # Best config details
    best = sorted_results[0]
    print(f"\n{'='*70}")
    print("BEST CONFIGURATION DETAILS:")
    print(f"{'='*70}")
    print(f"Name: {best.config_name}")
    print(f"Config: {json.dumps(best.config, indent=2)}")
    print(f"\nMetrics:")
    print(f"  Stability Score: {best.stability_score:.4f}")
    print(f"  Max Reward: {best.max_reward:.2f}")
    print(f"  Final Reward: {best.final_reward:.2f}")
    print(f"  Epochs until HOLD>70%: {best.epochs_until_hold_70 or 'never'}")
    print(f"  Epochs until entropy<3.5: {best.epochs_until_entropy_35 or 'never'}")
    print(f"  Final entropy: {best.final_entropy:.3f}")
    print(f"  Final HOLD%: {best.final_hold_pct:.1f}%")

    # Recommendations
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS:")
    print(f"{'='*70}")

    # Check if any config prevented collapse
    stable_configs = [r for r in results if r.epochs_until_hold_70 is None or r.epochs_until_hold_70 > 40]

    if stable_configs:
        print(f"\n{len(stable_configs)} config(s) maintained stability for 40+ epochs:")
        for r in stable_configs[:3]:
            print(f"  - {r.config_name}: sims_warmup={r.config['sims_warmup']}, entropy={r.config['entropy_coef']}")
    else:
        print("\nNo config maintained stability for 40+ epochs.")
        print("Consider implementing TASK_037 (ε-greedy MCTS) for stronger intervention.")

    # Check entropy patterns
    high_entropy_configs = [r for r in results if r.final_entropy > 3.5]
    if high_entropy_configs:
        print(f"\n{len(high_entropy_configs)} config(s) maintained entropy > 3.5:")
        for r in high_entropy_configs[:3]:
            print(f"  - {r.config_name}: final_entropy={r.final_entropy:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Warmup Configuration Sweep")
    parser.add_argument('--epochs', type=int, default=50,
                        help='Epochs per run (default: 50)')
    parser.add_argument('--output-dir', type=str, default=str(PROJECT_ROOT / 'logs' / 'sweep_results'),
                        help='Output directory (default: sweep_results)')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick subset of configs')
    parser.add_argument('--config', type=str, default=None,
                        help='Run single config by name')
    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"sweep_{timestamp}"
    run_dir.mkdir(exist_ok=True)

    print(f"Warmup Configuration Sweep")
    print(f"Output: {run_dir}")
    print(f"Epochs per run: {args.epochs}")
    print(f"Mode: {'quick' if args.quick else 'full'}")

    # Get configs
    configs = get_sweep_configs(quick=args.quick)

    if args.config:
        configs = [c for c in configs if c.name == args.config]
        if not configs:
            print(f"Config '{args.config}' not found!")
            sys.exit(1)

    print(f"Configurations to test: {len(configs)}")

    # Estimate runtime
    est_time = len(configs) * args.epochs * 1.5 / 60  # ~1.5 min per epoch
    print(f"Estimated runtime: {est_time:.1f} hours")

    # Run sweep
    results = []
    for i, config in enumerate(configs):
        metrics = run_single_config(config, args.epochs, run_dir, i)
        results.append(metrics)

        # Save intermediate results
        save_results(results, run_dir)

    # Final report
    print_final_report(results)
    save_results(results, run_dir)

    print(f"\nSweep complete! Results in: {run_dir}")


if __name__ == "__main__":
    main()
