#!/usr/bin/env python3
"""
Stability Intervention Sweep

Tests the new stability modules (TASK_037-043) systematically:
1. Individual interventions vs baseline
2. Combinations of interventions
3. Best configs from warmup_sweep + new interventions

Usage:
    python tools/sweeps/stability_sweep.py --mode quick      # 50 epochs, ~2h
    python tools/sweeps/stability_sweep.py --mode full       # 150 epochs, ~8h
    python tools/sweeps/stability_sweep.py --mode overnight  # 300 epochs, ~16h

Results saved to: logs/stability_sweep_<timestamp>/
"""

import subprocess
import json
import os
import sys
import time
import re
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class InterventionConfig:
    """Configuration for a stability intervention test."""
    name: str
    description: str = ""

    # Base settings (from kitchen_sink - best warmup config)
    sims_warmup: int = 300
    sims_min: int = 4
    sims_max: int = 24
    entropy_coef: float = 0.2
    entropy_hold_threshold: float = 0.55
    entropy_boost_low: float = 0.4
    entropy_boost_high: float = 0.6
    prediction_loss_weight: float = 0.5

    # TASK_037: ε-greedy MCTS
    epsilon_greedy: float = 0.0

    # TASK_039: Dirichlet noise
    dirichlet_aggressive: bool = False
    noise: float = 0.3
    noise_frac: float = 0.25

    # TASK_040: SAC entropy
    sac_entropy: bool = False
    sac_target_entropy_ratio: float = 0.5
    sac_alpha_lr: float = 3e-4
    sac_init_alpha: float = 0.01

    # TASK_041: Flow matching
    flow_matching: bool = False
    flow_velocity_weight: float = 0.8

    # TASK_038: Differential reward
    differential_reward: bool = False
    differential_weight: float = 1.0


# ============================================================================
# SWEEP CONFIGURATIONS
# ============================================================================

def get_baseline_config() -> InterventionConfig:
    """Baseline: kitchen_sink from warmup sweep (no new interventions)."""
    return InterventionConfig(
        name="baseline_kitchen_sink",
        description="Best config from warmup_sweep, no new interventions",
    )


def get_individual_interventions() -> List[InterventionConfig]:
    """Test each intervention individually against baseline."""
    return [
        # ε-greedy variations
        InterventionConfig(
            name="epsilon_010",
            description="ε-greedy 10%",
            epsilon_greedy=0.1,
        ),
        InterventionConfig(
            name="epsilon_020",
            description="ε-greedy 20%",
            epsilon_greedy=0.2,
        ),
        InterventionConfig(
            name="epsilon_030",
            description="ε-greedy 30%",
            epsilon_greedy=0.3,
        ),

        # Dirichlet aggressive
        InterventionConfig(
            name="dirichlet_aggressive",
            description="Aggressive Dirichlet (α=0.5, ε=0.5)",
            dirichlet_aggressive=True,
        ),

        # SAC entropy variations
        InterventionConfig(
            name="sac_entropy_050",
            description="SAC entropy, 50% target",
            sac_entropy=True,
            sac_target_entropy_ratio=0.5,
        ),
        InterventionConfig(
            name="sac_entropy_040",
            description="SAC entropy, 40% target",
            sac_entropy=True,
            sac_target_entropy_ratio=0.4,
        ),

        # Flow matching
        InterventionConfig(
            name="flow_matching_080",
            description="Flow matching, 80% velocity weight",
            flow_matching=True,
            flow_velocity_weight=0.8,
        ),
        InterventionConfig(
            name="flow_matching_100",
            description="Flow matching, 100% velocity weight",
            flow_matching=True,
            flow_velocity_weight=1.0,
        ),

        # Differential reward
        InterventionConfig(
            name="differential_reward_100",
            description="Differential reward, full weight",
            differential_reward=True,
            differential_weight=1.0,
        ),
        InterventionConfig(
            name="differential_reward_050",
            description="Differential reward, 50% weight",
            differential_reward=True,
            differential_weight=0.5,
        ),
    ]


def get_combination_configs() -> List[InterventionConfig]:
    """Test promising combinations of interventions."""
    return [
        # ε-greedy + Dirichlet (exploration boost)
        InterventionConfig(
            name="combo_exploration",
            description="ε-greedy 20% + aggressive Dirichlet",
            epsilon_greedy=0.2,
            dirichlet_aggressive=True,
        ),

        # ε-greedy + SAC entropy (entropy control)
        InterventionConfig(
            name="combo_entropy_control",
            description="ε-greedy 20% + SAC entropy",
            epsilon_greedy=0.2,
            sac_entropy=True,
            sac_target_entropy_ratio=0.5,
        ),

        # Flow matching + differential reward (prediction + reward shaping)
        InterventionConfig(
            name="combo_prediction_reward",
            description="Flow matching + differential reward",
            flow_matching=True,
            differential_reward=True,
        ),

        # Triple: ε-greedy + Dirichlet + SAC
        InterventionConfig(
            name="combo_triple_entropy",
            description="ε-greedy + Dirichlet + SAC entropy",
            epsilon_greedy=0.2,
            dirichlet_aggressive=True,
            sac_entropy=True,
        ),

        # All exploration interventions
        InterventionConfig(
            name="combo_all_exploration",
            description="All exploration: ε-greedy + Dirichlet + SAC + differential",
            epsilon_greedy=0.2,
            dirichlet_aggressive=True,
            sac_entropy=True,
            differential_reward=True,
        ),

        # Kitchen sink v2: all interventions
        InterventionConfig(
            name="kitchen_sink_v2",
            description="All interventions enabled",
            epsilon_greedy=0.2,
            dirichlet_aggressive=True,
            sac_entropy=True,
            flow_matching=True,
            differential_reward=True,
        ),

        # Conservative combo: mild settings
        InterventionConfig(
            name="combo_conservative",
            description="Conservative: ε-greedy 10% + flow matching",
            epsilon_greedy=0.1,
            flow_matching=True,
            flow_velocity_weight=0.5,
        ),

        # Aggressive combo: strong settings
        InterventionConfig(
            name="combo_aggressive",
            description="Aggressive: ε-greedy 30% + all exploration",
            epsilon_greedy=0.3,
            dirichlet_aggressive=True,
            sac_entropy=True,
            sac_target_entropy_ratio=0.6,
            differential_reward=True,
        ),
    ]


def get_quick_configs() -> List[InterventionConfig]:
    """Quick mode: baseline + top interventions only."""
    return [
        get_baseline_config(),
        InterventionConfig(
            name="epsilon_020",
            description="ε-greedy 20%",
            epsilon_greedy=0.2,
        ),
        InterventionConfig(
            name="dirichlet_aggressive",
            description="Aggressive Dirichlet",
            dirichlet_aggressive=True,
        ),
        InterventionConfig(
            name="sac_entropy_050",
            description="SAC entropy 50%",
            sac_entropy=True,
        ),
        InterventionConfig(
            name="flow_matching_080",
            description="Flow matching 80%",
            flow_matching=True,
        ),
        InterventionConfig(
            name="combo_exploration",
            description="ε-greedy + Dirichlet",
            epsilon_greedy=0.2,
            dirichlet_aggressive=True,
        ),
        InterventionConfig(
            name="kitchen_sink_v2",
            description="All interventions",
            epsilon_greedy=0.2,
            dirichlet_aggressive=True,
            sac_entropy=True,
            flow_matching=True,
            differential_reward=True,
        ),
    ]


# ============================================================================
# SWEEP RUNNER
# ============================================================================

def build_command(
    config: InterventionConfig,
    epochs: int,
    run_name: str,
    parallel_envs: int = 50,
    use_jax_env: bool = True,
) -> List[str]:
    """Build train_jax_fast.py command from config."""
    cmd = [
        "python", str(PROJECT_ROOT / "train_jax_fast.py"),
        "--synthetic-level", "0",
        "--synthetic-steps", "10000",
        "--epochs", str(epochs),
        "--batch-size", "384",
        "--model-size", "small",
        "--steps", "400",
        "--parallel-envs", str(parallel_envs),
    ]
    if use_jax_env:
        cmd.append("--jax-env")
    cmd.extend([
        "--bf16",
        "--popart-beta", "0.0001",
        "--use-prediction-loss",
        "--prediction-loss-weight", str(config.prediction_loss_weight),
        "--pred-head-size", "large",
        "--ds24-level", "3",
        "--ds24-intermediates-every", "5",
        "--regenerate-data-every", "50",
        "--fee", "0",
        "--slippage", "0",
        "--no-checkpoint",
        "--name", run_name,
        # Warmup settings
        "--sims", str(config.sims_max),
        "--sims-warmup", str(config.sims_warmup),
        "--sims-min", str(config.sims_min),
        # Entropy settings
        "--entropy-coef", str(config.entropy_coef),
        "--adaptive-entropy",
        "--entropy-hold-threshold", str(config.entropy_hold_threshold),
        "--entropy-boost-low", str(config.entropy_boost_low),
        "--entropy-boost-high", str(config.entropy_boost_high),
    ])

    # TASK_037: ε-greedy
    if config.epsilon_greedy > 0:
        cmd.extend(["--epsilon-greedy", str(config.epsilon_greedy)])

    # TASK_039: Dirichlet
    if config.dirichlet_aggressive:
        cmd.append("--dirichlet-aggressive")
    elif config.noise != 0.3 or config.noise_frac != 0.25:
        cmd.extend(["--noise", str(config.noise), "--noise-frac", str(config.noise_frac)])

    # TASK_040: SAC entropy
    if config.sac_entropy:
        cmd.extend([
            "--sac-entropy",
            "--sac-target-entropy-ratio", str(config.sac_target_entropy_ratio),
            "--sac-alpha-lr", str(config.sac_alpha_lr),
            "--sac-init-alpha", str(config.sac_init_alpha),
        ])

    # TASK_041: Flow matching
    if config.flow_matching:
        cmd.extend([
            "--flow-matching",
            "--flow-velocity-weight", str(config.flow_velocity_weight),
        ])

    # TASK_038: Differential reward
    if config.differential_reward:
        cmd.extend([
            "--differential-reward",
            "--differential-weight", str(config.differential_weight),
        ])

    return cmd


def parse_training_output(output: str) -> Dict[str, Any]:
    """Parse training output for metrics."""
    metrics = {
        'epochs_completed': 0,
        'max_reward': -999,
        'final_reward': -999,
        'hold_exceeded_70_at': None,
        'entropy_below_35_at': None,
        'final_entropy': 0,
        'final_hold_pct': 0,
        'crashed': False,
        'error': None,
    }

    lines = output.split('\n')
    epoch_rewards = []
    epoch_entropies = []
    epoch_hold_pcts = []

    for line in lines:
        # Parse epoch metrics
        epoch_match = re.search(r'Epoch\s+(\d+)', line)
        if epoch_match:
            metrics['epochs_completed'] = int(epoch_match.group(1))

        # Parse reward
        reward_match = re.search(r'Reward:\s*([-\d.]+)', line)
        if reward_match:
            reward = float(reward_match.group(1))
            epoch_rewards.append(reward)

        # Parse entropy
        entropy_match = re.search(r'Entropy:\s*([-\d.]+)', line)
        if entropy_match:
            entropy = float(entropy_match.group(1))
            epoch_entropies.append(entropy)

        # Parse HOLD%
        hold_match = re.search(r'HOLD:\s*([\d.]+)%', line)
        if hold_match:
            hold_pct = float(hold_match.group(1)) / 100
            epoch_hold_pcts.append(hold_pct)

        # Check for errors
        if 'Error' in line or 'Exception' in line or 'Traceback' in line:
            metrics['crashed'] = True
            metrics['error'] = line[:200]

    if epoch_rewards:
        metrics['max_reward'] = max(epoch_rewards)
        metrics['final_reward'] = epoch_rewards[-1]

    if epoch_entropies:
        metrics['final_entropy'] = epoch_entropies[-1]
        # Find when entropy dropped below 3.5
        for i, ent in enumerate(epoch_entropies):
            if ent < 3.5 and metrics['entropy_below_35_at'] is None:
                metrics['entropy_below_35_at'] = i + 1

    if epoch_hold_pcts:
        metrics['final_hold_pct'] = epoch_hold_pcts[-1]
        # Find when HOLD exceeded 70%
        for i, hold in enumerate(epoch_hold_pcts):
            if hold > 0.70 and metrics['hold_exceeded_70_at'] is None:
                metrics['hold_exceeded_70_at'] = i + 1

    return metrics


def compute_stability_score(metrics: Dict[str, Any], max_epochs: int) -> float:
    """Compute stability score (higher = better)."""
    if metrics['crashed']:
        return 0.0

    score = 0.0

    # Reward component (40%)
    reward_score = (metrics['max_reward'] + 50) / 100  # Normalize roughly
    score += 0.4 * min(max(reward_score, 0), 1)

    # HOLD delay component (30%)
    hold_epoch = metrics['hold_exceeded_70_at'] or max_epochs
    hold_score = hold_epoch / max_epochs
    score += 0.3 * hold_score

    # Entropy stability component (30%)
    entropy_epoch = metrics['entropy_below_35_at'] or max_epochs
    entropy_score = entropy_epoch / max_epochs
    score += 0.3 * entropy_score

    return score


def run_sweep(
    configs: List[InterventionConfig],
    epochs: int,
    output_dir: Path,
    timeout_minutes: int = 60,
    parallel_envs: int = 50,
    use_jax_env: bool = True,
) -> List[Dict[str, Any]]:
    """Run sweep and collect results."""
    results = []

    print(f"\n{'='*70}")
    print(f"STABILITY INTERVENTION SWEEP")
    print(f"{'='*70}")
    print(f"Configs: {len(configs)}")
    print(f"Epochs per config: {epochs}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")

    for i, config in enumerate(configs):
        run_name = f"sweep_{config.name}"
        print(f"\n[{i+1}/{len(configs)}] Running: {config.name}")
        print(f"    Description: {config.description}")

        cmd = build_command(config, epochs, run_name, parallel_envs, use_jax_env)
        print(f"    Command: {' '.join(cmd[:10])}...")

        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_minutes * 60,
            )
            output = result.stdout + result.stderr
            elapsed = time.time() - start_time

            metrics = parse_training_output(output)
            metrics['elapsed_seconds'] = elapsed
            metrics['stability_score'] = compute_stability_score(metrics, epochs)

            print(f"    ✓ Completed in {elapsed/60:.1f}m")
            print(f"    Max Reward: {metrics['max_reward']:.2f}")
            print(f"    HOLD>70% at: {metrics['hold_exceeded_70_at'] or 'never'}")
            print(f"    Entropy<3.5 at: {metrics['entropy_below_35_at'] or 'never'}")
            print(f"    Stability Score: {metrics['stability_score']:.4f}")

        except subprocess.TimeoutExpired:
            print(f"    ✗ Timeout after {timeout_minutes}m")
            metrics = {
                'crashed': True,
                'error': 'Timeout',
                'stability_score': 0.0,
            }
            output = ""
        except Exception as e:
            print(f"    ✗ Error: {e}")
            metrics = {
                'crashed': True,
                'error': str(e),
                'stability_score': 0.0,
            }
            output = ""

        # Save individual result
        result_entry = {
            'config': asdict(config),
            'metrics': metrics,
        }
        results.append(result_entry)

        # Save output log
        log_file = output_dir / f"{config.name}.log"
        with open(log_file, 'w') as f:
            f.write(output)

        # Save intermediate results
        results_file = output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

    return results


def print_summary(results: List[Dict[str, Any]], output_dir: Path):
    """Print and save sweep summary."""
    print(f"\n{'='*70}")
    print("SWEEP RESULTS SUMMARY")
    print(f"{'='*70}\n")

    # Sort by stability score
    sorted_results = sorted(
        results,
        key=lambda x: x['metrics'].get('stability_score', 0),
        reverse=True
    )

    print("TOP CONFIGURATIONS (by stability score):\n")
    print(f"{'Rank':<6}{'Config':<30}{'Score':<10}{'MaxRwd':<10}{'HOLD>70':<12}{'Ent<3.5':<12}")
    print("-" * 80)

    for i, result in enumerate(sorted_results[:10]):
        config = result['config']
        metrics = result['metrics']
        hold_at = metrics.get('hold_exceeded_70_at', 'never')
        ent_at = metrics.get('entropy_below_35_at', 'never')

        print(f"{i+1:<6}{config['name']:<30}{metrics.get('stability_score', 0):.4f}    "
              f"{metrics.get('max_reward', -999):.2f}     {str(hold_at):<12}{str(ent_at):<12}")

    # Best config details
    if sorted_results:
        best = sorted_results[0]
        print(f"\n{'='*70}")
        print("BEST CONFIGURATION:")
        print(f"{'='*70}")
        print(f"Name: {best['config']['name']}")
        print(f"Description: {best['config']['description']}")
        print("\nEnabled interventions:")
        if best['config']['epsilon_greedy'] > 0:
            print(f"  - ε-greedy: {best['config']['epsilon_greedy']}")
        if best['config']['dirichlet_aggressive']:
            print(f"  - Dirichlet aggressive: α=0.5, ε=0.5")
        if best['config']['sac_entropy']:
            print(f"  - SAC entropy: target={best['config']['sac_target_entropy_ratio']}")
        if best['config']['flow_matching']:
            print(f"  - Flow matching: velocity_weight={best['config']['flow_velocity_weight']}")
        if best['config']['differential_reward']:
            print(f"  - Differential reward: weight={best['config']['differential_weight']}")

    # Save summary
    summary_file = output_dir / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write("STABILITY INTERVENTION SWEEP RESULTS\n")
        f.write("=" * 70 + "\n\n")
        for i, result in enumerate(sorted_results):
            f.write(f"{i+1}. {result['config']['name']}: "
                   f"score={result['metrics'].get('stability_score', 0):.4f}, "
                   f"max_reward={result['metrics'].get('max_reward', -999):.2f}\n")

    print(f"\nResults saved to: {output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Stability Intervention Sweep')
    parser.add_argument('--mode', type=str, default='quick',
                       choices=['quick', 'full', 'overnight'],
                       help='Sweep mode: quick (50 epochs), full (150), overnight (300)')
    parser.add_argument('--timeout', type=int, default=60,
                       help='Timeout per config in minutes (default: 60)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: logs/stability_sweep_<timestamp>)')
    parser.add_argument('--parallel-envs', type=int, default=50,
                       help='Parallel environments (default: 50 for JAX env)')
    parser.add_argument('--no-jax-env', action='store_true',
                       help='Disable JAX env (use CPU envs, set --parallel-envs 4-12)')
    args = parser.parse_args()

    # Set epochs and timeout based on mode
    if args.mode == 'quick':
        epochs = 50
        configs = get_quick_configs()
        default_timeout = 30  # 30 min for 50 epochs
    elif args.mode == 'full':
        epochs = 150
        configs = [get_baseline_config()] + get_individual_interventions() + get_combination_configs()
        default_timeout = 90  # 90 min for 150 epochs
    else:  # overnight
        epochs = 300
        configs = [get_baseline_config()] + get_individual_interventions() + get_combination_configs()
        default_timeout = 270  # 4.5 hours for 300 epochs (~43 sec/epoch)

    # Use provided timeout or mode default
    timeout = args.timeout if args.timeout != 60 else default_timeout

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "logs" / f"stability_sweep_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_file = output_dir / "sweep_config.json"
    with open(config_file, 'w') as f:
        json.dump({
            'mode': args.mode,
            'epochs': epochs,
            'num_configs': len(configs),
            'configs': [asdict(c) for c in configs],
        }, f, indent=2)

    # Run sweep
    use_jax_env = not args.no_jax_env
    print(f"Environment: {'JAX (GPU)' if use_jax_env else 'CPU'} with {args.parallel_envs} parallel envs")
    print(f"Timeout per config: {timeout} minutes")

    results = run_sweep(
        configs, epochs, output_dir, timeout,
        parallel_envs=args.parallel_envs,
        use_jax_env=use_jax_env,
    )

    # Print summary
    print_summary(results, output_dir)


if __name__ == "__main__":
    main()
