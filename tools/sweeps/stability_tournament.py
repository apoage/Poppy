#!/usr/bin/env python3
# Add project root to path for imports
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))
"""
Stability Tournament

Run promising configurations for longer durations to determine true winners.
Uses statistical analysis to identify significantly better configs.

Features:
- Multiple runs per config for statistical significance
- Tracks variance and confidence intervals
- Saves checkpoints for best performers
- Real-time leaderboard updates

Usage:
    # Quick tournament (3 runs, 100 epochs each)
    python stability_tournament.py --mode quick

    # Full tournament (5 runs, 200 epochs each)
    python stability_tournament.py --mode full

    # From sweep results (auto-select top N)
    python stability_tournament.py --from-sweep logs/stability_sweep_*/results.json --top 5
"""

import subprocess
import json
import os
import sys
import time
import re
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path
import statistics


@dataclass
class TournamentConfig:
    """Configuration for tournament participant."""
    name: str
    description: str = ""

    # Base settings
    sims_warmup: int = 300
    sims_min: int = 4
    sims_max: int = 24
    entropy_coef: float = 0.2
    entropy_hold_threshold: float = 0.55
    entropy_boost_low: float = 0.4
    entropy_boost_high: float = 0.6
    prediction_loss_weight: float = 0.5

    # Interventions
    epsilon_greedy: float = 0.0
    dirichlet_aggressive: bool = False
    sac_entropy: bool = False
    sac_target_entropy_ratio: float = 0.5
    flow_matching: bool = False
    flow_velocity_weight: float = 0.8
    differential_reward: bool = False
    differential_weight: float = 1.0


# ============================================================================
# TOURNAMENT PARTICIPANTS
# ============================================================================

def get_tournament_configs() -> List[TournamentConfig]:
    """Default tournament participants - promising configs from research."""
    return [
        # Baseline (kitchen_sink from warmup)
        TournamentConfig(
            name="baseline",
            description="Kitchen sink from warmup_sweep (no new interventions)",
        ),

        # Individual winners (hypothesized)
        TournamentConfig(
            name="epsilon_020",
            description="ε-greedy 20% only",
            epsilon_greedy=0.2,
        ),

        TournamentConfig(
            name="sac_entropy",
            description="SAC entropy 50% target",
            sac_entropy=True,
            sac_target_entropy_ratio=0.5,
        ),

        # Best combinations (hypothesized)
        TournamentConfig(
            name="epsilon_dirichlet",
            description="ε-greedy 20% + aggressive Dirichlet",
            epsilon_greedy=0.2,
            dirichlet_aggressive=True,
        ),

        TournamentConfig(
            name="epsilon_sac",
            description="ε-greedy 20% + SAC entropy",
            epsilon_greedy=0.2,
            sac_entropy=True,
        ),

        TournamentConfig(
            name="full_stack",
            description="All interventions",
            epsilon_greedy=0.2,
            dirichlet_aggressive=True,
            sac_entropy=True,
            flow_matching=True,
            differential_reward=True,
        ),
    ]


def load_configs_from_sweep(sweep_results_path: str, top_n: int = 5) -> List[TournamentConfig]:
    """Load top configs from sweep results."""
    with open(sweep_results_path) as f:
        results = json.load(f)

    # Sort by stability score
    sorted_results = sorted(
        results,
        key=lambda x: x['metrics'].get('stability_score', 0),
        reverse=True
    )

    configs = []
    for result in sorted_results[:top_n]:
        cfg = result['config']
        configs.append(TournamentConfig(
            name=cfg['name'],
            description=cfg.get('description', ''),
            sims_warmup=cfg.get('sims_warmup', 300),
            sims_min=cfg.get('sims_min', 4),
            sims_max=cfg.get('sims_max', 24),
            entropy_coef=cfg.get('entropy_coef', 0.2),
            entropy_hold_threshold=cfg.get('entropy_hold_threshold', 0.55),
            entropy_boost_low=cfg.get('entropy_boost_low', 0.4),
            entropy_boost_high=cfg.get('entropy_boost_high', 0.6),
            prediction_loss_weight=cfg.get('prediction_loss_weight', 0.5),
            epsilon_greedy=cfg.get('epsilon_greedy', 0.0),
            dirichlet_aggressive=cfg.get('dirichlet_aggressive', False),
            sac_entropy=cfg.get('sac_entropy', False),
            sac_target_entropy_ratio=cfg.get('sac_target_entropy_ratio', 0.5),
            flow_matching=cfg.get('flow_matching', False),
            flow_velocity_weight=cfg.get('flow_velocity_weight', 0.8),
            differential_reward=cfg.get('differential_reward', False),
            differential_weight=cfg.get('differential_weight', 1.0),
        ))

    return configs


# ============================================================================
# TOURNAMENT RUNNER
# ============================================================================

def build_command(
    config: TournamentConfig,
    epochs: int,
    run_name: str,
    save_checkpoint: bool = False,
    parallel_envs: int = 50,
    use_jax_env: bool = True,
) -> List[str]:
    """Build training command."""
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

    if not save_checkpoint:
        cmd.append("--no-checkpoint")

    # Interventions
    if config.epsilon_greedy > 0:
        cmd.extend(["--epsilon-greedy", str(config.epsilon_greedy)])
    if config.dirichlet_aggressive:
        cmd.append("--dirichlet-aggressive")
    if config.sac_entropy:
        cmd.extend([
            "--sac-entropy",
            "--sac-target-entropy-ratio", str(config.sac_target_entropy_ratio),
        ])
    if config.flow_matching:
        cmd.extend([
            "--flow-matching",
            "--flow-velocity-weight", str(config.flow_velocity_weight),
        ])
    if config.differential_reward:
        cmd.extend([
            "--differential-reward",
            "--differential-weight", str(config.differential_weight),
        ])

    return cmd


def parse_training_output(output: str) -> Dict[str, Any]:
    """Parse training output for key metrics."""
    metrics = {
        'epochs_completed': 0,
        'max_reward': -999,
        'final_reward': -999,
        'avg_reward_last_10': -999,
        'hold_exceeded_70_at': None,
        'entropy_below_35_at': None,
        'final_entropy': 0,
        'final_hold_pct': 0,
        'crashed': False,
    }

    lines = output.split('\n')
    epoch_rewards = []
    epoch_entropies = []
    epoch_hold_pcts = []

    for line in lines:
        epoch_match = re.search(r'Epoch\s+(\d+)', line)
        if epoch_match:
            metrics['epochs_completed'] = int(epoch_match.group(1))

        reward_match = re.search(r'Reward:\s*([-\d.]+)', line)
        if reward_match:
            epoch_rewards.append(float(reward_match.group(1)))

        entropy_match = re.search(r'Entropy:\s*([-\d.]+)', line)
        if entropy_match:
            epoch_entropies.append(float(entropy_match.group(1)))

        hold_match = re.search(r'HOLD:\s*([\d.]+)%', line)
        if hold_match:
            epoch_hold_pcts.append(float(hold_match.group(1)) / 100)

        if 'Error' in line or 'Exception' in line:
            metrics['crashed'] = True

    if epoch_rewards:
        metrics['max_reward'] = max(epoch_rewards)
        metrics['final_reward'] = epoch_rewards[-1]
        metrics['avg_reward_last_10'] = np.mean(epoch_rewards[-10:])

    if epoch_entropies:
        metrics['final_entropy'] = epoch_entropies[-1]
        for i, ent in enumerate(epoch_entropies):
            if ent < 3.5 and metrics['entropy_below_35_at'] is None:
                metrics['entropy_below_35_at'] = i + 1

    if epoch_hold_pcts:
        metrics['final_hold_pct'] = epoch_hold_pcts[-1]
        for i, hold in enumerate(epoch_hold_pcts):
            if hold > 0.70 and metrics['hold_exceeded_70_at'] is None:
                metrics['hold_exceeded_70_at'] = i + 1

    return metrics


def compute_composite_score(metrics: Dict[str, Any], max_epochs: int) -> float:
    """Compute composite tournament score."""
    if metrics['crashed']:
        return 0.0

    score = 0.0

    # Reward component (40%)
    reward = metrics.get('avg_reward_last_10', metrics.get('final_reward', -50))
    reward_score = (reward + 50) / 100
    score += 0.4 * np.clip(reward_score, 0, 1)

    # Stability component - HOLD delay (30%)
    hold_epoch = metrics.get('hold_exceeded_70_at') or max_epochs
    score += 0.3 * (hold_epoch / max_epochs)

    # Entropy stability (30%)
    entropy_epoch = metrics.get('entropy_below_35_at') or max_epochs
    score += 0.3 * (entropy_epoch / max_epochs)

    return score


@dataclass
class ParticipantStats:
    """Statistics for a tournament participant."""
    name: str
    runs: List[Dict[str, Any]] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)

    @property
    def mean_score(self) -> float:
        return np.mean(self.scores) if self.scores else 0.0

    @property
    def std_score(self) -> float:
        return np.std(self.scores) if len(self.scores) > 1 else 0.0

    @property
    def mean_reward(self) -> float:
        rewards = [r['avg_reward_last_10'] for r in self.runs if 'avg_reward_last_10' in r]
        return np.mean(rewards) if rewards else -999

    @property
    def mean_hold_delay(self) -> Optional[float]:
        delays = [r['hold_exceeded_70_at'] for r in self.runs if r.get('hold_exceeded_70_at')]
        return np.mean(delays) if delays else None

    def confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """Compute confidence interval for mean score."""
        if len(self.scores) < 2:
            return (self.mean_score, self.mean_score)
        from scipy import stats
        ci = stats.t.interval(confidence, len(self.scores) - 1,
                             loc=self.mean_score, scale=self.std_score / np.sqrt(len(self.scores)))
        return ci


def run_tournament(
    configs: List[TournamentConfig],
    epochs: int,
    runs_per_config: int,
    output_dir: Path,
    timeout_minutes: int = 90,
    save_best_checkpoint: bool = True,
    parallel_envs: int = 50,
    use_jax_env: bool = True,
) -> Dict[str, ParticipantStats]:
    """Run tournament with multiple runs per config."""

    print(f"\n{'='*70}")
    print("STABILITY TOURNAMENT")
    print(f"{'='*70}")
    print(f"Participants: {len(configs)}")
    print(f"Runs per participant: {runs_per_config}")
    print(f"Epochs per run: {epochs}")
    print(f"Total runs: {len(configs) * runs_per_config}")
    print(f"{'='*70}\n")

    participants: Dict[str, ParticipantStats] = {
        cfg.name: ParticipantStats(name=cfg.name) for cfg in configs
    }

    total_runs = len(configs) * runs_per_config
    run_count = 0

    for run_idx in range(runs_per_config):
        print(f"\n{'='*70}")
        print(f"ROUND {run_idx + 1}/{runs_per_config}")
        print(f"{'='*70}")

        for config in configs:
            run_count += 1
            run_name = f"tournament_{config.name}_run{run_idx + 1}"

            print(f"\n[{run_count}/{total_runs}] {config.name} (run {run_idx + 1})")

            # Save checkpoint only for best performer on last run
            save_ckpt = save_best_checkpoint and run_idx == runs_per_config - 1

            cmd = build_command(
                config, epochs, run_name,
                save_checkpoint=save_ckpt,
                parallel_envs=parallel_envs,
                use_jax_env=use_jax_env,
            )

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
                score = compute_composite_score(metrics, epochs)

                participants[config.name].runs.append(metrics)
                participants[config.name].scores.append(score)

                print(f"    Score: {score:.4f} | Reward: {metrics.get('avg_reward_last_10', -999):.2f} | "
                      f"Time: {elapsed/60:.1f}m")

                # Save log
                log_file = output_dir / f"{config.name}_run{run_idx + 1}.log"
                with open(log_file, 'w') as f:
                    f.write(output)

            except subprocess.TimeoutExpired:
                print(f"    ✗ Timeout")
                participants[config.name].scores.append(0.0)
                participants[config.name].runs.append({'crashed': True, 'error': 'Timeout'})

            except Exception as e:
                print(f"    ✗ Error: {e}")
                participants[config.name].scores.append(0.0)
                participants[config.name].runs.append({'crashed': True, 'error': str(e)})

        # Print leaderboard after each round
        print_leaderboard(participants, run_idx + 1)

        # Save intermediate results
        save_results(participants, configs, output_dir)

    return participants


def print_leaderboard(participants: Dict[str, ParticipantStats], round_num: int):
    """Print current leaderboard."""
    print(f"\n--- LEADERBOARD (after round {round_num}) ---")

    sorted_p = sorted(participants.values(), key=lambda x: x.mean_score, reverse=True)

    print(f"{'Rank':<6}{'Name':<25}{'Mean Score':<12}{'Std':<10}{'Mean Reward':<12}")
    print("-" * 65)

    for i, p in enumerate(sorted_p):
        print(f"{i+1:<6}{p.name:<25}{p.mean_score:.4f}      {p.std_score:.4f}    {p.mean_reward:.2f}")


def save_results(
    participants: Dict[str, ParticipantStats],
    configs: List[TournamentConfig],
    output_dir: Path,
):
    """Save tournament results."""
    results = {
        'participants': {
            name: {
                'mean_score': p.mean_score,
                'std_score': p.std_score,
                'mean_reward': p.mean_reward,
                'runs': p.runs,
                'scores': p.scores,
            }
            for name, p in participants.items()
        },
        'configs': {cfg.name: asdict(cfg) for cfg in configs},
        'ranking': [
            {'rank': i + 1, 'name': p.name, 'mean_score': p.mean_score}
            for i, p in enumerate(sorted(participants.values(), key=lambda x: x.mean_score, reverse=True))
        ],
    }

    results_file = output_dir / "tournament_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)


def print_final_results(participants: Dict[str, ParticipantStats], output_dir: Path):
    """Print final tournament results with statistical analysis."""
    print(f"\n{'='*70}")
    print("FINAL TOURNAMENT RESULTS")
    print(f"{'='*70}\n")

    sorted_p = sorted(participants.values(), key=lambda x: x.mean_score, reverse=True)

    print("FINAL STANDINGS:\n")
    print(f"{'Rank':<6}{'Name':<25}{'Mean':<10}{'Std':<10}{'95% CI':<20}{'Reward':<10}")
    print("-" * 80)

    for i, p in enumerate(sorted_p):
        try:
            ci = p.confidence_interval()
            ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        except:
            ci_str = "N/A"

        print(f"{i+1:<6}{p.name:<25}{p.mean_score:.4f}    {p.std_score:.4f}    {ci_str:<20}{p.mean_reward:.2f}")

    # Winner details
    winner = sorted_p[0]
    print(f"\n{'='*70}")
    print(f"WINNER: {winner.name}")
    print(f"{'='*70}")
    print(f"Mean Score: {winner.mean_score:.4f} ± {winner.std_score:.4f}")
    print(f"Mean Reward: {winner.mean_reward:.2f}")
    print(f"Mean HOLD>70% Delay: {winner.mean_hold_delay or 'never'}")

    # Statistical significance test (if we have scipy)
    if len(sorted_p) >= 2:
        try:
            from scipy import stats
            # Compare winner to runner-up
            runner_up = sorted_p[1]
            if len(winner.scores) >= 2 and len(runner_up.scores) >= 2:
                t_stat, p_value = stats.ttest_ind(winner.scores, runner_up.scores)
                print(f"\nStatistical significance vs {runner_up.name}:")
                print(f"  t-statistic: {t_stat:.3f}")
                print(f"  p-value: {p_value:.4f}")
                if p_value < 0.05:
                    print("  Result: SIGNIFICANT (p < 0.05)")
                else:
                    print("  Result: Not significant (p >= 0.05)")
        except ImportError:
            pass

    print(f"\nResults saved to: {output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Stability Tournament')
    parser.add_argument('--mode', type=str, default='quick',
                       choices=['quick', 'full', 'overnight'],
                       help='Tournament mode')
    parser.add_argument('--from-sweep', type=str, default=None,
                       help='Load top configs from sweep results JSON')
    parser.add_argument('--top', type=int, default=5,
                       help='Number of top configs to use from sweep')
    parser.add_argument('--runs', type=int, default=None,
                       help='Override runs per config')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override epochs per run')
    parser.add_argument('--timeout', type=int, default=90,
                       help='Timeout per run in minutes')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory')
    parser.add_argument('--parallel-envs', type=int, default=50,
                       help='Parallel environments (default: 50 for JAX env)')
    parser.add_argument('--no-jax-env', action='store_true',
                       help='Disable JAX env (use CPU envs, set --parallel-envs 4-12)')
    args = parser.parse_args()

    # Set defaults based on mode
    if args.mode == 'quick':
        runs = args.runs or 3
        epochs = args.epochs or 100
    elif args.mode == 'full':
        runs = args.runs or 5
        epochs = args.epochs or 200
    else:  # overnight
        runs = args.runs or 5
        epochs = args.epochs or 300

    # Load configs
    if args.from_sweep:
        configs = load_configs_from_sweep(args.from_sweep, args.top)
        print(f"Loaded {len(configs)} configs from {args.from_sweep}")
    else:
        configs = get_tournament_configs()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"logs/tournament_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save tournament config
    with open(output_dir / "tournament_config.json", 'w') as f:
        json.dump({
            'mode': args.mode,
            'runs_per_config': runs,
            'epochs_per_run': epochs,
            'num_participants': len(configs),
            'configs': [asdict(c) for c in configs],
        }, f, indent=2)

    # Run tournament
    use_jax_env = not args.no_jax_env
    print(f"Environment: {'JAX (GPU)' if use_jax_env else 'CPU'} with {args.parallel_envs} parallel envs")

    participants = run_tournament(
        configs=configs,
        epochs=epochs,
        runs_per_config=runs,
        output_dir=output_dir,
        timeout_minutes=args.timeout,
        parallel_envs=args.parallel_envs,
        use_jax_env=use_jax_env,
    )

    # Print final results
    print_final_results(participants, output_dir)


if __name__ == "__main__":
    main()
