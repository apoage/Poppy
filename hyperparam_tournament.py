#!/usr/bin/env python3
"""
Hyperparameter Tournament - Successive Halving Search

Tournament structure:
  Round 1: All configs on L0 → top 50% advance
  Round 2: Survivors on L1 → top 50% advance
  Round 3: Survivors on L3 → top 50% advance
  Round 4: Survivors on L4 → top 50% advance
  Round 5: Survivors on real data → user decides

Each round runs short training (configurable epochs).
Configs are ranked by reward, top half advances.
"""

import os
import sys
import json
import subprocess
import itertools
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import csv

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TournamentConfig:
    """Tournament settings."""
    # Epochs per round (can increase for later rounds)
    epochs_per_round: Dict[int, int] = field(default_factory=lambda: {
        0: 50,   # L0: quick filter
        1: 75,   # L1: slightly more
        3: 100,  # L3: more serious
        4: 100,  # L4: full test
        -1: 150  # Real data
    })

    # Steps per episode
    steps: int = 100

    # Survival rate per round
    survival_rate: float = 0.5

    # Output directory
    output_dir: Path = Path("tournament_results")

    # Parallel runs (set based on GPU memory)
    parallel_runs: int = 1


@dataclass
class HyperparamConfig:
    """Single hyperparameter configuration."""
    name: str
    model_size: str = "small"
    lr: float = 3e-4
    sims: int = 32
    batch_size: int = 256
    h_cycles: int = 4
    l_cycles: int = 6
    use_prediction_loss: bool = False
    prediction_loss_weight: float = 0.5
    gumbel_scale: float = 3.0
    discount: float = 0.99
    # Always use adaptive entropy, not static
    adaptive_entropy: bool = True

    def to_args(self) -> List[str]:
        """Convert to command line arguments."""
        args = [
            f"--model-size={self.model_size}",
            f"--lr={self.lr}",
            f"--sims={self.sims}",
            f"--batch-size={self.batch_size}",
            f"--h-cycles={self.h_cycles}",
            f"--l-cycles={self.l_cycles}",
            f"--gumbel-scale={self.gumbel_scale}",
            f"--discount={self.discount}",
        ]
        # Use adaptive entropy (not static entropy_coef)
        if self.adaptive_entropy:
            args.append("--adaptive-entropy")
        if self.use_prediction_loss:
            args.append("--use-prediction-loss")
            args.append(f"--prediction-loss-weight={self.prediction_loss_weight}")
        return args


# =============================================================================
# Hyperparameter Grid
# =============================================================================

def generate_configs() -> List[HyperparamConfig]:
    """Generate all hyperparameter combinations to test."""

    # Define search space (no static entropy - we use adaptive)
    grid = {
        "model_size": ["small", "medium"],
        "lr": [1e-4, 3e-4],
        "sims": [16, 32],
        "batch_size": [128, 256],
        "h_cycles": [2, 4],
        "l_cycles": [4, 6],
        "use_prediction_loss": [False, True],
        "prediction_loss_weight": [0.3, 1.0],  # Only used if prediction enabled
    }

    configs = []
    config_id = 0

    # Generate combinations
    keys = list(grid.keys())
    for values in itertools.product(*[grid[k] for k in keys]):
        params = dict(zip(keys, values))

        # Skip prediction_loss_weight variations if prediction disabled
        if not params["use_prediction_loss"] and params["prediction_loss_weight"] != 0.3:
            continue

        config_id += 1
        name = f"config_{config_id:03d}"

        configs.append(HyperparamConfig(name=name, **params))

    return configs


def generate_focused_configs() -> List[HyperparamConfig]:
    """Generate focused set of 48 configs for tournament.

    Key variables (2×2×2×3×2 = 48):
    - sims: [16, 32] - MCTS depth vs exploration
    - lr: [1e-4, 3e-4] - Learning rate
    - h_cycles: [2, 4] - TRM reasoning depth
    - pred_weight: [0.0, 0.3, 1.0] - Physics prediction weight
    - gumbel_scale: [1.5, 3.0] - MCTS exploration temperature

    Uses adaptive entropy (not static).
    ~20 hours total for L0 @ 50 epochs each.

    Winners then tested on scaling (medium/large model, bigger batch).
    """

    configs = []

    # Key variables to sweep (2×2×2×3×2 = 48)
    sims_options = [16, 32]
    lr_options = [1e-4, 3e-4]
    h_cycles_options = [2, 4]
    pred_weight_options = [0.0, 0.3, 1.0]  # 0.0 = disabled
    gumbel_options = [1.5, 3.0]  # MCTS exploration

    config_id = 0
    for sims in sims_options:
        for lr in lr_options:
            for h_cycles in h_cycles_options:
                for pred_weight in pred_weight_options:
                    for gumbel in gumbel_options:
                        config_id += 1
                        use_pred = pred_weight > 0

                        # Generate descriptive name
                        pred_str = f"p{int(pred_weight*10)}" if use_pred else "np"
                        lr_str = "10" if lr == 1e-4 else "30"  # 1e-4 or 3e-4
                        gumbel_str = "g15" if gumbel == 1.5 else "g30"
                        name = f"s{sims}_lr{lr_str}_h{h_cycles}_{pred_str}_{gumbel_str}"

                        configs.append(HyperparamConfig(
                            name=name,
                            model_size="small",
                            sims=sims,
                            lr=lr,
                            h_cycles=h_cycles,
                            l_cycles=h_cycles * 2,  # L = 2*H
                            batch_size=256,
                            use_prediction_loss=use_pred,
                            prediction_loss_weight=pred_weight if use_pred else 0.5,
                            gumbel_scale=gumbel,
                            # Always use adaptive entropy
                            adaptive_entropy=True,
                        ))

    return configs


def generate_scaling_configs(base_config: HyperparamConfig) -> List[HyperparamConfig]:
    """Generate scaling variants of a winning config.

    Tests the same hyperparams with:
    - medium model (96-dim state, ~15M params)
    - large model (160-dim state, ~36M params)
    - larger batch sizes
    """
    from dataclasses import replace

    scaling_configs = []

    # Medium model variants
    scaling_configs.append(replace(
        base_config,
        name=f"{base_config.name}_medium",
        model_size="medium",
        batch_size=256,
    ))
    scaling_configs.append(replace(
        base_config,
        name=f"{base_config.name}_medium_b512",
        model_size="medium",
        batch_size=512,
    ))

    # Large model variants
    scaling_configs.append(replace(
        base_config,
        name=f"{base_config.name}_large",
        model_size="large",
        batch_size=256,
    ))
    scaling_configs.append(replace(
        base_config,
        name=f"{base_config.name}_large_b512",
        model_size="large",
        batch_size=512,
    ))

    return scaling_configs


# =============================================================================
# Tournament Runner
# =============================================================================

class Tournament:
    """Manages the hyperparameter tournament."""

    LEVELS = [0, 1, 3, 4, -1]  # -1 = real data

    def __init__(self, configs: List[HyperparamConfig], tournament_config: TournamentConfig):
        self.configs = configs
        self.tc = tournament_config
        self.results: Dict[str, Dict] = {}  # config_name -> results
        self.survivors: List[str] = [c.name for c in configs]
        self.current_round = 0

        # Create output directory
        self.tc.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.tc.output_dir / f"tournament_{self.run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Save initial configs
        self._save_configs()

    def _save_configs(self):
        """Save all configs to JSON."""
        config_data = {c.name: asdict(c) for c in self.configs}
        with open(self.run_dir / "configs.json", "w") as f:
            json.dump(config_data, f, indent=2)

    def _get_config(self, name: str) -> HyperparamConfig:
        """Get config by name."""
        for c in self.configs:
            if c.name == name:
                return c
        raise ValueError(f"Config not found: {name}")

    def run_single(self, config: HyperparamConfig, level: int, epochs: int) -> Dict:
        """Run a single training configuration."""

        # Build command - use tournament subdir for checkpoints
        tournament_ckpt_dir = self.run_dir / "checkpoints"
        tournament_ckpt_dir.mkdir(exist_ok=True)

        run_name = f"tourn_{self.run_id}_{config.name}_L{level}"

        cmd = [
            "python", "-u", "train_jax_fast.py",  # -u for unbuffered stdout
            f"--epochs={epochs}",
            f"--steps={self.tc.steps}",
            f"--name={run_name}",
        ]

        # Add level-specific args
        if level >= 0:
            cmd.append(f"--synthetic-level={level}")
        else:
            # Real data
            cmd.extend([
                "--db-path=trading_data.db",
                "--fee=16",
                "--slippage=5"
            ])

        # Add hyperparameter args
        cmd.extend(config.to_args())

        # Load from previous checkpoint if not first round
        if self.current_round > 0:
            prev_level = self.LEVELS[self.current_round - 1]
            prev_name = f"tourn_{self.run_id}_{config.name}_L{prev_level}"
            prev_ckpt = Path(f"jax_ckpt_fast/{prev_name}")
            if prev_ckpt.exists():
                cmd.extend([
                    f"--load-checkpoint={prev_ckpt}",
                    "--clear-buffer",
                    "--reset-optimizer"
                ])
            else:
                print(f"  WARNING: No checkpoint found at {prev_ckpt}")

        print(f"\n{'='*60}")
        print(f"Running: {config.name} on L{level}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}\n")

        # Run training with real-time output
        try:
            # Use Popen to stream output while capturing it
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
            )

            stdout_lines = []
            for line in process.stdout:
                print(line, end='')  # Real-time output
                stdout_lines.append(line)

            process.wait(timeout=3600)  # 1 hour max
            stdout = ''.join(stdout_lines)

            # Parse results from output
            metrics = self._parse_output(stdout, "")
            metrics["config_name"] = config.name
            metrics["level"] = level
            metrics["round"] = self.current_round
            metrics["success"] = process.returncode == 0

            if not metrics["success"]:
                print(f"\n>>> FAILED: {config.name} (exit code {process.returncode})")

            return metrics

        except subprocess.TimeoutExpired:
            process.kill()
            return {
                "config_name": config.name,
                "level": level,
                "round": self.current_round,
                "success": False,
                "error": "timeout"
            }
        except Exception as e:
            return {
                "config_name": config.name,
                "level": level,
                "round": self.current_round,
                "success": False,
                "error": str(e)
            }

    def _parse_output(self, stdout: str, stderr: str) -> Dict:
        """Parse training output to extract metrics."""
        metrics = {
            "avg_reward": None,
            "recent_reward": None,  # MA50
            "final_loss": None,
            "final_val_loss": None,
            "final_entropy": None,
            "hold_ratio": None,
            "activity": None,  # 1 - hold_ratio
            "epoch_time": None,  # Learning speed
        }

        # Track all epoch times for averaging
        epoch_times = []

        # Look for final metrics in output
        for line in stdout.split("\n"):
            # Reward metrics
            if "avg_reward=" in line:
                try:
                    metrics["avg_reward"] = float(line.split("avg_reward=")[1].split(",")[0])
                except:
                    pass
            if "recent_50_avg=" in line:
                try:
                    metrics["recent_reward"] = float(line.split("recent_50_avg=")[1].split(",")[0])
                except:
                    pass

            # Loss and entropy
            if "Loss:" in line and "Val:" in line:
                try:
                    parts = line.split("|")
                    for p in parts:
                        if "Loss:" in p:
                            metrics["final_loss"] = float(p.split(":")[1].strip())
                        if "Val:" in p:
                            metrics["final_val_loss"] = float(p.split(":")[1].strip().split()[0])
                        if "Ent:" in p:
                            metrics["final_entropy"] = float(p.split(":")[1].strip())
                except:
                    pass

            # Action distribution format: "S/H/B/LS/LB: 8/3/3/5/1"
            if "S/H/B/LS/LB:" in line:
                try:
                    parts = line.split("S/H/B/LS/LB:")[1].strip().split("|")[0].strip()
                    counts = [int(x) for x in parts.split("/")]
                    if len(counts) >= 5:
                        total = sum(counts)
                        if total > 0:
                            hold_count = counts[1]  # H is second
                            metrics["hold_ratio"] = hold_count / total
                            metrics["activity"] = 1.0 - metrics["hold_ratio"]
                except:
                    pass

            # Epoch timing: "Self-play:   2.0s"
            if "Self-play:" in line:
                try:
                    import re
                    time_match = re.search(r'Self-play:\s*(\d+\.?\d*)\s*s', line)
                    if time_match:
                        epoch_times.append(float(time_match.group(1)))
                except:
                    pass

        # Average epoch time
        if epoch_times:
            metrics["epoch_time"] = sum(epoch_times) / len(epoch_times)

        return metrics

    def run_round(self, level: int):
        """Run all surviving configs on a level."""
        print(f"\n{'#'*60}")
        print(f"# ROUND {self.current_round}: Level {level}")
        print(f"# Configs: {len(self.survivors)}")
        print(f"{'#'*60}\n")

        epochs = self.tc.epochs_per_round.get(level, 50)
        round_results = []

        for config_name in self.survivors:
            config = self._get_config(config_name)
            result = self.run_single(config, level, epochs)
            round_results.append(result)

            # Save intermediate results
            self._save_round_results(round_results)

        # Update global results
        for r in round_results:
            if r["config_name"] not in self.results:
                self.results[r["config_name"]] = {}
            self.results[r["config_name"]][f"L{level}"] = r

        return round_results

    def _save_round_results(self, results: List[Dict]):
        """Save current round results to CSV."""
        if not results:
            return

        csv_path = self.run_dir / f"round_{self.current_round}_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    def select_survivors(self, results: List[Dict]) -> List[str]:
        """Select top performers to advance."""
        # Filter successful runs
        successful = [r for r in results if r.get("success", False)]

        if not successful:
            print("WARNING: No successful runs in this round!")
            return []

        # Composite scoring:
        # - Primary: recent_reward (MA50) or avg_reward
        # - Secondary: activity (penalize excessive HOLD)
        # - Tertiary: epoch_time (prefer faster training)
        def get_score(r):
            reward = r.get("recent_reward") or r.get("avg_reward") or 0.0
            activity = r.get("activity") or 0.5  # Default 50% activity
            epoch_time = r.get("epoch_time") or 15.0  # Default 15s

            # Penalize HOLD-dominant strategies (activity < 20%)
            activity_penalty = 0.0 if activity >= 0.2 else (0.2 - activity) * 10.0

            # Normalize time (faster is better, but not weighted heavily)
            time_bonus = max(0, (20.0 - epoch_time) / 20.0) * 0.1

            return reward - activity_penalty + time_bonus

        sorted_results = sorted(successful, key=get_score, reverse=True)

        # Take top half
        n_survivors = max(1, int(len(sorted_results) * self.tc.survival_rate))
        survivors = [r["config_name"] for r in sorted_results[:n_survivors]]

        print(f"\nSurvivors ({n_survivors}/{len(results)}):")
        for i, r in enumerate(sorted_results[:n_survivors]):
            score = get_score(r)
            reward = r.get("recent_reward") or r.get("avg_reward") or 0.0
            hold = r.get("hold_ratio")
            hold_str = f"hold={hold:.0%}" if hold is not None else "hold=?"
            print(f"  {i+1}. {r['config_name']}: score={score:.2f}, reward={reward:.2f}, {hold_str}")

        print(f"\nEliminated:")
        for r in sorted_results[n_survivors:]:
            score = get_score(r)
            reward = r.get("recent_reward") or r.get("avg_reward") or 0.0
            hold = r.get("hold_ratio")
            hold_str = f"hold={hold:.0%}" if hold is not None else "hold=?"
            print(f"  - {r['config_name']}: score={score:.2f}, reward={reward:.2f}, {hold_str}")

        return survivors

    def run_tournament(self, start_round: int = 0, skip_scaling: bool = False):
        """Run the full tournament."""
        self.current_round = start_round

        for level in self.LEVELS[start_round:]:
            if not self.survivors:
                print("No survivors left! Tournament ended early.")
                break

            # Run round
            results = self.run_round(level)

            # Select survivors (except last round)
            if level != self.LEVELS[-1]:
                self.survivors = self.select_survivors(results)
                self.current_round += 1
            else:
                # Final round - show all results
                print("\n" + "="*60)
                print("TOURNAMENT COMPLETE - FINAL RESULTS")
                print("="*60)
                self._print_final_results(results)

        # Save intermediate summary
        self._save_summary()

        # Run scaling tests on winners
        if not skip_scaling and self.survivors:
            self.run_scaling_tests()

        return self.survivors

    def run_scaling_tests(self):
        """Test winning configs with larger models and batch sizes."""
        print("\n" + "#"*60)
        print("# SCALING TESTS - Testing winners with larger models")
        print("#"*60)

        scaling_results = []

        for config_name in self.survivors[:3]:  # Top 3 winners
            base_config = self._get_config(config_name)
            scaling_configs = generate_scaling_configs(base_config)

            print(f"\n>>> Testing scaling variants of: {config_name}")

            for scale_config in scaling_configs:
                # Add to our config list for tracking
                self.configs.append(scale_config)

                # Run with more epochs for scaling test
                result = self.run_single(
                    scale_config,
                    level=-1,  # Real data
                    epochs=self.tc.epochs_per_round.get(-1, 150)
                )
                scaling_results.append(result)

        # Save scaling results
        self._save_scaling_results(scaling_results)

        # Print scaling summary
        print("\n" + "="*60)
        print("SCALING TEST RESULTS")
        print("="*60)

        successful = [r for r in scaling_results if r.get("success", False)]
        sorted_results = sorted(
            successful,
            key=lambda r: r.get("recent_reward") or r.get("avg_reward") or float("-inf"),
            reverse=True
        )

        for i, r in enumerate(sorted_results):
            reward = r.get('recent_reward') or r.get('avg_reward') or 0.0
            hold = r.get('hold_ratio')
            epoch_time = r.get('epoch_time')

            print(f"\n{i+1}. {r['config_name']}")
            print(f"   Reward (MA50): {reward:.2f}")
            if hold is not None:
                print(f"   Hold Ratio: {hold:.0%}")
            if epoch_time:
                print(f"   Epoch Time: {epoch_time:.1f}s")

    def _save_scaling_results(self, results: List[Dict]):
        """Save scaling test results."""
        if not results:
            return

        csv_path = self.run_dir / "scaling_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        print(f"\nScaling results saved to: {csv_path}")

    def _print_final_results(self, results: List[Dict]):
        """Print final tournament results."""
        successful = [r for r in results if r.get("success", False)]
        sorted_results = sorted(
            successful,
            key=lambda r: r.get("recent_reward") or r.get("avg_reward") or float("-inf"),
            reverse=True
        )

        for i, r in enumerate(sorted_results):
            config = self._get_config(r["config_name"])
            reward = r.get('recent_reward') or r.get('avg_reward') or 0.0
            hold = r.get('hold_ratio')
            epoch_time = r.get('epoch_time')

            print(f"\n{i+1}. {r['config_name']}")
            print(f"   Reward (MA50): {reward:.2f}")
            if hold is not None:
                print(f"   Hold Ratio: {hold:.0%} (Activity: {1-hold:.0%})")
            if r.get('final_loss'):
                print(f"   Loss: {r.get('final_loss'):.4f}")
            if epoch_time:
                print(f"   Avg Epoch Time: {epoch_time:.1f}s")
            print(f"   Config: size={config.model_size}, sims={config.sims}, lr={config.lr}")
            pred_str = f"pred_weight={config.prediction_loss_weight}" if config.use_prediction_loss else "no_pred"
            print(f"           h={config.h_cycles}, l={config.l_cycles}, {pred_str}")

    def _save_summary(self):
        """Save tournament summary."""
        summary = {
            "run_id": self.run_id,
            "total_configs": len(self.configs),
            "final_survivors": self.survivors,
            "results": self.results
        }

        with open(self.run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\nResults saved to: {self.run_dir}")


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Hyperparameter Tournament")
    parser.add_argument("--mode", choices=["full", "focused", "custom"], default="focused",
                       help="Config generation mode")
    parser.add_argument("--start-round", type=int, default=0,
                       help="Start from this round (for resuming)")
    parser.add_argument("--epochs-multiplier", type=float, default=1.0,
                       help="Multiply default epochs by this factor")
    parser.add_argument("--dry-run", action="store_true",
                       help="Just print configs, don't run")
    parser.add_argument("--list-configs", action="store_true",
                       help="List all configs and exit")
    parser.add_argument("--skip-scaling", action="store_true",
                       help="Skip scaling tests on winners")

    args = parser.parse_args()

    # Generate configs
    if args.mode == "full":
        configs = generate_configs()
    elif args.mode == "focused":
        configs = generate_focused_configs()
    else:
        # Custom - edit this for specific tests
        configs = [
            HyperparamConfig(name="test1", model_size="small", sims=16),
            HyperparamConfig(name="test2", model_size="small", sims=32),
        ]

    print(f"Generated {len(configs)} configurations")

    if args.list_configs:
        for c in configs:
            print(f"\n{c.name}:")
            for k, v in asdict(c).items():
                if k != "name":
                    print(f"  {k}: {v}")
        return

    # Create tournament config
    tc = TournamentConfig()
    if args.epochs_multiplier != 1.0:
        tc.epochs_per_round = {k: int(v * args.epochs_multiplier) for k, v in tc.epochs_per_round.items()}

    if args.dry_run:
        print("\nDry run - would test these configs:")
        for c in configs:
            print(f"  {c.name}: {c.model_size}, sims={c.sims}, lr={c.lr}")
        print(f"\nEpochs per round: {tc.epochs_per_round}")
        return

    # Run tournament
    tournament = Tournament(configs, tc)
    winners = tournament.run_tournament(
        start_round=args.start_round,
        skip_scaling=args.skip_scaling
    )

    print(f"\n{'='*60}")
    print(f"WINNERS: {winners}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
