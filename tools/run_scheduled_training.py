#!/usr/bin/env python3
"""
Scheduled Training Runner
Executes training runs defined in training_schedule.json.
"""

import json
import os
import sys
import subprocess
import time
import logging
from datetime import datetime

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [SCHEDULE] - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'scheduled_training.log')),
        logging.StreamHandler()
    ]
)

def setup_cuda_env():
    """Build environment with correct CUDA library paths from pip-installed nvidia packages"""
    env = os.environ.copy()
    try:
        import site
        site_packages = site.getsitepackages()[0]
        nvidia_libs = [
            "cudnn", "cublas", "cuda_nvrtc", "cuda_runtime",
            "cuda_cupti", "cufft", "curand", "cusolver", "cusparse", "nccl"
        ]

        lib_paths = []
        for lib in nvidia_libs:
            lib_path = os.path.join(site_packages, "nvidia", lib, "lib")
            if os.path.exists(lib_path):
                lib_paths.append(lib_path)

        if lib_paths:
            existing = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = ":".join(lib_paths) + (":" + existing if existing else "")
    except Exception:
        pass
    return env

CUDA_ENV = setup_cuda_env()

def run_training_run(period_idx, run_idx, period_data, run_data, db_path, epochs=None, steps=200,
                     last_checkpoint=None, fee=26, slippage=5, sims=16, noise=0.3, noise_frac=0.25,
                     adaptive_calibration=False, value_scale=1.0, model_size='small'):
    """Execute a single training run"""
    
    start_ts = period_data['start']
    end_ts = period_data['end']
    duration_days = period_data['duration_days']
    assets = run_data['assets']
    
    # Determine epochs: use override if provided, else use duration_days
    actual_epochs = epochs if epochs is not None else duration_days
    
    # Generate unique run ID
    run_id = f"p{period_idx}_run{run_idx}"
    
    # --- PADDING LOGIC START ---
    # Ensure we always have 64 assets for consistent model shape
    TARGET_ASSETS = 64
    current_count = len(assets)
    if current_count < TARGET_ASSETS:
        needed = TARGET_ASSETS - current_count
        # Randomly sample existing assets to fill the gap
        # We assume assets list is unique strings
        import random
        padding = random.choices(assets, k=needed)
        assets.extend(padding)
        logging.info(f"⚠️  Padded asset list from {current_count} to {TARGET_ASSETS} using duplicates.")
        logging.info(f"   (Added: {padding})")
    elif current_count > TARGET_ASSETS:
        # Trim if somehow we have more (shouldn't happen with current map logic but safe to handle)
        assets = assets[:TARGET_ASSETS]
        logging.info(f"⚠️  Trimmed asset list from {current_count} to {TARGET_ASSETS}.")
    # --- PADDING LOGIC END ---
    
    logging.info(f"--- Starting Run {run_id} ---")
    logging.info(f"Period: {datetime.fromtimestamp(start_ts)} to {datetime.fromtimestamp(end_ts)} ({duration_days} days)")
    logging.info(f"Assets: {len(assets)} (First 5: {assets[:5]})")
    logging.info(f"Configuration: {actual_epochs} epochs, {steps} steps, Fee: {fee} bps, Slip: {slippage} bps")
    logging.info(f"MCTS: Sims={sims}, Noise={noise}, Frac={noise_frac}")
    logging.info(f"Model: {model_size}, Calibration: adaptive={adaptive_calibration}, scale={value_scale}")
    
    # Create temporary config
    config = {
        'experiment_name': run_id,
        'n_assets': len(assets),
        'asset_symbols': assets,
        'min_coverage': 0.1, # Relaxed for specific schedule
        'start_date': int(start_ts),
        'end_date': int(end_ts),
        'description': f"Scheduled run for period {period_idx}"
    }
    
    config_path = os.path.join('config', f'temp_schedule_{run_id}.json')
    os.makedirs('config', exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # CHECK FOR EXISTING CHECKPOINTS (RESUME LOGIC)
    ckpt_dir = os.path.join('jax_ckpt_fast', run_id)
    if os.path.exists(ckpt_dir):
        checkpoints = [d for d in os.listdir(ckpt_dir) if d.startswith('checkpoint_')]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split('_')[1]))
            latest = os.path.join(ckpt_dir, checkpoints[-1])
            logging.info(f"⏭️  Run {run_id} already completed. Found checkpoint: {latest}")
            logging.info("Skipping execution and using existing checkpoint.")
            return latest

    cmd = [
        "python", "train_jax_fast.py",
        "--config", config_path,
        "--epochs", str(actual_epochs),
        "--steps", str(steps),
        "--mode", "MARGIN",
        "--fee", str(fee),
        "--slippage", str(slippage),
        "--sims", str(sims),
        "--noise", str(noise),
        "--noise-frac", str(noise_frac),
        "--model-size", model_size
    ]

    # Adaptive calibration
    if adaptive_calibration:
        cmd.append("--adaptive-calibration")
    if value_scale != 1.0:
        cmd.extend(["--value-scale", str(value_scale)])

    if last_checkpoint:
        cmd.extend(["--load-checkpoint", last_checkpoint])
    
    logging.info(f"Executing: {' '.join(cmd)}")
    
    start_time = time.time()
    
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=CUDA_ENV
    ) as proc:
        for line in proc.stdout:
            # Filter output to reduce noise but keep progress
            if "Epoch" in line or "Error" in line or "CRITICAL" in line or "Checkpoint" in line:
                print(f"[{run_id}] {line.strip()}")
                
        return_code = proc.wait()
        
    duration = time.time() - start_time
    
    if return_code == 0:
        logging.info(f"Run {run_id} COMPLETED in {duration/60:.1f} minutes.")
        # Clean up temp config
        try:
            os.remove(config_path)
        except:
            pass
            
        # Find latest checkpoint
        ckpt_dir = os.path.join('jax_ckpt_fast', run_id)
        if os.path.exists(ckpt_dir):
            checkpoints = [d for d in os.listdir(ckpt_dir) if d.startswith('checkpoint_')]
            if checkpoints:
                # Sort by step number (checkpoint_100, checkpoint_200)
                checkpoints.sort(key=lambda x: int(x.split('_')[1]))
                latest = os.path.join(ckpt_dir, checkpoints[-1])
                logging.info(f"Found latest checkpoint: {latest}")
                return latest
        
        logging.warning("No checkpoint found for successful run.")
        return None
    else:
        logging.error(f"Run {run_id} FAILED with code {return_code}.")
        return None

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Scheduled Training Runner for Historical Data')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of runs (for testing)')
    parser.add_argument('--epochs', type=int, default=None, help='Epochs per run (default: duration of period)')
    parser.add_argument('--steps', type=int, default=200, help='Steps per episode')
    parser.add_argument('--adaptive-calibration', action='store_true', help='Enable adaptive value calibration')
    parser.add_argument('--value-scale', type=float, default=1.0, help='Initial value calibration scale')
    parser.add_argument('--model-size', default='small', choices=['small', 'medium', 'large'], help='Model size')
    parser.add_argument('--load-checkpoint', help='Initial checkpoint to load for first run')
    parser.add_argument('--period', type=int, default=None, help='Train specific period only (1-indexed)')
    parser.add_argument('--run', type=int, default=None, help='Train specific run within period only')
    parser.add_argument('--start-fee', type=int, default=0, help='Starting fee in bps (0-26)')
    parser.add_argument('--start-slip', type=int, default=0, help='Starting slippage in bps (0-5)')
    args = parser.parse_args()

    if not os.path.exists('training_schedule.json'):
        logging.error("training_schedule.json not found!")
        return
        
    with open('training_schedule.json') as f:
        schedule = json.load(f)
        
    logging.info(f"Loaded schedule with {len(schedule['periods'])} periods.")
    
    total_runs = schedule['summary']['total_unique_runs']
    logging.info(f"Total planned runs: {total_runs}")
    
    db_path = schedule['database']

    runs_completed = 0
    global_run_count = 0
    last_checkpoint = args.load_checkpoint  # Use provided checkpoint or None

    # Adaptive Difficulty State
    current_fee = args.start_fee
    current_slippage = args.start_slip
    target_fee = 26
    target_slippage = 5
    logging.info(f"Starting difficulty: Fee={current_fee}bps, Slip={current_slippage}bps")

    # Maniac Mode Configuration
    current_sims = 32
    current_noise = 0.5
    current_noise_frac = 0.5

    # Filter periods if --period specified
    periods = schedule['periods']
    if args.period is not None:
        if args.period < 1 or args.period > len(periods):
            logging.error(f"Period {args.period} out of range (1-{len(periods)})")
            return
        periods = [(args.period - 1, periods[args.period - 1])]
        logging.info(f"Training only Period {args.period}")
    else:
        periods = list(enumerate(periods))

    for p_idx_offset, period in periods:
        p_idx = p_idx_offset + 1 if args.period is None else args.period
        logging.info(f"=== Processing Period {p_idx}/{len(schedule['periods'])} ===")
        
        runs = period.get('training_runs', [])

        # Filter runs if --run specified
        if args.run is not None:
            if args.run < 1 or args.run > len(runs):
                logging.error(f"Run {args.run} out of range for period {p_idx} (1-{len(runs)})")
                return
            runs = [(args.run, runs[args.run - 1])]
            logging.info(f"Training only Run {args.run}")
        else:
            runs = list(enumerate(runs, start=1))

        for r_idx, run_data in runs:
            if args.limit and global_run_count >= args.limit:
                logging.info(f"Limit of {args.limit} runs reached. Exiting.")
                return

            # Pass all parameters including new calibration options
            latest_ckpt = run_training_run(
                p_idx, r_idx, period, run_data, db_path,
                epochs=args.epochs, steps=args.steps,
                last_checkpoint=last_checkpoint,
                fee=current_fee, slippage=current_slippage,
                sims=current_sims, noise=current_noise, noise_frac=current_noise_frac,
                adaptive_calibration=args.adaptive_calibration,
                value_scale=args.value_scale,
                model_size=args.model_size
            )
            
            if latest_ckpt:
                last_checkpoint = latest_ckpt
                runs_completed += 1
                
                # Update Difficulty Strategy based on performance
                try:
                    with open('last_run_metrics.json', 'r') as f:
                        metrics = json.load(f)
                        avg_reward = metrics.get('avg_reward', 0.0)

                        logging.info(f"📊 Run Metrics: Avg Reward={avg_reward:.2f}%")

                        if avg_reward > 50.0:
                            # TURBO: Model crushing it - jump to max difficulty
                            old_fee, old_slip = current_fee, current_slippage
                            current_fee = target_fee
                            current_slippage = target_slippage
                            logging.info(f"🚀 TURBO! Avg {avg_reward:.1f}% >> Jumping to max: Fee {old_fee}->{current_fee}, Slip {old_slip}->{current_slippage}")

                        elif avg_reward > 20.0:
                            # FAST: Model doing great - aggressive increase
                            old_fee, old_slip = current_fee, current_slippage
                            current_slippage = target_slippage  # Max slip immediately
                            current_fee = min(target_fee, current_fee + 8)  # +8 fee per run
                            logging.info(f"⚡ FAST! Avg {avg_reward:.1f}% >> Fee {old_fee}->{current_fee}, Slip {old_slip}->{current_slippage}")

                        elif avg_reward > 5.0:
                            # NORMAL: Level Up Logic - Slippage first, then Fee
                            if current_slippage < target_slippage:
                                current_slippage = min(target_slippage, current_slippage + 2)
                                logging.info(f"📈 Level Up! Slippage -> {current_slippage} bps")
                            elif current_fee < target_fee:
                                old_fee = current_fee
                                current_fee = min(target_fee, current_fee + 4)
                                logging.info(f"📈 Level Up! Fee {old_fee} -> {current_fee} bps")
                            else:
                                logging.info(f"🏆 Max Difficulty Reached (Fee {current_fee}, Slip {current_slippage})")

                        elif avg_reward > 2.0:
                             # SLOW: Modest gains
                             if current_fee < target_fee:
                                 current_fee = min(target_fee, current_fee + 2)
                                 logging.info(f"📈 Slow Progress. Fee -> {current_fee} bps")
                        else:
                            logging.info(f"🐢 Reward low ({avg_reward:.2f}%), maintaining Fee={current_fee}, Slip={current_slippage}")

                except Exception as e:
                    logging.warning(f"Could not load metrics to adjust difficulty: {e}")
                
            else:
                logging.warning(f"Run {p_idx}-{r_idx} failed or produced no checkpoint. Continuing without updating last_checkpoint...")
                
            global_run_count += 1
            
            # Optional cooldown
            time.sleep(2)
            
    logging.info(f"All scheduled runs finished. Completed: {runs_completed}/{total_runs}")

if __name__ == "__main__":
    main()
