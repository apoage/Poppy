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

def run_training_run(period_idx, run_idx, period_data, run_data, db_path, epochs=None, steps=200, last_checkpoint=None, fee=26, slippage=5, sims=16, noise=0.3, noise_frac=0.25):
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
        "--noise-frac", str(noise_frac)
    ]

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None, help='Limit number of runs (for testing)')
    parser.add_argument('--epochs', type=int, default=None, help='Epochs per run (default: duration of period)')
    parser.add_argument('--steps', type=int, default=200, help='Steps per episode')
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
    last_checkpoint = None
    
    # Adaptive Difficulty State
    current_fee = 0 
    current_slippage = 0 # Start frictionless
    target_fee = 26
    target_slippage = 5
    
    # Maniac Mode Configuration
    current_sims = 32
    current_noise = 0.5
    current_noise_frac = 0.5
    
    for p_idx, period in enumerate(schedule['periods'], start=1):
        logging.info(f"=== Processing Period {p_idx}/{len(schedule['periods'])} ===")
        
        runs = period.get('training_runs', [])
        for r_idx, run_data in enumerate(runs, start=1):
            if args.limit and global_run_count >= args.limit:
                logging.info(f"Limit of {args.limit} runs reached. Exiting.")
                return

            # Pass all parameters
            latest_ckpt = run_training_run(
                p_idx, r_idx, period, run_data, db_path, 
                epochs=args.epochs, steps=args.steps, 
                last_checkpoint=last_checkpoint, 
                fee=current_fee, slippage=current_slippage,
                sims=current_sims, noise=current_noise, noise_frac=current_noise_frac
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
                        
                        if avg_reward > 5.0:
                            # Level Up Logic: Slippage first, then Fee
                            if current_slippage < target_slippage:
                                current_slippage += 1
                                logging.info(f"📈 Level Up! Slippage increased: {current_slippage-1} -> {current_slippage} bps")
                            elif current_fee < target_fee:
                                old_fee = current_fee
                                current_fee = min(target_fee, current_fee + 2)
                                logging.info(f"📈 Level Up! Fee increased: {old_fee} -> {current_fee} bps")
                            else:
                                logging.info(f"🏆 Max Difficulty Reached (Fee {current_fee}, Slip {current_slippage})")
                                
                        elif avg_reward > 2.0:
                             # Slower increase
                             if current_slippage < target_slippage:
                                 pass # Hold slippage
                             elif current_fee < target_fee:
                                 current_fee = min(target_fee, current_fee + 1)
                                 logging.info(f"📈 Slow Progress. Fee increased: {current_fee-1} -> {current_fee} bps")
                        else:
                            logging.info(f"🐢 Reward low ({avg_reward:.2f}%), maintaining difficulty (Fee {current_fee}, Slip {current_slippage}).")
                            
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
