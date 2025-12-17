import psutil
import logging
import sys

def check_memory_safety(config):
    """
    Validates that the training configuration won't exhaust system RAM.
    Raises RuntimeError if unsafe.
    """
    print("="*60, flush=True)
    print("Memory Safety Check", flush=True)
    print("="*60, flush=True)
    
    # 1. Get system memory
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024**3)
    available_gb = mem.available / (1024**3)
    
    print(f"System RAM:              {total_gb:.1f} GB")
    print(f"Available RAM:           {available_gb:.1f} GB")
    
    # 2. Calculate expected usage
    # Replay buffer: buffer_size * segment_length * obs_size
    # Obs size (approx):
    # - Image: 27 * 64 * 64 * 4 bytes (float32) = 442 KB
    # - Action/Reward/Etc: ~1 KB
    # - Total per step: ~450 KB
    
    # Config values
    try:
        buffer_size = config.policy.replay_buffer_size
        segment_len = config.policy.game_segment_length
    except AttributeError:
        # Fallback if structure is different
        buffer_size = config.get('replay_buffer_size', 10000)
        segment_len = config.get('game_segment_length', 200)

    print(f"Replay buffer config:    {buffer_size} segments × {segment_len} steps")
    
    kb_per_step = 450
    mb_per_segment = (segment_len * kb_per_step) / 1024
    buffer_gb = (buffer_size * mb_per_segment) / 1024
    
    print(f"Expected replay buffer:  {buffer_gb:.1f} GB")
    
    # Overhead (Python processes, CUDA context, Model)
    # 4 collectors * 500MB + Learner * 2GB ~ 4GB
    overhead_gb = 4.0
    print(f"Process overhead:        {overhead_gb:.1f} GB")
    
    total_expected_gb = buffer_gb + overhead_gb
    print(f"Total expected usage:    {total_expected_gb:.1f} GB")
    
    # 3. Validation
    # Safety threshold: 80% of available RAM
    safe_limit_gb = available_gb * 0.8
    
    if total_expected_gb > safe_limit_gb:
        msg = (
            f"\n❌ MEMORY CONFIG UNSAFE!\n"
            f"Expected usage ({total_expected_gb:.1f} GB) exceeds safe limit ({safe_limit_gb:.1f} GB).\n"
            f"You MUST reduce replay_buffer_size or game_segment_length.\n"
            f"Current: size={buffer_size}, len={segment_len}"
        )
        print(msg)
        print("="*60)
        raise RuntimeError(msg)
        
    print(f"✓ PASSED: Config is safe ({total_expected_gb:.1f}GB < {safe_limit_gb:.1f}GB)")
    print("="*60)