#!/usr/bin/env python3
"""
Diagnostic script to check why GPU MCTS is not working
Run this to identify the exact issue
"""

import sys
import torch

print("="*80)
print("GPU MCTS Diagnostic Tool")
print("="*80)

# Test 1: CUDA availability
print("\n[Test 1] CUDA Availability")
cuda_available = torch.cuda.is_available()
print(f"  torch.cuda.is_available(): {cuda_available}")
if cuda_available:
    print(f"  GPU: {torch.cuda.get_device_properties(0).name}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  ✓ CUDA OK")
else:
    print(f"  ✗ CUDA NOT AVAILABLE")
    sys.exit(1)

# Test 2: LightZero installation
print("\n[Test 2] LightZero Installation")
try:
    import lzero
    print(f"  LightZero version: {lzero.__version__ if hasattr(lzero, '__version__') else 'unknown'}")
    print(f"  ✓ LightZero installed")
except ImportError as e:
    print(f"  ✗ LightZero not found: {e}")
    sys.exit(1)

# Test 3: C++ MCTS tree module
print("\n[Test 3] C++ MCTS Tree Module")
try:
    from lzero.mcts.tree_search import MuZeroMCTSCtree
    print(f"  ✓ C++ MCTS module found (MuZeroMCTSCtree)")
except ImportError as e:
    print(f"  ✗ C++ MCTS module not found: {e}")
# Test 4: MCTS C++ extension attributes
print("\n[Test 4] MCTS Extension Functions")
try:
    from lzero.mcts.tree_search.mcts_ctree import MCTS
    print(f"  MCTS class: {MCTS}")
    print(f"  ✓ MCTS class available")

    # Check for GPU-specific functions
    if hasattr(MCTS, 'run_gpu'):
        print(f"  ✓ GPU mode detected (run_gpu method exists)")
    else:
        print(f"  ⚠ GPU mode uncertain (no run_gpu method)")

except Exception as e:
    print(f"  ✗ MCTS class error: {e}")

# Test 5: Try to create MCTS tree with GPU
print("\n[Test 5] GPU MCTS Instantiation Test")
try:
    from lzero.mcts.tree_search.mcts_ctree_muzero import MuZeroMCTSCtree
    from easydict import EasyDict

    # Create minimal config
    config = EasyDict({
        'num_simulations': 10,
        'max_num_considered_actions': 192,
        'pb_c_base': 19652,
        'pb_c_init': 1.25,
        'root_dirichlet_alpha': 0.3,
        'root_noise_weight': 0.25,
        'discount_factor': 0.997,
    })

    print(f"  Creating MuZeroMCTSCtree with config...")
    mcts = MuZeroMCTSCtree(config)
    print(f"  ✓ MCTS tree created successfully")

    # Check if it has GPU support
    if hasattr(mcts, 'device'):
        print(f"  Device: {mcts.device}")
    else:
        print(f"  ⚠ No device attribute (may be CPU-only)")

except Exception as e:
    print(f"  ✗ MCTS instantiation failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Check config
print("\n[Test 6] Current Config Check")
try:
    sys.path.insert(0, '/home/lukas/Trader')
    from mcts_trader.config.trading_config import main_config

    print(f"  mcts_ctree (mcts): {main_config.policy.mcts.mcts_ctree}")
    print(f"  mcts_ctree (policy): {main_config.policy.mcts_ctree}")
    print(f"  use_rpc_wrapper: {main_config.policy.use_rpc_wrapper}")
    print(f"  cuda: {main_config.policy.cuda}")

    if main_config.policy.mcts.mcts_ctree and main_config.policy.mcts_ctree:
        print(f"  ✓ Config looks correct for GPU MCTS")
    else:
        print(f"  ✗ Config issue detected")

except Exception as e:
    print(f"  ⚠ Could not load config: {e}")

# Test 7: Runtime GPU MCTS check
print("\n[Test 7] Runtime GPU Usage Check")
try:
    import subprocess
    result = subprocess.run(
        ['nvidia-smi', 'pmon', '-c', '1'],
        capture_output=True,
        text=True,
        timeout=2
    )

    if 'python' in result.stdout.lower():
        print(f"  ✓ Python process using GPU detected")
    else:
        print(f"  ⚠ No Python GPU usage detected (may need training to be running)")

except Exception as e:
    print(f"  ⚠ Could not check GPU usage: {e}")

# Summary
print("\n" + "="*80)
print("DIAGNOSIS SUMMARY")
print("="*80)

print("""
If all tests passed (✓):
  → GPU MCTS is available
  → Issue may be runtime/config related
  → Check if training actually uses GPU (watch nvidia-smi while training)

If Test 3 failed (✗):
  → C++ MCTS extension not compiled with GPU support
  → Solution: Reinstall LightZero from source with CUDA
  → OR: Use CPU MCTS fallback (slower)

If tests are mixed:
  → Run: python profile_training.py
  → Check GPU utilization during training
  → If GPU < 50%, GPU MCTS is not working

Next steps:
  1. If GPU MCTS available: python main.py (should work)
  2. If not available: Edit config to use CPU MCTS (see QUICK_FIX_BOTTLENECK.md)
  3. Run test suite: ./TEST_SMALL_HARDWARE.sh
""")

print("="*80)
