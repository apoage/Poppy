# TASK_148: Google Colab/TPU Compatibility

**Status:** PLANNED

**Created:** 2026-02-10

**Note:** This plan is based on audit from 2026-02-10. Re-audit recommended before implementation as Colab environment and codebase may have changed.

---

## Executive Summary

The codebase is optimized for local NVIDIA CUDA hardware. Migration to Google Colab requires fixes across 4 categories:
1. Device-specific configuration (CRITICAL)
2. Path/filesystem assumptions (CRITICAL)
3. TPU-incompatible operations (HIGH)
4. Memory/performance tuning (MEDIUM)

---

## Audit Findings Summary

### Critical Blockers (12 issues)

| ID | Issue | Files | Severity |
|----|-------|-------|----------|
| C1 | GPU-only XLA flags | `train_easy.py:30-33`, `train_jax_fast.py:19-22` | CRITICAL |
| C2 | CUDA malloc allocator | `train_easy.py:31`, `train_jax_fast.py:20` | CRITICAL |
| C3 | Hardcoded local paths | Multiple files | CRITICAL |
| C4 | Venv python path | `e-dojo_v1.py:819` | CRITICAL |
| C5 | `jax.lax.top_k` on TPU | `path_solver.py:117,297,409`, `alpha_reward.py:39,116,174` | CRITICAL |
| C6 | Subprocess multi-process | `e-dojo_v1.py:814-843` | CRITICAL |
| C7 | JAX cache directories | `train_easy.py:24-28` | CRITICAL |
| C8 | Bella checkpoint path | `train_easy.py:817-819` | CRITICAL |
| C9 | CSV log directory | `train_easy.py:976-978` | HIGH |
| C10 | Dashboard port binding | `DSe/server.py:561` | HIGH |
| C11 | pgrep process lookup | `DSe/server.py:169` | HIGH |
| C12 | Float32 vs bfloat16 | `trm_flax.py:250` | HIGH |

### High Priority (8 issues)

| ID | Issue | Files |
|----|-------|-------|
| H1 | Default batch_size=8 too large | `streaming.py:15` |
| H2 | seq_length=1440 memory pressure | `generator.py:38` |
| H3 | BiMamba parallel mode 2x memory | `bimamba.py:27` |
| H4 | Mamba d_state=16 expensive | `mamba_flow.py:67-68` |
| H5 | No XLA cache clearing | `generator.py:88` |
| H6 | Path solver max_len=512 mismatch | `path_solver.py:35` |
| H7 | PyTorch cu121 vs Colab cu122 | `requirements.txt` |
| H8 | File handle leaks | `e-dojo_v1.py:928,937` |

### Medium Priority (6 issues)

| ID | Issue | Files |
|----|-------|-------|
| M1 | No Colab environment detection | All |
| M2 | `nn.remat` behavior differs on TPU | `mamba_flow.py:120,139,307` |
| M3 | `mode='edge'` padding inefficient | `mamba_block.py:27` |
| M4 | Signal handling race conditions | `e-dojo_v1.py:1395-1403` |
| M5 | VRAM fraction over-subscription | `e-dojo_v1.py:916-918` |
| M6 | No Drive mount validation | `train_easy.py` |

---

## Implementation Plan

### Phase 1: Environment Detection (Foundation)

**Goal:** Add Colab/TPU detection to enable conditional configuration.

#### 1.1 Create `utils/device_utils.py`

```python
"""Device detection utilities for Colab/TPU compatibility."""
import os
import sys
import jax

def is_colab() -> bool:
    """Detect Google Colab environment."""
    return 'COLAB_RELEASE_TAG' in os.environ or 'google.colab' in sys.modules

def is_tpu() -> bool:
    """Detect TPU backend."""
    return jax.default_backend() == 'tpu'

def is_gpu() -> bool:
    """Detect GPU backend."""
    return jax.default_backend() == 'gpu'

def get_python_executable() -> str:
    """Get current Python executable (works in Colab)."""
    return sys.executable

def get_default_checkpoint_dir() -> str:
    """Get appropriate checkpoint directory for environment."""
    if is_colab():
        # Check if Drive is mounted
        if os.path.exists('/content/drive/MyDrive'):
            return '/content/drive/MyDrive/trader_checkpoints'
        else:
            return '/content/checkpoints'  # Ephemeral
    else:
        return 'jax_ckpt_easy'

def get_cache_dir() -> str:
    """Get appropriate cache directory."""
    if is_colab():
        return '/tmp/.jax_cache'
    else:
        import pathlib
        return str(pathlib.Path(__file__).parent.parent / '.jax_cache')
```

#### 1.2 Update imports in main files

Add to `train_easy.py`, `train_jax_fast.py`:
```python
from utils.device_utils import is_colab, is_tpu, get_cache_dir
```

---

### Phase 2: Device-Specific Configuration (CRITICAL)

**Goal:** Make XLA/CUDA configuration conditional.

#### 2.1 Fix `train_easy.py:24-37`

**Current (BROKEN):**
```python
_trader_dir = os.path.dirname(os.path.abspath(__file__))
_xla_cache = os.path.join(_trader_dir, '.xla_cache')
_jax_cache = os.path.join(_trader_dir, '.jax_cache')
os.makedirs(_xla_cache, exist_ok=True)
os.makedirs(_jax_cache, exist_ok=True)

os.environ['XLA_FLAGS'] = f'--xla_gpu_autotune_level=4 --xla_gpu_per_fusion_autotune_cache_dir={_xla_cache}'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['JAX_COMPILATION_CACHE_DIR'] = _jax_cache
```

**Fixed:**
```python
from utils.device_utils import is_colab, is_tpu, get_cache_dir

_cache_dir = get_cache_dir()
os.makedirs(_cache_dir, exist_ok=True)

# Device-specific XLA configuration
if is_tpu():
    # TPU: No GPU flags, use TPU-specific settings
    os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
elif not is_colab():
    # Local GPU: Full CUDA optimization
    os.environ['XLA_FLAGS'] = f'--xla_gpu_autotune_level=4 --xla_gpu_per_fusion_autotune_cache_dir={_cache_dir}'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
# Colab GPU: Let JAX auto-configure

os.environ['JAX_COMPILATION_CACHE_DIR'] = _cache_dir
```

#### 2.2 Fix `train_jax_fast.py:14-37` (same pattern)

#### 2.3 Fix `diff_teacher/agent_evaluator.py:24`

**Current:**
```python
os.environ['JAX_PLATFORM_NAME'] = 'gpu'
```

**Fixed:**
```python
# Remove hardcoded device - let JAX auto-detect
# os.environ['JAX_PLATFORM_NAME'] = 'gpu'  # REMOVED
```

---

### Phase 3: Path/Filesystem Fixes (CRITICAL)

**Goal:** Make all paths work in Colab's `/content/` filesystem.

#### 3.1 Fix `train_easy.py` checkpoint paths

**Current (lines 817-827):**
```python
parser.add_argument('--bella-checkpoint', type=str,
                    default='checkpoints/diff_teacher/final/generator')
parser.add_argument('--output-dir', '-o', type=str, default='jax_ckpt_easy')
```

**Fixed:**
```python
from utils.device_utils import is_colab, get_default_checkpoint_dir

default_bella = '/content/drive/MyDrive/checkpoints/diff_teacher/final/generator' if is_colab() else 'checkpoints/diff_teacher/final/generator'
default_output = get_default_checkpoint_dir()

parser.add_argument('--bella-checkpoint', type=str, default=default_bella)
parser.add_argument('--output-dir', '-o', type=str, default=default_output)
```

#### 3.2 Fix `train_easy.py:976-978` CSV logging

**Fixed:**
```python
if is_colab():
    log_dir = Path('/content/drive/MyDrive/logs') if os.path.exists('/content/drive/MyDrive') else Path('/content/logs')
else:
    log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)
```

#### 3.3 Fix `e-dojo_v1.py:819-820` venv path

**Current:**
```python
self.python_path = str(self.project_root / "trader_env" / "bin" / "python")
```

**Fixed:**
```python
import sys
self.python_path = sys.executable  # Works in Colab and local
```

#### 3.4 Add Drive mount validation

Add to `train_easy.py` main():
```python
if is_colab():
    if not os.path.exists('/content/drive/MyDrive'):
        print("WARNING: Google Drive not mounted. Checkpoints will be ephemeral.")
        print("To mount: from google.colab import drive; drive.mount('/content/drive')")
```

---

### Phase 4: TPU-Incompatible Operations (CRITICAL for TPU)

**Goal:** Replace GPU-only JAX operations with TPU-compatible alternatives.

#### 4.1 Replace `jax.lax.top_k` in `path_solver.py`

**Current (line 117):**
```python
top_k_values, top_k_indices = jax.lax.top_k(masked_returns, k)
```

**Fixed:**
```python
def top_k_compatible(x, k):
    """TPU-compatible top_k using argsort."""
    # argsort works on both GPU and TPU
    sorted_indices = jnp.argsort(-x, axis=-1)  # Descending
    top_k_indices = sorted_indices[..., :k]
    top_k_values = jnp.take_along_axis(x, top_k_indices, axis=-1)
    return top_k_values, top_k_indices

top_k_values, top_k_indices = top_k_compatible(masked_returns, k)
```

#### 4.2 Same fix for `alpha_reward.py:39,116,174`

#### 4.3 Add dtype configuration for TPU

**In `easy_config.py`:**
```python
# Add to EasyConfig
compute_dtype: str = "float32"  # "bfloat16" for TPU

def get_compute_dtype(self):
    import jax.numpy as jnp
    if self.compute_dtype == "bfloat16":
        return jnp.bfloat16
    return jnp.float32
```

**In model creation (auto-detect):**
```python
from utils.device_utils import is_tpu

if is_tpu():
    config.compute_dtype = "bfloat16"  # 2-4x faster on TPU
```

---

### Phase 5: Memory Optimization (HIGH)

**Goal:** Reduce memory footprint to fit Colab limits.

#### 5.1 Create Colab-specific config presets

**Add to `easy_config.py`:**
```python
def get_colab_config() -> EasyConfig:
    """Memory-efficient config for Google Colab."""
    return EasyConfig(
        name="easy_colab",
        batch_size=128,        # Reduced from 256
        H_cycles=2,            # Reduced from 4
        L_cycles=3,            # Reduced from 6
        hidden_dim=128,        # Reduced from 256
        num_heads=4,           # Reduced from 8
        episodes_per_level=100,
    )

def get_colab_tpu_config() -> EasyConfig:
    """TPU-optimized config for Colab TPU runtime."""
    config = get_colab_config()
    config.name = "easy_colab_tpu"
    config.batch_size = 256   # TPU handles larger batches
    config.compute_dtype = "bfloat16"
    return config
```

#### 5.2 Reduce streaming defaults

**In `streaming.py`:**
```python
@dataclass
class StreamConfig:
    batch_size: int = 4 if is_colab() else 8
    buffer_size: int = 2 if is_colab() else 4
    seq_length: int = 360 if is_colab() else 1440
```

#### 5.3 Enable BiMamba sequential mode for Colab

**In `bimamba.py:27`:**
```python
sequential_mode: bool = True  # Changed from False - saves 50% memory
```

---

### Phase 6: Orchestration Compatibility (MEDIUM)

**Goal:** Make e-dojo work in Colab (single-process mode).

#### 6.1 Add Colab mode to e-dojo

**In `e-dojo_v1.py`:**
```python
class DojoOrchestrator:
    def __init__(self, ..., colab_mode: bool = None):
        if colab_mode is None:
            colab_mode = is_colab()
        self.colab_mode = colab_mode

        if self.colab_mode:
            print("[DOJO] Running in Colab mode (single-process)")
            # Disable subprocess-based training launcher
            self.launcher = None
```

#### 6.2 Create Colab notebook template

**Create `notebooks/train_colab.ipynb`:**
```python
# Cell 1: Setup
!pip install -q jax[tpu] flax optax orbax-checkpoint
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Clone repo (if needed)
!git clone https://github.com/your/trader.git /content/trader
%cd /content/trader

# Cell 3: Training
!python train_easy.py --preset colab --bella --epochs 100 \
    --output-dir /content/drive/MyDrive/checkpoints
```

---

### Phase 7: Dashboard Alternatives (LOW)

**Goal:** Provide visualization that works in Colab.

#### 7.1 Create Colab-native dashboard

**Create `tools/colab_dashboard.py`:**
```python
"""Colab-native training dashboard using IPython display."""
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import pandas as pd

def show_training_progress(csv_path: str, refresh_interval: int = 30):
    """Display live training metrics in Colab cell."""
    import time
    while True:
        clear_output(wait=True)
        df = pd.read_csv(csv_path)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        df['oracle_agreement'].plot(ax=axes[0,0], title='Agreement')
        df['policy_loss'].plot(ax=axes[0,1], title='Policy Loss')
        df['value_mse'].plot(ax=axes[1,0], title='Value MSE')
        df['agree_trade'].plot(ax=axes[1,1], title='Trade Agreement')
        plt.tight_layout()
        plt.show()

        time.sleep(refresh_interval)
```

---

## Testing Plan

### Pre-Implementation Audit Checklist

Before implementing, re-verify:
- [ ] Colab JAX version (may have changed from 0.4.x)
- [ ] Colab PyTorch version
- [ ] TPU firmware version
- [ ] Available Colab GPU types (T4, A100, etc.)
- [ ] Drive API changes

### Test Matrix

| Test | GPU Colab | TPU Colab | Local |
|------|-----------|-----------|-------|
| Import train_easy.py | | | |
| Device detection | | | |
| Bella checkpoint load | | | |
| 1 epoch training | | | |
| Checkpoint save/load | | | |
| 10 epoch training | | | |
| Memory usage < 80% | | | |

---

## Files to Modify

| File | Changes | Priority |
|------|---------|----------|
| `utils/device_utils.py` | **NEW FILE** | Phase 1 |
| `train_easy.py` | Device detection, paths, config | Phase 2-3 |
| `train_jax_fast.py` | Device detection, XLA flags | Phase 2 |
| `easy_config.py` | Colab presets, dtype config | Phase 5 |
| `mcts_trader/jax_agent/path_solver.py` | Replace top_k | Phase 4 |
| `mcts_trader/jax_agent/alpha_reward.py` | Replace top_k | Phase 4 |
| `diff_teacher/mamba/bimamba.py` | sequential_mode=True | Phase 5 |
| `diff_teacher/generation/streaming.py` | Colab defaults | Phase 5 |
| `tools/orchestration/e-dojo_v1.py` | Colab mode, sys.executable | Phase 6 |
| `notebooks/train_colab.ipynb` | **NEW FILE** | Phase 6 |
| `tools/colab_dashboard.py` | **NEW FILE** | Phase 7 |

---

## Rollback Plan

All changes should be backward-compatible with local development:
- Device detection defaults to local behavior when not in Colab
- Config presets are additive (don't modify existing presets)
- Path logic uses fallback chain: Colab → Local

---

## Success Criteria

- [ ] `train_easy.py --preset colab` runs on Colab GPU without errors
- [ ] `train_easy.py --preset colab_tpu` runs on Colab TPU without errors
- [ ] Checkpoints persist to Google Drive between sessions
- [ ] Training metrics visible via `colab_dashboard.py`
- [ ] Local development unchanged (no regressions)
- [ ] Memory usage < 12GB on Colab GPU, < 100GB on TPU

---

## Notes

- **Live trading (`real_trade_jax.py`)**: Intentionally NOT supported on Colab due to network restrictions and security concerns
- **e-dojo dual-branch**: Subprocess-based parallelism won't work on Colab; consider sequential training or Colab-native parallelism
- **Long training**: Colab has 12-hour limit; implement checkpoint-resume for multi-session training
