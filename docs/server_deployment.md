# Server Deployment Snapshot

Target: multi-GPU server (multiple RTX 3090), Ubuntu 22.04 LTS.
Source: workstation Ubuntu 25.10, RTX 3090 + CMP 90HX, kernel 6.17.

## OS — NOT bound

Nothing in this stack pins a specific OS. **Ubuntu 22.04 LTS is fine** (also 24.04). Any modern Linux x86_64 with a recent NVIDIA driver works.

## Hardware

- **GPU**: NVIDIA with ≥ 24 GB VRAM. Currently RTX 3090 confirmed working.
  - Code uses `XLA_PYTHON_CLIENT_MEM_FRACTION=0.7` (~17 GB on a 24 GB card).
  - Works at 16 GB but tight; <16 GB will OOM with the 17-slot pool.
- **RAM**: ≥ 45 GB free (17-slot pool + 60-step OHLCV in RAM + JIT caches; cf. `lore/notes_2026-04-24.md §61`).
- **Disk**: 50 GB project + 24 GB DB + 7.6 GB venv. Reserve **≥ 150 GB** for checkpoints / logs / caches.
- **CPU**: modern x86_64. CPU does price oracle / pool generation in parallel with GPU train.

## Multi-GPU note

Current training code is **single-GPU**. JAX runs on `jax.devices()[0]`. To use multiple 3090s on the new box, either:
1. **Run independent jobs per GPU** — easiest. Set `CUDA_VISIBLE_DEVICES=0` for run A, `=1` for run B, etc. Great for ablations / repeats / parallel oracle variants.
2. **Data-parallel via `pmap`** — code change required, none of the current training entries (`train_easy.py`, `train_phase2.py`) use `pmap` or `jax.distributed.initialize`. Not free.

For now, plan on (1): launch the queued runs (TASK_737 minimal repeat, A1+B1, TASK_741 cheat re-run, etc.) in parallel on separate cards.

**GPU index quirk** (workstation): `nvidia-smi` ordering is the *reverse* of CUDA/JAX ordering — on the workstation `nvidia-smi 0` is the CMP 90HX, but `CUDA_VISIBLE_DEVICES=0` selects the RTX 3090. On a homogeneous multi-3090 server this disappears, but always verify with `python -c "import jax; print(jax.devices())"` before launching.

## NVIDIA driver / CUDA

- **Driver ≥ 550** required by the bundled CUDA 12.1 wheels. Workstation runs 580.142.
- **CUDA toolkit comes from pip wheels**: `nvidia-cuda-runtime-cu12==12.1.105`, cuDNN 9.17, NCCL 2.21. **System CUDA toolkit not needed** and should not be on PATH while inside the venv (mixing system/wheel CUDA causes silent breakage).
- nvcc on the system is fine; JAX won't use it.

Ubuntu 22.04 install path:
```bash
# 1) Driver — use the graphics-drivers PPA or NVIDIA's .run installer.
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install -y nvidia-driver-550   # or 555/560
sudo reboot
# Verify: nvidia-smi shows all GPUs.
```

## Python

- **Python 3.10.14** (pinned in `.python-version`). Install via `pyenv` or `deadsnakes`:
  ```bash
  sudo add-apt-repository ppa:deadsnakes/ppa
  sudo apt install -y python3.10 python3.10-venv python3.10-dev build-essential
  ```
- 3.11/3.12 likely work but are untested here. **Stick to 3.10** for reproducibility.

## Key library versions (frozen on workstation)

```
jax==0.6.2                  jaxlib==0.6.2
jax-cuda12-plugin==0.6.2    jax-cuda12-pjrt==0.6.2
flax==0.10.7
optax==0.2.6                # PINNED — TASK_675 #15: rebuild_optimizer_state uses semi-private API
chex==0.1.90
mctx==0.0.5
torch==2.5.1+cu121          torchvision==0.20.1+cu121  torchaudio==2.5.1+cu121
numpy==1.26.4               pandas==2.3.3
numba==0.63.1               gymnasium==1.2.2
orbax-checkpoint==0.11.30
tensorflow-probability==0.25.0
tensorflow-datasets==4.9.9
```

`requirements.txt` is loose (`>=`). Before snapshot, freeze the working venv:
```bash
source trader_env/bin/activate
pip freeze > requirements.lock.txt
```
Ship `requirements.lock.txt` and use it on the server.

## Hard pins / gotchas

- `optax==0.2.6` — do not bump; semi-private API used.
- `numpy<2` — TF / numba compat.
- JAX env vars: keep `XLA_PYTHON_CLIENT_PREALLOCATE` default true, only set `MEM_FRACTION=0.7`. **Do not force `XLA_PYTHON_CLIENT_PREALLOCATE=false` or `XLA_PYTHON_CLIENT_ALLOCATOR=platform`** (degrades JIT and fragmentation behaviour — confirmed on this stack).
- Reentrant locks: `threading.RLock()` not `Lock()` (project convention, see CLAUDE.md).

## What to copy in the snapshot

| Path | Size | Required | Note |
|---|---|---|---|
| Source: all `.py`, `mcts_trader/`, `diff_teacher/`, `tools/`, `yarrow/` | few MB | yes | |
| `requirements.txt` + `requirements.lock.txt` | small | yes | |
| `config/` (incl. `foxglove_periods.json`, `benchmark_periods.json`) | small | yes | |
| `trading_data.db` | **24 GB** | yes | OHLCV training data |
| `jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl` | ~30–60 MB | yes | warm-start for Stage B |
| `jax_ckpt_easy_phase2_736/` | 166 MB | yes | reference baseline |
| `jax_ckpt_easy_phase2_737_minimal/` | 164 MB | yes | best provisional |
| `jax_ckpt_easy_phase2_739_dense_sign/` | 166 MB | optional | dense_sign result |
| `jax_ckpt_easy_phase2_a1b1/` | growing | yes if mid-run | currently training |
| `jax_ckpt_easy_phase1_cheat/` | varies | optional | diagnostic only, NEVER warmstart prod |
| `lore/`, `tasks/`, `docs/`, all `CLAUDE.md` | small | yes | context — required for orientation |
| `prompt-lib/` | small | yes | |
| `.jax_cache/` (982 MB), `.xla_cache/` | — | **skip** | rebuilds on first run |
| `trader_env/` (7.6 GB) | — | **skip** | recreate via pip on server |
| `__pycache__/`, `*.pyc` | — | skip | |
| `jax_ckpt_easy_phase2_739_dense_sign.eps1e-4/` | — | skip | failed run, kept for reference only |

## Reproduce on server

```bash
# 1) Driver + reboot (above), verify nvidia-smi shows all GPUs.

# 2) Python 3.10
sudo apt install -y python3.10 python3.10-venv python3.10-dev build-essential

# 3) Project
mkdir -p ~/Trader && cd ~/Trader
# (extract snapshot tarball here)

# 4) venv
python3.10 -m venv trader_env
source trader_env/bin/activate
pip install -U pip wheel
pip install -r requirements.lock.txt

# 5) Sanity
python -c "import jax; print(jax.devices())"          # expect [CudaDevice(id=0), CudaDevice(id=1), ...]
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"

# 6) Verify DB and warm-start ckpt are present
ls -lh trading_data.db jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl
```

## Suggested tarball command

```bash
cd /home/lukas
tar --exclude='Trader/trader_env' \
    --exclude='Trader/.jax_cache' \
    --exclude='Trader/.xla_cache' \
    --exclude='Trader/__pycache__' \
    --exclude='Trader/**/__pycache__' \
    --exclude='Trader/jax_ckpt_easy_phase2_739_dense_sign.eps1e-4' \
    -czf trader_snapshot_$(date +%F).tar.gz Trader
```

Expect ~30–35 GB compressed (mostly `trading_data.db` + active checkpoints).

## Per-GPU launch template (server)

```bash
# Run two ablations in parallel on a 2x3090 box:
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 \
  bash tools/run_a1b1_stacked.sh > a1b1.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 \
  bash tools/run_737_ablations.sh > 737.log 2>&1 &
```

Each run is single-GPU; they don't talk to each other. Disk I/O on `trading_data.db` is read-only and cached, so concurrency is fine.

## NOT bound to OS — bound to:

- NVIDIA driver ≥ 550
- Linux x86_64
- Python 3.10
- The pip-frozen venv (`requirements.lock.txt`)
