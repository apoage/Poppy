# TASK_688: CMP 90HX Unlock & Multi-GPU Integration

**Status**: PLANNED
**Priority**: Background (hardware project, independent of training)

## Hardware Specs

| Spec | Value |
|------|-------|
| GPU | GA102-100-A1 (same die as RTX 3090 Ti) |
| CUDA Cores | 6400 (vs 3090's 10496) |
| Tensor Cores | 200 (3rd gen, bfloat16 capable) |
| VRAM | 10 GB GDDR6X, 320-bit, 760 GB/s |
| TDP | 320W, 2× 8-pin |
| PCIe | **Locked to PCIe 1.0 x4** (~1 GB/s) |
| Display | None (mining card) |
| Board | PG132 (same PCB as RTX 3080) |
| CUDA Compute | 8.6 (Ampere) |

## Goal

Maximize the card's utility for ML training: Bella sideloading, potential data-parallel compute, PCIe unlock for bandwidth.

## Phase 1: Basic Integration (No Hardware Mods)

### 1a. Physical Install
- Slot: secondary x4 PCIe slot (motherboard dependent)
- Power: 2× 8-pin, 320W TDP — verify PSU headroom (3090 also draws 350W)
- Cooling: dual-slot, verify airflow clearance
- **Total system power**: 3090 (350W) + 90HX (320W) + system = ~800W minimum PSU

### 1b. Linux Driver Setup
- CMP cards work headless with standard NVIDIA drivers
- CUDA 8.6 compute capability — same as 3090
- Verify: `nvidia-smi` shows both GPUs
- Verify: `JAX` sees both: `jax.devices()` should list 2 GPUs
- May need: `CUDA_VISIBLE_DEVICES=0,1` environment variable

### 1c. Bella Sideloading (~20 LOC)
Pin Bella generator to GPU2, training stays on GPU1.

```python
# In train_easy.py or generator code:
bella_device = jax.devices()[1]  # CMP 90HX
train_device = jax.devices()[0]  # RTX 3090

# Move Bella params to GPU2
bella_params = jax.device_put(bella_params, bella_device)

# Generate on GPU2, transfer result to GPU1
with jax.default_device(bella_device):
    market_data = bella.generate(bella_params, rng)
market_data = jax.device_put(market_data, train_device)
```

Data transfer budget (even at PCIe 1.0 x4 = ~1 GB/s):
- Per episode: (400, 64, 5) × float32 = 512 KB
- Per buffer fill (32 episodes): ~16 MB → **16ms transfer** — negligible

### 1d. Benchmark
- Run Bella generation on GPU1 vs GPU2, compare throughput
- Measure actual PCIe transfer latency
- Confirm no PCIe errors in `dmesg`

## Phase 2: PCIe Unlock (Hardware Mod)

### Two-Layer Lockdown
The CMP 90HX has TWO layers of PCIe restriction:

1. **Firmware/fuse level**: Locks to PCIe Gen 1 speed (2.5 GT/s per lane)
2. **PCB level**: Missing AC coupling capacitors on lanes x4-x15, forcing x4 width

### 2a. PCIe Width Unlock (x4 → x16)
- **Method**: Solder 0402-size AC coupling capacitors on unpopulated pads near PCIe connector
- **Board**: PG132 — same as RTX 3080, schematics may be available
- **Skill**: Fine-pitch SMD soldering (0402 = 1.0mm × 0.5mm)
- **Risk**: Medium — if caps are wrong value or placement, card still works at x4
- **Reward**: 4× bandwidth (x16 at Gen1 = 4 GB/s vs x4 at Gen1 = 1 GB/s)
- **Resources**: [NVIDIA-patcher issue #45](https://github.com/dartraiden/NVIDIA-patcher/issues/45) discusses this mod

### 2b. PCIe Gen Unlock (Gen1 → Gen3/4)

#### Method 1: `setpci` Runtime Hack (try first — zero risk)

Check current link status:
```bash
# Find GPU PCIe address
lspci | grep -i nvidia
# e.g. output: 01:00.0 3D controller: NVIDIA Corporation GA102 ...

# Check current link speed and capability
sudo lspci -vv -s 01:00.0 | grep -i "lnk"
# LnkCap: Speed 2.5GT/s (Gen1)  ← hardware/firmware advertised max
# LnkSta: Speed 2.5GT/s         ← actual negotiated speed
```

Force link speed renegotiation:
```bash
# Read Link Control 2 register (offset 0x30 from PCIe capability)
sudo setpci -s 01:00.0 CAP_EXP+30.W

# Force target link speed to Gen3 (value 3) or Gen4 (value 4)
# Bits [3:0] = target link speed: 1=Gen1, 2=Gen2, 3=Gen3, 4=Gen4
sudo setpci -s 01:00.0 CAP_EXP+30.W=0003

# Trigger link retrain (set bit 5 in Link Control register at CAP_EXP+10)
sudo setpci -s 01:00.0 CAP_EXP+10.W=0020

# Check if it worked
sudo lspci -vv -s 01:00.0 | grep -i "lnk"
```

**Reality check**: This likely won't work on CMP cards because:
- The GPU's LnkCap probably reports Gen1 max (firmware-locked)
- NVIDIA driver overrides speed settings on load
- The [NVIDIA dev forum](https://forums.developer.nvidia.com/t/linux-enabling-persistent-pcie-gen-4-0-and-pcie-buswidth-16-for-l40s-gpu/338641) confirms no driver parameter for forcing Gen4
- `NVreg_EnablePCIeGen3=1` module param exists but may not override CMP firmware

But it costs nothing to try:
```bash
# Add to /etc/modprobe.d/nvidia.conf
options nvidia NVreg_EnablePCIeGen3=1
```

#### Method 2: Cross-Flash RTX 3080 VBIOS (the punk approach)

The CMP 90HX and RTX 3080 share critical specs:
| | CMP 90HX | RTX 3080 |
|---|----------|----------|
| Board | PG132 | PG132 |
| GPU | GA102-100-A1 | GA102-200-KD-A1 |
| VRAM | 10GB GDDR6X | 10GB GDDR6X |
| Bus width | 320-bit | 320-bit |
| CUDA cores | 6400 | 8704 |

Same board, same memory config. The GPU variant differs (100 vs 200) and CUDA cores differ (fuse-locked), but memory training and PCIe config should match.

**Tool**: [nvflashk](https://github.com/notfromstatefarm/nvflashk) — bypasses board ID mismatch checks
- Uses NVIDIA's own internal board ID override (forced always-on)
- Supports Ampere (GA102 confirmed)
- Only works with **signed** VBIOS files (stock RTX 3080 VBIOS = signed by NVIDIA)
- Cannot flash modified/unsigned VBIOS

**Procedure**:
```bash
# 1. ALWAYS backup first
nvflashk.exe --protectoff
nvflashk.exe -b factory_cmp90hx_backup.rom

# 2. Download RTX 3080 10GB VBIOS from TechPowerUp (pick same memory type)
# 3. Flash with board ID bypass
nvflashk.exe -6 rtx3080_vbios.rom

# 4. If GPU doesn't POST: reflash backup via CH341 hardware programmer
```

**What could happen**:
- **Best case**: GPU boots with RTX 3080 identity, PCIe Gen3/4 unlocked, 3D acceleration native
- **Likely case**: GPU boots but only 6400 cores active (fuse-locked), PCIe may or may not upgrade
- **Worst case**: GPU doesn't POST → CH341 reflash backup → back to CMP 90HX
- **Impossible**: Permanent brick — nvflashk author says "almost impossible to permanently brick from BIOS alone"

**Why RTX 3080 and NOT 3090 Ti**:
- 3090 Ti = PG136 board, 24GB, 384-bit — memory controller mismatch = guaranteed fail
- 3080 = PG132 board, 10GB, 320-bit — **identical memory layout**

#### Method 3: CH341 Hardware Programmer (backup/recovery)
- Read stock VBIOS as backup before any experiments
- If cross-flash bricks POST: reflash original via CH341 → guaranteed recovery
- Can also write modified VBIOS directly, but GPU boot-time signature check may reject it
- **Essential safety net for Method 2**

### 2c. Research Resources
- [dartraiden/NVIDIA-patcher](https://github.com/dartraiden/NVIDIA-patcher) — driver patches for 3D acceleration on CMP cards
- [NVIDIA-patcher issue #45](https://github.com/dartraiden/NVIDIA-patcher/issues/45) — CMP 90HX specific discussion
- [nvflashk](https://github.com/notfromstatefarm/nvflashk) — board ID bypass flash tool (Ampere support)
- [CMP 170HX deep teardown](https://niconiconi.neocities.org/tech-notes/nvidia-cmp-170hx-review/) — PCIe lockdown analysis
- [TechPowerUp CMP 90HX VBIOS](https://www.techpowerup.com/vgabios/240539/240539) — stock VBIOS for backup reference
- [TechPowerUp RTX 3080 VBIOS collection](https://www.techpowerup.com/vgabios/?architecture=NVIDIA+GA102&did=2216) — cross-flash candidates
- [NVIDIA PCIe speed blog](https://billauer.co.il/blog/2018/08/nvidia-low-pci-express-speed/) — lspci diagnostics
- Win-Raid forums, Overclock.net — VBIOS modding community

### 2d. Multi-Card Idea (4× CMP 90HX on PCIe 1.0)
At PCIe 1.0 x4 each, four cards fit in x16 worth of lanes. But:
- 4 × 320W = 1280W TDP just for GPUs — massive PSU needed
- 4 × 10GB = 40GB total — but JAX `pmap` needs equal memory per device
- For Bella sideloading: overkill — 1 card generates data faster than training consumes
- For gradient accumulation: PCIe 1.0 bandwidth is the real bottleneck (gradient sync)
- **Verdict**: Fun thought, not practical for our workload. One CMP 90HX is sufficient.

## Phase 3: Advanced Mods (Optional)

### 3a. Display Output
- PG132 board has unpopulated display connector pads
- Would need: populate DP/HDMI components + driver patch (dartraiden/NVIDIA-patcher)
- **Not needed for compute** — only useful if repurposing as gaming GPU

### 3b. VRAM Unlock (10GB → ?)
- GA102-100-A1 has 12 memory channels, PG132 board populates 10 (320-bit)
- Remaining 2 channels may have unpopulated GDDR6X pads
- If populatable: 10GB → 12GB (requires matching Micron GDDR6X chips + BGA rework)
- **High risk, marginal reward** — 10GB is sufficient for Bella + small model

### 3c. Power Limit Unlock
- Mining VBIOS may have different power tables than GeForce
- With hardware programmer: could modify power limit tables
- Already 320W — probably enough, but could be interesting to explore

## Phase 4: Dual-GPU Training (After PCIe Unlock)

If PCIe unlock succeeds (≥ Gen3 x4 = 4 GB/s):

### 4a. Gradient Accumulation
- Run micro-batches on GPU2 (batch=64-128, limited by 10GB)
- Accumulate gradients, transfer to GPU1 for optimizer step
- Effectively increases batch diversity without more VRAM on GPU1
- Transfer: ~100MB gradients (25.71M params × 4 bytes) at 4 GB/s = 25ms

### 4b. Model-Parallel (encoder on GPU2, reasoning on GPU1)
- Encoder: 0.56M params — fits easily on 10GB
- Transfer: state vector (batch × 128) per step — tiny
- Frees GPU1 VRAM for larger batch or wider model

## Priority Order

1. **Phase 1c**: Bella sideloading (immediate value, ~20 LOC)
2. **Phase 2a**: PCIe x4→x16 capacitor mod (moderate value, hardware work)
3. **Phase 2b**: PCIe Gen1→Gen3+ (high value, needs research)
4. **Phase 4a**: Gradient accumulation (only after PCIe unlock)
5. Everything else: nice-to-have

## PCIe Stability Concern

Previous PCIe/southbridge crashes were caused by incompatible NVMe drive (now removed, left for Windows). New motherboard — should be clean.
- Monitor: `sudo dmesg -w` during initial testing
- Start with Bella-only workload on GPU2 (low bus traffic)
- If crashes return: reduce GPU2 clock speeds, try different PCIe slot

## PCIe Lock Diagnosis (First Thing When Card Installed)

```bash
# This single command tells you everything:
sudo lspci -vv -s XX:00.0 | grep -i "lnkcap\|lnksta"
```

- `LnkCap: Speed 2.5GT/s` → Gen lock is firmware/fuse level (not just resistors)
- `LnkCap: Speed 8GT/s`, `LnkSta: Speed 2.5GT/s` → negotiation issue, fixable with setpci/strap mod

Also check: CMP 50HX cap-soldering attempt [failed to bring extra lanes online](https://www.tomshardware.com/pc-components/gpus/crypto-mining-version-of-rtx-2080-ti-crippled-by-pcie-lanes-in-gaming-test), suggesting firmware/fuse gating beyond just missing caps. Diagnose before soldering.

## Equipment Available

- CH341 hardware programmer (EEPROM read/write for VBIOS backup+recovery)
- Soldering station (for 0402 AC coupling caps and strap resistor mods)
- Both at work
