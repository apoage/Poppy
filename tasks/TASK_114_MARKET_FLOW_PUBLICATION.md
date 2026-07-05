# TASK_114: Market-Flow Publication Package

## Status: FUTURE (After Generator Complete)

## Prerequisites
- [ ] TASK_111 complete (CNN baseline trained, quality validated)
- [ ] TASK_112 complete (skeleton conditioning working)
- [ ] TASK_113 evaluated (Mamba vs CNN benchmark done)
- [ ] Winner architecture selected and trained

---

## Goal

Create a **standalone, publishable package** for the market generator:
- Inference only (no training loop)
- No RL integration
- Web viewer for generated data
- Direct-to-VRAM streaming capability
- HuggingFace-ready

---

## What to Publish

| Component | Include | Notes |
|-----------|---------|-------|
| Generator architecture | Yes | Core model code |
| Trained weights | Yes | Checkpoint files |
| Skeleton extraction | Yes | Utility functions |
| Evaluation suite | Yes | Stylized facts, metrics |
| Web viewer | Yes | Interactive visualization |
| VRAM streaming stub | Yes | For integration demos |
| Training loop | **No** | Keep private |
| RL integration | **No** | Keep private |
| Data pipeline | **No** | Keep private |
| Full diff_teacher | **No** | Keep private |

---

## Package Structure

```
market-flow-generator/
├── README.md                    # Overview, installation, usage
├── LICENSE                      # Apache 2.0 or MIT
├── pyproject.toml               # Package config
├── setup.py
│
├── market_flow/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── conv_generator.py    # CNN architecture (cleaned)
│   │   ├── mamba_generator.py   # Mamba architecture (if winner)
│   │   ├── flow_matching.py     # Flow matching inference
│   │   └── conditioning.py      # FiLM, AdaLN, skeleton encoder
│   │
│   ├── skeleton/
│   │   ├── __init__.py
│   │   ├── extraction.py        # extract_skeleton()
│   │   ├── builder.py           # SkeletonBuilder class
│   │   └── types.py             # MarketSkeleton dataclass
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── generator.py         # High-level generate() API
│   │   ├── streaming.py         # Direct VRAM output
│   │   └── sampling.py          # ODE solvers, Diffrax integration
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── stylized_facts.py    # Fat tails, vol clustering, etc.
│       ├── controllability.py   # Regime adherence, vol targeting
│       └── visualization.py     # Plotting utilities
│
├── web_viewer/
│   ├── app.py                   # Gradio or Streamlit app
│   ├── static/
│   └── templates/
│
├── configs/
│   ├── model_small.yaml
│   ├── model_medium.yaml
│   └── model_large.yaml
│
├── checkpoints/
│   └── README.md                # Instructions to download weights
│
├── notebooks/
│   ├── quickstart.ipynb         # Basic usage
│   ├── skeleton_design.ipynb    # Custom scenario creation
│   └── evaluation.ipynb         # Quality assessment
│
└── examples/
    ├── basic_generation.py
    ├── skeleton_conditioning.py
    ├── batch_generation.py
    └── vram_streaming.py
```

---

## Core API Design

### Simple Generation
```python
from market_flow import MarketFlowGenerator, load_checkpoint

# Load model
generator = MarketFlowGenerator.from_pretrained("market-flow-93m")
# or
generator = load_checkpoint("path/to/checkpoint")

# Generate unconditional
prices = generator.generate(
    n_samples=10,
    seq_length=1440,
    n_assets=64,
    seed=42
)
# Returns: Array[10, 1440, 64, 5] (batch, time, assets, OHLCV)
```

### Skeleton-Conditioned Generation
```python
from market_flow import SkeletonBuilder

# Build custom scenario
skeleton = SkeletonBuilder() \
    .set_regime_sequence(['sideways', 'crash', 'recovery']) \
    .set_volatility_percentile(0.8) \
    .set_correlation_mode('crisis')  # All assets correlated
    .build()

# Generate with skeleton
prices = generator.generate(skeleton=skeleton, n_samples=10)
```

### Direct VRAM Streaming
```python
from market_flow.streaming import VRAMGenerator

# For real-time applications / integration
vram_gen = VRAMGenerator(generator, device='cuda:0')

# Returns JAX array on GPU, no CPU roundtrip
prices_gpu = vram_gen.generate_on_device(
    skeleton=skeleton,
    return_device_array=True
)

# Use directly in downstream JAX computation
# (useful for RL training integration - but we don't publish that part)
```

---

## Web Viewer Spec

### Technology
- **Gradio** (preferred) or Streamlit
- Lightweight, easy to deploy
- HuggingFace Spaces compatible

### Features
1. **Generate Tab**
   - Regime selector (dropdown)
   - Volatility slider
   - Correlation mode selector
   - Number of samples
   - Generate button
   - Download CSV/Parquet

2. **Visualize Tab**
   - Multi-asset price chart (Plotly)
   - Returns distribution histogram
   - Correlation heatmap
   - Volatility time series

3. **Evaluate Tab**
   - Stylized facts report
   - Compare to reference statistics
   - Quality metrics display

4. **Skeleton Designer Tab**
   - Visual skeleton builder
   - Drag-and-drop regime blocks
   - Preview skeleton shape
   - Export skeleton config

### Minimal Gradio App
```python
import gradio as gr
from market_flow import MarketFlowGenerator, SkeletonBuilder

generator = MarketFlowGenerator.from_pretrained("market-flow-93m")

def generate_market(regime, volatility, n_samples, seed):
    skeleton = SkeletonBuilder() \
        .set_regime_sequence([regime]) \
        .set_volatility_percentile(volatility) \
        .build()

    prices = generator.generate(
        skeleton=skeleton,
        n_samples=n_samples,
        seed=seed
    )

    # Create visualization
    fig = plot_prices(prices[0])  # First sample
    stats = compute_stats(prices)

    return fig, stats

demo = gr.Interface(
    fn=generate_market,
    inputs=[
        gr.Dropdown(["crash", "rally", "sideways", "volatile", "calm"]),
        gr.Slider(0, 1, value=0.5, label="Volatility"),
        gr.Number(value=5, label="Samples"),
        gr.Number(value=42, label="Seed"),
    ],
    outputs=[
        gr.Plot(label="Generated Prices"),
        gr.JSON(label="Statistics"),
    ],
    title="Market-Flow Generator",
    description="Generate synthetic financial time series with skeleton conditioning."
)

demo.launch()
```

---

## Repository Setup

### Separate Branch vs Separate Repo

**Option A: Separate Repository** (Recommended)
- Clean separation from trading system
- Easy to open source
- Independent versioning
- No risk of accidentally exposing private code

**Option B: Branch in Trader Repo**
- Easier to sync updates
- Shared history
- Risk of leaking private code

**Decision**: Separate repository `market-flow-generator`

### Git Strategy
```bash
# Create new repo
mkdir market-flow-generator
cd market-flow-generator
git init

# Copy only public components (manually curated)
# DO NOT copy: training loops, RL code, data pipelines

# Set up remotes
git remote add origin git@github.com:username/market-flow-generator.git
```

---

## HuggingFace Integration

### Model Card
```markdown
---
license: apache-2.0
tags:
  - time-series
  - financial
  - generative
  - flow-matching
  - jax
library_name: market-flow
---

# Market-Flow Generator

Skeleton-conditioned financial time series generator using Flow Matching.

## Usage
...

## Architecture
...

## Evaluation
...

## Citation
...
```

### Checkpoint Upload
```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="checkpoints/market_flow_93m",
    repo_id="username/market-flow-93m",
    repo_type="model",
)
```

### Spaces Deployment
```yaml
# README.md for HuggingFace Spaces
---
title: Market-Flow Generator
emoji: 📈
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: web_viewer/app.py
pinned: false
---
```

---

## Documentation

### README Sections
1. Overview & Demo GIF
2. Installation
3. Quick Start
4. API Reference
5. Skeleton Conditioning Guide
6. Evaluation Metrics
7. Model Card
8. Citation
9. License

### Notebooks
1. `quickstart.ipynb` - 5-minute intro
2. `skeleton_design.ipynb` - Custom scenarios
3. `evaluation.ipynb` - Quality metrics
4. `advanced_usage.ipynb` - Batch, streaming, custom configs

---

## Implementation Steps

### Phase 1: Extract & Clean Code
- [ ] Extract generator architecture (remove training)
- [ ] Extract skeleton extraction utilities
- [ ] Extract evaluation metrics
- [ ] Remove all imports from private modules
- [ ] Add proper docstrings

### Phase 2: Package Structure
- [ ] Create pyproject.toml
- [ ] Set up package imports
- [ ] Write unit tests
- [ ] Verify clean install

### Phase 3: Web Viewer
- [ ] Build Gradio app
- [ ] Add visualization components
- [ ] Test locally
- [ ] Optimize for Spaces

### Phase 4: Documentation
- [ ] Write README
- [ ] Create notebooks
- [ ] Model card
- [ ] API docs

### Phase 5: Release
- [ ] Final code review (no private leaks)
- [ ] Upload checkpoints to HuggingFace
- [ ] Deploy to HuggingFace Spaces
- [ ] Announce

---

## Security Checklist

Before publishing, verify:

- [ ] No training loop code
- [ ] No RL integration code
- [ ] No data loading from trading_data.db
- [ ] No references to private config files
- [ ] No API keys or credentials
- [ ] No paths to private filesystem
- [ ] No diff_teacher training code
- [ ] No references to Trader repo internals
- [ ] Git history is clean (no private commits)

---

## Potential Value Add

### Direct VRAM Output
Publishing a clean "generate directly to GPU array" interface is useful for:
- Other researchers doing RL with market data
- Backtesting frameworks
- Real-time simulation systems

This is a **differentiator** - most generators return CPU arrays.

### Skeleton API
The skeleton conditioning concept is novel for finance:
- Stress testing use case
- Scenario analysis
- Curriculum learning

Clear documentation of this API adds significant value.

---

## Timeline

| Phase | Depends On | Effort |
|-------|------------|--------|
| Phase 1: Extract | TASK_111-113 complete | 1-2 days |
| Phase 2: Package | Phase 1 | 1 day |
| Phase 3: Web Viewer | Phase 2 | 1-2 days |
| Phase 4: Documentation | Phase 2-3 | 1 day |
| Phase 5: Release | All above | 1 day |

**Total**: ~1 week after generator is working

---

## Notes

- Keep it simple - this is a research release, not production software
- Focus on ease of use and clear examples
- The web viewer is the "wow factor" for HuggingFace
- VRAM streaming is the "serious researcher" feature
- Don't over-engineer - ship when it works
