# TASK_030: Training Animation Tools (DS56/DS78)

## Status: PLANNED

## Overview

Create visualization tools for animating neural network training dynamics over the full curriculum. Uses tensor snapshots collected during training to generate videos showing latent space evolution, TRM reasoning crystallization, and physics model discovery.

## Motivation

- Training dynamics are rarely visualized temporally (most tools show static snapshots)
- TRM recursive reasoning creates unique "crystallization" patterns as cycles specialize
- Physics model gradually discovers market structure (correlations, volatility regimes)
- Full curriculum training (L0→L14) creates a compelling "learning journey" narrative

## Data Source

Tensor snapshots collected via:
```bash
--ds24-save-tensors --ds24-snapshot-every 10
```

Files: `tensor_snapshots/<run_name>/epoch_NNNNNN.npz`

Contents per snapshot:
| Tensor | Shape | Description |
|--------|-------|-------------|
| `latent_states` | (batch, state_dim) | Latent space point cloud |
| `latent_trajectory` | (steps, state_dim) | Temporal trajectory |
| `latent_pca_2d` | (steps, 2) | PCA-projected 2D trajectory |
| `pca_explained_variance` | (2,) | Variance explained |
| `correlation_matrix` | (n_assets, n_assets) | Learned correlations |
| `value_head_weights` | varies | Weight matrices |
| `trm_cycle_utilization` | (H, L) | TRM heatmap |

## Proposed Tools

### DS56: Latent Dynamics Animator

`tools/DS56/animate_latent.py`

Features:
- 3D latent space visualization (PCA or UMAP)
- Particle trails showing trajectory over epochs
- Color coding: epoch (time rainbow) or performance (reward gradient)
- Camera orbit animation for depth perception

Output: `latent_evolution.mp4`

### DS78: Training Journey Composer

`tools/DS78/compose_training_video.py`

Multi-panel synchronized animation:
```
┌─────────────────┬─────────────────┐
│  Latent Space   │  Performance    │
│  (3D rotating)  │  (reward/alpha) │
├─────────────────┼─────────────────┤
│  TRM Heatmap    │  Physics Params │
│  (H×L cycles)   │  (α, β, GARCH)  │
└─────────────────┴─────────────────┘
         Timeline: Epoch 0 → 500
```

Optional audio track: Map training metrics to sound
- Loss → pitch
- Entropy → reverb
- TRM delta norms → percussion

### Shared Components

`tools/viz_common/`
- `snapshot_loader.py` - Load and validate .npz files
- `color_schemes.py` - Consistent color palettes
- `video_utils.py` - Frame generation, ffmpeg wrapper
- `audio_mapper.py` - Metrics-to-sound synthesis

## Implementation Phases

### Phase 1: Basic Frame Generation
- [ ] Load snapshot sequence
- [ ] Generate static frames for each epoch
- [ ] Basic 2D latent trajectory plot
- [ ] Correlation matrix heatmap sequence
- [ ] ffmpeg stitching to video

### Phase 2: 3D and Interactivity
- [ ] 3D latent space with matplotlib/plotly
- [ ] Camera path animation
- [ ] Interactive HTML export (plotly)
- [ ] Epoch slider for exploration

### Phase 3: Multi-Panel Composition
- [ ] Synchronized multi-chart layout
- [ ] Performance overlay with annotations
- [ ] Curriculum level transitions highlighted
- [ ] Export as single composed video

### Phase 4: Audio Synthesis (Optional/Fun)
- [ ] Map metrics to MIDI
- [ ] Generate audio track
- [ ] Sync with video frames

## Example Usage

```bash
# Generate latent space animation
python tools/DS56/animate_latent.py \
    --snapshots tensor_snapshots/curriculum_full/ \
    --output latent_evolution.mp4 \
    --fps 10 \
    --color-by reward

# Generate full training journey video
python tools/DS78/compose_training_video.py \
    --snapshots tensor_snapshots/curriculum_full/ \
    --csv logs/training_curriculum_full.csv \
    --output training_journey.mp4 \
    --resolution 1920x1080
```

## Dependencies

```
matplotlib>=3.7
plotly>=5.0
numpy
ffmpeg (system)
# Optional for audio:
midiutil
pydub
```

## Notes

- Start simple: 2D frames → ffmpeg video
- Optimize later: caching, parallel frame generation
- Consider Manim for more polished mathematical animations
- Could integrate with DS24 dashboard as "replay" feature

## Related

- TASK_008-012: Physics model (source of correlation data)
- DS24: Training dashboard (live metrics)
- TRM architecture: Source of cycle utilization data
