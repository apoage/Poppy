# TASK_045: JSON Training Configuration Files

## Status: COMPLETED (2026-01-23)

**Implementation**: JSON config files in `config/` directory
- `advanced_curriculum.json`, `benchmark_periods.json`, `dojo_override.json`
- `curriculum_adaptive_fees.json`, `pretrain_curriculum.json`
- Plus schedule files for training phases

## Motivation

Training command complexity has grown significantly:
- 30+ CLI arguments
- Hard to track which configs were used for which runs
- Difficult to reproduce experiments
- Sweep configs already use Python dataclasses

## Proposed Structure

```
config/
├── base/
│   ├── small_model.json      # Model size defaults
│   ├── medium_model.json
│   └── large_model.json
├── curriculum/
│   ├── level0_sine.json      # Curriculum-specific settings
│   ├── level1_lagged.json
│   └── ...
├── interventions/
│   ├── epsilon_030.json      # Stability interventions
│   ├── triple_entropy.json
│   └── ...
└── experiments/
    ├── sweep_20260106.json   # Full experiment configs
    └── eps030_dirichlet.json
```

## Example Config Format

```json
{
  "name": "eps030_dirichlet_longwarmup",
  "description": "Epsilon 0.30 + Dirichlet, long physics warmup",
  "base": "small_model",

  "data": {
    "synthetic_level": 0,
    "synthetic_steps": 10000,
    "regenerate_every": 50
  },

  "training": {
    "epochs": 500,
    "batch_size": 384,
    "parallel_envs": 50,
    "steps": 400
  },

  "model": {
    "use_popart": true,
    "popart_beta": 0.0001,
    "bf16": true
  },

  "mcts": {
    "sims": 24,
    "sims_warmup": 300,
    "sims_min": 4
  },

  "interventions": {
    "epsilon_greedy": 0.3,
    "dirichlet_aggressive": true,
    "sac_entropy": false
  },

  "prediction": {
    "enabled": true,
    "loss_weight": 0.2,
    "physics_warmup_steps": 20000
  },

  "entropy": {
    "adaptive": true,
    "hold_threshold": 0.55,
    "boost_low": 0.4,
    "boost_high": 0.6
  }
}
```

## Implementation

### Phase 1: Config Loader
- Add `--config path/to/config.json` to train_jax_fast.py
- Config values override defaults, CLI args override config
- Log full resolved config to experiment directory

### Phase 2: Config Inheritance
- `"base": "small_model"` loads and merges base config
- Multiple inheritance: `"base": ["small_model", "epsilon_030"]`

### Phase 3: Sweep Integration
- Sweep configs reference JSON files instead of Python dataclasses
- Auto-generate sweep configs from JSON templates

## Benefits

1. **Reproducibility** - exact config saved with each run
2. **Versioning** - configs can be git-tracked
3. **Sharing** - easy to share experiment setups
4. **Iteration** - quick tweaks without long CLI commands
5. **Documentation** - configs are self-documenting

## CLI Usage

```bash
# Run from config
python train_jax_fast.py --config config/experiments/eps030_dirichlet.json

# Override specific values
python train_jax_fast.py --config config/experiments/eps030_dirichlet.json --epochs 100

# Combine configs
python train_jax_fast.py --config config/base/small_model.json --config config/interventions/epsilon_030.json
```

## Priority

Medium - quality of life improvement, not blocking progress
