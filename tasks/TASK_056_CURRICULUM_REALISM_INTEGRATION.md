# TASK_056: Curriculum Realism Integration

## Status: PLANNING

## Overview

Integrate newly implemented market realism features into the curriculum system. These features should be progressively introduced to prevent overwhelming the agent early in training.

## New Features to Integrate

| Feature | Flag | Purpose | Curriculum Impact |
|---------|------|---------|-------------------|
| **TASK_047** | `--probabilistic-fills` | Distance-based fill probability | Adds uncertainty to limit orders |
| **TASK_049** | `--synthetic-noise` | Anti-memorization noise | Prevents overfitting to patterns |
| **TASK_052** | `--random-fill-priority` | Random asset fill order | Removes index bias |
| **TASK_053** | `--episode-variance` | Variable episode length | Anti-exploit |
| **TASK_055** | `--borrow-rate` | Short position cost | Realistic short selling |
| **TASK_024** | `--lr-schedule` | Learning rate decay | Stable convergence |

## Proposed Curriculum Phases

### Phase 1: Foundation (Levels 0-4)
**Goal**: Learn basic patterns without market friction

```
Level 0 (Sine):      noise=0.0, variance=0,  borrow=0,    prob_fills=OFF
Level 1 (Lagged):    noise=0.1, variance=10, borrow=0,    prob_fills=OFF
Level 2 (Dip):       noise=0.2, variance=20, borrow=0,    prob_fills=OFF
Level 3 (Trends):    noise=0.3, variance=30, borrow=0,    prob_fills=OFF
Level 4 (Rotation):  noise=0.3, variance=30, borrow=0,    prob_fills=OFF
```

### Phase 2: Market Realism (Levels 5-7)
**Goal**: Introduce execution uncertainty

```
Level 5 (Pairs):     noise=0.3, variance=30, borrow=0.02, prob_fills=ON
Level 6 (Triangle):  noise=0.4, variance=40, borrow=0.03, prob_fills=ON
Level 7 (Index):     noise=0.4, variance=40, borrow=0.03, prob_fills=ON
```

### Phase 3: Strategy Mastery (Levels 8-10)
**Goal**: Full market realism

```
Level 8 (Momentum):      noise=0.5, variance=50, borrow=0.05, prob_fills=ON
Level 9 (Mean Rev):      noise=0.5, variance=50, borrow=0.05, prob_fills=ON
Level 10 (Breakout):     noise=0.5, variance=50, borrow=0.05, prob_fills=ON
```

### Phase 4: Microstructure (Levels 11-14)
**Goal**: Expert-level with full friction

```
Level 11-14:  noise=0.5, variance=50, borrow=0.08, prob_fills=ON
```

## Learning Rate Schedule by Phase

| Phase | LR Schedule | Warmup | Min LR | Rationale |
|-------|-------------|--------|--------|-----------|
| Phase 1 | constant | N/A | N/A | Fast exploration |
| Phase 2 | cosine | 20 epochs | 1e-5 | Stable adaptation |
| Phase 3 | cosine | 10 epochs | 1e-6 | Fine-tuning |
| Phase 4 | cosine | 5 epochs | 1e-6 | Minimal change |

## Implementation: Hybrid YAML + Auto-scaling (Options B+C)

### Architecture

```
config/curriculum/
├── defaults.yaml          # Auto-scaling formulas and base values
├── level_00_sine.yaml     # Per-level overrides (optional)
├── level_01_lagged.yaml
├── ...
└── level_14_volatility.yaml

mcts_trader/training/
└── curriculum_config.py   # Config loader with precedence handling
```

### Precedence Order (highest to lowest)

1. **CLI arguments** - Always win
2. **Level YAML file** - `config/curriculum/level_XX.yaml`
3. **Auto-scaling defaults** - Formula-based fallback

### Config Loader Module

```python
# mcts_trader/training/curriculum_config.py

import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class CurriculumConfig:
    """Curriculum level configuration."""
    # Pattern
    level: int
    pattern: str

    # Realism (TASK_047, 049, 052, 053, 055)
    synthetic_noise: float = 0.0
    episode_variance: int = 0
    probabilistic_fills: bool = False
    borrow_rate: float = 0.0

    # Fees
    fee_bps: int = 0
    slippage_bps: int = 0

    # LR Schedule (TASK_024)
    lr_schedule: str = 'constant'
    lr_warmup_epochs: int = 10
    lr_min: float = 1e-6

    # MCTS
    sims: int = 24
    sims_min: int = 4
    sims_warmup: int = 300


def get_auto_defaults(level: int) -> Dict[str, Any]:
    """Auto-scaling defaults based on level."""

    if level <= 4:  # Foundation
        return {
            'synthetic_noise': level * 0.075,
            'episode_variance': level * 7,
            'borrow_rate': 0.0,
            'probabilistic_fills': False,
            'lr_schedule': 'constant',
        }
    elif level <= 7:  # Realism
        return {
            'synthetic_noise': 0.3 + (level - 4) * 0.033,
            'episode_variance': 30 + (level - 4) * 3,
            'borrow_rate': 0.02 + (level - 5) * 0.01,
            'probabilistic_fills': True,
            'lr_schedule': 'cosine',
            'lr_warmup_epochs': 20,
        }
    else:  # Mastery
        return {
            'synthetic_noise': 0.5,
            'episode_variance': 50,
            'borrow_rate': min(0.08, 0.05 + (level - 10) * 0.01),
            'probabilistic_fills': True,
            'lr_schedule': 'cosine',
            'lr_warmup_epochs': 10,
        }


def load_curriculum_config(
    level: int,
    config_dir: Path = Path('config/curriculum'),
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> CurriculumConfig:
    """Load curriculum config with precedence handling.

    Precedence: CLI > YAML file > auto-defaults
    """
    # Start with auto-scaled defaults
    config = get_auto_defaults(level)

    # Layer YAML file if exists
    yaml_path = config_dir / f'level_{level:02d}.yaml'
    if yaml_path.exists():
        with open(yaml_path) as f:
            yaml_config = yaml.safe_load(f)
            # Flatten nested structure
            if 'realism' in yaml_config:
                config.update(yaml_config['realism'])
            if 'fees' in yaml_config:
                config.update(yaml_config['fees'])
            if 'lr_schedule' in yaml_config:
                config.update(yaml_config['lr_schedule'])
            if 'mcts' in yaml_config:
                config.update(yaml_config['mcts'])

    # Apply CLI overrides (only non-None values)
    if cli_overrides:
        for k, v in cli_overrides.items():
            if v is not None:
                config[k] = v

    config['level'] = level
    return CurriculumConfig(**config)
```

### Example YAML Config File

```yaml
# config/curriculum/level_05_pairs.yaml
level: 5
pattern: pairs_trading
description: "Cointegrated pairs with market realism"

realism:
  synthetic_noise: 0.35
  episode_variance: 35
  probabilistic_fills: true
  borrow_rate: 0.025

fees:
  fee_bps: 10
  slippage_bps: 2

lr_schedule:
  type: cosine
  warmup_epochs: 25
  min_lr: 1e-5

mcts:
  sims: 24
  sims_min: 8
  sims_warmup: 200

# Optional: level-specific thresholds
promotion:
  min_alpha: 1.5
  min_epochs: 100
```

### CLI Integration

```python
# In train_jax_fast.py argparse

parser.add_argument('--curriculum-config', type=str, default='config/curriculum',
    help='Path to curriculum config directory')
parser.add_argument('--no-curriculum-defaults', action='store_true',
    help='Disable auto-scaling, use only explicit args')

# Usage in main():
if args.synthetic_level is not None and not args.no_curriculum_defaults:
    curriculum = load_curriculum_config(
        level=args.synthetic_level,
        config_dir=Path(args.curriculum_config),
        cli_overrides={
            'synthetic_noise': args.synthetic_noise if args.synthetic_noise != 0 else None,
            'episode_variance': args.episode_variance if args.episode_variance != 0 else None,
            'borrow_rate': args.borrow_rate if args.borrow_rate != 0 else None,
            'probabilistic_fills': args.probabilistic_fills if args.probabilistic_fills else None,
            'lr_schedule': args.lr_schedule if args.lr_schedule != 'constant' else None,
        }
    )
    # Apply curriculum config to args
    args = apply_curriculum_to_args(args, curriculum)
```

### Usage Examples

```bash
# Use auto-defaults for level 5
python train_jax_fast.py --synthetic-level 5

# Use YAML config (if exists) + auto-defaults
python train_jax_fast.py --synthetic-level 5 --curriculum-config config/curriculum

# Override specific parameter (CLI wins)
python train_jax_fast.py --synthetic-level 5 --borrow-rate 0.0

# Disable all curriculum defaults (explicit args only)
python train_jax_fast.py --synthetic-level 5 --no-curriculum-defaults \
    --synthetic-noise 0.5 --probabilistic-fills
```

## Migration Path

1. **Immediate**: Add new flags to existing commands (manual)
2. **Short-term**: Implement `get_curriculum_defaults()` function
3. **Medium-term**: Integrate into `dojo_dynamic_curriculum.py`
4. **Long-term**: Config file system for complex curriculum definitions

## Testing Requirements

- [ ] Verify each level's defaults produce stable training
- [ ] Test checkpoint compatibility across curriculum transitions
- [ ] Validate metric normalization with variable episode lengths
- [ ] Benchmark training speed with probabilistic fills enabled

## Files to Modify

- `train_jax_fast.py`: Add `--use-curriculum-defaults` flag
- `dojo_dynamic_curriculum.py`: Integrate realism parameters
- `mcts_trader/training/env_factory.py`: Accept curriculum defaults
- `CLAUDE.md`: Document new curriculum structure

## Related Tasks

- TASK_047: Probabilistic fills (IMPLEMENTED)
- TASK_049: Synthetic noise (IMPLEMENTED)
- TASK_052: Random fill priority (IMPLEMENTED)
- TASK_053: Episode variance (IMPLEMENTED)
- TASK_055: Borrow cost (IMPLEMENTED)
- TASK_024: LR schedule (IMPLEMENTED)
