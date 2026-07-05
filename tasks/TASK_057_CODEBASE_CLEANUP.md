# TASK_057: Codebase Cleanup & Folder Structure

## Status: COMPLETED (2026-01-11)

## Overview

Reorganize root folder into proper structure. Move files to appropriate directories and update any hardcoded paths in scripts.

## Target Folder Structure

```
Trader/
├── CLAUDE.md                    # Project documentation (root)
├── README.md                    # Project readme (root)
├── requirements.txt             # Dependencies (root)
├── trading_data.db              # Database (root)
├── train_jax_fast.py            # Main training entry point (root)
├── real_trade_jax.py            # Live trading entry point (root)
├── trader_env/                  # Virtual environment
│
├── config/
│   ├── curriculum/              # TASK_056 curriculum configs
│   └── training_schedule.json   # Training schedules
│
├── datasets/
│   └── delistedpairs.csv
│
├── docs/
│   ├── AI_bleed research/
│   └── synthetic_level_*_preview.png
│
├── jax_ckpt_fast/               # All checkpoints (unified)
│   ├── L0_*/
│   ├── L1_*/
│   └── dojo_*/                  # Moved from dojo_checkpoints/
│
├── logs/
│   ├── training_*.csv           # Training logs
│   ├── reports/                 # Analysis reports
│   ├── sweep_results/           # Sweep outputs
│   └── tournament_results/      # Tournament outputs
│
├── lore/                        # Design documents
│
├── mcts_trader/                 # Main package
│   ├── data/
│   ├── encoder/
│   ├── env/
│   ├── jax_agent/
│   └── training/
│
├── scripts/                     # Shell scripts only
│   ├── run_training.sh
│   ├── START_TRAINING.sh
│   └── START_TRAINING_PHASE2.sh
│
├── tasks/                       # Task specifications
│
├── tensor_snapshots/            # Debug snapshots (keep as is)
│
├── tests/                       # Unit tests (pytest)
│
└── tools/                       # Analysis & utility scripts
    ├── analysis/
    │   ├── analyze_sweep.py
    │   └── cross_validate.py
    ├── benchmarks/
    │   ├── measure_pci_bandwidth.py
    │   ├── measure_vram_bandwidth.py
    │   ├── profile_bottleneck.py
    │   └── profile_mcts_detailed.py
    ├── sweeps/
    │   ├── stability_sweep.py
    │   ├── stability_tournament.py
    │   ├── sweep_level1.py
    │   ├── warmup_sweep.py
    │   ├── hyperparam_tournament.py
    │   └── tuning_tournament.py
    ├── orchestration/
    │   ├── dojo_manager.py
    │   ├── dojo_dynamic_curriculum.py
    │   └── train_manager.py
    ├── debug/
    │   └── debug_fx_map.py
    └── testing/
        ├── test_task_implementations.py
        ├── test_batched_selfplay.py
        ├── test_dual_instance.sh
        └── test_dual_instance_15M.sh
```

## Files to Move

### To `tools/analysis/`
| File | From | Notes |
|------|------|-------|
| analyze_sweep.py | root | |
| cross_validate.py | root | |

### To `tools/benchmarks/`
| File | From | Notes |
|------|------|-------|
| measure_pci_bandwidth.py | root | |
| measure_vram_bandwidth.py | root | |
| profile_bottleneck.py | root | |
| profile_mcts_detailed.py | root | |

### To `tools/sweeps/`
| File | From | Notes |
|------|------|-------|
| stability_sweep.py | root | Update checkpoint paths |
| stability_tournament.py | root | Update checkpoint paths |
| sweep_level1.py | root | Update checkpoint paths |
| warmup_sweep.py | root | Update checkpoint paths |
| hyperparam_tournament.py | root | Update checkpoint paths |
| tuning_tournament.py | root | Update checkpoint paths |

### To `tools/orchestration/`
| File | From | Notes |
|------|------|-------|
| dojo_manager.py | root | Update checkpoint paths |
| dojo_dynamic_curriculum.py | root | Update checkpoint paths |
| train_manager.py | root | |

### To `tools/debug/`
| File | From | Notes |
|------|------|-------|
| debug_fx_map.py | root | |

### To `tools/testing/`
| File | From | Notes |
|------|------|-------|
| test_task_implementations.py | root | |
| test_batched_selfplay.py | root | |
| test_dual_instance.sh | root | |
| test_dual_instance_15M.sh | root | |

### To `scripts/`
| File | From | Notes |
|------|------|-------|
| run_training.sh | root | |
| START_TRAINING.sh | root | |
| START_TRAINING_PHASE2.sh | root | |

### To `logs/`
| File/Folder | From | Notes |
|-------------|------|-------|
| reports/ | root | Move entire folder |
| sweep_results/ | root | Move entire folder |
| tournament_results/ | root | Move entire folder |
| tuning_results_*.csv | root | |
| last_run_metrics.json | root | |

### To `jax_ckpt_fast/`
| File/Folder | From | Notes |
|-------------|------|-------|
| dojo_checkpoints/* | root | Merge into main checkpoint dir |

### To `docs/`
| File | From | Notes |
|------|------|-------|
| synthetic_level_*_preview.png | root | |

### To `config/`
| File | From | Notes |
|------|------|-------|
| training_schedule.json | root | |

### To `datasets/`
| File | From | Notes |
|------|------|-------|
| delistedpairs.csv | root | |

## Files to Delete

| File | Reason |
|------|--------|
| miner_capture.pcap | Old debug artifact |
| __pycache__/ | Regeneratable |

## Path Updates Required

After moving files, update hardcoded paths in these scripts:

### tools/sweeps/*.py
```python
# Old
checkpoint_dir = 'jax_ckpt_fast'
results_dir = 'sweep_results'

# New
checkpoint_dir = Path(__file__).parent.parent.parent / 'jax_ckpt_fast'
results_dir = Path(__file__).parent.parent.parent / 'logs' / 'sweep_results'
```

### tools/orchestration/dojo_*.py
```python
# Old
checkpoint_dir = 'dojo_checkpoints'

# New
checkpoint_dir = Path(__file__).parent.parent.parent / 'jax_ckpt_fast'
```

### scripts/*.sh
```bash
# Update relative paths to use $SCRIPT_DIR
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
```

## Implementation Steps

1. **Create directory structure**
   ```bash
   mkdir -p tools/{analysis,benchmarks,sweeps,orchestration,debug,testing}
   mkdir -p logs/{reports,sweep_results,tournament_results}
   ```

2. **Move files** (in order to avoid breaking imports)
   - Move logs/results first (no code dependencies)
   - Move tools second
   - Update paths in moved tools
   - Move scripts last

3. **Update CLAUDE.md** with new folder structure

4. **Test** that all tools still work:
   ```bash
   python tools/testing/test_task_implementations.py
   python tools/sweeps/stability_sweep.py --help
   bash scripts/run_training.sh --help
   ```

5. **Clean up**
   ```bash
   rm miner_capture.pcap
   rm -rf __pycache__
   rmdir dojo_checkpoints  # After merging
   ```

## Verification Checklist

- [ ] All tools import correctly from new locations
- [ ] Checkpoint paths resolve correctly
- [ ] Log outputs go to logs/ folder
- [ ] Shell scripts work from any directory
- [ ] CLAUDE.md updated with new structure
- [ ] No broken symlinks or references

## Rollback Plan

Before moving, create backup:
```bash
git stash  # If uncommitted changes
git checkout -b cleanup-backup
# Or: tar -czf ../Trader_backup.tar.gz .
```
