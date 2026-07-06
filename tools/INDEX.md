# Tools Index

**Last Updated**: December 26, 2025

Documentation for all tools in the `tools/` directory.

---

## Quick Reference

### Trading
| Tool | Description |
|------|-------------|
| [paper_trade_vectorbt.py](paper_trade_vectorbt.md) | Paper trading with QuantStats reports |

### Visualization
| Tool | Description |
|------|-------------|
| [visualize_synthetic.py](visualize_synthetic.md) | Preview synthetic data levels (0-3) |
| [visualize_mcts_tree.py](visualize_mcts_tree.md) | Analyze MCTS policy/value metrics |

### Data Management
| Tool | Description |
|------|-------------|
| [download_data.py](download_data.md) | Download Kraken OHLCV data |
| [update_dataset.py](update_dataset.md) | Incremental dataset updates |
| [validate_database.py](validate_database.md) | Validate assets against Kraken |
| [build_fx_database.py](build_fx_database.md) | Build FX rate database |
| [map_training_data.py](map_training_data.md) | Map training data segments |

### Diagnostics
| Tool | Description |
|------|-------------|
| [diagnose_jax_training.py](diagnose_jax_training.md) | Check JAX/GPU setup |
| [diagnose_gpu_mcts.py](diagnose_gpu_mcts.md) | GPU MCTS diagnostics |
| [test_mctx_jax.py](test_mctx_jax.md) | MCTS search benchmark |
| [test_trm_jax.py](test_trm_jax.md) | TRM model test |
| [test_model_init.py](test_model_init.md) | Model initialization test |
| [test_env_init.py](test_env_init.md) | Environment initialization test |
| [test_episode_length.py](test_episode_length.md) | Episode length test |

### Performance
| Tool | Description |
|------|-------------|
| [profile_training.py](profile_training.md) | Profile CPU/GPU/RAM |
| [profile_memory.py](profile_memory.md) | Memory profiling |
| [benchmark_parallelization.py](benchmark_parallelization.md) | Parallel strategy benchmark |

### Training
| Tool | Description |
|------|-------------|
| [run_scheduled_training.py](run_scheduled_training.md) | Execute scheduled training |
| [parallel_epoch_training.py](parallel_epoch_training.md) | Parallel epoch training |
| [generate_mcts_data.py](generate_mcts_data.md) | Generate MCTS data |

### Analysis
| Tool | Description |
|------|-------------|
| [log_parser.py](log_parser.md) | Parse training logs |
| [audit_codebase.py](audit_codebase.md) | Codebase quality audit |
| [audit_documentation.py](audit_documentation.md) | Documentation coverage |

### Database Utilities
| Tool | Description |
|------|-------------|
| [check_assets.py](check_assets.md) | Quick asset check |
| [debug_db_stats.py](debug_db_stats.md) | Database statistics |
| [validate_fx_rates.py](validate_fx_rates.md) | FX rate validation |

### Kraken API
| Tool | Description |
|------|-------------|
| [test_kraken_connection.py](test_kraken_connection.md) | API connection test |
| [check_wsname.py](check_wsname.md) | WebSocket name lookup |
| [debug_kraken_pairs.py](debug_kraken_pairs.md) | Pair metadata debug |

### Deprecated
| Tool | Status |
|------|--------|
| [debug_env.py](debug_env.md) | Use `test_env_init.py` instead |

---

## Common Workflows

### Initial Setup
```bash
python tools/download_data.py --days 7 --count 64
python tools/validate_database.py
python tools/build_fx_database.py
python tools/diagnose_jax_training.py
```

### Training Diagnostics
```bash
python tools/test_mctx_jax.py
python tools/profile_training.py --pid <PID>
python tools/visualize_mcts_tree.py logs/training_default.csv
```

### Paper Trading Audit
```bash
python tools/paper_trade_vectorbt.py \
    --checkpoint jax_ckpt_fast/default/checkpoint_250 \
    --synthetic-level 0 \
    --episodes 5 \
    --generate-report
```

---

## File Naming Convention

Each tool has a corresponding `.md` file with the same name:
- `tool_name.py` → `tool_name.md`
