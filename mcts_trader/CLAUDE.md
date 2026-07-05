# mcts_trader — Poppy

TRM + MuZero trader. JAX/Flax. Production target for live trading.

## Architecture (do not suggest transformer/attention changes to reasoning or dynamics)

- **TRM (Tiny Recursive Model)** — NOT a transformer
  - Encoder: cross-asset attention (the ONLY attention in the model)
  - Reasoning: MLP loops (Dense + LayerNorm + GELU) in H×L cycles
  - Dynamics: 2-layer MLP for MCTS state prediction
- **Actions**: 5 × 64 assets = 320 (0=Sell, 1=Hold, 2=Buy, 3=LimitSell, 4=LimitBuy)
- **Model sizes**: small=7M · medium=15M · large=36M params
- **Value head**: PopArt mandatory, 401 bins over [-10, +10]. Disabling PopArt requires range adjustment.
- **MCTS**: Gumbel MuZero default (via Google `mctx`). Standard MuZero needs Dirichlet noise tuning.

## Subdirectories

| Dir | Purpose |
|---|---|
| `jax_agent/` | Full learning pipeline: TRM encoder, MuZero networks, MCTS search, PopArt, ReDo, ATC, differential rewards, path solvers, replay buffer, schedules |
| `env/` | Trading envs. `jax_trading_env.py` is primary (vmap-safe, pure, immutable). `vectorbt_env.py` is legacy. `jax_encoder.py` — 27-channel market encoder |
| `training/` | CLI (`cli.py`), config (`config.py` — central `TrainingConfig` dataclass), `self_play.py` (serial/batched/JAX variants), `setup.py`, `safe_checkpoint.py`, `evaluator.py`, `env_factory.py`, `tree_search_setup.py`. **No main() here** — invoked from parent `train_manager.py` |
| `data/` | DB, Kraken client, market history, streaming, synthetic GPU generators |
| `exchange/` | Paper trading simulator, Kraken adapter |
| `config/` | Fees, assets, modes |
| `utils/` | JAX/NumPy compat, memory checks, transforms |
| `agent/` | **LEGACY** (PyTorch TRM + wrapper). Superseded by `jax_agent/`. Do not edit. |
| `encoder/` | **DEPRECATED** (numpy indicators). Superseded by `env/jax_encoder.py`. Do not edit. |

## Entry points

- **Training**: `training/cli.py::parse_args()` → `TrainingConfig`. Invoked from parent `train_manager.py` or `train_jax_fast.py`
- **Model**: `jax_agent/muzero_model.py::MuZeroNetworks`; TRM: `trm_flax.py::TinyRecursiveTradingModel`
- **MCTS**: `jax_agent/mctx_search.py::MuZeroMCTXSearch`
- **Self-play**: `training/self_play.py::run_self_play{,_batched,_jax}`
- **Env**: `env/jax_trading_env.py::reset/step` (pure JAX)

## Gotchas specific to this subtree

- **Parallel arrays, not dicts** — position state is `pos_active[i]`, `pos_side[i]`, `pos_size[i]`. Immutable. No in-place mutation.
- **No Python control flow in JAX code** — use `jnp.where`, `jax.lax.switch`. `step()` / `reset()` are pure, vmap-safe.
- **VALUE_SUPPORT** in `jax_agent/__init__.py` must match training targets. If PopArt off, raw range can hit [-750, +750].
- **Batch axis invariant** — encoder/dynamics/reward head expect leading batch dim (vmap assumption).
- **Encoder v2 vs v3** — `trm_encoder_v2.py` (flatten window→Dense) vs v3 (minGRU temporal). v3 is newer, preserves temporal ordering. Check which the active checkpoint uses before edits.
- **ReDo + ATC** are stability modules (TASK_042, TASK_037–043); not core to TRM but wired into training. Path solver + differential rewards (TASK_119, TASK_121) mandatory when curriculum is enabled.

## Common commands

```bash
source trader_env/bin/activate
python train_jax_fast.py --synthetic-level 0 --epochs 500            # train
python train_jax_fast.py --load-checkpoint jax_ckpt_fast/<run>        # resume
```
