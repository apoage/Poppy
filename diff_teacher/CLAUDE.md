# diff_teacher — Bella

Market generator. Belladonna → eye medicine → "vision of unseen markets."
Skeleton-conditioned flow matching on OHLCV sequences. JAX/Flax.

## Data + shapes

- Training window: `(window_size=400, n_assets=64, 5)` OHLCV
- Generation window: `1440` steps (context-dependent)
- Latent dim: 256 (encoder VAE), float16
- Skeleton: `MarketSkeleton` holds `trend_direction (n_ts, n_assets)`, `vol_regime (n_ts)`, `correlation_clusters (n_assets)`, `alpha_shape (n_ts)`, `regime_sequence` (int indices), sparse `Event` list
- Regimes: 5 (`crash`, `rally`, `sideways`, `volatile`, `calm`) + 1 unconditional pseudo-regime

## Architecture

- **Flow matching**: OT-CFM (optimal transport conditional flow matching), `sigma_min=1e-4`. Single-step inference.
- **Hybrid Mamba/Attention (Jamba-style)**: 1 attention block per 7 Mamba blocks in `HybridMambaFlow`
- **Two model families**:
  - **Conv** (default): 302M params, 1536d, 16L. Dilated causal WaveNet-style. O(n) memory → fits 1440 in 6GB VRAM
  - **Transformer**: 22–100M, 1024d, 12L. O(n²) attention. Rarely used (expensive)

## Subdirectories

| Dir | Contents |
|---|---|
| `skeleton/` | `types.py` (regime/event/skeleton defs), `encoder.py` (TrendEncoder, VolRegimeEncoder, ClusterEncoder, `SkeletonConditioner`), `extraction.py`, `builder.py` |
| `mamba/` | `ssm.py` (selective scan sequential/parallel), `mamba_block.py` (per-step input/output projections), `bimamba.py` (bidirectional + SelfAttention), `mamba_flow.py` (`HybridMambaFlow`, 1:7 attn:Mamba ratio) |
| `evaluation/` | `stylized_facts.py`, `fidelity.py`, `controllability.py`, `report.py` |
| `generation/` | `generator.py` (`MarketGenerator.from_checkpoint`, `.generate`), `streaming.py`. No `VRAMGenerator` |

## Root-level files worth knowing

- `train.py` — entry: `python -m diff_teacher.train --db-path trading_data.db --epochs 100`
- `teacher.py` — `DiffusionTeacher` (two-phase: encoder VAE → flow matching)
- `encoder.py` — `MarketEncoder` (VAE). Distinct from `skeleton/encoder.py` (conditioning). Both used.
- `conv_flow_matching.py` — Conv architecture (separate from transformer `flow_matching.py`)
- `curriculum_bridge.py` — adapter feeding generated markets to external RL agents
- `adversarial_training.py`, `agent_evaluator.py` — tie generation to RL loop
- `optimal_solver.py` — hindsight oracle (peak-valley or DP) for regret scoring
- `constants.py` — `OHLCV_IDX` mapping. Small but load-bearing.

## Gotchas specific to this subtree

- **JAX RNG** — use `jax.random.fold_in()` per step, NOT `split()`. Direct split breaks seedability (HIGH-009 fix, ubiquitous).
- **SSM parallel scan explodes memory** — `associative_scan` creates `(batch, seq, d_inner, d_state)` 4D tensor, ~18GB+. Default `use_parallel=False`; sequential is safe (CRIT-004).
- **BiMamba `sequential_mode=True`** if OOM — processes fwd then bwd separately, halves peak memory (HIGH-004).
- **AdaLN gate init** — `gate = sigmoid(x - 5.0)` so gates start near zero. Critical for stability (LOW-003).
- **Skeleton conditioning** — encoder returns `(gamma, beta, h0)` tuple for FiLM. Must handle `None` for unconditional path (CRIT-002).
- **Checkpoint double-nesting** — params can be `{params: {params: {...}}}` from Flax `TrainState`. `generator.py` unwraps automatically; your code may need to.
- **Gradient checkpointing** — `mamba_flow.py` uses `nn.remat()` per layer. Slower forward, ~50% less peak memory (HIGH-003).
- **MarketDataLoader is not thread-safe** (simple dict cache, HIGH-015).

## Common commands

```bash
python -m diff_teacher.train --db-path trading_data.db --epochs 100
```
