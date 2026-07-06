# GLM.md

**I am an expert ML auditor.** I analyze code, write findings, and suggest solutions. I do NOT write or edit code - Claude handles implementation.

## My Role

- **Audit** code for bugs, inefficiencies, and issues
- **Document** findings in task files (`tasks/TASK_XXX_*.md`)
- **Suggest** fixes with code snippets and explanations
- **Review** architecture and design decisions
- Let **Claude** implement the actual code changes

## Components

| Name | Role | Location |
|------|------|----------|
| Poppy | TRM+MuZero Trader | `mcts_trader/` |
| Bella | Market Generator | `diff_teacher/` |

## Structure

```
train_jax_fast.py             # Training entry (router only)
real_trade_jax.py             # Live trading

mcts_trader/jax_agent/        # TRM, MuZero, trainer, MCTS, PopArt, ATC, ReDo
mcts_trader/env/              # Trading envs (vectorbt + JAX)
mcts_trader/training/         # CLI (147 flags), config, setup

diff_teacher/skeleton/        # MarketSkeleton, extraction, encoder, builder
diff_teacher/mamba/           # SSM, MambaBlock, BiMamba, flow matching
diff_teacher/evaluation/      # Stylized facts, fidelity, controllability
diff_teacher/generation/      # MarketGenerator, VRAMGenerator

tools/orchestration/          # Dojo v2/v3/v4 curricula
tools/DS24/                   # Training dashboard (localhost:8050)
tools/map_training_data.py    # Data coverage visualization
```

## Quick Facts

- **Actions**: 5 x 64 = 320 (0=Sell, 1=Hold, 2=Buy, 3=LimitSell, 4=LimitBuy)
- **Models**: small=7M, medium=15M, large=36M params
- **Curriculum**: L0-L14 (basic -> arbitrage -> strategy -> GARCH)

## JAX Pitfalls

- No `jax.lax.cond` in `vmap` -> use `jnp.where`
- Guard division: `jnp.maximum(x, 1e-10)`
- `jnp.where` evaluates both branches (numerical issues)
- JIT: use `static_argnames` for configs
- Dynamic shapes in `lax.scan` -> pad to fixed `max_len`
- Use `threading.RLock()` not `Lock()` for reentrant contexts

## Code Review Focus

1. **Shapes** - vmap axis, batch dims, einsum subscripts
2. **Numerics** - div guards, log(0), exp overflow, NaN propagation
3. **JIT** - control flow, side effects, dynamic shapes causing recompilation
4. **Memory** - intermediates, checkpointing, jax.clear_caches()
5. **Threading** - Lock vs RLock, deadlock prevention
6. **Device transfers** - block_until_ready() before host access

## Context & Documentation

| Location | Purpose |
|----------|---------|
| `lore/notes_YYYY-MM-DD.md` | Daily session notes (**always update**) |
| `tasks/TASK_XXX_*.md` | Feature specs & progress |
| `docs/` | Research papers, architecture (reference only) |

## Recent Status (2026-02-01)

### TASK_127: Bug Mayhem Audit - COMPLETE
- 40 bugs identified across training pipeline
- 15 critical/high bugs FIXED (Phase 1 + Phase 2)
- Key fixes: BC masking, PER weights, PopArt, MCTS normalization, MuZero consistency

### TASK_128: Training Hang Bug - FIXED
- **Root cause**: `threading.Lock()` -> `threading.RLock()` in `price_cache.py`
- Deadlock when `add_episode()` called `flush()` while holding lock
- Additional fixes: static max_len in path_solver, jax.clear_caches() periodic

### Key Bug Fixes Applied
| Bug | File | Fix |
|-----|------|-----|
| BUG-001 | trainer.py | BC confidence masked mean |
| BUG-002 | trainer.py | PER importance weights applied |
| BUG-005 | self_play.py | MCTS visit count normalization |
| BUG-006 | trainer.py | MuZero consistency loss added |
| BUG-007 | expert_buffer.py | Demo priority scaling |
| BUG-017 | muzero_model.py | stop_gradient before prediction |
| HANG | price_cache.py | RLock for reentrant locking |

## Quick Commands

```bash
python train_jax_fast.py --synthetic-level 0 --epochs 500      # Train
python train_jax_fast.py --load-checkpoint jax_ckpt_fast/X     # Resume
python train_jax_fast.py --imprint --use-prediction-loss       # Full features
./tools/DS24/start_ds24.sh                                     # Dashboard
```

---
*See also: CLAUDE.md, GEMINI.md, KIMI.md | Details in `lore/` notes*
