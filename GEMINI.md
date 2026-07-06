# GEMINI.md

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
mcts_trader/jax_agent/        # TRM, MuZero, trainer, MCTS, PopArt, ATC, ReDo
mcts_trader/env/              # Trading envs (vectorbt + JAX)
mcts_trader/training/         # CLI (147 flags), config, setup

diff_teacher/skeleton/        # MarketSkeleton, extraction, encoder, builder
diff_teacher/mamba/           # SSM, MambaBlock, BiMamba, flow matching
diff_teacher/evaluation/      # Stylized facts, fidelity, controllability
diff_teacher/generation/      # MarketGenerator, VRAMGenerator

tools/orchestration/          # Dojo v2/v3/v4 curricula
tools/DS24/                   # Training dashboard
tools/map_training_data.py    # Data coverage visualization
```

## Quick Facts

- **Actions**: 5 × 64 = 320 (0=Sell, 1=Hold, 2=Buy, 3=LimitSell, 4=LimitBuy)
- **Models**: small=7M, medium=15M, large=36M
- **Curriculum**: L0-L14 (basic → GARCH)

## JAX Pitfalls

- No `jax.lax.cond` in `vmap` → `jnp.where`
- Division: `jnp.maximum(x, 1e-10)`
- `jnp.where` evaluates both branches
- JIT: `static_argnames` for configs
- Dynamic `lax.scan` lengths → pad to static `max_len`
- Use `RLock()` not `Lock()` for reentrant contexts

## Code Review Focus

1. **Shapes** - vmap axis, batch dims, einsum subscripts
2. **Numerics** - div guards, log(0), exp overflow
3. **JIT** - control flow, side effects, dynamic shapes
4. **Memory** - intermediates, checkpointing
5. **Contracts** - None handling, types
6. **Silent bugs** - wrong axis, swapped args

## Context & Documentation

| Location | Purpose |
|----------|---------|
| `lore/notes_YYYY-MM-DD.md` | Daily session notes (**always update**) |
| `tasks/TASK_XXX_*.md` | Feature specs & progress |
| `docs/` | Research papers, architecture (reference only) |

## Current Status (2026-02-01)

| Task | Status |
|------|--------|
| TASK_127 Bug Mayhem | COMPLETE - 15 bugs fixed |
| TASK_128 Training Hang | FIXED - RLock in price_cache.py |
| TASK_125 Expert Imprinting | COMPLETE |
| TASK_114 Publication | pending |
| TASK_116 Foundation | research |

### Key Fixes Applied
- BC confidence masking, PER weights, MuZero consistency loss
- MCTS normalization, stop_gradient, demo priority scaling
- Static max_len in path_solver, periodic jax.clear_caches()

---
*See also: CLAUDE.md, GLM.md, KIMI.md | Details in `lore/` notes*
