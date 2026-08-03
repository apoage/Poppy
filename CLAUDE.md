# CLAUDE.md

## Environment

Two execution environments share this repo:

- **Workstation** (detected by the presence of `.env-workstation` at repo root) — full
  context: `lore/`, `tasks/`, `bloom/`, `yarrow/`, `docs/` all live here.
- **Server agent on gpu-server** — `.env-workstation` does NOT exist. Many directories
  referenced below will be missing; that is intentional. **If you are reading this on
  the server, stop and read `SERVER.md` first** — it overrides the rules below.

---

**I am an expert ML engineer and programmer.** I read files before editing. I trace data flow before making changes.

Component-specific facts live in per-directory CLAUDE.md files (auto-loaded when working in that subtree). This file is for cross-cutting rules only.

## Components

| Name | Role | Location | Subtree CLAUDE.md |
|------|------|----------|-------------------|
| Poppy | TRM + MuZero trader — ACTIVE BUILD: vector-action rebuild (TASK_828) | `mcts_trader/` | [mcts_trader/CLAUDE.md](mcts_trader/CLAUDE.md) |
| Bella | Market generator (resident, not load-bearing — DECISIONS D5) | `diff_teacher/` | [diff_teacher/CLAUDE.md](diff_teacher/CLAUDE.md) |
| Foxglove | Stage-1 supervised imprint — RETIRING (TASK_829, D5) | `train_easy.py` + `tools/orchestration/e-dojo_v1.py` | [tools/orchestration/CLAUDE.md](tools/orchestration/CLAUDE.md) |

Dark factory runs this repo's implementation queue: `tasks/QUEUE.md` + `tasks/DECISIONS.md`,
config `.dark-factory/config.json`. This interactive session = planner lane.
Research line design: `docs/pathway.md` (rung ladder of experiments; verdicts →
`experiments/LEDGER.md`; rung verdicts gate GPU spend).
Yarrow/bloom/prompt-lib were pruned 2026-07-07 (`lore/prune_2026-07-07.md`).

## Top-level layout

```
train_jax_fast.py             # Poppy training entry (router)
train_phase2.py               # Poppy phase-2 trainer
train_easy.py                 # Foxglove entry (retiring — TASK_829)
real_trade_jax.py             # Live trading
trader/                       # numpy CTA engine (eval bar + factory test baseline)

tools/DS24/                   # Training dashboard (localhost:8050)
tools/benchmarks/             # Performance testing
tools/benchmark_easy.py       # Canonical eval scorecard
tools/map_training_data.py    # Visualize data coverage & gaps

fixtures/                     # dark-factory verification fixture (make_fixture.py)
config/benchmark_periods.json # Test periods
```

## Efficiency Philosophy

**Push for performance. Respect the memory hierarchy. No wasteful I/O.**

### Memory Hierarchy (Sacred to Disposable)

| Tier | Resource | Use For | Rules |
|------|----------|---------|-------|
| 1. GPU Cache | L1/L2 | JIT kernels | LEGENDARY — never pollute |
| 2. VRAM | GPU memory | Active tensors, compute | Use freely if available |
| 3. RAM | System memory | Batched data, frequent access | Second-class data lives here |
| 4. Disk | SSD/HDD | Control data, configs, tables | Load to RAM before compute |
| 5. Stream | Disk → RAM | Data too large for RAM | Last resort, adds latency |

### Placement rules

- Training data → generate on-the-fly in VRAM (never pre-generate to disk)
- Frequently needed → keep in RAM, batch transfers to VRAM
- Control/config → disk OK, but load to RAM for compute
- Large datasets → stream if doesn't fit, minimize disk touches
- Checkpoints, logs/metrics → disk (necessary)

### CPU/GPU split

If compute is CPU-effective + bandwidth available + frees GPU → use CPU + RAM.
- Price generation (SyntheticMarket) → CPU/RAM, transfer to VRAM
- Preprocessing, normalization → CPU while GPU trains
- Oracle path solving (if CPU-bound) → CPU parallel to GPU backward
- Logging, metrics aggregation → CPU (never burn GPU on this)

Goal: keep GPU saturated with gradient compute, offload everything else.

### Anti-patterns (NEVER)

- Pre-generate training data to npz/pkl then load
- Repeated disk reads in training loop
- Intermediate tensors on disk
- Any compute touching disk when RAM/VRAM available

## JAX Pitfalls

- No `jax.lax.cond` inside `vmap` → use `jnp.where`
- Guard division: `jnp.maximum(x, 1e-10)`
- `jnp.where` evaluates both branches
- Use `static_argnames` for configs in JIT
- Dynamic shapes in `lax.scan` → pad to static `max_len`
- Use `threading.RLock()` not `Lock()` for reentrant contexts
- Add `jax.block_until_ready()` before device-to-host transfers

## Naming (Apothecary Theme)

Medicinal plants, not dark — careful work with powerful tools.
- **Poppy**: dreams via MCTS planning
- **Bella**: vision of unseen markets (belladonna = eye medicine)
- **Foxglove**: cardiotonic — stabilizing pre-training phase
- **Yarrow**: wound herb — finds and cleans

## Context & Documentation

| Location | Purpose | Update Frequency |
|----------|---------|------------------|
| `lore/notes_YYYY-MM-DD.md` | Session notes — mid-session captures, pre-compact checkpoints, not end-of-day logs | During sessions |
| `tasks/TASK_XXX_*.md` | Feature specs & progress | Per task |
| `docs/` | Research papers, architecture docs, setup guides | Reference |
| Subtree `CLAUDE.md` | Component-specific facts | When component changes |

**Check `lore/` and `tasks/` at session start — do not rely on stale summaries in this file or in memory.**

## Activate

```bash
source trader_env/bin/activate
```

Component-specific commands live in the subtree CLAUDE.md files.

---
*Details in `lore/` notes, task specs in `tasks/`, component facts in subtree `CLAUDE.md` files.*
