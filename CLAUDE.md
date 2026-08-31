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

## Agent rules (`.claude/rules/`)

Operating rules for anyone working in this tree — implementers, judges, verifiers, the server
agent. Migrated out of local agent memory 2026-08-31 so subagents can actually read them
(`lore/adr/0008`).

| File | Covers |
|---|---|
| `00-shared-tree-concurrency.md` | **Hard prohibitions.** Never stash/checkout the live tree; never regenerate the gitignored fixture; sub-agents never deploy; no planner writes to shared surfaces while ARMED; the stop gate is blind to async agents. |
| `10-compute-allocation.md` | `safe_run` cgroup caps, VRAM envelope, single verified-idle GPU, tmux launches, JAX settings, keep gpu-server loaded. |
| `20-verification-rig.md` | HEAD-control worktree A/B mechanics and the two traps that make a clean worktree lie. |
| `30-agent-conduct.md` | Repo over summary; a refutation's replacement needs its own panel; transient 401s; lore tone. |

Host inventory, storage, uplink and DB-backup facts: `docs/infrastructure.md`.
Long-form experiment findings and diagnoses: `docs/findings/` (index: `docs/findings/README.md`).

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

### Where tests and verification RUN (operator rule, 2026-08-22)

**Test suites, eval sweeps, and verification runs go on `gpu-server`, on GPU, wrapped — NOT on
workstation CPU.** The workstation keeps orchestration, greps, reads, and syntax checks
(`py_compile`). Full contract, including the wrapper invocation and the "it was only the fixture
DB is not an exemption" clause: `.claude/rules/10-compute-allocation.md` and
`.claude/rules/20-verification-rig.md`.

### Anti-patterns (NEVER)

- Pre-generate training data to npz/pkl then load
- Repeated disk reads in training loop
- Intermediate tensors on disk
- Any compute touching disk when RAM/VRAM available
- **Bare `python -m pytest` / sweeps on the workstation** — wrap them and run them on the server
- **Fanning out parallel subagents that each run suites/sweeps.** Load multiplies invisibly because
  every agent looks small on its own. Factory run 40 put 8 unwrapped agents on the workstation and
  drove load average to 17.6 while `gpu-server` idled. Cap concurrency deliberately.
- **Mutating HEAD in the live tree** (`git stash`/`checkout`/`restore`) or regenerating
  `fixtures/trading_fixture.db` as a side effect — see `.claude/rules/00-shared-tree-concurrency.md`.

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
