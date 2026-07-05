---
status: accepted
date: 2026-01-13
deciders: Lukas
---

# 0009. Apothecary / medicinal-plants naming theme for components

## Status

Accepted.

## Context

The project crosses several domains (RL planner, generative market model, orchestration tooling, code review side project, on-chain DeFi pipeline). Generic names ("trainer", "generator", "dojo", "reviewer") collide and confuse — multiple components have a "trainer" inside them, multiple have a "generator", etc. Reference in lore notes and task specs would be ambiguous.

A coherent codename scheme was needed. The user picked an apothecary / medicinal-plants metaphor — careful work with powerful tools, neither dark-themed nor whimsical. Each component's plant maps to its function.

## Decision

Components are named after medicinal plants. The mapping (from top-level `CLAUDE.md`):

- **Poppy** (`mcts_trader/`) — TRM + MuZero trader. Dreams via MCTS planning. Production target.
- **Bella** (`diff_teacher/`) — market generator. Belladonna = "beautiful lady"; in herbalism, eye medicine — vision of unseen markets.
- **Foxglove** (`train_easy.py` + `tools/orchestration/e-dojo_v1.py`) — Stage-1 supervised imprint. Cardiotonic — stabilising pre-training phase.
- **Yarrow** (`yarrow/`) — multi-model code review side project. Wound herb — finds and cleans.
- **Aconite** — MEV extraction application (in `~/Iris-DEX/`, `docs/MEV/`).
- **Iris-DEX** (`~/Iris-DEX/`) — Reth archive node + DeFi data pipeline.
- **Bloom** (`bloom/`) — publishing pathway, the apothecary's lab notebook.

## Consequences

Gain:
- Unambiguous references in lore notes and task specs ("Poppy reads from Bella" is clearer than "the agent reads from the generator").
- The metaphor self-documents the relationship between components — Foxglove stabilises Poppy before MCTS goes live; Yarrow cleans wounds in any of them.
- Sufficient namespace headroom: there are far more medicinal plants than components we'll ever name.

Give up:
- A new contributor must learn the mapping. Acceptable: it's a solo project, and the table at the top of `CLAUDE.md` is cheap to consult.
- File and path names mostly do not match the codenames (`mcts_trader/` not `poppy/`, etc.) — historical, and renaming would break too much. The codenames are documentation/discussion-layer, not filesystem-layer.

## References

- Top-level `CLAUDE.md` § Naming (Apothecary Theme).
- Memory `MEMORY.md` § Related projects, § Side projects.
