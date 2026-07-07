# Changelog

The dark-factory release ritual appends one entry per shipped task (newest first).
Format: `## YYYY-MM-DD — <task> — <one-line what shipped>` + remainders if any.

## 2026-07-07 — factory setup — dark-factory config, QUEUE/DECISIONS surfaces, fixture, repo prune
- `.dark-factory/config.json` (invariants inlined), `tasks/QUEUE.md` (828 stages ranked), `tasks/DECISIONS.md` (D1–D5 open).
- `fixtures/make_fixture.py` → 51MB instant-load fixture (full daily panel + 90d minute slice, top-8 coverage).
- Repo pruned to the reasoning-trading-system core (manifest in `lore/prune_2026-07-07.md`).
