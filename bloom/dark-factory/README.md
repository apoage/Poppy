# dark-factory — workshop

> **STATUS (2026-07-05): v0.2.0 SHIPPED to [apoage/apothecary](https://github.com/apoage/apothecary)
> as `plugins/dark-factory/` (commit `e65ced7`).** `plugin/` in this directory is the
> master copy; edit here, then sync into the apothecary clone and push.

## Layout

```
plugin/            ← CANONICAL v0.2 — exactly what ships to apothecary/plugins/dark-factory/
│                     skill (+references incl. config template) · 4 agents · 5 hooks · docs
├── NOTES.md           ← working notes: weak points, overfit inventory, research synthesis,
│                         hook enforcement map — the design record behind v0.2
├── RESEARCH-PROMPT.md ← the deep-research prompt that produced the report
├── compass_*.md       ← the research report (ships as plugin/docs/RESEARCH.md)
│
└── skills/ agents/ sauce/   ← v0.1 videoED extraction, kept as historical reference
                                (vd-* names, prompt-enforced contract, hardcoded paths)
```

## What changed v0.1 → v0.2

1. **Hooks, not honor.** The contract's high-risk rules (single-writer lanes, commit
   needs verification, never push, exit ritual) moved from prompt-discipline into
   PreToolUse/Stop hooks the loop cannot ignore. Armed via `true # dark-factory-arm`,
   fast no-op everywhere else. Gate scenarios tested 37/37.
2. **Variables, not paint.** Every project specific became a `{{variable}}` resolving
   from `.dark-factory/config.json` (template + key-by-key reference ship in the skill).
   videoED survives only in `docs/examples/`.
3. **Roster renamed** `vd-*` → `dark-factory-*`; risk-tiered verification panel;
   HALT-after-3-failures; objective checks outrank LLM judges.

## Publishing flow

```bash
gh repo clone apoage/apothecary /tmp/apothecary
rsync -a --delete --exclude __pycache__ plugin/ /tmp/apothecary/plugins/dark-factory/
# bump version in plugin/.claude-plugin/plugin.json, update marketplace entry if the
# description changed, commit, push
```

**Naming (decided 2026-07-05):** the plugin keeps the **dark-factory** name — the agents
stay `dark-factory-*` (the judge remains a badass cyberpunk villain). **moonflower** is
its **codename**: the night-blooming datura — powerful, and it only opens in the dark —
its entry in the apothecary's garden ledger. Use the codename in theme/lore contexts, the
name in commands and code.
