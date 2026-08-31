# apothecary marketplace

> The user's public Claude Code plugin marketplace (github.com/apoage/apothecary) — where skills/plugins ship; dark-factory (codename moonflower) lives there

*Migrated from local agent memory 2026-08-31 (`project_apothecary_marketplace.md`) so subagents and the server agent
can read it — see `lore/adr/0008`. Content verbatim; only frontmatter and wiki-links changed.*
**Apothecary** = the user's public Claude Code plugin marketplace at
`github.com/apoage/apothecary` (repo was renamed from `ClaudeC-things`; old slug
redirects). Shelf as of 2026-07-05: `feverfew` (ML/quant experiment discipline),
`erythroxy` (JAX/GPU perf), `chandoff` (pre-compact rubric), `dark-factory` v0.2.0
(autonomous implementer loop — skill + 4 agents + 5 hooks, hook-enforced two-thread
contract, `{{variable}}` config layer).

Naming decision 2026-07-05: **dark-factory keeps its name** (user likes the cyberpunk
agent names — `dark-factory-judge`); **moonflower** (night-blooming datura) is the
**codename** for apothecary-theme/lore contexts.

**Master copy MOVED (2026-07-07):** `bloom/` was pruned from `~/Trader`, so
`~/Trader/bloom/dark-factory/plugin/` no longer exists and neither do
`bloom/dark-factory/README.md` (the rsync flow) or `bloom/dark-factory/NOTES.md` (design
record). The surviving master is the marketplace repo itself, `github.com/apoage/apothecary`;
a backup of the pruned tree exists per `lore/prune_2026-07-07.md`.

**Version drift observed 2026-08-31:** `dark-factory` is at **0.3.7** in the marketplace and in
`~/splatworld`, `~/openBella`, `~/factory-multiline`, but `~/Trader` is still pinned to **0.3.1**
(`claude plugin list`). The "v0.2.0" above is the 2026-07-05 shelf snapshot, not current.
