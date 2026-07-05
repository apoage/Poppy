# `.dark-factory/config.json` — key by key

The single place a project teaches the factory its specifics. Committed, planner-owned
(the lane guard denies the implementer session from editing it while armed). Every
`{{key}}` in the skill and the `dark-factory-*` agents resolves from here.

| Key | What it is | Example |
|---|---|---|
| `project_name` | Short name, used in reports | `"videoED"` |
| `owner` | The human who answers DECISIONS rows | `"Lukas"` |
| `commands.test` | Full test command, green = precondition for every report | `"source .venv/bin/activate && python -m pytest -q"` |
| `commands.lint` | Extra per-file checks the implementer runs on touched files | `["node --check {file}"]` |
| `commands.build` | Release build, `null` = no build step | `"pyinstaller app.spec"` |
| `commands.ui_test` | Deterministic UI smoke layer, `null` if no UI | `"python -m pytest -m ui -q"` |
| `commands.ui_boot` | Headless boot for the visual judge | `"python -m uvicorn app.server:app --port {port}"` |
| `paths.queue` / `paths.decisions` | The two shared surfaces | defaults `tasks/QUEUE.md`, `tasks/DECISIONS.md` |
| `paths.version_file` | What the release ritual bumps, `null` = no versioning | `"app/version.py"` |
| `paths.changelog` | Where ritual entries go | `"CHANGELOG.md"` |
| `paths.fixture` | Small instant-load fixture; the whole verification story leans on it — build one FIRST | `"fixtures/sample_project"` |
| `paths.ui_playbook` | The visual judge's playbook of record | `"tests/ui_smoke_playbook.md"` |
| `paths.design_principles` | The project's own design-system doc the visual judge grades against | `"docs/design.md"` |
| `paths.handoff_dir` | Where dated handoff/validation docs land | `"docs/"` |
| `invariants` | The 5–10 rules a wrong change must never violate. Inlined here — agents read them as law; never referenced by memory slug (dead pointers on fresh installs) | `["never modify source media", "all times = seconds in original clip timeline"]` |
| `protected` | Paths/volumes never written to or run against | `["/Volumes/Blue"]` |
| `ui.enabled` | Whether a visual judge exists for this project | `false` |
| `ui.engine` | Playwright engine, matched to the REAL runtime | `"webkit"` for a pywebview/WKWebView app |
| `lanes.planner_owns` | Dir prefixes only the planner writes | `["docs/", "prototype/", ".claude/"]` |
| `lanes.shared` | Both threads write here (banners/DECISIONS rows for the factory — discipline stays prompt-level inside shared dirs) | `["tasks/"]` |
| `lanes.implementer_doc_exceptions` | Filename substrings the factory MAY write inside planner lanes | `["handoff", "validation"]` |
| `guards.allow_push` | `false` = the bash gate denies every `git push` while armed | keep `false` |
| `guards.require_verdict` | `true` = commit/tag requires a fresh green verdict file | keep `true` |
| `guards.verdict_max_age_minutes` | How fresh the verdict must be at commit time | `60` |
| `guards.halt_after_failures` | Failed fix→verify cycles before the task HALTs to a DECISIONS row | `3` |
| `verification.risk_tiers` | Panel composition per declared task risk (`judge:<lens>`, `flow-verifier`) | see template |

## Design notes

- **Why inline invariants instead of a doc pointer?** Fresh installs have no project
  memory; slugs rot silently. Ten lines of law in the config survive any transplant.
- **Why risk tiers?** A constant 3-judge panel is the wrong cost model — verification
  effort should scale with blast radius (XS filler ≠ strategic change). Tier `low` on
  small mechanical tasks, `high` on anything touching gates, locks, money, or data
  destruction.
- **Why is the verdict freshness-checked?** A verdict older than the code it approves is
  a lie with a timestamp. The gate compares mtimes, not intentions.
