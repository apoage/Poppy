# Porting the factory to another project

The loop mechanics (queue → delegate → fan-out verify → release ritual → banner) are fully
generic. What you must swap is everything the videoED files hard-code. Checklist:

## 1. Install
```
your-repo/
├── .claude/
│   ├── skills/dark-factory/SKILL.md      ← from skills/
│   └── agents/vd-*.md                    ← from agents/ (rename the vd- prefix if you like)
├── tasks/
│   ├── QUEUE.md                          ← from sauce/templates/QUEUE.template.md
│   ├── DECISIONS.md                      ← from sauce/templates/DECISIONS.template.md
│   └── <date>-<slug>.md                  ← your task files (each = one implementer contract)
```
Run it with `/dark-factory` (optionally under `/loop` for unattended time).

## 2. Swap the project-specific sauce (grep the files for these)

| videoED-specific | Replace with |
|------------------|--------------|
| Hard invariants block (uv venv, `pipeline/tools.py`, EDL timebase, alpha WebM, fail-closed gates, synthetic-always-marked) | YOUR project's non-negotiables — the 5–10 rules a wrong change must never violate |
| `fixtures/antminer_s21` (instant-load baked fixture) | a small real-shaped fixture for your app; build one FIRST, the whole verification story leans on it |
| `tests/ui_smoke_playbook.md` + grading-suite design principles | your UI playbook + your design system's judgment criteria |
| `app/version.py` letter-suffix versioning + CHANGELOG + PyInstaller build | your release ritual |
| Playwright **webkit** (videoED's UI is pywebview = WKWebView) | whatever engine matches your real runtime |
| "not port 8765 — Thunderbird squats it" | check your own port squatters; always bind a free port |
| Blue / real-footage Tier-B overnight runs | your expensive real-data validation, still scheduled-not-polled |
| memory references (`videoed-*`) | delete or repoint; the contract doc in sauce/ carries the substance |

## 3. Keep these verbatim — they are the actual sauce

- Single-writer areas (planner vs implementer file ownership).
- The task file is the contract; implementer never re-derives scope, never gold-plates.
- Fan-out verification with named lenses; refutation-first judges that must list attempted
  attacks when they find nothing.
- Claims in task files are test cases for the flow verifier.
- STATUS banners mapped to CHANGELOG, never guessed; banners beat the queue on conflict.
- DECISIONS rows as walls; recommendation-first phrasing.
- Stop cleanly on human-decision boundaries; never build on guessed decisions.
- Visual judging produces findings with evidence, never bare pass/fail.

## 4. Minimum viable factory

If you want to start smaller: SKILL.md + vd-implementer + ONE vd-judge (correctness lens) +
QUEUE.md gets you 80% of the value. Add the flow verifier when your app writes artifacts worth
inspecting, and the visual judge when you have a UI.
