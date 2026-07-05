<!-- Live file from videoED, the project dark-factory was extracted from — kept verbatim as a worked example. -->
# Agentic UI verification playbook

The durable artifact of the agentic-UI-testing sidequest (`tasks/2026-07-02-sidequest-agentic-ui-testing.md`;
stance: memory `videoed-testing-stance`). videoED's UI is exploratory, so verification is **judgment-based**,
not an assertion suite. This playbook is what a **verification subagent** follows after a UI-touching change.
It is *review*, not CI: it produces a **findings list**, never a hard pass/fail (screenshot judgment is noisy).

## The two layers (run in this order)
1. **Deterministic smoke** (`pytest -m ui`, webkit) — a handful of FROZEN invariants: app boots, project
   loads, **zero console errors**, passes render, version badge matches `app/version.py`, L0/L1/L2 present.
   If this fails, STOP and report — that's structural breakage, not a judgment call. Keep this layer TINY;
   never grow it into selector-level assertions (they calcify while the UI is finding its shape).
2. **Agentic visual pass** (this playbook) — boot the app on the fixture (or a Blue project), LOOK at each
   screen, and judge against the CURRENT task file (what was the change supposed to do?).

## Inputs
- The **current task file** — the spec of what the change should produce. Judge against it.
- A loaded project: **Tier A** the fast fixture `fixtures/antminer_s21/` (instant-load, all `.videoED/`
  artifacts baked — no passes run); **Tier B** a real `/Volumes/Blue` project (anti-overfit, unattended, ok
  to be long). Rule: **fixture-green is necessary, Blue-green is sufficient.** Blue is READ-ONLY for the
  harness (memory `videoed-nondestructive`) — never organize-move/nuke, never write outside the project.

## Procedure
1. Boot: `python -m uvicorn app.server:app --port <FREE>` (venv active). Load the project via the real path
   (`POST /api/scan {folder: <abs path>}`) — same path the UI uses, so ingest/gating are exercised.
2. Run the deterministic smoke first. Fail → report breakage, stop.
3. Capture full-page screenshots (webkit) of **every** main screen + extra shots of the screens the change
   touched. (Phase-2's smoke already writes `ui_smoke_*.png` to the scratchpad; reuse or re-capture.)
4. **Judge each screen** (see per-tab checklist). For each, ask:
   - Layout intact? No overlap / clipping / overflow / z-index collisions / broken flex-grid?
   - Empty/missing where the fixture clearly has data? (A blank panel over a full `.videoED/` = a bug.)
   - Does the rendered state **contradict the artifacts on disk**? (e.g. Cut panel shows 0 beats but
     `decisions.json` has 22; Preview offers no `ai_cut` though `aicut/decisions.json` exists.)
   - Does the **change's intended effect** (per the task file) actually appear?
5. Interact where the change warrants: click the new control, round-trip a save (decisions/graphics-plan),
   scrub the preview, toggle a level. Watch the console the whole time (any error = finding).
6. Emit a **findings list**: `{severity, screen, screenshot, what's wrong, expected-vs-seen}`. Rank by
   severity. Empty list = clean. This goes to the orchestrator as advisory review, not an exit code.

## Per-tab checklist (what "good" looks like on the fixture)
- **Run** — pass rows render (count = `P.catalog()`); resolved engine/mode labels visible (STT/fps/LLM —
  memory `videoed-surface-active-state`); AI-cut L0-L3 checkboxes present (L2/L3 experimental), L4 disabled.
- **Structure** — lanes per camera from `sync.json`; per-clip sync status; the LOW/unsynced cameraA (TV
  `.mkv`) is flagged, not hidden.
- **Cut / Decisions** — placed beats with score + chosen/alt cam; pickups (never-filmed) surfaced; counts
  match `decisions.json`.
- **Plan / Graphics** — placed cards, timing source, PLACEHOLDER flags, overlap warnings from
  `graphics_plan.json`.
- **Preview** — program monitor paints; cut selector offers greenlight + ai_cut (both have decisions on the
  fixture); scrubbing loads media/thumbs. NOTE fixture blind spot: the cameraA `.mkv` may not play in
  WebKit `<video>` → thumb fallback is expected, not a bug.
- **Timeline** — the SVG spine renders the cut lanes.
- **Coherence / QC** — coherence warnings from `coherence.json`; video-review panel present (may show the
  "needs api backend" skip state — that's correct on the fixture/claude_code backend).
- **Settings** — resolved backend/model + STT engines shown; version badge matches; blank-key rule (never
  shows the real key).

## When it runs
- After **any** UI-touching change (the delegation-model verification step — memory
  `videoed-delegation-model`): implementation subagent lands → orchestrator spawns a verification subagent
  that runs this playbook on Tier A.
- Before a **release build** of a UI change.
- **Dark-factory**: an unattended overnight Tier-B run on a real Blue project → morning
  findings/divergence report. Record what diverged from the fixture run — that list is the anti-overfit
  payoff; fold fixture blind spots back into `make_fixture.py`.

## Explicitly NOT this playbook's job (stays human — Lukas)
Cut quality, best-take rightness, whether an L2 reorder reads well, color/grade look, real-footage feel —
the subjective/creative. This layer catches **breakage**, not **badness**.

## Out of scope
The packaged `.app` shell (pywebview window, menus, file dialogs) stays a manual build-and-click. No CI
service / GitHub. No pixel-diff/golden images — visual judgment is the agent's, by design.
