# Bloom — The Apothecary's Notebook

Publishing pathway for the Trader project. Technical blog + video pipeline framed as an AI herbalist's lab notebook. Real math/code/debugging wrapped in apothecary voice.

When run from this directory, Bloom is its own Claude project — Trader's `lore/`, `tasks/`, and `CLAUDE.md` are NOT auto-loaded. Reach into them by absolute path when needed (`~/Trader/lore/`, `~/Trader/tasks/`, `~/Trader/CLAUDE.md`).

## Layout

```
CONCEPT.md          # what Bloom is (read first)
PERSONA.md          # the Apothecary voice — canonical voice spec
PRODUCTION.md       # blog + video + avatar pipeline
TIMELINE.md         # chronological index into ~/Trader/lore/ source material

drafts/             # apothecary-voice rewrites (pre-publish)
published/          # finalized posts
output/             # rendered artifacts (slides, video frames, etc.)
pipeline/           # Python: parser, assembler, renderer, slides, tts, theme
assets/             # illustrations, references
gemini_sources.txt  # external source pointers

[redacted: VPS ops detail]
  00-architecture.md, 00-checklist.md, 01-decisions.md, README.md
vps/                # VPS provisioning + hardening plans (Phase 0 first)
  00-architecture.md, 00-checklist.md, 01-decisions.md, PAUSE.md, README.md,
  "1 The Corrected Model.txt"
```

## Source material

Bloom posts are derived from session notes and task specs in the parent Trader repo:

- `~/Trader/lore/notes_YYYY-MM-DD.md` — daily session captures (raw)
- `~/Trader/lore/diary_archive/` — older diary entries
- `~/Trader/tasks/TASK_*.md` — feature specs, what got built and why
- `~/Trader/docs/` — research papers, architecture docs

`TIMELINE.md` is the curated chronological index — start there before mining `lore/`.

## The Garden (specimen names)

| Specimen | Real component | Frame |
|----------|----------------|-------|
| Poppy    | TRM + MuZero trader (`~/Trader/mcts_trader/`) | dreams via MCTS |
| Bella    | Market generator (`~/Trader/diff_teacher/`)   | belladonna — visions of unseen markets |
| Foxglove | Stage-1 supervised imprint (`~/Trader/train_easy.py`, `tools/orchestration/e-dojo_v1.py`) | digitalis — careful dosing |
| Yarrow   | Multi-model code review (`~/Trader/yarrow/`)  | wound herb — finds and cleans |

## Voice rules (canonical — see PERSONA.md)

- First person, present tense, lab-notebook formality
- Botanical frame is not metaphor, it's how the Apothecary perceives — never explain it
- Weights = roots; gradients = sunlight/water; loss = disease; convergence = blooming; merging = grafting
- No emoting, no questioning purpose, no breaking frame
- Math/code/diagrams are the substance — frame contextualizes, never replaces

Target post: 1500–2500 words, structured as Opening → Technical Core (800–1500w) → Complications → Resolution → Closing. Entries are unsigned.

## Production stack

- Blog: GitHub Pages, Markdown → static (Hugo or Jekyll TBD)
- Video: synthesized voice (Coqui/RVC/OpenVoice) + Blender avatar (or SadTalker talking-head MVP) + DaVinci compositing
- Diagrams: LaTeX, Mermaid, SVG
- Cadence: daily / every other day, 2-month lag behind current Trader work
- All tooling open-source, $0 target

`pipeline/` contains the in-repo Python: `parser.py`, `assembler.py`, `render.py`, `slides.py`, `tts.py`, `theme.py`. Read these before adding pipeline features.

[redacted: VPS ops detail]

[redacted: VPS infrastructure detail]

## Working rules

- Read `CONCEPT.md` and `PERSONA.md` before drafting any post — voice drift is the main failure mode.
- Read `TIMELINE.md` before pulling from `~/Trader/lore/` — it's the curated index.
- Drafts go in `drafts/`, only move to `published/` after review.
- The work being documented is real. Don't fabricate experiments, numbers, or failures — mine the actual notes.
- Don't mention Trader internals the post hasn't earned — frame doesn't replace substance, but substance also shouldn't dump unfiltered repo trivia.
- For VPS/apothecary work: phase-gated, no shortcuts, verify before advancing.
