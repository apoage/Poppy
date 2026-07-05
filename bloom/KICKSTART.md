# Bloom — Kickstart for a fresh Claude session

When you first run `claude` from `~/Trader/bloom/`, this is a brand-new project (`-home-lukas-Trader-bloom`) with empty memory. Use this file to seed it.

## On first session — ask Claude to save these to memory

Paste this to Claude as your first message:

> Read `CLAUDE.md`, `CONCEPT.md`, `PERSONA.md`, `PRODUCTION.md`, and `TIMELINE.md`. Then save to your auto-memory:
>
> 1. **project memory** — Bloom is the publishing pathway for the Trader project (`~/Trader/`). It is a technical blog + video pipeline framed as the Apothecary's lab notebook. Real math/code, voice wrapper. Cadence: every 1–2 days, 2-month lag behind current Trader work.
> 2. **project memory** — Source material lives in `~/Trader/lore/notes_*.md` (daily session notes), `~/Trader/lore/diary_archive/` (older), `~/Trader/tasks/TASK_*.md` (specs), `~/Trader/docs/` (research). `TIMELINE.md` here is the curated chronological index — use it before mining `lore/` directly.
> 3. **project memory** — Bloom is a subfolder of the Trader repo. Trader's `CLAUDE.md`, `lore/`, `tasks/`, and Trader-project memory are NOT auto-loaded here. Reach in by absolute path when needed.
[redacted: VPS ops detail]
> 5. **feedback memory** — Voice canon is in `PERSONA.md`. Botanical frame is not metaphor — never explain it, never break frame, no emoting, no purpose-questioning. Math/code is the substance; frame contextualizes only.
> 6. **feedback memory** — Specimen mapping (Poppy=TRM+MuZero trader, Bella=market generator, Foxglove=stage-1 supervised, Yarrow=code review). Use these names in posts; reach to the real components in `~/Trader/<name>/` only when needed for accuracy.
> 7. **user memory** — User is the author of the Trader project (<redacted-email>). Bloom is their personal publishing pathway, not a client deliverable. Prefer terse, no trailing summaries.
> 8. **reference memory** — Pipeline Python lives in `bloom/pipeline/` (`parser.py`, `assembler.py`, `render.py`, `slides.py`, `tts.py`, `theme.py`). Read before extending.
>
> Then confirm what you saved and stop.

## What NOT to seed

Don't ask Claude to memorize:
- File listings or directory structures (re-derivable, rot fast)
- Current draft contents or specific post numbers (state changes daily)
- Trader-project technical details (read `~/Trader/CLAUDE.md` on demand instead)
- The full contents of CONCEPT/PERSONA/PRODUCTION (they're in-repo — Claude reads them when needed)

## Recurring session-start checklist (for Claude)

At the start of each Bloom session:

1. Check `drafts/` — what's in flight?
2. Check `published/` — what's most recently shipped (sets the next-post anchor)?
3. Glance at `TIMELINE.md` — where in the chronology are we?
4. If touching infra: check `apothecary/00-checklist.md` and `vps/00-checklist.md` for current phase; check `vps/PAUSE.md` for holds.
5. Trader changes since last session: `git -C ~/Trader log --since="<last session>" --oneline` if relevant for new posts.

## Useful absolute paths

```
~/Trader/CLAUDE.md                              # Trader cross-cutting rules
~/Trader/lore/                                  # raw session notes
~/Trader/tasks/                                 # feature specs
~/Trader/mcts_trader/CLAUDE.md                  # Poppy facts
~/Trader/diff_teacher/CLAUDE.md                 # Bella facts
~/Trader/tools/orchestration/CLAUDE.md          # Foxglove / dojos
~/Trader/yarrow/CLAUDE.md                       # Yarrow facts
~/.claude/projects/-home-lukas-Trader/memory/   # Trader project memory (read-only ref)
```
