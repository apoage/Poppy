<!-- Live file from videoED, the project dark-factory was extracted from — kept verbatim as a worked example. -->
# QUEUE — the ranked dark-factory work order

The factory's single entry point (planner-maintained). STATUS banners on task files remain the
truth for what's done; this file only orders what's OPEN. The factory takes from the top, skips
gated rows (noting why), and treats FILLER rows as parallel/anytime slices. Last groomed:
**2026-07-04** (post ladder-completion + verification sweep).

## Ready — take from the top

| # | Task | Size | Note |
|---|------|------|------|
| 1 | `2026-07-04-diag-report.md` | S–M | "Report a problem" bug-report flow; tee pass logs first |
| 2 | `2026-07-04-silence-trim.md` | S–M | VERIFY-FIRST: L0's "trim dead air" may be spec fiction — build or correct docs |
| 3 | `2026-07-04-l4-polish.md` | M | 5 slices; ai_cut LUT gap (item 4) is the highest-value single slice |
| 4 | `2026-07-04-overlay-producer.md` | M–L | the ANIMATION lane's missing producer; unblocks l4-polish item 3 (PiP) |
| 5 | `2026-07-03-pro-formats.md` Rung 2 | S–M | TC-jam sync (Rung 3 docs note rides along) |
| 6 | `2026-07-03-agent-mcp-surface.md` | M | the harness admits other agents; pairs with library work |
| 7 | ui-workspace: **spine phase 2** | L | CAM lanes · ENERGY curve · editable — the vibe-cutting on-ramp; biggest open build |
| 8 | sidequest **Phase 4**: overnight Blue UI-harness run | one-shot | schedule overnight/idle, morning findings doc |

## Filler — anytime, parallel-safe

- `2026-07-04-hygiene-slices.md` (6 independent XS–S slices; slice 3 coordinates with #4 above)

## Gated — do NOT start (named gate must open first)

| Task | Gate |
|------|------|
| `2026-07-03-streamer-usecase.md` (as-is experiment first) | recordings ORDERED, not arrived |
| library-gated L4 payoffs (b-roll retrieval quality, steer-cap tuning) | a REAL b-roll/meme library (Lukas curates) |
| `2026-07-03-fps-resolver-unify.md` remainder | 60fps footage |
| Moon landing (`2026-07-02-landings.md`) | DECISIONS **D3** |
| D6 portability fix | DECISIONS **D6** (one-word yes queues it) |
| CJF L3 real-content bonus validation | CJF project folder path from Lukas |
| `2026-07-03-emotion-orbit.md` · visual-sync rungs 2–3 · `2026-07-03-palmier-study.md` (planner) | wild-ideas shelf — explicit Lukas pull only |

## Grooming rules
Planner re-ranks after each factory run + banners; a row leaves this file only by shipping
(banner) or being explicitly parked. If the factory finds the top row blocked in practice, it
takes the next and records why in its wrap-up.
