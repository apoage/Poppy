# QUEUE — the ranked dark-factory work order

The factory's single entry point (planner-maintained). STATUS banners on task files remain the
truth for what's done; this file only orders what's OPEN. The factory takes from the top, skips
gated rows (noting why), and treats FILLER rows as parallel/anytime slices. Last groomed:
**YYYY-MM-DD**.

## Ready — take from the top

| # | Task | Size | Note |
|---|------|------|------|
| 1 | `tasks/<date>-<slug>.md` | S–M | one line of orientation the loop needs before opening it |
| 2 | `tasks/<date>-<slug>.md` | M | dependencies / what it unblocks |

## Filler — anytime, parallel-safe

- `tasks/<date>-<hygiene-or-debt>.md` (independent XS–S slices; safe to interleave)

## Gated — do NOT start (named gate must open first)

| Task | Gate |
|------|------|
| `tasks/<date>-<slug>.md` | DECISIONS **D<n>** open / real data missing / hardware not arrived |

## Grooming rules
Planner re-ranks after each factory run + banners; a row leaves this file only by shipping
(banner) or being explicitly parked. If the factory finds the top row blocked in practice, it
takes the next and records why in its wrap-up.
