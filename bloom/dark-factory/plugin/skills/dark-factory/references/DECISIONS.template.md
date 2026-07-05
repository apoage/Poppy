# DECISIONS — the single blocked-on-human surface

One row per call. Recommendation FIRST — the human should be able to answer with one word.
OPEN rows are walls: the factory never guesses past one; the planner seeds rows here instead of
scattering open questions across task files. Answered rows move to DECIDED with the date.

## OPEN

| # | Decision | Recommendation | Context |
|---|----------|----------------|---------|
| D1 | <the call, phrased as a yes/no or pick-one> | <rec + one-line why> | <task file / doc that needs it> |

## DECIDED

| # | Decision | Outcome | Date |
|---|----------|---------|------|
| D0 | <what was asked> | <what was chosen> | YYYY-MM-DD |
