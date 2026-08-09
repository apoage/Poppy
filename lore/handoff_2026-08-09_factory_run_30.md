# Factory run 30 handoff — TASK_883R SHIPPED (the D36 re-cut lands clean)

**Date:** 2026-08-09 · **Harness:** Kimi Code (k3) — first factory run on this harness; all
subagents inherit the session model (no per-spawn override exists on Kimi), so the
`models.*` escalation policy was advisory-only this run.

## Shipped

| Version/tag | Item | Verification |
|---|---|---|
| `task883r-universe-identity-fail-loud` (`2ea68c2`) | TASK_883R — `eval_vector.py` fails CLOSED on universe identity: refuses or disclaims; the positive label does not exist | High-tier panel, 3 cycles: 2 MAJORs + 5 MINORs found and closed, final fail-closed re-verify clean (10 attacks). 883R suite (all gates + 12-row state table + 11 fix legs) · 882 suite (4 checks + 2b) · 861 smoke 11 legs canon bit-identical · selftest 12/12 · identical-argv byte-identity A/B PASS on both pin paths, 3 independent runs |

What shipped, one paragraph: gates (i) refusal-before-panel (strace-proven zero I/O),
(ii) order-exact mismatch refusal from BOTH pin sources, (iii) the hatch with the permanent
`universe_pinned: false` stamp, (iv) byte-identity vs HEAD, the D36 realized-panel equality
assertion, and the inline `ledger_row` caveat naming the mechanism. `universe_pinned: true`
is not emitted anywhere; `certified: false` on every record. 861 smoke legs now pass the
hatch explicitly (a fresh-init teacher has no training universe — the honest flag); 882's
CHECK 2 became a refusal assertion with its premise re-proven in-process (2b).

## The two MAJORs (both fail-closed lens, both closed)

1. **Duplicate symbol in `--symbols` realized faithfully** — slots 0/1 both XBT/USD,
   invisible to the realized-panel assertion (realized == requested), graded to exit 0
   while the identical frame via pin file refused. Fixed cycle 2 (exact-string distinctness).
2. **The cycle-2 fix judged identity by exact string; every loader groups by
   `normalize_symbol_key`** — `XBT/USD` + `xbt/usd` sailed through, and a COLLUDING
   provenance earned `verified_against_checkpoint_provenance: true` on the collapsed
   universe. Fixed cycle 3: normalized distinctness at both pin sentinels
   (`enforce_universe_pin` + the pre-existing `_load_frame_provenance`), importing the
   loaders' own identity function, never a copy.

The cycle-2→3 shape is the run's lesson: **a guard must judge identity by the same
function the loaders judge identity by.** A guard that re-implements "same symbol?"
differently from the code it guards is a hole by construction.

## Remainders for the planner

- **D37 seeded (OPEN):** a NONEXISTENT `--checkpoint` dir grades to exit 0 with a
  paste-ready row (`check_checkpoint_arch` warns, `load_checkpoint_into` random-inits).
  Pre-existing at HEAD; recommendation: refuse, fold into TASK_886. Adjacent pre-existing
  nit recorded there too (`make_ledger_row` fixture caveat keyed on `tag.startswith`).
- **Queue drift:** QUEUE.md row 4 still reads "HALTED run 29 → RE-CUT" — banner on the task
  file is the truth (SHIPPED). Rows 5 (886), 6 (877 — its HOLD is now lifted: "HOLD until
  883R ships, then take it"), 7 (878), 8 (881) stand. 877 is the natural next row.
- **The run-29 halt stash (`stash@{0}`) is now fully superseded** — everything in it
  shipped in better form. Left in place for planner review; safe to drop after.
- **Harness issue (twice this run): background subagents ran `dark-factory-disarm` on
  their own** — the stop gate pressures a finished judge into disarming so its turn can
  end. I re-armed both times (state.json shows the gaps: 10:23–10:26, 10:33–10:33). The
  stop gate should probably not fire for subagent turns, or judges need "never
  arm/disarm" in their base prompt. Worth a plugin-level fix.
- **Kimi harness note:** subagent background tasks vanish from the task registry when the
  subagent's turn ends mid-wait (their processes survive as orphans). Twice I finished a
  subagent's in-flight proof myself. Build expensive A/B proofs as self-restoring scripts
  (stash-by-message + trap) — that pattern survived every interruption today.

## State at end of run

- Tree: clean except the seeded D37 row (committed with this handoff). Nothing pushed
  (allow_push=false; owner pushes after review).
- Verdict: `.dark-factory/verdicts/current.json` (green, 3 cycles, panel summary).
- 883R scratch under `/tmp/fx883r*`, `/tmp/ab883r_L0H2G0`, `/tmp/fv883r*` — disposable.
- Fixture untouched all run (size 79310848, sha pin `b0f8de8b…`); `trading_data.db` never opened.
