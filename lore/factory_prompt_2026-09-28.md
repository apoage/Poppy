# Factory prompt — 2026-09-28 profitability-recomposition window (~12h unattended)

Paste everything below the line into the implementer session.

---

Run the dark factory. Startup per the skill: read `.dark-factory/config.json`, `git status`
(MUST be clean — if dirty with anything you cannot attribute to this run's tasks, STOP and
report, do not commit around unknown edits), `tasks/DECISIONS.md` (OPEN rows are walls),
`tasks/QUEUE.md` (its "TONIGHT'S STEERING" block is this run's order). Then arm.

**Window**: ~12 hours, unattended. Budget guidance: 918 ≈ 1–2 cycles; 924 ≈ 2; 923 ≈ 2–3;
915 = build + fixture smoke + SCHEDULED wrapped launch only (never poll it in-loop).

**Steering (overrides queue rank, never invariants) — work exactly these rows, in order:**

1. **TASK_918** (amended 2026-09-28 — read-only staged loader; run STEP-0 FIRST and let its
   kill gate fire if it fires; the hull/oracle/blend numbers are the deliverable either way).
2. **TASK_924** (execution-feasibility instrument — READ-ONLY over
   `~/captures/capture_907_main.jsonl` on gpu-server via the shipped 914 streaming reader;
   its D81 wall is absolute: never edit the 907 probe tree).
3. **TASK_923** (post-listing event book phase A — SPEC sha BEFORE the first real run;
   MDE-vs-band at spec time; the cost-honest stack only).
4. **TASK_915** (amended takeable — the memo `lore/fable_direction_memo_2026-09-03.md` §4 IS
   the spec; build + fixture smoke in-loop; launch the real ≤2-CPU-day CPU run wrapped on
   gpu-server inside tmux, record the tmux tag + manifest, move on).

If a row halts (cycles exhausted or a wall hits), take the next IN THIS LIST. Rows 904B /
908 / 906 are NOT tonight's work.

**Hard walls (on top of DECISIONS OPEN rows — D81(a)/(b), D84, D80(ii)/(iii), D43):**

- **D81**: never patch or modify the 907 probe tree (`experiments/907_live_fill_probe/*`).
- **NO DB WRITES anywhere** — `trading_data.db` and every DB stay read-only; the 899 ingest
  is NOT authorized; 918/915 read the staged CSVs as FILES with in-loader seam re-verification.
- **srv standing rule**: gpu-server is shared — pure reads + `tools/safe_run.sh`-wrapped jobs
  only; disturb no process; no restarts/kills/renice; NVML driver mismatch is pre-existing,
  ignore it.
- **Suites run on gpu-server wrapped** (fresh scratch copy; genuine HEAD objects + tracked
  corpus staged in, per the 922 verification recipe in `lore/notes_2026-09-28.md` §1);
  bare `pytest` on the workstation is refused. `py_compile` on the workstation is fine.
- **No live orders, no exchange APIs, no keys.** Dry-run only.
- **Any A/B- or verdict-shaped number**: MDE vs its pre-registered band computed AT SPEC
  TIME (D85 as promoted by run 55); underpowered = report UNDERPOWERED, never a pass.
- **D83 conduct**: artifact-and-revision-per-number; no `git add -A`; one code-modifying
  subagent at a time; every agent gets a private `/var/tmp/<task>-<role>` scratch named in
  its brief; freeze before you lens.
- **Canon surfaces**: any new behavior in shared code is flag-gated default-OFF with OFF =
  bit-identical proven (the config invariant); new-module-only rows keep their files
  disjoint.

**Wrap-up**: dated handoff in `lore/`, STATUS banners on every touched task file, queue
groomed with tonight's outcomes, clean tree, nothing pushed, disarm. In the wrap-up
message: per-row verdicts with the decision-changing numbers (918's STEP-0 line especially),
what was scheduled-but-not-finished, and the 1–2 questions that unblock the next track.
