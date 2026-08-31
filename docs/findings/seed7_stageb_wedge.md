# seed7 stageb wedge

> Recurring wedge: seed-7 Stage-B (train_phase2) runs hang — process alive but log freezes + GPU 0% util — under 2-wide concurrency. 3+ occurrences.

*Migrated from local agent memory 2026-08-31 (`project_seed7_stageb_wedge.md`) so subagents and the server agent
can read it — see `lore/adr/0008`. Content verbatim; only frontmatter and wiki-links changed.*
**Pattern:** `train_phase2.py` Stage-B runs on **seed 7** intermittently **wedge** — the python
process stays alive and holds its ~20 GB of VRAM, but the train log stops advancing and GPU util
drops to 0% indefinitely. Always correlates with **concurrent (2-wide) Stage-B execution**.

**Occurrences (≥3 — pattern, not transient):**
- 775p2 s7 hung at **E135/200** (~19h), 2-wide with s13 (2026-05-29/30).
- An s7 **eval** process hung ~9h (earlier in the campaign).
- TASK_787 **w020 s7 wedged at E4** (2026-05-31 20:30 UTC), log frozen 22h, killed 2026-06-01.
  Was 2-wide with the w010 chain.

**Notes:**
- Wedge epoch is inconsistent (E4 → E135), so it's not a fixed data/step fault. Common factors are
  **seed 7 + Stage-B + 2-wide concurrency**.
- Possibly the same root as TASK_742 (rank-head deadlock under concurrent `train_phase2`), which was
  thought safe at ≤2-wide — but these wedges happen AT 2-wide, so 2-wide may not be fully safe for s7.

**How to apply / mitigations:**
- Treat a frozen log + 0% util + live process as THIS wedge; kill and move on (don't wait it out).
- Run **seed-7 Stage-B SOLO** (not concurrent) when its result is load-bearing, OR
- Add a **log-staleness watchdog** to orchestrators (kill+restart a cell whose log hasn't advanced
  in N minutes) so a single s7 wedge can't silently burn a GPU for a day.
- See `feedback_server_stays_under_load` (a wedged job = silent idle GPU). TASK_742 ref in MEMORY.md.
