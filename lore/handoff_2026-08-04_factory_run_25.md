# Handoff — dark factory run 25 (2026-08-04)

Implementer lane. Ready = TASK_870 → TASK_871, both SHIPPED. Nothing pushed; tree clean.

## Queue drift found at startup (planner, please reconcile)

The Ready **table** in `tasks/QUEUE.md` read `_Ready = EMPTY (factory run 24, 2026-08-03)_`, while
the three most recent grooming lines — all dated 2026-08-03, including the FINAL one — read
`Ready = TASK_870 → 871 unchanged; factory GREEN`. Both task files existed, unbannered, seeded per
commit `b20ac63`. I took the grooming lines as authoritative (banners beat the queue; task files
existed and were specced) and worked both rows. The table is now stale in the other direction —
both rows shipped.

## Shipped

| Tag | Task | Verification |
|---|---|---|
| `task870-canon-db-readonly-handle` (`4fb0d77`) | TASK_870 — canon panel loader off read-write handles | medium panel, **0 BLOCKER/MAJOR, 1 cycle**, no escalation |
| `task871-residual-localtime-parses` (`9790644`) | TASK_871 — residual local-time date parses (FILLER) | medium panel, **0 BLOCKER/MAJOR, 1 cycle**, no escalation |

Both declared XS/S-low in their specs; both were run at the **medium** tier anyway, because 870's
diff reached `eval_vector.py` (read-of-record operator) and `trader/` (eval-bar engine), and 871
inherited 870's fresh import wiring. Full detail in the CHANGELOG entries and task banners.

### TASK_870 — the hazard was real on the canon path, and the panel proved it

The flow-verifier's **CONTROL B** is the load-bearing artifact: the *actual pre-870*
`vectorbt_env.py` at HEAD, given nothing but a bare `SELECT` against a fixture copy carrying a
crashed rollback journal, silently rewrote the file (sha256 changed, size reverted
83419136 → 79310848) and unlinked the journal. Through the shared `_ro_connect` the same copy is
left byte-identical with the sidecar intact. Both judges independently forged their own valid hot
journal and reproduced the same split; the regression judge caught and discarded its own first
attempt as a false negative (zero-header journal neither side touched) before trusting it.

Bit-identity was **proven, not assumed** — three panel members, three independent `git worktree`s
at `858fda0`, per-callsite sha256 plus a full `create_jax_env_historical` panel
(`np.array_equal` True, `tobytes` sha256 `81d79e23…`).

**Operator-facing behavior change, now in the helper's docstring:** a hot journal *stops* a canon
run (`attempt to write a readonly database`, db untouched) where it used to be silently rolled
back. Intended trade on a 24 GB read-only asset whose only backup predates the 858 backfill. If it
fires, it means an earlier writer died mid-transaction — recover deliberately through
`tools/backfill_minute.py` after a backup, never by reverting a call site.

### TASK_871 — one comment was measured false before it shipped

The implementer justified importing `trader.backtest._to_epoch` into `diff_teacher/data_loader.py`
with "this does not touch mcts_trader, so it does not compromise this loader's independence
boundary". I measured it: `import diff_teacher.data_loader` went from pulling **zero**
`mcts_trader` modules to pulling `mcts_trader{,.utils,.utils.db}` (0.49s → 0.71s) — because
TASK_870 had rewired `trader/backtest.py`'s DB handle onto the shared helper *hours earlier the
same run*. The comment is now an explicit disclosed-cost block; both judges re-measured and
confirmed the corrected figures. Functionally nominal (the reached module is `os`+`sqlite3`; no new
JAX — that was already in the graph via `diff_teacher/__init__` → `.encoder`), but a false
disclosure in a load-bearing file is the class this factory treats as real.

Gate (i)'s sweep report had no durable home in the tree — the flow-verifier correctly flagged that
and independently re-derived `train_manager.py:321,330` as an unconverted same-class site. Not a
miss: the implementer had reported it with the `tools/orchestration/CLAUDE.md:19`
"Deprecated — DO NOT EDIT" justification (verified). The full sweep table now lives in the
CHANGELOG entry.

## Incident — the fixture was regenerated mid-review, and restored

`fixtures/trading_fixture.db` is **gitignored**, so `git status` cannot see it move. The 871
regression judge ran `fixtures/make_fixture.py` mid-review; the fixture went
**`b0f8de8b…` → `59525bd3…`** (same size, 79310848). Every committed baseline was promoted against
`b0f8de8b`, and TASK_869's own remainder had warned that *the next regen re-bases several 865
expectations* — i.e. this was a pending, planner-gated re-base event triggered by accident, not a
numerics change from either shipped task.

Restored to `b0f8de8b` with the operator's explicit authorization (the first attempt was correctly
denied by the permission classifier). The reference is well corroborated: identical `b0f8de8b`
copies exist from the 869-era run (`/var/tmp/t869/fixture.db`, 2026-08-03) and from a copy taken
at 00:23:39 — 47 seconds before the regen. Post-restore re-verification: 861 `smoke_e2e.py` +
863 `smoke_frame_pin.py` (result recorded in the run's wrap-up message).

**Process lesson worth a standing rule:** a gitignored fixture is invisible to the clean-tree
check the whole ritual leans on. Review agents are told not to touch `trading_data.db`, but nothing
stopped one from regenerating the fixture *from* it. Suggest the brief for every future panel
member include "do not run `make_fixture.py`", and/or that the factory record the fixture's sha256
at arm time and re-check it at wrap.

Both judges did in fact reach the protected DB — one via a bare `runpy` of `map_training_data.py`
(whose `__main__` hardcodes `build_training_schedule('trading_data.db', …)`), one via
`make_fixture.py`. **`trading_data.db` is unharmed: mtime/size unchanged at 2026-07-26 20:47:32 /
26245337088, only atime moved.** That is TASK_870's read-only handle doing exactly its job, on the
day it shipped — an unplanned live demonstration of the thing the task was for.

## Where it stands

Ready is now genuinely empty. No OPEN rows in `tasks/DECISIONS.md`. The 861 real-leg chain and the
reference re-measure remain planner/server lane. Under honest post-`858fda0` semantics the program
still has real in-window learning (+0.9914 median, seed-stable) and **no demonstrated OOS edge** —
neither task shipped tonight touches that; both are fail-closed hygiene on the read path.

## Recommended rows (unspecced, ranked)

1. **`mcts_trader/data/stream.py:157,162` — LIVE local-time parse.** The Kraken V2 WebSocket
   adapter strips a confirmed `Z` (UTC) suffix off `interval_begin`, then calls `.timestamp()` on
   the now-naive string, so real-time candle construction is machine-TZ-dependent. Same class as
   871 but **result-bearing today, not latent**, and on the live trading path. Fix looks XS
   (`replace(tzinfo=utc)`). Left alone only because it sits outside 871's declared surface.
   Highest-value of these.
2. **`mcts_trader/data/database.py` — the last known read-write reach** at the canon layer
   (aiosqlite, genuine live-write path; needs a real dual-connection refactor, beyond XS). 870
   closed everything else it could reach.
3. **A neutral home for the canonical date parse**, if `diff_teacher`'s documented independence
   from `mcts_trader` is meant strictly — today it routes through `trader.backtest` (scipy +
   `mcts_trader.utils.db`). Would also let `trader/data.py`'s parallel `_connect` and
   `837_walkforward`'s weaker `_ro_connect` collapse into one implementation; the repo currently
   carries three.
4. **`train_manager.py:321,330`** — same defect class, blocked only by the file's
   "Deprecated — DO NOT EDIT" directive. XS the day that lifts.
5. **DB backup remains an ops item.** The only backup of the 24 GB source of record still predates
   the 858 backfill. 870 makes a stale-journal *stop* more likely to be seen, not less — and the
   recovery path for one runs through a deliberate write.
