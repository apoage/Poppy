# Handoff — dark factory run 26 (2026-08-06)

One row SHIPPED, one row STOPPED on the halt rule with a new operator wall. Four Ready rows
untouched. Nothing pushed.

## Shipped

**TASK_872 — the live candle path resolves timestamps at UTC** (`task872-live-stream-utc`,
`f5a74db`). Detail in the task banner and the CHANGELOG entry; not repeated here.

The part worth carrying forward: **the two-line fix the row specified would have made a second,
pre-existing defect silent.** Attaching `tzinfo=utc` to a naive parse is correct — but the
nanosecond-truncation block upstream destroys a real numeric offset before `fromisoformat` sees
it, so the guard then stamped UTC onto a wall clock that was never UTC, wrong by exactly the
offset on every host, with no exception raised at ≥6 fractional digits. The first round's own
test asserted that wrong value as its oracle and passed. That is the failure mode to watch for
whenever a fix normalizes a value: a guard that makes a wrong answer *consistent* looks exactly
like a guard that makes it *correct*, and a test written alongside the fix will happily certify
either. What caught it was a judge instructed that TZ-invariant-but-wrong is still wrong.

Second-order lesson, same row: the first separator fix anchored on `find('T')`, which a fresh
judge broke with a lowercase `t`. The eventual fix is positional (index 11) after establishing
that Python 3.10 accepts *any* single character as the date/time separator — so case-insensitivity
would have closed one door and left `X` and `_` open. Widening the fix beyond the reported symptom
was the right instinct and should be encouraged.

## Stopped

**TASK_876 — score-based checkpoint retention.** Built, gate-green, halted after 3 fix→verify
cycles. Code in `git stash` ("TASK_876 … STOPPED on halt rule (protection model, D32)"), re-apply
verified clean on HEAD. 1,208 insertions across 6 files. Full detail in the task banner and D32.

**The mechanism genuinely works** — the collapse case is proven end-to-end on real flax saves,
which is the thing the row exists for. What stopped it is that Scope 4's "referenced by an
`experiments/` artifact is never garbage" **has no mechanical definition in this repo**, and the
factory must not invent one for an irreversible deleter. That is a real wall, not a rule being
followed mechanically.

Read D32 for the options. The single most important line for whoever picks this up:
`flax_keep = 1_000_000` combined with the pre-existing `overwrite=True` is a protection-blind mass
delete on any resume at a lower step — **this task enlarged its own blast radius**, and that is
the finding that argues against starting the long base-training program on the stashed code as-is.

### Process note the planner should absorb

The row was sized **S**. It is not, and the mis-sizing was consequential: three separate panels
each found real BLOCKERs *in the previous round's fixing code*. That pattern — defect density
staying constant while the diff grows — is exactly what the halt rule exists to catch, and it
showed up because the row bundles two very different things: score EMISSION (small, safe, and
verified sound in cycle 1) and an irreversible DELETION POLICY (large, and where every blocker
lived). Splitting those is option (d) in D32 and is probably right regardless of which protection
model is chosen.

## Untouched Ready rows, and why

- **TASK_878** (failed-fill semantics, S/M) — deliberately deferred, and this was a judgement call
  I want visible rather than buried. It rewrites execution semantics across the eval core with
  every parity gate in the repo riding on the seeded path-independent flip. After 876 demonstrated
  what an under-scoped row costs, I chose to finish one row properly rather than half-open a
  second. It is not blocked; it is next.
- **TASK_873** (fixture identity pin), **TASK_877** (XBT alias fold), **TASK_874** (imprint emits
  frame provenance) — not started, capacity only. 873 is worth taking early: I hand-guarded the
  fixture sha256 (`b0f8de8b…`) in every single subagent brief this run, which is precisely the
  manual discipline 873 exists to replace with a mechanism.

## Recommended new row (not specced, and it should land before the srv program)

**`tools/ckpt_archive.py` carries the same discovery defect as `ckpt_gc.py` did — on the tool that
deletes REMOTELY.** Its embedded `REMOTE_PROBE` (lines 41–45) defines
`CK = (^snapshot_ep\d+$, ^ckpt(_.*)?$)` plus a "any entry starting with `checkpoint_`" fallback, so
a run directory holding `checkpoint_<step>` subdirs is misidentified as a single leaf checkpoint,
the walk is pruned there, and the run's own `scores.jsonl` is read from the parent — never. The
pattern set also omits `checkpoint_\d+`, `checkpoint_epoch_\d+` and `checkpoint_golden`, and scores
are epoch-keyed only, so flax dirs named by `step_count` never resolve a score. **Failure mode:** an
entire scored run is classified UNSCORED, archived as one opaque blob or judged uninteresting, and
then deleted on `gpu-server` — this tool deletes remotely after local verification, so the blast
radius is a whole multi-day run's checkpoints on the training host, not a dry-run line. **Fix:**
port the three landed `ckpt_gc.py` changes into `REMOTE_PROBE` — `_has_ckpt_subdirs()` before the
listdir fallback, the full pattern set, and path-keyed score reading with the epoch key as
fallback. Out of scope for 876 (the task file excludes remote/off-host archival), hence a row.

## State

- Tree clean apart from `lore/handoff_2026-08-05_operator-rulings-rerun-doctrine-planner.md`,
  which is **untracked and not mine** — it predates my first tree write. I deliberately left it
  alone rather than committing another lane's work, per the run-24 incident where an agent
  destroyed an uncommitted planner edit.
- `fixtures/trading_fixture.db` sha256 `b0f8de8b…` unchanged all run, never regenerated (checked
  after every round). `trading_data.db` mtime `1785091652` / size `26245337088` unchanged.
- `git stash@{0}` holds the TASK_876 work. Do not drop it.
- Nothing pushed. Tag `task872-live-stream-utc` is local.
- The TASK_875 walk-forward battery on `gpu-server` was never touched — the factory surface was
  disjoint from it throughout.
