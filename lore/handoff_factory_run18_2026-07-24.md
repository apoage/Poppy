# Dark-factory handoff — run 18 (2026-07-24)

Neutral technical continuity record. Run 18 took the two run-17-seeded Ready rows (D22 + planner
filler), shipped both, and hit the stop condition (only human walls + missing real data remain).

## Shipped

| Version (tag) | Task | Verification outcome |
|---|---|---|
| `task858-minute-backfill` (`19f955c`) | TASK_858 — guarded append-only minute backfill tool | 4-member panel (correctness + fail-closed + regression + flow-verifier); fix round closed **2 BLOCKER + 2 MAJOR + 1 MINOR**; both re-verify lenses then clean. Acceptance (i) fixture 17/17 + (ii) append-only proof PASS (mutation-tested). selftest 12/12. |
| `task859-eval-activity-metadata` (`d32c64a`) | TASK_859 — trade-activity metadata in eval JSONs (XS filler) | Small panel (correctness + regression): **no findings** at either lens. Additive-only (diff 30/0), tiers-SHA + old-keyset SHA unchanged. selftest 12/12. |

## TASK_858 — what shipped and what is NOT done

`tools/backfill_minute.py` is the **single sanctioned write path** to `trading_data.db` (D22 opt (a)).
It is complete and gate-green on the fixture, but **no real ingest has happened** — `trading_data.db`
was never written (mtime unchanged). Acceptance (iii) real ingest STOPPED per the spec's own fallback:
no Kraken OHLCVT dumps exist anywhere on the filesystem, and they are not acquirable non-interactively.

The guard architecture was the load-bearing part and it was substantially wrong on first implementation.
The fix round (documented in CHANGELOG + commit) closed:
- **BLOCKER**: `conn.commit()` ran BEFORE the integrity + append-only guards → a failed guard left the
  write durable in the source-of-record. Fixed: one atomic deferred transaction (`isolation_level=''`),
  commit only after both guards pass, `rollback()+raise` on any failure. Verified atomic across files.
- **BLOCKER**: the append-only "proof" hashed the whole `[min,last]` row-set, so a legitimate interior
  gap-fill (the NORMAL case — Q4-2025 dumps overlap existing tails and fill missed minutes) false-aborted
  the run, and it could never detect a true mutation. Fixed: subset proof — capture the pre-existing
  `{ts→OHLCV}` map before insert (excluding this run's own new keys), assert every pre-existing key
  unchanged post-ingest; new keys allowed.
- **MAJOR**: backup gate was mtime-only fail-open (self-ref / 0-byte / touched-stale all passed) → hardened
  (realpath + inode/dev, non-empty, size ≥ DB, mtime). Residual: it is a freshness/size heuristic, not a
  24GB content-validity proof (impractical) — documented, cannot corrupt the DB.
- **MAJOR**: dump matching was unreachable for 11/64 universe spellings (leading-slash + BTC-spelled) →
  only 53/64, below acceptance (iii)'s ≥55. Fixed: `_canon_key` (uppercase + strip slashes + XBT→BTC)
  routed through the resolved universe→asset_id table; verified 64 distinct keys / zero collisions.
- MINOR: unmatched dump files now reported (`unmatched_files`/`files_unmatched`).

Fixture selftest extended 9→17; scenarios E (gap-fill), G (atomic rollback), J (spelling forms) are
mutation-confirmed to FAIL against the pre-fix code.

### Operator remainder (unblocks the binding OOS reads — TASK_858 (iii)+(iv))
This is a **manual data-acquisition step**, on the **workstation** (the DB source-of-record lives here,
not the server):
1. Download Kraken downloadable historical **OHLCVT** CSV dumps (Q4-2025 … Q3-2026) from Kraken Support;
   the 1-min series are `{PAIR}_1.csv`. Unzip each archive to its own dir.
2. Take a fresh `trading_data.db` backup (the gate enforces mtime ≥ DB, size ≥ DB, non-self-reference).
3. Run:
   ```
   python tools/backfill_minute.py run --db trading_data.db --backup <fresh-backup> \
     --dump-dir <Q4dir> <Q1dir> <Q2dir> [<Q3dir>] --universe runs/832_e5/frame_provenance_imprint.json
   ```
   (add `--tail-topup` for the ≤12h tail — opt-in live path, writes outside the append-only proof).
4. Read `runs/858_backfill/coverage_report.md`. Target: minute table ≥ 2026-06 for ≥55/64, zero
   integrity failures. Then the 848 OOS window instrument runs on the extended path (acceptance iv,
   planner-scheduled). Note: a single bad candle in any dump halts the whole run (intended fail-loud).

## TASK_859 — note for the planner
One by-design NIT: `n_long`/`n_short` in the new `activity` block count **action-code intent**, not
net-position direction — a Sell that CLOSES an existing long buckets as "short". This matches the spec's
"decoded action stream" wording, but read the activity split accordingly when compiling sweeps.

## Where it stands / open for the human
- **Ready queue EMPTY** after this run. Nothing else is unblocked for the factory.
- **D24 OPEN** (planner/operator) — the read-then-decide fork on whether the frozen 832 reward head gets
  co-distilled/un-frozen. Unchanged by this run; the 857 real distill + corridor reads that inform it are
  running on gpu-server (planner lane), not factory work.
- **Binding OOS reads** (imprint OOS, 856 top-m fresh-data confirmation, future 857 operator OOS) all
  funnel through the TASK_858 operator remainder above — they cannot proceed until the Kraken dumps are
  downloaded and the real ingest is run on the workstation.

## Unblocking questions (highest-value first)
1. **Kraken dumps**: can you request/download the Kraken OHLCVT quarterly dumps (Q4-2025…Q3-2026) so the
   workstation real ingest can run? That single step unblocks every binding OOS benchmark the program is
   currently gated on.
2. **D24**: pending the 857 corridor read verdict (planner-scheduled on srv) — no factory action; flagged
   so it isn't lost.
