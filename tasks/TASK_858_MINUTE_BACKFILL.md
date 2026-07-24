# TASK_858 — Guarded minute-table backfill 2025-10 → present (D22 ruled: option (a))

> **STATUS (2026-07-24): SHIPPED as `task858-minute-backfill` (tool + guards; real ingest = operator remainder).**
> `tools/backfill_minute.py` — the single sanctioned append-only write path to `trading_data.db`. Acceptance
> (i) fixture leg 17/17 + (ii) append-only proof PASS (both mutation-tested); (iii) real ingest STOPPED per spec
> (no Kraken dumps on disk, not acquirable non-interactively); (iv) 848 OOS smoke gated on (iii). Verified by a
> 4-member panel (correctness + fail-closed + regression + flow-verifier); a fix round closed **2 BLOCKER + 2
> MAJOR + 1 MINOR** in the guard architecture (commit-before-guards → atomic; whole-window proof → subset proof;
> mtime-only backup gate → hardened; 11/64 unreachable dump spellings → `_canon_key` routing, 64 distinct/zero
> collisions; unmatched-file reporting) — both re-verify lenses then returned no findings. selftest 12/12.
> **REMAINDER (operator, unblocks iii + the binding OOS reads)**: download Kraken OHLCVT CSV dumps
> (Q4-2025…Q3-2026, `{PAIR}_1.csv`), take a fresh DB backup, then run `backfill_minute.py run …` ON THE
> WORKSTATION (see CHANGELOG for the exact command). Residuals documented (backup gate = freshness/size
> heuristic not content proof; `--tail-topup` opt-in writes outside the proof; bad candle halts the run).

**Status**: READY 2026-07-24 (D22 RULED — operator: "ok prepare factory"; option (a) minute backfill).
**Size: S/M. Risk: medium** — this task WRITES to `trading_data.db`, the source-of-record that is
otherwise read-only by standing constraint. That is exactly why it was an operator wall (D22);
the ruling admits ONE guarded, append-only ingest path. Nothing else in the program writes to it.

## Why (LEDGER 853/numeric-leg, DECISIONS D22)
The program has NO honest OOS benchmark: the canon minute table dies ~2025-10 for most of the
64-symbol universe (thin n=31 read = the only honest OOS datum), and the daily table covers only
27/64 in-sample ⇒ daily-source OOS is unattributable for the minute-native, 64-slot-locked ckpt.
Extending the minute table 2025-10 → present unblocks, via the ALREADY-SHIPPED instruments
(848 OOS window + on-window teacher floor; 853 fixed-frame panel): the binding imprint OOS read
(~9 months, ~270 decisions), the fresh-data confirmation leg for 856/rule-grid (top-m), and any
future 857 operator OOS read. Everything downstream of "is this real out-of-sample" funnels here.

## Data-path reality (measured, drives the design)
The existing incremental fetcher (`tools/update_dataset.py` → `mcts_trader/data/fetcher.py`)
uses Kraken's public **OHLC endpoint (`0/public/OHLC`), which returns ONLY the last ~720 candles
per pair** regardless of `since` — 12 hours of 1-min data. It can NEVER backfill 9 months.
- **Primary path**: Kraken's quarterly OHLCVT CSV dumps (Q4-2025, Q1-2026, Q2-2026) + the
  EXISTING importer `tools/import_kraken_csv.py` (format `{SYMBOL}_{TIMEFRAME}.csv`, 1-min =
  `_1.csv`) — the DB's original build path.
- **Tail top-up**: the existing fetcher covers the last ≤12h after the dumps.
- **Fallback (document, do NOT build unless primary is blocked)**: Trades-endpoint pagination +
  1-min aggregation — slow (rate limits × 64 pairs × months), new code.

## Scope
1. **Backfill orchestrator** (`tools/backfill_minute.py` or extend the importer): takes dump
   dir(s) + the 64-symbol universe (from `runs/832_e5/frame_provenance_imprint.json` slot order),
   imports ONLY those symbols' `_1.csv` series, then runs the fetcher tail top-up.
2. **Guards (the load-bearing part — all fail-loud)**:
   - **Backup gate**: refuses to run unless a fresh backup of `trading_data.db` exists (path +
     mtime check; `zstd→rsync` does NOT detect source corruption — the backup is the recovery
     path, make it mandatory).
   - **Append-only invariant**: never UPDATE/DELETE existing rows; if an incoming candle's
     (symbol, timestamp) already exists with DIFFERENT values → abort loud with a diff sample
     (no silent overwrite, no silent skip-count).
   - **Integrity checks post-ingest**: per-symbol monotone timestamps, no duplicate keys,
     OHLC sanity (l ≤ o,c ≤ h; v ≥ 0), row counts before/after.
3. **Coverage report** (the acceptance artifact): per-symbol table — old last-timestamp, new
   last-timestamp, rows added, residual gaps > 1 day inside 2025-10 → present. Written to
   `runs/858_backfill/coverage_report.md` + JSON.
4. **Fixture leg**: the whole orchestrator (guards + report) exercised against a COPY of
   `fixtures/trading_fixture.db` with a synthetic dump dir (few symbols, few days) — including
   the conflicting-candle abort and the missing-backup refusal.

## Acceptance (gates)
- (i) Fixture leg green: guards demonstrably bite (backup-gate refusal, conflict abort), report
  schema stable.
- (ii) Append-only proven: pre/post row hash over a pre-existing sample window is IDENTICAL.
- (iii) Real ingest (needs the downloaded dumps — see below): coverage report shows the
  64-universe minute table reaching ≥ 2026-06 for ≥ 55/64 symbols, zero integrity failures.
  If the dumps can't be acquired non-interactively, STOP after (i)+(ii) and report the exact
  download requirement — data acquisition is allowed to be the operator's manual step.
- (iv) Post-ingest smoke: the 848 OOS window instrument runs on the extended minute path
  (fail-loud guards pass); NO verdict claimed — the binding OOS read stays planner-scheduled.

## Out of scope
The OOS reads themselves (planner, post-ship); daily-table anything (D20 scope stands); Trades
aggregation fallback (build only if primary is blocked AND the wall is re-seeded); touching
`ohlcv_daily`; any schema change.
