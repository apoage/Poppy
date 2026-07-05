# BUGHUNT deploy-integrity — sync-db.sh (commit 3a50e36)

Read-only adversarial review of `tools/deploy/sync-db.sh` (107 lines, new in
`3a50e36`) and the `tools/deploy/README.md` section it advertises. Scope: does
the script actually close the 2026-05-12 silent-corruption gap, or does it
have failure modes that re-open the same hole or open a new one.

Verdict at a glance: the workstation-side integrity gate is broadly sound,
but the script has two **P0** issues — (a) `set -euo pipefail` swallows
sqlite3 crashes silently before the user-visible failure path runs, and (b)
the printed server-side snippet `unzstd -k -f trading_data.db.zst`
**overwrites the live `trading_data.db` in-place**, contradicting the
"manual, no live-swap" safety promise the script's own docstring and the
README make. Three further **P1** issues around stale-`.zst` propagation,
stale WAL-mode data, and rerun-after-interrupted-compress. Several P2 / P3
papercuts.

## Findings

### F1 (P0): sqlite3 crash / OOM is silently swallowed under `set -e`
- `tools/deploy/sync-db.sh:51` — `RESULT="$(sqlite3 "$DB" 'PRAGMA integrity_check' 2>&1)"`
- Evidence: with `set -euo pipefail` at line 25, a command substitution on the
  RHS of a plain variable assignment (not `local`/`declare`) DOES propagate
  the inner exit status — if `sqlite3` segfaults, OOMs, or exits non-zero
  (e.g. file-locked), the script aborts on line 51 **before** lines 52-56
  run. The user sees only the `==> PRAGMA integrity_check on $DB ($SIZE)...`
  banner from line 50, then a silent exit. There is no
  "❌ integrity_check FAILED" message — that branch is unreachable when
  sqlite3 itself fails. From the caller's point of view it looks
  identical to a clean abort, which is exactly the misleading-error path
  the prompt asked about.
- A 24 GB DB + full integrity_check is a realistic OOM / kill-by-OOM-killer
  candidate. The current path conflates "DB corrupt" with "tool died" with
  "user hit ^C during the check" — all are bare exits.
- Fix sketch: capture exit code explicitly, e.g.
  ```
  set +e
  RESULT="$(sqlite3 "$DB" 'PRAGMA integrity_check' 2>&1)"
  RC=$?
  set -e
  if [[ $RC -ne 0 ]]; then
    echo "❌ sqlite3 itself failed (rc=$RC) — could not run integrity_check" >&2
    echo "$RESULT" | head -20 >&2
    exit 1
  fi
  if [[ "$RESULT" != "ok" ]]; then ... fi
  ```

### F2 (P0): printed server-side command live-swaps the DB it claims not to
- `tools/deploy/sync-db.sh:97-102` — the heredoc prints:
  ```
  ssh -p 22 operator@<redacted-ip> '
    cd Trader &&
    sha256sum -c trading_data.db.zst.sha256 &&
    unzstd -k -f trading_data.db.zst &&
    sqlite3 trading_data.db "PRAGMA integrity_check"
  '
  ```
- Evidence: `unzstd -k -f trading_data.db.zst` decompresses to the default
  output name `trading_data.db`, in `cd Trader/` — i.e. it writes directly
  on top of the live `trading_data.db` that train_phase2.py / train_easy.py
  may have open (mmap'd, file-handle reads). The `-f` forces overwrite.
  This **is** a live swap, despite the docstring (`tools/deploy/sync-db.sh:10-14`)
  and the README (`tools/deploy/README.md:18` — "manual server-side swap")
  both promising no live swap. The only "safety" is operator discipline:
  remember to verify training is idle before pasting. The script generates
  zero pre-checks (no `pgrep train_phase2.py`, no temp-name write, no
  atomic rename).
- Failure mode if pasted during training: SQLite reading via mmap on Linux
  can SIGBUS when the underlying file is replaced; writes via SQLite-WAL
  are likewise undefined. Best case: the trainer dies with a confusing
  error. Worst case: the trainer writes back through stale fd's, corrupting
  whatever the operator is in the middle of doing.
- Second failure: if `unzstd` aborts mid-write (e.g. .zst frame truncated by
  rsync interruption), the live `trading_data.db` is now a partial file.
  The subsequent `sqlite3 ... integrity_check` will fail — but the live DB
  is **already clobbered** with no rollback path. There's no `.bak` to
  restore from.
- Fix sketch: write to a temp name, integrity-check the temp, then atomically
  swap:
  ```
  unzstd -k trading_data.db.zst -o trading_data.db.NEW &&
  sqlite3 trading_data.db.NEW 'PRAGMA integrity_check' &&
  mv trading_data.db trading_data.db.OLD &&
  mv trading_data.db.NEW trading_data.db
  ```
  Plus a `pgrep -f 'train_(easy|phase2|jax_fast)\.py'` precheck.

### F3 (P1): interrupted compress poisons subsequent runs via mtime check
- `tools/deploy/sync-db.sh:62-68` (compress) + `tools/deploy/sync-db.sh:63`
  (`[[ ! -f "$ZST" ]] || [[ "$DB" -nt "$ZST" ]]` — only recompress if .zst
  missing OR DB newer than .zst).
- Evidence: `zstd -f --keep -19 -o "$ZST" "$DB"` at line 65. If zstd is
  ^C'd, SIGKILL'd by the OOM killer, or hits any error mid-write, the
  partial `.zst` is left on disk with mtime = the moment zstd started writing
  (i.e. **after** the DB's mtime). `set -e` then exits before
  `sha256sum` (step 3) runs.
- On the **next** invocation, the script evaluates line 63: `.zst` exists,
  and `$DB -nt $ZST` is false (DB mtime is older than the half-written
  .zst mtime). The compress block is skipped, the "skipping compress"
  message prints (line 67), step 3 then runs `sha256sum` on the corrupt
  .zst — which produces a deterministic hash that "matches itself" on
  server-side verify. The rsync ships the bad .zst.
- Server-side: `unzstd -k -f` will likely error out on a truncated zstd
  frame (zstd has frame-level checksums), so this is **detected** server
  side — but only if the operator runs the printed snippet. The
  workstation flow tells the user "==> Workstation sync done" with no
  warning.
- Compounding: there's no source-side sanity check that `$ZST` decompresses
  to a file matching `$DB`'s size (`zstd -t "$ZST"` would test the frame
  for free, takes seconds).
- Fix sketch: trap on EXIT in the compress branch to rm a half-written
  `.zst`, OR `zstd -t "$ZST"` after compress, OR generate the `.sha256`
  by decompressing-and-rehashing the DB (catches both transit and zstd-frame
  corruption — see also F8). Also: write to `.zst.tmp`, rename on success.

### F4 (P1): WAL-mode DBs may ship stale data (no checkpoint before compress)
- `tools/deploy/sync-db.sh:47-69` — integrity check + compress operate on
  `trading_data.db` only. No `PRAGMA wal_checkpoint(TRUNCATE)` is performed.
- Evidence: if `trading_data.db` is in WAL journal mode, recent committed
  transactions live in `trading_data.db-wal` until a checkpoint folds them
  into the main file. `PRAGMA integrity_check` correctly sees the combined
  view (DB + WAL) and reports "ok". But the script only compresses and
  ships `trading_data.db`, not the `-wal` / `-shm` files. The server
  receives a `trading_data.db` reflecting a stale checkpoint, missing
  whatever's in the unchecked-pointed WAL.
- Currently the workstation copy is January-frozen and has no `-wal` /
  `-shm` (verified: `ls trading_data.db-wal` → not found), so this is
  latent for now. But TASK_768 §"Staleness decision" explicitly considers
  re-fetching from Kraken to extend coverage before imprint — that
  fetch will run as a live writer and produce WAL state. At that point
  this becomes a silent staleness bug.
- The prior data-agent flow (`lore/notes_2026-05-08.md:47-49`) also
  didn't checkpoint, but it didn't claim to be the canonical
  integrity-gated tool either.
- Fix sketch: before line 50, `sqlite3 "$DB" 'PRAGMA wal_checkpoint(TRUNCATE);'`
  (no-op on non-WAL DBs).

### F5 (P1): `--skip-compress` ships stale `.zst` with no warning
- `tools/deploy/sync-db.sh:62` — `if [[ -z "$SKIP_COMPRESS" ]]; then` wraps
  the entire compress block, including the freshness check.
- Evidence: `--skip-compress` bypasses both the conditional recompress AND
  the freshness check. Step 3 recomputes `sha256sum "$ZST"` (matches itself),
  step 4 rsyncs. Server-side: sha256 verifies, decompress produces a clean
  but **stale** DB, integrity_check passes (the old DB was clean). The
  user sees green lights all the way down but has shipped data from
  whenever the .zst was last built.
- Currently `trading_data.db.zst` on disk is from 2026-01-20 (verified
  `stat`). If a user `--skip-compress`'s a sync today, they'd ship a
  4-month-old snapshot with no indication anything is amiss.
- The integrity_check on line 51 runs against the **current** `$DB` (good —
  catches the case where the current DB is corrupt), but it does NOT
  verify that the current `.zst` matches the current `.db`. Decoupled.
- Fix sketch: when `--skip-compress` is set, warn if `$DB -nt $ZST` is
  true, and refuse to proceed without an additional `--i-know-its-stale`
  flag. Or: drop `--skip-compress` entirely (the mtime-skip in the
  default path already covers the "skip if fresh" case).

### F6 (P1): integrity_check is the right tool; quick_check would be wrong
- `tools/deploy/sync-db.sh:51` uses `integrity_check`. TASK_768 line 98
  mandates `integrity_check` (full, not quick).
- Evidence: the May-12 incident was page-level corruption with unreadable
  `sqlite_master` (`lore/notes_2026-05-12.md:107-108`). Both `quick_check`
  and `integrity_check` would catch that. But `integrity_check` additionally
  catches: index/table content cross-reference inconsistencies, freelist
  errors, UNIQUE/NOT-NULL/CHECK violations on table rows. The May-12 mode
  is one of many corruption modes — adopting `quick_check` (faster, ~minutes
  on 24 GB) to save time would re-narrow the gate. The script picks the
  right tool.
- The script's commit cost on 24 GB integrity_check is one-time and not
  on a hot path, so the "minutes-long full scan" is acceptable. This is a
  **verified-clean** item; flagged here because the prompt asked about it
  explicitly.

### F7 (P2): stderr leakage into `RESULT` can false-positive
- `tools/deploy/sync-db.sh:51` — `RESULT="$(sqlite3 ... 2>&1)"` merges
  stdout + stderr.
- Evidence: if sqlite3 emits any stderr noise (load-extension warning,
  deprecation warning, locale warning from a future build, banner from an
  alias), the result becomes `<warning text>\nok` which strips to
  `<warning text>\nok` (trailing newline trimmed by `$( )` but the embedded
  newline stays). `[[ "$RESULT" != "ok" ]]` is true → script reports
  FALSE corruption.
- Today on this workstation: no known stderr noise. But the gate is
  brittle to environmental changes (new sqlite3 build, debug binary,
  SQLITE_DEBUG env, `~/.sqliterc` printing a banner). A `.sqliterc`
  with `.echo on` would absolutely break it.
- Fix sketch: drop `2>&1`, capture stderr separately, AND/OR check the
  last line of stdout: `[[ "$(echo "$RESULT" | tail -1)" != "ok" ]]`.
  Or use `sqlite3 -batch -bail "$DB" 'PRAGMA integrity_check'` to
  suppress init files.

### F8 (P2): `.zst.sha256` covers compressed-only — by design, but the script's
        own comment (line 72-73) understates the residual hole
- `tools/deploy/sync-db.sh:74-75` — the transit checksum is `sha256` of the
  compressed file; the script's comment ("catches rsync corruption only —
  NOT source-side SQLite corruption, which is what step 1 is for") is
  accurate for the WORKSTATION side.
- Evidence of residual hole: between step 1 (integrity check on `$DB`)
  and step 2 (`zstd -f --keep -19`), the DB could be written by another
  process (e.g. a Kraken-fetch script crontab kick), and the compressed
  artifact would reflect post-step-1 content. Then the integrity gate
  has run on a different byte stream than what ships. Probability is low
  but the script does nothing to lock the DB (e.g. `sqlite3 "$DB" 'BEGIN
  IMMEDIATE; ...'` to grab a write lock) for the duration of compress.
- Mitigation observation: the server-side `PRAGMA integrity_check` (when
  the operator runs it) would catch any post-step-1 corruption. So the
  end-to-end gate holds **provided the operator runs the printed snippet**.
  The README sells the chain as workstation-gated + manual server-gated,
  and both halves are needed; doing only the workstation half re-opens the
  hole.
- Fix sketch: add a `BEGIN IMMEDIATE` lock around the integrity_check +
  compress, OR (cheaper) hash the **uncompressed** DB into the `.sha256`
  metadata (`sha256sum trading_data.db` AS WELL AS the .zst), so the
  server-side flow can verify the decompressed file matches the gated
  bytes.

### F9 (P2): rsync target dir is unchecked; first-time deploy could surprise
- `tools/deploy/sync-db.sh:87-89` — rsync to `$SRV_USER@$SRV_HOST:$SRV_PATH/`,
  no `--mkpath`, no precheck that `~/Trader/` exists on server.
- Evidence: if `~/Trader/` does not exist on the server (first-time setup
  not yet run), rsync errors with "No such file or directory". This is
  expected to be solved by `tools/deploy/bootstrap-server.sh` (per README
  "First-time setup") — but sync-db.sh doesn't direct the user there or
  precheck it.
- Mitigation: realistically the operator has already run bootstrap by the
  time they need to sync a DB. Low impact.
- Fix sketch: `ssh -p "$SRV_PORT" "$SRV_USER@$SRV_HOST" "test -d $SRV_PATH"`
  before rsync, with a clear error pointing to bootstrap-server.sh.

### F10 (P2): no `--checksum` on rsync, no `zstd -t` after compress
- `tools/deploy/sync-db.sh:87` — `rsync -avP --partial` uses default
  mtime+size for the resume/skip heuristic.
- Evidence: in a re-sync where a same-size partial-but-corrupted `.zst`
  already exists on the server (e.g. interrupted previous run), rsync's
  default heuristic relies on its delta-transfer algorithm to detect
  differences — usually robust, but slower than `--checksum` and not
  rigorous against the case where the server-side mtime was touched
  manually. The transit `.sha256` catches this server-side. So the gate
  is held by the manual sha256 step, not by rsync's own integrity.
- Verified-clean-ish: the chain still detects this via the printed
  `sha256sum -c` server-side. P2 because reliance on the manual step
  is OK given the docs.
- Also: `-P` already implies `--partial`, the explicit `--partial` is
  redundant. Cosmetic only.

### F11 (P3): `--help` breaks when invoked from `tools/deploy/` with `./sync-db.sh`
- `tools/deploy/sync-db.sh:27-28` (`cd "$REPO_ROOT"`) + line 42 (`sed -n
  '2,21p' "$0"`).
- Evidence: line 28 cds to repo root before the arg loop runs. `$0` is
  whatever path the user typed. Three cases:
  - User typed `tools/deploy/sync-db.sh --help` from repo root → `$0 =
    tools/deploy/sync-db.sh`; after `cd $REPO_ROOT` (no-op), resolves
    → works.
  - User typed absolute path → works.
  - User typed `cd tools/deploy && ./sync-db.sh --help` → `$0 =
    ./sync-db.sh`, then `cd $REPO_ROOT` makes `./sync-db.sh` not exist
    in cwd. `sed -n '2,21p' ./sync-db.sh` → "No such file or directory",
    `--help` errors instead of printing the docstring.
- Trivial fix: `sed -n '2,21p' "$(realpath "$0" 2>/dev/null || echo "$0")"`,
  OR resolve `$0` before the `cd` to `$REPO_ROOT`.

### F12 (P3): unknown positional args are silently ignored
- `tools/deploy/sync-db.sh:37-45` — case statement has `--flag` patterns
[redacted: VPS ops detail]
- Evidence: a typo `sync-db.sh checkonly` (missing `--`) does not match
  any case pattern, hits no default, falls through. The script proceeds
  as if no flag had been passed. Quietly wrong.
- Fix sketch: add `*) echo "unknown argument: $arg" >&2; exit 2 ;;` after
  the `-*` arm.

### F13 (P3): `-h|--help` help slice is fragile to docstring edits
- `tools/deploy/sync-db.sh:42` — `sed -n '2,21p' "$0"`. Hardcoded line
  range. If anyone reorders/adds a docstring comment line, the help text
  silently chops off or includes wrong content. Minor maintenance hazard.
- Fix sketch: delimit the docstring with a sentinel comment (e.g.
  `# >>>HELP`/`# <<<HELP`) and sed between them.

### F14 (P2): documentation drift — README claim vs. printed snippet behavior
- `tools/deploy/README.md:9` ("manual server-side swap") + script docstring
  `tools/deploy/sync-db.sh:10-14` ("does NOT live-swap the DB on the
  server") vs. the actual printed snippet (F2): `unzstd -k -f
  trading_data.db.zst` overwrites the live `trading_data.db` in place.
- Evidence: every layer of documentation insists the swap is manual and
  safe, but the operator's "manual" step IS the swap, performed as a
  destructive in-place overwrite with no rollback. The promise the
  README makes does not match what the script tells the operator to run.
- Fix sketch: align by F2 (atomic temp-name + rename), then update the
  README to describe that flow.

## Verified clean

- The "ok" string comparison itself (line 52). Bash `$( )` strips a single
  trailing newline, so `RESULT == "ok"` exactly for a clean DB. Multi-line
  corruption reports → `[[ != "ok" ]]` correctly triggers. Confirmed by
  reading SQLite's PRAGMA integrity_check semantics (single-row "ok" or
  list of error rows, default mode — no column header).
- `sha256sum` output format (F0): GNU coreutils prints `<hash><sp><sp><name>`
  with bare basename when invoked as `sha256sum "$ZST"` (ZST = relative bare
  name). Verified the existing on-disk `.sha256` file: bare basename, two
  spaces, single line. Server-side `sha256sum -c $SHA` after `cd Trader`
  resolves correctly. No path-prefix bug.
- Heredoc variable expansion (lines 92-107): unquoted `EOF` → workstation
  shell substitutes `$SRV_PORT`, `$SRV_USER`, `$SRV_HOST`, `$SRV_PATH`,
  `$SHA`, `$ZST`, `$DB` into the printed snippet before the operator sees
  it. So the user copy-pastes literal values, not unset-on-server variable
  names. Verified mentally — values are populated by lines 30-32 + 78-81
  before the heredoc runs.
- Single-quote nesting in the printed ssh snippet: the outer `'...'` wraps
  the remote command. The inner `"PRAGMA integrity_check"` lives literally
  inside the single quotes (single quotes don't interpret double quotes),
  so the remote shell receives the double-quoted pragma argument intact.
  No quoting fight.
- `integrity_check` vs `quick_check` choice: integrity_check is strictly
  more thorough and is what TASK_768 mandates. See F6.
- `rsync --partial` resume semantics for an interrupted upload: rsync
  defaults already handle this correctly (resume from the partial on next
  invocation, truncate-on-shrink when the new file is smaller). Verified
  by rsync docs/standard behavior.
- Flag combinations: `--check-only` exits before compress/rsync (line 59);
  `--dry-run` is checked at every action site (lines 65, 75, 84); `--skip-
  compress` correctly bypasses the compress block. No silent skip + ship
  combination that ships **without** sha256 regeneration; the sha256
  always regenerates from the on-disk `.zst` after the compress branch.
  F5 covers the stale-content concern, not a flag-handling correctness
  one.
- `set -euo pipefail` (line 25): right tool for the job, except that the
  failure-message ergonomics around it are incomplete (F1).

## Inconclusive / out of scope

- Whether the operator actually runs the printed server-side snippet
  after every sync, or only "when they remember to". The script can only
  print instructions; this is a workflow / discipline issue. The lore
  note `lore/notes_2026-05-22.md:43` lists this as a standalone fix
  prerequisite for any base-model retrain, suggesting the gate is taken
  seriously, but no automation enforces it on the workstation side.
- Whether the server's `sqlite3` and `unzstd` binaries are present and a
  recent-enough version to interpret the workstation's `.zst` frame /
  PRAGMA output. The script doesn't probe. This was likely confirmed
  during the May-08 / May-12 sessions (`lore/notes_2026-05-08.md`)
  but is not re-verified by this script.
- The exact behavior of `mv` / rename atomicity on the server's
  filesystem during a future "fixed" live-swap flow (F2 fix sketch) —
  depends on whether `$SRV_PATH` and the temp filename are on the same
  device. On the same filesystem `mv` is atomic; cross-device it
  degrades to copy+unlink. The server agent's `~/Trader/` is on one
  device per `lore/notes_2026-05-08.md` so likely fine, but
  out-of-scope to verify without running anything.
- The integrity-check runtime on the workstation's specific 24 GB DB
  (read-only review, did not execute). May-22 verification was
  `quick_check` (TASK_768:95), not full `integrity_check`, so the script's
  cold-cache wallclock is unmeasured. Not a correctness issue, just an
  ergonomics one — if it routinely takes >10 minutes the operator will be
  tempted to `--skip-compress` (which has F5).
