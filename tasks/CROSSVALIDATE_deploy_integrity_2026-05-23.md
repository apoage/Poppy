# CROSSVALIDATE deploy-integrity — bug-hunt verdicts

Read-only independent re-derivation of `tasks/BUGHUNT_deploy_integrity_2026-05-23.md`
against the actual `tools/deploy/sync-db.sh` (commit `3a50e36`). No script
was executed; verdicts derive from script semantics + man pages + lore.

## Per-finding verdicts (F1-F14)

### F1: CONFIRMED — severity DOWNGRADE P0 → P1

**Mechanism** (independently re-derived). `set -euo pipefail` is in force.
Line 51 is `RESULT="$(sqlite3 "$DB" 'PRAGMA integrity_check' 2>&1)"`.

Bash grammar: a variable assignment with command substitution is a *simple
command*; its exit status equals the exit status of the last command
substitution performed. `man bash` (EXPANSION → Command Substitution + SHELL
GRAMMAR): "If one of the expansions contained a command substitution, the
exit status of the command is the exit status of the last command
substitution performed." The `local`/`declare` exception does not apply
(line 51 is a bare assignment in script scope). Therefore: if `sqlite3`
exits non-zero (SIGSEGV, OOM-killed, locked-DB error, missing binary,
PATH-issue, malformed CLI args), the entire line returns non-zero, and
`set -e` aborts the shell **before** the `if [[ "$RESULT" != "ok" ]]` test
on line 52 runs.

User-visible failure path: stdout shows the line 50 banner
`==> PRAGMA integrity_check on $DB (24G)...`, then the shell exits with
sqlite3's rc. **The `❌ integrity_check FAILED — refusing to sync:` branch
on lines 53-55 is unreachable when sqlite3 itself crashes** — that branch
is only for when sqlite3 succeeds AND prints non-"ok". The captured
RESULT (which has the stderr error text inside, due to `2>&1`) is
discarded with the shell.

So the bug-hunt's "silent abort" framing is correct: the operator sees
banner + bare exit, looking identical to a clean abort caused by some other
condition. The captured error text never reaches the operator's terminal.

**Severity sharpening**: P0 → P1.
- The **success path** is correct: sqlite3 succeeds + prints "ok" → script
  proceeds. sqlite3 succeeds + prints corruption rows → `RESULT != "ok"` →
  the explicit `❌` branch runs. The script's *primary* defensive purpose
  (detect a corrupt DB that PRAGMA flagged) is intact.
- The bug only manifests on sqlite3 itself dying (SIGSEGV / OOM / SIGBUS /
  missing binary). On 24 GB this is non-trivial — `PRAGMA integrity_check`
  is memory-bounded and can OOM-kill on a loaded workstation — but it is
  not the *common* case the way "DB intermittently corrupts" was the
  triggering 2026-05-12 incident.
- This is a "would-be-bad-under-rare-conditions" issue, not "running the
  script under realistic conditions corrupts a DB". It does NOT push
  corruption to the server (the script aborts; nothing reaches rsync).
  Worst outcome: operator is confused about why the script exited silently
  and may rerun without diagnosing.
- The hunt's P0 framing is defensible only if you consider "misleading
  failure mode of the gate itself" as blocking. By that bar, F1 is P0; by
  the "ships corruption to server" bar, F1 is P1. I take the latter.

**Recommended fix** (as the hunt sketched): capture rc separately:
```bash
set +e
RESULT="$(sqlite3 "$DB" 'PRAGMA integrity_check' 2>&1)"
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
  echo "❌ sqlite3 failed (rc=$RC) — could not run integrity_check" >&2
  echo "$RESULT" | head -20 >&2
  exit 1
fi
[[ "$RESULT" == "ok" ]] || { echo "❌ corrupt"; ...; exit 1; }
```

---

### F2: CONFIRMED on the literal claim — severity DOWNGRADE P0 → P1

The script's docstring (lines 10-14) says "this script does NOT live-swap
the DB on the server. It rsyncs the .zst + .sha256; the operator
decompresses + re-verifies MANUALLY when training is idle". The README
line 12 says "manual server-side swap". Lines 94-95 say
"Server-side live-swap is MANUAL — perform it ONLY when training is idle".

The printed command (lines 97-102):
```
unzstd -k -f trading_data.db.zst
```
zstd defaults: output filename is input minus `.zst` extension → writes to
`trading_data.db`. `-k` keeps the source `.zst`. `-f` forces overwrite of
any existing output file. So this **does** destructively overwrite the
live `trading_data.db` in place, with no temp-name flow, no `.bak`, no
atomic rename. man zstd confirms `-f, --force: disable input and output
checks. Allows overwriting existing files`.

**Two distinct claims to disambiguate**:

1. *Does the script itself perform a live swap?* No — the script's flow
   stops at rsync. The operator pastes the printed command manually. The
   docstring is honest about "the script doesn't do this".
2. *Is the printed command safe?* No — it is `unzstd -k -f` (force
   overwrite of the live file) with no temp-name, no atomic rename, no
   rollback. If the operator pastes during training (forgetting the
   "idleness" warning), mmap-readers in `train_phase2.py` / `train_easy.py`
   may SIGBUS, and any WAL-mode writers undefined. If `unzstd` itself
   aborts mid-frame, `trading_data.db` is left in a partial state with no
   restore path (the printed snippet writes the new .db **before** the
   sqlite3 integrity_check runs, so the integrity_check would catch a
   corrupted output — but the live DB is already clobbered).

So the documentation framing (~"safe, manual, no live swap") is *narrowly
true* (the script doesn't auto-execute) but *operationally misleading*
(the manual command IS a destructive in-place overwrite). The hunt's F14
captures this drift; F2 is the underlying mechanism.

**Severity sharpening**: P0 → P1.
- The script does not, on its own, corrupt a running server. The script
  prints instructions; the operator runs them.
- The script DOES warn explicitly (twice: docstring + post-rsync printed
  warning) about training-must-be-idle. This is operator discipline, not a
  defect the script *executes*.
- If the operator follows the warnings, the printed command works fine —
  unzstd writes to .db, sqlite3 reads .db (no concurrent writer), and the
  flow is sound.
- Risk profile: P1 (foot-gun pattern, downgraded from P0 because the
  damage requires an additional operator error layered on top of the
  script's actual behavior). Treat as a hardening item, not a blocker.

**Recommended fix** — option (c), a hybrid:
- (c1) Change the printed snippet to use a temp filename + atomic rename:
  ```bash
  unzstd -k trading_data.db.zst -o trading_data.db.NEW &&
  sqlite3 trading_data.db.NEW 'PRAGMA integrity_check' &&
  mv trading_data.db trading_data.db.OLD &&
  mv trading_data.db.NEW trading_data.db
  ```
  This makes the integrity_check gate the swap (failed integrity → no
  overwrite), and gives a `trading_data.db.OLD` rollback. The atomicity of
  the two `mv`s on the same filesystem is sound (rename(2) is atomic).
- (c2) Optionally add a pre-check the operator can run:
  `pgrep -f 'train_(easy|phase2|jax_fast)\.py' && echo "TRAINING IS RUNNING — abort"`
  prepended to the printed snippet.
- (c3) Update README line 12 + script docstring lines 10-14 to describe
  the new flow as "atomic server-side swap with integrity-gated rename"
  rather than "no live swap".

If only one is done: (c1) is the highest-leverage fix; it removes the
foot-gun pattern without operator behavior change.

---

### F3: CONFIRMED — P1

Line 65: `zstd -f --keep -19 -o "$ZST" "$DB"`. man zstd does not document
atomic-rename semantics; zstd's CLI streams compressed frames directly to
the output file (no `.tmp` + rename pattern in its source code). On SIGINT
/ SIGKILL / OOM mid-write, the output file is left in a partially-written
state. Its mtime is "now" (after the input DB's frozen mtime).

The next run hits line 63: `if [[ ! -f "$ZST" ]] || [[ "$DB" -nt "$ZST" ]]`.
The partial `.zst` exists (.zst file test true) and `.zst` mtime is NEWER
than `.db` mtime (the DB has not been written). So `$DB -nt $ZST` is
false. Both conditions are false → skip the compress block, print the
"skipping compress" message, proceed to step 3.

Step 3 (line 75): `sha256sum "$ZST" > "$SHA"` — produces a deterministic
hash of the corrupt .zst, which "matches itself" on server-side verify.
Step 4 rsyncs the corrupt .zst to the server. Workstation user sees
"==> Workstation sync done."

Server-side: `unzstd -k -f` will likely fail on the truncated zstd frame
(zstd has per-frame XXH64 checksums), printing an error. The hunt is
correct: the FAILURE is detected server-side, but only if the operator
runs the printed snippet. The workstation-side gate has shipped corrupt
data without complaint.

**Severity P1 confirmed**: real failure mode, requires interrupt during
compress (operator ^C, system suspend, OOM kill of zstd at -19), detection
is server-mediated and operator-discipline-gated. The May-12 incident
is exactly this class (silent ship of broken bytes).

**Recommended fix**: any of:
- (a) `zstd -t "$ZST"` (test frame integrity) immediately after compress
  succeeds; if -t fails, rm the .zst and re-compress.
- (b) Compress to `.zst.tmp`, then `mv` to `.zst` on success — atomic.
- (c) Trap on EXIT inside the compress branch to `rm -f "$ZST"` if zstd
  was running when the trap fires.

(b) is the smallest change.

---

### F4: CONFIRMED LATENT — P1 with a clarification

Workstation has no `trading_data.db-wal` or `trading_data.db-shm` files
(verified: directory listing shows only `trading_data.db`,
`trading_data.db.zst`, `trading_data.db.zst.sha256`). So the DB is
currently either in DELETE journal mode (default) or in WAL mode with a
clean checkpoint (no pending WAL pages). The script's omission of
`PRAGMA wal_checkpoint(TRUNCATE)` is currently a no-op concern.

If a future writer (e.g. Kraken re-fetch, per TASK_768 §"Staleness
decision") flips the DB into WAL mode and produces uncheckpointed pages:
- `PRAGMA integrity_check` correctly sees the WAL + main combined view
  and reports "ok" (so the gate doesn't false-negative).
- `zstd` compresses only `trading_data.db` (the main file) — NOT the
  `-wal` or `-shm` files. The shipped artifact would be a stale
  pre-WAL-pages snapshot.
- Server-side `PRAGMA integrity_check` after decompress would also report
  "ok" (the main file is internally consistent), so the gate doesn't
  catch the staleness either.
- The user would silently ship a DB missing the latest committed
  transactions.

So the gap is **silent staleness** in the WAL-mode case, not silent
corruption. Confirmed as latent. P1 is reasonable given TASK_768
explicitly anticipates a Kraken re-fetch.

**Recommended fix**: prepend before line 50:
```bash
sqlite3 "$DB" 'PRAGMA wal_checkpoint(TRUNCATE);' >/dev/null
```
Cheap (no-op on non-WAL DBs), and resolves the latent gap before it bites.

---

### F5: CONFIRMED — P1

Verified `.zst` mtime: `2026-01-20 23:22:06` (per `stat`). Current date:
2026-05-23. The .zst on disk is **4 months stale** relative to whatever
fresh DB content might be in `trading_data.db` (the .db itself shares the
mtime, but TASK_768 anticipates Kraken re-fetch which would touch the .db
and leave the .zst behind).

Re-tracing the `--skip-compress` flow:
- Line 62 `if [[ -z "$SKIP_COMPRESS" ]]; then` — skipped entirely when
  flag is set.
- Line 70 `[[ -f "$ZST" ]] || ...` — .zst exists; OK.
- Line 75 — regenerates sha256 from the (stale) .zst.
- Lines 87-89 — rsyncs the stale .zst + matching fresh sha256.

Server-side: sha256 verifies, decompress produces a stale-but-internally-clean
.db, integrity_check on the stale .db passes (it was clean when frozen in
Jan-20). All gates green. Stale data deployed silently.

**Severity P1 confirmed**: same shape as F4 (silent staleness, not silent
corruption). Easy to trigger: just run `sync-db.sh --skip-compress` after
the .db has been refreshed and forgotten to remove the old .zst.

**Recommended fix**: in the `--skip-compress` branch, if `$DB -nt $ZST`,
warn loudly and require an additional `--i-know-its-stale` flag. Or:
remove `--skip-compress` entirely — the default mtime-skip on line 63
already covers the "skip if already fresh" case.

---

### F6: VERIFIED CLEAN (was P1? → verified-clean)

The script picks `integrity_check`, which is the superset of `quick_check`.
SQLite docs: `quick_check` is integrity_check minus index/row-content
cross-reference verification and minus UNIQUE/NOT-NULL/CHECK row
verification. The May-12 corruption mode (unreadable `sqlite_master`)
would be caught by both, but other corruption modes (e.g. index out of
sync with table, freelist errors) are caught only by `integrity_check`.

TASK_768:98 mandates `integrity_check`. The script complies. The hunt
itself rates this as "verified-clean-ish but flagged because the prompt
asked"; my read is the same. Move to verified-clean.

---

### F7: CONFIRMED — P2

Line 51 `RESULT="$(sqlite3 "$DB" 'PRAGMA integrity_check' 2>&1)"` merges
stderr into RESULT. If sqlite3 emits any stderr text before/after the "ok"
(load-extension warning, locale warning, deprecation, banner from a
sourced `~/.sqliterc`, debug build verbosity), RESULT becomes
`<noise>\nok` (or `ok\n<noise>` trimmed of trailing newline). The exact
string comparison `[[ "$RESULT" != "ok" ]]` triggers → false corruption
report.

A `.sqliterc` with `.echo on` or `.headers on` would absolutely break
this; verified there's no `.sqliterc` setup mentioned in lore but the
script is brittle to one being introduced.

**Severity P2 confirmed**: brittle but not currently triggering. Fix
candidates: drop `2>&1`, OR check tail line only:
`[[ "$(echo "$RESULT" | tail -1)" != "ok" ]]`, OR invoke as
`sqlite3 -batch -bail "$DB" 'PRAGMA integrity_check'` (suppress init
files).

---

### F8: CONFIRMED — P2 / philosophical

The hunt accurately characterizes the residual hole: between step 1
(integrity_check) and step 2 (compress), another process could write to
the DB, making the integrity gate validate a different byte sequence than
what ships. Probability is low (no scheduled writer on the workstation
DB today), but the script doesn't take a SQLite write lock to bridge the
gap.

A complementary observation: the script's own comment on line 72-73
("catches rsync corruption only — NOT source-side SQLite corruption,
which is what step 1 is for") is accurate but understates the residual
hole — step 1's bytes are not bound to step 2's bytes.

**Severity P2 confirmed**: real but low-probability. Stronger fix would
be to hash the *uncompressed* .db into the .sha256 metadata as well, so
the server-side verify can confirm post-decompress bytes match what step
1 gated.

---

### F9: CONFIRMED — P2 (low impact)

Lines 87-89, rsync to `$SRV_USER@$SRV_HOST:$SRV_PATH/`. No `--mkpath`, no
precheck. If `~/Trader/` does not exist on server, rsync errors with
"failed to set times on …: No such file or directory" or similar. Per
README, this is solved by `tools/deploy/bootstrap-server.sh`; sync-db.sh
doesn't reference it.

Low impact (operator has already bootstrapped by the time they sync a
DB). Recommended fix: pre-rsync, `ssh $SRV_USER@$SRV_HOST "test -d
$SRV_PATH" || { echo "run bootstrap-server.sh first" >&2; exit 1; }`.

---

### F10: VERIFIED CLEAN-ISH — P2

The hunt's analysis is right. Default rsync uses mtime+size heuristic for
the resume/skip decision, but the per-file delta-transfer algorithm
catches changes. The server-side `sha256sum -c $SHA` is the final integrity
gate for transit corruption (since rsync uses TCP checksums + its own
rolling checksums internally, but the manual sha256 step is the canonical
guarantee). The chain holds.

`-P` does imply `--partial`; the explicit `--partial` is redundant.
Cosmetic.

Verified clean.

---

### F11: CONFIRMED — P3

Line 27 `REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"` resolves to an
absolute path regardless of how the script was invoked. Line 28 `cd
"$REPO_ROOT"` then changes to repo root. But `$0` is preserved by bash
as-typed: if user runs `./sync-db.sh --help` from `tools/deploy/`, then
`$0` = `./sync-db.sh`. After `cd "$REPO_ROOT"` we're in
`/home/lukas/Trader/`; `./sync-db.sh` is NOT in that cwd. `sed -n
'2,21p' "./sync-db.sh"` → "No such file or directory", exit 2 from sed
(under `set -e` → script exits without printing help).

**Severity P3 confirmed**: cosmetic, --help is the only affected path.

Fix: resolve `$0` to an absolute path before the cd, e.g.
`SCRIPT_PATH="$(realpath "$0" 2>/dev/null || echo "$0")"` early, then
use `"$SCRIPT_PATH"` in sed.

---

### F12: CONFIRMED — P3

Case statement on lines 38-44:
```
case "$arg" in
  --check-only)    ... ;;
  --skip-compress) ... ;;
  --dry-run)       ... ;;
  -h|--help)       ... ;;
  -*)              echo "unknown flag: $arg" >&2; exit 2 ;;
esac
```
A bareword `checkonly` (no leading hyphen) does not match `--check-only`,
`--skip-compress`, `--dry-run`, `-h`, `--help`, or `-*`. There is no
[redacted: VPS ops detail]
the script proceeds with default flag values.

**Severity P3 confirmed**: minor (typos like `checkonly` instead of
`--check-only` should fail loudly). Fix: add `*) echo "unknown arg:
$arg" >&2; exit 2 ;;` after `-*)`.

---

### F13: CONFIRMED — P3

Line 42 `sed -n '2,21p' "$0"` hardcodes the help-text line range. Add a
docstring line, all subsequent help slices shift. Trivial maintenance
hazard.

**Severity P3 confirmed**. Fix as the hunt sketched: delimit the help
block with sentinel comments and sed between them.

---

### F14: CONFIRMED — P2 (downstream of F2)

The README at `tools/deploy/README.md:12` describes the script as
"**integrity-gated**, manual server-side swap". The script docstring
(lines 10-14) says "this script does NOT live-swap the DB on the
server". The printed command IS a destructive in-place overwrite (see
F2). The drift between "no live swap" / "manual swap" framing and the
literal `unzstd -k -f` behavior is real.

If F2 is fixed via the temp-name + atomic-rename flow, the README/docstring
become accurate (manual but atomic, integrity-gated, no in-place
destruction). Until then, the documentation overstates safety.

**Severity P2 confirmed**: documentation honesty issue, fix paired with F2.

---

## Severity sharpening — final consensus list

| Finding | Hunt severity | Cross-validation severity | Reasoning |
|---|---|---|---|
| F1 | P0 | **P1** | Failure-path-only; doesn't ship corruption; success path works |
| F2 | P0 | **P1** | Script doesn't auto-execute; operator-discipline-gated; foot-gun pattern but warned twice |
| F3 | P1 | **P1** | Confirmed — real interrupted-compress hole, ships corrupt .zst |
| F4 | P1 | **P1** | Confirmed latent — bites when DB flips to WAL mode |
| F5 | P1 | **P1** | Confirmed — `--skip-compress` ships 4-month-stale .zst silently |
| F6 | P1? | **verified clean** | Right tool picked; was self-flagged by hunt |
| F7 | P2 | **P2** | Confirmed — brittle stderr handling |
| F8 | P2 | **P2** | Confirmed — TOCTOU between integrity-check and compress |
| F9 | P2 | **P2** | Confirmed — no SRV_PATH precheck |
| F10 | P2 | **verified clean** | Hunt self-marked, chain holds via manual sha256 |
| F11 | P3 | **P3** | Confirmed — --help breaks when invoked from tools/deploy/ |
| F12 | P3 | **P3** | Confirmed — bareword typos silently ignored |
| F13 | P3 | **P3** | Confirmed — fragile help-slice line range |
| F14 | P2 | **P2** | Confirmed — drift paired with F2 |

**Net delta vs hunt**: F1 and F2 demoted P0 → P1; F6 demoted P1? →
verified-clean; F10 confirmed as verified-clean (was P2 in hunt's header
but marked "verified-clean-ish" inline). Effective P1 list: F1, F2, F3,
F4, F5. P2: F7, F8, F9, F14. P3: F11, F12, F13.

### Should the script be blocked from merge / use?

Not on F1 or F2.

- **F1**: the only realistic trigger is sqlite3 dying (OOM/segfault) on a
  loaded workstation during integrity_check. If it triggers, the script
  exits silently — the operator notices the exit code and reruns, or
  investigates. *No corruption ships to the server.* Compare to "script
  ships corrupt DB silently" (the 2026-05-12 incident shape): F1 does
  not have that shape.
- **F2**: the script itself does not destroy the live server DB. The
  operator copies the printed command. The script warns about idleness
  twice. If a foot-gun command sequence is the bar for P0, then F2 is
  P0; if "running this script under documented use destroys data" is the
  bar, F2 is P1. I prefer the latter.

Both F1 and F2 deserve fixing — they sharpen the gate's safety margin —
but neither blocks the gate's *primary use*. The script's commit message
("closes the 2026-05-12 hole") is accurate: the source-side integrity
check IS the hole-closer, and that mechanism works on the success path.

If a P0 *did* exist, it would be: the script ships corruption to the
server under realistic conditions despite reporting success. F3 (silent
ship of corrupt .zst after interrupted compress) is closest to that
shape but requires an interrupt event to trigger — P1 is right.

---

## New findings (cross-validation sweep)

### N1 (P2): server-side disk-full on decompress is unprotected

The printed server-side snippet (`unzstd -k -f trading_data.db.zst`)
decompresses ~8 GB .zst → ~24 GB .db. If the server's `$HOME` filesystem
is short of ~25 GB free, unzstd fails partway, leaving a partial
`trading_data.db`. Coupled with F2's in-place overwrite, this clobbers
the live DB into a half-written state. The snippet has no
`df`-precheck.

Fix: prepend `df -h .` to the printed snippet, or have the snippet check
explicitly: `[[ "$(df -BG --output=avail . | tail -1 | tr -d ' G')" -gt 25 ]] || exit 1`.
The temp-name + atomic-rename fix from F2 would also bound the damage
(partial write lands in `.NEW`, not the live file).

### N2 (P3): `sha256sum -c` format compatibility note (not currently a bug)

The script generates the .sha256 via GNU coreutils `sha256sum "$ZST" > "$SHA"`,
which produces `<hash><sp><sp><filename>` text-mode format. The server-side
snippet uses `sha256sum -c` — also GNU. The server is Ubuntu 20.04, so GNU
coreutils is the default. **Verified clean for the current deployment**.

Flag for future portability only: if the receiving host ever changes to
BSD/macOS (where `shasum -a 256 -c` is the equivalent and has slightly
different output parsing), the `-c` step would fail. Not a current
concern. Document briefly if SRV_HOST is parameterized for general use.

### N3 (P2): rsync atomicity of `$SHA` may decouple from `$ZST`

rsync transfers files in the order specified, atomically renaming each
on completion (no `--inplace`). If the rsync is interrupted mid-flight,
one file may have landed and the other not. Specifically: `$ZST` is large
(8 GB), `$SHA` is tiny (~90 bytes). If rsync goes alphabetically,
`trading_data.db.zst` transfers first; if interrupted mid-`.zst` transfer,
the server has old `.zst` (atomic rename didn't happen) and old `.sha256`
— consistent old state. **Verified clean** for this ordering.

But if rsync ever transferred `.sha256` first (e.g. if the user listed
them in a different order, or in a future change), the server could end
up with a new `.sha256` mismatched against an old/partial `.zst`. The
server-side `sha256sum -c` would catch this (mismatch → operator notices
and reruns). Not a silent bug. Flag for awareness.

### N4 (P3): environment-variable defaults hardcode a specific host

Lines 78-81 default to `operator@<redacted-ip>:22:Trader`. Documented in
the help block. If the operator wants to deploy to a different host (e.g.
a backup server), they override via env vars. This is fine but undocumented
in the README — the help block mentions the env overrides but only
inline.

Not a defect; consider a one-liner in README's "Transports" table
mentioning the override pattern for future multi-target use.

### N5 (P2): `--check-only` → full sync race

Workflow: operator runs `sync-db.sh --check-only`, sees "ok", then
immediately reruns `sync-db.sh` (full) to ship. If a background process
(Kraken fetch, manual sqlite session) writes to the DB between the two
invocations, the second invocation's integrity_check (line 51) catches
it — but if it's a *fast* write that finishes between the second
invocation's integrity_check and zstd start, that's the F8 race again.

This is the same TOCTOU as F8, just with an extra inter-invocation gap
that widens the window. Same fix (write-lock the DB during the
integrity_check + compress span) applies.

Not a new mechanism, but an articulation of when F8's window is widest.

### N6 (P3): heredoc in the printed snippet uses outer-shell `cd $SRV_PATH`

Lines 92-107 print the server-side snippet with `cd $SRV_PATH &&` —
`$SRV_PATH` is expanded by the workstation shell (heredoc with unquoted
EOF). The remote shell sees `cd Trader && ...`. If a future change quotes
the EOF (`<<'EOF'`) without updating the snippet to use `${SRV_PATH}`
unconditionally on the remote side, the snippet breaks. Cosmetic
fragility.

### N7 (P3): `du -h | cut -f1` is locale-dependent

Line 49 `SIZE="$(du -h "$DB" | cut -f1)"`. In locales where `du -h`
outputs `24G` (en_US) vs `24G\n` vs comma-decimal locales (`24,5G` in
some German locales), the displayed banner could be malformed but doesn't
break anything. Cosmetic.

---

## Verified-clean re-confirmed

Re-affirming the hunt's verified-clean list and adding any that came up
in the cross-validation:

- The "ok" exact-match comparison (line 52): bash strips trailing newline
  from `$(...)` substitution, so `RESULT == "ok"` exactly for clean DBs.
  Multi-line corruption reports do not match. Confirmed independently.
- sha256sum output format (GNU coreutils): two-space separator, bare
  basename, single line. Verified on disk (file content:
  `d33eec...  trading_data.db.zst`). Server-side `sha256sum -c $SHA` after
  `cd Trader` resolves correctly. No path-prefix bug.
- Heredoc variable expansion (lines 92-107): unquoted EOF; workstation
  expands `$SRV_PORT`, `$SRV_USER`, `$SRV_HOST`, `$SRV_PATH`, `$SHA`,
  `$ZST`, `$DB` before showing the snippet. Operator copy-pastes literal
  values. Confirmed.
- Single-quote nesting in the printed ssh snippet: outer `'...'` wraps
  the remote command, the inner `"PRAGMA integrity_check"` is preserved
  literally inside single quotes. Remote receives the intact double-quoted
  pragma argument. Confirmed.
- `integrity_check` vs `quick_check` choice: integrity_check is the
  strict superset; TASK_768:98 mandates it. Confirmed (F6 cleared).
- `rsync --partial` resume semantics: rsync defaults handle this
  correctly (resume + truncate-on-shrink). The `-P` flag implies
  `--partial`. Confirmed.
- Flag combinations: `--check-only` exits before compress; `--dry-run`
  is checked at every action site (lines 65, 75, 84); `--skip-compress`
  correctly bypasses the compress block. sha256 always regenerates from
  on-disk .zst after the compress branch (no silent skip + ship without
  sha). Confirmed.
- `set -euo pipefail`: right tool for the job, modulo the F1 ergonomics
  gap. Confirmed.
- Path normalization (line 27): `cd "$(dirname "$0")/../.." && pwd`
  resolves to an absolute path regardless of how the script was invoked,
  modulo the F11 issue with `$0` itself (which is preserved as-typed).
  Confirmed.

---

## Out of scope (same as hunt)

- Whether the operator actually runs the printed server-side snippet
  after every sync (workflow discipline, not script defect).
- Whether the server's `sqlite3` / `unzstd` versions match the workstation's
  capabilities (no version probe in the script).
- Filesystem-level atomicity of the F2 fix's `mv` on the server (depends
  on same-device guarantee; per lore likely fine).
- Wallclock cost of integrity_check on a 24 GB DB (May-22 used
  quick_check; full integrity_check runtime unmeasured here; only an
  ergonomics concern if it routinely exceeds operator patience).

---

## Bottom line

Hunt is broadly correct on mechanism. The two P0s are real issues but the
severity grade is too hot — both are P1 by the "ships corruption to
server / destroys live data under documented use" bar that the script's
own commit message ("closes the 2026-05-12 hole") sets. F3 is the closest
to a true P0 shape (silent ship of corrupt .zst after interrupted
compress) and the hunt rates it P1 — defensibly, because the corruption
is detected server-side IF the operator runs the printed snippet.

The chain holds in practice. The script makes the 2026-05-12 incident
much harder to repeat. The findings above are hardening recommendations,
not blockers. Recommended fix priorities, in order: F3 (atomic .zst.tmp
+ rename), F2 (atomic .NEW + rename on server, change docstring), F1
(explicit rc capture), F4 (preemptive `wal_checkpoint(TRUNCATE)`), F5
(`--skip-compress` staleness warn). F7-F14 + N1-N7 are polish.
