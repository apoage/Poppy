#!/usr/bin/env bash
# test_run_queue_watchdog.sh — hermetic smoke for the TASK_841 checkpoint-save WEDGE watchdog
# in tools/deploy/run_queue.sh. Exercises the REAL run_queue.sh + REAL tools/safe_run.sh
# (real systemd --user scopes) with NO GPU and NO real training: a stub "job" writes a training
# CSV then STOPS writing while staying alive (the exact wedge signature — process alive, CSV
# frozen), so the watchdog must kill + archive + relaunch it.
#
# The watchdog is OPT-IN (default --stall-timeout 0 = OFF); every scenario ARMS it explicitly
# with a small --stall-timeout. Three scenarios:
#   A (always-wedge): first wedge -> WEDGED-RETRIED (auto-retry); second wedge on the same job
#                     -> FAILED row (wedged-failed) + non-zero exit (fail loud).
#   B (recover-on-retry): first wedge -> WEDGED-RETRIED; the retry keeps the CSV fresh and exits
#                     0 -> one clean row, WEDGED-RETRIED noted, queue exit 0.
#   C (CSV-less immunity): a job that produces NO training CSV (no csvglob:, no --name) must run
#                     past stall-timeout+grace and finish cleanly — NEVER killed (the eval/
#                     benchmark/data-prep workload must be immune even with the watchdog armed).
#
# Hermetic: the CSV lives in scratch (declared via the manifest `csvglob:` token), a per-job
# state file carries the attempt count across relaunches, RUN_QUEUE_LOCK_DIR redirects the
# host-global GPU locks into scratch, and the runs/<sweep> dir is removed on exit.
#
# Usage: tools/tests/test_run_queue_watchdog.sh   (exit 0 = all scenarios pass)
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RQ="$REPO_ROOT/tools/deploy/run_queue.sh"
[[ -x "$RQ" ]] || { echo "FATAL: $RQ not executable"; exit 2; }

if ! systemctl --user show-environment >/dev/null 2>&1; then
  echo "SKIP: systemd --user unavailable — run_queue launches every job via safe_run scopes,"
  echo "      which need systemd --user. Cannot exercise the watchdog here (not a pass/fail)."
  exit 0
fi

T="$(mktemp -d "${TMPDIR:-/tmp}/rqwatch.XXXXXX")"
export RUN_QUEUE_LOCK_DIR="$T/locks"
SWEEPS=(); SCOPE_GLOBS=()
cleanup() {
  local s g
  for g in "${SCOPE_GLOBS[@]:-}"; do [[ -n "$g" ]] && systemctl --user stop "$g" 2>/dev/null || true; done
  for s in "${SWEEPS[@]:-}"; do [[ -n "$s" ]] && rm -rf "$REPO_ROOT/runs/$s"; done
  rm -rf "$T"
}
trap cleanup EXIT

# ---- stub job: write a training CSV, then either wedge (stop writing, stay alive) or, on a
#      later attempt in "recover" mode, keep the CSV fresh and exit clean. ------------------
cat > "$T/stub.sh" <<'EOF'
#!/usr/bin/env bash
job="$1"; csv="$2"; state="$3"; mode="$4"
mkdir -p "$(dirname "$csv")" "$(dirname "$state")"
attempt=1
[[ -f "$state" ]] && attempt=$(( $(cat "$state") + 1 ))
echo "$attempt" > "$state"
echo "[stub] $job attempt=$attempt mode=$mode" >&2
printf 'epoch,reward\n%d,1.0\n' $((100 + attempt)) > "$csv"    # first CSV write (epoch field)
if [[ "$mode" == "recover" && "$attempt" -ge 2 ]]; then
  for i in $(seq 1 6); do
    printf 'epoch,reward\n%d,1.0\n' $((100 + attempt + i)) > "$csv"   # keep mtime fresh
    sleep 1
  done
  echo "[stub] $job recovered on attempt=$attempt -> exit 0" >&2
  exit 0
fi
echo "[stub] $job WEDGING (no more CSV writes), sleeping alive" >&2
sleep 120
EOF
chmod +x "$T/stub.sh"

PASS=0; FAIL=0
ok()  { echo "  PASS: $1"; PASS=$((PASS+1)); }
bad() { echo "  FAIL: $1"; FAIL=$((FAIL+1)); }

# run_scenario runs in the PARENT shell (not $(...)), so SWEEPS/SCOPE_GLOBS registrations and
# RC survive for cleanup + assertions. Results: $RC (exit code) and $SW (sweep dir name).
run_scenario() {  # $1 tag $2 mode
  local tag="$1" mode="$2"
  local job="w${tag}"
  local csv="$T/csv_${tag}/${job}.csv" state="$T/state_${tag}/${job}.attempt"
  local man="$T/${tag}.man"
  SW="_rqwatch_$$_${tag}"
  SWEEPS+=("$SW"); SCOPE_GLOBS+=("${job}_*.scope")
  # csvglob: token names the CSV to watch (stripped before the command runs).
  printf '%s cpu 1G csvglob:%s %s %s %s %s %s\n' \
    "$job" "$csv" "$T/stub.sh" "$job" "$csv" "$state" "$mode" > "$man"
  "$RQ" "$man" --poll 1 --stall-timeout 3 --stall-grace 2 --max-cpu 2 --sweep "$SW" \
    >"$T/${tag}.out" 2>&1
  RC=$?
}

# =====================================================================================
echo "================================================================================"
echo "SCENARIO A: job wedges every attempt -> WEDGED-RETRIED once, then FAILED (2nd wedge)"
echo "================================================================================"
run_scenario A wedge
SUM_A="$REPO_ROOT/runs/$SW/queue_summary.md"
echo "--- run_queue exit=$RC (stdout tail) ---"; tail -n 8 "$T/A.out"
echo "--- summary ---"; sed 's/^/    /' "$SUM_A" 2>/dev/null || echo "    (no summary)"
grep -q "WEDGE detected" "$T/A.out"    && ok "A wedge detected"                 || bad "A no 'WEDGE detected'"
grep -q "WEDGED-RETRIED" "$T/A.out"    && ok "A first wedge -> auto-retry fired" || bad "A no WEDGED-RETRIED (retry) log"
grep -q "WEDGE-FAILED"   "$T/A.out"    && ok "A second wedge -> fail-loud fired" || bad "A no WEDGE-FAILED log"
[[ "$RC" -ne 0 ]]                      && ok "A queue exit non-zero"             || bad "A expected non-zero exit, got $RC"
grep -q "wedged-failed"  "$SUM_A" 2>/dev/null && ok "A wedged-failed row in summary" || bad "A no wedged-failed row"
grep -q "auto-retried after a checkpoint-save wedge (WEDGED-RETRIED)" "$SUM_A" 2>/dev/null \
  && ok "A WEDGED-RETRIED count noted in summary" || bad "A no WEDGED-RETRIED summary note"
ls "$T/csv_A/"*_wedged_ep*.csv >/dev/null 2>&1 && ok "A partial CSV archived with _wedged_ep suffix" \
  || bad "A no _wedged_ep CSV archive"
ls "$(dirname "$SUM_A")/"*_wedged_ep*.log >/dev/null 2>&1 && ok "A run log archived with _wedged_ep suffix" \
  || bad "A no _wedged_ep run-log archive"
# exactly TWO wedges detected (attempt 1 + attempt 2), never a 3rd (retry is capped at 1)
nwedge="$(grep -c "WEDGE detected" "$T/A.out")"
[[ "$nwedge" -eq 2 ]] && ok "A exactly 2 wedges (retry capped at 1)" || bad "A expected 2 wedges, saw $nwedge"

# =====================================================================================
echo "================================================================================"
echo "SCENARIO B: job wedges once, its auto-retry recovers -> WEDGED-RETRIED, clean exit 0"
echo "================================================================================"
run_scenario B recover
SUM_B="$REPO_ROOT/runs/$SW/queue_summary.md"
echo "--- run_queue exit=$RC (stdout tail) ---"; tail -n 8 "$T/B.out"
echo "--- summary ---"; sed 's/^/    /' "$SUM_B" 2>/dev/null || echo "    (no summary)"
grep -q "WEDGED-RETRIED" "$T/B.out"    && ok "B first wedge -> auto-retry fired" || bad "B no WEDGED-RETRIED (retry) log"
! grep -q "WEDGE-FAILED" "$T/B.out"    && ok "B retry recovered (no WEDGE-FAILED)" || bad "B unexpected WEDGE-FAILED"
[[ "$RC" -eq 0 ]]                      && ok "B queue exit 0 (retry succeeded)"  || bad "B expected exit 0, got $RC"
! grep -q "wedged-failed" "$SUM_B" 2>/dev/null && ok "B no wedged-failed row"     || bad "B unexpected wedged-failed row"
grep -q "auto-retried after a checkpoint-save wedge (WEDGED-RETRIED)" "$SUM_B" 2>/dev/null \
  && ok "B WEDGED-RETRIED count noted in summary" || bad "B no WEDGED-RETRIED summary note"

# =====================================================================================
echo "================================================================================"
echo "SCENARIO C: CSV-less job, watchdog ARMED -> must run past timeout+grace and NOT be killed"
echo "================================================================================"
SW="_rqwatch_$$_C"; SWEEPS+=("$SW"); SCOPE_GLOBS+=("wC_*.scope")
MANC="$T/C.man"
# No csvglob:, no --name -> INFERRED as logs/training_default.csv, which this job never writes.
# It sleeps 8s (> stall-timeout 3 + grace 2 = 5s) then exits 0. Must survive untouched.
printf 'wC cpu 1G sleep 8\n' > "$MANC"
"$RQ" "$MANC" --poll 1 --stall-timeout 3 --stall-grace 2 --max-cpu 2 --sweep "$SW" >"$T/C.out" 2>&1
RC=$?
SUM_C="$REPO_ROOT/runs/$SW/queue_summary.md"
echo "--- run_queue exit=$RC (stdout tail) ---"; tail -n 6 "$T/C.out"
echo "--- summary ---"; sed 's/^/    /' "$SUM_C" 2>/dev/null || echo "    (no summary)"
grep -q "watchdog: armed" "$T/C.out"  && ok "C watchdog was armed"                 || bad "C watchdog not armed"
! grep -q "WEDGE detected" "$T/C.out" && ok "C CSV-less job NOT wedged (immune)"   || bad "C CSV-less job was wrongly wedged"
[[ "$RC" -eq 0 ]]                     && ok "C queue exit 0 (job ran clean)"        || bad "C expected exit 0, got $RC"
grep -qE '^\| wC \| 0 \|' "$SUM_C" 2>/dev/null && ok "C clean rc=0 row in summary"  || bad "C no clean row"

echo "================================================================================"
echo "RESULT: $PASS passed, $FAIL failed"
echo "================================================================================"
[[ $FAIL -eq 0 ]]
