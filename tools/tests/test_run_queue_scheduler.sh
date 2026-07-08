#!/usr/bin/env bash
# test_run_queue_scheduler.sh — committable mock harness for tools/deploy/run_queue.sh.
#
# Exercises the REAL run_queue.sh + REAL tools/safe_run.sh (real systemd --user scopes)
# with ONLY nvidia-smi stubbed, proving the fail-closed scheduler fixes (TASK_833):
#   #1 subshell GPU acquisition  — jobs distribute across the pool, not all on GPU 1
#   #2 reservation / no double-book — no two overlapping jobs ever share a GPU; a full
#      pool QUEUES the next job instead of double-booking
#   #5 unbounded idle-wait          — a never-idle pool times out + SKIPS (non-zero exit)
#   #4 malformed manifest lines     — reported as failed rows + force non-zero exit
#   #3 cleanup trap                 — SIGTERM mid-run stops the job's scope unit (no orphan
#      scope/child), writes a partial summary with an `interrupted` row, exits 130
#
# Isolation: each scenario resets the mock nvidia-smi state dir (so a SIGKILLed job's stale
# "busy" file can't pollute a later scenario), and RUN_QUEUE_LOCK_DIR redirects run_queue's
# host-global per-GPU locks into scratch so this test never contends on production locks.
#
# No real GPU work: nvidia-smi is a PATH stub reporting per-GPU memory from a state dir;
# the "jobs" are timed stubs that log `<job> START gpu=$CUDA_VISIBLE_DEVICES ts=...` and
# `END`, and mark their GPU busy in the state dir for the stub nvidia-smi to observe.
#
# Usage: tools/tests/test_run_queue_scheduler.sh   (exit 0 = all scenarios pass)
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RQ="$REPO_ROOT/tools/deploy/run_queue.sh"
[[ -x "$RQ" ]] || { echo "FATAL: $RQ not executable"; exit 2; }

T="$(mktemp -d "${TMPDIR:-/tmp}/rqtest.XXXXXX")"
STATE="$T/state"; mkdir -p "$T/bin" "$STATE"
export RUN_QUEUE_LOCK_DIR="$T/locks"   # keep run_queue's host-global GPU locks off the shared path
SWEEPS=()      # runs/<sweep> dirs to clean up
SCOPE_GLOBS=() # leftover systemd --user scope globs to reap on exit (belt-and-suspenders)
cleanup() {
  local s g
  for g in "${SCOPE_GLOBS[@]:-}"; do [[ -n "$g" ]] && systemctl --user stop "$g" 2>/dev/null || true; done
  for s in "${SWEEPS[@]:-}"; do [[ -n "$s" ]] && rm -rf "$REPO_ROOT/runs/$s"; done
  rm -rf "$T"
}
trap cleanup EXIT

# Reset the mock nvidia-smi state dir so a SIGKILLed job's stale "busy" file (its own cleanup
# never ran) can't leak into a later scenario. Call at the top of every scenario.
reset_state() { rm -rf "$STATE"; mkdir -p "$STATE"; }

# ---- stub nvidia-smi: memory.used per GPU from $STATE/gpu<N>.mem (idle=5 if absent) ----
cat > "$T/bin/nvidia-smi" <<EOF
#!/usr/bin/env bash
STATE="$STATE"
gpu=""
while [[ \$# -gt 0 ]]; do
  case "\$1" in
    -i) gpu="\$2"; shift 2;;
    -i*) gpu="\${1#-i}"; shift;;
    *) shift;;
  esac
done
f="\$STATE/gpu\${gpu}.mem"
if [[ -f "\$f" ]]; then cat "\$f"; else echo 5; fi
EOF
chmod +x "$T/bin/nvidia-smi"

# ---- stub job: mark GPU busy, log START/END with CUDA_VISIBLE_DEVICES, sleep ----
cat > "$T/stub_job.sh" <<'EOF'
#!/usr/bin/env bash
job="$1"; secs="$2"; state="$3"
g="${CUDA_VISIBLE_DEVICES:-none}"
mkdir -p "$state"
echo 8000 > "$state/gpu${g}.mem"           # this GPU now "busy" for the mock nvidia-smi
echo "$job START gpu=$g ts=$(date +%s.%N)"
python3 -c "import time,sys; time.sleep(float(sys.argv[1]))" "$secs"
echo "$job END gpu=$g ts=$(date +%s.%N)"
rm -f "$state/gpu${g}.mem"
EOF
chmod +x "$T/stub_job.sh"

export PATH="$T/bin:$PATH"   # shadow the real nvidia-smi

# ---- python interval parser / asserter ----
cat > "$T/parse.py" <<'PY'
import sys, os, re, glob
sweep_dir, min_distinct, expect_queued = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1"
iv = {}
for lg in glob.glob(os.path.join(sweep_dir, "*.log")):
    job = os.path.basename(lg)[:-4]
    s = e = g = None
    for line in open(lg):
        m = re.search(r"START gpu=(\S+) ts=([\d.]+)", line)
        if m: g, s = m.group(1), float(m.group(2))
        m = re.search(r"END gpu=(\S+) ts=([\d.]+)", line)
        if m: e = float(m.group(2))
    if s is not None and e is not None:
        iv[job] = (g, s, e)
print(f"  parsed {len(iv)} run interval(s):")
for j in sorted(iv, key=lambda k: iv[k][1]):
    g, s, e = iv[j]
    print(f"    {j:8s} gpu={g} span={e-s:.2f}s [{s:.3f} -> {e:.3f}]")
ok = True
# no double-book: no two overlapping intervals share a GPU
items = list(iv.items())
for i in range(len(items)):
    for k in range(i+1, len(items)):
        (ja, (ga, sa, ea)), (jb, (gb, sb, eb)) = items[i], items[k]
        if ga == gb and sa < eb and sb < ea:
            print(f"  FAIL double-book: {ja} & {jb} both on gpu {ga} overlap"); ok = False
distinct = len({v[0] for v in iv.values()})
print(f"  distinct GPUs used: {distinct} (need >= {min_distinct})")
if distinct < min_distinct:
    print("  FAIL: jobs did not distribute across the pool"); ok = False
# concurrency: any two intervals overlapping in time = ran in parallel
concurrent = any(items[i][1][1] < items[k][1][2] and items[k][1][1] < items[i][1][2]
                 for i in range(len(items)) for k in range(i+1, len(items)))
print(f"  observed parallel overlap: {concurrent}")
if min_distinct >= 2 and not concurrent:
    print("  FAIL: no parallelism observed (jobs serialized onto fewer GPUs)"); ok = False
if expect_queued:
    # at least one job started only after another finished => the pool queued it
    queued = any(iv[a][1] >= iv[b][2] - 1e-6 for a in iv for b in iv if a != b)
    print(f"  observed queued-after-free: {queued}")
    if not queued:
        print("  FAIL: expected a job to queue until a GPU freed"); ok = False
sys.exit(0 if ok else 1)
PY

PASS=0; FAIL=0
ok()   { echo "  PASS: $1"; PASS=$((PASS+1)); }
bad()  { echo "  FAIL: $1"; FAIL=$((FAIL+1)); }

# =====================================================================================
echo "================================================================================"
echo "SCENARIO A (findings #1/#2): 3 GPU jobs, 2-GPU pool {1,2} -> distribute + queue 3rd"
echo "================================================================================"
reset_state
SW="_rqtest_${$}_a"; SWEEPS+=("$SW")
MAN="$T/a.txt"
: > "$MAN"
for j in jobA jobB jobC; do printf '%s gpu 1G %s %s 3 %s\n' "$j" "$T/stub_job.sh" "$j" "$STATE" >> "$MAN"; done
"$RQ" "$MAN" --gpus 1,2 --poll 1 --idle-mib 1500 --sweep "$SW" >"$T/a.out" 2>&1; RC=$?
echo "--- run_queue exit=$RC (tail) ---"; tail -n 4 "$T/a.out"
[[ $RC -eq 0 ]] && ok "A run_queue exit 0" || bad "A run_queue exit $RC"
if python3 "$T/parse.py" "$REPO_ROOT/runs/$SW" 2 1; then ok "A intervals: distributed across 2 GPUs, 3rd queued, no double-book"; else bad "A interval checks"; fi

# =====================================================================================
echo "================================================================================"
echo "SCENARIO B (finding #2): full pool {1,2,3} + 4th job must WAIT, not double-book"
echo "================================================================================"
reset_state
SW="_rqtest_${$}_b"; SWEEPS+=("$SW")
MAN="$T/b.txt"
: > "$MAN"
for j in j1 j2 j3 j4; do printf '%s gpu 1G %s %s 3 %s\n' "$j" "$T/stub_job.sh" "$j" "$STATE" >> "$MAN"; done
"$RQ" "$MAN" --gpus 1,2,3 --poll 1 --idle-mib 1500 --sweep "$SW" >"$T/b.out" 2>&1; RC=$?
echo "--- run_queue exit=$RC (tail) ---"; tail -n 4 "$T/b.out"
[[ $RC -eq 0 ]] && ok "B run_queue exit 0" || bad "B run_queue exit $RC"
if python3 "$T/parse.py" "$REPO_ROOT/runs/$SW" 3 1; then ok "B intervals: 3 GPUs concurrent, 4th queued, no double-book"; else bad "B interval checks"; fi

# =====================================================================================
echo "================================================================================"
echo "SCENARIO C (finding #5): never-idle pool -> acquire-timeout SKIPS + non-zero exit"
echo "================================================================================"
reset_state
SW="_rqtest_${$}_c"; SWEEPS+=("$SW")
echo 9000 > "$STATE/gpu1.mem"   # GPU 1 permanently busy for the mock nvidia-smi
MAN="$T/c.txt"
: > "$MAN"
for j in c1 c2; do printf '%s gpu 1G %s %s 3 %s\n' "$j" "$T/stub_job.sh" "$j" "$STATE" >> "$MAN"; done
"$RQ" "$MAN" --gpus 1 --poll 1 --acquire-timeout 3 --idle-mib 1500 --sweep "$SW" >"$T/c.out" 2>&1; RC=$?
rm -f "$STATE/gpu1.mem"
echo "--- run_queue exit=$RC (tail) ---"; tail -n 5 "$T/c.out"
[[ $RC -ne 0 ]] && ok "C exited non-zero on timeout" || bad "C should exit non-zero"
grep -q "ACQUIRE TIMEOUT" "$T/c.out" && ok "C emitted acquire-timeout" || bad "C no timeout message"
grep -q "skipped" "$REPO_ROOT/runs/$SW/queue_summary.md" && ok "C skipped rows in summary" || bad "C no skipped rows"
[[ ! -f "$REPO_ROOT/runs/$SW/c1.log" ]] || ! grep -q START "$REPO_ROOT/runs/$SW/c1.log" 2>/dev/null && ok "C no job actually launched" || bad "C a job launched despite busy pool"

# =====================================================================================
echo "================================================================================"
echo "SCENARIO D (finding #4): malformed manifest lines -> failed rows + non-zero exit"
echo "================================================================================"
reset_state
SW="_rqtest_${$}_d"; SWEEPS+=("$SW")
MAN="$T/d.txt"
{
  printf '# comment line ignored\n'
  printf 'okjob cpu 1G %s okjob 1 %s\n' "$T/stub_job.sh" "$STATE"   # valid cpu job
  printf 'brokejob gpu\n'                                          # missing fields
  printf 'badev banana 1G echo hi\n'                               # bad device
} > "$MAN"
"$RQ" "$MAN" --gpus 1,2 --poll 1 --sweep "$SW" >"$T/d.out" 2>&1; RC=$?
echo "--- run_queue exit=$RC (tail) ---"; tail -n 6 "$T/d.out"
[[ $RC -ne 0 ]] && ok "D exited non-zero on malformed lines" || bad "D should exit non-zero"
grep -q "malformed" "$REPO_ROOT/runs/$SW/queue_summary.md" && ok "D malformed rows in summary" || bad "D no malformed rows"
grep -q "brokejob" "$REPO_ROOT/runs/$SW/queue_summary.md" && grep -q "badev" "$REPO_ROOT/runs/$SW/queue_summary.md" \
  && ok "D both bad lines reported" || bad "D missing a bad line"

# =====================================================================================
echo "================================================================================"
echo "SCENARIO E (finding #3): SIGTERM mid-run -> trap fires, no orphan scope/child, exit 130"
echo "================================================================================"
reset_state
if ! systemctl --user show-environment >/dev/null 2>&1; then
  echo "  SKIP: systemd --user unavailable — cannot exercise real scope teardown (not a pass, not a fail)"
else
  SW="_rqtest_${$}_e"; SWEEPS+=("$SW")
  EINT="eint_${$}"; SCOPE_GLOBS+=("${EINT}_*.scope")
  MAN="$T/e.txt"
  printf '%s gpu 1G %s %s 60 %s\n' "$EINT" "$T/stub_job.sh" "$EINT" "$STATE" > "$MAN"
  "$RQ" "$MAN" --gpus 1,2 --poll 1 --idle-mib 1500 --sweep "$SW" >"$T/e.out" 2>&1 &
  RQ_PID=$!
  # wait until the job has actually started (scope live + START logged), up to ~20s
  started=0
  for _ in $(seq 1 40); do
    [[ -f "$REPO_ROOT/runs/$SW/$EINT.log" ]] && grep -q START "$REPO_ROOT/runs/$SW/$EINT.log" 2>/dev/null && { started=1; break; }
    kill -0 "$RQ_PID" 2>/dev/null || break
    sleep 0.5
  done
  if (( started == 0 )); then
    bad "E job never started (cannot exercise interrupt)"
    kill -TERM "$RQ_PID" 2>/dev/null || true; wait "$RQ_PID" 2>/dev/null || true
  else
    scope_before="$(systemctl --user list-units --all --no-legend "${EINT}_*.scope" 2>/dev/null | awk '{print $1}' | head -1)"
    before_n="$(pgrep -cf "$T/stub_job.sh $EINT" 2>/dev/null || true)"
    kill -TERM "$RQ_PID" 2>/dev/null           # interrupt the queue mid-run
    wait "$RQ_PID" 2>/dev/null; RC=$?
    echo "--- run_queue exit=$RC (tail) ---"; tail -n 6 "$T/e.out"
    # (d) SIGTERM path exits 130
    [[ $RC -eq 130 ]] && ok "E exit code 130 (clean SIGTERM teardown)" || bad "E expected exit 130, got $RC"
    # (a) trap fired
    grep -q "INTERRUPTED" "$T/e.out" && ok "E cleanup trap fired" || bad "E trap did not fire"
    # (c) partial summary with an interrupted row
    grep -q "interrupted" "$REPO_ROOT/runs/$SW/queue_summary.md" 2>/dev/null \
      && ok "E interrupted row in partial summary" || bad "E no interrupted row in summary"
    # (b) no orphaned scope unit
    sleep 1
    if [[ -n "$scope_before" ]] && systemctl --user is-active "$scope_before" >/dev/null 2>&1; then
      bad "E scope $scope_before still active (orphaned)"
    else
      ok "E scope unit reaped (was: ${scope_before:-<none captured>})"
    fi
    # (b) no orphaned child process (whole cgroup dies when the scope is stopped)
    after_n="$(pgrep -cf "$T/stub_job.sh $EINT" 2>/dev/null || true)"
    echo "  stub_job procs matching '$EINT': before=${before_n:-0} after=${after_n:-0}"
    if [[ "${after_n:-0}" -eq 0 ]]; then
      ok "E no orphaned child process"
    else
      bad "E ${after_n} child process(es) orphaned"; pkill -9 -f "$T/stub_job.sh $EINT" 2>/dev/null || true
    fi
  fi
fi

echo "================================================================================"
echo "RESULT: $PASS passed, $FAIL failed"
echo "================================================================================"
[[ $FAIL -eq 0 ]]
