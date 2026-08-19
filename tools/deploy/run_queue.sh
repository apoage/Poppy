#!/usr/bin/env bash
# run_queue.sh — self-advancing multi-GPU job queue for gpu-server.
#
# Reads a whitespace job manifest and runs each job under tools/safe_run.sh (systemd
# --user cgroup RAM cap + GPU pinning). GPU jobs are scheduled across a pool of GPUs
# (default 1,2,3 — GPU 0 is VirtualBox-reserved on gpu-server), at most ONE job per GPU,
# and a candidate GPU is assigned only after a LIVE idle check (never blind --gpu 1,2,3).
# CPU jobs run without consuming a GPU slot. Per-job logs land in runs/<sweep>/<job>.log
# and a summary table is written at the end. Meant to run INSIDE a tmux session so it
# survives disconnect (one self-advancing orchestrator tmux per queue).
#
# Manifest format (whitespace-separated; blank lines and lines starting with # ignored):
#   <job_id>  <device: gpu|cpu>  <mem, e.g. 32G>  <command ...to end of line...>
# job_id/device/mem must contain no spaces; everything after the 3rd field is the command.
# The command is run via `bash -c` from the repo root; for gpu jobs the queue injects
# JAX_PLATFORMS=cuda (safe_run --gpu sets CUDA_VISIBLE_DEVICES + XLA MEM_FRACTION but the
# runners default JAX_PLATFORMS=cpu via setdefault, so cuda must be forced explicitly).
#
# --- Fail-closed scheduling contract (drives UNATTENDED training; a bug = 3am OOM) ---
# GPU RESERVATION (no double-book, no TOCTOU): a GPU is removed from the free pool at the
# moment it is ACQUIRED (before launch) and only returned when its job is reaped, so the
# live-idle check and the launch cannot race two jobs onto one GPU. Cross-instance safety
# comes from a per-GPU advisory lock file on a HOST-GLOBAL path (flock on
# runs/.locks/gpu<N>.lock — shared by EVERY run_queue on the host regardless of --sweep, so
# two queues with different sweeps and an overlapping pool cannot both take one GPU) held for
# the job's lifetime — a second run_queue (or foreign holder) cannot grab a GPU we own, and
# we skip any pool GPU whose lock another holder is holding. After launch we additionally poll
# until that GPU's VRAM actually rises (warmup confirmation) so the *next* idle reading is
# trustworthy and the multi-second JAX/CUDA warmup blind-spot cannot fool the scheduler.
# NOTE: XLA_PYTHON_CLIENT_MEM_FRACTION is INERT unless PREALLOCATE=true, so on the GPU
# launch path we also force XLA_PYTHON_CLIENT_PREALLOCATE=true — the frac cap is the VRAM
# backstop, not the idle check. Malformed manifest lines and unmet GPU-acquire OR CPU-slot
# waits (both bounded by --acquire-timeout, with periodic heartbeats) are reported as explicit
# FAILED/SKIPPED rows and force a non-zero exit (fail loud, never silent). A GPU acquire-
# timeout abandons only the remaining GPU jobs; independent CPU jobs still run.
#
# Usage:
#   tools/deploy/run_queue.sh <manifest> [options]
#     --gpus 1,2,3            GPU pool (default 1,2,3)
#     --gpu-frac 0.7          XLA_PYTHON_CLIENT_MEM_FRACTION per GPU job (default 0.7)
#     --idle-mib 1500         a pool GPU counts as idle if memory.used < this (default 1500)
#     --poll 10               seconds between scheduler polls (default 10)
#     --acquire-timeout 1800  give up waiting for a free GPU after this many sec (0=never)
#     --sweep <id>            runs/<id> for logs+summary (default: manifest basename sans ext)
#     --max-cpu N             max concurrent CPU jobs (default 2, must be >= 1)
#     --stall-timeout 0       checkpoint-save WEDGE watchdog (OPT-IN; default 0 = OFF). When armed
#                             with a positive <sec>, a TRAINING job whose training CSV goes
#                             unmodified for this many sec WHILE THE PROCESS IS ALIVE is declared
#                             WEDGED -> killed, its partial ckpt/logs archived with a _wedged_ep<N>
#                             suffix, and the SAME manifest line relaunched ONCE (counted
#                             WEDGED-RETRIED). A 2nd wedge on that job = FAILED row (fail loud).
#                             ONLY jobs that declare (csvglob:) or actually PRODUCE a training CSV
#                             are watched; CSV-less eval/benchmark/data-prep jobs are never killed.
#                             Default OFF -> behaviour + summary byte-identical to pre-watchdog.
#     --stall-grace 600       pre-first-CSV-write grace: before any CSV exists a job is judged
#                             wedged only after start + (stall-timeout + this) sec (JAX warmup).
#     --debug-stall-dump      on a wedge, best-effort thread dump before the kill (py-spy on the
#                             scope MainPID + SIGABRT so PYTHONFAULTHANDLER prints a Python
#                             traceback into the job log). Off by default; adds no hard dep.
#     --dry-run               print the schedule decisions without launching (stubs safe_run)
#
# CSV to watch (per job): an optional leading `csvglob:<glob>` token in the command names it
# explicitly (stripped before the command runs); else it is inferred from `--name <X>` in the
# command (logs/training_<X>.csv); else logs/training_<job_id>.csv.
#
# --- Interpreter contract (TASK_891 — D46, RATIFIED: ABSOLUTE-PATH SUBSTITUTION and nothing
#     else. Supersedes D42(a)'s python-only shim-dir + PATH-prepend design, which run 35 killed
#     TWICE: a job planting $SHIM_DIR/bash hijacked the job shell because `env … bash -c` and the
#     wrapper's own `#!/usr/bin/env bash` shebang both resolved through the prepended, job-writable
#     PATH — the prepend was itself the attack surface. Deleting the shim dir deletes that whole
#     class: no directory WE create ever enters the trust path, so there is nothing of ours to
#     hijack, forge, or cmp against, and no lifecycle (build/self-probe/cleanup) to get an
#     interrupt window wrong.
#
#     THREAT MODEL — read this before reading any guarantee into the labels below. THIS IS NOT A
#     SECURITY BOUNDARY, and it never was one. Every queued job runs under tools/safe_run.sh,
#     which is `exec systemd-run --user --scope` (safe_run.sh:50): the SAME uid, the same login
#     session, with only a cgroup RAM cap — there is no privilege boundary between this queue and
#     the jobs it launches. `trader_env/bin/python` is an ordinary same-user-writable path (a
#     lukas:lukas symlink inside lukas-owned drwxrwxr-x directories), and the `protected` list in
#     .dark-factory/config.json is an AGENT-WRITE POLICY convention with no enforcement code
#     anywhere in the repo — not an OS ACL. So "trader_env/ is PROTECTED" (D46's wording) means
#     "agents are told not to write it", NOT "jobs cannot write it". A job that actually wanted to
#     subvert an interpreter would not need the venv at all: it could rewrite this script,
#     tools/safe_run.sh, or the training code it is about to run. What the checks below buy is
#     therefore protection against ACCIDENT AND MISTAKE — a half-rebuilt venv, a `pip install`
#     that clobbered the interpreter, an earlier manifest row that removed what a later row needs
#     — and NOT defence against a hostile same-user process. Nothing here should be read as such
#     a defence. ---
# Three-tier honest contract (kept verbatim in shape from the halted D42(a) work; the CONTENT is
# re-derived for THIS mechanism, not copied from the shim's):
#   (1) CLAIMED. A LEADING bare `python`/`python3` token — the first whitespace-separated word of
#       the command — is rewritten, in the command STRING itself, to the absolute path
#       `<repo>/trader_env/bin/python`. The rewritten string IS what bash execs, so the row cannot
#       be silently redirected by a PATH that the job (or anything above it) set, and a broken
#       interpreter fails loudly (bash rc 126/127) instead of quietly resolving somewhere else.
#       What is CLAIMED is the path that will be exec'd — not the identity of the file at the end
#       of it. That file is re-checked (existence + identity) immediately before THIS row's own
#       launch, which is as close to the `execve` as a shell can get; the residual window is
#       documented under LIMITS below and is not claimed away.
#   (2) BEST-EFFORT, NOT CLAIMED — EMPTY under this mechanism. D42(a)'s middle tier was a PATH
#       prepend covering a row's SECOND bare python and any bare-python children; that prepend is
#       exactly what reopened the attack surface, so D46 deletes it with no replacement. There is
#       nothing left in this tier: a chained/child python either gets the absolute-path rewrite
#       (only ever the row's leading word) or it does not, with no half-measure in between.
#   (3) NOT COVERED AT ALL — no claim, no help, byte-untouched, labelled `passthrough:<leading
#       token>`: `env python ...`, `bash -c "python ..."`, `PATH=/x python ...` (the run-34 halt
#       case — a row that assigns its own PATH shadows nothing of ours, because nothing of ours is
#       ever injected), `sudo python`, quoted/escaped/`$VAR`-suffixed spellings, versioned
#       spellings (`python3.11`), absolute interpreter paths, a `#!/usr/bin/env python` shebang
#       script the row calls, metacharacter-glued leading spellings (`python;true` is one token,
#       not bare `python`, so it is passthrough too) — AND, now that tier (2) is gone, every
#       SECOND-OR-LATER bare `python`/`python3` in a chained row (`python a.py && python b.py`'s
#       second half) and any bare-python child process a job spawns. This is a priced, documented
#       cost (D46), and the one-time TASK_891 sweep MEASURED how much of the tree actually leans on
#       it. Result, scoped to what version control can show: all 1438 GIT-TRACKED files were scanned
#       BY CONTENT SHAPE (not by filename), 18 contain lines matching the run_queue row shape (15 are
#       real manifests; 3 are .py sources whose lines coincidentally match), and ZERO contain a
#       NOT-COVERED spelling — so there was nothing tracked to re-spell or comment and the sweep is
#       a no-op diff. KNOWN EXCEPTION, named because "tracked" is what excludes it: the only manifest
#       anywhere in the checkout carrying this shape is `runs/860_codistill/manifest_860.txt:17-19`
#       (`python distill.py && test -e … && python eval_vector.py`), and it is UNTRACKED —
#       `.gitignore:54` (`runs/*/*`) excludes it and it has zero git history. So it cannot be cited
#       as evidence the sweep landed, and a comment written into it would not survive a clone; its
#       non-durability is recorded on DECISIONS D48 and is NOT claimed fixed here.
#       NOT SILENT, though. An accepted limit still has to be VISIBLE, or an operator cannot tell
#       `python train.py && python eval.py` (whose eval half runs the ambient python2 on gpu-server)
#       from a harmless passthrough row — both used to print the same label. So a row carrying a
#       chained bare python in COMMAND POSITION is (a) WARNED at parse time by job name + manifest
#       line, (b) marked `not-covered-chained-python=<n>` on its dry-run/launch line, and (c) listed
#       in the summary. It is still SCHEDULED AS WRITTEN — D46 priced this limit as accepted, so the
#       queue does not start refusing rows the ratified contract permits. The detector is a narrow
#       QUOTE-AWARE token heuristic, not a shell parser (`_uncovered_chained_pythons` states the
#       exact rules and the mutation-proved table behind them): quoted spans are never scanned, so
#       an argument that merely reads `python` (`--tag python`, `--note "a && python b"`,
#       `--desc env python`) is never flagged; and it deliberately does not see a python inside
#       quotes, inside a substitution, behind a launcher it does not model (`xargs`/`time`/`sudo`),
#       or one a job spawns at runtime. So `n=0` means "none visible", never "none present" — the
#       record it feeds is a FLOOR on what is uncovered, not a census.
# Fail loud on a broken venv TWICE, at both ends of the deferral: once at PARSE TIME (a doomed
# manifest is refused before anything at all is scheduled, by name and manifest line) and again
# IMMEDIATELY BEFORE EACH claimed row's own launch (a row that sat for hours behind a GPU wait is
# not launched against an interpreter that broke in the meantime). Both ends run the SAME check:
# `trader_env/bin/python` must exist and be executable, and must answer a PROBE-TIME NONCE with its
# own major version, `sys.executable` and `sys.prefix` — an rc-only check would bless `/bin/true`
# (ignores its arguments, exits 0), and a check that only looked for the digit `3` would be
# satisfied by any fixed-output stub. The probe is bounded by `timeout` so a wedged interpreter can
# hang neither the parse nor a launch. A row that fails either end is REFUSED BY NAME (with its
# manifest line at parse time), recorded `venv-refused` in the summary, and never launched;
# passthrough rows are entirely unaffected at both ends.
#
# LIMITS of that check — stated here so nobody trusts it further than it goes:
#   - It is NOT authentication, and cannot become authentication. Measured against four forgeries
#     during the TASK_891 fix pass, not asserted: `/bin/true` FAILS (no output), a fixed-output stub
#     FAILS (the nonce is only known at probe time), a plain shell delegator that `exec`s the real
#     interpreter FAILS (`sys.executable` then reports the real path, not the path we invoked), and
#     some other python planted at that path FAILS (`sys.prefix` must be the venv root) — BUT a
#     two-line delegator that spoofs argv[0] (`exec -a <the path we invoke> <real python> "$@"`)
#     next to a three-line planted `pyvenv.cfg` PASSES ALL FOUR CHECKS. So the probe raises the bar
#     from "any broken thing" to "a deliberate forgery", and that is exactly where it stops: it does
#     not prove the file is the interpreter we believe it is, and per the THREAT MODEL above,
#     proving that is not its job.
#   - It does NOT close the TOCTOU window, because a shell cannot: there is always a gap between
#     the last check and the `execve`, and bash resolves the substituted absolute path at exec
#     time, i.e. after our final look. The launch-time re-probe SHRINKS that window from hours (a
#     row parsed, then queued behind a GPU wait) to milliseconds. Sized honestly: it reliably
#     catches the venv breaking MID-QUEUE, which is a realistic accident, and it catches nothing
#     that races the exec — which, per the THREAT MODEL above, it is not trying to.
# The `--dry-run` printout AND the live launch line both show the REWRITTEN command for a claimed
# row (the operator sees the absolute path, not a description of one), and the summary carries a
# durable `interpreter` block recording every claimed row's rewritten command, so a post-mortem
# does not depend on transient /proc state. Everything else prints
# `interpreter=passthrough:<leading token>` and is otherwise untouched.
set -uo pipefail    # NOTE: no -e; a single failed job must not abort the queue.

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

MANIFEST=""
GPUS_CSV="1,2,3"
FRAC="0.7"
IDLE_MIB="1500"
POLL="10"
ACQUIRE_TIMEOUT="1800"   # 30 min; 0 = wait forever (old behaviour, discouraged)
SWEEP=""
MAX_CPU="2"
DRY=0
STALL_TIMEOUT="0"        # checkpoint-save wedge watchdog (sec); 0 = OFF. OPT-IN: arms only when
                         # the operator passes --stall-timeout <sec>. Default OFF satisfies the
                         # project invariant "every new behaviour flag-gated, default OFF" and
                         # keeps CSV-less eval/benchmark jobs safe (authorized deviation from the
                         # TASK_841 spec's "default 2700" — recorded in DECISIONS by the orchestrator).
STALL_GRACE="600"        # extra grace before the FIRST CSV write (JAX/CUDA warmup)
DEBUG_DUMP=0             # --debug-stall-dump: thread-dump a wedged job before the kill

HEARTBEAT_EVERY=60       # emit a "still waiting" status line at most this often (sec)
WARMUP_TIMEOUT=30        # after launch, wait up to this long for VRAM to rise (sec)
MAX_WEDGE_RETRY=1        # auto-retries per job after a checkpoint-save wedge (spec: exactly 1)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)            GPUS_CSV="$2"; shift 2;;
    --gpu-frac)        FRAC="$2"; shift 2;;
    --idle-mib)        IDLE_MIB="$2"; shift 2;;
    --poll)            POLL="$2"; shift 2;;
    --acquire-timeout) ACQUIRE_TIMEOUT="$2"; shift 2;;
    --sweep)           SWEEP="$2"; shift 2;;
    --max-cpu)         MAX_CPU="$2"; shift 2;;
    --stall-timeout)   STALL_TIMEOUT="$2"; shift 2;;
    --stall-grace)     STALL_GRACE="$2"; shift 2;;
    --debug-stall-dump) DEBUG_DUMP=1; shift;;
    --dry-run)         DRY=1; shift;;
    -*)                echo "run_queue: unknown option $1" >&2; exit 2;;
    *)                 [[ -z "$MANIFEST" ]] && MANIFEST="$1" || { echo "run_queue: extra arg $1" >&2; exit 2; }; shift;;
  esac
done

[[ -z "$MANIFEST" ]] && { echo "usage: $0 <manifest> [--gpus 1,2,3] [--gpu-frac 0.7] [--idle-mib 1500] [--poll 10] [--acquire-timeout 1800] [--sweep id] [--max-cpu 2] [--stall-timeout 2700] [--stall-grace 600] [--debug-stall-dump] [--dry-run]" >&2; exit 2; }
[[ -f "$MANIFEST" ]] || { echo "run_queue: manifest not found: $MANIFEST" >&2; exit 1; }
[[ -x "$REPO_ROOT/tools/safe_run.sh" ]] || { echo "run_queue: tools/safe_run.sh missing/not executable" >&2; exit 1; }
[[ "$MAX_CPU" =~ ^[0-9]+$ && "$MAX_CPU" -ge 1 ]] || { echo "run_queue: --max-cpu must be a positive integer (got '$MAX_CPU')" >&2; exit 2; }
[[ "$ACQUIRE_TIMEOUT" =~ ^[0-9]+$ ]] || { echo "run_queue: --acquire-timeout must be a non-negative integer (got '$ACQUIRE_TIMEOUT')" >&2; exit 2; }
[[ "$POLL" =~ ^[0-9]+$ && "$POLL" -ge 1 ]] || { echo "run_queue: --poll must be a positive integer (got '$POLL')" >&2; exit 2; }
[[ "$STALL_TIMEOUT" =~ ^[0-9]+$ ]] || { echo "run_queue: --stall-timeout must be a non-negative integer (got '$STALL_TIMEOUT')" >&2; exit 2; }
[[ "$STALL_GRACE" =~ ^[0-9]+$ ]] || { echo "run_queue: --stall-grace must be a non-negative integer (got '$STALL_GRACE')" >&2; exit 2; }
if [[ -z "$SWEEP" ]]; then
  SWEEP="$(basename "$MANIFEST")"
  SWEEP="${SWEEP%.*}"
fi

RUN_DIR="$REPO_ROOT/runs/$SWEEP"
mkdir -p "$RUN_DIR"
SUMMARY="$RUN_DIR/queue_summary.md"

# Per-GPU advisory locks live on a HOST-GLOBAL path (NOT under $RUN_DIR) so that two
# run_queue instances with DIFFERENT --sweep but an overlapping GPU pool contend on the
# SAME lock file and cannot both grab one GPU (finding N1). All queues on the host share
# this repo checkout, so runs/.locks is host-global in practice. RUN_QUEUE_LOCK_DIR lets
# hermetic tests redirect the locks off the shared production path.
LOCK_DIR="${RUN_QUEUE_LOCK_DIR:-$REPO_ROOT/runs/.locks}"
mkdir -p "$LOCK_DIR"

IFS=',' read -r -a GPU_POOL <<< "$GPUS_CSV"
FREE_GPUS=("${GPU_POOL[@]}")

# nvidia-smi shim for --dry-run (no GPUs / deterministic idle)
_gpu_mem_used() {  # $1 = gpu index -> echoes MiB used (integer), or "" on failure
  if [[ "$DRY" -eq 1 ]]; then echo 0; return 0; fi
  nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]'
}

# NB: init with =() so empty ${#arr[@]}/${!arr[@]} expansions are set -u safe (bash treats a
# declare -A that never got an element as UNSET otherwise, aborting before the first launch).
declare -A PID2JOB=() PID2GPU=() PID2LOG=() GPU_LOCK_FD=()
# Watchdog per-pid state (TASK_841). PID2IDX ties a live pid back to its manifest row so a
# wedged job can be relaunched from the same spec; the mtime/change/start clocks drive the
# stall test; RETRY caps auto-retries at MAX_WEDGE_RETRY per job.
declare -A PID2IDX=() PID2CSVGLOB=() PID2STARTTS=() PID2LASTMTIME=() PID2LASTCHANGE=() RETRY=()
declare -A PID2EXPLICIT=() PID2BASELINE=()  # explicit=csvglob-declared; baseline=pre-launch stale mtime
WEDGE_RETRIED_COUNT=0  # jobs auto-retried after a wedge (surfaced in the summary only when >0)
LAST_LAUNCH_PID=""     # set by launch() so callers can register watchdog state for the new pid
CLAIMED_CMDS=()        # "job<TAB>rewritten command" per interpreter=venv launch (D46 clause 4:
                       # durable record of the absolute path a claimed row actually ran). Empty
                       # unless a claimed row launched -> summary stays byte-identical otherwise.
NOTCOVERED_ROWS=()     # "job<TAB>count<TAB>command" per launched row carrying a NOT-COVERED chained
                       # bare python (same emptiness property).
RESULTS=()            # "job<TAB>rc<TAB>log"  (rc: integer | "malformed" | "skipped" | "interrupted"
                      # | "wedged-failed" | "venv-refused")
ACQUIRED_GPU=""       # set by acquire_idle_gpu (finding #1: mutate PARENT state, no subshell)
ACQUIRE_SAW_VALID=0   # 1 if the last acquire pass got a valid nvidia-smi reading from any pool GPU
START_TS="$(date -Iseconds)"
END_TS=""
SUMMARY_WRITTEN=0

# ---- cleanup trap (finding #3): reap/stop scope units so children never orphan ----
_release_all_locks() {
  local gg fd
  ((${#GPU_LOCK_FD[@]})) || return 0
  for gg in "${!GPU_LOCK_FD[@]}"; do
    fd="${GPU_LOCK_FD[$gg]}"
    flock -u "$fd" 2>/dev/null || true
    eval "exec ${fd}>&-" 2>/dev/null || true
    unset 'GPU_LOCK_FD[$gg]'
  done
}

cleanup() {
  local ec=$?
  trap - INT TERM EXIT           # disarm first so this is idempotent / non-reentrant
  if ((${#PID2JOB[@]})); then
    echo "" >&2
    echo "[queue] INTERRUPTED — stopping ${#PID2JOB[@]} running job(s) + their scope units ..." >&2
    local pid job
    for pid in "${!PID2JOB[@]}"; do
      job="${PID2JOB[$pid]}"
      RESULTS+=("$job"$'\t'"interrupted"$'\t'"${PID2LOG[$pid]:--}")
      kill "$pid" 2>/dev/null || true
      # safe_run names the scope <NAME>_<pid>; stop by glob so we get it regardless of pid.
      systemctl --user stop "${job}_*.scope" 2>/dev/null || true
    done
    _release_all_locks
    write_summary
    echo "[queue] cleanup done; partial summary -> $SUMMARY" >&2
    (( ec == 0 )) && ec=130
  fi
  exit "$ec"
}

_cpu_running() {  # count live CPU jobs (PID2GPU unset for them)
  local pid n=0
  ((${#PID2JOB[@]})) || { echo 0; return 0; }
  for pid in "${!PID2JOB[@]}"; do [[ -z "${PID2GPU[$pid]:-}" ]] && n=$((n+1)); done
  echo "$n"
}

acquire_idle_gpu() {  # sets ACQUIRED_GPU + reserves the GPU (parent state); return 0 on success
  local i g used lock fd
  ACQUIRE_SAW_VALID=0
  ((${#FREE_GPUS[@]})) || return 1
  for i in "${!FREE_GPUS[@]}"; do
    g="${FREE_GPUS[$i]}"
    used="$(_gpu_mem_used "$g")"
    if [[ -n "$used" && "$used" =~ ^[0-9]+$ ]]; then
      ACQUIRE_SAW_VALID=1
      if [[ "$used" -lt "$IDLE_MIB" ]]; then
        # Reserve cross-instance: hold an advisory lock for the job's lifetime. If another
        # holder has it, this GPU is busy elsewhere — leave it in the pool and try the next.
        lock="$LOCK_DIR/gpu${g}.lock"
        if ! eval "exec {fd}>\"\$lock\""; then continue; fi
        if flock -n "$fd"; then
          GPU_LOCK_FD["$g"]="$fd"
          unset 'FREE_GPUS[i]'; FREE_GPUS=("${FREE_GPUS[@]}")   # compact — reserve in parent
          ACQUIRED_GPU="$g"
          return 0
        else
          eval "exec ${fd}>&-"   # release fd; GPU locked elsewhere, keep it in the pool
        fi
      fi
    fi
  done
  return 1
}

reap() {  # reap finished children, free+unlock their GPUs, record results
  local pid rc job g fd
  ((${#PID2JOB[@]})) || return 0
  for pid in "${!PID2JOB[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid"; rc=$?
      job="${PID2JOB[$pid]}"; g="${PID2GPU[$pid]:-}"
      RESULTS+=("$job"$'\t'"$rc"$'\t'"${PID2LOG[$pid]}")
      if [[ -n "$g" ]]; then
        FREE_GPUS+=("$g")
        fd="${GPU_LOCK_FD[$g]:-}"
        if [[ -n "$fd" ]]; then flock -u "$fd" 2>/dev/null || true; eval "exec ${fd}>&-" 2>/dev/null || true; unset 'GPU_LOCK_FD[$g]'; fi
        echo "[queue] reaped $job rc=$rc (freed GPU $g)"
      else
        echo "[queue] reaped $job rc=$rc (cpu)"
      fi
      unset 'PID2JOB[$pid]' 'PID2GPU[$pid]' 'PID2LOG[$pid]'
    fi
  done
}

confirm_vram_or_warn() {  # $1 gpu $2 job — best-effort: wait for VRAM to rise so the next idle read is trustworthy
  [[ "$DRY" -eq 1 ]] && return 0
  local g="$1" job="$2" t0=$SECONDS used pid still
  while (( SECONDS - t0 < WARMUP_TIMEOUT )); do
    used="$(_gpu_mem_used "$g")"
    if [[ -n "$used" && "$used" =~ ^[0-9]+$ && "$used" -ge "$IDLE_MIB" ]]; then
      echo "[queue] $job: GPU $g VRAM confirmed (${used}MiB) after $((SECONDS-t0))s"
      return 0
    fi
    reap   # if the job died during warmup, its GPU is freed — stop waiting
    still=0
    if ((${#PID2GPU[@]})); then for pid in "${!PID2GPU[@]}"; do [[ "${PID2GPU[$pid]}" == "$g" ]] && still=1; done; fi
    (( still == 0 )) && { echo "[queue] $job: GPU $g job ended during warmup (no VRAM confirm)"; return 0; }
    sleep 1
  done
  echo "[queue] WARN $job: GPU $g VRAM did not rise above ${IDLE_MIB}MiB within ${WARMUP_TIMEOUT}s (job may have failed fast; check $RUN_DIR/$job.log)" >&2
  return 0
}

_free_gpu() {  # $1 gpu -> return it to the pool and drop its advisory lock (mirrors reap())
  local g="$1" fd="${GPU_LOCK_FD[$g]:-}"
  FREE_GPUS+=("$g")
  if [[ -n "$fd" ]]; then
    flock -u "$fd" 2>/dev/null || true
    eval "exec ${fd}>&-" 2>/dev/null || true
    unset 'GPU_LOCK_FD[$g]'
  fi
}

launch() {  # $1 job  $2 dev  $3 mem  $4 gpu(""=cpu)  $5 cmd  $6 interp label (venv|passthrough:<tok>)
            # $7 count of NOT-COVERED chained bare pythons in the command (0 = none visible)
            # -> 0 = launched (LAST_LAUNCH_PID set), 1 = REFUSED, nothing launched.
  local job="$1" dev="$2" mem="$3" g="$4" cmd="$5" interp="${6:-}" nc="${7:-0}"
  local log="$RUN_DIR/$job.log"
  local sr=("$REPO_ROOT/tools/safe_run.sh" --name "$job" --mem "$mem")
  # Interpreter contract (TASK_891 — D46): a `venv` row's command is rewritten to the absolute
  # venv path IN THE STRING ITSELF, here, right before it is handed to safe_run — no env var, no
  # PATH entry, no wrapper. A `passthrough` row's cmd is untouched. Nothing else about the launch
  # path differs between the two, which is the whole content of gate (iv): a claimed row's
  # environment is byte-identical to a passthrough row's.
  if [[ "$interp" == "venv" ]]; then
    # RE-ASSERT the interpreter HERE, immediately before this row actually goes. The parse-time
    # check can be hours stale for a row that waited out a GPU queue, and trader_env/bin/python is
    # an ordinary same-user-writable path (see THREAT MODEL in the header) — so a mid-queue
    # ACCIDENT (a half-rebuilt venv, a `pip install` that clobbered the interpreter, an earlier row
    # that removed what this row needs) must refuse the row rather than run it under an
    # `interpreter=venv` label the code can no longer back. This SHRINKS the check->exec window
    # from hours to milliseconds; it does not close it (see LIMITS in the header), and it adds
    # nothing job-writable to the trust path, so it does not reopen what D46 deleted.
    if ! _venv_usable_ok; then
      echo "[queue] REFUSED at launch: job '$job' needs the repo venv interpreter (its command's leading token is bare python/python3) but $VENV_REASON — the venv passed the parse-time check, so something changed since. Refusing this row only; nothing was launched, other rows are unaffected." >&2
      RESULTS+=("$job"$'\t'"venv-refused"$'\t'"-")
      LAST_LAUNCH_PID=""
      [[ -n "$g" ]] && _free_gpu "$g"
      return 1
    fi
    cmd="$(_rewrite_leading "$cmd")"
    # Durable record of what a claimed row ACTUALLY ran (D46 clause 4): the launch line below goes
    # to the queue's stdout only, which in a tmux session is not durable, and the absolute path
    # otherwise survives only in the transient /proc cmdline. write_summary() replays this into
    # runs/<sweep>/queue_summary.md so a post-mortem can read it.
    CLAIMED_CMDS+=("$job"$'\t'"$cmd")
  fi
  # NOT-COVERED marker (D46's priced limit, made VISIBLE): without it, `interpreter=passthrough:
  # python` reads identically for a harmless row and for `python train.py && python eval.py`, whose
  # second half silently resolves to the ambient interpreter. Warned at parse time, marked here, and
  # recorded durably in the summary. Never a refusal.
  local ncmark=""
  if (( nc > 0 )); then
    ncmark=" not-covered-chained-python=$nc"
    NOTCOVERED_ROWS+=("$job"$'\t'"$nc"$'\t'"$cmd")
  fi
  # --debug-stall-dump: PYTHONFAULTHANDLER=1 makes a SIGABRT (sent on a detected wedge) print a
  # full Python traceback into the job log. Empty unless the flag is set -> canon path unchanged.
  local fh=()
  (( DEBUG_DUMP )) && fh=(PYTHONFAULTHANDLER=1)
  if [[ -n "$g" ]]; then
    # PREALLOCATE=true so XLA_PYTHON_CLIENT_MEM_FRACTION actually bounds VRAM (else inert).
    sr+=(--gpu "$g" --gpu-frac "$FRAC" -- env ${fh[@]+"${fh[@]}"} JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=true bash -c "cd $REPO_ROOT && $cmd")
  elif (( DEBUG_DUMP )); then
    sr+=(--cpu -- env "${fh[@]}" bash -c "cd $REPO_ROOT && $cmd")
  else
    sr+=(--cpu -- bash -c "cd $REPO_ROOT && $cmd")
  fi
  if [[ "$DRY" -eq 1 ]]; then
    echo "[queue][dry] $job dev=$dev gpu=${g:-cpu} mem=$mem interpreter=$interp$ncmark -> $cmd" | tee -a "$log"
    ( sleep 1 ) &   # stand-in child so scheduling logic exercises reap/free
  else
    # D46 clause 4: the LIVE launch line prints the REWRITTEN command for a claimed row, exactly as
    # --dry-run does — the operator sees the absolute path, not a description of one. Passthrough
    # rows keep the pre-891 line verbatim (nothing was rewritten, so there is nothing to show).
    if [[ "$interp" == "venv" ]]; then
      echo "[queue] launch $job dev=$dev gpu=${g:-cpu} mem=$mem interpreter=$interp$ncmark log=$log -> $cmd"
    else
      echo "[queue] launch $job dev=$dev gpu=${g:-cpu} mem=$mem interpreter=$interp$ncmark log=$log"
    fi
    ( "${sr[@]}" >"$log" 2>&1 ) &
  fi
  local pid=$!
  LAST_LAUNCH_PID="$pid"
  PID2JOB[$pid]="$job"; PID2LOG[$pid]="$log"
  [[ -n "$g" ]] && PID2GPU[$pid]="$g"
  return 0
}

# ---- checkpoint-save WEDGE watchdog (TASK_841) ---------------------------------------------
# _config_name_for / _resolve_glob: resolve the training identity of a manifest row. The CSV
# watched, and the ckpt dir archived, follow the trainer's own convention (logs/training_<name>
# .csv, jax_ckpt_fast/<name>) where <name> comes from `--name value` OR `--name=value` in the
# command, falling back to config.name='default' (mcts_trader/training/cli.py) when --name is
# absent; a `csvglob:` token (stripped during parse -> GLOB[idx]) overrides the CSV path.
# NOTE: already-archived `*_wedged_ep*` artifacts are excluded from every glob expansion below,
# so a broad csvglob never re-matches (and re-processes) its own prior wedge archives.
_config_name_for() {  # $1 idx -> echo config name (from --name value | --name=value | 'default')
  local idx="$1" cmd="${CMD[$idx]}"
  if [[ "$cmd" =~ (^|[[:space:]])--name[[:space:]]+([^[:space:]]+) ]]; then
    echo "${BASH_REMATCH[2]}"
  elif [[ "$cmd" =~ (^|[[:space:]])--name=([^[:space:]]+) ]]; then
    echo "${BASH_REMATCH[2]}"
  else
    echo "default"
  fi
}

_resolve_glob() {  # $1 idx -> echo the CSV glob to watch for this row
  local idx="$1" g="${GLOB[$idx]:-}"
  if [[ -n "$g" ]]; then echo "$g"; return 0; fi
  echo "logs/training_$(_config_name_for "$idx").csv"
}

_newest_mtime() {  # $1 glob -> echo epoch-seconds mtime of the newest matching file, or "" if none
  local g="$1" f newest="" m
  # shellcheck disable=SC2086  (deliberate glob expansion; no match -> loop body skipped)
  for f in $g; do
    [[ -f "$f" ]] || continue
    [[ "$f" == *_wedged_ep* ]] && continue   # never re-match our own archives
    m="$(stat -c %Y "$f" 2>/dev/null)" || continue
    if [[ -z "$newest" || "$m" -gt "$newest" ]]; then newest="$m"; fi
  done
  echo "$newest"
}

_csv_epoch() {  # $1 glob -> echo the last epoch number from the newest matching CSV, or "unknown"
  local g="$1" f newest="" nf="" m last field
  # shellcheck disable=SC2086
  for f in $g; do
    [[ -f "$f" ]] || continue
    [[ "$f" == *_wedged_ep* ]] && continue
    m="$(stat -c %Y "$f" 2>/dev/null)" || continue
    if [[ -z "$newest" || "$m" -gt "$newest" ]]; then newest="$m"; nf="$f"; fi
  done
  [[ -z "$nf" ]] && { echo "unknown"; return 0; }
  last="$(tail -n 1 "$nf" 2>/dev/null)"
  field="${last%%,*}"
  [[ "$field" =~ ^[0-9]+$ ]] && echo "$field" || echo "unknown"
}

_archive_wedged() {  # $1 job $2 name $3 wtag $4 glob $5 runlog -> move partials aside, best-effort
  # $wtag is the (already uniquified) archive suffix, e.g. _wedged_ep1581 (see handle_wedge).
  local job="$1" name="$2" wtag="$3" glob="$4" runlog="$5" f fastlog ckpt
  # CSV(s) matched by the glob (excluding prior archives) -> <stem>$wtag.<ext>
  # shellcheck disable=SC2086
  for f in $glob; do
    [[ -e "$f" ]] || continue
    [[ "$f" == *_wedged_ep* ]] && continue
    if [[ "$f" == *.* ]]; then mv "$f" "${f%.*}${wtag}.${f##*.}" 2>/dev/null || true
    else mv "$f" "${f}${wtag}" 2>/dev/null || true; fi
  done
  fastlog="logs/training_fast_${name}.log"
  [[ -e "$fastlog" ]] && mv "$fastlog" "logs/training_fast_${name}${wtag}.log" 2>/dev/null || true
  ckpt="jax_ckpt_fast/${name}"
  [[ -d "$ckpt" ]] && mv "$ckpt" "${ckpt}${wtag}" 2>/dev/null || true
  # queue per-job run log: move aside so the relaunch starts a fresh $RUN_DIR/<job>.log
  [[ -e "$runlog" ]] && mv "$runlog" "$RUN_DIR/${job}${wtag}.log" 2>/dev/null || true
}

_watch_register() {  # $1 idx -> arm the watchdog clocks for the pid launch() just created
  (( STALL_TIMEOUT > 0 )) || return 0        # watchdog OFF -> register nothing (never watched)
  local idx="$1" pid="$LAST_LAUNCH_PID" glob
  [[ -n "$pid" ]] || return 0
  glob="$(_resolve_glob "$idx")"
  PID2IDX["$pid"]="$idx"
  PID2CSVGLOB["$pid"]="$glob"
  # EXPLICIT = the operator declared this job's CSV via a csvglob: token. Only explicit jobs are
  # judged before any CSV exists (no-CSV grace branch); INFERRED jobs (--name / 'default') are
  # watched ONLY once they actually PRODUCE a fresh training CSV, so CSV-less eval/benchmark jobs
  # are immune from a false kill.
  PID2EXPLICIT["$pid"]=0; [[ -n "${GLOB[$idx]:-}" ]] && PID2EXPLICIT["$pid"]=1
  # BASELINE = newest matching mtime BEFORE launch (a stale CSV from a prior run). The strict
  # stall clock starts only after the live mtime advances past this baseline, so a leftover CSV
  # cannot defeat the warmup grace.
  PID2BASELINE["$pid"]="$(_newest_mtime "$glob")"
  PID2STARTTS["$pid"]=$SECONDS
  PID2LASTCHANGE["$pid"]=$SECONDS
  PID2LASTMTIME["$pid"]=""
}

_relaunch() {  # $1 idx -> re-run the same manifest row once (called after a wedge kill)
  local idx="$1" job="${JOB_ID[$idx]}" dev="${DEV[$idx]}" mem="${MEM[$idx]}" cmd="${CMD[$idx]}" g="" ws
  if [[ "$dev" == "gpu" ]]; then
    ws=$SECONDS
    while [[ -z "$g" ]]; do
      reap
      ACQUIRED_GPU=""
      if acquire_idle_gpu; then g="$ACQUIRED_GPU"; fi
      if [[ -z "$g" ]]; then
        if (( ACQUIRE_TIMEOUT > 0 && SECONDS - ws >= ACQUIRE_TIMEOUT )); then
          echo "[queue] WEDGE-RELAUNCH TIMEOUT: no free+idle GPU for retry of '$job' after ${ACQUIRE_TIMEOUT}s — FAILED" >&2
          RESULTS+=("$job"$'\t'"wedged-failed"$'\t'"-"); return 1
        fi
        sleep "$POLL"
      fi
    done
    # launch() returns non-zero if it REFUSED the row (launch-time venv re-probe): register no
    # watchdog state and wait for no VRAM that will never rise.
    if launch "$job" gpu "$mem" "$g" "$cmd" "$(_interp_for_idx "$idx")" "${CHAIN[$idx]:-0}"; then
      _watch_register "$idx"
      confirm_vram_or_warn "$g" "$job"
    fi
  else
    ws=$SECONDS
    while [[ "$(_cpu_running)" -ge "$MAX_CPU" ]]; do
      reap
      [[ "$(_cpu_running)" -lt "$MAX_CPU" ]] && break
      if (( ACQUIRE_TIMEOUT > 0 && SECONDS - ws >= ACQUIRE_TIMEOUT )); then
        echo "[queue] WEDGE-RELAUNCH TIMEOUT: no free CPU slot for retry of '$job' after ${ACQUIRE_TIMEOUT}s — FAILED" >&2
        RESULTS+=("$job"$'\t'"wedged-failed"$'\t'"-"); return 1
      fi
      sleep "$POLL"
    done
    launch "$job" cpu "$mem" "" "$cmd" "$(_interp_for_idx "$idx")" "${CHAIN[$idx]:-0}" && _watch_register "$idx"
  fi
}

handle_wedge() {  # $1 pid -> dump (opt) + archive + kill scope/pid, then relaunch-once or FAIL
  local pid="$1" idx job dev mem g runlog glob N name wtag unit mainpid dumpfile n archived_log
  idx="${PID2IDX[$pid]}"; job="${JOB_ID[$idx]}"; dev="${DEV[$idx]}"; mem="${MEM[$idx]}"
  g="${PID2GPU[$pid]:-}"; runlog="${PID2LOG[$pid]}"; glob="${PID2CSVGLOB[$pid]}"
  N="$(_csv_epoch "$glob")"
  name="$(_config_name_for "$idx")"
  # Uniquify the archive suffix so a SAME-epoch second wedge cannot nest into / bury the first
  # archive (mv dir-into-existing-dir). Keep the plain _wedged_ep<N> for the common single case.
  wtag="_wedged_ep${N}"
  if [[ -e "jax_ckpt_fast/${name}${wtag}" || -e "$RUN_DIR/${job}${wtag}.log" ]]; then
    wtag="_wedged_ep${N}_$(date +%s)"
  fi
  echo "[queue] WEDGE detected: '$job' — training CSV '$glob' unmodified >= ${STALL_TIMEOUT}s while pid=$pid alive (ep=$N)" >&2

  if (( DEBUG_DUMP )); then
    dumpfile="$RUN_DIR/${job}${wtag}.dump"
    { echo "=== wedge thread dump $(date -Iseconds) job=$job pid=$pid ep=$N ==="; } >"$dumpfile" 2>/dev/null || true
    unit="$(systemctl --user list-units --all --no-legend "${job}_*.scope" 2>/dev/null | awk '{print $1}' | head -1)"
    if [[ -n "${unit:-}" ]]; then mainpid="$(systemctl --user show -p MainPID --value "$unit" 2>/dev/null)"; fi
    if command -v py-spy >/dev/null 2>&1 && [[ -n "${mainpid:-}" && "${mainpid:-0}" != 0 ]]; then
      { echo "--- py-spy dump (MainPID=$mainpid) ---"; py-spy dump --pid "$mainpid" --subprocesses; } >>"$dumpfile" 2>&1 || true
    fi
    echo "[queue] WEDGE dump: SIGABRT -> ${job}_*.scope (faulthandler traceback -> ${runlog}); py-spy -> $dumpfile" >&2
    systemctl --user kill --signal=SIGABRT "${job}_*.scope" 2>/dev/null || true
    sleep 2
  fi

  # Kill the whole cgroup (scope) then the launching subshell; reap the zombie.
  systemctl --user stop "${job}_*.scope" 2>/dev/null || true
  kill "$pid" 2>/dev/null || true
  { wait "$pid"; } 2>/dev/null || true

  archived_log="$RUN_DIR/${job}${wtag}.log"
  _archive_wedged "$job" "$name" "$wtag" "$glob" "$runlog"

  # Free + unlock the GPU (mirror reap) and drop all tracking for this pid.
  if [[ -n "$g" ]]; then
    FREE_GPUS+=("$g")
    local fd="${GPU_LOCK_FD[$g]:-}"
    if [[ -n "$fd" ]]; then flock -u "$fd" 2>/dev/null || true; eval "exec ${fd}>&-" 2>/dev/null || true; unset 'GPU_LOCK_FD[$g]'; fi
  fi
  unset 'PID2JOB[$pid]' 'PID2GPU[$pid]' 'PID2LOG[$pid]' 'PID2IDX[$pid]' 'PID2CSVGLOB[$pid]' \
        'PID2STARTTS[$pid]' 'PID2LASTMTIME[$pid]' 'PID2LASTCHANGE[$pid]' 'PID2EXPLICIT[$pid]' 'PID2BASELINE[$pid]'

  n="${RETRY[$job]:-0}"
  if (( n < MAX_WEDGE_RETRY )); then
    RETRY["$job"]=$((n + 1)); WEDGE_RETRIED_COUNT=$((WEDGE_RETRIED_COUNT + 1))
    echo "[queue] WEDGED-RETRIED: '$job' killed + archived (ep=$N) -> relaunching (auto-retry $((n + 1))/$MAX_WEDGE_RETRY)" >&2
    _relaunch "$idx"
  else
    echo "[queue] WEDGE-FAILED: '$job' wedged AGAIN after its retry — FAILED (fail loud, no further retry)" >&2
    RESULTS+=("$job"$'\t'"wedged-failed"$'\t'"${archived_log#"$REPO_ROOT"/}")
  fi
}

check_stalls() {  # scan live watchdog-tracked jobs; a CSV frozen past the deadline -> handle_wedge
  (( STALL_TIMEOUT > 0 )) || return 0
  ((${#PID2JOB[@]})) || return 0
  local pid glob m baseline fresh since
  for pid in "${!PID2JOB[@]}"; do
    [[ -n "${PID2IDX[$pid]:-}" ]] || continue     # only jobs armed via _watch_register
    kill -0 "$pid" 2>/dev/null || continue        # dead pids are reap()'s job, not a wedge
    glob="${PID2CSVGLOB[$pid]}"
    m="$(_newest_mtime "$glob")"
    baseline="${PID2BASELINE[$pid]}"
    # "fresh" = a training CSV this job actually PRODUCED: it exists AND its mtime advanced past
    # any pre-launch stale baseline. Until then the job has not demonstrably started training.
    fresh=0
    if [[ -n "$m" ]] && { [[ -z "$baseline" ]] || (( m > baseline )); }; then fresh=1; fi
    if (( fresh )); then
      if [[ "$m" != "${PID2LASTMTIME[$pid]}" ]]; then
        PID2LASTMTIME["$pid"]="$m"; PID2LASTCHANGE["$pid"]=$SECONDS; continue   # (re)start strict clock
      fi
      since=$(( SECONDS - PID2LASTCHANGE[$pid] ))
      (( since >= STALL_TIMEOUT )) && handle_wedge "$pid"
    elif (( PID2EXPLICIT[$pid] )); then
      # Operator explicitly declared this job's CSV (csvglob:) but none has appeared yet -> judge
      # from process start + grace (JAX/CUDA warmup). INFERRED jobs get NO such branch: a job that
      # never produces a training CSV (eval/benchmark/data-prep) is never killed.
      since=$(( SECONDS - PID2STARTTS[$pid] ))
      (( since >= STALL_TIMEOUT + STALL_GRACE )) && handle_wedge "$pid"
    fi
  done
}

write_summary() {
  END_TS="$(date -Iseconds)"
  local r j rc lg fails=0
  {
    echo "# Queue summary — sweep \`$SWEEP\`"
    echo ""
    echo "- manifest: \`$MANIFEST\`"
    echo "- started: $START_TS"
    echo "- finished: $END_TS"
    echo "- gpu pool: {$GPUS_CSV} · frac $FRAC · idle<${IDLE_MIB}MiB · acquire-timeout ${ACQUIRE_TIMEOUT}s"
    echo ""
    echo "| job | rc | log |"
    echo "|-----|----|-----|"
    for r in "${RESULTS[@]}"; do
      IFS=$'\t' read -r j rc lg <<< "$r"
      local mark="$rc"
      [[ "$rc" != "0" ]] && { mark="**$rc**"; fails=$((fails+1)); }
      echo "| $j | $mark | \`${lg#"$REPO_ROOT"/}\` |"
    done
    echo ""
    echo "**${#RESULTS[@]} row(s) · $fails non-zero/failed.**"
    # Watchdog note — emitted ONLY when a wedge actually fired, so the no-stall summary stays
    # byte-identical to the pre-TASK_841 output on any manifest that never wedges.
    (( WEDGE_RETRIED_COUNT > 0 )) && { echo ""; echo "**$WEDGE_RETRIED_COUNT job(s) auto-retried after a checkpoint-save wedge (WEDGED-RETRIED).**"; }
    # Interpreter block (TASK_891 — D46 clause 4). Emitted ONLY when a row's leading bare
    # python/python3 was actually rewritten, so a manifest with no claimed row produces a summary
    # byte-identical to the pre-891 one. This is the DURABLE copy of the rewritten command: the
    # launch line lives on the queue's stdout (tmux scrollback, not durable) and the absolute path
    # is otherwise only in the transient /proc cmdline.
    if ((${#CLAIMED_CMDS[@]})); then
      echo ""
      echo "### interpreter: rewritten commands (${#CLAIMED_CMDS[@]} claimed launch(es))"
      echo ""
      echo "A row whose command's LEADING token was bare \`python\`/\`python3\` had that token"
      echo "rewritten to the absolute repo-venv path below before launch. Every other spelling ran"
      echo "byte-untouched (\`passthrough\`) — see the interpreter contract in run_queue.sh."
      echo ""
      local cj ccmd
      for r in "${CLAIMED_CMDS[@]}"; do
        IFS=$'\t' read -r cj ccmd <<< "$r"
        echo "- \`$cj\` -> \`$ccmd\`"
      done
    fi
    # NOT-COVERED chained pythons in rows that actually launched (D46's accepted limit, recorded
    # durably rather than only warned to stderr at parse time).
    if ((${#NOTCOVERED_ROWS[@]})); then
      echo ""
      echo "### interpreter: NOT COVERED (${#NOTCOVERED_ROWS[@]} row(s) with a chained bare python)"
      echo ""
      echo "The contract covers a row's LEADING token only (D46). These rows also invoke bare"
      echo "\`python\`/\`python3\` further along the command, where NOTHING rewrites it — it resolves"
      echo "through whatever PATH the job inherited (bare \`python\` is python2 on gpu-server). This"
      echo "is an accepted limit, not a failure: the rows ran as written."
      echo ""
      local nj nn ncmd
      for r in "${NOTCOVERED_ROWS[@]}"; do
        IFS=$'\t' read -r nj nn ncmd <<< "$r"
        echo "- \`$nj\` ($nn chained): \`$ncmd\`"
      done
    fi
  } > "$SUMMARY"
  SUMMARY_WRITTEN=1
}

# ---- interpreter contract helpers (TASK_891 — D46: ABSOLUTE-PATH SUBSTITUTION) ----------------
_leading_token() {  # $1 cmd -> first whitespace-delimited token (leading-whitespace-safe)
  local c="$1"
  [[ "$c" =~ ^[[:space:]]*([^[:space:]]+) ]] && { echo "${BASH_REMATCH[1]}"; return 0; }
  echo ""
}
_is_bare_python_leading() {  # $1 cmd -> 0 iff the LEADING token is exactly python or python3
  local t; t="$(_leading_token "$1")"
  [[ "$t" == "python" || "$t" == "python3" ]]
}
_ucp_visit() {  # HELPER FOR _uncovered_chained_pythons ONLY — nothing else may call it. Bash's
                # dynamic scoping is deliberate here: this reads/updates that function's locals
                # (tok/tq/started, n/idx/cmdpos) so the "consume one finished token" logic exists
                # ONCE and runs identically at a whitespace boundary and at end-of-string.
  local t="$tok" q="$tq" next=0
  # Count: a token AFTER the leading one, not from inside quotes, spelled exactly python/python3,
  # standing in command position.
  if (( idx > 0 && cmdpos && ! q )) && [[ "$t" == "python" || "$t" == "python3" ]]; then
    n=$((n+1))
  fi
  # Decide whether the NEXT token starts in command position. A QUOTED token never does this: a
  # quoted `&&` is a literal argument, not a separator.
  if (( ! q )); then
    case "$t" in
      '&&'|'||'|';'|'|'|'&'|'('|'{'|then|else|do) next=1;;
      # TRANSPARENT PREFIXES are transparent only when THEY are themselves in command position.
      # `--desc env python` is an argument list, so its `env` is a WORD, not the env(1) command, and
      # the `python` after it is not an invocation. (Cycle-3 false positive; `env python x.py`, where
      # `env` really is the command, still counts.)
      env) (( cmdpos )) && next=1;;
      *=*) [[ "$t" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] && (( cmdpos )) && next=1;;
    esac
  fi
  cmdpos=$next
  idx=$((idx+1))
  tok=""; tq=0; started=0
}
_uncovered_chained_pythons() {  # $1 cmd -> count of bare `python`/`python3` tokens in COMMAND
                    # POSITION after the leading one. These are the NOT-COVERED tier (D46 priced
                    # them, and this row does NOT start refusing them): nothing rewrites them, so
                    # they resolve through whatever PATH the job inherits — and on gpu-server bare
                    # `python` is python2 (the repo says so itself:
                    # experiments/884_rung1/manifest_884_battery.txt:6). Without this, a landmine row
                    # (`python train.py && python eval.py`) is indistinguishable at runtime from a
                    # harmless one, so the queue WARNS by name and line.
                    #
                    # WHAT THIS IS, EXACTLY. The count is not just a stderr warning: it lands in a
                    # DURABLE, operator-facing record (the NOT-COVERED block of
                    # runs/<sweep>/queue_summary.md), and a post-mortem has to be able to trust that
                    # record — so this comment states what the code does, not what would be nice:
                    #   * Tokenisation is QUOTE-AWARE and is NOT a shell parser. One character pass
                    #     splits on UNQUOTED whitespace and marks any token that took a character
                    #     from inside '...' / "..." / a backslash escape as QUOTED. A quoted token is
                    #     never counted AND never acts as a separator or prefix, so nothing inside a
                    #     quoted span is ever scanned: `--note "a && python b"` and
                    #     `--desc 'x && python y'` are arguments and stay arguments.
                    #   * COMMAND POSITION = the previous token is an UNQUOTED shell separator
                    #     (&& || ; | & ( {, or then/else/do), or a TRANSPARENT PREFIX (`env`, a
                    #     VAR=value assignment) that was ITSELF in command position. The start of the
                    #     command counts as a command position, so `env python x.py` and
                    #     `PATH=/x python x.py` count their python, while `--desc env python` and
                    #     `--opt FOO=bar python` do not.
                    #   * Consequently it NEVER fires on an argument that merely reads `python`
                    #     (`--tag python`, `--out "a python b"`, `--desc env python`). That is
                    #     asserted, not asserted-in-a-comment: gate NCT in
                    #     tools/tests/test_run_queue_interpreter.sh runs THIS function over a table
                    #     of 21 spellings and is mutation-proved.
                    #   * ACCEPTED BLIND SPOTS, deliberately not chased (a narrow heuristic is worth
                    #     more here than a half-built shell parser): a python inside quotes
                    #     (`bash -c "python x"`), inside a substitution (`$(python x)`, backticks),
                    #     behind a launcher this does not model (`xargs python`, `time python`,
                    #     `sudo python`, `nohup python`), an escaped spelling (`\python`), or one the
                    #     job spawns at runtime. So n=0 means "no chained one VISIBLE from here",
                    #     never "none present": the summary block is a FLOOR on what is uncovered,
                    #     not a census.
  local c="$1" len i ch
  local tok="" tq=0 started=0 sq=0 dq=0    # tq = this token took a character from inside quotes
  local n=0 idx=0 cmdpos=1                 # cmdpos: the start of the command IS a command position
  len=${#c}
  for (( i = 0; i < len; i++ )); do
    ch="${c:i:1}"
    if (( sq )); then                      # inside '...': everything is literal until the next '
      if [[ "$ch" == "'" ]]; then sq=0; else tok+="$ch"; fi
    elif (( dq )); then                    # inside "...": backslash escapes one char
      case "$ch" in
        '\') i=$((i+1)); tok+="${c:i:1}";;
        '"') dq=0;;
        *)   tok+="$ch";;
      esac
    else
      case "$ch" in
        "'")            sq=1; tq=1; started=1;;
        '"')            dq=1; tq=1; started=1;;
        '\')            i=$((i+1)); tok+="${c:i:1}"; tq=1; started=1;;
        ' '|$'\t'|$'\n') (( started )) && _ucp_visit;;   # UNQUOTED whitespace = token boundary
        *)              tok+="$ch"; started=1;;
      esac
    fi
  done
  (( started )) && _ucp_visit              # flush the final token (also covers an unclosed quote)
  echo "$n"
}
_interp_for_idx() {  # $1 idx -> the interpreter= label (must never claim more than the code did)
  if [[ "${NEED[$1]}" == "leading" ]]; then
    echo "venv"
  else
    echo "passthrough:$(_leading_token "${CMD[$1]}")"
  fi
}
_rewrite_leading() {  # $1 cmd -> the same command with its LEADING bare python/python3 token
                      # replaced by the absolute path of the repo venv interpreter. Only ever
                      # called for a row already classified `leading` (the token is exactly
                      # `python` or `python3`); %q leaves the rest of the command untouched.
  local c="$1" rest
  [[ "$c" =~ ^[[:space:]]*[^[:space:]]+(.*)$ ]] || { printf '%s' "$c"; return 0; }
  rest="${BASH_REMATCH[1]}"
  printf '%s%s' "$(printf '%q' "$VENV_PYTHON")" "$rest"
}
PROBE_TIMEOUT=10   # seconds; a wedged interpreter must never hang manifest parsing OR a launch
_venv_identity_ok() {  # $1 interpreter -> 0 if it behaves like THIS venv's working Python 3, else 1
                       # (sets VENV_REASON). Deliberately more than an rc check and more than "did
                       # it print 3": an rc-only check blesses `/bin/true` (ignores its arguments,
                       # exits 0) and a digit-only check blesses any fixed-output stub. So the probe
                       # passes a NONCE only known now and demands, in one line each: the major
                       # version, `sys.executable` (must be the path we invoked) and `sys.prefix`
                       # (must be the venv root), plus arithmetic on the nonce. Bounded by `timeout`
                       # so a wedged interpreter hangs neither the parse nor a launch.
                       # NOT AUTHENTICATION — see the LIMITS block in the header: a forgery that
                       # delegates to a real python3 passes this, and that is unfixable by probing.
  local interp="$1" out rc nonce want_root
  want_root="$(cd "$(dirname "$interp")/.." 2>/dev/null && pwd)"
  # Arithmetic (not string concatenation) so the nonce can never acquire a leading zero, which
  # bash's $(( )) would read as octal.
  nonce=$(( RANDOM * 32768 + RANDOM + $$ ))
  out="$(timeout "$PROBE_TIMEOUT" "$interp" -c 'import sys
sys.stdout.write("%d\n%s\n%s\n%d\n" % (sys.version_info[0], sys.executable, sys.prefix, int(sys.argv[1]) * 3 + 7))' "$nonce" 2>/dev/null)"
  rc=$?
  if (( rc == 124 )); then
    VENV_REASON="$interp did not answer within ${PROBE_TIMEOUT}s (wedged interpreter)"; return 1
  elif (( rc != 0 )); then
    VENV_REASON="$interp is not a working Python 3 interpreter (identity check exit $rc)"; return 1
  fi
  local major exe prefix answer
  { IFS= read -r major; IFS= read -r exe; IFS= read -r prefix; IFS= read -r answer; } <<< "$out"
  if [[ "${major:-}" != "3" ]]; then
    VENV_REASON="$interp did not identify itself as Python 3 (got '${major:-<no output>}')"; return 1
  elif [[ "${answer:-}" != "$(( nonce * 3 + 7 ))" ]]; then
    VENV_REASON="$interp did not evaluate the probe nonce (fixed-output stub? expected $(( nonce * 3 + 7 )), got '${answer:-<nothing>}')"; return 1
  elif [[ "${exe:-}" != "$interp" ]]; then
    VENV_REASON="$interp reports sys.executable='${exe:-<nothing>}' — not the interpreter we invoked"; return 1
  elif [[ -n "$want_root" && "${prefix:-}" != "$want_root" ]]; then
    VENV_REASON="$interp reports sys.prefix='${prefix:-<nothing>}' — not the repo venv root $want_root (a python, but not THIS venv's)"; return 1
  fi
  return 0
}
_venv_usable_ok() {  # -> 0 if $VENV_PYTHON exists, is executable, AND passes the identity probe.
                     # The SINGLE check used at BOTH ends of the deferral (parse time and again
                     # immediately before each claimed row's launch), so the two can never drift.
  [[ -n "${VENV_PYTHON:-}" ]] || { VENV_REASON="no repo venv interpreter path resolved"; return 1; }
  if [[ ! -x "$VENV_PYTHON" ]]; then
    VENV_REASON="repo venv python not found/executable at $VENV_PYTHON"; return 1
  fi
  _venv_identity_ok "$VENV_PYTHON"
}

# ---- parse manifest (finding #4: malformed lines become explicit FAILED rows) ----
# GLOB[idx] holds an explicit `csvglob:<glob>` watchdog override (stripped from the command so
# the job runs exactly as written); empty means "infer from --name / job_id" (see _resolve_glob).
# NEED[idx] is the interpreter-contract classification (TASK_891 — D46): `leading` (claimed,
# rewritten to the absolute venv path) or `passthrough` (everything else, byte-untouched).
# LINE[idx] is the 1-based manifest line number, so a refusal can name it.
# CHAIN[idx] is the count of NOT-COVERED chained bare pythons visible in the command (see
# _uncovered_chained_pythons) — warned about at parse time, marked on the launch line, and recorded
# in the summary. Never a refusal: D46 priced chained pythons as an ACCEPTED limit.
JOB_ID=(); DEV=(); MEM=(); CMD=(); GLOB=(); LINE=(); NEED=(); CHAIN=()
lineno=0
while read -r job dev mem cmd || [[ -n "$job" ]]; do
  lineno=$((lineno+1))
  [[ -z "${job// }" ]] && continue
  [[ "${job:0:1}" == "#" ]] && continue
  if [[ -z "$dev" || -z "$mem" || -z "$cmd" ]]; then
    echo "[queue] MALFORMED manifest line $lineno for job='$job' (need 4 fields: id device mem command)" >&2
    RESULTS+=("${job:-line$lineno}"$'\t'"malformed"$'\t'"-")
    continue
  fi
  case "$dev" in
    gpu|cpu) ;;
    *) echo "[queue] MALFORMED manifest line $lineno job '$job': bad device '$dev' (want gpu|cpu)" >&2
       RESULTS+=("$job"$'\t'"malformed"$'\t'"-"); continue;;
  esac
  glob=""
  if [[ "$cmd" == csvglob:* ]]; then
    glob="${cmd%% *}"; glob="${glob#csvglob:}"    # first token declares the CSV glob
    if [[ "$cmd" == *" "* ]]; then cmd="${cmd#* }"; else cmd=""; fi
    if [[ -z "$cmd" ]]; then
      echo "[queue] MALFORMED manifest line $lineno job '$job': csvglob: token but no command follows" >&2
      RESULTS+=("$job"$'\t'"malformed"$'\t'"-"); continue
    fi
  fi
  need="passthrough"
  _is_bare_python_leading "$cmd" && need="leading"
  # NOT-COVERED chained python: warn LOUD and BY NAME, but schedule the row (D46 accepted this
  # limit; refusing here would break manifests the ratified contract permits).
  chain="$(_uncovered_chained_pythons "$cmd")"
  if (( chain > 0 )); then
    echo "[queue] NOT-COVERED manifest line $lineno job '$job': $chain chained bare python/python3 invocation(s) after the leading token. The interpreter contract covers the LEADING token ONLY (D46) — the chained one(s) resolve through whatever PATH this job inherits, which on gpu-server is python2 for bare \`python\`. Spell them \`trader_env/bin/python\` if that is not what you want. Scheduling the row as written." >&2
  fi
  JOB_ID+=("$job"); DEV+=("$dev"); MEM+=("$mem"); CMD+=("$cmd"); GLOB+=("$glob"); LINE+=("$lineno"); NEED+=("$need"); CHAIN+=("$chain")
done < "$MANIFEST"

MALFORMED=${#RESULTS[@]}

# ---- interpreter contract: resolve the venv, ONLY if some row needs it (TASK_891 — D46) -------
# A manifest with no `leading` row never touches trader_env and never prints the venv: line below
# — behaviour is untouched for every sweep that doesn't reference bare python as a leading token
# (acceptance v). This is the FIRST of two checks, not the only one: it exists to refuse a doomed
# manifest up front, by name and manifest line, before anything is scheduled. Because
# trader_env/bin/python is an ordinary same-user-writable path (see the THREAT MODEL in the
# header), the same check is re-run in launch() immediately before each claimed row actually goes,
# so a row deferred for hours behind a GPU wait cannot be launched against an interpreter that
# broke after this point.
VENV_PYTHON=""
VENV_OK=0
VENV_REASON=""
NEED_VENV=0
for _i in "${!NEED[@]}"; do [[ "${NEED[$_i]}" == "leading" ]] && { NEED_VENV=1; break; }; done
if (( NEED_VENV )); then
  VENV_PYTHON="$REPO_ROOT/trader_env/bin/python"
  _venv_usable_ok && VENV_OK=1
  if (( VENV_OK )); then
    echo "[queue] venv: $VENV_PYTHON (identity OK: python3) — leading bare python/python3 rows are rewritten to this absolute path in the command string; nothing else (PATH, env, other rows) is touched"
  else
    echo "[queue] venv: UNUSABLE — $VENV_REASON" >&2
  fi
fi

# Fail loud, per row (never per manifest): a `leading` row with no usable venv is REFUSED and
# dropped from what gets scheduled; `passthrough` rows are untouched and still run.
VENV_REFUSED=0
if (( NEED_VENV )) && (( ! VENV_OK )); then
  # The surviving rows are re-indexed here, in lockstep, so nothing downstream misattributes a
  # row's index after some rows are dropped.
  F_JOB_ID=(); F_DEV=(); F_MEM=(); F_CMD=(); F_GLOB=(); F_LINE=(); F_NEED=(); F_CHAIN=()
  for _i in "${!JOB_ID[@]}"; do
    if [[ "${NEED[$_i]}" == "leading" ]]; then
      echo "[queue] REFUSED manifest line ${LINE[$_i]} job '${JOB_ID[$_i]}': command's leading token is bare python/python3, which needs the repo venv interpreter, but $VENV_REASON — refusing this row only (no silent fallback); other rows are unaffected" >&2
      RESULTS+=("${JOB_ID[$_i]}"$'\t'"venv-refused"$'\t'"-")
      VENV_REFUSED=$((VENV_REFUSED+1))
      continue
    fi
    F_JOB_ID+=("${JOB_ID[$_i]}"); F_DEV+=("${DEV[$_i]}"); F_MEM+=("${MEM[$_i]}"); F_CMD+=("${CMD[$_i]}"); F_GLOB+=("${GLOB[$_i]}"); F_LINE+=("${LINE[$_i]}"); F_NEED+=("${NEED[$_i]}"); F_CHAIN+=("${CHAIN[$_i]}")
  done
  JOB_ID=("${F_JOB_ID[@]}"); DEV=("${F_DEV[@]}"); MEM=("${F_MEM[@]}"); CMD=("${F_CMD[@]}"); GLOB=("${F_GLOB[@]}"); LINE=("${F_LINE[@]}"); NEED=("${F_NEED[@]}"); CHAIN=("${F_CHAIN[@]}")
fi

N=${#JOB_ID[@]}
echo "[queue] sweep=$SWEEP jobs=$N malformed=$MALFORMED $( ((VENV_REFUSED>0)) && echo "venv_refused=$VENV_REFUSED " )gpus={${GPUS_CSV}} frac=$FRAC idle_mib=$IDLE_MIB poll=${POLL}s acquire_timeout=${ACQUIRE_TIMEOUT}s max_cpu=$MAX_CPU dry=$DRY $( ((STALL_TIMEOUT>0)) && echo "watchdog: armed stall_timeout=${STALL_TIMEOUT}s grace=${STALL_GRACE}s max=${MAX_WEDGE_RETRY} auto-retry" || echo "watchdog OFF (opt-in --stall-timeout)")"

# Arm the cleanup trap only now that PID2JOB/RESULTS/etc. exist (set -u safe).
trap cleanup INT TERM EXIT

if [[ "$N" -eq 0 ]]; then
  echo "[queue] no runnable jobs"
  write_summary
  cat "$SUMMARY"
  # non-zero if malformed lines were dropped, else 1 (nothing to do is still a failure to run)
  exit 1
fi

# ---- schedule ----
SCHED_ABORTED=0
for idx in "${!JOB_ID[@]}"; do
  job="${JOB_ID[$idx]}"; dev="${DEV[$idx]}"; mem="${MEM[$idx]}"; cmd="${CMD[$idx]}"
  # A GPU acquire-timeout (SCHED_ABORTED) abandons only the remaining GPU jobs; CPU jobs
  # need no GPU and must still run (finding N2).
  if (( SCHED_ABORTED )) && [[ "$dev" == "gpu" ]]; then
    echo "[queue] SKIP $job (GPU scheduler aborted after acquire-timeout)" >&2
    RESULTS+=("$job"$'\t'"skipped"$'\t'"-")
    continue
  fi
  if [[ "$dev" == "cpu" ]]; then
    # Bound the CPU-slot wait the same way as the GPU acquire (finding N3): a hung CPU job
    # must not let the queue spin forever silently. On deadline, SKIP this job (fail loud).
    wait_start=$SECONDS
    last_hb=$SECONDS
    cpu_ok=1
    while [[ "$(_cpu_running)" -ge "$MAX_CPU" ]]; do
      reap
      check_stalls
      [[ "$(_cpu_running)" -lt "$MAX_CPU" ]] && break
      now=$SECONDS
      if (( ACQUIRE_TIMEOUT > 0 && now - wait_start >= ACQUIRE_TIMEOUT )); then
        echo "[queue] ACQUIRE TIMEOUT: no free CPU slot for '$job' after ${ACQUIRE_TIMEOUT}s (max_cpu=$MAX_CPU running_cpu=$(_cpu_running)) — skipping" >&2
        RESULTS+=("$job"$'\t'"skipped"$'\t'"-")
        cpu_ok=0
        break
      fi
      if (( now - last_hb >= HEARTBEAT_EVERY )); then
        echo "[queue] heartbeat: $((now-wait_start))s waiting for a free CPU slot for '$job' | cpu_running=$(_cpu_running)/$MAX_CPU total_running=${#PID2JOB[@]}"
        last_hb=$now
      fi
      sleep "$POLL"
    done
    # `&&`: launch() returns non-zero if it REFUSED the row (launch-time venv re-probe) — there is
    # then no pid to arm the watchdog against.
    (( cpu_ok )) && { launch "$job" cpu "$mem" "" "$cmd" "$(_interp_for_idx "$idx")" "${CHAIN[$idx]:-0}" && _watch_register "$idx"; }
  else
    g=""
    wait_start=$SECONDS
    last_hb=$SECONDS
    while [[ -z "$g" ]]; do
      reap
      check_stalls
      ACQUIRED_GPU=""
      if acquire_idle_gpu; then g="$ACQUIRED_GPU"; fi
      if [[ -z "$g" ]]; then
        now=$SECONDS
        if (( ACQUIRE_TIMEOUT > 0 && now - wait_start >= ACQUIRE_TIMEOUT )); then
          echo "[queue] ACQUIRE TIMEOUT: no free+idle GPU for '$job' after ${ACQUIRE_TIMEOUT}s (pool={$GPUS_CSV} free=[${FREE_GPUS[*]:-}] running=${#PID2JOB[@]}) — abandoning remaining GPU jobs" >&2
          RESULTS+=("$job"$'\t'"skipped"$'\t'"-")
          SCHED_ABORTED=1
          break
        fi
        if (( now - last_hb >= HEARTBEAT_EVERY )); then
          if (( ACQUIRE_SAW_VALID == 0 )) && ((${#FREE_GPUS[@]})); then
            echo "[queue] heartbeat: $((now-wait_start))s — nvidia-smi returned NO data for pool GPUs [${FREE_GPUS[*]:-}] (driver/permission issue?); job='$job'"
          else
            echo "[queue] heartbeat: $((now-wait_start))s waiting for a free+idle GPU for '$job' | free=[${FREE_GPUS[*]:-}] running=${#PID2JOB[@]}"
          fi
          last_hb=$now
        fi
        sleep "$POLL"
      fi
    done
    if [[ -n "$g" ]]; then
      # launch() returns non-zero if it REFUSED the row (launch-time venv re-probe); it has already
      # released the GPU it was handed, so there is nothing to arm or to wait for VRAM on.
      if launch "$job" gpu "$mem" "$g" "$cmd" "$(_interp_for_idx "$idx")" "${CHAIN[$idx]:-0}"; then
        _watch_register "$idx"
        confirm_vram_or_warn "$g" "$job"
      fi
    fi
  fi
done

# ---- drain ----
echo "[queue] all launched; draining ..."
while (( ${#PID2JOB[@]} > 0 )); do reap; check_stalls; (( ${#PID2JOB[@]} > 0 )) && sleep "$POLL"; done

# ---- summary ----
write_summary
echo "[queue] done. summary -> $SUMMARY"
cat "$SUMMARY"

# exit non-zero if any row is non-zero/failed/skipped/malformed, so a watcher notices
for r in "${RESULTS[@]}"; do IFS=$'\t' read -r _ rc _ <<< "$r"; [[ "$rc" != "0" ]] && exit 1; done
exit 0
