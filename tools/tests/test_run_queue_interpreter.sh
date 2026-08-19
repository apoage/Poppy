#!/usr/bin/env bash
# test_run_queue_interpreter.sh — TASK_891 (D46, RATIFIED: ABSOLUTE-PATH SUBSTITUTION).
#
# Verifies the re-cut contract in tools/deploy/run_queue.sh. D46 supersedes D42(a)'s
# python-only shim-dir + PATH-prepend design (killed twice at runs 34/35: the prepend was
# itself the attack surface — a job-writable shim dir let `env … bash -c` and the wrapper's
# own `#!/usr/bin/env bash` shebang both resolve through it). The new mechanism has NO
# job-writable component in the trust path at all: a claimed row's leading token is rewritten,
# in the command STRING, to the absolute path of the repo venv python, once, at parse time.
# There is nothing left to forge, wedge, or lose +x during a deferred window — which is why
# this suite is much shorter than the halted one: whole classes of shim-defense scenarios
# (wrapper byte-compare, launch-time re-probe, SIGINT-during-construction, provenance forgery,
# probe-scratch leaks) have no equivalent under this mechanism, because the thing they defended
# against (a job-writable intermediary) no longer exists.
#
#   (i)   a LEADING bare `python`/`python3` token resolves to the repo venv via absolute-path
#         substitution, even with a python2 stub shadowing `python` on the ambient PATH.
#   (ii)  every other spelling (env python, bash -c "python ...", PATH=/x python, a SECOND bare
#         python in a chained row, ...) passes through byte-untouched and is labelled
#         `passthrough:<leading token>` — never a resolution claim the code cannot back.
#   (iii) a broken venv (missing/not executable) REFUSES only the `leading` rows, by name and
#         manifest line, at PARSE time; passthrough rows are unaffected.
#   (iv)  the queue makes NO PATH/env change of any kind: a claimed row's environment is
#         byte-identical to a passthrough row's (env diff empty, modulo per-invocation noise
#         that has nothing to do with the interpreter contract).
#   (v)   a manifest with no leading-python row behaves like pre-891 (no venv:/REFUSED banner,
#         same summary/scheduling as the actual HEAD script), modulo the new interpreter= label.
#   (vi)  the identity check refuses BY NAME a program that ignores its arguments and exits 0
#         (`/bin/true`) — an rc-only check would bless it.
#   (P)   POISON-AFTER-CHECK (run-36 BLOCKER): the parse-time check passes, then an ordinary
#         passthrough row replaces trader_env/bin/python mid-queue, and the LATER claimed row is
#         REFUSED at its own launch instead of executing the forgery under an `interpreter=venv`
#         label. The window is real because every job runs as the SAME uid (systemd-run --user
#         --scope) and that path is ordinary same-user-writable — there is no privilege boundary
#         and `protected` in .dark-factory/config.json has no enforcement code. The re-probe does
#         not CLOSE the window (nothing in a shell can); it shrinks it from hours to milliseconds.
#   (NCW) an ACCEPTED limit is still VISIBLE: a row with a chained bare python (the
#         runs/860_codistill/manifest_860.txt shape) is warned by name+line, marked on its launch
#         line, recorded in the summary — and still SCHEDULED. `--tag python` (an argument, not an
#         invocation) is NOT flagged.
#   (NCT) the detector's documented contract is TRUE OF THE CODE: the real
#         `_uncovered_chained_pythons` is run over a 22-row table — the arguments that must never be
#         flagged (`--tag python`, `--note "a && python b"`, `--desc env python`), the chained
#         invocations that must be, and the accepted false negatives the header names. The count
#         feeds a DURABLE operator-facing record (queue_summary.md's NOT-COVERED block), so a false
#         entry there is a false post-mortem, which is why the table is a gate and not a comment.
#   (PG)  the GPU REFUSAL PATH: a claimed GPU row that has already ACQUIRED a GPU is refused at
#         launch (poisoned mid-queue as in gate P) and its GPU is RETURNED to the pool and never
#         double-booked — the two following GPU rows reacquire it, one at a time (a literal doubled
#         free is masked by the per-GPU flock; see the note in the scenario). Fake pool index +
#         stubbed nvidia-smi;
#         no real GPU is ever used. Before cycle 3 every interpreter scenario, gate P included, used
#         cpu rows only, so `_free_gpu` inside launch()'s refusal branch had no committed test.
#   (vii) every guard is killed at its REAL drop site: 12 independent mutations of a COPY of the
#         candidate script (the tracked file itself is never touched), each re-run against the
#         SAME scenario that proves the guard, which must go red. A mutation that cannot be BUILT
#         (its anchor text moved) is a suite FAILURE, never a kill — see expect_red.
#
# Real execution throughout (no --dry-run for the resolution proofs): real tools/safe_run.sh,
# real systemd --user scopes, real trader_env. NEVER touches the real trader_env or the host's
# real `python` — sandboxed roots (run_queue.sh under test + tools/safe_run.sh copied in,
# trader_env SYMLINKED to the real one, read-only reference) give every script variant a correct
# REPO_ROOT without ever writing to the tracked file or the real venv.
#
# Usage: tools/tests/test_run_queue_interpreter.sh   (exit 0 = every acceptance gate passes AND
#                                                       every mutation was killed)
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RQ_LIVE="$REPO_ROOT/tools/deploy/run_queue.sh"
SAFE_RUN_SRC="$REPO_ROOT/tools/safe_run.sh"
[[ -x "$RQ_LIVE" ]] || { echo "FATAL: $RQ_LIVE not executable"; exit 2; }
[[ -x "$SAFE_RUN_SRC" ]] || { echo "FATAL: $SAFE_RUN_SRC not executable"; exit 2; }
command -v systemctl >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1 || {
  echo "SKIP: systemd --user unavailable on this host — this suite needs real scope teardown to"
  echo "      observe real interpreter resolution and cannot meaningfully stub that away."
  exit 0
}
command -v timeout >/dev/null 2>&1 || {
  echo "SKIP: 'timeout' binary unavailable — the venv identity check depends on it (bounded probe)."
  exit 0
}

T="$(mktemp -d "${TMPDIR:-/tmp}/rqitest.XXXXXX")"
SWEEPS_LIVE=()   # runs/<sweep> under the LIVE repo (real-repo scenarios) to clean up
trap 'for s in "${SWEEPS_LIVE[@]:-}"; do [[ -n "$s" ]] && rm -rf "$REPO_ROOT/runs/$s" 2>/dev/null; done; rm -rf "$T"' EXIT

PASS=0; FAIL=0
ok()  { echo "  PASS: $1"; PASS=$((PASS+1)); }
bad() { echo "  FAIL: $1"; FAIL=$((FAIL+1)); }

echo "================================================================================"
echo "fixtures"
echo "================================================================================"

# ---- probe scripts (real files, not inline -c strings, to dodge quoting hell through the
#      manifest -> bash -c -> [nested bash -c] layers) ----
cat > "$T/probe.py" <<'PY'
import sys
print("MAJOR=" + str(sys.version_info.major))
print("PREFIX=" + sys.prefix)
PY

cat > "$T/envdump.py" <<'PY'
import os
for k, v in sorted(os.environ.items()):
    print(k + "=" + v)
PY

# ---- fake python2: a PATH stub, NOT the host's real python. Ignores its args and prints fixed
#      markers, so "did the stub run" is a trivial grep regardless of what args were passed. ----
mkdir -p "$T/py2stub"
cat > "$T/py2stub/python" <<'SH'
#!/bin/bash
echo "MAJOR=2"
echo "PREFIX=/fake/py2/stub/never-a-real-prefix"
exit 0
SH
chmod +x "$T/py2stub/python"

# ---- claim probe: print the launching `bash -c` ancestor's REAL /proc cmdline, so the test can
#      see WHICH interpreter path the queue actually put in the command it executed. argv[1] =
#      repo root, used to recognise the queue's own `cd <root> && ...` wrapper frame. ----
cat > "$T/claimprobe.py" <<'PY'
import os, sys
marker = "cd %s &&" % (sys.argv[1] if len(sys.argv) > 1 else "")
pid, found = os.getpid(), False
for _ in range(6):
    try:
        with open("/proc/%d/stat" % pid) as f:
            st = f.read()
        ppid = int(st[st.rindex(")") + 2:].split()[1])
        if ppid <= 1:
            break
        with open("/proc/%d/cmdline" % ppid, "rb") as f:
            cl = f.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
        if marker in cl:
            print("LAUNCHCMD=" + cl)
            found = True
            break
        pid = ppid
    except Exception as e:
        print("LAUNCHCMD-ERR=%r" % (e,))
        break
if not found:
    print("LAUNCHCMD=<launching bash -c frame not found>")
print("MAJOR=" + str(sys.version_info.major))
PY

# ---- poison fixtures (scenario_poison): a FORGED interpreter and the ordinary passthrough row that
#      installs it mid-queue. The forged "python" ignores its arguments and prints a marker, so
#      "did the queue execute the forgery" is a trivial grep. ----
cat > "$T/forged_python" <<'SH'
#!/bin/bash
echo "FORGED_PAYLOAD_RAN"
exit 0
SH
chmod +x "$T/forged_python"

# $1 = sandbox root, $2 = forged interpreter to install. SAFETY: $root/trader_env is a SYMLINK to
# the real venv; `mv` moves the LINK, never its target, and every later write lands in a fresh
# directory created here. The [[ -L ]] guard makes that structural, not a comment — if the entry is
# ever a real directory this refuses rather than writing into the real venv.
cat > "$T/poison.sh" <<'SH'
#!/bin/bash
root="$1"; forged="$2"
[[ -L "$root/trader_env" ]] || { echo "POISON-ABORT: $root/trader_env is not a symlink — refusing to write"; exit 3; }
mv "$root/trader_env" "$root/trader_env.real" || exit 3
mkdir -p "$root/trader_env/bin" || exit 3
cp "$forged" "$root/trader_env/bin/python" || exit 3
chmod +x "$root/trader_env/bin/python" || exit 3
echo "POISONED"
SH
chmod +x "$T/poison.sh"

# ---- gate PG fixtures: a FAKE nvidia-smi (always reports 0 MiB used, so a fake pool GPU reads as
#      idle) and a mutual-exclusion probe. NO REAL GPU IS EVER TOUCHED: the pool index is 7, which
#      does not exist on this host, and every GPU-row command below is a trivial shell script. ----
mkdir -p "$T/nvstub"
cat > "$T/nvstub/nvidia-smi" <<'SH'
#!/bin/bash
echo 0
SH
chmod +x "$T/nvstub/nvidia-smi"

# mkdir(2) as a mutex: if two GPU rows are ever live at once (a double-booked GPU), the second one
# finds the directory already there and says so in its own log.
cat > "$T/excl.sh" <<'SH'
#!/bin/bash
d="$1"
if mkdir "$d" 2>/dev/null; then
  sleep 2
  rmdir "$d" 2>/dev/null
  echo "EXCLUSIVE_OK"
else
  echo "OVERLAP: another GPU row held the mutex at the same time"
fi
SH
chmod +x "$T/excl.sh"

# ---- gate NCT fixtures: the false-positive/false-negative table for _uncovered_chained_pythons.
#      Kept in a TSV (expected<TAB>command) so the commands need no shell quoting at all — the
#      runner reads them as data, which is the only way to test spellings that are themselves made
#      of quotes, backticks and $( ). ----
cat > "$T/nc_cases.tsv" <<'EOF'
0	python train.py --note "step1 && python notes"
0	python train.py --desc "note ( python inline )"
0	python train.py --desc env python
0	python train.py --desc 'x && python y'
0	python train.py --opt FOO=bar python
0	python train.py --tag python
0	python train.py --out "a python b"
0	python train.py --name python3
1	python a.py && python b.py
1	python a.py ; python b.py
1	python a.py || python b.py
1	python a.py | python b.py
1	python a.py & python b.py
2	python a.py && python b.py && python c.py
1	env python x.py
1	PATH=/stub python x.py
0	bash -c "python x.py"
0	echo $(python x.py)
0	echo `python x.py`
0	ls | xargs python
0	time python x.py
0	sudo python x.py
EOF

# The runner SOURCES the interpreter-contract helper region out of the script under test and calls
# the REAL _uncovered_chained_pythons — no reimplementation, no stub, so a mutation of that function
# in the script shows up here.
cat > "$T/nctable_runner.sh" <<'SH'
#!/usr/bin/env bash
# $1 = extracted helper region  $2 = cases TSV -> one line per case; exit 1 if any case disagrees.
set -uo pipefail
source "$1"
rc=0
while IFS=$'\t' read -r exp cmd; do
  [[ -z "${exp:-}" ]] && continue
  got="$(_uncovered_chained_pythons "$cmd")"
  if [[ "$got" == "$exp" ]]; then
    printf '    n=%s (want %s)  <-  %s\n' "$got" "$exp" "$cmd"
  else
    printf '    MISMATCH n=%s (want %s)  <-  %s\n' "$got" "$exp" "$cmd"
    rc=1
  fi
done < "$2"
exit "$rc"
SH
chmod +x "$T/nctable_runner.sh"

BASE_PIP="$(pip --version 2>&1)"
echo "ambient pip (baseline, unrelated to the interpreter contract): $BASE_PIP"

# ---- sandbox root builder: tools/deploy/run_queue.sh (variant under test) + tools/safe_run.sh
#      (real, copied) + trader_env SYMLINKED to the real one (read-only reference; rm/mv on this
#      entry only ever touches the symlink, never $REPO_ROOT/trader_env's contents). ----
new_sandbox() {  # $1 name -> echoes the sandbox root path
  local root="$T/sbx_$1"
  mkdir -p "$root/tools/deploy" "$root/runs"
  cp "$SAFE_RUN_SRC" "$root/tools/safe_run.sh"; chmod +x "$root/tools/safe_run.sh"
  ln -s "$REPO_ROOT/trader_env" "$root/trader_env"
  echo "$root"
}
install_rq() {  # $1 sandbox root  $2 script content path -> installs it as that sandbox's run_queue.sh
  cp "$2" "$1/tools/deploy/run_queue.sh"; chmod +x "$1/tools/deploy/run_queue.sh"
}

CAND="$T/candidate.sh"
cp "$RQ_LIVE" "$CAND"   # frozen CONTENT source for mutate() below; the tracked file itself is
                        # never touched or executed as a mutation source

deploy_mutant() {  # $1 raw content path -> installs it into a FRESH sandbox, echoes the deployed
                    # tools/deploy/run_queue.sh path (needed by any scenario that does path
                    # arithmetic on $RQ, e.g. `dirname "$RQ"/../..`, and needs a real trader_env
                    # symlink alongside it — scenario_i/ii/chain/claim all require this).
  local raw="$1" root
  # Anchor drift: if mutate() wrote nothing there is nothing to deploy. Echo the (missing) raw path
  # so expect_red fails the suite on it by name, instead of emitting stray cp/chmod noise and then
  # scoring the scenario's inevitable failure as a kill.
  [[ -s "$raw" ]] || { echo "$raw"; return 1; }
  root="$(new_sandbox "mut_$$_$RANDOM")"
  install_rq "$root" "$raw"
  echo "$root/tools/deploy/run_queue.sh"
}

# ---- mutation engine: literal, asserted-unique substring replacement (python, not sed, to avoid
#      regex-escaping the file's own [[ ]] / $ / backtick-heavy bash) ----
mutate() {  # $1 src  $2 dst  $3 old  $4 new -> writes dst ONLY if the target text occurs exactly
            # once. On drift nothing is written, and expect_red turns that into a suite FAILURE.
  rm -f "$2"
  python3 - "$1" "$2" "$3" "$4" <<'PY'
import sys
src, dst, old, new = sys.argv[1:5]
text = open(src).read()
n = text.count(old)
if n != 1:
    print(f"MUTATION-BUILD-ERROR: expected exactly 1 occurrence of the target text, found {n}", file=sys.stderr)
    sys.exit(1)
open(dst, "w").write(text.replace(old, new, 1))
PY
}

mutate_hd() {  # $1 src  $2 dst  $3 file-with-OLD  $4 file-with-NEW. Same literal, assert-unique
               # replacement as mutate(), but the target text comes from a quoted heredoc file —
               # the only sane way to target lines that are themselves made of `'` and `"` (the
               # tokenizer's quote-handling cases). Trailing newline is not part of the target.
  rm -f "$2"
  python3 - "$1" "$2" "$3" "$4" <<'PY'
import sys
src, dst, oldf, newf = sys.argv[1:5]
old = open(oldf).read().rstrip("\n")
new = open(newf).read().rstrip("\n")
text = open(src).read()
n = text.count(old)
if n != 1:
    print(f"MUTATION-BUILD-ERROR: expected exactly 1 occurrence of the target text, found {n}", file=sys.stderr)
    sys.exit(1)
open(dst, "w").write(text.replace(old, new, 1))
PY
}

# ==================================================================================
# scenario functions — each takes the run_queue.sh PATH under test, runs it for real,
# prints diagnostics, and returns 0 (behaved as the contract requires) or a positive
# count of failed checks.
# ==================================================================================

# ---- scenario i: leading bare `python` resolves to the venv via absolute-path substitution,
#      even with a python2 stub shadowing `python` on the ambient PATH. ----
scenario_i() {
  local RQ="$1" root fails=0 SW MAN LOG out
  root="$(cd "$(dirname "$RQ")/../.." && pwd)"
  SW="rqi_$$_$RANDOM"
  MAN="$T/gate_i_$$_$RANDOM.txt"
  printf 'j1 cpu 1G python %s\n' "$T/probe.py" > "$MAN"
  PATH="$T/py2stub:$PATH" RUN_QUEUE_LOCK_DIR="$T/locks_i_$RANDOM" \
    "$RQ" "$MAN" --sweep "$SW" --poll 1 >"$T/i_$$.out" 2>&1
  out="$(cat "$T/i_$$.out")"
  LOG="$root/runs/$SW/j1.log"
  [[ "$root" == "$REPO_ROOT" ]] && SWEEPS_LIVE+=("$SW")
  if [[ -f "$LOG" ]] && grep -q "^PREFIX=$root/trader_env$" "$LOG" && grep -q "^MAJOR=3$" "$LOG"; then
    echo "    j1.log resolved venv python: PREFIX=$root/trader_env, MAJOR=3 (stub NOT used)"
  else
    echo "    MISMATCH: expected PREFIX=$root/trader_env + MAJOR=3 in $LOG; got:"
    sed 's/^/      /' "$LOG" 2>/dev/null || echo "      (no log file)"
    fails=$((fails+1))
  fi
  echo "$out" | grep -q "interpreter=venv" && echo "    launch line says interpreter=venv" \
    || { echo "    MISMATCH: launch line missing interpreter=venv"; fails=$((fails+1)); }
  [[ "$root" != "$REPO_ROOT" ]] && rm -rf "$root/runs/$SW" 2>/dev/null
  return "$fails"
}

# ---- scenario ii: env python / bash -c "python ..." / PATH=<dir> python — all run the AMBIENT
#      interpreter (here: the py2 stub, proving no silent redirect) and are labelled passthrough,
#      including the run-34 halt case (a row that assigns its own PATH). ----
scenario_ii() {
  local RQ="$1" root fails=0 SW MAN out
  root="$(cd "$(dirname "$RQ")/../.." && pwd)"
  SW="rqii_$$_$RANDOM"
  MAN="$T/gate_ii_$$_$RANDOM.txt"
  {
    printf 'e1 cpu 1G env python %s\n' "$T/probe.py"
    printf 'b1 cpu 1G bash -c "python %s"\n' "$T/probe.py"
    printf 'p1 cpu 1G PATH=%s python %s\n' "$T/py2stub" "$T/probe.py"
  } > "$MAN"
  PATH="$T/py2stub:$PATH" RUN_QUEUE_LOCK_DIR="$T/locks_ii_$RANDOM" \
    "$RQ" "$MAN" --sweep "$SW" --poll 1 >"$T/ii_$$.out" 2>&1
  out="$(cat "$T/ii_$$.out")"
  [[ "$root" == "$REPO_ROOT" ]] && SWEEPS_LIVE+=("$SW")
  local job tok
  for job in e1 b1 p1; do
    case "$job" in
      e1) tok="passthrough:env";;
      b1) tok="passthrough:bash";;
      p1) tok="passthrough:PATH=$T/py2stub";;
    esac
    if echo "$out" | grep -F "$job " | grep -q "interpreter=$tok"; then
      echo "    $job labelled interpreter=$tok"
    else
      echo "    MISMATCH: $job expected interpreter=$tok; launch lines were:"
      echo "$out" | grep -F "launch $job " | sed 's/^/      /'
      fails=$((fails+1))
    fi
    # ambient (stub) actually ran, i.e. no venv resolution claim was backed by real redirection
    if [[ -f "$root/runs/$SW/$job.log" ]] && grep -q "^MAJOR=2$" "$root/runs/$SW/$job.log"; then
      echo "    $job.log: ambient/stub ran (MAJOR=2), not the venv — matches the passthrough label"
    else
      echo "    MISMATCH: $job.log did not show the ambient stub running:"
      sed 's/^/      /' "$root/runs/$SW/$job.log" 2>/dev/null || echo "      (no log file)"
      fails=$((fails+1))
    fi
  done
  [[ "$root" != "$REPO_ROOT" ]] && rm -rf "$root/runs/$SW" 2>/dev/null
  return "$fails"
}

# ---- scenario chain: `python a.py && python b.py` — the LEADING python is claimed (gate i), the
#      SECOND is NOT COVERED under D46 (the PATH-prepend that used to reach it is deleted, D46's
#      priced cost). This must be a documented limit, not a silent mislabel: the row is labelled
#      `venv` (its leading token is bare python), and the second half runs whatever is ambient. ----
scenario_chain() {
  local RQ="$1" root fails=0 SW MAN LOG n_venv n_stub
  root="$(cd "$(dirname "$RQ")/../.." && pwd)"
  SW="rqchain_$$_$RANDOM"
  MAN="$T/gate_chain_$$_$RANDOM.txt"
  printf 'c2 cpu 1G python %s && python %s\n' "$T/probe.py" "$T/probe.py" > "$MAN"
  PATH="$T/py2stub:$PATH" RUN_QUEUE_LOCK_DIR="$T/locks_chain_$RANDOM" \
    "$RQ" "$MAN" --sweep "$SW" --poll 1 >"$T/chain_$$.out" 2>&1
  [[ "$root" == "$REPO_ROOT" ]] && SWEEPS_LIVE+=("$SW")
  LOG="$root/runs/$SW/c2.log"
  n_venv="$(grep -c "^PREFIX=$root/trader_env$" "$LOG" 2>/dev/null)"
  n_stub="$(grep -c '^MAJOR=2$' "$LOG" 2>/dev/null)"
  if [[ "${n_venv:-0}" == "1" && "${n_stub:-0}" == "1" ]]; then
    echo "    documented NOT-COVERED limit holds: leading half -> venv (1x), second half -> ambient stub (1x)"
  else
    echo "    MISMATCH: expected exactly 1 venv-prefix line and 1 MAJOR=2 line; got venv=${n_venv:-0} stub=${n_stub:-0}:"
    sed 's/^/      /' "$LOG" 2>/dev/null || echo "      (no log file)"
    fails=$((fails+1))
  fi
  if grep -F "c2 " "$T/chain_$$.out" | grep -q "interpreter=venv"; then
    echo "    row labelled venv (its LEADING token is bare python — the label the code CAN back)"
  else
    echo "    MISMATCH: expected interpreter=venv for the row's own label"
    fails=$((fails+1))
  fi
  [[ "$root" != "$REPO_ROOT" ]] && rm -rf "$root/runs/$SW" 2>/dev/null
  return "$fails"
}

# ---- scenario claim: THE CLAIM IS OBSERVABLE. The command bash actually executes for a `venv`
#      row must literally contain the ABSOLUTE path of the repo venv python, not the bare word
#      `python` re-resolved at exec time. Read out of the real process tree (no dry-run, no stub
#      standing in for the mechanism under test). ----
scenario_claim() {
  local RQ="$1" root fails=0 SW MAN LOG lc
  root="$(cd "$(dirname "$RQ")/../.." && pwd)"
  SW="rqclaim_$$_$RANDOM"
  MAN="$T/gate_claim_$$_$RANDOM.txt"
  # The trailing `; true` stops bash from exec-optimising its own frame away, so the frame the
  # probe reads is guaranteed to still be there while python runs.
  printf 'c1 cpu 1G python %s %s ; true\n' "$T/claimprobe.py" "$root" > "$MAN"
  PATH="$T/py2stub:$PATH" RUN_QUEUE_LOCK_DIR="$T/locks_claim_$RANDOM" \
    "$RQ" "$MAN" --sweep "$SW" --poll 1 >"$T/claim_$$.out" 2>&1
  [[ "$root" == "$REPO_ROOT" ]] && SWEEPS_LIVE+=("$SW")
  LOG="$root/runs/$SW/c1.log"
  lc="$(grep -m1 '^LAUNCHCMD=' "$LOG" 2>/dev/null)"
  if [[ "$lc" == *"$root/trader_env/bin/python $T/claimprobe.py"* ]]; then
    echo "    the command bash actually ran names the ABSOLUTE venv path, not bare 'python':"
    echo "      $lc"
  else
    echo "    MISMATCH: the leading token was not rewritten to the venv's absolute path; the"
    echo "              row's resolution was left to a PATH lookup at exec time. Got:"
    echo "      ${lc:-<no LAUNCHCMD line>}"
    fails=$((fails+1))
  fi
  if grep -q "^MAJOR=3$" "$LOG" 2>/dev/null; then
    echo "    ...and it landed on the venv (MAJOR=3)"
  else
    echo "    MISMATCH: the row did not end up on the venv interpreter:"
    sed 's/^/      /' "$LOG" 2>/dev/null || echo "      (no log file)"
    fails=$((fails+1))
  fi
  [[ "$root" != "$REPO_ROOT" ]] && rm -rf "$root/runs/$SW" 2>/dev/null
  return "$fails"
}

# ---- scenario iii: broken venv (sandboxed copy, trader_env RENAMED — never the real one) refuses
#      only the leading-python row (named job + line), passthrough rows still run. ----
scenario_iii() {
  local RQ_SRC="$1" fails=0
  local root; root="$(new_sandbox "iii_$$_$RANDOM")"
  install_rq "$root" "$RQ_SRC"
  local MAN="$T/gate_iii_$$_$RANDOM.txt"
  printf 'lead cpu 1G python %s\n' "$T/probe.py" > "$MAN"
  printf 'thru cpu 1G true\n' >> "$MAN"

  # --- sanity: venv present -> the leading row runs clean (proves the refusal below is really
  #     about the missing venv, not a broken test rig) ---
  RUN_QUEUE_LOCK_DIR="$T/locks_iii_ok_$RANDOM" "$root/tools/deploy/run_queue.sh" "$MAN" --sweep sane --poll 1 \
    >"$T/iii_sane_$$.out" 2>&1
  if grep -q "^PREFIX=$root/trader_env$" "$root/runs/sane/lead.log" 2>/dev/null; then
    echo "    sanity: venv present -> leading row resolves venv python ($root/trader_env)"
  else
    echo "    MISMATCH (sanity): leading row did not resolve the sandboxed venv:"
    sed 's/^/      /' "$root/runs/sane/lead.log" 2>/dev/null || echo "      (no log)"
    fails=$((fails+1))
  fi

  # --- venv missing: rename the SYMLINK (never the real trader_env behind it) ---
  mv "$root/trader_env" "$root/trader_env.bak"
  RUN_QUEUE_LOCK_DIR="$T/locks_iii_bad_$RANDOM" "$root/tools/deploy/run_queue.sh" "$MAN" --sweep broke --poll 1 \
    >"$T/iii_broke_$$.out" 2>&1
  local out; out="$(cat "$T/iii_broke_$$.out")"

  if echo "$out" | grep -q "REFUSED manifest line 1 job 'lead'"; then
    echo "    leading row refused BY NAME + LINE (line 1, job 'lead')"
  else
    echo "    MISMATCH: expected 'REFUSED manifest line 1 job '\''lead'\''' in output:"
    echo "$out" | sed 's/^/      /'
    fails=$((fails+1))
  fi
  # ...and refused at PARSE time specifically: the row is dropped before scheduling (jobs=1,
  # venv_refused=1 in the banner), not merely stopped later.
  if echo "$out" | grep -q "jobs=1 malformed=0 venv_refused=1"; then
    echo "    refusal happened at PARSE time (banner: jobs=1 venv_refused=1 — row never scheduled)"
  else
    echo "    MISMATCH: expected the banner to report jobs=1 malformed=0 venv_refused=1; got:"
    echo "$out" | grep -m1 '^\[queue\] sweep=' | sed 's/^/      /'
    fails=$((fails+1))
  fi
  if grep -qE '\|\s*lead\s*\|\s*\*\*venv-refused\*\*' "$root/runs/broke/queue_summary.md" 2>/dev/null; then
    echo "    summary: lead -> venv-refused"
  else
    echo "    MISMATCH: queue_summary.md missing a venv-refused row for 'lead':"
    sed 's/^/      /' "$root/runs/broke/queue_summary.md" 2>/dev/null || echo "      (no summary)"
    fails=$((fails+1))
  fi
  if [[ ! -f "$root/runs/broke/lead.log" ]]; then
    echo "    lead: nothing launched (no lead.log)"
  else
    echo "    MISMATCH: lead.log exists — something was launched despite the refusal"
    fails=$((fails+1))
  fi
  if grep -qE '\|\s*thru\s*\|\s*0\s*\|' "$root/runs/broke/queue_summary.md" 2>/dev/null; then
    echo "    thru (passthrough row): still scheduled and exit 0"
  else
    echo "    MISMATCH: thru row missing/non-zero — passthrough must be unaffected by the refusal:"
    sed 's/^/      /' "$root/runs/broke/queue_summary.md" 2>/dev/null || echo "      (no summary)"
    fails=$((fails+1))
  fi

  mv "$root/trader_env.bak" "$root/trader_env" 2>/dev/null || true
  return "$fails"
}

# ---- scenario iv: the queue makes NO env/PATH change at all — a claimed row's environment is
#      byte-identical to a passthrough row's. Normalises out the two expected per-invocation
#      artifacts that have nothing to do with the interpreter contract: systemd-run's own
#      INVOCATION_ID (fresh per scope) and safe_run's banner line (names the job). ----
scenario_iv() {
  local RQ="$1" root fails=0 SW MAN out
  root="$(cd "$(dirname "$RQ")/../.." && pwd)"
  SW="rqiv_$$_$RANDOM"
  MAN="$T/gate_iv_$$_$RANDOM.txt"
  {
    printf 'claimed cpu 1G python %s\n' "$T/envdump.py"   # leading python -> absolute-path rewrite
    printf 'plain cpu 1G env\n'                            # leading 'env' -> passthrough, untouched
  } > "$MAN"
  RUN_QUEUE_LOCK_DIR="$T/locks_iv_$RANDOM" "$RQ" "$MAN" --sweep "$SW" --poll 1 >"$T/iv_$$.out" 2>&1
  [[ "$root" == "$REPO_ROOT" ]] && SWEEPS_LIVE+=("$SW")

  local norm_claimed norm_plain
  # Normalised out, neither part of the interpreter contract: INVOCATION_ID (fresh per systemd
  # scope), the safe_run banner line (names the job), and `_` (bash's own "last command" special
  # var — it trivially differs because the two rows run different BINARIES by design; that is the
  # intended difference, not evidence of an injected PATH/env change).
  norm_claimed="$(sed -E '/^INVOCATION_ID=/d; /^\[safe_run\]/d; /^_=/d' "$root/runs/$SW/claimed.log" 2>/dev/null | sort)"
  norm_plain="$(sed -E '/^INVOCATION_ID=/d; /^\[safe_run\]/d; /^_=/d' "$root/runs/$SW/plain.log" 2>/dev/null | sort)"
  if [[ -n "$norm_claimed" && "$norm_claimed" == "$norm_plain" ]]; then
    echo "    env diff empty: a claimed row's environment matches a passthrough row's exactly"
    echo "      (modulo INVOCATION_ID, the safe_run banner line, and \$_ — none part of this contract)"
  else
    echo "    MISMATCH: claimed-row env diverged from the passthrough-row env:"
    diff <(echo "$norm_claimed") <(echo "$norm_plain") | sed 's/^/      /'
    fails=$((fails+1))
  fi
  [[ "$root" != "$REPO_ROOT" ]] && rm -rf "$root/runs/$SW" 2>/dev/null
  return "$fails"
}

# ---- scenario v: a manifest with NO leading-python row behaves like pre-891 — no venv:/REFUSED
#      banner, same summary/scheduling as the actual (unmodified) HEAD script, modulo the new
#      per-row interpreter= label (an intentional, minimal addition). ----
OLD_SANDBOX=""
scenario_v() {
  local RQ_NEW="$1" fails=0
  if [[ -z "$OLD_SANDBOX" ]]; then
    OLD_SANDBOX="$(new_sandbox "old_$$")"
    git -C "$REPO_ROOT" show HEAD:tools/deploy/run_queue.sh > "$T/head_run_queue.sh"
    install_rq "$OLD_SANDBOX" "$T/head_run_queue.sh"
  fi
  local new_root; new_root="$(new_sandbox "new_$$_$RANDOM")"
  install_rq "$new_root" "$RQ_NEW"

  # A SINGLE job: two+ concurrent CPU jobs reap in whatever order bash's associative-array
  # iteration happens to produce (pid-hash dependent) — pre-existing nondeterminism in
  # run_queue.sh itself, unrelated to this task, that would make an A/B stdout diff flaky
  # regardless of any interpreter-contract change. One job still exercises the full
  # parse -> schedule -> summary path.
  local MAN="$T/gate_v_$$_$RANDOM.txt"
  printf 'a1 cpu 1G true\n' > "$MAN"

  RUN_QUEUE_LOCK_DIR="$T/locks_v_old_$RANDOM" "$OLD_SANDBOX/tools/deploy/run_queue.sh" "$MAN" --sweep v --dry-run \
    >"$T/v_old_$$.out" 2>&1
  RUN_QUEUE_LOCK_DIR="$T/locks_v_new_$RANDOM" "$new_root/tools/deploy/run_queue.sh" "$MAN" --sweep v --dry-run \
    >"$T/v_new_$$.out" 2>&1

  if grep -qE 'venv:|REFUSED|venv_refused' "$T/v_new_$$.out"; then
    echo "    MISMATCH: candidate printed an interpreter-contract banner line for an all-passthrough manifest:"
    grep -E 'venv:|REFUSED|venv_refused' "$T/v_new_$$.out" | sed 's/^/      /'
    fails=$((fails+1))
  else
    echo "    no venv:/REFUSED banner line for a manifest with no leading-python row"
  fi

  local norm_old norm_new
  norm_old="$(sed -E "s#$OLD_SANDBOX##g; s/ interpreter=passthrough:[^ ]*//; s/^- (started|finished):.*/- \1: <ts>/" "$T/v_old_$$.out")"
  norm_new="$(sed -E "s#$new_root##g; s/ interpreter=passthrough:[^ ]*//; s/^- (started|finished):.*/- \1: <ts>/" "$T/v_new_$$.out")"
  if [[ "$norm_old" == "$norm_new" ]]; then
    echo "    HEAD-vs-candidate diff is confined EXACTLY to the interpreter= label (byte-identical otherwise)"
  else
    echo "    MISMATCH: HEAD vs candidate diverge beyond the interpreter= label:"
    diff <(echo "$norm_old") <(echo "$norm_new") | sed 's/^/      /'
    fails=$((fails+1))
  fi
  rm -rf "$new_root"
  return "$fails"
}

# ---- scenario b1: the venv python is a program that ignores its arguments and exits 0
#      (/bin/true). An rc-only probe blesses it; an IDENTITY check must refuse it BY NAME. The
#      sandbox venv here is FABRICATED (a real directory, never a symlink to the real venv), so
#      nothing in this scenario can reach the real venv. ----
scenario_b1() {
  local RQ_SRC="$1" fails=0 root MAN out
  root="$T/sbx_b1_$$_$RANDOM"
  mkdir -p "$root/tools/deploy" "$root/runs" "$root/trader_env/bin"
  cp "$SAFE_RUN_SRC" "$root/tools/safe_run.sh"; chmod +x "$root/tools/safe_run.sh"
  ln -s /bin/true "$root/trader_env/bin/python"      # executable, exits 0, is NOT Python
  install_rq "$root" "$RQ_SRC"
  MAN="$T/gate_b1_$$_$RANDOM.txt"
  printf 'lead cpu 1G python %s\n' "$T/probe.py" > "$MAN"
  printf 'thru cpu 1G true\n' >> "$MAN"
  RUN_QUEUE_LOCK_DIR="$T/locks_b1_$RANDOM" "$root/tools/deploy/run_queue.sh" "$MAN" --sweep b1 --poll 1 \
    >"$T/b1_$$.out" 2>&1
  out="$(cat "$T/b1_$$.out")"

  if echo "$out" | grep -q "venv: UNUSABLE"; then
    echo "    /bin/true rejected as the venv interpreter: $(echo "$out" | grep -m1 'venv: UNUSABLE')"
  else
    echo "    MISMATCH: the queue accepted /bin/true as the repo venv python. Output was:"
    echo "$out" | sed 's/^/      /'
    fails=$((fails+1))
  fi
  if echo "$out" | grep -q "REFUSED manifest line 1 job 'lead'"; then
    echo "    leading row refused BY NAME + LINE"
  else
    echo "    MISMATCH: expected the leading row to be REFUSED by name+line"
    fails=$((fails+1))
  fi
  if echo "$out" | grep -q "interpreter=venv"; then
    echo "    MISMATCH: a venv resolution claim was printed for a non-Python interpreter"
    fails=$((fails+1))
  else
    echo "    no venv claim printed anywhere (the label the code cannot back is never emitted)"
  fi
  if [[ -f "$root/runs/b1/lead.log" ]]; then
    echo "    MISMATCH: lead.log exists — the row was launched against a non-Python interpreter:"
    sed 's/^/      /' "$root/runs/b1/lead.log"
    fails=$((fails+1))
  else
    echo "    lead: nothing launched (no lead.log)"
  fi
  if grep -qE '\|\s*thru\s*\|\s*0\s*\|' "$root/runs/b1/queue_summary.md" 2>/dev/null; then
    echo "    thru (passthrough row): still scheduled and exit 0"
  else
    echo "    MISMATCH: passthrough row must be unaffected by the refusal:"
    sed 's/^/      /' "$root/runs/b1/queue_summary.md" 2>/dev/null || echo "      (no summary)"
    fails=$((fails+1))
  fi
  rm -rf "$root"
  return "$fails"
}

# ---- scenario poison: POISON-AFTER-CHECK — the run-36 BLOCKER, reproduced. The parse-time venv
#      check PASSES; then an ordinary `passthrough` row breaks the interpreter mid-queue; then a
#      LATER claimed row must be REFUSED at its own launch rather than executed under an
#      `interpreter=venv` label the code can no longer back. This is the check-vs-execute window: it
#      is real because trader_env/bin/python is same-user-writable (no privilege boundary — every
#      job runs under the same uid via systemd-run --user --scope), so the parse-time check can be
#      hours stale for a row that waited out a GPU queue.
#      SAFETY: everything happens inside a sandbox root whose `trader_env` is a SYMLINK to the real
#      venv. The poison step `mv`s the SYMLINK aside (never its target) behind a `[[ -L ]]` guard and
#      builds a fake directory in its place, so the real venv is never written to.
#      `--max-cpu 1` serialises the rows, so the poison provably lands BEFORE the victim launches.
scenario_poison() {
  local RQ_SRC="$1" fails=0 root MAN out
  root="$(new_sandbox "poison_$$_$RANDOM")"
  install_rq "$root" "$RQ_SRC"
  MAN="$T/gate_poison_$$_$RANDOM.txt"
  {
    printf 'poison cpu 1G bash %s %s %s\n' "$T/poison.sh" "$root" "$T/forged_python"
    printf 'victim cpu 1G python %s\n' "$T/probe.py"
  } > "$MAN"
  RUN_QUEUE_LOCK_DIR="$T/locks_poison_$RANDOM" "$root/tools/deploy/run_queue.sh" "$MAN" \
    --sweep poison --poll 1 --max-cpu 1 >"$T/poison_$$.out" 2>&1
  out="$(cat "$T/poison_$$.out")"

  # rig sanity 1: the parse-time check must have PASSED (otherwise this proves nothing about the
  # launch-time re-probe — it would just be gate iii again).
  if echo "$out" | grep -q "venv: .*identity OK: python3" && ! echo "$out" | grep -q "REFUSED manifest line"; then
    echo "    parse-time check PASSED (the venv was healthy when the manifest was parsed)"
  else
    echo "    MISMATCH (rig): the parse-time check did not pass, so this scenario is not testing the"
    echo "                    deferred window. Output was:"
    echo "$out" | sed 's/^/      /'
    fails=$((fails+1))
  fi
  # rig sanity 2: the poison actually landed, mid-queue, from an ordinary passthrough row.
  if grep -q '^POISONED$' "$root/runs/poison/poison.log" 2>/dev/null \
     && grep -q 'FORGED_PAYLOAD_RAN' "$root/trader_env/bin/python" 2>/dev/null; then
    echo "    poison landed: a plain passthrough row replaced trader_env/bin/python after the check"
  else
    echo "    MISMATCH (rig): the poison row did not install the forged interpreter:"
    sed 's/^/      /' "$root/runs/poison/poison.log" 2>/dev/null || echo "      (no poison.log)"
    fails=$((fails+1))
  fi

  # the guard itself: the claimed row is refused AT LAUNCH, by name.
  if echo "$out" | grep -q "REFUSED at launch: job 'victim'"; then
    echo "    victim REFUSED at launch by name: $(echo "$out" | grep -m1 "REFUSED at launch" | cut -c1-118)..."
  else
    echo "    MISMATCH: the claimed row was NOT refused at launch — the parse-time check was the only"
    echo "              check, so the forged interpreter got executed under an interpreter=venv label."
    echo "$out" | grep -E "launch victim|REFUSED" | sed 's/^/      /'
    fails=$((fails+1))
  fi
  # ...and nothing of the forgery ran: no victim log at all, and the payload marker nowhere.
  if [[ ! -f "$root/runs/poison/victim.log" ]]; then
    echo "    victim: nothing launched (no victim.log)"
  else
    echo "    MISMATCH: victim.log exists — the row was launched against the forged interpreter:"
    sed 's/^/      /' "$root/runs/poison/victim.log"
    fails=$((fails+1))
  fi
  if grep -rq 'FORGED_PAYLOAD_RAN' "$root/runs/poison/" 2>/dev/null; then
    echo "    MISMATCH: the forged payload EXECUTED (marker found under runs/poison/) — exploit live:"
    grep -rl 'FORGED_PAYLOAD_RAN' "$root/runs/poison/" | sed 's/^/      /'
    fails=$((fails+1))
  else
    echo "    the forged payload never executed (marker absent from every job log)"
  fi
  if grep -qE '\|\s*victim\s*\|\s*\*\*venv-refused\*\*' "$root/runs/poison/queue_summary.md" 2>/dev/null; then
    echo "    summary: victim -> venv-refused (durable record of the refusal)"
  else
    echo "    MISMATCH: queue_summary.md missing a venv-refused row for 'victim':"
    sed 's/^/      /' "$root/runs/poison/queue_summary.md" 2>/dev/null || echo "      (no summary)"
    fails=$((fails+1))
  fi

  rm -f "$root/trader_env.real"        # the symlink to the real venv — removed, never followed
  rm -rf "$root"
  return "$fails"
}

# ---- scenario ncwarn: an ACCEPTED limit must still be VISIBLE. A row with a chained bare python
#      (`python a && python b` — the runs/860_codistill/manifest_860.txt shape, whose second half is
#      the eval that reads the corridor result) is warned by name+line, marked on its launch line and
#      recorded in the summary — while still being SCHEDULED (D46 priced the limit as accepted).
#      The second row is the FALSE-POSITIVE guard: `--tag python` is an argument, not an invocation,
#      and must produce no warning and no marker. ----
scenario_ncwarn() {
  local RQ="$1" root fails=0 SW MAN out
  root="$(cd "$(dirname "$RQ")/../.." && pwd)"
  SW="rqnc_$$_$RANDOM"
  MAN="$T/gate_nc_$$_$RANDOM.txt"
  {
    printf 'nc1 cpu 1G python %s && python %s\n' "$T/probe.py" "$T/probe.py"
    printf 'nc0 cpu 1G python %s --tag python\n' "$T/probe.py"
    # FALSE-POSITIVE rows (cycle-3 MAJOR): each of these merely READS the word python in an
    # argument. The detector's own header comment promises it never fires on those, and the count
    # writes a DURABLE entry into queue_summary.md's NOT-COVERED block, so a false entry is a
    # false operator-facing record. fp1/fp2 need quote-aware tokenisation; fp3 is UNQUOTED and
    # needs the `env`-must-itself-be-in-command-position rule.
    printf 'fp1 cpu 1G python %s --note "step1 && python notes"\n' "$T/probe.py"
    printf 'fp2 cpu 1G python %s --desc "note ( python inline )"\n' "$T/probe.py"
    printf 'fp3 cpu 1G python %s --desc env python\n' "$T/probe.py"
  } > "$MAN"
  # py2stub on PATH so the CHAINED half of nc1 (which nothing rewrites — that is the point) resolves
  # to a deterministic ambient interpreter. Without it the second `python` is whatever pyenv resolves
  # for the job's cwd, which differs between the live repo root and a mutant sandbox (rc 127 there) —
  # an environment artifact that has nothing to do with the detector under test.
  PATH="$T/py2stub:$PATH" RUN_QUEUE_LOCK_DIR="$T/locks_nc_$RANDOM" "$RQ" "$MAN" --sweep "$SW" --poll 1 >"$T/nc_$$.out" 2>&1
  out="$(cat "$T/nc_$$.out")"
  [[ "$root" == "$REPO_ROOT" ]] && SWEEPS_LIVE+=("$SW")

  if echo "$out" | grep -q "NOT-COVERED manifest line 1 job 'nc1'"; then
    echo "    nc1 WARNED at parse time by name + manifest line"
  else
    echo "    MISMATCH: no NOT-COVERED warning naming nc1 (line 1) — a landmine row is indistinguishable"
    echo "              from a harmless one. Output was:"
    echo "$out" | sed 's/^/      /'
    fails=$((fails+1))
  fi
  if echo "$out" | grep -F "launch nc1 " | grep -q "not-covered-chained-python=1"; then
    echo "    nc1 launch line marked not-covered-chained-python=1"
  else
    echo "    MISMATCH: nc1's launch line carries no NOT-COVERED marker:"
    echo "$out" | grep -F "launch nc1 " | sed 's/^/      /'
    fails=$((fails+1))
  fi
  if grep -q '^- `nc1` (1 chained)' "$root/runs/$SW/queue_summary.md" 2>/dev/null; then
    echo "    summary records nc1 under 'interpreter: NOT COVERED' (durable, survives the tmux buffer)"
  else
    echo "    MISMATCH: queue_summary.md has no NOT COVERED entry for nc1:"
    sed 's/^/      /' "$root/runs/$SW/queue_summary.md" 2>/dev/null || echo "      (no summary)"
    fails=$((fails+1))
  fi
  # still SCHEDULED, not refused (D46's accepted limit — the warning must not become a refusal)
  if grep -qE '\|\s*nc1\s*\|\s*0\s*\|' "$root/runs/$SW/queue_summary.md" 2>/dev/null; then
    echo "    nc1 still ran (rc=0): the warning is a warning, not a refusal of a permitted row"
  else
    echo "    MISMATCH: nc1 did not run cleanly — the accepted limit must not turn into a refusal:"
    sed 's/^/      /' "$root/runs/$SW/queue_summary.md" 2>/dev/null || echo "      (no summary)"
    fails=$((fails+1))
  fi
  # false-positive guard: `--tag python` and the three cycle-3 shapes are all ARGUMENTS. Each must
  # produce NO parse-time warning, NO launch-line marker and NO summary entry — and still run.
  local fp
  for fp in nc0 fp1 fp2 fp3; do
    if echo "$out" | grep -q "job '$fp'" || echo "$out" | grep -F "launch $fp " | grep -q "not-covered"; then
      echo "    MISMATCH: $fp (an argument that merely reads \`python\`) was flagged — the detector over-fires:"
      echo "$out" | grep -E "job '$fp'|launch $fp " | sed 's/^/      /'
      fails=$((fails+1))
    elif grep -q "^- \`$fp\` (" "$root/runs/$SW/queue_summary.md" 2>/dev/null; then
      # `- <job> (<n> chained): ...` is the NOT-COVERED block's format; `- <job> -> ...` is the
      # rewritten-command block's and is expected for every one of these rows.
      echo "    MISMATCH: $fp got a durable NOT-COVERED entry in queue_summary.md for an argument:"
      grep -n "^- \`$fp\` (" "$root/runs/$SW/queue_summary.md" | sed 's/^/      /'
      fails=$((fails+1))
    elif ! grep -qE "\|\s*$fp\s*\|\s*0\s*\|" "$root/runs/$SW/queue_summary.md" 2>/dev/null; then
      echo "    MISMATCH: $fp did not run cleanly (rc must be 0 — these rows are ordinary rows):"
      sed 's/^/      /' "$root/runs/$SW/queue_summary.md" 2>/dev/null || echo "      (no summary)"
      fails=$((fails+1))
    else
      echo "    $fp NOT flagged (no warning, no marker, no summary entry) and ran rc=0"
    fi
  done
  # ...and the durable record contains EXACTLY the one true positive, nothing else.
  local nc_entries
  nc_entries="$(grep -c '^- `' "$root/runs/$SW/queue_summary.md" 2>/dev/null)"
  if [[ "${nc_entries:-0}" == "6" ]]; then
    # 5 claimed-row entries (nc1/nc0/fp1/fp2/fp3, all leading bare python) + 1 NOT-COVERED entry
    echo "    summary bullet count 6 = 5 rewritten-command entries + exactly 1 NOT-COVERED entry"
  else
    echo "    MISMATCH: expected 6 summary bullets (5 rewritten + 1 NOT-COVERED); got ${nc_entries:-0}:"
    grep -n '^- `' "$root/runs/$SW/queue_summary.md" 2>/dev/null | sed 's/^/      /'
    fails=$((fails+1))
  fi
  [[ "$root" != "$REPO_ROOT" ]] && rm -rf "$root/runs/$SW" 2>/dev/null
  return "$fails"
}

# ---- scenario nctable: the detector's header comment must be TRUE OF THE CODE. Runs the REAL
#      _uncovered_chained_pythons (sourced out of the script under test, not reimplemented) over a
#      table of 22 spellings: the cycle-3 false positives that contradicted the comment, the
#      arguments that were already correct, the true positives, and the ACCEPTED false negatives the
#      comment now names explicitly (quotes / substitutions / backticks / unmodelled launchers). The
#      table is the artifact that keeps the comment honest — a durable operator-facing record
#      (queue_summary.md's NOT-COVERED block) is only worth having if it does not lie. ----
scenario_nctable() {
  local RQ="$1" fails=0 helpers out
  helpers="$T/helpers_$$_$RANDOM.sh"
  awk '/^# ---- interpreter contract helpers/{f=1} /^PROBE_TIMEOUT=/{f=0} f' "$RQ" > "$helpers"
  if ! grep -q '^_uncovered_chained_pythons()' "$helpers"; then
    echo "    RIG ERROR: could not extract the interpreter-helper region from $RQ"
    return 1
  fi
  out="$(bash "$T/nctable_runner.sh" "$helpers" "$T/nc_cases.tsv" 2>&1)"
  local rc=$?
  echo "$out"
  if (( rc == 0 )); then
    echo "    all 22 spellings match: no false positive on an argument, true positives intact,"
    echo "    accepted false negatives exactly as the header documents them"
  else
    echo "    MISMATCH: the detector disagrees with its own documented contract (see MISMATCH lines)"
    fails=$((fails+1))
  fi
  rm -f "$helpers"
  return "$fails"
}

# ---- scenario poison_gpu: the GPU REFUSAL PATH, which no committed test reached before (every
#      interpreter scenario, gate P included, used cpu rows; the scheduler suite's GPU scenarios
#      never involve a bare-python leading token). Same poison-after-check window as gate P, but the
#      victim is a GPU row that has ALREADY ACQUIRED a GPU when launch() refuses it — so the
#      _free_gpu() call in the refusal branch is what stands between one bad row and a permanently
#      leaked GPU. Single-GPU pool, so the following GPU rows can only run if that GPU came back.
#      NO REAL GPU IS USED: pool index 7 does not exist on this host, nvidia-smi is a stub that
#      always reports 0 MiB, and every GPU-row command is a trivial shell script. The single-GPU
#      pool also SERIALISES the rows, which is what makes the poison provably land before the
#      victim launches (the same role --max-cpu 1 plays in gate P). ----
scenario_poison_gpu() {
  local RQ_SRC="$1" fails=0 root MAN out mutex
  root="$(new_sandbox "pgpu_$$_$RANDOM")"
  install_rq "$root" "$RQ_SRC"
  MAN="$T/gate_pgpu_$$_$RANDOM.txt"
  mutex="$T/pgpu_mutex_$$_$RANDOM"
  {
    printf 'poison gpu 1G bash %s %s %s\n' "$T/poison.sh" "$root" "$T/forged_python"
    printf 'victim gpu 1G python %s\n' "$T/probe.py"
    printf 'after1 gpu 1G bash %s %s\n' "$T/excl.sh" "$mutex"
    printf 'after2 gpu 1G bash %s %s\n' "$T/excl.sh" "$mutex"
  } > "$MAN"
  PATH="$T/nvstub:$PATH" RUN_QUEUE_LOCK_DIR="$T/locks_pgpu_$RANDOM" \
    "$root/tools/deploy/run_queue.sh" "$MAN" --sweep pgpu --gpus 7 --poll 1 --acquire-timeout 15 \
    >"$T/pgpu_$$.out" 2>&1
  out="$(cat "$T/pgpu_$$.out")"

  # rig 1: the parse-time check PASSED (else this is gate iii, not the deferred window).
  if echo "$out" | grep -q "venv: .*identity OK: python3" && ! echo "$out" | grep -q "REFUSED manifest line"; then
    echo "    parse-time check PASSED (venv healthy when the manifest was parsed)"
  else
    echo "    MISMATCH (rig): parse-time check did not pass — not testing the deferred window:"
    echo "$out" | sed 's/^/      /'
    fails=$((fails+1))
  fi
  # rig 2: the poison landed mid-queue, from an ordinary passthrough GPU row.
  if grep -q '^POISONED$' "$root/runs/pgpu/poison.log" 2>/dev/null \
     && grep -q 'FORGED_PAYLOAD_RAN' "$root/trader_env/bin/python" 2>/dev/null; then
    echo "    poison landed: a passthrough GPU row replaced trader_env/bin/python after the check"
  else
    echo "    MISMATCH (rig): the poison row did not install the forged interpreter:"
    sed 's/^/      /' "$root/runs/pgpu/poison.log" 2>/dev/null || echo "      (no poison.log)"
    fails=$((fails+1))
  fi
  # rig 3: the victim really did HOLD a GPU when it was refused — it was never starved of one.
  if echo "$out" | grep -q "ACQUIRE TIMEOUT.*'victim'"; then
    echo "    MISMATCH (rig): victim never acquired a GPU, so this does not exercise the refusal"
    echo "                    branch's _free_gpu at all:"
    echo "$out" | grep -F "ACQUIRE TIMEOUT" | sed 's/^/      /'
    fails=$((fails+1))
  else
    echo "    victim acquired the pool's only GPU (no acquire-timeout for it)"
  fi

  # the guard: refused AT LAUNCH, by name, nothing launched, nothing of the forgery ran.
  if echo "$out" | grep -q "REFUSED at launch: job 'victim'"; then
    echo "    victim REFUSED at launch by name (GPU row, post-acquire)"
  else
    echo "    MISMATCH: the claimed GPU row was not refused at launch:"
    echo "$out" | grep -E "launch victim|REFUSED" | sed 's/^/      /'
    fails=$((fails+1))
  fi
  if [[ -f "$root/runs/pgpu/victim.log" ]] || grep -rq 'FORGED_PAYLOAD_RAN' "$root/runs/pgpu/" 2>/dev/null; then
    echo "    MISMATCH: the victim launched against the forged interpreter:"
    sed 's/^/      /' "$root/runs/pgpu/victim.log" 2>/dev/null
    fails=$((fails+1))
  else
    echo "    victim: nothing launched, forged payload never executed"
  fi
  if grep -qE '\|\s*victim\s*\|\s*\*\*venv-refused\*\*' "$root/runs/pgpu/queue_summary.md" 2>/dev/null; then
    echo "    summary: victim -> venv-refused (durable record)"
  else
    echo "    MISMATCH: queue_summary.md missing a venv-refused row for 'victim':"
    sed 's/^/      /' "$root/runs/pgpu/queue_summary.md" 2>/dev/null || echo "      (no summary)"
    fails=$((fails+1))
  fi

  # THE GPU CAME BACK: with a one-GPU pool, the two following GPU rows can only run if the refused
  # row's GPU was returned to the pool AND its advisory lock dropped.
  local a
  for a in after1 after2; do
    if echo "$out" | grep -F "launch $a " | grep -q "gpu=7" \
       && grep -qE "\|\s*$a\s*\|\s*0\s*\|" "$root/runs/pgpu/queue_summary.md" 2>/dev/null; then
      echo "    $a reacquired GPU 7 and completed rc=0"
    else
      echo "    MISMATCH: $a did not reacquire GPU 7 / did not complete — the refused row leaked it:"
      echo "$out" | grep -E "launch $a |ACQUIRE TIMEOUT|SKIP $a" | sed 's/^/      /'
      fails=$((fails+1))
    fi
  done
  # ...and never handed out twice: the GPU is never held by two live rows (mkdir mutex inside the
  # job), and no row ever had to wait out the acquire timeout.
  # MEASURED, so this claim is not stronger than the evidence: a LITERAL duplicate `_free_gpu` call
  # in the refusal branch is NOT observable from outside. A duplicated pool entry is skipped by the
  # per-GPU advisory lock (flock(2) denies a second lock on the same file even from the same process,
  # via a different fd), so the queue never double-books on it. What these two checks therefore
  # prove is "the GPU came back and was never double-booked"; the mutation that goes red is the
  # MISSING free (M11), not a doubled one.
  if grep -rq 'OVERLAP' "$root/runs/pgpu/" 2>/dev/null; then
    echo "    MISMATCH: two GPU rows were live on the same GPU at once (pool double-counted it):"
    grep -rn 'OVERLAP' "$root/runs/pgpu/" | sed 's/^/      /'
    fails=$((fails+1))
  elif [[ "$(grep -rl 'EXCLUSIVE_OK' "$root/runs/pgpu/" 2>/dev/null | wc -l)" != "2" ]]; then
    echo "    MISMATCH: expected both following GPU rows to take the mutex exclusively; got:"
    grep -rn 'EXCLUSIVE_OK\|OVERLAP' "$root/runs/pgpu/" 2>/dev/null | sed 's/^/      /'
    fails=$((fails+1))
  else
    echo "    GPU 7 served the two following rows one at a time (no overlap): came back, never double-booked"
  fi
  if echo "$out" | grep -q "ACQUIRE TIMEOUT"; then
    echo "    MISMATCH: some row waited out the acquire timeout — the pool lost its GPU:"
    echo "$out" | grep -F "ACQUIRE TIMEOUT" | sed 's/^/      /'
    fails=$((fails+1))
  else
    echo "    no acquire-timeout anywhere: the pool never ran dry"
  fi

  rm -rf "$mutex" 2>/dev/null
  rm -f "$root/trader_env.real"        # the symlink to the real venv — removed, never followed
  rm -rf "$root"
  return "$fails"
}

# ==================================================================================
# acceptance driver — every scenario against the TRACKED candidate (never mutated)
# ==================================================================================
echo "================================================================================"
echo "acceptance gates (candidate = tools/deploy/run_queue.sh, unmutated)"
echo "================================================================================"

run_scenario() {  # $1 name  $2 fn  $3 RQ path
  local name="$1" fn="$2" rq="$3" rc
  echo "-- $name --"
  "$fn" "$rq"; rc=$?
  if [[ "$rc" -eq 0 ]]; then ok "$name"; else bad "$name ($rc check(s) failed)"; fi
}

# scenario_i/ii/chain/claim/iv run REAL manifests and do path arithmetic on $RQ to find the
# real trader_env symlink alongside it — they run directly against the TRACKED, deployed
# $RQ_LIVE (never mutated; read-only manifests, cleaned up via SWEEPS_LIVE). scenario_iii/b1/v
# instead take a CONTENT source and build their own internal sandbox, so $CAND (a frozen copy)
# is the right argument for those.
run_scenario "gate i   (leading bare python -> venv, PATH-stub shadowed)"        scenario_i      "$RQ_LIVE"
run_scenario "gate ii  (env/bash-c/PATH= -> passthrough, ambient runs)"          scenario_ii     "$RQ_LIVE"
run_scenario "gate NC  (chained python: leading claimed, second NOT COVERED)"    scenario_chain  "$RQ_LIVE"
run_scenario "gate cl  (the claim is observable: real /proc cmdline)"            scenario_claim  "$RQ_LIVE"
run_scenario "gate iii (broken venv: named refusal, passthrough unaffected)"     scenario_iii    "$CAND"
run_scenario "gate iv  (env diff empty: no PATH/env manipulation at all)"        scenario_iv     "$RQ_LIVE"
run_scenario "gate v   (no-leading-python manifest: byte-identical to HEAD)"     scenario_v      "$CAND"
run_scenario "gate vi  (identity, not exit status: /bin/true refused by name)"   scenario_b1     "$CAND"
run_scenario "gate P   (poison-after-check: claimed row refused AT LAUNCH)"      scenario_poison "$CAND"
run_scenario "gate NCW (chained python: warned+marked+recorded, still runs)"     scenario_ncwarn "$RQ_LIVE"
run_scenario "gate NCT (detector table: no false positive on an argument)"       scenario_nctable "$RQ_LIVE"
run_scenario "gate PG  (GPU refusal path: GPU returned, next rows reacquire it)" scenario_poison_gpu "$CAND"

# ==================================================================================
# gate (vii): every guard killed at its REAL drop site — 12 independent mutations of a COPY
# of the candidate script, each re-run against the SAME scenario that proves the guard. A mutation
# whose anchor no longer exists is reported as a BUILD FAILURE and fails the suite (it proves
# nothing); it is never scored as a kill.
# ==================================================================================
echo "================================================================================"
echo "gate (vii): mutation kills (each mutant is an independent copy; CAND is never touched)"
echo "================================================================================"

expect_red() {  # $1 mutation name  $2 mutant path  $3 scenario fn  -> the scenario must FAIL
                # THREE outcomes, not two. The mutant must EXIST first: if `mutate` could not find
                # its anchor (an edit moved the target text), no mutant is written, and running a
                # scenario against a non-existent script fails for a reason that has NOTHING to do
                # with the guard — which used to be scored as a KILL. That is exactly the meaningless
                # green this suite exists to prevent (found in cycle 3: the detector rewrite moved
                # M8's anchor, M8's mutant was never built, and the suite still reported 18/18). So a
                # BUILD failure is now a loud suite FAILURE in its own right: an unbuildable mutation
                # proves nothing, and the guard it was supposed to cover is UNPROVEN, not proven.
  local name="$1" mutant="$2" fn="$3" rc
  if [[ ! -s "$mutant" ]]; then
    bad "MUTATION BUILD FAILED — ANCHOR DRIFT: $name"
    echo "      no mutant script at '$mutant' (mutate() found != 1 occurrence of its target text, or"
    echo "      the deploy step had nothing to copy). The guard is UNPROVEN — re-anchor the mutation"
    echo "      to the current code. This is NOT a kill: a scenario 'failing' against a missing"
    echo "      script says nothing about the guard."
    return 0
  fi
  "$fn" "$mutant" >"$T/mut_$$.out" 2>&1; rc=$?
  if [[ "$rc" -ne 0 ]]; then
    ok "MUTATION KILLED: $name (scenario failed as required, $rc check(s))"
  else
    bad "MUTATION SURVIVED: $name (scenario passed against a broken script — the guard is not load-bearing)"
    sed 's/^/      /' "$T/mut_$$.out"
  fi
}

# M1 — classify EVERY row as `leading` (drop the passthrough/leading distinction). A row whose
#      leading token is NOT python/python3 (e.g. `env`) would then have that unrelated token
#      rewritten to the venv path -> broken command, caught by gate ii.
M1_RAW="$T/mutant_m1_raw.sh"
mutate "$CAND" "$M1_RAW" \
  'need="passthrough"' \
  'need="leading"'
M1="$(deploy_mutant "$M1_RAW")"
expect_red "M1 over-broad classification (every row claimed)" "$M1" scenario_ii

# M2 — the identity check always reports OK, regardless of what the interpreter actually is.
#      Lets /bin/true through -> caught by gate vi.
M2="$T/mutant_m2.sh"
mutate "$CAND" "$M2" \
  '  local interp="$1" out rc nonce want_root' \
  '  local interp="$1" out rc nonce want_root; return 0  # MUTATION: identity check bypassed'
expect_red "M2 identity check bypassed (always OK)" "$M2" scenario_b1

# M3 — the parse-time refuse-filter is disabled (a broken venv no longer drops `leading` rows).
#      Caught by gate iii (the refused-by-name message disappears; the row still gets scheduled).
M3="$T/mutant_m3.sh"
mutate "$CAND" "$M3" \
  'if (( NEED_VENV )) && (( ! VENV_OK )); then' \
  'if (( NEED_VENV )) && (( ! VENV_OK )) && false; then'
expect_red "M3 refuse-filter disabled (broken venv never refuses)" "$M3" scenario_iii

# M4 — the absolute-path rewrite is skipped for claimed rows (the label still says `venv`, but
#      the row's leading `python` is never actually rewritten). Caught by gate i (a python2 stub
#      shadowing PATH would then run instead of the venv) and gate cl.
M4_RAW="$T/mutant_m4_raw.sh"
mutate "$CAND" "$M4_RAW" \
  '    cmd="$(_rewrite_leading "$cmd")"' \
  '    : # MUTATION: rewrite disabled — the claim is made but never delivered'
M4="$(deploy_mutant "$M4_RAW")"
expect_red "M4 rewrite skipped (claim made, never delivered)" "$M4" scenario_i

# M5 — the rewrite is applied UNCONDITIONALLY, including to passthrough rows. Caught by gate ii
#      (env/bash passthrough rows would have their actual leading token replaced by the venv path).
M5_RAW="$T/mutant_m5_raw.sh"
mutate "$CAND" "$M5_RAW" \
  $'  if [[ "$interp" == "venv" ]]; then\n    # RE-ASSERT the interpreter HERE' \
  $'  if true; then  # MUTATION: claimed-only gate removed (rewrite touches passthrough rows too)\n    # RE-ASSERT the interpreter HERE'
M5="$(deploy_mutant "$M5_RAW")"
expect_red "M5 rewrite unconditional (passthrough no longer untouched)" "$M5" scenario_ii

# M6 — classification widens from "the LEADING token is exactly python/python3" to "the command
#      CONTAINS the substring python anywhere". Caught by gate ii (env/bash rows would then be
#      misclassified as `leading` and have their actual leading token wrongly rewritten).
M6_RAW="$T/mutant_m6_raw.sh"
mutate "$CAND" "$M6_RAW" \
  $'_is_bare_python_leading() {  # $1 cmd -> 0 iff the LEADING token is exactly python or python3\n  local t; t="$(_leading_token "$1")"\n  [[ "$t" == "python" || "$t" == "python3" ]]\n}' \
  $'_is_bare_python_leading() {  # MUTATION: substring match, not leading-token match\n  [[ "$1" == *python* ]]\n}'
M6="$(deploy_mutant "$M6_RAW")"
expect_red "M6 substring classification (over-claims non-leading spellings)" "$M6" scenario_ii

# M7 — the LAUNCH-TIME re-probe is removed at its real drop site, i.e. the code reverts to exactly
#      the parse-time-only design the run-36 fail-closed judge exploited. Caught by gate P: the
#      forged interpreter then executes under an `interpreter=venv` label. Note the parse-time check
#      is left fully intact by this mutation, which is the point — gate P must be killed by the
#      RE-PROBE's absence specifically, not by any venv check being gone.
M7="$T/mutant_m7.sh"
mutate "$CAND" "$M7" \
  '    if ! _venv_usable_ok; then' \
  '    if false; then  # MUTATION: launch-time re-probe removed (parse-time check only, as at run 36)'
expect_red "M7 launch-time re-probe removed (poison-after-check exploitable)" "$M7" scenario_poison

# M8 — the chained-python detector is blinded at its real drop site: an unquoted shell separator no
#      longer opens a command position, so a landmine row prints the same thing a harmless one does.
#      Caught by gate NCW (no warning, no marker, no summary entry for nc1) and by gate NCT (every
#      true positive collapses to 0). RE-ANCHORED in cycle 3: the old anchor
#      (`...|then|else|do|env) n=$((n+1));;`) no longer exists — the detector's counting moved into
#      _ucp_visit and `env` moved out of that case list — and because the anchor was stale the mutant
#      was never built while the suite still scored it a KILL. See expect_red's build-failure branch.
M8_RAW="$T/mutant_m8_raw.sh"
mutate "$CAND" "$M8_RAW" \
  "      '&&'|'||'|';'|'|'|'&'|'('|'{'|then|else|do) next=1;;" \
  "      '&&'|'||'|';'|'|'|'&'|'('|'{'|then|else|do) : ;;  # MUTATION: separators stop opening a command position"
M8="$(deploy_mutant "$M8_RAW")"
expect_red "M8 chained-python detector blinded (accepted limit becomes invisible)" "$M8" scenario_ncwarn

# M9 — QUOTE-BLIND TOKENISER: the `'`/`"` state transitions are dropped, so quotes become ordinary
#      characters and unquoted-looking whitespace inside a quoted span splits tokens again. This is
#      exactly the cycle-3 MAJOR: `--note "step1 && python notes"` then yields a bare `&&` separator
#      followed by a bare `python`, and the detector writes a FALSE entry into queue_summary.md's
#      NOT-COVERED block. Killed twice, deliberately: by the unit table (gate NCT) and end-to-end
#      through the real durable record (gate NCW), because the record is the thing that must not lie.
M9_RAW="$T/mutant_m9_raw.sh"
cat > "$T/m9_old.txt" <<'EOF'
        "'")            sq=1; tq=1; started=1;;
        '"')            dq=1; tq=1; started=1;;
EOF
cat > "$T/m9_new.txt" <<'EOF'
        "'")            tok+="$ch"; started=1;;
        '"')            tok+="$ch"; started=1;;
EOF
mutate_hd "$CAND" "$M9_RAW" "$T/m9_old.txt" "$T/m9_new.txt"
expect_red "M9 quote-blind tokeniser (quoted args become separators+invocations)" "$M9_RAW" scenario_nctable
M9_DEP="$(deploy_mutant "$M9_RAW")"
expect_red "M9 quote-blind tokeniser, end-to-end (false NOT-COVERED record)" "$M9_DEP" scenario_ncwarn

# M10 — `env` becomes an unconditionally transparent prefix again (the pre-cycle-3 rule), so
#       `--desc env python` — an UNQUOTED argument list — is read as an env(1) invocation. Caught by
#       gate NCT's fp3 row; quote-awareness alone does not save this one, which is why it is a
#       separate mutation from M9.
M10="$T/mutant_m10.sh"
mutate "$CAND" "$M10" \
  '      env) (( cmdpos )) && next=1;;' \
  '      env) next=1;;  # MUTATION: env transparent even as an argument'
expect_red "M10 env prefix transparent anywhere (argument read as invocation)" "$M10" scenario_nctable

# M11 — the refused row LEAKS its GPU: _free_gpu is removed from launch()'s refusal branch, at its
#       real drop site. Caught by gate PG only — with a one-GPU pool the following GPU rows can never
#       acquire anything, so they wait out the acquire timeout and are SKIPPED. This is the path the
#       cycle-2 re-verification had to check by hand because no committed test reached it.
M11="$T/mutant_m11.sh"
mutate "$CAND" "$M11" \
  '      [[ -n "$g" ]] && _free_gpu "$g"' \
  '      : # MUTATION: refused row keeps its GPU reservation forever'
expect_red "M11 refused GPU row leaks its GPU (no _free_gpu in the refusal branch)" "$M11" scenario_poison_gpu

# M12 — same as M10 for the OTHER transparent prefix: a VAR=value assignment is treated as
#       transparent wherever it appears, so `--opt FOO=bar python` is read as an invocation.
M12="$T/mutant_m12.sh"
mutate "$CAND" "$M12" \
  '      *=*) [[ "$t" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] && (( cmdpos )) && next=1;;' \
  '      *=*) [[ "$t" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] && next=1;;  # MUTATION: assignment prefix transparent anywhere'
expect_red "M12 assignment prefix transparent anywhere (argument read as invocation)" "$M12" scenario_nctable

echo "================================================================================"
echo "RESULT: $PASS passed, $FAIL failed"
echo "================================================================================"
[[ "$FAIL" -eq 0 ]]
