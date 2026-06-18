#!/usr/bin/env python3
"""TASK_804-B4 — seed-major self-advancing campaign orchestrator for the
token-frontend arms experiment (arms 1-3 x seeds 7/13/23).

Drives the 9-run campaign described in the TASK_804-B4 brief:

  3 arms   : raw  (canon, no token flags)
             real_fe (--use-token-frontend ... L1 vocab)
             shift_fe (same flags, CYCLIC-SHIFT NULL spec — TASK_804-B2.5)
  3 seeds  : 7, 13, 23

The null arm is `shift_fe` (TASK_804-B2.5): the corrected real-vs-null control.
Its "vocab file" is a cyclic-shift null SPEC (experiments/804/shift_null_spec.json,
built by build_shift_null.py) that names the REAL base vocab + per-asset roll
offsets; the hot-path rolls the real timeline per asset PRE-JOIN so the token<->bar
alignment is decoupled while the token marginal multiset is preserved. (This
REPLACES the dead-end π-permutation null `rand_fe` — a relabel of ids that left the
token sequence intact, "the control that couldn't fail".)

Hard orchestration contract (encoded here, NOT left to the operator):

  * SEED-MAJOR queue: seed 7 across all 3 arms first (raw_s7, real_s7,
    rand_s7), then seed 13 triple, then seed 23.  Within a seed, pack
    arms 2-wide.  The first complete paired triple lands ~1.5 slots in.
  * GPU contract: train <=2-wide on GPU 1 and 2 (TASK_742); GPU 3 for
    evals; GPU 0 untouchable.  Each run wrapped in
    `safe_run --gpu <1|2> --mem <cap> --gpu-frac 0.7` (PREALLOCATE default).
    Evals dispatch on GPU 3 as each seed's run finishes (eval-on-land).
  * Seed-7 log-staleness watchdog (MANDATORY): if a running train's log
    mtime is frozen AND its GPU is ~0% for > threshold, KILL it and flag
    (the `project_seed7_stageb_wedge` failure mode).
  * Per-run archival: each run dir gets the resolved config JSON, the
    deploy hash, the vocab sha, and (shift_fe) the shift seed.
  * Pairing/readout: after a seed's eval lands, compute Delta-alpha =
    arm-N - arm-1 *same-seed* (never vs historical 760), with trade-rate
    + Bal.Acc per arm; write incremental RESULT.md.
  * Stop/extend: KILL the campaign if a real arm hits mean Delta-alpha
    < -3pp OR null (shift) >= real consistently; EXTEND to seeds 31/42 iff
    3-seed mean Delta-alpha >= +1pp AND no seed < -3pp.

DRY-RUN MODE (`--dry-run`) is the audit gate.  The FULL queue executes
with stub commands (sleep + synthetic result JSON/.pkl) so the
scheduler, watchdog, eval-on-land, archival, pairing and stop/extend
logic all run against real inputs without a GPU, server, or tmux.

Real-mode SELF-ADVANCES: each safe_run job is spawned as a tracked child of the
one orchestrator tmux (Popen, venv-sourced, log tee'd) and the watchdog/poll loop
advances the queue (TASK_804-B4 emit->spawn fix). The 760 a1b1 canonical COMMON_BASE
train literals are finalized in `build_train_cmd` (SPEC.md:163-174, no --mcts-sims =>
default 32); eval is the eval-MCTS sims=16 joint scorecard.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

# --------------------------------------------------------------------------- #
# Campaign constants (pinned by the brief)
# --------------------------------------------------------------------------- #
DEPLOY_HASH = "57d05ac"               # pinned in TASK_804-B4 brief
ARMS = ("raw", "real_fe", "shift_fe")  # arm-1, arm-2, arm-3
REAL_ARMS = ("raw", "real_fe")        # "real" arms for the KILL rule
SHIFT_ARM = "shift_fe"                # the cyclic-shift NULL arm (TASK_804-B2.5)
BASELINE_ARM = "raw"                  # arm-1: Delta-alpha is computed vs this
SEEDS_PRIMARY = (7, 13, 23)
SEEDS_EXTEND = (31, 42)               # QUEUE ladder
SHIFT_SEED = 804                      # shift_null_spec.json per-asset offset seed
WARM_CKPT = "jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl"

# GPU contract
TRAIN_GPUS = (1, 2)                   # <=2-wide, TASK_742
EVAL_GPU = 3                          # evals on GPU 3
# GPU 0 is never referenced -> untouchable by construction.

# Stop / extend thresholds (pp on Delta-alpha)
KILL_REAL_MEAN_DA = -3.0             # real arm mean Delta-alpha < this -> KILL
EXTEND_MEAN_DA = 1.0                 # 3-seed mean Delta-alpha >= this ...
EXTEND_MIN_SEED_DA = -3.0           # ... AND no seed < this -> EXTEND

# Watchdog
WATCHDOG_STALE_SEC_DEFAULT = 20 * 60  # 20 min (brief threshold)
WATCHDOG_POLL_SEC_DEFAULT = 30        # how often the watchdog checks

# Token-frontend flags (arm-2 / arm-3).  Literal bodies are finalized by the
# operator at launch (B1 lands in parallel); these are the brief's values.
VOCAB_REAL = "experiments/804/L1_RESULT_vocab.json"
# shift_fe arm: the cyclic-shift null SPEC (build_shift_null.py), NOT a vocab — names the real base
# vocab + per-asset roll offsets; the hot-path rolls the real timeline pre-join (TASK_804-B2.5).
SHIFT_NULL_SPEC = "experiments/804/shift_null_spec.json"
TOKEN_LANGUAGE = "L1_bpe"
TOKEN_CACHE_DIR = "experiments/804/token_cache"


# --------------------------------------------------------------------------- #
# Run identity
# --------------------------------------------------------------------------- #
@dataclass
class Run:
    arm: str
    seed: int
    kind: str = "train"               # "train" | "eval"
    gpu: Optional[int] = None
    proc: Optional[subprocess.Popen] = None
    log_path: Optional[Path] = None
    run_dir: Optional[Path] = None
    started_at: float = 0.0
    finished_at: float = 0.0
    killed: bool = False
    kill_reason: str = ""
    failed: bool = False              # train exited non-zero / produced no best-ckpt
    fail_reason: str = ""

    @property
    def tag(self) -> str:
        return f"{self.arm}_seed{self.seed}"

    @property
    def name(self) -> str:
        return f"804_{self.kind}_{self.tag}"


# --------------------------------------------------------------------------- #
# Command templates (parameterized — operator finalizes literal bodies)
# --------------------------------------------------------------------------- #
def _arm_train_flags(arm: str) -> list[str]:
    """Arm-specific train flags appended to the common Stage-A->Stage-B body."""
    if arm == "raw":
        return []  # canon, no token flags
    if arm == "real_fe":
        return [
            "--use-token-frontend",
            "--token-frontend-vocab-file", VOCAB_REAL,
            "--token-frontend-language", TOKEN_LANGUAGE,
            "--token-frontend-cache-dir", TOKEN_CACHE_DIR,
        ]
    if arm == "shift_fe":
        return [
            "--use-token-frontend",
            # the cyclic-shift null spec is passed via the SAME vocab-file flag; the hot-path detects
            # null_type=="cyclic_shift" and rolls the real timeline (TASK_804-B2.5). Zero new flags.
            "--token-frontend-vocab-file", SHIFT_NULL_SPEC,
            "--token-frontend-language", TOKEN_LANGUAGE,
            "--token-frontend-cache-dir", TOKEN_CACHE_DIR,
        ]
    raise ValueError(f"unknown arm {arm!r}")


def build_train_cmd(arm: str, seed: int, run_dir: Path, gpu: int) -> list[str]:
    """Build the EXACT real-mode train command (argv list) for an arm+seed.

    Canonical Phase-2 Stage-A->Stage-B sequence, mirroring the 760 lineage
    (token params are inert in Stage A, learn in Stage B).  The COMMON body
    below is a placeholder for the 760 COMMON_BASE; the operator finalizes
    its literal contents at launch.  Identical across arms except the
    arm flags and the seed.
    """
    common = [
        "python", "-u", "train_phase2.py",
        # --- 760 a1b1 canonical COMMON_BASE (experiments/760_train_mcts_sims/SPEC.md:163-174;
        #     matches 741/746/734/757/758 + the 760 baseline). NO --mcts-sims => default 32 = a1b1.
        "--stage", "B", "--epochs", "200",
        "--periods-file", "config/foxglove_periods.json",
        "--n-slots", "17", "--max-runs-per-group", "10",
        "--factored", "--factored-temperature", "0.1",
        "--no-reset-value-head",
        "--value-target-mode", "nstep", "--td-discount", "0.95", "--value-weight", "0.5",
        "--bc-weight-min", "0.05", "--bc-max-trade-weight", "4.0",
        "--direction-weight", "0.5",
        "--encoder-lr", "3e-5", "--encoder-version", "3", "--embed-dim", "64", "--encoder-heads", "8",
        "--checkpoint", WARM_CKPT,           # warm-start (pinned a1b1 Phase-1 imprint)
        "--rank-loss", "mse", "--rank-weight", "1.0",
        "--aux-price-weight", "0.1",
        # -------------------------------------------------------------------
        "--seed", str(seed),
        "--output-dir", str(run_dir),
        "--name", f"804_{arm}_seed{seed}",
    ]
    return common + _arm_train_flags(arm)


def build_eval_cmd(arm: str, seed: int, run_dir: Path, ckpt: Path) -> list[str]:
    """Build the EXACT real-mode eval command (argv list) for an arm+seed.

    eval-MCTS sims=16 joint scorecard (760 lineage).  Operator finalizes
    the literal benchmark flags at launch.
    """
    return [
        "python", "tools/benchmark_easy.py",
        "--checkpoint", str(ckpt),
        "--db", "trading_data.db",
        "--json", str(run_dir / "benchmark.json"),
        # --- 760 eval scorecard placeholder (operator finalizes) -----------
        "--eval-mcts-sims", "16", "--eval-mcts-decode", "joint",
        # -------------------------------------------------------------------
    ]


def wrap_safe_run(name: str, gpu: int, mem: str, argv: list[str],
                  gpu_frac: float = 0.7) -> list[str]:
    """Wrap a child command in tools/safe_run.sh (RAM cgroup + GPU placement)."""
    return [
        "tools/safe_run.sh",
        "--name", name,
        "--mem", mem,
        "--gpu", str(gpu),
        "--gpu-frac", str(gpu_frac),
        "--", *argv,
    ]


def tmux_launch_line(window: str, argv: list[str], log_path: Path) -> str:
    """Emit the tmux launch line for real mode (NOT executed by this script).

    Real launches go in tmux (survive workstation-off, resumable) per the
    server-launch contract.  We only render the line; the operator (or the
    main agent on the server) executes it.
    """
    inner = " ".join(shlex.quote(a) for a in argv)
    piped = f"{inner} 2>&1 | tee {shlex.quote(str(log_path))}"
    return (
        f"tmux new-window -t 804_token_arms -n {shlex.quote(window)} "
        f"bash -c {shlex.quote('source trader_env/bin/activate && ' + piped)}"
    )


# --------------------------------------------------------------------------- #
# Archival
# --------------------------------------------------------------------------- #
def _sha256_file(path: Path) -> str:
    if not path.exists():
        return f"MISSING:{path}"
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def archive_run(run: Run, dry_run: bool) -> None:
    """Write the resolved config JSON + deploy hash + vocab sha (+ perm seed)
    into the run dir.  Verifiable post-dry-run."""
    rd = run.run_dir
    rd.mkdir(parents=True, exist_ok=True)

    vocab_file = None
    shift_seed = None
    if run.arm == "real_fe":
        vocab_file = VOCAB_REAL
    elif run.arm == "shift_fe":
        vocab_file = SHIFT_NULL_SPEC
        shift_seed = SHIFT_SEED

    vocab_sha = _sha256_file(Path(vocab_file)) if vocab_file else None

    cfg = {
        "task": "TASK_804-B4",
        "arm": run.arm,
        "seed": run.seed,
        "kind": run.kind,
        "gpu": run.gpu,
        "deploy_hash": DEPLOY_HASH,
        "warm_checkpoint": WARM_CKPT,
        "vocab_file": vocab_file,
        "vocab_sha256": vocab_sha,
        "shift_seed": shift_seed,
        "token_language": TOKEN_LANGUAGE if run.arm != "raw" else None,
        "train_cmd": build_train_cmd(run.arm, run.seed, rd, run.gpu or 0),
        "eval_cmd": build_eval_cmd(run.arm, run.seed, rd, rd / "phase2_stageB_best.pkl"),
        "dry_run": dry_run,
    }
    with open(rd / "resolved_config.json", "w") as fh:
        json.dump(cfg, fh, indent=2)
    # Standalone deploy-hash + vocab-sha markers for fast audit grep.
    (rd / "DEPLOY_HASH").write_text(DEPLOY_HASH + "\n")
    if vocab_sha is not None:
        (rd / "VOCAB_SHA256").write_text(f"{vocab_sha}  {vocab_file}\n")
    if shift_seed is not None:
        (rd / "SHIFT_SEED").write_text(f"{shift_seed}\n")


# --------------------------------------------------------------------------- #
# Stub command bodies (dry-run only)
# --------------------------------------------------------------------------- #
def _stub_train_script(run: Run, sleep_sec: float, stale_after: Optional[float],
                       synthetic: dict) -> str:
    """A self-contained python -c stub for a train run.

    It appends to its log on a heartbeat (so the watchdog has an mtime to
    track), then writes the fake phase2_stageB_best.pkl on completion.  If
    `stale_after` is set, it STOPS writing to the log after that many
    seconds (but keeps the process alive) to simulate the seed-7 wedge so
    the watchdog can detect + kill it.
    """
    rd = str(run.run_dir)
    return (
        "import time, os, json\n"
        f"rd = {rd!r}\n"
        f"sleep_sec = {sleep_sec!r}\n"
        f"stale_after = {stale_after!r}\n"
        f"synthetic = {json.dumps(synthetic)}\n"
        "os.makedirs(rd, exist_ok=True)\n"
        "log = os.path.join(rd, 'train.log')\n"
        "t0 = time.time()\n"
        "i = 0\n"
        "while time.time() - t0 < sleep_sec:\n"
        "    el = time.time() - t0\n"
        "    if stale_after is None or el < stale_after:\n"
        "        with open(log, 'a') as fh:\n"
        "            fh.write('epoch %d heartbeat t=%.1f\\n' % (i, el)); fh.flush()\n"
        "    # else: log goes silent -> watchdog should catch us\n"
        "    i += 1\n"
        "    time.sleep(0.5)\n"
        "# completion: write the fake best ckpt + a metrics stub\n"
        "with open(os.path.join(rd, 'phase2_stageB_best.pkl'), 'wb') as fh:\n"
        "    fh.write(b'STUB_CKPT')\n"
        "with open(os.path.join(rd, 'train_done.json'), 'w') as fh:\n"
        "    json.dump({'arm': synthetic['arm'], 'seed': synthetic['seed']}, fh)\n"
    )


def _stub_eval_script(run: Run, sleep_sec: float, synthetic: dict) -> str:
    """A self-contained python -c stub for an eval run: sleeps then writes a
    benchmark.json carrying the synthetic alpha / trade-rate / bal-acc so the
    pairing logic has real inputs."""
    rd = str(run.run_dir)
    return (
        "import time, os, json\n"
        f"rd = {rd!r}\n"
        f"sleep_sec = {sleep_sec!r}\n"
        f"synthetic = {json.dumps(synthetic)}\n"
        "os.makedirs(rd, exist_ok=True)\n"
        "log = os.path.join(rd, 'benchmark.log')\n"
        "t0 = time.time()\n"
        "while time.time() - t0 < sleep_sec:\n"
        "    with open(log, 'a') as fh:\n"
        "        fh.write('eval tick t=%.1f\\n' % (time.time()-t0)); fh.flush()\n"
        "    time.sleep(0.3)\n"
        "with open(os.path.join(rd, 'benchmark.json'), 'w') as fh:\n"
        "    json.dump(synthetic, fh, indent=2)\n"
    )


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
class Orchestrator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.dry_run: bool = args.dry_run
        self.root = Path("runs/804_token_arms")
        self.mem_cap = args.mem
        self.stale_sec = args.watchdog_stale_sec
        self.poll_sec = args.watchdog_poll_sec
        self.seeds = list(SEEDS_PRIMARY)

        # GPU 0% / log mtime probes for the watchdog.
        self.gpu_util_fn: Callable[[int], float] = self._gpu_util_real
        self.now_fn: Callable[[], float] = time.time

        # Scheduler state.
        self.train_slots: dict[int, Optional[Run]] = {g: None for g in TRAIN_GPUS}
        self.eval_slot: Optional[Run] = None
        self.eval_queue: list[Run] = []           # finished trains awaiting eval
        self.completed_evals: dict[str, dict] = {}  # tag -> benchmark dict
        self.train_done: set[str] = set()
        self.killed_runs: list[Run] = []
        self.failed_runs: list[Run] = []   # trains that exited bad / wrote no ckpt
        self.campaign_killed = False
        self.campaign_kill_reason = ""
        self.extend_triggered = False
        self.max_concurrent_trains = 0            # 2-wide invariant tracker

        # Synthetic Delta-alpha scenario for dry-run (drives stop/extend).
        self.scenario = args.scenario

        self._lock = threading.RLock()
        self._watchdog_stop = threading.Event()

    # ---- logging ---------------------------------------------------------- #
    def log(self, msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    # ---- GPU util probe --------------------------------------------------- #
    def _gpu_util_real(self, gpu: int) -> float:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits", "-i", str(gpu)],
                text=True, timeout=10,
            )
            return float(out.strip().splitlines()[0])
        except Exception:
            return 100.0  # fail safe: assume busy, don't false-kill

    # ---- build the seed-major queue --------------------------------------- #
    def build_queue(self, seeds: list[int]) -> list[Run]:
        """SEED-MAJOR: all 3 arms for seed s, then next seed.  Within a seed,
        arms are emitted in ARMS order (raw, real_fe, shift_fe) so raw_s7 and
        real_s7 start together 2-wide and rand_s7 follows as a slot frees."""
        q: list[Run] = []
        for s in seeds:
            for arm in ARMS:
                q.append(Run(arm=arm, seed=s, kind="train"))
        return q

    # ---- launch a single train run ---------------------------------------- #
    def _launch_train(self, run: Run, gpu: int) -> None:
        run.gpu = gpu
        run.run_dir = self.root / run.tag
        run.log_path = run.run_dir / "train.log"
        run.started_at = self.now_fn()
        archive_run(run, self.dry_run)  # config/hash/vocab-sha BEFORE launch

        if self.dry_run:
            sleep_sec, stale_after = self._dry_train_timing(run)
            synthetic = {"arm": run.arm, "seed": run.seed}
            script = _stub_train_script(run, sleep_sec, stale_after, synthetic)
            run.proc = subprocess.Popen([sys.executable, "-c", script])
            tag = " (STALE-INJECT)" if stale_after is not None else ""
            self.log(f"LAUNCH train {run.tag} on GPU{gpu} "
                     f"[stub {sleep_sec:.0f}s]{tag}")
        else:
            # Self-advancing: spawn the safe_run job as a tracked child of THIS
            # orchestrator (which itself runs in one tmux session per
            # feedback_server_launches_in_tmux). bash -c sources the venv + tees the
            # per-run log (the watchdog reads its mtime); the existing poll()/watchdog
            # machinery tracks completion + advances the queue.
            run.run_dir.mkdir(parents=True, exist_ok=True)
            argv = build_train_cmd(run.arm, run.seed, run.run_dir, gpu)
            wrapped = wrap_safe_run(run.name, gpu, self.mem_cap, argv)
            inner = " ".join(shlex.quote(a) for a in wrapped)
            bash_cmd = (f"source trader_env/bin/activate && {inner} "
                        f"2>&1 | tee {shlex.quote(str(run.log_path))} >/dev/null")
            run.proc = subprocess.Popen(["bash", "-c", bash_cmd])
            self.log(f"LAUNCH train {run.tag} on GPU{gpu} (pid {run.proc.pid})")
            self.log("  cmd: " + inner)
        self.train_slots[gpu] = run
        n = sum(1 for s in self.train_slots.values() if s is not None)
        self.max_concurrent_trains = max(self.max_concurrent_trains, n)

    # ---- launch a single eval run ----------------------------------------- #
    def _launch_eval(self, train_run: Run) -> None:
        ev = Run(arm=train_run.arm, seed=train_run.seed, kind="eval", gpu=EVAL_GPU)
        ev.run_dir = self.root / f"eval_{train_run.tag}"
        ev.log_path = ev.run_dir / "benchmark.log"
        ev.started_at = self.now_fn()
        ev.run_dir.mkdir(parents=True, exist_ok=True)

        if self.dry_run:
            synthetic = self._synthetic_benchmark(train_run.arm, train_run.seed)
            script = _stub_eval_script(ev, self.args.eval_stub_sec, synthetic)
            ev.proc = subprocess.Popen([sys.executable, "-c", script])
            self.log(f"EVAL-ON-LAND dispatch {ev.tag} on GPU{EVAL_GPU} "
                     f"[stub {self.args.eval_stub_sec:.0f}s]")
        else:
            ckpt = train_run.run_dir / "phase2_stageB_best.pkl"
            argv = build_eval_cmd(ev.arm, ev.seed, ev.run_dir, ckpt)
            wrapped = wrap_safe_run(ev.name, EVAL_GPU, self.mem_cap, argv)
            inner = " ".join(shlex.quote(a) for a in wrapped)
            bash_cmd = (f"source trader_env/bin/activate && {inner} "
                        f"2>&1 | tee {shlex.quote(str(ev.log_path))} >/dev/null")
            ev.proc = subprocess.Popen(["bash", "-c", bash_cmd])
            self.log(f"EVAL-ON-LAND dispatch {ev.tag} on GPU{EVAL_GPU} (pid {ev.proc.pid})")
            self.log("  cmd: " + inner)
        self.eval_slot = ev

    # ---- dry-run timing / synthetic data ---------------------------------- #
    def _dry_train_timing(self, run: Run) -> tuple[float, Optional[float]]:
        """Pick a stub duration; inject staleness on the designated seed-7 run
        so the watchdog fires.  We inject on raw_seed7 (the first run, highest
        attention) per the brief."""
        base = self.args.train_stub_sec
        stale_after = None
        if (self.args.inject_stale and run.arm == "raw" and run.seed == 7):
            # Go stale early, then "hang" far past its slot so the watchdog must
            # kill it; the queue then recovers and re-uses the freed slot.
            stale_after = max(1.0, base * 0.3)
            base = base * 100
        return base, stale_after

    def _synthetic_benchmark(self, arm: str, seed: int) -> dict:
        """Synthetic alpha/trade-rate/bal-acc per arm+seed to drive pairing +
        stop/extend.  Scenario picks the campaign-level outcome."""
        base_alpha = {7: 4.0, 13: 5.0, 23: 6.0, 31: 5.5, 42: 6.5}[seed]
        if self.scenario == "extend":
            # real_fe clearly positive, all seeds >= +1, none < -3 -> EXTEND.
            da = {"raw": 0.0,
                  "real_fe": {7: 2.5, 13: 3.0, 23: 2.0, 31: 2.2, 42: 2.8}[seed],
                  "shift_fe": {7: -0.5, 13: 0.2, 23: -1.0, 31: 0.0, 42: 0.3}[seed]}
        elif self.scenario == "kill":
            # real_fe collapses (mean << -3pp) -> campaign KILL.
            da = {"raw": 0.0,
                  "real_fe": {7: -4.0, 13: -5.0, 23: -3.5, 31: -4.0, 42: -4.0}[seed],
                  "shift_fe": {7: -1.0, 13: -0.5, 23: -1.5, 31: -1.0, 42: -1.0}[seed]}
        elif self.scenario == "null_ge_real":
            # null (shift) >= real consistently -> campaign KILL (token signal noise).
            da = {"raw": 0.0,
                  "real_fe": {7: 0.5, 13: 0.3, 23: 0.4, 31: 0.4, 42: 0.4}[seed],
                  "shift_fe": {7: 1.5, 13: 1.4, 23: 1.6, 31: 1.5, 42: 1.5}[seed]}
        else:  # "neutral" — finishes 3 seeds, no kill, no extend
            da = {"raw": 0.0,
                  "real_fe": {7: 0.5, 13: -0.5, 23: 0.2, 31: 0.0, 42: 0.0}[seed],
                  "shift_fe": {7: -0.3, 13: 0.1, 23: -0.4, 31: 0.0, 42: 0.0}[seed]}
        alpha = base_alpha + da[arm]
        return {
            "arm": arm,
            "seed": seed,
            "alpha": round(alpha, 4),
            "trade_rate": round(0.20 + 0.01 * (hash((arm, seed)) % 7), 4),
            "bal_acc": round(0.57 + 0.002 * (hash((seed, arm)) % 5), 4),
        }

    # ---- watchdog --------------------------------------------------------- #
    def _watchdog_loop(self) -> None:
        """Monitor every running train: if its log mtime is frozen AND its GPU
        is ~0% for > stale_sec, KILL it and flag.  Covers the seed-7 wedge."""
        while not self._watchdog_stop.is_set():
            with self._lock:
                for gpu, run in list(self.train_slots.items()):
                    if run is None or run.proc is None or run.killed:
                        continue
                    if run.proc.poll() is not None:
                        continue  # finished; reaped by scheduler
                    self._watchdog_check(run)
            self._watchdog_stop.wait(self.poll_sec)

    def _watchdog_check(self, run: Run) -> None:
        log = run.log_path
        now = self.now_fn()
        try:
            mtime = log.stat().st_mtime if log and log.exists() else run.started_at
        except OSError:
            mtime = run.started_at
        stale_for = now - mtime
        util = self.gpu_util_fn(run.gpu)
        if stale_for > self.stale_sec and util < 5.0:
            run.killed = True
            run.kill_reason = (
                f"log frozen {stale_for:.0f}s (>{self.stale_sec:.0f}s) "
                f"AND GPU{run.gpu} util {util:.0f}% (seed-{run.seed} wedge)"
            )
            try:
                run.proc.kill()
            except Exception:
                pass
            # run.proc is only the bash wrapper; the `systemd-run --scope` python survives
            # it as an orphan, holding its GPU and OOM-failing the NEXT launch on that device
            # (the 2026-06-18 cascade). Reap the scope so the slot's GPU is actually freed.
            self._reap_scope(run)
            self.killed_runs.append(run)
            self.log(f"WATCHDOG KILL {run.tag} on GPU{run.gpu}: {run.kill_reason}")
            self.log(f"WATCHDOG FLAG: {run.tag} flagged "
                     f"(project_seed7_stageb_wedge); slot freed, queue recovers")

    def _reap_scope(self, run: Run) -> None:
        """Stop the safe_run systemd --user scope backing `run`. run.proc.kill() reaps only
        the bash wrapper; safe_run launches the trainer in `systemd-run --user --scope
        --unit=<NAME>_<pid>` which outlives the wrapper. Stop it by the run.name glob so the
        scoped python (and its GPU memory) is actually released before the slot is re-used."""
        if self.dry_run:
            return
        try:
            subprocess.run(
                ["bash", "-c",
                 f"systemctl --user list-units --all --no-legend '{run.name}_*.scope' 2>/dev/null "
                 f"| awk '{{print $1}}' | grep -E '[.]scope$' | xargs -r -n1 systemctl --user stop 2>/dev/null; "
                 f"systemctl --user reset-failed '{run.name}_*.scope' 2>/dev/null || true"],
                timeout=25, check=False)
            self.log(f"  scope reaped for {run.tag} ({run.name}_*.scope)")
        except Exception as e:  # never let scope-reap failure crash the watchdog
            self.log(f"  scope-reap WARN for {run.tag}: {e}")

    # ---- pairing / readout ------------------------------------------------ #
    def _record_eval(self, ev: Run) -> None:
        bj = ev.run_dir / "benchmark.json"
        if bj.exists():
            with open(bj) as fh:
                data = json.load(fh)
        else:
            data = {"arm": ev.arm, "seed": ev.seed, "alpha": None}
        self.completed_evals[ev.tag] = data
        self.log(f"EVAL LANDED {ev.tag}: alpha={data.get('alpha')}")

    def _paired_da(self, seed: int) -> Optional[dict]:
        """Delta-alpha = arm - raw, same-seed.  None until all 3 arms eval'd."""
        if not all(f"{arm}_seed{seed}" in self.completed_evals for arm in ARMS):
            return None
        raw = self.completed_evals[f"raw_seed{seed}"]["alpha"]
        out = {"seed": seed, "raw_alpha": raw}
        for arm in ARMS:
            d = self.completed_evals[f"{arm}_seed{seed}"]
            out[arm] = {
                "alpha": d["alpha"],
                "da": round(d["alpha"] - raw, 4),
                "trade_rate": d.get("trade_rate"),
                "bal_acc": d.get("bal_acc"),
            }
        return out

    def _all_paired(self) -> list[dict]:
        out = []
        for s in list(SEEDS_PRIMARY) + list(SEEDS_EXTEND):
            p = self._paired_da(s)
            if p:
                out.append(p)
        return out

    def _write_result_md(self) -> None:
        rows = self._all_paired()
        lines = [
            "# TASK_804-B4 — Token Arms Campaign — RESULT (incremental)",
            "",
            f"Deploy hash: `{DEPLOY_HASH}`  |  arms: {', '.join(ARMS)}  |  "
            f"Delta-alpha = arm - raw (same-seed)"
            + ("  |  **DRY-RUN (synthetic)**" if self.dry_run else ""),
            "",
            "| seed | raw alpha | real_fe alpha (Da) | shift_fe alpha (Da) | "
            "real_fe trade | shift_fe trade |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for p in rows:
            lines.append(
                f"| {p['seed']} | {p['raw_alpha']:.2f} | "
                f"{p['real_fe']['alpha']:.2f} ({p['real_fe']['da']:+.2f}) | "
                f"{p['shift_fe']['alpha']:.2f} ({p['shift_fe']['da']:+.2f}) | "
                f"{p['real_fe']['trade_rate']} | {p['shift_fe']['trade_rate']} |"
            )
        for arm in ("real_fe", "shift_fe"):
            das = [p[arm]["da"] for p in rows]
            if das:
                lines.append("")
                lines.append(
                    f"- mean Delta-alpha **{arm}** = "
                    f"{sum(das)/len(das):+.2f}pp  (n={len(das)}, "
                    f"min seed {min(das):+.2f})"
                )
        if self.campaign_killed:
            lines += ["", f"## CAMPAIGN KILLED — {self.campaign_kill_reason}"]
        if self.extend_triggered:
            lines += ["", f"## EXTEND TRIGGERED — seeds {SEEDS_EXTEND} queued"]
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "RESULT.md").write_text("\n".join(lines) + "\n")
        self.log("RESULT.md updated")

    # ---- stop / extend ---------------------------------------------------- #
    def _check_stop_extend(self) -> None:
        paired = self._all_paired()
        if not paired:
            return
        # KILL: any real arm mean Da < -3pp.
        for arm in REAL_ARMS:
            if arm == BASELINE_ARM:
                continue
            das = [p[arm]["da"] for p in paired]
            mean_da = sum(das) / len(das)
            # QA1-[1]: never KILL on a single seed — one noisy seed-7 draw must not end
            # the campaign (the >=3-seed-noise contract; matches the null_ge kill below).
            if len(paired) >= 2 and mean_da < KILL_REAL_MEAN_DA:
                self.campaign_killed = True
                self.campaign_kill_reason = (
                    f"real arm {arm} mean Delta-alpha {mean_da:+.2f}pp "
                    f"< {KILL_REAL_MEAN_DA}pp (n={len(das)})"
                )
                self.log(f"STOP/KILL: {self.campaign_kill_reason}")
                return
        # KILL: null (shift) >= real consistently (all completed seeds, >=2).
        null_ge = [p[SHIFT_ARM]["da"] >= p["real_fe"]["da"] for p in paired]
        if len(paired) >= 2 and all(null_ge):
            self.campaign_killed = True
            self.campaign_kill_reason = (
                f"null (shift) arm >= real arm on all {len(paired)} seeds "
                f"(token signal is noise)"
            )
            self.log(f"STOP/KILL: {self.campaign_kill_reason}")
            return
        # EXTEND: only on the FULL primary 3-seed set, real_fe arm.
        primary_done = {p["seed"] for p in paired} >= set(SEEDS_PRIMARY)
        if primary_done and not self.extend_triggered:
            das = [self._paired_da(s)["real_fe"]["da"] for s in SEEDS_PRIMARY]
            mean_da = sum(das) / len(das)
            if mean_da >= EXTEND_MEAN_DA and min(das) >= EXTEND_MIN_SEED_DA:
                self.extend_triggered = True
                self.log(
                    f"EXTEND: real_fe 3-seed mean Delta-alpha {mean_da:+.2f}pp "
                    f">= {EXTEND_MEAN_DA}pp AND min seed {min(das):+.2f}pp "
                    f">= {EXTEND_MIN_SEED_DA}pp -> queue seeds {SEEDS_EXTEND}"
                )

    # ---- scheduler core --------------------------------------------------- #
    def run(self) -> int:
        self.log(f"=== TASK_804-B4 orchestrator START "
                 f"({'DRY-RUN' if self.dry_run else 'REAL'}, "
                 f"scenario={self.scenario}) ===")
        self.log(f"GPU contract: train on {TRAIN_GPUS} (2-wide), eval on "
                 f"GPU{EVAL_GPU}, GPU0 untouchable")
        if self.dry_run:
            self.gpu_util_fn = self._gpu_util_dry

        pending = self.build_queue(self.seeds)
        self.log("SEED-MAJOR queue: " + " -> ".join(r.tag for r in pending))

        wd = threading.Thread(target=self._watchdog_loop, daemon=True)
        wd.start()

        launched: list[Run] = []
        extend_appended = False

        while True:
            with self._lock:
                self._reap_finished()
                self._pump_evals()

                # EXTEND: append seeds 31/42 once, when triggered.
                if (self.extend_triggered and not extend_appended
                        and not self.campaign_killed):
                    ext = self.build_queue(list(SEEDS_EXTEND))
                    pending.extend(ext)
                    extend_appended = True
                    self.log("EXTEND queue appended: "
                             + " -> ".join(r.tag for r in ext))

                # Fill train slots 2-wide in seed-major order.
                if not self.campaign_killed:
                    for gpu in TRAIN_GPUS:
                        if self.train_slots[gpu] is None and pending:
                            run = pending.pop(0)
                            self._launch_train(run, gpu)
                            launched.append(run)

                idle = (all(s is None for s in self.train_slots.values())
                        and self.eval_slot is None and not self.eval_queue)
                if (not pending and idle) or (self.campaign_killed and idle):
                    break

            time.sleep(self.args.tick_sec)

        self._watchdog_stop.set()
        wd.join(timeout=2)
        self._write_result_md()
        self._final_report(launched)
        return 0

    def _reap_finished(self) -> None:
        """Reap finished/killed train slots; enqueue eval-on-land.  Killed runs
        are NOT auto-rerun (flagged for the operator); the slot just frees."""
        for gpu, run in list(self.train_slots.items()):
            if run is None:
                continue
            finished = run.proc is None or run.proc.poll() is not None
            if not finished:
                continue
            run.finished_at = self.now_fn()
            self.train_slots[gpu] = None
            if run.killed:
                self.log(f"SLOT FREED (killed) GPU{gpu}: {run.tag}")
                continue
            # QA1-[2]: a crashed / OOM-killed train must NOT be evaluated as success.
            # safe_run's cgroup OOM-killer reaps the scope with a non-zero exit; trusting
            # poll()!=None alone would dispatch an eval against a missing/stale best-ckpt
            # and poison the paired Delta-alpha. Require a clean exit AND the best-ckpt on
            # disk before queueing eval (ckpt check is real-mode only; dry-run stubs write none).
            rc = run.proc.returncode if run.proc is not None else 1
            ckpt_ok = self.dry_run or (run.run_dir / "phase2_stageB_best.pkl").exists()
            if rc != 0 or not ckpt_ok:
                run.failed = True
                run.fail_reason = (f"exit={rc}, best-ckpt="
                                   f"{'ok' if ckpt_ok else 'MISSING'}")
                self.failed_runs.append(run)
                self.log(f"TRAIN FAILED {run.tag} (GPU{gpu}): {run.fail_reason} "
                         f"-> NOT evaluated (flagged for operator)")
                continue
            self.train_done.add(run.tag)
            self.log(f"TRAIN DONE {run.tag} (GPU{gpu}) -> queue eval-on-land")
            self.eval_queue.append(run)

    def _pump_evals(self) -> None:
        """Dispatch one eval at a time on GPU 3 (eval-on-land), reap finished
        evals, record results, fire pairing + stop/extend after each land."""
        if self.eval_slot is not None:
            ev = self.eval_slot
            done = ev.proc is None or ev.proc.poll() is not None
            if done:
                ev.finished_at = self.now_fn()
                self._record_eval(ev)
                self.eval_slot = None
                self._write_result_md()
                self._check_stop_extend()
        if self.eval_slot is None and self.eval_queue:
            train_run = self.eval_queue.pop(0)
            self._launch_eval(train_run)

    # ---- dry-run GPU util mock -------------------------------------------- #
    def _gpu_util_dry(self, gpu: int) -> float:
        """In dry-run, report 0% for the stale-inject run (so the watchdog can
        fire) and busy for everything else.  Staleness is detected by log mtime
        in _watchdog_check; this supplies the 'GPU ~0%' half of the AND."""
        run = self.train_slots.get(gpu)
        if run is None:
            return 0.0
        if (self.args.inject_stale and run.arm == "raw" and run.seed == 7
                and not run.killed):
            return 0.0
        return 95.0

    # ---- final report ----------------------------------------------------- #
    def _final_report(self, launched: list[Run]) -> None:
        self.log("=== CAMPAIGN COMPLETE ===")
        self.log(f"trains launched: {len(launched)} | "
                 f"evals landed: {len(self.completed_evals)} | "
                 f"watchdog kills: {len(self.killed_runs)} | "
                 f"max concurrent trains: {self.max_concurrent_trains} (<=2 ok="
                 f"{self.max_concurrent_trains <= 2})")
        for r in self.killed_runs:
            self.log(f"  KILLED: {r.tag} — {r.kill_reason}")
        if self.campaign_killed:
            self.log(f"VERDICT: CAMPAIGN KILLED — {self.campaign_kill_reason}")
        elif self.extend_triggered:
            self.log(f"VERDICT: EXTEND to seeds {SEEDS_EXTEND}")
        else:
            self.log("VERDICT: primary 3-seed complete, no kill/extend")


# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TASK_804-B4 token-arms orchestrator")
    p.add_argument("--dry-run", action="store_true",
                   help="execute the full queue with stub commands (audit gate)")
    p.add_argument("--mem", default="40G", help="safe_run RAM cap per run")
    p.add_argument("--watchdog-stale-sec", type=float,
                   default=WATCHDOG_STALE_SEC_DEFAULT,
                   help="log-staleness kill threshold (sec)")
    p.add_argument("--watchdog-poll-sec", type=float,
                   default=WATCHDOG_POLL_SEC_DEFAULT,
                   help="watchdog poll interval (sec)")
    # Dry-run knobs.
    p.add_argument("--scenario",
                   choices=("neutral", "extend", "kill", "null_ge_real"),
                   default="neutral",
                   help="dry-run synthetic Delta-alpha scenario")
    p.add_argument("--train-stub-sec", type=float, default=3.0,
                   help="dry-run train stub duration")
    p.add_argument("--eval-stub-sec", type=float, default=1.5,
                   help="dry-run eval stub duration")
    p.add_argument("--inject-stale", action="store_true",
                   help="dry-run: make raw_seed7 log go stale to test watchdog")
    p.add_argument("--tick-sec", type=float, default=0.25,
                   help="scheduler poll interval")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        # Tighten watchdog timing for a fast dry-run audit if left at default.
        if args.watchdog_stale_sec == WATCHDOG_STALE_SEC_DEFAULT:
            args.watchdog_stale_sec = 2.0
        if args.watchdog_poll_sec == WATCHDOG_POLL_SEC_DEFAULT:
            args.watchdog_poll_sec = 0.5
    orch = Orchestrator(args)
    return orch.run()


if __name__ == "__main__":
    raise SystemExit(main())
