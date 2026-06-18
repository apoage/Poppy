#!/usr/bin/env python3
"""TASK_804 arms 4-5 (token-grounded DYNAMICS) solo runner on GPU 2.

Fills the idle GPU while the front-end retry runner (804_solo) owns GPU 1 (train) +
GPU 3 (eval). SELF-CONTAINED on GPU 2 (trains AND evals on 2) so it never contends
with the front-end runner. Per-run frozen-log watchdog + systemd-scope reap + retry,
identical discipline to solo_retry.py.

Arms (TASK_804 five-arm spec):
  arm-4 real_dyn  = real_fe front-end flags + TASK_802 --use-token-dynamics
  arm-5 shift_dyn = shift_fe (cyclic-shift null) flags + --use-token-dynamics  (placebo for 4)
Both ride the existing front-end wiring and reuse the prebuilt token-cache artifacts
(real timelines_d6a718dd*, shift timelines_b55f18f6*) -> cache HIT, no cold-annotate.

Seed-major: complete each seed's real/shift pair before moving on (real_dyn - shift_dyn
is the within-seed planning-side signal -> the branch-C "diagnose" test, TASK_804 line 75).
Reuses orchestrate_token_arms builders so the COMMON Stage-B body stays bit-identical to
arms 1-3 except the seed and the two dynamics flags.

Namespaces are disjoint from the front-end runner:
  run dirs   real_dyn_seed* / shift_dyn_seed* (vs raw_*/real_fe_*/shift_fe_*)
  units      804dyn_train_* / 804dyn_eval_*   (vs 804solo_*)
  GPU        train+eval on 2                  (vs 1 train / 3 eval)
"""
import sys, subprocess, shlex, time, os
from pathlib import Path
sys.path.insert(0, "experiments/804")
import orchestrate_token_arms as O

BASE = {"real_dyn": "real_fe", "shift_dyn": "shift_fe"}  # front-end arm whose flags we ride
RUNS = [("real_dyn", 7), ("shift_dyn", 7),
        ("real_dyn", 13), ("shift_dyn", 13),
        ("real_dyn", 23), ("shift_dyn", 23)]
ROOT = Path("runs/804_token_arms")
TRAIN_GPU, EVAL_GPU, MEM = 2, 2, "40G"
MAX_RETRY, STALE_SEC, POLL, TARGET = 3, 1200, 60, 195
DYN_FLAGS = ["--use-token-dynamics", "--token-dynamics-weight", "0.1"]


def gpu(g):
    try:
        return float(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits",
             "-i", str(g)], timeout=10).decode().strip())
    except Exception:
        return 100.0


def epochs(log):
    try:
        return int(subprocess.check_output(
            ["bash", "-c", f"grep -c consistency= {shlex.quote(str(log))} 2>/dev/null||echo 0"]
        ).decode().strip() or 0)
    except Exception:
        return 0


def reap(name):
    subprocess.run(["bash", "-c",
        f"systemctl --user list-units --all --no-legend '{name}_*.scope' 2>/dev/null"
        f"|awk '{{print $1}}'|grep -E '[.]scope$'|xargs -r -n1 systemctl --user stop 2>/dev/null"],
        timeout=25)


def build_dyn_train(dyn_arm, seed, rd, g):
    """Front-end arm command (correct COMMON body + token-frontend flags + output-dir)
    with the --name label fixed to *_dyn_* and the two TASK_802 dynamics flags appended."""
    cmd = O.build_train_cmd(BASE[dyn_arm], seed, rd, g)
    i = cmd.index("--name")
    cmd[i + 1] = f"804_{dyn_arm}_seed{seed}"
    return cmd + DYN_FLAGS


def main():
    for arm, seed in RUNS:
        tag = f"{arm}_seed{seed}"; rd = ROOT / tag; ok = False
        for att in range(1, MAX_RETRY + 1):
            rd.mkdir(parents=True, exist_ok=True)
            ck = rd / "phase2_stageB_best.pkl"
            if ck.exists():
                ck.unlink()
            name = f"804dyn_train_{tag}"; log = rd / "train.log"
            argv = O.wrap_safe_run(name, TRAIN_GPU, MEM, build_dyn_train(arm, seed, rd, TRAIN_GPU))
            inner = " ".join(shlex.quote(a) for a in argv)
            print(f"[dyn] TRAIN {tag} attempt {att}/{MAX_RETRY}", flush=True)
            p = subprocess.Popen(["bash", "-c",
                f"source trader_env/bin/activate && {inner} 2>&1 | tee {shlex.quote(str(log))} >/dev/null"])
            while True:
                time.sleep(POLL)
                if p.poll() is not None:
                    ep = epochs(log)
                    ok = ck.exists() and ep >= TARGET
                    print(f"[dyn] {tag} EXIT rc={p.returncode} ep={ep} ckpt={ck.exists()} "
                          f"-> {'OK' if ok else 'INCOMPLETE'}", flush=True)
                    break
                try:
                    age = time.time() - os.path.getmtime(log)
                except OSError:
                    age = 0
                if age > STALE_SEC and gpu(TRAIN_GPU) < 5.0:
                    print(f"[dyn] {tag} WEDGE (frozen {age:.0f}s, GPU{TRAIN_GPU} idle) -> kill+retry",
                          flush=True)
                    try:
                        p.kill()
                    except Exception:
                        pass
                    reap(name); time.sleep(5); break
            if ok:
                break
        if ok:
            erd = ROOT / f"eval_{tag}"; erd.mkdir(parents=True, exist_ok=True)
            eargv = O.wrap_safe_run(f"804dyn_eval_{tag}", EVAL_GPU, MEM,
                                    O.build_eval_cmd(BASE[arm], seed, erd, rd / "phase2_stageB_best.pkl"))
            einner = " ".join(shlex.quote(a) for a in eargv)
            print(f"[dyn] EVAL {tag}", flush=True)
            subprocess.run(["bash", "-c",
                f"source trader_env/bin/activate && {einner} 2>&1 | tee {shlex.quote(str(erd / 'benchmark.log'))} >/dev/null"])
            print(f"[dyn] EVAL {tag} done", flush=True)
        else:
            print(f"[dyn] GIVE UP {tag} after {MAX_RETRY} attempts", flush=True)
    print("[dyn] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
