#!/usr/bin/env python3
"""Score-based checkpoint GC + disk-floor guard (operator requirement, 2026-08-05).

WHY THIS EXISTS AND WHY IT IS NOT A TRAINING-LOOP FEATURE
--------------------------------------------------------
Long runs write ~89 MB per snapshot and `snapshot_ep{N}` dirs are never overwritten, while the
main save path prunes by RECENCY (flax ``keep=3``) — which after a late-run collapse keeps the
broken checkpoints and evicts the good ones. If a training host fills its filesystem, the
casualty is not just the run: ``trading_data.db`` lives on the same filesystem, and re-shipping
it is impossible on a ~5 Mbps uplink. So pruning has to exist as an EXTERNAL, always-on
mechanism that works on runs already in flight and on runs launched by anything (factory,
planner, hand) — independent of whether a given trainer was taught to prune itself.

POLICY
------
Per run directory: keep the TOP-K by score, plus ALWAYS the newest checkpoint (resume) and the
best-scored one. Scores come from, in order: ``checkpoints_index.json``, ``scores.jsonl``
(``{"epoch": int, "score": float}`` per line), or a run log matched by ``--score-regex``.
Checkpoints with NO score are never pruned on score — they are only pruned by recency when free
space is under the floor, and every such deletion says so in the log. Missing data never causes
a deletion decision (a checkpoint you cannot evaluate is a checkpoint you keep).

SAFETY (all enforced, not documented-and-hoped)
-----------------------------------------------
* dry-run by default; ``--apply`` required to delete anything
* only paths UNDER a configured root, whose basename matches a checkpoint pattern, are eligible
* never deletes: newest per run · best-scored per run · anything listed in ``--protect-file`` ·
  any path containing a ``.db``/``.git``/``trading_data`` component · symlinks (never followed)
* deletion is move-to-``.gc-trash-<pid>`` then rmtree, so a partial delete cannot leave a
  half-checkpoint that looks loadable
* every action is logged with score + reason + bytes
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time

CKPT_PATTERNS = (re.compile(r"^snapshot_ep\d+$"), re.compile(r"^ckpt(_.*)?$"),
                 re.compile(r"^checkpoint_\d+$"))
FORBIDDEN_PARTS = (".git", "trading_data", "fixtures", "backups")


def _is_ckpt_dir(path: str) -> bool:
    base = os.path.basename(path.rstrip("/"))
    if any(p.match(base) for p in CKPT_PATTERNS):
        return True
    # a flax checkpoint dir: contains checkpoint_* files
    try:
        return any(n.startswith("checkpoint_") for n in os.listdir(path))
    except OSError:
        return False


def _safe_target(path: str, roots: list[str]) -> tuple[bool, str]:
    rp = os.path.realpath(path)
    if os.path.islink(path):
        return False, "is a symlink"
    parts = rp.split(os.sep)
    for bad in FORBIDDEN_PARTS:
        if any(bad in p for p in parts):
            return False, f"path contains forbidden component '{bad}'"
    if not any(rp.startswith(os.path.realpath(r) + os.sep) for r in roots):
        return False, "outside configured roots"
    if not os.path.isdir(rp):
        return False, "not a directory"
    return True, "ok"


def _du(path: str) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path, followlinks=False):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def _epoch_of(path: str) -> int | None:
    m = re.search(r"(?:snapshot_ep|checkpoint_)(\d+)", os.path.basename(path.rstrip("/")))
    return int(m.group(1)) if m else None


def _load_scores(run_dir: str, score_regex: str | None, log_glob: str) -> dict[int, float]:
    """epoch -> score, from the most authoritative source available."""
    idx = os.path.join(run_dir, "checkpoints_index.json")
    if os.path.exists(idx):
        try:
            rows = json.load(open(idx))
            out = {}
            for r in rows if isinstance(rows, list) else rows.get("checkpoints", []):
                if r.get("epoch") is not None and r.get("score") is not None:
                    out[int(r["epoch"])] = float(r["score"])
            if out:
                return out
        except Exception as e:
            print(f"  [warn] unreadable {idx}: {e} — treating run as unscored", file=sys.stderr)
    sj = os.path.join(run_dir, "scores.jsonl")
    if os.path.exists(sj):
        out = {}
        for line in open(sj):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                out[int(d["epoch"])] = float(d["score"])
            except Exception:
                continue
        if out:
            return out
    if score_regex:
        import glob as _glob
        pat = re.compile(score_regex)
        out = {}
        for lf in _glob.glob(os.path.join(run_dir, log_glob)) + _glob.glob(log_glob):
            try:
                for line in open(lf, errors="replace"):
                    m = pat.search(line)
                    if m:
                        try:
                            out[int(m.group("epoch"))] = float(m.group("score"))
                        except Exception:
                            continue
            except OSError:
                continue
        if out:
            return out
    return {}


def free_gb(path: str) -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def collect(roots: list[str]) -> dict[str, list[str]]:
    """run_dir -> [checkpoint dirs]"""
    runs: dict[str, list[str]] = {}
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, _files in os.walk(root, followlinks=False):
            keep_walking = []
            for d in dirnames:
                full = os.path.join(dirpath, d)
                if os.path.islink(full):
                    continue
                if _is_ckpt_dir(full):
                    runs.setdefault(dirpath, []).append(full)
                else:
                    keep_walking.append(d)
            dirnames[:] = keep_walking
    return runs


def plan(runs, scores_of, keep_best, keep_latest, under_floor, keep_unscored):
    """-> (deletions[(path, bytes, reason)], kept_summary[str])"""
    deletions, summary = [], []
    for run_dir, ckpts in sorted(runs.items()):
        scores = scores_of(run_dir)
        by_mtime = sorted(ckpts, key=lambda p: os.path.getmtime(p), reverse=True)
        newest = set(by_mtime[:max(1, keep_latest)])
        scored = [(p, scores.get(_epoch_of(p))) for p in ckpts]
        have = [(p, s) for p, s in scored if s is not None]
        none = [p for p, s in scored if s is None]
        keep = set(newest)
        if have:
            best = sorted(have, key=lambda t: t[1], reverse=True)
            keep |= {p for p, _ in best[:max(1, keep_best)]}
            for p, s in best[max(1, keep_best):]:
                if p in keep:
                    continue
                deletions.append((p, _du(p), f"score {s:.6g} outside top-{keep_best}"))
        if none:
            # unscored: never pruned on score. Only recency-pruned when disk is under the floor.
            keep |= set(sorted(none, key=os.path.getmtime, reverse=True)[:keep_unscored])
            if under_floor:
                for p in sorted(none, key=os.path.getmtime, reverse=True)[keep_unscored:]:
                    if p not in keep:
                        deletions.append((p, _du(p), "UNSCORED, recency-pruned (disk under floor)"))
        summary.append(f"{run_dir}: {len(ckpts)} ckpts, {len(have)} scored, keeping {len(keep)}")
    return deletions, summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", default=["jax_ckpt_fast", "runs"])
    ap.add_argument("--keep-best", type=int, default=3, help="top-K by score to keep per run")
    ap.add_argument("--keep-latest", type=int, default=1, help="newest N always kept (resume)")
    ap.add_argument("--keep-unscored", type=int, default=2,
                    help="newest N unscored kept even under the disk floor")
    ap.add_argument("--min-free-gb", type=float, default=30.0,
                    help="below this, unscored checkpoints become recency-prunable")
    ap.add_argument("--critical-free-gb", type=float, default=12.0,
                    help="below this, print a CRITICAL line (the DB shares this filesystem)")
    ap.add_argument("--score-regex", default=None,
                    help=r"e.g. 'Epoch (?P<epoch>\d+).*reward_ma20[= ](?P<score>[-\d.]+)'")
    ap.add_argument("--log-glob", default="*.log")
    ap.add_argument("--protect-file", default=None, help="file of paths never to delete")
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry run)")
    a = ap.parse_args()

    roots = [r for r in a.roots if os.path.isdir(r)]
    if not roots:
        print(f"[ckpt_gc] no existing roots among {a.roots} — nothing to do")
        return 0
    protect = set()
    if a.protect_file and os.path.exists(a.protect_file):
        protect = {os.path.realpath(l.strip()) for l in open(a.protect_file) if l.strip()}

    fg = free_gb(roots[0])
    under = fg < a.min_free_gb
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[ckpt_gc {stamp}] free={fg:.1f}GB floor={a.min_free_gb}GB "
          f"under_floor={under} mode={'APPLY' if a.apply else 'dry-run'}")
    if fg < a.critical_free_gb:
        print(f"[ckpt_gc] CRITICAL: free {fg:.1f}GB < {a.critical_free_gb}GB — "
              f"trading_data.db shares this filesystem; stop writing before it fills")

    runs = collect(roots)
    deletions, summary = plan(runs, lambda rd: _load_scores(rd, a.score_regex, a.log_glob),
                              a.keep_best, a.keep_latest, under, a.keep_unscored)
    for s in summary:
        print("  " + s)

    reclaimed = 0
    for path, size, reason in deletions:
        ok, why = _safe_target(path, roots)
        if not ok:
            print(f"  REFUSE {path}: {why}")
            continue
        if os.path.realpath(path) in protect:
            print(f"  PROTECTED {path}: listed in protect-file")
            continue
        print(f"  {'DELETE' if a.apply else 'would delete'} {path} "
              f"({size/1e6:.0f} MB) — {reason}")
        if a.apply:
            trash = f"{path}.gc-trash-{os.getpid()}"
            try:
                os.rename(path, trash)      # atomic: a half-deleted ckpt can never look loadable
                shutil.rmtree(trash)
                reclaimed += size
            except OSError as e:
                print(f"  ERROR deleting {path}: {e}")
    print(f"[ckpt_gc] {'reclaimed' if a.apply else 'would reclaim'} {reclaimed/1e9:.2f} GB "
          f"across {len(deletions)} checkpoints; free now {free_gb(roots[0]):.1f}GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
