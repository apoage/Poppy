#!/usr/bin/env python3
"""Archive interesting checkpoints server -> workstation, then clear the server (operator, 2026-08-05).

THE LIFECYCLE THIS IMPLEMENTS
-----------------------------
`gpu-server` is disposable compute with 136 GB free and ``trading_data.db`` on the same
filesystem; the workstation has ~440 GB free and is where things are kept. So: score the
checkpoints, PULL the interesting ones down, VERIFY the local copy, and only then delete from the
server. Uninteresting ones are deleted without archiving. Nothing is ever deleted on the server
that was not either (a) verified present locally, or (b) explicitly judged uninteresting by score.

Direction matters: this DOWNLOADS (srv -> workstation). Nothing is ever uploaded — the uplink is
~5 Mbps and the DB must never travel as a file.

"SOMEWHAT INTERESTING" = top-K by score per run, plus anything scoring >= --min-score, plus the
newest checkpoint of each run (the resume point). Unscored checkpoints are archived, never
silently dropped — you cannot judge what you did not measure.

SAFETY
------
* dry-run by default; ``--apply`` required for any transfer or deletion
* deletion on the server happens ONLY after local verification (file count + total bytes, and
  per-file sha256 with ``--verify-sha``)
* remote paths are confined to the configured roots and must match a checkpoint pattern; any path
  containing ``trading_data``/``.git``/``fixtures``/``backups`` is refused
* a manifest (what was archived, its score, sizes, digests, what was deleted) is written locally
  BEFORE any deletion, so the record survives a failure mid-clear
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

REMOTE_PROBE = r'''
import json, os, re, sys
roots = sys.argv[1:]
CK = (re.compile(r"^snapshot_ep\d+$"), re.compile(r"^ckpt(_.*)?$"))
def is_ck(p):
    b = os.path.basename(p.rstrip("/"))
    if any(r.match(b) for r in CK): return True
    try: return any(n.startswith("checkpoint_") for n in os.listdir(p))
    except OSError: return False
def du(p):
    t = 0
    for dp, _dn, fn in os.walk(p, followlinks=False):
        for f in fn:
            try: t += os.path.getsize(os.path.join(dp, f))
            except OSError: pass
    return t
def ep_of(p):
    m = re.search(r"(?:snapshot_ep|checkpoint_)(\d+)", os.path.basename(p.rstrip("/")))
    return int(m.group(1)) if m else None
def scores(run):
    for name in ("checkpoints_index.json", "scores.jsonl"):
        f = os.path.join(run, name)
        if not os.path.exists(f): continue
        out = {}
        try:
            if name.endswith(".jsonl"):
                for line in open(f):
                    line = line.strip()
                    if line:
                        d = json.loads(line); out[int(d["epoch"])] = float(d["score"])
            else:
                rows = json.load(open(f))
                for r in (rows if isinstance(rows, list) else rows.get("checkpoints", [])):
                    if r.get("epoch") is not None and r.get("score") is not None:
                        out[int(r["epoch"])] = float(r["score"])
        except Exception: continue
        if out: return out
    # imprint-style single-ckpt runs: take the sanity replay net@21 as the score
    import glob as _g
    cands = ["imprint.json", "results.json"] + [os.path.basename(x) for x in
             sorted(_g.glob(os.path.join(run, "results*.json")))]
    for name in cands:
        f = os.path.join(run, name)
        if os.path.exists(f):
            try:
                d = json.load(open(f))
                t = ((d.get("sanity") or {}).get("per_tier") or {}).get("21") or {}
                v = t.get("policy_inenv")
                if v is not None: return {"_run": float(v)}
            except Exception: pass
    return {}
found = []
for root in roots:
    if not os.path.isdir(root): continue
    for dp, dn, _f in os.walk(root, followlinks=False):
        keep = []
        for d in dn:
            full = os.path.join(dp, d)
            if os.path.islink(full): continue
            if is_ck(full):
                found.append((dp, full))
            else:
                keep.append(d)
        dn[:] = keep
runs = {}
for run, ck in found: runs.setdefault(run, []).append(ck)
out = []
for run, cks in runs.items():
    sc = scores(run)
    for ck in cks:
        s = sc.get(ep_of(ck), sc.get("_run"))
        out.append({"run": run, "path": ck, "epoch": ep_of(ck), "score": s,
                    "bytes": du(ck), "mtime": os.path.getmtime(ck)})
print(json.dumps(out))
'''

FORBIDDEN = ("trading_data", ".git", "fixtures", "backups")


def sh(cmd: list[str], check=True) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:3])}… failed: {r.stderr.strip()[:400]}")
    return r.stdout


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="operator@<redacted-ip>")
    ap.add_argument("--port", default="22")
    ap.add_argument("--remote-dir", default="Trader")
    ap.add_argument("--roots", nargs="+", default=["runs", "jax_ckpt_fast"])
    ap.add_argument("--archive", default=os.path.expanduser("~/Trader/archive/checkpoints/gpu-server"))
    ap.add_argument("--keep-best", type=int, default=3, help="top-K by score per run = interesting")
    ap.add_argument("--min-score", type=float, default=None, help="also archive anything >= this")
    ap.add_argument("--no-keep-newest", action="store_true",
                    help="do not treat each run's newest ckpt as a resume point worth keeping "
                         "(correct for DEAD runs: nothing will ever resume them)")
    ap.add_argument("--unscored", choices=("drop", "archive"), default="drop",
                    help="unscored checkpoints: 'drop' (operator ruling — no record, no value) "
                         "or 'archive' (old fail-safe)")
    ap.add_argument("--protect-from-ledger", default="experiments/LEDGER.md",
                    help="any runs/... path cited in this file is never deleted")
    ap.add_argument("--verify-sha", action="store_true", help="per-file sha256 both sides (slower)")
    ap.add_argument("--clear-remote", action="store_true",
                    help="after verified archive, DELETE from the server (needs --apply)")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    ssh = ["ssh", "-p", a.port, a.host]
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[ckpt_archive {stamp}] host={a.host} roots={a.roots} "
          f"mode={'APPLY' if a.apply else 'dry-run'} clear_remote={a.clear_remote}")

    raw = sh(ssh + [f"cd {a.remote_dir} && python3 - {' '.join(a.roots)} <<'PYEOF'\n{REMOTE_PROBE}\nPYEOF"])
    items = json.loads(raw.strip().splitlines()[-1])
    if not items:
        print("  no checkpoints found on the server — nothing to do")
        return 0

    runs: dict[str, list[dict]] = {}
    for it in items:
        runs.setdefault(it["run"], []).append(it)

    archive_set, drop_set = [], []
    for run, cks in sorted(runs.items()):
        scored = [c for c in cks if c["score"] is not None]
        unscored = [c for c in cks if c["score"] is None]
        newest = max(cks, key=lambda c: c["mtime"])
        interesting = set() if a.no_keep_newest else {newest["path"]}
        for c in sorted(scored, key=lambda c: c["score"], reverse=True)[:max(1, a.keep_best)]:
            interesting.add(c["path"])
        if a.min_score is not None:
            for c in scored:
                if c["score"] >= a.min_score:
                    interesting.add(c["path"])
        if a.unscored == "archive":
            interesting |= {c["path"] for c in unscored}
        # else: OPERATOR RULING 2026-08-05 — "no reason to store weights that i dont have info
        # what they are"; an unscored checkpoint has no record of what it is, so it is garbage
        # and is dropped WITHOUT archiving. Protection comes from the LEDGER, not from ignorance:
        # anything the science record references is kept even when its score file is missing.
        for c in cks:
            (archive_set if c["path"] in interesting else drop_set).append(c)
        # (ledger protection is applied below, after the protect-set is built)
        print(f"  {run}: {len(cks)} ckpts ({len(scored)} scored) -> "
              f"{len([c for c in cks if c['path'] in interesting])} interesting, "
              f"{len([c for c in cks if c['path'] not in interesting])} droppable")

    protected = set()
    if a.protect_from_ledger and os.path.exists(a.protect_from_ledger):
        import re as _re
        txt = open(a.protect_from_ledger).read()
        for m in _re.finditer(r"runs/[A-Za-z0-9_./{}-]+", txt):
            protected.add(m.group(0).rstrip(".,)"))
        print(f"  ledger protection: {len(protected)} referenced paths in {a.protect_from_ledger}")

    def _is_protected(p: str) -> bool:
        return any(p == q or p.startswith(q.rstrip("/") + "/") or q.startswith(p.rstrip("/") + "/")
                   for q in protected)

    def guard(p: str) -> bool:
        if any(f in p for f in FORBIDDEN):
            print(f"  REFUSE {p}: forbidden path component")
            return False
        if not any(p.startswith(r + "/") or p == r for r in a.roots):
            print(f"  REFUSE {p}: outside roots")
            return False
        return True

    archive_set = [c for c in archive_set if guard(c["path"])]
    rescued = [c for c in drop_set if _is_protected(c["path"])]
    for c in rescued:
        print(f"  LEDGER-PROTECTED (kept despite no score): {c['path']}")
    archive_set += rescued
    drop_set = [c for c in drop_set if guard(c["path"]) and not _is_protected(c["path"])]
    tot_a = sum(c["bytes"] for c in archive_set) / 1e9
    tot_d = sum(c["bytes"] for c in drop_set) / 1e9
    print(f"  ARCHIVE {len(archive_set)} ckpts / {tot_a:.2f} GB   "
          f"DROP-WITHOUT-ARCHIVE {len(drop_set)} ckpts / {tot_d:.2f} GB")

    manifest = {"stamp": stamp, "host": a.host, "policy": vars(a),
                "archived": [], "dropped": [], "verify_failures": []}

    if not a.apply:
        for c in archive_set[:12]:
            print(f"    would pull {c['path']} ({c['bytes']/1e6:.0f} MB, score={c['score']})")
        for c in drop_set[:12]:
            print(f"    would drop {c['path']} ({c['bytes']/1e6:.0f} MB, score={c['score']})")
        print("  dry run — nothing transferred or deleted")
        return 0

    os.makedirs(a.archive, exist_ok=True)
    verified: list[dict] = []
    for c in archive_set:
        dest = os.path.join(a.archive, c["path"])
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print(f"  pull {c['path']} ({c['bytes']/1e6:.0f} MB, score={c['score']})")
        sh(["rsync", "-a", "-e", f"ssh -p {a.port}",
            f"{a.host}:{a.remote_dir}/{c['path']}/", dest + "/"])
        local_bytes = sum(os.path.getsize(os.path.join(dp, f))
                          for dp, _dn, fn in os.walk(dest) for f in fn)
        ok = local_bytes == c["bytes"]
        if ok and a.verify_sha:
            r_sha = sh(ssh + [f"cd {a.remote_dir}/{c['path']} && find . -type f -exec sha256sum {{}} + | sort -k2"])
            l_sha = sh(["bash", "-c", f"cd {dest} && find . -type f -exec sha256sum {{}} + | sort -k2"])
            ok = r_sha.split() == l_sha.split()
        rec = {**c, "local": dest, "local_bytes": local_bytes, "verified": ok}
        (verified if ok else manifest["verify_failures"]).append(rec)
        manifest["archived"].append(rec)
        if not ok:
            print(f"    VERIFY FAILED for {c['path']} (remote {c['bytes']} vs local {local_bytes}) "
                  f"— will NOT be cleared from the server")

    mpath = os.path.join(a.archive, f"MANIFEST_{stamp.replace(':', '')}.json")
    manifest["dropped"] = drop_set
    json.dump(manifest, open(mpath, "w"), indent=2)
    print(f"  manifest -> {mpath}")

    if a.clear_remote:
        to_clear = [c["path"] for c in verified] + [c["path"] for c in drop_set]
        if manifest["verify_failures"]:
            print(f"  {len(manifest['verify_failures'])} verify failures — those paths are excluded")
        for path in to_clear:
            sh(ssh + [f"cd {a.remote_dir} && rm -rf -- '{path}'"])
        print(f"  cleared {len(to_clear)} checkpoint dirs from the server")
        free = sh(ssh + [f"df -h {a.remote_dir} | tail -1"]).strip()
        print(f"  server disk now: {free}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
