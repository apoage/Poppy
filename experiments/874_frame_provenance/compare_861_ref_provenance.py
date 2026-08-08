#!/usr/bin/env python3
"""TASK_874 scope 3 — READ-ONLY comparison of the three hand-derived 861 reference provenances
against what ``imprint.py``'s emitter would now produce.

On 2026-08-03 the planner hand-derived ``runs/861_ref/frame_provenance_ref_s{0,1,2}.json`` from
the leg artifacts because ``imprint.py`` had no emitter (the gap TASK_874 closes). Scope 3 asks
whether those three files agree with the emitter, and to REPORT any difference rather than
overwrite. The first implementation pass did that ad hoc in-process and left nothing auditable;
this is the auditable entry point.

Method (nothing is re-derived by hand here):

  * for each seed, read ``runs/861_ref/results_s<i>.json`` — the leg's OWN training artifact,
    which records the window, db, n_assets, the pin it trained under, the pinned frame it
    actually walked and the panel frame SOURCE;
  * rebuild the emitter's input ``frame`` block from that artifact and call
    ``imprint.build_frame_provenance`` DIRECTLY (the shipped function, not a copy), so the
    comparison also exercises its scope-2 re-read of the pin
    ``runs/832_e5/frame_provenance_imprint.json``;
  * diff the resulting document field-by-field against the file on disk and print the result.

READ-ONLY, and it proves it: ``build_frame_provenance`` writes nothing by construction (the
write lives in ``write_frame_provenance``, which is never called here), and this script sha256s
every file in ``runs/861_ref/`` before and after the comparison and FAILS if a single byte moved.

Run: ``python experiments/874_frame_provenance/compare_861_ref_provenance.py``
Exit 0 = comparison completed (differences are REPORTED, not failed on — a difference here is a
planner read, not a test result). Exit non-zero only if an input is missing/unreadable, the
emitter REFUSED, or this script somehow perturbed a file.
"""
import hashlib
import importlib.util
import json
import os
import sys
from types import SimpleNamespace

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(EXP_DIR))
IMPRINT_PY = os.path.join(_REPO, "experiments", "832_linear_imprint", "imprint.py")
REF_DIR = os.path.join(_REPO, "runs", "861_ref")
PIN_REL = "runs/832_e5/frame_provenance_imprint.json"
SEEDS = (0, 1, 2)

# The `source` NOTE is expected to differ: the on-disk files carry the planner's hand-derivation
# note, the emitter carries its own. The task words scope 3 as "byte-for-byte modulo the `source`
# note", so it is reported separately from the substantive fields.
EXPECTED_DIFF_FIELDS = ("source",)


def _sha_tree(d):
    out = {}
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            with open(p, "rb") as fh:
                out[name] = hashlib.sha256(fh.read()).hexdigest()
    return out


def _imprint_module():
    spec = importlib.util.spec_from_file_location("imprint832_cmp874", IMPRINT_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _emitter_doc(imp, res, ckpt_path):
    """What ``imprint.py`` would emit for this leg, from the leg's own artifact."""
    args = SimpleNamespace(
        panel_frame_from_checkpoint=os.path.join(_REPO, res["panel_frame_from_checkpoint"]),
        panel_frame_source=res.get("panel_frame_source", "daily"),
        n_assets=int(res["n_assets"]),
        start=res["window"][0], end=res["window"][1],
        db=res["db"],
    )
    frame = {"frame_symbols": list(res["frame_symbols"])}
    _path, doc = imp.build_frame_provenance(ckpt_path, args, frame, None)
    return doc


def compare_seed(imp, seed):
    res_path = os.path.join(REF_DIR, f"results_s{seed}.json")
    prov_path = os.path.join(REF_DIR, f"frame_provenance_ref_s{seed}.json")
    with open(res_path) as fh:
        res = json.load(fh)
    with open(prov_path) as fh:
        on_disk = json.load(fh)
    # The legs ran on gpu-server, so results.checkpoint is that HOST's absolute path
    # (/home/operator/Trader/...). The emitter records a repo-relative path, so re-home it here;
    # the host difference is a deployment fact, not a frame fact, and is reported as such.
    ckpt_recorded = res["checkpoint"]
    ckpt_local = os.path.join(_REPO, "runs", "861_ref", f"ckpt_ref_s{seed}")
    doc = _emitter_doc(imp, res, ckpt_local)

    keys = sorted(set(on_disk) | set(doc))
    same, expected, unexpected = [], [], []
    for k in keys:
        if k in on_disk and k in doc and on_disk[k] == doc[k]:
            same.append(k)
        elif k in EXPECTED_DIFF_FIELDS:
            expected.append(k)
        else:
            unexpected.append(k)

    print(f"\n--- seed {seed}: {os.path.relpath(prov_path, _REPO)} vs the emitter")
    print(f"    artifact: {os.path.relpath(res_path, _REPO)}  "
          f"(pin {res['panel_frame_from_checkpoint']}, source "
          f"{res.get('panel_frame_source', 'daily')})")
    for k in same:
        v = on_disk[k]
        shown = (f"[{len(v)} symbols, first {v[0]!r} last {v[-1]!r}]"
                 if isinstance(v, list) and k == "frame_symbols" else repr(v))
        print(f"    IDENTICAL  {k}: {shown}")
    for k in expected:
        print(f"    DIFFERS (expected, the hand-derivation note) {k}:")
        print(f"        on disk : {str(on_disk.get(k))[:260]}")
        print(f"        emitter : {str(doc.get(k))[:260]}")
    for k in unexpected:
        if k not in on_disk:
            print(f"    ADDED BY THE EMITTER  {k} = {doc[k]!r} (absent from the hand-derived file)")
        elif k not in doc:
            print(f"    ONLY ON DISK  {k} = {on_disk[k]!r}")
        else:
            print(f"    DIFFERS  {k}:")
            print(f"        on disk : {str(on_disk[k])[:260]}")
            print(f"        emitter : {str(doc[k])[:260]}")
    print(f"    NOTE checkpoint path: the artifact records {ckpt_recorded!r} (gpu-server "
          f"absolute); the comparison used the repo-local {os.path.relpath(ckpt_local, _REPO)!r}, "
          f"which is what the on-disk provenance carries.")
    frame_ok = on_disk.get("frame_symbols") == doc["frame_symbols"]
    print(f"    => frame_symbols order-exact match: {frame_ok}; "
          f"{len(same)} field(s) identical, {len(expected)} expected difference(s), "
          f"{len(unexpected)} other difference(s)")
    return {"seed": seed, "identical_fields": same, "expected_differences": expected,
            "other_differences": unexpected, "frame_symbols_order_exact": frame_ok}


def main():
    missing = [p for p in
               [os.path.join(_REPO, PIN_REL)]
               + [os.path.join(REF_DIR, f"results_s{s}.json") for s in SEEDS]
               + [os.path.join(REF_DIR, f"frame_provenance_ref_s{s}.json") for s in SEEDS]
               if not os.path.exists(p)]
    if missing:
        raise SystemExit("[874 scope-3] missing input(s):\n  "
                         + "\n  ".join(os.path.relpath(p, _REPO) for p in missing))
    before = _sha_tree(REF_DIR)
    print("=" * 88)
    print("TASK_874 scope 3 — hand-derived 861 reference provenances vs the imprint.py emitter")
    print(f"  reference dir : {os.path.relpath(REF_DIR, _REPO)} (READ-ONLY; "
          f"{len(before)} file(s) sha256'd before and after)")
    print(f"  pin           : {PIN_REL}")
    print("=" * 88)
    imp = _imprint_module()
    rows = [compare_seed(imp, s) for s in SEEDS]

    after = _sha_tree(REF_DIR)
    moved = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    print("\n" + "=" * 88)
    print("SUMMARY")
    for r in rows:
        print(f"  seed {r['seed']}: frame_symbols order-exact {r['frame_symbols_order_exact']}; "
              f"identical {sorted(r['identical_fields'])}; expected-diff "
              f"{sorted(r['expected_differences'])}; other-diff "
              f"{sorted(r['other_differences'])}")
    print(f"  runs/861_ref untouched: {not moved}" + (f" (MOVED: {moved})" if moved else ""))
    print("=" * 88)
    if moved:
        raise SystemExit("[874 scope-3] FATAL: this read-only comparison perturbed "
                         f"runs/861_ref: {moved}")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
