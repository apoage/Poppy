#!/usr/bin/env python3
"""TASK_863 — fixture smoke for the imprint frame pin (``--panel-frame-from-checkpoint``).

Proves the task's acceptance gates on ``fixtures/trading_fixture.db`` only (CPU, ~80-iteration
imprints; NEVER trading_data.db):

  (i)   PARITY (lite) — a legacy run (no frame flag, unpinned collection) carries NO frame keys in
        its artifact. The BINDING parity evidence is the byte-level shipped-vs-post comparison run
        by hand against git tag ``task861-search-teacher-imprint`` (both legacy artifacts came out
        BYTE-IDENTICAL); this leg is the cheap in-repo guard that the additive blocks stay additive.
  (ii)  END-TO-END PASS + DISCLOSURE — a collection recorded by ``eval_vector.py
        --panel-frame-from-checkpoint`` imprints clean under the MATCHING pin, all the way through
        the policy-argmax sanity replay, and the inactive-slot disclosure lands in the imprint
        artifact. The pin is supplied via a DIFFERENT PATH to the same frame (the spec allows the
        path to differ; only the ordered symbol list may not). The replay-integrity check passing
        is the alignment proof: eval's masked walk and imprint's masked walk score identically.
  (iii) POSITIONAL HONESTY — a deliberately REORDERED, same-membership frame is REFUSED. Order is
        load-bearing (slot k IS asset k) and membership-only matching is the named failure mode.

Plus the fail-closed matrix scope 2 demands:
  * pinned collection + UNPINNED imprint          -> REFUSED, both directions named
  * UNPINNED collection + pinned imprint          -> REFUSED, both directions named
  * pinned collection + DIFFERENT-MEMBERSHIP pin  -> REFUSED
  * ``--panel-frame-from-checkpoint ""``          -> fails closed before any walk
  * frame pin at ``--bar-minutes`` != 1440        -> fails closed (the frame is read from
    ``ohlcv_daily``, the shipped 853 rule)

FIX-PASS legs (2026-07-28 verification panel), all IN-PROCESS (no training — they exercise the
loader / mask / gate directly, so they cost seconds and can assert exact cell sets):
  * FIX E — PARTIAL-slot mask: ``MONUSD`` (first ``ohlcv_daily`` row 2025-11-24 = panel row 71 of
    90) at slot 1 gives the exact masked set ``hold cells per slot [0 11 0 0 29 0 0 0]``, i.e.
    decisions k=0..10 force-HOLD at that slot and k=11 (the first real close) TRADEABLE. The
    boundary is asserted BOTH ways (k=10 held, k=11 free) and the expectation is shown to be
    sensitive to a one-row shift, so an off-by-one in ``frame_hold_mask`` fails this leg. No
    previous leg had a partial slot at all, and on the ``--teacher-vectors`` path the mask is
    behaviorally inert (eval's collector already recorded HOLD at those cells).
  * FIX C — inactive-slot FEATURE guard: the encoded features of a dead slot are asserted CONSTANT
    across all decisions and non-zero ONLY on the guarded vol-ratio channels 8-11 (0.2), plus TWO
    negative controls (a decision-varying cell / a non-zero cell on another channel must RAISE),
    so the claim has teeth instead of being a comment.
  * FIX B/D — provenance-gate matrix on the panel_source / panel_frame header fields: a
    frame-matched pinned pair recording ``panel_source: "daily"`` is ACCEPTED (the framed loader is
    dispatched before that branch, so the panel is bit-identical either way), a bogus source is
    refused with the pinned message, the UNPINNED refusal text is byte-unchanged, and a truthy
    NON-DICT ``panel_frame`` is a NAMED refusal instead of an AttributeError traceback.

ROUND-2 legs (2026-07-29 fail-closed judge):
  * FIX 2.5 — the degeneracy guard FIRING. Two POSITIVE legs (the guard previously shipped with
    negative evidence only, which is how FIX 2.1's hole survived): a teacher-side degenerate run
    (all-HOLD teacher, finite student) and a student-side one (all-HOLD teacher LEARNED), each
    asserting exit non-zero + ``sanity.degenerate_replay: true`` + a non-empty reason naming the
    RIGHT diagnosis + the ``DEGENERATE_REPLAY.json`` checkpoint stamp.
  * FIX 2.2/2.3/2.4 — extra cases on the FIX B/D in-process gate leg: a wrong-TYPED ``slot``
    (string and null) and a duplicate-slot coverage are NAMED refusals instead of raw
    ValueError/TypeError or false positional "proof"; a PRESENT but EMPTY ``panel_frame`` object is
    malformed rather than silently "unpinned"; an ABSENT or ``null`` ``panel_frame`` still reads as
    LEGACY.

Run: ``python experiments/863_imprint_frame_pin/smoke_frame_pin.py``
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(EXP_DIR))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
import fixtures.fixture_identity as fixture_identity          # noqa: E402  (TASK_873)

RUN_DIR = os.path.join(EXP_DIR, "smoke_runs")
EVAL_PY = os.path.join(_REPO, "experiments", "835_stage4_eval", "eval_vector.py")
IMPRINT_PY = os.path.join(_REPO, "experiments", "832_linear_imprint", "imprint.py")
FIXTURE_DB = os.path.join(_REPO, "fixtures", "trading_fixture.db")
# The 861 smoke's fresh-init teacher checkpoint (the pinned operator is prior_weight 0 + true-env
# leaf + no v-hat tail => net_dependent False, so the weights are never read; the checkpoint only
# satisfies the harness' load path). Reused, not re-minted.
CKPT_SRC = os.path.join(_REPO, "experiments", "861_search_teacher_imprint", "ckpt_teacher_src")
LEGACY_COLLECTION = os.path.join(_REPO, "experiments", "861_search_teacher_imprint",
                                 "chosen_vectors_fixture.json")

# The pinned-frame fixture panel. --bar-minutes 1440 is REQUIRED by the frame pin (the frame is
# read from ohlcv_daily). Over 2025-09-14..2025-12-12 the fixture's daily table gives T=90 rows
# (TASK_869: was 89 under local-time window resolution — UTC includes the end date's own
# midnight row; inclusivity unchanged, only the timezone moved); window_size 60 and the 819
# book's warm 31 put start_row at 60 => 29 decisions on both legs.
FRAME_PANEL = ["--start", "2025-09-14", "--end", "2025-12-12", "--bar-minutes", "1440",
               "--n-assets", "8"]
# The legacy (unpinned) fixture panel the 861 collection was recorded on — minute-sourced.
LEGACY_PANEL = ["--start", "2025-09-14", "--end", "2025-12-12", "--bar-minutes", "240",
                "--n-assets", "8"]
COLLECT_POSITION_SIZE_PCT = 0.1          # teacher of record (conditionally exempt under weight)

# 8 slots: 6 with full ohlcv_daily coverage in the window and 2 (`/EURGBP`, `/USDCAD`) with ZERO —
# deliberately interleaved at slots 1 and 4 so a slot-shift cannot coincidentally line up. This
# mirrors the real leg's situation (symbols in runs/832_e5/frame_provenance_imprint.json that lost
# coverage after the 858 backfill ride the 853 inactive-slot mechanics).
FRAME_SYMBOLS = ["XBT/USD", "/EURGBP", "ETH/USD", "LTC/USD",
                 "/USDCAD", "XMR/USD", "ZEC/USD", "ETC/USD"]
EXPECT_INACTIVE_SLOTS = [1, 4]
# FIX E: the SAME frame with a PARTIAL slot in place of one dead one. MONUSD's first ohlcv_daily
# row in the window is 2025-11-24 = panel row 71 of 90 (interior — unaffected by the TASK_869
# end-row shift); start_row is 60, so decisions k=0..10 have no real price at that slot and
# k=11 is its first tradeable decision. The dead slot masks all N=29 decisions (was 28 pre-869).
PARTIAL_SYMBOL = "MONUSD"
PARTIAL_SLOT = 1
PARTIAL_FRAME_SYMBOLS = [PARTIAL_SYMBOL if i == PARTIAL_SLOT else s
                         for i, s in enumerate(FRAME_SYMBOLS)]
EXPECT_PARTIAL_HOLD_PER_SLOT = [0, 11, 0, 0, 29, 0, 0, 0]
EXPECT_PARTIAL_FIRST_FREE_K = 11             # k=10 held, k=11 tradeable (the off-by-one boundary)
EXPECT_PARTIAL_FIRST_COVERED_ROW = 71
# FIX C: the ONLY channels a flat-zero (dead) slot may encode non-zero, and the value they carry
# (clip(1.0,0,5)/5 from the guarded 0/0 volume ratio). Measured, not assumed.
EXPECT_DEAD_SLOT_CHANNELS = [8, 9, 10, 11]
EXPECT_DEAD_SLOT_VALUE = 0.2

TEACHER_OPERATOR = [
    "--operator", "search", "--vector-proposal-scheme", "dithered-book",
    "--vector-k", "16", "--sims", "32", "--vector-hold-bias", "3.0",
    "--vector-dither-scale", "1.0", "--vector-proposal-temp", "1.0",
    "--vector-proposal-explore-slots", "2", "--prior-weight", "0.0",
    "--leaf-eval", "true-env", "--leaf-rollout-r", "3", "--no-leaf-vhat-tail",
    "--gumbel-scale", "3.0", "--vector-weight-rebalance",
]


def sh(cmd, log, expect_ok=True):
    t0 = time.time()
    print(f"  $ {' '.join(str(c) for c in cmd[1:])[:170]}")
    p = subprocess.run([str(c) for c in cmd], cwd=_REPO, capture_output=True, text=True)
    text = p.stdout + p.stderr
    with open(log, "w") as fh:
        fh.write(text)
    print(f"    -> exit {p.returncode} in {time.time() - t0:.1f}s  ({os.path.basename(log)})")
    if expect_ok and p.returncode != 0:
        print(text[-3000:])
        raise SystemExit(f"FATAL: expected success but got exit {p.returncode}: {log}")
    return p.returncode, text


def write_frame(path, symbols, note):
    with open(path, "w") as fh:
        json.dump({"frame_symbols": list(symbols), "n_assets": len(symbols),
                   "train_window": ["2025-09-14", "2025-12-12"],
                   "db": "fixtures/trading_fixture.db", "source": note}, fh, indent=1)
    return path


# D30 rider (b) reaches the DAILY pin as of the fix pass, and this suite's fixture frames are
# intentionally 6/8 (two zero-coverage slots ARE the subject of the inactive-slot legs), which is
# below the shared 0.9 floor. Every pinned leg here opts out EXPLICITLY — the default stays 0.9 for
# the real legs. (Unpinned legs never reach the guard, but passing it is harmless and keeps one argv
# shape.)
REQUIRE_ACTIVE_OVERRIDE = ["--require-active-slots", "0"]


def eval_cmd(out_dir, collect, frame=None):
    cmd = [sys.executable, EVAL_PY, "--db", FIXTURE_DB, *FRAME_PANEL,
           "--position-size-pct", str(COLLECT_POSITION_SIZE_PCT), *TEACHER_OPERATOR,
           "--checkpoint", CKPT_SRC, "--out-dir", out_dir, "--tag", "863fx",
           "--collect-chosen-vectors", collect, *REQUIRE_ACTIVE_OVERRIDE]
    if frame is not None:
        cmd += ["--panel-frame-from-checkpoint", frame]
    return cmd


def imprint_cmd(out_json, ckpt_out, collection, frame=None, panel=None, extra=()):
    cmd = [sys.executable, IMPRINT_PY, "--db", FIXTURE_DB, *(panel or FRAME_PANEL),
           "--vector-weight-rebalance", "--teacher-vectors", collection,
           "--allow-off-pin-teacher", "--iters", "80", "--seed", "0",
           "--ckpt-out", ckpt_out, "--out", out_json, *REQUIRE_ACTIVE_OVERRIDE]
    if frame is not None:
        cmd += ["--panel-frame-from-checkpoint", frame]
    return cmd + list(extra)


def degenerate_leg(name, R, frame, extra, expect_clauses, absent_clauses,
                   expect_null, expect_finite):
    """FIX 2.5 (round 2) — POSITIVE-firing evidence for the FIX 0 / FIX 2.1 degeneracy guard.

    The guard shipped with NEGATIVE evidence only (a healthy run carries the marker as false, and
    the legacy artifact has no such key), so nothing in the deliverable ever showed it FIRING —
    which is exactly how FIX 2.1's hole (only ``policy_inenv`` examined) survived a fix pass. This
    leg runs a deliberately DEGENERATE pinned imprint and asserts the whole refusal contract:
    exit non-zero, ``sanity.degenerate_replay: true`` + a non-empty reason in the artifact, the
    ``DEGENERATE_REPLAY.json`` stamp in the checkpoint dir, and — the FIX 2.1 point — that the
    reason names the RIGHT diagnosis (student-side collapse vs a teacher that never opened a
    position) rather than one boilerplate string for both.
    """
    out, ck = R(f"results_{name}.json"), R(f"ckpt_{name}")
    cmd = [sys.executable, IMPRINT_PY, "--db", FIXTURE_DB, *FRAME_PANEL,
           "--vector-weight-rebalance", "--imprint-mapping", "weight-native",
           "--batch-size", "32", "--panel-frame-from-checkpoint", frame,
           "--out", out, "--ckpt-out", ck, *REQUIRE_ACTIVE_OVERRIDE, *extra]
    rc, text = sh(cmd, R(f"log_{name}.txt"), expect_ok=False)
    res = json.load(open(out)) if os.path.exists(out) else {}
    sn = res.get("sanity") or {}
    tier = (sn.get("per_tier") or {}).get("21") or {}
    reason = sn.get("degenerate_replay_reason") or ""
    stamp_path = os.path.join(ck, "DEGENERATE_REPLAY.json")
    stamp = json.load(open(stamp_path)) if os.path.exists(stamp_path) else None
    checks = {
        "exit_non_zero": rc != 0,
        "refusal_printed": "[863] REFUSED" in text,
        "artifact_written_for_forensics": bool(res),
        "artifact_degenerate_replay_TRUE": sn.get("degenerate_replay") is True,
        "reason_non_empty": len(reason.strip()) > 80,
        "ckpt_stamp_present": stamp is not None,
        "ckpt_stamp_degenerate_true": bool(stamp) and stamp.get("degenerate_replay") is True,
        "ckpt_stamp_carries_reason": bool((stamp or {}).get("reason", "").strip()),
        "ledger_metric_is_null": (res.get("ledger_row") or {}).get(
            "imprint_corridor_position_net21") is None,
        "no_smoke_pass_banner": "SMOKE PASS" not in text,
    }
    for f in expect_null:                 # the fields the verdict claims are null really are
        checks[f"net21_{f}_is_null"] = tier.get(f, "missing") is None
    for f in expect_finite:               # ... and the ones it does not claim are NOT null
        checks[f"net21_{f}_is_finite"] = isinstance(tier.get(f), float)
    for c in expect_clauses:
        checks[f"reason_names_{c.replace(' ', '_')}"] = c in reason
    for c in absent_clauses:              # the diagnoses must be DISTINCT, not boilerplate
        checks[f"reason_omits_{c.replace(' ', '_')}"] = c not in reason
    for k, v in checks.items():
        print(f"    {'OK  ' if v else 'FAIL'} {k}")
    print(f"    reason: {reason[:300]}...")
    return {"leg": name, "passed": all(checks.values()), "checks": checks,
            "exit_code": rc, "reason": reason, "log": f"log_{name}.txt"}


def refusal_leg(name, needle, cmd, log):
    """A leg that MUST exit non-zero with ``needle`` in its output and train nothing."""
    rc, text = sh(cmd, log, expect_ok=False)
    hit = needle in text
    trained = "SMOKE PASS" in text
    # Every frame refusal must also state the RULE (both directions), not just what went wrong.
    rule = ("BOTH legs pin the frame, or NEITHER does" in text
            or "ordered symbol list may not" in text
            or "EMPTY path" in text or "requires --bar-minutes 1440" in text)
    ok = bool(rc != 0 and hit and rule and not trained)
    print(f"    {name}: exit {rc}, '{needle}' present={hit}, rule-stated={rule}, "
          f"trained={trained} -> {'REFUSED (PASS)' if ok else 'LEG FAILED'}")
    return {"leg": name, "passed": ok, "exit_code": rc, "needle": needle,
            "needle_present": hit, "rule_stated": rule, "trained": trained,
            "log": os.path.basename(log)}


# =============================================================================================
# fix-pass IN-PROCESS legs (FIX E / FIX C / FIX B+D) — loader, mask, feature guard and provenance
# gate exercised directly: seconds instead of an imprint, and exact cell sets can be asserted.
# =============================================================================================

def _imprint_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("imprint832_smoke863", IMPRINT_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pin_args(imp, frame_path, n_assets=8, bar_minutes=1440):
    """The namespace ``load_framed_panel`` / ``gate_teacher_vectors`` read, at the SAME values the
    subprocess legs use (imprint.py CLI defaults + FRAME_PANEL + --vector-weight-rebalance)."""
    from types import SimpleNamespace
    return SimpleNamespace(
        db=FIXTURE_DB, start="2025-09-14", end="2025-12-12", bar_minutes=bar_minutes,
        n_assets=n_assets, fee_bps=16.0, slip_bps=5.0,
        position_size_pct=0.5 / n_assets, reward_attributable=True, reward_hybrid=True,
        vector_weight_rebalance=True, vector_weight_unit=-1.0, vector_weight_max=4.0,
        vector_rebalance_deadband=-1.0, seed=0, max_decisions=None,
        allow_off_pin_teacher=True, shuffle_teacher_vectors=False,
        panel_frame_from_checkpoint=frame_path,
        # D30 rider (b) now applies to the DAILY pin too (fix pass). This suite's frames are
        # INTENTIONALLY inert (2 of 8 slots have zero ohlcv_daily coverage — that is what the
        # inactive-slot mechanics legs exist to exercise), i.e. 6/8 = 0.75 below the shared 0.9
        # floor. The suite therefore opts out EXPLICITLY, exactly as the 865 minute fixture legs
        # do; the DEFAULT is not weakened for the real legs.
        require_active_slots=0,
    )


def _framed_panel(imp, frame_path, n_assets=8):
    """(args, prices_j, env_cfg, frame, start_row, N, T) for a pinned panel — the same start_row /
    N arithmetic ``run()`` does, so the mask rows are the rows the imprint would walk."""
    args = _pin_args(imp, frame_path, n_assets)
    prices_j, logc, env_cfg, frame = imp.load_framed_panel(args)
    T = logc.shape[0]
    corridor = imp._load_corridor()
    _tgt, _pres, warm = corridor.book_target_signs(logc)
    start_row = max(int(warm), int(env_cfg.window_size))
    N = T - 1 - start_row
    return args, prices_j, env_cfg, frame, start_row, N, T


def leg_partial_slot_mask(R, imp):
    """FIX E — a PARTIAL slot's exact masked row set, asserted with the off-by-one boundary."""
    import numpy as np
    print("\n=== FIX E: PARTIAL-slot force-HOLD mask (exact row set + boundary) ===")
    fp = write_frame(R("frame_partial.json"), PARTIAL_FRAME_SYMBOLS,
                     "TASK_863 FIX E partial-slot frame (MONUSD lists mid-window)")
    args, prices_j, env_cfg, frame, start_row, N, T = _framed_panel(imp, fp)
    mask = imp.frame_hold_mask(prices_j, start_row, N)
    per_slot = [int(x) for x in np.asarray(mask).sum(axis=0)]
    held_rows = sorted(int(k) for k in np.nonzero(np.asarray(mask)[:, PARTIAL_SLOT])[0])
    cov = {int(c["slot"]): c for c in frame["coverage"]}
    part = cov[PARTIAL_SLOT]
    obs = imp.encode_training_obs(prices_j, start_row, N, env_cfg)
    # A one-row shift of the mask must NOT satisfy the expectation (the guard has teeth). NOTE the
    # per-slot COUNTS are invariant under a roll — the discriminating quantity is the exact held-row
    # SET at the partial slot (and hence the k=10/k=11 boundary), which is what the leg asserts.
    shifted_rows = sorted(int(k) for k in
                          np.nonzero(np.roll(np.asarray(mask), 1, axis=0)[:, PARTIAL_SLOT])[0])
    checks = {
        "panel_rows_90_decisions_29": (T == 90 and N == 29 and start_row == 60),
        "one_partial_slot_declared": frame["partial_slots"] == [PARTIAL_SLOT],
        "partial_slot_absent_from_inactive_slots": PARTIAL_SLOT not in frame["inactive_slots"],
        "partial_status_and_first_covered_row": (
            part.get("status") == "partial"
            and part.get("leading_inactive_rows") == EXPECT_PARTIAL_FIRST_COVERED_ROW),
        "hold_cells_per_slot_exact": per_slot == EXPECT_PARTIAL_HOLD_PER_SLOT,
        "held_rows_are_k0_to_k10": held_rows == list(range(EXPECT_PARTIAL_FIRST_FREE_K)),
        "boundary_k10_HELD": bool(mask[EXPECT_PARTIAL_FIRST_FREE_K - 1, PARTIAL_SLOT]),
        "boundary_k11_TRADEABLE": not bool(mask[EXPECT_PARTIAL_FIRST_FREE_K, PARTIAL_SLOT]),
        "first_real_close_at_panel_row_71": bool(
            np.asarray(prices_j)[EXPECT_PARTIAL_FIRST_COVERED_ROW, PARTIAL_SLOT] > 0
            and np.asarray(prices_j)[EXPECT_PARTIAL_FIRST_COVERED_ROW - 1, PARTIAL_SLOT] == 0),
        "obs_finite_no_nan_inf": bool(np.isfinite(obs).all()),
        "one_row_shift_would_FAIL": shifted_rows != list(range(EXPECT_PARTIAL_FIRST_FREE_K)),
    }
    for k, v in checks.items():
        print(f"    {'OK  ' if v else 'FAIL'} {k}")
    print(f"    hold cells per slot {per_slot} (expected {EXPECT_PARTIAL_HOLD_PER_SLOT}); "
          f"partial slot {PARTIAL_SLOT} held rows k={held_rows[0]}..{held_rows[-1]}, "
          f"k={EXPECT_PARTIAL_FIRST_FREE_K} tradeable; a one-row shift would give held rows "
          f"k={shifted_rows[0]}..{shifted_rows[-1]} (rejected)")
    return {"leg": "e_partial_slot_mask", "passed": all(checks.values()), "checks": checks,
            "hold_cells_per_slot": per_slot, "partial_held_rows": held_rows}


def leg_inactive_feature_guard(R, imp):
    """FIX C — the dead-slot feature claim, ASSERTED, with two negative controls."""
    import numpy as np
    print("\n=== FIX C: inactive-slot ENCODED-feature guard (+ negative controls) ===")
    fp = write_frame(R("frame_featureguard.json"), FRAME_SYMBOLS,
                     "TASK_863 FIX C feature-guard frame")
    args, prices_j, env_cfg, frame, start_row, N, T = _framed_panel(imp, fp)
    obs = imp.encode_training_obs(prices_j, start_row, N, env_cfg)
    rep = imp.check_inactive_slot_features(obs, frame["inactive_slots"])
    slot = int(frame["inactive_slots"][0])
    vals = sorted({v for s in rep["checked_slots"]
                   for v in s["nonzero_channel_values"].values()})

    def raises(mutate, needle):
        bad = np.array(obs, copy=True)
        mutate(bad)
        try:
            imp.check_inactive_slot_features(bad, frame["inactive_slots"])
        except RuntimeError as e:
            hit = needle in str(e)
            print(f"    negative control RAISED as required ('{needle}' present={hit})")
            return hit
        print(f"    negative control did NOT raise -> the guard is toothless")
        return False

    checks = {
        "both_dead_slots_checked": len(rep["checked_slots"]) == len(frame["inactive_slots"]) == 2,
        "constant_across_all_decisions": all(s["constant_across_decisions"]
                                            for s in rep["checked_slots"]),
        "nonzero_only_on_vol_ratio_channels": all(
            s["nonzero_channels"] == EXPECT_DEAD_SLOT_CHANNELS for s in rep["checked_slots"]),
        "nonzero_value_is_0.2_not_zero": bool(
            len(vals) == 1 and abs(vals[0] - EXPECT_DEAD_SLOT_VALUE) < 1e-6),
        "n_nonzero_cells_is_a_minority_not_zero": all(
            0 < s["n_nonzero_feature_cells"] < s["n_feature_cells"]
            for s in rep["checked_slots"]),
        # TEETH 1: a decision-varying cell on an ALLOWED channel must still be refused.
        "refuses_decision_varying_feature": raises(
            lambda a: a.__setitem__((3, EXPECT_DEAD_SLOT_CHANNELS[0], slot, 0), 0.25),
            "VARY across decisions"),
        # TEETH 2: a CONSTANT non-zero on any other channel must be refused (a real leak shape).
        "refuses_nonzero_on_other_channel": raises(
            lambda a: a.__setitem__((slice(None), 0, slot, slice(None)), 0.5),
            "outside the guarded vol-ratio set"),
    }
    for k, v in checks.items():
        print(f"    {'OK  ' if v else 'FAIL'} {k}")
    print(f"    per dead slot: {rep['checked_slots'][0]['n_nonzero_feature_cells']} of "
          f"{rep['checked_slots'][0]['n_feature_cells']} feature cells non-zero, channels "
          f"{rep['checked_slots'][0]['nonzero_channels']}, value(s) {vals} — flat-zero PANEL "
          f"cells, CONSTANT (not all-zero) ENCODED features")
    return {"leg": "c_inactive_slot_feature_guard", "passed": all(checks.values()),
            "checks": checks, "report": rep["checked_slots"]}


def leg_panel_source_and_type_gate(R, imp, collection):
    """FIX B + FIX D — the provenance gate's panel_source / panel_frame header handling."""
    print("\n=== FIX B/D: panel_source + panel_frame-type gate matrix (in-process) ===")
    fp = write_frame(R("frame_gate.json"), FRAME_SYMBOLS, "TASK_863 FIX B/D gate frame")
    args, prices_j, env_cfg, frame, start_row, N, T = _framed_panel(imp, fp)
    tier = env_cfg.replace(fee_bps=imp.FEE_BPS, slippage_bps=imp.SLIP_BPS)
    base = json.load(open(collection))

    def run_gate(mutate, pinned=True):
        doc = json.loads(json.dumps(base))
        mutate(doc["provenance"])
        # The UNPINNED leg reads the minute loader mirror, which cannot build a 1440 panel on the
        # fixture's minute slice (the disclosed fixture limitation) — so it runs at 240, exactly
        # like the subprocess mixed legs. Only the panel_source line is asserted from it.
        a = _pin_args(imp, fp if pinned else None, bar_minutes=1440 if pinned else 240)
        try:
            imp.gate_teacher_vectors(doc, collection, a, tier, start_row, N, T,
                                     frame=frame if pinned else None)
            return None
        except Exception as e:                      # RuntimeError = refusal; anything else = bug
            return f"{type(e).__name__}: {e}"

    accepted_minute = run_gate(lambda p: None)
    accepted_daily = run_gate(lambda p: p.update(panel_source="daily"))
    refused_bogus = run_gate(lambda p: p.update(panel_source="hourly")) or ""
    refused_unpinned = run_gate(lambda p: p.update(panel_source="daily"), pinned=False) or ""
    refused_badtype = run_gate(lambda p: p.update(panel_frame="fixed-frame")) or ""
    # ---- round-2 FIX 2.2 / 2.3 / 2.4: the remaining valid-JSON/wrong-shape header cases ----
    # 2.2: a wrong-TYPED slot used to reach int() and raise a raw ValueError ("abc") / TypeError
    # (null). 2.3: a slot multiset that cannot be a per-slot map (duplicate 0, no 1) sorted stably
    # into a plausible symbol list and was recorded as identity_matched proof. 2.4: a PRESENT but
    # EMPTY panel_frame object was dict-typed AND falsy, so it downgraded to the legacy path.
    refused_slot_str = run_gate(
        lambda p: p["panel_frame"]["coverage"][3].update(slot="abc")) or ""
    refused_slot_null = run_gate(
        lambda p: p["panel_frame"]["coverage"][3].update(slot=None)) or ""
    refused_dup_slots = run_gate(
        lambda p: p["panel_frame"]["coverage"][1].update(slot=0)) or ""
    refused_empty_frame = run_gate(
        lambda p: p.update(panel_frame={}, panel_frame_from_checkpoint=None)) or ""
    # ... and the both-directions control: an ABSENT key (or the `null` every UNPINNED collection
    # records) must still read as LEGACY, never as a pin. These two runs refuse on the unrelated
    # fixture panel mismatch (1440-recorded collection vs the 240 minute mirror); what matters is
    # that NEITHER refusal is a frame-pin/EMPTY one.
    legacy_absent_key = run_gate(lambda p: (p.pop("panel_frame", None),
                                            p.update(panel_frame_from_checkpoint=None)),
                                 pinned=False) or ""
    legacy_null_key = run_gate(lambda p: p.update(panel_frame=None,
                                                  panel_frame_from_checkpoint=None),
                               pinned=False) or ""
    legacy_text = ("panel_source: collection='daily'  imprint='minute'  -> this runner builds its "
                   "panel through create_jax_env_historical (the 1-min ohlcv table, PIN b); it "
                   "cannot reproduce a daily/framed panel")
    checks = {
        "pinned_pair_panel_source_minute_ACCEPTED": accepted_minute is None,
        # The previously-refused case: same frame, header says 'daily'.
        "pinned_pair_panel_source_daily_ACCEPTED": accepted_daily is None,
        "pinned_bogus_panel_source_REFUSED": ("panel_source" in refused_bogus
                                              and "daily-framed" in refused_bogus),
        "unpinned_refusal_text_UNCHANGED": legacy_text in refused_unpinned,
        "nondict_panel_frame_NAMED_refusal": ("panel_frame TYPE" in refused_badtype
                                              and "not a JSON object" in refused_badtype),
        "nondict_panel_frame_no_AttributeError": "AttributeError" not in refused_badtype,
        # FIX 2.2 — wrong-TYPED slot: a NAMED refusal, not a raw ValueError/TypeError traceback.
        "slot_str_NAMED_refusal": ("panel_frame.coverage MALFORMED" in refused_slot_str
                                   and "INT 'slot'" in refused_slot_str),
        "slot_str_no_raw_ValueError": "ValueError" not in refused_slot_str,
        "slot_null_NAMED_refusal": ("panel_frame.coverage MALFORMED" in refused_slot_null
                                    and "INT 'slot'" in refused_slot_null),
        "slot_null_no_raw_TypeError": "TypeError" not in refused_slot_null,
        # FIX 2.3 — duplicate slots are not positional proof.
        "duplicate_slots_NAMED_refusal": (
            "panel_frame.coverage MALFORMED" in refused_dup_slots
            and "complete per-slot map" in refused_dup_slots),
        # A malformed coverage must NOT be silently replaced by the loaded_symbols fallback.
        "malformed_coverage_does_not_fall_back_to_loaded_symbols": all(
            "loaded_symbols is NOT accepted in its place" in t
            for t in (refused_slot_str, refused_slot_null, refused_dup_slots)),
        # FIX 2.4 — a PRESENT but EMPTY panel_frame is malformed, not "unpinned".
        "empty_panel_frame_NAMED_refusal": ("panel_frame EMPTY" in refused_empty_frame
                                            and "TRUNCATED" in refused_empty_frame),
        # ... while an absent/null panel_frame stays LEGACY (no frame verdict at all).
        "absent_panel_frame_stays_LEGACY": not any(
            s in legacy_absent_key for s in ("panel_frame EMPTY", "panel_frame PIN MISMATCH",
                                             "panel_frame TYPE")),
        "null_panel_frame_stays_LEGACY": not any(
            s in legacy_null_key for s in ("panel_frame EMPTY", "panel_frame PIN MISMATCH",
                                           "panel_frame TYPE")),
        "all_refusals_are_RuntimeError": all(
            t.startswith("RuntimeError:") for t in
            (refused_bogus, refused_unpinned, refused_badtype, refused_slot_str,
             refused_slot_null, refused_dup_slots, refused_empty_frame,
             legacy_absent_key, legacy_null_key)),
    }
    for k, v in checks.items():
        print(f"    {'OK  ' if v else 'FAIL'} {k}")
    if accepted_daily is not None:
        print(f"    panel_source='daily' unexpectedly refused: {accepted_daily[:400]}")
    return {"leg": "bd_panel_source_and_frame_type_gate", "passed": all(checks.values()),
            "checks": checks}


def main():
    ap = argparse.ArgumentParser(description="TASK_863 frame-pin fixture smoke")
    ap.add_argument("--keep", action="store_true", help="keep a previous run dir's artifacts")
    args = ap.parse_args()
    # TASK_873: verify the live fixture against fixtures/FIXTURE_IDENTITY.json BEFORE any work
    # — a re-based fixture would otherwise silently move every gate below.
    fixture_identity.verify_or_die(FIXTURE_DB, label="[863 fixture-identity]")
    if not args.keep and os.path.isdir(RUN_DIR):
        shutil.rmtree(RUN_DIR)
    os.makedirs(RUN_DIR, exist_ok=True)
    if not os.path.exists(os.path.join(CKPT_SRC, "trainer_metadata.json")):
        raise SystemExit(
            f"missing the 861 fresh-init teacher checkpoint {CKPT_SRC}; run "
            f"`python experiments/861_search_teacher_imprint/smoke_e2e.py --quick` once first.")
    R = lambda *p: os.path.join(RUN_DIR, *p)                              # noqa: E731
    report = {"legs": []}
    t0 = time.time()

    frame = write_frame(R("frame.json"), FRAME_SYMBOLS, "TASK_863 fixture frame")
    # SAME frame, DIFFERENT path: the spec allows the provenance path to differ.
    frame_copy = write_frame(R("frame_same_other_path.json"), FRAME_SYMBOLS,
                             "TASK_863 identical frame reached by a different path")
    reordered = list(FRAME_SYMBOLS)
    reordered[0], reordered[2] = reordered[2], reordered[0]
    frame_reordered = write_frame(R("frame_reordered.json"), reordered,
                                  "TASK_863 (iii) REORDERED same-membership frame")
    membership = list(FRAME_SYMBOLS)
    membership[7] = "DASH/USD"
    frame_membership = write_frame(R("frame_membership.json"), membership,
                                   "TASK_863 different-membership frame")

    # ---------------------------------------------------------------- leg 1: pinned collection
    print("\n=== leg 1: record a FRAME-PINNED collection (eval_vector.py, shipped 853 path) ===")
    collection = R("chosen_vectors_pinned.json")
    sh(eval_cmd(R("evalout"), collection, frame=frame), R("log_collect_pinned.txt"))
    coll = json.load(open(collection))
    cp = coll["provenance"]
    coll_frame = [c["symbol"] for c in cp["panel_frame"]["coverage"]]
    assert coll_frame == FRAME_SYMBOLS, coll_frame
    assert cp["panel_frame"]["inactive_slots"] == EXPECT_INACTIVE_SLOTS, cp["panel_frame"]
    import numpy as np
    vecs = np.asarray(coll["decisions"]["chosen_vectors"])
    inactive_all_hold = bool((vecs[:, EXPECT_INACTIVE_SLOTS] == 1).all())
    print(f"  collection: {cp['n_decisions_collected']} decisions, rows from "
          f"{cp['start_row']}, n_rows {cp['n_rows']}, frame order matched, inactive slots "
          f"{cp['panel_frame']['inactive_slots']}, walk net@21 {cp['walk_net21']:+.6f}; "
          f"inactive slots all HOLD in the recorded stream = {inactive_all_hold}")
    report["collection"] = {"n_decisions": cp["n_decisions_collected"],
                            "n_rows": cp["n_rows"], "start_row": cp["start_row"],
                            "walk_net21": cp["walk_net21"],
                            "inactive_slots": cp["panel_frame"]["inactive_slots"],
                            "inactive_slots_all_hold": inactive_all_hold}

    # ------------------------------------------------- (ii) END-TO-END PASS + DISCLOSURE
    print("\n=== (ii) PASS LEG — matching pin (via a DIFFERENT path), end to end ===")
    out_pass = R("results_pass.json")
    sh(imprint_cmd(out_pass, R("ckpt_pass"), collection, frame=frame_copy),
       R("log_pass.txt"))
    res = json.load(open(out_pass))
    pf = res.get("panel_frame")
    gate = res["teacher_vectors"]["source"]["gate"]
    ri = res["teacher_vectors"]["replay_integrity"]
    checks = {
        "panel_frame_block_present": pf is not None,
        "frame_symbols_recorded_in_order": res.get("frame_symbols") == FRAME_SYMBOLS,
        "inactive_slots_disclosed": res.get("inactive_slots") == EXPECT_INACTIVE_SLOTS,
        "inactive_spans_disclosed": bool(pf and len(pf["inactive_spans"]) == 2),
        "n_active_slots_6_of_8": bool(pf and pf["n_active_slots"] == 6 and pf["n_slots"] == 8),
        "force_hold_cells_eq_2_per_decision": bool(
            pf and pf["n_force_hold_cells"] == 2 * res["n_decisions"]),
        "frame_pin_gate_identity_matched": gate["frame_pin"]["identity_matched"] is True,
        "frame_pin_gate_both_pinned": (gate["frame_pin"]["collection_pinned"] and
                                       gate["frame_pin"]["imprint_pinned"]),
        "provenance_paths_differ": (gate["frame_pin"]["collection_provenance_path"] !=
                                    gate["frame_pin"]["imprint_provenance_path"]),
        # The alignment proof: imprint's masked walk reproduces eval's masked walk score.
        "replay_integrity_checked_and_passed": bool(ri.get("checked") and ri.get("passed")),
        # ... all the way through the sanity replay, both cost tiers.
        "sanity_replay_both_tiers": all(
            res["sanity"]["per_tier"][t]["policy_inenv"] is not None for t in ("0", "21")),
        "losses_finite": res["imprint"]["all_losses_finite"],
        "policy_ce_decreased": res["imprint"]["policy_ce_decreased"],
    }
    # ---- fix-pass disclosure keys on the SAME artifact (FIX 0 marker + FIX A companions) ----
    sn = res["sanity"]
    ac = sn.get("active_cells_only") or {}
    fit = res["teacher_vectors"]["fit_vs_original_targets"]
    checks.update({
        # FIX 0: a healthy pinned run carries the marker as FALSE (862 can assert on it).
        "fix0_degenerate_replay_marker_false": sn.get("degenerate_replay") is False,
        "fix0_degenerate_reason_none": sn.get("degenerate_replay_reason") is None,
        # FIX A(1): active-cells-only companions present and DIFFERENT from the diluted figures.
        "fixA_masked_cell_fraction_recorded": (
            abs(sn.get("masked_cell_fraction", -1)
                - 2 * res["n_decisions"] / (8.0 * res["n_decisions"])) < 1e-9),
        "fixA_active_agreement_present": ac.get("teacher_action_agreement") is not None,
        "fixA_active_agreement_differs_from_reported": (
            ac.get("teacher_action_agreement") != sn["teacher_action_agreement"]),
        "fixA_active_hist_excludes_masked_holds": (
            ac.get("policy_action_hist", {}).get("1", -1)
            == sn["policy_action_hist"]["1"] - sn["n_masked_cells"]),
        "fixA_active_hist_total_is_active_cells": (
            sum((ac.get("policy_action_hist") or {}).values())
            == sn["n_decision_cells"] - sn["n_masked_cells"] == ac.get("n_active_cells")),
        "fixA_per_tier_active_avg_long_present": all(
            "avg_long_active_cells" in sn["per_tier"][t] for t in ("0", "21")),
        "fixA_861_fit_active_companion_present": bool(fit.get("active_cells_only")),
        "fixA_861_fit_active_cell_count": (
            (fit.get("active_cells_only") or {}).get("n_label_cells")
            == sn["n_decision_cells"] - sn["n_masked_cells"]),
        "fixA_panel_frame_masked_fraction": pf is not None and (
            abs(pf.get("masked_cell_fraction", -1) - sn["masked_cell_fraction"]) < 1e-12),
        # FIX A(3): partial slots discoverable at top level (none on THIS frame -> empty list, but
        # the key must exist so a consumer can never miss a partial slot by reading inactive only).
        "fixA_partial_slots_key_present": isinstance(res.get("partial_slots"), list),
        # FIX C: the feature guard ran and is recorded in the artifact.
        "fixC_feature_guard_recorded": bool(
            pf and pf.get("inactive_slot_feature_check", {}).get("checked_slots")),
        "fixC_guard_channels_are_vol_ratio": bool(pf and all(
            s["nonzero_channels"] == EXPECT_DEAD_SLOT_CHANNELS
            for s in pf["inactive_slot_feature_check"]["checked_slots"])),
        # FIX B: the gate recorded the collection's (default) panel_source vs this run's REAL one.
        "fixB_panel_source_disclosed": (
            gate["frame_pin"].get("imprint_panel_source_actual") == "daily-framed"
            and gate["frame_pin"].get("collection_panel_source_recorded") is not None),
    })
    for k, v in checks.items():
        print(f"    {'OK  ' if v else 'FAIL'} {k}")
    print(f"    reported teacher_action_agreement {sn['teacher_action_agreement']:.4f} vs "
          f"ACTIVE-CELLS-ONLY {ac.get('teacher_action_agreement')} "
          f"({sn.get('n_masked_cells')} of {sn.get('n_decision_cells')} cells masked = "
          f"{100.0 * sn.get('masked_cell_fraction', 0):.2f}%)")
    report["legs"].append({"leg": "ii_pass_end_to_end", "passed": all(checks.values()),
                           "checks": checks, "log": "log_pass.txt",
                           "reported_vs_active": {
                               "teacher_action_agreement": sn["teacher_action_agreement"],
                               "teacher_action_agreement_active":
                                   ac.get("teacher_action_agreement"),
                               "masked_cell_fraction": sn.get("masked_cell_fraction"),
                               "policy_action_hist": sn["policy_action_hist"],
                               "policy_action_hist_active": ac.get("policy_action_hist"),
                               "fit_argmax_agreement": fit.get("argmax_agreement_vs_original"),
                               "fit_argmax_agreement_active":
                                   (fit.get("active_cells_only") or {}).get(
                                       "argmax_agreement_vs_original"),
                               "fit_modal_action_rate": fit.get("modal_action_rate"),
                               "fit_modal_action_rate_active":
                                   (fit.get("active_cells_only") or {}).get("modal_action_rate")}})

    # ------------------------------------------------- (iii) POSITIONAL HONESTY
    print("\n=== (iii) REORDERED same-membership frame must be REFUSED ===")
    report["legs"].append(refusal_leg(
        "iii_reordered_same_membership", "panel_frame identity MISMATCH",
        imprint_cmd(R("results_reordered.json"), R("ckpt_reordered"), collection,
                    frame=frame_reordered), R("log_reordered.txt")))

    # ------------------------------------------------- scope-2 fail-closed matrix
    print("\n=== scope 2: fail-closed matrix ===")
    # NOTE (fixture limitation, disclosed): a PINNED panel requires --bar-minutes 1440 (ohlcv_daily)
    # while the fixture's minute slice can only build an UNPINNED panel at 240 — so on this DB the
    # two mixed legs necessarily also differ in bar_minutes. The gate collects every mismatch and
    # raises them TOGETHER (FIX 10), so the frame-pin refusal is still emitted and asserted here;
    # what the leg cannot do on the fixture is isolate the pin as the SOLE differing field.
    report["legs"].append(refusal_leg(
        "pinned_collection_unpinned_imprint",
        "panel_frame PIN MISMATCH (pinned collection, UNPINNED imprint)",
        imprint_cmd(R("results_coll_pinned_imp_unpinned.json"), R("ckpt_cpiu"), collection,
                    panel=LEGACY_PANEL),
        R("log_coll_pinned_imp_unpinned.txt")))
    report["legs"].append(refusal_leg(
        "unpinned_collection_pinned_imprint",
        "panel_frame PIN MISMATCH (PINNED imprint, unpinned collection)",
        imprint_cmd(R("results_coll_unpinned_imp_pinned.json"), R("ckpt_cupi"),
                    LEGACY_COLLECTION, frame=frame),
        R("log_coll_unpinned_imp_pinned.txt")))
    report["legs"].append(refusal_leg(
        "different_membership_frame", "panel_frame identity MISMATCH",
        imprint_cmd(R("results_membership.json"), R("ckpt_membership"), collection,
                    frame=frame_membership), R("log_membership.txt")))
    report["legs"].append(refusal_leg(
        "empty_frame_path_fails_closed", "was given an EMPTY path",
        imprint_cmd(R("results_empty.json"), R("ckpt_empty"), collection, frame=""),
        R("log_empty.txt")))
    report["legs"].append(refusal_leg(
        "frame_pin_requires_bar_minutes_1440", "requires --bar-minutes 1440",
        imprint_cmd(R("results_bar240.json"), R("ckpt_bar240"), collection, frame=frame,
                    panel=LEGACY_PANEL), R("log_bar240.txt")))

    # --------------------------------- FIX 2.5: the degeneracy guard FIRING (positive evidence)
    print("\n=== FIX 2.5: the FIX 0 / FIX 2.1 degeneracy guard FIRES (both diagnoses) ===")
    # TEACHER side (FIX 2.1): --vector-rebalance-deadband 10.0 makes every 819-book target a HOLD,
    # so the TEACHER never opens a position => teacher_floor_inenv and corridor_position (the
    # LEDGER metric) go null while the 1-iter student still trades and policy_inenv is FINITE.
    # Before FIX 2.1 this exact run exited 0 and recorded `degenerate_replay: false` beside a null
    # corridor_position — a POSITIVE health assertion on a degenerate run.
    report["legs"].append(degenerate_leg(
        "fix21_degenerate_teacher_side", R, frame,
        extra=["--vector-rebalance-deadband", "10.0", "--iters", "1", "--seed", "0"],
        expect_clauses=["TEACHER SIDE", "teacher_floor_inenv"], absent_clauses=["STUDENT SIDE"],
        expect_null=["teacher_floor_inenv", "corridor_position"],
        expect_finite=["policy_inenv"]))
    # STUDENT side (FIX 0, the MAJOR guard, which shipped with no positive evidence at all): the
    # same all-HOLD teacher with the smoke's 80 iters, so BC actually LEARNS total HOLD and the
    # policy replay is zero-variance too => policy_inenv null as well. This is the deterministic
    # consequence of an all-HOLD label set rather than a tuned iteration count (verified firing on
    # seeds 0 and 1); --max-decisions cannot be used for this on THIS runner, which refuses N < 8
    # ("panel too short") before any replay happens.
    report["legs"].append(degenerate_leg(
        "fix0_degenerate_student_side", R, frame,
        extra=["--vector-rebalance-deadband", "10.0", "--iters", "80", "--seed", "0"],
        expect_clauses=["STUDENT SIDE", "TEACHER SIDE", "total-HOLD policy"], absent_clauses=[],
        expect_null=["policy_inenv", "teacher_floor_inenv", "corridor_position"],
        expect_finite=[]))

    # ------------------------------------------------- fix-pass legs (in-process, seconds)
    imp = _imprint_module()
    report["legs"].append(leg_partial_slot_mask(R, imp))
    report["legs"].append(leg_inactive_feature_guard(R, imp))
    report["legs"].append(leg_panel_source_and_type_gate(R, imp, collection))

    # ------------------------------------------------- (i) parity (lite): additive blocks only
    print("\n=== (i) PARITY (lite) — a legacy unpinned run carries NO frame keys ===")
    out_legacy = R("results_legacy.json")
    sh(imprint_cmd(out_legacy, R("ckpt_legacy"), LEGACY_COLLECTION, panel=LEGACY_PANEL),
       R("log_legacy.txt"))
    leg = json.load(open(out_legacy))
    lchecks = {
        "no_panel_frame_key": "panel_frame" not in leg,
        "no_panel_frame_from_checkpoint_key": "panel_frame_from_checkpoint" not in leg,
        "no_inactive_slots_key": "inactive_slots" not in leg,
        "no_frame_symbols_key": "frame_symbols" not in leg,
        "no_frame_pin_in_gate": "frame_pin" not in leg["teacher_vectors"]["source"]["gate"],
        # fix pass: the new additive keys must ALSO be absent on the legacy path.
        "no_partial_slots_key": "partial_slots" not in leg,
        "no_degenerate_replay_key": "degenerate_replay" not in leg["sanity"],
        "no_masked_cell_keys": not any(k in leg["sanity"] for k in
                                      ("masked_cell_fraction", "active_cells_only",
                                       "n_masked_cells", "n_decision_cells", "dilution_note")),
        # TASK_868 (D31 a2) AMENDMENT: the causal open-gate mask rides the canon UNPINNED
        # minute path too (uniform with the pinned minute loader), so the legacy unpinned
        # panel now HAS masked cells and the active-cells companions legitimately appear
        # there. That is the ruling's intended re-base, not a frame-key leak — the identity
        # keys above stay absent and remain the additive-parity assertion.
        "active_cell_companions_disclose_the_868_mask": all(
            leg["sanity"]["per_tier"][t].get("avg_long_active_cells") is not None
            for t in ("0", "21")),
        "active_companion_in_861_fit_discloses_the_868_mask": all(
            k in (leg["teacher_vectors"]["fit_vs_original_targets"] or {})
            for k in ("active_cells_only", "masked_cell_fraction", "active_cells_note")),
    }
    for k, v in lchecks.items():
        print(f"    {'OK  ' if v else 'FAIL'} {k}")
    report["legs"].append({"leg": "i_parity_lite_no_frame_keys",
                           "passed": all(lchecks.values()), "checks": lchecks,
                           "log": "log_legacy.txt"})

    # ------------------------------------------------------------------------------- verdict
    report["all_passed"] = all(l["passed"] for l in report["legs"])
    report["elapsed_s"] = round(time.time() - t0, 1)
    report["note"] = ("The BINDING acceptance-(i) evidence is the byte-level comparison of the "
                      "legacy artifacts against git tag task861-search-teacher-imprint (run by "
                      "hand with identical invocations, incl. --ckpt-out): BYTE-IDENTICAL. Orbax "
                      "checkpoint bytes are nondeterministic run-to-run (per-save timestamps in "
                      "_CHECKPOINT_METADATA + OCDBT store naming) and are NOT a parity surface.")
    with open(R("report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print("\n" + "=" * 84)
    for l in report["legs"]:
        print(f"  {'PASS' if l['passed'] else 'FAIL'}  {l['leg']}")
    print(f"TASK_863 frame-pin smoke: {'ALL LEGS PASS' if report['all_passed'] else 'FAILURES'} "
          f"in {report['elapsed_s']}s -> {R('report.json')}")
    print("=" * 84)
    raise SystemExit(0 if report["all_passed"] else 1)


if __name__ == "__main__":
    main()
