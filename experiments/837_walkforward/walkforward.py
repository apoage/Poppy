#!/usr/bin/env python3
"""TASK_837 / R4 — walk-forward split harness (the graduation instrument).

The definition-of-working says UNSEEN YEARS; before this module nothing in the repo could
produce an unseen-years number (training and 835-eval both run the same pinned 2021-2025
window). This harness BUILDS THE RAILS — it trains nothing:

  1. FOLD MATERIALIZATION — per-fold periods-files in the exact 821 format
     (``experiments/821_planner_daily/period_821_train.json``: ``periods[0]``
     {name, start_date, end_date, assets}; train_jax_fast.py:223-231 and 835's
     ``--periods-file`` both read periods[0] ONLY), plus per-fold POINT-IN-TIME universes,
     plus job manifests consumable by ``tools/deploy/run_queue.sh`` (whitespace format:
     ``<job_id> <gpu|cpu> <mem> <command...>``) and by
     ``experiments/835_stage4_eval/eval_vector.py`` (its CLI; imported by manifest, never
     modified).
  2. PIT UNIVERSE (load-bearing) — ``trader/data.py:401 build_pit_universe(db_path,
     min_bars)`` carries NO as-of date: its coverage floor counts ALL 1-min bars
     (``SELECT COUNT(*) FROM ohlcv WHERE asset_id=?`` — unbounded in time), so it CANNOT
     derive a universe as-of train-end. The PIT derivation is therefore implemented HERE
     (read-only queries; ZERO edits to trader/data.py) — rule pre-registered in SPEC.md:
     build_pit_universe's USD-crypto filter (imported ``STABLE_FIAT``) + a coverage floor +
     coverage ranking, all TIME-BOUNDED at train-end, counted on ``ohlcv_daily`` (LOUD
     DEVIATION, see SPEC §PIT: daily bars instead of 1-min bars — the canonical audited
     daily store, and fixture-daily == db-daily is LEDGER-official per 830/E0-REAL, so fold
     universes are derivable and leak-assertable without touching trading_data.db).
     The pinned-64 universe is emitted alongside, labeled LEAKY-UNIVERSE (821 continuity).
  3. FOLD CORRIDOR — per-eval-window floor/ceiling/ref via the 830 corridor machinery
     (``experiments/830_corridor/corridor.py`` loaded by importlib, the 832 pattern; never
     modified). The eval windows are shorter than the panel loader's ``min_active=365``
     liquidity filter tolerates (the 831-EXT crash precedent: windows <366d zero the
     universe), so the corridor panel is padded BACKWARD (past data only — never beyond
     eval-end) by a pre-registered ``--pad-days`` and every corridor number is computed on
     the EVAL-WINDOW DECISION SLICE only:
       floor_inenv = 830 ``replay_floor`` rets sliced to eval rows (830 row convention);
       reference   = ``trader.backtest.window_metrics`` (the gate-2 scorer: positions on
                     the whole panel, metrics ONLY on ts in [eval_start, eval_end));
       ceiling     = 830 ``oracle_positions`` scored by ``trader.backtest.regime_series``
                     on the same [r_lo, r_hi) slice (identical audited accounting).
  4. AGGREGATION — per-fold 835 scorecards + per-fold corridors -> ONE walk-forward
     summary JSON + a LEDGER-convention row. Per-fold corridor positions are ALWAYS shown;
     the pooled block exists only underneath the per-fold rows (never a pooled Sharpe
     without the per-fold view).

LEAK-FREE GATES AS CODE (the deliverable): ``validate`` parses the GENERATED artifacts
(it never trusts construction) and FAILS LOUD if any fold violates:
  ASSERT-1  no train/eval date overlap in any generated fold artifact (train + eval
            periods-files re-parsed with BOTH the training loader's UTC-midnight parse
            (vectorbt_env.py:660 ``timestamp <= end_ts`` inclusive semantics; TASK_869:
            was local-midnight) and the
            backtest UTC parse — the conservative envelope must be disjoint);
  ASSERT-2  no eval-window data can enter a train periods-file: the loader's own ingest
            query is run against the db with the PARSED train-file bounds and the max
            ingested timestamp (minute AND daily tables) must precede eval-start;
            plus PIT re-derivation equality (universe unchanged when derived from data
            <= train-end only — a tampered universe is REJECTED), manifest cross-checks
            (a train job may only reference a fold train periods-file; an eval job must
            reference the fold's eval_pit file IDENTICALLY — same-window substitutes like
            the LEAKY file are rejected), and a corridor panel-bound check (corridor
            never loaded data past eval-end);
  POLICY-PIN every parsed fold window and universe-rule parameter is compared against the
            PRE-REGISTERED constants (POLICY) — never against the artifacts' own
            (tamperable) rule blocks; internally-consistent tampers and materialize
            parameter typos are rejected. The fixture smoke proves the gates in BOTH
            directions: an unmutated control clone validates CLEAN and five corrupted
            clones are each rejected BY THE TARGETED gate (named in the rejection).

Pre-registered REAL folds (planner, 2026-07-08 — the harness implements, NEVER invents):
  fold1: train 2021-01-01..2023-12-31 -> eval 2024-01-01..2024-12-31
  fold2: train 2021-01-01..2024-12-31 -> eval 2025-01-01..2025-09-01
Pre-registered SMOKE folds (sized to the fixture minute slice 2025-09-13..2025-12-12,
SPEC §smoke): smokefold1 train 2025-09-14..2025-10-31 -> eval 2025-11-01..2025-11-30;
smokefold2 train 2025-09-14..2025-11-30 -> eval 2025-12-01..2025-12-11.

FACTORY / SCHEDULED split (830/831/835 house style): this module + the fixture smoke are
the factory deliverable. Fold TRAININGS are scheduled (phase-2+ planner decision, out of
scope); the scheduled re-materialization against ``trading_data.db`` (read-only) must
reproduce the emitted fold universes content-identically (identity contract in SPEC §PIT:
volatile fields ``generated``/``db``/``minute_coverage_check`` excluded; periods-files
byte-identical). CPU-only (``JAX_PLATFORMS=cpu``), fixture db only in this task,
read-only queries, no npz.
"""
import os
# CPU-only, deterministic; keep JAX off the GPU (factory/smoke job — house style).
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import importlib.util
import json
import shutil
import sqlite3
import sys
from datetime import datetime, timedelta, timezone, date

import numpy as np

# Run from anywhere: repo root on the path so `trader` imports cleanly.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from trader import backtest, config as trader_config
from trader.config import STABLE_FIAT

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURE_DB = os.path.join(_REPO, "fixtures", "trading_fixture.db")
PINNED_PERIODS = os.path.join(_REPO, "experiments", "821_planner_daily",
                              "period_821_train.json")
CORRIDOR_830 = os.path.join(_REPO, "experiments", "830_corridor", "corridor.py")

# ---- pre-registered fold policy (planner 2026-07-08; TASK_837 spec verbatim) ----
REAL_FOLDS = (
    {"fold_id": "fold1", "train": ("2021-01-01", "2023-12-31"),
     "eval": ("2024-01-01", "2024-12-31")},
    {"fold_id": "fold2", "train": ("2021-01-01", "2024-12-31"),
     "eval": ("2025-01-01", "2025-09-01")},
)
# ---- pre-registered smoke folds (SPEC §smoke — sized to fixture minute coverage) ----
SMOKE_FOLDS = (
    {"fold_id": "smokefold1", "train": ("2025-09-14", "2025-10-31"),
     "eval": ("2025-11-01", "2025-11-30")},
    {"fold_id": "smokefold2", "train": ("2025-09-14", "2025-11-30"),
     "eval": ("2025-12-01", "2025-12-11")},
)

# PIT qualification rule (pre-registered, SPEC §PIT): USD-quoted, non-stable/fiat
# (build_pit_universe's filter, STABLE_FIAT imported), >= MIN_DAYS daily bars with
# timestamp <= as-of (train-end UTC midnight), ranked by that count DESC (tie: asset_id
# ASC), capped at N_ASSETS. TWO deviations from build_pit_universe, both LOUD in SPEC.md:
# daily-bar counts instead of its 1-min counts, and the coverage RANKING + cap are NEW
# policy (build_pit_universe has no ranking/cap — it returns ALL qualifying, asset_id-
# sorted; the ranking mirrors the training loader's own coverage ordering instead).
PIT_MIN_DAYS = 180
PIT_N_ASSETS = 64
SMOKE_PIT_N_ASSETS = 8

# Pre-registration PIN (verify-panel MAJOR-1 fix, 2026-07-09): ``validate`` re-checks
# every generated artifact against THESE constants — never against the (tamperable)
# artifact's own rule block or fold_manifest alone. An internally-consistent tamper
# (windows or universe rule edited coherently across all files) and an honest
# ``materialize --min-days 30`` typo are both REJECTED. Only pre-registered fold sets
# validate (the harness implements splits, it never invents them).
POLICY = {
    "real": {"folds": {f["fold_id"]: f for f in REAL_FOLDS},
             "n_assets": PIT_N_ASSETS, "min_days": PIT_MIN_DAYS},
    "smoke": {"folds": {f["fold_id"]: f for f in SMOKE_FOLDS},
              "n_assets": SMOKE_PIT_N_ASSETS, "min_days": PIT_MIN_DAYS},
}

# Corridor conventions (830 lineage).
PAD_DAYS = 366          # backward pad: covers strategy warmup AND the min_active=365 filter
GATE_COSTS = (0, 21)    # bp/side cost tiers — gross + net@21 (fee16+slip5), the 830 SPEC tiers
MIN_EVAL_DECISIONS = 6  # regime_series' own floor; fewer eval decisions = fail loud

# Training arm mirrored VERBATIM from experiments/828_stage3_train/jobs_phase1.manifest
# (the preflight-revised phase-1: cost0-first; fee16 is a CONTINUATION decision, out of
# scope here). The eval manifest carries the SAME operator (hold_bias 0.0 — 835's CLI
# default 3.0 is the jobs.manifest value, wrong for this arm; explicit beats default).
ARM = {
    "name": "cost0",
    "train_flags": ("--jax-env --db-path {db} --periods-file {pf} --bar-minutes 1440 "
                    "--vector-action --vector-proposal-scheme dithered-book --vector-k 16 "
                    "--vector-hold-bias 0.0 --vector-dither-scale 1.0 --reward-hybrid "
                    "--reward-attributable --model-size small --sims 32 --batch-size 256 "
                    "--parallel-envs 4 --steps 256 --fee 0 --slippage 0 --epochs 2000 "
                    "--seed {seed} --snapshot-every 100 --name {name}"),
    "eval_flags": ("--db {db} --periods-file {pf} --bar-minutes 1440 --n-assets {n_assets} "
                   "--vector-proposal-scheme dithered-book --vector-k 16 "
                   "--vector-hold-bias 0.0 --vector-dither-scale 1.0 --sims 32 "
                   "--checkpoint {ckpt} --eval-seed 0 --tag {tag} --out-dir {out_dir}"),
}
SEEDS = (0, 1, 2)
HOLD_COLLAPSE_PCT = 99.0   # the 821 signature (835 convention)


class LeakError(RuntimeError):
    """A generated fold artifact violates the leak-free gates. ALWAYS fatal."""


# =============================================================================
# date parsing — BOTH conventions, used conservatively
# =============================================================================

def utc_epoch(d):
    """UTC-midnight parse (trader.backtest._to_epoch — daily-store timestamps are UTC)."""
    return backtest._to_epoch(d)


def loader_epoch(d):
    """The training loader's parse (VectorBTTradingEnv._load_database_prices; the 835
    mirror). TASK_869: the loader's parse is now UTC-midnight (was local-timezone), so this
    is IDENTICAL to ``utc_epoch`` — the conservative min/max helpers below collapse to one
    bound, which is the point of the re-base. Loader bounds are INCLUSIVE:
    ``timestamp >= start AND timestamp <= end`` (vectorbt_env.py:660)."""
    return int(datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def _max_end_epoch(d):
    """Conservative LATEST instant a date string can mean under either parse."""
    return max(utc_epoch(d), loader_epoch(d))


def _min_start_epoch(d):
    """Conservative EARLIEST instant a date string can mean under either parse."""
    return min(utc_epoch(d), loader_epoch(d))


def _date_add(d, days):
    return (date.fromisoformat(d) + timedelta(days=days)).isoformat()


def _rel(p):
    """Repo-relative display path (committed artifacts never embed absolute paths)."""
    p = os.path.abspath(p)
    r = os.path.relpath(p, _REPO)
    return p if r.startswith("..") else r


def _ro_connect(db_path):
    return sqlite3.connect(f"file:{os.path.abspath(db_path)}?mode=ro", uri=True)


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=float)
    return path


def _f(x):
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


# =============================================================================
# PIT universe — time-bounded re-implementation of build_pit_universe's rule
# =============================================================================

def derive_pit_universe(db_path, as_of, n_assets=PIT_N_ASSETS, min_days=PIT_MIN_DAYS):
    """Universe as-of ``as_of`` (train-end) from data <= as_of ONLY.

    Rule (pre-registered; SPEC §PIT): symbol contains '/', quote == USD, base not in
    STABLE_FIAT (== trader/data.py:401 build_pit_universe's filter), coverage floor
    ``min_days`` DAILY bars with timestamp <= as_of (UTC midnight), ranked by that count
    DESC (tie: asset_id ASC), top ``n_assets``. LOUD DEVIATION from build_pit_universe:
    daily-bar counts (``ohlcv_daily``) instead of its unbounded 1-min counts — see module
    docstring #2 and SPEC.md. Read-only; never writes."""
    as_of_ts = utc_epoch(as_of)
    con = _ro_connect(db_path)
    rows = con.execute(
        """SELECT a.id, a.symbol, COUNT(*) AS nd
           FROM ohlcv_daily o JOIN assets a ON o.asset_id = a.id
           WHERE o.timestamp <= ?
           GROUP BY a.id""", (as_of_ts,)).fetchall()
    con.close()
    qual = []
    for aid, sym, nd in rows:
        if not sym or "/" not in sym:
            continue
        base, quote = sym.split("/", 1)
        if quote.upper() != "USD" or base.upper() in STABLE_FIAT:
            continue
        if nd >= min_days:
            qual.append((int(nd), int(aid), sym))
    qual.sort(key=lambda t: (-t[0], t[1]))
    sel = qual[:n_assets]
    if not sel:
        raise LeakError(
            f"PIT universe EMPTY as-of {as_of} on {_rel(db_path)} (rule min_days="
            f"{min_days}) — refusing to emit a fold without a point-in-time universe. "
            f"(On a minute-only db slice this is expected; materialize real folds "
            f"against a db whose ohlcv_daily covers the train window.)")
    return {
        "as_of": as_of,
        "as_of_ts_utc": as_of_ts,
        "rule": {
            "source_table": "ohlcv_daily",
            "filter": "symbol has '/', quote==USD, base not in trader.config.STABLE_FIAT "
                      "(build_pit_universe's filter, time-bounded at as_of)",
            "min_days": min_days,
            "rank": "daily-bar count <= as_of DESC, tie asset_id ASC",
            "n_assets_cap": n_assets,
            "deviation": "trader/data.py:401 build_pit_universe has NO as-of parameter "
                         "(counts ALL 1-min bars) — rule re-implemented here time-bounded "
                         "on the canonical daily store; trader/data.py untouched.",
        },
        "n_qualifying": len(qual),
        "n_selected": len(sel),
        "symbols": [s for (_, _, s) in sel],
        "asset_ids": [a for (_, a, _) in sel],
        "counts": {s: n for (n, _, s) in sel},
        "db": _rel(db_path),
    }


def load_pinned_universe():
    """The pinned 64-asset 821 universe, byte-order preserved (TASK_740 landmine: never
    reorder). Emitted only under the LEAKY-UNIVERSE label — selected with knowledge
    through 2025, survivorship-leaky for fold evals."""
    with open(PINNED_PERIODS) as fh:
        return json.load(fh)["periods"][0]["assets"]


# =============================================================================
# fold materialization — periods-files (821 format) + universes + job manifests
# =============================================================================

def _periods_file(name, start, end, assets, comment):
    """EXACT 821 schema: consumers (train_jax_fast.py:223-231, 835 eval_vector.py
    --periods-file) read periods[0] {start_date, end_date, assets} only."""
    return {"_comment": comment,
            "periods": [{"name": name, "start_date": start, "end_date": end,
                         "assets": list(assets)}]}


def _check_fold_definition(fold):
    """Fail loud BEFORE writing anything if a fold definition itself is malformed
    (the harness implements pre-registered splits; it never repairs them)."""
    ts0, ts1 = fold["train"]
    es0, es1 = fold["eval"]
    if not (ts0 < ts1 and es0 < es1):
        raise LeakError(f"{fold['fold_id']}: degenerate window (train {ts0}..{ts1}, "
                        f"eval {es0}..{es1})")
    if not _max_end_epoch(ts1) < _min_start_epoch(es0):
        raise LeakError(f"{fold['fold_id']}: train end {ts1} does not strictly precede "
                        f"eval start {es0} under the conservative dual parse")


def materialize(db_path, folds, fold_set, out_root, n_assets, min_days,
                jobs_db, seeds=SEEDS, verify_minute=True):
    """Emit all fold artifacts under ``out_root``:
      folds/<id>/universe_pit.json, period_train.json, period_eval.json,
      period_train_leaky.json, period_eval_leaky.json
      out_root/train_jobs.manifest, eval_jobs.manifest, fold_manifest.json
    Returns the fold_manifest dict. Read-only db access; validation is a SEPARATE pass
    (validate_folds) that re-parses these files — never trusts this construction."""
    pinned = load_pinned_universe()
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    fold_records, train_lines, eval_lines, notes = [], [], [], []

    for fold in folds:
        _check_fold_definition(fold)
        fid = fold["fold_id"]
        (t0, t1), (e0, e1) = fold["train"], fold["eval"]
        fdir = os.path.join(out_root, fid)
        os.makedirs(fdir, exist_ok=True)

        uni = derive_pit_universe(db_path, as_of=t1, n_assets=n_assets, min_days=min_days)
        uni.update({"fold_id": fid, "generated": generated})
        if verify_minute:
            uni["minute_coverage_check"] = _minute_coverage_note(
                db_path, uni["asset_ids"], uni["as_of_ts_utc"])
        upath = _write_json(os.path.join(fdir, "universe_pit.json"), uni)

        c_pit = (f"TASK_837 {fid} — POINT-IN-TIME universe as-of train-end {t1} "
                 f"(rule: universe_pit.json; NO data past {t1} used). "
                 f"821 periods[0] format; consumers read periods[0] only.")
        c_leaky = (f"TASK_837 {fid} — *** LEAKY-UNIVERSE *** pinned 64-asset 821 list "
                   f"(selected with knowledge through 2025 — survivorship leak for fold "
                   f"evals; emitted ONLY for continuity with 821-lineage numbers, order "
                   f"byte-identical, TASK_740 landmine: never reorder).")
        files = {
            "universe": upath,
            "train_pit": _write_json(os.path.join(fdir, "period_train.json"),
                                     _periods_file(f"837_{fid}_train_{t0}_{t1}", t0, t1,
                                                   uni["symbols"], c_pit)),
            "eval_pit": _write_json(os.path.join(fdir, "period_eval.json"),
                                    _periods_file(f"837_{fid}_eval_{e0}_{e1}", e0, e1,
                                                  uni["symbols"], c_pit)),
            "train_leaky": _write_json(os.path.join(fdir, "period_train_leaky.json"),
                                       _periods_file(f"837_{fid}_train_LEAKY_{t0}_{t1}",
                                                     t0, t1, pinned, c_leaky)),
            "eval_leaky": _write_json(os.path.join(fdir, "period_eval_leaky.json"),
                                      _periods_file(f"837_{fid}_eval_LEAKY_{e0}_{e1}",
                                                    e0, e1, pinned, c_leaky)),
        }

        train_jobs, eval_jobs = [], []
        for s in seeds:
            tj = f"837_{fid}_{ARM['name']}_s{s}"
            cmd = "python -u train_jax_fast.py " + ARM["train_flags"].format(
                db=jobs_db, pf=_rel(files["train_pit"]), seed=s, name=tj)
            train_lines.append(f"{tj} gpu 32G {cmd}")
            train_jobs.append(tj)
            ej = f"837_{fid}_eval_s{s}"
            ecmd = ("python -u experiments/835_stage4_eval/eval_vector.py "
                    + ARM["eval_flags"].format(
                        db=jobs_db, pf=_rel(files["eval_pit"]),
                        n_assets=uni["n_selected"],
                        ckpt=f"jax_ckpt_fast/{tj}", tag=ej,
                        out_dir=f"runs/837_walkforward/{fold_set}/{fid}"))
            eval_lines.append(f"{ej} gpu 32G {ecmd}")
            eval_jobs.append(ej)

        fold_records.append({
            "fold_id": fid, "train_window": [t0, t1], "eval_window": [e0, e1],
            "files": {k: _rel(v) for k, v in files.items()},
            "universe": {"as_of": t1, "n_selected": uni["n_selected"],
                         "n_qualifying": uni["n_qualifying"]},
            "jobs": {"train": train_jobs, "eval": eval_jobs},
        })

    hdr = [f"# TASK_837 walk-forward {fold_set} folds — generated {generated}",
           f"# arm mirrors experiments/828_stage3_train/jobs_phase1.manifest "
           f"(cost0 phase-1; fee16 continuation = planner phase-2+ decision)",
           f"# consumable by tools/deploy/run_queue.sh: <job_id> <gpu|cpu> <mem> <command>"]
    tman = os.path.join(out_root, "train_jobs.manifest")
    with open(tman, "w") as fh:
        fh.write("\n".join(hdr + train_lines) + "\n")
    eman = os.path.join(out_root, "eval_jobs.manifest")
    with open(eman, "w") as fh:
        fh.write("\n".join(hdr + ["# eval operator carried from the training arm "
                                  "(hold_bias 0.0 explicit — 835's CLI default 3.0 is "
                                  "the jobs.manifest arm, wrong here)"] + eval_lines) + "\n")

    manifest = {
        "task": "TASK_837", "fold_set": fold_set, "generated": generated,
        "universe_db": _rel(db_path), "jobs_db": jobs_db,
        "n_assets": n_assets, "min_days": min_days, "seeds": list(seeds),
        "folds": fold_records,
        "manifests": {"train": _rel(tman), "eval": _rel(eman)},
        "notes": notes + [
            "validate with: python experiments/837_walkforward/walkforward.py validate "
            f"--fold-root {_rel(out_root)} --db <db>",
            "REAL-fold universes derived from the fixture's full ohlcv_daily panel "
            "(fixture daily == db daily, LEDGER 830/E0-REAL); the scheduled "
            "re-materialization against trading_data.db must reproduce them "
            "content-identically (M4 identity contract, SPEC §PIT: periods-files "
            "byte-identical; universe_pit.json identical on the derivation fields, "
            "volatile generated/db/minute_coverage_check excluded)."
            if fold_set == "real" else
            "smoke fold set — never run these jobs; rails proof only.",
        ],
    }
    _write_json(os.path.join(out_root, "fold_manifest.json"), manifest)
    print(f"[837] materialized {len(fold_records)} {fold_set} fold(s) -> {_rel(out_root)}")
    return manifest


def _minute_coverage_note(db_path, asset_ids, as_of_ts):
    """Cross-check (SPEC §PIT): PIT members should have SOME minute coverage <= as_of,
    because the training loader reads the 1-min ``ohlcv`` table. On a db whose minute
    slice starts after as_of (the fixture, for the real folds) the check is VACUOUS and
    says so loudly instead of passing silently."""
    con = _ro_connect(db_path)
    any_min = con.execute("SELECT COUNT(*) FROM ohlcv WHERE timestamp <= ?",
                          (as_of_ts,)).fetchone()[0]
    if any_min == 0:
        con.close()
        return {"status": "VACUOUS", "detail":
                "db has ZERO minute bars <= as_of — minute-coverage check impossible "
                "here; re-run on the scheduled materialization against trading_data.db."}
    ph = ",".join("?" * len(asset_ids))
    cov = dict(con.execute(
        f"SELECT asset_id, COUNT(*) FROM ohlcv WHERE asset_id IN ({ph}) "
        f"AND timestamp <= ? GROUP BY asset_id", (*asset_ids, as_of_ts)).fetchall())
    con.close()
    missing = [a for a in asset_ids if cov.get(a, 0) == 0]
    return {"status": "CHECKED", "n_members_without_minute_bars": len(missing),
            "asset_ids_without_minute_bars": missing}


# =============================================================================
# leak-free gates AS CODE — parse the generated files, never trust construction
# =============================================================================

def _read_periods(path):
    with open(path) as fh:
        p = json.load(fh)["periods"][0]
    return p["start_date"], p["end_date"], p["assets"]


def _pf_from_cmd(cmd):
    toks = cmd.split()
    return toks[toks.index("--periods-file") + 1] if "--periods-file" in toks else None


def validate_folds(fold_root, db_path, verbose=True):
    """THE leak gates (module docstring ASSERT-1 / ASSERT-2 / POLICY-PIN). Parses
    fold_manifest.json + every generated periods-file/universe/manifest/corridor under
    ``fold_root`` and re-checks them against the db AND against the PRE-REGISTERED
    POLICY constants (never against the artifacts' own tamperable rule blocks). Raises
    LeakError listing EVERY violation (fail loud); returns the check records on
    success."""
    man_path = os.path.join(fold_root, "fold_manifest.json")
    if not os.path.exists(man_path):
        raise LeakError(f"no fold_manifest.json under {_rel(fold_root)}")
    with open(man_path) as fh:
        man = json.load(fh)
    con = _ro_connect(db_path)
    checks, fails = [], []

    def rec(fold, name, ok, detail):
        checks.append({"fold": fold, "check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            fails.append(f"[{fold}] {name}: {detail}")
        if verbose:
            print(f"  [{'PASS' if ok else 'FAIL'}] {fold} :: {name} — {detail}")

    pinned = load_pinned_universe()
    fold_train_files = {}
    fold_uni_syms = {}

    # ---- POLICY-PIN (MAJOR-1): everything below is checked against constants ----
    fold_set = man.get("fold_set")
    policy = POLICY.get(fold_set)
    if policy is None:
        rec("*", "POLICY-PIN fold set", False,
            f"unknown fold_set '{fold_set}' — only the pre-registered sets "
            f"{sorted(POLICY)} validate (the harness implements splits, never invents)")
    else:
        man_ids = sorted(f["fold_id"] for f in man["folds"])
        rec("*", "POLICY-PIN fold-set completeness", man_ids == sorted(policy["folds"]),
            f"manifest folds {man_ids} == pre-registered {sorted(policy['folds'])}")
        rec("*", "POLICY-PIN manifest rule params",
            man.get("n_assets") == policy["n_assets"]
            and man.get("min_days") == policy["min_days"],
            f"manifest n_assets={man.get('n_assets')} min_days={man.get('min_days')} == "
            f"pre-registered {policy['n_assets']}/{policy['min_days']}")

    for f in man["folds"]:
        fid = f["fold_id"]
        files = {k: os.path.join(_REPO, v) for k, v in f["files"].items()}
        try:
            tr0, tr1, tr_assets = _read_periods(files["train_pit"])
            ev0, ev1, ev_assets = _read_periods(files["eval_pit"])
            lt0, lt1, lt_assets = _read_periods(files["train_leaky"])
            le0, le1, le_assets = _read_periods(files["eval_leaky"])
        except Exception as e:  # unparseable artifact = leak-gate failure, not a crash
            rec(fid, "artifacts-parse", False, f"unreadable periods file: {e}")
            continue
        fold_train_files[fid] = {os.path.relpath(files["train_pit"], _REPO),
                                 os.path.relpath(files["train_leaky"], _REPO)}

        # ---- POLICY-PIN (MAJOR-1): parsed windows == the pre-registered constants ----
        if policy is not None:
            pol = policy["folds"].get(fid)
            rec(fid, "POLICY-PIN fold windows pre-registered",
                pol is not None and (tr0, tr1) == tuple(pol["train"])
                and (ev0, ev1) == tuple(pol["eval"]),
                (f"parsed train {tr0}..{tr1} / eval {ev0}..{ev1} == pre-registered "
                 f"{pol['train'][0]}..{pol['train'][1]} / {pol['eval'][0]}..{pol['eval'][1]}"
                 if pol else f"fold_id '{fid}' is NOT in the pre-registered {fold_set} set"))

        # ---- ASSERT-1: no train/eval overlap, from the PARSED files, dual parse ----
        t_end = max(_max_end_epoch(tr1), _max_end_epoch(lt1))
        e_start = min(_min_start_epoch(ev0), _min_start_epoch(le0))
        rec(fid, "ASSERT-1 train/eval windows disjoint", t_end < e_start,
            f"parsed train end {tr1} (<= {t_end}) vs eval start {ev0} (>= {e_start}); "
            f"loader bounds are end-INCLUSIVE (vectorbt_env.py:660)")
        rec(fid, "windows well-formed",
            tr0 < tr1 and ev0 < ev1 and (tr0, tr1) == (lt0, lt1) and (ev0, ev1) == (le0, le1),
            f"train {tr0}..{tr1}, eval {ev0}..{ev1}, leaky dates match pit dates")

        # ---- ASSERT-2a: what a training loader WOULD ingest from the train file ----
        s_ts = min(_min_start_epoch(tr0), _min_start_epoch(lt0))
        e_ts = t_end
        for table in ("ohlcv", "ohlcv_daily"):
            mx = con.execute(f"SELECT MAX(timestamp) FROM {table} "
                             f"WHERE timestamp >= ? AND timestamp <= ?",
                             (s_ts, e_ts)).fetchone()[0]
            if mx is None:
                detail = (f"VACUOUS — db has no {table} rows in the parsed train bounds "
                          f"(loader would load nothing here)")
                if table == "ohlcv" and fold_set == "real":
                    detail += ("; *** WARNING: these manifests are LAUNCH-READY against "
                               "trading_data.db, where this assert HAS teeth — run "
                               "`validate --db trading_data.db` (read-only) BEFORE "
                               "launching train jobs (SPEC pre-flight) ***")
                    print(f"  [WARN] {fid} :: ASSERT-2 ingest-max ohlcv is VACUOUS on "
                          f"this db — pre-flight `validate --db trading_data.db` "
                          f"required before any fold training launch")
                rec(fid, f"ASSERT-2 ingest-max {table}", True, detail)
            else:
                rec(fid, f"ASSERT-2 ingest-max {table}", mx < e_start,
                    f"max ingestable ts {datetime.fromtimestamp(mx, timezone.utc):%Y-%m-%d %H:%M} "
                    f"UTC < eval start (>= {e_start})")

        # ---- ASSERT-2b: PIT universe re-derivation from data <= train-end ONLY,
        # with the rule parameters PINNED to POLICY (MAJOR-1: the artifact's own rule
        # block is tamperable and is itself checked, never trusted) ----
        try:
            with open(files["universe"]) as fh:
                uni = json.load(fh)
            exp_n = policy["n_assets"] if policy else uni["rule"]["n_assets_cap"]
            exp_min = policy["min_days"] if policy else uni["rule"]["min_days"]
            if policy is not None:
                rec(fid, "POLICY-PIN universe rule",
                    uni["rule"]["n_assets_cap"] == exp_n
                    and uni["rule"]["min_days"] == exp_min,
                    f"artifact rule cap={uni['rule']['n_assets_cap']} min_days="
                    f"{uni['rule']['min_days']} == pre-registered {exp_n}/{exp_min} "
                    f"(re-derivation params are PINNED, never read from the artifact)")
            ok_asof = uni["as_of"] == tr1
            re_uni = derive_pit_universe(db_path, as_of=tr1, n_assets=exp_n,
                                         min_days=exp_min)
            same = re_uni["symbols"] == uni["symbols"]
            rec(fid, "ASSERT-2 PIT universe re-derivation", ok_asof and same,
                f"as_of=={tr1}: {ok_asof}; re-derived {re_uni['n_selected']} symbols "
                f"{'IDENTICAL (incl. order)' if same else 'MISMATCH — tampered or derived with post-train-end data'}")
            rec(fid, "periods-files carry the PIT universe",
                tr_assets == uni["symbols"] and ev_assets == uni["symbols"],
                "train/eval pit files carry universe_pit.json symbols verbatim "
                "(same order train==eval — TASK_740 landmine)")
            fold_uni_syms[fid] = uni["symbols"]
        except LeakError as e:
            rec(fid, "ASSERT-2 PIT universe re-derivation", False, str(e))
        rec(fid, "leaky files carry the pinned 64 byte-identically",
            lt_assets == pinned and le_assets == pinned,
            "LEAKY-UNIVERSE label + pinned order preserved")
        rec(fid, "leaky files labeled LOUDLY",
            "LEAKY-UNIVERSE" in json.load(open(files["train_leaky"]))["_comment"]
            and "LEAKY-UNIVERSE" in json.load(open(files["eval_leaky"]))["_comment"],
            "_comment contains LEAKY-UNIVERSE")

        # ---- corridor artifact bounds (if computed): never loaded past eval-end ----
        cpath = os.path.join(os.path.dirname(files["universe"]), "corridor.json")
        if os.path.exists(cpath):
            with open(cpath) as fh:
                cor = json.load(fh)
            ok = (cor["panel_window"][1] == ev1
                  and cor["panel_last_day_ts"] < _min_start_epoch(_date_add(ev1, 1))
                  and cor["eval_window"] == [ev0, ev1])
            rec(fid, "corridor panel bounded at eval-end", ok,
                f"panel {cor['panel_window'][0]}..{cor['panel_window'][1]}, last day ts "
                f"{cor['panel_last_day_ts']} < {ev1}+1d; eval slice {cor['eval_window']}")

    # ---- manifest cross-checks: a train job may ONLY reference a train periods-file ----
    for kind in ("train", "eval"):
        mpath = os.path.join(_REPO, man["manifests"][kind])
        if not os.path.exists(mpath):
            rec("*", f"{kind}_jobs.manifest exists", False, mpath)
            continue
        with open(mpath) as fh:
            lines = [ln.strip() for ln in fh
                     if ln.strip() and not ln.strip().startswith("#")]
        for ln in lines:
            job, dev, mem, cmd = ln.split(None, 3)
            fid = next((f["fold_id"] for f in man["folds"] if f"_{f['fold_id']}_" in job),
                       None)
            pf = _pf_from_cmd(cmd)
            if fid is None or pf is None:
                rec("*", f"manifest line parse ({job})", False,
                    f"cannot map job to a fold / no --periods-file in: {ln[:120]}")
                continue
            if not os.path.exists(os.path.join(_REPO, pf)):
                rec(fid, f"{kind} job {job} periods-file exists", False, pf)
                continue
            p0, p1, pf_assets = _read_periods(os.path.join(_REPO, pf))
            fdef = next(f for f in man["folds"] if f["fold_id"] == fid)
            if kind == "train":
                member = pf in fold_train_files.get(fid, set())
                dates_ok = (_max_end_epoch(p1)
                            < _min_start_epoch(fdef["eval_window"][0]))
                rec(fid, f"train job {job} references a fold TRAIN periods-file",
                    member and dates_ok,
                    f"--periods-file {pf}: membership in the fold's train files="
                    f"{member}; parsed window {p0}..{p1} ends before eval start "
                    f"{fdef['eval_window'][0]}={dates_ok}")
            else:
                # MAJOR-2 fix: window equality alone let ANY same-window file through
                # (incl. period_eval_leaky.json — the survivorship-leaky universe would
                # silently become the graded unseen-years number). Require file IDENTITY
                # with the fold's eval_pit AND assets == the PIT universe.
                exp_pf = fdef["files"]["eval_pit"]
                ident = pf == exp_pf
                win_ok = [p0, p1] == fdef["eval_window"]
                assets_ok = (fid in fold_uni_syms
                             and pf_assets == fold_uni_syms[fid])
                rec(fid, f"eval job {job} references the fold's eval_pit file",
                    ident and win_ok and assets_ok,
                    f"--periods-file {pf}: identity==files.eval_pit ({exp_pf})={ident}; "
                    f"window {p0}..{p1}=={fdef['eval_window']}={win_ok}; "
                    f"assets==universe_pit symbols={assets_ok}")
    con.close()

    if fails:
        raise LeakError("LEAK ASSERT FAILED — fold artifacts REJECTED:\n  "
                        + "\n  ".join(fails))
    print(f"[837] validate: ALL {len(checks)} checks PASS ({_rel(fold_root)})")
    return checks


# =============================================================================
# fold corridor — 830 machinery on the eval window (padded panel, eval-slice numbers)
# =============================================================================

def _load_corridor830():
    """Import the TASK_830 corridor module by path (the 832 importlib pattern — dir name
    '830_corridor' is not an identifier). Reused pieces: load_panel (audited daily panel +
    819 liquidity filter), book_target_signs (819 book via trader.strategy), replay_floor
    (the in-env floor walk), oracle_positions (path_solver ceiling instrument), FEE/SLIP
    constants. NEVER modified."""
    spec = importlib.util.spec_from_file_location("corridor830", CORRIDOR_830)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fold_corridor(db_path, fold, pad_days=PAD_DAYS, position_size_pct=None, c830=None):
    """Per-eval-window corridor {floor_inenv, reference, ceiling} per cost tier.

    Panel = [eval_start - pad_days, eval_end] via 830 ``load_panel`` (UNMODIFIED defaults:
    include_dead, zf_max=0.15, min_active=365 — the pad exists exactly because the raw
    eval windows are shorter than that filter tolerates; 831-EXT crash precedent). The pad
    is PAST data only; the panel NEVER extends beyond eval_end (asserted, and re-checked
    from the emitted artifact by validate_folds). Every reported number is computed on the
    eval-window decision slice only:
      floor_inenv — 830 replay_floor over the panel (the env book enters the eval window
        already positioned — a live floor strategy would too); per-decision rets index k
        maps to row warm+k (the 830 _regime_slices convention); slice = rows with
        day_ts in [eval_start, eval_end) (backtest half-open window convention).
      reference   — backtest.window_metrics (gate-2 scorer: frozen 819 positions on the
        WHOLE panel incl. warmup pre-history, metrics only on the slice).
      ceiling     — 830 oracle_positions (path_solver = measurement instrument ONLY, H5)
        scored by backtest.regime_series on [r_lo, r_hi) — identical audited accounting.
    Annualization everywhere = backtest._sharpe(., period=H_rebal*K=1440)."""
    if c830 is None:
        c830 = _load_corridor830()
    fid = fold["fold_id"]
    e0, e1 = fold["eval"]
    pan_start = _date_add(e0, -pad_days)
    day_ts, close, logc, syms, info = c830.load_panel(db_path, pan_start, e1)
    A = close.shape[1]
    if position_size_pct is None:
        position_size_pct = 0.5 / A                       # the 830 default sizing
    s_ts, e_ts = utc_epoch(e0), utc_epoch(e1)
    if int(day_ts[-1]) >= utc_epoch(_date_add(e1, 1)):    # leak-side guard, fail loud
        raise LeakError(f"{fid}: corridor panel extends past eval end {e1} "
                        f"(last day {int(day_ts[-1])})")

    tgt, pres, warm = c830.book_target_signs(logc)
    r_lo = int(np.searchsorted(day_ts, s_ts))
    r_hi = int(np.searchsorted(day_ts, e_ts))
    H = trader_config.STRATEGY.H_rebal
    period = H * trader_config.K

    # reference — gate-2 scorer on the eval slice (positions use the padded pre-history)
    ref = backtest.window_metrics(logc, day_ts, start_ts=s_ts, end_ts=e_ts)
    if ref is None or ref["n"] < MIN_EVAL_DECISIONS:
        raise LeakError(f"{fid}: eval window {e0}..{e1} too thin for the corridor "
                        f"(window_metrics n={None if ref is None else ref['n']})")

    # ceiling — oracle book on the identical slice, identical accounting
    pos_orc, orc_alpha = c830.oracle_positions(logc, pres)
    res = backtest.regime_series(logc, np.asarray(pos_orc), pres, r_lo, r_hi, H, warm)
    if res is None:
        raise LeakError(f"{fid}: oracle slice empty on {e0}..{e1}")
    og, ot, _ = res
    ceiling = {c: _f(backtest._sharpe(backtest.net_series(og, ot, c), period))
               for c in GATE_COSTS}

    # floor — 830 replay through step_vector, rets sliced to eval rows
    floors, floor_diag = {}, {}
    n_eval = None
    for c in GATE_COSTS:
        fee, slip = (0.0, 0.0) if c == 0 else (c830.FEE_BPS, c830.SLIP_BPS)
        rep = c830.replay_floor(close, tgt, warm, fee, slip, position_size_pct)
        dec_ts = day_ts[warm:warm + rep["n"]]
        sel = (dec_ts >= s_ts) & (dec_ts < e_ts)
        n_eval = int(sel.sum())
        if n_eval < MIN_EVAL_DECISIONS:
            raise LeakError(f"{fid}: only {n_eval} in-env eval decisions on {e0}..{e1}")
        floors[c] = _f(c830._sharpe(rep["rets"][sel]))
        floor_diag[c] = {"final_pv_panel": rep["final_pv"],
                         "bankrupt_at": rep["bankrupt_at"]}

    out = {
        "task": "TASK_837", "fold_id": fid, "eval_window": [e0, e1],
        "panel_window": [pan_start, e1], "pad_days": pad_days,
        "panel_last_day_ts": int(day_ts[-1]),
        "db": _rel(db_path), "n_assets": A, "n_days": int(day_ts.size), "warm": warm,
        "n_eval_decisions_env": n_eval, "n_eval_decisions_engine": int(ref["n"]),
        "position_size_pct": position_size_pct,
        "oracle_gross_alpha_panel": _f(orc_alpha),
        "per_tier": {str(c): {"floor_inenv": floors[c],
                              "reference": _f(ref["gross"] if c == 0 else ref[f"net{c}"]),
                              "ceiling": ceiling[c]} for c in GATE_COSTS},
        "bh_eval": _f(ref["bh"]),
        "floor_diag": {str(c): floor_diag[c] for c in GATE_COSTS},
        "provenance": ("830 corridor machinery (importlib, unmodified) on a "
                       f"{pad_days}d-back-padded panel; ALL numbers are eval-slice only; "
                       "universe = 830-pattern liquidity-filtered daily panel (NOT the "
                       "fold's 64) — same yardstick caveat as 835 SPEC #10. "
                       "Daily source: fixture daily == db daily (LEDGER 830/E0-REAL)."),
        # M6 (verify-panel): the exact edge mismatch vs the 835 fold scorecard, with sign.
        "slice_convention": (
            "corridor slice = backtest half-open [eval_start, eval_end) on UTC-midnight "
            "day_ts; the daily panel load is end-EXCLUSIVE, so the last graded corridor "
            "decision is eval_end-2d (the final panel row only labels the previous "
            "decision). The 835 fold scorecard's minute loader is end-INCLUSIVE at LOCAL "
            "midnight and consumes window_size (60) rows of the eval panel for obs "
            "warmup: its graded decisions run ~eval_start+60d .. ~eval_end-1d. SIGN of "
            "the mismatch: the corridor covers ~60 MORE daily decisions at the LEFT edge "
            "and ~1 FEWER at the RIGHT edge (tz skew <=1d). Neither instrument ever "
            "touches data past eval_end. Documented (SPEC §corridor), not tuned."),
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    print(f"[837] corridor {fid} ({e0}..{e1}, {A} assets, {n_eval} eval decisions): "
          f"net@21 [{out['per_tier']['21']['floor_inenv']}, "
          f"{out['per_tier']['21']['ceiling']}] ref {out['per_tier']['21']['reference']}")
    return out


def corridor_for_root(db_path, fold_root, folds, pad_days=PAD_DAYS):
    c830 = _load_corridor830()
    outs = []
    for fold in folds:
        cor = fold_corridor(db_path, fold, pad_days=pad_days, c830=c830)
        outs.append(_write_json(os.path.join(fold_root, fold["fold_id"],
                                             "corridor.json"), cor))
    return outs


# =============================================================================
# aggregation — per-fold scorecards + corridors -> walk-forward summary + LEDGER row
# =============================================================================

def _fmt(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:+.3f}"


def aggregate(fold_root, scorecard_map, out_path, tag):
    """Fold scorecards (835 result JSONs — real runs or smoke stubs) graded against each
    fold's OWN corridor. Per-fold corridor positions ALWAYS shown; the pooled block never
    exists without them (spec: no pooled Sharpe without the per-fold view). Partial
    rollouts withhold corridor booleans (the 835 completeness-guard convention)."""
    with open(os.path.join(fold_root, "fold_manifest.json")) as fh:
        man = json.load(fh)
    rows = []
    for f in man["folds"]:
        fid = f["fold_id"]
        cpath = os.path.join(fold_root, fid, "corridor.json")
        if not os.path.exists(cpath):
            raise LeakError(f"aggregate: no corridor.json for {fid} — run `corridor` "
                            f"first (a fold is never graded without its own corridor)")
        with open(cpath) as fh:
            cor = json.load(fh)
        sc_path = scorecard_map.get(fid)
        if sc_path is None or not os.path.exists(sc_path):
            raise LeakError(f"aggregate: no scorecard for {fid} (got {sc_path})")
        with open(sc_path) as fh:
            sc = json.load(fh)
        t21, t0 = sc["tiers"]["21"], sc["tiers"]["0"]
        partial = bool(t21.get("partial") or t0.get("partial"))
        net21, gross = _f(t21.get("sharpe_full")), _f(t0.get("sharpe_full"))
        fl, rf, ce = (cor["per_tier"]["21"][k] for k in
                      ("floor_inenv", "reference", "ceiling"))
        span = None if (fl is None or ce is None) else max(ce - fl, 1e-10)
        pct_hold = t21.get("pct_hold_agg")
        rows.append({
            "fold_id": fid,
            "train_window": f["train_window"], "eval_window": f["eval_window"],
            "scorecard": {"path": _rel(sc_path),
                          "checkpoint_label": sc.get("checkpoint_label"),
                          "stub": bool(sc.get("stub", False)),
                          "net21": net21, "gross": gross,
                          "pct_hold_agg_net21": pct_hold, "partial": partial,
                          "final_pv_net21": t21.get("final_pv")},
            "corridor": {"path": _rel(cpath), "per_tier": cor["per_tier"],
                         "n_eval_decisions_env": cor["n_eval_decisions_env"]},
            "position_in_corridor_net21": (None if partial or net21 is None or span is None
                                           else float((net21 - fl) / span)),
            "beats_floor_net21": (None if partial or net21 is None or fl is None
                                  else bool(net21 >= fl)),
            "beats_reference_net21": (None if partial or net21 is None or rf is None
                                      else bool(net21 >= rf)),
            "hold_collapse": (None if pct_hold is None
                              else bool(pct_hold > HOLD_COLLAPSE_PCT)),
        })

    # graded = non-partial folds with a number — the SAME set feeds mean/min/worst
    # (a partial fold must not name worst_fold while being excluded from the stats).
    graded = [r for r in rows
              if r["scorecard"]["net21"] is not None and not r["scorecard"]["partial"]]
    valid = [r["scorecard"]["net21"] for r in graded]
    pooled = {
        "n_folds": len(rows), "n_folds_graded": len(valid),
        "mean_net21": _f(np.mean(valid)) if valid else None,
        "min_net21": _f(np.min(valid)) if valid else None,
        "worst_graded_fold": (min(graded, key=lambda r: r["scorecard"]["net21"])
                              ["fold_id"] if graded else None),
        "all_beat_floor": (all(r["beats_floor_net21"] for r in rows)
                           if all(r["beats_floor_net21"] is not None for r in rows)
                           else None),
        "all_beat_reference": (all(r["beats_reference_net21"] for r in rows)
                               if all(r["beats_reference_net21"] is not None for r in rows)
                               else None),
        "note": "pooled view — NEVER read without the per-fold rows above "
                "(TASK_837: no pooled Sharpe without the per-fold view)",
    }
    is_stub = any(r["scorecard"]["stub"] for r in rows)
    per_fold_txt = " · ".join(
        f"{r['fold_id']} net@21 {_fmt(r['scorecard']['net21'])} in "
        f"[{_fmt(r['corridor']['per_tier']['21']['floor_inenv'])}, "
        f"{_fmt(r['corridor']['per_tier']['21']['ceiling'])}] "
        f"(ref {_fmt(r['corridor']['per_tier']['21']['reference'])}, "
        f"pos {_fmt(r['position_in_corridor_net21'])})"
        for r in rows)
    stub_note = (" FIXTURE-SMOKE with STUB scorecards — plumbing proof only, no verdict."
                 if is_stub else
                 " Scheduled-run numbers; R4 graduation verdict applied by the planner.")
    ledger_row = (
        f"| 837/R4 | {date.today().isoformat()} | H1 (learned) | Walk-forward ({tag}): "
        f"per-fold eval vs each window's OWN corridor "
        f"({' + '.join(r['fold_id'] for r in rows)}) | {per_fold_txt}; pooled mean "
        f"{_fmt(pooled['mean_net21'])} min {_fmt(pooled['min_net21'])} over "
        f"{pooled['n_folds_graded']}/{pooled['n_folds']} folds.{stub_note} | "
        f"per-fold positions above (never pooled alone) | {_rel(out_path)} |")

    summary = {"task": "TASK_837", "tag": tag, "fold_set": man["fold_set"],
               "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "folds": rows, "pooled": pooled, "ledger_row": ledger_row}
    _write_json(out_path, summary)

    print("=" * 90)
    print(f"WALK-FORWARD SUMMARY ({tag}) — per-fold FIRST (house rule)")
    print(f"{'fold':>12} | {'net@21':>8} {'gross':>8} | {'floor':>8} {'ref':>8} "
          f"{'ceil':>8} | {'pos':>6} {'>=floor':>7} {'>=ref':>5} {'collapse':>8}")
    for r in rows:
        pt = r["corridor"]["per_tier"]["21"]
        flag = {True: "YES", False: "no", None: "n/a"}
        print(f"{r['fold_id']:>12} | {_fmt(r['scorecard']['net21']):>8} "
              f"{_fmt(r['scorecard']['gross']):>8} | {_fmt(pt['floor_inenv']):>8} "
              f"{_fmt(pt['reference']):>8} {_fmt(pt['ceiling']):>8} | "
              f"{_fmt(r['position_in_corridor_net21']):>6} "
              f"{flag[r['beats_floor_net21']]:>7} {flag[r['beats_reference_net21']]:>5} "
              f"{flag[r['hold_collapse']]:>8}")
    print(f"pooled: mean net@21 {_fmt(pooled['mean_net21'])} min {_fmt(pooled['min_net21'])} "
          f"({pooled['n_folds_graded']}/{pooled['n_folds']} folds) — {pooled['note']}")
    print("LEDGER row:")
    print(ledger_row)
    print("=" * 90)
    return summary


# =============================================================================
# fixture smoke — end-to-end + the NEGATIVE leak tests (teeth)
# =============================================================================

def _stub_scorecard(path, fid, net21, gross, pct_hold):
    """Tiny 835-schema stub (the keys aggregate consumes) — smoke plumbing only."""
    def tier(sh):
        return {"sharpe_full": sh, "final_pv": 100000.0 * (1 + sh / 100),
                "pct_hold_agg": pct_hold, "partial": False,
                "steps_completed": 20, "steps_expected": 20,
                "avg_long": 1.0, "avg_short": 1.0, "bankrupt_at": -1}
    return _write_json(path, {
        "tag": f"837-smoke-stub-{fid}", "stub": True,
        "checkpoint_label": "STUB (no model — smoke aggregation plumbing)",
        "tiers": {"21": tier(net21), "0": tier(gross)}})


def _clone_fold_tree(smoke_root, case):
    """Copy the fold tree for a control/negative run and re-point EVERY repo-relative
    path inside it — fold_manifest.json AND both job manifests. MAJOR-3 fix: the job
    manifests were previously left pointing at the original tree, so even a PRISTINE
    copy was rejected on paths and the negative tests had zero discriminating power."""
    neg_root = os.path.join(smoke_root, "_negative", case)
    if os.path.exists(neg_root):
        shutil.rmtree(neg_root)
    shutil.copytree(smoke_root, neg_root, ignore=shutil.ignore_patterns("_negative"))
    old_rel, new_rel = _rel(smoke_root), _rel(neg_root)
    for name in ("fold_manifest.json", "train_jobs.manifest", "eval_jobs.manifest"):
        p = os.path.join(neg_root, name)
        with open(p) as fh:
            txt = fh.read()
        with open(p, "w") as fh:
            fh.write(txt.replace(old_rel, new_rel))
    return neg_root


def _noop_control(smoke_root, db_path):
    """CONTROL (MAJOR-3): an UNMUTATED clone must validate CLEAN — otherwise every
    'rejection' below is a path confound, not a leak gate, and the smoke FAILS."""
    root = _clone_fold_tree(smoke_root, "control_noop")
    try:
        checks = validate_folds(root, db_path, verbose=False)
    except LeakError as e:
        raise RuntimeError(
            f"NEGATIVE-TEST CONTROL FAILED: the UNMUTATED clone was rejected — the "
            f"negative tests are path-confounded, not gate-driven:\n{e}")
    print(f"[837][negative:control_noop] CLEAN as required ({len(checks)} checks PASS "
          f"on the unmutated clone)")
    return {"case": "control_noop", "clean": True, "n_checks": len(checks)}


def _corrupt_and_expect_reject(smoke_root, db_path, case, mutate, expect):
    """Clone the validated smoke fold tree, apply ``mutate``, and REQUIRE validate to
    (a) REJECT it AND (b) NAME the targeted gate (``expect`` substring) in the
    rejection. A silent pass OR a rejection by the wrong check is itself a failure —
    a refactor that de-tooths the targeted gate must break the smoke."""
    neg_root = _clone_fold_tree(smoke_root, case)
    mutate(neg_root)
    try:
        validate_folds(neg_root, db_path, verbose=False)
    except LeakError as e:
        msg = str(e)
        if expect not in msg:
            raise RuntimeError(
                f"NEGATIVE TEST '{case}': REJECTED but NOT by the targeted gate "
                f"'{expect}' — the rejection reads:\n{msg}")
        line = next((l.strip() for l in msg.splitlines() if expect in l), expect)
        print(f"[837][negative:{case}] REJECTED by targeted gate '{expect}' — {line}")
        return {"case": case, "rejected": True, "targeted_check": expect,
                "named_in_rejection": True, "message": line}
    raise RuntimeError(f"NEGATIVE TEST FAILED: corrupted fold '{case}' was ACCEPTED — "
                       f"the leak gates have no teeth")


def _neg_train_into_eval(root):
    """Corruption A: the train periods-file end_date is pushed INTO the eval window."""
    p = os.path.join(root, "smokefold1", "period_train.json")
    with open(p) as fh:
        d = json.load(fh)
    d["periods"][0]["end_date"] = "2025-11-15"      # 15 days inside the eval window
    with open(p, "w") as fh:
        json.dump(d, fh, indent=2)


def _neg_tampered_universe(root):
    """Corruption B: universe_pit.json symbol list tampered (rank-1 symbol replaced) —
    the re-derivation equality must reject it."""
    p = os.path.join(root, "smokefold2", "universe_pit.json")
    with open(p) as fh:
        d = json.load(fh)
    d["symbols"][0] = "NOT_A_PIT_MEMBER/USD"
    with open(p, "w") as fh:
        json.dump(d, fh, indent=2)


def _neg_train_job_on_eval_file(root):
    """Corruption C: a train manifest job re-pointed at the fold's EVAL periods-file."""
    p = os.path.join(root, "train_jobs.manifest")
    with open(p) as fh:
        txt = fh.read()
    with open(p, "w") as fh:
        fh.write(txt.replace("period_train.json", "period_eval.json", 1))


def _neg_consistent_retamper(root):
    """Corruption D (MAJOR-1 teeth): an INTERNALLY-CONSISTENT tamper — the universe is
    truncated to 3 with the rule block, counts AND both pit periods-files edited to
    match. Every self-consistency check passes; only the pre-registration PIN
    (POLICY-PIN universe rule + the PINNED-params re-derivation) can reject it."""
    fdir = os.path.join(root, "smokefold1")
    up = os.path.join(fdir, "universe_pit.json")
    with open(up) as fh:
        u = json.load(fh)
    syms = u["symbols"][:3]
    u["symbols"], u["asset_ids"] = syms, u["asset_ids"][:3]
    u["counts"] = {s: u["counts"][s] for s in syms}
    u["n_selected"] = 3
    u["rule"]["n_assets_cap"] = 3
    with open(up, "w") as fh:
        json.dump(u, fh, indent=2)
    for name in ("period_train.json", "period_eval.json"):
        p = os.path.join(fdir, name)
        with open(p) as fh:
            d = json.load(fh)
        d["periods"][0]["assets"] = syms
        with open(p, "w") as fh:
            json.dump(d, fh, indent=2)


def _neg_eval_job_on_leaky(root):
    """Corruption E (MAJOR-2 teeth): an eval manifest job re-pointed at the SAME-WINDOW
    period_eval_leaky.json — window equality alone would let the survivorship-leaky
    universe silently become the graded unseen-years number; the eval-file IDENTITY
    gate must reject it."""
    p = os.path.join(root, "eval_jobs.manifest")
    with open(p) as fh:
        txt = fh.read()
    with open(p, "w") as fh:
        fh.write(txt.replace("period_eval.json", "period_eval_leaky.json", 1))


def run_smoke(args):
    """Gate (a): materialize + validate + corridor + stub-aggregate on the fixture,
    then the negative battery (no-op CONTROL must validate CLEAN + five corrupted
    clones each REJECTED by its targeted gate), AND the real-fold rails (materialize +
    validate [+ corridor unless --skip-real-corridor]). Writes
    results_fixture_smoke.json."""
    db = FIXTURE_DB
    result = {"task": "TASK_837", "smoke": True, "db": _rel(db),
              "generated": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    # ---- 1. smoke folds: materialize -> validate (leak gates) ----
    smoke_root = os.path.join(EXP_DIR, "folds", "smoke")
    if os.path.exists(smoke_root):
        shutil.rmtree(smoke_root)
    materialize(db, SMOKE_FOLDS, "smoke", smoke_root,
                n_assets=SMOKE_PIT_N_ASSETS, min_days=PIT_MIN_DAYS, jobs_db=_rel(db))
    checks = validate_folds(smoke_root, db)
    result["smoke_folds"] = {"root": _rel(smoke_root), "n_checks": len(checks),
                             "all_pass": True}

    # ---- 2. per-fold corridors (830 machinery, eval-slice) ----
    corridor_for_root(db, smoke_root, SMOKE_FOLDS, pad_days=args.pad_days)
    checks2 = validate_folds(smoke_root, db, verbose=False)   # corridor bounds re-checked
    result["smoke_corridors"] = {
        f["fold_id"]: json.load(open(os.path.join(smoke_root, f["fold_id"],
                                                  "corridor.json")))["per_tier"]
        for f in SMOKE_FOLDS}
    result["smoke_folds"]["n_checks_with_corridor"] = len(checks2)

    # ---- 3. stub scorecards -> aggregation ----
    sc_map = {
        "smokefold1": _stub_scorecard(os.path.join(smoke_root, "smokefold1",
                                                   "scorecard_stub.json"),
                                      "smokefold1", net21=0.30, gross=0.55, pct_hold=95.0),
        "smokefold2": _stub_scorecard(os.path.join(smoke_root, "smokefold2",
                                                   "scorecard_stub.json"),
                                      "smokefold2", net21=-0.20, gross=0.05, pct_hold=99.5),
    }
    summary = aggregate(smoke_root, sc_map,
                        os.path.join(smoke_root, "walkforward_summary.json"),
                        tag="fixture-smoke-stub")
    result["smoke_aggregate"] = {"pooled": summary["pooled"],
                                 "ledger_row": summary["ledger_row"]}

    # ---- 4. NEGATIVE leak tests: no-op control CLEAN first (de-confound), then every
    # corrupted fold must be REJECTED *by its targeted gate* (named in the rejection) ----
    result["negative_control"] = _noop_control(smoke_root, db)
    result["negative_tests"] = [
        _corrupt_and_expect_reject(smoke_root, db, "train_into_eval",
                                   _neg_train_into_eval,
                                   "ASSERT-1"),
        _corrupt_and_expect_reject(smoke_root, db, "tampered_pit_universe",
                                   _neg_tampered_universe,
                                   "ASSERT-2 PIT universe re-derivation"),
        _corrupt_and_expect_reject(smoke_root, db, "train_job_on_eval_file",
                                   _neg_train_job_on_eval_file,
                                   "references a fold TRAIN periods-file"),
        _corrupt_and_expect_reject(smoke_root, db, "consistent_universe_retamper",
                                   _neg_consistent_retamper,
                                   "POLICY-PIN universe rule"),
        _corrupt_and_expect_reject(smoke_root, db, "eval_job_on_leaky_universe",
                                   _neg_eval_job_on_leaky,
                                   "references the fold's eval_pit file"),
    ]

    # ---- 5. the REAL fold rails (the default manifest deliverable) ----
    real_root = os.path.join(EXP_DIR, "folds", "real")
    materialize(db, REAL_FOLDS, "real", real_root,
                n_assets=PIT_N_ASSETS, min_days=PIT_MIN_DAYS, jobs_db="trading_data.db")
    if not args.skip_real_corridor:
        corridor_for_root(db, real_root, REAL_FOLDS, pad_days=args.pad_days)
    real_checks = validate_folds(real_root, db)
    result["real_folds"] = {
        "root": _rel(real_root), "n_checks": len(real_checks), "all_pass": True,
        "corridors": ({f["fold_id"]: json.load(open(os.path.join(
            real_root, f["fold_id"], "corridor.json")))["per_tier"]
            for f in REAL_FOLDS} if not args.skip_real_corridor else "SKIPPED"),
        "universe_note": "PIT universes from the fixture's FULL ohlcv_daily panel "
                         "(fixture daily == db daily, LEDGER 830/E0-REAL); scheduled "
                         "re-materialization vs trading_data.db must reproduce them "
                         "content-identically (M4 contract: volatile fields excluded — "
                         "SPEC §PIT).",
    }

    result["verdict"] = ("PASS — folds materialized with correct window boundaries; leak "
                         "asserts + POLICY pre-registration PIN enforced IN CODE (good "
                         "folds pass, the UNMUTATED control clone validates CLEAN, and "
                         "all 5 corrupted folds are REJECTED each BY ITS TARGETED gate); "
                         "corridors eval-slice-bounded; aggregation emits per-fold-first "
                         "summary + LEDGER row. Plumbing proof — no training, no "
                         "promotion claims.")
    out = os.path.join(EXP_DIR, "results_fixture_smoke.json")
    _write_json(out, result)
    print(f"\n[837] SMOKE PASS -> {_rel(out)}")
    return result


# =============================================================================
# CLI
# =============================================================================

def _fold_set(name):
    return {"real": REAL_FOLDS, "smoke": SMOKE_FOLDS}[name]


def main():
    ap = argparse.ArgumentParser(description="TASK_837 walk-forward split harness (R4)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("materialize", help="emit fold periods-files/universes/manifests")
    p.add_argument("--db", default=FIXTURE_DB,
                   help="db for PIT universe derivation (read-only; default fixture — "
                        "scheduled runs pass trading_data.db)")
    p.add_argument("--fold-set", choices=("real", "smoke"), default="real",
                   help="pre-registered fold set (default: the REAL folds)")
    p.add_argument("--out-root", default=None,
                   help="output root (default: <this dir>/folds/<fold-set>)")
    p.add_argument("--n-assets", type=int, default=None,
                   help="PIT cap (default 64 real / 8 smoke — pre-registered)")
    p.add_argument("--min-days", type=int, default=PIT_MIN_DAYS)
    p.add_argument("--jobs-db", default=None,
                   help="db path written into job manifests (default: trading_data.db "
                        "for real, the fixture for smoke)")

    p = sub.add_parser("validate", help="run the leak gates on generated fold artifacts")
    p.add_argument("--db", default=FIXTURE_DB)
    p.add_argument("--fold-root", required=True)

    p = sub.add_parser("corridor", help="per-fold eval-window corridors (830 machinery)")
    p.add_argument("--db", default=FIXTURE_DB)
    p.add_argument("--fold-set", choices=("real", "smoke"), default="real")
    p.add_argument("--fold-root", default=None)
    p.add_argument("--pad-days", type=int, default=PAD_DAYS)

    p = sub.add_parser("aggregate", help="fold scorecards + corridors -> summary")
    p.add_argument("--fold-root", required=True)
    p.add_argument("--scorecard", action="append", default=[],
                   metavar="FOLD_ID=PATH", help="835 result JSON per fold (repeatable)")
    p.add_argument("--out", default=None)
    p.add_argument("--tag", default="walkforward")

    p = sub.add_parser("smoke", help="fixture smoke: end-to-end + negative leak tests")
    p.add_argument("--pad-days", type=int, default=PAD_DAYS)
    p.add_argument("--skip-real-corridor", action="store_true",
                   help="skip the real-fold corridor computation (rails only)")

    args = ap.parse_args()
    try:
        if args.cmd == "materialize":
            out_root = args.out_root or os.path.join(EXP_DIR, "folds", args.fold_set)
            n_assets = args.n_assets or (PIT_N_ASSETS if args.fold_set == "real"
                                         else SMOKE_PIT_N_ASSETS)
            jobs_db = args.jobs_db or ("trading_data.db" if args.fold_set == "real"
                                       else _rel(args.db))
            materialize(args.db, _fold_set(args.fold_set), args.fold_set, out_root,
                        n_assets=n_assets, min_days=args.min_days, jobs_db=jobs_db)
            validate_folds(out_root, args.db)
        elif args.cmd == "validate":
            validate_folds(args.fold_root, args.db)
        elif args.cmd == "corridor":
            root = args.fold_root or os.path.join(EXP_DIR, "folds", args.fold_set)
            corridor_for_root(args.db, root, _fold_set(args.fold_set),
                              pad_days=args.pad_days)
            validate_folds(root, args.db, verbose=False)
        elif args.cmd == "aggregate":
            sc = dict(kv.split("=", 1) for kv in args.scorecard)
            out = args.out or os.path.join(args.fold_root, "walkforward_summary.json")
            aggregate(args.fold_root, sc, out, tag=args.tag)
        elif args.cmd == "smoke":
            run_smoke(args)
    except LeakError as e:
        print("\n" + "!" * 78, file=sys.stderr)
        print(f"LEAK GATE: {e}", file=sys.stderr)
        print("!" * 78, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
