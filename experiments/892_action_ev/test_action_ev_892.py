#!/usr/bin/env python3
"""TASK_892 regression suite — acceptance gates (i)-(v) plus the fix-pass gates. Standalone
script (835 house style: ``failures`` list, PASS/FAIL summary, exit 0/1), not pytest. Run:
    trader_env/bin/python experiments/892_action_ev/test_action_ev_892.py

FIX-PASS GATES (each one goes RED without its fix):
  * (boot)        BLOCKER 1 — a sample below the block bootstrap's power floor reports
                  INSUFFICIENT-N with a NULL CI, never a band and never a zero-width interval.
                  Exercised THROUGH ``ext_windows`` with a deliberately thin bucket, which the
                  shipped suite never did (it passed ``ext_windows=None`` everywhere).
  * (fail-window) finding 6 — an INSTRUMENT FAIL suppresses EVERY per-regime bucket's band.
  * (budget)      BLOCKER 2 — the budget check reads the walk's REALIZED allocation, refuses to
                  apply the dead ``position_size_pct`` under weight semantics, and a real
                  ``--vector-weight-rebalance`` CLI walk (SPEC.md's mandated mode, never
                  exercised before) reports ZERO violations.
  * (pin)         finding 5 — a real ``--checkpoint`` walk EMBEDS 886's mechanism-typed record
                  (886's verified/certified split inherited, not synthesized); a fresh-init
                  smoke walk says truthfully that it has none.
  * (disclose)    the pre-registration deviation is inline in BOTH artifact kinds, and
                  SPEC.md:46's promised third matched moment (book width) is computed.
  * (abbrev)      finding 8 — CLASS-CLOSING: every long option that existed at HEAD keeps its
                  minimal unique abbreviation (the new flag must steal none). Set-theoretic over
                  all HEAD options + behavioural against the real parser + 2 real-CLI pins.
  * (op-coherence) finding 9 — the null-operator flags REFUSE under an operator that cannot
                  honour them, instead of silently producing a complete-looking ordinary eval
                  that never opens the sidecar.

CYCLE-3 GATES (each one goes RED without its fix — the RED was reproduced against the shipped
code before the fix, and the reproduction is what each gate now asserts):
  * (schema)      BLOCKER 1 — report.py HARD-READS every input trace: an unrecognized/absent
                  ``schema`` and an absent ``provenance.budget_conservation_violations`` are
                  REFUSALS, not a silent "zero violations". RED: docs with neither field produced
                  band=EDGE, instrument_fail=[], violations=0.
  * (power)       finding 2 — the block bootstrap's banding bar is CALIBRATED against measured
                  true-null coverage, and a sample above the degeneracy floor but below that bar
                  reads LOW-POWER (CI reported, band refused). RED: 2 blocks banded freely at a
                  measured 78.5% coverage (22% of true nulls banding EDGE/ANTI-EDGE).
  * (zero-width)  finding 3 — a zero-width interval never carries a band, at ANY n. RED:
                  s=full(42,0.02) vs n=full(42,0.0) -> ci_lo == ci_hi == 0.02, band EDGE.
  * (race)        finding 4 — concurrent ``trace.py --smoke`` runs do not collide on one shared
                  default checkpoint dir. RED: 2 of 3 concurrent runs died in orbax.
  * (nan-budget)  finding 5 — a non-finite allocation is a budget violation (``NaN > x`` is
                  False, so the bare overdraft test could not see it). RED: [0.5, nan, 0.9] -> 0.
  * (walk-kind)   finding 6 — ``walk_kind`` is a VALIDATED ENUM, not an allow-list of two
                  positive raises. RED: walk_kind="bogus-walk-kind" emitted a trace whose note
                  claimed a fresh-init smoke walk over a real checkpoint_label.
  * (ci-args)     finding 7 — ``block``/``n_boot`` are validated with a clear message instead of
                  crashing inside numpy. RED: ZeroDivisionError / negative-dimensions /
                  IndexError.
  * (atomic)      cycle-3 minor — every artifact write goes through ``<path>.tmp`` + os.replace,
                  so a kill mid-dump cannot leave a truncated JSON at the trusted path.

D51-REVIEW FIX-PASS GATES (each one goes RED without its fix; both reproduced first):
  * (schema)/(fee-tier) MAJOR 1 — the cost tier is HARD-READ like the budget field:
                  ``tier_bp``/``fee_bps``/``slip_bps`` are in report.REQUIRED_TRACE_PROVENANCE and
                  subscripted, never ``.get(..., 0)``. RED: a GENUINE fee16/slip5 pair with
                  ``provenance.tier_bp`` deleted graded as a citable EDGE, instrument_fail=[], on
                  identical numbers (delta 0.0207906523) instead of DIAGNOSTIC-ONLY. Gate (schema)
                  closes the CLASS (every hard-read provenance key's omission is refused by name);
                  gate (fee-tier) adds the omitted-field case and the mislabeled-tier sibling
                  (tier_bp=0 over real fee_bps=16/slip_bps=5 is still a fee-tier read).
  * (subject-class) MAJOR 2 — the never-pooled CLAIM matches the WIRING: report.py's AST is read
                  for live ``assert_never_pooled`` calls (there are none — no aggregator exists),
                  and with none, every emitted report must carry
                  ``disclosures.NEVER_POOLED_OPEN_ITEM``. RED: report.py claimed the rule was
                  "enforced STRUCTURALLY here, not merely by convention" over a guard nothing
                  called, and no artifact said otherwise.

Gate (v) precedent (run-33 rule): the null operator (--decision-rule random-matched) is
exercised via a REAL subprocess CLI invocation of eval_vector.py, never by calling/stubbing its
internal functions directly. Everything else here calls the real, unstubbed module functions
(no monkeypatching anywhere in this file).
"""
import json
import os
import subprocess
import sys
import tempfile
import shutil

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
EV_PY = os.path.join(_REPO, "experiments", "835_stage4_eval", "eval_vector.py")
TRACE_PY = os.path.join(EXP_DIR, "trace.py")
BUILD_MARGINALS_PY = os.path.join(EXP_DIR, "build_action_marginals.py")
BASELINE_SMOKE_JSON = os.path.join(_REPO, "experiments", "835_stage4_eval",
                                   "results_fixture_smoke.json")
FIXTURE_DB = os.path.join(_REPO, "fixtures", "trading_fixture.db")

sys.path.insert(0, EXP_DIR)
import trace as trace_mod          # noqa: E402  (the module under light exercise, not stubbed)
import report as report_mod        # noqa: E402
import build_action_marginals as bam_mod   # noqa: E402
import disclosures as disclosures_mod      # noqa: E402


def _run(argv, timeout=280):
    return subprocess.run([sys.executable] + argv, cwd=_REPO, capture_output=True, text=True,
                          timeout=timeout)


# =============================================================================
# gate (i) — default OFF, byte-identical to HEAD (proxy: the TRACKED committed baseline
# results_fixture_smoke.json IS a HEAD-generated artifact this exact CLI invocation
# reproduces; a real worktree A/B was additionally run by hand for this task — see the task
# report — this test is the PERMANENT regression asset that future changes must not break).
# =============================================================================
_VOLATILE_TOP_KEYS = {"out_pointer", "ledger_row", "db"}


def _diff_ignoring_volatile(a, b, path=""):
    diffs = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if path == "" and k in _VOLATILE_TOP_KEYS:
                continue
            if k not in a:
                diffs.append(f"MISSING IN BASELINE {path}.{k}")
            elif k not in b:
                diffs.append(f"MISSING IN CANDIDATE {path}.{k}")
            else:
                diffs.extend(_diff_ignoring_volatile(a[k], b[k], f"{path}.{k}"))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"LEN DIFF {path}: {len(a)} != {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                diffs.extend(_diff_ignoring_volatile(x, y, f"{path}[{i}]"))
    else:
        if a != b:
            diffs.append(f"DIFF {path}: {a!r} != {b!r}")
    return diffs


def gate_i_default_off_parity(failures, work):
    if not os.path.exists(BASELINE_SMOKE_JSON):
        failures.append(f"gate (i): committed baseline missing: {BASELINE_SMOKE_JSON}")
        return
    r = _run([EV_PY, "--smoke", "--out-dir", work])
    ok = r.returncode == 0
    print(f"    [{'PASS' if ok else 'FAIL'}] gate (i): `eval_vector.py --smoke` "
         f"(default decision-rule=argmax, operator=search) exits 0")
    if not ok:
        failures.append(f"gate (i): eval_vector.py --smoke exit {r.returncode}: "
                        f"{r.stderr[-2000:]}")
        return
    with open(os.path.join(work, "results_fixture_smoke.json")) as fh:
        candidate = json.load(fh)
    with open(BASELINE_SMOKE_JSON) as fh:
        baseline = json.load(fh)
    diffs = _diff_ignoring_volatile(baseline, candidate)
    ok = not diffs
    print(f"    [{'PASS' if ok else 'FAIL'}] gate (i): every scored block byte-identical to "
         f"the committed baseline ({len(diffs)} diffs)")
    if not ok:
        failures.append("gate (i): --smoke default OFF path diverged from the committed "
                        f"baseline:\n  " + "\n  ".join(diffs[:20]))


# =============================================================================
# gate (ii) — the null is matched by construction (statistical property of
# random_matched_actions itself, at a scale where sampling noise cannot obscure it)
# =============================================================================

def gate_ii_matched_by_construction(failures):
    ev = trace_mod._load_eval_vector()
    n_assets, n_decisions = 6, 20000
    rng_marginals = np.random.default_rng(0)
    marginals = rng_marginals.dirichlet(np.ones(5), size=n_assets)
    draws = ev.random_matched_actions(marginals, n_decisions, n_assets, seed=42)
    empirical = np.stack([np.bincount(draws[:, a], minlength=5) / n_decisions
                          for a in range(n_assets)])
    max_dev_pp = 100.0 * float(np.max(np.abs(empirical - marginals)))
    # NOTE: SPEC.md's 0.5pp bar is the REPORT-LEVEL activity-match threshold at REAL-LEG scale
    # (~1673 decisions x 64 assets pooled over >=20 null seeds). This unit test demonstrates
    # the SAME convergence at a much smaller, CI-cheap n — binomial sampling noise at n=20000
    # per cell is ~0.3-0.35pp (1 sigma) even at p=0.5, so the bound here is intentionally looser
    # (1.0pp) to avoid a flaky test while still proving the mechanism converges to the input
    # marginals rather than to something else.
    ok = max_dev_pp <= 1.0
    print(f"    [{'PASS' if ok else 'FAIL'}] gate (ii): random_matched_actions empirical "
         f"per-asset distribution converges to the input marginals at n={n_decisions} "
         f"(max dev {max_dev_pp:.4f}pp, bound 1.0pp — see note)")
    if not ok:
        failures.append(f"gate (ii): matched-by-construction max deviation {max_dev_pp:.4f}pp "
                        f"> 1.0pp at n={n_decisions}")


# =============================================================================
# gate (iii) — path-independence: same seed -> bit-identical; different seed -> different,
# same marginals
# =============================================================================

def gate_iii_path_independence(failures):
    ev = trace_mod._load_eval_vector()
    rng = np.random.default_rng(1)
    marginals = rng.dirichlet(np.ones(5), size=4)
    a = ev.random_matched_actions(marginals, 500, 4, seed=7)
    b = ev.random_matched_actions(marginals, 500, 4, seed=7)
    c = ev.random_matched_actions(marginals, 500, 4, seed=8)
    same_seed_identical = np.array_equal(a, b)
    diff_seed_differs = not np.array_equal(a, c)
    print(f"    [{'PASS' if same_seed_identical else 'FAIL'}] gate (iii): same seed -> "
         f"bit-identical null walk")
    print(f"    [{'PASS' if diff_seed_differs else 'FAIL'}] gate (iii): different seed -> "
         f"different walk")
    if not same_seed_identical:
        failures.append("gate (iii): random_matched_actions not reproducible under a fixed seed")
    if not diff_seed_differs:
        failures.append("gate (iii): random_matched_actions identical across different seeds "
                        "(RNG not actually seeded per-call)")
    # same marginals: empirical distributions of a and c should each still track `marginals`
    emp_a = np.stack([np.bincount(a[:, i], minlength=5) / 500 for i in range(4)])
    emp_c = np.stack([np.bincount(c[:, i], minlength=5) / 500 for i in range(4)])
    close = (np.max(np.abs(emp_a - marginals)) < 0.15 and np.max(np.abs(emp_c - marginals)) < 0.15)
    print(f"    [{'PASS' if close else 'FAIL'}] gate (iii): both seeds draw from the SAME "
         f"marginals (empirical match within tolerance at n=500)")
    if not close:
        failures.append("gate (iii): seed-7 / seed-8 empirical distributions diverge from the "
                        "shared input marginals beyond sampling tolerance")


# =============================================================================
# gate (ii)/loader — fail-loud sidecar validation (load_action_marginals)
# =============================================================================

def gate_loader_fails_loud(failures):
    ev = trace_mod._load_eval_vector()
    work = tempfile.mkdtemp(prefix="t892_loader_")
    try:
        good = {"schema": ev.ACTION_MARGINALS_SCHEMA, "symbols": ["A", "B"],
                "marginals": [[0.2, 0.2, 0.2, 0.2, 0.2], [1, 0, 0, 0, 0]]}
        cases = {
            "wrong_schema": {**good, "schema": "not-a-real-schema"},
            "wrong_symbols": {**good, "symbols": ["B", "A"]},
            "bad_shape": {**good, "marginals": [[0.5, 0.5]]},
            "rows_dont_sum_to_1": {**good, "marginals": [[0.5, 0.5, 0.5, 0.5, 0.5],
                                                          [1, 0, 0, 0, 0]]},
            "negative_mass": {**good, "marginals": [[-0.2, 0.3, 0.3, 0.3, 0.3],
                                                     [1, 0, 0, 0, 0]]},
        }
        all_refused = True
        for name, doc in cases.items():
            p = os.path.join(work, f"{name}.json")
            with open(p, "w") as fh:
                json.dump(doc, fh)
            try:
                ev.load_action_marginals(p, ["A", "B"])
                all_refused = False
                failures.append(f"gate (loader): {name!r} sidecar was ACCEPTED (should refuse)")
            except RuntimeError:
                pass
        # the good doc must be ACCEPTED
        p = os.path.join(work, "good.json")
        with open(p, "w") as fh:
            json.dump(good, fh)
        accepted = True
        try:
            arr = ev.load_action_marginals(p, ["A", "B"])
            accepted = arr.shape == (2, 5)
        except RuntimeError as e:
            accepted = False
            failures.append(f"gate (loader): a VALID sidecar was refused: {e}")
        print(f"    [{'PASS' if all_refused else 'FAIL'}] gate (loader): every malformed "
             f"sidecar (wrong schema/symbols/shape/row-sum/negative mass) is refused")
        print(f"    [{'PASS' if accepted else 'FAIL'}] gate (loader): a valid sidecar is accepted")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    # missing path / rule-without-sidecar coherence, via the real CLI (gate v style)
    r = _run([EV_PY, "--smoke", "--operator", "policy-argmax", "--decision-rule",
             "random-matched"])
    ok = r.returncode != 0 and "--marginals-sidecar" in (r.stderr or "")
    print(f"    [{'PASS' if ok else 'FAIL'}] gate (loader): --decision-rule random-matched "
         f"without --marginals-sidecar refuses via the real CLI")
    if not ok:
        failures.append(f"gate (loader): missing-sidecar coherence check did not refuse as "
                        f"expected (exit {r.returncode}): {r.stderr[-500:]}")


# =============================================================================
# gate (v) — the null operator's DOCUMENTED CLI invocation is run for real, not stubbed
# =============================================================================

def gate_v_real_cli_not_stubbed(failures, work):
    subject_path = os.path.join(work, "trace_subject.json")
    r = _run([TRACE_PY, "--smoke", "--out", subject_path])
    ok = r.returncode == 0 and os.path.exists(subject_path)
    print(f"    [{'PASS' if ok else 'FAIL'}] gate (v): trace.py --smoke (subject, argmax) "
         f"real CLI run exits 0")
    if not ok:
        failures.append(f"gate (v): trace.py --smoke failed (exit {r.returncode}): "
                        f"{r.stderr[-2000:]}")
        return None

    sidecar_path = os.path.join(work, "sidecar.json")
    r = _run([BUILD_MARGINALS_PY, "--trace", subject_path, "--out", sidecar_path])
    ok = r.returncode == 0 and os.path.exists(sidecar_path)
    print(f"    [{'PASS' if ok else 'FAIL'}] gate (v): build_action_marginals.py real CLI run "
         f"exits 0")
    if not ok:
        failures.append(f"gate (v): build_action_marginals.py failed: {r.stderr[-2000:]}")
        return None

    # THE gate: eval_vector.py's OWN documented CLI, --decision-rule random-matched, run for
    # real (never eval_vector835.policy_decision_actions / random_matched_actions called
    # directly from this test to fake it).
    r = _run([EV_PY, "--smoke", "--operator", "policy-argmax", "--decision-rule",
             "random-matched", "--marginals-sidecar", sidecar_path,
             "--out-dir", os.path.join(work, "ev_null")])
    ok = r.returncode == 0
    print(f"    [{'PASS' if ok else 'FAIL'}] gate (v): eval_vector.py --smoke --decision-rule "
         f"random-matched --marginals-sidecar <sidecar> (the REAL documented CLI) exits 0")
    if not ok:
        failures.append(f"gate (v): eval_vector.py random-matched CLI failed (exit "
                        f"{r.returncode}): {r.stderr[-2000:]}")
        return None
    art = os.path.join(work, "ev_null", "results_fixture_smoke_policy-argmax.json")
    exists = os.path.exists(art)
    print(f"    [{'PASS' if exists else 'FAIL'}] gate (v): the real CLI wrote its scorecard "
         f"artifact")
    if not exists:
        failures.append(f"gate (v): expected artifact missing: {art}")

    null_path = os.path.join(work, "trace_null_seed0.json")
    r = _run([TRACE_PY, "--smoke", "--decision-rule", "random-matched", "--marginals-sidecar",
             sidecar_path, "--eval-seed", "0", "--out", null_path])
    ok = r.returncode == 0 and os.path.exists(null_path)
    print(f"    [{'PASS' if ok else 'FAIL'}] gate (v): trace.py --smoke --decision-rule "
         f"random-matched (null trace) real CLI run exits 0")
    if not ok:
        failures.append(f"gate (v): trace.py null run failed: {r.stderr[-2000:]}")
        return None
    return subject_path, sidecar_path, null_path


def gate_v_report_plumbing(failures, subject_path, null_path, work):
    if subject_path is None:
        failures.append("gate (v)/report: skipped — upstream CLI chain failed")
        return
    out_path = os.path.join(work, "report_smoke.json")
    r = _run([os.path.join(EXP_DIR, "report.py"), "--subject", subject_path, "--null",
             null_path, "--out", out_path])
    ok = r.returncode == 0 and os.path.exists(out_path)
    print(f"    [{'PASS' if ok else 'FAIL'}] gate (v): report.py real CLI run exits 0 and "
         f"writes a report")
    if not ok:
        failures.append(f"gate (v): report.py failed (exit {r.returncode}): "
                        f"{r.stderr[-2000:]}")
        return
    with open(out_path) as fh:
        rep = json.load(fh)
    required = {"band", "delta", "subject_activity", "null_activity",
               "instrument_fail_reasons", "per_window", "READ_THIS_FIRST",
               "pre_registration_deviations", "subject_book_width", "null_book_width",
               "warnings", "insufficient_n_windows", "min_n_for_bootstrap"}
    ok = required.issubset(rep.keys()) and rep["band"] in report_mod.RECOGNIZED_BANDS
    print(f"    [{'PASS' if ok else 'FAIL'}] gate (v): report schema carries the required keys "
         f"and a recognized band label ({rep.get('band')})")
    if not ok:
        failures.append(f"gate (v): report.py schema/band unexpected: {rep}")


# =============================================================================
# gate (iv) — the fixture smoke recovers the planted sign; a shuffled-trace falsification
# reads non-EDGE. Pure numpy synthetic construction (a controlled, deterministic "fixture
# walk" per SPEC.md scope 4) — no DB, no model, no subprocess; every function called is the
# real, unstubbed compute_forward_returns / report.compare.
# =============================================================================

def _make_synthetic_walk(n_decisions, n_assets, seed, foresight_matched=True):
    """Deterministic synthetic price path + a perfect-foresight 'policy' that buys an asset
    iff its price is about to rise over the next `horizon` rows and sells iff about to fall.
    Returns (prices, row_index, timestamps, chosen_vectors, realized_book)."""
    horizon = 5
    n_rows = n_decisions + horizon + 5
    rng = np.random.default_rng(seed)
    # A random walk with i.i.d. steps: the realized DIRECTION over the next `horizon` rows is
    # exactly what "foresight" reads off (no drift needed — this tests DIRECTION skill, not
    # beta).
    steps = rng.normal(0, 0.01, size=(n_rows, n_assets))
    log_prices = np.cumsum(steps, axis=0)
    prices = 100.0 * np.exp(log_prices)
    row_index = np.arange(0, n_decisions)
    chosen_vectors = np.full((n_decisions, n_assets), 1, dtype=np.int32)   # HOLD
    realized_book = np.zeros((n_decisions, n_assets), dtype=np.int32)     # flat
    for k in range(n_decisions):
        will_rise = prices[row_index[k] + horizon] > prices[row_index[k]]
        if foresight_matched:
            chosen_vectors[k] = np.where(will_rise, 2, 0)      # Buy / Sell — always non-HOLD
            realized_book[k] = np.where(will_rise, 1, 2)       # long / short
    timestamps = row_index * 60
    return prices, row_index, timestamps, chosen_vectors, realized_book, horizon


def _trace_doc_from_arrays(col_symbols, row_index, timestamps, chosen_vectors, realized_book,
                           prices, horizon, tag, decision_rule, eval_seed):
    """A synthetic (not-from-a-walk) trace doc. ``walk_kind`` is the smoke kind because there is
    no checkpoint here to pin — build_trace_doc REFUSES a checkpoint-kind doc with no
    mechanism-typed universe_pin record (finding 5). The realized allocation is the honest
    weight-semantics one for this synthetic book: every open slot holds exactly one weight unit
    (w_unit = 1/n_assets), so allocated = n_open/n_assets <= 1 by construction — never the dead
    position_size_pct product (BLOCKER 2)."""
    n_assets = len(col_symbols)
    book = np.asarray(realized_book)
    allocated = (book != 0).sum(axis=1) / float(n_assets)
    return trace_mod.build_trace_doc(
        tag=tag, checkpoint_label="synthetic", decision_rule=decision_rule,
        action_marginals_path=None, eval_seed=eval_seed, tier_bp=0, fee_bps=0.0, slip_bps=0.0,
        horizon=horizon, position_size_pct=0.1, col_symbols=col_symbols,
        row_index=row_index, timestamps=timestamps, chosen_vectors=chosen_vectors,
        realized_book=realized_book, prices=prices, universe_pin=None,
        walk_kind=trace_mod.WALK_KIND_SMOKE, allocated_fraction=allocated,
        allocation_semantics=trace_mod.WEIGHT_SEMANTICS, weight_unit=1.0 / n_assets,
        weight_max_mult=4.0)


def gate_iv_fixture_smoke_recovers_sign(failures):
    # n_decisions/n_assets/n_null_seeds chosen so the null's activity-rate/long-short-mix
    # sampling noise clears SPEC.md's 0.5pp instrument-fail bar reliably (pooling ~8x the
    # subject's own cell count keeps the mismatch well under 1 sigma of that bar).
    n_decisions, n_assets = 2000, 5
    col_symbols = [f"SYM{i}" for i in range(n_assets)]
    prices, row_index, ts, chosen, book, horizon = _make_synthetic_walk(
        n_decisions, n_assets, seed=3)
    subject_doc = _trace_doc_from_arrays(col_symbols, row_index, ts, chosen, book, prices,
                                         horizon, "synthetic-subject", "argmax", 0)

    # marginals derived from the subject's OWN attempt stream (matched by construction)
    marginals = bam_mod.marginals_from_chosen_vectors(chosen, n_assets)

    ev = trace_mod._load_eval_vector()
    work = tempfile.mkdtemp(prefix="t892_gate_iv_")
    try:
        subject_path = os.path.join(work, "subject.json")
        with open(subject_path, "w") as fh:
            json.dump(subject_doc, fh, default=float)

        null_paths = []
        for seed in range(10, 18):
            null_actions = ev.random_matched_actions(marginals, n_decisions, n_assets, seed)
            null_book = np.where(null_actions == 2, 1, np.where(null_actions == 0, 2, 0))
            null_doc = _trace_doc_from_arrays(col_symbols, row_index, ts, null_actions,
                                              null_book, prices, horizon,
                                              "synthetic-null", "random-matched", seed)
            p = os.path.join(work, f"null_{seed}.json")
            with open(p, "w") as fh:
                json.dump(null_doc, fh, default=float)
            null_paths.append(p)

        report = report_mod.compare(subject_path, null_paths, block=21, n_boot=2000, seed=0,
                                    ext_windows=None)
        ok = report["band"] == "EDGE"
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (iv): the planted-sign fixture walk "
             f"reads EDGE (got {report['band']}, delta={report['delta']['delta']}, "
             f"ci=[{report['delta']['ci_lo']}, {report['delta']['ci_hi']}])")
        if not ok:
            failures.append(f"gate (iv): planted-sign walk did not read EDGE: {report}")

        # Falsification: shuffle the subject's OWN action/book arrays across the decision axis
        # (prices/timestamps untouched) so the "foresight" no longer lines up with what
        # actually happened next.
        rng = np.random.default_rng(99)
        perm = rng.permutation(n_decisions)
        shuf_chosen = chosen[perm]
        shuf_book = book[perm]
        shuf_doc = _trace_doc_from_arrays(col_symbols, row_index, ts, shuf_chosen, shuf_book,
                                          prices, horizon, "synthetic-subject-shuffled",
                                          "argmax", 0)
        shuf_path = os.path.join(work, "subject_shuffled.json")
        with open(shuf_path, "w") as fh:
            json.dump(shuf_doc, fh, default=float)
        shuf_marginals = bam_mod.marginals_from_chosen_vectors(shuf_chosen, n_assets)
        shuf_null_paths = []
        for seed in (20, 21, 22):
            null_actions = ev.random_matched_actions(shuf_marginals, n_decisions, n_assets, seed)
            null_book = np.where(null_actions == 2, 1, np.where(null_actions == 0, 2, 0))
            null_doc = _trace_doc_from_arrays(col_symbols, row_index, ts, null_actions,
                                              null_book, prices, horizon,
                                              "synthetic-null-shuffled", "random-matched", seed)
            p = os.path.join(work, f"null_shuf_{seed}.json")
            with open(p, "w") as fh:
                json.dump(null_doc, fh, default=float)
            shuf_null_paths.append(p)
        shuf_report = report_mod.compare(shuf_path, shuf_null_paths, block=21, n_boot=2000,
                                         seed=0, ext_windows=None)
        ok2 = shuf_report["band"] != "EDGE"
        print(f"    [{'PASS' if ok2 else 'FAIL'}] gate (iv): the shuffled-targets falsification "
             f"reads non-EDGE (got {shuf_report['band']}, "
             f"delta={shuf_report['delta']['delta']})")
        if not ok2:
            failures.append(f"gate (iv): shuffled-targets falsification still read EDGE: "
                            f"{shuf_report}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


# =============================================================================
# gate (boot) — BLOCKER 1: an under-powered sample is a REPORTED CONDITION, never a band, and
# never a zero-width CI. Exercised BOTH directly and THROUGH ``ext_windows`` (the shipped suite
# passed ext_windows=None everywhere, so the SPEC's own predicted thin-n regime — per-regime
# buckets — was untested).
# =============================================================================

def _synthetic_pair_doc(work, name, n_decisions, n_assets, seed, ts_base=0, ts_step=60):
    """A tiny synthetic trace whose forward returns are well-defined at every non-HOLD cell."""
    rng = np.random.default_rng(seed)
    horizon = 5
    n_rows = n_decisions + horizon + 5
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(n_rows, n_assets)), axis=0))
    row_index = np.arange(n_decisions)
    chosen = np.full((n_decisions, n_assets), 2, dtype=np.int32)          # all Buy = all non-HOLD
    book = np.ones((n_decisions, n_assets), dtype=np.int32)               # all long
    ts = ts_base + row_index * ts_step
    doc = _trace_doc_from_arrays([f"S{i}" for i in range(n_assets)], row_index, ts, chosen, book,
                                 prices, horizon, name, "argmax", seed)
    p = os.path.join(work, f"{name}.json")
    with open(p, "w") as fh:
        json.dump(doc, fh, default=float)
    return p, ts


def gate_boot_thin_n_no_fake_ci(failures):
    # --- (a) the degenerate case, directly. Pre-fix this returned ci_lo == ci_hi == delta.
    rng = np.random.default_rng(11)
    s = rng.normal(0, 0.01, size=15)
    n = rng.normal(0, 0.01, size=15)
    ci = report_mod.block_bootstrap_delta_ci(s, n, block=21, n_boot=10000, seed=0)
    band = report_mod.band_for(ci)
    ok = (ci["ci_lo"] is None and ci["ci_hi"] is None and ci.get("insufficient_n") is True
          and band == report_mod.BAND_INSUFFICIENT_N and ci["n_boot"] == 0
          and ci["delta"] is not None)
    print(f"    [{'PASS' if ok else 'FAIL'}] gate (boot): n=15 vs 15 at block=21 reports "
         f"INSUFFICIENT-N with a NULL CI and a point-estimate delta "
         f"(band={band}, ci=[{ci['ci_lo']}, {ci['ci_hi']}], delta={ci['delta']})")
    if not ok:
        failures.append(f"gate (boot): thin sample did not refuse a CI/band: {ci} band={band}")

    # --- (b) the boundary: n == block is STILL degenerate (one wrapped block covers every index
    # exactly once => every resample is a rotation => a mean is rotation-invariant).
    for n_side in (21, 41):
        ci_b = report_mod.block_bootstrap_delta_ci(
            rng.normal(0, 0.01, size=n_side), rng.normal(0, 0.01, size=n_side),
            block=21, n_boot=500, seed=0)
        ok_b = ci_b["ci_lo"] is None and ci_b.get("insufficient_n") is True
        print(f"    [{'PASS' if ok_b else 'FAIL'}] gate (boot): n={n_side} (< "
             f"{report_mod.MIN_BOOTSTRAP_BLOCKS} full blocks of 21) still refuses a CI")
        if not ok_b:
            failures.append(f"gate (boot): n={n_side} emitted a CI: {ci_b}")
    # ... and the first n at/above the bar DOES get a real, non-degenerate CI.
    ci_ok = report_mod.block_bootstrap_delta_ci(
        rng.normal(0, 0.01, size=42), rng.normal(0, 0.01, size=42), block=21, n_boot=2000, seed=0)
    ok_c = (ci_ok["ci_lo"] is not None and ci_ok["ci_hi"] > ci_ok["ci_lo"]
            and ci_ok.get("insufficient_n") is False)
    print(f"    [{'PASS' if ok_c else 'FAIL'}] gate (boot): n=42 (= {report_mod.MIN_BOOTSTRAP_BLOCKS}"
         f" full blocks) yields a real, POSITIVE-WIDTH CI "
         f"([{ci_ok['ci_lo']:.6g}, {ci_ok['ci_hi']:.6g}])")
    if not ok_c:
        failures.append(f"gate (boot): n=42 did not produce a usable CI: {ci_ok}")
    # AMENDMENT 1 / D51 RE-CUT — SUPERSEDES cycle-3 finding 2's old assertion. Clearing the
    # DEGENERACY floor used to NEVER authorize a band on its own (a fixed CALIBRATED_BOOTSTRAP_
    # BLOCKS=30 bar sat between the floor and any band, and n=42/2-blocks always read LOW-POWER
    # regardless of what the calibration harness might have found). Amendment 1 replaces that
    # fixed bar with a MEASURED, per-block-count calibrated width correction
    # (disclosures.CALIBRATION_TABLE): if 2 blocks is itself in the table and ACHIEVED (it is,
    # measured k~3.49 with validated coverage ~0.94 -- see CALIBRATION_TABLE), a sample there is
    # now CORRECTLY bandable, calibration-widened. The gate now checks THAT calibration was
    # actually consulted and applied (calibration_achieved True, ci_lo_raw/ci_hi_raw present and
    # narrower than the final, widened ci_lo/ci_hi) rather than asserting the old blanket refusal.
    ok_d = (ci_ok.get("low_power") is False and ci_ok.get("calibration_achieved") is True
           and ci_ok.get("calibration_k") is not None and ci_ok["calibration_k"] > 1.0
           and (ci_ok["ci_hi"] - ci_ok["ci_lo"]) > (ci_ok["ci_hi_raw"] - ci_ok["ci_lo_raw"]))
    print(f"    [{'PASS' if ok_d else 'FAIL'}] gate (boot): ...and that n=42 (2 blocks) is now "
         f"CALIBRATED (Amendment 1) rather than unconditionally refused: k="
         f"{ci_ok.get('calibration_k')!r}, widened CI [{ci_ok['ci_lo']:.6g}, "
         f"{ci_ok['ci_hi']:.6g}] vs raw [{ci_ok['ci_lo_raw']:.6g}, {ci_ok['ci_hi_raw']:.6g}], "
         f"band={report_mod.band_for(ci_ok)}")
    if not ok_d:
        failures.append(f"gate (boot): n=42 calibration was not applied as Amendment 1 "
                        f"requires: {ci_ok}")

    # --- (c) THROUGH ext_windows: a deliberately THIN regime bucket must not carry a band.
    work = tempfile.mkdtemp(prefix="t892_boot_")
    try:
        # 200 decisions x 4 assets = 800 non-HOLD cells pooled (well past the bar), timestamps
        # 0, 60, ... 11940. The THIN window admits 3 decisions (12 cells); the WIDE one admits all.
        subj_p, ts = _synthetic_pair_doc(work, "boot_subject", 200, 4, seed=5)
        null_p, _ = _synthetic_pair_doc(work, "boot_null", 200, 4, seed=6)
        ext_windows = {
            "THIN_BUCKET": {"window": ["thin-start", "thin-end"],
                            "start_ts": int(ts[0]), "end_ts": int(ts[2])},
            "WIDE_BUCKET": {"window": ["wide-start", "wide-end"],
                            "start_ts": int(ts[0]), "end_ts": int(ts[-1])},
            "MISSING_BUCKET": None,
        }
        rep = report_mod.compare(subj_p, [null_p], block=21, n_boot=2000, seed=0,
                                 ext_windows=ext_windows)
        thin = rep["per_window"]["THIN_BUCKET"]
        wide = rep["per_window"]["WIDE_BUCKET"]
        ok_thin = (thin["band"] == report_mod.BAND_INSUFFICIENT_N
                   and thin["delta"]["ci_lo"] is None and thin["delta"]["ci_hi"] is None
                   and thin["excluded_from_banding"] is True
                   and "THIN_BUCKET" in rep["insufficient_n_windows"])
        print(f"    [{'PASS' if ok_thin else 'FAIL'}] gate (boot)/ext_windows: the THIN bucket "
             f"(n_subject={thin['n_subject']}) reads {thin['band']} with a NULL CI and is "
             f"excluded from banding")
        if not ok_thin:
            failures.append(f"gate (boot)/ext_windows: thin bucket emitted a band/CI: {thin}")
        ok_wide = (wide["band"] in ("EDGE", "ANTI-EDGE", "NULL")
                   and wide["delta"]["ci_hi"] > wide["delta"]["ci_lo"])
        print(f"    [{'PASS' if ok_wide else 'FAIL'}] gate (boot)/ext_windows: the WIDE bucket "
             f"(n_subject={wide['n_subject']}) still gets a real band ({wide['band']}) with a "
             f"positive-width CI")
        if not ok_wide:
            failures.append(f"gate (boot)/ext_windows: wide bucket lost its band/CI: {wide}")
        # NO bucket anywhere may carry a zero-width interval.
        degenerate = [k for k, v in rep["per_window"].items()
                      if isinstance(v.get("delta"), dict)
                      and v["delta"].get("ci_lo") is not None
                      and v["delta"]["ci_lo"] == v["delta"]["ci_hi"]]
        ok_z = not degenerate
        print(f"    [{'PASS' if ok_z else 'FAIL'}] gate (boot)/ext_windows: no bucket emits a "
             f"ZERO-WIDTH CI ({len(degenerate)} found)")
        if not ok_z:
            failures.append(f"gate (boot)/ext_windows: zero-width CI buckets: {degenerate}")
        # every bucket, including the missing-file one, carries the top-level disclaimer
        ok_note = all(v.get("band_note") == report_mod.BAND_NOTE
                      for v in rep["per_window"].values())
        print(f"    [{'PASS' if ok_note else 'FAIL'}] gate (boot)/ext_windows: every bucket "
             f"carries the band_note disclaimer")
        if not ok_note:
            failures.append("gate (boot)/ext_windows: a bucket carries no band_note")
    finally:
        shutil.rmtree(work, ignore_errors=True)


# =============================================================================
# gate (fail-window) — finding 6: when the instrument has FAILED, no per-regime bucket may
# publish a substantive band. Pre-fix a report whose top level read INSTRUMENT FAIL still
# emitted per_window["ALL"]["band"] == "EDGE".
# =============================================================================

def gate_instrument_fail_suppresses_windows(failures):
    work = tempfile.mkdtemp(prefix="t892_failwin_")
    try:
        # 60 decisions x 1 asset, subject all-long vs a null with a DELIBERATELY different
        # long/short mix (all short) => activity/mix instrument fail, with n past the boot bar.
        rng = np.random.default_rng(21)
        horizon, n_dec, n_assets = 5, 60, 1
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0.002, 0.01,
                                                     size=(n_dec + horizon + 5, n_assets)), axis=0))
        rows, ts = np.arange(n_dec), np.arange(n_dec) * 60
        subj = _trace_doc_from_arrays(
            ["S0"], rows, ts, np.full((n_dec, n_assets), 2, dtype=np.int32),
            np.ones((n_dec, n_assets), dtype=np.int32), prices, horizon, "fw-subject",
            "argmax", 0)
        null = _trace_doc_from_arrays(
            ["S0"], rows, ts, np.full((n_dec, n_assets), 0, dtype=np.int32),
            np.full((n_dec, n_assets), 2, dtype=np.int32), prices, horizon, "fw-null",
            "random-matched", 1)
        sp, np_ = os.path.join(work, "s.json"), os.path.join(work, "n.json")
        for p, d in ((sp, subj), (np_, null)):
            with open(p, "w") as fh:
                json.dump(d, fh, default=float)
        ext_windows = {"W": {"window": ["a", "b"], "start_ts": int(ts[0]),
                             "end_ts": int(ts[-1])}}
        rep = report_mod.compare(sp, [np_], block=21, n_boot=2000, seed=0,
                                 ext_windows=ext_windows)
        top_failed = rep["band"] == report_mod.BAND_INSTRUMENT_FAIL and rep[
            "instrument_fail_reasons"]
        buckets_clean = all(v["band"] == report_mod.BAND_INSTRUMENT_FAIL
                            and v.get("excluded_from_banding") is True
                            for v in rep["per_window"].values())
        ok = bool(top_failed) and buckets_clean
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (fail-window): a failed instrument "
             f"({rep['instrument_fail_reasons']}) suppresses EVERY per-window band "
             f"(ALL={rep['per_window']['ALL']['band']}, W={rep['per_window']['W']['band']})")
        if not ok:
            failures.append(f"gate (fail-window): per_window leaked a band past an instrument "
                            f"fail: top={rep['band']} per_window={rep['per_window']}")
        ok2 = rep["n_windows_banded"] == 0
        print(f"    [{'PASS' if ok2 else 'FAIL'}] gate (fail-window): n_windows_banded == 0 on a "
             f"failed instrument (got {rep['n_windows_banded']})")
        if not ok2:
            failures.append(f"gate (fail-window): n_windows_banded={rep['n_windows_banded']}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


# =============================================================================
# gate (budget) — BLOCKER 2: the budget check is computed from the REALIZED allocation and is
# MODE-AWARE. Includes the real CLI run under --vector-weight-rebalance (SPEC.md's mandated mode,
# which the shipped suite never exercised at all).
# =============================================================================

def gate_budget_mode_aware(failures, work):
    # (a) SPEC.md's own book shape at real-leg scale: 64 assets, ~57 open slots. The OLD form
    # (open slots x position_size_pct, 57 x 0.1 = 5.7 >> 1.0) flagged EVERY decision; the
    # realized-allocation form sees a legal book and flags none. This is the red-without-the-fix
    # assertion at the scale the defect actually bites.
    n_dec, n_assets, n_open = 30, 64, 57
    book = np.zeros((n_dec, n_assets), dtype=np.int64)
    book[:, :n_open] = 1
    legacy_n, legacy_alloc = trace_mod.budget_conservation_violations(
        book, allocation_semantics=trace_mod.PCT_SEMANTICS, position_size_pct=0.1)
    realized = np.full(n_dec, n_open / n_assets)          # one weight unit per open slot
    real_n, real_alloc = trace_mod.budget_conservation_violations(
        book, allocated_fraction=realized)
    ok = legacy_n == n_dec and real_n == 0
    print(f"    [{'PASS' if ok else 'FAIL'}] gate (budget): on SPEC.md's ~{n_open}-slot book the "
         f"DEAD position_size_pct form reports {legacy_n}/{n_dec} violations "
         f"(alloc={legacy_alloc[0]:.2f}) while the REALIZED-allocation form reports {real_n} "
         f"(alloc={real_alloc[0]:.4f})")
    if not ok:
        failures.append(f"gate (budget): expected legacy={n_dec} realized=0, got "
                        f"{legacy_n}/{real_n}")

    # (b) mode-awareness: under weight semantics the dead field may not be used AT ALL.
    refused = False
    try:
        trace_mod.budget_conservation_violations(
            book, allocation_semantics=trace_mod.WEIGHT_SEMANTICS, position_size_pct=0.1)
    except ValueError:
        refused = True
    print(f"    [{'PASS' if refused else 'FAIL'}] gate (budget): weight semantics with no "
         f"realized allocation REFUSES to fall back to position_size_pct")
    if not refused:
        failures.append("gate (budget): weight semantics silently used position_size_pct")

    # (c) THE real CLI run in SPEC.md's mandated mode. A healthy walk must report ZERO budget
    # violations, and the artifact must say which semantics were in force.
    out = os.path.join(work, "trace_weight_mode.json")
    r = _run([TRACE_PY, "--smoke", "--vector-weight-rebalance", "--out", out,
              "--smoke-ckpt-dir", os.path.join(work, "wk_ckpt")])
    ok_run = r.returncode == 0 and os.path.exists(out)
    print(f"    [{'PASS' if ok_run else 'FAIL'}] gate (budget): trace.py --smoke "
         f"--vector-weight-rebalance (SPEC.md's mandated mode) real CLI run exits 0")
    if not ok_run:
        failures.append(f"gate (budget): weight-mode CLI run failed (exit {r.returncode}): "
                        f"{r.stderr[-2000:]}")
        return
    with open(out) as fh:
        doc = json.load(fh)
    prov = doc["provenance"]
    alloc = np.asarray(doc["decisions"]["allocated_fraction"], dtype=np.float64)
    ok2 = (prov["budget_conservation_violations"] == 0
           and prov["allocation_semantics"] == trace_mod.WEIGHT_SEMANTICS
           and prov["position_size_pct_is_dead_code"] is True
           and prov["vector_weight_unit_effective"] is not None
           and float(alloc.max()) <= 1.0 + 1e-6)
    print(f"    [{'PASS' if ok2 else 'FAIL'}] gate (budget): the weight-mode walk reports ZERO "
         f"budget violations (semantics={prov['allocation_semantics']}, "
         f"max_allocated={float(alloc.max()):.6f}, "
         f"dead_field_flagged={prov['position_size_pct_is_dead_code']})")
    if not ok2:
        failures.append(f"gate (budget): weight-mode walk unhealthy/mislabelled: "
                        f"{prov['budget_conservation_violations']} violations, "
                        f"semantics={prov['allocation_semantics']}, max_alloc={alloc.max()}")
    # finding 7: --smoke-ckpt-dir was HONOURED (it used to be inherited into --help and then
    # ignored, hardcoding EXP_DIR/smoke_ckpt).
    ok3 = os.path.isdir(os.path.join(work, "wk_ckpt"))
    print(f"    [{'PASS' if ok3 else 'FAIL'}] gate (budget)/scratch: --smoke-ckpt-dir was "
         f"honoured (the throwaway checkpoint landed in the given directory)")
    if not ok3:
        failures.append("gate (budget)/scratch: --smoke-ckpt-dir was ignored")


# =============================================================================
# gate (pin) — finding 5: a real --checkpoint walk EMBEDS 886's mechanism-typed record; a
# fresh-init smoke walk says truthfully that it has none.
# =============================================================================

def gate_universe_pin_record(failures, work):
    ckpt = os.path.join(work, "pin_ckpt")
    trace_smoke = os.path.join(work, "trace_pin_smoke.json")
    r = _run([TRACE_PY, "--smoke", "--smoke-ckpt-dir", ckpt, "--out", trace_smoke])
    if r.returncode != 0 or not os.path.exists(trace_smoke):
        failures.append(f"gate (pin): the smoke run that produces the test checkpoint failed: "
                        f"{r.stderr[-2000:]}")
        return
    with open(trace_smoke) as fh:
        smoke_doc = json.load(fh)
    sprov = smoke_doc["provenance"]
    ok = (sprov["universe_pin"] is None
          and sprov["walk_kind"] == trace_mod.WALK_KIND_SMOKE
          and isinstance(sprov["universe_pin_note"], str)
          and "smoke" in sprov["universe_pin_note"])
    print(f"    [{'PASS' if ok else 'FAIL'}] gate (pin): the fresh-init smoke walk carries "
         f"universe_pin=null and a note that TRUTHFULLY describes a smoke walk")
    if not ok:
        failures.append(f"gate (pin): smoke walk pin/note wrong: {sprov['universe_pin']} / "
                        f"{sprov['universe_pin_note']}")

    # THE gate: a genuine --checkpoint walk (not --smoke) over the same fixture universe.
    ev = trace_mod._load_eval_vector()
    trace_ck = os.path.join(work, "trace_pin_ckpt.json")
    r = _run([TRACE_PY, "--checkpoint", ckpt, "--db", FIXTURE_DB,
              "--symbols", ",".join(ev.SMOKE_SYMBOLS), "--n-assets", str(len(ev.SMOKE_SYMBOLS)),
              "--start", ev.SMOKE_START, "--end", ev.SMOKE_END,
              "--bar-minutes", str(ev.SMOKE_BAR_MINUTES),
              "--max-decisions", str(ev.SMOKE_MAX_DECISIONS),
              "--vector-weight-rebalance", "--tag", "pin-ckpt-walk", "--out", trace_ck])
    ok_run = r.returncode == 0 and os.path.exists(trace_ck)
    print(f"    [{'PASS' if ok_run else 'FAIL'}] gate (pin): a real --checkpoint walk "
         f"(non-smoke, documented CLI) exits 0")
    if not ok_run:
        failures.append(f"gate (pin): real --checkpoint walk failed (exit {r.returncode}): "
                        f"{r.stderr[-2500:]}")
        return
    with open(trace_ck) as fh:
        prov = json.load(fh)["provenance"]
    pin = prov["universe_pin"]
    ok2 = (isinstance(pin, dict)
           and pin.get("mechanism") in ev.CERTIFIABLE_MECHANISMS
           and "verified" in pin and "verified_reason" in pin
           and pin.get("n_slots") == len(ev.SMOKE_SYMBOLS)
           and pin.get("realized_matches_requested") is True
           and prov["walk_kind"] == trace_mod.WALK_KIND_CHECKPOINT
           and prov["universe_pin_note"] is None)
    print(f"    [{'PASS' if ok2 else 'FAIL'}] gate (pin): the --checkpoint walk EMBEDS the 886 "
         f"mechanism-typed record inline (mechanism={pin and pin.get('mechanism')}, "
         f"verified={pin and pin.get('verified')}, "
         f"reason={pin and pin.get('verified_reason')}) and no smoke note")
    if not ok2:
        failures.append(f"gate (pin): --checkpoint walk did not embed a mechanism-typed record: "
                        f"{pin}")
    # 886's HONEST split is preserved, not overwritten: this checkpoint carries no provenance of
    # its own, so verified/certified MUST still be False with the reason spelled out.
    ok3 = (pin.get("verified") is False and pin.get("certified") is False
           and pin.get("verified_reason"))
    print(f"    [{'PASS' if ok3 else 'FAIL'}] gate (pin): 886's PROVEN/NOT-PROVEN split is "
         f"inherited verbatim — a provenance-less checkpoint stays verified=False with a reason")
    if not ok3:
        failures.append(f"gate (pin): 886's verified/certified split was not preserved: {pin}")
    # and build_trace_doc REFUSES to emit a checkpoint-kind trace with no record at all
    refused = False
    try:
        trace_mod.build_trace_doc(
            tag="t", checkpoint_label="c", decision_rule="argmax", action_marginals_path=None,
            eval_seed=0, tier_bp=0, fee_bps=0.0, slip_bps=0.0, horizon=2, position_size_pct=0.1,
            col_symbols=["A"], row_index=[0], timestamps=[0],
            chosen_vectors=np.array([[2]]), realized_book=np.array([[1]]),
            prices=np.array([[100.0], [101.0], [102.0], [103.0]]), universe_pin=None,
            walk_kind=trace_mod.WALK_KIND_CHECKPOINT, allocated_fraction=np.array([0.5]))
    except ValueError:
        refused = True
    print(f"    [{'PASS' if refused else 'FAIL'}] gate (pin): build_trace_doc REFUSES a "
         f"--checkpoint-kind trace with universe_pin=null (never a false note)")
    if not refused:
        failures.append("gate (pin): a checkpoint-kind trace with no pin record was accepted")


# =============================================================================
# gate (disclose) — MAJOR 3 + the pre-registration deviation: SPEC.md:46's third matched moment
# is computed and reported, and the deviation notice is inline in BOTH artifact kinds.
# =============================================================================

def gate_disclosure_and_book_width(failures, subject_path, null_path, work):
    if subject_path is None:
        failures.append("gate (disclose): skipped — upstream CLI chain failed")
        return
    with open(subject_path) as fh:
        tr = json.load(fh)
    ok = (tr.get("READ_THIS_FIRST") == disclosures_mod.READ_THIS_FIRST
          and tr.get("pre_registration_deviations")
          and any(d["id"] == "PRD-1" and "counterfactual" in d["delivered"]
                  and "OPTIMISTIC" in d["bias_direction"] and "EXACTLY" in d["cost0_impact"]
                  for d in tr["pre_registration_deviations"]))
    print(f"    [{'PASS' if ok else 'FAIL'}] gate (disclose): every TRACE carries the "
         f"pre-registration deviation inline (delivered estimator, cost0 neutrality, bias "
         f"direction)")
    if not ok:
        failures.append("gate (disclose): a trace does not disclose the pre-registration "
                        "deviation")

    out_path = os.path.join(work, "report_disclose.json")
    r = _run([os.path.join(EXP_DIR, "report.py"), "--subject", subject_path, "--null",
              null_path, "--out", out_path, "--n-boot", "500"])
    if r.returncode != 0:
        failures.append(f"gate (disclose): report.py failed: {r.stderr[-2000:]}")
        return
    with open(out_path) as fh:
        rep = json.load(fh)
    ok2 = (rep.get("READ_THIS_FIRST") == disclosures_mod.READ_THIS_FIRST
           and rep.get("pre_registration_deviations")
           and rep.get("bootstrap_power_disclosure") == disclosures_mod
           .BOOTSTRAP_POWER_DISCLOSURE
           and rep.get("book_width_disclosure") == disclosures_mod.BOOK_WIDTH_DISCLOSURE)
    print(f"    [{'PASS' if ok2 else 'FAIL'}] gate (disclose): every REPORT carries the same "
         f"deviation notice plus the bootstrap-power and book-width disclosures")
    if not ok2:
        failures.append("gate (disclose): a report is missing an inline disclosure block")
    # SPEC.md:46's third matched moment, promised and previously never computed.
    sw, nw = rep.get("subject_book_width"), rep.get("null_book_width")
    ok3 = (isinstance(sw, dict) and isinstance(nw, dict)
           and sw["mean"] is not None and nw["mean"] is not None
           and sw["std"] is not None and nw["std"] is not None
           and rep["book_width_mismatch_slots"] is not None
           and rep["book_width_mismatch_frac_of_universe"] is not None
           and "warnings" in rep)
    print(f"    [{'PASS' if ok3 else 'FAIL'}] gate (disclose): mean AND spread of realized book "
         f"width reported for both sides (subject {sw and sw['mean']} +/- {sw and sw['std']}, "
         f"null {nw and nw['mean']} +/- {nw and nw['std']}, "
         f"mismatch {rep.get('book_width_mismatch_slots')} slots)")
    if not ok3:
        failures.append(f"gate (disclose): book width not computed/reported: {sw} / {nw}")
    # a book-width mismatch is a WARNING, never one of SPEC.md's frozen instrument-fail reasons
    ok4 = not any("book-width" in reason for reason in rep["instrument_fail_reasons"])
    print(f"    [{'PASS' if ok4 else 'FAIL'}] gate (disclose): a book-width mismatch is never "
         f"folded into SPEC.md's frozen instrument_fail list")
    if not ok4:
        failures.append("gate (disclose): book-width mismatch was added to instrument_fail")


# =============================================================================
# gate (abbrev) — finding 8: NO new flag may steal a long-option abbreviation that was
# unambiguous at HEAD. `--action-marginals` (the original spelling of --marginals-sidecar) made
# `--a` ambiguous with HEAD's only `--a*` option, `--allow-unpinned-universe`. This is the
# CLASS-CLOSING test: it does not just pin `--a`, it enumerates EVERY HEAD long option, computes
# its minimal unique abbreviation AS OF HEAD, and asserts each still resolves to the same option.
# Pattern mirrored from tools/tests/test_ckpt_881i_manifest_ingestion.py's
# test_head_option_abbreviations_still_resolve (TASK_881I closed the identical class on
# tools/ckpt_gc.py) — two layers, set-theoretic + behavioural, because either alone is weak.
# =============================================================================

# FROZEN SNAPSHOT of eval_vector.py's long options at commit 63dad5c (the HEAD this task branched
# from), with each one's kind. Hardcoded deliberately: deriving it from the CURRENT HEAD would be
# self-defeating the moment this change ships.
HEAD_LONG_OPTIONS = {
    "--allow-unpinned-universe": "flag", "--bar-minutes": "value", "--checkpoint": "value",
    "--checkpoint-list": "value", "--collect-chosen-vectors": "value", "--db": "value",
    "--decision-m": "value", "--decision-rule": "value", "--decision-tau": "value",
    "--discount": "value", "--end": "value", "--equiv-close-rtol": "value",
    "--equivalence-gate": "flag", "--eval-seed": "value", "--fill-prob": "value",
    "--gumbel-scale": "value", "--h-cycles": "value", "--l-cycles": "value",
    "--leaf-eval": "value", "--leaf-rollout-r": "value", "--max-decisions": "value",
    "--max-positions": "value", "--min-cash-threshold": "value", "--model-size": "value",
    "--n-assets": "value", "--no-leaf-vhat-tail": "value", "--operator": "value",
    "--out-dir": "value", "--panel-frame-from-checkpoint": "value",
    "--panel-frame-source": "value", "--panel-source": "value", "--periods-file": "value",
    "--position-size-pct": "value", "--prior-weight": "value",
    "--require-active-slots": "value", "--sims": "value", "--smoke": "flag",
    "--smoke-ckpt-dir": "value", "--start": "value", "--symbols": "value", "--tag": "value",
    "--teacher-floor": "flag", "--value-scale": "value", "--value-shift": "value",
    "--vector-budget-pct": "value", "--vector-budget-step": "flag",
    "--vector-dither-scale": "value", "--vector-hold-bias": "value", "--vector-k": "value",
    "--vector-per-node-resample": "flag", "--vector-proposal-explore-slots": "value",
    "--vector-proposal-scheme": "value", "--vector-proposal-temp": "value",
    "--vector-rebalance-deadband": "value", "--vector-weight-max": "value",
    "--vector-weight-rebalance": "flag", "--vector-weight-unit": "value",
}


def _unambiguous_prefixes(options):
    """{prefix: option} for every "--" prefix (>= 1 character after the dashes, up to and
    including the full spelling) that matches EXACTLY one option in `options` — argparse's own
    abbreviation rule."""
    out = {}
    for o in options:
        for k in range(3, len(o) + 1):
            pref = o[:k]
            matches = [x for x in options if x.startswith(pref)]
            if len(matches) == 1:
                out[pref] = matches[0]
    return out


def gate_abbrev_head_options_still_resolve(failures):
    import contextlib
    import io
    ev = trace_mod._load_eval_vector()
    ap = ev.build_argparser()
    # `--help` is part of the option set argparse resolves against (it exists at HEAD too), so it
    # must be in BOTH sets or `--h` would look uniquely abbreviated here while the real parser
    # calls it ambiguous. Short flags (`-h`) play no part in long-option abbreviation.
    candidate = {o for o in ap._option_string_actions if o.startswith("--")}
    head_options = set(HEAD_LONG_OPTIONS) | {"--help"}

    missing = set(HEAD_LONG_OPTIONS) - candidate
    ok0 = not missing
    print(f"    [{'PASS' if ok0 else 'FAIL'}] gate (abbrev): every one of the "
         f"{len(HEAD_LONG_OPTIONS)} HEAD long options still exists ({len(candidate)} total now)")
    if not ok0:
        failures.append(f"gate (abbrev): HEAD options disappeared: {sorted(missing)}")
        return

    # --- layer 1: set-theoretic, exhaustive.
    head_pref = _unambiguous_prefixes(head_options)
    cand_pref = _unambiguous_prefixes(candidate)
    stolen = {p: (o, cand_pref.get(p)) for p, o in head_pref.items() if cand_pref.get(p) != o}
    ok1 = not stolen
    print(f"    [{'PASS' if ok1 else 'FAIL'}] gate (abbrev): all {len(head_pref)} abbreviations "
         f"that were unambiguous at HEAD still resolve to the SAME option "
         f"({len(stolen)} stolen)")
    if not ok1:
        failures.append(f"gate (abbrev): abbreviations stolen (None = now ambiguous): {stolen}")

    # --- layer 2: behavioural, against the REAL parser (not our model of argparse). Each HEAD
    # option's MINIMAL abbreviation is actually parsed; argparse's own resolution decides.
    minimal = {}
    for pref, o in head_pref.items():
        if o not in minimal or len(pref) < len(minimal[o]):
            minimal[o] = pref
    bad = []
    for o in sorted(HEAD_LONG_OPTIONS):
        # An option that is itself a strict prefix of another (--checkpoint vs
        # --checkpoint-list, --smoke vs --smoke-ckpt-dir, ...) has NO unambiguous abbreviation;
        # argparse resolves it by its EXACT-match rule instead. Probe the full spelling for those
        # — there was no abbreviation to steal, but the exact spelling must still resolve.
        pref = minimal.get(o, o)
        act = ap._option_string_actions[o]
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err):
                ns, _rest = ap.parse_known_args([f"{pref}=Zz"])
            # accepted: the sentinel must have landed in THIS option's dest
            got = getattr(ns, act.dest, None)
            # action="append" options (e.g. --checkpoint) land the sentinel inside a list.
            if got != "Zz" and got != ["Zz"]:
                bad.append(f"{pref} parsed but did not populate {o} (dest {act.dest} = {got!r})")
        except SystemExit:
            msg = err.getvalue()
            if "ambiguous option" in msg:
                bad.append(f"{pref} became AMBIGUOUS: {msg.strip().splitlines()[-1]}")
            elif f"argument {o}" not in msg:
                bad.append(f"{pref} no longer resolves to {o}: {msg.strip().splitlines()[-1]}")
    ok2 = not bad
    print(f"    [{'PASS' if ok2 else 'FAIL'}] gate (abbrev): the real parser resolves every HEAD "
         f"option's minimal abbreviation to that same option ({len(bad)} failures)")
    if not ok2:
        failures.append(f"gate (abbrev): behavioural failures: {bad[:8]}")

    # --- the reported instance, pinned by name through the REAL CLI (subprocess), so the
    # in-process model above is validated against the shipped command line.
    r = _run([EV_PY, "--a"])
    ok3 = "ambiguous option" not in (r.stderr or "")
    print(f"    [{'PASS' if ok3 else 'FAIL'}] gate (abbrev): `--a` is still unambiguous on the "
         f"REAL CLI (resolves to --allow-unpinned-universe)")
    if not ok3:
        failures.append(f"gate (abbrev): --a is ambiguous again: {r.stderr[-300:]}")
    r = _run([EV_PY, "--smoke", "--action-marginals", "x.json"])
    ok4 = r.returncode != 0 and "unrecognized arguments" in (r.stderr or "")
    print(f"    [{'PASS' if ok4 else 'FAIL'}] gate (abbrev): the colliding original spelling "
         f"`--action-marginals` is GONE (unrecognized), not merely aliased")
    if not ok4:
        failures.append(f"gate (abbrev): --action-marginals still accepted: exit "
                        f"{r.returncode} {r.stderr[-300:]}")


# =============================================================================
# gate (op-coherence) — finding 9: --decision-rule random-matched / --marginals-sidecar must
# REFUSE under an operator that cannot honour them. Pre-fix, the whole dispatch (including
# load_action_marginals' fail-loud validation) sat inside `if args.operator == "policy-argmax":`,
# so a forgotten `--operator policy-argmax` silently produced a complete-looking ORDINARY eval
# and never opened the sidecar at all — even a NONEXISTENT one.
# =============================================================================

def gate_op_coherence_refuses(failures):
    r = _run([EV_PY, "--smoke", "--decision-rule", "random-matched",
             "--marginals-sidecar", "/nonexistent/does_not_exist.json"])
    ok = (r.returncode != 0 and "policy-argmax" in (r.stderr or "")
          and "silently ignored" in (r.stderr or ""))
    print(f"    [{'PASS' if ok else 'FAIL'}] gate (op-coherence): random-matched + a NONEXISTENT "
         f"sidecar under the default `search` operator REFUSES (exit {r.returncode}) instead of "
         f"writing a complete-looking ordinary eval")
    if not ok:
        failures.append(f"gate (op-coherence): not refused (exit {r.returncode}): "
                        f"{(r.stdout + r.stderr)[-800:]}")
    # and the sidecar flag alone is refused under the wrong operator too
    r = _run([EV_PY, "--smoke", "--decision-rule", "argmax",
             "--marginals-sidecar", "/nonexistent/does_not_exist.json"])
    ok2 = r.returncode != 0
    print(f"    [{'PASS' if ok2 else 'FAIL'}] gate (op-coherence): --marginals-sidecar without "
         f"random-matched is still refused (exit {r.returncode})")
    if not ok2:
        failures.append("gate (op-coherence): stray --marginals-sidecar accepted")
    # the CORRECT invocation still runs (the guard refuses misuse, not use) — proven by gate (v),
    # which runs `--operator policy-argmax --decision-rule random-matched` end-to-end.


# =============================================================================
# CYCLE-3 gate (schema) — BLOCKER 1: report.py hard-reads its inputs. An input trace with no
# declared schema and no budget field is a REFUSAL, never "zero violations".
# =============================================================================

def _minimal_trace_doc(n_dec, n_assets, fwd_level, *, schema=trace_mod.TRACE_SCHEMA,
                       with_budget_field=True, with_cost_tier_fields=True, seed=0):
    """A hand-built doc carrying exactly what report.compare reads — deliberately NOT built by
    build_trace_doc, because the point is to feed report.py a document a DIFFERENT producer might
    have written (that is the only route to the BLOCKER-1 defect)."""
    rng = np.random.default_rng(seed)
    fwd = (fwd_level + rng.normal(0, 0.001, size=(n_dec, n_assets))).tolist()
    # [A1] market_return: an independent (unrelated-magnitude) undirected series, same shape/
    # mask as forward_return — this helper is deliberately NOT build_trace_doc, so it fabricates
    # its own value rather than re-deriving one.
    mkt = (0.5 * fwd_level + rng.normal(0, 0.001, size=(n_dec, n_assets))).tolist()
    # position_delta: first decision opens from flat (delta=+1 on a constant-long book), every
    # later decision is a side-preserving re-trim (delta=0) — book side never changes here.
    pos_delta = np.zeros((n_dec, n_assets), dtype=float)
    if n_dec:
        pos_delta[0, :] = 1.0
    prov = {"col_symbols": [f"S{i}" for i in range(n_assets)], "n_assets": n_assets,
            "n_decisions": n_dec}
    if with_budget_field:
        prov["budget_conservation_violations"] = 0
    # FIX PASS (MAJOR 1): the cost triple is hard-required on the trace-read path too — a
    # document that omits it is refused, never read as cost0. Omittable here for exactly that
    # gate, present by default (cost0) so every other gate keeps exercising a gradeable doc.
    if with_cost_tier_fields:
        prov.update({"tier_bp": 0, "fee_bps": 0.0, "slip_bps": 0.0})
    doc = {
        "provenance": prov,
        "decisions": {
            "row_index": list(range(n_dec)),
            "timestamp": [k * 60 for k in range(n_dec)],
            "chosen_vectors": np.full((n_dec, n_assets), 2, dtype=int).tolist(),
            "realized_book": np.ones((n_dec, n_assets), dtype=int).tolist(),
            "non_hold": np.ones((n_dec, n_assets), dtype=bool).tolist(),
            "forward_return": fwd,
            "market_return": mkt,
            "net_exposure": [1.0] * n_dec,
            "position_delta": pos_delta.tolist(),
            "allocated_fraction": [0.5] * n_dec,
        },
    }
    if schema is not None:
        doc["schema"] = schema
    return doc


def gate_report_hard_reads_its_inputs(failures):
    work = tempfile.mkdtemp(prefix="t892_schema_")
    try:
        def _write(name, doc):
            p = os.path.join(work, f"{name}.json")
            with open(p, "w") as fh:
                json.dump(doc, fh)
            return p

        # (a) THE reproduction: no schema key, no budget field. Pre-fix this produced a COMPLETE
        # EDGE verdict with violations=0 — a trace that never asserted its budget was checked read
        # as clean.
        s = _write("no_schema_subject", _minimal_trace_doc(200, 4, 0.02, schema=None,
                                                           with_budget_field=False, seed=1))
        n = _write("no_schema_null", _minimal_trace_doc(200, 4, 0.0, schema=None,
                                                        with_budget_field=False, seed=2))
        refused = None
        try:
            rep = report_mod.compare(s, [n], block=21, n_boot=500, seed=0)
        except RuntimeError as e:
            refused = str(e)
        ok = refused is not None and "schema" in refused
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (schema): a trace with NO schema key and NO "
             f"budget field is REFUSED (not banded)")
        if not ok:
            failures.append(f"gate (schema): an unvalidated trace was graded: "
                            f"{rep if refused is None else refused}")

        # (b) the schema check is exact — a plausible-but-wrong schema is refused too.
        s2 = _write("wrong_schema", _minimal_trace_doc(200, 4, 0.02,
                                                       schema="892/action-ev-trace/v1", seed=3))
        n2 = _write("ok_null", _minimal_trace_doc(200, 4, 0.0, seed=4))
        refused2 = False
        try:
            report_mod.compare(s2, [n2], block=21, n_boot=500, seed=0)
        except RuntimeError:
            refused2 = True
        print(f"    [{'PASS' if refused2 else 'FAIL'}] gate (schema): a WRONG (older) trace "
             f"schema is refused, not coerced")
        if not refused2:
            failures.append("gate (schema): a wrong trace schema was accepted")

        # (c) the budget field specifically: correct schema, field missing => refusal naming it.
        # This is the one that used to be a `.get(..., 0)`.
        s3 = _write("no_budget_subject", _minimal_trace_doc(200, 4, 0.02, with_budget_field=False,
                                                            seed=5))
        n3 = _write("no_budget_null", _minimal_trace_doc(200, 4, 0.0, seed=6))
        msg = ""
        try:
            report_mod.compare(s3, [n3], block=21, n_boot=500, seed=0)
        except RuntimeError as e:
            msg = str(e)
        ok3 = "budget_conservation_violations" in msg
        print(f"    [{'PASS' if ok3 else 'FAIL'}] gate (schema): an ABSENT "
             f"provenance.budget_conservation_violations is a refusal that NAMES the field "
             f"(absence is not zero violations)")
        if not ok3:
            failures.append(f"gate (schema): missing budget field not refused by name: {msg!r}")

        # (d) the same absence on a NULL trace is refused too (both sides are hard-read).
        s4 = _write("ok_subject2", _minimal_trace_doc(200, 4, 0.02, seed=7))
        n4 = _write("null_no_budget", _minimal_trace_doc(200, 4, 0.0, with_budget_field=False,
                                                          seed=8))
        refused4 = False
        try:
            report_mod.compare(s4, [n4], block=21, n_boot=500, seed=0)
        except RuntimeError:
            refused4 = True
        print(f"    [{'PASS' if refused4 else 'FAIL'}] gate (schema): the NULL side is hard-read "
             f"too (a null trace missing the budget field is refused)")
        if not refused4:
            failures.append("gate (schema): null-side budget field absence accepted")

        # (d2) FIX PASS (MAJOR 1) — CLASS-CLOSING over the whole hard-read provenance list, not
        # just the budget field: for EVERY key in report.REQUIRED_TRACE_PROVENANCE, omitting that
        # one key from an otherwise-valid schema-v3 subject must be a refusal that NAMES it. This
        # is the check that would have caught `tier_bp` being added as a soft `.get(..., 0)` while
        # its sibling was hard-subscripted two lines away, and it catches the next such addition.
        # the list itself is PINNED first — a loop over REQUIRED_TRACE_PROVENANCE alone could be
        # satisfied by shrinking the list, which is exactly how `tier_bp` came to be read softly.
        must_be_required = {"budget_conservation_violations", "tier_bp", "fee_bps", "slip_bps"}
        missing_req = sorted(must_be_required - set(report_mod.REQUIRED_TRACE_PROVENANCE))
        ok_req = not missing_req
        print(f"    [{'PASS' if ok_req else 'FAIL'}] gate (schema): every field a REFUSAL depends "
             f"on is in REQUIRED_TRACE_PROVENANCE (budget + the whole cost triple)")
        if not ok_req:
            failures.append(f"gate (schema): {missing_req} not in REQUIRED_TRACE_PROVENANCE — a "
                            f"refusal-gating field read with a soft default")
        for key in report_mod.REQUIRED_TRACE_PROVENANCE:
            doc_missing = _minimal_trace_doc(200, 4, 0.02, seed=90)
            del doc_missing["provenance"][key]
            sp = _write(f"missing_prov_{key}", doc_missing)
            nn = _write(f"ok_null_for_{key}", _minimal_trace_doc(200, 4, 0.0, seed=91))
            msg_k = ""
            try:
                report_mod.compare(sp, [nn], block=21, n_boot=300, seed=0)
            except RuntimeError as e:
                msg_k = str(e)
            ok_k = key in msg_k
            print(f"    [{'PASS' if ok_k else 'FAIL'}] gate (schema): an ABSENT "
                 f"provenance.{key} is a refusal that NAMES the field (every hard-read "
                 f"provenance field, not just the budget one)")
            if not ok_k:
                failures.append(f"gate (schema): missing provenance.{key} was not refused by "
                                f"name: {msg_k!r}")

        # (e) and a fully valid pair still grades (the guard refuses malformed input, not input).
        graded = False
        try:
            rep_ok = report_mod.compare(s4, [n3], block=21, n_boot=500, seed=0)
            graded = rep_ok["band"] in report_mod.RECOGNIZED_BANDS
        except RuntimeError as e:
            failures.append(f"gate (schema): a VALID pair was refused: {e}")
        print(f"    [{'PASS' if graded else 'FAIL'}] gate (schema): a well-formed pair still "
             f"grades normally")
    finally:
        shutil.rmtree(work, ignore_errors=True)


# =============================================================================
# CYCLE-3 gate (power) — finding 2: the banding bar is CALIBRATED, and the calibration is
# re-measured here (cheaply) rather than taken on faith from the constant.
# =============================================================================

def _measure_null_coverage(n_blocks, block=21, n_boot=400, trials=200, base_seed=7000,
                           field="raw"):
    """Coverage of the REAL block_bootstrap_delta_ci under a true null (independent i.i.d. sides,
    true Delta = 0). Cheap settings — enough to separate 78% from 93%, not to re-derive the table
    (disclosures.NULL_COVERAGE_TABLE is the recorded measurement; the derivation script settings
    are stated there). ``field="raw"`` (default) measures the RAW, UNCALIBRATED percentile
    interval (ci_lo_raw/ci_hi_raw) -- this is what the historical cycle-3 finding is ABOUT, and
    stays correct regardless of whether Amendment 1's calibration table happens to cover
    n_blocks. ``field="calibrated"`` measures the FINAL, possibly-widened ci_lo/ci_hi -- the
    quantity that actually gets banded."""
    lo_key, hi_key = (("ci_lo_raw", "ci_hi_raw") if field == "raw" else ("ci_lo", "ci_hi"))
    hit = 0
    for t in range(trials):
        rng = np.random.default_rng(base_seed + t)
        n = n_blocks * block
        ci = report_mod.block_bootstrap_delta_ci(
            rng.normal(0, 1, size=n), rng.normal(0, 1, size=n), block=block, n_boot=n_boot, seed=t)
        if ci[lo_key] is not None and ci[lo_key] <= 0.0 <= ci[hi_key]:
            hit += 1
    return hit / trials


def gate_calibrated_power_bar(failures):
    bar = report_mod.CALIBRATED_BOOTSTRAP_BLOCKS
    ok0 = bar > report_mod.MIN_BOOTSTRAP_BLOCKS
    print(f"    [{'PASS' if ok0 else 'FAIL'}] gate (power): the BANDING bar "
         f"({bar} blocks) is strictly above the DEGENERACY floor "
         f"({report_mod.MIN_BOOTSTRAP_BLOCKS} blocks) — two different guarantees, two bars")
    if not ok0:
        failures.append(f"gate (power): banding bar {bar} <= degeneracy floor "
                        f"{report_mod.MIN_BOOTSTRAP_BLOCKS}")

    # (a) the recorded tables must be real measurements of THIS code, re-measured at the two ends.
    tbl = disclosures_mod.NULL_COVERAGE_TABLE
    tbl_ov = disclosures_mod.NULL_COVERAGE_TABLE_OVERLAPPING
    ok1 = (isinstance(tbl, dict) and len(tbl) >= 6
           and all(0.0 < v <= 1.0 for v in tbl.values())
           and isinstance(tbl_ov, dict) and len(tbl_ov) >= 6
           and all(0.0 < v <= 1.0 for v in tbl_ov.values()))
    print(f"    [{'PASS' if ok1 else 'FAIL'}] gate (power): BOTH measured true-null coverage tables "
         f"are recorded in the disclosure (i.i.d.: "
         f"{len(tbl) if isinstance(tbl, dict) else 0} rows, overlapping-horizon: "
         f"{len(tbl_ov) if isinstance(tbl_ov, dict) else 0} rows)")
    if not ok1:
        failures.append(f"gate (power): coverage tables not usable: {tbl} / {tbl_ov}")
        return
    # the bar must be the MORE DEMANDING of the two derivations — the overlapping scenario is the
    # one the real forward returns live in, so it may never be the optimistic table that sets it.
    ok1b = (max(tbl_ov.values()) < max(tbl.values())
            and disclosures_mod.measured_null_coverage(bar, scenario="overlapping") is not None)
    print(f"    [{'PASS' if ok1b else 'FAIL'}] gate (power): the realistic overlapping-horizon "
         f"scenario covers strictly WORSE than i.i.d. at its best "
         f"({100 * max(tbl_ov.values()):.1f}% vs {100 * max(tbl.values()):.1f}%) and the bar has a "
         f"measured value in that table ("
         f"{100 * disclosures_mod.measured_null_coverage(bar, scenario='overlapping'):.1f}%)")
    if not ok1b:
        failures.append("gate (power): the overlapping-scenario calibration is missing or "
                        "implausibly better than i.i.d.")
    # the residual the bar CANNOT fix must be escalated in words, in the artifact.
    esc = disclosures_mod.NULL_COVERAGE_ESCALATION
    ok1c = isinstance(esc, str) and "ESCALATED" in esc and "block = 21" in esc
    print(f"    [{'PASS' if ok1c else 'FAIL'}] gate (power): the residual under-coverage that no "
         f"sample size can close is ESCALATED in the disclosure, not silently absorbed")
    if not ok1c:
        failures.append("gate (power): NULL_COVERAGE_ESCALATION missing/incomplete")

    floor_cov_raw = _measure_null_coverage(report_mod.MIN_BOOTSTRAP_BLOCKS, field="raw")
    bar_cov_raw = _measure_null_coverage(bar, field="raw")
    # The RAW/UNCALIBRATED degeneracy floor must be measurably under-covering (the historical
    # finding this whole mechanism exists to correct) — this is unaffected by whether Amendment
    # 1's calibration table happens to cover this block count, because ci_lo_raw/ci_hi_raw are
    # the pre-widening percentiles. Tolerance: 200 trials -> SE ~2pp, so allow 3pp.
    ok2 = floor_cov_raw < 0.90
    print(f"    [{'PASS' if ok2 else 'FAIL'}] gate (power): RAW (uncalibrated) coverage AT THE "
         f"DEGENERACY FLOOR ({report_mod.MIN_BOOTSTRAP_BLOCKS} blocks) is "
         f"{100 * floor_cov_raw:.1f}%, far below the nominal 95% — this is the historical "
         f"finding Amendment 1's calibrated-width correction exists to fix")
    if not ok2:
        failures.append(f"gate (power): RAW floor coverage {floor_cov_raw:.3f} unexpectedly "
                        f"high — the historical finding this bar rests on no longer reproduces")
    ok3 = bar_cov_raw >= disclosures_mod.NULL_COVERAGE_BAND_MIN - 0.03
    print(f"    [{'PASS' if ok3 else 'FAIL'}] gate (power): RAW coverage AT THE OLD FIXED BAR "
         f"({bar} blocks) is {100 * bar_cov_raw:.1f}% >= the disclosed minimum "
         f"{100 * disclosures_mod.NULL_COVERAGE_BAND_MIN:.0f}% (-3pp MC tolerance) — the "
         f"historical NULL_COVERAGE_TABLE_OVERLAPPING regression tripwire, unaffected by "
         f"Amendment 1 superseding this bar as a live gate")
    if not ok3:
        failures.append(f"gate (power): RAW coverage at the old bar ({bar} blocks) measured "
                        f"{bar_cov_raw:.3f} < disclosed {disclosures_mod.NULL_COVERAGE_BAND_MIN}")

    # AMENDMENT 1's OWN CLAIM, checked the same way: the CALIBRATED (widened) interval at the
    # DEGENERACY FLOOR (2 blocks — the thinnest calibrated stratum, disclosures.CALIBRATION_
    # TABLE[2]) now reaches the >= 93% target, where the RAW interval measured above does not.
    # This is the whole point of the re-cut: a fixed bar could only ever say "n is big enough";
    # calibration can certify even the thinnest stratum SPEC.md names.
    floor_cov_calibrated = _measure_null_coverage(report_mod.MIN_BOOTSTRAP_BLOCKS,
                                                  field="calibrated")
    ok2b = floor_cov_calibrated >= disclosures_mod.CALIBRATION_TARGET_COVERAGE - 0.05
    print(f"    [{'PASS' if ok2b else 'FAIL'}] gate (power): AMENDMENT 1 — the CALIBRATED "
         f"(widened) coverage at the SAME degeneracy floor is now {100 * floor_cov_calibrated:.1f}%"
         f" (target {100 * disclosures_mod.CALIBRATION_TARGET_COVERAGE:.0f}%, -5pp MC tolerance "
         f"at this small a trial count) — the fixed bar's job now done by measured widening, "
         f"not by refusing the stratum outright")
    if not ok2b:
        failures.append(f"gate (power): calibrated floor coverage {floor_cov_calibrated:.3f} "
                        f"does not reach the Amendment 1 target — the shipped calibration table "
                        f"does not reproduce")

    # (b) BEHAVIOUR: Amendment 1 supersedes the old claim "a sample between the floor and the
    # bar always reads LOW-POWER" — the whole point of the re-cut is that a block count WITH an
    # achieved calibration entry (2 blocks now measures k~3.49, validated coverage ~0.94 — see
    # CALIBRATION_TABLE) is now WIDENED and CORRECTLY BANDED, not blanket-refused. Demonstrated
    # at MIN_BOOTSTRAP_BLOCKS itself (the thinnest stratum this instrument will ever emit a CI
    # for): a real signal there now bands EDGE, via a properly widened (not raw) interval.
    rng = np.random.default_rng(31)
    n_lp = report_mod.MIN_BOOTSTRAP_BLOCKS * 21
    ci_lp = report_mod.block_bootstrap_delta_ci(
        rng.normal(0.5, 0.01, size=n_lp), rng.normal(0.0, 0.01, size=n_lp),
        block=21, n_boot=500, seed=0)
    band_lp = report_mod.band_for(ci_lp)
    ok4 = (ci_lp["low_power"] is False and ci_lp["calibration_achieved"] is True
          and band_lp == "EDGE" and ci_lp["ci_lo"] is not None and ci_lp["ci_lo"] > 0
          and (ci_lp["ci_hi"] - ci_lp["ci_lo"]) > (ci_lp["ci_hi_raw"] - ci_lp["ci_lo_raw"]))
    print(f"    [{'PASS' if ok4 else 'FAIL'}] gate (power): AMENDMENT 1 — a hugely positive delta "
         f"at {report_mod.MIN_BOOTSTRAP_BLOCKS} blocks now correctly bands {band_lp} through a "
         f"CALIBRATED (widened) CI [{ci_lp['ci_lo']:.4g}, {ci_lp['ci_hi']:.4g}], k="
         f"{ci_lp['calibration_k']!r} (raw, unwidened, was [{ci_lp['ci_lo_raw']:.4g}, "
         f"{ci_lp['ci_hi_raw']:.4g}])")
    if not ok4:
        failures.append(f"gate (power): the thinnest calibrated stratum did not band as "
                        f"Amendment 1 requires: {band_lp} {ci_lp}")

    # A DIRECT MECHANISM TEST of the path Amendment 1's own restricted, cheap-precision table
    # (disclosures.CALIBRATION_PRECISION_DISCLOSURE) cannot currently exercise end-to-end: every
    # probed block count ACHIEVED, so calibrated_k_for's "no achieved entry found" / LOW-POWER
    # branch is not reachable through any in-range query against the SHIPPED table today. Proven
    # directly against the real function via its own supported ``table=`` parameter (NOT
    # monkeypatching disclosures.CALIBRATION_TABLE — a distinct, explicit input to the same,
    # unmodified function) with an entry that failed calibration, exactly as a genuinely
    # uncalibratable stratum would appear in a future, fuller sweep.
    unachieved_table = {2: {"k": 5.0, "achieved": False, "reason": "did not reach target "
                            "(synthetic, for this mechanism test only)"}}
    k_none, achieved_none, src_none = disclosures_mod.calibrated_k_for(2, table=unachieved_table)
    ok4b = (k_none is None and achieved_none is False and src_none is None)
    print(f"    [{'PASS' if ok4b else 'FAIL'}] gate (power): calibrated_k_for correctly reports "
         f"NO achieved calibration (k=None, achieved=False) when the (real, unmodified) "
         f"function is given a table whose only candidate entry failed to achieve target — "
         f"the LOW-POWER mechanism, proven live even though the SHIPPED table has no such "
         f"entry in its currently-probed range")
    if not ok4b:
        failures.append(f"gate (power): calibrated_k_for mishandled an unachieved-only table: "
                        f"{k_none}/{achieved_none}/{src_none}")

    # (c) at/above the bar the same signal DOES band (the bar refuses weak evidence, not signal).
    n_ok = bar * 21
    ci_hi_pow = report_mod.block_bootstrap_delta_ci(
        rng.normal(0.5, 0.01, size=n_ok), rng.normal(0.0, 0.01, size=n_ok),
        block=21, n_boot=500, seed=0)
    band_hi = report_mod.band_for(ci_hi_pow)
    ok5 = ci_hi_pow["low_power"] is False and band_hi == "EDGE"
    print(f"    [{'PASS' if ok5 else 'FAIL'}] gate (power): the SAME signal at the bar "
         f"({bar} blocks) does band ({band_hi})")
    if not ok5:
        failures.append(f"gate (power): the bar suppressed a well-powered EDGE: {ci_hi_pow}")

    # (d) THROUGH the per-regime buckets, which is where SPEC.md says thinness bites. With
    # Amendment 1's calibration covering 2 blocks (achieved), this bucket is now CORRECTLY
    # bandable rather than blanket-refused — the mechanism test above already covers the
    # (currently unreachable through the shipped table) LOW-POWER path directly.
    work = tempfile.mkdtemp(prefix="t892_power_")
    try:
        subj_p, ts = _synthetic_pair_doc(work, "pw_subject", 200, 4, seed=5)
        null_p, _ = _synthetic_pair_doc(work, "pw_null", 200, 4, seed=6)
        # a bucket admitting 12 decisions x 4 assets = 48 cells => 2 blocks: the thinnest
        # stratum this instrument's degeneracy floor allows, and (per CALIBRATION_TABLE) now
        # calibrated/achieved.
        ext = {"MID_BUCKET": {"window": ["m0", "m1"], "start_ts": int(ts[0]),
                              "end_ts": int(ts[11])},
               "WIDE_BUCKET": {"window": ["w0", "w1"], "start_ts": int(ts[0]),
                               "end_ts": int(ts[-1])}}
        rep = report_mod.compare(subj_p, [null_p], block=21, n_boot=500, seed=0, ext_windows=ext)
        mid = rep["per_window"]["MID_BUCKET"]
        ok6 = (mid["band"] in report_mod.SUBSTANTIVE_BANDS
              and mid["delta"]["calibration_achieved"] is True
              and mid["excluded_from_banding"] is False)
        print(f"    [{'PASS' if ok6 else 'FAIL'}] gate (power)/ext_windows: a bucket with "
             f"{mid['delta']['n_blocks_subject']} block(s) is now CALIBRATED and bands "
             f"({mid['band']}) rather than being blanket-refused (Amendment 1)")
        if not ok6:
            failures.append(f"gate (power)/ext_windows: mid bucket did not calibrate/band: {mid}")
        wide = rep["per_window"]["WIDE_BUCKET"]
        ok7 = wide["band"] in report_mod.SUBSTANTIVE_BANDS
        print(f"    [{'PASS' if ok7 else 'FAIL'}] gate (power)/ext_windows: the WIDE bucket "
             f"({wide['delta']['n_blocks_subject']} blocks) keeps its band ({wide['band']})")
        if not ok7:
            failures.append(f"gate (power)/ext_windows: wide bucket lost its band: {wide}")
        # the report states the bar and the table so the number is auditable from the artifact
        ok8 = (rep["calibrated_min_blocks"] == bar
               and rep["measured_null_coverage_table"] == tbl
               and rep["measured_null_coverage_table_overlapping"] == tbl_ov
               and rep["bootstrap_calibration_escalation"] == esc
               and "COVERAGE" in disclosures_mod.BOOTSTRAP_POWER_DISCLOSURE.upper())
        print(f"    [{'PASS' if ok8 else 'FAIL'}] gate (power): the report carries the bar, BOTH "
             f"measured coverage tables, the escalation, and the disclosure that explains them")
        if not ok8:
            failures.append("gate (power): the report does not carry its own calibration evidence")
    finally:
        shutil.rmtree(work, ignore_errors=True)


# =============================================================================
# CYCLE-3 gate (zero-width) — finding 3: report.py:234 used to claim "Nothing here can emit a
# zero-width interval". The block floor guarantees enough BLOCKS, not enough VARIANCE.
# =============================================================================

def gate_zero_width_ci_never_bands(failures):
    s = np.full(42, 0.02)
    n = np.full(42, 0.0)
    ci = report_mod.block_bootstrap_delta_ci(s, n, block=21, n_boot=1000, seed=0)
    band = report_mod.band_for(ci)
    ok = (ci["zero_width_ci"] is True and band == report_mod.BAND_ZERO_WIDTH
          and ci["insufficient_n"] is False and "ZERO-WIDTH-CI" in ci["note"])
    print(f"    [{'PASS' if ok else 'FAIL'}] gate (zero-width): constant inputs AT the block floor "
         f"(ci=[{ci['ci_lo']}, {ci['ci_hi']}], width={ci['ci_width']}) read {band}, never EDGE")
    if not ok:
        failures.append(f"gate (zero-width): a zero-width CI carried band {band}: {ci}")

    # and at a HIGH block count too — this is not a thin-sample condition, it is a variance one.
    s2, n2 = np.full(21 * 40, 0.02), np.full(21 * 40, 0.0)
    ci2 = report_mod.block_bootstrap_delta_ci(s2, n2, block=21, n_boot=500, seed=0)
    band2 = report_mod.band_for(ci2)
    ok2 = ci2["zero_width_ci"] is True and band2 == report_mod.BAND_ZERO_WIDTH
    print(f"    [{'PASS' if ok2 else 'FAIL'}] gate (zero-width): the same degeneracy at 40 blocks "
         f"(well past every power bar) still reads {band2}")
    if not ok2:
        failures.append(f"gate (zero-width): a 40-block zero-width CI banded {band2}: {ci2}")

    # a normal, varying sample is untouched: it keeps a positive-width, bandable CI.
    rng = np.random.default_rng(77)
    n_ok = 21 * report_mod.CALIBRATED_BOOTSTRAP_BLOCKS
    ci3 = report_mod.block_bootstrap_delta_ci(rng.normal(0, 0.01, size=n_ok),
                                              rng.normal(0, 0.01, size=n_ok),
                                              block=21, n_boot=500, seed=0)
    ok3 = ci3["zero_width_ci"] is False and ci3["ci_width"] > 0
    print(f"    [{'PASS' if ok3 else 'FAIL'}] gate (zero-width): a normal varying sample keeps a "
         f"positive-width CI (width={ci3['ci_width']:.6g}) and is unaffected")
    if not ok3:
        failures.append(f"gate (zero-width): a healthy CI was flagged zero-width: {ci3}")

    # D51 RE-CUT — THE NEAR CASE, not just the exact one (gate vii's transferable rule, and the
    # cycle-3 killer this row was cut to fix): bit-DISTINCT per-block jitter, not bit-identical
    # inputs. Reproduced independently of the factory orchestrator's own cross-validation (which
    # measured the shipped ``not (hi > lo)`` guard missing this exact construction at CI width
    # ~6.7e-10, reading EDGE): here, per-block jitter at 1e-9 and 1e-6 scale must now read
    # ZERO-WIDTH-CI (excluded from banding), while a per-block jitter at a REALISTIC scale (with
    # a real, non-degenerate mean level) must still band.
    def _near_duplicate(n_blocks, jitter_std, base_s=0.02, base_n=0.0, block=21, seed=0):
        rng = np.random.default_rng(seed)
        jit_s = rng.normal(0.0, jitter_std, size=n_blocks)
        jit_n = rng.normal(0.0, jitter_std, size=n_blocks)
        s_ = np.repeat(base_s + jit_s, block)
        n_ = np.repeat(base_n + jit_n, block)
        return s_, n_

    for jitter_std, n_blocks, expect_zero_width in ((1e-9, 35, True), (1e-6, 35, True),
                                                    (1e-9, 33, True), (1e-6, 33, True)):
        s_nd, n_nd = _near_duplicate(n_blocks, jitter_std, seed=int(jitter_std * 1e12) + n_blocks)
        ci_nd = report_mod.block_bootstrap_delta_ci(s_nd, n_nd, block=21, n_boot=2000, seed=0)
        band_nd = report_mod.band_for(ci_nd)
        ok_nd = (ci_nd["insufficient_n"] is False and ci_nd["zero_width_ci"] is expect_zero_width
                 and (band_nd == report_mod.BAND_ZERO_WIDTH) is expect_zero_width)
        print(f"    [{'PASS' if ok_nd else 'FAIL'}] gate (zero-width, NEAR case): per-block "
             f"jitter std={jitter_std:g} at {n_blocks} blocks (bit-DISTINCT, not bit-identical: "
             f"width={ci_nd['ci_width']:.3e}) -> zero_width_ci={ci_nd['zero_width_ci']} "
             f"band={band_nd} (expected zero_width_ci={expect_zero_width})")
        if not ok_nd:
            failures.append(
                f"gate (zero-width, NEAR case): jitter_std={jitter_std:g} n_blocks={n_blocks} "
                f"width={ci_nd['ci_width']!r} zero_width_ci={ci_nd['zero_width_ci']} "
                f"band={band_nd} (a near-duplicate, bit-DISTINCT input slipped past the "
                f"rebuilt guard the same way the bit-identical-only guard let it through in "
                f"cycle 3)")

    # a REAL, non-degenerate delta with realistic-scale dispersion must NOT be caught by the
    # near-duplicate rebuild (no over-exclusion): this is the D51 "guard vs band threshold"
    # boundary — the fix must not turn into a second, disguised power floor.
    rng_real = np.random.default_rng(55)
    n_real = 21 * 35
    s_real = 0.02 + rng_real.normal(0, 0.01, size=n_real)
    n_real_arr = 0.0 + rng_real.normal(0, 0.01, size=n_real)
    ci_real = report_mod.block_bootstrap_delta_ci(s_real, n_real_arr, block=21, n_boot=2000,
                                                  seed=0)
    ok_real = ci_real["zero_width_ci"] is False and ci_real["ci_width"] > report_mod.ZERO_WIDTH_FLOOR
    print(f"    [{'PASS' if ok_real else 'FAIL'}] gate (zero-width, NEAR case): a REALISTIC-scale "
         f"(std=0.01) sample at 35 blocks is NOT swept up by the rebuilt guard "
         f"(width={ci_real['ci_width']:.4g}, zero_width_ci={ci_real['zero_width_ci']})")
    if not ok_real:
        failures.append(f"gate (zero-width, NEAR case): a realistic-scale sample was "
                        f"over-excluded as zero-width: {ci_real}")

    # THE COMMENT IS NOW TRUE: no path in this module may return a zero-width interval that a
    # caller could band. Assert the invariant over the union of both refusal paths.
    for label, (a, b) in {"constant-at-floor": (np.full(42, 1.0), np.full(42, 1.0)),
                          "constant-thin": (np.full(10, 1.0), np.full(10, 1.0))}.items():
        c = report_mod.block_bootstrap_delta_ci(a, b, block=21, n_boot=200, seed=0)
        bandable = report_mod.band_for(c) in report_mod.SUBSTANTIVE_BANDS
        okz = not bandable
        print(f"    [{'PASS' if okz else 'FAIL'}] gate (zero-width): {label} is never bandable")
        if not okz:
            failures.append(f"gate (zero-width): {label} produced a bandable zero-width CI: {c}")


# =============================================================================
# CYCLE-3 gate (race) — finding 4: the --smoke default checkpoint dir is unique per invocation.
# The shipped default was one literal path, and this suite's own two back-to-back --smoke
# subprocesses were nondeterministic because of it.
# =============================================================================

def gate_smoke_ckpt_dir_no_race(failures, work):
    scratch = os.path.join(EXP_DIR, "_scratch")
    before = set(os.listdir(scratch)) if os.path.isdir(scratch) else set()
    procs = []
    outs = [os.path.join(work, f"race_trace_{i}.json") for i in range(3)]
    for o in outs:
        procs.append(subprocess.Popen([sys.executable, TRACE_PY, "--smoke", "--out", o],
                                      cwd=_REPO, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, text=True))
    results = []
    for p in procs:
        so, se = p.communicate(timeout=600)
        results.append((p.returncode, se))
    n_ok = sum(1 for (rc, _), o in zip(results, outs) if rc == 0 and os.path.exists(o))
    ok = n_ok == 3
    print(f"    [{'PASS' if ok else 'FAIL'}] gate (race): 3 CONCURRENT `trace.py --smoke` runs "
         f"with no --smoke-ckpt-dir all succeed ({n_ok}/3) — the default checkpoint dir is "
         f"private per invocation")
    if not ok:
        first_err = next((se for (rc, se) in results if rc != 0), "")
        failures.append(f"gate (race): only {n_ok}/3 concurrent --smoke runs survived; "
                        f"first error: {first_err[-800:]}")
    # and each run's private checkpoint dir is cleaned up after its save->load roundtrip, so three
    # concurrent runs leave the scratch directory exactly as they found it (measured against a
    # snapshot taken before the runs, so a pre-existing scratch entry cannot fail this).
    after = set(os.listdir(scratch)) if os.path.isdir(scratch) else set()
    leftovers = sorted(after - before)
    ok2 = not leftovers
    print(f"    [{'PASS' if ok2 else 'FAIL'}] gate (race): the three runs left NO new entry in "
         f"_scratch/ — each private checkpoint dir is cleaned up ({len(leftovers)} found)")
    if not ok2:
        failures.append(f"gate (race): leftover default checkpoint dirs: {leftovers}")
    # an EXPLICIT --smoke-ckpt-dir is still honoured verbatim (finding 7 stays closed) — proven by
    # gate (budget)/scratch and gate (pin), which both depend on that exact path surviving.


# =============================================================================
# CYCLE-3 gate (nan-budget) — finding 5: `NaN > x` is False, so the bare overdraft test could not
# see a non-finite allocation, which made PRD-2's "cannot miss an overdraft/leverage defect" claim
# false as written.
# =============================================================================

def gate_nan_allocation_is_a_violation(failures):
    book = np.ones((3, 2), dtype=np.int64)
    cases = {"nan": np.array([0.5, np.nan, 0.9]),
             "posinf": np.array([0.5, np.inf, 0.9]),
             "neginf": np.array([0.5, -np.inf, 0.9])}
    for name, alloc in cases.items():
        n_viol, _ = trace_mod.budget_conservation_violations(book, allocated_fraction=alloc)
        ok = n_viol == 1
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (nan-budget): a {name} allocation counts as "
             f"a budget violation (got {n_viol})")
        if not ok:
            failures.append(f"gate (nan-budget): {name} allocation counted {n_viol} violations")
    # a healthy book is still clean, and a real overdraft is still caught
    n_clean, _ = trace_mod.budget_conservation_violations(
        book, allocated_fraction=np.array([0.5, 0.9, 1.0]))
    n_over, _ = trace_mod.budget_conservation_violations(
        book, allocated_fraction=np.array([0.5, 1.5, 0.9]))
    ok2 = n_clean == 0 and n_over == 1
    print(f"    [{'PASS' if ok2 else 'FAIL'}] gate (nan-budget): a legal book still reports 0 and a "
         f"real overdraft still reports 1 (got {n_clean} / {n_over})")
    if not ok2:
        failures.append(f"gate (nan-budget): clean/overdraft behaviour changed: {n_clean}/{n_over}")
    # the fail-closed path is end-to-end: a NaN allocation makes the REPORT read INSTRUMENT FAIL
    work = tempfile.mkdtemp(prefix="t892_nanbudget_")
    try:
        rng = np.random.default_rng(12)
        n_dec, n_assets, horizon = 200, 4, 5
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01,
                                                     size=(n_dec + horizon + 5, n_assets)), axis=0))
        rows, ts = np.arange(n_dec), np.arange(n_dec) * 60
        alloc = np.full(n_dec, 0.5)
        alloc[7] = np.nan
        subj = trace_mod.build_trace_doc(
            tag="nan-budget", checkpoint_label="synthetic", decision_rule="argmax",
            action_marginals_path=None, eval_seed=0, tier_bp=0, fee_bps=0.0, slip_bps=0.0,
            horizon=horizon, position_size_pct=0.1,
            col_symbols=[f"S{i}" for i in range(n_assets)], row_index=rows, timestamps=ts,
            chosen_vectors=np.full((n_dec, n_assets), 2, dtype=np.int32),
            realized_book=np.ones((n_dec, n_assets), dtype=np.int32), prices=prices,
            universe_pin=None, walk_kind=trace_mod.WALK_KIND_SMOKE, allocated_fraction=alloc)
        null = _trace_doc_from_arrays(
            [f"S{i}" for i in range(n_assets)], rows, ts,
            np.full((n_dec, n_assets), 2, dtype=np.int32),
            np.ones((n_dec, n_assets), dtype=np.int32), prices, horizon, "nan-null",
            "random-matched", 1)
        sp, np_ = os.path.join(work, "s.json"), os.path.join(work, "n.json")
        for p, d in ((sp, subj), (np_, null)):
            with open(p, "w") as fh:
                json.dump(d, fh, default=float)
        rep = report_mod.compare(sp, [np_], block=21, n_boot=500, seed=0)
        ok3 = (rep["band"] == report_mod.BAND_INSTRUMENT_FAIL
               and rep["budget_conservation_violations"] == 1)
        print(f"    [{'PASS' if ok3 else 'FAIL'}] gate (nan-budget): a walk with a NaN allocation "
             f"makes the whole report read {rep['band']} "
             f"({rep['budget_conservation_violations']} violation(s)) instead of a band")
        if not ok3:
            failures.append(f"gate (nan-budget): NaN allocation did not fail the instrument: "
                            f"{rep['band']} / {rep['budget_conservation_violations']}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


# =============================================================================
# CYCLE-3 gate (walk-kind) — finding 6: two positive-match raises are an ALLOW-LIST; a third value
# slipped through and inherited the smoke branch's note.
# =============================================================================

def gate_walk_kind_is_a_validated_enum(failures):
    base = dict(tag="t", checkpoint_label="runs/861_ref/ckpt_ref_s0", decision_rule="argmax",
                action_marginals_path=None, eval_seed=0, tier_bp=0, fee_bps=0.0, slip_bps=0.0,
                horizon=2, position_size_pct=0.1, col_symbols=["A"], row_index=[0], timestamps=[0],
                chosen_vectors=np.array([[2]]), realized_book=np.array([[1]]),
                prices=np.array([[100.0], [101.0], [102.0], [103.0]]),
                allocated_fraction=np.array([0.5]))
    for bad_kind in ("bogus-walk-kind", "", None, "smoke", trace_mod.WALK_KIND_SMOKE.upper()):
        refused = False
        emitted = None
        try:
            emitted = trace_mod.build_trace_doc(universe_pin=None, walk_kind=bad_kind, **base)
        except ValueError:
            refused = True
        print(f"    [{'PASS' if refused else 'FAIL'}] gate (walk-kind): walk_kind={bad_kind!r} is "
             f"REFUSED up front (not silently accepted with a note from another branch)")
        if not refused:
            failures.append(f"gate (walk-kind): walk_kind={bad_kind!r} was accepted and emitted "
                            f"note={emitted['provenance']['universe_pin_note']!r}")
    # the two legitimate kinds still work exactly as before
    smoke_doc = trace_mod.build_trace_doc(universe_pin=None,
                                          walk_kind=trace_mod.WALK_KIND_SMOKE, **base)
    ok = (smoke_doc["provenance"]["walk_kind"] == trace_mod.WALK_KIND_SMOKE
          and smoke_doc["provenance"]["universe_pin"] is None
          and "smoke" in smoke_doc["provenance"]["universe_pin_note"])
    print(f"    [{'PASS' if ok else 'FAIL'}] gate (walk-kind): the SMOKE kind still emits its "
         f"truthful null-pin note")
    if not ok:
        failures.append("gate (walk-kind): the smoke kind regressed")
    ck_doc = trace_mod.build_trace_doc(
        universe_pin={"mechanism": "explicit-symbols", "verified": False,
                      "verified_reason": "test"},
        walk_kind=trace_mod.WALK_KIND_CHECKPOINT, **base)
    ok2 = (ck_doc["provenance"]["walk_kind"] == trace_mod.WALK_KIND_CHECKPOINT
           and ck_doc["provenance"]["universe_pin_note"] is None)
    print(f"    [{'PASS' if ok2 else 'FAIL'}] gate (walk-kind): the CHECKPOINT kind still embeds "
         f"its record with no smoke note")
    if not ok2:
        failures.append("gate (walk-kind): the checkpoint kind regressed")


# =============================================================================
# CYCLE-3 gate (ci-args) — finding 7: block/n_boot validation. Unreachable via the pinned
# invocation, so the pre-fix behaviour was a raw crash rather than a wrong number — but a crash
# from inside numpy is not a message.
# =============================================================================

def gate_bootstrap_arg_validation(failures):
    s = np.arange(100.0)
    n = np.arange(100.0)
    cases = {"block=0": {"block": 0}, "block=-5": {"block": -5}, "n_boot=0": {"n_boot": 0},
             "n_boot=-1": {"n_boot": -1}}
    for name, kw in cases.items():
        got = None
        try:
            report_mod.block_bootstrap_delta_ci(s, n, **kw)
        except ValueError as e:
            got = str(e)
        except Exception as e:                                    # noqa: BLE001 — the RED case
            got = f"{type(e).__name__}: {e}"
        ok = got is not None and got.startswith("block_bootstrap_delta_ci:")
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (ci-args): {name} raises a NAMED ValueError "
             f"({(got or 'accepted')[:70]})")
        if not ok:
            failures.append(f"gate (ci-args): {name} -> {got!r} (expected a named ValueError)")
    # the pinned SPEC.md invocation is unaffected
    ok2 = report_mod.block_bootstrap_delta_ci(s, n, block=21, n_boot=100, seed=0)["block"] == 21
    print(f"    [{'PASS' if ok2 else 'FAIL'}] gate (ci-args): the pinned block=21 / n_boot "
         f"invocation is unaffected")
    if not ok2:
        failures.append("gate (ci-args): the pinned invocation broke")


# =============================================================================
# CYCLE-3 gate (atomic) — every artifact write is <path>.tmp + os.replace, so a kill mid-dump
# cannot leave a truncated JSON at the path a reader trusts.
# =============================================================================

def gate_atomic_artifact_writes(failures, subject_path, null_path, work):
    # (a) the helper itself: the temp file is gone and the target is complete + valid JSON.
    p = os.path.join(work, "atomic_probe.json")
    trace_mod.write_json_atomic(p, {"a": [1, 2, 3]}, indent=2)
    with open(p) as fh:
        got = json.load(fh)
    ok = got == {"a": [1, 2, 3]} and not os.path.exists(p + ".tmp")
    print(f"    [{'PASS' if ok else 'FAIL'}] gate (atomic): write_json_atomic lands complete JSON "
         f"and leaves no .tmp behind")
    if not ok:
        failures.append(f"gate (atomic): helper misbehaved: {got}, tmp_exists="
                        f"{os.path.exists(p + '.tmp')}")

    # (b) an interrupted write cannot damage an EXISTING good artifact — the .tmp takes the hit.
    trace_mod.write_json_atomic(p, {"good": True})
    try:
        trace_mod.write_json_atomic(p, {"bad": {1, 2}})            # a set is not JSON-serializable
    except TypeError:
        pass
    # A non-atomic writer leaves TRUNCATED JSON here, so the reload itself must be guarded — an
    # escaping JSONDecodeError would abort the whole suite instead of reporting one failed check.
    try:
        with open(p) as fh:
            survived = json.load(fh)
    except (json.JSONDecodeError, OSError) as e:
        survived = f"<UNREADABLE: {type(e).__name__}: {e}>"
    ok2 = survived == {"good": True}
    print(f"    [{'PASS' if ok2 else 'FAIL'}] gate (atomic): a write that dies mid-dump leaves the "
         f"PREVIOUS artifact intact (got {survived})")
    if not ok2:
        failures.append(f"gate (atomic): a failed write damaged the target — the write is NOT "
                        f"atomic: {survived}")

    # (c) all three producers use it — the artifacts the real CLI wrote are complete JSON with no
    # .tmp siblings anywhere.
    if subject_path is None:
        failures.append("gate (atomic): skipped the producer check — upstream CLI chain failed")
        return
    sidecar = os.path.join(work, "atomic_sidecar.json")
    r = _run([BUILD_MARGINALS_PY, "--trace", subject_path, "--out", sidecar])
    rep_out = os.path.join(work, "atomic_report.json")
    r2 = _run([os.path.join(EXP_DIR, "report.py"), "--subject", subject_path, "--null",
               null_path, "--out", rep_out, "--n-boot", "200"])
    produced = [subject_path, null_path, sidecar, rep_out]
    ok3 = r.returncode == 0 and r2.returncode == 0
    for path in produced:
        if not os.path.exists(path):
            ok3 = False
            continue
        if os.path.exists(path + ".tmp"):
            ok3 = False
        try:
            with open(path) as fh:
                json.load(fh)
        except Exception:                                          # noqa: BLE001
            ok3 = False
    print(f"    [{'PASS' if ok3 else 'FAIL'}] gate (atomic): trace.py / build_action_marginals.py "
         f"/ report.py all wrote complete JSON with no .tmp siblings left")
    if not ok3:
        failures.append(f"gate (atomic): a producer left a .tmp or an unreadable artifact "
                        f"(build rc={r.returncode}, report rc={r2.returncode})")


# =============================================================================
# D51 RE-CUT / AMENDMENT 1 gate (leg) — the OOS binding leg vs the sanity leg is a STRUCTURAL
# marker (validate_leg_window), not a comment. A report tagged with a leg whose data does not
# match that leg's pre-registered window must be REFUSED, not silently mislabeled.
# =============================================================================

def gate_leg_binding_is_structural(failures):
    work = tempfile.mkdtemp(prefix="t892_leg_")
    try:
        # decisions timestamped well inside the OOS binding window
        oos_start = report_mod.LEG_WINDOW_EPOCH[report_mod.LEG_BINDING][0]
        subj_p, ts = _synthetic_pair_doc(work, "leg_subject_oos", 60, 3, seed=1,
                                         ts_base=oos_start + 86400, ts_step=3600)
        null_p, _ = _synthetic_pair_doc(work, "leg_null_oos", 60, 3, seed=2,
                                        ts_base=oos_start + 86400, ts_step=3600)
        rep = report_mod.compare(subj_p, [null_p], block=21, n_boot=300, seed=0,
                                 leg=report_mod.LEG_BINDING)
        ok1 = (rep["leg"] == report_mod.LEG_BINDING and rep["leg_is_binding"] is True
              and "BINDING" in rep["leg_role_note"])
        print(f"    [{'PASS' if ok1 else 'FAIL'}] gate (leg): decisions inside the OOS window "
             f"accept --leg oos-binding and the report stamps leg_is_binding=True")
        if not ok1:
            failures.append(f"gate (leg): correct-window leg declaration mishandled: "
                            f"{rep.get('leg')}/{rep.get('leg_is_binding')}")

        # the SAME data declared as the SANITY leg must be REFUSED — it is outside that window.
        refused = False
        try:
            report_mod.compare(subj_p, [null_p], block=21, n_boot=300, seed=0,
                               leg=report_mod.LEG_SANITY)
        except ValueError:
            refused = True
        print(f"    [{'PASS' if refused else 'FAIL'}] gate (leg): OOS-window decisions declared "
             f"--leg sanity are REFUSED (structural mismatch), not silently relabeled")
        if not refused:
            failures.append("gate (leg): a leg/window mismatch was silently accepted")

        # no --leg at all: reported, but explicitly NOT citable as the binding read.
        rep_noleg = report_mod.compare(subj_p, [null_p], block=21, n_boot=300, seed=0, leg=None)
        ok2 = (rep_noleg["leg"] is None and rep_noleg["leg_is_binding"] is False
              and "NO --leg DECLARED" in rep_noleg["leg_role_note"])
        print(f"    [{'PASS' if ok2 else 'FAIL'}] gate (leg): with no --leg declared, the report "
             f"says plainly it cannot be cited as the D47 binding read")
        if not ok2:
            failures.append(f"gate (leg): undeclared leg did not disclose its own non-binding "
                            f"status: {rep_noleg.get('leg_role_note')}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


# =============================================================================
# D51 RE-CUT / AMENDMENT 1 gate (fee-tier) — D49(a): fee16/slip5 is a REPORTED DIAGNOSTIC ONLY.
# Enforced in band_for's force_diagnostic, not merely in prose: a fee16-tier trace pair whose
# raw delta/CI would otherwise read EDGE must come back BAND_FEE_TIER_DIAGNOSTIC.
# =============================================================================

def _tiered_trace_doc(col_symbols, row_index, timestamps, chosen_vectors, realized_book, prices,
                      horizon, tag, decision_rule, eval_seed, tier_bp):
    n_assets = len(col_symbols)
    book = np.asarray(realized_book)
    allocated = (book != 0).sum(axis=1) / float(n_assets)
    fee, slip = (16.0, 5.0) if tier_bp else (0.0, 0.0)
    return trace_mod.build_trace_doc(
        tag=tag, checkpoint_label="synthetic", decision_rule=decision_rule,
        action_marginals_path=None, eval_seed=eval_seed, tier_bp=tier_bp, fee_bps=fee,
        slip_bps=slip, horizon=horizon, position_size_pct=0.1, col_symbols=col_symbols,
        row_index=row_index, timestamps=timestamps, chosen_vectors=chosen_vectors,
        realized_book=realized_book, prices=prices, universe_pin=None,
        walk_kind=trace_mod.WALK_KIND_SMOKE, allocated_fraction=allocated,
        allocation_semantics=trace_mod.WEIGHT_SEMANTICS, weight_unit=1.0 / n_assets,
        weight_max_mult=4.0)


def gate_fee_tier_is_diagnostic_only(failures):
    work = tempfile.mkdtemp(prefix="t892_feetier_")
    try:
        n_decisions, n_assets = 800, 4
        prices, row_index, chosen, book, horizon = None, None, None, None, None
        rng = np.random.default_rng(9)
        horizon = 5
        n_rows = n_decisions + horizon + 5
        # a strong, unambiguous planted long-side drift so cost0 would read EDGE.
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0.003, 0.005, size=(n_rows, n_assets)),
                                          axis=0))
        row_index = np.arange(n_decisions)
        chosen = np.full((n_decisions, n_assets), 2, dtype=np.int32)     # all Buy
        book = np.ones((n_decisions, n_assets), dtype=np.int32)          # all long
        ts = row_index * 60
        col_symbols = [f"S{i}" for i in range(n_assets)]

        for tier_bp, expect_diagnostic in ((0, False), (21, True)):
            subj = _tiered_trace_doc(col_symbols, row_index, ts, chosen, book, prices, horizon,
                                     f"fee-subj-{tier_bp}", "argmax", 0, tier_bp)
            # this gate only needs a SUBSTANTIVE would-be band from block_bootstrap_delta_ci to
            # exercise the fee-tier override — an EXACT activity-rate/long-short-mix TWIN of the
            # subject's own book (100% non-HOLD, 100% long, both sides) keeps the SPEC.md 0.5pp
            # instrument-fail check at exactly 0pp, so a genuine NULL/EDGE band is what reaches
            # band_for, not an unrelated INSTRUMENT FAIL swallowing the signal this gate probes.
            null_ = _tiered_trace_doc(col_symbols, row_index, ts, chosen, book, prices, horizon,
                                      f"fee-null-{tier_bp}", "random-matched", 1, tier_bp)
            sp = os.path.join(work, f"subj_{tier_bp}.json")
            npth = os.path.join(work, f"null_{tier_bp}.json")
            for p, d in ((sp, subj), (npth, null_)):
                with open(p, "w") as fh:
                    json.dump(d, fh, default=float)
            rep = report_mod.compare(sp, [npth], block=21, n_boot=1000, seed=0)
            is_diag = rep["band"] == report_mod.BAND_FEE_TIER_DIAGNOSTIC
            ok = (is_diag is expect_diagnostic and rep["is_fee_tier_diagnostic"] is
                 expect_diagnostic and rep["tier_bp"] == tier_bp)
            print(f"    [{'PASS' if ok else 'FAIL'}] gate (fee-tier): tier_bp={tier_bp} -> "
                 f"band={rep['band']} delta={rep['delta']['delta']:.4g} "
                 f"is_fee_tier_diagnostic={rep['is_fee_tier_diagnostic']} "
                 f"(expected diagnostic-only={expect_diagnostic})")
            if not ok:
                failures.append(f"gate (fee-tier): tier_bp={tier_bp} band={rep['band']} "
                                f"is_fee_tier_diagnostic={rep['is_fee_tier_diagnostic']}")
            if expect_diagnostic:
                ok_disc = rep.get("fee_tier_disclosure") == disclosures_mod.FEE_TIER_DIAGNOSTIC_DISCLOSURE
                print(f"    [{'PASS' if ok_disc else 'FAIL'}] gate (fee-tier): the D49(a) "
                     f"disclosure is carried inline on a fee-tier report")
                if not ok_disc:
                    failures.append("gate (fee-tier): fee_tier_disclosure missing/wrong on a "
                                    "fee16 report")

        # ------------------------------------------------------------------------------------
        # FIX PASS (MAJOR 1) — THE OMITTED-FIELD CASE, which the two cases above could not see.
        # RED against the pre-fix code: `provenance.get("tier_bp", 0)` accepted a schema-v3
        # document that simply OMITS the field, so a GENUINE fee16/slip5 pair (tier_bp=21,
        # fee_bps=16, slip_bps=5 — the very documents built above) came back as a full citable
        # EDGE with instrument_fail: [] on identical numbers, instead of DIAGNOSTIC-ONLY. The
        # field is now hard-required and hard-subscripted, so the same document is REFUSED.
        # ------------------------------------------------------------------------------------
        subj21 = _tiered_trace_doc(col_symbols, row_index, ts, chosen, book, prices, horizon,
                                   "fee-subj-omitted", "argmax", 0, 21)
        null21 = _tiered_trace_doc(col_symbols, row_index, ts, chosen, book, prices, horizon,
                                   "fee-null-omitted", "random-matched", 1, 21)
        assert subj21["provenance"]["tier_bp"] == 21          # a REAL fee16/slip5 trace ...
        del subj21["provenance"]["tier_bp"]                   # ... with the one gating key gone
        sp_om = os.path.join(work, "subj_tier_omitted.json")
        np_om = os.path.join(work, "null_tier_present.json")
        for p, d in ((sp_om, subj21), (np_om, null21)):
            with open(p, "w") as fh:
                json.dump(d, fh, default=float)
        msg_om, banded_om = "", None
        try:
            banded_om = report_mod.compare(sp_om, [np_om], block=21, n_boot=1000, seed=0)
        except RuntimeError as e:
            msg_om = str(e)
        ok_om = "tier_bp" in msg_om
        print(f"    [{'PASS' if ok_om else 'FAIL'}] gate (fee-tier): a fee16/slip5 trace that "
             f"OMITS provenance.tier_bp is REFUSED by name — an absent cost tier is not cost0 "
             f"(pre-fix: band={None if banded_om is None else banded_om['band']})")
        if not ok_om:
            failures.append(
                f"gate (fee-tier): a fee16 trace with tier_bp OMITTED was graded as "
                f"band={None if banded_om is None else banded_om['band']} "
                f"is_fee_tier_diagnostic="
                f"{None if banded_om is None else banded_om['is_fee_tier_diagnostic']} — the "
                f"D49(a) carve-out is defeatable by deleting one field")

        # ... and the reviewer's exact reproduction: the key deleted on BOTH sides, so pre-fix
        # nothing was even mismatched — both sides read as cost0, the carve-out never fired, and
        # the fee16 numbers came back as a substantive citable band with instrument_fail: [].
        both = []
        for name, rule, sd in (("subj", "argmax", 0), ("null", "random-matched", 1)):
            d = _tiered_trace_doc(col_symbols, row_index, ts, chosen, book, prices, horizon,
                                  f"fee-{name}-omitted-both", rule, sd, 21)
            del d["provenance"]["tier_bp"]
            p = os.path.join(work, f"omitted_both_{name}.json")
            with open(p, "w") as fh:
                json.dump(d, fh, default=float)
            both.append(p)
        msg_b, rep_b = "", None
        try:
            rep_b = report_mod.compare(both[0], [both[1]], block=21, n_boot=1000, seed=0)
        except RuntimeError as e:
            msg_b = str(e)
        ok_b = "tier_bp" in msg_b
        print(f"    [{'PASS' if ok_b else 'FAIL'}] gate (fee-tier): the same omission on BOTH "
             f"sides (pre-fix: a cost0-looking, fully citable band on fee16 numbers) is REFUSED "
             f"too (pre-fix: band={None if rep_b is None else rep_b['band']}, instrument_fail="
             f"{None if rep_b is None else rep_b['instrument_fail_reasons']})")
        if not ok_b:
            failures.append(
                f"gate (fee-tier): fee16 traces with tier_bp omitted on BOTH sides graded as "
                f"band={None if rep_b is None else rep_b['band']} — the D49(a) carve-out never "
                f"fired")

        # ... and the sibling spelling: a document whose LABEL says cost0 while it carries real
        # per-decision costs is still a fee-tier read (the carve-out gates on the whole
        # (tier_bp, fee_bps, slip_bps) triple, not on the label alone).
        incoherent = []
        for name, rule, sd in (("subj", "argmax", 0), ("null", "random-matched", 1)):
            d = _tiered_trace_doc(col_symbols, row_index, ts, chosen, book, prices, horizon,
                                  f"fee-{name}-mislabeled", rule, sd, 21)
            d["provenance"]["tier_bp"] = 0        # label lies; fee_bps=16/slip_bps=5 remain
            p = os.path.join(work, f"mislabeled_{name}.json")
            with open(p, "w") as fh:
                json.dump(d, fh, default=float)
            incoherent.append(p)
        rep_mis = report_mod.compare(incoherent[0], [incoherent[1]], block=21, n_boot=1000, seed=0)
        ok_mis = (rep_mis["band"] == report_mod.BAND_FEE_TIER_DIAGNOSTIC
                  and rep_mis["is_fee_tier_diagnostic"] is True
                  and rep_mis["fee_bps"] == 16.0 and rep_mis["slip_bps"] == 5.0)
        print(f"    [{'PASS' if ok_mis else 'FAIL'}] gate (fee-tier): tier_bp=0 MISLABELED over "
             f"real fee_bps=16/slip_bps=5 still reads {report_mod.BAND_FEE_TIER_DIAGNOSTIC!r} "
             f"(got band={rep_mis['band']}, fee_bps={rep_mis['fee_bps']})")
        if not ok_mis:
            failures.append(f"gate (fee-tier): a cost0-LABELED trace carrying real fee16/slip5 "
                            f"costs was graded band={rep_mis['band']}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


# =============================================================================
# D51 RE-CUT / AMENDMENT 1 gate (subject-class) — the 887 A/D subject class must never be pooled
# with the imprint primaries; the 832 contrast leg is excluded from the D47 evidence base. The
# EXCLUSION is structural (validated enum + a per-report membership stamp). The NEVER-POOLED rule
# is NOT: assert_never_pooled is implemented and tested but has no live caller, so part (e) below
# (fix pass, MAJOR 2) holds the module's CLAIM coherent with its WIRING.
# =============================================================================

def gate_subject_class_never_pooled(failures):
    base = dict(tag="t", checkpoint_label="runs/861_ref/ckpt_ref_s0", decision_rule="argmax",
               action_marginals_path=None, eval_seed=0, tier_bp=0, fee_bps=0.0, slip_bps=0.0,
               horizon=2, position_size_pct=0.1, col_symbols=["A"], row_index=[0], timestamps=[0],
               chosen_vectors=np.array([[2]]), realized_book=np.array([[1]]),
               prices=np.array([[100.0], [101.0], [102.0], [103.0]]),
               allocated_fraction=np.array([0.5]),
               universe_pin={"mechanism": "explicit-symbols", "verified": False,
                            "verified_reason": "test"},
               walk_kind=trace_mod.WALK_KIND_CHECKPOINT)

    # (a) a bogus class is refused, same discipline as walk_kind.
    refused = False
    try:
        trace_mod.build_trace_doc(subject_class="not-a-real-class", **base)
    except ValueError:
        refused = True
    print(f"    [{'PASS' if refused else 'FAIL'}] gate (subject-class): an unrecognized "
         f"subject_class is REFUSED by build_trace_doc")
    if not refused:
        failures.append("gate (subject-class): a bogus subject_class was accepted")

    # (b) each recognized class round-trips through provenance and report.compare's stamp.
    docs = {}
    for cls in trace_mod.SUBJECT_CLASSES:
        docs[cls] = trace_mod.build_trace_doc(subject_class=cls, **base)
    ok_rt = all(docs[cls]["provenance"]["subject_class"] == cls for cls in trace_mod.SUBJECT_CLASSES)
    print(f"    [{'PASS' if ok_rt else 'FAIL'}] gate (subject-class): every recognized class "
         f"round-trips into provenance.subject_class")
    if not ok_rt:
        failures.append("gate (subject-class): a class did not round-trip")

    # (c) D47 evidence-base membership: primaries and 887 A/D are IN; the 832 contrast and the
    # search-secondary capacity-ceiling leg are OUT.
    for cls in (trace_mod.SUBJECT_CLASS_PRIMARY, trace_mod.SUBJECT_CLASS_887_A,
               trace_mod.SUBJECT_CLASS_887_D):
        ok_in = report_mod.in_d47_evidence_base(cls) is True
        print(f"    [{'PASS' if ok_in else 'FAIL'}] gate (subject-class): {cls} IS in the D47 "
             f"evidence base")
        if not ok_in:
            failures.append(f"gate (subject-class): {cls} should be in the D47 evidence base")
    for cls in (trace_mod.SUBJECT_CLASS_832_CONTRAST, trace_mod.SUBJECT_CLASS_SEARCH_SECONDARY):
        ok_out = report_mod.in_d47_evidence_base(cls) is False
        print(f"    [{'PASS' if ok_out else 'FAIL'}] gate (subject-class): {cls} is EXCLUDED from "
             f"the D47 evidence base")
        if not ok_out:
            failures.append(f"gate (subject-class): {cls} should be excluded from D47 evidence")

    # (d) assert_never_pooled REFUSES combining two D47-evidence classes (the primary + an 887
    # arm), but does NOT refuse a primary alongside the excluded 832 contrast (832 simply never
    # counts toward the evidence base — it is not "pooled" in the sense this guard forbids).
    reports_conflict = [{"subject_class": trace_mod.SUBJECT_CLASS_PRIMARY},
                        {"subject_class": trace_mod.SUBJECT_CLASS_887_A}]
    raised = False
    try:
        report_mod.assert_never_pooled(reports_conflict)
    except ValueError:
        raised = True
    print(f"    [{'PASS' if raised else 'FAIL'}] gate (subject-class): assert_never_pooled "
         f"REFUSES pooling the primary class with an 887 arm")
    if not raised:
        failures.append("gate (subject-class): primary + 887_A was not refused as pooled")

    reports_887_ad = [{"subject_class": trace_mod.SUBJECT_CLASS_887_A},
                      {"subject_class": trace_mod.SUBJECT_CLASS_887_D}]
    raised2 = False
    try:
        report_mod.assert_never_pooled(reports_887_ad)
    except ValueError:
        raised2 = True
    print(f"    [{'PASS' if raised2 else 'FAIL'}] gate (subject-class): assert_never_pooled "
         f"REFUSES pooling the 887 A-arm with the 887 D-arm")
    if not raised2:
        failures.append("gate (subject-class): 887_A + 887_D was not refused as pooled")

    reports_832_ok = [{"subject_class": trace_mod.SUBJECT_CLASS_PRIMARY},
                      {"subject_class": trace_mod.SUBJECT_CLASS_832_CONTRAST}]
    no_raise = True
    try:
        report_mod.assert_never_pooled(reports_832_ok)
    except ValueError:
        no_raise = False
    print(f"    [{'PASS' if no_raise else 'FAIL'}] gate (subject-class): a primary alongside the "
         f"EXCLUDED 832 contrast does not trip the never-pooled guard (832 is excluded, not "
         f"'pooled')")
    if not no_raise:
        failures.append("gate (subject-class): primary + 832 incorrectly raised as pooled")

    # ----------------------------------------------------------------------------------------
    # (e) FIX PASS (MAJOR 2) — THE CLAIM MUST MATCH THE WIRING. `assert_never_pooled` has NO live
    # caller anywhere in report.py (there is no multi-report aggregator: `main` takes one
    # --subject, `compare` grades one subject trace), so the module must not claim pooling is
    # structurally impossible — it must ship the honest OPEN ITEM instead. This gate reads
    # report.py's own AST for live calls and holds the two states coherent BOTH ways: with no
    # live call, the emitted artifact must carry disclosures.NEVER_POOLED_OPEN_ITEM; once a real
    # aggregator wires the guard in, this gate goes RED and forces the wording to be upgraded to
    # the structural claim it can then support. RED against the pre-fix code: report.py claimed
    # "enforced STRUCTURALLY here, not merely by convention" over a guard nothing called, and no
    # emitted report said otherwise.
    # ----------------------------------------------------------------------------------------
    import ast
    with open(report_mod.__file__) as fh:
        tree = ast.parse(fh.read())
    live_calls = [n for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and
                  ((isinstance(n.func, ast.Name) and n.func.id == "assert_never_pooled") or
                   (isinstance(n.func, ast.Attribute) and n.func.attr == "assert_never_pooled"))]

    work = tempfile.mkdtemp(prefix="t892_pooling_")
    try:
        paths = []
        for name, level, sd in (("subj", 0.02, 31), ("null", 0.0, 32)):
            p = os.path.join(work, f"{name}.json")
            with open(p, "w") as fh:
                json.dump(_minimal_trace_doc(200, 4, level, seed=sd), fh)
            paths.append(p)
        rep = report_mod.compare(paths[0], [paths[1]], block=21, n_boot=300, seed=0)

        if live_calls:
            ok_e = (rep.get("never_pooled_enforcement") is not None
                    and "OPEN ITEM" not in str(rep.get("never_pooled_enforcement")))
            print(f"    [{'PASS' if ok_e else 'FAIL'}] gate (subject-class): "
                 f"assert_never_pooled now HAS {len(live_calls)} live call site(s) — the report's "
                 f"never_pooled_enforcement must be upgraded from OPEN ITEM to the structural "
                 f"claim it can now support")
            if not ok_e:
                failures.append(
                    f"gate (subject-class): assert_never_pooled is now called from "
                    f"{len(live_calls)} site(s) in report.py, but the artifact still reports "
                    f"never_pooled_enforcement={rep.get('never_pooled_enforcement')!r} — update "
                    f"the disclosure to match the wiring")
        else:
            ok_e = (str(rep.get("never_pooled_enforcement", "")).startswith("OPEN ITEM")
                    and rep.get("never_pooled_open_item") ==
                    disclosures_mod.NEVER_POOLED_OPEN_ITEM)
            print(f"    [{'PASS' if ok_e else 'FAIL'}] gate (subject-class): with NO live caller "
                 f"of assert_never_pooled in report.py, every emitted report declares the "
                 f"never-pooled rule an OPEN ITEM (not a structural guarantee)")
            if not ok_e:
                failures.append(
                    f"gate (subject-class): assert_never_pooled has no live caller, but the "
                    f"artifact reports never_pooled_enforcement="
                    f"{rep.get('never_pooled_enforcement')!r} / open_item carried="
                    f"{rep.get('never_pooled_open_item') == disclosures_mod.NEVER_POOLED_OPEN_ITEM}"
                    f" — a 'structurally impossible' claim over a guard nothing calls")

        # ... and the residual gap the guard cannot close is NAMED, not left implicit: an
        # untagged subject is stamped as untagged, kept out of the evidence base, and the
        # disclosure says in words that a MISTAGGED primary would not trip the guard either.
        ok_untagged = (rep["subject_class"] is None and rep["subject_class_is_untagged"] is True
                       and rep["in_d47_evidence_base"] is False
                       and "MISTAGGED" in disclosures_mod.NEVER_POOLED_OPEN_ITEM)
        print(f"    [{'PASS' if ok_untagged else 'FAIL'}] gate (subject-class): an UNTAGGED "
             f"subject is stamped untagged + outside the D47 evidence base, and the open-item "
             f"disclosure names the mistagged-primary gap explicitly")
        if not ok_untagged:
            failures.append("gate (subject-class): the untagged/mistagged-subject gap is not "
                            "stamped on the report or not named in the disclosure")
    finally:
        shutil.rmtree(work, ignore_errors=True)


# =============================================================================
# D51 RE-CUT / AMENDMENT 1 gate (co-primary) — entries/exits vs maintenance re-trims, banded
# SEPARATELY; plus net_exposure / market_return present and sane.
# =============================================================================

def gate_co_primary_split_and_market_return(failures):
    # book side sequence (asset 0 only): flat->long->long->long->long->short. DELIBERATELY
    # ASYMMETRIC (1 entry/exit-shaped decision at index 0, 3 re-trims at 1-3, 1 exit+re-entry at
    # 4) so n_entry_exit != n_re_trim — a test with EQUAL counts on both sides cannot catch a
    # mutation that swaps the two masks (both counts would still "match" by accident). position_
    # delta = [1, 0, 0, 0, -2]: entry/exit (!=0) = decisions {0, 4}; re-trim (==0) = {1, 2, 3}.
    n_decisions, n_assets, horizon = 5, 1, 1
    n_rows = n_decisions + horizon + 1
    prices = np.array([[100.0], [101.0], [99.0], [102.0], [97.0], [98.0], [95.0]])[:n_rows]
    row_index = np.arange(n_decisions)
    ts = row_index * 60
    chosen = np.full((n_decisions, n_assets), 2, dtype=np.int32)     # non-HOLD throughout
    book = np.array([[1], [1], [1], [1], [2]], dtype=np.int32)       # long x4, then short
    doc = _trace_doc_from_arrays(["A"], row_index, ts, chosen, book, prices, horizon,
                                 "coprimary", "argmax", 0)
    flat = report_mod._non_hold_forward_returns(doc)
    entry_exit_ts = set(flat["timestamps_entry_exit"].tolist())
    re_trim_ts = set(flat["timestamps_re_trim"].tolist())
    expected_entry_exit_ts = {int(ts[0]), int(ts[4])}
    expected_re_trim_ts = {int(ts[1]), int(ts[2]), int(ts[3])}
    ok = (flat["n_entry_exit"] == 2 and flat["n_re_trim"] == 3
         and entry_exit_ts == expected_entry_exit_ts and re_trim_ts == expected_re_trim_ts
         and "market_return" in doc["decisions"] and "net_exposure" in doc["decisions"]
         and doc["schema"] == trace_mod.TRACE_SCHEMA)
    print(f"    [{'PASS' if ok else 'FAIL'}] gate (co-primary): position_delta=[1,0,0,0,-2] "
         f"splits by DECISION IDENTITY into entry_exit@{sorted(entry_exit_ts)} "
         f"(expected {sorted(expected_entry_exit_ts)}) / re_trim@{sorted(re_trim_ts)} "
         f"(expected {sorted(expected_re_trim_ts)})")
    if not ok:
        failures.append(f"gate (co-primary): entry/exit vs re-trim split wrong (by identity, "
                        f"not just count): entry_exit@{sorted(entry_exit_ts)} "
                        f"re_trim@{sorted(re_trim_ts)}")
    # net_exposure: fully long at decisions 0-3 (mean side=+1), fully short at decision 4 (-1).
    ok2 = doc["decisions"]["net_exposure"][:4] == [1.0, 1.0, 1.0, 1.0] and \
        doc["decisions"]["net_exposure"][4] == -1.0
    print(f"    [{'PASS' if ok2 else 'FAIL'}] gate (co-primary): net_exposure tracks realized "
         f"book side ({doc['decisions']['net_exposure']})")
    if not ok2:
        failures.append(f"gate (co-primary): net_exposure wrong: "
                        f"{doc['decisions']['net_exposure']}")
    # market_return is the UNDIRECTED counterpart of forward_return on the SAME mask.
    fwd = doc["decisions"]["forward_return"]
    mkt = doc["decisions"]["market_return"]
    ok3 = all((fwd[k][0] is None) == (mkt[k][0] is None) for k in range(n_decisions))
    print(f"    [{'PASS' if ok3 else 'FAIL'}] gate (co-primary): market_return shares "
         f"forward_return's exact NaN mask (undirected vs signed, same held cells)")
    if not ok3:
        failures.append(f"gate (co-primary): market_return/forward_return masks disagree: "
                        f"{fwd} / {mkt}")
    # for a held, defined cell, market_return is the SIGNED forward_return with direction removed
    # (recall forward_return = direction*raw - cost; at cost0, market_return should equal
    # direction*forward_return exactly).
    defined_idx = [k for k in range(n_decisions) if fwd[k][0] is not None]
    direction = doc["decisions"]["direction"]
    ok4 = all(abs(mkt[k][0] - direction[k][0] * fwd[k][0]) < 1e-9 for k in defined_idx)
    print(f"    [{'PASS' if ok4 else 'FAIL'}] gate (co-primary): market_return = direction * "
         f"forward_return at cost0 (undirected vs signed, verified numerically)")
    if not ok4:
        failures.append("gate (co-primary): market_return does not match direction*forward_return")


def main():
    failures = []
    before = (os.path.getsize(FIXTURE_DB), os.path.getmtime(FIXTURE_DB))
    # finding 4: nothing this suite runs may create an artifact in the TRACKED experiment
    # directory. `_scratch/` and `__pycache__/` are the two gitignored exceptions.
    _ALLOWED_NEW = {"_scratch", "__pycache__"}
    exp_before = set(os.listdir(EXP_DIR))
    work = tempfile.mkdtemp(prefix="t892_action_ev_")
    try:
        print("[892] gate (i) — default OFF byte-identical to the committed HEAD baseline ...")
        gate_i_default_off_parity(failures, work)

        print("[892] gate (ii) — matched-by-construction (random_matched_actions fidelity) ...")
        gate_ii_matched_by_construction(failures)

        print("[892] gate (iii) — path independence (seeded, deterministic) ...")
        gate_iii_path_independence(failures)

        print("[892] gate (loader) — load_action_marginals fails loud on every malformed "
             "sidecar, accepts a valid one ...")
        gate_loader_fails_loud(failures)

        print("[892] gate (v) — the null operator's real, undocumented-nowhere-stubbed CLI ...")
        result = gate_v_real_cli_not_stubbed(failures, work)
        subject_path, null_path = (result[0], result[2]) if result else (None, None)
        gate_v_report_plumbing(failures, subject_path, null_path, work)

        print("[892] gate (iv) — fixture smoke recovers the planted sign; falsification "
             "reads non-EDGE ...")
        gate_iv_fixture_smoke_recovers_sign(failures)

        print("[892] gate (boot) — BLOCKER 1: thin samples report INSUFFICIENT-N, never a band "
             "or a zero-width CI (exercised THROUGH ext_windows) ...")
        gate_boot_thin_n_no_fake_ci(failures)

        print("[892] gate (fail-window) — finding 6: an instrument fail suppresses every "
             "per-regime band ...")
        gate_instrument_fail_suppresses_windows(failures)

        print("[892] gate (budget) — BLOCKER 2: the budget check reads the REALIZED allocation "
             "and is mode-aware (real --vector-weight-rebalance CLI run) ...")
        gate_budget_mode_aware(failures, work)

        print("[892] gate (pin) — finding 5: a real --checkpoint walk embeds 886's "
             "mechanism-typed record; a smoke walk says so truthfully ...")
        gate_universe_pin_record(failures, work)

        print("[892] gate (disclose) — the pre-registration deviation is inline in both artifact "
             "kinds; SPEC.md:46's book width is computed ...")
        gate_disclosure_and_book_width(failures, subject_path, null_path, work)

        print("[892] gate (abbrev) — finding 8: no new flag steals an abbreviation that was "
             "unambiguous at HEAD (class-closing, all HEAD options) ...")
        gate_abbrev_head_options_still_resolve(failures)

        print("[892] gate (op-coherence) — finding 9: the null-operator flags REFUSE under an "
             "operator that cannot honour them ...")
        gate_op_coherence_refuses(failures)

        print("[892] gate (schema) — CYCLE-3 BLOCKER 1: report.py hard-reads every input trace "
             "(no soft budget default, schema validated) ...")
        gate_report_hard_reads_its_inputs(failures)

        print("[892] gate (power) — CYCLE-3 finding 2: the banding bar is CALIBRATED against "
             "measured true-null coverage; below it, LOW-POWER and no band ...")
        gate_calibrated_power_bar(failures)

        print("[892] gate (zero-width) — CYCLE-3 finding 3: a zero-width CI never carries a "
             "band, at any n ...")
        gate_zero_width_ci_never_bands(failures)

        print("[892] gate (race) — CYCLE-3 finding 4: the --smoke default checkpoint dir is "
             "private per invocation (concurrent runs do not collide) ...")
        gate_smoke_ckpt_dir_no_race(failures, work)

        print("[892] gate (nan-budget) — CYCLE-3 finding 5: a non-finite allocation is a budget "
             "violation ...")
        gate_nan_allocation_is_a_violation(failures)

        print("[892] gate (walk-kind) — CYCLE-3 finding 6: walk_kind is a validated ENUM, not an "
             "allow-list ...")
        gate_walk_kind_is_a_validated_enum(failures)

        print("[892] gate (ci-args) — CYCLE-3 finding 7: block/n_boot are validated with a named "
             "message ...")
        gate_bootstrap_arg_validation(failures)

        print("[892] gate (atomic) — CYCLE-3 minor: every artifact write is tmp + os.replace ...")
        gate_atomic_artifact_writes(failures, subject_path, null_path, work)

        print("[892] gate (leg) — D51/[A1]: the OOS binding leg is structurally validated, not "
             "a comment ...")
        gate_leg_binding_is_structural(failures)

        print("[892] gate (fee-tier) — D49(a)/[A1]: fee16/slip5 is a reported diagnostic, never "
             "a band condition, enforced in code ...")
        gate_fee_tier_is_diagnostic_only(failures)

        print("[892] gate (subject-class) — [A1]: the 887 A/D class is never pooled with the "
             "primaries; the 832 contrast is excluded from the D47 evidence base ...")
        gate_subject_class_never_pooled(failures)

        print("[892] gate (co-primary) — [A1]: entries/exits vs maintenance re-trims split "
             "correctly; net_exposure/market_return computed ...")
        gate_co_primary_split_and_market_return(failures)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    new_entries = (set(os.listdir(EXP_DIR)) - exp_before) - _ALLOWED_NEW
    if new_entries:
        failures.append(f"the suite created {sorted(new_entries)} inside the TRACKED directory "
                        f"experiments/892_action_ev/ — every default write must land in the "
                        f"gitignored _scratch/ (finding 4, the run-34 pattern)")
    else:
        print(f"[892] tracked directory clean: no new entries in experiments/892_action_ev/ "
              f"(only {sorted(_ALLOWED_NEW)} permitted)")

    after = (os.path.getsize(FIXTURE_DB), os.path.getmtime(FIXTURE_DB))
    if before != after:
        failures.append(f"fixtures/trading_fixture.db changed size/mtime during this run "
                        f"({before} -> {after}) — INVARIANT VIOLATION, never regenerate it.")
    else:
        print(f"[892] fixture unchanged: size={before[0]} mtime={before[1]}")

    print("=" * 84)
    if failures:
        print(f"[892] FAIL — {len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[892] PASS — gates (i), (ii), (iii), (loader), (v), (iv), (boot), (fail-window), "
          "(budget), (pin), (disclose), (abbrev), (op-coherence), (schema), (power), "
          "(zero-width), (race), (nan-budget), (walk-kind), (ci-args), (atomic), (leg), "
          "(fee-tier), (subject-class), (co-primary) green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
