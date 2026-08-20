#!/usr/bin/env python3
"""TASK_892 / action-EV audit — scope 2: the per-decision TRACE instrument.

887 proved the hold collapse is adaptive; it could not tell whether the policy's non-HOLD
actions carry no edge (a correct refusal) or edge the policy fails to FIND (a search/selection
failure). ``experiments/892_action_ev/SPEC.md`` pre-registers the distinguishing measurement:
compare the realized forward return after a policy's non-HOLD decisions against a
matched-activity NULL that sees the same states at the same costs. This module is the
instrument that RECORDS one walk (subject or null) as a per-decision, per-asset trace; the
comparison itself is ``report.py``.

ZERO canon / training-path code diffs, same house style as 849/leaf_probe.py: every piece of
the model/env/search/panel machinery is IMPORTED from ``experiments/835_stage4_eval/
eval_vector.py`` (which owns edits to that module) — this file only OBSERVES a walk that
eval_vector's own ``rollout()`` already knows how to produce, then computes the forward-return
attribution eval_vector has no reason to know about.

READ FIRST — PRE-REGISTRATION DEVIATION. The forward return this module delivers is NOT the
quantity SPEC.md pre-registers ("the env's own accounting; PV conservation exact per walk"). It
is a counterfactual hold-through-H, ENTRY-COST-ONLY event-study return (contract below). The
deviation, its exact cost0 neutrality, and the concrete direction of its bias at fee16/slip5 are
recorded INLINE in every emitted trace (``READ_THIS_FIRST`` / ``pre_registration_deviations``,
one definition in ``disclosures.py``). SPEC.md is a FROZEN pre-registration and is deliberately
NOT amended to match this build — the planner rules on the deviation.

NUMERIC CONTRACT (read before citing a trace's forward_return numbers):
  * ``direction[k,a]`` is the book side AFTER decision k (+1 long / -1 short / 0 flat), read
    from ``rollout()``'s own ``collect["realized_book"]`` (env accounting, not re-derived).
  * ``forward_return[k,a]`` is a COUNTERFACTUAL "hold the direction opened at decision k for
    exactly ``--horizon`` further decisions" price return: ``direction * (price[k+H]/price[k]
    - 1) - entry_cost``, entry_cost = (fee_bps+slip_bps)/1e4 charged ONCE (the decision's own
    entry; whatever future decision eventually closes the position bears its own exit cost —
    that P&L is attributed to THAT decision, not this one). This is a standard event-study
    forward return, not a replay of the walk's own compounding trajectory: it deliberately does
    NOT stop early if a later decision flips or flattens the position first, because the
    question this audit asks is "was THIS decision's direction correct", not "what did the
    walk's path-dependent PV do next". NaN where k+H falls outside the walk or no position was
    opened (direction == 0 — nothing to attribute).
  * ``budget_conservation_violations`` counts decisions at which the walk's OWN REALIZED
    allocation exceeded 100% of portfolio value. The input is ``1 - cash/PV`` recorded per step
    by ``eval_vector.rollout``'s opt-in ``"allocated_fraction"`` collector, off the same env
    state that produced the PV; since ``compute_portfolio_value`` is ``cash + Σ position_values
    + reserved``, that fraction IS Σ(committed capital)/PV under WHATEVER allocation semantics
    the config selected, and a violation is exactly "cash went negative". It is NOT a bar-by-bar
    reconciliation of the compounded PV path (a documented scope limitation, not an oversight;
    the CONSERVATIVE half of SPEC.md's "PV violation" instrument-fail check).
    FIX PASS, BLOCKER 2 — this check used to be ``open_slot_count * position_size_pct``, a field
    that is DEAD CODE under the ``--vector-weight-rebalance`` semantics SPEC.md MANDATES for all
    three primary checkpoints (``jax_trading_env.py`` ~L1806 sizes from ``w_unit *
    portfolio_value``; ``experiments/882_instrument_reconciliation/
    test_instrument_agreement.py:19`` records it and proves an 8x sweep inert). On SPEC.md's own
    ~26L/31S book that computed 57 * 0.1 = 5.7 >> 1.0 and reported a violation on essentially
    every decision, which would have made the real leg read INSTRUMENT FAIL on this check alone
    and swallowed the EDGE/ANTI-EDGE/NULL verdict. ``budget_conservation_violations`` is now
    MODE-AWARE and REFUSES to apply the dead field under weight semantics.

NON-SMOKE USAGE (undocumented before this pass): a real ``--checkpoint`` walk needs ``--n-assets
N`` passed EXPLICITLY, or TASK_888R's refusing resolver correctly rejects the load with a
"FOREIGN SHAPES" error (the default n_assets does not match the checkpoint's trained universe
size). That refusal is right — this note only makes it discoverable. Real walks also need a
universe: ``--symbols``/``--panel-frame-from-checkpoint`` (or the ``--allow-unpinned-universe``
hatch), enforced by ``ev.enforce_universe_pin`` before any panel is built.

SCRATCH: with no ``--out``, ``--smoke`` writes its trace (and its throwaway checkpoint) under
``experiments/892_action_ev/_scratch/``, which is GITIGNORED. This tool never writes into the
tracked experiment directory by default (fix pass, finding 4 — the run-34 "a write nobody thinks
of as an edit" pattern); ``--smoke-ckpt-dir`` redirects the checkpoint and is honoured (finding 7).
CYCLE-3 finding 4: the default checkpoint directory is now UNIQUE PER INVOCATION
(``mkdtemp`` under the scratch dir, removed as soon as the save->load roundtrip is done). It used
to be the ONE literal path ``_scratch/smoke_ckpt`` for every ``--smoke`` run, so two concurrent or
back-to-back runs raced on the same orbax directory — measured: 2 of 3 concurrent ``--smoke`` runs
died with ``FileNotFoundError``/``ocdbt read failure``, and the acceptance suite's own two
back-to-back ``trace.py --smoke`` subprocesses were nondeterministic (a run FAILED, an immediate
re-run PASSED). A suite whose green depends on disk timing is not evidence.

ATOMIC WRITES (cycle-3 minor): every artifact is written to ``<path>.tmp`` and ``os.replace``d
into place, so a kill mid-``json.dump`` can never leave a truncated JSON where a complete trace
is expected. Every refusal in this module still happens strictly BEFORE the write.

Fixture smoke: ``python trace.py --smoke`` — fresh-init UNTRAINED model on
``fixtures/trading_fixture.db`` (identical universe/window to eval_vector.py --smoke), full
trace schema emitted, budget check run. ``--smoke --decision-rule random-matched
--marginals-sidecar <sidecar>`` exercises the null path end-to-end through the REAL
eval_vector.py CLI machinery (no stubs). Untrained numbers are near-noise by construction —
plumbing proof, not a verdict.

FACTORY / SCHEDULED split (835 house style): this module + the fixture smoke are the factory
deliverable. The real leg (3 ref checkpoints x 2 cost tiers + ~60 null walks) is a SCHEDULED
run, never inside the factory loop — see SPEC.md "Cost".
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)
EV_PY = os.path.join(_REPO, "experiments", "835_stage4_eval", "eval_vector.py")

# TASK_892 fix pass (finding 4 / finding 7): NOTHING this tool writes by default lands in the
# TRACKED experiment directory. `experiments/892_action_ev/` is tracked and its .gitignore
# covered only __pycache__/*.pyc/smoke_ckpt*/, so the old `--smoke`-with-no-`--out` default
# (EXP_DIR/trace_{tag}_{rule}.json) was a write nobody thinks of as an edit — the run-34
# pattern. Every default scratch path now goes under this ONE gitignored directory.
SCRATCH_DIR = os.path.join(EXP_DIR, "_scratch")

import disclosures                                            # noqa: E402  (path set above)

TRACE_SCHEMA = "892/action-ev-trace/v3"          # v3 (D51 re-cut, SPEC Amendment 1 [A1]):
# + decisions.net_exposure (n_decisions,), + decisions.market_return (n_decisions, n_assets).
# A v2 trace lacks both; report.py's schema check already refuses any non-current schema, so an
# old trace is refused loudly rather than silently read with the new fields missing.
DEFAULT_HORIZON = 21          # SPEC.md: H=21 forward decisions

WALK_KIND_SMOKE = "fixture-smoke-fresh-init"
WALK_KIND_CHECKPOINT = "checkpoint"
# CYCLE-3 finding 6: the CLOSED set of walk kinds. ``build_trace_doc`` validates against THIS
# tuple up front instead of pattern-matching the two known values positively — an allow-list of
# positive raises lets a THIRD value through, and the third value inherits the smoke branch's
# note, which then describes a run that never happened.
WALK_KINDS = (WALK_KIND_SMOKE, WALK_KIND_CHECKPOINT)

# [A1] SPEC.md Amendment 1's subject classes. A CLOSED vocabulary, the same discipline as
# WALK_KINDS: "never pooled" is enforced downstream (report.py's D47_EVIDENCE_CLASSES /
# assert_never_pooled) by comparing this TAGGED, validated field, not by trusting a caller to
# keep files in separate directories.
SUBJECT_CLASS_PRIMARY = "861_ref_primary"                 # 3 clean-semantics ref checkpoints
SUBJECT_CLASS_887_A = "887_A_collapsed"                   # 887 A-arm ep300 (collapsed)
SUBJECT_CLASS_887_D = "887_D_uncollapsed"                 # 887 D-arm ep300 (un-collapsed)
SUBJECT_CLASS_SEARCH_SECONDARY = "861_ref_search_secondary"   # +8.02-class search operator leg
SUBJECT_CLASS_832_CONTRAST = "832_e5_contrast"            # EXCLUDED from the D47 evidence base
SUBJECT_CLASSES = (SUBJECT_CLASS_PRIMARY, SUBJECT_CLASS_887_A, SUBJECT_CLASS_887_D,
                   SUBJECT_CLASS_SEARCH_SECONDARY, SUBJECT_CLASS_832_CONTRAST)


def write_json_atomic(path, doc, **dump_kwargs):
    """Write ``doc`` to ``path`` via ``<path>.tmp`` + ``os.replace`` (cycle-3 minor: non-atomic
    artifact writes). A kill during ``json.dump`` then leaves the temp file, never a truncated
    JSON at the path a reader trusts. THE ONE definition — ``report.py`` and
    ``build_action_marginals.py`` import it from here rather than each growing their own."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, **dump_kwargs)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def _load_eval_vector():
    """Import eval_vector.py by path (849's pattern) — never edit it, never re-derive it."""
    spec = importlib.util.spec_from_file_location("eval_vector835_892", EV_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compute_forward_returns(chosen_vectors, realized_book, prices, row_index, horizon,
                             fee_bps, slip_bps):
    """The trace instrument's core numeric contract (module docstring). Pure numpy — no model/
    env/search dependency — so the SAME function grades a real CLI-driven walk (``run_walk``)
    and a hand-built synthetic fixture trace (acceptance gate iv) identically.

    chosen_vectors: (n_decisions, n_assets) int action codes, the ATTEMPT ([0]Sell [1]Hold
        [2]Buy [3]LimitSell [4]LimitBuy; HOLD==1).
    realized_book: (n_decisions, n_assets) int, POST-decision book side (0 flat/1 long/2
        short) — the rollout() OUTCOME.
    prices: (n_rows, n_assets) float64 panel prices, the array the walk actually read.
    row_index: (n_decisions,) int, panel row for each decision (prices[row_index[k]] is
        decision k's mark; decisions are one panel row apart, so row_index[k]+horizon is
        exactly `horizon` decisions forward).
    horizon: H (SPEC: 21).
    fee_bps, slip_bps: this tier's one-way entry cost in basis points.

    Returns a dict of (n_decisions, n_assets) arrays: direction, position_delta, non_hold,
    forward_return (NaN where undefined — see module docstring), and (SPEC Amendment 1 [A1])
    market_return — the UNDIRECTED forward H-return of the held asset over the same horizon and
    NaN mask as forward_return (``direction * raw`` vs plain ``raw``: the two share every masking
    decision, differing only in sign/cost, by construction) — so a directional artifact (a long
    book realized during a broad rally) is separable from timing skill: forward_return earns
    credit only if the SIGNED bet was right, market_return reports what the position's own
    underlying asset did regardless of side."""
    chosen_vectors = np.asarray(chosen_vectors)
    realized_book = np.asarray(realized_book, dtype=np.int64)
    prices = np.asarray(prices, dtype=np.float64)
    row_index = np.asarray(row_index, dtype=np.int64)
    n_decisions, n_assets = chosen_vectors.shape
    if realized_book.shape != (n_decisions, n_assets):
        raise ValueError(f"realized_book shape {realized_book.shape} != chosen_vectors shape "
                         f"{chosen_vectors.shape}")
    if row_index.shape != (n_decisions,):
        raise ValueError(f"row_index shape {row_index.shape} != (n_decisions,)={(n_decisions,)}")

    side_of_code = np.array([0.0, 1.0, -1.0])          # book code -> {flat, long, short}
    if realized_book.size and (realized_book.max() > 2 or realized_book.min() < 0):
        raise ValueError("realized_book must contain only {0,1,2} (flat/long/short)")
    side = side_of_code[realized_book]                  # (n_decisions, n_assets)
    prev_side = np.zeros_like(side)
    prev_side[1:] = side[:-1]                           # v1 book = flat/HOLD start (DEC-3)
    position_delta = side - prev_side

    HOLD = 1
    non_hold = chosen_vectors != HOLD

    cost = (float(fee_bps) + float(slip_bps)) / 1e4
    n_rows = prices.shape[0]
    forward_return = np.full((n_decisions, n_assets), np.nan, dtype=np.float64)
    market_return = np.full((n_decisions, n_assets), np.nan, dtype=np.float64)
    for k in range(n_decisions):
        r0 = int(row_index[k])
        rH = r0 + int(horizon)
        if rH >= n_rows or r0 >= n_rows:
            continue
        d = side[k]
        if not np.any(d != 0):
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = prices[rH] / np.maximum(prices[r0], 1e-12) - 1.0
        fr = np.where(d != 0, d * raw - cost, np.nan)
        forward_return[k] = np.where(np.isfinite(fr), fr, np.nan)
        # [A1] market_return: the SAME held-cell mask as forward_return, UNDIRECTED and
        # cost-free — "what did this asset itself do", not "was the signed bet right".
        mr = np.where(d != 0, raw, np.nan)
        market_return[k] = np.where(np.isfinite(mr), mr, np.nan)
    # [A1] net_exposure: per-decision directional tilt of the realized book, mean(side) over
    # assets in [-1, 1] (fully short .. fully long; 0 = flat or long/short-balanced). Computed
    # from `side`, not from `chosen_vectors` — the REALIZED book the walk actually carried, not
    # the attempted action — so a rejected/no-op attempt cannot masquerade as exposure.
    net_exposure = np.mean(side, axis=1)
    return {
        "direction": side, "position_delta": position_delta,
        "non_hold": non_hold, "forward_return": forward_return,
        "market_return": market_return, "net_exposure": net_exposure,
    }


WEIGHT_SEMANTICS = "vector-weight-rebalance"          # TASK_840 weight env (SPEC.md's mandate)
PCT_SEMANTICS = "position-size-pct"                   # canon / 839 budget path


def budget_conservation_violations(realized_book, *, allocated_fraction=None,
                                   allocation_semantics=WEIGHT_SEMANTICS,
                                   position_size_pct=None, tol=1e-6):
    """CONSERVATIVE budget check (module docstring: not a bar-by-bar PV reconciliation) — the
    book may never have allocated more than 100% of capital. Returns
    ``(n_violations: int, allocated_fraction: (n_decisions,) float array)``.

    FIX PASS, BLOCKER 2 — THIS CHECK USED TO BE COMPUTED FROM A DEAD FIELD. It multiplied the
    open-slot COUNT by ``position_size_pct``, but SPEC.md's Subjects section mandates
    ``--vector-weight-rebalance`` for all three primary checkpoints, and under weight semantics
    ``mcts_trader/env/jax_trading_env.py`` (the TASK_840 block, ~L1806) sizes every position from
    ``w_unit * portfolio_value`` (``vector_weight_unit``, default ``1/n_assets``, clipped at
    ``vector_weight_max * w_unit``) — ``position_size_pct`` is NEVER READ for sizing there. The
    repo already knew this: ``experiments/882_instrument_reconciliation/
    test_instrument_agreement.py:19`` records ``position_size_pct`` as DEAD CODE under weight
    semantics and proves an 8x sweep of it is inert. Measured pre-fix: with SPEC.md's own
    ~26L/31S book (~57 open slots) at the 0.1 default, the old form computed 57 * 0.1 = 5.7 >> 1.0
    and reported a violation on ESSENTIALLY EVERY DECISION, which ``report.py`` sums into
    ``instrument_fail_reasons`` — so the real leg was guaranteed to read INSTRUMENT FAIL on this
    check alone and swallow the EDGE/ANTI-EDGE/NULL verdict entirely.

    THE FIX — measure the REALIZED allocation, and be MODE-AWARE so a dead field can never be
    silently applied:
      * ``allocated_fraction`` (preferred, and what ``run_walk`` always supplies) is the walk's
        OWN per-decision ``1 - cash/PV``, recorded by ``eval_vector.rollout``'s opt-in
        ``"allocated_fraction"`` collector off the same env state that produced the PV. Since
        ``compute_portfolio_value`` is ``cash + Σ position_values + reserved``, this fraction IS
        Σ(committed capital)/PV under WHATEVER semantics the config selected — nothing is
        re-derived, and a violation is exactly "committed capital exceeded portfolio value"
        (equivalently: cash went negative), the env's own conservation identity.
      * with no realized allocation and ``PCT_SEMANTICS``, the legacy slot-count * pct form is
        still available (it is the correct allocation model THERE).
      * with no realized allocation under ``WEIGHT_SEMANTICS`` -> RAISE. There is no honest
        substitute, and silently falling back to the dead field is the defect this fix removes.

    CYCLE-3 finding 5 — NON-FINITE ALLOCATIONS COUNT AS VIOLATIONS. The check used to be the bare
    ``allocated > 1.0 + tol``, and ``NaN > x`` is ``False`` in IEEE/numpy, so a NaN allocation
    (measured: ``[0.5, nan, 0.9]`` -> ``0`` violations) was silently NOT a violation — which made
    ``disclosures.PRD-2``'s stated guarantee ("it cannot miss an overdraft/leverage defect")
    FALSE AS WRITTEN. Not proven reachable today (a bankruptcy clamps cash to 0), but a
    fail-closed guard may not rest on the unreachability of its own blind spot: an allocation that
    is not a finite number is not a bounded allocation, so it is counted."""
    realized_book = np.asarray(realized_book)
    if allocated_fraction is not None:
        allocated = np.asarray(allocated_fraction, dtype=np.float64)
        if allocated.shape != (realized_book.shape[0],):
            raise ValueError(
                f"allocated_fraction shape {allocated.shape} != (n_decisions,)="
                f"{(realized_book.shape[0],)}")
    elif allocation_semantics == PCT_SEMANTICS:
        if position_size_pct is None:
            raise ValueError("position-size-pct semantics need position_size_pct")
        allocated = ((realized_book != 0).astype(np.float64).sum(axis=1)
                     * float(position_size_pct))
    else:
        raise ValueError(
            f"budget_conservation_violations: allocation_semantics={allocation_semantics!r} "
            f"sizes positions from vector_weight_unit * PV, so position_size_pct is DEAD CODE "
            f"(experiments/882_instrument_reconciliation/test_instrument_agreement.py:19) and "
            f"cannot be used to bound the book. Pass the walk's realized allocated_fraction "
            f"(eval_vector.rollout's opt-in 'allocated_fraction' collector).")
    # finding 5: `NaN > x` is False, so the overdraft test ALONE cannot see a non-finite
    # allocation. `~np.isfinite` closes it — NaN/+-inf are counted as violations, which is the
    # fail-closed reading (an allocation that is not a finite number is not a bounded one).
    bad = (allocated > (1.0 + tol)) | ~np.isfinite(allocated)
    return int(bad.sum()), allocated


def _f(x):
    x = float(x)
    return x if np.isfinite(x) else None


def build_trace_doc(*, tag, checkpoint_label, decision_rule, action_marginals_path, eval_seed,
                    tier_bp, fee_bps, slip_bps, horizon, position_size_pct, col_symbols,
                    row_index, timestamps, chosen_vectors, realized_book, prices,
                    universe_pin=None, walk_kind=WALK_KIND_CHECKPOINT,
                    allocated_fraction=None, allocation_semantics=WEIGHT_SEMANTICS,
                    weight_unit=None, weight_max_mult=None, subject_class=None):
    """Assemble the trace JSON: schema, provenance (incl. the 886 mechanism-typed
    universe_pin record inline — TASK_892 scope 2's "first consumer"), and the per-decision
    arrays.

    ``walk_kind`` decides what the universe_pin FIELDS ARE ALLOWED TO SAY (fix pass, finding 5).
    Pre-fix, ``main()`` never called ``ev.enforce_universe_pin`` at all, so ``universe_pin`` was
    ``None`` on EVERY walk — including real ``--checkpoint`` walks — while the emitted note
    asserted "no checkpoint provenance to certify (fresh-init --smoke walk)". A note describing a
    different run than the one that produced it is worse than emitting nothing, and it would have
    covered all three real-leg ref checkpoints x 2 tiers. Now:
      * ``WALK_KIND_SMOKE`` (fresh-init fixture walk) — ``universe_pin: None`` is CORRECT and the
        note says exactly that, truthfully (eval_vector.py's own ``run_smoke`` certifies nothing
        either; there is no checkpoint provenance in existence to certify against).
      * ``WALK_KIND_CHECKPOINT`` — a finished, mechanism-typed 886 record is REQUIRED; a missing
        one RAISES rather than being explained away. The record is produced by
        ``ev.enforce_universe_pin`` + ``ev.finalize_universe_pin_record`` — the SAME functions
        ``evaluate_checkpoint`` uses, so 886's honest PROVEN / NOT-PROVEN split is inherited
        verbatim: ``verified``/``certified`` stay False (with ``verified_reason`` naming the first
        unmet conjunct) unless 886's own predicate is satisfied. Nothing here synthesizes a
        positive claim."""
    # CYCLE-3 finding 6: VALIDATE THE ENUM FIRST, then branch. The two refusals below used to be
    # the only walk_kind logic in this function — two POSITIVE-match raises, i.e. an allow-list,
    # so a THIRD value satisfied neither and slipped straight through: measured,
    # ``walk_kind="bogus-walk-kind"`` with a real ``checkpoint_label`` SUCCEEDED and emitted a
    # ``universe_pin_note`` asserting "fresh-init fixture --smoke walk", a note describing a run
    # that never happened. Today's two CLI call sites hardcode the constants, but this function is
    # documented as a hard contract and is called DIRECTLY by other code and by the tests, so the
    # contract is enforced here rather than assumed at the call sites.
    if walk_kind not in WALK_KINDS:
        raise ValueError(
            f"build_trace_doc: walk_kind={walk_kind!r} is not one of {WALK_KINDS!r}. Every "
            f"downstream claim about provenance (universe_pin, universe_pin_note) is keyed off "
            f"this value; refusing to emit a trace whose walk kind this module cannot describe.")
    # [A1] same closed-vocabulary discipline as walk_kind, for the SAME reason: an unvalidated
    # subject_class is exactly how a pooling guard downstream could be fed a string it does not
    # recognize and silently treat as "no class" (report.py's D47_EVIDENCE_CLASSES).
    if subject_class is not None and subject_class not in SUBJECT_CLASSES:
        raise ValueError(
            f"build_trace_doc: subject_class={subject_class!r} is not one of "
            f"{SUBJECT_CLASSES!r} (or None). Refusing to tag a trace with a class the "
            f"never-pooled enforcement downstream cannot recognize.")
    fr = compute_forward_returns(chosen_vectors, realized_book, prices, row_index, horizon,
                                 fee_bps, slip_bps)
    n_viol, allocated = budget_conservation_violations(
        realized_book, allocated_fraction=allocated_fraction,
        allocation_semantics=allocation_semantics, position_size_pct=position_size_pct)
    n_decisions, n_assets = np.asarray(chosen_vectors).shape
    if walk_kind == WALK_KIND_CHECKPOINT and universe_pin is None:
        raise ValueError(
            "build_trace_doc: a --checkpoint walk MUST carry the 886 mechanism-typed "
            "universe_pin record (TASK_892 scope 2's 'first consumer'); refusing to emit a "
            "trace with universe_pin: null behind a note that explains the absence away.")
    if walk_kind == WALK_KIND_SMOKE and universe_pin is not None:
        raise ValueError(
            "build_trace_doc: a fresh-init --smoke walk has no checkpoint provenance to "
            "certify; refusing to attach a universe_pin record to it.")
    pin_note = (None if walk_kind == WALK_KIND_CHECKPOINT else
                "fresh-init fixture --smoke walk: there is no checkpoint provenance in "
                "existence to certify against, so universe_pin is null (eval_vector.py's own "
                "run_smoke certifies nothing either). A real --checkpoint walk embeds the "
                "mechanism-typed record ev.enforce_universe_pin + "
                "ev.finalize_universe_pin_record returned, with 886's verified/certified split "
                "untouched.")
    return {
        "schema": TRACE_SCHEMA,
        # DISCLOSURES FIRST — a reader of a TRACE must meet the same deviation notice a reader
        # of the REPORT meets (disclosures.py is the one definition of both).
        "READ_THIS_FIRST": disclosures.READ_THIS_FIRST,
        "pre_registration_deviations": disclosures.PRE_REGISTRATION_DEVIATIONS,
        "provenance": {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "producer": "experiments/892_action_ev/trace.py",
            "tag": tag,
            "walk_kind": walk_kind,
            # [A1] the subject-class tag (None on random-matched null walks and on --smoke —
            # neither has a class of its own to pool); validated above against SUBJECT_CLASSES.
            "subject_class": subject_class,
            "checkpoint_label": checkpoint_label,
            "decision_rule": decision_rule,
            "action_marginals_sidecar": action_marginals_path,
            "eval_seed": int(eval_seed),
            "tier_bp": int(tier_bp), "fee_bps": float(fee_bps), "slip_bps": float(slip_bps),
            "horizon": int(horizon),
            # TASK_892 fix pass (BLOCKER 2): allocation semantics are now EXPLICIT, and
            # `position_size_pct` is stamped with what it actually is on this walk. Under the
            # weight path it is DEAD CODE (never read for sizing) and the budget check does NOT
            # use it — the check reads the realized 1 - cash/PV recorded by the walk itself.
            "allocation_semantics": allocation_semantics,
            "position_size_pct": float(position_size_pct),
            "position_size_pct_is_dead_code": bool(allocation_semantics == WEIGHT_SEMANTICS),
            "vector_weight_unit_effective": (None if weight_unit is None else float(weight_unit)),
            "vector_weight_max_mult": (None if weight_max_mult is None
                                       else float(weight_max_mult)),
            "vector_weight_max_effective": (None if (weight_unit is None or weight_max_mult is None)
                                            else float(weight_unit) * float(weight_max_mult)),
            "budget_check_input": ("realized allocated_fraction (1 - cash/PV) recorded per step "
                                   "by eval_vector.rollout's opt-in collector"
                                   if allocated_fraction is not None else
                                   f"open-slot count x position_size_pct "
                                   f"({PCT_SEMANTICS} semantics only)"),
            "n_assets": int(n_assets), "n_decisions": int(n_decisions),
            "col_symbols": list(col_symbols),
            # TASK_886 (D36/886): the mechanism-typed universe record, inline — TASK_892 scope
            # 2's "first consumer". Produced by ev.enforce_universe_pin +
            # ev.finalize_universe_pin_record (the same pair evaluate_checkpoint uses, so 886's
            # verified/certified split is inherited, never re-derived); REQUIRED on a
            # --checkpoint walk (build_trace_doc raises otherwise) and correctly null on a
            # fresh-init --smoke walk, whose note says so truthfully.
            "universe_pin": universe_pin,
            "universe_pin_note": pin_note,
            "budget_conservation_violations": n_viol,
            "forward_return_note": ("counterfactual hold-through-H price return, entry-cost "
                                    "only, NaN where k+horizon exceeds the walk or no position "
                                    "was opened — see trace.py module docstring for the exact "
                                    "contract, and READ_THIS_FIRST / "
                                    "pre_registration_deviations for how it DIFFERS from the "
                                    "quantity SPEC.md pre-registers"),
        },
        "decisions": {
            "row_index": [int(r) for r in row_index],
            "timestamp": [int(t) for t in timestamps],
            "chosen_vectors": np.asarray(chosen_vectors, dtype=np.int32).tolist(),
            "realized_book": np.asarray(realized_book, dtype=np.int32).tolist(),
            "direction": fr["direction"].tolist(),
            "position_delta": fr["position_delta"].tolist(),
            "non_hold": fr["non_hold"].tolist(),
            "forward_return": [[_f(x) if np.isfinite(x) else None for x in row]
                              for row in fr["forward_return"]],
            # [A1] the undirected forward market return of the held asset, same mask as
            # forward_return — separates directional-artifact (beta) from timing (see
            # compute_forward_returns docstring).
            "market_return": [[_f(x) if np.isfinite(x) else None for x in row]
                              for row in fr["market_return"]],
            # [A1] per-decision net exposure, mean(direction) over assets in [-1, 1].
            "net_exposure": [_f(x) for x in fr["net_exposure"]],
            "allocated_fraction": allocated.tolist(),
        },
    }


def run_walk(args, ev, universe_pin=None, universe_requested=None):
    """Build panel + model (fresh-init --smoke, or a real --checkpoint), run ev.rollout()
    under args.decision_rule with collect=..., and assemble the trace doc. Every model/env/
    search/panel call is ev's OWN — this function only threads args through and post-processes
    the recording (module docstring: ZERO canon diffs).

    ``universe_pin``/``universe_requested`` are ``ev.enforce_universe_pin``'s two returns for THIS
    checkpoint (``main()`` calls it BEFORE this function, mirroring eval_vector.main()'s own
    ordering: a refusal must fire before any panel is built). On a real ``--checkpoint`` walk this
    function runs the SAME post-panel/post-load assertions eval_vector.main() runs
    (``_assert_realized_panel_universe``, ``_assert_provenance_binds_loaded_params``) and then
    finishes the record through ``ev.finalize_universe_pin_record`` — the one shared definition
    (fix pass, finding 5: the record used to be ``None`` on every walk while the emitted note
    claimed the run was a smoke walk)."""
    panel = ev.build_panel(args)
    prices, env_cfg0, state_init, row_ts, col_symbols, panel_info = panel
    warm = int(env_cfg0.window_size)
    n_decisions = int(panel_info["n_decisions"])

    if args.smoke:
        model, trainer = ev.build_model_and_trainer(args, args.n_assets, init_seed=0)
        # finding 7: --smoke-ckpt-dir is inherited into --help from ev.build_argparser(), so it
        # must WORK here (it was hardcoded away, silently misleading anyone redirecting it for a
        # clean tree or parallel runs). Same precedence as eval_vector.py:4489, with the default
        # moved into the gitignored scratch dir (finding 4).
        #
        # CYCLE-3 finding 4 — THE DEFAULT MUST BE UNIQUE PER INVOCATION. The default used to be
        # the ONE literal path `_scratch/smoke_ckpt` shared by every --smoke run, and each run
        # does `save_checkpoint(..., overwrite=True)` then a reload: two runs overlapping (or one
        # tearing down an orbax tmp dir while the next writes it) collide. Measured pre-fix: 2 of
        # 3 concurrent `trace.py --smoke` runs died (`FileNotFoundError` on
        # `checkpoint_0.orbax-checkpoint-tmp/_METADATA`), and the acceptance suite's own two
        # back-to-back --smoke subprocesses were nondeterministic. An explicit --smoke-ckpt-dir is
        # still honoured VERBATIM (the harness owns that path); only the DEFAULT is made private,
        # and it is removed as soon as the save->load roundtrip is finished, so the scratch dir
        # does not grow without bound.
        os.makedirs(SCRATCH_DIR, exist_ok=True)
        explicit_ckpt_dir = bool(args.smoke_ckpt_dir)
        ckpt_dir = (os.path.abspath(args.smoke_ckpt_dir) if explicit_ckpt_dir
                    else tempfile.mkdtemp(prefix=f"smoke_ckpt_{os.getpid()}_", dir=SCRATCH_DIR))
        try:
            os.makedirs(ckpt_dir, exist_ok=True)
            trainer.save_checkpoint(ckpt_dir, overwrite=True)
            model, trainer = ev.build_model_and_trainer(args, args.n_assets, init_seed=1)
            ev.load_checkpoint_into(trainer, ckpt_dir)
        finally:
            if not explicit_ckpt_dir:
                shutil.rmtree(ckpt_dir, ignore_errors=True)
        checkpoint_label = "untrained_fresh_init(save->load roundtrip)"
        walk_kind, pin_record = WALK_KIND_SMOKE, None
    else:
        # ev's --checkpoint is action="append" (a list, possibly-multi in eval_vector.py's own
        # CLI); this instrument grades exactly one walk per invocation.
        ckpt_path = args.checkpoint[0]
        # D36's realized-panel equality assertion, in the SAME position eval_vector.main() puts
        # it (immediately after build_panel): refuse if the panel returned anything other than
        # exactly the requested universe.
        ev._assert_realized_panel_universe(universe_requested, panel)
        model, trainer = ev.build_model_and_trainer(args, args.n_assets, init_seed=0)
        checkpoint_load = ev.load_checkpoint_into(trainer, ckpt_path)
        # TASK_886 BLOCKER 3b, again in eval_vector.main()'s position: the instant weights exist,
        # refuse a provenance whose recorded fingerprint describes DIFFERENT weights.
        ev._assert_provenance_binds_loaded_params(ckpt_path, universe_pin, checkpoint_load)
        pin_record = ev.finalize_universe_pin_record(
            universe_pin, panel_info, col_symbols, universe_requested, checkpoint_load)
        checkpoint_label = ckpt_path
        walk_kind = WALK_KIND_CHECKPOINT
    params = trainer.state.params

    imp = ev.load_imprint_module()
    _tgt, book_warm_op = ev.book_target_signs(prices)
    ev.guard_book_warm(book_warm_op, warm)

    if args.decision_rule == "random-matched":
        marginals = ev.load_action_marginals(args.action_marginals, col_symbols)
        policy_actions = ev.random_matched_actions(
            marginals, n_decisions, int(env_cfg0.n_assets), args.eval_seed)
    else:
        obs_all = imp.encode_training_obs(prices, warm, n_decisions, env_cfg0)
        policy_actions = ev.policy_decision_actions(
            model, params, obs_all, int(env_cfg0.n_assets),
            args.decision_rule, args.decision_tau, args.decision_m)

    # Minimal defensive guard (TASK_878's (A), reused verbatim as a HOLD-code): nothing
    # quotable is nothing attemptable. This module deliberately does NOT reimplement 878's
    # full frozen-cell open-gate (that lives in evaluate_checkpoint and is out of this
    # instrument's scope), only the zero-coverage floor every operator in this repo applies.
    dec_prices = np.asarray(prices[warm:warm + n_decisions])
    zero_coverage = dec_prices <= 0
    if zero_coverage.any():
        policy_actions = np.array(policy_actions, copy=True)
        policy_actions[zero_coverage] = ev.HOLD

    fee, slip = ((ev.FEE_BPS, ev.SLIP_BPS) if args.tier_bp else (0.0, 0.0))
    tier_cfg = env_cfg0.replace(fee_bps=fee, slippage_bps=slip)
    # BLOCKER 2: opt into rollout's third recorder key so the budget check reads the walk's OWN
    # realized allocation (1 - cash/PV) instead of a field the weight path never reads.
    collect = {"chosen_vectors": [], "realized_book": [], "allocated_fraction": []}
    ev.rollout(params, None, prices, tier_cfg, state_init, int(env_cfg0.n_assets), n_decisions,
              args.eval_seed, args.max_positions, args.min_cash_threshold,
              policy_actions=policy_actions, collect=collect)

    n_coll = len(collect["chosen_vectors"])
    rows = list(range(warm, warm + n_coll))
    w_unit = (float(tier_cfg.vector_weight_unit) if tier_cfg.vector_weight_unit > 0.0
              else 1.0 / int(env_cfg0.n_assets))
    doc = build_trace_doc(
        tag=args.tag, checkpoint_label=checkpoint_label, decision_rule=args.decision_rule,
        action_marginals_path=args.action_marginals, eval_seed=args.eval_seed,
        tier_bp=args.tier_bp, fee_bps=fee, slip_bps=slip, horizon=args.horizon,
        position_size_pct=float(env_cfg0.position_size_pct), col_symbols=col_symbols,
        row_index=rows, timestamps=[int(row_ts[r]) for r in rows],
        chosen_vectors=collect["chosen_vectors"], realized_book=collect["realized_book"],
        prices=prices, universe_pin=pin_record, walk_kind=walk_kind,
        allocated_fraction=collect["allocated_fraction"],
        allocation_semantics=(WEIGHT_SEMANTICS if tier_cfg.vector_weight_rebalance
                              else PCT_SEMANTICS),
        weight_unit=(w_unit if tier_cfg.vector_weight_rebalance else None),
        weight_max_mult=(float(tier_cfg.vector_weight_max) if tier_cfg.vector_weight_rebalance
                         else None),
        # [A1] only a --checkpoint walk carries a class; random-matched/--smoke walks have none
        # of their own to pool.
        subject_class=(getattr(args, "subject_class", None) if walk_kind == WALK_KIND_CHECKPOINT
                       else None))
    return doc


def main():
    # TASK_892: the shared arg surface (--db/--symbols/--decision-rule/--marginals-sidecar/
    # --checkpoint/--smoke/--eval-seed/... ) comes VERBATIM from eval_vector.build_argparser()
    # — one parser, one set of defaults, no drift between the two modules (the same reuse
    # discipline as 849/leaf_probe.py importing eval_vector's model/panel/search machinery
    # instead of re-declaring it). Only trace.py's OWN concepts (H, the cost tier, the output
    # path) get a second, thin parser; ``parse_known_args`` lets both coexist on one argv.
    trace_ap = argparse.ArgumentParser(add_help=False)
    trace_ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON,
                          help="forward decisions H (SPEC.md: 21)")
    trace_ap.add_argument("--tier-bp", type=int, choices=[0, 21], default=0, dest="tier_bp",
                          help="SPEC.md primary tier is cost0 (separates edge from fees)")
    trace_ap.add_argument("--out", default=None,
                          help="output trace JSON path (required unless --smoke, which defaults "
                               "to experiments/892_action_ev/_scratch/ — GITIGNORED; this tool "
                               "never writes into the tracked experiment directory by default)")
    trace_ap.add_argument("--subject-class", choices=SUBJECT_CLASSES, default=None,
                          dest="subject_class",
                          help="[A1] which SPEC.md Amendment 1 subject class a --checkpoint "
                               "walk belongs to (861_ref_primary / 887_A_collapsed / "
                               "887_D_uncollapsed / 861_ref_search_secondary / "
                               "832_e5_contrast). Stamped into provenance.subject_class; "
                               "report.py's never-pooled enforcement is keyed off this field. "
                               "Ignored on --smoke walks (no class of their own to pool).")
    trace_args, remaining_argv = trace_ap.parse_known_args()

    ev = _load_eval_vector()
    ap = ev.build_argparser()
    args = ap.parse_args(remaining_argv)
    args.horizon, args.tier_bp, args.out = trace_args.horizon, trace_args.tier_bp, trace_args.out
    args.subject_class = trace_args.subject_class
    # This instrument never searches — the null/subject walks are both the 848 policy-argmax
    # replay (rollout() bypasses search whenever policy_actions is given regardless of this
    # value, but build_search is only skipped for 'policy-argmax' at call sites that check it).
    args.operator = "policy-argmax"

    if args.decision_rule == "random-matched" and not args.action_marginals:
        ap.error("--decision-rule random-matched requires --marginals-sidecar <sidecar.json>")

    if args.smoke:
        args.db = ev.FIXTURE_DB
        args.symbols = args.symbols or ev.SMOKE_SYMBOLS
        args.start = args.start or ev.SMOKE_START
        args.end = args.end or ev.SMOKE_END
        args.bar_minutes = ev.SMOKE_BAR_MINUTES
        args.n_assets = len(args.symbols)
        args.max_decisions = args.max_decisions or ev.SMOKE_MAX_DECISIONS
        args.tag = args.tag or "action-ev-fixture-smoke"
    else:
        if not args.checkpoint:
            ap.error("--checkpoint is required unless --smoke")
        if args.symbols:
            args.symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        args.tag = args.tag or "action-ev"

    out_path = args.out
    if out_path is None:
        if not args.smoke:
            ap.error("--out is required unless --smoke")
        # finding 4: the default lands in the GITIGNORED scratch dir, never in the tracked
        # experiment directory (the run-34 "a write nobody thinks of as an edit" pattern).
        os.makedirs(SCRATCH_DIR, exist_ok=True)
        out_path = os.path.join(SCRATCH_DIR, f"trace_{args.tag}_{args.decision_rule}.json")

    # TASK_883/886 (finding 5): the universe guard runs BEFORE any panel is built, exactly as
    # eval_vector.main() orders it, and hands the mechanism-typed record to the walk. --smoke is
    # a fresh-init fixture walk with no checkpoint to pin (eval_vector.py's own run_smoke returns
    # before its guard too), so it is the one path that legitimately carries no record.
    universe_pin = universe_requested = None
    if not args.smoke:
        universe_pin_info, universe_requested = ev.enforce_universe_pin(
            [args.checkpoint[0]], args)
        universe_pin = universe_pin_info[args.checkpoint[0]]

    doc = run_walk(args, ev, universe_pin=universe_pin,
                   universe_requested=universe_requested)
    # cycle-3 minor: atomic. Every refusal above happens strictly BEFORE this line, so the only
    # remaining failure mode was a kill mid-`json.dump` leaving a truncated trace behind.
    write_json_atomic(out_path, doc, indent=2, default=float)
    n_nh = int(np.asarray(doc["decisions"]["non_hold"]).sum())
    prov = doc["provenance"]
    _pin = prov["universe_pin"]
    print(f"[892/trace] wrote {out_path} "
         f"(decision_rule={args.decision_rule}, n_decisions={len(doc['decisions']['row_index'])}, "
         f"non_hold_cells={n_nh}, allocation_semantics={prov['allocation_semantics']}, "
         f"max_allocated_fraction="
         f"{max(doc['decisions']['allocated_fraction'], default=0.0):.6f}, "
         f"budget_violations={prov['budget_conservation_violations']})")
    if _pin is None:
        pin_desc = "null (fresh-init smoke walk — see universe_pin_note)"
    else:
        pin_desc = (f"mechanism={_pin.get('mechanism')} verified={_pin.get('verified')} "
                    f"reason={_pin.get('verified_reason')}")
    print(f"[892/trace] walk_kind={prov['walk_kind']} universe_pin: {pin_desc}")
    print(f"[892/trace] {disclosures.READ_THIS_FIRST}")


if __name__ == "__main__":
    main()
