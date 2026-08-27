"""TASK_907 gate (iv), run-45 halting MAJOR — DECLARED-vs-OBSERVED pair reconciliation.

The defect these tests pin, and the family it belongs to. Three cycles produced three real
defects and all three were *silent absence presented as completeness* at successively finer
granularity (per-sample/per-gap -> cumulative/per-capture -> per-pair total absence). The case
that escaped cycle 3:

    A DECLARED pair with ZERO data-bearing events for the entire capture.

It never enters `data_ts`, so:

  * defense 1 (per-jump over all timestamps) has no jump to find — heartbeats flow normally;
  * defense 2 (per-jump per pair over book/trade) has no timeline for that pair at all;
  * defense 3 (`coverage_by_pair`) has no coverage record for it to find degraded — the map is
    built over `data_ts` keys, so the dead pair is not in it and `degraded_coverage_pairs()`
    never iterates over it;
  * `missing_channels()` is a CAPTURE-WIDE sum, so the healthy pair keeps it silent.

`assert_complete()` passed. The pair was named in the top-line provenance and then silently
absent from every downstream table — a fabricated completeness claim, which is exactly what gate
(iv) exists to prevent.

WHY THE DECLARED SET MUST COME FROM THE SUBSCRIPTION CONFIG (D66 condition 1). `meta.pairs` is a
UNION of declared-and-observed (`meta_start`'s pairs, unioned with every pair seen in a data
event). Reconciling that union against `coverage_by_pair` is circular in one specific, checkable
way: a pair OBSERVED but never DECLARED is in the union AND in the coverage map, so the union can
never see it — and the union carries no channel information at all, so per-cell reconciliation is
impossible from it. `test_the_union_is_blind_where_the_declared_set_is_not` pins both halves.

Every fixture here is built with the real production modules — no monkeypatching, no bypass. The
tests that hand-craft a `CaptureMetadata` say so and say which shape they are closing.
"""
import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture import (  # noqa: E402
    DEFAULT_MAX_UNCOVERED_FRACTION, CaptureFormatError, CaptureMetadata, PairCoverage,
    read_capture,
)
from recorder import (  # noqa: E402
    SUBSCRIBED_CHANNELS, LiveRecorder, RecorderConfig, _subscribe_messages,
)
from render import render_report  # noqa: E402
import cli  # noqa: E402

T0 = 1_700_000_000.0

#: 5s cadence against a 15s coverage window: every fixture pair that HAS data is fully covered, so
#: defense 3 is provably not what is firing in any refusal below.
DENSE_CADENCE_S = 5.0


def _write_capture(path, data, *, declared_pairs=None, span_s=1800.0,
                    cadence_s=DENSE_CADENCE_S, channels=("book", "trade"),
                    heartbeat_every_s=30.0, t0=T0, pairs_field=..., include_pairs_field=True):
    """Write a synthetic capture where the DECLARED pair set and the pairs that actually carry
    data are set INDEPENDENTLY — which is the whole point of this file.

    `data` maps pair -> the tuple of channels to emit for it (so a pair can be given trades but no
    book, or nothing at all). `declared_pairs` defaults to `sorted(data)`; `pairs_field` overrides
    the raw JSON value written for `pairs` (for shape tests); `include_pairs_field=False` omits it.
    """
    if declared_pairs is None:
        declared_pairs = sorted(data)
    rows = []
    for pair, pair_channels in data.items():
        t = 0.0
        while t <= span_s:
            for offset, channel in enumerate(pair_channels):
                rows.append((t0 + t + 0.001 * offset, channel, pair))
            t += cadence_s
    t = 0.0
    while t <= span_s:
        rows.append((t0 + t, "heartbeat", None))
        t += heartbeat_every_s
    rows.sort(key=lambda r: (r[0], r[1]))

    with open(path, "w") as fh:
        meta = {"type": "meta_start", "capture_id": "decl", "format_version": 2,
                "clock_basis": "local_receipt"}
        if include_pairs_field:
            meta["pairs"] = sorted(declared_pairs) if pairs_field is ... else pairs_field
        if channels is not None:
            meta["channels"] = list(channels)
        fh.write(json.dumps(meta) + "\n")
        for ts, kind, pair in rows:
            if kind == "book":
                fh.write(json.dumps({"type": "book", "pair": pair, "ts": ts, "bid": 50000.0,
                                      "ask": 50001.0, "bid_sz": 1.0, "ask_sz": 1.0}) + "\n")
            elif kind == "trade":
                fh.write(json.dumps({"type": "trade", "pair": pair, "ts": ts, "price": 50000.5,
                                      "volume": 0.1, "side": "buy"}) + "\n")
            else:
                fh.write(json.dumps({"type": "heartbeat", "ts": ts}) + "\n")
        fh.write(json.dumps({"type": "meta_end", "ts": t0 + span_s}) + "\n")
    return str(path)


BOTH = ("book", "trade")


# --- THE HALTING MAJOR ---------------------------------------------------------------------------

def test_a_declared_pair_with_zero_data_events_is_refused(tmp_path):
    """THE REPORTED HALTING MAJOR. Before this fix, this exact capture gave
    `is_complete() -> True` and a full rendered artifact whose provenance line named ETH/USD while
    no table below it contained a single ETH/USD number.
    """
    path = _write_capture(tmp_path / "dead_pair.jsonl",
                           {"BTC/USD": BOTH},
                           declared_pairs=["BTC/USD", "ETH/USD"])
    meta = read_capture(path)

    # The BEHAVIOURAL assertions come first, deliberately: they are phrased only in terms of the
    # pre-existing public surface, so this test fails against the PRE-FIX code with "is_complete()
    # is True / assert_complete did not raise" — the defect itself — rather than with an
    # AttributeError about a field that did not exist yet. A test that can only fail on a missing
    # attribute proves the attribute is new, not that anything was ever broken.
    assert not meta.is_complete(), (
        "a declared pair with zero data-bearing events must not read as a complete capture")
    with pytest.raises(CaptureFormatError, match="DECLARED/OBSERVED pair reconciliation failed"):
        meta.assert_complete()
    assert "ETH/USD" not in meta.coverage_by_pair, (
        "the dead pair gets no coverage record at all — that absence IS the defect, and if it "
        "ever starts getting one the isolation test below stops isolating anything")

    assert meta.declared_pairs == ["BTC/USD", "ETH/USD"]
    assert meta.observed_pairs == ["BTC/USD"]
    assert meta.pair_reconciliation_gaps()["declared_without_data"] == ["ETH/USD"]


def test_the_dead_pair_defeats_all_three_prior_defenses(tmp_path):
    """Isolation, in the style of the 59s-cadence proof: assert that defenses 1-3 are all CLEAN on
    this fixture, so the refusal above is attributable to defense 4 and to nothing else.
    """
    path = _write_capture(tmp_path / "dead_pair2.jsonl",
                           {"BTC/USD": BOTH},
                           declared_pairs=["BTC/USD", "ETH/USD"])
    meta = read_capture(path)

    assert meta.unrecorded_gaps == []            # defense 1: heartbeats flow, socket never dark
    assert meta.unrecorded_data_gaps == []       # defense 2: no ETH timeline to find a jump in
    assert meta.degraded_coverage_pairs() == []  # defense 3: no ETH coverage record to degrade
    assert meta.missing_channels() == []         # capture-wide sums: BTC keeps both non-zero
    assert meta.coverage_evaluated
    assert meta.coverage_by_pair["BTC/USD"].uncovered_fraction < DEFAULT_MAX_UNCOVERED_FRACTION

    with pytest.raises(CaptureFormatError, match="DECLARED/OBSERVED pair reconciliation failed"):
        meta.assert_complete()


def test_the_refusal_message_names_the_remedy_and_forbids_dropping_the_pair(tmp_path):
    """House style: the refusal must state declared vs observed, say why silent absence is worse
    than a crash, and prescribe re-cut/re-run — explicitly NOT 'drop the pair' or 'widen'."""
    path = _write_capture(tmp_path / "dead_pair3.jsonl", {"BTC/USD": BOTH},
                           declared_pairs=["BTC/USD", "ETH/USD"])
    meta = read_capture(path)
    with pytest.raises(CaptureFormatError) as exc:
        meta.assert_complete()
    msg = str(exc.value)
    assert "GATE (iv) REFUSAL" in msg
    assert "ETH/USD" in msg and "Declared" in msg and "Observed" in msg
    assert "893" in msg                      # why silent absence is worse than a crash
    assert "re-cut" in msg and "re-run" in msg
    assert "drop the pair" in msg and "widen the bound" in msg


def test_no_number_renders_from_a_capture_with_a_dead_declared_pair(tmp_path):
    """The gate is only worth anything if it reaches the artifact: `render_report` must refuse."""
    path = _write_capture(tmp_path / "dead_pair4.jsonl", {"BTC/USD": BOTH},
                           declared_pairs=["BTC/USD", "ETH/USD"])
    meta = read_capture(path)
    with pytest.raises(CaptureFormatError, match="DECLARED/OBSERVED pair reconciliation failed"):
        render_report([meta], {}, {})


# --- one granularity down: the declared (pair, channel) cell -------------------------------------

def test_a_declared_channel_of_one_pair_with_zero_events_is_refused(tmp_path):
    """The next granularity, closed in the same pass. ETH/USD is declared, observed, densely
    covered and perfectly reconciled at PAIR level — but its `book` subscription was dead from
    t=0. `missing_channels()` is a capture-wide sum (BTC's book keeps it non-zero) and per-pair
    coverage POOLS book+trade (ETH's dense trades keep it covered), so the cell is absent from the
    cadence table and from everything else with nothing flagging it.
    """
    path = _write_capture(tmp_path / "dead_cell.jsonl",
                           {"BTC/USD": BOTH, "ETH/USD": ("trade",)})
    meta = read_capture(path)

    assert meta.declared_pairs == ["BTC/USD", "ETH/USD"]
    assert meta.observed_pairs == ["BTC/USD", "ETH/USD"], "reconciles at PAIR level"
    assert meta.observed_pair_channels["ETH/USD"] == ["trade"]
    assert meta.missing_channels() == []
    assert meta.degraded_coverage_pairs() == []
    assert meta.unrecorded_data_gaps == []
    assert [c.channel for c in meta.cadence if c.pair == "ETH/USD"] == ["trade", "book+trade"], (
        "the dead `book` cell has no cadence row either — absence, not a zero. BTC/USD, which "
        "really has both, gets a `book` row; ETH/USD simply does not appear with one")
    assert [c.channel for c in meta.cadence if c.pair == "BTC/USD"] \
        == ["book", "trade", "book+trade"]

    assert meta.pair_reconciliation_gaps()["declared_channel_without_data"] == ["ETH/USD:book"]
    assert not meta.is_complete()
    with pytest.raises(CaptureFormatError, match="ETH/USD:book"):
        meta.assert_complete()


def test_an_undeclared_channels_capture_still_reconciles_at_pair_level(tmp_path):
    """The `channels=None` fallback is the module's existing, disclosed precedent (synthetic and
    v1 fixtures omit it). It may soften the per-CELL check — the artifact prints
    `channels=undeclared` so the reader knows which check was available — but it must NOT soften
    the pair-level one, or it becomes a bypass.
    """
    ok = _write_capture(tmp_path / "nochan_ok.jsonl", {"BTC/USD": ("trade",)}, channels=None)
    meta_ok = read_capture(ok)
    assert meta_ok.channels is None
    assert meta_ok.pair_reconciliation_gaps()["declared_channel_without_data"] == []
    meta_ok.assert_complete()

    bad = _write_capture(tmp_path / "nochan_bad.jsonl", {"BTC/USD": BOTH},
                          declared_pairs=["BTC/USD", "ETH/USD"], channels=None)
    meta_bad = read_capture(bad)
    assert not meta_bad.is_complete()
    with pytest.raises(CaptureFormatError, match="DECLARED/OBSERVED pair reconciliation failed"):
        meta_bad.assert_complete()


# --- D66 condition 1: the declared set is the SUBSCRIPTION CONFIG, not the union -----------------

def test_the_union_is_blind_where_the_declared_set_is_not(tmp_path):
    """CONDITION 1, proven rather than argued. `meta.pairs` is declared UNION observed, so a pair
    that was OBSERVED but never DECLARED sits in the union AND in `coverage_by_pair` — reconciling
    the union against the coverage map can never see it. Reconciling the DECLARED set can.

    This is the mutation guard on condition 1: re-implement `declared_pairs` as the union and the
    first assertion goes red; reconcile against `meta.pairs` instead and the refusal disappears.
    """
    path = _write_capture(tmp_path / "undeclared.jsonl", {"BTC/USD": BOTH, "ETH/USD": BOTH},
                           declared_pairs=["BTC/USD"])
    meta = read_capture(path)

    # behavioural first (see the note in the halting-MAJOR test above)
    assert not meta.is_complete()
    with pytest.raises(CaptureFormatError, match="DECLARED/OBSERVED pair reconciliation failed"):
        meta.assert_complete()
    assert meta.pairs == ["BTC/USD", "ETH/USD"], "meta.pairs keeps its union meaning (back-compat)"
    # the union's blindness, stated as an assertion: every union member has a coverage record, so a
    # union-vs-coverage reconciliation is vacuous here.
    assert all(p in meta.coverage_by_pair for p in meta.pairs)

    assert meta.declared_pairs == ["BTC/USD"], "declared comes from meta_start ALONE"
    assert meta.observed_pairs == ["BTC/USD", "ETH/USD"]
    gaps = meta.pair_reconciliation_gaps()
    assert gaps["observed_undeclared"] == ["ETH/USD"]
    assert gaps["coverage_undeclared"] == ["ETH/USD"]
    assert not meta.is_complete()
    with pytest.raises(CaptureFormatError, match="DECLARED/OBSERVED pair reconciliation failed"):
        meta.assert_complete()


def test_the_subscribe_frames_and_meta_start_declare_the_same_pair_set(tmp_path):
    """CONDITION 1, closed at the WRITE side. `RecorderConfig.pairs` is the subscription config: it
    goes out as `symbol` on both public subscribe frames AND into `meta_start`'s `pairs`. Today
    that is a single source of truth by construction and by nothing else — so assert it, and a
    future edit that splits the two goes red here instead of silently making the declared side of
    gate (iv) describe a different subscription than the socket actually asked for.

    Driven through the real `LiveRecorder.run` path with an injected fake socket (no network, no
    injected writer — the writer is the production one built from `cfg.pairs`).
    """
    out = str(tmp_path / "coupling.jsonl")
    configured = ["BTC/USD", "ETH/USD", "SOL/USD"]
    sent = []

    class _WS:
        async def send(self, msg):
            sent.append(json.loads(msg))

        async def recv(self):
            raise RuntimeError("fake connection exhausted")

    class _CM:
        async def __aenter__(self):
            return _WS()

        async def __aexit__(self, *a):
            return False

    async def _fast_sleep(_s):
        return None

    cfg = RecorderConfig(pairs=list(configured), out_path=out, capture_id="coupling",
                          heartbeat_timeout_s=5.0, backoff_base_s=0.001, backoff_max_s=0.01)
    rec = LiveRecorder(cfg, lambda url: _CM(), sleep_fn=_fast_sleep)
    asyncio.run(rec.run(max_iterations=1))
    rec.close(clean=False)

    subscribed = [frame["params"]["symbol"] for frame in sent
                  if frame.get("method") == "subscribe"]
    assert len(subscribed) == len(_subscribe_messages(cfg)) == 2, (
        "both public subscribe frames must be accounted for; a new one appearing here without a "
        "symbol list is a hole in this coupling")
    assert all(sym == configured for sym in subscribed)

    with open(out) as fh:
        meta_start = json.loads(fh.readline())
    assert meta_start["type"] == "meta_start"
    assert meta_start["pairs"] == configured
    for sym in subscribed:
        assert sym == meta_start["pairs"], (
            "the pairs DECLARED into the capture must be exactly the symbols the socket "
            "subscribed to — the declared side of gate (iv) is only meaningful if it is the "
            "subscription config")

    # and the reader's declared set is that same list, normalized
    assert read_capture(out).declared_pairs == sorted(configured)

    # The CHANNEL side of the same coupling. `channels=None` is a legitimate, disclosed shape (the
    # writer only emits the field when its caller supplies it, and synthetic fixtures omit it),
    # which SOFTENS the per-cell reconciliation. Pin here that the real recorder always declares
    # them and always declares exactly the channels it subscribed to — so the softening can only
    # ever apply to a synthetic capture, never to one this probe actually recorded.
    assert meta_start["channels"] == list(SUBSCRIBED_CHANNELS)
    assert sorted(frame["params"]["channel"] for frame in sent
                   if frame.get("method") == "subscribe") == sorted(SUBSCRIBED_CHANNELS)


# --- fail-CLOSED on a missing / malformed declaration -------------------------------------------

def test_a_capture_declaring_no_pairs_at_all_is_refused(tmp_path):
    """`CaptureWriter` writes `pairs` into `meta_start` unconditionally at EVERY format version, so
    a capture without it is a shape this recorder cannot emit. Fail-closed: falling back to
    "declared := observed" would let a hand-crafted capture declare nothing and reconcile
    perfectly against whatever it contains.
    """
    absent = _write_capture(tmp_path / "nopairs.jsonl", {"BTC/USD": BOTH},
                             include_pairs_field=False)
    meta = read_capture(absent)
    assert not meta.is_complete()          # behavioural first (see the halting-MAJOR test above)
    with pytest.raises(CaptureFormatError, match="DECLARES NO PAIR SET"):
        meta.assert_complete()
    assert meta.declared_pairs is None
    assert meta.declared_pair_set_missing()

    empty = _write_capture(tmp_path / "emptypairs.jsonl", {"BTC/USD": BOTH}, pairs_field=[])
    meta_empty = read_capture(empty)
    assert not meta_empty.is_complete()
    with pytest.raises(CaptureFormatError, match="DECLARES NO PAIR SET"):
        meta_empty.assert_complete()
    assert meta_empty.declared_pairs == []


def test_a_malformed_pairs_declaration_is_refused_at_the_source(tmp_path):
    """A bare string would be iterated character-by-character into a declared set of single letters
    that no observed pair can ever match. Refuse the shape where it is read, not downstream."""
    for bad in ("BTC/USD", [""], [1], [None], {"BTC/USD": 1}, ["BTC/USD", ""]):
        path = _write_capture(tmp_path / f"bad_{abs(hash(str(bad)))}.jsonl",
                               {"BTC/USD": BOTH}, pairs_field=bad)
        with pytest.raises(CaptureFormatError, match="must be a list of non-empty strings"):
            read_capture(path)


# --- the shape-trust gap: hand-crafted `coverage_by_pair` shapes that used to sail through -------

def _hand_built(**kwargs) -> CaptureMetadata:
    """A metadata object that has cleared every check EXCEPT the one under test. `coverage_evaluated`
    is set True on purpose: these tests exist to prove the reconciliation is a real cross-check and
    not just a second name for the `coverage_evaluated` fence.
    """
    base = dict(path="hand-built", had_meta_start=True, n_book=5, n_trade=5,
                 start_ts=T0, end_ts=T0 + 100.0, coverage_evaluated=True,
                 declared_pairs=["BTC/USD"], channels=["book", "trade"],
                 observed_pair_channels={"BTC/USD": ["book", "trade"]},
                 coverage_by_pair={"BTC/USD": PairCoverage(
                     pair="BTC/USD", n_data_events=10, window_s=15.0, span_s=100.0,
                     recorded_gap_s=0.0, accounted_s=100.0, covered_s=100.0)})
    base.update(kwargs)
    return CaptureMetadata(**base)


def test_the_control_hand_built_shape_passes(tmp_path):
    """Non-vacuity: the helper's default shape must PASS, or the tests below prove nothing."""
    meta = _hand_built()
    assert meta.is_complete()
    meta.assert_complete()


@pytest.mark.parametrize("shape,expect", [
    # declared pair with no coverage record and no observed events — the halting MAJOR, hand-built
    (dict(declared_pairs=["BTC/USD", "ETH/USD"]), "declared_without_data"),
    # declared + observed, but the coverage map does not cover it: `assert_complete` used to trust
    # the SHAPE of `coverage_by_pair` rather than verifying it covered what it should have
    (dict(declared_pairs=["BTC/USD", "ETH/USD"],
          observed_pair_channels={"BTC/USD": ["book", "trade"],
                                   "ETH/USD": ["book", "trade"]}), "declared_without_coverage"),
    # a coverage record for a pair nobody declared
    (dict(coverage_by_pair={"ETH/USD": PairCoverage(
        pair="ETH/USD", n_data_events=10, window_s=15.0, span_s=100.0, recorded_gap_s=0.0,
        accounted_s=100.0, covered_s=100.0)}), "coverage_undeclared"),
    # observed data attributed to a pair nobody declared
    (dict(observed_pair_channels={"BTC/USD": ["book", "trade"],
                                   "ETH/USD": ["trade"]}), "observed_undeclared"),
    # an observed pair whose channel list is empty is NOT observed — the pair-level view must not
    # disagree with the grid it is derived from
    (dict(observed_pair_channels={"BTC/USD": []}), "declared_without_data"),
    # the declared cell, hand-built
    (dict(observed_pair_channels={"BTC/USD": ["trade"]}), "declared_channel_without_data"),
])
def test_hand_crafted_metadata_shapes_no_longer_bypass_the_cross_check(shape, expect):
    meta = _hand_built(**shape)
    gaps = meta.pair_reconciliation_gaps()
    assert gaps[expect], f"expected {expect} to fire for shape {shape}"
    assert not meta.is_complete()
    with pytest.raises(CaptureFormatError, match="GATE .iv. REFUSAL"):
        meta.assert_complete()


def test_a_hand_built_shape_cannot_evade_by_declaring_nothing():
    """The bypass the fail-closed rule closes: blank the declaration and every set difference is
    empty, so a naive reconciliation would report a clean capture."""
    meta = _hand_built(declared_pairs=[],
                        observed_pair_channels={"ETH/USD": ["book", "trade"]},
                        coverage_by_pair={})
    assert meta.pair_reconciliation_gaps()["declared_without_data"] == []
    assert not meta.is_complete()
    with pytest.raises(CaptureFormatError, match="DECLARES NO PAIR SET"):
        meta.assert_complete()


# --- the boolean twin must never disagree with the raising one -----------------------------------

def test_is_complete_and_assert_complete_never_disagree(tmp_path):
    """`is_complete()` is the boolean twin of `assert_complete()`. If any input can make one True
    while the other raises, that divergence is itself the defect: the gate would pass in one call
    site and refuse in another.
    """
    cases = [
        read_capture(_write_capture(tmp_path / "a.jsonl", {"BTC/USD": BOTH})),
        read_capture(_write_capture(tmp_path / "b.jsonl", {"BTC/USD": BOTH, "ETH/USD": BOTH})),
        read_capture(_write_capture(tmp_path / "c.jsonl", {"BTC/USD": BOTH},
                                     declared_pairs=["BTC/USD", "ETH/USD"])),
        read_capture(_write_capture(tmp_path / "d.jsonl", {"BTC/USD": BOTH, "ETH/USD": ("trade",)})),
        read_capture(_write_capture(tmp_path / "e.jsonl", {"BTC/USD": BOTH, "ETH/USD": BOTH},
                                     declared_pairs=["BTC/USD"])),
        read_capture(_write_capture(tmp_path / "f.jsonl", {"BTC/USD": BOTH},
                                     include_pairs_field=False)),
        read_capture(_write_capture(tmp_path / "g.jsonl", {"BTC/USD": ("trade",)}, channels=None)),
        _hand_built(),
        _hand_built(declared_pairs=["BTC/USD", "ETH/USD"]),
        _hand_built(declared_pairs=[]),
        _hand_built(observed_pair_channels={"BTC/USD": ["trade"]}),
        _hand_built(coverage_evaluated=False),
    ]
    n_complete = 0
    for meta in cases:
        try:
            meta.assert_complete()
            raised = False
        except CaptureFormatError:
            raised = True
        assert meta.is_complete() == (not raised), (
            f"is_complete()={meta.is_complete()} but assert_complete() "
            f"{'raised' if raised else 'passed'} for {meta.path} "
            f"(declared={meta.declared_pairs}, observed={meta.observed_pairs})")
        n_complete += int(not raised)
    assert 0 < n_complete < len(cases), "the table must contain both passing and failing cases"


# --- non-vacuity: a healthy multi-pair capture must still pass and render ------------------------

def test_a_healthy_multi_pair_capture_clears_the_reconciliation_and_renders(tmp_path):
    """If this fails, the new defense is simply refusing everything and proves nothing."""
    path = _write_capture(tmp_path / "healthy.jsonl", {"BTC/USD": BOTH, "ETH/USD": BOTH})
    meta = read_capture(path)
    assert meta.declared_pairs == meta.observed_pairs == ["BTC/USD", "ETH/USD"]
    assert not any(meta.pair_reconciliation_gaps().values())
    assert meta.is_complete()
    meta.assert_complete()

    report = render_report([meta], {}, {})
    assert "declared_pairs=['BTC/USD', 'ETH/USD']" in report
    assert "observed_pairs=['BTC/USD', 'ETH/USD']" in report


# --- and no dial ---------------------------------------------------------------------------------

def test_the_reconciliation_is_not_reachable_from_the_command_line():
    """A dial on a fail-loud bound is a bypass with extra steps. There must be no flag that lets an
    operator narrow the declared set, skip the reconciliation, or tolerate a dead pair.
    """
    flags = {a.dest for a in cli.build_argparser()._actions}
    for forbidden in ("declared_pairs", "skip_pair_reconciliation", "allow_dead_pairs",
                       "require_all_pairs", "allow_undeclared_pairs"):
        assert forbidden not in flags
