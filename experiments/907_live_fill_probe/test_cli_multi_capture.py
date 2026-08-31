"""TASK_907 ADDENDUM B (run 45/46) — B4 (the `--capture` set -> the report) and B5 (the
`--pairs` operator filter -> the report), exercised at the `cli.main` level end to end.

Both defects were reproduced first-hand against the pre-B addendum tree (run-45 banner):

  * B4 — two single-pair `BTC/USD` captures with DIFFERING markout distributions, rendered
    together, silently picked the LAST `--capture` on the command line and discarded the other
    whole, under a provenance block naming both files. Order-dependent: swapping the two
    `--capture` flags flipped which capture's numbers rendered.
  * B5 — `--pairs SOL/USD` against a capture that never declared or observed `SOL/USD` exited 0
    with a full gate-(iv)-passing disclosure box, zero estimand sections, and no line anywhere
    saying a filter had been applied.

`cli.main` is called directly (no subprocess) so these tests can capture rc and stderr cheaply.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture import CaptureWriter  # noqa: E402
import cli  # noqa: E402


def _write_single_pair_capture(path, pair, base_price, drift_per_tick, n_ticks=24,
                                tick_s=10.0, capture_id="cap"):
    """A capture through the REAL production writer: one pair, `n_ticks` book+trade events
    `tick_s` apart, price walking by `drift_per_tick` each tick (so two captures built with
    opposite-signed drift produce genuinely different markout distributions, per the orchestrator's
    own B4 repro table).
    """
    w = CaptureWriter(str(path), [pair], capture_id, channels=["book", "trade"],
                       heartbeat_timeout_s=30.0)
    t, price = 0.0, base_price
    for _ in range(n_ticks):
        w.write_book(pair, t, price, price + 0.2, 1.0, 1.0)
        w.write_trade(pair, t + 0.001, price, 0.1, "buy")
        price += drift_per_tick
        t += tick_s
    w.close(clean=True)
    return str(path)


def test_B4_two_captures_sharing_a_pair_REFUSE_naming_both_files_not_a_silent_clobber(
        tmp_path, capsys):
    """The orchestrator's own repro, at the `--capture`/`--capture` boundary: two captures both
    carrying `BTC/USD`. The old behaviour let the LAST `--capture` on the command line win
    silently; the fix is a loud refusal naming both files — never a silent pool, never a
    last-wins overwrite, regardless of flag order.
    """
    cap_a = _write_single_pair_capture(tmp_path / "a.jsonl", "BTC/USD", 100.0, +0.05,
                                        capture_id="capA")
    cap_b = _write_single_pair_capture(tmp_path / "b.jsonl", "BTC/USD", 100.0, -0.05,
                                        capture_id="capB")
    out = str(tmp_path / "out.md")

    rc = cli.main(["--capture", cap_a, "--capture", cap_b, "--out", out])
    err = capsys.readouterr().err
    assert rc == 1
    assert not Path(out).exists(), "a refusal must not leave a partial/clobbered artifact behind"
    assert "B4 REFUSAL" in err
    assert cap_a in err and cap_b in err, "both files are named, not just the one that lost"

    # order-independence: swapping the flags produces the SAME refusal, not a different winner
    out2 = str(tmp_path / "out2.md")
    rc2 = cli.main(["--capture", cap_b, "--capture", cap_a, "--out", out2])
    err2 = capsys.readouterr().err
    assert rc2 == 1
    assert not Path(out2).exists()
    assert "B4 REFUSAL" in err2


def test_B4_two_captures_with_disjoint_pairs_BOTH_render_with_correct_provenance(tmp_path):
    """The non-overlapping case must not be caught by the same net: two captures naming
    DIFFERENT pairs both contribute their numbers, and the artifact states which capture backed
    which pair — this is the B4 requirement, not just the refusal half of it.
    """
    cap_a = _write_single_pair_capture(tmp_path / "a.jsonl", "BTC/USD", 100.0, +0.05,
                                        capture_id="capA")
    cap_b = _write_single_pair_capture(tmp_path / "b.jsonl", "ETH/USD", 3000.0, -0.10,
                                        capture_id="capB")
    out = str(tmp_path / "out.md")

    rc = cli.main(["--capture", cap_a, "--capture", cap_b, "--out", out])
    assert rc == 0
    report = Path(out).read_text()
    assert f"## BTC/USD (source capture: `{cap_a}`)" in report
    assert f"## ETH/USD (source capture: `{cap_b}`)" in report
    assert f"captures declared on the command line: 2" in report
    assert f"captures that actually contributed at least one pair's numbers: 2" in report


def test_B5_pairs_naming_a_pair_no_supplied_capture_declared_or_observed_REFUSES(
        tmp_path, capsys):
    """The orchestrator's repro: `--pairs SOL/USD` against a capture that only ever had
    `BTC/USD`. Old behaviour: exit 0, full gate-(iv) disclosure box, zero estimand sections, no
    line stating a filter was applied. Required: a loud refusal, not a silent empty report.
    """
    cap = _write_single_pair_capture(tmp_path / "a.jsonl", "BTC/USD", 100.0, +0.05)
    out = str(tmp_path / "out.md")

    rc = cli.main(["--capture", cap, "--pairs", "SOL/USD", "--out", out])
    err = capsys.readouterr().err
    assert rc == 1
    assert not Path(out).exists()
    assert "B5 REFUSAL" in err
    assert "SOL/USD" in err


def test_B5_pairs_naming_a_declared_and_observed_pair_renders_and_STATES_the_filter(tmp_path):
    """The positive case: `--pairs` naming a pair the supplied captures actually reconcile on
    renders normally AND the artifact says a filter was applied — the filter must not be
    invisible downstream (the run-45 defect: `shadow_engine.run_shadow` filtered results while
    nothing in the artifact said so).
    """
    cap_a = _write_single_pair_capture(tmp_path / "a.jsonl", "BTC/USD", 100.0, +0.05,
                                        capture_id="capA")
    cap_b = _write_single_pair_capture(tmp_path / "b.jsonl", "ETH/USD", 3000.0, -0.10,
                                        capture_id="capB")
    out = str(tmp_path / "out.md")

    rc = cli.main(["--capture", cap_a, "--capture", cap_b, "--pairs", "BTC/USD", "--out", out])
    assert rc == 0
    report = Path(out).read_text()
    assert "## BTC/USD" in report
    assert "## ETH/USD" not in report, "the filter actually excluded the other pair"
    assert "`--pairs` filter applied: ['BTC/USD']" in report
