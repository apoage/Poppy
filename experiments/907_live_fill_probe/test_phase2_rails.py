import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase2_rails import (  # noqa: E402
    Phase2Args, Phase2RefusalError, RailViolation, auto_cancel_due, build_argparser,
    check_arming, check_kill_file, check_max_open_orders, check_per_order_cap, check_total_cap,
    enforce_post_only, place_order, run_phase2_dry,
)


# --- arming: refuses in BOTH directions ---

def test_arming_refuses_flag_without_cap():
    with pytest.raises(Phase2RefusalError, match="budget-cap-usd"):
        check_arming(Phase2Args(arm_live=True, budget_cap_usd=None, max_order_usd=10.0))


def test_arming_refuses_cap_without_flag():
    with pytest.raises(Phase2RefusalError, match="arm-live"):
        check_arming(Phase2Args(arm_live=False, budget_cap_usd=50.0))


def test_arming_refuses_neither():
    with pytest.raises(Phase2RefusalError):
        check_arming(Phase2Args(arm_live=False, budget_cap_usd=None))


def test_arming_refuses_nonpositive_cap():
    with pytest.raises(Phase2RefusalError):
        check_arming(Phase2Args(arm_live=True, budget_cap_usd=0.0, max_order_usd=10.0))
    with pytest.raises(Phase2RefusalError):
        check_arming(Phase2Args(arm_live=True, budget_cap_usd=-10.0, max_order_usd=10.0))


def test_arming_passes_with_both():
    check_arming(Phase2Args(arm_live=True, budget_cap_usd=25.0, max_order_usd=10.0))  # must not raise


# --- D68(b)/TASK_910 scope 4: BOTH notional caps are mandatory to arm, not just budget_cap_usd
# --- (run 44's MAJOR finding: a budget cap alone armed the probe while the per-order cap and the
# --- cumulative-spend check stayed opt-in and defaulted OFF).

def test_arming_refuses_budget_cap_without_max_order_usd():
    with pytest.raises(Phase2RefusalError, match="max-order-usd"):
        check_arming(Phase2Args(arm_live=True, budget_cap_usd=50.0, max_order_usd=None))


def test_arming_refuses_max_order_usd_without_budget_cap():
    with pytest.raises(Phase2RefusalError, match="budget-cap-usd"):
        check_arming(Phase2Args(arm_live=True, budget_cap_usd=None, max_order_usd=10.0))


# --- rails ---

def test_kill_file_rail_fires(tmp_path):
    kf = tmp_path / "KILL"
    kf.write_text("stop")
    with pytest.raises(RailViolation, match="kill file"):
        check_kill_file(str(kf))


def test_kill_file_rail_passes_when_absent(tmp_path):
    check_kill_file(str(tmp_path / "does_not_exist"))  # must not raise
    check_kill_file(None)


def test_per_order_cap_fires():
    with pytest.raises(RailViolation, match="per-order cap"):
        check_per_order_cap(notional_usd=100.0, cap_usd=50.0)
    check_per_order_cap(notional_usd=50.0, cap_usd=50.0)  # at the cap is fine


def test_total_cap_fires():
    with pytest.raises(RailViolation, match="total cap"):
        check_total_cap(cumulative_notional_usd=101.0, cap_usd=100.0)
    check_total_cap(cumulative_notional_usd=100.0, cap_usd=100.0)


def test_max_open_orders_fires():
    with pytest.raises(RailViolation, match="max_open_orders"):
        check_max_open_orders(open_count=2, max_open_orders=2)
    check_max_open_orders(open_count=1, max_open_orders=2)


def test_auto_cancel_due():
    assert auto_cancel_due(bars_open=2, auto_cancel_bars=2) is True
    assert auto_cancel_due(bars_open=1, auto_cancel_bars=2) is False


def test_post_only_enforced():
    enforce_post_only("limit_post_only")  # must not raise
    with pytest.raises(RailViolation, match="not post-only"):
        enforce_post_only("market")
    with pytest.raises(RailViolation, match="not post-only"):
        enforce_post_only("limit")  # a bare limit WITHOUT the post-only tag is also refused


# --- the live transmission path never runs ---

def test_place_order_always_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="NOT implemented"):
        place_order()
    with pytest.raises(NotImplementedError):
        place_order(pair="BTC/USD", side="buy", price=1.0, volume=1.0)  # args don't matter


def test_run_phase2_dry_passes_every_rail_then_hits_the_unimplemented_transmit():
    args = Phase2Args(arm_live=True, budget_cap_usd=100.0, max_order_usd=20.0,
                       total_cap_usd=100.0, max_open_orders=3, kill_file=None)
    with pytest.raises(NotImplementedError):
        run_phase2_dry(args, notional_per_order_usd=10.0, cumulative_before_usd=0.0,
                        open_count=0)


def test_run_phase2_dry_still_refuses_arming_before_reaching_any_rail():
    args = Phase2Args(arm_live=False, budget_cap_usd=None)
    with pytest.raises(Phase2RefusalError):
        run_phase2_dry(args, notional_per_order_usd=10.0, cumulative_before_usd=0.0,
                        open_count=0)


def test_run_phase2_dry_rail_violation_stops_before_transmit(tmp_path):
    kf = tmp_path / "KILL"
    kf.write_text("stop")
    args = Phase2Args(arm_live=True, budget_cap_usd=100.0, max_order_usd=50.0, kill_file=str(kf))
    with pytest.raises(RailViolation):
        run_phase2_dry(args, notional_per_order_usd=10.0, cumulative_before_usd=0.0,
                        open_count=0)


# --- D68(b)/TASK_910 scope 4: `check_total_cap` is now called UNCONDITIONALLY at the order
# --- boundary, bound to the mandatory --budget-cap-usd, even when no separate --total-cap-usd
# --- was ever supplied. Before this fix, `check_total_cap` was only reached when the operator
# --- ALSO passed --total-cap-usd, so a declared budget cap never bounded anything on its own —
# --- this reproduces run 44's MAJOR demonstration exactly (a $1,000,000 order against $9,999,999
# --- already spent, against a declared $50 budget cap, cleared every rail).

def test_run_phase2_dry_enforces_the_budget_cap_as_the_total_cap_with_no_total_cap_usd_given():
    args = Phase2Args(arm_live=True, budget_cap_usd=50.0, max_order_usd=1_000_000.0,
                       total_cap_usd=None)
    with pytest.raises(RailViolation, match="total cap"):
        run_phase2_dry(args, notional_per_order_usd=1_000_000.0,
                        cumulative_before_usd=9_999_999.0, open_count=0)


def test_run_phase2_dry_enforces_the_max_order_cap_unconditionally():
    args = Phase2Args(arm_live=True, budget_cap_usd=10_000_000.0, max_order_usd=50.0,
                       total_cap_usd=None)
    with pytest.raises(RailViolation, match="per-order cap"):
        run_phase2_dry(args, notional_per_order_usd=1_000_000.0,
                        cumulative_before_usd=0.0, open_count=0)


# --- MAJOR (run-43): a NaN/inf budget cap must NOT satisfy the arming gate ---
#
# `args.budget_cap_usd <= 0` is False for both nan and inf, and argparse's `type=float` accepts
# both spellings — so `--budget-cap-usd nan` and `--budget-cap-usd inf` both ARMED, while the
# docstring claimed arming required "a positive budget cap".

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_arming_refuses_a_nonfinite_budget_cap(bad):
    with pytest.raises(Phase2RefusalError, match="FINITE|> 0"):
        check_arming(Phase2Args(arm_live=True, budget_cap_usd=bad, max_order_usd=10.0))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_arming_refuses_nonfinite_per_order_and_total_caps(bad):
    with pytest.raises(Phase2RefusalError, match="FINITE"):
        check_arming(Phase2Args(arm_live=True, budget_cap_usd=50.0, max_order_usd=bad))
    with pytest.raises(Phase2RefusalError, match="FINITE"):
        check_arming(Phase2Args(arm_live=True, budget_cap_usd=50.0, max_order_usd=10.0,
                                 total_cap_usd=bad))


def test_argparse_accepts_nan_so_the_check_is_the_only_thing_standing_between(tmp_path):
    """Non-vacuity: proves the parser really does hand through a non-finite float, so the
    refusal above is load-bearing rather than guarding an impossible input."""
    ns = build_argparser().parse_args(
        ["--arm-live", "--budget-cap-usd", "nan", "--max-order-usd", "10"])
    assert ns.budget_cap_usd != ns.budget_cap_usd  # it IS a NaN
    with pytest.raises(Phase2RefusalError):
        check_arming(Phase2Args(arm_live=ns.arm_live, budget_cap_usd=ns.budget_cap_usd,
                                 max_order_usd=ns.max_order_usd))


def test_arming_refuses_a_nonsensical_max_open_orders():
    with pytest.raises(Phase2RefusalError, match="max-open-orders"):
        check_arming(Phase2Args(arm_live=True, budget_cap_usd=50.0, max_order_usd=10.0,
                                 max_open_orders=0))


def test_cap_rails_refuse_a_nonfinite_notional():
    """`nan > cap` is False, so a NaN notional silently PASSED the cap it was supposed to hit."""
    with pytest.raises(RailViolation, match="not a finite number"):
        check_per_order_cap(notional_usd=float("nan"), cap_usd=50.0)
    with pytest.raises(RailViolation, match="not a finite number"):
        check_per_order_cap(notional_usd=float("inf"), cap_usd=50.0)
    with pytest.raises(RailViolation, match="not a finite number"):
        check_total_cap(cumulative_notional_usd=float("nan"), cap_usd=50.0)
    with pytest.raises(RailViolation, match="not a finite number"):
        check_total_cap(cumulative_notional_usd=10.0, cap_usd=float("nan"))


# --- MINOR 3: the kill file must fail CLOSED ---

def test_dangling_kill_file_symlink_means_stop(tmp_path):
    """`os.path.exists` follows symlinks and is False for a broken one, so "kill file present
    but target deleted" read as "no kill file, proceed" — the kill switch failed OPEN."""
    target = tmp_path / "target"
    target.write_text("stop")
    link = tmp_path / "KILL"
    link.symlink_to(target)
    target.unlink()  # now a dangling symlink

    import os
    assert os.path.exists(str(link)) is False   # the old check saw nothing here
    assert os.path.lexists(str(link)) is True   # the path IS present
    with pytest.raises(RailViolation, match="kill file present"):
        check_kill_file(str(link))


def test_kill_file_as_a_directory_means_stop(tmp_path):
    d = tmp_path / "KILL"
    d.mkdir()
    with pytest.raises(RailViolation, match="kill file present"):
        check_kill_file(str(d))


def test_kill_file_absent_still_proceeds(tmp_path):
    check_kill_file(str(tmp_path / "nope"))  # must not raise
    check_kill_file(None)
    check_kill_file("")


def test_unreadable_kill_file_path_means_stop(tmp_path):
    """If we cannot determine the kill file's state, we do not get to assume consent."""
    import os
    locked = tmp_path / "locked"
    locked.mkdir()
    kf = locked / "KILL"
    os.chmod(str(locked), 0o000)
    try:
        if os.geteuid() == 0:
            pytest.skip("running as root — permission bits are not enforced")
        with pytest.raises(RailViolation, match="cannot determine the state"):
            check_kill_file(str(kf))
    finally:
        os.chmod(str(locked), 0o700)


# --- MINOR 2: no safety flag may be reachable by a prefix abbreviation ---

def test_arm_flag_cannot_be_reached_by_abbreviation():
    """`--arm` prefix-matched `--arm-live` with argparse's default allow_abbrev=True: a
    two-character typo armed real-money trading. The sibling 881D row shipped this same defect
    class, so abbreviation resistance is a hard requirement here."""
    p = build_argparser()
    with pytest.raises(SystemExit):
        p.parse_args(["--arm"])
    with pytest.raises(SystemExit):
        p.parse_args(["--arm", "--budget-cap-usd", "50"])
    with pytest.raises(SystemExit):
        p.parse_args(["--budget", "50"])
    # the full, explicit spellings still work
    ns = p.parse_args(["--arm-live", "--budget-cap-usd", "50"])
    assert ns.arm_live is True and ns.budget_cap_usd == 50.0


def test_no_flag_on_the_phase2_parser_is_abbreviation_reachable():
    """Belt and braces: every declared long option, truncated by one character, must be
    rejected — so this cannot regress for a flag added later."""
    p = build_argparser()
    long_opts = [o for a in p._actions for o in a.option_strings if o.startswith("--")]
    assert "--arm-live" in long_opts
    for opt in long_opts:
        if opt in ("--help",):
            continue
        truncated = opt[:-1]
        with pytest.raises(SystemExit), _suppress_stderr():
            p.parse_args([truncated, "1"])


class _suppress_stderr:
    def __enter__(self):
        import io
        import sys
        self._orig = sys.stderr
        sys.stderr = io.StringIO()

    def __exit__(self, *exc):
        import sys
        sys.stderr = self._orig
        return False
