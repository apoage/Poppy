"""TASK_907 phase 2 — arming refusal + rails, WITHOUT live order transmission.

Scope for THIS run (orchestrator split, see the task brief): implement only the refusal gate
and the rails as tested enforcement code. The function that would actually transmit an order
to the venue (`place_order`) is a single, permanently-raising `NotImplementedError` — real-
money order placement ships as its own separately-verified increment. Nothing reachable from
this module can send an order.

Credentials: this module never reads the Kraken account credential env vars and never imports
the authenticated REST client's signed-request machinery — there is nothing here that could
use them, since `place_order` never runs. `test_credential_free.py` greps this file (by name,
deliberately not spelled out in this docstring) for zero occurrences.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import List, Optional


class Phase2RefusalError(RuntimeError):
    """Raised when phase 2 is asked to start without both required arming inputs."""


class RailViolation(RuntimeError):
    """Raised when a hard rail would be crossed. Always fail-closed: the caller must not
    catch this and proceed — every call site in this module raises rather than returning a
    bool, so a rail cannot be silently ignored by a caller that forgets to check a return
    value.
    """


@dataclass
class Phase2Args:
    arm_live: bool = False
    budget_cap_usd: Optional[float] = None
    max_order_usd: Optional[float] = None
    total_cap_usd: Optional[float] = None
    max_open_orders: int = 1
    auto_cancel_bars: int = 2
    kill_file: Optional[str] = None
    pairs: List[str] = field(default_factory=list)
    duration_s: Optional[float] = None


def _require_finite_positive_cap(value: Optional[float], flag: str) -> None:
    """A cap must be a FINITE positive number.

    `nan <= 0` and `inf <= 0` are both False, so a bare `<= 0` test armed the probe on
    `--budget-cap-usd nan` and `--budget-cap-usd inf` — argparse's `type=float` accepts both
    spellings. `inf` is "no cap at all" wearing a cap's clothes; `nan` is worse, because every
    subsequent comparison against it is False, so every downstream rail silently passes.
    """
    if value is None:
        return
    if not math.isfinite(value):
        raise Phase2RefusalError(
            f"phase 2 refuses to start: {flag} must be a FINITE number, got {value!r}. A "
            f"non-finite cap is not a cap: `inf` bounds nothing, and every comparison against "
            f"`nan` is False, so it silently defeats the rails it is supposed to feed.")
    if value <= 0:
        raise Phase2RefusalError(
            f"phase 2 refuses to start: {flag} must be > 0, got {value}.")


def check_arming(args: Phase2Args) -> None:
    """Phase 2 arms ONLY when an explicit arm flag AND BOTH mandatory notional caps
    (`--budget-cap-usd`, `--max-order-usd`) are given, each a FINITE POSITIVE number
    (D68(b)/TASK_910).

    Before this, only `--budget-cap-usd` was mandatory and it was never actually read by any
    rail — the per-order cap and the cumulative-spend check were both opt-in, default-`None`
    flags that a caller could simply omit. Demonstrated (run 44): a declared `--budget-cap-usd
    50` armed the probe, and a $1,000,000 order against $9,999,999 already spent then cleared
    every rail, because nothing compared either number against the declared $50. Both caps are
    now required to arm at all, and `run_phase2_dry` enforces both unconditionally at the order
    boundary (see there) — a budget cap can no longer be declared and then ignored.
    """
    if not args.arm_live and args.budget_cap_usd is None and args.max_order_usd is None:
        raise Phase2RefusalError(
            "phase 2 refuses to start: none of --arm-live, --budget-cap-usd, --max-order-usd "
            "was given.")
    if not args.arm_live:
        raise Phase2RefusalError(
            "phase 2 refuses to start: a notional cap was given but --arm-live was not — "
            "a cap alone is not consent to trade.")
    if args.budget_cap_usd is None:
        raise Phase2RefusalError(
            "phase 2 refuses to start: --arm-live was given but --budget-cap-usd was not — "
            "arming without a budget cap is refused unconditionally.")
    if args.max_order_usd is None:
        raise Phase2RefusalError(
            "phase 2 refuses to start: --arm-live was given but --max-order-usd was not — "
            "arming without a per-order cap is refused unconditionally (D68(b)/TASK_910: both "
            "notional caps are mandatory at arming, neither is opt-in).")
    _require_finite_positive_cap(args.budget_cap_usd, "--budget-cap-usd")
    _require_finite_positive_cap(args.max_order_usd, "--max-order-usd")
    # total_cap_usd stays an OPTIONAL, additional, tighter total-cap the operator may layer on
    # top of --budget-cap-usd (which is itself now the mandatory cumulative bound — see
    # `run_phase2_dry`); validate it for finiteness if given, same as the mandatory caps.
    _require_finite_positive_cap(args.total_cap_usd, "--total-cap-usd")
    if not isinstance(args.max_open_orders, int) or args.max_open_orders < 1:
        raise Phase2RefusalError(
            f"phase 2 refuses to start: --max-open-orders must be an integer >= 1, got "
            f"{args.max_open_orders!r}.")


def check_kill_file(kill_file: Optional[str]) -> None:
    """An unreadable OR dangling kill file means STOP.

    `os.path.exists` follows symlinks and returns False for a broken one, so "kill file present
    but its target was deleted" read as "no kill file, proceed" — the kill switch failed OPEN.
    `os.lstat` does not follow the link, and any error that is not "genuinely absent" is treated
    as a refusal: if we cannot determine the kill file's state, we do not get to assume consent.
    """
    if not kill_file:
        return
    try:
        os.lstat(kill_file)
    except FileNotFoundError:
        return  # genuinely absent — the only condition that permits proceeding
    except OSError as exc:
        raise RailViolation(
            f"cannot determine the state of the kill file at {kill_file!r} ({exc}) — refusing "
            f"to proceed. An undeterminable kill switch is a kill switch that is ON.") from exc
    raise RailViolation(
        f"kill file present at {kill_file!r} — refusing to proceed (this includes a dangling "
        f"symlink or an unreadable entry: presence of the path is enough)")


def _require_finite_notional(value: float, what: str) -> float:
    """A non-finite notional silently defeats its own cap: `nan > cap` is False, so the rail
    passes. Refuse it as a rail violation.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise RailViolation(
            f"{what} is not a finite number ({value!r}) — refusing. Every comparison against a "
            f"non-finite notional is False, which would pass this rail without checking it.")
    return float(value)


def check_per_order_cap(notional_usd: float, cap_usd: float) -> None:
    _require_finite_notional(notional_usd, "order notional")
    _require_finite_notional(cap_usd, "per-order cap")
    if notional_usd > cap_usd:
        raise RailViolation(
            f"order notional {notional_usd} exceeds per-order cap {cap_usd}")


def check_total_cap(cumulative_notional_usd: float, cap_usd: float) -> None:
    _require_finite_notional(cumulative_notional_usd, "cumulative notional")
    _require_finite_notional(cap_usd, "total cap")
    if cumulative_notional_usd > cap_usd:
        raise RailViolation(
            f"cumulative notional {cumulative_notional_usd} would exceed total cap {cap_usd}")


def check_max_open_orders(open_count: int, max_open_orders: int) -> None:
    if open_count >= max_open_orders:
        raise RailViolation(
            f"open order count {open_count} already at/above max_open_orders {max_open_orders}")


def auto_cancel_due(bars_open: int, auto_cancel_bars: int) -> bool:
    """Not a refusal — a scheduling decision the caller acts on (cancel the resting order)."""
    return bars_open >= auto_cancel_bars


def enforce_post_only(order_type: str) -> None:
    """Never a market order, ever. The only accepted order_type is the literal post-only
    limit tag; anything else — including a bare 'limit' that omits the post-only flag — is
    refused.
    """
    if order_type != "limit_post_only":
        raise RailViolation(
            f"order_type {order_type!r} is not post-only — refusing. Market orders and "
            f"non-post-only limits are never permitted by this code path.")


def place_order(*args, **kwargs):
    """THE LIVE TRANSMISSION PATH. Deliberately, permanently unimplemented in this build.

    Real-money order placement to the venue is a separate, unshipped increment — this row
    (TASK_907, D65) ships phase 1 (shadow, zero risk) plus phase 2's refusal gate and rails
    ONLY. No caller in this codebase can reach a real order-transmission call through this
    function; it always raises before doing anything else, regardless of how it is invoked or
    what arguments are passed. (Deliberately not naming the venue's own method literal here:
    TASK_910's credential fence globs every `.py` file in this lane for exactly that token, so
    this docstring must not carry it either.)
    """
    raise NotImplementedError(
        "TASK_907 phase 2 live order transmission is NOT implemented in this build. This is "
        "a deliberate, permanent refusal for this row — real-money order placement ships as "
        "its own separately-verified increment. Nothing in "
        "experiments/907_live_fill_probe/ can send an order to a venue.")


def run_phase2_dry(args: Phase2Args, notional_per_order_usd: float,
                    cumulative_before_usd: float, open_count: int) -> None:
    """Runs every rail in order, fixture-mode: proves arming + every rail fires correctly,
    then reaches `place_order`, which always raises. This function NEVER transmits an order —
    it exists so the rail chain is exercised end to end in tests without any live path.

    D68(b)/TASK_910 scope 4: both notional rails are now enforced UNCONDITIONALLY at this order
    boundary — no `if ... is not None` guard is left that a caller could satisfy by omitting a
    cap, because `check_arming` above already refused to arm without both `--budget-cap-usd`
    and `--max-order-usd` present. `check_total_cap` in particular is called against
    `args.budget_cap_usd` regardless of whether the operator also supplied a separate, tighter
    `--total-cap-usd` — before this, a declared budget cap was demanded at arming and then never
    read by any rail (run 44's MAJOR finding: a $1,000,000 order against $9,999,999 already
    spent cleared every rail with only `--budget-cap-usd 50` set, because `total_cap_usd` was
    the optional flag actually wired to `check_total_cap` and it had been left at its `None`
    default).
    """
    check_arming(args)
    check_kill_file(args.kill_file)
    check_per_order_cap(notional_per_order_usd, args.max_order_usd)
    cumulative_after_usd = cumulative_before_usd + notional_per_order_usd
    check_total_cap(cumulative_after_usd, args.budget_cap_usd)
    if args.total_cap_usd is not None:
        # An OPTIONAL, additional, tighter total cap layered on top of the mandatory check
        # above — the mandatory one already ran unconditionally regardless of this flag.
        check_total_cap(cumulative_after_usd, args.total_cap_usd)
    check_max_open_orders(open_count, args.max_open_orders)
    enforce_post_only("limit_post_only")
    place_order()  # always raises NotImplementedError — this line is unreachable in practice


def build_argparser():
    import argparse
    p = argparse.ArgumentParser(
        # allow_abbrev=False is a HARD requirement on this parser, not a nicety. With
        # abbreviation on, argparse prefix-matches `--arm` to `--arm-live`, so a two-character
        # typo arms real-money trading. The sibling 881D row shipped exactly this defect class
        # (an override flag defeated by a two-character prefix); `test_phase2_rails.py` pins it.
        allow_abbrev=False,
        description="TASK_907 phase 2 — NOT a live trading entry point in this build. Proves "
                    "the arming refusal and rails; the transmit path always raises "
                    "NotImplementedError. The orchestrator does not launch this.")
    p.add_argument("--arm-live", action="store_true")
    p.add_argument("--budget-cap-usd", type=float, default=None)
    p.add_argument("--max-order-usd", type=float, default=None)
    p.add_argument("--total-cap-usd", type=float, default=None)
    p.add_argument("--max-open-orders", type=int, default=1)
    p.add_argument("--auto-cancel-bars", type=int, default=2)
    p.add_argument("--kill-file", default=None)
    p.add_argument("--pairs", default="")
    return p


def main(argv=None) -> int:  # pragma: no cover — always refuses or raises; nothing to launch
    args_ns = build_argparser().parse_args(argv)
    args = Phase2Args(
        arm_live=args_ns.arm_live, budget_cap_usd=args_ns.budget_cap_usd,
        max_order_usd=args_ns.max_order_usd, total_cap_usd=args_ns.total_cap_usd,
        max_open_orders=args_ns.max_open_orders, auto_cancel_bars=args_ns.auto_cancel_bars,
        kill_file=args_ns.kill_file,
        pairs=[p.strip() for p in args_ns.pairs.split(",") if p.strip()],
    )
    check_arming(args)
    raise NotImplementedError(
        "arming checks passed, but live order transmission is not implemented in this "
        "build — see place_order(). Nothing further will run.")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
