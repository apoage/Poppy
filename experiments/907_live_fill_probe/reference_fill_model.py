"""TASK_907 — the declared REFERENCE fill model under test.

This is an INDEPENDENT reimplementation of the rule the 806->809->897 lineage documents in
`experiments/897_corrected_809/sweep_gated_maker.py` ("OHLC-touch: limit posted at close(d),
fills iff bar d+1's range touches it, plus a flat adverse-selection bps penalty on filled
notional"). It is deliberately NOT imported from that lineage — the whole point of this probe
is an independent measurement of the assumption that code embodies, so a latent defect in the
897 implementation cannot silently launder itself into this probe's "ground truth" via a shared
import. If this file's touch rule disagrees with 897's someday, that disagreement is itself a
finding, not a bug to reconcile away by importing one from the other.

This module holds ONLY the rule being measured (what counts as a touch, what the model assumes
the adverse-selection cost is). It does not decide anything, place any order, or read the
capture format directly.
"""
from __future__ import annotations

from dataclasses import dataclass

# The model's own flat adverse-selection assumption, in bp, as carried by the 897 grid
# (`GRID_ADV_BP` in sweep_gated_maker.py covers {0, 5, 10, ...}; the three the row's framing
# names explicitly are reproduced here for the artifact's side-by-side comparison).
MODEL_ADVERSE_SELECTION_BP_TIERS = (0.0, 5.0, 10.0)


@dataclass(frozen=True)
class VirtualLimit:
    side: int          # +1 = buy, -1 = sell
    price: float        # posted at the bar's close
    post_ts: float       # bar-close time the limit was posted


def touch_fills(limit: VirtualLimit, next_bar_high: float, next_bar_low: float) -> bool:
    """The reference OHLC-touch rule: a resting limit posted at close(d) is deemed filled iff
    bar d+1's range touches or crosses it.

    A BUY limit rests below or at the market and fills if price trades down to it
    (next_bar_low <= price). A SELL limit rests above or at the market and fills if price
    trades up to it (next_bar_high >= price).
    """
    if limit.side == 1:
        return next_bar_low <= limit.price
    if limit.side == -1:
        return next_bar_high >= limit.price
    raise ValueError(f"side must be +1 or -1, got {limit.side}")


def trade_crosses(limit: VirtualLimit, trade_prices: "list[float]") -> bool:
    """A stronger fill proxy than OHLC-touch: true iff a REAL captured trade printed at or
    through the limit price (not merely a bar high/low, which can be a synthesized extreme
    the shadow engine derives from the same trades — this uses the raw prints directly).

    Still an OPTIMISTIC bound: it says nothing about whether our hypothetical resting size
    would have been ahead of or behind that print in the real order queue.
    """
    if limit.side == 1:
        return any(p <= limit.price for p in trade_prices)
    if limit.side == -1:
        return any(p >= limit.price for p in trade_prices)
    raise ValueError(f"side must be +1 or -1, got {limit.side}")
