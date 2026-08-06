# TASK_872 — The LIVE candle path resolves timestamps in machine-local time (871 remainder, result-bearing today)

> **STATUS (2026-08-06): SHIPPED as `task872-live-stream-utc`.** Both `fromisoformat` sites in
> `KrakenWebSocket._handle_message`'s V2 OHLC adapter now resolve at UTC; regression suite
> `tools/tests/test_stream_utc.py` 58/58; selftest 12/12; fixture `b0f8de8b…` and
> `trading_data.db` byte-untouched. Medium panel (correctness + regression) over 3 cycles,
> **2 MAJOR found and closed**, fix rounds escalated to opus.
>
> **The fix is larger than the two-line change this row specified, because the naive-datetime fix
> alone would have made a second, pre-existing defect SILENT.** The nanosecond-truncation block
> does `frac_part[:6]` and rebuilds the string without re-appending anything past the slice, which
> DESTROYS a real numeric offset before `fromisoformat` ever sees it — so the new `tzinfo is None`
> guard stamped UTC onto a wall clock that was never UTC. Measured: `2024-07-15T12:34:56.123+05:30`
> → `1721046896.0` against a truth of `1721027096.123`, wrong by exactly the 19,800 s offset, on
> every host. Not confined to the fallback: at ≥6 fractional digits the offset is sliced clean off
> and the PRIMARY branch parses with no exception at all. **The first round's own test asserted
> that wrong value as its oracle and passed** — the defect class the round was sent to close.
> Closed by detaching any offset suffix before truncation and re-appending it at both call sites.
>
> **Pre-existing, verified:** that truncation defect is byte-identical at
> `git show 67c8176:mcts_trader/data/stream.py`, where the same input already returned `1721046896`
> under `TZ=UTC` and `1721039696` under `Europe/Prague`. It was NOT introduced here; it is closed
> here because this diff re-touched that exact block and because 872's own guard is what converted
> it from a naive-local error into a *silent* wrong-UTC stamp.
>
> **The offset-detach scan is POSITIONAL (from index 11), not separator-matched.** A fresh judge
> caught the first attempt anchoring on `max(find('T'), find(' '))`: Python 3.10 accepts a lowercase
> `t` separator, so `_sep == -1`, the detach was skipped, and the silent mis-stamp reproduced
> exactly. Widening to case-insensitive would still have missed `X`/`_` — 3.10 accepts ANY single
> character there. The positional rule is separator-agnostic by construction and structurally cannot
> reach the date's own hyphens at indices 4 and 7. Verified through the real adapter: `T`/`t`/space/
> `X`/`_` all yield `1721027096`, delta 0.0. Sub-12-char timestamps skip the detach deliberately —
> safe by construction, and `2024-07-15+05:30` is a date whose `+` is the SEPARATOR, parsing to
> 05:30 (`1721021400`, unchanged).
>
> **DECISION-CHANGING, for the planner — this row's stated priority is wrong.** The header below
> calls this "the only known defect that is result-bearing on the LIVE path rather than latent".
> The regression judge established that both named consumers are DEAD CODE: `real_trade_jax.py` is
> self-documented "BROKEN / LEGACY … Do not use this file for live/paper trading", its
> `RealPortfolioManager` import fails (`_REAL_PORTFOLIO_AVAILABLE = False`) and `__main__`
> `sys.exit()`s before `RealTrader` is constructed; `PaperExchange.set_data_source` is called
> nowhere (its only caller, `paper_trade_jax.py`, was deleted in `c4a27e0`). `KrakenWebSocket` is
> instantiated only by that dead file and the new test. **The STOP condition is firmly not
> triggered — no persistence is reachable — but the reason is stronger than "decision-path only":
> today's live blast radius is zero.** The fix is correct and worth having for whenever the live
> path is revived; its ranking justification should be restated.
>
> **Accepted, disclosed not fixed:** colon-less offsets (`+0530`) parse in neither branch on 3.10
> and now FAIL LOUD (logged, candle dropped) rather than silently mis-stamping — better than the
> pre-round silent `1721046896`, but not full handling. Sub-second precision is discarded downstream
> regardless (`str(int(ts))`) — pre-existing. `int(ts) + 60` hardcodes a 60 s bar end: checked
> CONSISTENT (`stream.py:50,75` and `real_trade_jax.py:268` all pin `interval=1`), but it breaks on
> a non-1m subscription.
>
> **Spec correction (binding):** Acceptance (iii) below assumes a nanosecond-precision payload is
> what reaches the `ValueError` fallback. That is empirically FALSE, and was false pre-fix too — no
> well-formed `Z`-or-offset payload at any fractional length reaches it (always exactly one
> `fromisoformat` call). Both sites are covered and proven fixed, but the fallback is exercised only
> by malformed input. The wording should be amended where authored.

**Status**: READY 2026-08-05 (seeded from the run-25 wrap, question 1 — planner: YES, take it).
**Size: XS. Risk: low** (a two-line semantics fix plus its regression test; no numerics on any
committed baseline). Surface: `mcts_trader/data/stream.py` (+ its regression test).
**Priority: Ready #1 — this is the only known defect in the program that is result-bearing on
the LIVE path rather than latent.**

## Why (planner-verified 2026-08-05, on top of the 871 banner remainder)

`stream.py:145-146` strips a confirmed `Z` from Kraken's `interval_begin`, and `:157`/`:162`
then call `datetime.fromisoformat(ts_str).timestamp()` on the now-NAIVE string. Python resolves
a naive datetime's `.timestamp()` in **machine-local** time, so every live candle's epoch is
shifted by the host's UTC offset — **−7200 s on this workstation (CEST)**, a different number
on `gpu-server` or after a DST transition. Downstream, `:168-169` writes that epoch as the
candle's Time/EndTime, so live bar bucketing and any alignment against historical (UTC) bars
are off by the offset: 2 bars on hourly, 120 on minute.

Consumers confirmed live, not hypothetical: `real_trade_jax.py:42,374`
(`KrakenWebSocket(on_ohlc=self.handle_ohlc)`) and the paper simulator
(`mcts_trader/exchange/paper.py:80`).

**Scoping fact, planner-measured — this is NOT a source-of-record contamination path:**
`real_trade_jax.handle_ohlc` only updates the in-memory `market_state` dict
(`real_trade_jary.py:304-314` region — no INSERT/execute on any window inspected), and the
paper simulator likewise does not persist. So **no historical rows in `trading_data.db` are
implicated and no data cleanup is in scope** — the damage is confined to live/paper decision
alignment. (If verification finds ANY persistence path from this callback, that is a STOP and a
finding: it would mean shifted rows reached the source of record.)

Python here is **3.10.14**, where `fromisoformat` cannot parse `Z` — which is why the strip
exists. The fix must therefore attach UTC explicitly rather than rely on 3.11 parsing.

## Scope

1. **Make the conversion timezone-correct**: after parsing, if the result is naive, attach
   `timezone.utc` before `.timestamp()` (Kraken `interval_begin` is documented UTC); if the
   string carried a real offset, `fromisoformat` already returns an aware datetime and it must
   be left alone. Applies to BOTH conversion sites (`:157` primary and `:162` fallback) — the
   fallback is the one a nanosecond-precision payload actually takes, so fixing only the first
   would leave the live path broken.
2. **Regression test**: adapt one recorded V2 OHLC payload (with `Z` and with 9-digit
   fractional seconds, i.e. both branches) under at least `TZ=UTC`, `TZ=Europe/Prague`,
   `TZ=Pacific/Chatham` (non-hour offset) and a DST-interior date — assert the produced epoch
   is IDENTICAL across zones and equals the hand-computed UTC epoch. Follow the 869/871 pattern:
   the probe must be shown to FAIL against the pre-fix code (a HEAD control), so it is not
   vacuous.
3. **Sweep the module**: any other naive `.timestamp()` / `fromisoformat` on this adapter's
   surface (ticker, trade, book handlers) gets the same treatment or an explicit
   "intentionally-local" note. Report what was found either way.

## Acceptance (gates)

- (i) TZ-invariance proven across the four zones, with a differing pre-fix control.
- (ii) The epoch equals the independently computed UTC value (not merely self-consistent).
- (iii) Both conversion sites covered — demonstrate the fallback branch is exercised by a
  nanosecond payload.
- (iv) Selftest green; no committed baseline moves (this path has none — if any artifact
  changes, STOP and report).

## Out of scope

`tools/orchestration/train_manager.py:321,330` (same class, but that file is marked
"Deprecated — DO NOT EDIT", `tools/orchestration/CLAUDE.md:19`); the `fromisoformat` →
strict `%Y-%m-%d` narrowing in `load_benchmark_periods` (871 remainder, unreachable today);
`mcts_trader/data/database.py`'s live-write RW handle (870 remainder — needs a dual-connection
refactor, its own row); any change to live trading behavior beyond the timestamp semantics.
