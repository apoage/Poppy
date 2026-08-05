# TASK_872 — The LIVE candle path resolves timestamps in machine-local time (871 remainder, result-bearing today)

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
