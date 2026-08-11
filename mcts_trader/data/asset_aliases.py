"""TASK_868 (DECISIONS D31, opt (a1)) — asset identity repair: normalized-alias resolution +
fail-loud union loading across twin ``assets`` rows. TASK_877 changes exactly two things: the
normalization gains an explicit Kraken ticker-alias TABLE (``XBT``->``BTC``, ``XDG``->``DOGE`` —
see "TICKER ALIASES" below), and the agreement gate gains an EVIDENCE FLOOR
(``MIN_ALIAS_OVERLAP_TS``) so a union it never got to test refuses instead of passing silently.

THE DEFECT: the 858 minute backfill ingested 13 series a SECOND time under spelling-variant
``assets`` rows (``/USDTEUR`` id 1016 vs ``USDTEUR`` id 1144, twins added 2025-12-12), splitting
one feed's minute history across two ``asset_id``s. On the 861 leg-1 frame 6 of 64 slots
(/USDTEUR, /USDTCHF, /USDC/USD, /USDCGBP, /USDC/USDT, /USDTAUD) consequently read near-empty
2025-03-31 -> 2025-10-01 while the twin id carries 128k-368k real rows spanning exactly that
gap — and in the measured overlap the two closes agree with max relative diff 0.00 (same feed,
split key; independently confirmed by the run-23 orchestrator and again for this task).

THE REPAIR (loader-level, NO DB WRITES): at load time, resolve each requested symbol to the SET
of ``assets`` rows whose NORMALIZED form matches, and union their rows into one series:

  * NORMALIZATION is slash-stripping PLUS a known Kraken ticker-alias fold (TASK_877; see
    "TICKER ALIASES" below): ``/USDTEUR`` == ``USDTEUR``, and now also ``XBT/CHF`` ==
    ``BTC/CHF`` == ``XBTCHF`` == ``BTCCHF``.
  * AGREEMENT GATE (fail-loud): where aliased series overlap in time, any close disagreement
    (rel > 1e-12) on a shared timestamp REFUSES with both asset_ids and the first disagreeing
    timestamp named — never average, never prefer silently.
  * EVIDENCE FLOOR (fail-CLOSED, TASK_877): a union whose two live ids share FEWER than
    ``MIN_ALIAS_OVERLAP_TS`` timestamps has not been TESTED by the gate above, so it refuses
    instead of blessing it. See ``alias_agreement_gate`` for the exact rule, the threshold's
    derivation, and what the gate does and does not prove.
  * DEDUPE: an agreeing duplicate timestamp is written twice into the same panel cell. No write
    ORDER is imposed or claimed (the loaders' streaming query carries no ``ORDER BY``), because
    the gate makes order immaterial: every pair it admits agrees to <= 1e-12 relative, ~5 orders
    below float32's ~1.19e-7 spacing, so both candidates round into the same float32 cell (worst
    case, if they straddle a rounding boundary, one ULP) — and on every pair measured to date the
    two closes are float64-EQUAL (max rel diff exactly 0.0), i.e. literally the same number.
  * CANONICAL ID ORDER: the exact-spelling id first, then remaining ids ascending
    (``resolve_requested_symbols``). Deterministic, and it is what the disclosure records — it is
    NOT a claim about who wins an overlapping timestamp, because per the DEDUPE bullet nobody
    needs to win one.
  * DISCLOSED: every consumer records per-slot ``asset_ids`` used + per-id row counts in the
    artifact (the D31 "disclosed" rider).

TICKER ALIASES (TASK_877): this module's normalization was ORIGINALLY (868) slash-only, and its
docstring reasoned that folding ``XBT``->``BTC`` would wrongly merge ``BTCCHF`` (id 1085) with
``XBTCHF`` (id 1175) as "DIFFERENT feeds". THAT REASONING WAS MEASURED WRONG: over their 48,780
shared timestamps the two closes agree to max relative diff 0.0 — the identical D31 signature —
and ``XBTCHF`` carries the real rows inside BTCCHF's supposed "genuine sparse" gap
2025-03-31->2025-10-01. GATE (i), MEASURED IN BOTH WINDOW CONVENTIONS (2026-08-11, direct query
— this repo resolves window dates at UTC midnight since TASK_869 / D31 (a3), and the task text's
own numbers were LOCAL):
    UTC   2025-03-31..2025-10-01: BTCCHF alone 667 rows -> union 92,637 (XBTCHF 92,636, of
                                  which 666 timestamps are shared)
    LOCAL (the task text, and the pre-877 prose): BTCCHF alone 697 -> union 92,639
Both say the same thing — the "genuine sparse gap" is ~99.3% recoverable — and the difference is
ONLY the convention. The local member of each pair came from a naive
``datetime.strptime(...).timestamp()`` in ``runs/868_asset_identity_repair/harvest_gate_i.py``,
the one never-migrated diagnostic that still carried the pre-869 defect; TASK_877 migrated it,
and it now agrees with the loaders cell for cell. Every number in the shipped suites is UTC. The
full before/after registration lives in ``experiments/865_minute_frame_pin/
smoke_minute_frame_pin.py`` ("TASK_877 RE-REGISTRATION"), which is measured through the suite's
own UTC path. Kraken spells some instruments with legacy ticker codes
(``XBT`` for bitcoin, ``XDG`` for dogecoin) that do not match the spelling used elsewhere in
``assets.symbol`` for the SAME instrument. ``TICKER_ALIASES`` folds ONLY these known, explicit
tokens — table-driven, never a blanket ``str.replace`` across the whole symbol (that would
wrongly fold e.g. ``AIXBT/EUR`` id 78, a DIFFERENT, unrelated ticker that merely CONTAINS "xbt",
or ``XBTPY/USD`` id 1090, whose base token is ``XBTPY`` not ``XBT``): folding is TOKEN-EXACT (see
``normalize_symbol_key`` / ``_fold_ticker_aliases``). Full-``assets``-table sweep (TASK_877, both
the real DB and the fixture, every one of the 1202-1205 rows checked, not sampled): exactly ONE
new merge with rows on BOTH sides — ``BTCCHF``/``XBTCHF`` — matching this task's own scope-2
claim; ``XDG``->``DOGE`` creates NO merge at all today (checked, not assumed: no ``DOGE``-spelled
row exists in either database, so the fold only relabels the pre-existing ``XDG/USD``+``XDGUSD``
pair's key and touches zero other rows). ``XDG``->``DOGE`` IS THEREFORE A FORWARD DECLARATION —
a rule that folds nothing until a ``DOGE``-spelled row is ingested — and it is kept ONLY because
the evidence floor below makes such a rule unable to fire unreviewed: the day that row arrives it
either shows >= ``MIN_ALIAS_OVERLAP_TS`` agreeing shared timestamps against the ``XDG`` rows (the
same signature that identified every twin this repo has folded) or the load REFUSES and a human
looks. Without that floor the entry would be exactly the unreviewed auto-fold it is here to
prevent, and the right call would be to delete it.

BOUNDARY, DISCLOSED: Kraken's OTHER legacy spelling — the REST asset-pair codes ``XXBTZ<FIAT>``
(``XXBTZUSD`` id 1139, ``XXBTZEUR``, ...) — is deliberately NOT folded. ``_fold_ticker_aliases``
matches whole leading tokens, and ``XXBTZUSD`` starts with ``XX``, not ``XBT``; the remainder
``ZUSD`` is not a quote code either, so BOTH of its conditions miss. Those rows carry no data on
either database today, so the boundary costs nothing now — but it IS a boundary: if they are ever
backfilled they will read as separate series until a future task measures them and extends the
table. ``runs/877_xbt_alias_fold/test_xbt_alias_fold.py`` pins ``XXBTZUSD`` as MUST-NOT-FOLD so
the boundary cannot move silently.

GUARD KIND (repo convention: a normalizer looks identical to a corrector — state which this is).
This module ships both, and they are deliberately DIFFERENT functions: ``normalize_symbol_key`` /
``_fold_ticker_aliases`` are the NORMALIZER — they rewrite two spellings of what the alias table
asserts is the SAME instrument onto one identity key, so downstream grouping treats them as one
series. ``alias_agreement_gate`` is the CORRECTOR that sits downstream of it: it never rewrites a
price, never averages, never silently prefers one side — it either lets the union proceed or
REFUSES, naming both asset_ids and the first disagreeing timestamp. A disagreement means the
normalizer's "same instrument" premise was wrong for that group, and the only safe response is to
stop, not to guess. TASK_877 CHANGED THE CORRECTOR in exactly one way — it added the evidence
floor (``MIN_ALIAS_OVERLAP_TS``), because a corrector that passes an untested group as
confidently as a measured one is not a guard. What it does and does not prove is stated on
``alias_agreement_gate`` / ``_identity_evidence``: agreement is necessary, never sufficient.

APPLIED BY, EXACTLY (all of them read the MINUTE ``ohlcv`` table): the framed minute loader
``load_minute_prices_framed`` and its row-axis mirror ``load_row_timestamps`` (both in
``experiments/835_stage4_eval/eval_vector.py``), plus the canon unpinned minute loaders
``VectorBTTradingEnv._load_database_prices`` / ``._load_database_prices_for_symbols``.

NOT APPLIED BY THE DAILY LOADERS — stated plainly because 868 shipped the opposite claim in this
spot and the first TASK_877 pass carried it forward. ``eval_vector.load_daily_prices``,
``eval_vector.load_daily_prices_framed`` and ``eval_vector._close_by_date`` each resolve their
symbols with a literal ``SELECT a.id, a.symbol FROM assets a WHERE a.symbol IN (...)`` — an EXACT
``assets.symbol`` string match. ``resolve_requested_symbols`` / ``alias_agreement_gate`` /
``normalize_symbol_key`` are never invoked on that path. (The repair does reach a daily run in ONE
narrow place — UNIVERSE SELECTION: with no ``--symbols``, ``load_daily_prices`` ranks over the
MINUTE table via ``_minute_ranked_symbols`` -> ``ranked_minute_universe``, which IS grouped. That
chooses which SPELLINGS get requested; it does not change how the daily rows behind them resolve.)

CONSEQUENCE, DISCLOSED. On the daily path a slot whose REQUESTED spelling carries no
``ohlcv_daily`` rows reads INACTIVE even when its alias twin holds history. MEASURED LIVE
(2026-08-11, real DB, the shipped 64-symbol frame ``runs/832_e5/frame_provenance_imprint.json``),
slot 11: ``/USDC/USD`` id 996 has **0** ``ohlcv_daily`` rows, while ``USDCUSD`` id 1145 — same
``normalize_symbol_key`` group, and one of the six named 868 split slots — has **721** daily rows
spanning 2024-07-15..2026-07-05. ``load_daily_prices_framed`` therefore records that slot
``{"status": "inactive", "reason": "zero ohlcv_daily coverage in window"}``, while the MINUTE path
unions the identical slot correctly (116,685 shared timestamps, max rel diff 0.0). It is the ONLY
slot of the 64 in that state (all checked, not sampled). This PRE-DATES TASK_877 — TASK_868 never
wired the daily loaders in — and 877 neither introduces nor fixes it: wiring them is a SEPARATE
task row, deliberately not done here, because it would move daily-pinned numerics.

WHAT DOES HOLD ON ``ohlcv_daily`` — re-measured by direct query for this fix pass, not carried
forward on trust (2026-08-11, every duplicate group of BOTH databases): in EVERY group **at most
ONE member carries ``ohlcv_daily`` rows at all** — 0 groups with two daily-live members, 28 with
exactly one, over all 56 real-DB / 57 fixture groups. So IF the daily loaders were ever wired,
the union there would be a PURE RELABEL: no two series blended, every group's daily column exactly
one asset_id's rows under a second spelling, and nothing for the agreement gate to compare (all
daily overlaps: 0 shared timestamps — including 996/1145's). Read that claim for exactly what it
says: "no blending" is NOT "the requested spelling reaches the data", and the ``/USDC/USD``
miscount above is precisely the gap between the two. (The pre-877 wording here — "only ``USDCUSD``
id 1145 (721 rows) and ``USDGUSD`` id 1182 (609)" carry ``ohlcv_daily`` rows — was WRONG, and was
already wrong when 868 shipped it: 28 groups have a daily-carrying member, among them ``XDG/USD``
id 1102 with 2,391 daily rows, a member of the very group ``XDG``->``DOGE`` renames.) TASK_877
changes none of it: ``BTCCHF`` id 1085 and ``XBTCHF`` id 1175 both carry ZERO daily rows.

Table-wide duplicate sweep (D31 scope 3) is ``duplicate_sweep`` below: read-only, report-only.
"""
from __future__ import annotations

import sqlite3

# The agreement gate's tolerance. Overlapping closes of the same feed agree to float64-exact 0.0
# on every measured pair (125,737 shared rows on the largest pair); 1e-12 admits float dust only.
ALIAS_AGREEMENT_REL_TOL = 1e-12

# TASK_877 — the agreement gate's EVIDENCE FLOOR: the fewest shared timestamps a pair of live
# aliased ids must have before their union is allowed to proceed. WHY 64, exactly:
#   * BELOW every currently-shipped fold, with an order of magnitude to spare, so it cannot
#     retroactively refuse one (measured 2026-08-11, every duplicate group of both databases,
#     not sampled): the smallest overlap any BOTH-LIVE pair has is 720 shared timestamps on the
#     fixture (TWXUSD ids 90001/90002) and 10,657 on the real DB (USDGUSD ids 1001/1182);
#     the TASK_877 pair BTCCHF/XBTCHF has 48,780; the largest, USDTEUR, has 125,737.
#     64 is 11x under the binding fixture number and 167x under the real DB's.
#   * ABOVE the vacuous region the floor exists to refuse — 0 shared timestamps (the premise was
#     never tested at all: two spans that never meet) and 1 (a single sample).
# It is deliberately NOT set near the measured minimum: a floor tuned to today's data would turn
# an ordinary coverage change into a refusal. Overridable per call (``min_shared_ts``) only so
# the tests can drive it red/green; no shipped caller passes it.
MIN_ALIAS_OVERLAP_TS = 64

# Unbounded window for the full-history evidence query — the same SQL as the in-window checks,
# with bounds that cannot exclude a row (SQLite integers are 64-bit).
_TS_MIN, _TS_MAX = -(2 ** 63), 2 ** 63 - 1


class AliasDisagreementError(RuntimeError):
    """Two aliased asset_ids carry DIFFERENT closes for the same timestamp. Never averaged,
    never silently preferred — the load refuses."""


class AliasInsufficientOverlapError(RuntimeError):
    """Two aliased asset_ids BOTH carry rows, but they share too few timestamps for the
    agreement gate to have tested the "same instrument" premise their union rests on
    (< ``MIN_ALIAS_OVERLAP_TS``). Fail-CLOSED: the load refuses rather than silently unioning
    two series on unmeasured identity. Distinct from ``AliasDisagreementError``, which means the
    premise WAS tested and FAILED; this one means it was never really tested."""


# TASK_877 — known Kraken ticker aliases: two DB spellings for the SAME instrument. An explicit,
# closed TABLE (not a blanket string rewrite, gate iii): keys are whole tokens, matched EXACTLY
# by ``normalize_symbol_key`` / ``_fold_ticker_aliases`` below, never as a substring pattern —
# see those functions' docstrings for why ``AIXBT/EUR`` (id 78) and ``XBTPY/USD`` (id 1090) must
# NOT fold even though both contain the literal characters "xbt".
TICKER_ALIASES = {
    # Kraken's native bitcoin ticker. Measured collision: BTCCHF (id 1085) / XBTCHF (id 1175) —
    # 48,780 shared timestamps, max relative close diff 0.0 (same feed, split key; the identical
    # D31 signature). This is the fold this task exists to ship.
    "XBT": "BTC",
    # Kraken's native dogecoin ticker. Checked (TASK_877 full-``assets``-table sweep, both real
    # DB and fixture): no ``DOGE``-spelled row exists anywhere today, so this folds zero live
    # rows — a documented no-op, kept explicit for the same reason the table exists at all: the
    # day a ``DOGE...`` row is ingested under that spelling, it unions with the ``XDG...`` rows
    # instead of silently staying split the way BTCCHF/XBTCHF did.
    "XDG": "DOGE",
}

# The quote-currency tokens this DB actually uses — derived from ``assets.symbol``'s own
# slash-delimited and concatenated spellings elsewhere in the table (fiat + BTC + the stable
# dollars this DB carries). Consulted ONLY by ``_fold_ticker_aliases``, to bound where a
# SLASH-LESS alias token's boundary is allowed to fall — see its docstring.
_KNOWN_QUOTE_CODES = frozenset((
    "USD", "EUR", "GBP", "CAD", "AUD", "JPY", "CHF", "AED", "DAI",
    "USDT", "USDC", "USDQ", "USDR", "BTC",
))


def _fold_ticker_aliases(key: str) -> str:
    """Fold a KNOWN alias token (``TICKER_ALIASES``) at the START of a slash-less, upper-cased
    key — ONLY when the token is a genuine PREFIX of ``key`` AND everything after it is an EXACT
    member of ``_KNOWN_QUOTE_CODES``. Both conditions matter for gate iii (never fold a mere
    substring): a bare prefix check alone already survives every substring trap actually present
    in this table (``AIXBT...`` does not START WITH ``XBT`` — the token sits at position 2, not
    0), but the quote-code check is what stops a hypothetical ``XBTxyz`` with no recognized quote
    from folding on a guess instead of a known instrument shape. Returns ``key`` UNCHANGED when
    no alias token matches — the safe default is never to fold.

    DISCLOSED BOUNDARY (module docstring, "BOUNDARY"): Kraken's REST asset-pair spelling
    ``XXBTZ<FIAT>`` (``XXBTZUSD`` id 1139, ...) does NOT fold here and is not meant to — it fails
    BOTH conditions (it starts with ``XX``, and its remainder ``ZUSD`` is not a quote code). Those
    rows are data-empty on both databases today, and the 877 test pins ``XXBTZUSD`` as
    must-not-fold so the boundary cannot move by accident. If they are ever backfilled they will
    read as a separate series until a task measures them and extends ``TICKER_ALIASES``."""
    for alias, canon in TICKER_ALIASES.items():
        if key.startswith(alias) and key[len(alias):] in _KNOWN_QUOTE_CODES:
            return canon + key[len(alias):]
    return key


def normalize_symbol_key(symbol: str) -> str:
    """Normalized match key: slash-stripped, upper-cased, with the known Kraken ticker aliases
    (``TICKER_ALIASES``) folded (TASK_877 — see the module docstring for why). Folding is
    TOKEN-EXACT everywhere, never a blanket ``str.replace`` (gate iii):

      * Exactly one ``/``: split into (base, quote) and fold EACH SIDE independently, only on an
        EXACT dict match, then join. ``XBT/CHF`` and ``BTC/CHF`` both key to ``BTCCHF``;
        ``AIXBT/EUR`` (base ``AIXBT``) and ``XBTPY/USD`` (base ``XBTPY``) do not equal the alias
        key ``XBT`` and are left untouched — a symbol merely CONTAINING the token never folds.
      * No ``/`` (Kraken's concatenated spelling, e.g. ``XBTCHF``): delegates to
        ``_fold_ticker_aliases``, which folds only a LEADING token whose remainder is a
        recognized quote code.
      * Two or more ``/`` (pre-existing ingestion artifacts, e.g. ``/EUROP/EUR`` — none involve
        XBT/XDG, checked): every slash is stripped exactly as the pre-877 slash-only rule always
        did, so these are numerically UNCHANGED by this task.
    """
    s = str(symbol).upper()
    if s.count("/") == 1:
        base, _, quote = s.partition("/")
        return TICKER_ALIASES.get(base, base) + TICKER_ALIASES.get(quote, quote)
    if "/" in s:
        return s.replace("/", "")
    return _fold_ticker_aliases(s)


def group_asset_rows(asset_rows):
    """[(id, symbol), ...] -> {normalized_key: [(id, symbol), ...] sorted by id}."""
    groups = {}
    for aid, sym in asset_rows:
        groups.setdefault(normalize_symbol_key(sym), []).append((int(aid), sym))
    for g in groups.values():
        g.sort()
    return groups


def resolve_requested_symbols(asset_rows, requested_symbols):
    """Resolve each requested symbol to its alias group.

    Returns (resolution, unknown): one entry per requested symbol (order preserved) ::
    {"symbol", "asset_ids" (canonical order: exact-spelling id first, then ascending),
     "db_symbols", "exact_id"}; ``unknown`` lists requested symbols with NO group at all
    (callers decide: the canon loader warns+skips, the framed loaders refuse)."""
    groups = group_asset_rows(asset_rows)
    resolution, unknown = [], []
    for s in requested_symbols:
        g = groups.get(normalize_symbol_key(s))
        if not g:
            unknown.append(s)
            continue
        exact = [aid for aid, sym in g if sym == s]
        exact_id = exact[0] if exact else None
        ids = ([exact_id] if exact_id is not None else []) \
            + [aid for aid, _ in g if aid != exact_id]
        resolution.append({"symbol": s, "asset_ids": ids, "exact_id": exact_id,
                           "db_symbols": [sym for _, sym in g]})
    return resolution, unknown


def ranked_minute_universe(cur, s_ts, e_ts, n_assets):
    """The canon coverage-ranked universe, REPAIRED: per-id in-window minute row counts grouped
    by normalized key and SUMMED, so a split series ranks on its whole history instead of each
    twin ranking separately (D31: "the split slots' coverage ranking is itself distorted today").

    Returns up to ``n_assets`` groups, ordered by unioned count desc (ties: lowest member id —
    deterministic), each ::
    {"symbol" (representative = the in-window max-count member's DB spelling, tie -> lowest id),
     "asset_ids" (ascending), "count", "per_id_counts"}."""
    rows = cur.execute(
        """SELECT a.id, a.symbol, COUNT(*) FROM ohlcv o JOIN assets a ON o.asset_id = a.id
           WHERE o.timestamp >= ? AND o.timestamp <= ? GROUP BY a.id""",
        (s_ts, e_ts)).fetchall()
    groups = {}
    for aid, sym, cnt in rows:
        g = groups.setdefault(normalize_symbol_key(sym), {"members": [], "count": 0})
        g["members"].append((int(aid), sym, int(cnt)))
        g["count"] += int(cnt)
    ordered = sorted(groups.values(),
                     key=lambda g: (-g["count"], min(m[0] for m in g["members"])))
    out = []
    for g in ordered[:int(n_assets)]:
        members = sorted(g["members"], key=lambda m: (-m[2], m[0]))
        out.append({"symbol": members[0][1],
                    "asset_ids": sorted(m[0] for m in members),
                    "count": g["count"],
                    "per_id_counts": {m[0]: m[2] for m in members}})
    return out


def _first_disagreement(conn, table, a, b, s_ts, e_ts, rel_tol):
    """First (lowest-timestamp) shared row in [s_ts, e_ts] where the two ids' closes differ by
    more than ``rel_tol`` relative, or None. (timestamp, close_a, close_b)."""
    return conn.execute(
        f"""SELECT a.timestamp, a.close, b.close FROM {table} a
            JOIN {table} b ON a.timestamp = b.timestamp
            WHERE a.asset_id = ? AND b.asset_id = ?
              AND a.timestamp >= ? AND a.timestamp <= ?
              AND a.close IS NOT NULL AND b.close IS NOT NULL
              AND a.close != 0 AND b.close != 0
              AND ABS(a.close - b.close)
                  / MAX(ABS(a.close), ABS(b.close)) > ?
            ORDER BY a.timestamp LIMIT 1""",
        (a, b, s_ts, e_ts, rel_tol)).fetchone()


def _pair_overlap(conn, table, a, b, s_ts, e_ts):
    """(shared timestamp count, max relative close difference or None) for the two ids over
    [s_ts, e_ts]."""
    n, mx = conn.execute(
        f"""SELECT COUNT(*), MAX(ABS(a.close - b.close)
                  / MAX(ABS(a.close), ABS(b.close))) FROM {table} a
            JOIN {table} b ON a.timestamp = b.timestamp
            WHERE a.asset_id = ? AND b.asset_id = ?
              AND a.timestamp >= ? AND a.timestamp <= ?
              AND a.close IS NOT NULL AND b.close IS NOT NULL
              AND a.close != 0 AND b.close != 0""",
        (a, b, s_ts, e_ts)).fetchone()
    return int(n), (float(mx) if mx is not None else None)


def _carries_rows(conn, table, asset_id):
    """Does this asset_id carry ANY usable close in ``table``, over its WHOLE history? Indexed
    existence probe (LIMIT 1), not a count — this runs on the gate's fallback path only."""
    return conn.execute(
        f"SELECT 1 FROM {table} WHERE asset_id = ? AND close IS NOT NULL AND close != 0 LIMIT 1",
        (asset_id,)).fetchone() is not None


def _identity_evidence(conn, table, a, b, shared_in_window, rel_tol, min_shared_ts):
    """TASK_877 EVIDENCE FLOOR — decide whether this pair's union has been tested enough to be
    allowed, and return the JSON-safe evidence record. Raises rather than bless an untested one.

    The rule, in the order it is applied (cheapest first):

      1. ``shared_ts`` inside the requested window already >= ``min_shared_ts`` -> basis
         ``"window"``. Free: the count is the one the disclosure already computed. This is the
         path every shipped fold takes on a wide window (all 14 real-DB live pairs, and the
         fixture's one live pair, clear the floor in-window on the 865 frame window).
      2. Otherwise, is the union LOAD-BEARING for this pair at all? If either id carries no
         usable close ANYWHERE in ``table`` (not merely in this window), the group is a pure
         relabel — one spelling has rows, the other is an empty ``assets`` stub, nothing is
         blended and nothing can be substituted -> basis ``"relabel_only"``, allowed. (42 of the
         56 real-DB duplicate groups are exactly this; refusing them would refuse 11 of the 18
         aliased frame slots.)
      3. Otherwise BOTH ids carry rows, so the column really is a blend of two feeds. Fall back
         to the pair's WHOLE shared history, because "are these two ids one instrument?" is a
         property of the ids, not of the window someone happened to ask for: a window can
         legitimately contain both sides with no shared minute at all (measured: on
         2025-09-14..2025-12-12 SEVEN shipped twin pairs share exactly 0 timestamps — the twin's
         rows stop where the original's resume). If the full history clears the floor AND agrees
         within ``rel_tol`` -> basis ``"full_history"``, allowed.
      4. Anything else REFUSES, naming both ids and the shared-timestamp count.

    WHAT THIS PROVES, EXACTLY — and what it does not. It proves the union was TESTED: at least
    ``min_shared_ts`` (64 by default) timestamps on which BOTH ids printed, every one of them
    agreeing within ``rel_tol``. It does NOT prove the two ids are the
    same instrument. Agreement is necessary, never sufficient, and on THIS table it is weak
    evidence for a whole class of symbols: two genuinely different pegged pairs (the stablecoin
    crosses that dominate the duplicate groups) sit at ~1.0000 and would agree to 1e-12 over any
    number of shared timestamps. The identity claim itself comes from ``TICKER_ALIASES`` and the
    slash rule — i.e. from a HUMAN asserting the spelling equivalence — and this floor's whole
    job is to guarantee that a fold which has never been measured cannot slip through unnoticed
    on the day the data arrives. A refusal here is a request for review, not a data error."""
    if shared_in_window >= min_shared_ts:
        return {"basis": "window", "shared_ts": int(shared_in_window),
                "min_required": int(min_shared_ts)}
    if not (_carries_rows(conn, table, a) and _carries_rows(conn, table, b)):
        return {"basis": "relabel_only", "shared_ts": int(shared_in_window),
                "min_required": int(min_shared_ts)}
    n_all, mx_all = _pair_overlap(conn, table, a, b, _TS_MIN, _TS_MAX)
    if n_all >= min_shared_ts and mx_all is not None and mx_all <= rel_tol:
        return {"basis": "full_history", "shared_ts": int(shared_in_window),
                "min_required": int(min_shared_ts), "full_history_shared_ts": int(n_all),
                "full_history_max_rel_diff": float(mx_all)}
    bad = _first_disagreement(conn, table, a, b, _TS_MIN, _TS_MAX, rel_tol)
    if bad is not None:
        raise AliasDisagreementError(
            f"alias agreement gate: asset_ids {a} and {b} (same normalized symbol, table "
            f"`{table}`) DISAGREE on shared timestamp {int(bad[0])}: closes {bad[1]!r} vs "
            f"{bad[2]!r} (rel tol {rel_tol}). The disagreement is OUTSIDE the requested window "
            f"— found while falling back to the pair's whole history for the identity evidence "
            f"this window does not carry ({shared_in_window} shared timestamp(s) in window). "
            f"Refusing to average or silently prefer one. Resolve the assets rows first.")
    raise AliasInsufficientOverlapError(
        f"alias evidence floor: asset_ids {a} and {b} (same normalized symbol, table `{table}`) "
        f"BOTH carry rows, so their union really does blend two series — but they share only "
        f"{n_all} timestamp(s) in their WHOLE history ({shared_in_window} inside the requested "
        f"window), under the {min_shared_ts}-timestamp floor "
        f"(asset_aliases.MIN_ALIAS_OVERLAP_TS). The agreement gate has therefore not TESTED the "
        f"'same instrument' premise this union rests on, and a blessing it cannot justify is "
        f"worse than a stop: refusing (fail-closed). Confirm the two `assets` rows name one "
        f"instrument (and widen the window if the evidence exists outside it), or remove the "
        f"alias that groups them.")


def alias_agreement_gate(conn, table, id_groups, s_ts, e_ts, rel_tol=ALIAS_AGREEMENT_REL_TOL,
                         min_shared_ts=MIN_ALIAS_OVERLAP_TS):
    """Fail-loud agreement gate, checked ONCE per load (not per streamed row): for every alias
    group with >1 id, every pair's shared-timestamp closes in [s_ts, e_ts] must agree within
    ``rel_tol`` (relative). A disagreement REFUSES with both asset_ids and the first disagreeing
    timestamp named. Returns a JSON-safe per-group disclosure of the checked overlaps.

    TASK_877 adds the EVIDENCE FLOOR (``min_shared_ts``): agreement over ZERO shared timestamps
    used to pass exactly as confidently as agreement over 48,780, which made the gate silent
    precisely where it knew least. A pair that really blends two feeds and cannot show
    ``min_shared_ts`` agreeing timestamps now REFUSES (``AliasInsufficientOverlapError``, naming
    both ids and the count). ``_identity_evidence`` holds the rule and the derivation; each
    pair's disclosure carries an ``evidence`` record saying which basis let it through.

    GUARD KIND: CORRECTOR, not normalizer (see the module docstring). It never averages a close,
    never silently prefers one asset_id's price over the other's, and never repairs a group; it
    either passes a group through untouched or RAISES. ``normalize_symbol_key`` is this module's
    NORMALIZER — the function that ASSERTS two spellings name the same instrument; this gate is
    the check on that assertion.

    WHAT IT PROVES: that on every timestamp where both ids printed inside the window, they
    printed the same close, and (877) that there were enough such timestamps for that to mean
    something. WHAT IT DOES NOT PROVE: that the two ids ARE one instrument. Agreement is
    necessary, not sufficient — two different pegged pairs both sitting at 1.0000 agree
    perfectly forever. It also says nothing about timestamps only one side printed, which is
    exactly where the union's value (and its risk) lives: the gate cannot see a substitution on
    a span only one id covers. Treat a PASS as "no contradiction found", never as a proof."""
    disclosure = []
    for group in id_groups:
        ids = [int(i) for i in group]
        if len(ids) < 2:
            continue
        pairs = [(ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids))]
        overlap = []
        for a, b in pairs:
            bad = _first_disagreement(conn, table, a, b, s_ts, e_ts, rel_tol)
            if bad is not None:
                raise AliasDisagreementError(
                    f"alias agreement gate: asset_ids {a} and {b} (same normalized symbol, "
                    f"table `{table}`) DISAGREE on shared timestamp {int(bad[0])}: closes "
                    f"{bad[1]!r} vs {bad[2]!r} (rel tol {rel_tol}). An ingest identity split is "
                    f"only union-safe where the two feeds are the SAME series; refusing to "
                    f"average or silently prefer one. Resolve the assets rows first.")
            n, mx = _pair_overlap(conn, table, a, b, s_ts, e_ts)
            overlap.append({"pair": [a, b], "shared_ts": n, "max_rel_diff": mx,
                            "evidence": _identity_evidence(conn, table, a, b, n, rel_tol,
                                                           min_shared_ts)})
        disclosure.append({"asset_ids": ids, "overlaps": overlap})
    return disclosure


def real_bar_mask(raw_nz, n_steps, step):
    """(n_steps, ...) bool — did this slot print a REAL close anywhere inside panel row r's BAR?
    ``raw_nz`` is the 1-min-resolution non-zero mask (1-D or 2-D with rows on axis 0). Panel row
    r is the ``[::step]`` sample at 1-min index ``r*step``, so its bar is the ``step`` minutes
    ``((r-1)*step, r*step]`` (row 0 is the single minute 0). Shared by the canon loader
    (env_factory, on the raw mask captured before the fill) and eval_vector's framed minute
    loader (its private ``_real_bar_mask`` delegates here — see its docstring for why the
    exact-minute reading is wrong on real data)."""
    import numpy as np
    nz = np.asarray(raw_nz, dtype=bool)
    if int(step) <= 1:
        return nz[:int(n_steps)]
    n_full = nz.shape[0]
    cs = np.concatenate([np.zeros((1,) + nz.shape[1:], dtype=np.int64),
                         np.cumsum(nz, axis=0, dtype=np.int64)], axis=0)  # cs[i] = prints in [0,i)
    r = np.arange(int(n_steps), dtype=np.int64)
    hi = np.minimum(r * int(step), n_full - 1)
    lo = np.where(r == 0, 0, (r - 1) * int(step) + 1)
    return (cs[hi + 1] - cs[lo]) > 0


def duplicate_sweep(db_path, timeout_s=60):
    """D31 scope 3 — the table-wide duplicate sweep. READ-ONLY (mode=ro URI), report-only:
    every normalized-duplicate ``assets`` group with per-id row counts + spans in BOTH price
    tables and the pairwise overlap agreement. Fix nothing — this is the measurement the wrap
    cites. Returns a JSON-safe dict."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=timeout_s)
    try:
        rows = conn.execute("SELECT id, symbol FROM assets").fetchall()
        groups = group_asset_rows(rows)
        dups = {k: v for k, v in groups.items() if len(v) > 1}
        out = {"db": str(db_path), "normalization": "upper() + slash-stripped + known Kraken "
               "ticker-alias fold (TASK_877: XBT->BTC, XDG->DOGE; token-exact — see "
               "asset_aliases.normalize_symbol_key)",
               "n_assets_rows": len(rows), "n_duplicate_groups": len(dups), "groups": []}
        for key in sorted(dups):
            members = []
            for aid, sym in sorted(dups[key]):
                rec = {"asset_id": aid, "symbol": sym}
                for table in ("ohlcv", "ohlcv_daily"):
                    n, mn, mx = conn.execute(
                        f"SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM {table} "
                        f"WHERE asset_id = ? AND close IS NOT NULL AND close != 0",
                        (aid,)).fetchone()
                    rec[table] = {"rows": int(n), "first_ts": mn, "last_ts": mx}
                members.append(rec)
            ids = [m["asset_id"] for m in members]
            overlaps = []
            for table in ("ohlcv", "ohlcv_daily"):
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        n, mx = conn.execute(
                            f"""SELECT COUNT(*), MAX(ABS(a.close - b.close)
                                      / MAX(ABS(a.close), ABS(b.close))) FROM {table} a
                                JOIN {table} b ON a.timestamp = b.timestamp
                                WHERE a.asset_id = ? AND b.asset_id = ?
                                  AND a.close IS NOT NULL AND b.close IS NOT NULL
                                  AND a.close != 0 AND b.close != 0""",
                            (ids[i], ids[j])).fetchone()
                        overlaps.append({"table": table, "pair": [ids[i], ids[j]],
                                         "shared_ts": int(n),
                                         "max_rel_diff": (float(mx) if mx is not None else None)})
            n_live = sum(1 for m in members
                         if m["ohlcv"]["rows"] or m["ohlcv_daily"]["rows"])
            out["groups"].append({
                "normalized_key": key, "members": members, "overlaps": overlaps,
                "classification": ("empty_stub_pair" if n_live <= 1 else "live_twin_pair")})
        return out
    finally:
        conn.close()
