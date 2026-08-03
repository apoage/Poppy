#!/usr/bin/env python3
"""TASK_865 — fixture smoke for the MINUTE frame pin (``--panel-frame-source minute``).

Proves the task's acceptance gates on ``fixtures/trading_fixture.db`` (CPU, ~80-iteration
imprints). The ONE leg that reads ``trading_data.db`` is acceptance (iii), it is OPT-IN
(``--real-db-coverage``), it opens the file ``mode=ro`` through a URI, and it runs a single
read-only COUNT query — no panel build, no training, no GPU.

  (i)   PARITY (lite) — an UNPINNED run and a DAILY-pinned run carry NO minute-pin keys; the
        minute keys are additive and reachable only under --panel-frame-source minute. (The
        BINDING acceptance-(i) evidence is the byte-level comparison of default / unpinned /
        daily-pinned artifacts against the task864-parity-baseline-hygiene HEAD, run by hand with
        identical invocations; this leg is the cheap in-repo guard that additive stays additive.)
  (ii)  END-TO-END + ASSERTED DISCLOSURE — a collection recorded by ``eval_vector.py
        --panel-frame-from-checkpoint --panel-frame-source minute`` imprints clean under the
        MATCHING pin (supplied via a DIFFERENT path to the same frame), all the way through the
        policy-argmax sanity replay; the inactive-slot / partial-slot / stale-tail disclosures are
        ASSERTED, not merely printed (the 863 lesson). Replay integrity passing is the alignment
        proof: eval's masked minute walk and imprint's masked minute walk score identically.
        SOURCE FIDELITY is proven separately: with the frame set to the coverage-ranked universe
        in rank order, the framed panel is BIT-IDENTICAL to the canon unpinned minute panel
        except on the pre-listing rows the 853 leak-free rule deliberately leaves flat-zero.
        BOTH refusal directions are exercised:
          * IDENTITY SCRAMBLING — different membership, an unknown symbol, a duplicated symbol and
            a wrong-length frame are each refused (the loader NEVER drops or re-indexes a slot,
            unlike VectorBTTradingEnv's ``specific_symbols`` branch, which drops with a warning).
          * SAME-MEMBERSHIP REORDER — the same 8 symbols in a different order is refused.
        Plus the fail-closed matrix: pinned/unpinned mixes both ways, an empty frame path, and
        --panel-frame-source minute WITHOUT a pin (on both runners).
        Plus the TASK_865 SOURCE gate: a daily-pinned collection may not be imprinted under a
        minute pin, or vice versa (D20 — same frame, different price series).
  (iii) COVERAGE HONESTY (opt-in, read-only real DB) — the frame-slot coverage of the minute
        `ohlcv` table on the 861 leg-1 window, under BOTH definitions, with the dead slots NAMED.
  (iv)  DEGENERATE-RUN DETECTOR — the 863 FIX-0 guard fires on the MINUTE-pinned path with
        ``--smoke-assert`` OFF (the real leg's configuration: the default for any non-fixture DB).

Run: ``python experiments/865_minute_frame_pin/smoke_minute_frame_pin.py``
     ``python experiments/865_minute_frame_pin/smoke_minute_frame_pin.py --real-db-coverage``
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(EXP_DIR))
RUN_DIR = os.path.join(EXP_DIR, "smoke_runs")
EVAL_PY = os.path.join(_REPO, "experiments", "835_stage4_eval", "eval_vector.py")
IMPRINT_PY = os.path.join(_REPO, "experiments", "832_linear_imprint", "imprint.py")
FIXTURE_DB = os.path.join(_REPO, "fixtures", "trading_fixture.db")
# The 861 smoke's fresh-init teacher checkpoint (pinned operator = prior_weight 0 + true-env leaf
# + no v-hat tail => net_dependent False, so the weights are never read). Reused, not re-minted.
CKPT_SRC = os.path.join(_REPO, "experiments", "861_search_teacher_imprint", "ckpt_teacher_src")
LEGACY_COLLECTION = os.path.join(_REPO, "experiments", "861_search_teacher_imprint",
                                 "chosen_vectors_fixture.json")

# The MINUTE-pinned fixture panel: the same window/stride/size the 861 legacy panel uses (the
# minute pin has NO --bar-minutes 1440 requirement — that rule belongs to the daily table), so
# the panel is the canon minute one with the universe PINNED instead of coverage-ranked.
# TASK_868 RE-BASE (2026-08): the fixture was regenerated (fixtures/make_fixture.py) from the
# CURRENT trading_data.db — the 858 backfill densified the majors' minute history since the old
# fixture was cut (2025-07-07), so the minute axis moved from 27,181 to 127,602 distinct
# minutes on this window: T = 532 rows, window_size 60, book warm 31 -> start_row 60 ->
# 471 decisions on both legs. ZEC/USD, ETC/USD (and DASH/REP, not in this frame) still die
# 2025-09-30, so the panel now carries LONG forward-filled stale tails.
MINUTE_PANEL = ["--start", "2025-09-14", "--end", "2025-12-12", "--bar-minutes", "240",
                "--n-assets", "8"]
# The DAILY-pinned panel (the shipped 863 configuration) — used by the parity leg and the
# cross-source refusal leg. --bar-minutes 1440 is REQUIRED there. (The daily table's smoke
# windows are UNCHANGED by the regen — daily paths stay bit-identical, gate (v).)
DAILY_PANEL = ["--start", "2025-09-14", "--end", "2025-12-12", "--bar-minutes", "1440",
               "--n-assets", "8"]
COLLECT_POSITION_SIZE_PCT = 0.1          # teacher of record (conditionally exempt under weight)

# 8 slots over the fixture's minute table: 6 symbols with minute coverage in the window and 2
# (`/EURGBP`, `/USDCAD`) with ZERO — interleaved at slots 1 and 4 so a slot shift cannot
# coincidentally line up. `/EURGBP` is ALSO the fixture's one live alias pair (asset_ids
# [370, 1198] — both empty on this window, so the union is a numeric no-op but the D31 (a1)
# disclosure has something to disclose). Three slots list one bar late (PARTIAL heads) and two
# (`ZEC/USD`, `ETC/USD` — minute data ends 2025-09-30) are STALE TAILS: the fixture analogue of
# the real leg's FTM->S / MATIC->POL ticker migrations, which is why the stale-tail disclosure
# exists at all.
FRAME_SYMBOLS = ["XBT/USD", "/EURGBP", "ETH/USD", "LTC/USD",
                 "/USDCAD", "XMR/USD", "ZEC/USD", "ETC/USD"]
EXPECT_INACTIVE_SLOTS = [1, 4]
# Slot 7 lists one bar late (mid-window listing) => PARTIAL: its pre-listing rows are
# flat-zero and force-HOLD. (TASK_869: under UTC window resolution slots 3 and 5's first real
# prints re-bin INTO bar 0, so they are covered from open and no longer partial — measured;
# slot 7 still lists one bar late.) Slots 6 and 7 are the STALE TAILS: their trailing rows are
# FORWARD-FILLED at the last real close (D30 opt (i), PV-exact for VALUATION) and, as of
# TASK_868 (D31 a2), force-HELD for OPENING by the causal open-gate — at decision time a
# delisting and an interior hole are indistinguishable without look-ahead. Slot 7 is BOTH
# partial (head) and stale-tailed; slot 6 is a tail-only slot and is in NEITHER inactive_slots
# NOR partial_slots — it is disclosed on its own axis (stale_tail_slots).
EXPECT_PARTIAL_SLOTS = [7]
EXPECT_STALE_TAIL_SLOTS = [6, 7]
EXPECT_STALE_TAIL_SYMBOLS = ["ZEC/USD", "ETC/USD"]
EXPECT_TAIL_ONLY_SLOTS = [6]               # stale-tailed and NOT partial
EXPECT_T_ROWS = 532
EXPECT_DECISIONS = 471
EXPECT_START_ROW = 60
EXPECT_DEAD_SLOT_CHANNELS = [8, 9, 10, 11]
# The stale tails, MEASURED (fixture, TASK_868 regen; TASK_869 UTC re-measurement): ZEC/USD and
# ETC/USD both print their last real minute close 2025-09-30; under UTC window resolution the
# LAST PANEL BAR carrying a real print is row 102 of 531 (was 103 under local-time resolution —
# the +2h/+1h window shift re-bins the boundary bar), so rows 103..531 are FROZEN (429 rows
# each). Decisions run rows 60..530 (k = row - 60), so decisions k=43..470 (428 cells per slot)
# carry a FROZEN tail price. Under D30 those decisions were TRADEABLE at the frozen price;
# under TASK_868 (D31 a2) they are force-HELD — the WHOLE action (the causal open-gate) —
# while VALUATION still reads the fill: the polarity this suite's stale-tail legs now assert.
EXPECT_STALE_TAIL_LAST_ROW = 102
EXPECT_STALE_TAIL_TRAILING_ROWS = 429
EXPECT_STALE_TAIL_FROZEN_K = list(range(43, 471))
# Per-slot force-HOLD decision cells: 471 each for the two ZERO-CVERAGE slots (inert) + 428
# each for the two stale tails (open-gated) = 1,798 of 3,768.
EXPECT_HOLD_CELLS_PER_SLOT = [0, 471, 0, 0, 471, 0, 428, 428]
EXPECT_TOTAL_HOLD_CELLS = 1798

# ---- FIX PASS: FROZEN-CELL ACCOUNTING on the fixture grid, and the definition it turns on ------
# (TASK_868 RE-BASE: numbers below are re-measured on the regenerated fixture — see MINUTE_PANEL.
#  TASK_869: re-measured again under UTC window resolution — the boundary-bar re-bin moved the
#  stale tails one decision cell each; INERT is unchanged, heads being pre-listing.)
# On THIS fixture frame, MEASURED under the bar rule: 942 INERT + 856 FROZEN (0 interior + 856
# trailing) = 1,798 not real, 1,970 (52.28%) LIVE. NB the pre-868 fixture's "25% undisclosed-frozen"
# was the exact-minute reading of the old sparse grid — the bar rule is the shipped definition, the
# exact-minute count is kept as a labelled DIAGNOSTIC (``no_print_at_sampled_minute_rows``).
EXPECT_FROZEN_INTERIOR_CELLS = 0
EXPECT_FROZEN_TAIL_CELLS = 856
EXPECT_FROZEN_CELLS_TOTAL = 856
EXPECT_LIVE_CELLS = 1970
EXPECT_INERT_CELLS = 942
EXPECT_NOT_REAL_CELLS = 1798
EXPECT_FROZEN_PER_SLOT = {6: 428, 7: 428}                 # interior + trailing, per slot
EXPECT_FROZEN_SLOTS = [6, 7]
# The exact-minute DIAGNOSTIC on the same grid (the panel's number), asserted so the two readings
# are both recorded and can never be silently conflated again. (TASK_869 re-measure: the +2h/+1h
# UTC window shift re-bins every sampled minute — slot 0 now prints on ALL its sampled rows.)
EXPECT_NO_PRINT_AT_SAMPLED_MINUTE_ROWS = {0: 0, 2: 3, 3: 100, 5: 157, 6: 495, 7: 514}

# ---- a SYNTHETIC interior gap, through the REAL loader ----------------------------------------
# The fixture's own minute table has NO multi-bar interior hole under the bar rule, so the interior
# disclosure is proven the way the PV leg proves PV conservation: a purpose-made minute DB with one
# explicit interior gap, read through ``load_minute_prices_framed``. Slot 1 goes dark for bars
# 30..59 and RESUMES at a different price — the re-anchor that makes an interior span categorically
# unlike a trailing one.
GAP_SYMBOLS = ["GAP0/USD", "GAP1/USD"]
GAP_N_BARS = 100
GAP_STEP = 10                       # 10 minutes per bar (keeps the synthetic DB tiny)
GAP_SLOT = 1
GAP_FIRST_DARK_BAR = 30
GAP_LAST_DARK_BAR = 59
GAP_PRE_PRICE = 100.0
GAP_POST_PRICE = 160.0              # +60% re-anchor, delivered in ONE bar

# ---- FIX PASS 2: the CLIPPED grid, and a slot with more spans than the artifact shows ----------
# Two defects that only appear when the decision grid does not run to the end of the panel, or when
# a slot has many sub-gaps — i.e. exactly D31's regime, and exactly what --max-decisions (a
# first-class manifest window key, scorecard.py:959-960) produces on the result-producing path.
#   * INTERIOR-vs-TRAILING used to be decided PER SPAN against the LAST DECISION ROW
#     (``stale_tail and b >= last_dec_row``). Shorten the grid so it ends INSIDE an interior span
#     and the whole span was relabelled TRAILING: the banner printed "0 INTERIOR (a hole that
#     RE-ANCHORS: it CAN fabricate PnL)" over 51 of exactly those cells.
#   * the span list itself used to be truncated to 64 AT THE SOURCE, and the cell count was summed
#     off the truncated list: 200 true frozen rows -> 34 counted decision cells of a true 170, and
#     ``live_decision_cell_pct`` 95.01% against a true 75.07%, with the same record's
#     ``n_frozen_rows`` contradicting it.
# Both are pinned on synthetic minute DBs read through the REAL loader (the PV/gap leg's method).
CLIP_SYMBOLS = ["CLIP0/USD", "CLIP1/USD"]
CLIP_N_BARS = 400
CLIP_STEP = 10
CLIP_SLOT = 1
CLIP_WINDOW = ("2025-01-02", "2025-01-06")
CLIP_WARM = 60
CLIP_PRE_PRICE = 100.0
CLIP_POST_PRICE = 150.0             # +50% re-anchor out of the interior hole
CLIP_DARK = (100, 200)              # interior dark bars (inclusive), slot 1
CLIP_DELIST_BAR = 280               # ... and then it delists for good
EXPECT_CLIP_PANEL_ROWS = 401
EXPECT_CLIP_SPANS = [[100, 200], [281, 400]]
EXPECT_CLIP_LAST_COVERED_ROW = 280
EXPECT_CLIP_FULL_CELLS = (221, 101, 120)     # (total, INTERIOR, trailing) on the full grid
CLIP_SHORT_DECISIONS = 91                    # grid ends at row 150 — INSIDE the interior span
EXPECT_CLIP_SHORT_CELLS = (51, 51, 0)
EXPECT_CLIP_SHORT_CELLS_PREFIX_BUG = (51, 0, 51)   # what the per-span rule reported
# the many-span slot: dark on every ODD panel bar over the same 401-row panel
EXPECT_DENSE_FROZEN_ROWS = 200
EXPECT_DENSE_FROZEN_SPANS = 200
EXPECT_DENSE_FROZEN_CELLS = 170              # odd rows in [60, 401)
EXPECT_DENSE_ARTIFACT_SPANS = 64             # == eval_vector.FROZEN_SPANS_ARTIFACT_MAX

# ---- D30 amendment 6: the VOID bar, kept as data so the reconciliation is checkable -----------
# ``walk_net21 = 11.32152396918238`` was the PASS bar recorded on the ZEROED-tail build, i.e. it
# carries fabricated PnL and D30 declared it VOID. D30 predicted +17.1242 with the tail filled.
# Neither number exists anywhere else in the tree; the (i-b) leg re-derives the value on the shipped
# panel and reports both deltas with the reason each differs.
VOID_WALK_NET21 = 11.32152396918238
D30_PREDICTED_WALK_NET21 = 17.1242
# The PRE-FIX stale-tail boundary (``last_real_full // step``) and the panel's measured
# pure-channel replay numbers under it — HISTORICAL RECORD only as of TASK_868: they belong to
# the OLD fixture panel (T=111) and are not reproducible on the regenerated fixture (T=532).
# The (i-b) leg keeps the structural assertions (the pure valuation channel is SMALL; the masked
# replay is the post-868 semantics) and records fresh numbers.
VOID_LAST_COVERED_ROW = 102
PANEL_MEASURED_FILLED_REPLAY = 14.949671921997371
PANEL_MEASURED_ZEROED_REPLAY = 15.952853341299857
# D30 rider (b): this 8-slot fixture frame is 6/8 covered BY DESIGN (the inactive slots are the
# point of the disclosure legs), which is below the shared default floor — so every leg here opts
# out EXPLICITLY rather than the default being weakened for the real legs.
REQUIRE_ACTIVE_OVERRIDE = ["--require-active-slots", "0"]

# ---- (ii-a4) DAILY-PIN DISCLOSURE PARITY (D30 amendment 4) ---------------------------------
# The SHIPPED daily pin has the mirror defect in DISCLOSURE terms: it forward-fills a mid-window
# delisting to the window end at a frozen price (which D30 opt (i) now sanctions as the semantics
# for both paths, so its NUMERICS do not move) and discloses it in NEITHER inactive_slots NOR
# partial_slots, while the slot stays inside n_active_slots and weight-eligible. Fixture proof, on
# a window that straddles a real delisting in `ohlcv_daily`: FTM/USD's last daily close is
# 2025-03-31, so over 2025-01-01..2025-06-30 (181 daily rows) 90 rows are real and 91 are fabricated.
DAILY_STALE_FRAME = ["XBT/USD", "/EURGBP", "ETH/USD", "LTC/USD",
                     "/USDCAD", "FTM/USD", "ZEC/USD", "ETC/USD"]
DAILY_STALE_SLOT = 5
DAILY_STALE_SYMBOL = "FTM/USD"
DAILY_STALE_WINDOW = ("2025-01-01", "2025-06-30")
# MEASURED under the loader's (now UTC) window convention: _to_epoch(end) is UTC MIDNIGHT of the
# end date and the bound is INCLUSIVE, so 2025-06-30's own daily row (timestamp == midnight) is
# INSIDE: 181 panel rows, 90 real + 91 forward-filled — exactly D30's "91 of 181". (Pre-TASK_869
# local-time resolution put the end bound at local midnight = 23:00Z the previous day, excluding
# that row: 180 rows / 90 trailing. Inclusivity unchanged; only the timezone moved.)
DAILY_STALE_EXPECT_ROWS = 181
DAILY_STALE_EXPECT_TRAILING = 91
# FIX PASS 2 — the DAILY path's frozen/interior disclosure had ZERO coverage. Replacing the whole
# daily ``_frozen_spans`` block (eval_vector.load_daily_prices_framed) with ``pass`` left EVERY leg
# of both suites green: the daily STALE-TAIL half has teeth, the FROZEN/INTERIOR half shipped
# unguarded, and D30 amendment 4 makes the semantics uniform across both pins. Two additions here:
#   * the FTM window above now asserts the daily frozen FIELDS exactly (they are the same span,
#     seen on the frozen axis rather than the stale-tail one);
#   * a second window with real INTERIOR daily holes — the case the daily path had no test for at
#     all. MEASURED on fixtures/trading_fixture.db over 2015-07-20..2015-08-20 (31 daily rows):
#     ETH/USD lists mid-window and then misses 2 interior days; LTC/USD misses 5 across 3 runs.
DAILY_STALE_EXPECT_FROZEN_ROWS = 91
DAILY_STALE_EXPECT_FROZEN_SPANS = [[90, 180]]
DAILY_STALE_EXPECT_INTERIOR_ROWS = 0
DAILY_STALE_EXPECT_REAL_ROWS = 90
DAILY_INTERIOR_WINDOW = ("2015-07-20", "2015-08-20")
DAILY_INTERIOR_EXPECT_ROWS = 32      # TASK_869: UTC end bound includes the end date's own row (was 31)
DAILY_INTERIOR_EXPECT_SPANS = {2: [[20, 21]], 3: [[7, 7], [10, 10], [14, 16]]}
DAILY_INTERIOR_EXPECT_FROZEN_ROWS = {2: 2, 3: 5}
DAILY_INTERIOR_EXPECT_CELLS = (7, 7, 0)      # total, INTERIOR, trailing — over the whole panel

# ---- (0) PV CONSERVATION — the ground-truth invariant (D30 amendment 2) --------------------
# The discriminating assertion the original 19 legs structurally missed: every one of them asserts
# ACTIONS, none asserted PORTFOLIO VALUE, and force-HOLD gates OPENING, not VALUATION. On a FLAT
# price series at fee 0 / slippage 0 no book can make or lose a cent — a long is marked
# ``size * price`` and a short at ``2*size*entry - size*price`` (jax_trading_env.py:246-258), both
# exactly the entry notional when the price never moves — so LONG and SHORT books must BOTH end at
# exactly the initial cash. Any deviation is fabricated PnL, and it is measured, not argued.
# Built on a purpose-made 4-slot synthetic DB (flat closes, one slot delisting mid-panel) read
# through the REAL loader, so it tracks whatever fill rule the loader actually applies.
PV_SYMBOLS = ["PV0/USD", "PV1/USD", "PV2/USD", "PV3/USD"]
PV_FLAT_PRICE = 100.0
PV_N_ROWS = 120
PV_DELIST_SLOT = 2
PV_LAST_REAL_ROW = 89                    # slot 2's last real close; rows 90..119 have NO data
PV_START_ROW = 60                        # == window_size, so decisions run rows 60..118
PV_INITIAL_CASH = 100_000.0
PV_WINDOW = ("2025-01-02", "2025-01-03")

# ---- acceptance (iii): the 861 real leg-1 window + the frame of record --------------------
REAL_DB = os.path.join(_REPO, "trading_data.db")
REAL_FRAME_JSON = os.path.join(_REPO, "runs", "832_e5", "frame_provenance_imprint.json")
REAL_WINDOW = ("2021-01-01", "2025-10-04")
REAL_BAR_MINUTES = 1440                  # the leg-1 stride (TASK_865 "Real leg")
# D30 amendment 5 re-specifies acceptance (iii) from SLOT level to CELL level: the honest number is
# the inert share of the DECISION GRID, not a slot count. MEASURED read-only on trading_data.db over
# the leg-1 window at stride 1440 with warm 60 (decision rows = panel rows - warm):
#   panel rows 1735, decision rows 1675, grid 1675 x 64 = 107,200 cells
#   inert HEAD cells 9,046 + inert TAIL cells 375 = 9,421 = 8.79% of the decision grid
# (D30 quotes ~9.6% / 110,976 because it counted head cells over ALL panel rows, i.e. including the
# 60-row warmup: 9,046 + 59*21 ~ 10,285 ~ its "~10,320", and its tail count 374 ~ 375. Same
# measurement, different row convention — NOT the amendment-5 STOP condition. Both are reported.)
# The gate is a BAND around the measured decision-grid share plus the structural facts: no slot is
# wholly inert, the migrations are stale-tailed, and every stale tail is disclosed.
# FIX PASS — the gate is RE-SPECIFIED onto the HONEST definition and split onto TWO axes.
# The old single number folded head+tail into one band and was BLIND to interior frozen spans by
# construction (it was derived from each slot's MIN/MAX timestamp only). Two consequences, both
# measured: 2,013 interior frozen cells (1.88%) were excluded from the STOP gate entirely, and the
# +/-1.0pp band on a folded total is loose enough that one extra mid-window delisting (~+0.78pp)
# passes unnoticed. So:
#   * INERT (no price at all: pre-listing heads + wholly-uncovered slots) keeps its own band;
#   * FROZEN (forward-filled: interior spans + delisted tails) gets its OWN axis with a TIGHT
#     tolerance, because that is the number a new delisting or a new data hole actually moves;
#   * NOT REAL (inert + frozen) is reported as the headline and banded too.
# MEASURED read-only on trading_data.db over 2021-01-01..2025-10-04 at stride 1440, warm 60
# (panel rows 1734, decision rows 1674, grid 1674 x 64 = 107,136 cells):
#   INERT     9,047  =  8.4444%
#   FROZEN    2,383  =  2.2243%   (2,013 = 1.8789% INTERIOR + 370 = 0.3454% trailing)
#   NOT REAL 11,430  = 10.6687%
#   LIVE     95,706  = 89.3313%
# The interior share sits on 13 slots at ~182 cells each — the 11 near-empty FX/BTC-cross slots the
# panel named (ETHBTC, /USDTEUR, /USDTCHF, /USDC/USD, BTCGBP, BTCCHF, /USDCGBP, /USDC/USDT,
# /USDTAUD, BTCCAD, XRPBTC) plus STX/USD (8) and single cells on BCH/USD and XRP/USD.
# (D30's "~9.6% / 110,976" counted head cells over ALL panel rows incl. the 60-row warmup and had
# no interior term at all; the old 8.79% is this measurement's head+tail under a 1735-row axis.
# Same instrument, different row convention and a missing term — both are reported below.)
# TASK_868 RE-REGISTRATION (gate (i), measured this task through the ALIAS UNION — see
# runs/868_asset_identity_repair/harvest_gate_i.py). Pre-868 registration, kept for the record:
#   INERT 9,047 = 8.4444%   FROZEN 2,383 = 2.2243% (2,013 = 1.8789% INTERIOR + 370 trailing)
#   NOT REAL 11,430 = 10.6687%   LIVE 95,706 = 89.3313%
# Post-868, same instrument, alias-unioned (17/64 frame slots resolve to twin groups; the 6
# span-carriers agree to max rel diff 0.00 over 70k-126k shared timestamps):
#   INERT     9,047  =  8.4444%   (heads are pre-listing — unchanged, as expected)
#   FROZEN    1,381  =  1.2890%   (1,011 = 0.9437% INTERIOR + 370 = 0.3454% trailing, unchanged)
#   NOT REAL 10,428  =  9.7334%   LIVE 96,708 = 90.2666%
# The interior term is now the D31-predicted ~1% residue: the 5 genuine sparse crosses
# (ETHBTC 183, BTCGBP/BTCCHF/BTCCAD/XRPBTC 182 = 911) + /USDC/USD 90 (its twin's own span has
# sub-gaps — the one split slot NOT fully repaired) + STX/USD 8 + BCH/USD 1 + XRP/USD 1.
EXPECT_INERT_DECISION_CELL_PCT = 8.44
EXPECT_INERT_PCT_TOLERANCE = 1.0
EXPECT_FROZEN_DECISION_CELL_PCT = 1.29        # its OWN axis, TIGHT band (RE-REGISTERED post-868)
EXPECT_FROZEN_PCT_TOLERANCE = 0.30            # one extra 186-row delisting = +0.17pp => visible
EXPECT_NOT_REAL_DECISION_CELL_PCT = 9.73
EXPECT_NOT_REAL_PCT_TOLERANCE = 1.0
EXPECT_REAL_INTERIOR_FROZEN_PCT = 0.94        # the term D31 (a1) shrinks 2,013 -> ~1k cells
EXPECT_REAL_INTERIOR_PCT_TOLERANCE = 0.30
# The 6 ingest-split slots D31 (a1) repairs, with the twin ids the union must name. Post-union
# they read as REAL data: 5 of 6 to 0 interior cells; /USDC/USD keeps a 90-cell residue (its
# twin id 1145's own 2025-01-01..2025-10-01 span has sub-gaps) — the residual is NAMED, with a
# small band so the gate stays a STOP on a coverage regression without wobbling on boundary rows.
EXPECT_SPLIT_SYMBOLS = {"/USDTEUR": [1016, 1144], "/USDTCHF": [1015, 1172],
                        "/USDC/USD": [996, 1145], "/USDCGBP": [128, 1160],
                        "/USDC/USDT": [997, 1153], "/USDTAUD": [1013, 1176]}
EXPECT_SPLIT_RESIDUAL_CELLS = {"/USDC/USD": 90}
# The residual interior-frozen slots post-union (>= ~167 cells each): exactly the 5 genuine
# sparse crosses D31 kept. A change in THIS list is a data-coverage event, not a rounding wobble.
EXPECT_INTERIOR_FROZEN_SYMBOLS = ["BTCCAD", "BTCCHF", "BTCGBP", "ETHBTC", "XRPBTC"]
EXPECT_INTERIOR_FROZEN_CELLS_EACH = 182       # each of the 5, measured
# Row counts under the LOADER's own window convention: eval_vector._to_epoch(end) is MIDNIGHT of the
# end date, so 2025-10-04's intraday minutes are OUTSIDE the window -> 2,495,590 distinct minutes ->
# 1734 panel rows -> 1674 decision rows. An end-of-day convention adds ~1499 minutes = one more
# panel row (1735/1675) and one more frozen tail row per stale slot; the cell SHARE is 8.79% either
# way. Tolerance +/-1 row so the boundary convention cannot make this leg brittle.
EXPECT_N_PANEL_ROWS = 1734
EXPECT_N_DECISION_ROWS = 1674
EXPECT_PANEL_ROW_TOLERANCE = 1
REAL_WARM_ROWS = 60                       # EnvConfig.window_size on every 835/832 framed panel
# The two ticker migrations D29/D30 name. Both are ALSO late-listing (MATIC head 77 rows, FTM head
# 457), i.e. partial at BOTH ends — a stale-tail slot must not be assumed head-complete.
EXPECT_MIGRATION_SYMBOLS = ["FTM/USD", "MATIC/USD"]
# There can be MORE stale tails than the two named migrations, and the count is BOUNDARY-SENSITIVE:
# a slot whose last real close sits one strided row short of the axis end (e.g. slot 36 XMR/EUR)
# is a 1-cell stale tail under an end-of-day window convention and not one under the loader's
# midnight convention. So the disclosure rule is "ANY tail of >= 1 panel row" — no minimum-length
# threshold, because a threshold makes silence indistinguishable from absence — and the GATE is a
# BAND plus the structural facts, never "exactly the 2 migrations" (which would raise a FALSE STOP
# on a healthy run). Slots that are one bar short of the end are reported as boundary-sensitive.
EXPECT_STALE_TAIL_SLOT_BAND = (2, 4)
EXPECT_MIN_TAIL_ROWS_DISCLOSED = 1
# Successor tickers of the two migrations. Reported ONLY — splicing a successor into a
# predecessor's slot is a UNIVERSE change needing a planner ruling on cross-ticker splice
# provenance, so this leg looks them up and says so, and does nothing with them.
SUCCESSOR_TICKERS = {"FTM/USD": "S/USD", "MATIC/USD": "POL/USD"}

TEACHER_OPERATOR = [
    "--operator", "search", "--vector-proposal-scheme", "dithered-book",
    "--vector-k", "16", "--sims", "32", "--vector-hold-bias", "3.0",
    "--vector-dither-scale", "1.0", "--vector-proposal-temp", "1.0",
    "--vector-proposal-explore-slots", "2", "--prior-weight", "0.0",
    "--leaf-eval", "true-env", "--leaf-rollout-r", "3", "--no-leaf-vhat-tail",
    "--gumbel-scale", "3.0", "--vector-weight-rebalance",
]


def sh(cmd, log, expect_ok=True):
    t0 = time.time()
    print(f"  $ {' '.join(str(c) for c in cmd[1:])[:170]}")
    p = subprocess.run([str(c) for c in cmd], cwd=_REPO, capture_output=True, text=True)
    text = p.stdout + p.stderr
    with open(log, "w") as fh:
        fh.write(text)
    print(f"    -> exit {p.returncode} in {time.time() - t0:.1f}s  ({os.path.basename(log)})")
    if expect_ok and p.returncode != 0:
        print(text[-3000:])
        raise SystemExit(f"FATAL: expected success but got exit {p.returncode}: {log}")
    return p.returncode, text


def write_frame(path, symbols, note):
    with open(path, "w") as fh:
        json.dump({"frame_symbols": list(symbols), "n_assets": len(symbols),
                   "train_window": ["2025-09-14", "2025-12-12"],
                   "db": "fixtures/trading_fixture.db", "source": note}, fh, indent=1)
    return path


def eval_cmd(out_dir, collect, frame=None, source="minute", panel=None):
    cmd = [sys.executable, EVAL_PY, "--db", FIXTURE_DB, *(panel or MINUTE_PANEL),
           "--position-size-pct", str(COLLECT_POSITION_SIZE_PCT), *TEACHER_OPERATOR,
           "--checkpoint", CKPT_SRC, "--out-dir", out_dir, "--tag", "865fx",
           "--collect-chosen-vectors", collect, *REQUIRE_ACTIVE_OVERRIDE]
    if frame is not None:
        cmd += ["--panel-frame-from-checkpoint", frame]
    if source is not None:
        cmd += ["--panel-frame-source", source]
    return cmd


def imprint_cmd(out_json, ckpt_out, collection, frame=None, source="minute", panel=None,
                iters=250, extra=()):
    # TASK_868 re-base note (measured on the regenerated fixture): the decision grid is now
    # 471x8 with 1,798 force-HOLD cells (47.7% — the two zero-coverage slots + the two long
    # open-gated stale tails; TASK_869's UTC window shift moved the stale tails' last real
    # panel row 103 -> 102, adding one frozen decision cell per tail slot, 1,796 -> 1,798).
    # Collapse band, MEASURED per grid: at the old 80 iters the student collapses to a
    # total-HOLD policy (the degenerate-replay refusal fires, correctly); 150 was the escape
    # on the pre-869 grid but 150 AND 200 collapse on the post-869 one — 250 and 300 are
    # measured HEALTHY (seed 0, this frame/teacher). The band is configuration-dependent by
    # design; the degenerate legs below still exercise the collapse on purpose.
    cmd = [sys.executable, IMPRINT_PY, "--db", FIXTURE_DB, *(panel or MINUTE_PANEL),
           "--vector-weight-rebalance", "--teacher-vectors", collection,
           "--allow-off-pin-teacher", "--iters", str(iters), "--seed", "0",
           "--ckpt-out", ckpt_out, "--out", out_json, *REQUIRE_ACTIVE_OVERRIDE]
    if frame is not None:
        cmd += ["--panel-frame-from-checkpoint", frame]
    if source is not None:
        cmd += ["--panel-frame-source", source]
    return cmd + list(extra)


def _default_require_active(imp):
    """The shared default floor, read from eval_vector so the smoke can never drift from it."""
    return imp._load_eval_module().REQUIRE_ACTIVE_SLOTS_DEFAULT


_EV_MOD = None


def _load_eval_module():
    """eval_vector, by path (cached) — for the legs that need it without an imprint handle."""
    global _EV_MOD
    if _EV_MOD is None:
        _EV_MOD = _imprint_module()._load_eval_module()
    return _EV_MOD


def refusal_leg(name, needle, cmd, log, rule_tokens=()):
    """A leg that MUST exit non-zero with ``needle`` in its output and train nothing."""
    rc, text = sh(cmd, log, expect_ok=False)
    hit = needle in text
    trained = "SMOKE PASS" in text
    rule = ("BOTH legs pin the frame, or NEITHER does" in text
            or "ordered symbol list may not" in text
            or "EMPTY path" in text or "requires --bar-minutes 1440" in text
            or "requires --panel-frame-from-checkpoint" in text
            or "cannot be dropped or" in text
            or "Pass the SAME --panel-frame-source on both legs" in text
            or any(t in text for t in rule_tokens))
    ok = bool(rc != 0 and hit and rule and not trained)
    print(f"    {name:<38} exit {rc} | '{needle[:40]}' {hit} | rule-stated {rule} | "
          f"trained {trained} -> {'REFUSED (PASS)' if ok else 'LEG FAILED'}")
    return {"leg": name, "passed": ok, "exit_code": rc, "needle": needle,
            "needle_present": hit, "rule_stated": rule, "trained": trained,
            "log": os.path.basename(log)}


# =============================================================================================
# in-process legs — the loader / mask / gate exercised directly (seconds, exact cell sets)
# =============================================================================================

def _imprint_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("imprint832_smoke865", IMPRINT_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pin_args(frame_path, source="minute", n_assets=8, bar_minutes=240):
    from types import SimpleNamespace
    return SimpleNamespace(
        db=FIXTURE_DB, start="2025-09-14", end="2025-12-12", bar_minutes=bar_minutes,
        n_assets=n_assets, fee_bps=16.0, slip_bps=5.0,
        position_size_pct=0.5 / n_assets, reward_attributable=True, reward_hybrid=True,
        vector_weight_rebalance=True, vector_weight_unit=-1.0, vector_weight_max=4.0,
        vector_rebalance_deadband=-1.0, seed=0, max_decisions=None,
        allow_off_pin_teacher=True, shuffle_teacher_vectors=False,
        panel_frame_from_checkpoint=frame_path, panel_frame_source=source,
        # D30 rider (b): 6 of 8 slots are covered BY DESIGN here (the inactive slots are what the
        # disclosure legs assert on), which is below the shared default floor -> opt out EXPLICITLY.
        require_active_slots=0,
    )


def _framed_panel(imp, frame_path, source="minute", bar_minutes=240):
    args = _pin_args(frame_path, source=source, bar_minutes=bar_minutes)
    prices_j, logc, env_cfg, frame = imp.load_framed_panel(args)
    T = logc.shape[0]
    corridor = imp._load_corridor()
    _tgt, _pres, warm = corridor.book_target_signs(logc)
    start_row = max(int(warm), int(env_cfg.window_size))
    N = T - 1 - start_row
    return args, prices_j, env_cfg, frame, start_row, N, T


def _write_pv_db(path):
    """A 4-slot synthetic minute DB: FLAT closes everywhere, and slot ``PV_DELIST_SLOT`` stops
    dead after ``PV_LAST_REAL_ROW`` (a delisting / ticker migration, mid-panel). Two tables, the
    fixture's own schema, ~470 rows. Written under the smoke's run dir; never near a real DB."""
    import sqlite3
    from datetime import datetime, timezone
    if os.path.exists(path):
        os.remove(path)
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE assets (id INTEGER PRIMARY KEY, symbol VARCHAR(20) UNIQUE NOT NULL);"
        "CREATE TABLE ohlcv (asset_id INTEGER, timestamp INTEGER NOT NULL, open REAL, high REAL,"
        " low REAL, close REAL, volume REAL, trade_count INTEGER,"
        " PRIMARY KEY (asset_id, timestamp));")
    # TASK_869: UTC, matching the loaders' window resolution (was local time).
    t0 = int(datetime.strptime(PV_WINDOW[0], "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    p = PV_FLAT_PRICE
    for i, sym in enumerate(PV_SYMBOLS):
        con.execute("INSERT INTO assets (id, symbol) VALUES (?, ?)", (i + 1, sym))
        last = PV_LAST_REAL_ROW if i == PV_DELIST_SLOT else PV_N_ROWS - 1
        con.executemany(
            "INSERT INTO ohlcv (asset_id, timestamp, open, high, low, close, volume, "
            "trade_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(i + 1, t0 + 60 * r, p, p, p, p, 1.0, 1) for r in range(last + 1)])
    con.commit()
    con.close()
    return path


def leg_pv_conservation(R, imp):
    """(0) PV CONSERVATION — the D30 ground-truth invariant, asserted FIRST and independently of
    every action-level check in this file.

    Flat prices, fee 0, slippage 0, 4 slots, one slot delisting at panel row 89. A LONG-biased
    book and a SHORT-biased book are walked through the SAME ``walk_env`` + ``frame_hold_mask``
    the imprint pipeline uses. Both MUST end at exactly ``PV_INITIAL_CASH`` to the cent: on a
    constant series a position is worth its entry notional under both marks, so no book can move
    the portfolio by a cent, whatever it holds and whichever slot dies.

    Structural control (proves the leg really exercises the delisting rather than passing
    vacuously): the same two books walked on a panel whose delisted tail has been ZEROED must
    BOTH break the invariant, and in OPPOSITE directions — the long is marked to price 0 (wiped),
    the short pockets a 100% windfall (``2*size*entry - 0``). That is the pre-D30 behaviour, and
    it is the reason force-HOLD cannot fix this: HOLD gates OPENING, not VALUATION."""
    import numpy as np
    print("\n=== (0) PV CONSERVATION — flat prices, fee 0: every book ends at 100,000.00 ===")
    ev = imp._load_eval_module()
    db = _write_pv_db(R("pv_conservation.db"))
    prices_np, syms, row_ts, cov, gate = ev.load_minute_prices_framed(
        db, PV_WINDOW[0], PV_WINDOW[1], PV_SYMBOLS, 1)   # 4/4 slots covered: no override needed
    px = np.asarray(prices_np)
    n_assets = len(PV_SYMBOLS)
    N = PV_N_ROWS - 1 - PV_START_ROW
    cfg = imp.EnvConfig(
        n_assets=n_assets, initial_cash=PV_INITIAL_CASH, fee_bps=0.0, slippage_bps=0.0,
        max_steps=8, window_size=PV_START_ROW, position_size_pct=0.5 / n_assets,
        vector_action=True, vector_weight_rebalance=True, vector_weight_unit=-1.0,
        vector_weight_max=4.0, vector_rebalance_deadband=-1.0)

    def book(code):
        """One unit of target weight on EVERY slot at decision 0, then HOLD — so each slot is
        1/n_assets of the book (25% here) and a wiped slot costs exactly 25% of PV."""
        def fn(k, state):
            return np.full(n_assets, code if k == 0 else 1, dtype=np.int32)
        return fn

    def walk(px_panel):
        pj = imp.jnp.asarray(px_panel, dtype=imp.jnp.float32)
        mask = imp.frame_hold_mask(pj, PV_START_ROW, N)
        out = {}
        for side, code in (("long", 2), ("short", 0)):
            rep = imp.walk_env(pj, cfg, PV_START_ROW, N, book(code), hold_mask=mask)
            out[side] = rep
        return out

    live = walk(px)
    # --- structural control: the pre-D30 zeroed tail, on the SAME books ---
    px_zeroed = px.copy()
    px_zeroed[PV_LAST_REAL_ROW + 1:, PV_DELIST_SLOT] = 0.0
    ctl = walk(px_zeroed)
    # --- TASK_868 gate (iv), first half: PV conservation WITH THE OPEN-GATE MASK ON. The
    # delisted slot's trailing cells are open-gated (no print in the bar), so the full
    # hold_mask (inert|gated) force-HOLDs them — and the invariant must STILL hold to the
    # cent, because the gate touches OPENING only and the books opened at k=0 (a live cell).
    pj = imp.jnp.asarray(px, dtype=imp.jnp.float32)
    mask868 = imp.frame_hold_mask(pj, PV_START_ROW, N, open_gate=gate)
    masked868 = {side: imp.walk_env(pj, cfg, PV_START_ROW, N, book(code), hold_mask=mask868)
                 for side, code in (("long", 2), ("short", 0))}
    n_gated = int((~np.asarray(gate)[PV_START_ROW:PV_START_ROW + N, PV_DELIST_SLOT]).sum())
    cent = lambda v: round(float(v), 2)                                       # noqa: E731
    stale = [c for c in cov if c.get("stale_tail")]
    checks = {
        "panel_is_flat_apart_from_the_delisting": bool(
            (px[:, [i for i in range(n_assets) if i != PV_DELIST_SLOT]] == PV_FLAT_PRICE).all()),
        "delisting_is_disclosed_as_a_stale_tail": (
            [int(c["slot"]) for c in stale] == [PV_DELIST_SLOT]
            and stale[0].get("last_covered_row") == PV_LAST_REAL_ROW),
        "books_really_opened_positions": (live["long"]["avg_long"] > 0
                                          and live["short"]["avg_short"] > 0),
        "no_bankruptcy_either_side": (live["long"]["bankrupt_at"] == -1
                                      and live["short"]["bankrupt_at"] == -1),
        # ---- THE INVARIANT ----
        "LONG_book_ends_at_exactly_initial_cash": cent(live["long"]["final_pv"]) == cent(
            PV_INITIAL_CASH),
        "SHORT_book_ends_at_exactly_initial_cash": cent(live["short"]["final_pv"]) == cent(
            PV_INITIAL_CASH),
        # ---- structural control: the zeroed tail MUST break it, in OPPOSITE directions ----
        "control_zeroed_tail_breaks_LONG_downward": cent(ctl["long"]["final_pv"]) < cent(
            PV_INITIAL_CASH),
        "control_zeroed_tail_breaks_SHORT_upward": cent(ctl["short"]["final_pv"]) > cent(
            PV_INITIAL_CASH),
        "control_error_is_the_delisted_slots_whole_weight": all(
            abs(abs(cent(ctl[s]["final_pv"]) - PV_INITIAL_CASH)
                - PV_INITIAL_CASH / n_assets) < 50.0 for s in ("long", "short")),
        # ---- TASK_868 gate (iv): THE INVARIANT HOLDS WITH THE OPEN-GATE MASK ON ----
        "open_gate_masks_the_delisted_slots_tail_cells": n_gated == (
            (PV_N_ROWS - 1) - (PV_LAST_REAL_ROW + 1)),
        "MASK_ON_long_book_ends_at_exactly_initial_cash": cent(
            masked868["long"]["final_pv"]) == cent(PV_INITIAL_CASH),
        "MASK_ON_short_book_ends_at_exactly_initial_cash": cent(
            masked868["short"]["final_pv"]) == cent(PV_INITIAL_CASH),
    }
    for k, v in checks.items():
        print(f"    {'OK  ' if v else 'FAIL'} {k}")
    print(f"    LOADER PANEL (the shipped fill rule): LONG final_pv "
          f"{live['long']['final_pv']:,.2f}  SHORT final_pv {live['short']['final_pv']:,.2f}  "
          f"(both MUST be {PV_INITIAL_CASH:,.2f})")
    print(f"    MASK ON (TASK_868 open-gate)    : LONG final_pv "
          f"{masked868['long']['final_pv']:,.2f}  SHORT final_pv "
          f"{masked868['short']['final_pv']:,.2f}  (both MUST be {PV_INITIAL_CASH:,.2f}; "
          f"{n_gated} tail cells force-HELD)")
    print(f"    CONTROL, tail zeroed by hand : LONG final_pv "
          f"{ctl['long']['final_pv']:,.2f}  SHORT final_pv {ctl['short']['final_pv']:,.2f}  "
          f"(fabricated: -/+ one slot's whole {100.0 / n_assets:.0f}% weight)")
    return {"leg": "0_pv_conservation", "passed": all(checks.values()), "checks": checks,
            "long_final_pv": round(float(live["long"]["final_pv"]), 2),
            "short_final_pv": round(float(live["short"]["final_pv"]), 2),
            "control_long_final_pv": round(float(ctl["long"]["final_pv"]), 2),
            "control_short_final_pv": round(float(ctl["short"]["final_pv"]), 2),
            "mask_on_long_final_pv": round(float(masked868["long"]["final_pv"]), 2),
            "mask_on_short_final_pv": round(float(masked868["short"]["final_pv"]), 2),
            "open_gated_tail_cells": n_gated,
            "initial_cash": PV_INITIAL_CASH, "n_decisions": N,
            "delisted_slot": PV_DELIST_SLOT, "last_real_row": PV_LAST_REAL_ROW}


def leg_positional_loader(R, imp):
    """The loader keeps EVERY recorded slot POSITIONAL — the contrast with the env loader's
    ``specific_symbols`` branch, which drops an unknown/uncovered symbol with a logging.warning
    and silently re-indexes every slot after it."""
    import numpy as np
    print("\n=== (ii-a) POSITIONAL LOADER — zero-coverage slots stay COLUMNS, never dropped ===")
    fp = write_frame(R("frame_positional.json"), FRAME_SYMBOLS, "TASK_865 positional-loader frame")
    args, prices_j, env_cfg, frame, start_row, N, T = _framed_panel(imp, fp)
    px = np.asarray(prices_j)
    mask = imp.frame_hold_mask(prices_j, start_row, N, open_gate=frame["open_gate"])
    per_slot = [int(x) for x in np.asarray(mask).sum(axis=0)]
    cov = {int(c["slot"]): c for c in frame["coverage"]}
    obs = imp.encode_training_obs(prices_j, start_row, N, env_cfg)
    rep = imp.check_inactive_slot_features(obs, frame["inactive_slots"])
    checks = {
        "panel_is_T532_x_8_with_471_decisions": (T == EXPECT_T_ROWS and px.shape[1] == 8
                                                 and N == EXPECT_DECISIONS
                                                 and start_row == EXPECT_START_ROW),
        "every_recorded_symbol_kept_in_order": (
            list(frame["frame_symbols"]) == FRAME_SYMBOLS),
        "zero_coverage_slots_are_FLAT_ZERO_columns": all(
            not (px[:, s] != 0).any() for s in EXPECT_INACTIVE_SLOTS),
        "inactive_slots_exact": frame["inactive_slots"] == EXPECT_INACTIVE_SLOTS,
        "partial_slots_exact": frame["partial_slots"] == EXPECT_PARTIAL_SLOTS,
        "stale_tail_slots_exact": (
            frame["panel_frame"]["stale_tail_slots"] == EXPECT_STALE_TAIL_SLOTS),
        "stale_tail_symbol_named": all(
            c["symbol"] in EXPECT_STALE_TAIL_SYMBOLS
            for c in frame["panel_frame"]["stale_tail_spans"]),
        # D30 opt (i) made a TAIL-ONLY slot non-'partial' (every row of it carries a
        # forward-filled price), so it is in NEITHER inactive_slots NOR partial_slots and
        # stale_tail_slots is the ONLY place it appears. NB a slot CAN be both — on the 861 real
        # leg both migration slots are late-listing too, and on this fixture slot 7 (ETC/USD)
        # is partial (1-row head) AND stale-tailed; slot 6 (ZEC/USD) is the tail-only case.
        "tail_only_slot_is_NOT_partial_but_IS_stale_disclosed": all(
            s not in frame["partial_slots"]
            and cov[s]["status"] == "active"
            and s in frame["panel_frame"]["stale_tail_slots"] for s in EXPECT_TAIL_ONLY_SLOTS),
        "partial_AND_stale_slot_is_in_both_lists": (
            7 in frame["partial_slots"] and 7 in frame["panel_frame"]["stale_tail_slots"]),
        "hold_cells_per_slot_exact": per_slot == EXPECT_HOLD_CELLS_PER_SLOT,
        # Every partial HEAD is pre-warm (so it costs no decision cell); the partial TAIL is the
        # stale-tail slot and IS inside the decision range — asserted in leg (ii-a3).
        "partial_leading_row_is_before_start_row": all(
            cov[s].get("leading_inactive_rows", 0) < EXPECT_START_ROW
            for s in EXPECT_PARTIAL_SLOTS),
        "panel_frame_source_recorded_minute": (
            frame["panel_frame"]["panel_frame_source"] == "minute"
            and frame["panel_frame"]["bar_minutes"] == 240),
        "obs_finite_no_nan_inf": bool(np.isfinite(obs).all()),
        "dead_slot_features_constant_and_vol_ratio_only": (
            len(rep["checked_slots"]) == 2
            and all(s["constant_across_decisions"] for s in rep["checked_slots"])
            and all(s["nonzero_channels"] == EXPECT_DEAD_SLOT_CHANNELS
                    for s in rep["checked_slots"])),
    }
    for k, v in checks.items():
        print(f"    {'OK  ' if v else 'FAIL'} {k}")
    print(f"    hold cells per slot {per_slot} (expected {EXPECT_HOLD_CELLS_PER_SLOT}); "
          f"stale-tail slot {EXPECT_STALE_TAIL_SLOTS[0]} last real close ts "
          f"{cov[EXPECT_STALE_TAIL_SLOTS[0]]['last_real_ts']} = panel row "
          f"{cov[EXPECT_STALE_TAIL_SLOTS[0]]['last_covered_row']}, "
          f"{cov[EXPECT_STALE_TAIL_SLOTS[0]]['trailing_stale_rows']} trailing rows "
          f"FORWARD-FILLED at that close (VALUATION) and force-HELD for OPENING (D31 a2)")
    return {"leg": "iia_positional_loader", "passed": all(checks.values()), "checks": checks,
            "hold_cells_per_slot": per_slot}


def leg_stale_tail_guard(R, imp):
    """THE STALE-TAIL CONTRACT under D30 opt (i) + TASK_868 (D31 a2), asserted with a negative
    control (the 863 lesson: every guard is shown to FAIL when disabled). The polarity of this
    leg has now inverted TWICE, each time for a measured reason:
      * pre-D30 (zeroed tail): tail cells force-HELD, PV fabricated (the (0) PV leg's control);
      * D30 opt (i) (forward-filled tail): tail cells TRADEABLE at the frozen price — a constant
        tail cannot fabricate PnL;
      * TASK_868 (D31 a2) (forward-filled tail + causal open-gate): tail cells force-HELD for
        OPENING again, but for the NEW reason — the mask is CAUSAL, and at decision time a
        delisting and an interior hole are indistinguishable without look-ahead into whether
        the gap later resumes. VALUATION still reads the fill (PV-exact, the (0) leg with the
        mask ON); the fill itself is UNCHANGED from D30.
    Positive: the tail rows carry the LAST REAL CLOSE forward-filled (boundary asserted both
    ways), the tail is DISCLOSED on its own axis, and the frozen decisions k=44..470 are
    force-HELD while the pre-tail decisions k=0..43 stay openable.
    Negative control: rebuild the mask WITHOUT the open-gate (the D30-only rule) and show the
    frozen decisions become non-held — i.e. this leg really exercises the 868 gate."""
    import numpy as np
    print("\n=== (ii-a3) STALE TAIL — filled for VALUATION, force-HELD (whole action, D31 a2) ===")
    fp = write_frame(R("frame_stale_tail.json"), FRAME_SYMBOLS, "TASK_865 stale-tail frame")
    args, prices_j, env_cfg, frame, start_row, N, T = _framed_panel(imp, fp)
    px = np.asarray(prices_j)
    s = EXPECT_STALE_TAIL_SLOTS[0]
    mask = np.asarray(imp.frame_hold_mask(prices_j, start_row, N,
                                          open_gate=frame["open_gate"]))
    held = sorted(int(k) for k in np.nonzero(mask[:, s])[0])
    last_real_px = float(px[EXPECT_STALE_TAIL_LAST_ROW, s])
    # --- negative control: the D30-only mask (no open-gate) leaves the tail unheld ---
    mask_d30 = np.asarray(imp.frame_hold_mask(prices_j, start_row, N))
    held_d30 = sorted(int(k) for k in np.nonzero(mask_d30[:, s])[0])
    cov = {int(c["slot"]): c for c in frame["coverage"]}
    entry = cov[s]
    tail = px[EXPECT_STALE_TAIL_LAST_ROW + 1:, s]
    checks = {
        # The tail is a single FROZEN value carried to the window end. NB it is the slot's LAST real
        # 1-min close, which need not equal the STRIDED row 102 value (the stride samples one minute
        # per bar, the fill runs at 1-min resolution) — hence "constant and non-zero".
        "tail_rows_are_ONE_frozen_nonzero_value": bool(
            tail.size and (tail == tail[0]).all() and tail[0] > 0),
        "last_real_close_row_is_nonzero": last_real_px > 0,
        "no_tail_row_is_flat_zero": bool((px[EXPECT_STALE_TAIL_LAST_ROW + 1:, s] != 0).all()),
        "trailing_row_count_exact": (
            px.shape[0] - 1 - EXPECT_STALE_TAIL_LAST_ROW == EXPECT_STALE_TAIL_TRAILING_ROWS),
        # TASK_868 (D31 a2): every FROZEN decision is force-HELD (the whole tail, and ONLY it)
        "frozen_decisions_are_force_HELD_exactly": held == EXPECT_STALE_TAIL_FROZEN_K,
        "pre_tail_decisions_stay_openable": all(
            not mask[k, s] for k in range(EXPECT_STALE_TAIL_FROZEN_K[0])),
        # ... and the tail is DISCLOSED on its own axis: the frozen span is named, with its row
        "tail_is_DISCLOSED_in_coverage": (
            entry.get("stale_tail") is True
            and entry.get("last_covered_row") == EXPECT_STALE_TAIL_LAST_ROW
            and entry.get("trailing_stale_rows") == EXPECT_STALE_TAIL_TRAILING_ROWS),
        "tail_is_DISCLOSED_in_the_artifact_block": (
            frame["panel_frame"]["stale_tail_slots"] == EXPECT_STALE_TAIL_SLOTS
            and frame["panel_frame"]["n_stale_tail_slots"] == len(EXPECT_STALE_TAIL_SLOTS)),
        "disclosure_text_says_FORWARD_FILLED_and_PV_exact": all(
            t in frame["panel_frame"]["stale_tail_rule"]
            for t in ("FORWARD-FILLED", "PV-exact", "force-HELD")),
        # --- NEGATIVE CONTROL: without the open-gate (the D30-only rule) nothing is held ---
        "neg_control_D30_only_mask_holds_NOTHING": held_d30 == [],
        "neg_control_proves_the_gate_is_what_holds_them": held_d30 != held,
    }
    for k, v in checks.items():
        print(f"    {'OK  ' if v else 'FAIL'} {k}")
    print(f"    D31 a2 ON: slot {s} force-HELD on k={held[0]}..{held[-1]} ({len(held)} cells); "
          f"rows {EXPECT_STALE_TAIL_LAST_ROW + 1}..{px.shape[0] - 1} frozen at "
          f"{float(tail[0]):.4f} (VALUATION still reads the fill — PV-exact, (0) leg MASK ON)")
    print(f"    control (D30-only mask, no open-gate): slot {s} force-HELD on k={held_d30} — the "
          f"tail was TRADEABLE under D30; the 868 causal gate is what holds it now")
    return {"leg": "iia3_stale_tail_disclosed_filled_masked", "passed": all(checks.values()),
            "checks": checks, "held_decisions": held, "held_decisions_d30_only_mask": held_d30,
            "last_real_row": EXPECT_STALE_TAIL_LAST_ROW, "frozen_price": float(tail[0]),
            "trailing_rows": EXPECT_STALE_TAIL_TRAILING_ROWS}


def _write_gap_db(path):
    """A 2-slot synthetic minute DB with ONE explicit INTERIOR gap. Slot ``GAP_SLOT`` prints every
    minute up to bar ``GAP_FIRST_DARK_BAR``-1, goes completely dark for bars
    ``GAP_FIRST_DARK_BAR``..``GAP_LAST_DARK_BAR``, then RESUMES at a different price. Slot 0 prints
    throughout (so the row axis covers the gap). Read back through the REAL loader, so it tracks
    whatever fill/disclosure rule the loader actually applies."""
    import sqlite3
    from datetime import datetime, timezone
    if os.path.exists(path):
        os.remove(path)
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE assets (id INTEGER PRIMARY KEY, symbol VARCHAR(20) UNIQUE NOT NULL);"
        "CREATE TABLE ohlcv (asset_id INTEGER, timestamp INTEGER NOT NULL, open REAL, high REAL,"
        " low REAL, close REAL, volume REAL, trade_count INTEGER,"
        " PRIMARY KEY (asset_id, timestamp));")
    # TASK_869: UTC, matching the loaders' window resolution (was local time).
    t0 = int(datetime.strptime(PV_WINDOW[0], "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    rows = []
    for sid, sym in enumerate(GAP_SYMBOLS, start=1):
        con.execute("INSERT INTO assets (id, symbol) VALUES (?, ?)", (sid, sym))
        for m in range(GAP_N_BARS * GAP_STEP):
            bar = m // GAP_STEP
            if sid - 1 == GAP_SLOT and GAP_FIRST_DARK_BAR <= bar <= GAP_LAST_DARK_BAR:
                continue                          # DARK: no row at all, the interior gap
            px = (GAP_PRE_PRICE if (sid - 1 != GAP_SLOT or bar < GAP_FIRST_DARK_BAR)
                  else GAP_POST_PRICE)
            rows.append((sid, t0 + m * 60, px, px, px, px, 1.0, 1))
    con.executemany("INSERT INTO ohlcv VALUES (?,?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()
    return path


def _write_clip_db(path, dark_pred, post=None, delist_bar=None):
    """A 2-slot synthetic minute DB on the CLIP grid. Slot 0 prints every minute (it carries the
    row axis); slot ``CLIP_SLOT`` is dark on every panel BAR for which ``dark_pred(bar)`` is True,
    prices at ``CLIP_POST_PRICE`` from ``post`` onward, and stops printing after ``delist_bar``.

    'bar' here is the PANEL row whose bar ``((r-1)*step, r*step]`` contains the minute — the same
    convention ``eval_vector._real_bar_mask`` uses — so a dark bar really is a frozen panel row
    (dropping minutes by ``m // step`` instead leaves the previous bar's prints inside the mask
    window and produces no frozen row at all)."""
    import sqlite3
    from datetime import datetime, timezone
    if os.path.exists(path):
        os.remove(path)
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE assets (id INTEGER PRIMARY KEY, symbol VARCHAR(20) UNIQUE NOT NULL);"
        "CREATE TABLE ohlcv (asset_id INTEGER, timestamp INTEGER NOT NULL, open REAL, high REAL,"
        " low REAL, close REAL, volume REAL, trade_count INTEGER,"
        " PRIMARY KEY (asset_id, timestamp));")
    # TASK_869: UTC, matching the loaders' window resolution (was local time).
    t0 = int(datetime.strptime(CLIP_WINDOW[0], "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    rows = []
    for sid, sym in enumerate(CLIP_SYMBOLS, start=1):
        con.execute("INSERT INTO assets (id, symbol) VALUES (?, ?)", (sid, sym))
        for m in range(CLIP_N_BARS * CLIP_STEP + 1):
            bar = (m + CLIP_STEP - 1) // CLIP_STEP if m else 0
            px = CLIP_PRE_PRICE
            if sid - 1 == CLIP_SLOT:
                if delist_bar is not None and bar > delist_bar:
                    continue
                if dark_pred(bar):
                    continue
                if post is not None and bar >= post:
                    px = CLIP_POST_PRICE
            rows.append((sid, t0 + m * 60, px, px, px, px, 1.0, 1))
    con.executemany("INSERT INTO ohlcv VALUES (?,?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()
    return path


def leg_interior_frozen_spans(R, imp):
    """(ii-a6) FIX PASS, FIRST-CLASS — INTERIOR FROZEN SPANS are measured, disclosed, and counted.

    THE class of defect this fix pass closes: ``_ffill_interior_only`` fills from the first real
    value to the END, so an INTERIOR hole is filled by exactly the same rule as a tail — but
    staleness was derived from the LAST REAL ROW only, so a slot that RESUMED before the window end
    got no ``stale_tail``, no ``partial``, no ``inactive``, and read ``covered_rows == n_rows``. It
    was invisible on every axis. On the 861 leg-1 window that hides 2,013 cells (1.88%) on 11
    near-empty slots, with re-anchor jumps up to +64.52%.

    Two parts, because the fixture alone cannot prove it:
      (1) THE FIXTURE GRID, under the shipped definition (a row is REAL when the slot printed
          anywhere inside its BAR). Asserts the honest cell partition, the span reconstruction
          against an INDEPENDENT raw-table derivation, and that ``covered_rows`` is real rows and
          not the degenerate filled count. ALSO records the exact-minute DIAGNOSTIC — the reading
          under which this same grid is 109 frozen / 191 live — so the two are never conflated.
      (2) A SYNTHETIC INTERIOR GAP through the REAL loader (the PV leg's method): slot 1 goes dark
          for 30 bars and resumes +60% higher. Asserts the span is disclosed as INTERIOR (not a
          tail, not partial, not inactive), that ``frozen_slots`` is the ONLY list it appears in
          (the negative control for the pre-fix behaviour), and that the re-anchor really does
          deliver the whole gap move in ONE bar — the reason an interior span is categorically
          unlike a trailing one.
    It asserts NOTHING about what the env should DO with those cells: tradeability/valuation of an
    interior frozen span is a D30-class semantics ruling (DECISIONS D31), deliberately untouched."""
    import numpy as np
    print("\n=== (ii-a6) INTERIOR FROZEN SPANS — measured, disclosed, counted (FIX PASS) ===")
    ev = _load_eval_module()
    fp = write_frame(R("frame_interior.json"), FRAME_SYMBOLS, "TASK_865 interior-frozen frame")
    args, prices_j, env_cfg, frame, start_row, N, T = _framed_panel(imp, fp)
    px = np.asarray(prices_j)
    cov = {int(c["slot"]): c for c in frame["coverage"]}
    step = 240
    # INDEPENDENT re-derivation straight off the RAW table (not the loader's coverage report), under
    # BOTH readings, so the disclosure is checked against something else and the definition the
    # shipped code uses is pinned by an assertion rather than by trust.
    raw, ts_full = _raw_minute_matrix(FIXTURE_DB, FRAME_SYMBOLS, "2025-09-14", "2025-12-12")
    raw_nz = raw != 0
    n_panel = px.shape[0]
    real_bar = np.zeros((n_panel, len(FRAME_SYMBOLS)), dtype=bool)
    for r in range(n_panel):
        lo = 0 if r == 0 else (r - 1) * step + 1
        hi = min(r * step, raw_nz.shape[0] - 1)
        real_bar[r] = raw_nz[lo:hi + 1].any(axis=0)
    real_exact = raw_nz[::step]
    filled = px != 0
    frozen_indep = filled & ~real_bar
    frozen_exact = filled & ~real_exact
    frozen_from_spans = np.zeros_like(frozen_indep)
    for c in frame["coverage"]:
        for a, b in (c.get("frozen_spans") or []):
            frozen_from_spans[a:b + 1, int(c["slot"])] = True
    fz = ev.frozen_decision_cells(frame["coverage"], start_row, N)
    dec_inert = int((px[start_row:start_row + N] <= 0).sum())
    n_cells = N * len(FRAME_SYMBOLS)
    D = slice(start_row, start_row + N)
    per_slot = {int(c["slot"]): int(c["n_frozen_rows"]) for c in frame["coverage"]
                if c.get("n_frozen_rows")}
    dec_per_slot = {r["slot"]: r["frozen_decision_cells"] for r in fz["slots"]}
    checks = {
        "spans_reconstruct_the_frozen_mask_EXACTLY": bool(
            (frozen_from_spans == frozen_indep).all()),
        "shipped_definition_is_the_BAR_rule_not_the_exact_minute_rule": bool(
            frozen_indep.sum() != frozen_exact.sum()
            and (frozen_from_spans == frozen_indep).all()),
        "n_frozen_rows_matches_independent_count": all(
            int(frozen_indep[:, s].sum()) == per_slot.get(s, 0)
            for s in range(len(FRAME_SYMBOLS))),
        "frozen_decision_cells_per_slot_exact": dec_per_slot == EXPECT_FROZEN_PER_SLOT,
        "frozen_interior_cells_exact": fz["interior"] == EXPECT_FROZEN_INTERIOR_CELLS,
        "frozen_tail_cells_exact": fz["tail"] == EXPECT_FROZEN_TAIL_CELLS,
        "frozen_total_cells_exact": fz["total"] == EXPECT_FROZEN_CELLS_TOTAL,
        "inert_cells_exact": dec_inert == EXPECT_INERT_CELLS,
        "not_real_cells_exact": dec_inert + fz["total"] == EXPECT_NOT_REAL_CELLS,
        "live_cells_exact": n_cells - dec_inert - fz["total"] == EXPECT_LIVE_CELLS,
        "the_three_categories_partition_the_grid": (
            dec_inert + fz["total"] + EXPECT_LIVE_CELLS == n_cells),
        "frozen_slots_disclosed_at_top_level": (
            sorted(frame["panel_frame"]["frozen_slots"]) == EXPECT_FROZEN_SLOTS),
        # the exact-minute DIAGNOSTIC is recorded per slot, so the panel's reading is not lost
        "exact_minute_diagnostic_recorded_per_slot": all(
            cov[s]["no_print_at_sampled_minute_rows"] == v
            for s, v in EXPECT_NO_PRINT_AT_SAMPLED_MINUTE_ROWS.items()),
        # covered_rows is REAL panel rows on both pin paths now (was the degenerate filled count)
        "covered_rows_is_REAL_panel_rows_not_filled": all(
            c["covered_rows"] == c["real_panel_rows"] for c in frame["coverage"]
            if c["status"] != "inactive"),
        "covered_rows_is_NOT_degenerate_on_a_frozen_slot": all(
            cov[s]["covered_rows"] < cov[s]["filled_rows"] for s in EXPECT_FROZEN_SLOTS),
        # the rule text no longer claims a frozen span cannot fabricate PnL in general
        "rule_text_distinguishes_INTERIOR_from_trailing": all(
            t in frame["panel_frame"]["stale_tail_rule"]
            for t in ("INTERIOR", "re-anchor", "cannot fabricate PnL")),
    }
    print(f"    fixture grid {N}x{len(FRAME_SYMBOLS)} = {n_cells}: INERT {dec_inert} + FROZEN "
          f"{fz['total']} ({fz['interior']} INTERIOR + {fz['tail']} trailing) => LIVE "
          f"{n_cells - dec_inert - fz['total']} "
          f"({100.0 * (n_cells - dec_inert - fz['total']) / n_cells:.2f}%)")
    print(f"    DIAGNOSTIC, same grid under the EXACT-MINUTE reading: {int(frozen_exact[D].sum())} "
          f"frozen / {n_cells - dec_inert - int(frozen_exact[D].sum())} live. That reading measures "
          f"minute-bar SPARSITY, not staleness (on trading_data.db it calls 51.22% of the leg-1 "
          f"grid frozen); it is recorded, not gated on.")

    # ---------- (2) the SYNTHETIC interior gap, through the real loader ----------
    gdb = _write_gap_db(R("gap_minute.db"))
    gpx, gsyms, gts, gcov, ggate = ev.load_minute_prices_framed(
        gdb, PV_WINDOW[0], PV_WINDOW[1], GAP_SYMBOLS, GAP_STEP, require_active_slots=0)
    g = next(c for c in gcov if c["slot"] == GAP_SLOT)
    gcol = np.asarray(gpx)[:, GAP_SLOT]
    # The slot has NO row for minutes [GAP_FIRST_DARK_BAR*step, (GAP_LAST_DARK_BAR+1)*step). Bar b
    # stands for minutes ((b-1)*step, b*step], so bar GAP_FIRST_DARK_BAR still catches the prints
    # from the minutes just before the gap opens: the FROZEN bars are GAP_FIRST_DARK_BAR+1 ..
    # GAP_LAST_DARK_BAR, and bar GAP_LAST_DARK_BAR+1 catches the first post-gap print (the
    # re-anchor). Derived here rather than hardcoded so the expectation follows the definition.
    dark = list(range(GAP_FIRST_DARK_BAR + 1, GAP_LAST_DARK_BAR + 1))
    spans = g.get("frozen_spans") or []
    jump = float(gcol[dark[-1] + 1]) / float(gcol[dark[-1]]) - 1.0
    checks.update({
        "GAP_slot_has_a_disclosed_frozen_span": bool(spans),
        "GAP_span_is_exactly_the_dark_bars": spans == [[dark[0], dark[-1]]],
        "GAP_span_is_classified_INTERIOR_not_tail": (
            g.get("frozen_interior_rows") == len(dark) and not g.get("stale_tail")),
        # THE negative control for the pre-fix behaviour: under the OLD rule this slot was in NONE
        # of the three lists, so it could not be found at all.
        "GAP_slot_is_in_NO_pre_existing_list": (
            GAP_SLOT not in [c["slot"] for c in gcov if c["status"] == "inactive"]
            and GAP_SLOT not in [c["slot"] for c in gcov if c["status"] == "partial"]
            and not g.get("stale_tail")),
        "GAP_slot_IS_in_frozen_slots": bool(g.get("n_frozen_rows")),
        "GAP_dark_bars_carry_the_PRE_gap_price": bool(
            all(abs(float(gcol[b]) - GAP_PRE_PRICE) < 1e-6 for b in dark)),
        # ... and this is WHY an interior span is not a trailing one: the whole move lands in 1 bar
        "GAP_reanchor_delivers_the_whole_move_in_ONE_bar": abs(
            jump - (GAP_POST_PRICE / GAP_PRE_PRICE - 1.0)) < 1e-6,
        "GAP_covered_rows_excludes_the_dark_bars": (
            g["covered_rows"] == g["n_rows"] - len(dark)),
        # TASK_868 (D31 a2): the loader's open-gate masks exactly the dark bars (and nothing
        # else) on the interior span — the causal mask the re-anchor repro is gated by.
        "GAP_open_gate_masks_exactly_the_dark_bars": bool(
            (~np.asarray(ggate)[:, GAP_SLOT]).sum() == len(dark)
            and all(not ggate[b, GAP_SLOT] for b in dark)
            and bool(ggate[dark[0] - 1, GAP_SLOT]) and bool(ggate[dark[-1] + 1, GAP_SLOT])),
    })

    # ---------- (3) FIX PASS 2 — the CLIPPED grid and the elided span list ----------
    # (3a) an INTERIOR span must stay INTERIOR when the decision grid is shortened. This is the
    # ONE axis DECISIONS D31 turns on, and --max-decisions is on the result-producing path.
    cdb = _write_clip_db(R("clip_minute.db"), lambda b: CLIP_DARK[0] <= b <= CLIP_DARK[1],
                         post=CLIP_DARK[1] + 1, delist_bar=CLIP_DELIST_BAR)
    cpx, csyms, cts, ccov, cgate = ev.load_minute_prices_framed(
        cdb, CLIP_WINDOW[0], CLIP_WINDOW[1], CLIP_SYMBOLS, CLIP_STEP, require_active_slots=0)
    cc = next(c for c in ccov if c["slot"] == CLIP_SLOT)
    n_full_dec = int(cpx.shape[0]) - CLIP_WARM
    fz_full = ev.frozen_decision_cells(ccov, CLIP_WARM, n_full_dec)
    fz_short = ev.frozen_decision_cells(ccov, CLIP_WARM, CLIP_SHORT_DECISIONS)
    trip = (fz_full["total"], fz_full["interior"], fz_full["tail"])
    trip_s = (fz_short["total"], fz_short["interior"], fz_short["tail"])
    # the PRE-FIX rule, re-implemented here so the leg is a real discriminator and not a
    # restatement of the shipped code: classify per SPAN against the LAST DECISION ROW.
    last_dec = CLIP_WARM + CLIP_SHORT_DECISIONS - 1
    old_tot = old_it = old_tl = 0
    for a, b in cc["frozen_spans"]:
        n = max(0, min(int(b), last_dec) - max(int(a), CLIP_WARM) + 1)
        if not n:
            continue
        old_tot += n
        if cc.get("stale_tail") and int(b) >= last_dec:
            old_tl += n
        else:
            old_it += n

    # (3b) a slot with 200 frozen spans: the COUNT is off the complete list, the ARTIFACT list is
    # elided, and feeding the elided list back into the accounting is a hard STOP.
    ddb = _write_clip_db(R("dense_minute.db"), lambda b: b % 2 == 1)
    dpx, dsyms, dts, dcov, dgate = ev.load_minute_prices_framed(
        ddb, CLIP_WINDOW[0], CLIP_WINDOW[1], CLIP_SYMBOLS, CLIP_STEP, require_active_slots=0)
    dc = next(c for c in dcov if c["slot"] == CLIP_SLOT)
    fz_dense = ev.frozen_decision_cells(dcov, CLIP_WARM, int(dpx.shape[0]) - CLIP_WARM)
    dcov_art = ev.coverage_for_artifact(dcov)
    dc_art = next(c for c in dcov_art if c["slot"] == CLIP_SLOT)
    try:
        ev.frozen_decision_cells(dcov_art, CLIP_WARM, int(dpx.shape[0]) - CLIP_WARM)
        elided_refused = False
    except RuntimeError:
        elided_refused = True
    n_dense_cells = (int(dpx.shape[0]) - CLIP_WARM) * len(CLIP_SYMBOLS)
    checks.update({
        "CLIP_panel_and_spans_are_the_expected_shape": (
            int(cpx.shape[0]) == EXPECT_CLIP_PANEL_ROWS
            and cc["frozen_spans"] == EXPECT_CLIP_SPANS
            and cc.get("last_covered_row") == EXPECT_CLIP_LAST_COVERED_ROW
            and cc.get("stale_tail") is True),
        "CLIP_full_grid_split_exact": trip == EXPECT_CLIP_FULL_CELLS,
        # THE assertion: shorten the grid so it ends INSIDE the interior span and the cells stay
        # INTERIOR. Under the per-span rule they all became TRAILING.
        "CLIP_interior_STAYS_interior_on_a_shortened_grid": trip_s == EXPECT_CLIP_SHORT_CELLS,
        "CLIP_the_PRE_FIX_rule_really_did_mislabel_them": (
            (old_tot, old_it, old_tl) == EXPECT_CLIP_SHORT_CELLS_PREFIX_BUG),
        "CLIP_the_two_rules_disagree_so_this_leg_discriminates": (
            (old_it, old_tl) != (fz_short["interior"], fz_short["tail"])),
        # (3b) elision is display-only
        "DENSE_slot_has_the_expected_frozen_load": (
            dc["n_frozen_rows"] == EXPECT_DENSE_FROZEN_ROWS
            and dc["n_frozen_spans"] == EXPECT_DENSE_FROZEN_SPANS),
        "DENSE_in_memory_span_list_is_COMPLETE": (
            len(dc["frozen_spans"]) == EXPECT_DENSE_FROZEN_SPANS),
        "DENSE_cell_count_is_off_the_COMPLETE_list": (
            fz_dense["total"] == EXPECT_DENSE_FROZEN_CELLS),
        "DENSE_live_pct_is_honest": abs(
            100.0 * (n_dense_cells - fz_dense["total"]) / n_dense_cells - 75.0733) < 0.01,
        "DENSE_artifact_span_list_is_elided_AND_SAYS_SO": (
            len(dc_art["frozen_spans"]) == EXPECT_DENSE_ARTIFACT_SPANS
            and dc_art.get("frozen_spans_truncated") is True
            and dc_art.get("n_frozen_rows") == EXPECT_DENSE_FROZEN_ROWS),
        "DENSE_counting_from_the_ELIDED_list_is_a_HARD_STOP": elided_refused,
        "DENSE_elision_is_flagged_in_the_cell_report": all(
            r["frozen_spans_truncated"] for r in fz_dense["slots"]),
    })
    for k, v in checks.items():
        print(f"    {'OK  ' if v else 'FAIL'} {k}")
    print(f"    CLIP grid: slot {CLIP_SLOT} spans {cc['frozen_spans']}, last_covered_row "
          f"{cc.get('last_covered_row')} — full grid ({n_full_dec} dec) {trip[0]} cells = "
          f"{trip[1]} INTERIOR + {trip[2]} trailing; clipped to {CLIP_SHORT_DECISIONS} dec "
          f"{trip_s[0]} = {trip_s[1]} INTERIOR + {trip_s[2]} trailing "
          f"(PRE-FIX rule: {old_tot} = {old_it} INTERIOR + {old_tl} trailing)")
    print(f"    DENSE slot: {dc['n_frozen_rows']} frozen rows in {dc['n_frozen_spans']} spans -> "
          f"{fz_dense['total']} decision cells, "
          f"{100.0 * (n_dense_cells - fz_dense['total']) / n_dense_cells:.2f}% live; artifact "
          f"shows {len(dc_art['frozen_spans'])} spans and flags the elision; recomputing cells "
          f"from the elided list is refused = {elided_refused}")
    print(f"    synthetic gap: slot {GAP_SLOT} dark on bars {dark[0]}..{dark[-1]} "
          f"({len(dark)} frozen at {GAP_PRE_PRICE}), re-anchors {jump:+.2%} in ONE bar at bar "
          f"{dark[-1] + 1}; disclosed as frozen_spans={spans}, stale_tail={bool(g.get('stale_tail'))}")
    for r in fz["slots"]:
        print(f"      fixture slot {r['slot']:>2} {r['symbol']:<10} {r['frozen_decision_cells']:>3} "
              f"frozen decision cells ({r['interior']} interior + {r['tail']} trailing) over "
              f"{r['n_frozen_spans']} span(s)")
    return {"leg": "iia6_interior_frozen_spans", "passed": all(checks.values()), "checks": checks,
            "n_decision_cells": n_cells, "n_inert": dec_inert, "n_frozen": fz["total"],
            "n_frozen_interior": fz["interior"], "n_frozen_tail": fz["tail"],
            "n_live": n_cells - dec_inert - fz["total"],
            "exact_minute_diagnostic_frozen_cells": int(frozen_exact[D].sum()),
            "frozen_decision_cells_per_slot": dec_per_slot,
            "frozen_panel_rows_per_slot": per_slot,
            "synthetic_gap": {"span": spans, "n_dark_bars": len(dark),
                              "reanchor_jump_pct": round(100.0 * jump, 4),
                              "interior_rows": g.get("frozen_interior_rows"),
                              "stale_tail": bool(g.get("stale_tail"))},
            # fix pass 2
            "clipped_grid": {"spans": cc["frozen_spans"],
                             "last_covered_row": cc.get("last_covered_row"),
                             "full_grid_decisions": n_full_dec,
                             "full": {"total": trip[0], "interior": trip[1], "tail": trip[2]},
                             "short_grid_decisions": CLIP_SHORT_DECISIONS,
                             "short": {"total": trip_s[0], "interior": trip_s[1],
                                       "tail": trip_s[2]},
                             "short_under_the_PRE_FIX_per_span_rule": {
                                 "total": old_tot, "interior": old_it, "tail": old_tl}},
            "dense_span_slot": {"n_frozen_rows": dc["n_frozen_rows"],
                                "n_frozen_spans": dc["n_frozen_spans"],
                                "n_spans_in_memory": len(dc["frozen_spans"]),
                                "n_spans_in_artifact": len(dc_art["frozen_spans"]),
                                "frozen_decision_cells": fz_dense["total"],
                                "live_decision_cell_pct": round(
                                    100.0 * (n_dense_cells - fz_dense["total"]) / n_dense_cells, 4),
                                "elided_list_refused_by_the_accounting": elided_refused}}


def _raw_daily_matrix(db, symbols, row_ts):
    """(raw (T, A) float64) straight off ``ohlcv_daily`` on the loader's OWN returned row axis,
    with NO fill — the daily twin of ``_raw_minute_matrix`` and the independent basis the DAILY
    frozen-span disclosure is checked against (fix pass 2: that half of the daily path had no
    coverage at all). Read-only (mode=ro)."""
    import sqlite3
    import numpy as np
    con = sqlite3.connect(f"file:{os.path.abspath(db)}?mode=ro", uri=True, timeout=60)
    try:
        ph = ",".join("?" * len(symbols))
        sym_to_id = {r[1]: r[0] for r in con.execute(
            f"SELECT id, symbol FROM assets WHERE symbol IN ({ph})", symbols).fetchall()}
        col = {sym_to_id[s]: i for i, s in enumerate(symbols) if s in sym_to_id}
        ts_idx = {int(t): i for i, t in enumerate(row_ts)}
        ids = list(col)
        ph2 = ",".join("?" * len(ids))
        raw = np.zeros((len(row_ts), len(symbols)), dtype=np.float64)
        lo, hi = int(min(row_ts)), int(max(row_ts))
        for aid, t, c in con.execute(
            f"""SELECT asset_id, timestamp, close FROM ohlcv_daily WHERE asset_id IN ({ph2})
                AND timestamp >= ? AND timestamp <= ?""", (*ids, lo, hi)):
            i, j = ts_idx.get(int(t)), col.get(aid)
            if i is not None and j is not None and c:
                raw[i, j] = c
    finally:
        con.close()
    return raw


def _raw_minute_matrix(db, symbols, start, end):
    """(raw (n_full, A) float64, timestamps) straight off the minute table — the loader's row axis
    and slot order, with NO fill. The independent basis the frozen-span disclosure is checked
    against, so the check is not the loader agreeing with itself. Read-only (mode=ro)."""
    import sqlite3
    import numpy as np
    ev = _load_eval_module()
    s_ts, e_ts = ev._to_epoch(start, 0), ev._to_epoch(end, 9999999999)
    con = sqlite3.connect(f"file:{os.path.abspath(db)}?mode=ro", uri=True, timeout=60)
    try:
        ph = ",".join("?" * len(symbols))
        sym_to_id = {r[1]: r[0] for r in con.execute(
            f"SELECT id, symbol FROM assets WHERE symbol IN ({ph})", symbols).fetchall()}
        ids = [sym_to_id[s] for s in symbols if s in sym_to_id]
        # covered ids only — the loader's own row axis
        cov_ids = ev._minute_coverage_ids(db, ids, s_ts, e_ts)
        cov_ordered = [sym_to_id[s] for s in symbols
                       if s in sym_to_id and sym_to_id[s] in cov_ids]
        ph2 = ",".join("?" * len(cov_ordered))
        ts = [r[0] for r in con.execute(
            f"""SELECT DISTINCT timestamp FROM ohlcv WHERE asset_id IN ({ph2})
                AND timestamp >= ? AND timestamp <= ? ORDER BY timestamp""",
            (*cov_ordered, s_ts, e_ts)).fetchall()]
        ts_idx = {t: i for i, t in enumerate(ts)}
        col = {sym_to_id[s]: i for i, s in enumerate(symbols) if s in sym_to_id}
        raw = np.zeros((len(ts), len(symbols)), dtype=np.float64)
        for aid, t, c in con.execute(
            f"""SELECT asset_id, timestamp, close FROM ohlcv WHERE asset_id IN ({ph2})
                AND timestamp >= ? AND timestamp <= ?""", (*cov_ordered, s_ts, e_ts)):
            i, j = ts_idx.get(t), col.get(aid)
            if i is not None and j is not None and c:
                raw[i, j] = c
    finally:
        con.close()
    return raw, ts


def leg_canon_minute_equivalence(R, imp):
    """SOURCE FIDELITY — with the frame set to the coverage-ranked universe IN RANK ORDER, the
    minute-framed panel must BE the canon unpinned minute panel: same row axis, same
    ``[::bar_minutes]`` stride, same interior forward fill. The ONLY licensed divergences are the
    two leak-free rules, and they are the SAME rule at the two ends of a slot's listed span:
      * HEAD (853): the canon loader BACK-fills a slot's pre-listing rows with its first real
        close; the framed loader leaves them flat-zero so they read INACTIVE.
      * TAIL: BOTH loaders forward-fill a delisted slot at its last real close (D30 opt (i) made
        this uniform), so the tail is NOT a divergence — it is asserted IDENTICAL below, and the
        frozen span is disclosed instead of masked.
    Under D30 opt (i) the TAIL divergence is GONE — both loaders forward-fill a delisted slot to the
    window end — so the framed panel now differs from canon ONLY on pre-listing HEAD rows, and in
    every differing cell the canon price is > 0 while the framed price is 0 (never the other way
    round, and never a differing VALUE anywhere from the slot's first real close onward). The pin
    changes the UNIVERSE and the pre-listing region, nothing else."""
    import numpy as np
    import jax
    from mcts_trader.training.env_factory import create_jax_env_historical
    print("\n=== (ii-a2) SOURCE FIDELITY — framed minute panel == canon minute panel ===")
    ev = imp._load_eval_module()
    start, end, bm, n = "2025-09-14", "2025-12-12", 240, 8
    _, px_canon, _ = create_jax_env_historical(
        db_path=FIXTURE_DB, start_date=start, end_date=end, n_envs=1, n_assets=n,
        specific_symbols=None, fee_bps=0.0, slippage_bps=0.0, max_episode_steps=400,
        position_size_pct=0.1, bar_minutes=bm, vector_action=True,
        rng_key=jax.random.PRNGKey(0))
    px_canon = np.asarray(px_canon)
    ranked = ev._minute_ranked_symbols(FIXTURE_DB, start, end, n)
    px_f, syms, ts, cov, gate = ev.load_minute_prices_framed(
        FIXTURE_DB, start, end, ranked, bm, require_active_slots=0)
    dr, dc = np.nonzero(px_canon != px_f)
    # Per slot, the listed span [first real, last real] as PANEL rows (from the coverage record).
    span = {}
    for c in cov:
        s = int(c["slot"])
        if c["status"] == "inactive":
            span[s] = (px_f.shape[0], -1)                    # empty span
        else:                                                # first real close -> window END
            span[s] = (int(c.get("leading_inactive_rows", 0)), px_f.shape[0] - 1)
    # "listed span" now runs from a slot's first real close to the END of the panel (the tail is
    # filled, not masked), so anything at or after span[c][0] must MATCH canon exactly.
    heads = sorted({int(r) for r, c in zip(dr, dc) if int(r) < span[int(c)][0]})
    post_listing_diffs = [(int(r), int(c)) for r, c in zip(dr, dc) if int(r) >= span[int(c)][0]]
    stale_slots = sorted(int(c["slot"]) for c in cov if c.get("stale_tail"))
    tail_cells_identical = all(
        (px_canon[int(cv["last_covered_row"]) + 1:, int(cv["slot"])]
         == px_f[int(cv["last_covered_row"]) + 1:, int(cv["slot"])]).all()
        for cv in cov if cv.get("stale_tail"))
    checks = {
        "same_shape": px_canon.shape == px_f.shape,
        "frame_is_the_ranked_universe_in_order": list(syms) == list(ranked),
        "NO_differing_cell_from_a_slots_first_real_close_onward": not post_listing_diffs,
        "every_differing_cell_is_canon_gt0_and_framed_0": all(
            px_canon[r, c] > 0 and px_f[r, c] == 0
            for r, c in zip(dr, dc)),
        "head_divergence_is_pre_listing_only": all(r < max(span[c][0] for c in span)
                                                   for r in heads),
        # D30 opt (i): the stale tail is no longer a divergence at all.
        "there_IS_a_stale_tail_slot_to_test": bool(stale_slots),
        "stale_tail_cells_are_IDENTICAL_to_canon": tail_cells_identical,
    }
    for k, v in checks.items():
        print(f"    {'OK  ' if v else 'FAIL'} {k}")
    print(f"    {int((px_canon != px_f).sum())} of {px_f.size} cells differ, ALL of them "
          f"pre-listing HEAD rows {heads} (canon back-fills, the framed loader leaves them "
          f"flat-zero = the 853 leak fix). Post-delisting TAIL cells on slots {stale_slots} are "
          f"BIT-IDENTICAL to canon — both loaders forward-fill (D30 opt (i)).")
    return {"leg": "iia2_canon_minute_equivalence", "passed": all(checks.values()),
            "checks": checks, "n_differing_cells": int((px_canon != px_f).sum()),
            "head_rows": heads, "stale_tail_slots": stale_slots, "n_cells": int(px_f.size),
            "n_post_listing_value_diffs": len(post_listing_diffs)}


def leg_require_active_slots(R, imp):
    """D30 rider (b) — the BREADTH refusal, MUTATION-TESTED at the exact configurations that matter.

    Before this, ``load_minute_prices_framed`` refused only when EVERY slot was uncovered, so a
    4-of-8-inert fixture run exited 0 with a publishable ``net@21 +16.996`` and the real
    2021-01-01..2022-01-01 walk-forward fold (50 of 64 slots live) would have run green —
    reintroducing the exact 64->33 breadth confound D29 exists to prevent.

    The guard is exercised directly on synthetic coverage reports (the boundary is arithmetic, so
    it is tested arithmetically rather than by building six panels) AND end to end: the real 6/8
    fixture frame is REFUSED at the default floor and ACCEPTED under the explicit override, and the
    refusal names the flag, both counts and the dead slots."""
    print("\n=== (ii-b0) BREADTH REFUSAL — --require-active-slots at the configurations that matter ===")
    ev = imp._load_eval_module()

    def cov(n_active, n_slots):
        return ([{"slot": i, "symbol": f"S{i}", "status": "active"} for i in range(n_active)]
                + [{"slot": i, "symbol": f"DEAD{i}", "status": "inactive"}
                   for i in range(n_active, n_slots)])

    def guard(n_active, n_slots, req):
        try:
            ev._guard_active_slots(cov(n_active, n_slots), [f"s{i}" for i in range(n_slots)],
                                   req, "2021-01-01", "2025-10-04")
            return None
        except (RuntimeError, ValueError) as e:
            # ValueError too: an unsatisfiable / malformed floor is a RequireActiveSlotsError,
            # which subclasses ValueError + argparse.ArgumentTypeError, not RuntimeError.
            return str(e)

    d = ev.REQUIRE_ACTIVE_SLOTS_DEFAULT
    leg1 = guard(64, 64, d)                    # the 861 leg-1 pin: 64/64 slots carry rows
    wf = guard(50, 64, d) or ""                # the 2021-2022 WF fold: 50/64 live
    fixture = guard(6, 8, d) or ""             # this smoke's frame: 6/8 by design
    checks = {
        "leg1_coverage_64of64_PASSES_at_the_default": leg1 is None,
        "wf_fold_50of64_REFUSES_at_the_default": bool(wf),
        "refusal_names_the_flag": "--require-active-slots" in wf,
        "refusal_states_both_counts": "only 50 of 64" in wf,
        "refusal_names_the_dead_slots": "DEAD50" in wf,
        "refusal_names_the_breadth_confound": "breadth confound" in wf,
        "fixture_frame_6of8_REFUSES_at_the_default": bool(fixture),
        "explicit_zero_DISABLES_the_check": guard(6, 8, 0) is None,
        "explicit_override_below_coverage_PASSES": guard(50, 64, 0.5) is None,
        "absolute_count_form_works": (guard(50, 64, 60) or "").startswith(
            "--require-active-slots 60") and guard(50, 64, 50) is None,
        # the 62/64 minute coverage D29 turns on must NOT be refused by the default
        "62of64_minute_coverage_PASSES": guard(62, 64, d) is None,
        # FIX PASS 2 — an ABSOLUTE floor above the frame's own slot count is REFUSED BY NAME. It
        # used to be CLAMPED (``min(floor, n_slots)``), so ``--require-active-slots 65`` on a
        # 64-slot frame silently became 64: the run was graded against a floor it never asked for
        # while the artifact recorded 65. Both the fully-covered and the under-covered case, so the
        # refusal is about SATISFIABILITY and not just about failing the count.
        "absolute_floor_ABOVE_the_frame_size_is_REFUSED_not_clamped": (
            "can NEVER be satisfied" in (guard(64, 64, 65) or "")
            and "can NEVER be satisfied" in (guard(50, 64, 100) or "")
            and "can NEVER be satisfied" in (guard(8, 8, 9) or "")),
        "an_absolute_floor_EQUAL_to_the_frame_size_still_passes": guard(64, 64, 64) is None,
        # a FRACTION can never be unsatisfiable (ceil(f*n) <= n for f <= 1) and must not be caught
        "fraction_1_point_0_is_NOT_read_as_unsatisfiable": guard(64, 64, 1.0) is None,
    }
    # ... and end to end on the real fixture frame, through the loader and through the CLI.
    fp = write_frame(R("frame_require_active.json"), FRAME_SYMBOLS, "TASK_865 breadth-refusal frame")
    try:
        ev.load_minute_prices_framed(FIXTURE_DB, "2025-09-14", "2025-12-12", FRAME_SYMBOLS, 240)
        checks["loader_default_REFUSES_the_6of8_fixture_frame"] = False
        loader_msg = ""
    except RuntimeError as e:
        loader_msg = str(e)
        checks["loader_default_REFUSES_the_6of8_fixture_frame"] = (
            "--require-active-slots" in loader_msg and "only 6 of 8" in loader_msg)
    ok_px, _, _, _, _ = ev.load_minute_prices_framed(
        FIXTURE_DB, "2025-09-14", "2025-12-12", FRAME_SYMBOLS, 240, require_active_slots=0)
    checks["explicit_override_lets_the_fixture_frame_load"] = ok_px.shape == (EXPECT_T_ROWS, 8)

    # ---- FIX PASS 1: the floor holds on the DAILY pin too -------------------------------------
    # It shipped MINUTE-ONLY, so a daily-pinned run at 5/8 active slots exited 0 with a full
    # scorecard and no ``require_active_slots`` key at all, while the identical frame on the minute
    # source was refused. The real daily pin is 33/64 = 0.52 and flipping the manifest's source to
    # "daily" is a one-word change, so a minute-only floor is no floor.
    daily_msg = ""
    try:
        ev.load_daily_prices_framed(FIXTURE_DB, DAILY_STALE_WINDOW[0], DAILY_STALE_WINDOW[1],
                                    DAILY_STALE_FRAME)
        checks["DAILY_pin_default_REFUSES_an_inert_frame"] = False
    except RuntimeError as e:
        daily_msg = str(e)
        checks["DAILY_pin_default_REFUSES_an_inert_frame"] = (
            "--require-active-slots" in daily_msg and " of 8 frame slots" in daily_msg)
    d_px, _, _, d_cov = ev.load_daily_prices_framed(
        FIXTURE_DB, DAILY_STALE_WINDOW[0], DAILY_STALE_WINDOW[1], DAILY_STALE_FRAME,
        require_active_slots=0)
    checks["DAILY_explicit_override_lets_it_load"] = d_px.shape[1] == 8
    n_daily_active = sum(1 for c in d_cov if c["status"] != "inactive")
    checks["DAILY_frame_really_is_below_the_floor"] = n_daily_active < 8 * d

    # ---- FIX PASS 2: nonsense floors are REFUSED BY NAME, not failed open ---------------------
    # MEASURED before the fix: -1, -0.0, False, 1e-300 and "0" ALL admitted a 50/64 panel while the
    # artifact recorded a floor as though one were in force (``floor = max(1, ceil(req*n))`` collapses
    # them to "at least 1 active slot"), and nan/inf raised a bare ValueError/OverflowError from
    # inside np.ceil instead of a named refusal. None was 'disabled' here and 'the shared default'
    # one function over.
    def bad(v):
        try:
            ev.parse_require_active_slots(v)
            return False
        except ValueError:
            return True

    def guard_refuses_value(v):
        try:
            ev._guard_active_slots(cov(50, 64), [f"s{i}" for i in range(64)], v,
                                   "2021-01-01", "2025-10-04")
            return False                              # admitted a 50/64 panel: FAIL OPEN
        except (ValueError, RuntimeError):
            return True

    # REFUSED outright: each of these used to be admitted (or to blow up unnamed) while a floor was
    # recorded as though one were in force.
    nonsense = [-1, False, True, 1e-300, "x", float("nan"), float("inf"), 1.5]
    # NORMALISED to exactly 0.0 instead: these mean "disabled", and the artifact must record 0.0 —
    # not "-0.0"/"0", which is what made a disabled check look like an active one.
    normalised_to_zero = [0, 0.0, -0.0, "0"]
    checks["nonsense_floors_refused_by_the_validator"] = all(bad(v) for v in nonsense)
    checks["nonsense_floors_cannot_reach_the_guard_and_fail_open"] = all(
        guard_refuses_value(v) for v in nonsense)
    checks["disable_forms_normalise_to_exactly_0_point_0"] = all(
        ev.parse_require_active_slots(v) == 0.0
        and repr(ev.parse_require_active_slots(v)) == "0.0" for v in normalised_to_zero)
    checks["normalised_disable_forms_do_NOT_refuse_a_50of64_panel"] = all(
        not guard_refuses_value(v) for v in normalised_to_zero)
    checks["None_is_REFUSED_by_the_guard_not_read_as_disabled"] = guard_refuses_value(None)
    checks["None_at_args_level_means_the_SHARED_DEFAULT"] = (
        ev._resolve_require_active(_ns(require_active_slots=None)) == d)
    checks["valid_values_still_accepted"] = all(
        not bad(v) for v in (0, 0.5, 1, 1.0, 58, 64.0))
    for k, v in checks.items():
        print(f"    {'OK  ' if v else 'FAIL'} {k}")
    import math
    print(f"    default floor {d} => ceil({d} * 64) = {math.ceil(d * 64)} of 64 slots required; "
          f"leg-1 64/64 passes, WF fold 50/64 refuses, fixture 6/8 refuses")
    print(f"    loader refusal (minute): {loader_msg[:180]}...")
    print(f"    loader refusal (DAILY, {n_daily_active}/8 active): {daily_msg[:180]}...")
    print(f"    REFUSED floor values: {nonsense}")
    print(f"    NORMALISED to 0.0 (disabled, recorded honestly): {normalised_to_zero}")
    return {"leg": "iib0_require_active_slots", "passed": all(checks.values()), "checks": checks,
            "default": d, "loader_refusal": loader_msg[:600],
            "daily_loader_refusal": daily_msg[:600], "daily_n_active_slots": n_daily_active,
            "refused_floor_values": [repr(v) for v in nonsense],
            "disable_forms_normalised_to_zero": [repr(v) for v in normalised_to_zero]}


def _ns(**kw):
    from types import SimpleNamespace
    return SimpleNamespace(**kw)


def leg_daily_disclosure_parity(R, imp):
    """(ii-a4) D30 amendment 4 — the DAILY pin's stale tail is DISCLOSED, its numerics unchanged.

    Read-only loader call (seconds, no imprint, no GPU). Asserts, on a fixture window that straddles
    a real `ohlcv_daily` delisting: the trailing rows really are FORWARD-FILLED at the last real
    close (so the numerics are what they always were), the slot keeps status 'active' and stays out
    of BOTH inactive_slots and partial_slots (so the pre-existing fields did not move), and the
    stale tail is nevertheless discoverable through the NAMED additive keys.

    FIX PASS 2 — this leg also owns the DAILY path's FROZEN / INTERIOR disclosure, which had NO
    coverage anywhere in either suite: replacing the whole daily ``_frozen_spans`` block with
    ``pass`` left every leg of both suites green, because only the stale-tail half was ever
    asserted. Two more parts:
      (2) the SAME window, on the FROZEN axis: exact ``frozen_spans`` / ``n_frozen_rows`` /
          ``frozen_interior_rows`` / ``real_panel_rows`` / ``filled_rows``, plus the
          INTERIOR-vs-TRAILING split ``frozen_decision_cells`` derives from them.
      (3) a SECOND fixture window carrying REAL INTERIOR daily holes (ETH/USD 2 rows, LTC/USD 5
          across 3 runs) — the case the daily path was never tested on at all. Spans are checked
          against an INDEPENDENT re-derivation straight off ``ohlcv_daily``, so the check is not
          the loader agreeing with itself."""
    import numpy as np
    print("\n=== (ii-a4) DAILY DISCLOSURE PARITY — a delisted daily slot is disclosed, not moved ===")
    ev = imp._load_eval_module()
    # require_active_slots=0: this frame is 6/8 covered on the daily table BY DESIGN (the inactive
    # slots are the subject of the disclosure legs), and as of the fix pass the D30 rider (b) floor
    # applies to the DAILY pin too — so this leg opts out EXPLICITLY, like every other fixture leg.
    px, syms, ts, cov = ev.load_daily_prices_framed(
        FIXTURE_DB, DAILY_STALE_WINDOW[0], DAILY_STALE_WINDOW[1], DAILY_STALE_FRAME,
        require_active_slots=0)
    entry = next(c for c in cov if c["slot"] == DAILY_STALE_SLOT)
    stale = [c for c in cov if c.get("stale_tail")]
    inactive = [c["slot"] for c in cov if c["status"] == "inactive"]
    partial = [c["slot"] for c in cov if c["status"] == "partial"]
    last = int(entry.get("last_covered_row", -1))
    col = px[:, DAILY_STALE_SLOT]
    checks = {
        "window_has_the_expected_row_count": px.shape[0] == DAILY_STALE_EXPECT_ROWS,
        "slot_is_the_expected_symbol": syms[DAILY_STALE_SLOT] == DAILY_STALE_SYMBOL,
        "slot_IS_stale_tailed": entry.get("stale_tail") is True,
        "trailing_fabricated_row_count_exact": (
            entry.get("trailing_stale_rows") == DAILY_STALE_EXPECT_TRAILING),
        # NUMERICS UNCHANGED: the tail really is the last real close, carried forward
        "tail_is_the_frozen_last_real_close": bool(
            last >= 0 and (col[last + 1:] == col[last]).all() and col[last] > 0),
        "no_tail_row_is_zero": bool(last >= 0 and (col[last + 1:] != 0).all()),
        # PRE-EXISTING FIELDS UNMOVED: it is still 'active', in neither list
        "status_is_still_active": entry["status"] == "active",
        "NOT_in_inactive_slots": DAILY_STALE_SLOT not in inactive,
        "NOT_in_partial_slots": DAILY_STALE_SLOT not in partial,
        # ... and DISCLOSED on its own axis
        "disclosed_in_stale_tail_list": [c["slot"] for c in stale] == [DAILY_STALE_SLOT],
        "disclosure_names_the_symbol": [c["symbol"] for c in stale] == [DAILY_STALE_SYMBOL],
        "last_real_ts_recorded": isinstance(entry.get("last_real_ts"), int),
        # ---- (2) the DAILY FROZEN axis, exact. Without these the whole daily _frozen_spans block
        # can be deleted and every leg of both suites still passes (measured).
        "DAILY_frozen_rows_exact": entry.get("n_frozen_rows") == DAILY_STALE_EXPECT_FROZEN_ROWS,
        "DAILY_frozen_spans_exact": entry.get("frozen_spans") == DAILY_STALE_EXPECT_FROZEN_SPANS,
        "DAILY_frozen_interior_rows_exact": (
            entry.get("frozen_interior_rows") == DAILY_STALE_EXPECT_INTERIOR_ROWS),
        "DAILY_real_panel_rows_exact": (
            entry.get("real_panel_rows") == DAILY_STALE_EXPECT_REAL_ROWS),
        "DAILY_filled_rows_is_the_whole_window": entry.get("filled_rows") == px.shape[0],
        # the span really is the fabricated region, checked against the price column
        "DAILY_frozen_span_covers_exactly_the_fabricated_rows": bool(
            last >= 0 and entry.get("frozen_spans") == [[last + 1, px.shape[0] - 1]]),
    }
    # this delisting is a pure TAIL: frozen_decision_cells must call all 90 cells trailing, none
    # interior (the axis D31 turns on).
    fz_tail = ev.frozen_decision_cells(cov, 0, px.shape[0])
    checks["DAILY_cells_split_all_TRAILING_none_interior"] = (
        fz_tail["total"] == DAILY_STALE_EXPECT_FROZEN_ROWS
        and fz_tail["tail"] == DAILY_STALE_EXPECT_FROZEN_ROWS and fz_tail["interior"] == 0)

    # ---------- (3) REAL INTERIOR daily holes, on a second fixture window ----------
    ipx, isyms, its, icov = ev.load_daily_prices_framed(
        FIXTURE_DB, DAILY_INTERIOR_WINDOW[0], DAILY_INTERIOR_WINDOW[1], DAILY_STALE_FRAME,
        require_active_slots=0)
    # INDEPENDENT re-derivation: the raw ohlcv_daily rows on the loader's own axis, no fill.
    iraw = _raw_daily_matrix(FIXTURE_DB, DAILY_STALE_FRAME, its)
    ireal = iraw != 0
    ifilled = np.asarray(ipx) != 0
    ifrozen_indep = ifilled & ~ireal
    ifrozen_spans = np.zeros_like(ifrozen_indep)
    for c in icov:
        for a, b in (c.get("frozen_spans") or []):
            ifrozen_spans[a:b + 1, int(c["slot"])] = True
    ispans = {int(c["slot"]): c["frozen_spans"] for c in icov if c.get("n_frozen_rows")}
    irows = {int(c["slot"]): int(c["n_frozen_rows"]) for c in icov if c.get("n_frozen_rows")}
    ifz = ev.frozen_decision_cells(icov, 0, ipx.shape[0])
    checks.update({
        "DAILY_INTERIOR_window_row_count": ipx.shape[0] == DAILY_INTERIOR_EXPECT_ROWS,
        "DAILY_INTERIOR_spans_exact": ispans == DAILY_INTERIOR_EXPECT_SPANS,
        "DAILY_INTERIOR_frozen_rows_exact": irows == DAILY_INTERIOR_EXPECT_FROZEN_ROWS,
        # the disclosure reconstructs the independently derived frozen mask, cell for cell
        "DAILY_INTERIOR_spans_reconstruct_the_INDEPENDENT_mask": bool(
            (ifrozen_spans == ifrozen_indep).all()),
        "DAILY_INTERIOR_cells_are_INTERIOR_not_trailing": (
            (ifz["total"], ifz["interior"], ifz["tail"]) == DAILY_INTERIOR_EXPECT_CELLS),
        # ... and the pre-fix invisibility is the negative control: NONE of these slots is stale-
        # tailed, so on the old (last-real-row) rule they were disclosed on no axis whatsoever.
        "DAILY_INTERIOR_slots_are_in_NO_stale_tail_list": not any(
            c.get("stale_tail") for c in icov if int(c["slot"]) in DAILY_INTERIOR_EXPECT_SPANS),
        "DAILY_INTERIOR_holes_carry_the_PRE_hole_price": all(
            bool((np.asarray(ipx)[a:b + 1, s] == np.asarray(ipx)[a - 1, s]).all())
            for s, sp in ispans.items() for a, b in sp if a > 0),
    })
    for k, v in checks.items():
        print(f"    {'OK  ' if v else 'FAIL'} {k}")
    print(f"    DAILY frozen axis, {DAILY_STALE_WINDOW[0]}..{DAILY_STALE_WINDOW[1]}: slot "
          f"{DAILY_STALE_SLOT} spans {entry.get('frozen_spans')} = {entry.get('n_frozen_rows')} "
          f"frozen rows ({entry.get('frozen_interior_rows')} interior) -> cells "
          f"{fz_tail['total']} = {fz_tail['interior']} INTERIOR + {fz_tail['tail']} trailing")
    print(f"    DAILY INTERIOR holes, {DAILY_INTERIOR_WINDOW[0]}..{DAILY_INTERIOR_WINDOW[1]} "
          f"({ipx.shape[0]} rows): " + "; ".join(
              f"slot {s} {isyms[s]} {ispans[s]} = {irows[s]} rows" for s in sorted(ispans))
          + f"  -> cells {ifz['total']} = {ifz['interior']} INTERIOR + {ifz['tail']} trailing")
    print(f"    daily slot {DAILY_STALE_SLOT} {DAILY_STALE_SYMBOL}: "
          f"{entry.get('trailing_stale_rows')} of {px.shape[0]} rows FABRICATED (forward-filled at "
          f"{float(col[last]) if last >= 0 else float('nan'):.4f} from panel row {last}), status "
          f"'{entry['status']}', in inactive_slots={DAILY_STALE_SLOT in inactive}, "
          f"in partial_slots={DAILY_STALE_SLOT in partial} — previously disclosed NOWHERE")
    return {"leg": "iia4_daily_disclosure_parity", "passed": all(checks.values()),
            "checks": checks, "window": list(DAILY_STALE_WINDOW), "n_rows": int(px.shape[0]),
            "slot": DAILY_STALE_SLOT, "symbol": DAILY_STALE_SYMBOL,
            "trailing_stale_rows": entry.get("trailing_stale_rows"),
            "last_covered_row": last,
            # fix pass 2 — the DAILY frozen/interior axis, previously unasserted anywhere
            "daily_frozen_rows": entry.get("n_frozen_rows"),
            "daily_frozen_spans": entry.get("frozen_spans"),
            "daily_frozen_cells": {"total": fz_tail["total"], "interior": fz_tail["interior"],
                                   "tail": fz_tail["tail"]},
            "daily_interior_window": list(DAILY_INTERIOR_WINDOW),
            "daily_interior_n_rows": int(ipx.shape[0]),
            "daily_interior_spans": {str(s): ispans[s] for s in sorted(ispans)},
            "daily_interior_frozen_rows": {str(s): irows[s] for s in sorted(irows)},
            "daily_interior_cells": {"total": ifz["total"], "interior": ifz["interior"],
                                     "tail": ifz["tail"]}}


def leg_teacher_floor_masked(R, imp):
    """(ii-a5) D30 rider (a) — the on-window TEACHER FLOOR must not go bankrupt on a framed panel,
    and must read the SAME price-validity rule as the other two operators.

    ``compute_teacher_floor`` walks the weight-native ladder straight through ``step_vector``, so it
    bypasses the env's ``price_valid = current_prices > 0`` action mask that the search operator gets
    for free and that the policy-argmax replay already re-applies by hand. ``--teacher-floor`` is
    hardwired ON in the read-of-record operator (scorecard.py, OPERATOR_ARGS), so a bankrupt floor
    would ship silently: its Sharpe is FINITE, so no degenerate detector fires.

    FOUR measurements, all on the fixture minute pin:
      * the loader's panel (forward-filled tail), MASKED: NO bankruptcy, finite Sharpe — the ship
        condition.
      * the SAME panel UNMASKED: **BIT-IDENTICAL** to the masked walk. This is the honest statement
        of what the rider buys (fix pass): on any panel this loader can emit the mask is PROVABLY
        INERT — `_ffill_interior_only` makes a real->0 transition impossible, only LEADING zeros
        survive, and on a leading flat-zero run `book_target_signs` returns sign 0 so the ladder
        emits HOLD anyway. The rider is DEFENCE IN DEPTH against a future loader that could emit a
        non-leading zero, not a live rescue.
      * CONTROL, tail zeroed (the pre-D30 build) UNMASKED: bankrupt at decision 45, final_pv 0.00,
        Sharpe -15.83 — D30's measured defect, reproduced, so this leg is not vacuous. NOTE what it
        proves: the -15.83/0.00/45 the rider was justified with is a property of the ZEROED-tail
        build, which D30 opt (i) replaced. It is NOT evidence about the shipped panel.
      * the same zeroed panel MASKED: no bankruptcy. So the mask WOULD fix it, independently of the
        tail fill — the mechanism is real, it just cannot be reached from here any more.
    """
    import numpy as np
    print("\n=== (ii-a5) TEACHER FLOOR — masked on the framed path, no bankruptcy (D30 rider (a)) ===")
    ev = imp._load_eval_module()
    import jax.numpy as jnp
    px, syms, ts, cov, gate = ev.load_minute_prices_framed(
        FIXTURE_DB, "2025-09-14", "2025-12-12", FRAME_SYMBOLS, 240, require_active_slots=0)
    cfg = ev.EnvConfig(
        n_assets=8, initial_cash=100_000.0, fee_bps=0.0, slippage_bps=0.0, max_steps=400,
        window_size=EXPECT_START_ROW, position_size_pct=COLLECT_POSITION_SIZE_PCT,
        vector_action=True, vector_weight_rebalance=True, vector_weight_unit=-1.0,
        vector_weight_max=4.0, vector_rebalance_deadband=-1.0)
    warm, N = EXPECT_START_ROW, EXPECT_DECISIONS

    def floor_of(arr, masked):
        dec = arr[warm:warm + N]
        m = (dec <= 0) if masked else None
        f, _ = ev.compute_teacher_floor(jnp.asarray(arr, dtype=jnp.float32), cfg, warm, N, 240.0,
                                        hold_mask=m)
        return f

    filled = np.asarray(px)
    zeroed = filled.copy()
    zeroed[EXPECT_STALE_TAIL_LAST_ROW + 1:, EXPECT_STALE_TAIL_SLOTS[0]] = 0.0
    live = floor_of(filled, True)
    live_unmasked = floor_of(filled, False)
    ctl_unmasked = floor_of(zeroed, False)
    ctl_masked = floor_of(zeroed, True)
    finite = lambda f: all(isinstance(v["sharpe"], float) for v in f.values())   # noqa: E731
    # FIX PASS: the ladder's action histogram on a wholly-inactive slot, instrumented directly —
    # sign 0 on a constant log-price column => HOLD on every decision, so the -26 log-return the
    # rider was written against cannot be produced from a leading flat-zero head either.
    _tgt, _bw = ev.book_target_signs(jnp.asarray(filled, dtype=jnp.float32))
    dead_slot_signs = np.asarray(_tgt)[warm:warm + N, EXPECT_INACTIVE_SLOTS[0]]
    checks = {
        "shipped_panel_no_bankruptcy_any_tier": all(
            v["bankrupt_at"] == -1 for v in live.values()),
        "shipped_panel_final_pv_positive": all(v["final_pv"] > 0 for v in live.values()),
        "shipped_panel_sharpes_finite": finite(live),
        # FIX PASS — the rider is INERT on the shipped panel, stated as an assertion instead of a
        # claim: masked and unmasked are bit-identical, so nothing about the recorded floor numbers
        # depends on the mask, and the docstring may not say otherwise.
        "MASKED_equals_UNMASKED_on_the_shipped_panel": all(
            live[c] == live_unmasked[c] for c in live),
        "wholly_inactive_slot_gets_ladder_sign_ZERO_ie_HOLD": bool(
            (dead_slot_signs == 0).all()),
        # control: the pre-D30 zeroed tail, UNMASKED -> D30's measured bankruptcy, reproduced
        "control_zeroed_unmasked_IS_bankrupt": all(
            v["bankrupt_at"] >= 0 and v["final_pv"] == 0.0 for v in ctl_unmasked.values()),
        # The control's EXACT numbers are a property of where the tail is zeroed, and the fix pass
        # moved that boundary (last_covered_row is now derived from the panel's real-BAR mask:
        # 102 -> 103, so the control zeroes rows 104+ instead of 103+). D30's -15.83 / bankrupt_at
        # 45 belong to the OLD boundary; re-derived here as -13.21 / 46 and asserted as a BAND plus
        # the structural facts, so this leg proves the mechanism without pinning a number that
        # depends on an arbitrary boundary choice.
        "control_bankruptcy_is_early_in_the_tail": all(
            EXPECT_STALE_TAIL_LAST_ROW - EXPECT_START_ROW
            <= v["bankrupt_at"] <= EXPECT_DECISIONS for v in ctl_unmasked.values()),
        "control_sharpe_is_FINITE_so_no_detector_fires": finite(ctl_unmasked),
        "control_sharpe_is_catastrophic_not_marginal": all(
            v["sharpe"] < -10.0 for v in ctl_unmasked.values()),
        "control_is_an_order_of_magnitude_worse_than_the_shipped_panel": (
            ctl_unmasked[0]["sharpe"] < 5.0 * live[0]["sharpe"]),
        # ... and the mask alone fixes it
        "mask_alone_removes_the_bankruptcy": all(
            v["bankrupt_at"] == -1 and v["final_pv"] > 0 for v in ctl_masked.values()),
    }
    for k, v in checks.items():
        print(f"    {'OK  ' if v else 'FAIL'} {k}")
    for label, f in (("loader panel (ffilled tail), MASKED  ", live),
                     ("loader panel (ffilled tail), UNMASKED", live_unmasked),
                     ("control zeroed tail, UNMASKED       ", ctl_unmasked),
                     ("control zeroed tail, MASKED         ", ctl_masked)):
        print(f"    {label}: " + "  ".join(
            f"tier{c}bp sharpe {f[c]['sharpe']:.4f} pv {f[c]['final_pv']:,.2f} "
            f"bankrupt_at {f[c]['bankrupt_at']}" for c in sorted(f)))
    print("    => on the SHIPPED panel the rider (a) mask is a NO-OP (bit-identical). It is "
          "defence in depth against a loader that could emit a non-leading zero; the -15.83/0.00/"
          "45 it was justified with belongs to the ZEROED-tail build D30 opt (i) replaced.")
    return {"leg": "iia5_teacher_floor_masked", "passed": all(checks.values()), "checks": checks,
            "floor_shipped": {str(c): live[c] for c in live},
            "floor_shipped_unmasked": {str(c): live_unmasked[c] for c in live_unmasked},
            "mask_is_inert_on_shipped_panel": all(live[c] == live_unmasked[c] for c in live),
            "floor_control_unmasked": {str(c): ctl_unmasked[c] for c in ctl_unmasked},
            "floor_control_masked": {str(c): ctl_masked[c] for c in ctl_masked}}


def leg_daily_teacher_floor_masked(R, imp):
    """(ii-a6) TASK_869 (D31 (a3) scope 2) — the DAILY-framed teacher floor now reads the SAME
    price-validity rule as the minute pin. The minute-path inertness finding does NOT transfer
    by assumption; it is MEASURED here on the daily path:

      * the loader's DAILY framed panel (FTM/USD stale tail forward-filled, inactive slots
        flat-zero), MASKED vs UNMASKED: **BIT-IDENTICAL** — the mask is INERT on the shipped
        daily panel, for the same structural reason as the minute path (the daily fill is
        interior-only, so only LEADING zeros survive, and a leading flat-zero run reads as
        ladder sign 0 => HOLD anyway). It ships as DEFENCE IN DEPTH, the 865 precedent.
      * CONTROL, mutation both ways: the same panel plus a synthetic 9th slot that is FLAT
        (sign 0 => HOLD, never held) while real and then ZEROED over a synthetic stale tail —
        the non-leading zero span no post-D30 loader can emit. The UNMASKED walk reads the
        fabricated real->0 transition and MOVES (catastrophic); the MASKED walk is
        bit-identical to the unmutated walk (the slot is never held, so the zeroed rows move
        no valuation either). So the mask demonstrably bites exactly where the defect class
        lives, and costs nothing elsewhere.
    """
    import numpy as np
    print("\n=== (ii-a6) DAILY TEACHER FLOOR — masked like the minute pin (TASK_869 scope 2) ===")
    ev = imp._load_eval_module()
    import jax.numpy as jnp
    px, syms, ts, cov = ev.load_daily_prices_framed(
        FIXTURE_DB, DAILY_STALE_WINDOW[0], DAILY_STALE_WINDOW[1], DAILY_STALE_FRAME,
        require_active_slots=0)
    filled = np.asarray(px)
    T = filled.shape[0]

    def mk_cfg(n):
        return ev.EnvConfig(
            n_assets=n, initial_cash=100_000.0, fee_bps=0.0, slippage_bps=0.0, max_steps=400,
            window_size=60, position_size_pct=COLLECT_POSITION_SIZE_PCT,
            vector_action=True, vector_weight_rebalance=True, vector_weight_unit=-1.0,
            vector_weight_max=4.0, vector_rebalance_deadband=-1.0)
    warm, N = 60, T - 1 - 60

    def floor_of(arr, masked):
        cfg = mk_cfg(arr.shape[1])
        dec = arr[warm:warm + N]
        m = (dec <= 0) if masked else None
        f, _ = ev.compute_teacher_floor(jnp.asarray(arr, dtype=jnp.float32), cfg, warm, N, 1440.0,
                                        hold_mask=m)
        return f

    # Control: a 9th synthetic slot, FLAT at 100.0 while real (ladder sign 0 => HOLD on every
    # decision, so the walk NEVER carries it), ZEROED from row z0 to the panel end — a synthetic
    # stale daily span (non-leading zeros, the panel no post-D30 loader emits). z0 sits strictly
    # inside the decision range so the fabricated real->0 transition is walked.
    z0 = warm + 20
    synth_real = np.full((T, 1), 100.0)
    base9 = np.concatenate([filled, synth_real], axis=1)          # unmutated 9-slot panel
    ctl9 = base9.copy()
    ctl9[z0:, 8] = 0.0
    live = floor_of(filled, True)
    live_unmasked = floor_of(filled, False)
    base_masked = floor_of(base9, True)
    base_unmasked = floor_of(base9, False)
    ctl_unmasked = floor_of(ctl9, False)
    ctl_masked = floor_of(ctl9, True)
    finite = lambda f: all(isinstance(v["sharpe"], float) for v in f.values())   # noqa: E731
    moved = [c for c in base_unmasked if ctl_unmasked[c] != base_unmasked[c]]
    checks = {
        "shipped_daily_panel_no_bankruptcy_any_tier": all(
            v["bankrupt_at"] == -1 for v in live.values()),
        "shipped_daily_panel_sharpes_finite": finite(live),
        # The MEASURED inertness on the daily path (not assumed from the minute path):
        "MASKED_equals_UNMASKED_on_the_shipped_daily_panel": all(
            live[c] == live_unmasked[c] for c in live),
        "MASKED_equals_UNMASKED_on_the_9slot_base_panel": all(
            base_masked[c] == base_unmasked[c] for c in base_masked),
        # Mutation proof, both directions:
        "synthetic_zeroed_span_MOVES_the_unmasked_walk": bool(moved),
        "synthetic_zeroed_span_does_NOT_move_the_masked_walk": all(
            ctl_masked[c] == base_masked[c] for c in base_masked),
        # ... and the unmasked divergence is the synthetic slot's whole weight destroyed, not
        # noise (the defect class the rider exists against: acting on a cell with no market).
        # MEASURED: final_pv ratio 0.931 on BOTH tiers (the flat slot's notional plus costs,
        # bounded by the ladder's per-slot weight — a daily bar caps the move, unlike the
        # minute rider's zeroed-tail bankruptcy).
        "control_unmasked_destroys_the_synthetic_slots_weight": all(
            ctl_unmasked[c]["final_pv"] < 0.95 * base_unmasked[c]["final_pv"]
            for c in moved),
    }
    for k, v in checks.items():
        print(f"    {'OK  ' if v else 'FAIL'} {k}")
    for label, f in (("daily panel (ffilled tail), MASKED  ", live),
                     ("daily panel (ffilled tail), UNMASKED", live_unmasked),
                     ("control zeroed span, UNMASKED      ", ctl_unmasked),
                     ("control zeroed span, MASKED        ", ctl_masked)):
        print(f"    {label}: " + "  ".join(
            f"tier{c}bp sharpe {f[c]['sharpe']:.4f} pv {f[c]['final_pv']:,.2f} "
            f"bankrupt_at {f[c]['bankrupt_at']}" for c in sorted(f)))
    print("    => the daily mask is INERT on the shipped panel (defence in depth, 865 "
          "precedent) and BITES on the synthetic stale span — both directions measured.")
    return {"leg": "iia6_daily_teacher_floor_masked", "passed": all(checks.values()),
            "checks": checks, "zeroed_rows": [z0, T - 1],
            "floor_shipped": {str(c): live[c] for c in live},
            "floor_shipped_unmasked": {str(c): live_unmasked[c] for c in live_unmasked},
            "mask_is_inert_on_shipped_daily_panel": all(
                live[c] == live_unmasked[c] for c in live),
            "floor_control_unmasked": {str(c): ctl_unmasked[c] for c in ctl_unmasked},
            "floor_control_masked": {str(c): ctl_masked[c] for c in ctl_masked}}


def leg_loader_identity_refusals(R, imp):
    """IDENTITY SCRAMBLING, at the LOADER: unknown symbol / duplicate symbol / wrong length are
    each a NAMED refusal — never a silent drop, pad or re-index."""
    print("\n=== (ii-b1) IDENTITY SCRAMBLING — unknown / duplicate / wrong-length frames ===")

    def refuses(path, symbols, needle):
        write_frame(path, symbols, "TASK_865 identity-scramble frame")
        try:
            _framed_panel(imp, path)          # --n-assets stays 8 for every scramble
        except RuntimeError as e:
            hit = needle in str(e)
            print(f"    {'OK  ' if hit else 'FAIL'} refused ('{needle}' present={hit})")
            return hit
        print(f"    FAIL not refused: {needle}")
        return False

    unknown = list(FRAME_SYMBOLS)
    unknown[3] = "NOT/A/SYMBOL"
    dupe = list(FRAME_SYMBOLS)
    dupe[3] = dupe[0]
    checks = {
        "unknown_symbol_refused_not_dropped": refuses(
            R("frame_unknown.json"), unknown, "are not in the `assets` table"),
        "duplicate_symbol_refused": refuses(
            R("frame_dupe.json"), dupe, "DUPLICATE symbol(s)"),
        "wrong_length_frame_refused": refuses(
            R("frame_short.json"), FRAME_SYMBOLS[:6], "recorded frame has 6 symbols"),
    }
    return {"leg": "iib1_identity_scrambling_refused", "passed": all(checks.values()),
            "checks": checks}


def leg_source_gate(R, imp, collection):
    """The TASK_865 SOURCE gate, in-process and ISOLATED: the SAME frame, the SAME window, the
    SAME 50 rows — only the TABLE the pin was read from differs. Both directions refuse."""
    print("\n=== (ii-b2) SOURCE GATE — daily-pinned vs minute-pinned frames may not be mixed ===")
    fp = write_frame(R("frame_source_gate.json"), FRAME_SYMBOLS, "TASK_865 source-gate frame")
    args, prices_j, env_cfg, frame, start_row, N, T = _framed_panel(imp, fp)
    tier = env_cfg.replace(fee_bps=imp.FEE_BPS, slippage_bps=imp.SLIP_BPS)
    base = json.load(open(collection))

    def run_gate(mutate, imp_source="minute"):
        doc = json.loads(json.dumps(base))
        mutate(doc["provenance"])
        a = _pin_args(fp, source=imp_source)
        f = frame if imp_source is not None else None
        try:
            imp.gate_teacher_vectors(doc, collection, a, tier, start_row, N, T, frame=f)
            return None
        except Exception as e:                      # RuntimeError = refusal; anything else = bug
            return f"{type(e).__name__}: {e}"

    matched = run_gate(lambda p: None)
    # The collection claims DAILY (the shipped 853 block carries no panel_frame_source key at
    # all, so "absent == daily" is exercised by DELETING the key) while this run pins MINUTE.
    coll_daily = run_gate(
        lambda p: p["panel_frame"].pop("panel_frame_source", None)) or ""
    # ... and the reverse: a MINUTE-pinned collection imprinted under a DAILY pin.
    coll_minute_imp_daily = run_gate(lambda p: None, imp_source="daily") or ""
    checks = {
        "matched_minute_pair_ACCEPTED": matched is None,
        "daily_collection_minute_imprint_REFUSED": (
            "panel_frame SOURCE MISMATCH" in coll_daily and "'daily'" in coll_daily),
        "minute_collection_daily_imprint_REFUSED": (
            "panel_frame SOURCE MISMATCH" in coll_minute_imp_daily
            and "'minute'" in coll_minute_imp_daily),
        "refusal_names_D20_cross_source_rule": all(
            "D20" in t for t in (coll_daily, coll_minute_imp_daily)),
        "absent_panel_frame_source_reads_as_DAILY": "'daily'" in coll_daily,
        "all_refusals_are_RuntimeError": all(
            t.startswith("RuntimeError:") for t in (coll_daily, coll_minute_imp_daily)),
    }
    for k, v in checks.items():
        print(f"    {'OK  ' if v else 'FAIL'} {k}")
    if matched is not None:
        print(f"    matched pair unexpectedly refused: {matched[:400]}")
    return {"leg": "iib2_source_gate", "passed": all(checks.values()), "checks": checks}


def degenerate_leg(name, R, frame, extra, expect_clauses, expect_null, expect_finite):
    """Acceptance (iv) — the 863 FIX-0 / FIX-2.1 degeneracy guard on the MINUTE-pinned path, with
    ``--smoke-assert`` explicitly OFF (the default for any non-fixture DB, i.e. the REAL leg's
    configuration). The whole refusal contract is asserted: exit non-zero, the artifact's
    ``sanity.degenerate_replay: true`` + a reason naming the right diagnosis, and the
    ``DEGENERATE_REPLAY.json`` stamp in the checkpoint dir."""
    out, ck = R(f"results_{name}.json"), R(f"ckpt_{name}")
    cmd = [sys.executable, IMPRINT_PY, "--db", FIXTURE_DB, *MINUTE_PANEL,
           "--vector-weight-rebalance", "--imprint-mapping", "weight-native",
           "--batch-size", "32", "--panel-frame-from-checkpoint", frame,
           "--panel-frame-source", "minute", "--no-smoke-assert",
           *REQUIRE_ACTIVE_OVERRIDE, "--out", out, "--ckpt-out", ck, *extra]
    rc, text = sh(cmd, R(f"log_{name}.txt"), expect_ok=False)
    res = json.load(open(out)) if os.path.exists(out) else {}
    sn = res.get("sanity") or {}
    tier = (sn.get("per_tier") or {}).get("21") or {}
    reason = sn.get("degenerate_replay_reason") or ""
    stamp_path = os.path.join(ck, "DEGENERATE_REPLAY.json")
    stamp = json.load(open(stamp_path)) if os.path.exists(stamp_path) else None
    checks = {
        "smoke_assert_was_OFF": "smoke_assert=False" in text,
        "exit_non_zero_anyway": rc != 0,
        "refusal_printed": "[863] REFUSED" in text,
        "artifact_written_for_forensics": bool(res),
        "artifact_degenerate_replay_TRUE": sn.get("degenerate_replay") is True,
        "reason_non_empty": len(reason.strip()) > 80,
        "ckpt_stamp_present": stamp is not None,
        "ckpt_stamp_degenerate_true": bool(stamp) and stamp.get("degenerate_replay") is True,
        "ledger_metric_is_null": (res.get("ledger_row") or {}).get(
            "imprint_corridor_position_net21") is None,
        "no_smoke_pass_banner": "SMOKE PASS" not in text,
        "ran_on_the_MINUTE_pin": res.get("panel_frame_source") == "minute",
    }
    for f in expect_null:
        checks[f"net21_{f}_is_null"] = tier.get(f, "missing") is None
    for f in expect_finite:
        checks[f"net21_{f}_is_finite"] = isinstance(tier.get(f), float)
    for c in expect_clauses:
        checks[f"reason_names_{c.replace(' ', '_')}"] = c in reason
    for k, v in checks.items():
        print(f"    {'OK  ' if v else 'FAIL'} {k}")
    print(f"    reason: {reason[:260]}...")
    return {"leg": name, "passed": all(checks.values()), "checks": checks,
            "exit_code": rc, "reason": reason, "log": f"log_{name}.txt"}


def leg_walk_net21_reconciliation(R, imp, collection_path, coll):
    """D30 amendment 6 — RE-DERIVE the VOID bar and account for the difference, per field.

    The recorded PASS bar ``walk_net21 = 11.32152396918238`` was declared VOID by D30: it was
    produced on the ZEROED-tail build, i.e. it contains fabricated PnL. D30 predicted +17.1242 with
    the tail filled. Neither number exists anywhere in the tree any more, so this leg RE-DERIVES the
    value on the shipped panel and states the reconciliation as data rather than prose.

    TASK_868 NOTE: the pre-868 PANEL_MEASURED_* replay pins (14.949671921997371 /
    15.952853341299857) belonged to the OLD fixture panel (T=111) and are not reproducible on
    the regenerated fixture (T=532); the structural assertions stand — the SAME recorded stream
    replayed on the filled and hand-zeroed panels isolates the pure valuation channel, which
    must stay SMALL against the historical +16.26 mis-attribution — and the freshly measured
    numbers are recorded below.
    """
    import numpy as np
    import jax.numpy as jnp
    print("\n=== (i-b) walk_net21 RECONCILIATION — the VOID bar re-derived and decomposed ===")
    ev = imp._load_eval_module()
    cp = coll["provenance"]
    recorded = cp["walk_net21"]
    px, _, _, cov, gate = ev.load_minute_prices_framed(
        FIXTURE_DB, "2025-09-14", "2025-12-12", FRAME_SYMBOLS, 240, require_active_slots=0)
    filled = np.asarray(px)
    zeroed = filled.copy()
    zeroed[EXPECT_STALE_TAIL_LAST_ROW + 1:, EXPECT_TAIL_ONLY_SLOTS[0]] = 0.0
    vecs = np.asarray(coll["decisions"]["chosen_vectors"], dtype=np.int32)
    warm, N = EXPECT_START_ROW, EXPECT_DECISIONS
    cfg = ev.EnvConfig(
        n_assets=8, initial_cash=100_000.0, fee_bps=ev.FEE_BPS, slippage_bps=ev.SLIP_BPS,
        max_steps=400, window_size=warm, position_size_pct=COLLECT_POSITION_SIZE_PCT,
        vector_action=True, vector_weight_rebalance=True, vector_weight_unit=-1.0,
        vector_weight_max=4.0, vector_rebalance_deadband=-1.0)

    def replay(arr, hold_mask=None):
        w = imp.walk_env(jnp.asarray(arr, dtype=jnp.float32), cfg, warm, N,
                         lambda k, state: vecs[k], hold_mask=hold_mask)
        return float(ev._sharpe(w["rets"], 240.0)), float(w["final_pv"])

    s_filled, pv_filled = replay(filled)
    s_zeroed, pv_zeroed = replay(zeroed)
    # TASK_868 (a2): the SAME stream under the causal open-gate — the masked replay is the
    # post-868 semantics (frozen cells force-HELD for OPENING).
    mask868 = imp.frame_hold_mask(jnp.asarray(filled, dtype=jnp.float32), warm, N, open_gate=gate)
    s_masked, pv_masked = replay(filled, hold_mask=mask868)
    n_gated = int(mask868.sum()) if mask868 is not None else 0
    pure = s_filled - s_zeroed
    # the pre/post panel delta really is only the tail cells — stated, not assumed
    delta_cells = int((filled != zeroed).sum())
    checks = {
        "recorded_walk_net21_is_finite": isinstance(recorded, float) and recorded == recorded,
        "panel_delta_is_ONLY_the_stale_tail_cells": delta_cells == (
            filled.shape[0] - 1 - EXPECT_STALE_TAIL_LAST_ROW),
        "same_stream_replays_on_BOTH_panels": True,     # both replays completed above
        "pure_valuation_channel_is_SMALL_vs_the_recorded_swing": abs(pure) < 5.0,
        "shipped_panel_replay_is_finite_and_solvent": (
            s_filled == s_filled and pv_filled > 0),
        # TASK_868 (a2): the masked replay (post-868 semantics) is finite and solvent — and is
        # IDENTICAL to the unmasked replay, which is exactly the end-to-end alignment proof: the
        # collection stream was itself recorded UNDER the open-gate, so it never acts on a
        # frozen cell and the mask cannot bite on it. (The biting case — a stream that DOES act
        # mid-gap — is the 868-iv leg's synthetic repro.)
        "masked_replay_is_finite_and_solvent": (s_masked == s_masked and pv_masked > 0),
        "masked_replay_EQUALS_unmasked_on_a_mask_consistent_stream": (
            s_masked == s_filled and pv_masked == pv_filled
            and not bool((vecs[np.asarray(mask868)] != 1).any())),
    }
    for k, v in checks.items():
        print(f"    {'OK  ' if v else 'FAIL'} {k}")
    print(f"    RE-RECORDED walk_net21 on the shipped (forward-filled) panel: {recorded:+.12f}")
    print(f"    VOID value recorded on the ZEROED-tail build (D30)          : "
          f"{VOID_WALK_NET21:+.12f}  (historical record, old fixture)")
    print(f"    D30's predicted value with the tail filled                  : "
          f"{D30_PREDICTED_WALK_NET21:+.12f}  (historical record, old fixture)")
    print(f"    same recorded stream replayed: filled {s_filled:+.12f} (pv {pv_filled:,.2f}) vs "
          f"zeroed {s_zeroed:+.12f} (pv {pv_zeroed:,.2f}) -> pure channel {pure:+.6f} Sharpe "
          f"over {delta_cells} changed cells")
    print(f"    TASK_868 masked replay (open-gate, {n_gated} cells force-HELD): "
          f"{s_masked:+.12f} (pv {pv_masked:,.2f}) — the post-868 semantics")
    return {"leg": "ib_walk_net21_reconciliation", "passed": all(checks.values()),
            "checks": checks,
            "walk_net21_rederived": recorded,
            "walk_net21_void_recorded_on_zeroed_tail": VOID_WALK_NET21,
            "walk_net21_d30_prediction": D30_PREDICTED_WALK_NET21,
            "delta_vs_void": (None if recorded is None else recorded - VOID_WALK_NET21),
            "delta_vs_d30_prediction": (None if recorded is None
                                        else recorded - D30_PREDICTED_WALK_NET21),
            "same_stream_replay": {
                "filled_sharpe": s_filled, "filled_final_pv": pv_filled,
                "zeroed_sharpe": s_zeroed, "zeroed_final_pv": pv_zeroed,
                "pure_valuation_channel_sharpe": pure,
                "n_changed_panel_cells": delta_cells,
                "masked_sharpe_open_gate": s_masked, "masked_final_pv": pv_masked,
                "open_gated_cells": n_gated},
            "note": (
                "D30 amendment 6 + TASK_868 re-base. The VOID bar 11.32152396918238 (zeroed-tail "
                "build) and D30's 17.1242 prediction are historical records tied to the OLD "
                "fixture panel; the pre-868 PANEL_MEASURED replay pins are likewise not "
                "reproducible on the regenerated fixture. The pure valuation/tradeability "
                "channel (SAME recorded stream, filled vs hand-zeroed panel) is reported "
                "fresh and stays small; the masked replay is the post-868 (D31 a2) semantics.")}


def leg_real_db_coverage():
    """Acceptance (iii) + TASK_868 gate (i) — the 861 leg-1 frame read through the ALIAS UNION,
    READ-ONLY (one ``mode=ro`` URI connection, SELECTs only: no panel build, no write, no DDL,
    no GPU). Every frame symbol resolves to its alias GROUP (the D31 (a1) repair this task
    ships in the loaders), the per-bar real mask is UNIONED across member ids, and the
    agreement gate is asserted on the measured overlaps.

    What this measured pre-868 (the numbers this leg was registered against): INERT 9,047
    (8.4444%), FROZEN 2,383 (2.2243%) = 2,013 INTERIOR (1.8789%) + 370 trailing, NOT REAL
    11,430 (10.6687%). The interior term sat on 13 slots — 11 near-empty ones, of which 6
    were the INGEST IDENTITY SPLIT D31 repaired (twins agreeing to max rel diff 0.00).

    MEASURED post-868 (re-registered below, this run): the 6 split slots read as REAL data —
    5 of 6 fully (/USDTEUR, /USDTCHF, /USDCGBP, /USDC/USDT, /USDTAUD: 0 interior cells),
    /USDC/USD with a 90-cell residue (its twin's own span has sub-gaps) — and the frozen-
    interior count drops 2,013 -> 1,011 (0.9437%), i.e. the ~1% 5-slot residue D31 predicted
    (ETHBTC 183 + BTCGBP/BTCCHF/BTCCAD/XRPBTC 182 each = 911, plus /USDC/USD 90, STX/USD 8,
    BCH/USD 1, XRP/USD 1). INERT (9,047, heads) and trailing (370, migrations) are unchanged.

    A materially different share on ANY axis is a STOP."""
    import sqlite3
    import numpy as np
    from datetime import datetime, timezone
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)
    from mcts_trader.data.asset_aliases import (
        alias_agreement_gate, resolve_requested_symbols)
    print("\n=== (iii) COVERAGE HONESTY + ALIAS UNION (CELL LEVEL) — read-only, gate (i) ===")
    prov = json.load(open(REAL_FRAME_JSON))
    frame = prov.get("frame_symbols") or prov.get("symbols")
    # TASK_869: UTC, matching the loaders' window resolution (was local time).
    s_ts = int(datetime.strptime(REAL_WINDOW[0], "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    e_ts = int(datetime.strptime(REAL_WINDOW[1], "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    step = REAL_BAR_MINUTES
    con = sqlite3.connect(f"file:{REAL_DB}?mode=ro", uri=True, timeout=60)
    try:
        # TASK_868 (a1): each frame symbol -> its alias GROUP (the loader's own resolver).
        all_assets = con.execute("SELECT id, symbol FROM assets").fetchall()
        resolution, unknown = resolve_requested_symbols(all_assets, frame)
        id_groups = [r["asset_ids"] for r in resolution]
        db_syms = [r["db_symbols"] for r in resolution]
        aliased = [(k, frame[k], id_groups[k], db_syms[k])
                   for k in range(len(frame)) if len(id_groups[k]) > 1]
        # ... and the FAIL-LOUD agreement gate, measured: every alias pair's shared-timestamp
        # closes in the window. The 6 span-carriers must agree to 0.00 (same feed, split key).
        gate = alias_agreement_gate(con, "ohlcv", id_groups, s_ts, e_ts)
        gate_by_ids = {tuple(d["asset_ids"]): d["overlaps"] for d in gate}
        ids = sorted({a for g in id_groups for a in g})
        ph2 = ",".join("?" * len(ids))
        rows = con.execute(
            f"""SELECT asset_id, COUNT(*), MIN(timestamp), MAX(timestamp) FROM ohlcv
                WHERE asset_id IN ({ph2}) AND timestamp >= ? AND timestamp <= ?
                  AND close IS NOT NULL AND close != 0
                GROUP BY asset_id""", (*ids, s_ts, e_ts)).fetchall()
        # The loader's ROW AXIS over the UNIONED member ids (~2.5M int64 = ~20MB).
        axis = np.fromiter((r[0] for r in con.execute(
            f"""SELECT DISTINCT timestamp FROM ohlcv
                WHERE asset_id IN ({ph2}) AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp""", (*ids, s_ts, e_ts))), dtype=np.int64)
        n_full = int(axis.size)
        n_panel = len(range(0, n_full, step))
        panel_ts = axis[::step]
        idx_of = {int(t): i for i, t in enumerate(axis)}
        col_of = {a: k for k, g in enumerate(id_groups) for a in g}
        real_bar = np.zeros((n_panel, len(frame)), dtype=bool)
        real_exact = np.zeros((n_panel, len(frame)), dtype=bool)
        per_id_inwin = {a: 0 for a in ids}
        for aid, t in con.execute(
            f"""SELECT asset_id, timestamp FROM ohlcv WHERE asset_id IN ({ph2})
                AND timestamp >= ? AND timestamp <= ? AND close IS NOT NULL AND close != 0""",
                (*ids, s_ts, e_ts)):
            per_id_inwin[aid] += 1
            i = idx_of.get(t)
            if i is None:
                continue
            c = col_of[aid]
            # minute index i belongs to bar ceil(i/step) (bar b covers ((b-1)*step, b*step]);
            # index 0 belongs to bar 0.
            b = (i + step - 1) // step if i else 0
            if b < n_panel:
                real_bar[b, c] = True
            if i % step == 0 and i // step < n_panel:
                real_exact[i // step, c] = True
        # Successor tickers — LOOKED UP AND REPORTED ONLY (never spliced; see SUCCESSOR_TICKERS).
        sym_to_id = {s: g[0] for s, g in zip(frame, id_groups)}
        succ = {}
        for pred, sy in SUCCESSOR_TICKERS.items():
            r = con.execute(
                """SELECT a.id, COUNT(o.rowid), MIN(o.timestamp), MAX(o.timestamp)
                   FROM assets a LEFT JOIN ohlcv o ON o.asset_id = a.id
                   WHERE a.symbol = ? GROUP BY a.id""", (sy,)).fetchone()
            succ[pred] = None if r is None else {
                "symbol": sy, "asset_id": r[0], "rows": r[1], "first_ts": r[2], "last_ts": r[3]}
        whole = {}
        for pred in SUCCESSOR_TICKERS:
            aid = sym_to_id.get(pred)
            r = con.execute(
                """SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM ohlcv
                   WHERE asset_id = ? AND close IS NOT NULL AND close != 0""", (aid,)).fetchone()
            whole[pred] = {"rows": r[0], "first_ts": r[1], "last_ts": r[2]}
    finally:
        con.close()
    n_dec_rows = n_panel - REAL_WARM_ROWS
    n_cells = n_dec_rows * len(frame)
    per_id = {r[0]: (r[1], r[2], r[3]) for r in rows}
    fmt = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d")  # noqa: E731
    # filled / tail masks from each slot's UNIONED first & last real print (the loader's rule:
    # the union's first real close onward is filled; the union's last real close ends the tail)
    filled = np.zeros_like(real_bar)
    tailm = np.zeros_like(real_bar)
    for k, g in enumerate(id_groups):
        spans = [(per_id[a][1], per_id[a][2]) for a in g if per_id.get(a, (0,))[0]]
        if not spans:
            continue
        filled[:, k] = panel_ts >= min(s[0] for s in spans)
        tailm[:, k] = panel_ts > max(s[1] for s in spans)
    frozen = filled & ~real_bar
    frozen_exact = filled & ~real_exact
    D = slice(REAL_WARM_ROWS, n_panel)
    head_cells = int((~filled[D]).sum())
    fz_tail = int((frozen & tailm)[D].sum())
    fz_int = int((frozen & ~tailm)[D].sum())
    fz_tot = fz_int + fz_tail
    not_real = head_cells + fz_tot
    inert_m = ~filled[D]
    frozen_m = (filled & ~real_bar)[D]
    live_m = (filled & real_bar)[D]
    live = int(live_m.sum())
    partition_overlap = int((inert_m & frozen_m).sum() + (inert_m & live_m).sum()
                            + (frozen_m & live_m).sum())
    partition_uncovered = int((~(inert_m | frozen_m | live_m)).sum())
    partition_is_a_partition = (
        partition_overlap == 0 and partition_uncovered == 0
        and int(inert_m.sum()) == head_cells and int(frozen_m.sum()) == fz_tot
        and head_cells + fz_tot + live == n_cells)
    frozen_split_is_a_partition = (
        int(((frozen & tailm) & (frozen & ~tailm))[D].sum()) == 0
        and fz_int + fz_tail == int(frozen_m.sum()))
    pct = lambda n: 100.0 * n / max(n_cells, 1)                                   # noqa: E731
    slots, zero_cov, heads, tails, interiors = [], [], [], [], []
    for k, sym in enumerate(frame):
        n = sum(per_id_inwin[a] for a in id_groups[k])
        spans = [(per_id[a][1], per_id[a][2]) for a in id_groups[k] if per_id.get(a, (0,))[0]]
        mn = min(s[0] for s in spans) if spans else None
        mx = max(s[1] for s in spans) if spans else None
        rec = {"slot": k, "symbol": sym, "rows": n,
               "first": fmt(mn) if mn else None, "last": fmt(mx) if mx else None,
               "asset_ids": list(id_groups[k]),
               "per_id_rows": {a: per_id_inwin[a] for a in id_groups[k]},
               "inert_head_cells": int((~filled[D])[:, k].sum()),
               "frozen_interior_cells": int((frozen & ~tailm)[D][:, k].sum()),
               "frozen_tail_cells": int((frozen & tailm)[D][:, k].sum()),
               "no_print_at_sampled_minute_cells": int((frozen_exact[D])[:, k].sum())}
        rec["not_real_cells"] = (rec["inert_head_cells"] + rec["frozen_interior_cells"]
                                 + rec["frozen_tail_cells"])
        rec["not_real_pct_of_its_column"] = round(100.0 * rec["not_real_cells"]
                                                  / max(n_dec_rows, 1), 1)
        if n == 0:
            zero_cov.append(rec)
        else:
            rec["tail_panel_rows"] = int(tailm[:, k].sum())
            rec["head_panel_rows"] = int((~filled[:, k]).sum())
            if rec["inert_head_cells"]:
                heads.append(rec)
            if rec["frozen_tail_cells"] >= EXPECT_MIN_TAIL_ROWS_DISCLOSED:
                rec["days_stale"] = round((int(panel_ts[-1]) - mx) / 86400.0, 1)
                tails.append(rec)
            if rec["frozen_interior_cells"]:
                interiors.append(rec)
        slots.append(rec)
    interiors.sort(key=lambda r: -r["frozen_interior_cells"])
    print(f"  frame: {len(frame)} slots from {os.path.relpath(REAL_FRAME_JSON, _REPO)}  "
          f"window {REAL_WINDOW[0]}..{REAL_WINDOW[1]}  stride {step}  (read-only, mode=ro)")
    print(f"  ALIAS UNION (TASK_868 a1): {len(aliased)} of {len(frame)} slots resolved to "
          f"MULTIPLE assets rows (D31: 11 empty stubs + 6 span-carriers):")
    for k, sym, g, dsy in aliased:
        ov = gate_by_ids.get(tuple(g), [])
        ov_s = ", ".join(f"pair {o['pair']}: {o['shared_ts']} shared ts, max rel diff "
                         f"{o['max_rel_diff']}" for o in ov)
        print(f"      slot {k:>3} {sym:<12} asset_ids {g} spellings {dsy}  [{ov_s}]")
    print(f"  ROW AXIS: {n_full} distinct minute timestamps -> {n_panel} panel rows at [::{step}] "
          f"-> {n_dec_rows} DECISION rows after the {REAL_WARM_ROWS}-row env window")
    print(f"  DECISION GRID: {n_dec_rows} x {len(frame)} = {n_cells} cells")
    print(f"    INERT  (no price at all; pre-listing heads)          : {head_cells} "
          f"({pct(head_cells):.4f}%)  on {len(heads)} slot(s)")
    print(f"    FROZEN (forward-filled; force-HELD for OPENING, D31 a2): {fz_tot} "
          f"({pct(fz_tot):.4f}%)")
    print(f"      ... INTERIOR (a hole that RE-ANCHORS; CAN fabricate PnL): {fz_int} "
          f"({pct(fz_int):.4f}%) on {len(interiors)} slot(s)   <-- 2,013 pre-868 (1.8789%)")
    print(f"      ... TRAILING (delisted to the window end)               : {fz_tail} "
          f"({pct(fz_tail):.4f}%) on {len(tails)} slot(s)")
    print(f"    LIVE   (the slot printed inside that bar)            : {live} ({pct(live):.4f}%)")
    print(f"    NOT REAL DATA, total                                 : {not_real} of {n_cells} "
          f"= {pct(not_real):.4f}%  (pre-868: 11,430 = 10.6687%; the honest inert share "
          f"re-recorded)")
    print(f"    DIAGNOSTIC, exact-minute reading (NOT gated on): "
          f"{int(frozen_exact[D].sum())} 'frozen' ({pct(int(frozen_exact[D].sum())):.4f}%) — that "
          f"reading measures minute-bar SPARSITY, not staleness")
    for rec in interiors:
        print(f"      INTERIOR FROZEN  slot {rec['slot']:>3}  {rec['symbol']:<12} "
              f"{rec['frozen_interior_cells']} cells forward-filled mid-window "
              f"({rec['rows']} real minute rows over the whole window, ids {rec['asset_ids']})")
    for rec in tails:
        print(f"      STALE TAIL  slot {rec['slot']:>3}  {rec['symbol']:<12} last real close "
              f"{rec['last']} -> {rec['frozen_tail_cells']} frozen decision cells "
              f"({rec['days_stale']} days stale at the window end)"
              + ("  [named ticker migration]" if rec["symbol"] in EXPECT_MIGRATION_SYMBOLS else
                 "  [NOT named in D30 — reported, not thresholded away]"))
    for pred, sy in succ.items():
        w = whole[pred]
        print(f"      migration   {pred}: whole-table span {fmt(w['first_ts'])}..{fmt(w['last_ts'])}"
              f" ({w['rows']} real rows) — dead in the WHOLE table after that date")
        if sy and sy["rows"]:
            print(f"      successor   {pred} -> {sy['symbol']}: PRESENT (asset_id "
                  f"{sy['asset_id']}, {sy['rows']} rows, {fmt(sy['first_ts'])}.."
                  f"{fmt(sy['last_ts'])}) — REPORTED ONLY, NOT spliced (universe change, needs a "
                  f"planner ruling)")
        else:
            print(f"      successor   {pred} -> {SUCCESSOR_TICKERS[pred]}: ABSENT from the DB")
    tail_syms = sorted(r["symbol"] for r in tails)
    interior_syms = sorted(r["symbol"] for r in interiors)
    big_interior = sorted(r["symbol"] for r in interiors
                          if r["frozen_interior_cells"] >= EXPECT_INTERIOR_FROZEN_CELLS_EACH - 15)
    # The 6 split slots (D31's measured identity split): their alias groups must be the named
    # twin pairs, the gate must record a large ZERO-diff overlap, and post-union they must
    # read as REAL data (interior cells ~0; /USDC/USD's residual sub-gap residue is named).
    split_recs = {r["symbol"]: r for r in slots if r["symbol"] in EXPECT_SPLIT_SYMBOLS}
    gate_zero = {}
    for k, sym, g, dsy in aliased:
        for o in gate_by_ids.get(tuple(g), []):
            gate_zero.setdefault(sym, []).append(
                o["shared_ts"] == 0 or o["max_rel_diff"] == 0.0)
    checks = {
        "no_unknown_symbols_in_assets": not unknown,
        "no_slot_is_wholly_inert": len(zero_cov) == 0,
        f"panel_rows_is_{EXPECT_N_PANEL_ROWS}_pm{EXPECT_PANEL_ROW_TOLERANCE}": abs(
            n_panel - EXPECT_N_PANEL_ROWS) <= EXPECT_PANEL_ROW_TOLERANCE,
        f"decision_rows_is_{EXPECT_N_DECISION_ROWS}_pm{EXPECT_PANEL_ROW_TOLERANCE}": abs(
            n_dec_rows - EXPECT_N_DECISION_ROWS) <= EXPECT_PANEL_ROW_TOLERANCE,
        # --- axis 1: INERT (no price at all) — unchanged by the union (heads are pre-listing) ---
        "inert_cell_share_matches_pre_registration": abs(
            pct(head_cells) - EXPECT_INERT_DECISION_CELL_PCT) <= EXPECT_INERT_PCT_TOLERANCE,
        # --- axis 2: FROZEN — RE-REGISTERED post-868 (was 2.2243%) ---
        "frozen_cell_share_matches_post868_registration_TIGHT": abs(
            pct(fz_tot) - EXPECT_FROZEN_DECISION_CELL_PCT) <= EXPECT_FROZEN_PCT_TOLERANCE,
        "INTERIOR_frozen_share_matches_post868_registration_TIGHT": abs(
            pct(fz_int) - EXPECT_REAL_INTERIOR_FROZEN_PCT) <= EXPECT_REAL_INTERIOR_PCT_TOLERANCE,
        # --- axis 3: the headline total — RE-REGISTERED post-868 (was 10.6687%) ---
        "not_real_cell_share_matches_post868_registration": abs(
            pct(not_real) - EXPECT_NOT_REAL_DECISION_CELL_PCT) <= EXPECT_NOT_REAL_PCT_TOLERANCE,
        # --- structure ---
        "the_three_categories_partition_the_grid": partition_is_a_partition,
        "the_frozen_split_INTERIOR_xor_TRAILING_is_disjoint": frozen_split_is_a_partition,
        "stale_tail_slot_count_in_band": (
            EXPECT_STALE_TAIL_SLOT_BAND[0] <= len(tails) <= EXPECT_STALE_TAIL_SLOT_BAND[1]),
        "both_named_migrations_are_stale_tailed": all(
            m in tail_syms for m in EXPECT_MIGRATION_SYMBOLS),
        "migrations_are_dead_in_the_WHOLE_table_after_2025_03_31": all(
            fmt(whole[m]["last_ts"]) == "2025-03-31" for m in SUCCESSOR_TICKERS),
        "both_migrations_are_ALSO_late_listing": all(
            any(h["symbol"] == m and h["inert_head_cells"] > 0 for h in heads)
            for m in EXPECT_MIGRATION_SYMBOLS),
        # --- TASK_868 gate (i): the identity repair, measured ---
        "all_17_frame_slots_alias_groups_resolved": len(aliased) == 17,
        "the_6_split_slots_are_aliased_to_the_named_twins": all(
            sorted(split_recs[s]["asset_ids"]) == sorted(EXPECT_SPLIT_SYMBOLS[s])
            for s in EXPECT_SPLIT_SYMBOLS),
        "the_6_split_slots_read_as_REAL_data": all(
            split_recs[s]["frozen_interior_cells"]
            <= EXPECT_SPLIT_RESIDUAL_CELLS.get(s, 0) + 5 for s in EXPECT_SPLIT_SYMBOLS),
        "agreement_gate_records_ZERO_diff_overlaps_for_the_6_twins": all(
            any(gate_zero.get(s, [])) for s in EXPECT_SPLIT_SYMBOLS),
        "frozen_interior_is_the_5_slot_residue_plus_named_small_terms": (
            sorted(set(big_interior)) == sorted(set(EXPECT_INTERIOR_FROZEN_SYMBOLS))),
        "every_named_residual_interior_symbol_is_disclosed": all(
            s in interior_syms for s in EXPECT_INTERIOR_FROZEN_SYMBOLS),
        "no_successor_is_spliced": True,          # this leg never writes; splicing is out of scope
    }
    for k, v in checks.items():
        print(f"    {'OK  ' if v else 'FAIL'} {k}")
    print(f"  GATE (i) SUMMARY: frozen-interior 2,013 (1.8789%) -> {fz_int} ({pct(fz_int):.4f}%); "
          f"the 6 split slots read as REAL data via ids "
          f"{[EXPECT_SPLIT_SYMBOLS[s] for s in EXPECT_SPLIT_SYMBOLS]}; honest not-real share "
          f"10.6687% -> {pct(not_real):.4f}%")
    return {"leg": "iii_real_db_coverage_cell_level", "passed": all(checks.values()),
            "checks": checks, "db": "trading_data.db (mode=ro)", "window": list(REAL_WINDOW),
            "bar_minutes": step, "warm_rows": REAL_WARM_ROWS, "n_slots": len(frame),
            "n_distinct_minute_timestamps": n_full, "n_panel_rows": n_panel,
            "n_decision_rows": n_dec_rows, "n_decision_cells": n_cells,
            "inert_cells": head_cells, "inert_decision_cell_pct": round(pct(head_cells), 4),
            "frozen_cells": fz_tot, "frozen_decision_cell_pct": round(pct(fz_tot), 4),
            "frozen_interior_cells": fz_int,
            "frozen_interior_decision_cell_pct": round(pct(fz_int), 4),
            "frozen_tail_cells": fz_tail,
            "frozen_tail_decision_cell_pct": round(pct(fz_tail), 4),
            "not_real_cells": not_real, "not_real_decision_cell_pct": round(pct(not_real), 4),
            "live_cells": live, "live_decision_cell_pct": round(pct(live), 4),
            "partition_overlap_cells": partition_overlap,
            "partition_uncovered_cells": partition_uncovered,
            "exact_minute_diagnostic_frozen_cells": int(frozen_exact[D].sum()),
            "exact_minute_diagnostic_pct": round(pct(int(frozen_exact[D].sum())), 4),
            "n_zero_coverage_slots": len(zero_cov), "n_late_listing_slots": len(heads),
            "n_interior_frozen_slots": len(interiors), "interior_frozen_symbols": interior_syms,
            "n_stale_tail_slots": len(tails), "stale_tail_symbols": tail_syms,
            "min_tail_rows_disclosed": EXPECT_MIN_TAIL_ROWS_DISCLOSED,
            "n_aliased_slots": len(aliased),
            "aliased_slots": [{"slot": k, "symbol": sym, "asset_ids": g, "db_symbols": dsy,
                               "overlap_agreement": gate_by_ids.get(tuple(g), [])}
                              for k, sym, g, dsy in aliased],
            "per_slot": [{k: (v if not isinstance(v, dict) or k == "per_id_rows" else v)
                          for k, v in r.items()} for r in slots],
            "interior_frozen_slots": interiors, "stale_tails": tails,
            "migrations_whole_table_span": {
                m: {"rows": whole[m]["rows"], "first": fmt(whole[m]["first_ts"]),
                    "last": fmt(whole[m]["last_ts"])} for m in whole},
            "successors_present_reported_only": succ,
            "pre_868_registration": {
                "inert_pct": 8.4444, "frozen_pct": 2.2243, "frozen_interior_pct": 1.8789,
                "not_real_pct": 10.6687, "frozen_interior_cells": 2013},
            "definitions": {
                "real": "the slot's alias-UNION printed at least one real close inside the bar "
                        "((r-1)*step, r*step] that panel row r stands for",
                "inert": "no price at all at that (decision row, slot): a pre-listing head or a "
                         "wholly uncovered slot; force-HOLD under every operator (853 leak fix)",
                "frozen_interior": "a forward-filled price on a hole that RE-ANCHORS before the "
                                   "window end; force-HELD for OPENING (TASK_868 D31 a2), still "
                                   "forward-filled for VALUATION (D30 opt (i), PV-exact)",
                "frozen_trailing": "a forward-filled price after a delisting, constant to the "
                                   "window end; force-HELD for OPENING under the same causal gate",
                "exact_minute_diagnostic": "cells whose EXACT sampled minute carries no print. "
                                           "Normal on sparse minute data (the canon [::bar_minutes] "
                                           "operator samples one minute, it does not aggregate the "
                                           "bar) — reported, NOT counted as frozen",
                "row_convention": "decision rows = panel rows - warm(60)"}}



# =============================================================================================
# TASK_868 (D31 ruled opt (a)) — the identity repair + the causal frozen-cell mask, fixture-proven
# =============================================================================================
# The fixture (fixtures/make_fixture.py) carries the two synthetic phenomena, on 2025-09-10..
# 2025-09-12 (BEFORE the copied slice, so no smoke window's universe or panel moves):
#   * TWX/USD (asset_id 90001) prints minutes 0..1439 of the window then goes dark; its twin
#     TWXUSD (90002) prints minutes 720..2879 — AGREEING in the 720..1439 overlap and carrying
#     the ORPHANED span 1440..2879 the original lacks. The D31 ingest-split pattern.
#   * HOLE/USD (90003) prints every minute EXCEPT minutes 1000..1599 (a 600-minute multi-bar
#     interior hole), re-anchoring 100 -> 160 (+60%) on resumption.
T868_WINDOW = ("2025-09-10", "2025-09-12")
T868_TWIN = ("TWX/USD", 90001, 90002)
T868_HOLE = "HOLE/USD"
T868_HOLE_ID = 90003
T868_FILLER = "XBT/USD"          # a dense canon slot, so a frame is not 100% the phenomenon


def leg_alias_union_and_agreement_gate(R, imp):
    """TASK_868 gate (ii) — the ALIAS UNION and the FAIL-LOUD AGREEMENT GATE, demonstrated BOTH
    ways on the fixture: the agreeing twins union cleanly (the orphaned span becomes REAL data);
    a tampered disagreeing overlap REFUSES, naming both asset_ids and the first disagreeing
    timestamp. Both the FRAMED loader and the CANON unpinned loader are exercised (D31 (a1)
    applies to both)."""
    import numpy as np
    import shutil
    import sqlite3
    print("\n=== (868-ii) ALIAS UNION + AGREEMENT GATE — the fixture twin pair, both ways ===")
    ev = _load_eval_module()
    sym, id_a, id_b = T868_TWIN
    frame_symbols = [sym, T868_FILLER]
    # --- the agreeing union (the fixture as shipped) ---
    px, syms, ts, cov, gate = ev.load_minute_prices_framed(
        FIXTURE_DB, T868_WINDOW[0], T868_WINDOW[1], frame_symbols, 10,
        require_active_slots=0)
    c0 = next(c for c in cov if c["slot"] == 0)
    col = np.asarray(px)[:, 0]
    n_bars = px.shape[0]
    fz = ev.frozen_decision_cells(cov, 0, n_bars)
    # --- the canon unpinned loader on the same symbol ---
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)
    from mcts_trader.env.vectorbt_env import VectorBTTradingEnv
    import logging
    logging.disable(logging.CRITICAL)
    env = VectorBTTradingEnv(db_path=FIXTURE_DB, start_date=T868_WINDOW[0],
                             end_date=T868_WINDOW[1], specific_symbols=[sym],
                             initial_cash=10000.0, max_steps=100)
    logging.disable(logging.NOTSET)
    canon_col = np.asarray(env.prices)[:, 0]
    # --- the tampered copy: one overlapping close moved by 0.5% (> rel 1e-12) ---
    tdb = R("tampered_alias.db")
    shutil.copyfile(FIXTURE_DB, tdb)
    tamper_ts = 1000  # minute 1000 is inside the 720..1439 overlap
    con = sqlite3.connect(tdb)
    t0 = con.execute("SELECT MIN(timestamp) FROM ohlcv WHERE asset_id = ?",
                     (id_a,)).fetchone()[0]
    con.execute("UPDATE ohlcv SET close = close * 1.005 WHERE asset_id = ? AND timestamp = ?",
                (id_b, t0 + 60 * tamper_ts))
    con.commit()
    con.close()

    def refuses(db):
        try:
            ev.load_minute_prices_framed(db, T868_WINDOW[0], T868_WINDOW[1],
                                         frame_symbols, 10, require_active_slots=0)
            return None
        except Exception as e:                    # AliasDisagreementError (a RuntimeError)
            return f"{type(e).__name__}: {e}"

    framed_refusal = refuses(tdb) or ""

    def canon_refuses(db):
        try:
            logging.disable(logging.CRITICAL)
            VectorBTTradingEnv(db_path=db, start_date=T868_WINDOW[0], end_date=T868_WINDOW[1],
                               specific_symbols=[sym], initial_cash=10000.0, max_steps=100)
            logging.disable(logging.NOTSET)
            return None
        except Exception as e:
            logging.disable(logging.NOTSET)
            return f"{type(e).__name__}: {e}"

    canon_refusal = canon_refuses(tdb) or ""
    checks = {
        # --- BOTH WAYS, part 1: agreeing twins union CLEANLY ---
        "framed_slot_resolves_to_BOTH_twin_ids": c0["asset_ids"] == [id_a, id_b],
        # TASK_869: the fixture's synthetic rows are anchored at the (local-time) epoch the
        # fixture was CUT with; under UTC window resolution the T868 window's start bound
        # lands 120 minutes INTO the original's span, so id_a contributes 1320 of its 1440
        # rows (the twin's 720..2879 span is fully inside either way). Window-SEMANTICS
        # unchanged — the anchor moved, not the rule.
        "framed_per_id_row_counts_exact": c0["per_id_rows"] == {str(id_a): 1320, str(id_b): 2160},
        "framed_overlap_agreement_disclosed_ZERO_diff": any(
            o["shared_ts"] == 720 and o["max_rel_diff"] == 0.0
            for o in c0["overlap_agreement"]),
        "framed_union_covers_the_ORPHANED_span_as_real": (
            c0["covered_rows"] == c0["n_rows"] and fz["total"] == 0),
        "framed_no_frozen_cells_anywhere_on_the_union": fz["total"] == 0,
        "framed_open_gate_all_open_on_the_unioned_slot": bool(
            np.asarray(gate)[:, 0].all()),
        "framed_prices_continuous_through_the_orphaned_span": bool((col == 100.0).all()),
        "canon_slot_discloses_both_ids": (
            env.asset_aliases_[0]["asset_ids"] == [id_a, id_b]
            and env.asset_aliases_[0]["aliased"] is True),
        "canon_union_covers_the_ORPHANED_span": bool((canon_col == 100.0).all()),
        # --- BOTH WAYS, part 2: a tampered disagreeing overlap REFUSES, naming ids + ts ---
        "tampered_overlap_REFUSED_by_the_framed_loader": (
            "AliasDisagreementError" in framed_refusal
            and f"asset_ids {id_a} and {id_b}" in framed_refusal
            and str(t0 + 60 * tamper_ts) in framed_refusal),
        "tampered_overlap_REFUSED_by_the_canon_loader": (
            "AliasDisagreementError" in canon_refusal
            and f"asset_ids {id_a} and {id_b}" in canon_refusal
            and str(t0 + 60 * tamper_ts) in canon_refusal),
    }
    for k, v in checks.items():
        print(f"    {'OK  ' if v else 'FAIL'} {k}")
    print(f"    fixture twins {sym} ids [{id_a}, {id_b}]: original minutes 0..1439, twin "
          f"720..2879 (orphaned span 1440..2879) -> ONE continuous 2880-minute series, "
          f"0 frozen cells, gate overlap 720 shared ts at max rel diff 0.0")
    print(f"    tampered copy (twin close x1.005 at ts {t0 + 60 * tamper_ts}): framed loader -> "
          f"{framed_refusal.splitlines()[0][:120] if framed_refusal else 'NOT REFUSED'}")
    return {"leg": "868ii_alias_union_and_agreement_gate",
            "passed": all(checks.values()), "checks": checks,
            "twin_ids": [id_a, id_b], "overlap_shared_ts": 720,
            "framed_refusal": framed_refusal[:400], "canon_refusal": canon_refusal[:400]}


def leg_open_gate_causality(R, imp):
    """TASK_868 gate (iii) — the open-gate mask is CAUSAL: the mask at decision t reads only
    data <= t. Probe (the ruling's own example): for a slot whose gap RESUMES at t+k, the mask
    over every bar <= t is IDENTICAL whether or not the rows after t exist in the DB copy.
    Uses the fixture's HOLE/USD (600-minute interior hole at 1000..1599, resuming at 1600).
    Also asserts the masked bars are exactly the hole bars (interior AND trailing are gated
    alike — at decision time they are indistinguishable)."""
    import numpy as np
    print("\n=== (868-iii) OPEN-GATE CAUSALITY — the mask at t reads only data <= t ===")
    ev = _load_eval_module()
    step = 10
    frame_symbols = [T868_HOLE, "TWX/USD"]   # dense companion drives the axis
    # Full window: the hole resumes at minute 1600 (bar 160).
    px_f, _, ts_f, cov_f, gate_f = ev.load_minute_prices_framed(
        FIXTURE_DB, T868_WINDOW[0], T868_WINDOW[1], frame_symbols, step,
        require_active_slots=0)
    gate_full = np.asarray(gate_f)[:, 0]
    # Truncated window: ends INSIDE the hole (minute 1200 = bar 120), so the hole's resumption
    # rows do not exist in this read at all.
    px_t, _, _, cov_t, gate_t = ev.load_minute_prices_framed(
        FIXTURE_DB, T868_WINDOW[0], "2025-09-10T20:00", frame_symbols, step,
        require_active_slots=0)
    gate_trunc = np.asarray(gate_t)[:, 0]
    n_common = min(gate_trunc.shape[0], gate_full.shape[0])
    # The hole bars, derived (not hardcoded): bar b covers AXIS rows ((b-1)*step, b*step]; the
    # hole is minutes 1000..1599 RELATIVE TO THE FIXTURE'S OWN ANCHOR (the epoch the fixture was
    # cut with — local-midnight at the time), while the axis starts at the window's first
    # distinct minute, which under TASK_869's UTC resolution is 120 minutes into that anchor.
    # Derive the hole's AXIS-relative indices from the DB itself (HOLE's MIN timestamp + the
    # loaded axis), so the expectation tracks the window resolution instead of pinning one.
    import sqlite3 as _sq
    _con = _sq.connect(f"file:{FIXTURE_DB}?mode=ro", uri=True, timeout=60)
    try:
        t0_hole = _con.execute("SELECT MIN(timestamp) FROM ohlcv WHERE asset_id = ?",
                               (T868_HOLE_ID,)).fetchone()[0]
    finally:
        _con.close()
    axis_off = (int(ts_f[0]) - int(t0_hole)) // 60          # fixture-minute of axis row 0
    d0, d1 = 1000 - axis_off, 1599 - axis_off               # the dark span, in axis rows
    hole_bars = [b for b in range(px_f.shape[0]) if (b - 1) * step >= d0 and b * step <= d1]
    masked_bars = sorted(int(b) for b in np.nonzero(~gate_full)[0])
    checks = {
        # THE PROBE: identical masks over the shared prefix, whether or not the resumption exists
        "mask_over_shared_prefix_is_IDENTICAL_with_and_without_future_rows": bool(
            (gate_full[:n_common] == gate_trunc[:n_common]).all()),
        "truncation_did_remove_the_resumption_rows": (
            gate_trunc.shape[0] < gate_full.shape[0]
            and 1600 > (gate_trunc.shape[0] - 1) * step),
        # ... and the mask really does gate exactly the hole bars, on the full read
        "masked_bars_are_exactly_the_hole_bars": masked_bars == hole_bars,
        "the_hole_is_multi_bar": len(hole_bars) > 1,
        "the_bar_before_and_after_the_hole_are_open": bool(
            gate_full[hole_bars[0] - 1] and gate_full[hole_bars[-1] + 1]),
        # the disclosure side agrees: the hole is one INTERIOR frozen span (not a tail)
        "hole_is_disclosed_as_ONE_interior_frozen_span": (
            next(c for c in cov_f if c["slot"] == 0).get("frozen_spans")
            == [[hole_bars[0], hole_bars[-1]]]
            and next(c for c in cov_f if c["slot"] == 0).get("frozen_interior_rows")
            == len(hole_bars)),
    }
    for k, v in checks.items():
        print(f"    {'OK  ' if v else 'FAIL'} {k}")
    print(f"    HOLE/USD minutes 1000..1599 dark: bars {hole_bars[0]}..{hole_bars[-1]} gated "
          f"({len(hole_bars)} bars); mask over the shared {n_common} bars is byte-identical "
          f"with the resumption rows present vs deleted (full {gate_full.shape[0]} bars vs "
          f"truncated {gate_trunc.shape[0]})")
    return {"leg": "868iii_open_gate_causality", "passed": all(checks.values()),
            "checks": checks, "hole_bars": [hole_bars[0], hole_bars[-1]],
            "n_bars_full": int(gate_full.shape[0]), "n_bars_truncated": int(gate_trunc.shape[0])}


def leg_tz_offset_iso_probe(R, imp):
    """(869-fix) TZ-AWARE ISO-8601 inputs resolve to the CORRECT INSTANT — never silently
    re-labeled UTC. The strict eval validator (eval_vector._validate_eval_date) accepts
    ISO-8601, so --start/--end can carry an explicit offset; the run-25 panel found the
    fromisoformat fallback clobbering it (``.replace(tzinfo=utc)`` unconditionally = a
    "2025-06-01T00:00:00+05:00" window silently shifted 5h). Probed at BOTH levels:
      * the mirror (eval_vector._to_epoch): offset-bearing -> the offset's instant;
        naive -> UTC (unchanged); plain %Y-%m-%d -> UTC midnight (unchanged).
      * the canon loader (VectorBTTradingEnv on the fixture): an offset-bearing start_date
        yields the SAME panel as the equivalent epoch-seconds start, and a DIFFERENT panel
        than the same digits read naive-UTC (the probe has teeth: 2h apart)."""
    import numpy as np
    from datetime import datetime, timezone
    print("\n=== (869-fix) TZ-AWARE ISO PROBE — explicit offsets resolve to the correct instant ===")
    ev = _load_eval_module()
    off = ev._to_epoch("2025-06-01T00:00:00+05:00", None)
    naive = ev._to_epoch("2025-06-01T00:00:00", None)
    plain = ev._to_epoch("2025-06-01", None)
    want_off = int(datetime(2025, 5, 31, 19, 0, tzinfo=timezone.utc).timestamp())
    want_naive = int(datetime(2025, 6, 1, 0, 0, tzinfo=timezone.utc).timestamp())
    # --- canon loader level: offset-bearing start == the same instant as epoch seconds ---
    # (XBT's first fixture print is 2025-09-13T23:00Z, so the pair is chosen to straddle real
    # prints: the offset start 22:30Z includes 23:00Z+ rows the naive same-digits start
    # 00:30Z excludes — the teeth.)
    from mcts_trader.env.vectorbt_env import VectorBTTradingEnv
    import logging
    epoch_instant = int(datetime(2025, 9, 13, 22, 30, tzinfo=timezone.utc).timestamp())

    def panel(start):
        logging.disable(logging.CRITICAL)
        env = VectorBTTradingEnv(db_path=FIXTURE_DB, start_date=start,
                                 end_date="2025-09-16", specific_symbols=["XBT/USD"],
                                 initial_cash=10000.0, max_steps=10)
        logging.disable(logging.NOTSET)
        return np.asarray(env.prices)

    p_offset = panel("2025-09-14T00:30:00+02:00")      # == 2025-09-13T22:30Z == epoch_instant
    p_epoch = panel(epoch_instant)
    p_naive = panel("2025-09-14T00:30:00")             # naive -> 00:30Z, 2h LATER (the teeth)
    checks = {
        "offset_iso_resolves_to_the_offsets_instant": off == want_off,
        "offset_iso_is_NOT_the_naive_reading": off != naive,
        "naive_iso_still_labeled_UTC": naive == want_naive,
        "plain_date_still_UTC_midnight": plain == want_naive,
        "loader_offset_start_EQUALS_epoch_start": (
            p_offset.shape == p_epoch.shape and bool((p_offset == p_epoch).all())),
        "loader_offset_start_DIFFERS_from_naive_same_digits": (
            p_offset.shape != p_naive.shape or not bool((p_offset == p_naive).all())),
    }
    for k, v in checks.items():
        print(f"    {'OK  ' if v else 'FAIL'} {k}")
    print(f"    '2025-06-01T00:00:00+05:00' -> {off} (2025-05-31T19:00Z, the correct instant; "
          f"the pre-fix re-label gave {naive} = 5h late); naive and plain dates -> UTC "
          f"midnight, unchanged. Loader: '+02:00' start == epoch start, != naive start.")
    return {"leg": "869fix_tz_offset_iso_probe", "passed": all(checks.values()),
            "checks": checks, "offset_epoch": off, "naive_epoch": naive}


def leg_masked_reanchor_nets_zero(R, imp):
    """TASK_868 gate (iv), second half — the pre-fix fabricated re-anchor repro now nets 0 on
    the masked slot. A book that OPENS on the gapped slot DURING the frozen span and HOLDs
    through the resumption captures the whole +60% re-anchor at fee 0 when unmasked (the
    fabrication D31 exists to remove); under the causal open-gate the same action stream is
    forced to HOLD at every frozen decision, so it never opens at a price no market offered
    and nets EXACTLY 0 (final PV == initial cash to the cent). The VALUATION fill is the same
    in both walks — only OPENING is gated."""
    import numpy as np
    import jax.numpy as jnp
    print("\n=== (868-iv) MASKED RE-ANCHOR — the fabrication now nets 0 on the masked slot ===")
    ev = _load_eval_module()
    gdb = _write_gap_db(R("gap_minute_868.db"))
    px, _, _, cov, gate = ev.load_minute_prices_framed(
        gdb, PV_WINDOW[0], PV_WINDOW[1], GAP_SYMBOLS, GAP_STEP, require_active_slots=0)
    filled = np.asarray(px)
    n_assets = len(GAP_SYMBOLS)
    # decisions span the whole panel so the frozen span is inside the walk
    start_row, N = 20, filled.shape[0] - 1 - 20
    initial = 100_000.0
    cfg = imp.EnvConfig(
        n_assets=n_assets, initial_cash=initial, fee_bps=0.0, slippage_bps=0.0,
        max_steps=8, window_size=start_row, position_size_pct=0.5 / n_assets,
        vector_action=True, vector_weight_rebalance=True, vector_weight_unit=-1.0,
        vector_weight_max=4.0, vector_rebalance_deadband=-1.0)
    dark = list(range(GAP_FIRST_DARK_BAR + 1, GAP_LAST_DARK_BAR + 1))
    k_open = dark[5] - start_row                    # a decision INSIDE the frozen span

    def book(k_open):
        def fn(k, state):
            a = np.full(n_assets, 1, dtype=np.int32)          # HOLD everywhere
            if k == k_open:
                a[GAP_SLOT] = 2                                # BUY the gapped slot mid-gap
            return a
        return fn

    pj = jnp.asarray(filled, dtype=jnp.float32)
    mask = imp.frame_hold_mask(pj, start_row, N, open_gate=gate)
    unmasked = imp.walk_env(pj, cfg, start_row, N, book(k_open))
    masked = imp.walk_env(pj, cfg, start_row, N, book(k_open), hold_mask=mask)
    expect_jump = GAP_POST_PRICE / GAP_PRE_PRICE - 1.0
    cent = lambda v: round(float(v), 2)                                       # noqa: E731
    checks = {
        "the_open_decision_is_inside_the_frozen_span": (
            not np.asarray(gate)[dark[5], GAP_SLOT]
            and bool(np.asarray(mask)[k_open, GAP_SLOT])),
        "UNMASKED_book_captures_the_reanchor_fabrication": (
            initial * 1.05 < unmasked["final_pv"] < initial * (1.0 + expect_jump)),
        "MASKED_book_is_forced_to_HOLD_at_the_frozen_decision": (
            masked["actions"][k_open, GAP_SLOT] == 1),
        "MASKED_book_nets_EXACTLY_zero_to_the_cent": (
            cent(masked["final_pv"]) == cent(initial)),
        "both_walks_read_the_SAME_forward_filled_panel": True,
    }
    for k, v in checks.items():
        print(f"    {'OK  ' if v else 'FAIL'} {k}")
    print(f"    slot {GAP_SLOT} dark bars {dark[0]}..{dark[-1]}, re-anchor {expect_jump:+.0%}; "
          f"book opens mid-gap at decision {k_open}: UNMASKED final_pv "
          f"{unmasked['final_pv']:,.2f} (fabricated: the gap move on 1/{n_assets} of the book) "
          f"vs MASKED {masked['final_pv']:,.2f} (== initial {initial:,.2f} to the cent)")
    return {"leg": "868iv_masked_reanchor_nets_zero", "passed": all(checks.values()),
            "checks": checks, "unmasked_final_pv": round(float(unmasked["final_pv"]), 2),
            "masked_final_pv": round(float(masked["final_pv"]), 2),
            "reanchor_pct": round(100.0 * expect_jump, 2), "open_decision": k_open}


def main():
    ap = argparse.ArgumentParser(description="TASK_865 minute frame-pin fixture smoke")
    ap.add_argument("--keep", action="store_true", help="keep a previous run dir's artifacts")
    ap.add_argument("--real-db-coverage", action="store_true",
                    help="acceptance (iii): READ-ONLY coverage count over trading_data.db "
                         "(mode=ro URI, one GROUP BY, no panel build, no GPU)")
    ap.add_argument("--only-real-db-coverage", action="store_true",
                    help="run ONLY the acceptance-(iii) leg")
    ap.add_argument("--only-pv-conservation", action="store_true",
                    help="run ONLY the (0) PV-conservation leg (seconds, no GPU, no fixture DB) — "
                         "the D30 ground-truth invariant on its own")
    args = ap.parse_args()
    if args.only_real_db_coverage:
        r = leg_real_db_coverage()
        raise SystemExit(0 if r["passed"] else 1)
    if args.only_pv_conservation:
        os.makedirs(RUN_DIR, exist_ok=True)
        r = leg_pv_conservation(lambda *p: os.path.join(RUN_DIR, *p), _imprint_module())
        print(json.dumps(r, indent=1, default=float))
        raise SystemExit(0 if r["passed"] else 1)
    if not args.keep and os.path.isdir(RUN_DIR):
        shutil.rmtree(RUN_DIR)
    os.makedirs(RUN_DIR, exist_ok=True)
    if not os.path.exists(os.path.join(CKPT_SRC, "trainer_metadata.json")):
        raise SystemExit(
            f"missing the 861 fresh-init teacher checkpoint {CKPT_SRC}; run "
            f"`python experiments/861_search_teacher_imprint/smoke_e2e.py --quick` once first.")
    R = lambda *p: os.path.join(RUN_DIR, *p)                              # noqa: E731
    report = {"legs": []}
    t0 = time.time()

    # (0) FIRST, before anything else: the PV-conservation ground truth (D30 amendment 2). It is
    # the only leg here that discriminates between the candidate stale-tail semantics, so it runs
    # ahead of every action-level check and needs neither the fixture DB nor a checkpoint.
    imp = _imprint_module()
    report["legs"].append(leg_pv_conservation(R, imp))

    frame = write_frame(R("frame.json"), FRAME_SYMBOLS, "TASK_865 fixture minute frame")
    # SAME frame, DIFFERENT path (the spec allows the path to differ; the ordered list may not).
    frame_copy = write_frame(R("frame_same_other_path.json"), FRAME_SYMBOLS,
                             "TASK_865 identical frame reached by a different path")
    reordered = list(FRAME_SYMBOLS)
    reordered[0], reordered[2] = reordered[2], reordered[0]
    frame_reordered = write_frame(R("frame_reordered.json"), reordered,
                                  "TASK_865 REORDERED same-membership frame")
    membership = list(FRAME_SYMBOLS)
    membership[7] = "DASH/USD"
    frame_membership = write_frame(R("frame_membership.json"), membership,
                                   "TASK_865 different-membership frame")

    # ------------------------------------------------- leg 1: MINUTE-pinned collection
    print("\n=== leg 1: record a MINUTE-FRAME-PINNED collection (eval_vector.py) ===")
    collection = R("chosen_vectors_minute_pinned.json")
    sh(eval_cmd(R("evalout"), collection, frame=frame), R("log_collect_minute.txt"))
    coll = json.load(open(collection))
    cp = coll["provenance"]
    cpf = cp["panel_frame"]
    coll_frame = [c["symbol"] for c in cpf["coverage"]]
    cchecks = {
        "frame_order_preserved": coll_frame == FRAME_SYMBOLS,
        "inactive_slots_exact": cpf["inactive_slots"] == EXPECT_INACTIVE_SLOTS,
        "panel_frame_source_minute": cpf.get("panel_frame_source") == "minute",
        "bar_minutes_recorded": cpf.get("bar_minutes") == 240,
        "stale_tail_disclosed_in_collection": (
            cpf.get("n_stale_tail_slots") == len(EXPECT_STALE_TAIL_SLOTS)
            and cpf.get("stale_tail_slots") == EXPECT_STALE_TAIL_SLOTS
            and [c["symbol"] for c in cpf["stale_tail_spans"]] == EXPECT_STALE_TAIL_SYMBOLS),
        # TASK_868 (a1): the alias-union disclosure is in the artifact; /EURGBP (slot 1)
        # resolved to BOTH its assets rows (370 + 1198 — both empty on this window, so the
        # union is a numeric no-op but the disclosure has something to disclose).
        "alias_disclosure_present_and_names_the_twin_ids": (
            cpf.get("n_aliased_slots") == 1
            and [c for c in cpf["coverage"] if c["slot"] == 1][0]["asset_ids"] == [370, 1198]
            and [c for c in cpf["coverage"] if c["slot"] == 1][0]["per_id_rows"]
            == {"370": 0, "1198": 0}),
        "open_gate_mask_disclosed": (
            cpf.get("n_open_gated_decision_cells") == EXPECT_FROZEN_CELLS_TOTAL
            and "open_gate_rule" in cpf),
        "n_rows_is_the_canon_minute_panel": cp["n_rows"] == EXPECT_T_ROWS,
        "n_decisions_collected": cp["n_decisions_collected"] == EXPECT_DECISIONS,
    }
    import numpy as np
    vecs = np.asarray(coll["decisions"]["chosen_vectors"])
    cchecks["inactive_slots_all_HOLD_in_the_stream"] = bool(
        (vecs[:, EXPECT_INACTIVE_SLOTS] == 1).all())
    # TASK_868 (D31 a2): the stale-tail slots' FROZEN decisions are force-HELD by the causal
    # open-gate — the polarity of these checks inverted again (they were HELD pre-D30,
    # TRADEABLE under D30 opt (i), force-HELD — whole action — under D31 (a2); VALUATION is
    # untouched).
    cchecks["stale_tail_slots_force_HELD_on_every_frozen_decision"] = all(
        (vecs[EXPECT_STALE_TAIL_FROZEN_K, s] == 1).all() for s in EXPECT_STALE_TAIL_SLOTS)
    cchecks["stale_tail_slot_traded_BEFORE_its_tail"] = bool(
        (vecs[:EXPECT_STALE_TAIL_FROZEN_K[0], EXPECT_TAIL_ONLY_SLOTS[0]] != 1).any())
    cchecks["stale_tail_disclosed_in_the_collection_provenance"] = (
        cpf.get("stale_tail_slots") == EXPECT_STALE_TAIL_SLOTS)
    for k, v in cchecks.items():
        print(f"    {'OK  ' if v else 'FAIL'} {k}")
    report["legs"].append({"leg": "i_minute_pinned_collection", "passed": all(cchecks.values()),
                           "checks": cchecks, "walk_net21": cp["walk_net21"],
                           "log": "log_collect_minute.txt"})
    report["legs"].append(leg_walk_net21_reconciliation(R, imp, collection, coll))

    # ------------------------------------ (ii) END-TO-END PASS + ASSERTED DISCLOSURE
    print("\n=== (ii) PASS LEG — matching MINUTE pin (via a DIFFERENT path), end to end ===")
    out_pass = R("results_pass.json")
    sh(imprint_cmd(out_pass, R("ckpt_pass"), collection, frame=frame_copy), R("log_pass.txt"))
    res = json.load(open(out_pass))
    pf = res.get("panel_frame")
    gate = res["teacher_vectors"]["source"]["gate"]
    ri = res["teacher_vectors"]["replay_integrity"]
    sn = res["sanity"]
    ac = sn.get("active_cells_only") or {}
    checks = {
        # --- the pin itself ---
        "panel_frame_block_present": pf is not None,
        "panel_frame_source_minute": (res.get("panel_frame_source") == "minute"
                                      and pf["panel_frame_source"] == "minute"),
        "price_source_is_the_minute_table": "minute `ohlcv`" in pf["panel_price_source"],
        "frame_symbols_recorded_in_order": res.get("frame_symbols") == FRAME_SYMBOLS,
        # --- disclosures ASSERTED, not merely printed (the 863 lesson) ---
        "inactive_slots_disclosed": res.get("inactive_slots") == EXPECT_INACTIVE_SLOTS,
        "inactive_spans_disclosed": len(pf["inactive_spans"]) == 2,
        "partial_slots_disclosed": res.get("partial_slots") == EXPECT_PARTIAL_SLOTS,
        "stale_tail_slots_disclosed": res.get("stale_tail_slots") == EXPECT_STALE_TAIL_SLOTS,
        "stale_tail_span_names_the_symbol": (
            [c["symbol"] for c in pf["stale_tail_spans"]] == EXPECT_STALE_TAIL_SYMBOLS),
        # D30 opt (i): a TAIL-ONLY slot is not partial — it is disclosed on its own axis and
        # must be in NEITHER of the other two lists. (Slot 7 on this fixture is partial AND
        # stale-tailed — the both-lists case is asserted in the positional-loader leg.)
        "tail_only_slot_in_NEITHER_inactive_NOR_partial": all(
            s not in res.get("partial_slots", [])
            and s not in res.get("inactive_slots", []) for s in EXPECT_TAIL_ONLY_SLOTS),
        "n_active_slots_6_of_8": pf["n_active_slots"] == 6 and pf["n_slots"] == 8,
        # 471 per zero-coverage slot x2 + 427 open-gated stale-tail cells x2 (the gate is LIVE
        # end to end: inert AND frozen cells are force-HELD, D31 a2).
        "force_hold_cells_exact_incl_stale_tail": (
            pf["n_force_hold_cells"] == EXPECT_TOTAL_HOLD_CELLS),
        "inactive_feature_guard_recorded": bool(
            pf.get("inactive_slot_feature_check", {}).get("checked_slots")),
        "feature_guard_channels_are_vol_ratio": all(
            s["nonzero_channels"] == EXPECT_DEAD_SLOT_CHANNELS
            for s in pf["inactive_slot_feature_check"]["checked_slots"]),
        # --- the 863 companions still hold on the minute path ---
        "degenerate_replay_marker_false": sn.get("degenerate_replay") is False,
        "active_cells_only_present": ac.get("teacher_action_agreement") is not None,
        "active_cells_only_differs_from_diluted": (
            ac.get("teacher_action_agreement") != sn["teacher_action_agreement"]),
        "active_cell_count_matches_mask": (
            ac.get("n_active_cells") == sn["n_decision_cells"] - sn["n_masked_cells"]),
        # --- the gate ---
        "frame_pin_gate_identity_matched": gate["frame_pin"]["identity_matched"] is True,
        "frame_pin_gate_both_pinned": (gate["frame_pin"]["collection_pinned"] and
                                       gate["frame_pin"]["imprint_pinned"]),
        "frame_pin_gate_sources_recorded": (
            gate["frame_pin"]["collection_frame_source"] == "minute"
            and gate["frame_pin"]["imprint_frame_source"] == "minute"
            and gate["frame_pin"]["imprint_panel_source_actual"] == "minute-framed"),
        "provenance_paths_differ": (gate["frame_pin"]["collection_provenance_path"] !=
                                    gate["frame_pin"]["imprint_provenance_path"]),
        # --- alignment: eval's masked minute walk == imprint's masked minute walk ---
        "replay_integrity_checked_and_passed": bool(ri.get("checked") and ri.get("passed")),
        # --- all the way through the sanity replay, both tiers ---
        "sanity_replay_both_tiers": all(
            res["sanity"]["per_tier"][t]["policy_inenv"] is not None for t in ("0", "21")),
        "losses_finite": res["imprint"]["all_losses_finite"],
        "policy_ce_decreased": res["imprint"]["policy_ce_decreased"],
    }
    for k, v in checks.items():
        print(f"    {'OK  ' if v else 'FAIL'} {k}")
    print(f"    reported teacher_action_agreement {sn['teacher_action_agreement']:.4f} vs "
          f"ACTIVE-CELLS-ONLY {ac.get('teacher_action_agreement')} "
          f"({sn.get('n_masked_cells')} of {sn.get('n_decision_cells')} cells masked)")
    report["legs"].append({"leg": "ii_pass_end_to_end_minute", "passed": all(checks.values()),
                           "checks": checks, "log": "log_pass.txt",
                           "sanity_net21": res["sanity"]["per_tier"]["21"]["policy_inenv"],
                           "replay_integrity": ri})

    # ------------------------------------------------- in-process legs (seconds)
    report["legs"].append(leg_positional_loader(R, imp))
    report["legs"].append(leg_stale_tail_guard(R, imp))
    report["legs"].append(leg_interior_frozen_spans(R, imp))
    report["legs"].append(leg_canon_minute_equivalence(R, imp))
    report["legs"].append(leg_teacher_floor_masked(R, imp))
    report["legs"].append(leg_daily_teacher_floor_masked(R, imp))
    report["legs"].append(leg_daily_disclosure_parity(R, imp))
    report["legs"].append(leg_require_active_slots(R, imp))
    report["legs"].append(leg_loader_identity_refusals(R, imp))
    report["legs"].append(leg_source_gate(R, imp, collection))
    # TASK_868 (D31 opt (a)): the identity repair + the causal frozen-cell mask, fixture-proven
    report["legs"].append(leg_alias_union_and_agreement_gate(R, imp))
    report["legs"].append(leg_open_gate_causality(R, imp))
    report["legs"].append(leg_tz_offset_iso_probe(R, imp))
    report["legs"].append(leg_masked_reanchor_nets_zero(R, imp))

    # ------------------------------------------------- (ii-b) refusal matrix (subprocess)
    print("\n=== (ii-b3) REFUSAL MATRIX — reorder, membership, pin mixes, empty/incoherent flags ===")
    report["legs"].append(refusal_leg(
        "reordered_same_membership", "panel_frame identity MISMATCH",
        imprint_cmd(R("results_reordered.json"), R("ckpt_reordered"), collection,
                    frame=frame_reordered, iters=5), R("log_reordered.txt")))
    report["legs"].append(refusal_leg(
        "different_membership_frame", "panel_frame identity MISMATCH",
        imprint_cmd(R("results_membership.json"), R("ckpt_membership"), collection,
                    frame=frame_membership, iters=5), R("log_membership.txt")))
    report["legs"].append(refusal_leg(
        "pinned_collection_unpinned_imprint",
        "panel_frame PIN MISMATCH (pinned collection, UNPINNED imprint)",
        imprint_cmd(R("results_cpiu.json"), R("ckpt_cpiu"), collection,
                    frame=None, source=None, iters=5), R("log_cpiu.txt")))
    report["legs"].append(refusal_leg(
        "unpinned_collection_pinned_imprint",
        "panel_frame PIN MISMATCH (PINNED imprint, unpinned collection)",
        imprint_cmd(R("results_cupi.json"), R("ckpt_cupi"), LEGACY_COLLECTION,
                    frame=frame, iters=5), R("log_cupi.txt")))
    report["legs"].append(refusal_leg(
        "empty_frame_path_fails_closed", "was given an EMPTY path",
        imprint_cmd(R("results_empty.json"), R("ckpt_empty"), collection, frame="", iters=5),
        R("log_empty.txt")))
    report["legs"].append(refusal_leg(
        "imprint_source_without_pin_fails_closed",
        "--panel-frame-source minute requires",
        imprint_cmd(R("results_nopin.json"), R("ckpt_nopin"), collection,
                    frame=None, source="minute", iters=5), R("log_nopin.txt")))
    report["legs"].append(refusal_leg(
        "eval_source_without_pin_fails_closed",
        "--panel-frame-source minute requires",
        eval_cmd(R("evalout"), R("never_written.json"), frame=None, source="minute"),
        R("log_eval_nopin.txt")))
    # The shipped 853/863 daily rule is untouched: a DAILY pin at a non-1440 stride still refuses.
    # D30 rider (b) at the CLI: the same 6/8 frame, the shared DEFAULT floor instead of the smoke's
    # explicit override (argparse last-wins, so `extra` re-sets it) -> REFUSED before any training.
    report["legs"].append(refusal_leg(
        "breadth_refusal_fires_on_the_result_producing_path",
        "--require-active-slots", imprint_cmd(
            R("results_breadth.json"), R("ckpt_breadth"), collection, frame=frame, iters=5,
            extra=("--require-active-slots", str(_default_require_active(imp)))),
        R("log_breadth.txt"), rule_tokens=("only 6 of 8",)))
    report["legs"].append(refusal_leg(
        "daily_pin_still_requires_1440", "requires --bar-minutes 1440",
        imprint_cmd(R("results_daily240.json"), R("ckpt_daily240"), collection,
                    frame=frame, source="daily", iters=5), R("log_daily240.txt")))

    # --------------------------------------------- (iv) degeneracy guard on the MINUTE pin
    print("\n=== (iv) DEGENERATE-RUN DETECTOR on the MINUTE pin, --smoke-assert OFF ===")
    # --vector-rebalance-deadband 10.0 makes every 819-book target a HOLD, so the TEACHER never
    # opens a position => teacher_floor_inenv and corridor_position (the LEDGER metric) go null
    # while the 1-iter student still trades and policy_inenv stays FINITE.
    report["legs"].append(degenerate_leg(
        "iv_degenerate_teacher_side_minute", R, frame,
        extra=["--vector-rebalance-deadband", "10.0", "--iters", "1", "--seed", "0"],
        expect_clauses=["TEACHER SIDE", "teacher_floor_inenv"],
        expect_null=["teacher_floor_inenv", "corridor_position"],
        expect_finite=["policy_inenv"]))
    # ... and the STUDENT side: the same all-HOLD teacher LEARNED over 80 iters collapses the
    # policy replay too, so policy_inenv goes null as well.
    report["legs"].append(degenerate_leg(
        "iv_degenerate_student_side_minute", R, frame,
        extra=["--vector-rebalance-deadband", "10.0", "--iters", "80", "--seed", "0"],
        expect_clauses=["STUDENT SIDE", "TEACHER SIDE", "total-HOLD policy"],
        expect_null=["policy_inenv", "teacher_floor_inenv", "corridor_position"],
        expect_finite=[]))

    # ------------------------------------------------- (i) parity (lite)
    print("\n=== (i) PARITY (lite) — unpinned and DAILY-pinned runs carry NO minute keys ===")
    out_legacy = R("results_legacy.json")
    sh(imprint_cmd(out_legacy, R("ckpt_legacy"), LEGACY_COLLECTION, frame=None, source=None,
                   iters=80, panel=MINUTE_PANEL), R("log_legacy.txt"))
    leg = json.load(open(out_legacy))
    lchecks = {
        "legacy_no_panel_frame_key": "panel_frame" not in leg,
        "legacy_no_panel_frame_source_key": "panel_frame_source" not in leg,
        "legacy_no_stale_tail_key": "stale_tail_slots" not in leg,
        "legacy_no_frame_pin_in_gate": "frame_pin" not in leg["teacher_vectors"]["source"]["gate"],
    }
    # ... and a DAILY-pinned run: the shipped 863 block, with none of the minute keys.
    daily_frame = write_frame(R("frame_daily.json"), FRAME_SYMBOLS, "TASK_865 daily-pin control")
    daily_coll = R("chosen_vectors_daily_pinned.json")
    sh(eval_cmd(R("evalout_daily"), daily_coll, frame=daily_frame, source="daily",
                panel=DAILY_PANEL), R("log_collect_daily.txt"))
    out_daily = R("results_daily.json")
    sh(imprint_cmd(out_daily, R("ckpt_daily"), daily_coll, frame=daily_frame, source="daily",
                   panel=DAILY_PANEL, iters=80), R("log_daily.txt"))
    dly = json.load(open(out_daily))
    dpf = dly["panel_frame"]
    dgate = dly["teacher_vectors"]["source"]["gate"]["frame_pin"]
    lchecks.update({
        "daily_pin_block_is_the_863_one": (
            dpf["source"] == "checkpoint-provenance-fixed-frame (TASK_853 semantics, "
                             "TASK_863 port)"
            and dpf["panel_price_source"] == "ohlcv_daily via "
                                             "eval_vector.load_daily_prices_framed"),
        "daily_pin_has_no_panel_frame_source_key": "panel_frame_source" not in dpf,
        # D30 amendment 4: the daily pin now DOES carry the stale-tail disclosure (additive keys,
        # numerics unchanged) — the mirror defect it shipped with was that a mid-window delisting
        # was forward-filled at a frozen price and disclosed NOWHERE. Fixture frame slot 5 FTM/USD:
        # its minute/daily data ends mid-window, so this is a live assertion, not a vacuous one.
        "daily_pin_HAS_the_additive_stale_tail_keys": all(
            k in dpf for k in ("n_stale_tail_slots", "stale_tail_slots", "stale_tail_spans",
                               "stale_tail_rule")),
        # This frame/window has NO daily delisting (all 89 rows covered on every live slot), so the
        # list is legitimately empty here and only the SHAPE is checkable. The LIVE daily stale tail
        # is asserted on its own window by leg (ii-a4).
        "daily_pin_stale_tail_shape_consistent": (
            dpf["stale_tail_slots"] == [c["slot"] for c in dpf["stale_tail_spans"]]
            and dpf["n_stale_tail_slots"] == len(dpf["stale_tail_slots"])),
        "daily_pin_stale_tail_slot_is_in_NEITHER_other_list": all(
            sl not in dpf["inactive_slots"] and sl not in dpf["partial_slots"]
            for sl in dpf["stale_tail_slots"]),
        "daily_pin_result_top_level_discloses_stale_tails": (
            dly.get("stale_tail_slots") == dpf["stale_tail_slots"]),
        "daily_pin_result_has_no_MINUTE_keys": "panel_frame_source" not in dly,
        "daily_pin_gate_has_no_source_keys": not any(
            k in dgate for k in ("collection_frame_source", "imprint_frame_source")),
        "daily_pin_gate_actual_source_unchanged": (
            dgate["imprint_panel_source_actual"] == "daily-framed"),
        "daily_pin_collection_block_has_no_minute_keys": not any(
            k in json.load(open(daily_coll))["provenance"]["panel_frame"]
            for k in ("panel_frame_source", "bar_minutes")),
        "daily_pin_collection_block_HAS_the_stale_tail_disclosure": all(
            k in json.load(open(daily_coll))["provenance"]["panel_frame"]
            for k in ("n_stale_tail_slots", "stale_tail_slots", "stale_tail_rule")),
    })
    for k, v in lchecks.items():
        print(f"    {'OK  ' if v else 'FAIL'} {k}")
    report["legs"].append({"leg": "i_parity_lite_additive_only", "passed": all(lchecks.values()),
                           "checks": lchecks, "log": "log_legacy.txt / log_daily.txt"})

    # ------------------------------------------------- (iii) real-DB coverage (opt-in)
    if args.real_db_coverage:
        report["legs"].append(leg_real_db_coverage())
    else:
        print("\n=== (iii) COVERAGE HONESTY — SKIPPED (pass --real-db-coverage to run it) ===")

    report["all_passed"] = all(l["passed"] for l in report["legs"])
    report["elapsed_s"] = round(time.time() - t0, 1)
    report["real_db_coverage_leg_run"] = bool(args.real_db_coverage)
    report["note"] = (
        "The BINDING acceptance-(i) evidence is the byte-level comparison of the DEFAULT, "
        "UNPINNED and DAILY-PINNED artifacts against git tag task864-parity-baseline-hygiene, "
        "run by hand with identical invocations; this file's parity leg is the in-repo guard "
        "that the minute keys stay additive. Orbax checkpoint bytes are nondeterministic "
        "run-to-run and are NOT a parity surface.")
    with open(R("report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    print("\n" + "=" * 84)
    for l in report["legs"]:
        print(f"  {'PASS' if l['passed'] else 'FAIL'}  {l['leg']}")
    print(f"TASK_865 minute frame-pin smoke: "
          f"{'ALL LEGS PASS' if report['all_passed'] else 'FAILURES'} in {report['elapsed_s']}s "
          f"-> {R('report.json')}")
    print("=" * 84)
    raise SystemExit(0 if report["all_passed"] else 1)


if __name__ == "__main__":
    main()
