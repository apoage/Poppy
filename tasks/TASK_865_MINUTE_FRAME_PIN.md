# TASK_865 — Frame pin on the MINUTE path (D29 ruled opt (a): the only basis comparable to +0.839 on universe, source, and breadth at once)

> **STATUS (2026-07-31): SHIPPED — D30 opt (i) landed, with the real leg newly GATED on DECISIONS D31.**
> Factory run 23. Resumed from the `git stash` verified in run 22, landed every D30 amendment, then a
> 3-lens panel (security / fail-closed / correctness) found **2 BLOCKER + 2 MAJOR + ~12 MINOR**; a
> FRESH fail-closed judge over that fix pass found **4 MAJOR + 4 MINOR** more. **All closed over TWO
> fix rounds**, every new guard mutation-tested (7 mutations, each shown to make a leg FAIL, then
> reverted and md5-verified). Suites on the final tree: **selftest 12/12 · 863 13/13 (run twice) ·
> 865 25/25 · 861 e2e 10/10 · scorecard 8/8.** Parity measured against `git archive HEAD`: canon
> unpinned **195/195** shared fields identical, daily-pinned **331/331** — only absolute-path stamps
> differ.
>
> **Round 2's headline catch:** an INTERIOR frozen span was reported as TRAILING whenever the decision
> grid ended inside it — the banner printed "0 INTERIOR" over a window holding 51 such cells, silencing
> the exact axis D31 exists to expose. Also: frozen cells were counted off a span list TRUNCATED at 64
> (200 true → 64 counted, 92% "live" against a true 75%); the DAILY half of the new disclosure had ZERO
> coverage; and the partition assertion was vacuous (`live` was the complement, so it was identically
> true).
>
> **What shipped.** The minute-framed loader (62/64 live on the leg-1 window vs the daily path's
> 33/64), with the stale tail FORWARD-FILLED per D30 opt (i) and disclosed. PV conservation is now a
> first-class smoke leg and it is exact: flat prices at fee 0, a LONG and a SHORT book each end at
> **100,000.00 to the cent**, while the pre-fix zeroed-tail control reproduces D30's **75,002.50 /
> 124,997.50** — reproduced independently by the orchestrator. Riders (a) masked teacher floor,
> (b) `--require-active-slots` breadth refusal (now on BOTH pin paths), (c) `scorecard.py` plumbs
> `--panel-frame-source` or refuses. Daily-pin disclosure parity with numerics untouched:
> **207/207 shared numeric fields identical + 78 named additive keys**; canon unpinned **0/208**.
>
> **The two BLOCKERs the panel caught, both demonstrated, both fixed.** (1) The loaders opened the
> 24 GB `trading_data.db` READ-WRITE; SQLite hot-journal recovery rewrote a copy's sha256 and size on
> a pure SELECT, no write SQL involved. All 11 sites now go through `_ro_connect`
> (`file:...?mode=ro`, timeout 60, realpath, refuses a nonexistent path instead of creating a 0-byte
> db). Orchestrator-verified: hot-journal db byte-identical, probe raises `attempt to write a
> readonly database`. (2) **INTERIOR frozen spans were tradeable and disclosed NOWHERE** — staleness
> was keyed off `last_real_full`, so a slot RESUMING before the window end appeared in no list and
> read `covered_rows == n_rows`. On the real leg-1 window the minute table is near-empty
> **2025-03-31 → 2025-10-01 on 11 of 64 slots** = **2,013 frozen interior cells (1.88%)**, with
> re-anchor jumps ETHBTC **+64.52%**, BTCCAD **+33.88%**, BTCGBP **+32.69%**. Now disclosed
> (`frozen_spans`), counted, and gated on their own axis. **The honest inert share is 10.67%, not the
> 8.79% recorded in D29/D30** (which counted head+tail only).
>
> **Definition correction, load-bearing.** "Frozen" means *the slot did not print anywhere inside the
> bar*, NOT *at the exact sampled minute*. The exact-minute reading is a stride artifact reporting a
> meaningless 51.22% frozen. **Retracted with it:** the panel's "the fixture is 25%
> undisclosed-frozen / 192 of 400 live" was the exact-minute reading — under the shipped definition
> the fixture has **6 frozen cells (all trailing) and 294 live**. The REAL-LEG finding stands in full.
>
> **Two recorded numbers corrected where authored.** D30's `walk_net21 ≈ +17.1242` is SUPERSEDED by
> **+14.9497** (+17.1242 is what replaying the PRE-FIX action stream against filled prices yields; a
> real re-run lets the search operator re-decide) — and it never lived in a tracked file. And
> `sanity_net21`'s −4.2446 → +12.0134 swing is NOT "the stale-tail slot's last 7 decisions are now
> tradeable": the pure valuation channel is **−1.00 Sharpe, opposite in sign and ~1/16 the
> magnitude**; the swing is dominated by the operator re-deciding and the student retraining.
>
> **Rider (a) is inert, and says so now.** `book_target_signs` returns sign 0 on a constant column, so
> the ladder HOLDs a flat-zero slot, and `_ffill_interior_only` makes the `real→0` transition its
> docstring blamed impossible. Masked == unmasked bit-identical. It ships as defence-in-depth with an
> honest docstring, not as a fix that fires.
>
> **REAL LEG IS BLOCKED ON D31**, which this task's own disclosure work made visible: interior spans
> can fabricate PnL (a re-anchoring constant series is not the fee-drag-only case opt (i) reasoned
> about), and the canon UNPINNED loader (`vectorbt_env.py:645-655`) fills identically — so the
> **+0.839 bar itself is likely contaminated the same way.** Do not launch the 861 chain until ruled.
>
> **Prior status (2026-07-30): STOPPED — NOT SHIPPED, BLOCKED on DECISIONS D30.**
> This is a STOP, not a ship: no commit, no tag. The implementation lives in `git stash`
> ("TASK_865 minute frame pin — REVIEWED, BLOCKED on D30"), carrying the 4 tracked edits plus the
> untracked `experiments/865_minute_frame_pin/` (978-line smoke). Do NOT apply it until D30 is ruled —
> the tail semantics it encodes is the thing under decision, and re-deriving the numerics is part of
> the ruling's cost.
>
> **What PASSED** (independently re-verified by the orchestrator, not just self-reported):
> acceptance (i) parity 9/9 artifacts identical vs `task864-parity-baseline-hygiene` once wall-clock
> stamps are normalised (`created_utc`; plus a `ledger_row` date that rolled at midnight mid-run) —
> re-proven on the final tree; selftest 12/12; 861 e2e 10/10 incl. the new `vi_minute_frame_pin` leg;
> 863 suite 13/13; acceptance (ii) 19/19 smoke legs incl. both refusal directions; (iv) the 863
> degenerate detector demonstrated firing on the minute path with `--smoke-assert` OFF, teacher- and
> student-side. Regression judge: ZERO byte regression across 21 legs in two isolated worktrees.
> Flow-verifier: positional fidelity proven cell-by-cell against independent per-symbol SQL
> (0/888 mismatches, plus a permutation test) — slot i really is `frame_symbols[i]`.
>
> **What BLOCKED it** (D30, 3 of 4 panel members CONFIRMING independently): the stale-tail treatment
> FABRICATES PnL. Force-HOLD gates opening, not valuation, so a position open at the boundary is
> marked at price 0 (long wiped / short paid a 100% windfall) and then frozen there. Ground truth on
> flat prices at fee 0, where any book must end at exactly 100,000.00: zeroed+guard gives
> 75,002.50 / 124,997.50. **The panel's unanimous prescription — force HOLD on the last real row —
> was implemented as a probe and is byte-identical to the broken behaviour**, because a HOLD leg's
> target under `--vector-weight-rebalance` is `clip(w_cur)` (keep the weight), not zero. Only
> forward-filling measures PV-exact. Consequently the smoke's own recorded bar
> `walk_net21 = 11.32152396918238` IS the fabricated number, and every numeric this task reported is
> VOID pending the ruling. Separately: `compute_teacher_floor` walks UNMASKED and goes bankrupt
> (Sharpe −15.83, PV 0.00) while `--teacher-floor` is hardwired ON in the read-of-record operator.
>
> **Root cause of the near-miss, worth carrying:** all 19 smoke legs assert ACTIONS; none asserted
> PORTFOLIO VALUE, and the `iia3` negative control compared held-decision sets in both arms, so it was
> structurally incapable of catching this. The flat-price / fee-0 / long-vs-short invariant belongs in
> the smoke as a first-class leg BEFORE any tail behaviour is chosen — it is the only assertion that
> discriminates between the three D30 options.
>
> **Spec drift found while building** (fix where authored): `create_jax_env_historical` is in
> `mcts_trader/training/env_factory.py:321`, not `vectorbt_env.py`; the `ORDER BY COUNT(*) DESC LIMIT`
> block is `vectorbt_env.py:578-596` (task file says 579-590) and the `specific_symbols` drop-branch is
> ~553-577; the `[::bar_minutes]` stride lives in `env_factory.py:~405`. Acceptance (iii)'s "62/64 …
> two inactive slots" conflates two definitions — under the 853 rule the loader applies it is **64/64**
> and the two named slots are PARTIAL, not inactive; and the Real-leg section's "≈3% of cells" is
> ~9.6% once late-listing heads are counted (see D30).

> **D30 RULING ADDENDUM (2026-07-30, planner ruled opt (i); operator may override) — RESUME FROM
> THE STASH.** Stale-tail semantics = **FORWARD-FILL + disclosure, uniform on BOTH pin paths**
> (the only treatment measured PV-exact; the constant-tail objection cannot fabricate PnL and is
> a strict fee drag above fee 0 — biased against the bar, the safe direction). Scope amendments,
> all mandatory:
> 1. **Resume `git stash` ("TASK_865 … BLOCKED on D30") FIRST** — before any 866/867 work; the
>    stash is the verified implementation and landing hygiene edits under it destroys it.
> 2. **PV-conservation smoke leg ships FIRST-CLASS and BEFORE the tail fix lands** (flat prices,
>    fee 0, long AND short books end at exactly 100,000.00) — the discriminating assertion 19
>    action-level legs structurally missed. It must be shown RED on the stashed behavior and
>    GREEN after.
> 3. **D30 riders (a)/(b)/(c), none optional**: (a) thread the price-validity mask into
>    `compute_teacher_floor` (fixture proof: no bankruptcy, −1.50/96,094 not −15.83/0.00);
>    (b) `--require-active-slots N` run-time refusal on the result-producing path (default
>    tuned so the leg-1 config passes and a 50/64 WF fold refuses without an explicit
>    override); (c) `scorecard.py` plumbs `--panel-frame-source` through the manifest +
>    `OPERATOR_PIN` or REFUSES a frame-pinned manifest — leg 4 must not silently read
>    `ohlcv_daily`.
> 4. **Daily-pin disclosure parity**: the shipped daily pin keeps its forward-fill numerics
>    UNCHANGED but gains the same stale-tail disclosure (additive fields; fixture proof on frame
>    slot 5 `FTM/USD`, 91/181 filled rows disclosed). Acceptance gate (i) is amended accordingly:
>    numerics bit-identical to `task864-parity-baseline-hygiene` HEAD; artifacts may gain the
>    NAMED additive disclosure fields on the daily path, per-field diff shown.
> 5. **Coverage-honesty gate (iii) re-specified to CELL level**: the read-only real-DB leg
>    reports inert DECISION CELLS on the leg-1 window — expected **≈9.6%** (22 late-listing
>    heads ≈10,320 cells + 374 tail cells of 110,976; TIA/USD 59.5%, PEPE/USD + PEPE/EUR 49.8%,
>    ARB/USD 46.7% named), with the 2 migration slots PARTIAL under the loader's own definition
>    and the loader banner fixed to disclose partial/stale-tail slots (no more "64 ACTIVE /
>    0 INACTIVE" over a 9.6%-inert grid). Materially different share = STOP.
> 6. **Re-derive every VOID numeric** across the five suites and re-promote the fixture
>    baselines (incl. the fabricated `walk_net21 = 11.32152396918238`) through the 864 promote
>    gate, per-field diff shown. Tail expectations invert as recorded in D30 (107→100 force-HOLD
>    cells, `partial_slots` [5,6,7]→[5,7], `iia3` + negative control swap polarity).
> 7. Spec-drift corrections from the banner are accepted as authoritative (`env_factory.py:321`,
>    `vectorbt_env.py:578-596`, stride at `env_factory.py:~405`).
> TRUE LIQUIDATION (D30 opt (ii)) is PARKED as its own future row — do NOT build it here.

**Status**: READY — RESUME FROM STASH — 2026-07-30 (D30 RULED — planner, opt (i); operator may
override). Previous: READY 2026-07-29 (D29 RULED — planner, opt (a)).
**Size: S/M. Risk: medium-low** (a new framed loader on a proven pattern; both existing pin
paths — 853 daily in `eval_vector.py`, 863 in `imprint.py` — stay untouched in behavior when the
new path is not selected).

## Why (DECISIONS D29 — measured during factory run 21)

The 861 real-leg chain is blocked a second time, for a precise reason: TASK_863's pin is sound
but its only available loader is DAILY (`load_daily_prices_framed`/`create_daily_panel_framed`),
and `ohlcv_daily` covers **33/64** frame slots on the leg-1 window (31 slots zero rows → forced
HOLD all window, ~48% of decision cells structurally inert), while the 858 backfill brought the
MINUTE table to **62/64** for the same frame. The +0.839 bar is minute-panel-native with a
64-asset book (frame provenance `source`: "835 minute-panel slot order"; LEDGER 832/E5-imprint:
29L/32S, 97.5% agreement). Pinning through daily therefore substitutes a SOURCE confound (D20:
|Δ|=0.0555 coverage-symmetric; cross-source comparisons RULED forbidden, daily = within-panel
grading only) plus a **64→33 breadth confound** (first-order: ST3 measures 24-asset breadth at
2.76× single-asset Sharpe; sign vs the bar unestablished) for the 59/64 universe drift D28
removed. A minute-framed loader is the only option that makes leg 3 apples-to-apples on universe
AND source AND breadth simultaneously — i.e. the only one that delivers what D28 opt (a) was
chosen to achieve.

## Scope

1. **Minute-framed loader**: teach the PIN-b `create_jax_env_historical` minute path (the
   `ORDER BY COUNT(*) DESC LIMIT n_assets` derivation at `vectorbt_env.py:579-590` and its
   consumers) to accept the recorded frame — fixed 64-slot panel in the frame's positional
   order over the minute `ohlcv` table, mirroring the SHIPPED 853/863 semantics exactly:
   zero-coverage and pre-listing slots INACTIVE and dual-disclosed (flat-zero features +
   price-validity force-HOLD), fail-loud on identity scrambling, same-membership reorder
   REFUSED, `active_cells_only` companion scalars emitted. Reuse the shipped validator rather
   than re-deriving it.
2. **Wire both consumers**: `eval_vector.py --panel-frame-from-checkpoint` on a minute-bar run
   uses the new loader (today it is daily-only); `imprint.py`'s 863 pin path likewise when the
   run is minute-sourced. Selection must be explicit and recorded in the artifact
   (`panel_source` + frame block); no silent source switching.
3. **Guard the known trap on the real leg's exact configuration**: the 863 panel's MAJOR
   (degenerate pinned run exits 0 with null Sharpes when `--smoke-assert` is off — the default
   for any non-fixture DB) was fixed pin-gated on the daily path; the minute path must carry the
   same degenerate-run detector from day one.
4. **Fixture smoke**: minute-framed collect → imprint → sanity replay end-to-end; reorder
   refused; inactive-slot disclosure asserted (the 863 lesson: disclosures are ASSERTED, not
   just printed); the 861 e2e suite extended to cover the minute-pinned path.

## Acceptance (gates)

- (i) All defaults / unpinned / daily-pinned paths bit-identical to
  `task864-parity-baseline-hygiene` HEAD (the shipped 863 daily pin included); selftest + 861
  e2e green.
- (ii) Fixture smoke per scope 4, including both refusal directions and the asserted disclosure.
- (iii) Coverage honesty: a read-only real-DB check leg (no GPU) prints the frame-slot coverage
  of the minute table on the leg-1 window — expected **62/64** with the two inactive slots
  NAMED (the FTM→S / MATIC→POL ticker migrations); a materially different count is a STOP, not
  a warning.
- (iv) Degenerate-run detector demonstrated to fire on the minute-pinned path with
  `--smoke-assert` off (the real leg's configuration).

## Real leg (planner post-ship — re-registers the 861 chain, third and final framing)

TASK_861 legs 1+2 run minute-pinned on `runs/832_e5/frame_provenance_imprint.json`
(`--bar-minutes 1440` over minute `ohlcv`, 62/64 active slots, 2 disclosed-inactive ≈ 3% of
cells — accepted as nominal vs the daily path's 48%). Leg 2/3 = 3 imprint seeds, median > +0.839
with ≥2/3 seeds above (unchanged from the D28 addendum). Leg 4 (OOS/WF) reuses the same frame
provenance. Per D17: benchmark reads only.

## Out of scope

Grading bars (TASK_861 owns them); the ExIt-continuation frame pin (row D — PARKED until an ExIt
continuation is actually scheduled); baseline/pin hygiene (TASK_866/867); any change to the
daily loaders' behavior.
